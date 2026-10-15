# 1. 入口

从实现层面，将 cuda graph 接入到正式的训练修改。由于 cuda graph 替换的是某个前反向处理流程，可以在该处理的流程最外层的 `forward` 函数处增加 cuda graph 分支，该分支会调用一个独立的 graph manager 进行 ve cuda graph 前反向处理流程。

```python
def forward(self, images, grid_thw=None, prefix_mask=None):
    images = images.to(device=self.device, dtype=self.dtype)
    graph_manager = getattr(self, "_ve_cuda_graph_manager", None)
    if graph_manager is None:
        image_features = self.vision_tower(images, grid_thw, prefix_mask)['last_hidden_state']
    else:
        if grid_thw is None:
            raise ValueError("VE CUDA Graph requires grid_thw")
        if prefix_mask is not None:
            raise ValueError("VE CUDA Graph does not support prefix_mask")
        image_features = graph_manager.forward(images, grid_thw)
    return image_features
```

原本 ve forward 的 vision_tower 流程被 graph manager 接管，而 graph manager 则是在初始化 trainer 时就已经创建的：

```python
class HybridVLMainModel:
    def _configure_ve_cuda_graph(self, visual_model):
        active_config = active_frozen_config(self._args)
        graph = active_config.model.multimodal.ve_cuda_graph
        if graph.mode == "off":
            return
        vision_encoder = unwrap_model(visual_model).vision_encoder
        if type(vision_encoder).__name__ != "AGIv2VisionTowerNative":
            raise ValueError(
                "VE CUDA Graph supports AGIv2VisionTowerNative only; "
                f"got {type(vision_encoder).__name__}"
            )
        from cybertron.models.multimodal.ve_cuda_graph import VECudaGraphManager

        vision_encoder._ve_cuda_graph_manager = VECudaGraphManager(
            vision_encoder.vision_tower,
            int(graph.max_tokens),
            int(graph.max_segments),
            int(graph.max_inflight),
            visual_model.zero_grad_buffer,
            visual_model.no_sync,
            graph.mode,
            capture_stream=torch.cuda.current_stream(),
            defer_capture=(
                (
                    graph.mode == "forward_backward"
                    and active_config.distributed.ddp.rebuild_grad_buffer
                )
                or (
                    graph.mode == "forward"
                    and active_config.framework.cuda_graph_impl == "transformer_engine"
                    and bool(
                        {"full_layer", "vpp_stage"}.intersection(
                            active_config.framework.cuda_graph_scope or ()
                        )
                    )
                )
            ),
        )
```

这里的 graph.mode 用于控制是否 capture 反向的 graph，因为在实际训练中，某些阶段会 freeze ve/ae 的更新，此时仅需要 forward graph；defer_capture 表示是否延迟捕获 cuda graph，当为 false 时，首次调用 ve 就会自行 warmup 并 capture，否则只是执行普通前向并等待 trainer 后续统一安排捕获 ve graph 的实际和顺序，目的是实现 ve graph 和 llm graph 的显存池共享和显存复用。显存复用并不是 ve cuda graph 本身的主体功能必要能力，而是在主体功能的基础上新增的显存优化，利用了 ve 和 llm 的前反向计算在时间上完全错开的特点，可以让同一块显存空间同时被 ve graph 和 llm graph 使用，只要确保使用时不存在时间上的 overlap，就不会出现数据踩踏。

# 2. 变长序列处理

vision encoder 处理的序列是变长的，但是 cuda graph 要求输入数据的 shape 必须静态、固定不变，因为需要分配固定的显存给特定的 graph。解决这个问题的方式之一是，给定一个固定的 max tokens 值和 max segments 值，作为 graph 进行前反向计算时 capture 所用的静态 shape。实际的 tokens 小于等于固定的 max tokens，图片的数量也小于等于 max segments，当有任何一个超出时，框架均会报错。包含在 max tokens 内但不属于实际有效 tokens 的部分都被 padding 处理。

因此会有单独的模块来进行变长的处理，其包裹在原本的 ve forward 之外，在处理好一个序列长度为 max tokens 的 input tensor 中真实 tokens 的部分和 padding 部分后，将 input tensor 输入给 ve forward。处理 padding 的逻辑类似于：

```python
def set_layout(self, images: torch.Tensor, grid_thw: torch.Tensor) -> int:
    tokens = int(images.shape[0])
    grid_cpu = grid_thw.detach().to(device="cpu")
    lengths = torch.repeat_interleave(grid_cpu[:, 1] * grid_cpu[:, 2], grid_cpu[:, 0]).to(
        torch.int32
    )
    described_tokens = int(lengths.sum())
    real_segments = int(lengths.numel())
    self.real_segments = real_segments

    padding_tokens = self.max_tokens - tokens
    padding_segments = max(self.max_segments - real_segments, 1)
    quotient, remainder = divmod(padding_tokens, padding_segments)
    dummy_lengths = torch.full((padding_segments,), quotient, dtype=torch.int32)
    dummy_lengths[:remainder] += 1
    if real_segments < self.max_segments:
        dummy_lengths = torch.cat((dummy_lengths, dummy_lengths.new_zeros(1)))
    endpoints = torch.nn.functional.pad(torch.cat((lengths, dummy_lengths)).cumsum(0), (1, 0))
    if endpoints.numel() != self.max_segments + 2 or int(endpoints[-1]) != self.max_tokens:
        raise RuntimeError("invalid fixed-capacity VE attention layout")

    self.cu_seqlens_cpu.copy_(endpoints)
    self.cu_seqlens.copy_(endpoints, non_blocking=True)
    rope = self.tower._maybe_compare_rot_pos_emb(grid_thw, images.device)
    self.rope[:tokens].copy_(rope)
    self.rope[tokens:].zero_()
    self.valid_tokens[:tokens].fill_(True)
    self.valid_tokens[tokens:].fill_(False)
```

处理完成后，序列长度为 max tokens 的输入以及 cu seqlen 长度为 max segments + 2 的输入会给到 ve 做 forward：

```python
def forward(self, images: torch.Tensor) -> torch.Tensor:
    hidden = self.tower.preprocessor(images)
    for block in self.tower.trunk.blocks:
        block_forward = partial(
            block,
            mask=None,
            rotary_pos_emb=self.rope,
            cu_seqlens=self.cu_seqlens,
            cu_seqlens_cpu=self.cu_seqlens_cpu,
            max_seqlen=self.max_tokens,
        )
        if (
            self.tower.trunk.ve_gradient_checkpointing
            and hidden.requires_grad
            and self.recompute_policy[1] == "full"
        ):
            hidden = checkpoint(
                block_forward, hidden, use_reentrant=True, preserve_rng_state=False
            )
        else:
            hidden = block_forward(hidden)
    return self.tower.trunk.post_trunk_norm(hidden)
```

这里提到的输入序列长度为 max tokens，比较好理解，其直接对输入数据进行了限制；而输入的 cu_seqlen 的长度之所以是 max srgments + 2，是因为原本 cu_seqlen 本身的长度就是图片数量加1，其首元素为 0，并且还需要考虑到一个边界情况，即真实的 segments 数量就是 max segments，但真是的  tokens 数量小于 max tokens，此时被 padding 的部分也是一段 segment，因此额外需要一个 cu seqlen 的槽位，此时 cu seqlen 的长度就是 max segments + 1 + 1。

# 3. 建图与显存缓冲区如何保存

## 3.1 初始化阶段

第二节提到的抽象对象是通过 padding 的方式实现定长的 ve 前反向计算，但是不涉及对 graph 进行 capture 和 replay 的行为，后者是更加上层的单个 graph 的行为管理，而且前者则主要是具体的执行层。因此我们需要抽象出一个新的类，重点关注如何捕获 graph、保存显存缓冲区、replay。在前向 replay 完成后，反向还需要使用前向计算留下的中间结果，如果这些中间结果占用的显存缓冲区被下一批数据占用了，反向计算就会出错，要新抽象出的这个类可以负责将这些缓冲区与 graph 的 pending 状态放在一起，集中管理缓冲区“什么时候可以使用、什么时候必须等待、什么时候可以清理”，可以定义如下 `_GraphSlot` 类：

```python
class _GraphSlot:
    def __init__(
        self,
        fixed_ve: _FixedCapacityVE,
        images: torch.Tensor,
        zero_grad_buffer: Callable[[], None],
        no_sync: Callable,
        capture_backward: bool,
        stream: torch.cuda.Stream | None = None,
        pool=None,
        defer_capture: bool = False,
    ) -> None:
        self.fixed_ve = fixed_ve
        self.fa_saved_outputs = []
        self.parameters = fixed_ve.parameters_for_backward
        if capture_backward and not self.parameters:
            raise ValueError("forward_backward VE CUDA Graph requires trainable VE parameters")
        self.images = images.new_zeros((fixed_ve.max_tokens, images.shape[1]))
        self.images[: images.shape[0]].copy_(images)
        self.pending = False
        self.requires_grad_init = False
        self.grad_initialized = False
        self.capture_backward = capture_backward
        self.zero_grad_buffer = zero_grad_buffer
        self.no_sync = no_sync
        if stream is None:
            stream = torch.cuda.Stream(device=images.device)
        self.stream = stream
        stream.wait_stream(torch.cuda.current_stream(images.device))

        self.counters = [
            module.assigned_tokens_per_expert
            for module in fixed_ve.modules()
            if hasattr(module, "assigned_tokens_per_expert")
        ]
        self.saved_counters = [counter.clone() for counter in self.counters]
        with (
            torch.cuda.stream(stream),
            no_sync(),
            _mask_router_padding(fixed_ve, fixed_ve.valid_tokens),
            self._selective_forward_context(CudaGraphPhase.WARMUP),
        ):
            for _ in range(3):
                zero_grad_buffer()
                output = fixed_ve(self.images)
                if capture_backward:
                    output.backward(torch.ones_like(output) * fixed_ve.valid_tokens.unsqueeze(1))
            del output
            stream.synchronize()
            zero_grad_buffer()

        if not defer_capture:
            self.capture_forward_graph(pool)
            if capture_backward:
                self.capture_backward_graph(pool if pool is not None else self.forward_graph.pool())
            self.finish_capture()
```

graph slot 在定长 ve 的上层，其在 capture 和 replay graph 的时候会直接调用定长 ve 的前反向运算，因此初始化中包含了保存定长 ve 以及用于反向传播的信息，其中 `fa_saved_outputs` 用于保留特定 FlashAttention 重计算路径需要的张量引用，保证后续反向仍然能够使用它们的显存：

```python
self.fixed_ve = fixed_ve
self.fa_saved_outputs = []
self.parameters = fixed_ve.parameters_for_backward
```

保存用于反向传播要使用的权重参数也是必要的，如果要求捕获反向，却没有可训练参数，就报错：

```python
if capture_backward and not self.parameters:
    raise ValueError(...)
```

输入数据的显存缓冲区也需要在 graph slot 初始化阶段创建。`new_zeros` 创建与原始 `images` **设备、dtype 相同**的张量，形状为 [max_tokens, 每个 patch token 的特征维度]：

```python
self.images = images.new_zeros((fixed_ve.max_tokens, images.shape[1]))
self.images[: images.shape[0]].copy_(images)
```

此外，`pending` 用于标记是否存在已经执行前向，但尚未完成反向的任务；`requires_grad_init` 用于标记是否要求反向前先执行梯度缓冲区初始化图，**用于共享显存场景**；`grad_initialized` 用于标记当前这次前向对应的梯度缓冲区初始化是否已经完成。

初始化阶段要做的一个非常重要的事是 warmup。具体来说，首先要先手机并备份 MoE 专家计数，方便最后恢复，避免 warmup 和 capture 污染训练统计：

```python
self.counters = [
    module.assigned_tokens_per_expert
    for module in fixed_ve.modules()
    if hasattr(module, "assigned_tokens_per_expert")
]
self.saved_counters = [counter.clone() for counter in self.counters]
with (
    torch.cuda.stream(stream),                              # 在指定 stream 上执行
    no_sync(),                                              # 暂停 DDP 梯度同步
    _mask_router_padding(fixed_ve, fixed_ve.valid_tokens),  # 让 router 识别 padding
    self._selective_forward_context(CudaGraphPhase.WARMUP), # 设置选择性重计算的执行上下文
):
    for _ in range(3):
        zero_grad_buffer()
        output = fixed_ve(self.images)
        if capture_backward:
            output.backward(torch.ones_like(output) * fixed_ve.valid_tokens.unsqueeze(1))
    del output
    stream.synchronize()
    zero_grad_buffer()
```

其中，最后一个上下文在没有启用选择性重计算时就是空上下文。warmup 会进行三轮循环，每一轮依次进行：清空梯度 → 前向 → 如果需要反向，则执行反向。反向使用模拟的上游梯度中，真实 token 对应位置为 1，padding 位置为 0。在 warmup 结束后需要删除 output 的引用，使用 stream.synchronize() 阻塞 CPU，等待 warmup 完成，再清空残留梯度。

graph slot 的入参 `defer_capture` 会绝对是否立即进行 graph 捕获。不推迟捕获时，先捕获前向，再按需捕获反向。反向使用外部指定的显存池；没有指定时，使用前向图的显存池，让两者具备共享池内显存的条件。`finish_capture()` 负责流之间的依赖衔接、清理梯度，以及恢复之前备份的 MoE 计数。这里的 `defer_capture` 只负责推迟 capture，前面三轮 warmup 是始终会进行的。通常 cuda graph manager 的 `prepare_capture()` 创建 graph slot 时默认直接传入 True，再由 manager 分步骤调用后续的捕获方法。

## 3.2 捕获前向 graph

graph slot 最核心的功能之一是 capture graph，并且前向和反向 graph 是分开的，首先讨论 capture forward graph 的流程:

```python
def capture_forward_graph(
    self, pool=None, before_capture: Callable[[], None] | None = None
) -> None:
    if hasattr(self, "forward_graph"):
        raise RuntimeError("VE forward graph is already captured")
    self.forward_graph = torch.cuda.CUDAGraph()
    with self._capture_context(), self._te_capture():
        with torch.cuda.graph(self.forward_graph, pool=pool, stream=self.stream):
            if before_capture is not None:
                before_capture()
            with retain_fa_capture_outputs(self.fa_saved_outputs):
                self.output = self.fixed_ve(self.images)
    self.grad_output = torch.zeros_like(self.output)
```

capture graph 主要做了三件事。首先创建了前向图对象 `self.forward_graph`；然后在指定的 stream 和显存池中捕获 `self.fixed_ve(self.iamges)` 并保存输出 `self.outputs`；捕获结束后，创建相同形状的梯度缓冲区 `self.grad_outputs`，供后续反向使用。

在这个主流程中，使用到了大量的上下文管理器，需要逐个来看一下。首先是 `_capture_context()`，这个上下文管理器主要负责准备 ve 的执行环境，涉及到的操作包括获取 `pinned_memory_capture_lock`，避免使用同一把锁的后台线程在捕获期间分配 pinned memory，干扰 capture；切换到初始化阶段创建的用于 capture graph 的 stream；通过 `no_sync()` 暂停 DDP 梯度同步；设置 router 的 padding 掩码；为选择性重计算设置 `CAPTURE` 阶段标记：

```python
@contextmanager
def _capture_context(self):
    with (
        pinned_memory_capture_lock,
        torch.cuda.stream(self.stream),
        self.no_sync(),
        _mask_router_padding(self.fixed_ve, self.fixed_ve.valid_tokens),
        self._selective_forward_context(CudaGraphPhase.CAPTURE),
    ):
        yield
```

第二个上下文管理器是 `self._te_capture()`，它负责设置 transformer_engine 和 pytorch autograd 的捕获状态。`te_set_capture_start()` 可以将 TE 的全局”正在捕获“标记为 True，让相关代码能够识别这个阶段。 `torch.autograd.graph.set_override_stale_capture_stream(True)` 允许 autograd 将未参与当前捕获的旧 stream 引用，替换为相应的捕获 stream，避免这些旧引用导致捕获失败。退出时，`finally` 语句会关闭这个开关，并调用 `te_set_capture_end()`，**即使中间发生异常也会执行清理**。但是这个 context 并没有真正去控制 cuda graph captute 的开始与结束，后者是由 `torch.cuda.graph()` 控制的：

```python
@contextmanager
def _te_capture(self):
    from megatron.core.transformer.cuda_graphs import te_set_capture_end, te_set_capture_start

    te_set_capture_start()
    torch.autograd.graph.set_override_stale_capture_stream(True)
    try:
        yield
    finally:
        torch.autograd.graph.set_override_stale_capture_stream(False)
        te_set_capture_end()
```

第三个重要的自定义上下文管理器是 `retain_fa_capture_outputs()`，它用于保留 ve 的 fa4 选择性重计算路径中，前向捕获产生的两类张量：一是 `out`，即 FlashAttention 的输出；二是 `lse`，即 softmax 的 log-sum-exp 统计量，反向计算需要它。在这个上下文管理器中， `self.fa_saved_outputs` 会被临时登记为收集列表，在执行到对应的 FlashAttention 前向时，会执行  `outputs.extend((out.detach(), lse.detach()))`，其中 `detach()` 与原张量共享显存，所以这里只增加引用，不复制数据。这里保留的原因是，cuda graph 记录了这些张量的显存地址，保留引用可以防止它们的存储过早被分配器复用，保证之后反向 replay 仍能读取正确的数据。每轮前向 replay 都会将新计算的 `out` 和 `lse` 写入这些固定地址。退出 上下文管理器是，只会恢复收集上下文，列表中的引用将继续保留，直到 `slot.reset()` 清空。 

## 3.3 捕获反向 graph

与捕获前向 graph 相对应的，当训练会涉及到反向传播时，也需要捕获反向 graph，供训练时反复 replay。下面展示的是剔除了一系列正确性检查后的主体执行流程：

```python
def capture_backward_graph(
    self, pool=None, before_capture: Callable[[], None] | None = None
) -> None:
    self.backward_graph = torch.cuda.CUDAGraph()
    with self._capture_context(), self._te_capture():
        with torch.cuda.graph(self.backward_graph, pool=pool, stream=self.stream):
            if before_capture is not None:
                before_capture()
            self.output.backward(self.grad_output * self.fixed_ve.valid_tokens.unsqueeze(1))
    self.parameter_grads = tuple(parameter.grad for parameter in self.parameters)
    self.grad_added_to_main_grad = tuple(
        parameter.grad_added_to_main_grad for parameter in self.parameters
    )
```

与捕获正向 graph 非常类似，反向也是首先实例化一个 `backward_graph`，并且调用多个上下文管理器以进入捕获环境，其中真正捕获范围由内层的 `torch.cuda.graph()` 确定。在捕获前向图阶段没有具体提的是，上下文管理器有一个重要的输入参数 `pool`，它用来指定捕获所用的显存池。调用方可以传入前向图的显存池或统一的共享显存池，让相关图具备复用池内显存的条件；具体哪些内存可以复用，还取决于执行顺序和数据的存活时间。这个函数本身不会在 pool=None 时主动选择前向图的池。捕获反向图时的 stream 指定了使用 `self.stream`，这与捕获前向图时使用的 stream 是同一个，而后续调用 `replay()` 时，图会在调用时的当前 cuda stream 上启动，并非捕获图时的 stream。

与前向图捕获流程中同样存在的，反向捕获流程中也会调用 `before_capture()`，这个函数位于 `torch.cuda.graph()` 内部，并且在反向主体之前执行，其中符合捕获条件的 GPU 操作也会被记录到这张反向图里。普通 python 语句仍然只在捕获时执行，不会因为 replay 而重新执行。当前 ve 在捕获前反向图并调用这个函数时，没有传入这个回调，因此是一个空回调，不会有任何实际执行。举一个可以实际执行的例子，假设 `before_capture()` 中传入的是以下函数，其中 `buffer` 是已经准备好的 CUDA 张量：

```python
state = {"calls": 0}

def callback():
    print("进入 callback")
    state["calls"] += 1
    buffer.zero_()
```

捕获时，代码走到 `before_capture()`，python 就会调用这个 `callback()`：首先在终端打印一次“进入 callback”，这个打印行为只会在本次捕获时执行，后续 replay 不会执行，因为这一步代码逻辑不会被捕获到 graph 中；然后 python 字段 state["calls"] 会增加 1，这一步也只会在此次捕获时执行，不会被捕获；最后会执行 `buffer.zero_()`，这一步 python 会调用 pytorch，提交的 GPU 清零操作被记录到图中，并且这个 GPU 操作会被捕获到图中，并且后续的 replay 会执行。这里的核心区别是，虽然三步处理都是 python 行为，但是第三步会向 GPU 提交操作，而前两者不会，GPU 操作是可以被 cuda graph 记录的，因此只有第三步会被捕获入图。

真正的反向行为的代码行为与 warmup 时的写法类似，即将用于回传的梯度对象先进行 mask，只保留真实 token 对应位置的 grad，padding 位置的 grad 数据全部用 0 覆盖；然后调用 `backward()`，进行反向传播计算，这里涉及到的张量的 shape 为：

```python
self.output                         [max_tokens, hidden_dim]
self.grad_output                    [max_tokens, hidden_dim]
self.fixed_ve.valid_tokens          [max_tokens]
valid_tokens.unsqueeze(1)           [max_tokens, 1]
```

mask 的乘法也在捕获范围内。需要注意的是，**mask 清零的是 padding 位置的上游梯度，并不意味着反向图会跳过所有 padding 计算**。参与 mask 乘法计算的 `self.grad_output` 是在前向捕获结束时用  `torch.zeros_like(self.output)` 创建的，所以反向捕获时它最初的状态是全为 0，但这并不妨碍捕获；autograd 仍会沿着前向建立的关系生成反向操作，梯度数值为 0 并不会取消这些操作。实际训练时，外部传回来的真实梯度会先复制到同一个 `self.grad_output` 缓冲区，再 replay。固定的是缓冲区地址和执行结构，里面的梯度数值每次都可以更新。 **此函数没有捕获优化器更新**。

在捕获了反向图后，还有创建了 `self.parameter_grads` 和 `self.grad_added_to_main_grad` 两个对象。前者是创建了所有被更新参数的梯度张量引用，会用于后续的更新；后者是一个重要的属性，由于很多框架都会 Megatron 框架对 `main_grad` 的处理，其是累积参数梯度的缓冲区，而 `grad_added_to_main_grad` 会告诉 DDP 这个参数的梯度是否已经由融合计算路径加入 `main_grad`，从而决定是否还需要执行通常的 `.grad → main_grad` 累加。在捕获阶段，融合算子的 python 代码可以把这个标记为 true，但之后 **cuda graph replay 只会重放 GPU 操作，不会重新执行这条 pyhton 属性赋值**。因此，这里先保存每个参数在捕获后的标记值；实际反向 replay 后，再由 **_ForwardBackwardReplay.backward()** 将这些值逐个恢复到参数上。否则，在融合算子已经完成累加的情况下，DDP 仍可能根据错误的标记再累加一次，造成梯度重复累积。注意这里保存的是每个参数各自的状态，并不是无条件把所有参数都设为 True。

最后展示一下实际 replay 的主体行为，理解捕获反向 graph 中准备的各个东西是如何使用的。这段实现取自 `_ForwardBackwardReplay.backward()`中与本函数直接相关的逻辑，省略检查和清理逻辑。这其中的 `gradient` 来自当前训练过程对 ve 输出的反向传播。`capture_backward_graph()` 准备的反向图、固定梯度输入缓冲区以及两组参数状态，都在这段 replay 流程中被使用：

```python
# 先把本轮真实的上游梯度写进固定缓冲区。
slot.grad_output.zero_()
slot.grad_output[:ctx.tokens].copy_(gradient)
# 执行已经捕获好的 VE 反向 GPU 操作。
slot.backward_graph.replay()
# GPU replay 不会执行 Python 属性赋值，因此手动恢复 DDP 标记。
for parameter, value in zip(
    slot.parameters, slot.grad_added_to_main_grad, strict=True
):
    parameter.grad_added_to_main_grad = value
# 前三个 None 对应 slot、images、grid_thw；后面对应参数的梯度。
return (None, None, None, *slot.parameter_grads)
```

## 3.4 finish_capture

除了前面 capture graph 的逻辑外，框架一般还有捕获完成后的收尾操作，通常是衔接 stream 的执行顺序、整理输出和梯度状态、恢复专家计数器，均为后续正式训练做准备。这部分收尾处理通常是框架的自定义行为，完整代码如下：

```python
def finish_capture(self) -> None:
    torch.cuda.current_stream(self.images.device).wait_stream(self.stream)
    if self.capture_backward:
        self.output = self.output.detach()
    for parameter in self.parameters:
        parameter.grad = None
    self.zero_grad_buffer()
    for counter, value in zip(self.counters, self.saved_counters, strict=True):
        counter.copy_(value)
```

在进行收尾工作之前，首先要确保捕获流程已经完成，因此首先进行了 `current_stream` 和用于捕获的 `self.stream` 的流同步管理，确保捕获流上已经提交的操作全部完成。这个同步是 cuda stream 之间的同步，不会阻塞 CPU 一直等到 GPU 完成。确保捕获流已提交任务被完成后，首先会用 detach 将输出与捕获阶段的 autograd 关系断开，断开计算图的 `self.output` 引用是一个持有与原本 `self.output` 同一块显存地址但是不在计算图内的对象，后续可以通过该对象，长期地访问该内存地址上的真实 token 输出数据。后续训练通过 `_ForwardBackwardReplay` 这个自定义 autograd 函数接入反向，并调用 `backward_graph.replay()`，此时完成 replay 后，在返回自定义模块的真实输出 token 时，就可以通过 `self.output` 来访问这块内存。这里要区分两件事：autograd 的反向关系用于生成反向操作，已经捕获好的 CUDA Graph 用于重放这些操作。 detach() 不会删除已经捕获好的 CUDA Graph。另外，detach() 不复制输出数据，新张量仍与原张量共享存储，因此前向 replay 仍能向同一块输出缓冲区写入新结果。这一步只在 capture_backward=True 时执行。

接下里的收尾行为是清理参数的 `.grad`，再清理梯度缓存区，这是两个不同层面的操作。`parameter.grad = None` 清除参数对象的 `.grad` 属性引用，不会把原梯度张量的数值清零。前面 `capture_backward_graph()` 已经用 `self.parameter_grads` 保存了这些对象的引用，所以把属性设为 None，也不意味着那些仍被引用的张量会随之消失。`self.zero_grad_buffer()` 则调用外部传入的梯度清理函数，负责清空框架管理的梯度缓冲区，避免准备阶段的梯度状态混入正式训练。实际清理哪些缓冲区和相关状态，由这个回调的实现决定；在框架的梯度累积机制中，还需要考虑 `main_grad`，不能仅靠把 `.grad` 设成 None 来完成清理。

最后一个收尾操作是恢复 MoE 专家的 token 计数器。在 `__init__()` 中，graph slot 收集了各模块的 assigned_tokens_per_expert，并在 warmup 之前通过 clone() 备份了它们的数值，即 `self.saved_counters`。warmup 等准备过程会经过 router，可能改变专家分配统计。因此这里将每个计数器恢复到准备之前的值，避免把这些准备工作统计成正式训练数据。这里是恢复备份值，不一定是清零：如果备份时已经有计数，就恢复那些计数。copy_() 在原计数器张量上修改数值，保留原来的对象和存储；strict=True 要求计数器与备份数量一致，否则会报错。

## 3.5 prepare

在做完了捕获的收尾操作后、真正执行 replay 操作前，还需要少量 replay 前的准备工作，这部分行为被整理到了 `prepare` 函数中。区别于建图行为，每一个 graph slot 只需要建图一次，但是 replay 的准备操作是依赖于 replay 的，因此多次 replay 意味着 `prepare` 会被反复多次调用：

```python
tokens = slot.prepare(images, grid_thw)
slot.forward_graph.replay()
```

忽略部分检查行为，准备工作的第一项是根据本轮的实际输入更新变长的布局，这里的 `set_layout` 函数在前面讲解定长 ve class 中已经提到了，核心行为是将长度为 max tokens 的 rope、长度为 max tokens 的包含真实 token 数量的 valid_token 和 长度为 max segments + 2 的 cu_seqlen 用实际输入数据填充，并将未填充部分 padding。**这些静态缓冲区的形状和地址保持不变，但内容会随本轮数据更新**，因此同一张图能够处理容量范围内不同的 token 数和序列布局。:

```python
tokens = self.fixed_ve.set_layout(images, grid_thw)
```

长度为 max tokens 的输入张量 images 也在 `prepare` 中被真实 token 数据复制，剩下的 token 位置则直接置 0。最终返回的是本次计算的真实 token 数量。

# 4. cuda graph 接入训练流程

## 4.1 自定义模块的前向

cuda graph 接入训练流程是通过一个自定义模块实现的，该自定义模块的前向依赖真实的输入 images 和用于控制 replay 的 slot，而反向会返回 replay 后得到的权重参数梯度，这就要求前向函数输入中有权重参数的站位，因此该类的前向函数可以定义为：

```python
class _ForwardBackwardReplay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, slot, images, grid_thw, *parameters):
        if slot.pending:
            raise RuntimeError("VE CUDA Graph slot already has an outstanding forward")
        tokens = slot.prepare(images, grid_thw)
        slot.forward_graph.replay()
        slot.pending = True
        ctx.slot = slot
        ctx.tokens = tokens
        return slot.output[:tokens].clone()
```

graph slot 在自定义模块输入函数中的角色非常重要，首先会判断当前传入的 graph slot 是否还在等待上一次的反向，若该 graph slot 的反向还没有处理完，那会拦截本次前向 replay 操作。因为反向需要对应前向保存在固定缓冲区里的中间结果。如果此时再次使用同一个 slot 做前向，新结果可能覆盖旧结果，让上一次反向读到错误的数据。因此，必须等对应反向处理完、释放 slot 后才能复用。

`forward()` 的输入 `images` 和 `grid_thw` 均为真实数据，因此在 replay 之前必须首先将真实数据放在 replay 使用的固定缓存区，即 `slot.prepaer()` 函数的作用，前一小节详细解释过，该函数会检查输入是否兼容，更新本轮的序列边界、位置编码和有效 token 编码，将 `images` 复制到 graph slot 的固定输入缓冲区，并将 padding 部分清零。 `prepare()` 返回的 tokens 是本轮真实的 token 数，也就是 `images.shape[0]`，虽然 replay 会按照固定的 `max_tokens` 容量运行，但代码会记住真实数据占多少位置。

完成了前序准备工作后，直接调用 graph slot 维护的前向图进行 replay 即可，这才正式发起了本轮 ve 的前向 GPU 计算，前向图读取准备好的输入 `images` 和布局，以及当前参数，写出本轮输出和反向所需的中间结果。图的输出写入 slot 持有的固定输出存储，所以后面的 `slot.output` 能读到这次的结果。这里已经进入 replay 阶段，不再进行捕获。

`forward()` 函数中剩下的处理则是服务于反向函数 `backward()` 和状态记录，即设置当前使用的 graph slot 的 `pending` 为 true，并将 graph slot 和 真实的 token 数都保存给反向使用，确保反向处理时知道梯度应该写入固定显存缓冲区的哪些位置。后续自定义的 `backward()` 函数在处理反向后，会将状态标记 `pending` 恢复成 false。需要注意的一点是，**前向函数最终返回结果时，返回的是真实 token 对应的输出，去掉了 padding 部分**。对返回的结果进行 `clone()` 也是必要的，因为当前使用的 graph slot 有可能后续被再次使用，并且被使用时，此前已有的输出结果可能还未被完全使用完成（例如多次 ve forward 前向完成后，再将多次 forward 的结果一起传给 llm backbone 部分处理），此时再次调用 `forward()` 会覆盖已有的输出，因此**有必要通过** `clone()` **复制成独立的 tensor**。

`forward()` 函数的输入中还有一个重要的参数是解包的 `*parameters`。被更新权重参数虽然不会在 `forward()` 函数中被显示使用，并且也不会被通过 ctx 保留给反向函数使用，但是依然要在这里传入，主要的原因有两个：首先，前向函数的输入中只有 images 拥有 `requires_grad` 属性，并且通常是 false，这意味着这个自定义模块不会创建对应的 autograd 反向节点，它的反向函数不会被调用执行，反向图不会进行 replay，这与预期的训练流程是完全不符的，因此提供被更新参数作为输入，可以将该自定义模块加入到训练的反向计算图中；第二，在后续进行反向 replay 时，真实计算得到的权重参数梯度有两种可能的存放位置，一个是写入到了捕获时绑定的参数梯度存储中，并在本自定义模块的反向返回，由外层 autograd/DDP 继续处理并累加到 `main_grad` 中，另一个存放位置时在 GPU 算子执行时直接将结果累加到 `main_grad` 中，此时本自定义模块返回的参数权重梯度主要起占位作用，配合后续待执行的 autograd/DDP hook，真实的梯度已经在 `main_grad` 中。

## 4.2 自定义模块的反向

自定义模块的反向依然依赖 graph slot 对显存管理和控制反向图 replay 的能力，因此首先取回 `forward()` 函数保存的 graph slot 对象，并进行基本的状态检查，确保当前 graph slot 是已经经过了前向流程、实际 token 数量的匹配检验、共享显存池是否已经进行了清零（展开来说，最后的 grad 检查与显存复用有关。共享图池中的梯度存储在其他计算阶段可能被 LLM 临时数据占用；进入视觉分支反向前，需要先将相应的梯度缓冲区清零，再开始累加本轮梯度。捕获阶段完成分配和绑定，正式运行时 replay 已记录的初始化操作）：

```python
class _ForwardBackwardReplay(torch.autograd.Function):
    @staticmethod
    def backward(ctx, gradient):
        slot = ctx.slot
        if not slot.pending:
            raise RuntimeError("VE CUDA Graph backward has no matching forward")
        if slot.requires_grad_init and not slot.grad_initialized:
            raise RuntimeError("VE backward requires grad-init replay before adapter backward")
        if gradient.shape != slot.grad_output[: ctx.tokens].shape:
            raise ValueError("VE CUDA Graph output gradient shape changed")
        slot.grad_output.zero_()
        slot.grad_output[: ctx.tokens].copy_(gradient)
        try:
            slot.backward_graph.replay()
            # Replay cannot update Python-side DDP bookkeeping captured in the graph.
            for parameter, value in zip(slot.parameters, slot.grad_added_to_main_grad, strict=True):
                parameter.grad_added_to_main_grad = value
            return (None, None, None, *slot.parameter_grads)
        finally:
            slot.pending = False
            slot.grad_initialized = False
```

`backward()` 函数首先接收下游回传的梯度 `gradient`，这个梯度张量的序列长度是真实 token 数量，对应前向 `forward()` 函数的输出。当前反向图是包裹的整个 vision encoder，而在 ve 和 llm backbone 之间其实还存在一个规模很小的 vision adaptor，在执行到 ve 反向图时，vision adaptor 的反向计算已经完成，因此此时的梯度缓冲区中已经存储了 vision adaptor 的梯度数据，无法在反向图开始 replay 前对梯度缓存区进行清零，否则会抹去 vision adaptor 的梯度数据。正确的清理时间点是在进行整个 vision model 的反向前，在完成了 llm backbone 的反向后，进行梯度缓存区的清理：

```text
调度层在视觉分支反向前执行 grad-init
    → adapter 反向，写入 adapter 参数梯度
    → 到达这里的 VE backward，检查初始化标记
    → VE 反向 replay，写入 VE 参数梯度
```

接来下会把真实回传的梯度 `gradient` 赋值到固定分配的回传梯度的缓冲区，具体行为就是先把后者置零再进行原地拷贝 `copy_()`。此时反向图 replay 的所需准备工作就已经全部完成了，可以通过 graph slot 直接调用反向图的 replay。反向图 replay 后，由于对图进行 capture 时仅捕捉到 GPU 操作，因此 param 的一些关键属性存在丢失，所以需要将 capture 阶段记录留存的 `grad_added_to_main_grad` 属性重新挂载在 param 上，后续 DDP hook 会据此决定是否还需要把普通 `.grad` 累加进 `main_grad`，避免将已经直接累加的真实梯度重复计入。这里 `strict=True` 则要求两个序列长度一致，防止参数与标记数量不匹配时被 `zip()` 静默截断。

最后 `backward()` 返回 `parameter_grads` 的原因已经在上一小节提到了，不管其是真实的梯度数据还是只是占位，都会与后续的 DDP hook 进行配合。返回退出时需要同步进行的是清理本轮 graph slot 状态，将 `pending` 和 `grad_initialized` 都置为 false。设置这两个布尔值不会清零梯度，也不是 GPU 同步。replay 通常是异步提交，GPU 操作的先后关系仍由 CUDA stream 和相应同步机制保证。注意，前面的状态检查、形状检查和梯度复制位于 try 外面；如果在那里报错，这个 finally 不会执行。即使 try 内报错后标记被重置，也不代表 GPU 或训练状态已自动恢复，可以直接继续训练。

## 4.3 graph slot 中的前反向流程

前两个小节的自定义模块 `_ForwardBackwardReplay` 由 graph slot 直接调用：

```python
def forward_backward(self, images: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    return _ForwardBackwardReplay.apply(self, images, grid_thw, *self.parameters)
```

但这个自定义模块仅会在 ve 训练流程既包含前向传播又包含反向传播时被调用。实际训练还有另一种场景，即 freeze encoder，仅更新 adaptor 部分的权重，此时 capture 环节只会捕捉前向图，训练流程中也不涉及反向图的 replay，此时框架也无需用自定义模块包裹该流程，直接回放前向图即可：

```python
def forward_only(self, images: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        tokens = self.prepare(images, grid_thw)
        self.forward_graph.replay()
        # VA backward saves this input, so it must not alias the next graph replay.
        return self.output[:tokens].clone()
```

# 5. ve cuda graph manager 维护 cuda graph 的完整生命周期

## 5.1 初始化 manager

ve cuda graph manager 在初始化环节会保存一系列参数配置，建立尚未捕获图的初始状态：

```python
class VECudaGraphManager:
    def __init__(
        self,
        tower: torch.nn.Module,
        max_tokens: int,
        max_segments: int,
        max_inflight: int,
        zero_grad_buffer: Callable[[], None],
        no_sync: Callable,
        mode: str,
        capture_stream: torch.cuda.Stream | None = None,
        pool=None,
        defer_capture: bool = False,
    ) -> None:
        if mode not in {"forward", "forward_backward"}:
            raise ValueError(f"unsupported VE CUDA Graph mode: {mode}")
        if max_inflight != 1:
            raise ValueError("VE CUDA Graph requires max_inflight=1")
        self.tower = tower
        self.max_tokens = int(max_tokens)
        self.max_segments = int(max_segments)
        self.max_inflight = int(max_inflight)
        self.zero_grad_buffer = zero_grad_buffer
        self.no_sync = no_sync
        self.mode = mode
        self.capture_stream = capture_stream
        self.pool = pool
        self.defer_capture = defer_capture
        self.slots: list[_GraphSlot] = []
        self.graphs_ready = False
        self.grad_init_graph = None
        self._capture_template = None
```

初始化环节首先做的第一件事是检查执行模式和允许的未完成的前向数量，即 `mode` 和 `max_inflight` 的取值。在第四章节有提到过目前允许的有效 `mode` 类型，分别是 `forward` 和 `forward_backward`。实际 yaml 中支持的可选 value 还有 `off`，但是取该值时，ve cuda graph manager 的外层函数 `_configure_ve_cuda_graph()` 会直接返回，不会创建 ve cuda graph manager。`max_inflight` 表示最多允许多少次前向已经执行且对应反向还没有处理完。通常使用时设置为 1，因此每个 manager 只创建一个 graph slot。它与 world size 和 segment number 都无关，也不表示一个 graph slot 只能持有一张图：训练模式的 slot 同时持有前向图和反向图。

初始化环节做的第二件事是保存要运行的模型以及固定容量配置，即 `tower`、`max_tokens`、`max_segments` 和 `max_inflight`。前者是已有的 vision encoder 模型，例如有些框架的 `tower` 其实是 `vision_encoder.vision_tower`，这里仅仅保留对其的引用，不会复制任何模型或权重，并且该 encoder 模块并不包含 adaptor 模块。

第三件事是保存梯度清理和 DDP 同步控制的回调。在某些框架中，`zero_grad_buffer` 是 `visual_mode.zero_grad_buffer` 这个可调用的函数对象，而 `no_sync` 是 `visual_model.no_sync`。前者用来清理框架的梯度缓冲区，graph slot 后续在 warmup 和捕获收尾等位置调用它，避免准备阶段产生的梯度混入正式训练；后者被调用后会提供一个上下文管理器，用于控制 DDP 的梯度同步行为，让准备和捕获阶段按所需方式运行。它不是 CUDA stream 的同步函数，不等价于“禁止所有通信”。这两个函数在 manager 初始化环节只是保留两个回调，不会在这里执行。

第四件事是保存捕获使用的 stream、内存池和延迟捕获设置。具体来说，`capture_stream` 是之后传给 graph slot、用于 warmup 和 capture 的 CUDA stream，不会在 replay 环节被使用，默认值为 None；manager 初始化环节不会创建 stream，只提供参数传导功能，若传入给 graph slot 的 stream 为 None，则会在后续 graph slot 中自行创建。在共享图池路径中（下一章节会详细讲解，是一种显存优化手段），`prepare_shared_capture_inputs()` 会重新选择与 TE 捕获配合的 stream，所以它不是初始化后就永远不变的配置。内存池 `pool` 是 CUDA Graph 内存池的句柄，后续传给 `torch.cuda.graph(..., pool=...)`，这个在此前讲解 graph slot 的章节中有展示，让图捕获时的显存分配可以使用指定的池。它不是预先申请多少 GiB 显存的数值。默认值 None 表示这里尚未指定池；非共享路径的反向捕获会使用前向图的池，共享路径则由外层提供共同的池。延迟捕获设置是 `defer_capture`，用于决定 manager 是否把捕获时机交给外层协调，默认为 false，此时首次需要运行时调用 `_capture()` 建图，然后继续执行本次正式前向 replay；当设置为 true 时，manager 会保存输入模版并执行普通前向，等待外层分阶段捕获，图就绪后才改用 replay，因此取 false 并不意味着在本初始化函数内进行捕获，只是之后首次 `forward()` 可以自行触发捕获。

初始化函数做的最后一件事是初始化生命周期状态，包括 `slots`、`graphs_ready`、`grad_init_graph` 和 `_capture_template`。`slots` 中的元素均为 `_GraphSlot` 对象，后者通过 ve cuda graph manager 的 `prepaer_capture()` 函数创建 wrapper 和 graph slot 并放入本列表；`graphs_ready` 在最初初始化时被标记成 false，表示建图流程尚未完成，在 manager 的 `finish_capture()` 完成了 graph slot 收尾后，会将其设置成 true；`grad_init_graph` 在初始化时为 None，表示尚无独立的梯度初始化图，后续共享捕获的训练路径中通过 `capture_grad_init()` 创建并在参数梯度累加前开始 replay；最后一个对象是 `_capture_template`，初始化时也为 None，表示尚未保存用于建图的输入样例，后续在延迟捕获时首次 `forward()` 保存输入副本，或在必要时通过 `ensure_capture_template()` 构造最小样例。

## 5.2 整体前向行为

ve cuda graph manager 会根据当前状态选择普通前向、建图或前向 replay，这些行为均由 `forward()` 函数负责。具体来说，这个函数是 manager 处理本次 ve 输入的入口，决定用哪条路径计算：普通前向交给 tower，图的准备与 replay 则交给对应方法和 graph slot。返回的是本次输入的 ve 特征，后续 adapter 等计算在外层继续进行。

```python
def forward(self, images: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    if self.defer_capture and not self.graphs_ready:
        if self._capture_template is None:
            self._capture_template = (images.detach().clone(), grid_thw.detach().clone())
        return self.tower(images, grid_thw)["last_hidden_state"]
    if not self.slots:
        self._capture(images, grid_thw)
    if self.mode == "forward":
        return self.slots[0].forward_only(images, grid_thw)
    for slot in self.slots:
        if not slot.pending:
            return slot.forward_backward(images, grid_thw)
    raise RuntimeError(f"VE CUDA Graph has more than {self.max_inflight} outstanding forwards")
```

`images` 是本次的视觉 patch token，`grid_thw` 描述对应图像或视频的网格布局。理解这段代码时，始终把“本次真实输入”和“保存下来用于建图的模板”分开。

首先，若要求延迟捕获且图未就绪时，会保存捕获的模版并直接调用 `self.tower` 进行普通前向，此时 manager 不会自行触发捕获，共享图池路径需要外层安排 ve、ae、llm 等图的捕获顺序，所以这里线执行普通前向，等待合适的建图时机。所谓的模版，其实就是保存一份当前输入，通过 detach 切断保存的模版与原输入的 autograd 关系，剥离出计算图；通过 clone 为模版分配独立存储，原来的输入被修改或复用后，不会改变保存的模版。

然后是 `forward()` 函数中最重要的行为，即在需要使用图但还没有任何 graph slot 时，先进行图捕获，会调用 ve cuda graph manager 的另一个核心功能函数 `_capture()` 来完成这个事，这个函数会在后续的小节展开说，这里只需要提一点的是，该核心功能函数完成了捕获前向图、反向图以及捕获完成后的后处理所有操作，并且执行完后并不会直接 return，后续才会调用 graph slot 执行本轮正式的前向 replay。

`forward()` 中正式执行 replay 的详细执行步骤在此前的 graph slot 中有提到过，这里是更外层的处理，即完整的调用链是 ve cuda graph manager 调用 graph slot 的前向或前反向计算功能函数，再由功能函数内部进行 replay。ve cuda graph manager 根据 `mode` 来决定调用 graph slot 的哪个功能函数，`mode` 为 `forward` 时调用 `slot.forward_only()`，为 `forward_backward` 时调用 `slot.forward_backward()`。需要注意的是，这里进行前反向 replay 时会对 graph slot 的 `pending` 的状态进行检查，之前有提到过，执行完前向但没有执行反向的 graph slot 的 `pending` 会被标记成 true，该 graph slot 将不被允许被用作另一次 replay，直到完成了反向图的 replay，原因是另一次的 replay 的前向图输出会与前一次 replay 的前向图的输出踩踏，影响前一次 replay 的反向计算。

目前部分框架虽然开放了 `max_inflight` 给用户设置，但是普遍设置为 1，因此 `self.slots` 列表中其实只包含一个 graph slot，在训练遍历前者并找到 `pending` 标记符合要求的 grpah slot 时，实际也只会检查一个 graph slot，目前的写法并不代表当前已经支持同时存在多个未完成的 ve 前向。最后如果没有找到可用的 graph slot 时会直接报错，比如所有的 graph slot 的  `pending` 标记均为 true。

到目前为止，整个一次正常训练的调用已经可以串起来了。这个流程中，每次正式调用使用的都是本次的实际数据。图保存运算及固定地址，模板帮助完成建图；在已支持的容量和输入规格内，真实 token 数及布局的变化由 slot.prepare() 和固定容量 wrapper 处理：

```text
manager.forward(images, grid_thw)
    → 必要时先完成直接捕获；已就绪则复用现有图
    → 选择空闲 slot
    → slot.prepare() 写入本次真实输入和布局
    → 前向 replay，pending=True
    → 返回 VE 特征，外层继续 adapter、LLM 和 loss 计算
    → 外层安排必要的梯度缓冲区初始化，并进行相应反向
    → 梯度到达 VE 输出，触发自定义 backward()
    → 反向 replay，退出时 pending=False
```

## 5.3 _capture()

`_capture()` 函数负责把几个已经拆开的建图阶段按顺序执行。它接收用于本次建图的 `images` 和 `grid_thw`，最终让 manager 持有可用的 graph slot 和 CUDA Graph。它的返回类型是 None，不负责返回本轮训练的 ve 特征：

```python
def _capture(self, images: torch.Tensor, grid_thw: torch.Tensor) -> None:
    self.prepare_capture(images, grid_thw)
    self.capture_forward()
    if self.mode == "forward_backward":
        self.capture_backward()
    self.finish_capture()
```

结构非常清晰，首先是 capture graph 之前的准备工作，通过 ve cuda graph manager 的功能函数 `prepare_capture()` 完成：

```python
def prepare_capture(self, images: torch.Tensor, grid_thw: torch.Tensor) -> None:
    if self.slots:
        raise RuntimeError("VE CUDA Graph slots are already prepared")
    slot_count = self.max_inflight if self.mode == "forward_backward" else 1
    for _ in range(slot_count):
        fixed_ve = _FixedCapacityVE(
            self.tower, self.max_tokens, self.max_segments, images, grid_thw
        )
        self.slots.append(
            _GraphSlot(
                fixed_ve,
                images,
                self.zero_grad_buffer,
                self.no_sync,
                capture_backward=self.mode == "forward_backward",
                stream=self.capture_stream,
                pool=self.pool,
                defer_capture=True,
            )
        )
```

准备环节会首先检查是否需要进行 capture，只会在 `self.slots` 为空时允许。`slot_count` 在仅需进行前向时被硬编码成 1，在需要进行前反向计算时会按照 `max_inflight` 取值，目前框架通常也指定为 1。然后按照 `slot_count` 进行定长 ve 的实例化以及 graph slot 的实例化，两者是一对一的，每个 graph slot 会维护一个定长 ve，并最终由 `self.slots` 维护所有的 graph slot。这里实际涉及的准备工作很多，主要是在实例化定长 ve 和实例化 graph slot 时，会涉及到包括固定缓冲区的准备、warmup 以及准备好定长计算对象等。

准备工作之后，ve cuda graph manager 会调用第二个功能函数 `capture_forward()` 进行前向图的捕获，其最终调用层级是走到了 `slot.capture_forward_graph()`，后者在此前已经讲解过了，内部在指定 stream 和 pool 上执行了定长 ve 的前向，并完成了前向图的 capture。

```python
def capture_forward(self, before_capture: Callable[[], None] | None = None) -> None:
    for index, slot in enumerate(self.slots):
        slot.capture_forward_graph(
            self.pool, before_capture=before_capture if index == 0 else None
        )
```

然后是 `capture_backward()` 函数，其最终调用的也是 slot graph 的 `capture_backward_graph()`，在指定 stream 和 pool 上完成了反向图的捕获。**与前向图类似，捕获操作会遍历** `slots` **中所有 graph slot 并全部执行**。

```python
def capture_backward(self, before_capture: Callable[[], None] | None = None) -> None:
    if self.mode != "forward_backward":
        raise RuntimeError("forward-only VE manager has no backward graph")
    for index, slot in enumerate(self.slots):
        pool = self.pool if self.pool is not None else slot.forward_graph.pool()
        slot.capture_backward_graph(pool, before_capture=before_capture if index == 0 else None)
```

流程的最后一步是 `finish_capture()` 函数，其也是遍历所有 graph slot 并调用后者的 `finish_capture()` 函数，一个额外的操作是将标记参数 `graphs_ready` 设置为 true，表明本 ve cuda graph manager 已经完成了图捕获行为：

```python
def finish_capture(self) -> None:
    for slot in self.slots:
        slot.finish_capture()
    self.graphs_ready = True
```

最后要补充说明的是，ve cuda graph manager 还有一个 `reset()` 函数，将当前 manager 的图状态重置：

```python
def reset(self) -> None:
    if any(slot.pending for slot in self.slots):
        raise RuntimeError("cannot reset VE CUDA Graph with an outstanding forward")
    for slot in self.slots:
        slot.reset()
    self.slots.clear()
    if self.grad_init_graph is not None:
        self.grad_init_graph.reset()
        self.grad_init_graph = None
    self.graphs_ready = False
```

其会将 ve cuda graph manager 的标记 `graphs_ready` 重置为 false，并遍历每一个 graph slot 进行图重置：

```python
def reset(self) -> None:
    if hasattr(self, "forward_graph"):
        self.forward_graph.reset()
    if hasattr(self, "backward_graph"):
        self.backward_graph.reset()
    self.fa_saved_outputs.clear()
```

# 6. 共享显存链路

一些多模态训练框架会让 VE、AE 和 LLM 的 CUDA Graph 共用显存池，以复用不同计算阶段的临时存储。本节沿用前文的 manager 和 graph slot，以一组实现片段为例，讲解捕获编排、梯度缓冲区和调度器怎样配合完成复用。下面的函数名、配置和调用顺序都属于这个示例，阅读时关注各部分的职责与协作关系即可。

## 6.1 共享显存的目的与捕获入口

前面第五节的 `_capture()` 会连续完成 VE 前向和反向图的捕获。这里的共享显存方案会在两者之间插入 LLM 的捕获，使显存分配器能够按照各阶段的存储生命周期安排复用，因此通过 `defer_capture=True` 将捕获时机交给外层。**共享的是 CUDA Graph 显存池中可以复用的存储，各模型仍持有自己的权重和图。**

以下示例以需要训练 VE、并启用梯度缓冲区重建的路径为主，对应配置为 `ve_cuda_graph.mode=forward_backward` 和 `distributed.ddp.rebuild_grad_buffer=True`。示例中的 `_validate_args()` 还要求配合 TE（Transformer Engine）的 `full_layer` 或 `vpp_stage` CUDA Graph、pipeline executor、至少一次训练 warmup，以及 `PYTORCH_CUDA_ALLOC_CONF` 中的 `expandable_segments:True`；这些是该实现的配置约束。在同一示例中，冻结 VE 的 `forward` 模式也可以共享 LLM 图池，但不捕获 VE 的梯度初始化图和反向图，adapter 仍通过普通 autograd 训练。

有些框架会由 trainer 统一协调 encoder 与 LLM 的捕获时机。例如，下面的 trainer 代码通过两个回调，把 encoder 的捕获接入 LLM 的建图流程；片段省略了外层轮次判断及其他训练逻辑：

```python
if use_ve_shared_graph_pool:
    ve_llm_graph_pool = cuda_graph_helper.new_shared_pool()
    with self._model_main.capture_encoder_cuda_graphs(ve_llm_graph_pool) as (
        start_ve_capture,
        finish_ve_capture,
    ):
        cuda_graph_helper.create_cudagraphs(
            before_model_capture=start_ve_capture,
            before_optimizer_capture=finish_ve_capture,
        )
```

`new_shared_pool()` 创建本次捕获共用的 pool 句柄；pool 不是预先指定的显存容量。模型侧的 `capture_encoder_cuda_graphs()` 是上下文管理器，它把两个回调交给负责 LLM 的 TE graph manager，分别在 LLM 捕获之前和之后执行。示例中的 `create_cudagraphs()` 按以下顺序调用，若配置了 optimizer graph，则最后才捕获 optimizer：

```text
start_ve_capture()
    → LLM 前向、反向图捕获
    → 清理对应 scope 已不再需要的捕获缓存
    → finish_ve_capture()
    → optimizer graph 捕获（若启用）
```

这里的两个回调是外层捕获流程的 Python 调用，不是前面 graph slot 中的 `before_capture` 参数，也不会因为之后调用某张图的 `replay()` 而重新执行。LLM 部分内部仍遵循它自己的 microbatch 和 pipeline 捕获顺序，本节只关注它与 encoder 的边界。

## 6.2 统一 pool 和 stream，再捕获 encoder 前向

模型侧的 `capture_encoder_cuda_graphs()` 首先收集已有且设置了 `defer_capture` 的 encoder manager，记作 `participants`，再把其中 `mode=forward_backward` 的成员放入 `backward_participants`。示例按 VE 在前、AE 在后的顺序组织成员；只有 VE 时就是一个成员。没有参与者时直接交回两个 None 回调，有参与者但没有共享 pool 时则报错。

下面是这个上下文管理器内部的第一个完整回调：

```python
def start_ve_capture():
    nonlocal started
    if started:
        return
    torch.cuda.synchronize()
    if backward_participants:
        self._zero_encoder_graph_optimizer_grads(
            wrapper for wrapper, _ in backward_participants
        )
    # Prepare every encoder before capturing any new graph. Preparation
    # may reset old graphs, which can clear stream-local BLAS caches.
    for _, manager in participants:
        manager.prepare_shared_capture_inputs(pool)
    # Existing execution order: VE F -> AE F, then LLM F/B.
    for _, manager in participants:
        manager.capture_forward()
    for wrapper, _ in backward_participants:
        release_model_grad_buffer(wrapper, force=True)
    started = True
```

这个回调先等待此前 GPU 工作完成，并清除相关 optimizer 持有的梯度引用，再准备所有 encoder。`_zero_encoder_graph_optimizer_grads()` 调用对应 optimizer 的 `zero_grad(set_to_none=True)`；示例将这条共享重建路径限定为 encoder 使用独立的 Adam optimizer，不能与 LLM 共用同一个叶子 optimizer，以便单独管理 encoder 的梯度引用。

准备工作与前向捕获被拆成两个循环，是因为准备时可能重置旧图并清理 stream 上的 BLAS 缓存。先把所有准备完成，才能避免准备第二个 encoder 时影响已经捕获好的第一个 encoder。VE manager 的准备函数如下：

```python
def prepare_shared_capture_inputs(self, pool) -> None:
    if not self.defer_capture:
        raise RuntimeError("VE CUDA Graph manager is not configured for shared capture")
    self.ensure_capture_template()
    self.reset()
    self.pool = pool
    # TE uses torch.cuda.graph without an explicit stream. Constructing the
    # same context (without entering capture) selects its process-wide default
    # side stream, so allocator blocks are reusable across VE and TE captures.
    capture_context = torch.cuda.graph(torch.cuda.CUDAGraph(), pool=pool)
    self.capture_stream = capture_context.capture_stream
    images, grid_thw = self._capture_template
    self.prepare_capture(images, grid_thw)
```

`ensure_capture_template()` 保证存在建图输入：通常沿用延迟捕获期间保存的输入，若该 rank 的 encoder 在 warmup 时一直没有数据，则构造最小样例。随后 `reset()` 清理旧图状态，但保留模板；`prepare_capture()` 继续执行第五节介绍的 wrapper、slot 创建和 warmup。

这里除了统一 `pool`，还专门选择了与 LLM 一致的捕获 stream。示例中的 TE graph manager 使用未显式指定 stream 的 `torch.cuda.graph()`，因此 VE 也先构造相同的上下文对象，读取其默认捕获 stream。**这里只创建上下文对象，没有进入** `with`**，所以还没有开始捕获。** 后续 VE 的前向、梯度初始化和反向捕获都会使用这个 stream，便于分配器跨 VE、LLM 捕获复用显存块；正式 replay 仍在调用时的当前 stream 上提交，不由这个属性指定。

所有准备完成后，回调才依次捕获 VE、AE 前向，并对需要反向的 encoder wrapper 强制释放梯度缓冲区。此时还没有本轮正式训练的 encoder 梯度需要保留；把梯度存储的重新分配推迟到 LLM 捕获之后，才能让它有机会复用 LLM 阶段结束后可用的图池存储。

## 6.3 LLM 捕获之后，先初始化梯度再捕获 encoder 反向

LLM 图捕获完成后，TE graph manager 调用第二个回调 `finish_ve_capture()`。示例的 `vpp_stage` 路径还会先释放部分 MoE 通信管理器的捕获缓存引用，为后续梯度缓冲区重建提供复用机会。这个回调仍定义在模型侧的 `capture_encoder_cuda_graphs()` 内部：

```python
def finish_ve_capture():
    nonlocal finished
    if finished:
        return
    start_ve_capture()
    for wrapper, manager in backward_participants:
        manager.capture_grad_init(
            lambda wrapper=wrapper: rebuild_model_grad_buffer(wrapper, force=True)
        )
    # Existing joint backward executes AE before VE.
    for _, manager in reversed(backward_participants):
        manager.capture_backward()
    for _, manager in participants:
        manager.finish_capture()
    if backward_participants:
        self._set_cuda_graph_static_grad_buffers(True)
    for wrapper, manager in backward_participants:
        wrapper._cuda_graph_prepare_grad_buffer = manager.replay_grad_init
    finished = True
    logger.info("[encoder_cuda_graph] coordinated capture complete")
```

`started` 和 `finished` 防止两个回调重复执行，`finish_ve_capture()` 开头再调用一次 `start_ve_capture()` 是为了保证前向准备已经完成。接着，代码先为所有需要训练的 encoder 捕获梯度初始化图，再按 AE、VE 的逆序捕获反向图。只有 VE 时，核心顺序就是 VE 前向 → LLM 前反向 → VE 梯度初始化 → VE 反向。

独立的梯度初始化图由 `VECudaGraphManager.capture_grad_init()` 创建：

```python
def capture_grad_init(self, rebuild_grad_buffer: Callable[[], None]) -> None:
    """Allocate/bind once; replay the captured zero before eager VA backward."""
    if not self.defer_capture or len(self.slots) != 1:
        raise RuntimeError("VE grad-init graph requires shared capture with one slot")
    if self.grad_init_graph is not None:
        raise RuntimeError("VE grad-init graph is already captured")
    self.grad_init_graph = torch.cuda.CUDAGraph()
    with (
        pinned_memory_capture_lock,
        torch.cuda.graph(self.grad_init_graph, pool=self.pool, stream=self.capture_stream),
    ):
        rebuild_grad_buffer()
    self.slots[0].requires_grad_init = True
```

传入的 `rebuild_grad_buffer` 是绑定了当前 DDP wrapper 的回调。因为上一阶段已释放旧的梯度缓冲区，这次在图捕获上下文中执行重建时，会从共享 pool 分配新存储，绑定各参数的 `main_grad`，并把清零操作记录到 `grad_init_graph`。随后 VE 反向捕获使用的就是这次绑定的地址。`requires_grad_init=True` 则告诉第四节的 autograd bridge：每轮反向开始前，必须已经完成梯度初始化。

最后，各 manager 按第五节的逻辑完成 `finish_capture()`，并标记图已就绪。外层同时启用梯度缓冲区的静态地址保护，把 `manager.replay_grad_init` 挂到 wrapper 的 `_cuda_graph_prepare_grad_buffer` 上，供调度器在正式训练中调用。初始化图与反向图分开，是为了让清零发生在整个视觉分支反向之前，而不是等到 VE 反向图即将 replay 时才清零。

## 6.4 捕获阶段怎样释放、重建和绑定梯度缓冲区

本节展开前两小节已经调用的梯度管理接口。先看释放端：6.2 的 `start_ve_capture()` 在 **encoder 前向图捕获完成之后、LLM 图捕获开始之前**，调用 `release_model_grad_buffer(wrapper, force=True)`。这个模型级入口检查 DDP 类型和静态保护状态，然后遍历普通参数与 expert parallel 参数的梯度缓冲区，逐个调用下面的 `release_grad_buffer()`：

```text
start_ve_capture()
    → release_model_grad_buffer(wrapper, force=True)
    → 遍历 wrapper 中的梯度 buffer
    → release_grad_buffer(wrapper, buffer)
```

因此，`release_grad_buffer(model, buffer)` 中的 `model` 就是上面的 DDP `wrapper`，`buffer` 是它管理的某一个梯度缓冲区；这个函数只处理单个 buffer。调用方的 `force=True` 使本次建图时的释放绕过静态保护，保护机制在 6.5 展开。

`buffer.grad_data` 持有连续的梯度存储，`bucket.grad_data` 和参数的 `main_grad` 是其中不同范围的视图。在这个示例中，视觉模型的 wrapper 同时包含 VE tower 和 adapter，因此需要一起管理它们的梯度存储。单个缓冲区的释放实现如下：

```python
def release_grad_buffer(model, buffer):
    if buffer.grad_data is None:
        return
    # release grad buffer
    buffer.grad_data = None
    cur_bucket_id = -1
    for param in buffer.param_index_map.keys():
        data_start_index, data_end_index, bucket_id = buffer.param_index_map[param]
        if bucket_id != cur_bucket_id:
            assert bucket_id == (cur_bucket_id + 1)
            bucket = buffer.buckets[bucket_id]
            # detach bucket view reference to grad_data
            bucket.grad_data = None
            cur_bucket_id = bucket_id
        # detach main_grad view reference to grad_data
        param.main_grad = None
    _refresh_grouped_main_grads(model, buffer)
```

释放时不能只把 `buffer.grad_data` 设成 None，因为 bucket 和参数的视图也持有底层存储。这里依次解除这些引用，再刷新 TE grouped weight 缓存的梯度视图，使它们跟随本次释放。它没有把梯度搬到 CPU，也没有释放模型权重；存储能否被分配器回收，取决于相关引用是否都已解除。

对应的重建入口在 6.3 的 `finish_ve_capture()` 中：此时 **LLM 前反向图已捕获，encoder 反向图尚未捕获**。它把重建回调传给 `manager.capture_grad_init()`，由后者在指定共享 pool 的捕获上下文中执行：

```text
finish_ve_capture()
    → manager.capture_grad_init(重建回调)
    → 在 torch.cuda.graph(...) 内调用回调
    → rebuild_model_grad_buffer(wrapper, force=True)
    → 遍历 wrapper 中的梯度 buffer
    → rebuild_grad_buffer(wrapper, buffer)
```

这里有两个同名位置需要区分：`capture_grad_init()` 的形参 `rebuild_grad_buffer` 保存的是前文传入的无参 lambda 回调；它通过模型级入口 `rebuild_model_grad_buffer()`，才调用到下面接收 `model, buffer` 两个参数的具体函数。这个函数先分配并清零连续存储，再按原来的参数和 bucket 索引恢复视图：

```python
def rebuild_grad_buffer(model, buffer):
    if buffer.grad_data is not None:
        return
    assert buffer.grad_data is None, "grad data is not none"
    # rebuild grad_data
    buffer.grad_data = torch.empty(
        buffer.numel,
        dtype=buffer.grad_dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )
    buffer.grad_data.zero_()
    cur_bucket_id = -1
    for param in buffer.param_index_map.keys():
        data_start_index, data_end_index, bucket_id = buffer.param_index_map[param]
        if bucket_id != cur_bucket_id:
            assert bucket_id == (cur_bucket_id + 1)
            bucket = buffer.buckets[bucket_id]
            bucket_start_index, bucket_end_index = buffer.bucket_indices[bucket_id]
            # remap main_grad of the param
            bucketed_grad_data = buffer._get(
                torch.Size([bucket_end_index - bucket_start_index]),
                bucket_start_index,
                buffer_type=BufferType.GRAD,
            )
            bucket.grad_data = bucketed_grad_data
            cur_bucket_id = bucket_id
        # remap main_grad of the param
        param.main_grad = buffer._get(
            param.data.shape, data_start_index, buffer_type=BufferType.GRAD
        )
    _refresh_grouped_main_grads(model, buffer)
```

`buffer._get()` 从连续存储中取出对应区间并恢复所需 shape，不为每个参数另复制一份梯度。`_refresh_grouped_main_grads()` 再把 TE 内部缓存的视图绑定到新存储，使后续 fused 算子也能写到正确的 `main_grad`。

**在这条共享捕获路径中，释放旧存储、分配新 Tensor 和绑定视图都属于建图阶段；正式 replay 不会重新运行这些 Python 代码。** 初始化图重放的是已经记录的 GPU 清零操作，反向图重放的是向已绑定地址计算、累加梯度的 GPU 操作。因此，两张图建立以后，不能再随意释放这块存储或把 `main_grad` 换到别的地址；6.5 继续说明如何保护这些地址，并在每轮需要时清零。

共享显存的关键也在这里：捕获 LLM 时，它的一些临时存储在自己的前反向结束后已可复用；随后分配的 encoder 梯度缓冲区可能落在这些地址上。正式运行时，LLM 图仍可以先向那些地址写入临时数据，之后再由初始化图清零，交给 encoder 反向使用。并不是图池中的全部张量都能这样重叠，例如 VE 前向保存、反向仍要读取的激活必须跨过 LLM 阶段保持有效；仅仅传入同一个 pool，并不代表所有显存都会自动省下来。

## 6.5 正式运行时保持地址固定，在反向边界 replay 清零

为了在捕获完成后固定梯度地址，示例通过 `_set_cuda_graph_static_grad_buffers(True)` 给 LLM 的 DDP chunks 和参与共享反向捕获的 encoder DDP wrapper 设置 `_cuda_graph_static_grad_buffer=True`。`release_model_grad_buffer()` 和 `rebuild_model_grad_buffer()` 都检查这个标记；下面节选前者通过模型类型检查之后的部分，后者使用相同的静态保护条件：

```python
if getattr(model, "_cuda_graph_static_grad_buffer", False) and not force:
    return
for buffer in model.buffers:
    release_grad_buffer(model, buffer)
if hasattr(model, "expert_parallel_buffers"):
    for buffer in model.expert_parallel_buffers:
        release_grad_buffer(model, buffer)
```

因此，捕获阶段传入 `force=True` 可以真正释放和重建缓冲区；进入正式训练后，普通 release、rebuild 调用会直接返回，不再改变图依赖的地址。这个保护同时覆盖普通参数和 expert parallel 参数的缓冲区。Tensor 引用被保留下来，并不会取消捕获时已经形成的地址复用关系。

需要在反向前清零时，调度器调用的是专门的入口 `prepare_model_grad_buffer_for_backward()`：

```python
def prepare_model_grad_buffer_for_backward(model):
    """Scheduler boundary: rebuild eager buffers or zero static graph buffers."""
    if isinstance(model, list):
        for m in model:
            prepare_model_grad_buffer_for_backward(m)
        return
    prepare = getattr(model, "_cuda_graph_prepare_grad_buffer", None)
    if getattr(model, "_cuda_graph_static_grad_buffer", False) and prepare is not None:
        prepare()
    else:
        rebuild_model_grad_buffer(model)
```

普通模型继续走原来的重建路径；具有静态标记且挂有准备回调的 encoder，则调用前面安装的 `manager.replay_grad_init()`。LLM chunk 虽然也会被标记为静态，但这里没有为它安装 encoder 的准备回调，不能把静态标记理解成“所有模型都调用 VE 的初始化图”。

```python
def replay_grad_init(self) -> None:
    if not self.graphs_ready or self.grad_init_graph is None:
        raise RuntimeError("VE grad-init graph is not ready")
    slot = self.slots[0]
    if slot.pending and slot.grad_initialized:
        raise RuntimeError("VE grad-init graph was already replayed for this forward")
    # Even an image-free iteration must discard LLM data at aliased grad addresses.
    self.grad_init_graph.replay()
    slot.grad_initialized = slot.pending
```

清零后，如果该 slot 有等待反向的前向，才把 `grad_initialized` 置为 True。同一次待完成的前向不允许重复初始化，否则可能擦掉已累加的梯度；反向 bridge 会检查这个标记，并在退出反向时复位。即使本轮没有图像、`pending=False`，也仍然需要清零，因为这些梯度地址可能已经被 LLM 写入临时数据，不能把残留内容当成本轮梯度。

这里清理的是参数的梯度缓冲区，与第四节的 `slot.grad_output` 不同：后者存放下游传给 VE 输出的梯度，在 VE 反向 replay 前复制；前者用于累加 VE 和 adapter 的参数梯度，必须在它们开始写入之前初始化。

## 6.6 调度顺序怎样保证共享存储不会被误用

有些框架会把梯度缓冲区的处理边界作为显式操作插入 pipeline 调度计划。例如，调度器中的 `add_encoder_grad_buffer_ops()` 在满足“encoder 前向全部位于 LLM 之前、encoder 反向全部位于 LLM 之后”的计划中，插入 `VEGRADBUFFERRELEASE` 和 `VEGRADBUFFERREBUILD`。模型侧的 `validate_ve_cuda_graph_shared_schedule()` 再校验共享训练路径恰好各有一个这样的操作，并检查它们与所有 encoder、LLM 计算的位置关系，不符合时直接报错。

调度器负责安排操作的位置，pipeline executor 负责执行。下面是示例中 rebuild 调度操作的完整处理函数，普通 pipeline 和 virtual pipeline 的 executor 都采用这个逻辑：

```python
def encoder_grad_buffer_rebuild_action(self, op: ScheduleOp):
    if not self.rebuild_grad_buffer:
        return
    if not self.encoder_models:
        return
    prepare_model_grad_buffer_for_backward(self.encoder_models)
```

虽然调度操作的名字仍然叫 rebuild，图已就绪时实际执行的是上一小节的初始化图 replay。release 操作仍调用 `release_model_grad_buffer()`，但对静态缓冲区不会真正释放。省略通信等无关操作后，这个示例在正式训练中的视觉分支沿以下路径执行：

```text
VE 前向 replay，随后完成 adapter 前向
    → VEGRADBUFFERRELEASE：静态缓冲区保持绑定
    → LLM 前向、反向：可能改写与 encoder 梯度存储重叠的地址
    → VEGRADBUFFERREBUILD
        → prepare_model_grad_buffer_for_backward()
        → manager.replay_grad_init()：清零参数梯度缓冲区
    → VEBACKWARD：进入视觉分支的 autograd 反向
        → adapter 反向，累加 adapter 参数梯度
        → VE 的 _ForwardBackwardReplay.backward()
        → 检查初始化状态，replay VE 反向图，累加 VE 参数梯度
    → 后续梯度同步与 optimizer 处理
```

这个边界必须放在 **adapter 反向之前**。adapter 的梯度也写入该 DDP wrapper 的缓冲区，如果把清零放到 VE 反向图内部，执行到那里时 adapter 已经完成反向，它的梯度就会被清掉。反过来，只在 iter 开头清零也不够，因为后面的 LLM 计算可能再次向重叠地址写入数据。

复用要求 GPU 上的实际读写顺序遵守这些生命周期约束，不能因为有多个 graph 对象就任意并发 replay；CPU 上设置 `pending`、`grad_initialized` 只是在检查流程状态，GPU 工作之间的先后关系仍由 stream 和相应的同步依赖保证。VE 前向需要保留给反向的数据、LLM 尚未消费完的数据，都不能提前让给其他阶段覆盖。

捕获失败时，`capture_encoder_cuda_graphs()` 的异常分支会关闭静态标记并移除准备回调，重置参与的 encoder manager，再强制重建其梯度缓冲区，最后继续抛出异常；正常退出但尚未调用完成回调时，则补调 `finish_ve_capture()`。这些是捕获过程的收尾逻辑，不是每轮 replay 都会执行的操作。

# 7. ve cuda graph 兼容 ve 选择重计算

# 8. attention 模块与 MoE 算子

# 9. Q&A

