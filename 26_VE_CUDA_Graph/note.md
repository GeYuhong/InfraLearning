
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

`forward()` 函数中剩下的处理则是服务于反向函数 `backward()` 和状态记录，即设置当前使用的 graph slot 的 `pending` 为 true，并将 graph slot 和 真实的 token 数都保存给反向使用，确保反向处理时知道梯度应该写入固定显存缓冲区的哪些位置。后续自定义的 `backward()` 函数在处理反向后，会将状态标记 `pending` 恢复成 false。需要注意的一点是，**前向函数最终返回结果时，返回的是真实 token 对应的输出，去掉了 padding 部分**。对返回的结果进行 `clone()` 也是必要的，因为当前使用的 graph slot 有可能后续被再次使用，并且被使用时，此前已有的输出结果可能还未被完全使用完成（例如多次 ve forward 前向完成后，再将多次 forward 的结果一起传给 llm backbone 部分处理），此时再次调用 `forward()` 会覆盖已有的输出，因此**有必要通过 `clone()` 复制成独立的 tensor**。

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

## 5.3 _capture()

## 5.4 prepaer_capture()

## 5.5 captue_forward()

## 5.6 capture_backward()

## 5.7 finish_capture()

## 5.8 reset()



读 VECudaGraphManager，把完整生命周期穿起来
先看 __init__() 和 forward()，然后沿它们的调用读 prepaer_capture()、captue_forward()、capture_backward()、finish_capture()。此时将前面读过的组件拼起来：manager 决定捕获时机，slot 持有图和缓冲区，固有容量 wrapper 执行 ve，autograd bridge 连续训练反向。共性显存相关的 prepare_shared_capture_inputs()、capture_grad_init()、replay_grad_init()留到下一步一起读。


# 6. 共享显存链路

读共享显存链路，理解捕获顺序和梯度缓冲区生命周期。
按下面的顺序跨文件阅读：
* model_hybrid_vlm.py 的 capture_encoder_cuda_graphs()（model_hybrid_vlm.py:598）：先掌握 ve、llm 和梯度初始化图的捕获顺序。
* 回到 manager 的三个共享捕获/梯度初始化函数，理解同一个 pool、capture stream和grad-init图如何建立。
* common/release_grad_util.py:179：看release、rebuild以及prepaer_model_grad_buffer_for_backward()。
这一步最重要的是区分：捕获时真正分配和绑定梯度缓冲区；正式运行时保持地址固定，在指定边界 replay 清零。
接着看 




# 7. ve cuda graph 兼容 ve 选择重计算


# 8. attention 模块与 MoE 算子