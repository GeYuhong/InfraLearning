# 1. 捕获边界与入口

音频编码器（AE）的输入长度随音频时长变化，而 CUDA Graph 重复执行时需要保持已记录的计算结构和显存地址。音频预处理与卷积可能涉及按长度分桶，各桶的样本数也会变化；后续 Transformer 则可以通过补齐 token 和分段信息，形成固定容量的输入。因此，一些框架保留预处理与卷积的普通执行路径，只把中间的 Transformer 捕获成图。下面以这种实现为例，说明变长适配、前反向衔接和共享显存。

本文的 capture（捕获）指沿计算流程运行 Python 代码，将其中提交的 GPU 操作记录成图；replay（重放）指执行这些已记录的 GPU 操作。replay 不会重新执行捕获时的 Python 函数。

## 1.1 图接管哪一段计算

先明确一次音频前向的数据流。下面是流程示意，不是源码：

```text
波形 → 分段与特征提取 → 卷积 stem、conv_out
     → packing 与 RoPE 准备
     → [Transformer layers → 最终 layer norm]  ← AE 图的边界
     → 输出整理与 token merge → audio adapter → LLM
```

这里的 packing 表示取出不同音频段的有效 token，沿第一维拼接成 `[tokens, hidden_size]` 的二维张量；这种表示称为 packed，段与段的边界由额外的长度信息记录。RoPE 准备则是按各段的位置生成旋转位置编码所需的 `cos`、`sin` 系数；Transformer 内部对 Q/K 应用这些系数的计算仍在图中。

图覆盖所有 Transformer layers 和最终 layer norm，整体捕获为一张前向图，训练模式下再对应一张反向图。卷积 stem 是前端卷积模块，`conv_out` 把卷积输出投影到 Transformer 的特征维度；二者在图前执行。图后的 token merge 合并时间方向上的 token，audio adapter 再将特征映射到 LLM 所需的表示。它们以及 encoder 的跨 rank 数据分发都在这两张图之外。

卷积输出准备好后，音频模型调用 `_dispatch_transformer()`，参考示例如下所示，省略了函数参数列表和后面的普通 Transformer 调用：

```python
manager = getattr(self, "_ae_cuda_graph_manager", None)
if manager is not None:
    if not return_packed or cu_seqlens_q is None:
        raise ValueError("AE CUDA Graph requires packed input and output")
    if output_attentions or output_hidden_states:
        raise ValueError("AE CUDA Graph does not support attention/hidden-state outputs")
    return manager.forward(hidden_states, (lengths, rotary_cos, rotary_sin)), None, None
# 未安装 manager 时，继续调用原来的 _forward_transformer(...)。
```

传给 `manager.forward()` 的 `hidden_states` 只包含真实 token，manager 内部才会补到图的固定容量，返回时再去掉尾部 padding。`return_packed=True` 要求返回拼接形式的有效 token，不表示返回固定容量的 padding；`cu_seqlens_q` 则提供各段的累计边界。

第二个检查禁止额外返回各层的中间特征和 attention 结果，最终一层的特征仍正常返回。返回值末尾的两个 `None` 对应这两类未提供的附加结果。这样，下游仍接收原来的有效 token 表示，变长处理集中在图的入口和出口。

## 1.2 manager 在模型装配时创建

模型管理层先构建音频模型、设置训练参数并完成 DDP 包装，再调用 `_configure_ae_cuda_graph(audio_model)`。DDP 负责数据并行训练中的参数梯度管理与同步，`audio_model` 是经过 DDP 包装的音频分支，包含编码器与 adapter。因此创建 manager 时，已经可以取得 DDP 的梯度清理方法 `zero_grad_buffer` 和暂时关闭梯度同步的上下文 `no_sync`，但此时还没有捕获图。

这个函数从原音频模型中取出 `speech_encoder`，在代码中命名为 `tower`。它是一个 `WhisperEncoder` 对象，持有前端卷积、Transformer 层列表 `layers` 和最终归一化层 `layer_norm`。函数把 `tower` 传给 `AECudaGraphManager`，创建 manager 后再执行：

```python
tower._ae_cuda_graph_manager = manager
```

这次赋值使 1.1 中的 `_dispatch_transformer()` 能找到 manager；manager 也保存着 `tower`，以便调用原 Transformer。二者互相保存对象引用，不会重新构造彼此，也不复制模型权重。后面的 `fixed_ae` 和 slot 要等到准备捕获时才创建。

示例有三种模式：`off` 不安装 manager；`forward` 要求冻结整个音频编码器，并训练图后的 audio adapter；`forward_backward` 捕获 Transformer 的前向和反向，且要求 Transformer 中存在可训练参数。后者还需要把输入梯度传回图外卷积，这一点将在第四节展开。

# 2. 变长音频如何放入固定容量

## 2.1 token、segment 与静态缓冲区

manager 用 `_FixedCapacityAE` 包装原来的 Transformer，创建的对象称为 `fixed_ae`。设固定容量为 `T = max_tokens`、`S = max_segments`，每次真实输入有 `t` 个 token、`s` 个 segment。一个 segment 是一段独立做 attention 的序列；一条音频经过 chunk 拆分后可能产生多个 segment。

**这里的 token 数是输出 merge 前的 Transformer 行数。** 它既不是波形采样点数，也不能直接用最终送入 LLM 的 audio token 数代替。

容量按**单个执行 AE 的 rank、单次 Transformer 调用**计算。rank 是训练进程，通常对应一张 GPU；如果数据经过负载均衡和重新分发，应以该 rank 最终送入 Transformer 的 token 总数和 segment 总数为准。框架可能把多批音频合并后执行一次 AE 前向，这就是这里所说的“融合调用”，此时容量必须容纳合并后的全部输入。

例如，同一张卡上两批音频经过卷积后，分别产生 `3000 token、10 segment` 和 `4000 token、12 segment`。若融合成一次调用，就需要 `max_tokens ≥ 7000`、`max_segments ≥ 22`；若分两次调用，同一缓冲区可以先后复用，这个例子中容量达到 `4000 token、12 segment` 即可分别容纳两批输入。

`fixed_ae` 创建固定长度的 RoPE 系数、分段边界和有效位掩码；存放 token 特征的静态输入由第三节的 slot 持有。这里的“静态”指 shape 和存储地址固定，内容可以在每次 replay 前更新。下面节选这些缓冲区的创建，省略容量与 RoPE 检查：

```python
self.rotary_cos = cos.new_ones((self.max_tokens, *cos.shape[1:]))
self.rotary_sin = sin.new_zeros((self.max_tokens, *sin.shape[1:]))
self.cu_seqlens = torch.full(
    (self.max_segments + 2,), self.max_tokens, device=hidden.device, dtype=torch.int32
)
self.cu_seqlens_cpu = torch.full(
    (self.max_segments + 2,), self.max_tokens, dtype=torch.int32
)
self.valid_tokens = torch.zeros(self.max_tokens, device=hidden.device, dtype=torch.bool)
```

`cu_seqlens` 存各段的累计边界，例如段长 `[3, 2]` 对应 `[0, 3, 5]`。GPU attention 根据它限制每个 token 能看到的范围，因此把 padding 放在独立段中，可以避免真实音频与 padding 相互 attention。

## 2.2 为什么要额外保留一个 segment

每次 replay 前，slot 调用 `set_layout(hidden, layout)`，其中 `layout` 是 `(lengths, cos, sin)`。函数先检查 hidden 是二维、各真实段长为正、段长之和等于 token 数、容量未超限，并核对 RoPE 的 shape、dtype 和 device。下面节选通过检查后的 padding 构造：

```python
padding_segments = max(self.max_segments - len(lengths), 1)
quotient, remainder = divmod(self.max_tokens - tokens, padding_segments)
dummy_lengths = torch.full((padding_segments,), quotient, dtype=torch.int32)
dummy_lengths[:remainder] += 1
if len(lengths) < self.max_segments:
    dummy_lengths = torch.cat((dummy_lengths, dummy_lengths.new_zeros(1)))
endpoints = torch.nn.functional.pad(
    torch.cat((torch.tensor(lengths, dtype=torch.int32), dummy_lengths)).cumsum(0), (1, 0)
)
```

dummy segments 是只用来容纳 padding 的补齐段。`T - t` 个 padding token 被尽量平均分给这些段。即使真实段数已经达到 `S`，token 总数仍可能小于 `T`，因此至少要额外保留一个补齐段，避免把 padding 接到真实音频段里。

当 `s < S` 时，代码先构造 `S - s` 个 dummy segments，再追加一个零长度段；当 `s == S` 时，直接使用一个 dummy segment。两条分支都使总段数固定为 `S + 1`，加上开头的零边界，`cu_seqlens` 始终有 `S + 2` 项。零长度段不增加有效计算数据，它保留的是固定的边界槽位。

例如 `T=12、S=3`，真实段长为 `[3, 2]`，补齐后段长是 `[3, 2, 7, 0]`，累计边界是 `[0, 3, 5, 12, 12]`。如果真实段长为 `[3, 2, 1]`，则变成 `[3, 2, 1, 6]`，边界仍然有五项。

构造好边界后，函数原地更新已有缓冲区，末尾返回有效 token 数：

```python
self.cu_seqlens_cpu.copy_(endpoints)
self.cu_seqlens.copy_(endpoints, non_blocking=True)
self.rotary_cos[:tokens].copy_(cos)
self.rotary_cos[tokens:].fill_(1)
self.rotary_sin[:tokens].copy_(sin)
self.rotary_sin[tokens:].zero_()
self.valid_tokens[:tokens].fill_(True)
self.valid_tokens[tokens:].fill_(False)
return tokens
```

更新的是同一块存储里的内容，因此不同音频可以使用不同段长和 RoPE，同时保持图所记录的地址不变。padding 的 `cos=1、sin=0` 对应不旋转；`valid_tokens` 标记前 `tokens` 行为有效数据，后面的预热和反向图会据此把 padding 位置的输出梯度设为零。

## 2.3 fixed_ae 执行原 Transformer

完成布局更新后，图内真正执行的是：

```python
def forward(self, hidden):
    return self.tower._forward_transformer(
        hidden,
        self.cu_seqlens,
        self.cu_seqlens,
        self.max_tokens,
        self.max_tokens,
        self.rotary_cos,
        self.rotary_sin,
    )[0]
```

Q 和 KV 使用同一套 GPU 分段边界；`max_seqlen` 使用固定容量上界，避免真实最大段长变成每次不同的 Python 参数。这里虽然也维护了 CPU 边界副本，但该调用传给 attention 的是 GPU 副本。

`_forward_transformer()` 内部仍然遍历原有 layers，按配置使用层内的激活重计算（gradient checkpointing），最后执行 layer norm。它直接执行 Transformer，不再经过 `_dispatch_transformer()`，因此不会递归回到 manager。`fixed_ae` 负责用固定容量的输入调用原模型，捕获和重放则由下一节的 slot 管理。

# 3. slot 如何保存图和训练所需的存储

## 3.1 静态输入与输入梯度

manager 的 `prepare_capture()` 创建 `fixed_ae`，再将它传入新建的 `_GraphSlot`。一个 slot 包含前向图、可选的反向图、输入输出缓冲区和占用状态；它不是只代表一张图。

在训练 AE 的模式下，图前面的卷积也可能需要梯度，因此 slot 除了静态输入，还要准备 Transformer 对输入的梯度。下面节选初始化中的相关代码；`capture_backward=True` 对应前反向捕获模式：

```python
self.hidden = hidden.new_zeros((fixed_ae.max_tokens, hidden.shape[1]))
self.hidden[: hidden.shape[0]].copy_(hidden.detach())
self.hidden.requires_grad_(capture_backward)
self.dx = torch.zeros_like(self.hidden) if capture_backward else None
if capture_backward:
    self.hidden.grad = self.dx
self.pending = False
self.requires_grad_init = False
self.grad_initialized = False
```

`self.hidden` 是图固定读取的输入存储，和这次卷积输出的存储分开。`detach()` 避免准备图时把临时的卷积 autograd 链带进捕获；正式训练中缺少的连接会由第四节的自定义 autograd Function 补上。

`self.dx` 是反向图写入的静态输入梯度，形状同样是 `[T, hidden_size]`。将它赋给 `self.hidden.grad` 后，捕获反向时就会记录向这块地址累积梯度的 GPU 操作。

`pending` 标记是否已有一次前向等待反向，第四节用它防止提前覆盖中间结果。另两个标记用于第六节的共享显存路径：`requires_grad_init` 表示反向前需要执行梯度初始化图，`grad_initialized` 表示本次是否已经执行。

除了输入和输入梯度，slot 还保存图内可训练参数。`_GraphSlot.__init__()` 中有下面这行初始化代码：

```python
self.parameters = fixed_ae.parameters_for_backward
```

`parameters_for_backward` 是 `_FixedCapacityAE` 的属性，返回 `tower.layers` 和 `tower.layer_norm` 中 `requires_grad=True` 的参数元组。因此，`self.parameters` 保存的是这些原模型权重的引用。第四节的 `slot.forward_backward()` 会把它们作为自定义 autograd Function 的输入，反向时按同一顺序返回对应的参数梯度。

stem、`conv_out` 在图外执行，其参数不放进 `self.parameters`。它们的梯度通过另一条路径计算：Transformer 反向图先得到前面介绍的输入梯度 `self.dx`，再由自定义 autograd Function 将有效部分传回图外，沿 packing、投影和卷积的普通 autograd 链继续反向，得到这些图外参数的梯度。具体衔接见第四节。

## 3.2 warmup 先让执行路径准备好

slot 将预热和捕获放在同一个 CUDA stream 上执行；stream 是 GPU 操作的执行队列，未由外层提供时由 slot 创建。开始前先等待当前 stream 上已提交的输入准备操作，确保能读到完整输入。随后执行三轮固定容量 warmup：

```python
with torch.cuda.stream(stream), no_sync(), self._forward_context(CudaGraphPhase.WARMUP):
    for _ in range(3):
        zero_grad_buffer()
        if capture_backward:
            self.dx.zero_()
        output = fixed_ae(self.hidden)
        if capture_backward:
            output.backward(torch.ones_like(output) * fixed_ae.valid_tokens.unsqueeze(1))
    del output
    stream.synchronize()
    zero_grad_buffer()
```

这里的 warmup 是为后续图捕获额外执行的预热计算：用固定 shape 触发所需的执行准备，包括启用编译时的相应路径。输入可以来自真实 batch，但反向使用的全一输出梯度是人为构造的，只用于运行反向算子，掩码使 padding 位置梯度为零。它不是正式训练 loss 的梯度，预热结束后会清理梯度，不会据此更新参数。

`zero_grad_buffer()` 清理音频 DDP 的参数梯度，`self.dx.zero_()` 单独清理 Transformer 的输入梯度。`no_sync()` 暂时关闭 DDP 的梯度同步，避免把这些额外计算当作一次需要跨 rank 同步的训练反向。

`_forward_context(CudaGraphPhase.WARMUP)` 将当前执行标记为 CUDA Graph 预热阶段。需要捕获反向时，它开启 autograd 记录，并设置 `owns_backward_graph=True`，告诉层内 checkpoint 等逻辑：这次前向需要建立可供反向使用的计算关系；仅捕获前向时则关闭 autograd 记录，并标记当前按冻结编码器执行。这些设置描述当前计算的用途，不会创建新的 stream，也不会修改模型参数的冻结配置。

## 3.3 先捕获前向，再捕获反向

warmup 后，manager 先调用 slot 的 `capture_forward_graph()`，需要训练 Transformer 时再调用 `capture_backward_graph()`。独立捕获时二者顺序执行；共享池模式会在中间插入 LLM 捕获与梯度缓冲区准备，第六节再展开。下面节选两者的核心，省略重复捕获检查和可选的 `before_capture` 回调：

```python
# capture_forward_graph
self.forward_graph = torch.cuda.CUDAGraph()
with self._capture_context(), self._te_capture():
    with torch.cuda.graph(self.forward_graph, pool=pool, stream=self.stream):
        self.output = self.fixed_ae(self.hidden)
self.grad_output = torch.zeros_like(self.output)

# capture_backward_graph，仅用于 forward_backward 模式
self.backward_graph = torch.cuda.CUDAGraph()
with self._capture_context(), self._te_capture():
    with torch.cuda.graph(self.backward_graph, pool=pool, stream=self.stream):
        self.dx.zero_()
        self.output.backward(self.grad_output * self.fixed_ae.valid_tokens.unsqueeze(1))
self.parameter_grads = tuple(parameter.grad for parameter in self.parameters)
self.grad_added_to_main_grad = tuple(
    parameter.grad_added_to_main_grad for parameter in self.parameters
)
```

前向捕获留下 `self.output` 及其 autograd 关系，反向捕获借助这条关系记录反向 GPU 操作。`grad_output` 存放下游传回的输出梯度：捕获时先用零值建立反向图，正式训练时再复制真实梯度。`dx.zero_()` 也被记录在图中，因此每次反向 replay 都先清空上一轮输入梯度，再计算本轮贡献。

两个图使用同一个 `self.stream` 捕获。`_capture_context()` 还暂时关闭 DDP 梯度同步、将执行阶段标记为 `CAPTURE`，并持有 `pinned_memory_capture_lock`。pinned memory 是用于 CPU/GPU 数据传输的锁页 CPU 内存，其分配可能触发 CUDA 调用；这把锁让配合使用它的后台线程避开捕获时段，防止并发分配干扰全局 capture。

后续 slot 直接调用 `graph.replay()`，由调用处的当前 stream 提交，不要求回到 capture stream。stream 决定执行顺序，Tensor 的设备地址不属于某个 stream；只要相关存储仍有效、读写依赖已正确同步，换一个 stream 不会改变图记录的地址。

`_te_capture()` 通知 Transformer Engine（TE，提供 Transformer 算子和图支持的库）当前进入捕获，并在退出时结束这个状态。捕获中的 Python 分支只决定本次记录哪些 GPU 操作，不会因 replay 自动重跑。

这份 DDP 实现把多个参数的梯度集中存入连续缓冲区，每个参数的 `main_grad` 是其中对应部分的视图。末尾的两个元组只是 Python 保存动作：`parameter_grads` 保留各参数 `.grad` Tensor 的引用；`grad_added_to_main_grad` 保存 DDP 的标记值，用于判断梯度是否已直接累加到 `main_grad`。它们都不属于反向图，第四节会说明 replay 后如何使用。

## 3.4 清理捕获状态，保留 replay 所需的地址

所有图记录好后，manager 调用各 slot 的 `finish_capture()`：

```python
def finish_capture(self) -> None:
    torch.cuda.current_stream(self.hidden.device).wait_stream(self.stream)
    if self.capture_backward:
        self.output = self.output.detach()
    for parameter in self.parameters:
        parameter.grad = None
    self.zero_grad_buffer()
```

开头的 `wait_stream()` 让当前 stream 后续操作等待捕获 stream 的已有工作完成。`detach()` 随后得到一个不带旧 autograd 关系、但共享同一输出存储的 Tensor。已经记录好的反向图通过固定地址运行，不再依赖这个 Tensor 的 `grad_fn`；正式训练由第四节的自定义 autograd Function 调用反向图，并把梯度交回图外。

`parameter.grad = None` 只解除普通 `.grad` 属性的引用，不会清零前面介绍的 DDP 梯度存储，所以还要调用 `zero_grad_buffer()`。`parameter_grads` 元组仍保留着此前的梯度 Tensor，供第四节返回给 autograd。这里清理的是额外建图计算产生的梯度，正式训练会另行 replay，详见第五节。

# 4. replay 如何接回卷积、adapter 和参数梯度

## 4.1 前向先填缓冲区，再返回独立输出

图准备好后，manager 选择空闲 slot，调用其 `forward_backward()`。该方法通过自定义 `torch.autograd.Function` 把 GPU 图接入普通 PyTorch 计算图：

```python
return _ForwardBackwardReplay.apply(self, hidden, layout, *self.parameters)
```

这里传入的 `hidden` 是图外卷积与 packing 得到的真实 Tensor，参数也是原模型参数。它们作为 Function 的输入，使外层 autograd 能知道这个自定义节点与哪些上游 Tensor、权重相关。

Function 的前向为：

```python
@staticmethod
def forward(ctx, slot, hidden, layout, *parameters):
    if slot.pending:
        raise RuntimeError("AE CUDA Graph slot already has an outstanding forward")
    tokens = slot.prepare(hidden, layout)
    slot.forward_graph.replay()
    slot.pending = True
    ctx.slot = slot
    ctx.tokens = tokens
    return slot.output[:tokens].clone()
```

`prepare()` 在 `no_grad` 下检查 feature shape、dtype 和 device，更新第二节的 layout，再把真实 hidden 复制到静态输入前 `tokens` 行，将尾部清零。随后 replay 使用这些地址中的新数据执行前向。

返回时先裁掉 padding，再 `clone()`。下游 adapter 的反向可能保存这个输入；若直接返回静态输出的视图，下一次 replay 会覆盖它。复制保护了图外消费者，但图内反向所需的中间结果仍由 slot 保存，因此 `pending=True` 期间禁止另一次前向占用同一 slot。

冻结 AE 的 `forward_only()` 也会复制输出：即使 AE 不反向，训练中的 adapter 仍可能需要保存其输入。

## 4.2 反向返回两类梯度

下游 adapter 和输出整理的普通反向到达这个 Function 后，传入 Transformer 输出的梯度 `gradient`。入口先检查前反向是否配对；共享显存模式还要求本轮已完成梯度清零，这对应第六节的两个初始化标记。反向代码如下：

```python
@staticmethod
def backward(ctx, gradient):
    slot = ctx.slot
    if not slot.pending:
        raise RuntimeError("AE CUDA Graph backward has no matching forward")
    if slot.requires_grad_init and not slot.grad_initialized:
        raise RuntimeError("AE backward requires grad-init replay before adapter backward")
    if gradient.shape != slot.grad_output[: ctx.tokens].shape:
        raise ValueError("AE CUDA Graph output gradient shape changed")
    slot.grad_output.zero_()
    slot.grad_output[: ctx.tokens].copy_(gradient)
    try:
        slot.backward_graph.replay()
        # Replay cannot update Python-side DDP bookkeeping captured in the graph.
        for parameter, value in zip(slot.parameters, slot.grad_added_to_main_grad, strict=True):
            parameter.grad_added_to_main_grad = value
        return (None, slot.dx[: ctx.tokens].clone(), None, *slot.parameter_grads)
    finally:
        slot.pending = False
        slot.grad_initialized = False
```

外部梯度只覆盖有效 token，先清零整个 `grad_output` 可以去除上一轮尾部内容。反向图读取它，写入静态 `dx` 和捕获时确定的参数梯度地址。

返回值与 `forward(slot, hidden, layout, *parameters)` 一一对应：`slot`、`layout` 不求导；`dx[:tokens].clone()` 是原始 hidden 的梯度，沿 packing、卷积和投影的普通 autograd 链继续传播；后面的 `parameter_grads` 对应图内 Transformer 权重。

CUDA Graph 记录了向哪些地址写梯度，但外层 autograd 仍需要通过 Function 的输入和返回值，建立这个节点与原模型权重的梯度关系，并触发 DDP 注册的梯度处理回调（hook）。因此，参数既作为前向输入，也有对应的反向返回值。

某些融合算子会把参数梯度直接累加到 `main_grad`，而不是先完整写入 `.grad` 再由 DDP 搬运。`grad_added_to_main_grad` 标记这种累加是否已经发生；GPU replay 不会重新执行修改该 Python 属性的代码，所以这里恢复捕获时保存的值，让 DDP 避免再次累加同一份贡献。返回的 `parameter_grads` 用来衔接 autograd，实际用于更新权重的梯度也可能已在 `main_grad` 中。

`dx` 也要复制，因为反向 replay 下一次还会清零并写入同一缓冲区。`finally` 解除 slot 占用，并清掉本次梯度初始化标记，让下一次前向重新开始完整配对。

# 5. manager 何时捕获，何时执行正式训练

底层 slot 已经能建图与 replay，manager 负责决定这些行为的时机。建图需要一份输入示例，manager 会保存输入特征与分段、RoPE 信息作为“捕获模板”。其 `forward()` 函数体如下；`_collecting_template` 表示正在专门生成这份示例，通常的训练调用不走这个分支：

```python
if self._collecting_template or (self.defer_capture and not self.graphs_ready):
    self._remember_template(hidden, layout)
    if self._collecting_template:
        raise _CaptureTemplateReady
    lengths, cos, sin = layout
    cu_seqlens = torch.nn.functional.pad(
        torch.tensor(lengths, device=hidden.device, dtype=torch.int32).cumsum(0), (1, 0)
    ).to(torch.int32)
    return self.tower._forward_transformer(
        hidden, cu_seqlens, cu_seqlens, max(lengths), max(lengths), cos, sin
    )[0]
if not self.slots:
    self._capture(hidden.detach(), layout)
if self.mode == "forward":
    return self.slots[0].forward_only(hidden, layout)
for slot in self.slots:
    if not slot.pending:
        return slot.forward_backward(hidden, layout)
raise RuntimeError(f"AE CUDA Graph has more than {self.max_inflight} outstanding forwards")
```

独立模式首次遇到有效输入时，`_capture()` 按 `prepare_capture → capture_forward → capture_backward（如需要）→ finish_capture` 建图。`prepare_capture()` 创建 `fixed_ae` 和 slot，并完成第三节的预热；两个 capture 方法分别委托 slot 记录 GPU 操作；`finish_capture()` 清理捕获状态，并把 `graphs_ready` 设为 `True`。

`_capture()` 完成后没有提前返回，当前输入会继续进入下面的正式 replay。第三节清掉的是额外的捕获梯度，随后 replay 计算出的正式梯度才参与这次训练。

`defer_capture=True` 则把 AE 的建图时机交给训练循环的管理者 trainer：图尚未准备好时，保存首次输入及其布局的独立副本，不保留该次输入的 autograd 关系，然后执行普通 Transformer。**这些普通前反向可以属于有效训练迭代。** 等 trainer 为 encoder 和 LLM 安排好共享池后，才统一捕获；后续正式调用改为 replay。

如果准备共享捕获时仍没有模板，例如此前该 rank 没有收到真实音频，`prepare_shared_capture_inputs()` 会调用 `ensure_capture_template()` 补齐。后者设置 `_collecting_template=True`，再调用初始化时提供的 `template_provider`：用一秒全零波形执行特征提取、卷积和 packing。到达 Transformer 入口后，manager 保存输入，并抛出内部异常 `_CaptureTemplateReady`；`ensure_capture_template()` 接住它，结束本次准备。因此这条示例音频只用于建立模板，不继续执行 Transformer、adapter，也不作为额外训练样本。

示例目前强制 `max_inflight=1`，即一个 manager 只有一个 slot，训练时最多允许一次 AE 前向等待对应反向。`graphs_ready` 表示整套捕获完成；`slot.pending` 表示 slot 正被一次尚未结束的前反向占用。已有图可用时，manager 仍需检查 slot 是否空闲。

# 6. 与 LLM 共享显存的完整链路

## 6.1 trainer 用两个回调安排捕获顺序

独立 AE 图已经能训练，共享池是在此基础上减少显存占用：LLM 的部分中间结果用完后，它们占据的显存可以再用于 encoder 梯度。这里依赖明确的计算顺序——encoder 前向先完成，随后运行 LLM 前反向，最后运行 encoder 反向。示例会校验调度计划，确保要复用同一地址的数据不会同时需要保留。

`cuda_graph_helper` 是 trainer 持有的 LLM 图管理对象，负责组织 LLM 和可选的 optimizer 图捕获。它通过 `new_shared_pool()` 创建图内存池的句柄；模型管理层再提供两个函数，让它在指定时机执行 encoder 捕获。下面节选共享池分支，省略外层捕获时机判断：

```python
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

名字中的 `ve` 是沿用视觉编码器（VE）的已有接口，实际参与者可以包含 VE 和 AE。进入 `with` 时，`capture_encoder_cuda_graphs()` 找到需要捕获的 encoder，定义两个函数，再通过 `yield` 将它们交给 `as` 后面的两个变量。此时拿到的是函数本身。

`create_cudagraphs()` 接收这两个回调：在 LLM 建图前调用 `start_ve_capture()`，在 LLM 建图完成、可选的 optimizer 建图之前调用 `finish_ve_capture()`。这就是先“提供函数”、后“在指定位置调用”的关系；下面的流程按这条正常调用路径展开。

仅前向的 encoder 不参加梯度初始化和反向阶段：

```text
start_ve_capture
  → 准备所有 encoder 的 slot、静态输入与 warmup
  → 捕获 VE 前向，再捕获 AE 前向（按实际参与者）
  → 解除训练型 encoder 旧 DDP 梯度缓冲区的引用
LLM graph manager
  → 按流水线计划捕获 LLM 前向与反向
finish_ve_capture
  → 在共享池捕获各 encoder 的梯度初始化图
  → 捕获 AE 反向，再捕获 VE 反向
  → 清理捕获状态、固定梯度地址、保存运行期回调
optimizer graph（启用时）
```

encoder 使用与 TE 相同的 pool 和 capture stream。pool 句柄由 PyTorch 提供，传给 `torch.cuda.graph(..., pool=pool)` 后，捕获中新分配的 CUDA 张量可以由该图池分配存储。已有 Tensor 仍使用原地址，不会因传入句柄而自动搬迁；能否复用某块显存，还取决于大小、分配顺序以及该存储是否已可复用。

这套捕获流程只在首次建图或第七节的扩容重建时执行。trainer 完成捕获后会关闭本次捕获触发条件，正式训练主要执行已经记录好的图。

## 6.2 释放的是哪一块旧梯度存储

这里的 `wrapper` 指经过 DDP 包装的 encoder 模型（DDP wrapper），音频分支对应 1.2 中的 `audio_model`。

上面 `start_ve_capture()` 在 encoder 前向图捕获后，调用 `release_model_grad_buffer(wrapper, force=True)`，再由它逐个调用底层 `release_grad_buffer()`。旧存储就是第三节介绍的 DDP 连续梯度缓冲区，初次在 DDP 初始化时分配，也可能经过此前普通执行路径的重建。slot 的 warmup 使用这块现有存储。

释放前，`buffer.grad_data` 持有整块存储；DDP 按梯度通信需要划分的 bucket，以及各参数的 `main_grad`，都通过视图引用其中一部分。`release_grad_buffer()` 把 `buffer.grad_data`、`bucket.grad_data` 和 `param.main_grad` 逐一设为 `None`，解除这些对象对旧存储的持有。这个操作不清零数据；没有其他持有者后，分配器才能回收该存储供以后使用，也不一定立即把显存交还 GPU 驱动。

若模型使用 TE 的 grouped weight，一个注册的 Parameter 下面还可能缓存多组权重各自的梯度视图。只清理父参数的 `main_grad`，这些缓存仍可能指向旧存储。因此 release 最后调用 `_refresh_grouped_main_grads()` 清理成员视图；后面 rebuild 也调用它，使缓存改为引用新存储，避免后续算子向旧地址写梯度。

之所以放弃旧存储，是因为它此前已经在共享捕获之外分配，继续持有它就无法让这份 encoder 梯度改用图池中的空间。释放后先捕获 LLM，使分配器建立 LLM 的存储布局，再为 encoder 梯度分配新存储。

LLM 捕获链路也会处理已用完的中间结果。例如，TE 启用 `_reuse_graph_input_output_buffers` 时，会在对应计算完成后处理部分前向输出和反向输入梯度的引用：保留按原地址访问的能力，同时不再靠这些引用占住存储，使图池能复用其中的空间。仍需保留的数据，例如等待 optimizer 使用的 LLM 参数梯度，不能提前覆盖。

## 6.3 梯度初始化图只在捕获时分配和绑定

LLM 建图结束后，`finish_ve_capture()` 把重建音频 DDP 梯度缓冲区的操作交给 `manager.capture_grad_init()`。具体而言，它传入一个回调，调用时执行 `rebuild_model_grad_buffer(wrapper, force=True)`。manager 在指定共享 pool 的 `torch.cuda.graph(...)` 上下文中调用它，后者再逐个调用底层 `rebuild_grad_buffer()`。

重建包含三件事：分配新存储、将数值清零、让 DDP 和参数的梯度视图指向新存储。其中 GPU 清零操作被记录为一张独立的 `grad_init_graph`，不放进 AE 前向图或反向图。

底层 rebuild 的关键操作如下，省略重复检查存储为空的断言，以及逐个 bucket 和参数的视图重绑循环：

```python
if buffer.grad_data is not None:
    return
buffer.grad_data = torch.empty(
    buffer.numel,
    dtype=buffer.grad_dtype,
    device=torch.cuda.current_device(),
    requires_grad=False,
)
buffer.grad_data.zero_()
# 将 bucket.grad_data、param.main_grad 重新绑定到新存储的对应切片。
_refresh_grouped_main_grads(model, buffer)
```

**分配 Tensor、设置 Python 属性和重绑视图发生在捕获阶段；运行期 replay 执行的是已记录的 GPU 清零操作。** 每轮不会重新运行这个 Python 函数，也不会重新申请一块梯度显存。

因为新分配发生在指定 pool 的捕获上下文里，它有机会取得 6.2 中已可复用的 LLM 临时存储。之后再捕获 AE 反向，反向算子记录的就是这些新 `main_grad` 地址。参数本身、图外静态输入等既有存储仍可能位于其他池中；共享的是满足条件的存储，不要求所有张量都在同一个池里。

## 6.4 每轮在整个音频分支反向前清零

捕获完成后，`finish_ve_capture()` 调用 `_set_cuda_graph_static_grad_buffers(True)`，将参与模型的 `_cuda_graph_static_grad_buffer` 标记设为 `True`。常规的 release/rebuild 方法读到该标记就直接返回，从而保留反向图已经记录的梯度地址。随后，它把初始化图的 replay 方法保存到音频 DDP wrapper 上：

```python
wrapper._cuda_graph_prepare_grad_buffer = manager.replay_grad_init
```

右侧没有 `()`，因此这里只保存方法，没有执行清零。正式训练中，流水线执行器（pipeline executor）在 LLM 计算结束后执行 encoder 梯度准备动作，调用 `prepare_model_grad_buffer_for_backward(self.encoder_models)`。该函数逐个处理 encoder 的 DDP wrapper，核心分支如下：

```python
prepare = getattr(model, "_cuda_graph_prepare_grad_buffer", None)
if getattr(model, "_cuda_graph_static_grad_buffer", False) and prepare is not None:
    prepare()
else:
    rebuild_model_grad_buffer(model)
```

局部变量 `prepare` 此时指向 `manager.replay_grad_init`，调用它会执行 `self.grad_init_graph.replay()`，将 6.3 中分配好的梯度存储清零。这发生在 LLM 计算之后、整个音频分支反向之前。第四节的 `slot.prepare(hidden, layout)` 是另一个方法，只负责前向输入填充。

清零不能等到 AE Transformer 的反向图内部才做：音频 DDP 的梯度缓冲区还服务于图外 adapter 和卷积，adapter 会先产生梯度。若那之后才清整个缓冲区，adapter 的有效贡献就会丢失。第三节中的 `dx.zero_()` 仅清 Transformer 输入梯度，是另一块存储、另一种职责。

正式训练沿用同一组地址，使用顺序如下；这是运行流程示意，不是源码：

```text
encoder 前向 → LLM 前反向，使用部分共享地址存放中间结果
            → 梯度初始化图清零 encoder 梯度区域
            → adapter、AE、卷积反向，向梯度区域写入并累加
            → 梯度归约与 optimizer 消费梯度
            → 下一轮可再次覆盖这些地址
```

调度计划中仍可能保留 release/rebuild 动作，但静态标记会让常规释放跳过，让准备动作转为上面的清零图。不会每轮重新解除引用和申请存储。LLM 下一轮仍可使用与 encoder 梯度重叠的地址，因为两边图的地址在捕获时已经确定，replay 无需重新向 Python 分配器申请；正确性依赖上图中各份数据的使用顺序。

`replay_grad_init()` 即使遇到本轮没有音频前向也会执行清零，避免这些地址上残留的 LLM 数据被当成音频梯度。若 `slot.pending=True`，它还把 `grad_initialized` 设为 `True`，表示这次前向的反向已具备干净的梯度存储。对同一次尚未完成反向的前向重复初始化，会在该方法中报错；漏掉初始化则由第四节的反向入口报错，反向结束后该标记重新置为 `False`。

# 7. 容量扩展与正确性边界

固定容量保证一套图的地址与结构稳定，容量不够时则需要扩大并重新建图。示例由 trainer 在开始本轮计算前检查已准备好的数据：根据多批音频如何合并调用、如何分给各 rank、如何拆成音频段，以及无真实音频时是否执行占位输入，计算每次 Transformer 调用的 token 数和 segment 数，再分别取本轮最大值。这一步读取已有的 CPU 长度和形状信息，不重新取样，也不推进到下一批数据。

超限时调用的增长规则为：

```python
def capacity_for(max_tokens: int, max_segments: int, tokens: int, segments: int) -> tuple[int, int]:
    if tokens < 0 or segments < 0:
        raise ValueError("encoder CUDA Graph requirements must be nonnegative")
    if tokens > max_tokens:
        target = max((5 * max_tokens + 3) // 4, (11 * tokens + 9) // 10)
        max_tokens = (target + 1023) // 1024 * 1024
    if segments > max_segments:
        max_segments = max((5 * max_segments + 3) // 4, (11 * segments + 9) // 10)
    return max_tokens, max_segments
```

两个维度独立增长，只扩大超限的一维：至少比旧容量增加约 25%，且覆盖本次需求的约 110%；token 容量再向上对齐到 1024。各 rank 保留各自容量，不把所有 rank 一起放大。

还没捕获时可直接更新待捕获容量；已有独立图时重建该 encoder 的图；已有共享图时，需要全体训练 rank 协调，等本轮计算尚未开始且上一轮相关工作已结束，再删除并重建相关 encoder、LLM 和已启用的 optimizer 图。这样才能一起重新安排可能重叠的地址。重建保留模型参数、optimizer 状态和本轮数据；有未完成反向的 slot 时禁止直接 reset。若绕过 trainer 的预检直接调用 manager，超限仍由 `set_layout()` 报错。

这个示例要求使用 BF16、RoPE 和支持 packed 输入的 FlashAttention，并将 dropout、随机跳过 Transformer 层的 LayerDrop 设为零。它还要求关闭 DDP 的梯度归约与计算重叠，源码将原因标为已知的 DDP 预热问题；这是本实现的限制，不能推成 CUDA Graph 的通用要求。

Transformer 层内和卷积内部的激活重计算可以按配置保留；在整个音频编码器或 adapter 外再包一层 checkpoint 的功能则要求关闭，对应 `ckpt_audio_encoder`、`ckpt_audio_encoder_use_reentrant` 和 `ckpt_audio_adapter`。

评估收益时需要把首次编译、warmup、capture 和扩容成本与稳定 replay 分开。较大容量能减少重捕获，但也增加固定缓冲区与 padding 工作。本文说明源码机制，不据此给出吞吐或数值一致性的实测结论。

# 8. Q&A

## 1. 3.4 中为什么要对 `self.output` 做 `detach()`？保留原来的 autograd 关系会有什么问题？

捕获前向时，`self.output` 不仅持有输出数据，还通过 `grad_fn` 关联着捕获时建立的 PyTorch autograd 图。随后捕获反向时，已经沿这条关系调用过一次 `backward()`，把相应的 GPU 操作记录下来。

正式训练时，每次 `_ForwardBackwardReplay.apply(...)` 会在外层建立本次调用的自定义 autograd 节点。反向到达该节点后，由它调用 `backward_graph.replay()`，执行捕获好的 GPU 操作，不需要再沿 `self.output` 的旧 `grad_fn` 求导。

因此，`detach()` 让 slot 持有的 `self.output` 只承担静态输出缓冲区的职责：保留同一块存储，去掉捕获时的旧求导关系。前向 replay 仍向这块存储写入数据，已经记录好的反向图也不会因此失效。

如果保留旧关系，slot 会继续持有不必要的旧图引用；如果后续某段普通 PyTorch 代码沿它求导，还可能进入已经执行过反向的旧图。旧图中部分保存的中间结果已经释放，此时可能出现重复反向的报错。

**不能据此断言删掉 detach 就一定立刻报错。** 当前自定义 `Function.forward()` 内部默认不记录普通 autograd 运算，也起到了隔离作用。这里的 detach 主要用于清理捕获阶段的旧关系，明确运行期静态缓冲区的职责。

## 2. 4.1 中为什么返回 `slot.output[:tokens].clone()`？不 clone 会有什么问题？

`slot.output` 是反复使用的固定输出缓冲区，`slot.output[:tokens]` 只是共享其存储的视图。若直接返回这个视图，调用者拿到的结果仍会受到 slot 后续写入的影响。下面是存储变化示意，不是源码：

```text
第一次 replay：把 batch 1 的结果写入 slot.output
返回 y1：y1 与 slot.output 共享存储
同一 slot 后续再次 replay：把 batch 2 的结果写入 slot.output
外部仍保存着的 y1：其内容也变成了 batch 2 的结果
```

外部对返回值做原地修改，也可能反过来改坏 slot 的内部输出。`clone()` 给调用者一份独立存储，使返回值保持“这是本次调用的结果”的含义；下游可以按普通 Tensor 使用或保存它，不必追踪 slot 的复用时机。

这里的 `pending` 会阻止同一个 slot 在配对反向完成前再次前向，因此不能说下一次前向一定会覆盖尚未完成反向的输出。不过，`pending` 不保证外部在反向完成后就不再持有这个输出，也不防止外部通过共享视图修改内部数据。

这一处保护的是输出存储，第一问的 detach 处理的是求导关系：**detach 去掉旧关系但仍共享存储，clone 才会复制数据。**

## 3. 4.2 中为什么返回输入梯度时也要用 `slot.dx[:ctx.tokens].clone()`？

`slot.dx` 是 Transformer 输入 hidden 的固定梯度缓冲区。每次反向图 replay 都会先清零它，再把本次输入梯度写入同一块存储。返回的梯度随后交给外层 autograd，沿 packing、卷积等图外运算继续传播。

如果只返回切片，外层拿到的梯度仍与 `slot.dx` 共享存储。这个反向函数在 `finally` 中把 `slot.pending` 设为 `False`，允许后续复用 slot；若外层仍在使用或保存这份梯度，后续对 `dx` 的清零和重写就可能影响已经返回的数据。autograd 接收到一个梯度 Tensor，并不保证立即替调用者复制一份。

`clone()` 把“本次交给外层的梯度”与“slot 下次要复用的梯度缓冲区”分开。它与前向 clone 的目的相同，只是保护的数据从输出变成了输入梯度。在严格限制执行顺序和引用生命周期的实现中，某些复制可能省掉；当前实现通过复制，避免要求图外消费者承担这些限制。

## 4. 6.1 中的 `new_shared_pool()` 具体创建什么？共享显存池的大小能控制吗？

`new_shared_pool()` 是示例中 LLM 图管理对象的方法。它先检查已配置共享池、且当前没有已创建的图，然后执行下面两行；这里省略前面的检查分支：

```python
self._graph_pool = torch.cuda.graph_pool_handle()
return self._graph_pool
```

它获取的是 **PyTorch CUDA Graph 显存池的标识，也就是 pool 句柄**。这是 PyTorch 提供的能力，不是 TE 独有的接口；调用本身也不会立即申请一块指定大小的显存。

后续捕获不同图时，把同一个句柄传给 `torch.cuda.graph(..., pool=shared_pool)`，这些捕获过程中的显存分配就可以使用同一个图池。在满足执行顺序和生命周期要求时，其中的显存块可以复用。传入句柄不会把之前已经分配好的 Tensor 自动搬进这个池。

**当前 `new_shared_pool()` 和它调用的 `graph_pool_handle()` 都没有指定池大小的参数。** 显存需求由捕获中的实际分配决定，可以通过最大 token 容量、模型规模、捕获图的数量以及张量存活时间等因素间接调整。这个句柄用于选择共享哪个池，不用于指定该池的容量上限。
