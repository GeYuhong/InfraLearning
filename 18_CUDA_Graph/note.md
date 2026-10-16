# 1. 入口与捕获粒度

LLM 训练会反复执行相似的前反向计算。CUDA Graph 将一段执行记录下来，后续向固定输入地址写入新数据，再 replay 已记录的 GPU 工作，减少逐个算子从 CPU 发起的开销。它保留计算与通信依赖，不会自动减少矩阵乘法的计算量。

一些框架会用 Transformer Engine（TE）负责建图和 autograd 衔接，再由框架的图管理器（manager）管理捕获时机、流水线顺序、输入模板和共享显存。下面以这样的实现为例，重点看完整 Transformer 层与 VPP chunk 两条路径。类名、配置限制和调度约定属于这个示例。

先明确流水线中的几个对象：流水线并行（PP）把模型的不同部分放到不同 rank 上；虚拟流水线（VPP）让一个 rank 持有多个模型 chunk，交错执行它们。**chunk 是一段模型，microbatch 是一次送入这段模型的小批数据**；一个训练 iter 通常要执行多个 microbatch 的前反向，再更新参数。

## 1.1 trainer 选择 manager，但不立即捕获

trainer 在训练循环前，根据捕获范围选择 manager。下面节选 TE 分支，省略其他实现的选择和配置检查：

```python
if use_te_vpp_stage_cuda_graph:
    cuda_graph_helper_cls = TEVPPStageGraphManager
elif use_te_managed_cuda_graph:
    cuda_graph_helper_cls = TELayerGraphManager
else:
    cuda_graph_helper_cls = TECudaGraphHelper
```

trainer 用选出的类创建 `cuda_graph_helper` 对象，传入本 rank 已有的模型 chunks、序列长度、microbatch size、optimizer 和 microbatch 数量计算器。本文后面说的 LLM manager 就是这个对象；它持有模型的引用，负责建图，没有重新创建一套模型权重。

这里 `use_te_managed_cuda_graph` 对应 `full_layer` 或 `vpp_stage`，第一个分支已先处理 `vpp_stage`。这两类 manager 共用 `_TEWholeModuleGraphManager` 的生命周期；第三个 helper 处理其他 TE 捕获范围，本文不展开。

初始化主要定位待捕获模块、为它们准备调用接口并保存配置，此时 `graphs_created()` 仍为假。trainer 先进行普通训练预热，等真实流水线计划和输入模板具备后，再调用 manager 的 `create_cudagraphs()`。创建 manager 和真正捕获图是两个时刻。

## 1.2 full_layer 与 vpp_stage 的边界

manager 需要把捕获范围交给 TE。TE 接收的是可以像函数一样调用的对象，代码称为 **callable**。本例用 `nn.Module` 包装原来的模型模块：包装对象整理输入并调用原模块，使 TE 能从一个入口捕获所选范围。

| 范围 | 交给 TE 的基本 callable | 一次前向覆盖的主体 |
| --- | --- | --- |
| `full_layer` | 包装单个 Transformer layer 的模块 | 该层 attention、MLP/MoE、残差等计算 |
| `vpp_stage` | `_VPPStageCallableWrapper` | 一个 chunk 的整个 decoder `TransformerBlock` |

`_VPPStageCallableWrapper` 保存 chunk 中原有的 decoder block，并在捕获时调用它。下面只节选 block 调用，省略输入整理、上下文设置，以及后面的重算登记与清理；第二节再说明输入如何整理：

```python
output = self.block.forward(
    hidden_states=hidden_states, llm_bitwise_mode=self.llm_bitwise_mode, **kwargs
)
```

这个包装对象使用原 block 的权重。对从上游接收激活的 block，它还会临时把 `block.input_tensor` 设为建图用的输入，调用结束后恢复，防止 block 读取此前普通训练留下的输入。

TE 会为各 callable 的每次 microbatch 调用准备前向图和反向图。因此 `vpp_stage` 一张前向图覆盖的是一个 microbatch 在一个 chunk 的 decoder 中的计算；`full_layer` 则需要依次执行该 chunk 各层的前向图，反向顺序相反。

manager 还把这些 callable 按调度职责组成 **unit**，由 `_GraphUnit` 记录所属 chunk、原模块和包装对象。一个 full-layer unit 可以包含同一 chunk 的多个层 callable；一个 VPP decoder unit 通常只有一个 block callable。第三节会用 unit 描述调度顺序，unit 本身不等于一张 CUDA Graph。

## 1.3 decoder 之外的可选捕获范围

只选择 `vpp_stage` 时，embedding、MTP、LM head 和 optimizer 不会自动被纳入 decoder 图。示例可以为前三者增加模型 unit，并另外捕获 optimizer 更新：

| 范围 | 图内边界 |
| --- | --- |
| embedding | 主模型 embedding 调用 |
| MTP | 预测未来 token 的 MTP block |
| LM head | hidden 到 logits 的投影，以及逐 token 的交叉熵 |
| optimizer | 支持的本地参数更新步骤，独立于模型前反向图 |

例如 `LMHeadCallable` 返回逐 token loss。下面是其前向源码：

```python
def forward(self, hidden, labels):
    self.bind_capture()
    logits = compute_language_model_logits(
        self.output_layer,
        hidden,
        weight=self.output_weight,
        runtime_gather_output=self.gather_output,
        output_dtype=self.output_dtype,
    )
    return (self.compute_loss(labels, logits),)
```

`labels` 的移动或整理、最终 loss 归约、MTP loss 缩放和日志仍由外层处理。示例要求 embedding、MTP、head 与 `vpp_stage` 配合；`full_layer` 和 `vpp_stage` 互斥，也不能把这些整块范围与局部层范围任意混合。另一种 `full_iteration` 实现属于不同路径，本文不展开。

# 2. 静态输入与梯度边界

## 2.1 一次调用需要保存哪些东西

捕获前，manager 为每次 callable 调用构造 `sample_args` 和 `sample_kwargs`：前者保存位置参数，后者保存关键字参数。这里的 sample 是建图用的输入 Tensor，正式训练会把本批数据复制到这些固定地址。

TE 随后把参数容器中的 Tensor 按约定顺序列成一个序列，称为“展平”。这一步整理的是 Python 容器，不是把 hidden Tensor 的 shape 压成一维；关键字参数也按固定键顺序排列，以便捕获和 replay 一一对应。

一个前反向调用涉及以下对象：

| 对象 | 保存什么 | replay 时的用途 |
| --- | --- | --- |
| `static_input_surface` | 用户输入、Tensor kwargs、模块参数 | 前向读取输入和当前权重 |
| `static_outputs` | 捕获前向得到的输出 | 向外部返回前向结果 |
| `static_grad_outputs` | 外部传来的输出梯度缓冲区 | 反向图的梯度入口 |
| `static_grad_inputs` | 与完整输入列表逐项对应的梯度，不求导的位置为 `None` | 接回外层 autograd |

`static_input_surface` 是上述显式输入加上模块参数组成的完整输入列表。下面节选组合步骤，其中 `flatten_sample_args[i]` 已包含位置参数和关键字参数：

```python
per_callable_static_input_surfaces = [
    flatten_sample_args[i] + per_callable_module_params[i]
    for i in range(len(flatten_sample_args))
]
```

这里的索引 `i` 对应某个 callable 的一次 microbatch 调用，所以同一模型模块可以对应多组静态输入输出，权重仍是同一套。TE 还会根据预热反向结果剔除未产生梯度的模块参数，避免为它们错误触发梯度 hook。

用户调用 `decoder(hidden)` 时不必显式传权重，但本例的 TE 实现会把权重传入自定义 autograd Function。这样 Function 的反向返回值就有对应的参数位置，能接入 PyTorch 的参数梯度累积和分布式训练流程；第四节会把这条链路展开。

**固定的是图记录的存储地址和执行结构，不是 Tensor 数值。** optimizer 可以原地更新已有参数，下一次 replay 会读到新权重；若直接替换参数对象或底层存储，图不会自动改为读取新地址。

## 2.2 变长信息要进入可更新的 Tensor 边界

LLM 的输入激活 `hidden` 按捕获配置确定 shape。常见形式是 `[sequence, batch, hidden]`，三个维度分别表示序列长度、批大小和特征宽度。packed 输入会把多条序列的有效 token 连续排列；上下文并行（CP）会把序列计算分到多个 rank。这些配置以及隐藏通道扩展可能改变模块实际接收的布局。

例如 VPP 的 pipeline 边界可能仍是三维，而内部第一层使用扩展后的四维表示。manager 必须构造 block 真正接收的输入，不能直接把内部层模板拿来当 chunk 输入。

attention 还需要知道每段 token 的边界、旋转位置编码（RoPE）、哪些位置允许相互关注，以及 CP 中的序列分配信息。需要逐批变化的部分，被整理为固定 shape 的 Tensor kwargs，例如用累计长度 Tensor 表示 packed 分段边界。

`_VPPStageCallableWrapper` 在捕获时把这些 Tensor 重新放进 attention 所需的参数结构，再调用 block。replay 只更新其中 Tensor 的内容，不重新构造 Python 对象，也不会重新选择 Python 分支。

因此“本批数据的分段边界变化”与“更换输入 shape、增加未捕获的 kwarg、改变 Python 控制分支”是不同的变化。前者可以由已实现的静态元信息缓冲区承载；后者需要重新验证，必要时重捕获。示例会检查 replay 的 kwarg 是否属于捕获时的集合，并拒绝改变已固定的选项。

# 3. 从流水线计划得到捕获顺序

## 3.1 执行器是顺序的来源

有了 callable 和输入模板，还不能随意逐层建图：多个 microbatch 的激活会同时存活，前向和反向又会交错，共享池必须按真实生命周期分配地址。

pipeline executor（流水线执行器）负责按本 rank 的计划发起各 chunk 的计算和流水线通信，并在每个训练 step 开始前发布计划。manager 的 `_build_graph_plan()` 读取它在普通训练中发布的计划，作为捕获顺序的依据。下面是核心源码，省略两个异常的完整提示字符串：

```python
schedule = published_executor_schedule()
if schedule is None:
    raise RuntimeError(...)  # 必须先有 executor 发布的训练计划。
num_microbatches = self._num_microbatches()
if schedule.num_microbatches != num_microbatches:
    raise RuntimeError(...)  # 捕获与执行计划的 microbatch 数必须一致。
return GraphSchedulePlan.build(schedule, self._graph_layout(units))
```

这里的 `units` 就是第一节介绍的 callable 分组。`GraphSchedulePlan.build()` 将 executor 的 chunk 前反向操作展开成这些 unit 的事件；开启 embedding、MTP、head 时，也把相应 unit 放到实际执行位置。

同一 chunk 的前向按 unit 的模型顺序展开，反向反序展开；若 head 在独立的 loss 操作中执行，就放在该操作处。每个事件记录方向、unit、microbatch 和 slot。一个事件可能需要启动多张层图，例如 full-layer unit 内各层的前向图按层序依次执行。

## 3.2 带符号的 order 与 microbatch slot

`GraphSchedulePlan.build()` 把事件编码为 TE 接收的 `_order`：

```python
events = tuple(cls._expand(schedule, layout))
order = tuple(event.unit + 1 if event.forward else -(event.unit + 1) for event in events)
```

正数代表某个 unit 的前向，负数代表其反向，编号加一是为了避开零。同一个正数可以重复出现，由每个 unit 的前向计数区分 microbatch；负数也按该 unit 的反向计数配对。

**slot 是 unit 内这次调用的索引**，用于选中该 microbatch 对应的图和缓冲区。它按前向出现的次序从 0 编号，反向沿用对应编号。这里的 slot 是计划中的整数，不是 encoder 实现里持有图与缓冲区的 `_GraphSlot` 对象。

例如只有一个 unit、两个 microbatch，示意顺序可以是 `[1, 1, -1, -1]`：先执行两个前向，再按 microbatch 0、1 的顺序反向。这只是说明编码方式，真实顺序必须来自本 rank 的 executor。

这里 **每个 unit 的反向按 FIFO 匹配此前前向**，不等于所有 unit 都按同一个全局编号顺序执行。计划会检查每个 unit 的调用次数、microbatch 顺序和前反向配对，发现不一致就在捕获前拒绝。

## 3.3 输入模板也按生命周期复用

manager 的 `_get_sample_arguments()` 顺着这个 order 构造第二节的 sample 输入。每遇到一次前向，它寻找已在计划中完成对应反向、且规格相同的输入缓冲区；找到就复用，否则新建。规格包括 shape、dtype、Tensor 布局，以及关键字参数的名称和 Tensor 规格。

这里是在捕获前推演生命周期，没有提前执行正式训练。前向输入可能被内部 autograd 保存并用于求梯度，所以对应反向完成前，不能让后面的前向覆盖它。manager 据此给不同调用分配或共用 sample Tensor，后续 capture 便记录这些地址。第六节还会介绍 TE 对输出和输入梯度的另一层复用。

## 3.4 两种 warmup 与真正的 capture

trainer 的预热轮次使用真实数据，正常计算 loss、反向和更新参数；同时让 executor 发布计划、可选 head 记录输入规格、optimizer 建立动量等持久状态。

随后 `create_cudagraphs()` 调用 TE。TE 在 capture 前还会对模型 callable 做额外预热，默认三次：使用 sample 输入执行前向，再提供临时构造的输出梯度执行反向，准备算子工作空间和模块状态。这里的梯度不是由真实训练 loss 算出的；这段模型预热不调用 optimizer 更新参数。

TE 预热和模型捕获产生的梯度、统计与临时状态，会在建图结束时清理。正式训练仍需用本轮数据进行 replay，所得结果才参与本轮参数更新。第七节会说明这段清理在 `create_cudagraphs()` 中的位置。

trainer 在达到 `capture_at_iteration` 且准备条件满足时建图，成功后将阈值设为 `float("inf")`。所以捕获入口虽然位于训练循环中，正常情况下也不会每轮调用。后续第八节的重捕获条件会重新设置这个时机。

# 4. TE 怎样把图接回 autograd

## 4.1 capture 留下固定输入输出和梯度

manager 将包装原模型的 callable、sample 输入、order 和 pool 句柄一起交给 TE 的 `make_graphed_callables()`。pool 指定捕获期间新分配显存所用的图内存池，第六节再解释如何共享。下面节选建图调用，省略前后用于计算通信重叠的 hooks 安装与恢复：

```python
graphs = make_graphed_callables(
    tuple(wrapper for unit in units for wrapper in unit.wrappers),
    sample_args,
    **kwargs,
)
```

其中 `kwargs` 包含 `_order`、`pool`、`sample_kwargs`，以及 TE 支持时的每个 unit 的 callable 数量和输入输出缓冲区复用选项。TE 返回的 graphed callable 可以像原模块一样被调用，其内部负责输入复制、图 replay 和 autograd 衔接。

TE 按 order 依次处理前向和反向。前向捕获的核心是：

```python
with _graph_context_wrapper(fwd_graph, pool=mempool):
    outputs = func(*args, **kwargs)
flatten_outputs, spec = _tree_flatten(outputs)
per_callable_static_outputs[per_callable_fwd_idx] = tuple(flatten_outputs)
per_callable_output_unflatten_spec[per_callable_fwd_idx] = spec
```

`_graph_context_wrapper()` 包装 `torch.cuda.graph()`，让 `func` 的 GPU 操作记录进 `fwd_graph`，并管理捕获期间的 Python 垃圾回收。`func` 的 Python 前向也会在此时运行，建立用于捕获反向的内部 autograd 关系。`spec` 保存输出原来的容器结构，后续把展平后的结果恢复成原调用接口。

到了匹配的反向事件，TE 取出这些输出，并准备固定地址的输出梯度缓冲区 `static_grad_outputs`。捕获时其中的值不代表真实训练 loss 的梯度；正式反向会先把实际梯度写到这些地址，再 replay。反向捕获的核心如下：

```python
inputs = tuple(i for i in static_input_surface if i.requires_grad)
with _none_grad_context_wrapper(inputs), _graph_context_wrapper(
    bwd_graph, pool=mempool
):
    torch.autograd.backward(
        tuple(o for o in static_outputs if o is not None and o.requires_grad),
        grad_tensors=tuple(o for o in static_grad_outputs if o is not None),
        retain_graph=retain_graph_in_backward,
    )
    grad_inputs = tuple(input.grad for input in inputs)
```

`_none_grad_context_wrapper(inputs)` 临时保存各输入原来的 `.grad` 并设为 `None`，退出时恢复；这样代码能单独取得这次反向产生的 `.grad` 引用。`_graph_context_wrapper()` 则把反向 GPU 操作记录进 `bwd_graph`。

随后 TE 为不求导的输入补上 `None`，得到与完整输入列表逐项对应的 `static_grad_inputs`。这个元组保存梯度 Tensor 的引用，供正式反向向外返回。捕获阶段用 autograd 找出反向计算；replay 直接执行已记录的 GPU 工作，不再逐层运行内部的 Python 反向代码。

## 4.2 前向 replay 只更新输入并执行图

TE 为这些图生成自定义 `Graphed(torch.autograd.Function)`。下面节选其前向，保留“输入复制—执行—返回”的路径，省略 FP8 状态和不同 stream 的同步分支：

```python
for i in range(len_user_args):
    if (
        isinstance(static_input_surface[i], torch.Tensor)
        and static_input_surface[i].data_ptr() != inputs[i].data_ptr()
    ):
        static_input_surface[i].copy_(inputs[i])

# 此处展示 replay stream 等于当前 stream 的分支。
fwd_graph.replay()
# 省略输出容器类型检查。
return tuple(o.detach() if o is not None else o for o in static_outputs)
```

复制范围是用户输入和 Tensor kwargs，参数本身沿用模块已有地址。输入已经位于对应静态地址时，可以省掉复制。

`detach()` 返回与静态输出共享存储的 Tensor，切断捕获时旧的 autograd 关系；外层调用 `Graphed.apply(...)` 又为这次正式训练建立新的自定义节点。后续反向进入的是 `Graphed.backward()`，不需要被 detach 的输出继续连着旧的内部计算图。

这里没有自动 `clone()` 整个输出。返回 Tensor 和静态输出指向同一块存储；后续图若改写该地址，前面返回的 Tensor 读到的内容也会改变。消费者必须在覆盖前用完结果，第六节会解释如何保证这个顺序。

## 4.3 反向返回输入梯度和参数梯度

反向收到外层 loss 传来的梯度后，先填固定梯度入口，再 replay。下面同样节选当前 stream 路径，省略形状数量检查、跨 stream 同步和 FP8 更新：

```python
for g, grad in zip(static_grad_outputs, grads):
    if g is not None:
        if g.data_ptr() != grad.data_ptr():
            g.copy_(grad)

bwd_graph.replay()
# 省略 FP8 状态更新和返回容器类型检查。
return (None, None, None) + tuple(
    b.detach() if b is not None else b for b in static_grad_inputs
)
```

前三个 `None` 对应 Function 的控制参数，后面依次对应 hidden、其他 Tensor 输入和模块参数。TE 在调用 Function 前实际拼接的是：

```python
func_args = tuple(flatten_user_args) + tuple(flatten_user_kwargs) + module_params
out = Graphed.apply(
    skip_fp8_weight_update, cuda_graph_stream, cuda_graph_event, *func_args
)
```

这里 `module_params` 是 callable 中参与求导的权重参数，前向把它们传入 `Graphed.apply()`，反向按相同位置返回参数梯度。反向图的 GPU 操作负责把梯度数值写到捕获地址；返回 Tensor 引用则告诉 autograd：这些数值属于哪些输入和参数。输入激活的梯度由此继续传向图外的上游模块，参数梯度由此进入累积与更新流程。

分布式数据并行的模型包装层（DDP wrapper）还可能为参数提供 `main_grad`，即指向集中梯度缓冲区的视图。开启融合梯度累积时，GPU 算子可以直接把结果累加到 `main_grad`；Function 返回的普通 `.grad` 此时还用于接通 autograd 和梯度 hook，不一定保存最终累积值。框架需要标记哪些梯度已写入 `main_grad` 并通知归约流程，避免漏累加或重复累加。

# 5. replay 如何接入流水线训练

## 5.1 安装 callable，按 microbatch 选择图

捕获完成后，manager 调用 `_install_unit_graphs()`，把 TE 返回的 graphed callable 保存到原模块的 `cuda_graphs` 列表，按 microbatch 和模块位置组织。正式训练仍从原模型的前向入口进入；入口检查配置、训练状态和图是否已安装，满足条件才调用对应 callable。

manager 还根据第三节的计划创建 `GraphReplayTracker`，保存到相关模块的 `_te_graph_tracker` 属性。tracker 用来比较实际执行与捕获计划，并在需要时安装梯度保护 hook；把它保存到属性本身不会执行检查。

VPP block 的 replay 核心如下，省略函数签名和开头的已捕获检查：

```python
cg_index = self.current_microbatch % len(self.cuda_graphs)
cudagraph_args, cudagraph_kwargs = self._get_vpp_stage_graph_replay_args(
    hidden_states, kwargs
)

tracker = getattr(self, "_te_graph_tracker", None)
if tracker is not None:
    tracker.before_launch(self._te_graph_unit, 0, self.current_microbatch)

self._register_te_vpp_stage_backward_completion_hook(cudagraph_args[0])
for hook, hook_args in self.cuda_graph_manual_hooks:
    hook(*hook_args)

cuda_graph_output = self.cuda_graphs[cg_index](*cudagraph_args, **cudagraph_kwargs)
if not torch.is_tensor(cuda_graph_output):
    cuda_graph_output = tuple(cuda_graph_output)
    assert len(cuda_graph_output) == 1, (
        "vpp_stage CUDA graph expects TransformerBlock to return one tensor, "
        f"got {len(cuda_graph_output)} outputs."
    )
    cuda_graph_output = cuda_graph_output[0]
if tracker is not None:
    tracker.after_launch(self._te_graph_unit, 0, 1, cuda_graph_output)
return cuda_graph_output
```

`_get_vpp_stage_graph_replay_args()` 把当前 hidden 和元信息整理成第二节的 Tensor 参数格式；`cg_index` 再选中当前 microbatch 对应的 callable。这里取模只是计算索引，执行顺序仍由计划约束。

`tracker.before_launch()` 核对 unit、microbatch 和模块位置。`after_launch()` 在 unit 的最后一个前向图执行后安装 hooks：反向开始时核对顺序，反向返回梯度时按需复制数据。此处 VPP unit 只有一个 block callable，所以模块索引为 0、数量为 1；第六节再解释为什么可能需要复制梯度。

full-layer 路径还要核对一个 unit 内的层顺序；同一模块存在多个逻辑调用点时，索引同时包含 microbatch 和调用点。不能把它简化成“每层只有一套可任意重复使用的图”。

## 5.2 Python 状态与 hooks 仍需外层处理

GPU graph replay 不会重新运行原模块内部的 Python forward hooks，因此框架把每轮仍需执行的回调保存在 `cuda_graph_manual_hooks` 中，在 replay 前手动调用。例如分片参数的 gather，需要在计算前把本次所需参数收集到位。

`_register_te_vpp_stage_backward_completion_hook()` 则在本次输入激活上注册梯度 hook。相关重叠配置启用时，输入梯度到达会触发权重梯度完成通知等处理。第四节的 autograd Function 把梯度传回外层，这些 hooks 再更新分布式训练所需的参数就绪状态；两部分共同接通正式训练。

指标也有相同问题。TE 的额外预热可能设置“本轮已写入”的 Python 标记；如果不清掉，capture 会跳过本应记录的统计 GPU 操作，导致正式 replay 一直没有新统计。manager 提供 `post_warmup_hook`，由 TE 在预热后、捕获前调用，用来清除这些标记。

捕获结束还会丢弃准备过程中产生的统计值。这样图中保留了正式运行需要的统计操作，日志又不会把 sample 输入产生的计数和 loss 当成本轮训练结果。

# 6. 共享显存与梯度保护

## 6.1 pool 指定分配归属，顺序决定能否复用

manager 把 `torch.cuda.graph_pool_handle()` 得到的 pool 句柄交给多个 capture 上下文，让 PyTorch 图分配器在这些捕获中使用同一个内存池。pool 句柄由 PyTorch 提供，TE 将它传给底层捕获上下文。池中不再被占用的存储，有机会用于后续捕获的新 Tensor。

句柄不是搬迁命令：捕获前已经分配的参数、输入或梯度缓冲区，传入 pool 后仍在原地址。即使所有图使用同一 pool，也不保证任意两个 Tensor 都能共用地址，分配器还受大小、stream 和仍存活引用等条件约束。

capture 期间 Python 分配和释放决定了地址布局；replay 使用已经记录的地址，不再逐个重做这些 Python 分配。这意味着即使后续 Tensor 对象仍指向某个地址，另一张图也可能按捕获布局写入同一地址。安全性依赖先前的内容已不再需要，而不是依赖 replay 时分配器再次阻止覆盖。

## 6.2 TE 如何让输出和输入梯度的存储可以复用

第二节的 `static_outputs` 和 `static_grad_inputs` 由 TE 保存。若这些 Tensor 一直维持对原存储的引用，分配器就不能把存储分给其他 Tensor。开启 `_reuse_graph_input_output_buffers` 后，TE 在特定捕获位置替换这些引用，为后续分配留下复用机会。

具体做法是：匹配的反向捕获完成后，处理该次前向输出；下一次反向捕获完成后，再处理前一次反向的输入梯度。下面节选同一 unit 内的逻辑，省略跨 unit 最后一个反向的对应处理：

```python
per_callable_static_outputs[per_callable_bwd_idx] = make_weak_ref(static_outputs)

if previous_per_callable_bwd_idx is not None:
    idx = previous_per_callable_bwd_idx
    per_callable_static_grad_inputs[idx] = make_weak_ref(
        per_callable_static_grad_inputs[idx]
    )
previous_per_callable_bwd_idx = per_callable_bwd_idx
```

`make_weak_ref()` 在这里保留访问原地址的能力，但不再通过这些保存项维持原分配的生命周期。这就是本节所说的“弱引用”：地址仍被记录，原存储却可以在其他维持它的引用也消失后被池复用。它不复制 Tensor 数据，也不能据此认为 TE 把所有输入输出都释放了。

处理前向输出，是因为对应反向已经捕获，后续建图不再需要靠它建立这次反向关系。输入梯度还要交给上游反向或流水线发送，因此多保留一段时间。这个捕获顺序约定了正式运行时的使用期限：replay 会重新生成数据，消费者必须在后续图覆盖其地址前读完。

`static_grad_outputs` 还有另一种复用：同规格的反向梯度入口可以共用缓冲区，因为每次反向前都会把本次梯度写进去。这是在多个调用之间共用一个入口 Tensor，与上面解除引用、让存储重新参与分配的机制不同。

## 6.3 图执行结束，通信不一定读完了

假设反向图 B0 产生输入梯度 G，executor 发起到上游 rank 的异步点对点（P2P）发送，随后执行 B1。按上面的捕获布局，B1 之后的某张图可能复用 G 的存储；但真实发送此时可能还在读取 G。后续图一旦覆盖该地址，通信就可能读到错误数据。**梯度已经算完，不代表消费者已经读完。**

类似问题也出现在多模态首个 PP stage：LLM 的输入梯度可能要一直保存到全部 LLM microbatch 结束，才交给 encoder 反向。不能因为某个 decoder backward 返回了，就判断这些梯度可丢弃。

所以 executor 发布的计划除了 F/B 顺序，还记录 `grad_hold_until`，表示该操作返回的梯度要保留到哪个调度位置。对重叠发送，需要保留到 executor 等待相应通信完成；对 encoder 输入梯度，可能要保留到整个 LLM 阶段结束。

`GraphSchedulePlan._borrowed()` 比较这个使用期限与 TE 的存储复用时机，把可能过早被覆盖的 `(unit, slot)` 记录为需要保护的位置。代码用 `borrowed` 表示这组输入梯度位置，用 `borrowed_parameters` 记录需要保护的共享参数梯度位置。后者也可能需要等待其他图的梯度贡献才能完成累积。

## 6.4 对使用时间较长的梯度做保护性复制

第五节的 tracker 使用这些标记安装保护 hook。unit 的第一个前向图对应最靠近其输入的计算；该图的反向返回时，产生 unit 的输入梯度。tracker 在这个 autograd 节点上注册 hook，检查各返回项对应的是输入激活还是共享参数，再查计划决定是否复制。

hook 在前向阶段注册，实际复制发生在反向算出梯度之后、梯度交给后续消费者之前。下面节选其返回部分，`copy_edges` 保存各返回项是否需要复制：

```python
return tuple(
    grad.clone() if copy and grad is not None else grad
    for grad, copy in zip(grad_inputs, copy_edges, strict=True)
)
```

复制后，executor 或梯度汇合节点持有独立存储，TE 后续覆写原地址不会破坏这份数据。不需要跨越复用边界的梯度仍直接返回，避免为所有图输出统一增加复制。

这层保护依赖 executor 如实报告数据使用期限。每个训练 step 开始时，`tracker.check_schedule()` 检查新计划；若与捕获时不同，就重新推导图的执行顺序、输入梯度保护集合和共享参数梯度保护集合，再与原计划比较。

**通信操作的位置可以变化，只要上述图顺序和保护决定没有改变。** 若这些结果改变，检查才会报错，要求删除并重新捕获；不能把“计划内容有任何差异”都理解成旧图必然失效。

## 6.5 stream 管执行依赖，不拥有显存地址

TE replay 默认使用调用处的当前 stream；调用方也可以提供 `cuda_graph_stream`。不同 stream 时，TE 先让 replay stream 等待当前 stream 的输入准备，再提交 replay，之后由当前 stream 等待对应 stream 或指定 event。

这解释了为什么 capture stream 与 replay stream 可以不同：设备地址不会因为换 stream 而变化。真正必须保证的是生产输入、执行图、消费输出之间的先后关系，以及不同图对共享地址的使用不发生非法重叠。

这里 P2P 的发送和接收由图外 executor 调度。decoder 内部的张量并行（TP）、上下文并行（CP）和专家并行（EP）通信，在实现支持时也可以进入图中；这些通信仍要满足捕获限制，并保证相关 rank 以匹配顺序发起通信。

# 7. 将 encoder 与 optimizer 接入同一捕获生命周期

## 7.1 create_cudagraphs 的两个回调位置

前面解释了 LLM 内部建图，共享 encoder 时还需要把整段流程放进更大的顺序中。`create_cudagraphs()` 接受两个回调：LLM 捕获前调用第一个，LLM 捕获及缓存清理后、optimizer 捕获前调用第二个。

下面节选生命周期骨架，省略 optimizer 图的 TE 参数构造和函数末尾的 GC/cache 清理：

```python
for step in self.optimizer_graph_steps:
    step.require_warmup()
    step.validate()
with self._capture_scope() as start_time:
    if before_model_capture is not None:
        before_model_capture()
    self._create_cudagraphs()
    self._release_capture_caches()
    if before_optimizer_capture is not None:
        before_optimizer_capture()
    # 若配置 optimizer 图，在这里用同一 pool 捕获，num_warmup_iters=0。
    self._finish_capturing(start_time)
# 省略 GC 与按条件清理缓存。
self._graphs_created = True
```

`_capture_scope()` 标记当前处于捕获阶段，通过 barrier 协调各 rank，并管理捕获期指标。它还持有 `pinned_memory_capture_lock`，与使用同一把锁的后台页锁定内存分配互斥，避免全局 CUDA capture 期间发生不允许的主机内存分配。

`_release_capture_caches()` 在 VPP 实现中解除部分 MoE 通信管理器保存的捕获缓存引用，让相关池内存储有机会用于后面的 encoder 梯度缓冲区重建。full-layer 的同名方法没有执行这套清理，所以这里不是统一释放所有模型临时 Tensor。

最后，`_finish_capturing()` 清零 DDP 梯度缓冲区，并清理 optimizer 的梯度引用以及准备期间产生的统计和临时状态。退出捕获上下文并完成清理后，`create_cudagraphs()` 将 `_graphs_created` 设为真，告诉 trainer 图已准备好，后续训练可以使用第五节的 replay 入口。

## 7.2 为什么 encoder 梯度要在 LLM 之后重建

多模态模型管理层通过上下文管理器定义 `start_ve_capture`、`finish_ve_capture`，trainer 将它们作为上面的两个参数传入。名称沿用 VE，实际可以同时处理 VE 和 AE。上下文管理器负责定义、交出回调和退出清理；真正按时调用回调的是 LLM manager。

该共享方案要求运行期各 encoder 前向先完成，再执行 LLM 前反向，最后执行 encoder 反向。捕获按对应的资源生命周期安排：

```text
start_ve_capture
  → 准备 encoder 输入和缓冲区，执行额外预热
  → 捕获 encoder 前向
  → 解除 encoder 旧 DDP 梯度存储的引用
LLM manager
  → 按 executor 计划捕获 LLM 前反向
  → 清理符合条件的 capture 缓存
finish_ve_capture
  → 在共享池捕获 encoder 梯度初始化图
  → 捕获 encoder 反向，固定梯度地址
LLM manager
  → 捕获可选的 optimizer 图，清理准备阶段的梯度与统计
```

这里释放的是 encoder 的 DDP 梯度缓冲区。DDP 此前为集中存放参数梯度分配了 `grad_data`；通信分组 bucket、各参数的 `main_grad`，以及部分 TE grouped 模块缓存的梯度视图，都可能引用这块存储。release 解除这些引用，让旧分配不再被它们占用。

传入 pool 不能把旧存储搬进图池，因此 `finish_ve_capture()` 要在使用共享 pool 的 capture 上下文中重新分配 `grad_data`，再让 bucket 和参数视图指向新存储。release 和 rebuild 都调用 `_refresh_grouped_main_grads()` 更新 TE grouped 模块缓存的视图，避免模块继续通过旧视图写梯度。

LLM 建图完成后，符合第六节生命周期的临时存储已经可以复用，新 encoder 梯度缓冲区才有机会占用这些块。之后捕获 encoder backward，记录的是新地址。仍要供后续反向使用的 LLM 或 encoder 激活不能因此随意释放。

**重建只在捕获阶段执行；正式训练每轮 replay 的梯度初始化图主要负责清零。** LLM 可能刚在重叠地址留下临时数据，encoder 梯度又使用累加，因此必须先清零。清零安排在整个 encoder/adapter 反向之前，不能等 adapter 已产生梯度后再清整块缓冲区。

具体调用关系是：`finish_ve_capture()` 将 encoder manager 的 `replay_grad_init` 方法保存到 encoder DDP wrapper 的 `_cuda_graph_prepare_grad_buffer` 属性。这个方法用于 replay 梯度初始化图；赋值只保存方法，不执行它。正式训练中，executor 在 encoder 反向前调用 `prepare_model_grad_buffer_for_backward()`，由该函数读取属性并调用，才真正执行图中的清零操作。

共享模式还将这些梯度缓冲区标为静态地址，常规 release/rebuild 路径不再每轮解除引用或重新分配。下一轮 LLM 仍可按图中记录的地址写入这些块，无需先解除 encoder 的 Tensor 引用；前提是上一轮梯度已用完，本轮 LLM 临时数据也在 encoder 清零前用完。完整的 encoder 调用关系见 [AE 笔记第六节](../28_AE_CUDA_Graph/note.md#6-与-llm-共享显存的完整链路)。

## 7.3 optimizer 图保留参数更新的原有时机

optimizer 图是独立的本地更新 callable，正式训练仍由 optimizer 的原有入口调用，在梯度准备与归约条件满足后执行。示例支持的是特定 PlannedMuon 本地路径，并非任意 optimizer 都能直接开启。

`OptimizerGraphStep` 是包装本地 optimizer 更新的可调用模块。它检查并保存参数、梯度、动量等状态的对象及规格，用于之后判断更新图能否继续使用。真实预热步骤先建立这些状态并标记 `warmed`；进入建图阶段时检查预热已完成，再用 `num_warmup_iters=0` 捕获更新，不额外执行一次 optimizer 预热。

学习率通过固定设备 Tensor 更新，其外层调用节选如下：

```python
for group, lr in zip(self.optimizer.param_groups, module.learning_rates, strict=True):
    lr.fill_(group["lr"])
    group["step"] = (group.get("step") or 0) + 1
module()
module.warmed = True
```

这里 `module` 就是 `OptimizerGraphStep` 对象：建图前调用它会执行普通参数更新，建图后调用它会进入 TE 安装的 replay 入口。每次更新前，外层把新学习率写入固定 CUDA Tensor，并增加 Python 中的 `step` 计数；其他已固定的超参数和状态规格仍须检查。模型参数和动量要跨 iter 保留，因此不能像临时激活一样让其他计算覆盖其存储。

# 8. 重捕获与正确性边界

当 global batch rampup 改变 microbatch 数量时，原来的图数量、slot 和流水线计划不再匹配，trainer 需要删除旧图，重新安排预热与捕获。共享 encoder 的容量增长也可能改变池布局，因此要在没有未完成反向的训练边界协调重建 encoder、LLM 和已启用的 optimizer 图。

删除图会移除安装的 callable 和 tracker，清理相关捕获状态；模型权重与 optimizer 持久状态继续保留，是否还需真实预热由重建路径检查。只改变静态输入的数值不需要重捕获，但 shape、地址、调度顺序或数据使用期限发生变化，不能假定旧图仍有效。

重算也受边界约束。捕获 callable 内部已支持的重算，可以被记录进相应图中。例如某些选择性重算实现会在 VPP block 内登记要保留的输入，并在反向前恢复所需中间量。普通执行中的任意 checkpoint 组合不能直接照搬到图边界两侧，因为它们可能依赖 Python 对象、旧输出或不同的调用顺序。

评估这类优化至少要区分首次 warmup/capture 成本和稳定 replay 成本，并核对前向结果、输入梯度、参数梯度、参数更新与训练统计。共享池还需覆盖未完成 microbatch、异步发送、共享权重和重捕获等生命周期情况。少了 Python launch 并不保证所有配置都更快；收益还受计算占比、输入复制、梯度保护复制、捕获粒度和显存布局影响。

本文依据源码解释这些机制，没有将仓库中的既有性能记录当成本次测量，也未据此宣称数值或吞吐已经实测验证。

# 9. Q&A
