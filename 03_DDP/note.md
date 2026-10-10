# DDP 原理及代码走读

<a id="sec-01"></a>
## 1. DDP 概念

**DDP（Distributed Data Parallel，分布式数据并行）的核心思想是：让多个 GPU 上的模型副本处理不同的数据，再把它们算出的梯度同步起来，共同训练同一个模型。**

先区分三个名称，避免后面混淆：

| 名称 | 含义 |
|---|---|
| 数据并行，Data Parallelism | 一种并行训练思想：不同副本处理不同数据 |
| PyTorch DDP | PyTorch 对这种思想的具体实现，即 `torch.nn.parallel.DistributedDataParallel` |
| Megatron / DDP | 面向大模型训练定制的实现，额外涉及连续缓冲区、分布式优化器、流水线调度等 |

它们共享基本原理，但接口、内存布局和通信调度并不完全相同。

<a id="sec-02"></a>
## 2. 从单卡训练到多卡训练

假设在训练一个语言模型，输入是一批文本。一次最基本的训练更新通常包含四个步骤：前向计算（forward）、计算损失（loss）、反向计算（backward）、优化器更新（optimizer step）。把模型参数记作 $\theta$，梯度记作 $g$，学习率记作 $\eta$，最简单的 SGD 更新是：

$$
\theta_{\text{new}}=\theta_{\text{old}}-\eta g
$$

实际的训练是哟哦难过的优化器通常不是 SGD，而是更为复杂的 AdamW、Muon 优化器，但是执行逻辑都是通过 loss 来计算得到反传的梯度。这个流程在单卡训练中适用，在多卡训练中也同样会严格遵循。在多卡训练中，DDP 会让两张 GPU 拥有完全一致的参数副本，但是处理的数据不一样。假设当前有两份数据 A、B，再遵循前面提到的基本训练流程的情况下，GPU 0 处理数据 A 并且算出了梯度 $g_0$，GPU 1 处理数据 B 并且算出了梯度 $g_1$，此时不会立刻更新参数，因为两个 GPU 上算出的梯度数据并不一样，它们是由不同的数据 A、B 分别算出的，直接更新参数会导致两个 GPU 上的权重参数在更新后不一样。因此，两张 GPU 在更新权重参数之前，会先对梯度进行通信，每个 GPU 得到了所有 GPU 梯度的平均梯度后，每个 GPU 再更新本地存放的模型，此时就可以保证更新后的模型权重、优化器状态等数据在不同 GPU 之前依然保持完全一致。**只要起始参数、优化器状态、更新规则等保持一致，使用相同的梯度就能让各个模型副本持续保持一致。**因此，在普通 DDP 的常规训练步骤中，各张卡可以自行执行优化器更新，无须每次更新后再广播完整模型参数。

对于每个副本处理等量样本、loss 使用相同平均方式的情况，这在数学上对应于用合并后的大 batch 计算梯度。实际浮点结果不一定与单卡大 batch 逐 bit 相同，因为归约顺序、随机算子等会影响数值细节。因此，在使用完全相同的数据但 DDP size 大小不同时，训练计算得到的 loss, grad norm 往往达不到 bitwise 一致，但是数学上是等价的。

在当下使用多卡进行分布式训练时，通常采用“一张 GPU 对应一个训练进程”。每个进程运行自己的 Python 训练程序，持有自己的模型、数据加载器和优化器。假设使用两台机器进行训练，每台机器 8 张卡，那么一共有 16 训练进程。如果只使用普通的 DDP，那么 DP size 就是 16。**如果同时使用张量并行、流水线并行等技术，DP size 就不一定等于 world size。**

<a id="sec-03"></a>
## 3. DDP 的作用以及显存方面的边界

DDP 最直接的作用就是提高训练的吞吐量。假设一张 GPU 每秒可以处理 1000 个 token，使用更多 GPU 可以同时处理更多数据，即数据并行。但是增加 GPU 的数量通常不意味着训练速度线性增长，通信、数据加载、各个 GPU 之间计算不均衡，都可能降低扩展 GPU 数量的吞吐增长收益。

我们这里聊的 DDP 是最朴素的概念，与实际的分布式训练框架（如 Megatron）中的 DDP 不完全一致，一个很重要的区别是，**普通 DDP 会复制模型训练状态，不会自动把它们均分到多张卡**。普通的 DDP 会要求每个 GPU 上保存一份模型参数、一份参数梯度、一份本地优化器状态、当前数据对应的中间激活、一些临时计算和通信缓存区。这里优化器状态是否复制，取决于所使用的优化器方案；DDP wrapper 本身并不负责管理优化器状态。因此，如果模型参数和优化器状态本身已经超过单卡容量，仅增加普通 DDP 的卡数，通常无法解决问题。固定全局 batch 时，增加卡数可以减少每卡处理的数据，从而减少激活显存，但不会自动分片参数和优化器状态。以一种常见的混合精度 Adam 训练布局为例：

| 数据 | 每个参数大致占用 |
|---|---:|
| BF16 计算参数 | 2 字节 |
| 梯度 | 2 或 4 字节 |
| FP32 主参数 | 4 字节 |
| Adam 的两份 FP32 状态 | 8 字节 |
| 合计 | 约 16～18 字节 |

这只是特定布局的估算，不是所有框架的固定公式。例如，PyTorch autocast 并不意味着参数本身一定以 BF16 常驻。按上述布局估算，7B 模型仅这些训练状态就约为 **112～126 GB**，还没算激活和额外缓冲区。这也是大模型训练经常需要优化器状态分片、FSDP、TP、PP 或 offload 的原因。

<a id="sec-04"></a>
## 4. DDP 中的通信操作

### 4.1 常规通信

**DDP 一定涉及通信，最核心的通信对象是梯度。** 最常见的通信操作叫 AllReduce。普通 PyTorch DDP 默认实现的效果是对梯度进行平均。底层通信可以先执行求和，再由框架完成相应缩放；不要把“NCCL 的 AllReduce”直接理解成“总是自动取平均”。使用自定义通信 hook，或者其他框架的定制归约逻辑时，缩放规则也可能变化。

实际生产使用的 Megatron 或 dots-score 分布式训练框架的 DDP 通信会更加复杂一些，通信类型也更加多，后者是基于前者扩展的自定义 DDP，均区别于 Pytorch 的 `torch.nn.parallel.DistributedDataParallel`。接下来将介绍主要可能出现的几种通信情况。

**第一种：普通梯度 AllReduce。**

当相关参数没有被后面介绍的 Even 路径接管，且同时满足：

```text
use_distributed_optimizer = False
grad_reduce_to_owner = False
```

DDP 对梯度执行 **AllReduce**。通信后，同一数据并行组内的每个 rank 都拿到完整的归约后梯度。这里的 rank 可以先理解成一个参与训练的进程，通常对应一张 GPU。这个执行路径与之前提到的朴素 DDP 的逻辑基本一致。

**第二种：标准 Distributed Optimizer 的梯度 ReduceScatter。**

当启用 `use_distributed_optimizer=True` 时，常规 DDP bucket 路径对梯度执行 **ReduceScatter**，使每个 rank 拿到自己负责的梯度分片。这个非常常用，因为使用分布式优化器时，不同 DP 间不会重复保存同一份优化器状态，而优化器状态又是静态显存中开销最可观的部分，因此可以节省不少显存。具体来说，优化器状态的不同分片分散在不同的 DP rank 上，彼此之间保存的数据不会有重复，在每个 rank 计算得到本地的梯度 grad 后，会对 grad 进行 reducescatter 通信而不是 allreduce，每个 rank 会获得进行了规约计算后的 grad 的一个分片，并且该分片和每个 rank 保存的优化器状态分片是同一个分片，因此做完 reducescatter 后每个 rank 能够更新当前分片所对应的优化器状态分片和 fp32 权重分片，之后再对 fp32 权重分片进行一次 allgather，让每个 rank 依然能够获取到完整且完全相同的权重。这个流程进行了一次 reducescatter 和一次 allgather，与第一种情况中进行一次 allreduce 通信的开销是一样的，但是显存得到了节省。这里有一个可以进行性能优化的点，就是参数的 allgather 可以与后续的前向计算 overlap，因此需要确保某一层使用参数之前，该层所需的参数已经准备就绪。

【todo】
- param_and_grad_buffer.py：标准 Distributed Optimizer 的参数 AllGather。路径：`/newcpfs/user/yimeng/keep_learning/dots-score/megatron/core/distributed/param_and_grad_buffer.py`（`start_param_sync()` 函数入口第 290 行；实际 AllGather 调用第 325 行。原第 288 行标注有误，更正说明见回复记录 06）。

此外，如果还配置了多个 Distributed Optimizer instance，流程会更复杂：先在 instance 内 ReduceScatter，再在 instance 之间对对应的梯度分片执行 AllReduce。

**第三种：梯度 Reduce 到参数的 owner。**

当相关参数没有被 Even 接管，且启用 `grad_reduce_to_owner=True` 时，相关梯度通过 **Reduce** 归约到负责该参数的 owner rank。

这里的 Reduce 与 AllReduce 不同：归约结果只需要在 owner 上可用。该开关不能与 `use_distributed_optimizer=True` 同时启用。

**第四种：Even 路径接管梯度通信。**

启用 `layer_wise_muon_even_collectives=True` 时，相关 DDP bucket 会跳过常规梯度归约，由优化器中的 Even 通信逻辑接管。这条路径是 dots-score 框架自定义的一个实现，使用 **ReduceScatter**；如果使用 Hybrid 配置，还会在 duplicate 维度上对对应梯度分片执行 **AllReduce**。组大小为 1 时，相应通信可能省略。因此，**不能只看到 `use_distributed_optimizer=False`，就断言 dots-score 一定使用 AllReduce**，还必须检查 owner 和 Even 等配置。

源码入口【todo】：

- param_and_grad_buffer.py：梯度同步调度函数 `start_grad_sync()`。路径：`/newcpfs/user/yimeng/keep_learning/dots-score/megatron/core/distributed/param_and_grad_buffer.py`（函数从第 399 行开始；第 474 行进入 owner Reduce 分支，第 497 行调用 ReduceScatter，第 505 行调用普通 AllReduce，第 527 行调用跨 instance 的分片 AllReduce）。
- layer_wise_owner_reduce.py：Reduce 到 owner。路径：`/newcpfs/user/yimeng/keep_learning/dots-score/megatron/core/distributed/layer_wise_owner_reduce.py`（第 399 行）。
- layer_wise_even_collectives.py：Even 梯度通信。路径：`/newcpfs/user/yimeng/keep_learning/dots-score/megatron/core/optimizer/layer_wise_even_collectives.py`（`launch_prepared_grad_sync()` 函数入口第 274 行；实际 ReduceScatter 调用第 297 行；本函数中直接提交的 Hybrid AllReduce 调用第 326 行，延后提交 AllReduce 的路径不执行这一行）。

### 4.2 确保 DP 间参数副本一致性的 broadcast 通信

不同框架的 DDP 实现细节是有差异的。在 DDP 相关的通信行为中，Pytorch 会在默认的训练流程构建中识别是否启用了 `data_parallel_random_init`，当启用时则会调用 DDP 的 `broadcast_params()`，将参数广播到对应的数据并行组内，使副本的初始参数一致。但是这个 broadcast 操作在 Megatron 框架和 dots-score 框架中并不是唯一确保参数副本一致的方法。以 dots-score 框架为例，考虑随机初始化和 load checkpoint 两种场景：当随机初始化权重时，模型的 llm backbone 部分是通过伪随机数生成器（可以理解为给定一个起始状态，生成器就按确定的顺序产生一串数。两个进程使用相同种子、相同初始化算法，并以相同顺序、相同形状调用随机操作，就能各自生成相同的参数张量，不需要先由一个进程生成再发送给另一个进程。）进行初始化的，实际随机种子的代码位于：

【todo】
```text
/newcpfs/user/yimeng/keep_learning/dots-score/cybertron/cybertron/trainer/initialize.py
```

第 478 行是 `_set_random_seed()` 的定义；第 488 行计算基础种子，第 490～491 行决定是否加上 DP rank 偏移：

【todo】
```python
seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
if data_parallel_random_init:
    seed = seed + (10 * mpu.get_data_parallel_rank())
```

在框架使用时，通常只保留 PP rank 偏移，没有 DP rank 偏移。比如，同在 PP stage 1 的两个 DP 副本，基础种子都是 `1234 + 100 × 1 = 1334`。它们处理的训练数据可以不同，初始化这份模型分片时却从相同随机状态出发。每个进行相应的 Python、NumPy、PyTorch 的随机种子也基于此设置，以及调用的 Megatron 的 CUDA 随机状态初始化函数也再次被设置，其在设置全局的 seed 时，还会维护不同用途的随机流：

【todo】
```text
/newcpfs/user/yimeng/keep_learning/dots-score/megatron/core/tensor_parallel/random.py
```

使用相同 seed 进行随机初始化需要注意，如果某个 rank 提前多消耗了一段相同随机流，或者按 DP rank 走了不同的参数初始化分支，后续随机参数就可能不同，无法保证 dp 间的参数副本一致。

**而对于 load checkpoint 场景，不同 dp rank 会进行分工，分别 load 不同的 checkpoint，然后以 param 为粒度进行逐参数的 broadcast 通信。**

区别于 llm backbone，目前 dots-score 框架会通过 broadcast encoder 部分的权重参数以确保 dp 间参数副本一致。具体的代码是：

【todo】
```text
/newcpfs/user/yimeng/keep_learning/dots-score/cybertron/cybertron/models/multimodal/gptv_model_hybrid_omni.py
```

第 86～88 行限制构造阶段／执行 rank；第 90 行构造 `AudioModel`；**第 92 行第一次调用 `broadcast_visual_model(audio_model, role="audio")`**。这个函数名称虽然带有 visual，但这里明确复用于音频模型。

接着，第 106 行加载额外音频权重；第 107 行检查 `not ae_use_megatron_fsdp`，当前成立；**第 108 行再次广播整个 audio model**。

这两个组件最终共用的广播实现位于：

【todo】
```text
/newcpfs/user/yimeng/keep_learning/dots-score/cybertron/cybertron/utils/hybrid_util.py
```

如果 vision model 和 audio model 中的部分权重选择 load checkpoint，而另一部分权重选择随机初始化的话，目前的流程是先进行统一的随机初始化，然后进行 broadcast 通信，这两步和前面提到的 from scratch 场景的做法是一致的，最后会多出一个 load  执行checkpoint 的行为。此时没有 load checkpoint 的权重参数通过 broadcast 保证参数副本一致，而 load checkpoint 的权重参数则是通过确保 load 的数据是同源的以确保副本一致。

### 4.3 非 Even 的 Layer-wise owner 路径的参数 broadcast 通信【todo】

对于使用 owner 梯度归约的这条路径，可以这样理解：

```text
各卡计算梯度
    → 梯度 Reduce 到 owner
    → owner 更新参数
    → owner 将更新后的参数 Broadcast 给其他相关 rank
```

- layer_wise_optimizer.py：更新后发布参数。路径：`/newcpfs/user/yimeng/keep_learning/dots-score/megatron/core/optimizer/layer_wise_optimizer.py`（`_publish_model_params()` 函数入口第 571 行；第 575 行调用 runtime 的 `sync_params()`；第 577 行调用非 Even 路径的 `self.broadcast_params()`。后两处是参数同步的调度调用，底层 collective 在其他函数中执行）。
- param_contiguous_buffer.py：连续参数 buffer 的广播。路径：`/newcpfs/user/yimeng/keep_learning/dots-score/megatron/core/optimizer/param_contiguous_buffer.py`（`broadcast_rank_params()` 函数入口第 536 行；连续 buffer 分支的实际 `torch.distributed.broadcast(...)` 调用第 569 行）。

### 4.4 Even 路径中的参数 allgather【todo】

Even 相关流程可以概括为：

```text
梯度 ReduceScatter
    → Hybrid 配置下，再对对应梯度分片 AllReduce
    → 更新各自负责的参数
    → 参数 AllGather
```

- layer_wise_even_collectives.py：Even 参数 AllGather。路径：`/newcpfs/user/yimeng/keep_learning/dots-score/megatron/core/optimizer/layer_wise_even_collectives.py`（`_launch_param_bucket()` 函数入口第 330 行；实际 `self.all_gather_func(...)` 调用第 349 行）。

Hybrid 中需要分别理解两个维度：梯度 ReduceScatter 沿 shard 维度执行，对应梯度分片的 AllReduce 沿 duplicate 维度执行；**参数 AllGather 只沿 shard 维度执行**。

不要将非 Even 的参数 Broadcast 与 Even 的参数 AllGather 混为同一条路径。

### 4.5 配套统计和初始化的通信【todo】

dots-score 框架中还存在一些其他和 DDP 相关的通信，主要涉及 有效 token 数、MoE expert token 计数、梯度范数和溢出标记等辅助通信。这些行为需要按调用位置区分：有些发生在初始化，有些受模型或优化器配置控制，有些只在调试时执行，不能全算成每一步 DDP 的固定通信开销。

- finalize_model_grads.py：有效 token 数同步。路径：`/newcpfs/user/yimeng/keep_learning/dots-score/megatron/core/distributed/finalize_model_grads.py`（第 477 行检查 `num_tokens is not None`；第 482 行在 PP 组内 Broadcast；第 485 行在 DP+CP 组内 AllReduce）。