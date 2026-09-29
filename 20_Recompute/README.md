# 激活检查点与选择性重算

本目录研究 activation checkpointing 和选择性重算：前向时减少保存的中间激活，在反向需要时重新计算，以降低显存占用。结合 `dots-score` 的实现和提交历史，梳理重算边界、张量生命周期和梯度语义。这里的 checkpoint 指激活重算机制。

## 重点内容

- **重算策略**：整层、局部和选择性重算的边界，分析哪些中间结果保留、哪些重新生成。
- **无输出检查点**：checkpoint without output 的接口、输出释放与重建，以及下游消费者的依赖。
- **Autograd 与随机数**：前反向衔接、RNG 状态、dropout 等随机操作的重现，以及梯度一致性。
- **生命周期与副作用**：激活引用、缓冲区复用、统计和状态更新在重算中的行为。
- **组合与取舍**：与并行执行、CUDA Graph、CPU offload 和融合算子的交互，衡量节省的显存与新增计算。

## 与其他专题的关系

- [ForwardContext](../19_ForwardContext/README.md) 定义重算阶段与副作用约定；本目录关注重算策略及其执行机制。
- [CPU offload](../21_CPU_offload/README.md) 通过数据搬移管理显存，两者可以在不同张量上组合使用。
- [通用检查点](../33_Universal_checkpoint/README.md) 和[分布式检查点](../34_Distributed_checkpoint/README.md) 研究训练状态的保存与恢复，与本目录的激活重算含义不同。

后续资料重点记录激活生命周期、重算前反向时序、相关 commit，以及显存、运行时间和梯度对照。
