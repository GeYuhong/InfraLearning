# Muon、PlannedMuon 与 Even Muon 优化器

本目录研究 Muon 优化器及 PlannedMuon、Even Muon 等相关实现。结合 `dots-score` 的实现和提交历史，梳理参数更新的数值过程、分布式参数组织方式，以及算法、通信和工程优化之间的关系。

## 重点内容

- **算法与数值过程**：Muon 更新、Newton–Schulz 迭代、精度选择和数值稳定性，以及实现与公式的对应关系。
- **参数处理**：参数分组、矩阵形状、QKV/SwiGLU 拆分等规则，以及这些规则对更新结果和执行效率的影响。
- **分布式实现**：PlannedMuon、Even Muon 的组织方式，layer-wise owner、bucket 和参数/梯度分布，以及更新所需的通信。
- **实现优化**：Triton 等算子优化、批处理与缓冲区管理，以及计算效率和显存占用的取舍。
- **正确性与演进**：数值监控、分布式结果一致性、边界条件和相关 bugfix，串联不同实现的开发与适配过程。

## 与其他专题的关系

- [VPP 中的 DP 通信重叠](../02_DP_overlap_in_vpp/README.md) 负责通信与 VPP 执行的具体配合；本目录负责解释优化器需要哪些数据、如何更新，以及 owner/bucket 的含义。
- [BigMac](../01_BigMac/README.md) 负责整体流水线调度；涉及两者交互时，本目录记录优化器的调用条件和生命周期要求。

后续资料可围绕更新公式、参数与 owner 布局、关键代码入口、相关 commit，以及数值对照和性能分析展开。
