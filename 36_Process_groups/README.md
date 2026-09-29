# 进程组、NCCL/Gloo 与通信拓扑

本目录研究分布式训练的进程组、通信后端和拓扑组织。结合 `dots-score` 的实现和提交历史，梳理 rank 如何映射到不同并行维度，以及通信参与者、初始化顺序和后端约束如何影响整个训练框架。

## 重点内容

- **并行状态与分组**：parallel state、TP/PP/DP/CP/EP 等维度的 rank 映射，以及进程组的创建、查询和生命周期。
- **通信网格**：多维 communication grid、节点内外的通信关系，以及组与组之间的交叠。
- **通信后端**：NCCL、Gloo 和相关后端的职责、接口、配置与适用条件。
- **初始化与顺序**：多机 rendezvous、端口、建组顺序、collective 发起顺序及 NCCL launch-order 相关问题。
- **正确性与性能**：rank 映射错误、通信不匹配、等待或死锁的定位，以及拓扑对通信成本的影响。

## 与其他专题的关系

- [DDP](../03_DDP/README.md)、[上下文并行](../06_Context_parallel/README.md) 和[Dispatch overlap](../08_Dispatch_overlap/README.md) 使用这些通信组实现具体算法，本目录说明底层分组与后端约定。
- [启动与缓存](../35_Startup/README.md) 研究整体启动过程，本目录细化分布式通信资源的初始化。

后续资料可包含 rank/进程组映射图、多机初始化时序、相关 commit，以及通信顺序和拓扑问题的排查案例。
