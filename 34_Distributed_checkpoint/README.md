# 分布式检查点、异步保存与 USRBIO

本目录研究分布式训练状态的保存、加载和 I/O 执行，包括异步保存与 3FS USRBIO。结合 `dots-score` 的实现和提交历史，梳理多个 rank 如何协作读写 checkpoint，以及训练计算与持久化操作之间的生命周期和一致性要求。

## 重点内容

- **分布式保存与加载**：DCP/dist-ckpt、fully parallel save/load、状态分片和 rank 间的任务组织。
- **异步保存**：writer、spawn 进程、数据暂存、后台写入和完成通知，以及训练状态在保存期间的可变性。
- **I/O 后端**：文件系统 reader/writer、读写流水线、direct I/O 和 3FS USRBIO 的接口与资源管理。
- **恢复与故障处理**：不完整保存、读写失败、resume 调试和资源清理，明确何时可以认为一次保存已完成。
- **性能与验证**：数据量、并发度、带宽、CPU/GPU 内存占用，以及恢复后参数和训练进度的一致性。

## 与其他专题的关系

- [Universal Checkpoint](../33_Universal_checkpoint/README.md) 研究统一状态描述和恢复语义；本目录研究如何执行这些保存与加载操作。
- [FSDP/HSDP](../04_FSDP_HSDP/README.md) 提供分片状态背景；[数据管线](../32_Data_pipeline/README.md) 说明数据进度对恢复的要求。
- [激活重算](../20_Recompute/README.md) 研究反向所需的中间激活，本目录研究训练状态的持久化。

后续资料可包含保存与加载时序、异步 writer 生命周期、相关 commit，以及 I/O 性能和故障恢复案例。
