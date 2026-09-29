# MoE Dispatch 后端与通信重叠

本目录研究 MoE 的 token dispatch/combine，以及 DeepEP、HybridEP、UltraEP 等专家并行通信后端。结合 `dots-score` 的实现和提交历史，梳理 token 如何到达目标专家、计算结果如何返回，以及这些通信如何与专家和 shared expert 的计算重叠。

## 重点内容

- **数据流与接口**：token 的重排、发送、接收和还原；dispatch/combine 的输入输出、路由元数据和前反向对应关系。
- **通信后端**：DeepEP、HybridEP、UltraEP 的接入方式、接口差异和适用条件，以及专家并行中的 all-to-all 等通信。
- **重叠执行**：通信与专家计算、shared expert 计算的依赖和时序，以及异步执行所需的同步机制。
- **反向与重算**：dispatch/combine 的梯度路径、重算行为，以及与延迟权重梯度计算等机制的配合。
- **组合适配与性能**：CUDA Graph 下的 stream 安全、缓冲区生命周期，以及通信量、等待时间和重叠收益的分析。

## 与其他专题的关系

- [MoE](../07_MoE/README.md) 负责模型、路由和损失；本目录从路由结果出发，研究 token 的设备间传输与执行。
- [EPLB](../09_EPLB/README.md) 负责逻辑专家到物理位置及副本的映射；本目录关注 dispatcher 如何使用这些映射。
- [VPP 中的 DP 通信重叠](../02_DP_overlap_in_vpp/README.md) 处理梯度归约和参数收集；本目录处理专家并行中的 token 通信。

后续资料可围绕 dispatch/combine 流程图、后端接口对照、相关 commit，以及通信时序、故障复现和性能测试展开。
