# 多 Token 预测（MTP）

本目录研究多 Token 预测（Multi-Token Prediction，MTP）的模型结构和训练接入。结合 `dots-score` 的实现和提交历史，梳理预测层、目标 token、辅助 loss 和共享参数的关系，以及它们如何参与分布式训练。

## 重点内容

- **模型与目标**：MTP 层的输入输出、预测步数和目标对齐方式，以及与主模型计算的衔接。
- **Loss 与梯度**：辅助 loss 的归一化、mask 和梯度传播，关注序列末尾及文档边界的处理。
- **共享参数**：embedding 等共享权重的使用和梯度累积，分析 first-microbatch 等执行边界。
- **并行接入**：CP、PP、VPP 下的数据布局、层分布和依赖，以及跨 stage 的参数与梯度处理。
- **工程适配**：checkpoint、权重映射、CUDA Graph 和重算相关的改动与正确性验证。

## 与其他专题的关系

- [BigMac](../01_BigMac/README.md) 负责流水线调度；本目录研究 MTP 接入流水线的具体要求。
- [文档 Mask](../30_Document_mask/README.md) 说明目标和边界语义；[CUDA Graph](../18_CUDA_Graph/README.md) 研究包含 MTP 的捕获与回放。

后续资料可围绕 MTP 数据流、目标对齐示例、loss 与共享梯度推导、相关 commit 和回归案例展开。
