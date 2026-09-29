# Document Mask、Packed THD 与序列边界

本目录研究文档 mask、packed 序列和序列边界的统一语义。结合 `dots-score` 的实现和提交历史，梳理多个文档或样本拼接后，attention、位置、loss 和多模态内容如何保持正确的归属关系。

## 重点内容

- **文档边界**：样本与文档的起止、可见范围，以及拼接后 attention 应满足的隔离条件。
- **Packed 表示**：THD（token、head、head dimension）布局、cu_seqlens 和其他序列元数据的构造与传播。
- **Padding 语义**：逻辑长度、物理存储、padding token 和有效 token 的区别及相互转换。
- **Loss 与 Media 归属**：训练目标、mask、图像/音频内容与原样本之间的对应，以及 SFT packing 等场景的边界处理。
- **Token 计数与并行适配**：有效 token accounting、CP 切分和数据重分布后的边界恢复，检查计数、loss 和梯度一致性。

## 与其他专题的关系

- [注意力后端](../15_Attention/README.md) 执行 mask 和变长计算；本目录定义传递给后端的文档与序列语义。
- [编码器 Packing](../29_Encoder_balance/README.md) 研究任务组织和分配；[数据管线](../32_Data_pipeline/README.md) 研究这些数据的读取和准备。
- [MTP](../17_MTP/README.md) 涉及目标 token 偏移，需要结合本目录明确跨文档和序列末尾的处理。

后续资料可包含 packed 样本示意、cu_seqlens 与 mask 示例、相关 commit，以及 token/loss 计数和边界回归案例。
