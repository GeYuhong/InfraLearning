# 注意力后端、滑动窗口注意力与 Mask 计算

本目录研究通用 attention 计算后端、滑动窗口注意力（SWA）和 mask 的计算语义。结合 `dots-score` 的实现和提交历史，梳理不同后端如何接收模型输入、实现注意力计算，并处理精度、布局和版本兼容问题。

## 重点内容

- **后端接口**：FlashAttention 各版本、PyTorch SDPA、Transformer Engine attention 等实现的调用方式与能力差异。
- **Mask 与窗口**：causal mask、滑动窗口、attention sink 等机制的作用范围、边界和计算含义。
- **布局与元数据**：Q/K/V 布局、变长输入、序列长度信息及不同后端的接口约定。
- **数值与梯度**：softmax、精度转换、前反向一致性，以及后端切换时需要验证的行为。
- **兼容与性能**：版本适配、后端选择、融合及运行开销，追踪相关 bugfix 和性能变化。

## 与其他专题的关系

- [文档 Mask 与序列边界](../30_Document_mask/README.md) 定义文档、padding 和 packed 序列的语义；本目录关注 attention 后端如何执行这些约束。
- [DSA](../11_DSA/README.md) 研究稀疏索引与专用 core；[上下文并行](../06_Context_parallel/README.md) 研究跨 rank 的 attention 通信。
- [RoPE](../16_RoPE/README.md) 负责位置编码的构造与应用，本目录记录其与后端布局的接口关系。

后续资料可包含后端能力对照、mask 示例、关键代码入口、相关 commit，以及数值和吞吐测试案例。
