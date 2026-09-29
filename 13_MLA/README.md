# MLA、O-LoRA 与 Q/K 归一化控制

本目录研究多头潜在注意力（Multi-head Latent Attention，MLA）、O-LoRA 及 Q/K 相关的归一化和数值控制。结合 `dots-score` 的实现和提交历史，梳理压缩表示、投影与注意力计算之间的关系，以及结构和数值改动如何影响训练。

## 重点内容

- **MLA 结构**：Q、K、V 的投影关系、KV 压缩维度、注意力头布局，以及前反向张量形状。
- **O-LoRA**：输出侧低秩投影的参数组织、计算路径，以及与门控、重算和融合实现的配合。
- **归一化与数值控制**：Q/K 归一化、qk clip、K/RoPE norm 等机制的作用位置和梯度语义。
- **实现与适配**：head split、投影融合、参数布局及相关接口变化，分析不同实现的等价条件。
- **正确性与演进**：数值稳定性、参数梯度、边界 shape 和相关 bugfix，串联结构变化与兼容适配。

## 与其他专题的关系

- [上下文并行](../06_Context_parallel/README.md) 研究 Latent TP/CP 的跨 rank 布局与通信；本目录解释相关模型结构。
- [RoPE](../16_RoPE/README.md) 研究位置编码；[DSA](../11_DSA/README.md) 研究相关结构上的稀疏选择与计算。
- [Hyper-Connections 与门控](../14_HyperConnections/README.md) 整理连接和门控机制，与本目录的投影路径相互引用。

后续资料重点记录投影结构图、张量 shape、前反向推导、相关 commit，以及数值与性能对照。
