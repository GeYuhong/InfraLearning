# Encoder Packing、VE/AE 负载均衡与 VARP

本目录研究多模态编码器的 packing、视觉/音频任务负载均衡和 VARP 相关机制。结合 `dots-score` 的实现和提交历史，梳理 encoder unit 如何组织、分配到不同 rank，并在编码后恢复与原样本的对应关系。

## 重点内容

- **任务单元与 Packing**：modality/sample packing、encoder unit 和 pack factor 的含义，以及它们对批处理形状与计算量的影响。
- **任务分配**：视觉与音频任务的负载衡量、rank 分配和 VARP（Vision/Audio Rank Partitioning）相关策略。
- **数据重分布**：跨 rank 的 all-to-all、元数据传递、输出还原和反向梯度路由。
- **跨列与窗口**：跨列窗口及相关分配策略如何影响执行顺序、可用任务和负载差异。
- **正确性与性能**：样本和 media 的归属、空任务和不均匀任务的处理，以及通信成本、padding 和吞吐的取舍。

## 与其他专题的关系

- [视觉编码器](../25_Vision_encoder/README.md) 和[音频编码器](../27_Audio_encoder/README.md) 负责实际计算；本目录研究任务的组织、分发与结果归还。
- [文档 Mask](../30_Document_mask/README.md) 定义序列与 media 边界；本目录需要在任务重分布后维持这些关系。
- [EPLB](../09_EPLB/README.md) 平衡专家部署，本目录平衡视觉和音频编码任务，二者的任务对象和映射机制不同。

后续资料重点记录 unit/rank 分配图、通信与还原时序、相关 commit，以及负载统计和性能案例。
