# 上下文并行、Latent TP/CP 与 Ulysses

本目录研究上下文并行（CP）及相关的 Latent TP/CP、Ulysses 实现。结合 `dots-score` 的实现和提交历史，梳理长序列如何分布到不同 rank，以及分布式 attention 如何交换数据并保持位置、mask 和梯度语义一致。

## 重点内容

- **序列切分与布局**：连续切片等分布方式、rank 与 token 的对应关系，以及不同布局之间的转换。
- **Attention 通信**：CP 所需的数据交换、Ulysses 的 all-to-all 与序列/注意力头布局转换，以及通信与计算重叠。
- **Latent TP/CP**：潜在表示及投影相关计算的并行划分、张量布局和通信边界。
- **复杂序列语义**：packed THD、变长序列、序列边界、位置编码、mask，以及滑动窗口 attention 的 halo 数据交换。
- **正确性与性能**：重分片和反向梯度恢复、跨 rank 边界条件，以及长上下文下的通信开销、显存和吞吐。

## 与其他专题的关系

- [张量、序列与词表并行](../05_Vocab_parallel/README.md) 提供 TP/SP 的基础及算子语义；本目录关注它们与 CP 组合时的布局和通信。
- MLA、注意力后端、位置编码及文档 mask 的细节，按本专题需要引用；这里重点说明它们在跨 rank 序列计算中的适配要求。

后续资料可围绕 token/rank 布局图、attention 通信流程、相关 commit，以及不同切分方式下的正确性和性能案例展开。
