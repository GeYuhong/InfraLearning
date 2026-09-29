# DSA 稀疏注意力与 Indexer/Core 后端

本目录研究 DSA 稀疏注意力的计算路径，包括 indexer、top-k 选择及 attention core 后端。结合 `dots-score` 的实现和提交历史，梳理如何选择参与注意力计算的 token，以及稀疏索引如何传递给具体计算后端。

## 重点内容

- **稀疏计算流程**：indexer 打分、top-k 选择、索引组织和稀疏 attention 的前反向关系。
- **Indexer 与 Core 接口**：张量布局、索引格式、序列边界和元数据约定，以及索引生成与核心计算的职责划分。
- **后端接入**：dots-key、cuDNN、FlashMLA 等相关实现的接口、能力差异和兼容条件。
- **算子与性能**：稀疏计算中的融合、数据搬运和计算开销，以及稀疏度与性能之间的关系。
- **正确性与观测**：索引合法性、recall、确定性路径、数值对照和边界条件，追踪相关 bugfix。

## 与其他专题的关系

- [DSA warmup](../12_DSA_warmup/README.md) 研究 indexer 的训练目标和阶段切换；本目录研究稀疏选择与计算实现。
- [MLA](../13_MLA/README.md) 提供相关的 attention 结构与投影背景；[注意力后端](../15_Attention/README.md) 整理通用后端和 mask 语义。

后续资料重点记录 indexer/core 数据流、后端接口对照、相关 commit，以及稀疏计算的正确性和性能案例。
