# mHC、iHC、SimpleHC 与 Attention Gate

本目录研究 Hyper-Connections 相关的 mHC、iHC、SimpleHC（SHC）以及 attention gate。结合 `dots-score` 的实现和提交历史，理解残差流、子层输入输出和门控如何组织，以及这些结构对参数、梯度和数值行为的影响。

## 重点内容

- **连接结构**：不同 HC 变体的流布局、混合方式和参数组织，梳理数据在子层之间的传递过程。
- **共享与简化形式**：两子层共享、collapsed 形式和宽度设置等变化的计算含义与适用条件。
- **Attention Gate**：门控的位置、形状、参数初始化和梯度路径，以及与 attention 输出投影的关系。
- **精度与稳定性**：RMS epsilon、BF16 门控参数及优化器处理等细节对训练数值的影响。
- **实现演进与适配**：融合、重算、并行执行及权重映射相关的变化，检查结构改动后的参数和梯度一致性。

## 与其他专题的关系

- [MLA](../13_MLA/README.md) 研究 attention 内部投影；本目录研究流连接和门控结构。
- [ForwardContext](../19_ForwardContext/README.md) 与[激活重算](../20_Recompute/README.md) 提供执行阶段和重算背景；本目录关注 HC/Gate 接入时的具体约束。

后续资料可包含流连接示意图、门控公式、关键代码入口、相关 commit，以及不同变体的数值对照。
