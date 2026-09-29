# DSA Warmup、FlashKL/LightningKL 与阶段切换

本目录研究 DSA indexer 的 warmup 训练及其向稀疏训练阶段的切换。结合 `dots-score` 的实现和提交历史，理解教师信号、KL 损失和梯度路径如何训练 indexer，以及切换阶段时哪些状态和执行行为需要调整。

## 重点内容

- **训练目标**：teacher attention、KL 与 log-sum-exp（LSE）相关信息的含义，以及 indexer 学习目标的构造。
- **KL 实现**：FlashKL、LightningKL 的输入输出、数值处理、前反向计算和接口适配。
- **梯度与统计**：warmup loss、梯度归一化、grad norm 和参数更新范围，排查梯度遗漏或重复计算。
- **阶段切换**：warmup 到 sparse 的配置、状态和执行路径变化，以及切换边界的恢复与验证。
- **组合适配**：与 MoE、EPLB、重算和图执行的交互，利用参考结果检查不同阶段的计算语义。

## 与其他专题的关系

- [DSA](../11_DSA/README.md) 研究稀疏 indexer/core 的计算机制；本目录重点是训练目标和阶段生命周期。
- [启动与缓存](../35_Startup/README.md) 中的 warmup 指运行时预热；本目录的 warmup 是 DSA 的训练阶段。
- [ForwardContext](../19_ForwardContext/README.md) 说明执行阶段与副作用管理，可用于分析 warmup 统计在重算和图执行中的行为。

后续资料可围绕 KL 公式、梯度路径、阶段切换时序、相关 commit 和参考数值对照展开。
