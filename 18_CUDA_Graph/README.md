# LLM、Transformer Engine 与 VPP CUDA Graph

本目录研究语言模型训练中的 CUDA Graph 捕获与回放，包括 Transformer Engine（TE）和 VPP 相关实现。结合 `dots-score` 的实现和提交历史，梳理图的捕获范围、输入输出和存储生命周期，以及图执行如何接入训练调度。

## 重点内容

- **捕获粒度**：单层、完整层、VPP stage，以及 embedding、MTP、LM head、optimizer 等捕获边界的设计与演进。
- **捕获与回放**：静态输入、shape、地址稳定性、预热和重新捕获的要求，分析 eager 与图执行的衔接。
- **显存与梯度**：共享 graph pool、输出复用、借用梯度的所有权和生命周期，以及存储释放与重建。
- **调度与通信**：图执行与流水线、异步通信、多 stream 和计算依赖之间的关系。
- **正确性与性能**：统计副作用、重算、参数更新和回放一致性，以及捕获开销、显存和运行收益。

## 与其他专题的关系

- [视觉编码器图执行](../26_VE_CUDA_Graph/README.md) 和[音频编码器图执行](../28_AE_CUDA_Graph/README.md) 分别记录编码器特有的布局、缓冲区和回放机制。
- [ForwardContext](../19_ForwardContext/README.md) 统一执行阶段语义；[BigMac](../01_BigMac/README.md) 提供 VPP 调度背景。
- [启动与缓存](../35_Startup/README.md) 研究预热和编译准备，本目录关注它们满足的捕获条件。

后续资料重点记录捕获范围图、显存与梯度生命周期、相关 commit，以及 eager/graph 对照和性能案例。
