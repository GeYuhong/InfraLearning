# 音频编码器 CUDA Graph

本目录研究音频编码器的 CUDA Graph 捕获与回放，重点关注 packed Transformer、共享显存池和梯度缓冲区。结合 `dots-score` 的实现和提交历史，梳理音频图执行如何与 LLM、视觉编码器和整体训练步骤协作。

## 重点内容

- **捕获边界**：packed Transformer 的图输入输出、张量布局、shape 和序列元数据约定。
- **回放与数据更新**：静态缓冲区的填充、输入长度变化和回放顺序，以及 eager 路径与图路径的衔接。
- **显存和梯度复用**：共享 graph pool、参数和梯度缓冲区的生命周期，以及地址与存储约束。
- **联合执行**：与 LLM、VE 图共享资源或组合执行时的依赖、多 stream 协调和兼容性。
- **正确性与性能**：前反向数值、参数梯度、padding 和序列边界的验证，以及图执行收益与显存变化。

## 与其他专题的关系

- [音频编码器](../27_Audio_encoder/README.md) 解释音频模型与预处理；本目录细化图捕获和回放的执行机制。
- [LLM CUDA Graph](../18_CUDA_Graph/README.md) 与[VE CUDA Graph](../26_VE_CUDA_Graph/README.md) 提供联合图执行的对应背景。
- [ForwardContext](../19_ForwardContext/README.md) 说明捕获、回放和重算阶段的副作用语义。

后续资料重点记录音频图的数据流、共享存储和梯度时序、相关 commit，以及 eager/graph 和联合执行的回归案例。
