# 音频编码器、音频预处理与 TTS/DiT

本目录研究音频编码器（Audio Encoder，AE）、音频预处理，以及相关的 TTS/DiT 模型接入。结合 `dots-score` 的实现和提交历史，梳理音频如何转成模型输入、编码表示或语音生成目标，并理解各阶段的训练与计算语义。

## 重点内容

- **音频预处理**：log-Mel 等特征、音频切块、长度和 mask 信息的生成，以及预处理与模型输入的接口。
- **编码器实现**：DotsAudioEncoder、Whisper 等相关模块的卷积、Transformer 和输出表示。
- **批处理与优化**：audio chunk、卷积 batching、变长输入和缓冲区组织对效率与正确性的影响。
- **语音任务**：voice tokenizer、文本转语音（TTS）、扩散 Transformer（DiT）及音视频任务中的模型和数据关系。
- **训练与适配**：梯度、归一化、精度、序列边界和分布式执行的检查，串联相关功能开发与 bugfix。

## 与其他专题的关系

- [AE CUDA Graph](../28_AE_CUDA_Graph/README.md) 研究音频编码器图执行；本目录研究音频计算和输入语义。
- [VLM/Omni](../24_VLM_Omni/README.md) 研究音频模块接入整体模型；[编码器负载均衡](../29_Encoder_balance/README.md) 研究音频任务分配和 packing。

后续资料可包含音频处理链路、长度与 mask 示例、关键代码入口、相关 commit，以及数值和性能对照。
