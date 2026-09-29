# InfraLearning

本仓库围绕 `dots-score` 分布式训练框架组织学习资料。以下 36 个专题来自截至 2026-09-29 的 `main` 分支提交历史梳理，编号与已筛选的 feature 清单对应。

每个目录的 `README.md` 说明研究范围、重点内容和相关专题。后续可在对应目录积累原理说明、代码阅读、commit 演进、问题复现和性能分析。跨专题的内容通过相互引用关联。

## 专题导航

| 目录 | 主题 |
|---|---|
| [01_BigMac](01_BigMac/README.md) | PP/VPP 与 BigMac 调度执行 |
| [02_DP_overlap_in_vpp](02_DP_overlap_in_vpp/README.md) | VPP 中的优化器 DP 通信重叠 |
| [03_DDP](03_DDP/README.md) | DDP、梯度同步与参数缓冲区 |
| [04_FSDP_HSDP](04_FSDP_HSDP/README.md) | FSDP/HSDP 分片训练 |
| [05_Vocab_parallel](05_Vocab_parallel/README.md) | 张量、序列与词表并行 |
| [06_Context_parallel](06_Context_parallel/README.md) | 上下文并行、Latent TP/CP 与 Ulysses |
| [07_MoE](07_MoE/README.md) | MoE 架构、路由与负载均衡损失 |
| [08_Dispatch_overlap](08_Dispatch_overlap/README.md) | DeepEP 等后端与 Dispatch 重叠 |
| [09_EPLB](09_EPLB/README.md) | 专家放置、副本与负载均衡 |
| [10_Muon](10_Muon/README.md) | Muon、PlannedMuon 与 Even Muon |
| [11_DSA](11_DSA/README.md) | DSA 稀疏注意力与 Indexer/Core 后端 |
| [12_DSA_warmup](12_DSA_warmup/README.md) | DSA Warmup、FlashKL/LightningKL 与阶段切换 |
| [13_MLA](13_MLA/README.md) | MLA、O-LoRA 与 Q/K 归一化控制 |
| [14_HyperConnections](14_HyperConnections/README.md) | mHC、iHC、SimpleHC 与 Attention Gate |
| [15_Attention](15_Attention/README.md) | 注意力后端、滑动窗口与 Mask 计算 |
| [16_RoPE](16_RoPE/README.md) | RoPE、YaRN 与位置编码 |
| [17_MTP](17_MTP/README.md) | 多 Token 预测 |
| [18_CUDA_Graph](18_CUDA_Graph/README.md) | LLM、Transformer Engine 与 VPP CUDA Graph |
| [19_ForwardContext](19_ForwardContext/README.md) | 统一执行上下文与重算语义 |
| [20_Recompute](20_Recompute/README.md) | 激活检查点与选择性重算 |
| [21_CPU_offload](21_CPU_offload/README.md) | 激活、优化器状态卸载与显存优化 |
| [22_Mixed_precision](22_Mixed_precision/README.md) | FP8、MXFP8 与 BF16 混合精度 |
| [23_Fused_kernels](23_Fused_kernels/README.md) | 融合算子与底层 Kernel 优化 |
| [24_VLM_Omni](24_VLM_Omni/README.md) | VLM/Omni 多模态模型与训练集成 |
| [25_Vision_encoder](25_Vision_encoder/README.md) | 视觉编码器、MoEViT 与性能优化 |
| [26_VE_CUDA_Graph](26_VE_CUDA_Graph/README.md) | 视觉编码器图捕获与回放 |
| [27_Audio_encoder](27_Audio_encoder/README.md) | 音频编码器、预处理与 TTS/DiT |
| [28_AE_CUDA_Graph](28_AE_CUDA_Graph/README.md) | 音频编码器图捕获与回放 |
| [29_Encoder_balance](29_Encoder_balance/README.md) | Encoder Packing、VE/AE 负载均衡与 VARP |
| [30_Document_mask](30_Document_mask/README.md) | Document Mask、Packed THD 与序列边界 |
| [31_SamplePlan](31_SamplePlan/README.md) | SamplePlan、分层采样与多数据源策略 |
| [32_Data_pipeline](32_Data_pipeline/README.md) | 数据预处理、加载、预取与 Mmap/Cache |
| [33_Universal_checkpoint](33_Universal_checkpoint/README.md) | 通用检查点 |
| [34_Distributed_checkpoint](34_Distributed_checkpoint/README.md) | 分布式检查点、异步保存与 USRBIO |
| [35_Startup](35_Startup/README.md) | 启动、预热与编译/Autotune 缓存 |
| [36_Process_groups](36_Process_groups/README.md) | 进程组、NCCL/Gloo 与通信拓扑 |

## 基础笔记

- [FlashAttention 与 GPU 基础笔记](flash_attention.md)
- [待进一步了解的问题](need_to_know.txt)
