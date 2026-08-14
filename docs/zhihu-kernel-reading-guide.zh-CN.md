# 知乎「算子优化」收藏夹调研与 Kernel 阅读路线

## 调研范围

本文整理知乎收藏夹[「算子优化」](https://www.zhihu.com/collection/1003017222)
在 2026-08-14 可见的 31 条内容。其中 30 条与 GPU Kernel、编译器或推理系统
相关；《2026 年，美元潮汐怎么玩不动了？》与主题无关，未纳入分析。

这 30 篇文章并不都属于 MegaKernel。为了避免把“Kernel 相关”误写成
“MegaKernel 核心工作”，本文将它们分为五条阅读路线：

1. MegaKernel、持久化调度与算通融合；
2. GEMM、Attention 与算子优化；
3. Triton、CuTe、TileLang、CUTLASS 与编译抽象；
4. 性能分析、设备内 Profiling 与资源分区；
5. MoE、推理引擎与模型/系统协同。

主 README 的精选目录只收录第一类中具有直接 MegaKernel 证据的文章；其余条目
保留在本导航中，作为实现和评测所需的知识底座。

## 先读：MegaKernel、持久化调度与算通融合

### [MegaKernel是创新还是传销？](https://www.zhihu.com/question/2013258505231050695/answer/2071314457918183125)

作者“是小肖啊”实现 MegaRTP，并在 B300 上将其与 Fusion、Multistream、PDL
逐层对照。文章最重要的贡献是收益归因：独立分支并行和 Consumer 提前准备并非
MegaKernel 专属；真正需要 MegaKernel 的是跨算子 Tile 依赖、设备内 Tile
调度与常驻 CTA 软件流水。该案例中 `M=16`、`blockM=16`，关键 GEMM 只有一个
M Tile，因此细粒度依赖几乎没有额外释放窗口。

阅读边界：测量的是 GLM-5.2-FP8 Decode 中进入 Sparse FlashMLA 之前的一段
Attention-pre 子图，不包含 TopK、Sparse FlashMLA 或完整模型端到端延迟。

### [如何看待 Hazy Research 团队将 1B 模型的 Forward 融合为一个 MegaKernel？](https://www.zhihu.com/question/1911094042047000841/answer/1924145663601542224)

从动机、执行模型、代码结构、价值与局限五个角度解读 Hazy Research 的
Llama-1B Megakernel。适合作为中文入口，并与 Hazy 原始博客和仓库交叉阅读。

阅读边界：它是第三方代码解读，事实与性能结论应回到 Hazy 的原始文章和制品
核对；“整个 Forward”也需要检查 LM Head、采样等阶段是否处于同一 Kernel。

### [megakernel 的 sync 开销](https://zhuanlan.zhihu.com/p/2054985214791775283)

对比 KOG MI300X Monokernel 与 Fleet 的同步层级：Naive 方案经 HBM，KOG 尽量
落在 LLC，Fleet 进一步利用 LLC/L2 层级。文章同时提出必要的反证：同步状态本身
的数据量很小，不能仅凭缓存层级就断言它是主瓶颈。作者在 H20、DeepSeek-V3、
`bs=2` Decode 的一次测量中观察到等待 Dispatch、GEMM0 等同步阶段合计约占
总时长 2%。

阅读边界：这组 H20 测量不能直接反驳 AMD 多 Die 场景。需要继续区分轮询延迟、
同步元数据流量、缓存一致性、跨 Die 访问和全局 Barrier 对关键路径的影响。

### [Megakernel](https://zhuanlan.zhihu.com/p/2059950781344789216)

以 MegaQwen 仓库为对象，对照 `csrc/kernels/` 的逐算子实现与
`csrc/megakernel/` 的 Transformer-block 融合实现。它的价值在于用同一仓库
展示两种执行模型，而不是比较彼此无关的代码基线。

阅读边界：MegaQwen 是模型和硬件针对性较强的学习制品；需要分别检查主
Persistent Kernel、LM Head 和其他输出阶段的真实边界。

### [MegaMoE 不止 MegaKernel](https://zhuanlan.zhihu.com/p/2061030607040390226)

把 MegaMoE 放回更长的 Fused Computation–Collective 谱系中，对比
Embedding + All-to-All、GEMV + AllReduce、GEMM + All-to-All 等先行路线。
核心提醒是：计算—通信重叠不是 MegaKernel 的定义；要判断 MegaMoE 的增量，
还需分析 Token/Tile 何时可见、通信由谁推进、同步和数据布局是否发生变化。

阅读边界：文中引用的性能数字来自不同工作，不能脱离硬件、拓扑、精度、
Batch 和基线配置横向比较。

### [如何评价 MiMo-V2.5-Pro UltraSpeed？1000 TPS 是怎么实现的？](https://www.zhihu.com/question/2047628080479524844/answer/2055753118420309625)

从 AI Infra 系统层解释超低延迟/高 TPS 路径，涉及更快 GEMM、大尺度 Fusion、
调度和推理系统协同。适合用于理解“Kernel 更快”只是系统吞吐的一部分。

阅读边界：产品模式的 TPS 必须结合并发、输入输出长度、投机策略、并行规模和
服务质量解释，不能直接等同于单请求延迟或单个 MegaKernel 的贡献。

## GEMM、Attention 与算子优化

### [CUDA 算子优化：Roofline 到 Tensor Core 体系化指南](https://zhuanlan.zhihu.com/p/2056836857451786262)

从 NCU 指标和 Roofline 判断 Memory-bound、Compute-bound 与 Latency-bound，
再连接到 Tensor Core、数据搬运和流水线优化。适合作为后续所有算子文章的诊断
入口：先定位限制，再选择优化，而不是先套模板。

### [MARLIN-W4A16 Mixed GEMM 技术分析](https://zhuanlan.zhihu.com/p/1956185856130916477)

围绕 W4A16 Mixed GEMM 分析 Marlin 类实现中的量化权重布局、解包/反量化与
Tensor Core 主循环协同。重点是把低比特节省的带宽与反量化、指令和布局成本
放在同一模型中理解。

### [[CUDA 优化实战] 纯手搓 Flash Decoding SM120（上）](https://zhuanlan.zhihu.com/p/2030745157620998667)

强调 Decode Attention 与 Prefill 的优化目标不同，并以 Blackwell SM120 上的
CUDA C++/PTX 实现为线索追求显存带宽利用。适合观察小 Batch、长 KV 下的切分、
加载和归约设计。

### [SM120 上的 Triton TMA 入门：从简单 Attention 开始](https://zhuanlan.zhihu.com/p/2055704132523184169)

用简单 Attention 解释 Blackwell 上 Triton 的 TMA 数据搬运。重点不只是 API，
而是把异步 Copy、Descriptor、Shared Memory 与计算流水联系起来。

### [Warp Specialization for Blackwell](https://zhuanlan.zhihu.com/p/1941232816185644696)

结合 CUTLASS Blackwell GEMM 源码分析 Warp Specialization，让 Loader、MMA、
Epilogue 等角色并行推进。它与 MegaKernel 的交集在于常驻 CTA 内的软件流水，
但 Warp Specialization 本身不是 MegaKernel。

### [FlashKL：像 FlashAttention 一样 One-Pass 计算 Attention KL Loss](https://zhuanlan.zhihu.com/p/2048512989352010720)

面向稀疏 Attention 的 Warm-up/训练过程，把 Attention KL Loss 改写成 One-pass
流式计算，减少中间矩阵物化。它展示了 FlashAttention 思路如何迁移到新的损失
计算，而不是简单复用 Attention Kernel。

### [LeetGPU Hard：Linear Self-Attention](https://zhuanlan.zhihu.com/p/2062598279435653172)

以线性 Attention 竞赛题记录 Kernel 设计与优化过程。适合观察算法结构变化如何
重塑状态更新、并行切分和内存访问，而不仅是对标准 MHA 做局部调优。

### [Tensor Core 编程与优化深度技术解析——GTC 2026](https://zhuanlan.zhihu.com/p/2051796929890218453)

基于 NVIDIA GTC 演讲整理 Tensor Core 编程与性能优化。作为二手讲解较易读，
涉及具体能力或限制时应回查对应 NVIDIA 演讲和架构文档。

## 编译器、DSL 与布局抽象

### [看懂 Triton 编译器的 IR：从 Python 到 PTX](https://www.zhihu.com/pin/2046346684142184193)

解释 Triton 从 Python DSL 经多级 MLIR 方言降到 PTX 的路径，以及编译器如何
保留 Block-level 语义来进行 Warp 切分、Shared-memory 布局和流水线选择。

### [local_tile 与 local_partition：从 CuTe 视角理解 TileLang](https://zhuanlan.zhihu.com/p/2050250641411462141)

用 CuTe 的 `local_tile`/`local_partition` 作为标尺，解释逻辑张量、CTA Tile
和线程级数据分工之间的两层映射。适合解决“一个逻辑 Tile 最终由哪个线程搬哪
个元素”的核心问题。

### [用 CuTe DSL 读懂 Hopper GEMM](https://zhuanlan.zhihu.com/p/2051749321079525874)

从单个 CTA Tile 出发连接 TMA、WGMMA 与 Pipeline，是理解现代 GEMM Mainloop
以及 MegaKernel 中可复用算子主体的实践入口。

### [[拆解 CuTeDSL] 怎么求最合适的 Epilogue](https://zhuanlan.zhihu.com/p/2051777797560054194)

从 TMEM/SMEM/TMA 限制推导 Epilogue Tile，进一步决定 `stmatrix` 或普通 Store。
重点是把 Epilogue 选择写成约束推导，而非经验常量。

### [[拆解 CuTeDSL] 怎么求最合适的 Epi Tile](https://zhuanlan.zhihu.com/p/2051719645745484663)

聚焦 Blackwell Helper 的 `compute_epilogue_tile_size`，分析 TMEM Load、寄存器
和 SMEM Store 之间的 Tile 选择。与上一篇主题接近，但更偏具体函数和算法拆解。

### [CUTLASS 笔记（10）：CUTLASS GEMM API](https://zhuanlan.zhihu.com/p/2044122416549474840)

基于 CUTLASS 4.5、SM90 介绍 GEMM API 和可配置模块，适合不直接手写 CuTe 时
构建高性能融合 GEMM。版本和目标架构是理解示例可迁移性的必要上下文。

### [如何正确地写 Layout：TileLang Layout Inference](https://zhuanlan.zhihu.com/p/2054144124052415821)

从源码角度分析 TileLang Layout Inference，说明布局如何在算子表达、线程映射
和最终代码之间传播。作者明确声明并非核心开发者，因此应把它作为源码导读而非
官方规范。

### [pegainfer：cuTile C++ 初尝试](https://zhuanlan.zhihu.com/p/2046657603057477134)

中文 cuTile C++ 实践笔记，记录如何把模型算子表达为 Tile 级程序。适合观察
新 DSL 的易用性与生成路径，但具体 API 和性能状态可能随工具版本快速变化。

## 性能分析、设备内 Profiling 与资源分区

### [CUDA Profile：从 Nsight Systems 时间线到 Nsight Compute 拆 Kernel](https://zhuanlan.zhihu.com/p/2045510649489387694)

提供从 Timeline 定位瓶颈、进入单 Kernel 指标、修改代码再验证的闭环。它是
评估 MegaKernel 的必要基础：整段关键路径、单 Kernel 时间求和和活跃 SM 覆盖率
回答的是不同问题，不能混用。

### [CuTeDSL 如何 Profile FA4](https://zhuanlan.zhihu.com/p/2047367828135752651)

围绕 `cutez.trace` 讨论 Device-side Range Profiling，用于观察 FA4 内部阶段。
重点是弥补只看 Kernel 外部时间线无法分辨 CTA 内流水的问题。

### [CuTeDSL IKET 尝试 Profile FA4](https://zhuanlan.zhihu.com/p/2054142037549642734)

继续比较自研 Range Profiling 与 CUTLASS 4.6 公开的 Device-side Profiling 工具，
反映同一需求在项目工具和官方实现中的不同权衡。

### [TIRx Notes：Plug-in In-kernel Profiler](https://zhuanlan.zhihu.com/p/2054305616391304228)

介绍 TIRx 的可插拔 CUDA Kernel 内 Profiler，并输出 Perfetto Trace 观察一个 CTA
内部不同线程组的重叠。它直接服务于 Warp-specialized Kernel 和 MegaKernel 的
阶段归因：外部 Timeline 看不到 Loader/MMA/Epilogue 的内部交叠。

### [CUDA 如何调度 Kernel 到指定的 SM？](https://www.zhihu.com/question/652642080/answer/2052753282947347717)

以 CUDA Green Contexts、Driver API 和 PDMux 为线索讨论 SM 资源分区。需要注意：
Green Context 更接近把一组 SM 资源分配给执行上下文，并不等同于逐次 Launch
精确钉死某个 Kernel 的具体 SM。文章声明使用模型整理官方资料，关键语义应回查
NVIDIA 官方文档。

## MoE、推理引擎与模型/系统协同

### [pegainfer（6）：从零开始的 MoE Expert Parallelism](https://zhuanlan.zhihu.com/p/2042737521881244854)

把问题缩小到单层 MoE，解释 Expert Parallelism 的数据流和实现难点。它是理解
MegaMoE 的前置材料：先明确 Token Dispatch、Expert GEMM、Combine 和通信归属，
再讨论是否应把它们放进一个持久化 Kernel。

### [vLLM 和 SGLang 的真正区别是什么？](https://www.zhihu.com/question/2045055313053843631/answer/2051092309400409916)

从源码和数据结构讨论 Prefix Cache、Block 组织及调度差异。它属于推理引擎而非
Kernel 文章，但能解释相同 Kernel 为什么在不同请求调度与缓存策略下呈现不同
系统性能。文章也明确提示其源码观察可能不是最新版本。

### [nanoPD：一个 LLM P/D 分离推理引擎的实现笔记](https://zhuanlan.zhihu.com/p/2026307825358382436)

通过自建小型引擎理解 Prefill/Decode 分离、KV 传输和调度。价值在于把论文概念
落实为可执行系统；它不应被误分类为单个 Kernel 优化。

### [如何看待 Qwen 的 Parallel Scaling？](https://www.zhihu.com/question/1907422978985169131/answer/1907565157103694086)

第一作者从 Idea 来源解释 Parallel Scaling。它不是 Kernel 实现文章，但模型
并行和结构选择会改变每卡 GEMM Shape、通信比例和可形成的流水，是 Kernel/系统
联合优化的上游约束。

## 推荐阅读顺序

如果目标是研究 MegaKernel，可按以下顺序：

1. `CUDA Profile` 与 `Roofline`：先建立正确的性能归因方法；
2. `Hopper GEMM`、`Warp Specialization`、`TMA`：理解高性能算子主体；
3. Hazy 1B MegaKernel 与 MegaQwen 对照：理解持久化执行模型；
4. MegaRTP：用强基线区分 Multistream/PDL 与 Tile 级专属收益；
5. Sync、Fleet、TIRx Profiling：分析运行时开销和设备内时间线；
6. MegaMoE 与 Expert Parallelism：研究真正持续的算通 Tile 流水。

最重要的评测原则是：相对单流串行基线的全部收益不能都记在 MegaKernel 名下。
应先建立 Fusion + 分支 DAG + PDL 的强基线，再报告 Tile 级依赖、设备内调度和
常驻 CTA 流水的增量价值。

