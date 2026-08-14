# MegaKernel 的专属能力：不要把所有重叠都归功于 MegaKernel

## 结论

通信—计算重叠、独立算子并行、下游 Kernel 提前启动和权重预取，都不是
MegaKernel 的专属能力。普通 CUDA Kernel 配合 Multistream、CUDA Graph、
Event 与 Programmatic Dependent Launch（PDL）也能表达这些优化。

MegaKernel 真正具有区分度的机制是：

> 在一次设备常驻 Kernel 内，将跨算子的依赖、调度与软件流水下沉到
> Tile/CTA 粒度，并让 GPU 持续执行这些细粒度任务。

“发生了重叠”只是结果。只有当这部分重叠依赖跨算子的 Tile 级释放、GPU 内
Tile 调度或常驻 CTA 状态复用时，才应归因于 MegaKernel 的专属能力。

## 哪些能力不是 MegaKernel 专属

| 能力 | 普通 CUDA 实现路径 |
| :--- | :--- |
| 无依赖分支并行 | Multistream 或 CUDA Graph 分支 DAG |
| 通信—计算重叠 | Stream、Event、异步通信原语 |
| Consumer 提前进入 GPU | PDL |
| 等待输入时完成 Prologue | PDL + Kernel 内阶段拆分 |
| 提前加载权重与 Scale | PDL + 独立 Loader |
| 降低 Host Launch 开销 | CUDA Graph、Kernel Fusion |

这些机制同样可能带来很大的端到端收益。因此，比较 MegaKernel 与普通 Kernel
时，基线至少应完成合理的 Fusion、DAG 并行和 PDL；否则加速比主要说明基线
存在粗粒度串行，而不能证明 MegaKernel 的独有机制有效。

## MegaKernel 的三项专属能力

### 1. 跨算子的 Tile-to-Tile 依赖

普通 Kernel 依赖通常以整个 Grid 完成为边界：

```text
Kernel A 全部完成 → Kernel B 开始
```

MegaKernel 可以把完成状态细化到 Tile：

```text
A.tile[0] 完成 → B.tile[0] 开始
A.tile[1..n] 继续执行
```

Producer 每完成一个输出区域就更新 Event Counter；Consumer 只等待自己读取区域
对应的阈值，而不必等待 Producer 的全部 CTA 退出。这项能力能够消除跨算子的
全 Grid Barrier。

### 2. GPU 内部的跨算子 Tile 调度

MegaKernel 启动固定数量的常驻 Worker CTA。每个 Worker 从任务队列读取来自
不同算子的 Tile Instruction：

```text
Worker 0: Norm tile → GEMM tile → Attention tile
Worker 1: GEMM tile → Communication tile → GEMM tile
```

因此，调度对象不再是完整 Kernel，而是跨算子的 CTA/Tile。设备运行时可以更
直接地表达 Tile 优先级、分支资源份额和局部就绪关系。Multistream 只能提交
Kernel 级工作，具体并发顺序与资源分配仍主要由硬件调度器决定。

### 3. 常驻 CTA 内跨 Instruction 延续软件流水

普通 Kernel 结束时，其 CTA 状态、共享内存布局和软件流水随之消失。常驻 CTA
则可以在当前 Instruction 计算时准备下一条 Instruction：

```text
当前 Instruction: Tensor Core 计算 / Epilogue
下一条 Instruction: 读取描述、初始化 Barrier、准备 Descriptor、预取权重
```

更细的实现还可以逐页交接 Shared Memory：上一条 Instruction 释放一个物理页，
下一条立即使用该页预取，而不必等待上一条任务完全结束。这是跨算子延续 CTA
内部流水与片上资源生命周期的能力。

## 专属机制何时才能产生专属收益

Tile 级能力存在，并不等于它一定缩短关键路径。通常需要同时满足：

1. Producer 的输出能被 Consumer 按更小区域独立消费；
2. Producer 持续多个 Wave，第一批 Tile 就绪明显早于整个 Grid 完成；
3. GPU 仍有空闲资源运行 Consumer；
4. 形成的 Tile 流水位于最终关键路径；
5. 收益能够覆盖指令分发、同步和常驻运行时的固定成本。

MoE 的 Dispatch → Expert GEMM → Combine 是典型候选：某个专家的一批 Token 到达
后即可计算，后续 Token 仍在通信，已完成的结果还能继续 Combine，因而可能形成
持续的 Tile 级算通流水。

相反，如果 `M=16` 且 `blockM=16`，M 方向只有一个 Tile；Consumer 又需要完整
的 16 行输入，那么 Tile 级 Counter 实际上接近“Producer 完成”信号。此时
MegaKernel 的专属机制虽然存在，却没有更早释放主计算的窗口。

## 如何判断一项加速是否应归功于 MegaKernel

建议使用逐层基线：

1. 优化单 Kernel；
2. 做必要的 Kernel Fusion；
3. 用 Multistream 表达真实的分支 DAG；
4. 用 PDL 提前执行 Consumer 的 Prologue 与预取；
5. 最后比较 Tile 级依赖、设备内调度和常驻 CTA 流水的增量收益。

如果第 3、4 步已经复现绝大部分收益，就应把这些收益归因于 Kernel 级 DAG
并行和提前启动。MegaKernel 的专属收益是第 5 步相对这一强基线的增量，而不是
相对单流串行基线的全部差值。

## 案例：MegaRTP 与 Multistream + PDL

知乎文章《MegaKernel是创新还是传销？》实现了一个名为 MegaRTP 的设备常驻
运行时，并在 B300 上分析 GLM-5.2-FP8 Decode 的一段 Attention-pre 子图。
作者进一步使用相同的 Standalone Kernel，通过 Multistream + PDL 重建计算图。

文章的关键认识不是“MegaKernel 无效”，而是：该工作负载的大部分重叠来自
独立分支并行与 Consumer 提前准备；由于主要 GEMM 在 M 方向只有一个 Tile，
MegaKernel 特有的细粒度依赖几乎没有缩短关键路径。这个案例说明，评价
MegaKernel 时必须把通用的重叠收益与 Tile 级专属收益分开归因。

## 延伸阅读

- [MegaKernel是创新还是传销？——是小肖啊](https://www.zhihu.com/question/2013258505231050695/answer/2071314457918183125)
- [Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- [A Framework for Fine-Grained Synchronization of Dependent GPU Kernels](https://conf.researchr.org/details/cgo-2024/cgo-2024-main-conference/14/A-Framework-for-Fine-Grained-Synchronization-of-Dependent-GPU-Kernels)

