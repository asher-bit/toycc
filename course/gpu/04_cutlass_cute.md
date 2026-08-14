# GPU 第 4 章：CUTLASS / CuTe——高性能 GEMM 的软件分层

## 1. 本章目标

- 能回答"cuBLAS、CUTLASS、Triton、手写 CUDA 四选一"的判断问题，并给出理由；
- 能说清 GEMM 五层分解（threadblock tile / warp tile / MMA / copy / epilogue）每层的对象和典型尺寸；
- 能用手算解释"为什么必须分层"——naive GEMM 与分块 GEMM 的算术强度差多少；
- 能读懂 CuTe layout 的 shape/stride，手算一个元素在内存里的偏移；
- 能列出用 CUTLASS profiler 做公平对比时必须固定的变量。

前置：第 1 章的线程/内存层次、第 3 章的 occupancy 与寄存器账。跑实验需要 NVIDIA GPU + CUTLASS（源码构建或预编译）；本章所有手算不需要 GPU。

## 2. 工作中的问题长什么样

性能库方向的两类日常问题：

```text
"这个 attention 的 epilogue 要融合 bias+量化，cuBLAS 不支持，怎么办？"
"CUTLASS 的模板报错一百行，到底哪里不对？"
"为什么 CUTLASS 的 GEMM 能到 700+ TFLOPS，我手写的只有 1/10？"
```

三个问题分别对应：**抽象边界**（什么活该在哪层干）、**模板阅读**（错误怎么定位）、**分层复用**（性能差从哪来）。本章先算性能账，再用这个账解释 CUTLASS 为什么长成"五层分解"的样子。

## 3. 先算账：为什么朴素 GEMM 注定慢两个数量级

取 fp16 的 4096³ GEMM（M=N=K=4096）。**算术强度 = 计算量 ÷ 数据搬运量**（FLOP/B）。

### 3.1 无复用的账

每算一个输出元素：2K 次运算，读 A 的 K 个元素 + B 的 K 个元素：

```text
强度 = 2K FLOP / (2K × 2 B) = 1 / 2 = 0.5 FLOP/B
```

对照 H100 的 roofline 拐点（Tensor Core fp16 约 989 TFLOPS，HBM3 约 3.35 TB/s）：

```text
拐点 = 989e12 / 3.35e12 ≈ 295 FLOP/B
0.5 ≪ 295 → 无复用的 GEMM 是带宽受限, 离算力墙差 ~590 倍
```

### 3.2 分级复用的账

把数据分块装进寄存器/共享内存后，DRAM 只需要搬运每份数据一次：A、B 各 M·K、N·K 个元素（fp16，2 B），C 是 M·N 个（fp32，4 B）：

```text
搬运 = 2×(M·K + N·K)×2 B + M·N×4 B
     = 2×(3.36e7 + 3.36e7)×2 + 1.68e7×4 ≈ 2.69e8 + 6.7e7 ≈ 3.36e8 B
计算 = 2·M·N·K = 2×4096³ ≈ 1.37e11 FLOP
强度 = 1.37e11 / 3.36e8 ≈ 407 FLOP/B  →  越过 295 拐点 → 算力受限 ✓
```

两笔账的差在哪：**无复用版本把同一份 A 元素读了 N 遍、同一份 B 元素读了 M 遍，而分级复用各读一遍**。这个"复用"落到代码里就是五层分解——它不是 CUTLASS 发明的美学，而是算术强度的账逼出来的结构。

## 4. GEMM 的五层分解

CUTLASS 不是图编译器，也不是 runtime，它是一套**CUDA C++ 可组合、可特化的高性能线性代数模板库**：把数据搬移、线程组织、矩阵乘累加、数据类型、布局和 epilogue 拆成可复用组件，让你按需拼装。CUTLASS 3.x 的底层抽象叫 **CuTe**（第 5 节）。五层分解以 `C = A × B + C` 为例：

| 层 | 一句话定义 | 典型尺寸 / 对象 |
|---|---|---|
| threadblock tile | 一个 block 负责输出矩阵 C 的一块 | 128×128；`CtaTile` 形状参数 |
| warp tile | block 内各 warp 再分工 | 64×64（8 个 warp 时） |
| MMA atom | 一条 Tensor Core 指令的形状 | fp16 为 m16n8k16 |
| copy | 数据从 global → shared → register 的搬移 | `cp.async`（异步拷贝指令）/ TMA（Hopper 硬件拷贝引擎） |
| epilogue | 累加结果转类型、加 bias、激活、量化、写回 | fp32 累加器 → 输出 dtype |

### 4.1 MMA atom 的账

**Tensor Core 的 mma 指令**吃固定形状的矩阵乘。fp16 的 `mma.sync.m16n8k16` 一条指令算：A 16×16、B 16×8，产出 C 16×8（fp32 累加）：

```text
每指令 MAC 数 = 16×8×16 = 2048 → FLOP = 4096
对照: 一条普通 FMA 只有 2 FLOP → 一条 mma = 2048 条 FMA 的工作量
```

这就是"为什么必须用 Tensor Core"的微观答案：H100 的 fp16 Tensor Core（约 989 TFLOPS）是 CUDA core fp16（约 67 TFLOPS）的 ~15 倍，而 mma 指令的固定形状决定了数据必须以它要求的布局到达寄存器——布局不匹配的代价是额外搬移，这正是 copy 层存在的意义。

### 4.2 copy 层的账

数据要进 Tensor Core 寄存器，路上两级搬运：global → shared（块级复用，用 `cp.async` 异步拷贝，或 Hopper 用 TMA 让硬件搬运引擎直接干），shared → register（每线程取自己的片段，按 mma 的布局取）。搬错布局的直接后果是**bank conflict 或额外的 shared 往返**（第 1 章第 8 节的手算在这里变成布局检查）。

### 4.3 epilogue 融合的账

不融合时，C 的 128×128 块要先写回 global 再读回来做 bias+激活再写回：

```text
不融合: 写 64KB + 读 64KB + 写 64KB = 192KB 的 DRAM 往返
融合:   epilogue 在寄存器里做完, 只写一次 64KB → 省 128KB/块
```

epilogue 层就是把"输出前的最后一串逐元素操作"留在片上完成——与第 1 章 roofline 结论一致：带宽受限的活，优化永远从"少搬"开始。

### 4.4 调优的耦合性

tile 大小、pipeline stage 数、dtype、布局、寄存器与 shared 用量**同时**改变 occupancy、访存事务和 Tensor Core 利用率（第 3 章第 4 节的四项限制在这里全部生效）。只调 tile 不看寄存器账，常见结局是 occupancy 掉一半、性能不升反降——所以 CUTLASS 的 profiler 才把资源用量和性能一起报（第 7 节）。

## 5. CuTe：layout algebra 的最小直觉

**CuTe** 是 CUTLASS 3.x 的基础抽象库，核心对象是 **Layout**。一个 Layout 描述"逻辑坐标 → 内存偏移"的映射，由两部分组成：

- **Shape**：每个维度的长度；
- **Stride**：每个维度走一步，内存偏移前进多少。

```cpp
// 示意代码(概念级, 非完整可编译程序): 8×16 的矩阵
auto layout = make_layout(make_shape(Int<8>{}, Int<16>{}),
                          make_stride(Int<16>{}, Int<1>{}));
auto tensor = make_tensor(ptr, layout);
```

手算这个 layout：元素 `(i, j)` 的内存偏移 = `i×stride0 + j×stride1`：

```text
shape = (8, 16), stride = (16, 1)   → 偏移(i,j) = 16i + j   = 行主序(第 i 行连续 16 个元素)
shape = (8, 16), stride = (1, 8)    → 偏移(i,j) = i + 8j     = 列主序
```

同样一块数据，换一个 stride 就是另一种布局——**布局不搬数据，只换"怎么解释坐标"**。这就是"layout transform"的本质（toycc 的 layout pass、第 2 章 PTX 里的转置，都是同一个概念的不同层次实现）。

线程映射同样用 Layout 表达：一个 128×128 的 C tile 分给 256 个线程，可以用"每线程 tile = tile 形状 ÷ 线程形状"的除法组合出来；copy 与 MMA 的输入输出布局是否匹配，就是"两边 Layout 是否相容"的检查。阅读 CUTLASS 代码时先问三个问题：

```text
1. 当前 layout 的 shape/stride 是什么？
2. 当前线程 / warp / CTA 如何切分这个 layout？
3. copy / MMA atom 的输入输出布局是否匹配？
```

模板报错难读的根源也在这里：Layout 是编译期类型，错配时错误在实例化点层层展开。定位顺序反过来——**从最内层的 static_assert 往外读**，第一行说"哪个 Layout 不匹配"通常就是根因。

## 6. API 选择：四种工具的边界

| 场景 | 优先选择 | 理由 |
|---|---|---|
| 标准 GEMM/conv，已有成熟算法 | cuBLAS / cuDNN | 算法已调优多年，自写重造轮子 |
| 要融合 epilogue、特殊 dtype/layout | CUTLASS / CuTe | 组件可拼装，epilogue 是它的强项 |
| 想快速表达自定义 tile 算法 | Triton | 块级抽象，开发迭代快（第 5 章） |
| 要完全控制 ISA、异步机制或新硬件 | CUDA C++ / PTX / 自研后端 | 抽象层以下的东西只有它能碰 |
| 只是图级组合 | TVM / MLIR / 框架编译器 | 图级问题在图级解决，别从 CUTLASS 开始 |

判断顺序一句话：**先问"问题在图的哪一层"，再选工具**——图级优化用图编译器，tile 级算法用 Triton/CUTLASS，指令级控制才轮到手写 CUDA。

## 7. Profiler：公平对比的固定清单

CUTLASS 自带 `cutlass_profiler` 命令行工具（示例命令，具体参数以 `--help` 为准）：

```bash
./cutlass_profiler --operation=Gemm \
  --m=4096 --n=4096 --k=4096 \
  --A=f16:row --B=f16:col --C=f32:row
```

它的输出不是"一个数字"，而是候选配置矩阵：不同 problem size、dtype、tile、split-K、epilogue 组合下的时间、**TFLOP/s、带宽、寄存器、shared、occupancy**。对比两组配置时，以下变量必须固定，否则数字不可比：

```text
GPU 与时钟/功耗状态、数据布局(row/col)、warmup、重复次数、
stream、workspace 大小、校验方式(是否真校验了数值)
```

对比的正确姿势：候选生成 → 编译实例化 → 数值校验 → warmup → 多次计时取中位数 → 记录资源账 → 与 baseline 比。**只比 wall time、不记资源的对比没有归因能力**——慢了，是 spill 了还是 occupancy 掉了，要靠资源账回答（第 3 章的账在这里变成 profiler 的列）。

## 8. 源码阅读地图

CUTLASS 仓库按层次组织（3.x 版本结构，具体以当前版本为准）：

- `include/cute/`：Layout、tensor、copy/MMA atom 等基础抽象；
- `include/cutlass/`：device kernel、矩阵乘、数据类型与工具；
- `examples/`：各组件的最小可用示例；
- `tools/profiler/`：命令行 profiling 入口；
- `test/unit/`：组件级与端到端测试。

读源码顺序：`examples → Collective/Kernel → TiledMma/TiledCopy → atom → arch wrapper`，**不要从最深的模板报错开始**——先在上层把数据流画出来，再逐层下沉。

## 9. 常见错误与归因

| 现象 | 根因 | 定位手段 |
|---|---|---|
| 模板报错上百行 | Layout 类型错配，在实例化点展开 | 从最内层 static_assert 往外读 |
| GEMM 只有理论峰值 1/10 | 数据没走 mma 布局 / copy 绕路 / spill | profiler 看 TFLOP/s 与资源账 |
| 换了 epilogue 数值误差变大 | fp32 累加器提前转低精度 / 校验缺失 | 逐层关 epilogue 找引入误差的转换 |
| profiler 数字每次跑都不同 | 时钟/功耗状态、warmup 不固定 | 按第 7 节清单固定变量 |
| 调大 tile 反而变慢 | 寄存器超限 → spill 或 occupancy 掉 | 对照第 3 章的四项限制手算 |

## 10. 检查点

完成以下四项才算通过本章：

1. 手算：fp16 的 2048³ GEMM（M=N=K=2048）在无复用与分级复用两种假设下的算术强度，并判断它在 H100（拐点约 295 FLOP/B）上属于哪种受限；
2. 给 Layout `shape=(16, 32), stride=(32, 1)` 手算出元素 `(i, j)` 的偏移公式，并写出它的列主序版本；
3. 画出 fp16 GEMM 一个 128×128 threadblock tile 里，数据从 DRAM 到 mma 寄存器经过的路径（标出 cp.async / TMA、shared、寄存器）；
4. 列出一份"对比 CUTLASS 两种 epilogue 配置"的 profiler 固定清单。

## 11. 下一步与扩展阅读

本章的账全部建立在第 1、3 章的线程/内存/寄存器模型上。下一章（GPU 05：Triton 编译器）回答"如果不手写这些分层，让编译器来做会怎样"——Triton 把 block 级抽象变成语言原语，本章的 tile/布局/copy 都变成编译器 IR 里的对象。

- 官方：[CUTLASS Overview](https://docs.nvidia.com/cutlass/latest/overview.html)、[Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)、[CuTe 文档](https://docs.nvidia.com/cutlass/latest/media/docs/cute/00_quickstart.html)；
- 与本课程的关系：CUTLASS 的 epilogue 融合与 toycc 的算子融合是同一思想（省中间往返）在不同层（指令层 vs 图层）的实现；CuTe 的 Layout 与 toycc 的 layout pass 处理的是同一个"布局变换"概念。

**导航**：⬅ [上一章](03_gpu_isa_registers_abi.md)（GPU ISA、寄存器分配与 ABI）　｜　[下一章](05_triton_compiler.md)（Triton 编译器）➡
