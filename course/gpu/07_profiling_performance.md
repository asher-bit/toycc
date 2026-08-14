# GPU 第 7 章：性能分析——从 benchmark 到 Nsight 指标

## 1. 本章目标

- 能按六步协议写一个可复现的 benchmark，并解释每一步防的是什么错误；
- 能区分 kernel 时间、端到端时间、launch 开销与同步时间四者；
- 能用 roofline 算出一个 kernel 的理论上限，并用它建立/否定优化假设；
- 能说清 Nsight Systems 与 Nsight Compute 的分工，以及"时间花在哪"和"kernel 为什么这样跑"两个问题的区别；
- 能把 profiler 指标翻译成具体的下一步代码修改，而不是只收集数字。

前置：第 1 章的 roofline 与带宽账、第 6 章的 stream/event 计时。工具：Nsight Systems / Nsight Compute（随 CUDA Toolkit 或独立安装）、`compute-sanitizer`（查正确性）。

## 2. 工作中的问题长什么样

性能方向的日常问题：

```text
"同一段代码跑两次，时间差 10%，到底以哪次为准？"
"Nsight Compute 说 stall 占 80%，下一步改什么？"
"单个 GEMM 优化了 20%，模型端到端怎么只快了 3%？"
```

三问对应三个方法论：**测量协议**（数字可复现的前提）、**指标归因**（指标 → 瓶颈 → 代码动作）、**分层回归**（各层收益不会自动传导）。本章建立这三个方法论，工具只是它们的载体。

## 3. 正确测速的六步协议

```text
① 固定 GPU / 电源和时钟条件
② 预热, 排除首次 JIT / 分配
③ 明确同步边界
④ 多次重复, 报告 median / p90 / min
⑤ 分离 H2D、kernel、D2H 和总时间
⑥ 同时做 correctness check
```

每一步防一个具体的错：

**① 固定条件**：GPU 时钟随负载/温度动态变化，同一代码两次跑可以差出 ±10%。锁时钟、固定功耗模式、独占 GPU，否则任何"优化了 5%"的结论都不成立。

**② 预热**：第一次运行混入 JIT（第 2 章）、context 创建、cudaMalloc、冷缓存——都不是 kernel 本身的时间。先跑一轮丢弃，再开始计时。

**③ 同步边界**：CUDA kernel 异步发射（第 6 章）。用 host wall clock 包住 launch 而不同步，量到的可能只是 enqueue 时间。kernel 区间用 `cudaEventRecord` + `cudaEventElapsedTime` 量，端到端时间用 Nsight Systems 看。这是【可运行代码】骨架：

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start); cudaEventCreate(&stop);
for (int i = 0; i < warmup; i++) kernel<<<grid, block>>>(...);   // 预热
cudaDeviceSynchronize();
cudaEventRecord(start);                       // 记录在 stream 上的时间点
for (int i = 0; i < iters; i++) kernel<<<grid, block>>>(...);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms = 0; cudaEventElapsedTime(&ms, start, stop);
double time_per_iter = ms / iters;            // 单次 kernel 时间
```

**④ 多次重复取 median/p90/min**：mean 会被偶发抖动（OS 抢占、时钟抖动）拉偏；**median 给典型值，p90 给抖动上限，min 给干净条件下的下界**。三者一起报，比一个平均数诚实得多。

**⑤ 分离各段时间**：H2D、kernel、D2H、launch 各记各的。四段混在一起，任何一段的优化都被另三段稀释，归因无从谈起。

**⑥ 正确性同跑**：每次 benchmark 同时跑一遍数值校验（对参考实现算 max|Δ|）。"快了 30% 但结果错了"的优化毫无意义——toycc 每个 pass 后都做对比，这里是同一纪律。

## 4. Roofline：上限账与假设检验

Roofline 模型（第 1 章已用于 vector_add，这里把它写成通用协议）：

```text
arithmetic intensity(算术强度) = FLOPs / bytes moved
可达性能上限 ≈ min(峰值计算吞吐, 峰值带宽 × 算术强度)
```

手算一遍 vector_add 在 A100 上的上限（fp32，峰值算力 19.5 TFLOPS、带宽 1.55 TB/s）：

```text
强度 = 1 FLOP / 12 B ≈ 0.08 FLOP/B (第 1 章已推)
带宽墙 = 1.55e12 × 0.08 = 1.24e11 FLOP/s ≈ 124 GFLOP/s
算力墙 = 19.5e12 = 19500 GFLOP/s
上限 = min(19500, 124) = 124 GFLOP/s → 只有峰值算力的 0.6%
```

**Roofline 是建立假设的工具，不是自动证明**。它告诉你"如果瓶颈是带宽，上限在哪"；真实 kernel 还受指令混合、cache 命中、同步、occupancy、发散、Tensor Core 使用与 launch 影响。正确用法是：先算出理论上限 → 实测 → 如果实测远低于上限，按第 6 节的指标表找"为什么低于"；如果实测接近上限，说明这个算法在此硬件上已到头，换算法而不是抠实现。

## 5. 两类 Nsight 工具的分工

**Nsight Systems 看时间线，回答"整个程序时间花在哪里"**；**Nsight Compute 看单个 kernel 的硬件行为，回答"这个 kernel 为什么这样跑"**。两者不互相替代。

### 5.1 Nsight Systems：读时间线

时间线上每行是一个 host 线程或 GPU stream，色块是区间（kernel、memcpy、同步）。读法：

- CPU 与 GPU 之间的大段空白 = host 在等 GPU 或 GPU 在等 host（同步点）；
- H2D/D2H/kernel 块交错与否 = stream 是否真的重叠（第 6 章第 6 节的账在这里变成图形）；
- 一串极短的小 kernel = launch 开销主导（第 6 章 28% 的账）；
- 通信块（NCCL）与计算块重叠与否 = 通信计算重叠的质量。

用 NVTX 区间给代码段打标签，时间线上就能看到"哪个函数阶段"而不只是"哪个 kernel"。典型排查问题清单：CPU 等 GPU？多 stream 真重叠吗？有大量小 kernel 吗？有隐式同步吗？有 JIT/分配尖峰吗？

### 5.2 Nsight Compute：读 kernel 报告

报告的入口是 **Speed of Light** 部分：SM 利用率（计算侧）与 DRAM 吞吐（带宽侧）各占理论峰值的百分比。两个数字先判断大头在哪一侧，再往下钻：

- **Warp State 统计**：一个 warp 的时间花在哪——`stall long scoreboard`（等内存返回）、`stall wait`（等固定延迟）、`stall barrier`（等同步）、真正在发射指令的占比。**"stall 80%"本身不是 bug**：GPU 靠大量 warp 互相隐藏延迟（第 3 章 occupancy 的用途），关键是"stall 的主因"指向哪个资源；
- **内存部分**：访存事务数、L1/L2 命中率、合并程度（第 1 章第 7 节的事务账在这里变成实测数字）；
- **资源部分**：寄存器/spill/occupancy（第 3 章的账变成报告的列）。

指标到行动的翻译表（第 6 节）就是把"报告里看到的现象"映射成"下一步改什么"。

## 6. 指标到行动的映射

| 观察 | 可能原因 | 下一步实验 |
|---|---|---|
| kernel 很短但总时间很长 | launch/同步/调度开销 | 融合、批量化、CUDA Graphs、减少 host 往返 |
| DRAM 吞吐高、计算利用率低 | memory-bound | 合并访问、tile、复用、压缩、换布局 |
| local load/store 多 | register spill（第 3 章） | 缩小 live range、调 tile/unroll、查内联 |
| eligible warps 少、stall 以 memory 为主 | 延迟没被隐藏或访存慢 | 加并行度、改善访问、重算 occupancy |
| shared 吞吐/波形异常 | bank conflict 或 barrier 过密 | 改 layout、padding、减少 barrier |
| Tensor Core 利用率低 | dtype/layout/shape/指令路径不匹配 | 查 MMA tile、对齐、编译目标、epilogue |
| 多 stream 不重叠 | 依赖、NULL stream、隐式同步（第 6 章） | event 图、stream 语义、copy engine |

两条纪律：**每次只改一个主要变量**（改两个就不知道归因给谁）；**记录全部条件**——代码、编译参数、输入 shape、GPU、driver 版本、报告本身，供以后复现与回归对比。

## 7. 分层性能回归：为什么 GEMM 快 20% ≠ 模型快 20%

性能测试分三层，各测各的：

```text
1. microbenchmark:  一个 kernel 的吞吐与资源
2. operator benchmark: 一个算子 = 多个 kernel + layout transform + workspace
3. model benchmark:  端到端 = 前端 + 编译 + runtime + 通信 + 调度
```

手算一个"收益不传导"的例子：

```text
模型端到端 9 ms = 前端 2 ms + layout transform 2 ms + 三个 kernel 共 5 ms
把 GEMM kernel 优化 20%: kernel 5 ms → 4 ms
端到端 = 2 + 2 + 4 = 8 ms → 只快 11%, 不是 20%
```

被优化的部分只占总时间的一部分，收益按占比缩水。所以性能报告要**同时标出优化收益和新增成本**（比如为了新 layout 多付了一次 transform），并分层记录——micro 层数字用于归因，model 层数字用于决策。

## 8. 常见错误与归因

| 现象 | 根因 | 修正 |
|---|---|---|
| 两次跑差 10% | 时钟/功耗未固定、没预热 | 协议第 ① ② 步 |
| "优化"后反而更慢 | 只跑一次、或改了多个变量 | 协议第 ④ 步 + 一次一个变量 |
| 量到的 kernel 时间异常短 | host 计时没同步，量到 enqueue | 协议第 ③ 步，用 cudaEvent |
| 指标收集一堆但没有行动 | 没做"指标 → 瓶颈 → 代码"翻译 | 第 6 节映射表逐行过 |
| 微基准漂亮、端到端不变 | 收益被其他层稀释 | 第 7 节分层回归 |
| 报告无法复现 | 条件没记录 | 记录代码/参数/shape/GPU/driver 版本 |

## 9. 检查点

完成以下四项才算通过本章：

1. 写出六步测速协议，并给每步配一句"防的是什么错误"；
2. 手算 fp16 4096³ GEMM（强度约 407 FLOP/B，见第 4 章）在 H100（TC fp16 约 989 TFLOPS、HBM3 约 3.35 TB/s）上的 roofline 上限，判断它受哪面墙限制；
3. 读一个 Nsight Compute 报告的 Speed of Light 部分：SM 利用率 15%、DRAM 吞吐 85% 时，下一步查哪一类指标；
4. 用第 7 节的例子结构，解释"kernel 快 20% 而模型只快 11%"的换算。

## 10. 下一步与扩展阅读

本章把第 1~6 章的所有账变成了可测数字。下一章（GPU 08：NCCL 与多 GPU）把这些方法用到多卡上——通信时间进入时间线、通信计算重叠成为新的优化轴。硬件计数器如何变成 profiler 数字的底层机制，见主教材第 30 课。

- 官方：[Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/)、[Nsight Compute 文档](https://docs.nvidia.com/nsight-compute/)、[CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)；
- 与本课程的关系：toycc 的硬件模型（缓存/延迟表）是"教学版性能模型"，本章的 profiler 是"真实硬件性能模型"——两者问的是同一个问题：时间去哪了。

**导航**：⬅ [上一章](06_runtime_driver.md)（Runtime / Driver）　｜　[下一章](08_multi_gpu_nccl.md)（NCCL 与多 GPU）➡
