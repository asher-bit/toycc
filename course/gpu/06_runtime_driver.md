# GPU 第 6 章：Runtime / Driver——模块、内存、Stream、Event 与提交

## 1. 本章目标

- 能说清 CUDA Runtime API 与 Driver API 的分层关系，以及框架什么时候必须用 Driver API；
- 能画出"cubin 文件 → 设备上执行的 kernel"的完整对象生命周期（context / module / function / memory / stream / event）；
- 能解释"错误为什么在 launch 之后很久才报"（sticky error 机制）以及哪些调用会隐式同步；
- 能手算 launch 开销账，并解释它对小 kernel 吞吐的杀伤力；
- 能用 stream/event 搭出一个 copy 与 kernel 重叠的最小流水，并说出重叠的收益上限。

前置：第 2 章的产物（PTX/cubin/fatbin）与第 1 章的异步执行事实。跑实验需要 NVIDIA GPU 与 CUDA Toolkit；手算不需要。

## 2. 工作中的问题长什么样

runtime/driver 方向的日常问题：

```text
"kernel 才跑 2 微秒，为什么吞吐只有理论的 28%？"
"报错的位置离出错的 kernel 差了十行，怎么定位？"
"我开了 4 条 stream，为什么 copy 和 kernel 完全没有重叠？"
```

三问对应三个机制：**launch 开销**（每发射一次要付多少固定账）、**异步错误**（错误何时暴露）、**同步边界**（谁在悄悄同步）。本章把"编译产物如何真正执行"整条链拆开，这三问的答案都在链上。

## 3. 完整生命周期：一次 kernel 执行要经过哪些对象

```text
host application
  → ① 初始化与 context     cuInit / cuCtxCreate
  → ② 模块加载             cuModuleLoad(cubin/PTX/fatbin)
  → ③ 函数查找             cuModuleGetFunction
  → ④ 资源准备             device memory / stream / event
  → ⑤ kernel launch        cuLaunchKernel
  → ⑥ 异步执行与同步       stream 排队、event、cudaDeviceSynchronize
  → ⑦ 错误查询             cudaGetLastError / cuGetLastError
```

逐环节定义与对象：

**① context（上下文）**：一个 context 是"某个设备上的一组执行资源与状态"的容器——所有 module、memory、stream 都挂在某个 context 下。创建 context 的代价在数百毫秒量级（驱动要做设备初始化），所以应用一次创建、长期复用；**错误状态也是 per-context 的**（第 5 节 sticky error 的载体）。

**② module（模块）**：把 cubin/PTX/fatbin 加载进 driver 后的设备代码对象。加载时 driver 做一轮校验：目标架构是否匹配（第 2 章第 7 节的兼容性账在这里兑现）、driver 版本是否支持、kernel 符号与资源元数据是否齐全。**"编译成功"只意味着存在一个可加载候选，加载这一步才是它第一次面对真实设备**。

**③ function（函数句柄）**：按 kernel 名字在 module 里查出的入口符号。对象是 `CUfunction`，之后每次 launch 都拿它作为"要执行谁"。名字对应第 2 章 PTX 里 `.visible .entry` 的符号。

**④ 内存 / stream / event**：launch 之前的资源准备——数据要先进 device 内存（第 7 节），提交要挂在 stream 上（第 4 节），测量/依赖靠 event。

**⑤⑥⑦ launch → 异步 → 查错**：launch 把"哪个函数、什么 grid/block、哪些参数、挂哪条 stream"打包提交给硬件队列，**host 侧立即返回**；执行是异步的。想拿到结果或错误，必须在同步边界上等（第 5 节）。

## 4. Runtime API 与 Driver API：一层封装之差

**CUDA Runtime API**（`cudaMalloc`、`cudaMemcpy`、`cudaLaunchKernel`...）是高层封装：替你管理 context、隐式处理一些资源，适合应用代码。**CUDA Driver API**（`cuInit`、`cuModuleLoad`、`cuLaunchKernel`...）是更底层、对象更显式的接口：context、module、function、虚拟内存全部手动管理。

下面是【示意代码】（省略错误检查与版本细节；真实代码必须逐个检查返回码）：

```cpp
CUcontext ctx;   CUmodule module;   CUfunction fn;
cuInit(0);                                  // 初始化 driver
cuCtxCreate(&ctx, 0, device);               // ① 建 context
cuModuleLoad(&module, "kernel.cubin");      // ② 加载模块(架构校验在这里)
cuModuleGetFunction(&fn, module, "add_kernel");  // ③ 按名字查 kernel 入口
cuLaunchKernel(fn,                          // ⑤ 发射
               grid_x, 1, 1, block_x, 1, 1, // grid/block 三维配置
               shared_bytes, stream,        // shared 与 stream
               args, nullptr);              // 参数数组
```

分层使用的判断：**应用层用 Runtime；框架、JIT、模块管理、插件、虚拟内存等细粒度控制用 Driver**。TVM/MLIR 系的运行时在做"动态加载新编译的 kernel"时，走的就是 Driver 路径——因为它们拿到的不是写死在源码里的 kernel，而是运行时才编出来的 cubin。

## 5. 异步执行：launch 开销账与 sticky error

### 5.1 每次 launch 的固定账

一次 kernel launch 的固定开销（近似值，随机器与 driver 变化）：

```text
用户态命令构造        ~1 μs
系统调用(进 driver)   ~1~3 μs
门铃 + 硬件响应       ~0.5 μs
────────────────────────────
合计                  ~3~5 μs / launch
```

对一个大 kernel（跑 1 ms）这 5 μs 可以忽略；对一个 2 μs 的小 kernel：

```text
有用时间占比 = 2 / (2 + 5) ≈ 28%
```

**70% 的时间花在发射而不是计算上**。这就是"小 kernel 吞吐上不去"的第一嫌疑：不是 kernel 慢，是 launch 太密。解法有三个方向：把活合并进一个 kernel（编译器的算子融合，第 1 章 roofline 结论的运行时版）、用 CUDA Graphs 把一串 launch 录制成图一次提交、或持久 kernel（常驻循环取活）。launch 开销在硬件侧的完整路径（门铃、命令缓冲）见主教材第 30 课，本章只用到这个总数。

### 5.2 为什么错误在十行之后才报：sticky error

launch 是异步的：`cuLaunchKernel` 返回成功只说明"命令进了队列"，kernel 是否跑挂要到执行时才暴露。driver 的做法是 **sticky error**：错误被记录在 context 的错误状态里，**粘住不放**，直到下一次同步点或错误查询（`cudaGetLastError`）才吐出来。两个后果：

```text
1. 报错的位置不是出错的位置 —— 报错在"下一次同步/查询", 出错在更早的 launch
2. 同一个错误可能被多次查到 —— GetLastError 查询后会清掉, PeekAtLastError 不清
```

排查纪律：**在明确边界上同步并查错**（每个阶段后查一次），而不是程序结尾统一查——否则错误现场早已丢失。

### 5.3 隐式同步：谁在偷偷等你

有些调用表面不叫"同步"，实际会等设备。常见清单：`cudaMemcpy`（默认同步，特别是 D2H）、`cudaMemset`、`cudaDeviceSynchronize`（显式）、以及 legacy default stream（NULL stream）上的操作会与其他 stream 隐式同步。**"开了 4 条 stream 却毫无重叠"的第一排查项就是：是否所有操作都挂在了 NULL stream 上**——NULL stream 是全局串行点，挂在它上面等于没开 stream。

## 6. Stream 与 Event：顺序从哪来

**stream（流）**是设备上的一条命令队列。规则只有两条：

```text
同一 stream 内: 保持提交顺序(先进先执行)
不同 stream 间: 没有任何先后保证
```

**event（事件）**是 stream 上的一个记录点。两个用途：当依赖（别的 stream 可以等它）、当时钟（量 GPU 时间）。一个最小重叠流水：

```text
stream A: H2D(chunk 0) → kernel(chunk 0) → record event E0
stream B: wait E0       → kernel(chunk 1) ...
```

重叠的收益账（示例值）：假设每块数据 copy 4 ms、kernel 4 ms，两块数据串行执行要 `2×(4+4) = 16 ms`；重叠后，kernel 与下一块 copy 并行，总时间压到接近 `4 + 4 + 4 = 12 ms`——**收益上限 = 串行总时间 − 最长单一阶段的时间**（本例省 4 ms，正好一个 copy 的长度）。重叠程度最终受限于较慢的一侧：copy 4 ms vs kernel 4 ms 时各吃一半带宽/算力；一边远慢于另一边时，另一边的时间被完全藏掉。

计时规则：用 `cudaEventRecord` + `cudaEventElapsedTime` 量 GPU 时间，**不要用 host 侧的 wall clock 计时**——host 返回不代表 device 完成，wall clock 量到的是"提交时间 + 排队时间 + 执行时间"的混合物。

## 7. 内存管理：六对概念逐个定义

| 概念对 | 一句话定义与差异 |
|---|---|
| device allocation vs pinned host memory | 前者在 GPU 显存；后者是**页锁定**的 host 内存，DMA 引擎可直达。普通 pageable host 内存要先 staging copy 进 pinned 区，H2D 带宽可能差一倍以上（近似值）——所以高频 H2D 要 `cudaMallocHost` |
| synchronous vs async copy | 同步版调用返回即完成；异步版（`cudaMemcpyAsync`）只入队，必须挂 stream 并配合 event/sync 使用 |
| memory pool vs 直接分配 | `cudaMalloc` 每次分配的驱动往返约几十微秒量级（近似）；pool 从预分配块里切，降到亚微秒——与"内存池"在第 1 章内存规划思想同构 |
| virtual address vs physical allocation | VMM API 把"保留地址范围"与"提交物理页"分开，支持惰性提交与细粒度管理 |
| unified memory vs 显式拷贝 | UM 让 host/device 共享同一地址，缺页时驱动自动迁移；代价是**首次触达的 page fault**（每页几十微秒量级，近似）。迁移/预取（`cudaMemPrefetchAsync`）可以显式控制 |
| stream-ordered allocation | 分配本身挂在 stream 上排队执行，与使用它的操作天然保序，避免"提前释放"类错误 |

一句话总结：**性能问题常常不是 kernel 本身，而是频繁分配、隐式同步、page fault、挂错 stream 或 host 内存没固定**。kernel 只负责算，这六对概念负责"数据能不能及时、正确地到达它手上"。

## 8. 错误排查矩阵

| 现象 | 首先检查 | 为什么 |
|---|---|---|
| module load failed | driver/toolkit 版本、架构、PTX/cubin、符号 | 加载期校验全在这里（第 2 章兼容性账） |
| invalid argument | 参数类型、指针、对齐、grid/block、shared bytes | launch 配置与 cubin 元数据不符（第 3 章 ABI 第 6 项） |
| launch 后结果错 | 边界、同步、stream、生命周期、data race | 异步语义下"读结果"的时机错了 |
| launch 返回很快但总体慢 | 隐式同步、H2D/D2H、频繁 launch、JIT | 时间花在发射与搬运，不在 kernel |
| 错误在很晚才出现 | 异步 sticky error | 错误粘在 context，等下一次查询才暴露 |
| 多 stream 结果不稳定 | 缺 event 依赖、复用 buffer、host 生命周期 | 跨 stream 无顺序保证，依赖没建立 |

## 9. 检查点

完成以下四项才算通过本章：

1. 画出 `cuInit → cuCtxCreate → cuModuleLoad → cuModuleGetFunction → cuLaunchKernel` 的对象链，在每一步标出"谁被创建、谁校验什么"；
2. 手算：kernel 2 μs、launch 5 μs，10000 次独立 launch 的吞吐利用率（有用时间/总时间）是多少；
3. 解释 sticky error 的两个后果，并写出"分阶段同步查错"的排查模板；
4. 设计一个双 stream 流水：chunk copy 3 ms、kernel 7 ms、两块数据，手算串行时间与重叠后时间的下限。

## 10. 下一步与扩展阅读

本章从 API 视角把一次执行串起来了。两个方向的深水区：命令提交的硬件路径（门铃、命令缓冲、GPU MMU）见主教材第 30 课；code object 的格式细节（ELF、重定位、JIT）见第 29 课。下一章（GPU 07：性能分析与 Nsight）回答"这些账怎么在真机上测出来"。

- 官方：[CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)、[CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)、[CUDA 异步并发章节](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-concurrent-execution)；
- 与本课程的关系：toycc 的 runtime（numpy 参考执行器）只解决"对不对"，本章的 runtime 解决"怎么在真设备上执行"——两者合起来才是完整 runtime 的定义（执行 + 正确性裁判）。

**导航**：⬅ [上一章](05_triton_compiler.md)（Triton 编译器）　｜　[下一章](07_profiling_performance.md)（性能分析与 Nsight）➡
