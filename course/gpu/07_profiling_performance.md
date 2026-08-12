# GPU 第 7 章：性能分析——从 benchmark 到 Nsight 指标

## 1. 本章目标

- 建立可复现的 GPU benchmark；
- 区分 kernel 时间、端到端时间、launch 开销和同步时间；
- 用 Nsight Systems 定位时间线问题，用 Nsight Compute 定位 kernel 瓶颈；
- 把 profiler 指标转成具体代码修改，而不是只收集数字。

## 2. 正确测速的最小协议

```text
固定 GPU / 电源和时钟条件
 → 预热，排除首次 JIT/分配
 → 明确同步边界
 → 多次重复，报告 median/p90/min
 → 分离 H2D、kernel、D2H 和总时间
 → 同时做 correctness check
```

CUDA kernel 是异步发射的。用 host wall clock 包住 launch 而没有同步，测到的可能只是 enqueue 时间。CUDA events 更适合测同一设备时间线中的 kernel 区间，但端到端报告仍需要 host 时间和 Nsight Systems。

## 3. Roofline 与瓶颈假设

```text
arithmetic intensity = FLOPs / bytes moved
可达性能 ≈ min(峰值计算吞吐, 峰值带宽 × arithmetic intensity)
```

Roofline 是建立假设的工具，不是自动证明。真实 kernel 还受指令混合、cache、同步、占用率、发散、Tensor Core 使用、访存事务和 launch 影响。

## 4. 两类 Nsight 工具

### 4.1 Nsight Systems：看系统时间线

适合回答：

- CPU 是否在等待 GPU；
- H2D/D2H、kernel、通信和同步如何交错；
- 多 stream 是否真的重叠；
- 是否存在大量小 kernel、隐式同步、JIT 或分配；
- NCCL 通信和计算是否重叠。

### 4.2 Nsight Compute：看单个 kernel

适合回答：

- 内存吞吐和 cache 命中如何；
- warp stall 原因是什么；
- issue/execute/pipe 利用率如何；
- 寄存器、shared memory 和 occupancy 是否成为限制；
- 分支发散、访存合并和 Tensor Core 指令是否符合预期。

不要把两个工具互相替代：Systems 解决“整个程序时间花在哪里”，Compute 解决“这个 kernel 为什么这样跑”。

## 5. 指标到行动的映射

| 观察 | 可能原因 | 下一步实验 |
|---|---|---|
| kernel 很短但总时间很长 | launch/同步/调度开销 | 融合、批量化、CUDA Graph、减少 host round trip |
| DRAM throughput 高、计算利用率低 | memory-bound | 合并访问、tile、复用、压缩、布局 |
| local load/store 多 | register spill | 减少 live range、tile/unroll、检查内联 |
| eligible warps 少、stall memory | 延迟未隐藏或访存慢 | 增加并行、改善访问、检查 occupancy |
| shared throughput/波动异常 | bank conflict 或同步 | 改 layout、padding、减少 barrier |
| Tensor Core 利用率低 | dtype/layout/shape/指令路径不匹配 | 检查 MMA tile、对齐、编译目标和 epilogue |
| 多 stream 不重叠 | 依赖、默认 stream、内存带宽或隐式同步 | event 图、stream 语义、copy engine |

每次只改变一个主要变量，并保存代码、编译参数、输入 shape、GPU、driver 和报告。

## 6. 端到端性能回归

性能测试至少分三层：

1. microbenchmark：一个 kernel 的吞吐和资源；
2. operator benchmark：包含 layout、workspace、多个 kernel；
3. model benchmark：包含前端、编译、runtime、通信和端到端延迟。

单个 GEMM 变快，不代表模型变快；可能因为 layout transform、kernel launch 或通信成本被放大。性能报告要同时标出优化收益和新增成本。

## 7. 源码与工具地图

- 第 19、23 课：benchmark、roofline、kernel 优化；
- 第 30 课：硬件计数器到 profiler；
- Nsight Systems：timeline、NVTX、CPU/GPU/通信关联；
- Nsight Compute：kernel report、sections、metrics、source correlation；
- CUDA events：设备时间线计时；
- `compute-sanitizer`：越界、竞争和同步问题，不负责替代 profiler。

## 8. 练习

1. 为 vector add、reduction、GEMM 写统一 benchmark harness；
2. 用 Systems 找出一个隐式同步；
3. 用 Compute 找出一个寄存器 spill 或访存不合并；
4. 写一页性能复盘：现象、假设、证据、修改、回归风险。

参考：[Nsight Systems](https://docs.nvidia.com/nsight-systems/)、[Nsight Compute](https://docs.nvidia.com/nsight-compute/)、[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。

