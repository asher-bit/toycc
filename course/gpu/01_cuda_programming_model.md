# GPU 第 1 章：CUDA 编程模型——线程、内存与同步

## 1. 本章目标

- 看懂一个 CUDA kernel 的 grid/block/thread 映射；
- 区分 global、shared、local、constant 等地址空间；
- 理解 warp、分支发散、barrier 和原子操作的边界；
- 能把一个性能问题归因到并行划分、访存、同步或 launch。

## 2. CUDA 不是“把 C++ 放到 GPU 上运行”

CUDA 程序有 host 和 device 两个执行世界。host 代码负责选择设备、分配内存、准备参数和发射 kernel；device kernel 由大量线程执行。一个 kernel 的线程组织成 grid，grid 由 block 组成，block 内有 thread。block 可以被调度到任意 SM，因此不同 block 默认不能依赖隐式的执行顺序。

```cuda
__global__ void vector_add(const float* a, const float* b,
                           float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

int threads = 256;
int blocks = (n + threads - 1) / threads;
vector_add<<<blocks, threads>>>(a, b, out, n);
```

索引公式同时连接了三个层次：编程模型中的线程 ID、编译器里的地址计算、硬件里的内存事务。写 kernel 时要同时检查边界、数据类型、对齐和线程映射。

## 3. warp 与 SIMT

GPU 通常以 warp 为调度和执行的基本粒度；CUDA 源码看起来是“每线程一份控制流”，硬件实际会将同一 warp 的线程分组执行。一个 warp 内线程走不同分支时会发生发散：硬件可能串行执行不同路径，再用活动掩码屏蔽不参与的线程。

```cuda
if ((threadIdx.x & 1) == 0) {
  // 一半线程走这里
} else {
  // 另一半线程走这里
}
```

发散不等于一定错误，但会降低有效吞吐。更危险的是在部分线程执行后调用要求全体线程到达的 `__syncthreads()`，可能造成死锁或未定义行为。同步的正确性必须先于性能讨论。

## 4. 内存空间与可见性

| 空间 | 典型生命周期/可见性 | 常见用途 |
|---|---|---|
| Register | 每线程，最快但容量有限 | 累加器、临时值 |
| Local | 每线程逻辑私有，可能落到显存 | 寄存器 spill、局部数组 |
| Shared | block 内共享 | tile、归约、中间交换 |
| Global | 全设备可访问，延迟高 | 输入、输出、权重 |
| Constant/Texture | 特定缓存与访问语义 | 广播只读数据、纹理访问 |

`__syncthreads()` 主要建立 block 内的控制与共享内存协作边界；它不是全 grid barrier，也不能自动修复 global memory 的所有可见性问题。跨 block 协作要使用分阶段 kernel、cooperative launch 或更高层的同步机制。

## 5. 全局内存合并访问

理想情况下，一个 warp 的相邻线程访问相邻地址，硬件可以用较少的内存事务服务请求。把二维 tile 按错误的 stride 访问、让线程随机跳跃或使用过大的结构体，都会放大事务数量。

```text
连续访问：thread lane k → base + k       通常易合并
跨行访问：thread lane k → base + k*stride  可能产生更多事务
```

不要只看“带宽利用率”一个数：还要看访问是否对齐、L1/L2 命中、请求是否被缓存、实际有效字节和 kernel 的算术强度。

## 6. 共享内存与 bank conflict

共享内存划分成 bank。一个 warp 的线程访问不同 bank 时可以并行；多个线程访问同一 bank 的不同地址时可能产生冲突，访问被拆成多个阶段。广播同一地址在某些架构上有特殊处理，不能简单套用“相同地址必然冲突”。

常见修复包括改变 tile 布局、padding、转置加载或让线程映射与后续计算保持一致。CUTLASS、CuTe 和 Triton 的 layout 抽象，实质上都在帮助程序员表达这种线程—数据映射。

## 7. 同步、原子和内存序

- block barrier：协作访问 shared memory 时常用；
- warp-level primitive：需要明确参与 mask 和独立线程调度语义；
- atomic：解决竞争更新，但代价是序列化、争用和内存序约束；
- fence：约束内存操作的观察顺序，不能代替线程到达同步；
- stream ordering：同一 stream 中的操作有顺序，跨 stream 需要 event 或显式依赖。

用“谁写、谁读、何时读、如何证明已可见”四个问题检查并发代码，比背 API 名称更可靠。

## 8. 源码和工具地图

- CUDA 源码：`.cu` kernel、host wrapper、错误检查宏、stream/event 管理；
- 编译器：`nvcc` 编译 host/device，`nvrtc` 运行时编译；
- 运行时：`cudaMalloc`、`cudaMemcpyAsync`、kernel launch、event；
- 观测：`cudaGetLastError`、`cudaPeekAtLastError`、CUDA events、Nsight。

```bash
nvcc -O3 -lineinfo -arch=sm_XX vector_add.cu -o vector_add
compute-sanitizer ./vector_add
```

`sm_XX` 只是占位符，实际应换成目标 GPU 支持的架构。`compute-sanitizer` 主要帮助发现越界、竞争和同步问题，但它不是性能工具。

## 9. 练习

1. 写 vector add，并分别使用 128、256、512 threads/block；
2. 写 block-level reduction，解释每次 `__syncthreads()` 的必要性；
3. 构造一个 shared-memory bank conflict，再用 padding 修复；
4. 故意制造一个越界访问，用 sanitizer 定位；
5. 记录 kernel 时间、有效带宽和理论带宽，不要只比较 wall time。

参考：[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)、[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。

