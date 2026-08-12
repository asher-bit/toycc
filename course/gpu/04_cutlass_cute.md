# GPU 第 4 章：CUTLASS / CuTe——高性能 GEMM 的软件分层

## 1. 本章目标

- 知道什么时候应该调用库、复用 CUTLASS、写 Triton，什么时候才值得手写 CUDA；
- 看懂 GEMM 的层次化 tiling、copy、MMA 和 epilogue；
- 理解 CuTe 的 layout algebra 与线程/数据映射；
- 能用 CUTLASS profiler 做候选 kernel 的公平比较。

## 2. CUTLASS 解决什么问题

CUTLASS 不是图编译器，也不是 CUDA runtime。它提供 CUDA C++ 中可组合、可特化的高性能线性代数抽象，尤其适合 GEMM 及其相关计算。它把数据移动、线程组织、矩阵乘累加、数据类型、布局和 epilogue 拆成可复用组件。

```text
问题层：GEMM / convolution / attention block
算法层：tile shape、pipeline、split-K、epilogue
线程层：warp/thread tile、copy atom、MMA atom
硬件层：shared/register/Tensor Core/async copy
```

## 3. GEMM 的五层分解

以 `C = A × B + C` 为例：

1. threadblock tile：一个 block 负责输出矩阵的一块；
2. warp tile：block 内不同 warp 分工；
3. MMA tile：warp 使用 Tensor Core 或目标矩阵指令；
4. copy tile：线程把 global 数据搬到 shared，再搬到 register；
5. epilogue：累加结果转类型、加 bias、激活、量化并写回。

调优不能只改变矩阵 tile。tile 大小、stage 数、数据类型、布局、寄存器和 shared memory 会一起改变 occupancy、访存和 Tensor Core 利用率。

## 4. CuTe 的核心直觉

CuTe 把 shape、stride、layout、tensor 和 atom 组合起来表达“哪一个线程拿哪一块数据”。这种表达比手写大量 `threadIdx`/`warp` 下标更适合复用，但模板错误也会很难读。

```cpp
// 示意：不是完整可编译程序
auto layout = make_layout(make_shape(Int<8>{}, Int<16>{}),
                          make_stride(Int<16>{}, Int<1>{}));
auto tensor = make_tensor(ptr, layout);
```

阅读 CUTLASS 时先问三个问题：

- 当前 layout 的 shape/stride 是什么；
- 当前线程、warp、CTA 如何切分这个 layout；
- copy/MMA atom 的输入输出布局是否匹配。

## 5. API 选择与决策

| 场景 | 优先选择 |
|---|---|
| 标准 GEMM/conv，已有成熟算法 | cuBLAS/cuDNN 或调用框架已有 kernel |
| 需要融合 epilogue、特殊 dtype/layout | CUTLASS/CuTe |
| 想快速表达自定义 tile 算法 | Triton |
| 需要完全控制 ISA、异步机制或新硬件 | CUDA C++ / PTX / 后端 |
| 只是图级组合 | TVM/MLIR/框架编译器，不要直接从 CUTLASS 开始 |

## 6. Profiler 与实例选择

CUTLASS 的 profiler 不只是跑一个数字，它帮助你比较不同 problem size、dtype、tile、split-K 和 epilogue 组合。记录实验时必须固定：GPU、时钟/功耗状态、数据布局、warmup、重复次数、stream、workspace 和校验方式。

```text
候选生成 → 编译/实例化 → correctness check → warmup
 → 多次计时 → 记录 TFLOP/s、带宽、寄存器、shared、occupancy
 → 与 baseline 比较
```

## 7. 源码阅读地图

CUTLASS 仓库通常按以下层次组织：

- `include/cute/`：layout、tensor、copy/MMA atom 等基础抽象；
- `include/cutlass/`：device kernel、矩阵乘、数据类型和工具；
- `examples/`：可运行的 kernel 使用方式；
- `tools/profiler/`：命令行 profiling；
- `test/unit/`：组件和完整计算测试。

读源码顺序建议是 `example → Collective/Kernel → TiledMma/TiledCopy → atom → arch wrapper`，不要从最深的模板错误开始。

## 8. 练习

1. 用 CUTLASS profiler 找一个 FP16 GEMM baseline；
2. 改变 epilogue，加入 bias 或 activation，并检查数值误差；
3. 用 layout 图画出一个线程到矩阵元素的映射；
4. 对比 CUTLASS、Triton 和手写 CUDA 的代码量、可控性、编译时间和性能。

参考：[CUTLASS Overview](https://docs.nvidia.com/cutlass/latest/overview.html)、[CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)。

