# GPU 工具链专题

这一专题补齐从 CUDA kernel 到 GPU 硬件、编译器后端、运行时和性能工具之间的连接。第 21~24、27~30 课负责建立全景和系统地基；本专题负责把工作中最常见的 NVIDIA GPU 生态名词拆开，并给出可复用的排查方法。

```text
CUDA C++ / Triton / CUTLASS
          ↓
      nvcc / Triton compiler / template instantiation
          ↓
      PTX → ptxas → cubin / fatbin → module loading
          ↓
      Driver / Runtime / Stream / Event / Launch
          ↓
      SM / warp / register / shared memory / cache / Tensor Core
          ↓
      Nsight / counters / benchmark / NCCL / multi-GPU
```

## 分章学习

1. [CUDA 编程模型：线程、内存与同步](01_cuda_programming_model.md)
2. [CUDA 工具链与 PTX：从 `.cu` 到 `cubin`](02_cuda_toolchain_ptx.md)
3. [GPU ISA、寄存器分配与 ABI](03_gpu_isa_registers_abi.md)
4. [CUTLASS / CuTe：高性能 GEMM 的软件分层](04_cutlass_cute.md)
5. [Triton：从 Python kernel 到 GPU 代码](05_triton_compiler.md)
6. [Runtime / Driver：模块、内存、Stream、Event 与提交](06_runtime_driver.md)
7. [性能分析：从 benchmark 到 Nsight 指标](07_profiling_performance.md)
8. [多 GPU、NCCL 与通信计算重叠](08_multi_gpu_nccl.md)
9. [端到端案例：一个算子从模型到 GPU](09_end_to_end.md)

## 推荐顺序

- Kernel 开发：01 → 07 → 04/05；
- 编译器后端：01 → 02 → 03 → 06；
- Runtime/驱动：02 → 03 → 06 → 07 → 08；
- 性能工程：01 → 04 → 07 → 08；
- 自研 GPU：01 → 03 → 06 → 第 27~30 课 → 09。

## 版本边界

CUDA、PTX、GPU 架构、驱动和 profiler 的具体选项会随 toolkit 和 compute capability 改变。文中的命令和指标用于建立方法，执行时以当前机器上的 `--help`、目标架构文档和实际报告为准；不要把某一代 NVIDIA GPU 的数值硬编码成所有 GPU 都成立的规律。

## 官方入口

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/overview.html)
- [Triton Documentation](https://triton-lang.org/main/index.html)
- [Nsight Compute](https://docs.nvidia.com/nsight-compute/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)

