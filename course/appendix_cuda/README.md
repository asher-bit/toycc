# 附录 C：CUDA 编程指南

> 本附录译自 NVIDIA 官方《CUDA Programming Guide》，按官方章节分文件存放，便于按需查阅与续译。
> 完整与最新内容以 [官方原文](https://docs.nvidia.com/cuda/cuda-programming-guide/) 为准。
> 本仓库的 CUDA 实战在第 22~24、34 课与 GPU 工具链专题 01~05；本附录只回答"官方指南里都有什么"，不重复讲 CUDA 概念。

---

## CUDA 与 CUDA 编程指南

CUDA 是 NVIDIA 开发的并行计算平台和编程模型，通过利用 GPU 的算力实现显著的性能提升。它让开发者能够加速计算密集型应用，被广泛用于深度学习、科学计算和高性能计算（HPC）等领域。

《CUDA 编程指南》是 CUDA 编程模型和 GPU 代码编写的官方权威资料，内容覆盖从 CUDA 编程模型和 CUDA 平台，到语言扩展的细节，再到如何使用具体软硬件特性的各个方面。它为初学者提供进入 CUDA 的路径，也是开发者在构建 CUDA 应用时的必备参考资料。

## 本指南的组织

即使主要使用库、框架或领域特定语言（DSL）的开发者，理解 CUDA 编程模型以及 GPU 如何执行代码，也有助于看懂抽象层背后正在发生什么。因此本指南从一章与具体编程语言无关的 CUDA 编程模型讲起，任何想了解 GPU 和 GPU 上代码执行概念的人——即便不是开发者——都能从中受益。

本指南分为五个主要部分：

### 第 1 部分：引言与编程模型抽象

以与语言无关的方式概述 CUDA 编程模型，并简要介绍 CUDA 平台。

适合任何想了解 GPU 以及在 GPU 上执行代码概念的人阅读，即使并非开发者。

### 第 2 部分：用 CUDA 在 GPU 上编程

介绍用 C++ 和 Python 进行 GPU 编程的基础。

适合任何想入门 GPU 编程的人阅读。本部分面向教学而非求全，讲解 CUDA 编程中最重要和最常使用的部分，并涉及一些常见的性能考量。

### 第 3 部分：CUDA 进阶

介绍 CUDA 的一些更进阶的特性，它们既能带来更细粒度的控制，也提供更多性能优化机会，包括在单个应用中使用多 GPU。

本部分末尾对第 4 部分涵盖的功能做一段简要导览，按开发者何时以及为何会需要某项特性来排序。

### 第 4 部分：CUDA 特性

本部分完整覆盖 CUDA 的特定特性，例如 CUDA Graphs、动态并行（Dynamic Parallelism）、与图形 API 的互操作，以及统一内存（Unified Memory）。

当需要了解某个 CUDA 特性的完整图景时，应查阅本部分。在可能的情况下，前面章节已经先介绍并说明本部分涉及的特性及其动机。

### 第 5 部分：技术附录

技术附录提供 CUDA 对 C++ 高级语言支持的参考文档、硬件相关的规格，以及其它技术规范，作为对 CUDA 各元素语法、语义和技术行为的具体参考。

---

## 三个部分之间的关系

第 1~3 部分为 CUDA 新手提供一段有引导的学习体验，对任何经验水平的 CUDA 开发者也能提供新的认识与信息。

第 4、5 部分提供关于特定特性与详细主题的大量信息，作为开发者在编写 CUDA 应用时需要深入了解细节时的精选且组织良好的参考。

---

## 本附录分章索引

| 文件 | 对应官方章节 | 内容 |
|---|---|---|
| [01_intro.md](01_intro.md) | 1.1 Introduction | GPU 的由来、相对 CPU 的设计取舍、库/框架/DSL 三种利用 GPU 的方式 |
| [02_programming_model.md](02_programming_model.md) | 1.2 Programming Model | 异构系统、SM/grid/block/warp、SIMT、Tile 编程、GPU 内存层次 |
| [03_platform.md](03_platform.md) | 1.3 The CUDA Platform | 计算能力、Toolkit/驱动、PTX、cubin/fatbin、二进制与 PTX 兼容性、JIT |
| [04_intro_cuda_cpp.md](04_intro_cuda_cpp.md) | 2.1 Intro to CUDA C++ | NVCC 编译、kernel 发射、内存管理、同步、错误检查、集群 |
| [05_intro_cuda_python.md](05_intro_cuda_python.md) | 2.2 Intro to CUDA Python | CUDA Python 生态、SIMT kernel、cuPy ndarray、同步、错误检查 |
| [06_writing_simt_kernels.md](06_writing_simt_kernels.md) | 2.3 Writing SIMT Kernels | 线程层级、设备内存空间、内存性能/合并访问/bank 冲突、原子、占用率 |

图片占位清单见 [images/README.md](images/README.md)。

---

**导航**：CUDA 主线见第 22 课（GPU 编译器）、第 23 课（Kernel 与性能分析）、第 24 课（自研 GPU 工具链）和第 34 课（Triton 与 CUTLASS），以及 GPU 工具链专题第 1~5 章。