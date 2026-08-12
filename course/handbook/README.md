# 机器学习编译器与 GPU 工具链新人手册

## 这本手册要解决什么问题

工作中经常会同时听到 TVM、LLVM、MLIR、CUDA、PTX、CUTLASS、Triton、寄存器、occupancy、runtime、driver 等词。它们不是同一层的东西：有的是编译器基础设施，有的是 GPU 编程模型，有的是高性能库，有的是硬件概念，还有的是运行时或调试工具。

本手册的目标，是让新员工能够把一个问题放回正确的层次：

```text
模型 / 算子
   ↓ 前端与语义
高层 IR / 图编译器（TVM、StableHLO、MLIR 等）
   ↓ 优化、布局、调度、bufferization
循环 / kernel IR（TIR、Linalg、Triton IR 等）
   ↓ 代码生成与后端
LLVM IR / PTX / GPU ISA / 目标文件
   ↓
Runtime / Driver / Library / Profiler
   ↓
GPU 硬件：线程层次、寄存器、共享内存、缓存、Tensor Core、互联
```

## 推荐的总分类

| 知识域 | 主要回答的问题 | 本仓库现有内容 | 后续优先补充 |
|---|---|---|---|
| 0. 入职导航 | 我该先学什么、遇到问题找哪一层？ | 本手册、知识地图 | 岗位路径、术语索引、任务模板 |
| 1. 编程与编译器地基 | C++、Linux、构建、测试、IR、Pass 是什么？ | 第 0~2、10、16 课，附录 | CMake/Bazel、调试、数据流与支配分析 |
| 2. 模型与前端 | 模型如何进入编译器？动态形状和控制流怎么办？ | 第 17 课 | PyTorch FX/Export、ONNX、StableHLO、前端合法化 |
| 3. IR 与编译器中间层 | 图、张量、循环、内存、机器指令如何表示？ | 第 1、7、11、14、25、26 课 | 类型系统、布局/形状、符号、LLVM/MLIR 核心源码 |
| 4. 优化与调度 | 为什么能改、改了是否正确、怎样变快？ | 第 3~6、12、13、18、19 课 | 依赖分析、别名分析、向量化、并行化、代价模型 |
| 5. GPU 硬件与微架构 | 一个线程块如何在 SM 上执行？瓶颈在哪里？ | 第 15、21、22 课 | 指令吞吐、缓存、Tensor Core、异步拷贝、TMA、互联 |
| 6. GPU 编程模型与 Kernel | CUDA kernel、线程、warp、同步和内存访问如何写？ | 第 23 课、GPU 专题 1~3 | CUDA C++、CUDA Runtime/Driver、PTX、内联汇编 |
| 7. Kernel DSL 与高性能库 | CUDA、Triton、CUTLASS 分别解决什么问题？ | GPU 专题 4~5 | Triton IR/编译链、CUTLASS/CuTe、cuBLAS/cuDNN/NCCL |
| 8. 编译器后端与目标工具链 | IR 怎样变成 PTX、SASS、汇编和目标文件？ | 第 24~26 课，LLVM/MLIR 专题 | nvcc/nvrtc/ptxas、寄存器分配、ABI、链接与 fatbin |
| 9. Runtime、Driver 与系统软件 | 代码怎样加载、分配内存、发射 kernel、同步和报错？ | 第 2、7、17、24、29、30 课、GPU 专题 6 | CUDA Driver API、模块加载、stream/event、图执行、设备运行时 |
| 10. 性能、正确性与调试 | 如何证明正确、测量性能、定位瓶颈？ | 第 2、16、19、23、30 课、GPU 专题 7 | Nsight Systems/Compute、指标体系、微基准、数值误差、性能回归 |
| 11. 工程与生产化 | 怎样把研究代码变成可维护的工具链？ | 第 16、20、24、27~30 课 | CI、兼容性、版本矩阵、发布、回滚、二分和故障手册 |
| 12. 多 GPU 与分布式 | 通信、切分、集合通信如何进入编译器？ | GPU 专题 8~9 | NCCL、collective、sharding、流水并行、通信计算重叠 |

## 这些名词应该放在哪里

| 名词 | 正确定位 | 不要把它误认为 |
|---|---|---|
| TVM | AI 编译器框架，包含 Relax、TensorIR、编译与运行时组件 | 一个单独的“GPU 后端” |
| LLVM | 通用编译器基础设施，包含 LLVM IR、优化器、后端和 MC | 只能编译 CPU 的编译器 |
| MLIR | 可扩展、多层 IR 与编译器基础设施 | LLVM IR 的新名字 |
| CUDA | NVIDIA GPU 的编程模型、工具链和运行时生态 | 只有 CUDA C++ 语法 |
| PTX | NVIDIA GPU 的虚拟指令集/中间汇编 | 最终硬件机器码 SASS |
| CUTLASS / CuTe | CUDA 上的高性能线性代数模板与布局/线程组织抽象 | 一个通用图编译器 |
| Triton | 面向并行 kernel 的语言和编译器 | CUDA runtime 或数学库 |
| 寄存器 | 硬件资源，也是后端寄存器分配的约束对象 | 只属于 CUDA 的概念 |
| Runtime | 加载、内存、队列、同步、发射和执行支持 | 编译器后端本身 |
| Profiler | 观测执行时间、吞吐、访存、占用率和依赖 | 只看一个 kernel 的 wall time |

## 新员工的四条学习路径

不要要求所有人把所有章节按同一深度学完。先完成公共地基，再按岗位选择主路径：

- [编译器 Pass / IR 路径](paths.md#路径-a编译器与-ir)
- [GPU Kernel / 性能路径](paths.md#路径-bgpu-kernel-与性能)
- [后端 / 自研芯片工具链路径](paths.md#路径-c后端与自研芯片工具链)
- [Runtime / 系统软件路径](paths.md#路径-druntime驱动与系统软件)

## 现有课程怎么放进新体系

现有 `lesson00~26.md` 不删除，作为“按周推进的旧主线”保留；新的侧边栏按知识域排列，避免新人把编号误读成技术层次。TVM、LLVM、MLIR 的专题目录继续保留，并作为对应知识域的深度材料。

## 官方资料入口

- [Apache TVM Documentation](https://tvm.apache.org/docs/)
- [LLVM Documentation](https://llvm.org/docs/)
- [MLIR Documentation](https://mlir.llvm.org/docs/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/overview.html)
- [Triton Documentation](https://triton-lang.org/main/index.html)

