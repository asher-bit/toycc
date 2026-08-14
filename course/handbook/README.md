# 编译器与 GPU 工具链新人手册（偏编译器方向）

## 这本手册要解决什么问题

工作中经常会同时听到 TVM、LLVM、MLIR、CUDA、PTX、CUTLASS、Triton、寄存器、occupancy、runtime、driver 等词。它们并不处在同一层：有的是编译器基础设施，有的是 GPU 编程模型，有的是高性能库，还有的是硬件资源、运行时组件或调试工具。

新人最容易遇到的问题，不是某个术语没背下来，而是不知道问题应该放在哪一层。本手册先建立一张共同地图，再带你沿着主教材逐步深入：

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

## 先分清三种材料

仓库里的内容分成三层，阅读方式不同：

1. **主教材**：`lesson00~35.md`。按顺序阅读，负责建立从模型到硬件的连续叙事；
2. **专题课**：`tvm/`、`llvm/`、`mlir/`、`gpu/`。主教材遇到某个技术域后，再按岗位选择深入；
3. **参考资料**：词汇表、知识地图、源码路径和官方文档。遇到术语或代码问题时查询，不要求一次读完。

主教材回答“这件事为什么存在、它和前后环节有什么关系”；专题课回答“这个组件内部是怎么实现的”；参考资料回答“我现在卡住的名词或路径是什么”。

## 推荐的总分类

| 知识域 | 主要回答的问题 | 本仓库现有内容 | 后续优先补充 |
|---|---|---|---|
| 0. 入职导航 | 我该先学什么、遇到问题找哪一层？ | 本手册、知识地图、[岗位路径](paths.md)、[词汇表](../glossary.md) | 任务模板、常见故障手册 |
| 1. 编程与编译器地基 | C++、Linux、构建、测试、IR、Pass 是什么？ | 第 0~2、10、16 课，附录 | CMake/Bazel、调试、数据流与支配分析 |
| 2. 模型与前端 | 模型如何进入编译器？动态形状和控制流怎么办？ | 第 17 课 | PyTorch FX/Export、ONNX 深读、StableHLO、前端合法化 |
| 3. IR 与编译器中间层 | 图、张量、循环、内存、机器指令如何表示？ | 第 1、7、11、14、25、26 课，LLVM/MLIR 专题 | 类型系统、符号形状、LLVM/MLIR 核心源码深读 |
| 4. 优化与调度 | 为什么能改、改了是否正确、怎样变快？ | 第 3~6、12、13、18、19 课 | 依赖分析、别名分析、向量化、并行化、代价模型 |
| 5. GPU 硬件与微架构 | 一个线程块如何在 SM 上执行？瓶颈在哪里？ | 第 15、21、22 课、GPU 专题 3 | 指令吞吐、缓存、Tensor Core 深入、异步拷贝、TMA、互联 |
| 6. GPU 编程模型与 Kernel | CUDA kernel、线程、warp、同步和内存访问如何写？ | 第 23 课、GPU 专题 1~3、6 | 内联 PTX、cooperative groups 进阶、CUDA C++ 完整语法 |
| 7. Kernel DSL 与高性能库 | CUDA、Triton、CUTLASS 分别解决什么问题？ | GPU 专题 4~5、8 | cuBLAS/cuDNN 库边界、CUTLASS 实例化调试、Triton 源码深读 |
| 8. 编译器后端与目标工具链 | IR 怎样变成 PTX、SASS、汇编和目标文件？ | 第 24~26、29 课，LLVM/MLIR 专题、GPU 专题 2~3 | nvrtc、relocatable device code、调试信息与符号 |
| 9. Runtime、Driver 与系统软件 | 代码怎样加载、分配内存、发射 kernel、同步和报错？ | 第 2、7、17、24、29、30 课、GPU 专题 6 | CUDA Graphs、设备端运行时、统一内存进阶 |
| 10. 性能、正确性与调试 | 如何证明正确、测量性能、定位瓶颈？ | 第 2、16、19、23、30 课、GPU 专题 7 | Nsight 真实报告实战、数值误差定位、性能回归自动化 |
| 11. 工程与生产化 | 怎样把研究代码变成可维护的工具链？ | 第 16、20、24、27~30 课 | CI、兼容性、版本矩阵、发布、回滚、二分和故障手册 |
| 12. 多 GPU 与分布式 | 通信、切分、集合通信如何进入编译器？ | 第 32 课、GPU 专题 8~9 | 集合通信算法与拓扑细节、sharding 实测、通信重叠测量 |

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

## 每一章应该怎样阅读

每章都按同一条学习节奏展开：

1. 先看本章要解决的工程问题；
2. 用一个最小例子建立直觉；
3. 回到 toycc 或真实项目，确认这个直觉对应哪些数据结构和函数；
4. 运行实验或阅读输出，验证自己的判断；
5. 完成章末检查点，再决定是否进入扩展阅读。

如果一段内容暂时没有代码或实验，它应该明确说明“这是背景知识”以及“后面哪一章会用到它”。

## 现有课程怎么放进新体系

`lesson00~35.md` 是主教材，不是旧内容。它按“编译器基础 → TVM → 硬件与 GPU → 工具链系统 → LLM 推理与分布式”逐步推进。TVM、LLVM、MLIR 和 GPU 专题是主教材的纵向深入，不要求所有岗位按照同样深度学习。toycc 是主教材前半段的动手教具，课程整体不绑定任何一个框架。

更具体的编辑原则见[教材写作与维护规范](editorial_guide.md)。

## 官方资料入口

- [Apache TVM Documentation](https://tvm.apache.org/docs/)
- [LLVM Documentation](https://llvm.org/docs/)
- [MLIR Documentation](https://mlir.llvm.org/docs/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/overview.html)
- [Triton Documentation](https://triton-lang.org/main/index.html)

