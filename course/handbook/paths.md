# 新员工学习路径与工作任务映射

## 公共地基：所有岗位都要完成

建议入职前两周完成以下内容：

1. 跑通 `python -m toycc.examples.demo`，理解 IR → Pass → Codegen → Runtime；
2. 学会阅读一个 Python/C++ 编译器仓库：入口、数据结构、Pass 注册、测试和日志；
3. 掌握 Linux shell、Git、CMake、编译/链接、gdb/lldb 和单元测试基本用法；
4. 能解释 SSA、基本块、控制流、数据流、内存层次、线程/warp/block；
5. 能用一个小例子同时看出“正确性结果”和“性能结果”的区别。

对应材料：第 0~2、10、15、16 课，附录 A/B，外加[知识地图](knowledge_map.md)。

## 路径 A：编译器与 IR

适合：Relax/TIR、MLIR、LLVM、图优化、算子编译、Pass 开发。

```text
计算图与 IR
 → Pass 与数据流
 → TIR / 调度
 → TVM 源码精读
 → MLIR IR / Dialect / Conversion
 → LLVM IR / New PM / Backend
```

重点任务：

- 给 toycc 增加一个算子和一个 Pass；
- 为 Pass 写 verifier、单测和 FileCheck 风格回归测试；
- 追踪一个算子从高层 IR 到循环、内存和 LLVM IR 的下降；
- 解释一个优化为什么合法，以及改变了哪些分析结果。

主要材料：第 1~14、17、25~26 课，`tvm/`、`llvm/`、`mlir/` 专题。

## 路径 B：GPU Kernel 与性能

适合：Kernel 开发、算子优化、CUDA/Triton、性能分析、库开发。

```text
GPU 微架构
 → CUDA 线程/内存/同步
 → PTX 与编译流程
 → GEMM / Convolution / Attention
 → Triton 或 CUTLASS
 → Nsight / roofline / 微基准
```

重点任务：

- 写一个向量加、归约、矩阵乘和 fused kernel；
- 用寄存器、共享内存、L2、全局内存和 Tensor Core 解释性能差异；
- 比较 CUDA、Triton、CUTLASS 实现同一个 GEMM 的抽象边界；
- 用 profiler 证明瓶颈是计算、访存、同步、发散还是 launch overhead。

主要材料：第 15、19、21~23 课。后续必须补充 CUDA、CUTLASS、Triton 专题。

## 路径 C：后端与自研芯片工具链

适合：LLVM backend、PTX/ISA、汇编器、指令选择、寄存器分配、编译器移植。

```text
ISA / ABI / 寄存器文件
 → LLVM IR / Machine IR
 → TableGen / 指令选择
 → 调度 / 寄存器分配 / 栈帧
 → 汇编器 / 反汇编器 / 链接器
 → Driver / Runtime / 性能工具
```

重点任务：

- 给一个目标增加指令描述、寄存器类和编码测试；
- 解释一个 kernel 的寄存器分配和 spill；
- 从 LLVM IR 追到目标汇编和目标文件重定位；
- 定义自研 GPU 的编译器、汇编器、运行时和调试器边界。

主要材料：第 22、24~26 课，LLVM/MLIR 专题。后续必须补充 PTX/SASS、ABI、链接和设备 ISA 专题。

## 路径 D：Runtime、驱动与系统软件

适合：Runtime、Driver、内存管理、模块加载、设备执行、框架接入。

```text
Host API
 → Context / Device
 → Module / Kernel 加载
 → Memory / Stream / Event
 → Launch / Synchronization
 → Error / Profiling / Multi-device
```

重点任务：

- 解释一个 kernel 从编译产物到设备执行的完整生命周期；
- 分清编译期、链接期、模块加载期和运行期错误；
- 设计 stream/event、异步拷贝、内存池和错误传播；
- 给一个新后端接入 runtime，并写端到端测试。

主要材料：第 2、7、16、17、24 课。后续必须补充 CUDA Driver API、设备运行时、NCCL 和多 GPU。

## 30/60/90 天交付目标

| 时间 | 应达到的能力 | 建议交付物 |
|---|---|---|
| 0~30 天 | 能跑、能读、能定位入口 | 一张个人知识地图；一个 toycc 小改动；一个性能/正确性测试 |
| 31~60 天 | 能改 Pass、Kernel 或 Runtime 的局部模块 | 一个带测试的真实仓库改动；一份 IR 或 profiler 分析报告 |
| 61~90 天 | 能独立处理跨层问题 | 一份从模型/算子到 kernel/runtime 的端到端问题复盘 |

