# 知识地图与补课优先级

## P0：新人必须掌握

这些内容决定新人能不能读代码、定位问题和参与评审：

- C/C++：指针、引用、模板、继承、RAII、迭代器、lambda、宏和编译链接；
- Linux 工具：shell、Git、CMake、gdb/lldb、readelf/objdump、grep/rg；
- 编译器基础：词法/语法、AST、IR、SSA、CFG、支配关系、数据流、Pass、验证器；
- ML 编译器基础：算子、形状、布局、动态形状、量化、内存规划、代码生成；
- GPU 基础：线程层次、warp、SIMT、寄存器、共享内存、缓存、同步、合并访问；
- 正确性与性能：参考实现、误差、benchmark、warmup、吞吐/延迟、roofline。

## P1：进入 GPU 工具链后必须补齐

- CUDA C++：kernel 参数、grid/block/thread、memory spaces、cooperative groups、stream/event；
- CUDA 工具链：`nvcc`、`nvrtc`、PTX、`ptxas`、fatbin、模块加载和兼容性；
- PTX/目标代码：虚拟 ISA 与真实 ISA 的关系、寄存器、predication、内存语义、调试信息；
- Kernel 性能：occupancy、寄存器压力、shared-memory bank conflict、访存事务、指令吞吐、launch 开销；
- 高性能库：cuBLAS/cuDNN/NCCL 的调用边界、算法选择、workspace、layout 和 stream 语义；
- CUTLASS/CuTe：层次化 tiling、layout algebra、copy/MMA atom、epilogue、模板实例化和 profiler；
- Triton：程序实例、mask、布局、TTIR/TTGIR、LLVM/PTX lowering、autotune 和调试；
- 性能工具：Nsight Systems、Nsight Compute、CUDA events、硬件计数器和微基准设计。

## P2：成为编译器/后端专家需要掌握

- LLVM：IR、Alias Analysis、MemorySSA、New PM、SelectionDAG、GlobalISel、MIR、TableGen、MC；
- MLIR：Operation/Region/Block、Dialect、ODS、Traits/Interfaces、Pattern Rewrite、Conversion、Bufferization、GPU/NVVM/LLVM Dialect；
- TVM：Relax、TensorIR、PrimFunc、schedule、meta-schedule、runtime、BYOC 和目标后端；
- 编译器优化：循环变换、依赖分析、向量化、并行化、算子融合、代价模型、自动调优、缓存/内存复用；
- 自研芯片：ISA、ABI、寄存器文件、调用约定、地址空间、DMA/异步拷贝、编译器/driver/runtime/调试器协同；
- 生产工程：版本矩阵、二分、性能回归、编译缓存、可复现构建、CI、发布和故障诊断。

## P3：高级方向

- 多 GPU：NCCL、collective、通信拓扑、切分/sharding、通信计算重叠；
- 稀疏与压缩：结构化稀疏、块稀疏、稀疏布局和硬件支持；
- 动态与不规则工作负载：动态形状、控制流、稀疏张量、图捕获和运行时调度；
- 新型硬件：异构核、片上互联、统一虚拟内存、近存计算、FP8/FP4 等低精度格式；
- 自动优化：搜索空间、代价模型、强化学习/贝叶斯搜索、跨硬件迁移和调优缓存；
- 可观测性：编译器 remark、IR dump、kernel trace、性能数据库和自动回归分析。

## 现在最值得继续补的章节

GPU 工具链专题已经把第一批缺口落成章节；后续应继续把这些章节扩展为可运行实验和真实源码精读：

1. 为 CUDA/PTX 章节加入真实 `.cu` 和 PTX 实验；
2. 为 CUTLASS/Triton 章节加入可复现的 GEMM 对比；
3. 为 Runtime/Driver 章节加入最小 module loader 和 stream/event 样例；
4. 为 Nsight 章节加入真实报告截图、指标解释和性能回归脚本；
5. 为 NCCL 章节加入 2 GPU/多 GPU collective 和通信重叠实验；
6. 把端到端案例接到 TVM/MLIR 的真实 lowering pipeline。

## 每个新专题必须采用的章节模板

每个专题不要只写概念介绍，统一按下面的结构：

1. 它在整条工具链中的位置；
2. 核心对象和术语；
3. 一个最小可运行示例；
4. 源码仓库的目录与入口函数；
5. 一次真实的编译/运行/性能观测；
6. 常见错误和错误归因方法；
7. 与 TVM、LLVM、MLIR、CUDA 或硬件的边界；
8. 练习、验收标准和进一步阅读。

