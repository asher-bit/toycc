- [课程首页](README.md)
- [新人手册总入口](handbook/README.md)
- [岗位学习路径](handbook/paths.md)
- [知识地图](handbook/knowledge_map.md)
- [教材写作与维护规范](handbook/editorial_guide.md)

**第一阶段：编译器基本闭环**

- [第 0 课 · 编译器全景](lesson00.md)
- [第 1 课 · 计算图与 IR](lesson01.md)
- [第 2 课 · 参考执行器与正确性](lesson02.md)
- [第 3 课 · 算子融合](lesson03.md)
- [第 4 课 · 布局优化](lesson04.md)
- [第 5 课 · 常量折叠](lesson05.md)
- [第 6 课 · 内存规划](lesson06.md)
- [第 7 课 · 代码生成](lesson07.md)

**第二阶段：TVM 与调度**

- [第 8 课 · 如何阅读真实 TVM Pass](lesson08.md)
- [第 9 课 · 在真实 TVM 中跑通流程](lesson09.md)
- [第 10 课 · 从阅读到动手](lesson10.md)
- [第 11 课 · TIR 与调度](lesson11.md)
- [第 12 课 · 优化全景](lesson12.md)
- [第 13 课 · 自动调度](lesson13.md)
- [第 14 课 · IR 家族](lesson14.md)

**第三阶段：硬件、模型与性能**

- [第 15 课 · 硬件基础](lesson15.md)
- [第 16 课 · 工程开发流程](lesson16.md)
- [第 17 课 · 模型导入与下降](lesson17.md)
- [第 18 课 · 量化与数值精度](lesson18.md)
- [第 19 课 · 性能与算子](lesson19.md)
- [第 20 课 · 知识地图与自测](lesson20.md)

**第四阶段：GPU 编译器与工具链**

- [第 21 课 · GPU 芯片架构](lesson21.md)
- [第 22 课 · GPU 编译器](lesson22.md)
- [第 23 课 · Kernel 与性能分析](lesson23.md)
- [第 24 课 · 自研 GPU 工具链](lesson24.md)
- [第 25 课 · LLVM 深入](lesson25.md)
- [第 26 课 · MLIR 深入](lesson26.md)

**第五阶段：工具链系统软件**

- [第 27 课 · 模拟器](lesson27.md)
- [第 28 课 · 内存模型与并发](lesson28.md)
- [第 29 课 · 二进制与模块加载](lesson29.md)
- [第 30 课 · 驱动与命令提交](lesson30.md)

**第六阶段：高性能部实战**

- [第 31 课 · LLM 推理性能工程](lesson31.md)
- [第 32 课 · 分布式并行与通信](lesson32.md)
- [第 33 课 · 生产级量化](lesson33.md)
- [第 34 课 · Triton 与 CUTLASS](lesson34.md)
- [第 35 课 · 前沿专题速览](lesson35.md)

**专题课 · 按知识域深入**

专题课不是主教材的额外编号，而是沿主教材中的某个概念继续往下读。

**TVM**

- [TVM 源码精读目录](tvm/README.md)
- [`fuse_ops.cc` · Relax 算子融合](tvm/fuse_ops.md)
- [`fold_constant.cc` · Relax 常量折叠](tvm/fold_constant.md)
- [经典 Relax Pass 学习路线](tvm/pass_roadmap.md)

**LLVM**

- [LLVM 专题目录](llvm/README.md)
- [LLVM IR、SSA 与验证器](llvm/01_ir_ssa.md)
- [Analysis、Pass 与新 Pass Manager](llvm/02_analysis_passes.md)
- [写 Pass、接入 opt 与测试](llvm/03_write_pass_and_tests.md)
- [后端、ABI、TableGen 与 MC](llvm/04_backend_abi_mc.md)

**MLIR**

- [MLIR 专题目录](mlir/README.md)
- [Operation、Region、Block 与 Value](mlir/01_ir_core.md)
- [Dialect、ODS 与 TableGen](mlir/02_dialect_ods.md)
- [Pattern Rewrite 与 Dialect Conversion](mlir/03_rewrite_and_conversion.md)
- [Bufferization、Lowering 与测试](mlir/04_bufferize_lowering_tests.md)

**模拟器**

- [模拟器专题目录](sim/README.md)
- [性能模型的设计](sim/01_performance_models.md)
- [周期模型的设计](sim/02_cycle_models.md)

**GPU 工具链**

- [GPU 工具链专题目录](gpu/README.md)
- [CUDA 编程模型](gpu/01_cuda_programming_model.md)
- [CUDA 工具链与 PTX](gpu/02_cuda_toolchain_ptx.md)
- [GPU ISA、寄存器与 ABI](gpu/03_gpu_isa_registers_abi.md)
- [CUTLASS / CuTe](gpu/04_cutlass_cute.md)
- [Triton 编译器](gpu/05_triton_compiler.md)
- [Runtime / Driver](gpu/06_runtime_driver.md)
- [性能分析与 Nsight](gpu/07_profiling_performance.md)
- [NCCL 与多 GPU](gpu/08_multi_gpu_nccl.md)
- [端到端案例](gpu/09_end_to_end.md)
- [GEMM 优化案例](gpu/10_gemm_optimization_case.md)
- [浮点误差与数值验证](gpu/11_float_error.md)
- [版本兼容与部署排查](gpu/12_compat_matrix.md)
- [真机实验目录](gpu/experiments/README.md)

**附录 C · CUDA 编程指南**

- [总览与导读](appendix_cuda/README.md)
- [1.1 引言](appendix_cuda/01_intro.md)
- [1.2 编程模型](appendix_cuda/02_programming_model.md)
- [1.3 CUDA 平台](appendix_cuda/03_platform.md)
- [2.1 CUDA C++ 入门](appendix_cuda/04_intro_cuda_cpp.md)
- [2.2 CUDA Python 入门](appendix_cuda/05_intro_cuda_python.md)
- [2.3 编写 SIMT Kernel](appendix_cuda/06_writing_simt_kernels.md)
- [2.4 编写 Tile Kernel](appendix_cuda/07_writing_tile_kernels.md)
- [2.5 异步执行](appendix_cuda/08_asynchronous_execution.md)
- [2.6 统一内存与系统内存](appendix_cuda/09_unified_system_memory.md)
- [2.7 NVCC 编译器](appendix_cuda/10_nvcc.md)

**参考资料**

- [C++ 阅读手册](appendix_cpp.md)
- [开发环境搭建](appendix_env.md)
- [词汇表](glossary.md)
