# GPU 第 9 章：端到端案例——一个算子从模型到 GPU

## 1. 本章目标

把前面各章串成一条工作流：面对“某个算子在 GPU 上慢/错/加载失败”，能够从模型输入一直追到 kernel、code object、runtime 和硬件指标，而不是一上来只改 tile。

## 2. 案例：带 bias 和激活的 GEMM

```text
模型图：matmul + bias + activation
   ↓
前端/高层 IR：形状、dtype、layout、融合机会
   ↓
调度或 kernel 生成：TIR / Triton / CUTLASS / CUDA
   ↓
LLVM IR / PTX / cubin
   ↓
module load + stream launch
   ↓
GPU：global → shared → register → MMA → epilogue
   ↓
correctness + Nsight + end-to-end benchmark
```

## 3. 正确性检查顺序

1. 对比高精度 reference，检查绝对/相对误差；
2. 覆盖非整 tile shape、零维/空输入、非连续 layout 和不同 dtype；
3. 检查边界 mask、stride、参数布局和输出写入范围；
4. 用 sanitizer 或 debug 编译排除越界、竞争和未同步；
5. 在多 stream/异步 pipeline 下重复测试，排除时序问题。

## 4. 性能检查顺序

```text
先确认不是重复编译/JIT/分配
 → 再分离 H2D/kernel/D2H/通信
 → 再看 kernel 的 memory/compute/launch
 → 再看寄存器、shared、occupancy、warp stall
 → 最后改 tile、layout、pipeline、dtype 或融合
```

如果 kernel 很快但模型很慢，优先查 layout transform、host/device copy、launch 数量、stream 依赖和框架调度；如果单 kernel 很慢，再查访存、Tensor Core、寄存器和指令路径。

## 5. 一份问题复盘模板

```text
问题：哪个输入、哪个 GPU、哪个版本、慢/错多少？
基线：commit、driver、toolkit、编译参数、shape、dtype、layout。
定位：IR / kernel / PTX / module / runtime / hardware / profiler 哪一层？
证据：IR dump、PTX/SASS、错误码、timeline、kernel metrics。
修复：改了什么，为什么合法，为什么预期能改善？
验证：correctness、microbenchmark、operator、model、回归矩阵。
风险：架构兼容、数值误差、资源使用、其他 shape、其他 stream。
```

## 6. 新员工验收项目

选择 vector add、reduction 或 fused GEMM 之一，完成：

- CUDA 或 Triton 实现；
- reference implementation 和随机/边界测试；
- PTX 或生成代码检查；
- Runtime/stream/event 的异步执行；
- Nsight Systems 时间线；
- Nsight Compute kernel 报告；
- 一份包含瓶颈假设和证据的性能复盘；
- 如果是多 GPU 版本，再增加 NCCL collective 和通信重叠分析。

## 7. 与整个仓库的关系

- toycc：帮助理解 IR、Pass、codegen 和 reference execution；
- TVM：帮助理解图优化、TensorIR、调度和 runtime 接入；
- MLIR：帮助理解多层 IR、Dialect、conversion 和 GPU lowering；
- LLVM：帮助理解 IR、后端、寄存器、汇编和目标文件；
- 第 27~30 课：帮助理解模拟器、并发、二进制和驱动命令提交；
- 本专题：把 CUDA/GPU 生态和上述基础设施连成一个可执行工作流。

