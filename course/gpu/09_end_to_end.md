# GPU 第 9 章：端到端案例——一个算子从模型到 GPU 的完整排查

## 1. 本章目标

本章把第 1~8 章串成一条工作流：面对"某个算子在 GPU 上慢 / 错 / 加载失败"，能从模型输入一路追到 kernel、code object、runtime 和硬件指标，而不是一上来只改 tile。完成本章后，面对一个新的算子问题，你能按层写出排查顺序、在每层用对工具、并输出一份带证据的复盘。

案例设定：带 bias + relu 激活的 fp16 GEMM，M=N=K=4096，目标 A100。现象：端到端只有预期性能的 60%。

## 2. 完整链路：每一层看什么对象

```text
层 0  模型/图:        matmul + bias + relu(融合机会)
层 1  高层 IR/kernel: Triton / CUTLASS / CUDA 的实现选择
层 2  PTX/SASS:      寄存器、spill、指令调度
层 3  module/runtime: 加载、launch、异步错误
层 4  硬件执行:      合并访问、bank、occupancy、Tensor Core
层 5  性能观测:      Nsight Systems / Compute
层 6  (多卡)NCCL:    通信量与重叠
```

每一层都有自己"该看的东西"：层 0 看图的融合机会，层 2 看 `ptxas -v` 的资源账，层 3 看错误码与 launch 开销，层 4 看事务数与 stall，层 5 看时间线与 Speed of Light，层 6 看通信时间。**排查的原则是"先分层，再动手"**：60% 性能问题不在 kernel 本身，而在层 0（没融合）、层 3（launch 太密）或层 5 才看得见的同步等待上。

## 3. 案例走查：假设 → 证据 → 行动 → 验证

以"端到端只有 60%"为例，把排查走成四步循环。A100 的 fp16 Tensor Core 峰值约 312 TFLOPS，因此这个 GEMM 的 roofline 上限（强度约 407 FLOP/B，第 4 章已推）就是算力墙，60% 意味着有约 40% 的算力没兑现。

| 轮 | 假设 | 证据（工具 + 数字） | 行动 | 验证 |
|---|---|---|---|---|
| 1 | 三个算子没融合，中间结果来回 global | 层 0：图上 matmul/bias/relu 是三个节点 | 把 bias+relu 融进 epilogue（第 4 章账：每 tile 省 128 KB 往返） | 端到端 60% → 72% |
| 2 | 寄存器超限，出现 spill | 层 2：`ptxas -v` 显示 "Used 168 registers, 512 bytes spill" | 缩小 warp tile（8×8 → 4×4 累加器），spill 归零 | 72% → 78% |
| 3 | 共享内存 bank conflict | 层 4：Nsight Compute 的 shared 部分波形异常 | 改 shared 布局 / padding（第 1 章数格子法） | 78% → 82% |
| 4 | launch 太密或隐式同步 | 层 5：Nsight Systems 时间线上一串短 kernel 与空白段 | 合并 launch / 修 NULL stream（第 6 章） | 82% → 87% |

每轮都遵守三条纪律：**一次只改一个变量**（改完立刻验证，否则归因失效）；**证据优先于直觉**（每层都有明确的工具产出，不是"感觉"）；**改动前后同条件对比**（第 7 章六步协议）。四轮下来，"慢 40%"被分解成四个可定位的独立问题，而不是一个"tile 没调好"。

## 4. 正确性检查顺序（错的时候）

1. 对高精度 reference 算绝对/相对误差（toycc 的 max|Δ| 纪律；fp16 对 fp32 reference 通常允许 1e-2 量级的相对误差，具体阈值按算子定）；
2. 覆盖非整 tile shape、零维/空输入、非连续 layout、不同 dtype——**错误最爱藏在边界**（第 1 章 mask 的账）；
3. 检查边界 mask、stride、参数布局与输出写入范围；
4. `compute-sanitizer` 或 debug 编译，排除越界、竞争、未同步；
5. 多 stream/异步 pipeline 下重复测试，排除时序问题（第 6 章 sticky error）。

## 5. 性能检查顺序（慢的时候）

```text
① 先确认不是重复编译 / JIT / 分配(第 2、6 章)
② 再分离 H2D / kernel / D2H / 通信(第 7 章六步协议)
③ 再看 kernel 的 memory / compute / launch 三类开销
④ 再看寄存器、shared、occupancy、warp stall(第 3、7 章)
⑤ 最后才改 tile、layout、pipeline、dtype 或融合
```

两条分流规则：**kernel 很快但模型很慢** → 优先查 layout transform、host/device copy、launch 数量、stream 依赖与框架调度（问题在层 0/3/5）；**单 kernel 很慢** → 查访存、Tensor Core、寄存器与指令路径（问题在层 2/4）。第 3 节的四轮走查就是这条顺序的展开。

## 6. 一份问题复盘模板

```text
问题: 哪个输入、哪个 GPU、哪个版本、慢/错多少？
基线: commit、driver、toolkit、编译参数、shape、dtype、layout。
定位: IR / kernel / PTX / module / runtime / hardware / profiler 哪一层？
证据: IR dump、PTX/SASS、错误码、timeline、kernel metrics。
修复: 改了什么，为什么合法，为什么预期能改善？
验证: correctness、microbenchmark、operator、model、回归矩阵。
风险: 架构兼容、数值误差、资源使用、其他 shape、其他 stream。
```

"定位"和"证据"两栏强制回答"问题在第 2 节的哪一层、用什么工具看见的"——没有这两栏的复盘只是叙事，不是排查记录。

## 7. 新员工验收项目

选 vector add、reduction 或 fused GEMM 之一，完成以下全部才算通过：

- CUDA 或 Triton 实现；
- reference implementation 与随机/边界测试（含非整 tile shape）；
- PTX 或生成代码检查（`ptxas -v` 资源账 + SASS 抽查）；
- Runtime/stream/event 的异步执行（第 6 章对象链）；
- Nsight Systems 时间线（标出每段属于哪一层）；
- Nsight Compute kernel 报告（Speed of Light + stall 归因）；
- 一份含"假设 → 证据 → 行动 → 验证"至少两轮的性能复盘；
- 多 GPU 版本再增加：NCCL collective 的通信量手算 + 通信计算重叠分析。

## 8. 与整个仓库的关系

- toycc：IR、Pass、codegen 与 reference execution 的教学版——第 2 节层 0 的对象在它身上可以动手改；
- TVM：图优化、TensorIR、调度与 runtime 接入的真实框架——层 0/1 的工业实现；
- MLIR：多层 IR、Dialect、conversion 与 GPU lowering——层 1→2 的下降链载体；
- LLVM：IR、后端、寄存器分配、汇编与目标文件——层 2 的通用基础设施；
- 主教材第 27~30 课：模拟器、并发、二进制与驱动命令提交——层 3/4 之下的硬件真相；
- 本专题：把 CUDA/GPU 生态与上述基础设施连成一条可执行的工作流。

## 9. 检查点

完成以下四项才算通过本章：

1. 把第 2 节的七层链路画出来，在每层标出"看什么对象 + 用什么工具"；
2. 给"端到端慢但单 kernel 快"列三个第一优先的排查项（按层标注）；
3. 按第 3 节格式，为"fp16 GEMM 数值误差偏大"写出第一轮的假设、证据、行动、验证；
4. 用第 6 节模板写一份你上一个真实问题（或验收项目）的复盘草稿。

## 10. 下一步与扩展阅读

本专题到这里闭环：CUDA 编程模型 → 工具链/PTX → ISA/ABI → CUTLASS → Triton → Runtime → Profiler → NCCL → 端到端。两个继续深入的方向：沿层 2 向下进 LLVM/MLIR 专题（编译器基础设施）；沿层 3 向下进主教材第 27~30 课（自研工具链四柱）。

- 官方：[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)、[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)；
- 本专题入口：[GPU 工具链专题目录](README.md)。

**导航**：⬅ [上一章](08_multi_gpu_nccl.md)（NCCL 与多 GPU）　｜　本专题完，返回 [专题目录](README.md) ➡
