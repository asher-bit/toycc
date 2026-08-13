# 机器学习编译器与 GPU 工具链：主教材

这部分是仓库的连续主线。它不要求你一开始掌握 TVM、LLVM、CUDA 或硬件，而是从一个能运行的 toycc 出发，逐步回答：

```text
模型如何表示？
  → 编译器如何保证变换正确？
  → 如何把计算变成循环和 kernel？
  → 硬件为什么会快或慢？
  → 代码如何经过后端、加载器和驱动，最终在 GPU 上执行？
```

完成主教材后，再根据岗位进入 TVM、LLVM、MLIR 或 GPU 工具链专题。专题是纵向深入，不是另一套互相竞争的课程。

## 怎么用这门课

```
项目根目录:
├── toycc/               ← 我们亲手写的迷你 AI 编译器(已可运行)
│   ├── ir/              ← 计算图 IR
│   ├── passes/          ← 优化 pass(融合/布局/常量折叠/内存)
│   ├── codegen/         ← 代码生成(C 和 Python 双后端)
│   └── runtime/         ← numpy 参考执行器(正确性的裁判)
├── course/              ← 本课程
│   ├── README.md        ← 你正在看的首页
│   ├── runner.py        ← 每课的实验运行器
│   ├── lesson00~35.md   ← 三十六课
│   ├── handbook/         ← 新员工总手册、岗位路径、知识地图
│   ├── gpu/              ← CUDA/PTX/CUTLASS/Triton/Runtime/Profiler/NCCL 专题
│   ├── tvm/              ← 真实 TVM 源码精读专题
│   ├── llvm/             ← LLVM 源码与实践分章
│   ├── mlir/             ← MLIR 源码与实践分章
│   └── glossary.md      ← 词汇表
└── out.c / out.py       ← 代码生成实验的输出
```

**每节课的固定节奏：**

1. 先读本课开头，明确本课要解决的问题和完成标准；
2. 跟着最小例子推一遍，先形成直觉，再记术语；
3. 运行实验 `python -m course.runner XX`，对照输入、输出和中间结果；
4. 回到 `toycc/` 或真实项目源码，确认直觉对应哪些对象和函数；
5. 完成章末检查点；只有需要更深源码背景时，再进入专题课。

> 有问题随时停下：先自己推理，再看答案——这是编译器开发者的日常训练。

## 主教材的结构

> 主教材按学习依赖组织，而不是按项目名称组织：编译器基本闭环 → TVM 与调度 → 硬件、模型与性能 → GPU 编译器 → 工具链系统软件。
> 进入仓库时，先看[新人手册总入口](handbook/README.md)了解全貌，再回到下面的 36 课按顺序学习。

## 先看这里：新人手册总入口

- [知识体系总览](handbook/README.md)
- [按岗位选择学习路径](handbook/paths.md)
- [知识地图与补课优先级](handbook/knowledge_map.md)

主教材先建立共同语言，再根据岗位深入编译器、Kernel、后端、Runtime 或硬件。TVM、LLVM、MLIR、CUDA、CUTLASS、Triton 都是工具链中的不同层，不把其中任何一个当成整套体系的代名词。

### ① 入门地基：看懂"一个 AI 编译器长什么样"

| 课 | 主题 | 学到什么 | 动手 |
|---|---|---|---|
| 00 | 编译器全景 | 编译流水线的完整地图 | 先读本课并跑通 demo |
| 01 | 计算图与 IR | 读 `graph.py` 每一行；拓扑序/消费者 | `runner 1` |
| 02 | 参考执行器 | 正确性怎么保证；`max|Δ|` | `runner 2` |

### ② 核心 pass：编译器的主体

| 课 | 主题 | 学到什么 | 动手 |
|---|---|---|---|
| 03 | **算子融合** | 融合动机 + 逐行读 `fusion.py` + 翻车现场 | `runner 3` |
| 04 | 布局优化 | NCHW/NHWC 排布、缓存/SIMD、布局传播 | `runner 4` |
| 05 | 常量折叠 | 编译期 vs 运行时；折叠也有成本 | `runner 5` |
| 06 | 内存规划 | 生命周期分析、缓冲区复用、省 47% | `runner 6` |

### ③ 后端与真实 TVM：从"玩具"跨越到"真框架"

| 课 | 主题 | 学到什么 | 动手 |
|---|---|---|---|
| 07 | 代码生成 | 下标展平手算、多后端、读生成的 C | `runner 7` |
| 08 | 真实 TVM(上) | 三遍读法；精读 `fuse_ops.cc`/`fold_constant.cc` | `runner 8` |
| 09 | 真实 TVM(下) | 剩余源码 + 装 tvm + 跑真实管线 | `runner 9` |

#### 真实 TVM 源码精读专题

- [`fuse_ops.cc`：Relax 算子融合源码详解](tvm/fuse_ops.md)
- [`fold_constant.cc`：Relax 常量折叠源码详解](tvm/fold_constant.md)
- [经典 Relax Pass 学习路线](tvm/pass_roadmap.md)

### ④ 上手与进阶：能动手 + 参与社区讨论

| 课 | 主题 | 学到什么 | 动手 |
|---|---|---|---|
| 10 | **从看懂到上手** | 给 toycc 加功能 → 写 TVM pass → 首次贡献 | 课后任务 |
| 11 | **TIR 与调度** | 调度原语 + 亲手调度 matmul（toycc 模拟器） | `runner 11` |
| 12 | **优化全景** | DCE/CSE/化简/内联 + op_pattern + 支配分析 | `runner 12` |
| 13 | **自动调度** | meta_schedule：搜索空间/成本模型/tune 流程 | `runner 13` |
| 14 | **IR 家族** | LLVM IR / MLIR / TIR / PTX 真实示例精读 | `runner 14` |

### ⑤ 硬件与系统：硬件/数据/工程三维度

| 课 | 主题 | 学到什么 | 动手 |
|---|---|---|---|
| 15 | **硬件必修** | 缓存/寄存器/内存层次 + 缓存模拟器实验 | `runner 15` |
| 16 | **工程流程** | build/测试/调试/CI + 第一个任务演练 | `runner 16` |
| 17 | **模型导入** | ONNX、legalize 下降、动态形状/控制流、运行时 VM | `runner 17` |
| 18 | **量化与精度** | int8/fp16/bf16、scale/zero-point、PTQ/QAT + 量化实验 | `runner 18` |
| 19 | **性能与算子** | roofline、benchmark 方法论、im2col/Winograd/GEMM | `runner 19` |

### ⑥ 知识地图（检查点）

| 课 | 主题 | 学到什么 | 动手 |
|---|---|---|---|
| 20 | **知识地图** | 完整知识全景 + 进阶路径 + 六阶段计划 | 自测清单 |

### ⑦ GPU 芯片专项：自研 GPU 岗位专属（你的核心目标）

| 课 | 主题 | 学到什么 | 动手 |
|---|---|---|---|
| 21 | **GPU 芯片架构** | SM/warp/共享内存/寄存器/张量核/合并访问/分支发散/bank conflict | `runner 21` |
| 22 | **GPU 编译器** | SIMT 编译、PTX→SASS、指令调度、占用率、发散处理 | `runner 22` |
| 23 | **Kernel 开发** | 写 kernel、benchmark、profiler、roofline 实战 | `runner 23` |
| 24 | **工具链全景** | compiler/driver/runtime/profiler + 加新后端五步 | `runner 24` |

### ⑧ LLVM / MLIR 深入：现代编译器基础设施

| 课 | 主题 | 学到什么 | 动手 |
|---|---|---|---|
| 25 | **LLVM 深入** | 基本块/phi、pass 体系、后端流水线、MC 层写汇编器 | `runner 25` |
| 26 | **MLIR 深入** | 方言/ODS/TableGen、pattern rewrite、渐进式下降链 | `runner 26` |

### LLVM / MLIR 源码与实践分章

第 25、26 课负责建立 LLVM/MLIR 的位置感；下面的专题再把它们拆成源码与实践章节：

- [LLVM 深入专题](llvm/README.md)：IR/SSA → Analysis/Pass → 写 Pass 与测试 → 后端/ABI/MC；
- [MLIR 深入专题](mlir/README.md)：IR 核心对象 → Dialect/ODS → Rewrite/Conversion → Bufferization/Lowering/测试。

专题章节包含学习目标、源码阅读地图、最小代码片段、命令实验和练习题。主教材完成到第 14 课后，可以先读专题目录，再决定是否进入深读。

### ⑨ 工具链四柱：流片前的四大地基（岗位专属）

> 第 24 课画了工具链的"盒子"，这四课填"盒子里的东西"——
> 自研芯片没有真硬件时，靠这四根柱子撑起全部开发。

| 课 | 主题 | 学到什么 | 动手 |
|---|---|---|---|
| 27 | **模拟器** | ISS/周期模型/RTL 三层、差分测试、覆盖率门禁 | `runner 27` |
| 28 | **内存模型与并发** | 原子/fence/弱内存序、同步四层级、bar 死锁 | `runner 28` |
| 29 | **二进制与加载** | ELF/cubin/fatbin、重定位手算、运行时 JIT、稳定中间层 | `runner 29` |
| 30 | **驱动与命令提交** | 命令缓冲/门铃、context/stream/event、GPU MMU、launch 开销手算 | `runner 30` |

### GPU 工具链专题

第 21~24、27~30 课建立 GPU 芯片和工具链的主线；[GPU 工具链专题](gpu/README.md)再补充 CUDA、PTX、寄存器/ABI、CUTLASS、Triton、Runtime/Driver、Nsight、NCCL 和端到端排查。

### ⑩ 高性能部实战：会议室语言（岗位专属）

> 前五组课让你"会造编译器"；这五课让你"会参与讨论"——
> 高性能部日常 80% 的话题（模型在卡上跑多快/几卡/什么精度）都在这五课的账里。

| 课 | 主题 | 学到什么 | 动手 |
|---|---|---|---|
| 31 | **LLM 推理性能工程** | prefill/decode 两阶段账、KV cache 容量账、PagedAttention、FlashAttention 内存分析 | `runner 31` |
| 32 | **分布式并行与通信** | DP/TP/PP/EP 切法、ring allreduce 带宽模型、NVLink/IB 层级、通信重叠 | `runner 32` |
| 33 | **生产级量化** | W4A16 位宽比账、per-group/GPTQ/AWQ/SmoothQuant、dequant 融合、FP8 | `runner 33` |
| 34 | **Triton 与 CUTLASS** | 块级抽象 vs 线程级、三级 tile 模板层次、三条 kernel 路线的分工 | `runner 34` |
| 35 | **前沿专题速览** | 结构化稀疏、MoE、投机解码、MLPerf 四问、框架接入三路径 | `runner 35` |

### 附录（随时查）

| 附录 | 内容 | 什么时候用 |
|---|---|---|
| 附录A | C++ 阅读手册 | 读 08 课源码之前必看 |
| 附录B | Windows → WSL2 → 装 TVM 环境 | 现在就做 |
| 附录C | CUDA 编程指南（分章） | 想知道官方指南里都有什么、该读哪一部分 |

## 前置要求

- Python 3.10+（本机 3.14）
- 只需要 `numpy`（已装）
- 会用命令行，看得懂基础 Python
- 完全不需要机器学习基础

## 一分钟先跑起来

```bash
python -m course.runner 1      # 第1课实验（第0课没有单独实验）
python -m toycc.examples.demo  # 完整流水线一键演示
```

demo 会打印：10 个算子 → 融合/布局/常量折叠 → 6 个算子，
内存省 47%，生成的代码与参考执行结果完全一致（max|Δ|=0）。

## 默认入职路径

不要把“学完所有内容”当成入职前置条件。推荐先完成公共地基，再按岗位选择一条主路径：

```text
公共地基（IR / Pass / C++ / Linux / GPU 基础）
  ├─ 编译器与 IR：TVM → MLIR → LLVM
  ├─ GPU Kernel 与性能：GPU → CUDA → Triton/CUTLASS → Profiler
  ├─ 后端与自研芯片：ISA/ABI → LLVM Backend → 汇编器/Runtime
  └─ Runtime 与系统软件：Driver → Module → Memory/Stream/Event → 多 GPU
```

具体的 30/60/90 天目标见[学习路径与工作任务](handbook/paths.md)；术语、源码路径和专题选择见[知识地图](handbook/knowledge_map.md)。

## 附录（按需查）

- `appendix_cpp.md` — 读编译器源码的最小 C++ 手册（从基础 C 起步）
- `appendix_env.md` — Windows → WSL2 → 装 TVM 保姆级指南
- `appendix_cuda/` — NVIDIA CUDA Programming Guide 中文译文（按 1.1/1.2/1.3 分章，README 含导读与索引）

## 心态建议

- **看不懂很正常**。本课反复用同一张地图讲不同房间，第一遍混个脸熟，第二遍就通了。
- **跟着代码走**，不要跳。每课都是"读代码 → 跑实验 → 想为什么"。
- **自己改一遍**。学完前 7 课，去做第 10 课的任务 A——亲手加一个算子，
  比你再看十遍都有用。
- 卡住就来问我。我可以陪你精读任意一课、帮你改 bug、带你做课后任务。
