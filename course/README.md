# 机器学习编译器与 GPU 工具链新人手册

> 这个仓库的最终目标不是"看懂概念"，而是——
> **把一个零基础的新人，完整带入编译器开发流程**：
> 所有基本概念都懂、常见代码看得明白、能上手写 pass、能参与开发讨论、
> **能负责一颗自研 GPU 芯片的工具链/编译器开发**。

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
│   ├── lesson00~30.md   ← 三十一课
│   ├── handbook/         ← 新员工总手册、岗位路径、知识地图
│   ├── gpu/              ← CUDA/PTX/CUTLASS/Triton/Runtime/Profiler/NCCL 专题
│   ├── tvm/              ← 真实 TVM 源码精读专题
│   ├── llvm/             ← LLVM 源码与实践分章
│   ├── mlir/             ← MLIR 源码与实践分章
│   └── glossary.md      ← 词汇表
└── out.c / out.py       ← 代码生成实验的输出
```

**每节课的固定节奏：**

1. 读 `lessonXX.md`（代码驱动，逐行拆解，带手算例子）
2. 运行实验 `python -m course.runner XX`，边跑边看输出
3. 对照它讲到的 `toycc` 源码，逐行读一遍
4. 做"思考题"，再对"答案"
5. 最后几节打开真实 TVM 源码，完成"从玩具到真框架"的跨越

> 有问题随时停下：先自己推理，再看答案——这是编译器开发者的日常训练。

## 学习路径

> 现在的主导航按知识域组织：公共地基 → 模型/IR/优化 → TVM → GPU 硬件 → CUDA/Kernel/库 → LLVM/MLIR/后端 → Runtime/Driver → 性能与工程。
> 原有 31 课仍然保留，作为按周推进的兼容主线；新人优先从[新人手册总入口](handbook/README.md)开始。

## 先看这里：新人手册总入口

- [知识体系总览](handbook/README.md)
- [按岗位选择学习路径](handbook/paths.md)
- [知识地图与补课优先级](handbook/knowledge_map.md)

这里的“主线”不是把所有人培养成同一种工程师，而是先建立共同语言，再根据岗位深入编译器、Kernel、后端、Runtime 或硬件。TVM、LLVM、MLIR、CUDA、CUTLASS、Triton 都是工具链中的不同层，不再把其中任何一个当成整套体系的代名词。

下面是保留的“31 课编号主线”，适合按周推进；如果你是新员工，优先从上面的[新人手册总入口](handbook/README.md)进入，再根据岗位选择路径。

### ① 入门地基：看懂"一个 AI 编译器长什么样"

| 课 | 主题 | 学到什么 | 动手 |
|---|---|---|---|
| 00 | 总览：编译器到底在干嘛 | 编译流水线的完整地图 | 跑通 `runner 1` |
| 01 | 计算图与 IR | 读 `graph.py` 每一行；拓扑序/消费者 | `runner 1` |
| 02 | 参考执行器 | 正确性怎么保证；`max|Δ|` | `runner 2` |

### ② 核心 pass：编译器的主体

| 课 | 主题 | 学到什么 | 动手 |
|---|---|---|---|
| 03 | **算子融合** | 融合动机 + 逐行读 `fusion.py` + 翻车现场 | `runner 3` |
| 04 | 布局优化 | NCHW/NHWC 排布、缓存/SIMD、布局传播 | `runner 4` |
| 05 | 常量折叠 | 编译期 vs 运行时；折叠也有成本 | `runner 5` |
| 06 | 内存规划 | 生命周期分析、缓冲区复用、省 70% | `runner 6` |

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

第 25、26 课是总览；下面的专题把内容继续拆成可逐章完成的源码与实践课程：

- [LLVM 深入专题](llvm/README.md)：IR/SSA → Analysis/Pass → 写 Pass 与测试 → 后端/ABI/MC；
- [MLIR 深入专题](mlir/README.md)：IR 核心对象 → Dialect/ODS → Rewrite/Conversion → Bufferization/Lowering/测试。

每章都包含学习目标、源码阅读地图、最小代码片段、命令实验和练习题。

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

第 21~24、27~30 课建立 GPU 芯片和工具链全景；[GPU 工具链专题](gpu/README.md)继续补充 CUDA、PTX、寄存器/ABI、CUTLASS、Triton、Runtime/Driver、Nsight、NCCL 和端到端排查。

### 附录（随时查）

| 附录 | 内容 | 什么时候用 |
|---|---|---|
| 附录A | C++ 阅读手册 | 读 08 课源码之前必看 |
| 附录B | Windows → WSL2 → 装 TVM 环境 | 现在就做 |

## 前置要求

- Python 3.10+（本机 3.14）
- 只需要 `numpy`（已装）
- 会用命令行，看得懂基础 Python
- 完全不需要机器学习基础

## 一分钟先跑起来

```bash
python -m course.runner 1      # 第1课实验
python -m toycc.examples.demo  # 完整流水线一键演示
```

demo 会打印：10 个算子 → 融合/布局/常量折叠 → 6 个算子，
内存省 70%，生成的代码与参考执行结果完全一致（max|Δ|=0）。

## 默认入职路径

不要把“学完所有内容”当成入职前置条件。推荐先完成公共地基，再按岗位选择一条主路径：

```text
公共地基（IR / Pass / C++ / Linux / GPU 基础）
  ├─ 编译器与 IR：TVM → MLIR → LLVM
  ├─ GPU Kernel 与性能：GPU → CUDA → Triton/CUTLASS → Profiler
  ├─ 后端与自研芯片：ISA/ABI → LLVM Backend → 汇编器/Runtime
  └─ Runtime 与系统软件：Driver → Module → Memory/Stream/Event → 多 GPU
```

具体的 30/60/90 天目标见[学习路径与工作任务](handbook/paths.md)；缺失内容和补课优先级见[知识地图](handbook/knowledge_map.md)。

## 附录（按需查）

- `appendix_cpp.md` — 读编译器源码的最小 C++ 手册（从基础 C 起步）
- `appendix_env.md` — Windows → WSL2 → 装 TVM 保姆级指南

## 心态建议

- **看不懂很正常**。本课反复用同一张地图讲不同房间，第一遍混个脸熟，第二遍就通了。
- **跟着代码走**，不要跳。每课都是"读代码 → 跑实验 → 想为什么"。
- **自己改一遍**。学完前 7 课，去做第 10 课的任务 A——亲手加一个算子，
  比你再看十遍都有用。
- 卡住就来问我。我可以陪你精读任意一课、帮你改 bug、带你做课后任务。
