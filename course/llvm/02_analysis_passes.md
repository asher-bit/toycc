# LLVM 第 2 章：Analysis、Pass 与新 Pass Manager——分析如何被缓存、失效与复用

## 1. 本章目标

- 能区分 analysis pass（算事实）与 transform pass（改 IR），并说出缓存存在的意义；
- 能解释 `PreservedAnalyses` 的三个档位（`all()` / `none()` / `preserve<X>()`）各自的语义与代价；
- 能说出 DominatorTree / LoopInfo / ScalarEvolution / Alias Analysis / MemorySSA 各自回答什么问题；
- 能按"四问清单"判断一个 transform pass 改完 IR 后能保留哪些分析；
- 能沿一条 `opt -passes='...'` pipeline 追出每个 pass 需要哪些分析。

前置：第 1 章的对象模型（Module/Function/BasicBlock/Instruction）与支配关系。工具：LLVM ≥ 15。

## 2. 工作中的问题长什么样

写 Pass 的人最典型的三个翻车现场：

```text
"我的 pass 单独跑没问题，插进 O2 pipeline 就随机出错？"
"返回 PreservedAnalyses::all() 到底有什么风险？"
"别人说我的 pass 太慢，为什么？"
```

三个问题的答案都在**分析缓存契约**上：分析结果被 AnalysisManager 缓存复用，正确性的前提是每个 transform pass 诚实声明"我改了什么、哪些分析还成立"。谎报 = 后续 pass 拿到过期事实 = 错误优化；过度保守 = 分析反复重算 = 编译变慢。本章建立这个契约的完整模型。

## 3. Pass 的两种：算事实与改 IR

**analysis pass（分析）**只计算事实、不修改 IR，结果交给 AnalysisManager 缓存；**transform pass（变换）**读取 IR 与分析结果，可能修改 IR，并在结束时**声明哪些分析仍然有效**。

```text
Function F
  ├─ DominatorTreeAnalysis → 支配关系(算一次, 缓存)
  ├─ LoopAnalysis         → 循环结构
  ├─ MemorySSAAnalysis    → 内存访问依赖
  └─ TransformPass        → 修改 F, 并返回 PreservedAnalyses 声明保留了什么
```

缓存的价值账：一条 O2 pipeline 有几十个 pass，其中十几个都要用支配树。支配树对每个函数算一次是 O(N) 量级——**不缓存，每个 pass 各自重算，编译时间被同一个分析反复吃掉；缓存，一次计算大家复用**。缓存唯一的正确性前提，就是第 5 节的失效契约。

## 4. 新 Pass Manager 的核心接口：一个最小 pass 逐行看

下面是一个**只读**的 transform pass（其实什么都没变换），统计函数里的 add 指令数：

```cpp
struct CountAddPass : llvm::PassInfoMixin<CountAddPass> {
  llvm::PreservedAnalyses run(
      llvm::Function &F,
      llvm::FunctionAnalysisManager &AM) {
    unsigned Count = 0;
    for (llvm::BasicBlock &BB : F)
      for (llvm::Instruction &I : BB)
        if (llvm::isa<llvm::BinaryOperator>(&I) &&
            I.getOpcode() == llvm::Instruction::Add)
          ++Count;

    llvm::errs() << F.getName() << ": " << Count << " adds\n";
    return llvm::PreservedAnalyses::all();
  }
};
```

逐行对应对象：

- `PassInfoMixin<CountAddPass>`：把本结构注册成一个可用名字查找的 pass（CRTP 混入）；
- `run(Function &F, FunctionAnalysisManager &AM)`：pass 的入口。函数级 pass 收 `Function&`，模块级收 `Module&`，循环级收 `Loop&`——**pass 的粒度由入口参数决定**；
- `AM`：本函数所属的分析管理器。pass 要用哪个分析，就从它那里取（`AM.getResult<SomeAnalysis>(F)`），取到的结果可能是缓存命中，也可能是现场计算；
- `return PreservedAnalyses::all()`：声明"所有分析都还有效"。本例只读不写，所以这个声明是**真的**。

返回值的三档语义与代价：

| 返回值 | 语义 | 代价 |
|---|---|---|
| `all()` | 什么都没动，全部分析有效 | 零失效成本；但**只要改了一行 IR 就是撒谎** |
| `none()` | 全部失效 | 最安全；后续每个分析都重算，编译时间最贵 |
| `preserveSet<CFGAnalyses>()` 等 | 精确声明保留哪些 | 写起来要动脑；正确性和性能的最佳平衡 |

判断规则一句话：**改了什么，就只保留与它无关的**。删指令可能保留支配树（支配关系没变），改 CFG 边就必须让支配树、循环信息失效。

## 5. 五个常用分析各回答什么问题

### 5.1 DominatorTree（支配树）

回答"从入口到某节点的所有路径是否都经过另一节点"（第 1 章第 5 节的定义）。对象是 `DominatorTree`，典型消费者：mem2reg（找 phi 插入点）、LICM（判断能否外提）、代码下沉。第 1 章 verifier 的 `does not dominate all uses` 检查就是它的一条硬约束。

### 5.2 LoopInfo 与 ScalarEvolution

`LoopInfo` 回答"循环嵌套结构"：哪个块是头、哪些块在循环里、循环怎么嵌套。对象是 `Loop`/`LoopInfo`。

`ScalarEvolution`（SCEV）回答"循环里的表达式如何随迭代变化"。手算一个归纳变量的 SCEV 记号：

```text
for (i = 0; i < n; i++) 的 i 在 SCEV 里写成 {0, +, 1}<%loop>
读法: 进入 %loop 时是 0, 每绕回 %loop 一次加 1
```

这个记号让"循环跑几趟""地址按什么步长走"变成可计算的表达式：`for (i = 0; i < n; i += 2)` 是 `{0, +, 2}`；地址 `a[i]` 的 SCEV 是 `a + 4×{0,+,1}`（i32 每步 4 字节）。循环优化（展开、向量化）的第一件事就是向 SCEV 要这些表达式来**证明变换的边界**——"这个循环能展开吗"要先变成"SCEV 能证明步长和次数吗"。

### 5.3 Alias Analysis 与 MemorySSA

Alias Analysis（别名分析）回答"两个内存访问是否可能指向同一位置"，答案是保守的近似（`NoAlias` / `MayAlias` / `MustAlias`）。对象是 `AAResults`。

MemorySSA 把内存读写也组织成类似 SSA 的 def-use 链：每个 `load` 连到"能提供它值的那个 store/入口"。对象是 `MemorySSA`。两者的分工：别名分析判断"**可不可能**别名"，MemorySSA 回答"**这个 load 看见谁写的值**"。它们共同服务 DSE（判断一个 store 是否死）、LICM（判断 load 能否外提）、循环变换。

## 6. 分析失效：正确性的组成部分

假设一个 pass 删掉了一条 CFG 边，却仍然返回 `preserveSet<CFGAnalyses>()`，后续 pass 拿到的就是**过期的支配树**。后果不是立刻崩溃，而是更危险的东西——**基于错误事实的优化**：比如按过期支配关系把指令插到不该插的位置，产出的 IR 可能通过编译、结果却错。完整契约：

```text
修改 IR
  → 旧分析可能过期
  → pass 返回 PreservedAnalyses(诚实声明)
  → AnalysisManager 把"未声明保留"的结果全部作废
  → 后续 pass 需要时重新计算
```

判断一个改动后能否保留分析，四问清单：

1. 是否改变了基本块数量或 CFG 边？（是 → 支配树/循环信息大概率失效）
2. 是否改变了指令的 opcode、操作数、类型或内存行为？（是 → 依赖指令的分析失效）
3. 是否改变了循环、支配关系或函数调用图？
4. 是否**主动更新**了某个 analysis result，而不是简单丢弃？（有些 pass 会调用分析结果的 update 接口，比重建便宜）

## 7. Pipeline 与调试：把失效链变成可观察对象

现代 `opt` 用字符串描述 pipeline（【可运行代码】）：

```bash
opt -passes='mem2reg,instcombine,simplifycfg' input.ll -S -o output.ll   # 三个变换串行跑
opt -passes='verify' input.bc -disable-output                            # 只跑验证器
opt -passes='default<O2>' input.ll -S -o output.ll                       # 完整 O2 pipeline
```

观察每个 pass 的分析行为：

```bash
opt -passes='default<O2>' -debug-pass-manager input.ll -disable-output
# 输出每个 pass 的运行与失效信息: Running pass ... / Invalidating analysis: ...
opt -passes='instcombine' -print-before-all -print-after-all input.ll -disable-output
# 打印 pass 前后的 IR, 定位"是哪一步改坏了"
```

排查 pass 问题的工作流：**先缩小到单个函数**（`-print-before-all` 的输出非常庞大），再二分 pipeline 找第一个出错的 pass，最后看它前后的 IR diff。`llvm-reduce` 可以自动把出错输入缩小到最小复现。

## 8. 源码阅读地图

- `llvm/include/llvm/IR/PassManager.h`：Pass、AnalysisManager、PreservedAnalyses 的核心模板；
- `llvm/lib/IR/PassManager.cpp`：部分运行时逻辑（分析缓存的实现在这里）；
- `llvm/lib/Passes/PassBuilder.cpp`：默认 pipeline 的组装与字符串解析注册；
- `llvm/lib/Analysis/Dominators.cpp`：支配树实现（对照第 1 章手算）；
- `llvm/lib/Analysis/MemorySSA.cpp`：MemorySSA 的构建与更新；
- `llvm/include/llvm/Analysis/`：各分析接口与结果类型；
- `llvm/lib/Transforms/`：变换 pass 的实现（每个 `run` 返回的 PreservedAnalyses 是重点）。

阅读顺序：从一个具体 pass 的 `run` 进入 → 跟进它调用的 `AM.getResult<SomeAnalysis>(IR)` → 回到对应 analysis 的 result 类。不要一上来通读 `PassManager.h` 模板。

## 9. 常见错误与归因

| 现象 | 根因 | 修正 |
|---|---|---|
| pass 单独跑对、进 pipeline 就错 | 谎报 PreservedAnalyses，后续 pass 用过期的分析 | 按四问清单重新声明 |
| 只读 pass 返回 `none()` | 过度保守 | 只读可返回 `all()` |
| 编译时间被自己 pass 拖慢 | 每个 pass 都触发全量分析重算 | 精确 preserve，或主动 update |
| `-debug-pass-manager` 输出看不懂 | 不熟悉"运行/失效"两行成对出现 | 先读失效行，找到谁作废了谁 |
| 改了 CFG 却保留 CFGAnalyses | 忘了边的改变会连带支配关系 | 四问第 1 问：改过边吗 |

## 10. 本章检查点

完成以下四项才算通过本章：

1. 把 `CountAddPass` 改成统计每个基本块的指令数（保持只读、返回 `all()`）；
2. 写一个删除空基本块的 pass，按四问清单逐条判断：CFG 边变了吗？支配树能保留吗？给出结论与理由；
3. 手写 `for (i = 1; i < n; i += 3)` 的 SCEV 记号，并解释每个字段；
4. 用 `opt -passes='instcombine' -print-before-all -print-after-all` 跑一个函数，指出输出里 pass 前后 IR 的差异点，并说出 instcombine 做了哪种变换。

## 11. 本章小结与下一步

本章把"分析"从名词变成了一条有契约的链：算 → 缓存 → 被变换作废 → 按需重算。下一章（LLVM 03：写 Pass 与测试）把本章的 `CountAddPass` 变成真正能改 IR、能进 `opt`、有 FileCheck 测试的 pass——本章的失效契约在那里第一次进入实战。

**导航**：⬅ [上一章](01_ir_ssa.md)（IR、SSA 与验证器）　｜　[下一章](03_write_pass_and_tests.md)（写 Pass、接入 opt 与测试）➡
