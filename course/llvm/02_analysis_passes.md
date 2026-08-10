# LLVM 第 2 章：Analysis、Pass 与新 Pass Manager

## 1. 本章目标

- 区分 analysis pass 与 transform pass；
- 理解 `AnalysisManager`、`PreservedAnalyses` 和分析失效；
- 知道何时使用 `DominatorTree`、`LoopInfo`、`ScalarEvolution`、Alias Analysis 和 MemorySSA；
- 能沿着一个 Pass pipeline 追到它需要的分析。

## 2. Pass 的本质

一个 transform pass 读取 IR 和分析结果，可能修改 IR；一个 analysis pass 计算事实并缓存结果。缓存的价值在于多个变换可以复用昂贵分析，但前提是变换准确声明哪些分析仍然有效。

```text
Function F
  ├─ DominatorTreeAnalysis → 支配关系
  ├─ LoopAnalysis          → 循环结构
  ├─ MemorySSAAnalysis     → 内存访问依赖
  └─ TransformPass         → 修改 F，并声明保留了什么
```

## 3. 新 Pass Manager 的核心接口

一个简化的函数 Pass 如下：

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

    llvm::errs() << F.getName() << ": " << Count << " adds\\n";
    return llvm::PreservedAnalyses::all();
  }
};
```

这个例子只读不写，所以返回 `all()` 是合理的。如果 Pass 插入、删除或替换指令，就不能机械地返回 `all()`；最保守的做法是返回 `none()`，更好的做法是准确声明仍然有效的分析。

## 4. 分析结果怎么选

### 4.1 DominatorTree

回答“从入口到某个节点的所有路径是否都经过另一个节点”。它用于判断定义是否支配使用、放置代码、构造 SSA 和理解循环入口。

### 4.2 LoopInfo 与 ScalarEvolution

`LoopInfo` 提供循环嵌套结构；`ScalarEvolution` 尝试表达 induction variable、步长和迭代次数。循环优化通常先用它们证明变换的边界，再修改 CFG 或指令。

### 4.3 Alias Analysis 与 MemorySSA

Alias Analysis 估计两个内存访问是否可能指向同一位置；MemorySSA 把内存读写也组织成类似 SSA 的 def-use 链。它们经常共同服务于 load/store 优化、DSE、LICM 和循环变换。

## 5. 分析失效是正确性的组成部分

假设 Pass 改了 CFG，但仍然把旧的 `DominatorTree` 当作有效结果返回，后续 Pass 看到的就是过期事实。错误未必立刻崩溃，更危险的是产生错误优化。

```text
修改 IR
  → 旧分析可能过期
  → Pass 返回 PreservedAnalyses
  → AnalysisManager 使未保留结果失效
  → 后续 Pass 重新计算需要的分析
```

判断是否可以保留分析时，至少问：

1. 是否改变了基本块数量或 CFG 边？
2. 是否改变了指令的 opcode、操作数、类型或内存行为？
3. 是否改变了循环、支配关系或函数调用图？
4. 是否需要主动更新某个 analysis result，而不是简单丢弃？

## 6. Pipeline 与调试

现代 `opt` 可以用字符串描述 Pass pipeline：

```bash
opt -passes='mem2reg,instcombine,simplifycfg' input.ll -S -o output.ll
opt -passes='verify' input.bc -disable-output
opt -passes='default<O2>' input.ll -S -o output.ll
```

定位某个 Pass 的影响时，常用：

```bash
opt -passes='default<O2>' -debug-pass-manager input.ll -disable-output
opt -passes='instcombine' -print-before-all -print-after-all input.ll -disable-output
```

输出非常大时，应先缩小到一个函数，再用 `llvm-reduce` 或手工删除无关代码。

## 7. 源码阅读地图

- `llvm/include/llvm/IR/PassManager.h`：Pass、AnalysisManager、PreservedAnalyses 的核心模板；
- `llvm/lib/IR/PassManager.cpp`：部分运行时逻辑；
- `llvm/lib/Passes/PassBuilder.cpp`：默认 pipeline 和 pipeline 解析注册；
- `llvm/lib/Analysis/Dominators.cpp`：支配树实现；
- `llvm/lib/Analysis/MemorySSA.cpp`：MemorySSA 构建与更新；
- `llvm/include/llvm/Analysis/`：分析接口和结果类型；
- `llvm/lib/Transforms/`：各类变换 Pass 的实现。

源码阅读技巧：先从一个具体 Pass 的 `run` 进入，再跟进它调用的 `AM.getResult<SomeAnalysis>(IR)`，最后回到对应 analysis 的 result 类，而不是一开始通读整个 `PassManager.h`。

## 8. 练习

1. 把 `CountAddPass` 改成统计每个基本块的指令数；
2. 使用 `DominatorTreeAnalysis`，打印入口块支配的块数量；
3. 找一个会改 CFG 的 Pass，分析为什么不能保留旧的支配树；
4. 比较 `mem2reg` 前后的 IR，说明它依赖哪些 SSA 事实。

参考：[Using the New Pass Manager](https://llvm.org/docs/NewPassManager.html)、[LLVM Analysis and Transform Passes](https://llvm.org/docs/Passes.html)、[MemorySSA](https://llvm.org/docs/MemorySSA.html)。

