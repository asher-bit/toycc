# LLVM 第 1 章：IR、SSA 与验证器

## 1. 本章目标

完成本章后，你应当能：

- 说明 `Module`、`Function`、`BasicBlock`、`Instruction`、`Value` 和 `Type` 的关系；
- 读懂带分支、循环和 `phi` 的 LLVM IR；
- 解释 SSA、支配关系和为什么需要 verifier；
- 用 `llvm-as`、`llvm-dis`、`opt`、`llc` 观察 IR 的变化。

## 2. LLVM IR 的最小心智模型

LLVM IR 是一种带类型的 SSA 中间表示。一个模块可以包含全局变量、函数声明和函数定义；函数由基本块组成，基本块由指令组成。大多数产生值的指令会产生一个 SSA 名字，这个名字在其定义点之后被使用。

```llvm
target triple = "x86_64-pc-linux-gnu"

define i32 @sum(i32 %a, i32 %b) {
entry:
  %s = add i32 %a, %b
  ret i32 %s
}
```

逐行看：

- `target triple` 描述目标平台；它会影响默认 ABI、指令选择和可用特性；
- `define i32 @sum` 定义一个返回 `i32` 的函数；
- `%a`、`%b` 是函数参数，也是 SSA 值；
- `add i32` 明确指定操作数和结果类型；
- `ret` 结束当前基本块并返回结果。

LLVM 新版本使用 opaque pointer 时，指针本身不再携带指向元素的类型；访问内存时仍要通过 `load`、`store`、GEP 等指令的类型信息表达语义。读代码时不要把“LLVM IR 有类型”和“每个指针都有具体 pointee type”混为一谈。

## 3. 基本块、终结指令与 SSA 合流

一个基本块通常以 `br`、`switch` 或 `ret` 等终结指令结束。控制流边决定了哪些块可能到达哪些块；数据流则决定某个 SSA 值在哪些位置可用。

```llvm
define i32 @select_value(i1 %cond, i32 %x, i32 %y) {
entry:
  br i1 %cond, label %then, label %else

then:
  br label %join

else:
  br label %join

join:
  %v = phi i32 [ %x, %then ], [ %y, %else ]
  ret i32 %v
}
```

`phi` 不是运行时执行的一条普通机器指令。它表示：沿哪条前驱边进入 `join`，就选择对应边上携带的值。这里的关键约束是“每个 incoming value 与一个前驱块一一对应”。

### 3.1 为什么要看支配关系

如果一个定义要在某个使用点可见，那么该定义必须支配这个使用点；对于循环和多路分支，合流点往往需要插入 `phi`。很多 LLVM Pass 的第一步不是改指令，而是从 `DominatorTree` 或 `LoopInfo` 获取控制流事实。

### 3.2 `getelementptr` 是地址计算

```llvm
%p = getelementptr i32, ptr %base, i64 %index
%x = load i32, ptr %p, align 4
```

GEP 计算地址，不读取内存；真正读取内存的是 `load`。这一区分在别名分析、向量化和后端寻址模式匹配中都很重要。

## 4. 目标信息不是装饰

模块里的 target triple 和 `DataLayout` 会告诉优化器：指针宽度、整数和浮点类型的 ABI 对齐、地址空间、端序以及某些类型的布局。一个只在“抽象 IR”上看起来合法的变换，如果忽略目标布局，可能在后端产生错误代码。

```text
IR 语义正确性  →  verifier
目标相关合法性  →  DataLayout / TargetTransformInfo / target hooks
机器级可实现性  →  TargetLowering / instruction selector
```

## 5. 工具实验

把上面的代码保存为 `sum.ll` 后，可以按顺序执行：

```bash
llvm-as sum.ll -o sum.bc
llvm-dis sum.bc -o -
opt -passes='verify' sum.bc -disable-output
llc sum.ll -o sum.s
```

`llvm-as` 检查文本 IR 并生成 bitcode；`llvm-dis` 反向打印；`verify` 检查 IR 不变量；`llc` 将 IR 交给目标相关代码生成器。故意删掉 `phi` 的一个 incoming edge，再运行 verifier，观察错误信息。

## 6. 源码阅读地图

建议按以下顺序打开源码：

1. `llvm/include/llvm/IR/Value.h`：所有 SSA 值共享的基础接口；
2. `llvm/include/llvm/IR/Instruction.h`：指令是 Value，同时属于某个基本块；
3. `llvm/include/llvm/IR/BasicBlock.h`：指令链表和 CFG 关系；
4. `llvm/include/llvm/IR/Function.h`、`Module.h`：函数和模块容器；
5. `llvm/lib/IR/Verifier.cpp`：把 LangRef 中的约束变成可执行检查；
6. `llvm/include/llvm/IR/DataLayout.h`：目标数据布局查询接口。

读每个类时，重点记录三件事：对象由谁拥有、迭代器遍历什么、修改后哪些分析结果会失效。

## 7. 练习

1. 写一个带循环的 `sum_to_n`，找出循环头的 `phi`；
2. 用 `opt -passes='mem2reg'` 比较带 `alloca/load/store` 的 IR 与 SSA IR；
3. 对一个非法 IR 运行 verifier，整理错误对应的 IR 不变量；
4. 思考：为什么 GEP 与 load 分开，为什么这有利于优化？

## 8. 本章小结

读 LLVM 源码时，先把 IR 当成“带控制流约束的数据结构”，再把每个 Pass 看作对这些结构的不变量进行维护。只看指令名字而不看支配关系、类型、布局和分析失效规则，很容易得到错误的结论。

参考：[LLVM Language Reference](https://llvm.org/docs/LangRef.html)、[LLVM Programmer’s Manual](https://llvm.org/docs/ProgrammersManual.html)。

