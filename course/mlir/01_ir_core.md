# MLIR 第 1 章：Operation、Region、Block 与 Value——一个可扩展的层次化 IR

## 1. 本章目标

- 能画出 `Operation → Region → Block → Operation` 的嵌套结构，并说出 operands / results / attributes / successors / traits 各是什么；
- 能读懂带 `scf.if` 的 MLIR 代码，解释 block argument 如何承载"从外面传进来的值"；
- 能区分三种 region 语义（CFG / graph / SSACFG），并知道如何从 operation 的约束判断；
- 能说清 Type / Attribute / Location / MLIRContext 四个对象的职责；
- 能用 `mlir-opt` 打印 IR、跑验证器，并做一次负向实验。

前置：LLVM 专题第 1 章的 SSA/支配概念（MLIR 的 SSA 是同一套规则在嵌套结构上的推广）。工具：MLIR（`mlir-opt`），LLVM ≥ 15 自带。

## 2. 工作中的问题长什么样

MLIR 方向的三个入门问题：

```text
"MLIR 和 LLVM IR 到底差在哪？不都是 SSA 吗？"
"%arg0 是什么？它没有定义它的指令啊？"
"这个 region 里有两个 block，它表达的是控制流吗？"
```

三个问题的答案：**可扩展性**（语义由 Dialect 定义，容器是通用的）、**block argument**（值从"入口"进入，而不是由指令产生）、**region 语义**（不是所有 region 都是 CFG）。本章建立这三个认知。

## 3. 最小对象模型：通用容器 + 方言定义语义

MLIR 的核心设计：**IR 只提供通用容器，具体语义由 Dialect（方言）定义**。容器结构：

```text
Operation(操作: IR 的基本单元)
  ├─ operands: Value(输入的操作数值)
  ├─ results: Value(本操作产生的值)
  ├─ attributes: Attribute(编译期静态数据)
  ├─ regions(嵌套的 region 列表)
  │    └─ blocks(region 内的 block 列表)
  │         └─ operations(block 内的操作列表)
  ├─ successors(CFG 语义操作的跳转目标)
  ├─ traits / interfaces(声明行为: 可交换? 有副作用?)
  └─ location(源码位置, 用于诊断)
```

逐对象一句话定义：**Operation** 是 IR 的基本单元，名字形如 `arith.addi`（Dialect 名.操作名），名字决定语义实现；**Value** 是所有 SSA 值的基类，分两类——operation 的 result 与 block 的 argument；**Region** 是 block 的容器；**Block** 是 operation 的容器，可以有 block arguments 与终结操作。**同一个容器能表达函数、循环、if 分支和嵌套计算**——`ModuleOp` 包含函数，函数包含 region，region 含 block，block 含操作，全部是同一套基础设施。

## 4. 一个可读的 MLIR 示例：逐行拆解

```mlir
module {
  func.func @double(%arg0: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %x = arith.addi %arg0, %c1 : i32
    return %x : i32
  }
}
```

逐行对象：

- `module { ... }`：`ModuleOp`，模块级容器（对应 LLVM 的 Module）；
- `func.func @double(%arg0: i32) -> i32`：`func.func` 操作定义一个函数；`@double` 是符号名；**`%arg0` 是 block argument**——函数体的第一个 block 把参数声明为入口值，**它不由任何指令产生**，是"从外面传进来的"；
- `%c1 = arith.constant 1 : i32`：`arith.constant` 是 arith 方言的操作，`%c1` 是它的 result，`1` 是它的 attribute（静态值），`i32` 是 type；
- `%x = arith.addi %arg0, %c1 : i32`：`arith.addi` 的 operands 是 `%arg0, %c1`，result 是 `%x`；
- `return`：func 方言的终结操作。

与 LLVM IR 的关键对比：LLVM 的指令种类是**固定的 opcode 枚举**（add/sub/br...），加新语义要改核心；MLIR 的"指令种类"是**operation 名字**，`arith.addi` 与 `scf.if` 只是两个不同 Dialect 注册的名字，**新方言 = 新名字 + 新语义实现，核心 IR 不动**。这就是"可扩展 IR"的含义：容器通用，语义外挂。

## 5. 嵌套 region 与 block argument：scf.if 的例子

把示例扩出控制流：

```mlir
func.func @select(%cond: i1, %x: i32, %y: i32) -> i32 {
  %r = scf.if %cond -> i32 {        ; scf.if 的 region 产生一个结果 i32
    scf.yield %x : i32              ; then 分支: 把 %x 交出去
  } else {
    scf.yield %y : i32              ; else 分支: 把 %y 交出去
  }
  return %r : i32
}
```

拆解：`scf.if` 操作带一个 region（then）和一个可选 region（else）；两个 region 各自用 `scf.yield` 把值交回 `scf.if`，`scf.if` 的 result `%r` 就是"哪个分支 yield 了哪个值"。**region 里直接使用外层定义的 `%x`/`%y`**——SSA 的支配规则在嵌套结构上依然成立：外层值支配内层使用点（LLVM 第 1 章的规则原样适用，只是"路径"变成了跨 region 的嵌套路径）。

另一个形态：循环携带值用 **block argument** 而不是 phi：

```mlir
func.func @sum(%n: i32) -> i32 {
  %init = arith.constant 0 : i32
  %res = scf.for %i = %init to %n step %c1 iter_args(%acc = %init) -> i32 {
    %next = arith.addi %acc, %i : i32
    scf.yield %next : i32            ; 交回的值成为下一轮 %acc, 最后一轮成为 %res
  }
  return %res : i32
}
```

对照 LLVM 的 phi：**MLIR 把"沿边选值"写成了 block argument**——`iter_args(%acc = %init)` 声明"入口时 %acc 取 %init，每次回边取 `scf.yield` 交回的值"。这是同一个 SSA 合流问题的两种语法，理解其一即可迁移。

## 6. Region 与 Block 的区别：不是所有 region 都是 CFG

**Region** 是一个或多个 block 的容器；**Block** 是带 block arguments 与操作列表的基本控制流单元。关键区别在于 **region 的语义种类**：

| 种类 | 语义 | 例子 |
|---|---|---|
| CFG region | region 内 block 是控制流节点，有跳转 | 通用分支结构 |
| Graph region | 操作之间只有 SSA 依赖，无隐含顺序 | 图方言的 region |
| SSACFG region | 结构化控制流：单入口、block 有明确分支 | `scf.if` 的 then/else |

**不要看到 region 就当成 LLVM 的 basic block**。判断方法：看该 operation 的 verifier、traits 与接口——`scf.if` 的 region 是否允许两个 block、block 能不能有跳转，都是 `scf.if` 自己声明的约束，不是 IR 容器的默认行为。

## 7. Type、Attribute、Location 与 Context

| 对象 | 一句话定义 | 例子 |
|---|---|---|
| `Type` | 值的编译期类型 | `i32`、`tensor<4xf32>`、`memref<...>` |
| `Attribute` | 附着在操作/类型上的**静态**数据 | `1`（常量值）、布局描述、符号引用 |
| `Location` | 源码位置，用于诊断与调试 | 文件名:行:列 |
| `MLIRContext` | 全库的上下文：uniqued storage、dialect 加载、线程设施 | 每个 IR 树挂在一个 context 下 |

第一近似："**值是数据，属性是静态描述**"（`arith.constant 1` 里的 `1` 是 attribute，因为它在编译期已知、不参与运行时数据流）。但具体 Dialect 可能把复杂静态信息编码成自定义 Attribute——这条近似是入口，不是全部。

## 8. 验证与打印：负向实验

```bash
mlir-opt input.mlir                        # 解析并打印(往返)
mlir-opt input.mlir -verify-diagnostics    # 把报错也当测试断言(配合 expected-error 注释)
mlir-opt input.mlir -mlir-print-ir-before-all -mlir-print-ir-after-all   # 每个 pass 前后打印
mlir-opt input.mlir -o output.mlir         # 写回文件
```

验证器检查三类东西：operation 自己的约束（`arith.addi` 两个操作数同类型）、region 结构（`scf.if` 的 region 数）、通用 SSA 不变量（支配、use-def）。**解析成功不等于语义正确**——负向实验：把 `arith.addi %arg0, %c1` 的两个操作数改成不同类型（`i32` 与 `f32`），跑 `mlir-opt`，预期 verifier 报类型不匹配。Pass 之间打开 verifier（`-verify-each`）可以定位是哪一步破坏了 IR。

## 9. 源码阅读地图

- `mlir/include/mlir/IR/Operation.h`：操作核心接口（operands/results/regions/attributes）；
- `mlir/include/mlir/IR/Region.h`、`Block.h`：嵌套容器与 CFG；
- `mlir/include/mlir/IR/Value.h`：result 与 block argument 的基类；
- `mlir/include/mlir/IR/MLIRContext.h`：上下文与 uniqued 存储；
- `mlir/lib/IR/Operation.cpp`、`Region.cpp`：实现细节；
- `mlir/tools/mlir-opt/`：命令行工具与 pass 注册。

从四个高频 API 追起：`Operation::walk`（遍历嵌套树）、`getUsers`（use 列表）、`replaceAllUsesWith`（替换值）、region/block 的插入删除接口——变换 pass 几乎都建立在这四个操作上。

## 10. 常见错误与归因

| 现象 | 根因 | 修正 |
|---|---|---|
| verifier 报类型不匹配 | 操作数类型与操作约束不符 | 对照该操作的 verifier 约束 |
| 值在 region 里不可见 | 违背 SSA 支配规则（内层值用在外层） | 用 block argument / yield 把值传出去 |
| 把 region 当 CFG 读 | 该 region 是 graph 语义 | 查 operation 的 traits/verifier |
| 解析成功但语义错 | verifier 不检查的语义错误 | 语义测试 + 正确性参考 |
| pass 之间 IR 被破坏 | 某个 pass 违反不变量 | `-verify-each` 定位破坏步骤 |

## 11. 本章检查点

完成以下四项才算通过本章：

1. 给第 4 节示例加一个 `scf.if`，画出完整嵌套树（Operation/Region/Block 三层），标出每个 value 属于 result 还是 block argument；
2. 解释 `scf.for ... iter_args(%acc = %init)` 里 `%acc` 的取值规则，并对照 LLVM phi 写出等价表达；
3. 做一次负向实验：制造一个类型不匹配的 `arith.addi`，记录 verifier 报错原文，判断它属于三类检查中的哪一类；
4. 用一句话回答"MLIR 与 LLVM IR 都叫 IR，本质差别是什么"，并各举一个具体对象佐证。

## 12. 本章小结与下一步

本章建立了 MLIR 的最小对象模型：容器通用、语义外挂、block argument 承载合流。下一章（MLIR 02：Dialect 与 ODS）回答"arith 这种方言是怎么定义出来的"——operation 的名字、约束、验证器如何用 TableGen 声明式生成。

**导航**：⬅ 上一章：无（本专题第一章，先看 [专题目录](README.md)）　｜　[下一章](02_dialect_ods.md)（Dialect、ODS 与 TableGen）➡
