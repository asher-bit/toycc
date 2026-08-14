# MLIR 第 4 章：Bufferization、Lowering 与测试——从值语义的张量到内存

## 1. 本章目标

- 能说清 tensor IR（值语义）与 memref IR（内存语义）的职责差异，并解释"bufferization 不是把 tensor 换成指针"；
- 能手算一个 in-place vs out-of-place 的 bufferization 决策；
- 能设计并跑通一条从高层 op 到 LLVM Dialect 的 lowering 链，逐 pass 说出作用；
- 能为每一层写 FileCheck 回归测试，并用负向实验验证测试有效。

前置：第 3 章的 conversion 框架（本章的 lowering 就是一组 conversion pass 的串联）。工具：MLIR（`mlir-opt`、lit）。

## 2. 工作中的问题长什么样

bufferization/lowering 方向的两个典型问题：

```text
"bufferize 之后结果错了，但 IR 里每个 op 看起来都对？"
"下降链里有一百个 pass，出错了该从哪一层查起？"
```

两个问题的答案：**决策复杂度**（in-place 判断依赖别名与生命周期，不是机械替换）与**分层测试**（每层单独钉住，错误定位就从"大海捞针"变成"二分查找"）。本章建立这两件事。

## 3. Tensor 与 MemRef：两种语义

**tensor** 表达**值语义**：一个 op 产生一个逻辑张量，变换只关心形状、广播、算子组合，不关心"数据放在哪块内存"。**memref** 表达**内存语义**：有内存地址、布局、偏移、索引与访问方式。中间的桥就是 bufferization：

```text
tensor<4xf32>  ——值/形状语义——→  memref<4xf32>  ——地址/布局语义——→ LLVM
```

**为什么 bufferization 不是"把 tensor 换成指针"**：值语义里"两个 op 输入同一个张量"没有任何内存后果；内存语义里"两个输入指向同一块 buffer"是别名问题，直接决定改写是否合法。bufferization 要回答四个问题：

1. 一个 tensor 能否**原地写回**某个 buffer（in-place）；
2. 两个值是否可能**别名**（指向同一内存）；
3. 何时必须**分配新 buffer 并插入拷贝**（out-of-place）；
4. buffer 的**所有权与生命周期**在函数边界如何处理。

所以 bufferization 是一组依赖别名、读写与生命周期分析的**决策过程**，one-shot bufferize 里这些决策由分析与接口共同完成。

## 4. 手算一个 in-place 决策

例子：`%y = tensor.add %x, %c`（c 是常量张量）。决策顺序：

```text
问 1: %x 之后还有别的使用者吗？
  没有 → %y 可以直接复用 %x 的 buffer(in-place), 省一次分配 + 一次拷贝
  有   → 不能覆盖 %x(别的使用者还要读原值) → 分配新 buffer(out-of-place)
问 2(别名): 两个输入 %a、%b 是同一个 buffer 吗？
  是 → %y = add(%a, %a) 若 in-place 写 %a, 会边读边写同一块内存 → 必须 out-of-place
```

一笔账：一个 `tensor<4096xf32>`（16 KB）的 add，in-place 省掉"分配 16 KB + 拷贝 16 KB"；如果整条链上的每个 op 都多做一次分配+拷贝，端到端多出一倍以上的内存往返——**in-place 优化的收益就是 bufferization 分析存在的意义**。这个"谁死了谁的内存能复用"的逻辑，与 toycc 内存规划 pass 的缓冲区复用是同一个思想：内存语义下，生命周期分析决定复用是否合法。

## 5. 一条典型下降链：逐 pass 说作用

具体 pipeline 随项目与版本变化，概念链是：

```text
自定义/算子 Dialect
  → linalg / tensor(结构化操作、值语义张量)
  → one-shot bufferization(tensor → memref, 含 in-place 决策)
  → memref / scf / affine(内存 + 循环)
  → loops / cf / func / arith(标量控制流)
  → llvm dialect(接近 LLVM IR 的方言)
  → LLVM IR
```

命令示意（【可运行代码】，pass 名随版本变化，先 `mlir-opt --help`）：

```bash
mlir-opt input.mlir \
  -one-shot-bufferize \
  -convert-linalg-to-loops \
  -convert-scf-to-cf \
  -convert-arith-to-llvm \
  -convert-func-to-llvm \
  -convert-memref-to-llvm \
  -convert-cf-to-llvm \
  -o lowered.mlir
```

每个 pass 只做一层：`-one-shot-bufferize` 完成 tensor→memref（第 3/4 节的决策）；`-convert-linalg-to-loops` 把结构化 op 展开成循环；`-convert-scf-to-cf` 把结构化控制流转成 CFG；后面的 `-convert-*-to-llvm` 把 arith/func/memref/cf 各方言逐一映射到 LLVM 方言。**每一步之间都跑一次 verifier**（`mlir-opt` 默认开启或显式加 `-verify-each`），IR 在哪一步被破坏，错误就停在哪一步——这就是"从哪一层查起"的答案：逐层验证，而不是等到最后对着 LLVM IR 发呆。

## 6. 每一层为什么要单独测试

只测最终 LLVM IR 时，错误无法归因——算子语义错、布局错、bufferization 决策错、循环生成错、LLVM lowering 错，五种错误长得一模一样。至少分五层测试：

```text
1. Dialect verifier/parser 测试        (op 约束是否正确)
2. canonicalization/folding 测试       (同层改写是否正确)
3. conversion 后的结构测试             (目标方言结构对不对)
4. bufferization 的 in-place 决策测试  (第 4 节的决策对不对)
5. LLVM Dialect / 最终 IR 的 ABI 测试  (调用与类型边界)
```

每层一个 FileCheck 用例，出错时先跑最底层（语义层）再逐层向上——**回归测试的层级 = 排查的二分区间**。

## 7. lit 与 FileCheck 的测试形状

```mlir
// RUN: mlir-opt %s -canonicalize | FileCheck %s

func.func @add_zero(%arg0: i32) -> i32 {
  %zero = arith.constant 0 : i32
  %r = arith.addi %arg0, %zero : i32
  return %r : i32
}

// CHECK-LABEL: func.func @add_zero
// CHECK-NOT: arith.addi
// CHECK: return %arg0 : i32
```

三行 CHECK 的语义：`CHECK-LABEL` 把匹配范围钉在这个函数；`CHECK-NOT: arith.addi` 断言"从 LABEL 到下一 CHECK 之间**不得**出现 `arith.addi`"（负向断言——canonicalize 后 `x+0` 应该消失）；`CHECK: return %arg0` 断言简化结果就是直接返回 `%arg0`。**关键纪律（与 LLVM 专题第 3 章相同）**：先做负向实验——故意让测试应失败一次（比如注释掉 canonicalize），确认测试真的会拦；只测"命令退出码 0"的测试等于没测。诊断类测试用 `-verify-diagnostics` 约定，在输入里用 `expected-error` 注释标出期望报错位置。

## 8. 源码阅读地图

- `mlir/include/mlir/Dialect/Bufferization/`：bufferization 的公共接口；
- `mlir/lib/Dialect/Bufferization/`：分析、接口与 pass 实现（in-place 决策的核心）；
- `mlir/docs/Bufferization/`：设计约束与术语；
- `mlir/lib/Conversion/`：各方言之间的转换 pass；
- `mlir/lib/Dialect/LLVMIR/`：LLVM 方言；
- `mlir/test/`：按 Dialect/Conversion/Transforms 组织的 lit 测试。

读一个 lowering pass 的顺序：找 `runOnOperation`/`run` → 找 `populate*Patterns` → 看 conversion target 与 TypeConverter → 回到测试目录验证合法性边界是否被覆盖。

## 9. 常见错误与归因

| 现象 | 根因 | 修正 |
|---|---|---|
| bufferize 后结果错、IR 看似都对 | in-place 决策错误（别名/生命周期漏判） | 第 4 节四问逐条核对 + in-place 决策测试 |
| 最终 IR 错但不知从哪查 | 没有分层测试 | 第 6 节五层各补 FileCheck |
| 某 pass 后 IR 非法 | 该 pass 的转换不完整 | `-verify-each` 定位到具体 pass |
| 测试永远通过 | 只断言退出码 / CHECK 写太宽 | 负向实验 + CHECK-NOT |
| in-place 优化没生效 | bufferization 分析被保守限制 | 检查别名信息是否被保守截断 |

## 10. 本章检查点

完成以下四项才算通过本章：

1. 对 `%y = tensor.add %x, %c` 写出 in-place 合法的两个前提条件（生命周期 + 别名），并各举一个反例；
2. 把第 5 节的下降链跑一遍，记录每个 pass 之后 IR 里剩余的方言种类；
3. 写一个 FileCheck 测试确认"某个 illegal op 在 conversion 后消失"，并做一次负向实验；
4. 手算：一个 `tensor<4096xf32>` 的 add，in-place 相比 out-of-place 省下多少字节的内存往返（分配 + 拷贝各算一次）。

## 11. 本章小结与下一步

MLIR 专题到这里闭环：IR 核心对象 → 方言定义 → 改写与转换 → 内存落地与测试。至此 LLVM + MLIR 专题 8 章全部完成。两条继续深入的路线：回到主教材第 26 课把本章与"自研芯片前端"接起来；或进入主教材第 27~30 课，看这些编译产物如何被模拟器、加载器与驱动消费。

**导航**：⬅ [上一章](03_rewrite_and_conversion.md)（Pattern Rewrite 与 Dialect Conversion）　｜　本专题完，返回 [专题目录](README.md) ➡
