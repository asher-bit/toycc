# MLIR 第 4 章：Bufferization、Lowering 与测试

## 1. 本章目标

- 理解 tensor IR 与 memref IR 的职责差异；
- 说明 bufferization 为什么不仅是“把 tensor 换成指针”；
- 设计一条从高层 operation 到 LLVM Dialect 的 lowering 链；
- 用 `lit` 和 `FileCheck` 为每一层建立回归测试。

## 2. Tensor 与 MemRef

Tensor 更适合表达值语义：一个 operation 产生一个逻辑张量，变换可以关注形状、广播和算子组合。MemRef 更接近内存语义：包含内存、布局、偏移、索引和访问方式。

```text
tensor<4x f32>  ——值/形状语义——→  memref<4xf32>  ——地址/布局语义——→ LLVM
```

Bufferization 要回答的问题包括：

- 一个 tensor 是否可以原地写回某个 buffer；
- 两个值是否可能别名；
- 何时需要分配新 buffer 和插入拷贝；
- buffer 的所有权、生命周期和函数边界如何处理。

所以 bufferization 是一组依赖别名、读写和生命周期分析的决策，而不是简单的文本替换。

## 3. 一条典型的下降链

具体 pipeline 会随项目和版本变化，概念上可以是：

```text
自定义/算子 Dialect
  → linalg / tensor
  → one-shot bufferization
  → memref / scf / affine
  → loops / cf / func / arith
  → llvm dialect
  → LLVM IR
```

命令示意：

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

这些 pass 名称和依赖会随 MLIR 版本、构建选项及输入 Dialect 改变，实际执行前先运行 `mlir-opt --help`，并用 verifier 检查每个阶段的结果。

## 4. 每一层为什么要单独测试

如果只测试最终 LLVM IR，出现错误时很难判断是算子语义、布局、bufferization、循环生成还是 LLVM lowering 造成的。推荐至少分成：

1. Dialect verifier/parser 测试；
2. canonicalization/folding 测试；
3. conversion 后的 operation 结构测试；
4. bufferization 的 in-place/out-of-place 决策测试；
5. LLVM Dialect 或最终 LLVM IR 的 ABI 测试。

## 5. `lit` 与 `FileCheck` 的测试形状

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

测试应当检查语义相关结构，而不是把所有无关属性和编号都写死。需要诊断测试时，可以使用 MLIR 的 `-verify-diagnostics` 约定，在输入中标出期望的错误位置。

## 6. 源码阅读地图

- `mlir/include/mlir/Dialect/Bufferization/`：bufferization 的公共接口；
- `mlir/lib/Dialect/Bufferization/`：分析、接口和 pass 实现；
- `mlir/docs/Bufferization/`：设计约束和术语；
- `mlir/lib/Conversion/`：从一个 Dialect 到另一个 Dialect 的转换；
- `mlir/lib/Dialect/LLVMIR/`：LLVM Dialect；
- `mlir/test/`：按 Dialect、Conversion、Transforms、Pass 等组织的 lit 测试。

读 lowering pass 时，先找 pass 的 `runOnOperation` 或 `run`，再找 `populate*Patterns`，然后看 conversion target 和 pass pipeline。最后回到测试目录，验证每个合法性边界是否真的被覆盖。

## 7. 练习

1. 对同一个 tensor 程序比较 in-place 与 out-of-place bufferization；
2. 在 lowering 链中每完成一个阶段就运行 verifier，并记录剩余 Dialect；
3. 写一个 FileCheck 测试，确认某个 illegal op 已经消失；
4. 把一个最终失败的测试拆成“输入 IR”“中间 IR”“最终 IR”三个最小回归用例。

参考：[Bufferization](https://mlir.llvm.org/docs/Bufferization/)、[MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)、[MLIR Pass Management](https://mlir.llvm.org/docs/PassManagement/)。

