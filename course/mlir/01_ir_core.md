# MLIR 第 1 章：Operation、Region、Block 与 Value

## 1. 本章目标

- 读懂 MLIR 的层次化 IR；
- 区分 operation、region、block、value、type、attribute 和 location；
- 理解 SSA 在嵌套区域中的作用；
- 能用 `mlir-opt` 打印和观察 IR。

## 2. MLIR 的最小对象模型

MLIR 不把所有语义塞进固定的一组指令。它提供通用容器，具体语义由 Dialect 定义：

```text
Operation
  ├─ operands: Value
  ├─ results: Value
  ├─ attributes: Attribute
  ├─ regions
  │    └─ blocks
  │         └─ operations
  ├─ successors（有 CFG 语义的 operation 可用）
  ├─ traits / interfaces
  └─ location
```

一个 `ModuleOp` 可以包含函数，函数包含 region，region 包含 block，block 再包含操作。这个结构让高层控制流、循环、函数和嵌套计算都能用同一套 IR 基础设施表达。

## 3. 一个可读的 MLIR 示例

```mlir
module {
  func.func @double(%arg0: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %x = arith.addi %arg0, %c1 : i32
    return %x : i32
  }
}
```

- `func.func` 和 `arith.constant` 是不同 Dialect 的 operation；
- `%arg0` 是 block argument，不是由某条指令产生的结果；
- `%c1`、`%x` 是 operation results；
- `i32` 是 type；
- operation 名称负责选择语义实现，属性和 region 提供额外结构。

## 4. Region 与 Block 的关键区别

Region 是一个或多个 block 的容器；block 是带有 block arguments 和 operation 列表的基本控制流单元。并非所有 region 都必须表达 CFG：有些 region 是图式或顺序语义，是否允许多个 block 由 operation 的约束决定。

因此读 MLIR 源码时，不要看到 region 就立刻把它当作 LLVM 的 basic block。要查看该 operation 的 verifier、traits 和接口，确认 region 的语义。

## 5. 类型、属性、位置与上下文

- `Type` 描述值的编译期类型，如 `i32`、`tensor<4xf32>`、`memref<...>`；
- `Attribute` 是附着在 operation 或 type 上的静态数据，如整数常量、布局、字符串和符号引用；
- `Location` 保存诊断、源码映射和调试信息；
- `MLIRContext` 管理 uniqued storage、dialect 加载和线程相关基础设施。

“值是数据，属性是静态描述”是一个很有用的第一近似，但具体 Dialect 仍可能把复杂静态信息编码成自定义 Attribute。

## 6. 验证与打印

```bash
mlir-opt input.mlir -verify-diagnostics
mlir-opt input.mlir -mlir-print-ir-before-all -mlir-print-ir-after-all
mlir-opt input.mlir -o output.mlir
```

验证器负责检查 operation 自己的约束、类型关系、region 结构和通用 IR 不变量。解析成功不等于语义一定正确；Pass 之间打开 verifier，有助于定位破坏 IR 的步骤。

## 7. 源码阅读地图

- `mlir/include/mlir/IR/Operation.h`：operation 的核心接口；
- `mlir/include/mlir/IR/Region.h`、`Block.h`：嵌套结构和 CFG 容器；
- `mlir/include/mlir/IR/Value.h`：结果值和 block argument；
- `mlir/include/mlir/IR/MLIRContext.h`：上下文与存储；
- `mlir/lib/IR/Operation.cpp`、`Region.cpp`、`PatternMatch.cpp`：实现细节；
- `mlir/tools/mlir-opt/`：命令行工具和 pass 注册。

建议从 `Operation::walk`、`getUsers`、`replaceAllUsesWith`、region/block 插入删除接口开始追，因为变换 Pass 几乎都会依赖这些操作。

## 8. 练习

1. 给示例增加一个 `scf.if`，观察嵌套 region 和 block argument；
2. 用 `mlir-opt` 打印每个 Pass 前后的 IR；
3. 找一个 verifier 错误，判断它属于类型约束、region 约束还是通用 SSA 约束；
4. 对比 MLIR 的 `Value` 与 LLVM 的 `Value`：它们都表达 SSA，但所在 IR 的结构和扩展方式有什么不同？

参考：[MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)。

