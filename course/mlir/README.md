# MLIR 深入专题

这一专题是[第 26 课 · MLIR 深入](../lesson26.md)的分章实践。核心问题是：如何定义一种 IR，如何让它可解析、可验证、可重写，最后如何逐步下降到 LLVM。

```text
Operation / Region / Block / Value
          ↓
Dialect + ODS + TableGen
          ↓
Pattern Rewrite / Dialect Conversion
          ↓
Tensor → MemRef → Loops/CFG → LLVM
```

## 学习顺序

1. [第 1 章：Operation、Region、Block 与 Value](01_ir_core.md)
2. [第 2 章：Dialect、ODS 与 TableGen](02_dialect_ods.md)
3. [第 3 章：Pattern Rewrite 与 Dialect Conversion](03_rewrite_and_conversion.md)
4. [第 4 章：Bufferization、Lowering 与测试](04_bufferize_lowering_tests.md)

## 阅读方法

先用 `mlir-opt` 观察 IR，再进入 `mlir/include/mlir/IR/` 看通用对象，接着挑一个小 Dialect 追 ODS 生成代码，最后沿一个 lowering pass 进入 `DialectConversion` 或 bufferization 基础设施。每章都把“概念—源码—命令—练习”放在一起。

## 官方入口

- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [Defining Dialects](https://mlir.llvm.org/docs/DefiningDialects/)
- [Pass Management](https://mlir.llvm.org/docs/PassManagement/)
- [Pattern Rewriter](https://mlir.llvm.org/docs/PatternRewriter/)
- [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)
- [Bufferization](https://mlir.llvm.org/docs/Bufferization/)

