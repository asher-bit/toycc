# MLIR 第 3 章：Pattern Rewrite 与 Dialect Conversion

## 1. 本章目标

- 写出一个基本的 `RewritePattern`；
- 区分 canonicalization、folding 和 lowering；
- 理解 `ConversionTarget`、`TypeConverter` 与合法性；
- 追踪一次 operation 替换如何维护 SSA use-def 链。

## 2. Pattern Rewrite 的基本形状

```cpp
struct AddZeroPattern : mlir::OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      arith::AddIOp op, mlir::PatternRewriter &rewriter) const override {
    auto cst = op.getRhs().getDefiningOp<arith::ConstantIntOp>();
    if (!cst || cst.value() != 0)
      return mlir::failure();

    rewriter.replaceOp(op, op.getLhs());
    return mlir::success();
  }
};
```

匹配成功并不代表可以随意改写。需要确认常量类型、整数语义属性和 operation 的副作用模型。`replaceOp` 会把旧结果的 uses 重定向到新值，并从 IR 中移除旧 operation；复杂场景要使用 `rewriter` 的通知接口，不能直接操作底层链表绕过它。

## 3. Greedy Rewrite、Canonicalization 与 Folding

- Folding 通常是 operation 自己根据已知常量计算结果；
- Canonicalization 是跨 operation 的局部规范化，包括 pattern、folding、删除无用结构等；
- Lowering 是把一种语义转换成另一种更低层语义，通常需要明确的合法性和类型转换。

一个好的 canonicalization pattern 应该简单、终止、不会在两个形态间来回震荡，并且不要偷偷改变用户可观察的语义。

## 4. Dialect Conversion 的四个核心对象

```text
ConversionTarget  → 哪些 op 合法、哪些必须消失
RewritePattern    → 某个 op 如何转换
TypeConverter     → 类型如何从源类型映射到目标类型
Conversion driver  → 调度转换、处理残留 op 和 materialization
```

示意代码：

```cpp
struct ToyAddLowering : mlir::OpConversionPattern<toy::AddOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      toy::AddOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::AddFOp>(
        op, adaptor.getLhs(), adaptor.getRhs());
    return mlir::success();
  }
};

mlir::ConversionTarget target(getContext());
target.addLegalDialect<mlir::arith::ArithDialect>();
target.addIllegalOp<toy::AddOp>();
```

`ConversionTarget` 是“完成”的定义：如果一个 illegal op 留在结果中，full conversion 应失败。partial conversion 则允许一部分源 Dialect 保留，适合分阶段 lowering。

## 5. 类型转换和 materialization

当源 operation 的结果类型与目标 operation 不一致时，不能只替换 operation 名称。`TypeConverter` 负责描述类型映射；如果边界上需要临时转换值，框架可能要求 materialization。张量到 memref、索引到整数、指针封装等场景都可能遇到这个问题。

## 6. 源码阅读地图

- `mlir/include/mlir/IR/PatternMatch.h`：PatternRewriter 基础接口；
- `mlir/lib/IR/PatternMatch.cpp`：替换、删除、通知等实现；
- `mlir/include/mlir/Transforms/DialectConversion.h`：转换 API；
- `mlir/lib/Transforms/Utils/DialectConversion.cpp`：转换 driver 的核心实现；
- `mlir/lib/Transforms/Canonicalizer.cpp`：canonicalization pass；
- `mlir/lib/Dialect/` 下各 Dialect 的 `*Patterns.cpp`：真实 pattern 集合。

追源码时按“谁声明 pattern → 谁把 pattern 加入 RewritePatternSet → 哪个 pass 启动 driver → 哪些 op 被标记 legal/illegal”的顺序走。

## 7. 练习

1. 把 `arith.addi x, 0` 的 pattern 扩展到常量在左侧；
2. 写一个 `toy.constant` 到 `arith.constant` 的 conversion；
3. 故意不把源 op 标记为 illegal，观察 conversion 为什么可能“成功但没有完成 lowering”；
4. 给一个类型变化的 conversion 增加 `TypeConverter`，记录每个边界值的类型。

参考：[Pattern Rewriter](https://mlir.llvm.org/docs/PatternRewriter/)、[Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)。

