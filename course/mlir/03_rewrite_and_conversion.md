# MLIR 第 3 章：Pattern Rewrite 与 Dialect Conversion——改写如何保持正确

## 1. 本章目标

- 能写出一个基本的 `RewritePattern`，并说清 `matchAndRewrite` 两步各自干什么；
- 能区分 folding、canonicalization、lowering 三种改写的边界；
- 能说清 `ConversionTarget`、`TypeConverter`、`ConversionPattern`、conversion driver 四个对象的分工；
- 能手算一次 `toy.add → arith.addf` 的转换过程，包括边界值的类型处理；
- 能解释为什么"匹配成功"不等于"可以随便改"，以及 pattern 必须终止的原因。

前置：第 1 章的对象模型、第 2 章的 ODS（本章的 pattern 就写在 op 定义旁边）。工具：MLIR。

## 2. 工作中的问题长什么样

改写方向的三个典型翻车：

```text
"我的 pattern 匹配了，也 replaceOp 了，为什么程序行为变了？"
"conversion 跑完没报错，但源方言的 op 还留在 IR 里？"
"两个 pattern 互相触发，死循环了，怎么破？"
```

三个问题对应：**改写的前提条件**（匹配成功 ≠ 改写合法）、**合法性定义**（ConversionTarget 才是"完成"的标准）、**终止性**（pattern 集合必须收敛）。本章建立这三个约束。

## 3. Pattern Rewrite 的基本形状：逐行拆解

一个"`addi x, 0` → `x`"的 pattern：

```cpp
struct AddZeroPattern : mlir::OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      arith::AddIOp op, mlir::PatternRewriter &rewriter) const override {
    auto cst = op.getRhs().getDefiningOp<arith::ConstantIntOp>();  // ① 看右操作数是不是常量
    if (!cst || cst.value() != 0)
      return mlir::failure();                                      // ② 不是"加 0", 拒绝

    rewriter.replaceOp(op, op.getLhs());                           // ③ 用左操作数替换整个 add
    return mlir::success();                                        // ④ 报告成功
  }
};
```

逐行对象：`OpRewritePattern<arith::AddIOp>` 声明"这个 pattern 只匹配 `arith.addi`"；`matchAndRewrite` 两步合一——先 match（判断形状是否符合：右操作数是值为 0 的常量）再 rewrite（替换）；`return failure()` 表示"不匹配，换下一个 pattern"，**failure 不是错误**，只是"这个 pattern 不接手"；`replaceOp(op, op.getLhs())` 做两件事：把旧结果的所有 uses 重定向到新值（use-def 链维护，等价于 LLVM 的 `replaceAllUsesWith`），再把旧 op 从 IR 里摘除。

**匹配成功 ≠ 可以随便改**。改写前要确认三件事：类型是否一致（`addi` 的操作数与结果同类型才安全）；语义属性（`addi nsw` 的溢出标志在 `replaceOp` 时怎么办——丢了标志可能让下游优化做出错误假设）；副作用模型（有副作用的 op 不能当没副作用地删）。复杂场景必须用 `rewriter` 的通知接口（`notifyOperationRemoved` 等），**不要直接操作底层链表绕过 rewriter**——绕过去，分析缓存与 listener 就收不到更新。

## 4. Greedy Rewrite 与终止性：为什么 pattern 必须收敛

**Greedy rewriter** 是驱动 pattern 的默认引擎：反复扫描 IR，把所有 pattern 应用到不再变化（fixpoint）。它的一个硬前提：**pattern 集合必须终止**。反例——两个方向相反的 pattern：

```text
pattern A: mul x, 2  →  add x, x        (把乘 2 写成加法)
pattern B: add x, x  →  mul x, 2        (把双加写成乘法)
```

A 触发 B、B 触发 A，driver 永远到不了 fixpoint。所以 canonicalization pattern 的铁律：**简单、终止、不在两个形态间震荡、不改变可观察语义**——每个 pattern 都要回答"我输出的是否比输入更'规范'，且不会再被别的 pattern 变回来"。

## 5. 三种改写：Folding / Canonicalization / Lowering

| 种类 | 一句话定义 | 例子 |
|---|---|---|
| Folding | op **自己**根据已知常量算出结果，替换为常量 | `arith.addi %c1, %c2` → `%c3`（常量折叠，toycc 的 constfold pass 同思想） |
| Canonicalization | 跨 op 的**局部规范化**（同层语义内变"标准形"） | `addi x, 0` → `x`；删无用 op |
| Lowering | 把语义**转换到另一层**（更低层/另一方言） | `toy.add` → `arith.addf`；`tensor` → `memref` |

三者的分界：folding 是"算出来"，canonicalization 是"同层变标准形"，lowering 是"跨层换方言"。canonicalization 不进 lowering——它只做局部的、不改变抽象层次的清理。

## 6. Dialect Conversion：四个对象的分工

```text
ConversionTarget  → 转换后什么 op 合法、什么 op 必须消失(合法性的定义)
ConversionPattern → 某个源 op 如何转换(规则)
TypeConverter     → 类型如何从源类型映射到目标类型
conversion driver → 调度规则、处理残留 op、在边界插入 materialization
```

最小示例（toy.add → arith.addf，两个方言都表达加法）：

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
target.addLegalDialect<mlir::arith::ArithDialect>();   // arith 是合法终点
target.addIllegalOp<toy::AddOp>();                      // toy.add 必须消失
```

关键对象语义：**`ConversionTarget` 就是"完成"的定义**——full conversion 结束时，任何 illegal op 残留 = 转换失败；**`OpAdaptor`** 是"操作数已经转换过"的视图（pattern 不用自己先递归转换操作数，driver 已经按 TypeConverter 换好了）。**partial conversion 允许一部分源方言保留**，适合分阶段 lowering（先降一批，再降一批）。

## 7. 类型转换与 materialization：为什么不能只换名字

当源 op 与目标 op 的**类型不同**时（`tensor<4xf32>` → `memref<4xf32>`），替换不是换个 op 名那么简单。`TypeConverter` 描述类型映射；两个已转换 op 之间如果插着类型不匹配的边界值，driver 会要求一个 **materialization**（临时转换值的生成回调）把边界值接上。手算一条链：

```text
toy.constant : i32          (源类型 i32, 目标也是 i32 → 直接映射)
toy.add : (i32, i32) → i32  (→ arith.addi, 类型一致)
tensor<4xf32> → memref<...> (类型不同 → 边界处必须 materialize: 分配 memref + 拷贝)
```

判断规则：**类型没变 → 换名即可；类型变了 → TypeConverter 映射 + 边界 materialization**。张量到 memref、index 到整数、指针封装都是"类型变了"的典型场景。

## 8. 源码阅读地图

- `mlir/include/mlir/IR/PatternMatch.h`：PatternRewriter 基础接口；
- `mlir/lib/IR/PatternMatch.cpp`：替换、删除、通知的实现；
- `mlir/include/mlir/Transforms/DialectConversion.h`：conversion API；
- `mlir/lib/Transforms/Utils/DialectConversion.cpp`：conversion driver 核心（materialization 调度在这里）；
- `mlir/lib/Transforms/Canonicalizer.cpp`：canonicalization pass；
- `mlir/lib/Dialect/` 下各方言的 `*Patterns.cpp`：真实 pattern 集合。

追踪顺序：**谁声明 pattern → 谁把 pattern 加进 RewritePatternSet → 哪个 pass 启动 driver → 哪些 op 被标记 legal/illegal**。

## 9. 常见错误与归因

| 现象 | 根因 | 修正 |
|---|---|---|
| 改写后行为变了 | 匹配成功但没检查类型/语义属性/副作用 | 按第 3 节三问补检查 |
| conversion "成功"但源 op 还在 | 用的是 partial conversion，或忘了 `addIllegalOp` | 明确 full/partial 语义 + 检查 Target 声明 |
| pattern 死循环 | 两个 pattern 互相触发 | 检查终止性（第 4 节铁律） |
| 边界值类型错 | 类型变了但没配 TypeConverter/materialization | 第 7 节判断规则 |
| 改了 IR 但分析结果过期 | 绕过 rewriter 直接操作链表 | 全部经由 rewriter 接口 |

## 10. 本章检查点

完成以下四项才算通过本章：

1. 把 `addi x, 0` 的 pattern 扩展到"常量在左侧"（`addi 0, x`），并写出两个测试用例（一个触发、一个不触发）；
2. 手算 `toy.constant → arith.constant` 的 conversion：写出 ConversionTarget 的 legal/illegal 声明，并说明 constant 的属性（APIntAttr）在转换中怎么搬；
3. 故意不把 `toy.add` 标记为 illegal，跑 conversion，观察并解释"成功但没有完成 lowering"的现象；
4. 给一个类型变化的转换（如 `toy.reshape : tensor → tensor`）写出 TypeConverter 的映射表，标出边界上需要 materialization 的位置。

## 11. 本章小结与下一步

本章建立了"改写必须守约束"的模型：匹配只是入场券，合法性由 ConversionTarget 定义，终止性由 pattern 集合自己保证。下一章（MLIR 04：Bufferization、Lowering 与测试）回答 MLIR 最著名的痛点——"值语义的张量如何落到内存"，以及这些改写如何被测试钉住。

**导航**：⬅ [上一章](02_dialect_ods.md)（Dialect、ODS 与 TableGen）　｜　[下一章](04_bufferize_lowering_tests.md)（Bufferization、Lowering 与测试）➡
