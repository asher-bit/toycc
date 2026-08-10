# TVM Relax Pass 源码课程目录

这套课程围绕 Relax 编译器中的经典 Pass 展开，按照“IR 基础 → 分析清理 → 常量与重写 → Legalize → 融合 → TIR → Codegen”的顺序学习。

## 当前章节

1. [第 27 课：FuseOps——Relax 算子融合源码详解](lesson27.md)
2. [第 28 课：FoldConstant——Relax 常量折叠源码详解](lesson28.md)
3. [第 29 课：Relax Pass 学习路线与经典 Pass 课程](lesson29.md)

## 推荐阅读顺序

```text
normalize.cc
  → canonicalize_bindings.cc
  → dead_code_elimination.cc
  → fold_constant.cc
  → decompose_ops.cc
  → legalize_ops.cc
  → call_tir_rewrite.cc
  → fuse_ops.cc
  → fuse_tir.cc
  → run_codegen.cc
```

## 课程总目标

能够从一个 Relax Function 出发，解释它如何经过：

```text
高层 Relax IR
  → 规范化
  → 类型和依赖分析
  → 常量折叠
  → 算子 Legalize
  → 算子融合
  → TIR 生成
  → 后端 Codegen
```

并能够独立阅读、调试和修改 TVM Relax Transform Pass。
