# 真实 TVM 源码精读专题目录

这套课程围绕 Relax 编译器中的经典 Pass 展开，按照“IR 基础 → 分析清理 → 常量与重写 → Legalize → 融合 → TIR → Codegen”的顺序学习。

## 当前章节

1. [`fuse_ops.cc`：Relax 算子融合源码详解](../fuse_ops.md)
2. [`fold_constant.cc`：Relax 常量折叠源码详解](../fold_constant.md)
3. [经典 Relax Pass 学习路线](../pass_roadmap.md)

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
