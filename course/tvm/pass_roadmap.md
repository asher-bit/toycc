# TVM Relax 源码精读专题：Pass 学习路线

## 课程定位

这是一套面向 TVM Relax 编译器开发者的 Pass 源码阅读课程。课程目标不是背 API，而是建立一条从 Relax IR、分析、重写、算子 lowering、融合到后端代码生成的完整链路。

本专题已经包含：

- [`fuse_ops.cc` 源码详解](fuse_ops.md)
- [`fold_constant.cc` 源码详解](fold_constant.md)
- [专题目录与推荐阅读顺序](README.md)

## 为什么选择这些 Pass

Relax 的 Pass 可以按编译器职责分成几层：

```text
IR 规范化
  ↓
分析和清理
  ↓
常量折叠与表达式重写
  ↓
高层算子分解和 Legalize
  ↓
算子融合
  ↓
TIR 生成与后端 Codegen
  ↓
参数绑定、内存规划和部署
```

学习源码时，最好沿着这条依赖链阅读，而不是随机打开一个优化文件。

## 推荐课程顺序

### 模块一：IR 访问和规范化基础

#### 1. `src/relax/transform/normalize.cc`

学习重点：

- `BlockBuilder` 如何维护当前作用域；
- `Emit` 与 `EmitOutput` 的区别；
- 表达式如何被规范化成 binding；
- DataflowBlock、BindingBlock、SeqExpr 的关系。

这是后面所有 ExprMutator 和 Pass 的基础。

#### 2. `src/relax/transform/canonicalize_bindings.cc`

学习重点：

- 如何简化 binding；
- 如何合并或消除不必要的变量；
- 为什么 canonicalization 会影响后续 pattern matching；
- 变量 identity 和 structural equality 的区别。

#### 3. `src/relax/transform/topological_sort.cc`

学习重点：

- binding 的数据依赖如何确定；
- 为什么重写 IR 后仍必须保持拓扑顺序；
- 多个 dataflow block 中的变量如何重新排序。

### 模块二：清理和局部优化

#### 4. `src/relax/transform/dead_code_elimination.cc`

学习重点：

- 如何判断 binding 是否仍然被使用；
- DataflowVar 和 block 输出的处理差异；
- 为什么 ConstantFolder 最后要调用 `RemoveAllUnused`；
- DCE 与 FuseOps 的前后关系。

#### 5. `src/relax/transform/eliminate_common_subexpr.cc`

学习重点：

- structural equality 如何识别公共子表达式；
- 纯函数属性如何影响 CSE 的安全性；
- 为什么有些表达式不能被合并；
- CSE 对算子融合和常量折叠的影响。

#### 6. `src/relax/transform/remove_unused_parameters.cc`

学习重点：

- 函数参数使用分析；
- 参数删除后调用点如何同步修改；
- 函数签名变化如何在 IRModule 中传播。

### 模块三：常量传播和高层表达式重写

#### 7. `src/relax/transform/fold_constant.cc`

本课程的第 2 章已经详细覆盖。

重点是理解：

- 常量输入检测；
- PrimFunc 构建缓存；
- `CallPacked` 编译期执行；
- 单输出和多输出 Tensor 物化；
- shape 与 tensor 之间的特殊转换。

#### 8. `src/relax/transform/decompose_ops.cc`

学习重点：

- 高层复合算子如何分解为基础 Relax 算子；
- decomposition table 或注册机制如何工作；
- 为什么 `tensor_to_shape` 等操作需要在特定 Pass 前后处理；
- 分解后如何给后续 Legalize 和 FuseOps 提供更细的粒度。

#### 9. `src/relax/transform/compute_prim_value.cc`

学习重点：

- 编译期计算 PrimExpr；
- 形状表达式和符号表达式的求值；
- 常量折叠与动态形状推导的边界。

### 模块四：从 Relax Op 到 TIR

#### 10. `src/relax/transform/legalize_ops.cc`

这是理解 Relax 后端路径最重要的 Pass 之一。

学习重点：

- `FLegalize` 如何把高层 Relax Op 转成 `call_tir`；
- legalization 函数如何使用 BlockBuilder；
- 高层类型、shape、dtype 如何传给 TIR；
- 为什么 FoldConstant 要读取 `FLegalize` 属性表。

建议把它和 `fold_constant.cc` 放在一起对照阅读：

```text
Relax Op
  ↓ LegalizeOps
call_tir
  ↓ FoldConstant（如果输入全是常量）
Constant
```

#### 11. `src/relax/transform/call_tir_rewrite.cc`

学习重点：

- `call_tir` 的参数和输出类型如何被规范化；
- Tuple 输入和 Tuple 输出如何展开或重组；
- 调用约定如何从 Relax 表示变成 TIR 可接受的形式。

#### 12. `src/relax/transform/fuse_ops.cc`

本课程的第 1 章已经详细覆盖。

学习重点是：

- indexed forward graph；
- post-dominator；
- OpPatternKind；
- union-find group；
- FunctionCreator；
- 多输出 fused function；
- DFPattern 驱动的 composite function。

#### 13. `src/relax/transform/fuse_tir.cc`

这是从 Relax fused function 进入 TIR 的关键 Pass。

学习重点：

- Relax binding 如何转换为 PrimFunc；
- 多个 `call_tir` 如何合并为一个 TIR PrimFunc；
- buffer、shape、dtype 和 layout 如何被传递；
- FuseOps 生成的 `kPrimitive` 函数如何被识别；
- TIR schedule 和后续 MetaSchedule 的衔接点。

建议的调用链是：

```text
LegalizeOps
  → FoldConstant
  → FuseOps
  → FuseTIR
  → RunCodegen
```

### 模块五：后端和部署相关 Pass

#### 14. `src/relax/transform/run_codegen.cc`

学习重点：

- `kCodegen` 和 `GlobalSymbol` 如何驱动后端；
- composite function 如何交给外部编译器；
- IRModule 中的 Relax 函数、TIR PrimFunc 和外部模块如何汇合。

#### 15. `src/relax/transform/attach_global_symbol.cc`

学习重点：

- GlobalSymbol 如何生成；
- 函数名如何映射到 runtime 可调用符号；
- codegen wrapper 与实际实现之间的关系。

#### 16. `src/relax/transform/bind_params.cc`

学习重点：

- 模型参数如何绑定到函数；
- 参数绑定后 ConstantFolder、FuseOps 的输入会发生什么变化；
- 为什么参数绑定通常会影响常量折叠收益。

#### 17. `src/relax/transform/bind_symbolic_vars.cc`

学习重点：

- 动态 shape 符号如何绑定；
- 符号替换如何保持类型和形状一致；
- `tir_vars` 与 Relax symbolic variable 的关系。

#### 18. `src/relax/transform/static_plan_block_memory.cc`

学习重点：

- Tensor 生命周期分析；
- workspace 和临时 buffer 规划；
- dataflow block 中的内存复用；
- 算子融合后内存规划为什么会改变。

## 经典 Pass 的优先级

如果时间有限，建议优先阅读下面 8 个：

| 优先级 | Pass | 需要掌握的核心概念 |
|---|---|---|
| 1 | `normalize.cc` | BlockBuilder、binding、SeqExpr |
| 2 | `dead_code_elimination.cc` | use-def、变量生命周期 |
| 3 | `fold_constant.cc` | 编译期执行、常量传播、shape |
| 4 | `legalize_ops.cc` | Relax Op 到 call_tir |
| 5 | `call_tir_rewrite.cc` | 调用约定和类型重写 |
| 6 | `fuse_ops.cc` | 图分区、post-dominator、函数封装 |
| 7 | `fuse_tir.cc` | Relax 到 TIR |
| 8 | `run_codegen.cc` | 后端 codegen 和 runtime 符号 |

如果要继续学习更复杂的优化，再阅读：

- `combine_parallel_matmul.cc`：并行矩阵乘法合并；
- `adjust_matmul_order.cc`：矩阵乘法顺序调整；
- `reorder_take_after_matmul.cc`：针对 matmul 和 take 的重排；
- `reorder_permute_dims_after_concat.cc`：布局和维度变换重排；
- `convert_layout.cc`：布局转换；
- `to_mixed_precision.cc`：混合精度转换；
- `eliminate_common_subexpr.cc`：公共子表达式消除；
- `inline_functions.cc`：Relax 函数内联；
- `expand_tuple_arguments.cc`：Tuple 参数展开；
- `to_non_dataflow.cc`：dataflow 到普通 binding 的转换。

当前 `src/relax/transform` 目录还包含内存规划、布局转换、参数绑定、代码生成和各种硬件特化 Pass。可以在官方源码目录中按上述顺序继续扩展阅读。

## 推荐的源码阅读方法

### 第一步：先看输入输出 IR

不要一开始就逐行看 C++。先写一个最小 Relax 函数，打印 Pass 前后的 IR：

```python
print(mod)
mod = relax.transform.FoldConstant()(mod)
print(mod)
```

先回答：

- 哪个 binding 消失了；
- 哪个表达式变成了 Constant；
- 函数属性有没有变化；
- 新增了哪些 GlobalVar；
- Tuple 和 shape 是否被改写。

### 第二步：找 Pass 的唯一入口

通常先找：

```text
Pass Xxx()
```

再从入口追踪：

```text
Pass factory
  → mutator / visitor
  → helper functions
  → builder update
```

### 第三步：记录状态变量

阅读每个 Mutator 时，重点列出：

- 当前模块是谁；
- 当前作用域是什么；
- 变量绑定保存在哪里；
- 是否有缓存；
- 是否会创建新函数；
- 是否会改变 GlobalVar；
- 是否会改变函数属性。

### 第四步：区分“分析”和“改写”

例如 `FuseOps` 中：

- `GraphCreator` 是分析；
- `GraphPartitioner` 是决策；
- `OperatorFusor` 是改写。

`FoldConstant` 中：

- `MatchConstShape`、`MatchConstArrayArgs` 是分析；
- `GetCachedBuild` 和 `CallPacked` 是编译期执行；
- `VisitExpr_` 是 IR 改写。

### 第五步：最后看 Pass 的注册方式

注册代码决定了：

- Pass 接收 Function 还是 IRModule；
- PassContext 是否参与配置；
- Pass 从 Python 侧如何调用；
- Pass 是否有依赖的前置 Pass。

## 建议的综合实践项目

实现一个最小 Relax 编译流水线：

```text
Normalize
  → LegalizeOps
  → FoldConstant
  → DeadCodeElimination
  → FuseOps
  → FuseTIR
  → RunCodegen
```

然后针对同一个模型分别观察：

1. Legalize 前后的高层算子；
2. FoldConstant 后减少了哪些 binding；
3. DCE 删除了哪些无用变量；
4. FuseOps 创建了哪些 fused 函数；
5. FuseTIR 生成了哪些 PrimFunc；
6. RunCodegen 创建了哪些 runtime symbol。

最终要建立这样的心智模型：

```text
Relax IR 是高层程序表示
分析 Pass 读取它的依赖、类型和属性
重写 Pass 改变它的结构
Legalize Pass 把高层算子降到 call_tir
Fusion Pass 合并计算边界
FuseTIR 把 Relax 计算变成 TIR 实现
Codegen Pass 把实现交给目标后端
```

## 常见错误与归因

| 现象 | 根因 | 定位手段 |
|---|---|---|
| 自写 pass 放进 pipeline 就错 | pass 顺序错了（如 Fusion 前没 AnnotateTIROpPattern） | 对照本文的标准顺序逐位排查 |
| 观察不到 pass 效果 | 没在关键节点 dump IR | 每步之后打印 mod，比较 binding 变化 |
| 混淆分析 pass 与重写 pass | 把"读取事实"当"修改 IR" | 按"输入输出"判断：分析不改图、重写改图 |
| 跑通 pipeline 但结果错 | 中间某 pass 语义不合法 | 逐 pass 差分对比数值结果 |

## 本章检查点

完成以下四项才算通过本章：

1. 默写标准 pipeline 顺序（AnnotateTIROpPattern → FuseOps → FuseTIR → FoldConstant → LegalizeOps），并说出每一环"为什么必须在下一环之前"；
2. 把 toycc 的 `run_passes(g, ("fusion", "layout", "constfold"))` 与本文 pipeline 逐项对照，列出语义对应的三项与 toycc 没有的一项；
3. 在真实 TVM 上跑一遍本文的观察清单（legalize 前后算子、FoldConstant 减 binding、DCE 删变量），记录每步的 IR 行数变化；
4. 用一句话回答"为什么读 pass 要先读 pipeline 而不是先读单个 pass 源码"。
