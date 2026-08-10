# TVM Relax 源码精读专题：`fuse_ops.cc`

## 专题说明

> 本章对应 Apache TVM `main` 分支中的 `src/relax/transform/fuse_ops.cc`。
> 源码链接：[fuse_ops.cc](https://github.com/apache/tvm/blob/main/src/relax/transform/fuse_ops.cc)
>
> 说明：`main` 分支会持续变化，本文中的行号以撰写时版本为准。

## 1. 本章学习目标

读完本章后，应当能够理解：

- Relax dataflow block 是如何被转换成临时数据流图的；
- `OpPatternKind` 如何影响算子融合；
- post-dominator 为什么可以解决 diamond-shaped 分支融合问题；
- union-find 如何管理融合 group；
- 多个 binding 如何被封装成新的 Relax 函数；
- fused 函数的参数、输出、Tuple 和动态形状如何处理；
- `FuseOps` 和 `FuseOpsByPattern` 的区别；
- `kPrimitive`、`kComposite` 和 `kCodegen` 属性分别起什么作用。

## 2. 先理解整个执行流程

这份文件包含两套融合入口。

普通算子融合的流程是：

```text
IRModule
  ↓
GraphCreator：把 Relax 函数转成数据流图
  ↓
GraphPartitioner：根据算子模式和后置支配关系进行分组
  ↓
OperatorFusor：生成 fused 函数并改写原函数
```

基于模式的融合流程是：

```text
FusionPattern
  ↓
PatternBasedPartitioner：匹配 DFPattern
  ↓
OperatorFusor：生成 composite 函数
  ↓
CompositeFunctionAnnotator：可选地添加 Codegen 外层函数
```

假设原始 Relax IR 是：

```text
lv1 = conv2d(x, weight)
lv2 = relu(lv1)
return lv2
```

融合后大致变成：

```text
fused_conv2d_relu(x, weight):
    lv1 = conv2d(x, weight)
    lv2 = relu(lv1)
    return lv2

main(x, weight):
    return fused_conv2d_relu(x, weight)
```

这份文件的核心思想可以概括为：

```text
GraphCreator 负责“看懂图”
GraphPartitioner 负责“决定怎么分组”
FunctionCreator 负责“生成新函数”
OperatorFusor 负责“改写原函数”
CompositeFunctionAnnotator 负责“接入后端代码生成”
```

## 3. 文件头和依赖：第 1～57 行

第 1～17 行是 Apache 许可证声明，没有运行逻辑。

第 18～26 行的文件注释说明了设计目标：

- 处理 Relax 函数中的 dataflow block；
- 把多个 binding 分成若干组；
- 为每一组创建新的 Relax 函数；
- 用对新函数的调用替换原来的 binding；
- 后续由 `FuseTIR` 把这些 Relax 函数转成 TIR 函数。

第 27～45 行是头文件引用：

- `tvm/ffi/cast.h`：提供 FFI 类型转换；
- `reflection/registry.h`：注册 Python/FFI 可调用对象；
- `relax/analysis.h`：Relax 分析工具；
- `dataflow_matcher.h`、`dataflow_pattern.h`：数据流模式匹配；
- `expr_functor.h`：表达式访问器和修改器；
- `transform.h`：Pass 定义；
- `type.h`：Relax 类型；
- `utils.h`：Relax 工具函数；
- `logging.h`：日志和警告；
- `tirx/analysis.h`：TIR 表达式分析；
- `tirx/expr_functor.h`：TIR 表达式遍历；
- `tirx/function.h`：TIR PrimFunc 类型；
- `optional`：处理可选结果；
- `arena.h`：Arena 内存分配器；
- `graph_partitioner.h`：后置支配分析和图分区；
- 当前目录的 `utils.h`：当前 transform 模块内部工具。

第 47～48 行进入 `tvm::relax` 命名空间，后面的实现都位于这个命名空间中。

第 50～52 行定义 `ExprIdentityLess`。它比较的是两个表达式对象的地址，而不是表达式的数学内容。这样 `Expr` 就可以作为 `std::set` 的 key，并且同一个 AST 对象具有稳定的排序。

第 54～57 行是 FFI 静态初始化代码：

- 注册 `FusionPatternNode` 的反射信息；
- 注册 `PatternCheckContextNode` 的反射信息；
- 使这些对象可以通过 TVM FFI 从 Python 或其他语言访问。

## 4. 融合算法说明：第 59～93 行

这里的注释描述了整个自动融合算法。

考虑下面的 diamond-shaped 数据流：

```text
       conv2d
      /   |   \
    op    op    op
      \   |   /
      elemwise_add
```

`conv2d` 后面有多个分支，最终在 `elemwise_add` 汇合。

问题是：当遍历到 `conv2d` 时，程序不一定已经知道未来的所有路径都会在 `elemwise_add` 汇合。因此代码使用 post-dominator，也就是“后置支配节点”。

如果从节点 A 出发的所有后续路径都必须经过节点 B，那么 B 就是 A 的后置支配节点。上图中，`elemwise_add` 是 `conv2d` 的后置支配节点。

源码列出的算法步骤是：

1. 构造数据流 DAG；
2. 构造 post-dominator tree；
3. 根据 post-dominator 信息执行融合。

`GraphPartitioner` 中的 `Group` 使用 union-find 管理融合组，同时保存父节点、算子模式、组内节点数量、参数数量和附加属性。详细定义见 [`graph_partitioner.h`](https://raw.githubusercontent.com/apache/tvm/main/src/relax/analysis/graph_partitioner.h)。

第 95 行将 `support::LinkNode` 引入当前命名空间，用于创建图中的边。

第 97 行定义默认最大融合算子数：

```cpp
constexpr uint32_t kMaxFusedOps = 256;
```

第 98 行注册 PassContext 配置项：

```text
relax.FuseOps.max_depth
```

它允许调用者修改单个 fused 函数允许包含的最大算子数量。

## 5. `GraphCreator`：第 99～351 行

`GraphCreator` 的职责是：

> 遍历 Relax IRModule，把每个需要参与融合的表达式转换为 `IndexedForwardGraph`。

它继承 `ExprVisitor`，所以本身是一个只读表达式访问器。

### 5.1 `Create`：第 108～132 行

`Create` 是静态入口函数。它接收输入 `IRModule` 和 Arena，创建一个临时的 `GraphCreator`，然后遍历模块中的全局函数。

具有以下属性的函数会被跳过：

- 不是 Relax Function 的对象；
- 已经带有 `attr::kPrimitive` 的函数；
- 已经带有 `attr::kCodegen` 的函数。

这样可以避免之前生成的 fused 函数被再次普通融合。

对符合条件的 Relax Function，代码调用访问器开始遍历函数体。

遍历结束后，代码检查三件事情的数量必须一致：

- `node_map` 中的节点数量；
- `post_dfs_order` 中的节点数量；
- 已初始化 pattern 的节点数量。

这几个检查用于保证每个图节点都被正确创建、登记和初始化。

### 5.2 构造函数：第 135～136 行

构造函数保存：

- 当前 IRModule；
- Arena 分配器。

这里使用 `std::move`，避免不必要的对象复制。

### 5.3 访问 Relax 函数：第 137～153 行

访问一个 Relax Function 时，代码先处理函数参数：

- 每个参数建立一个图节点；
- 参数标记为外部引用；
- 参数 pattern 设置为 `kOpaque`；
- 参数加入 post-DFS 顺序。

参数来自函数外部，不是当前函数内部生成的中间结果，因此不能被融合为中间算子。

如果函数有 `kNumInput` 属性，代码还会记录从 `kNumInput` 开始的参数。这些参数通常与 packed model parameter 有关，后面处理 `TupleGetItem` 时会使用。

最后调用基类 `ExprVisitor::VisitExpr_`，继续访问函数体、binding block 和具体表达式。

### 5.4 访问 MatchCast：第 154～158 行

`MatchCast` 用于给变量附加类型或形状约束。这里为它创建一个节点，pattern 设为 `kOpaque`，并放入 post-DFS 顺序。

当前融合算法不会把 MatchCast 当成普通可融合算子。

### 5.5 访问 VarBinding：第 160～179 行

对于：

```text
lv = value
```

代码先为左值变量建立图节点。

如果左值不是 `DataflowVar`，它通常是 dataflow block 的输出变量，因此被标记为 `extern_ref`。这意味着它被 block 外部使用，不能简单地作为内部中间节点处理。

右值根据类型分派：

- `CallNode` 调用 `VisitCall`；
- `TupleGetItemNode` 调用 `VisitTupleGetItem`；
- 其他表达式调用 `VisitUnsupportedNode`。

最后把左值节点加入 post-DFS 顺序。

### 5.6 访问 Call：第 182～217 行

这个函数处理：

```text
lv = call(...)
```

代码特别识别两个算子：

- `relax.call_tir`；
- `relax.call_tir_inplace`。

普通情况下，当前算子的 pattern 默认为 `kOpaque`，表示不允许融合。

对于 `call_tir`，代码从 Call 的第一个参数拿到 TIR PrimFunc 的 GlobalVar，再从 IRModule 中查找 PrimFunc。

然后从 PrimFunc 的 `op_pattern` 属性读取算子模式。如果没有该属性，则使用 `kOpaque`。

典型模式包括：

- elementwise；
- broadcast；
- injective；
- reduction；
- out-elementwise-fusable；
- opaque。

之后代码遍历所有 Call 参数，要求它们必须是叶子表达式或 Tuple。对于每个参数，调用 `VisitLeaf` 建立数据流边。

### 5.7 访问 TupleGetItem：第 218～230 行

`TupleGetItem` 默认被视为 `kInjective`，因为它只是从 Tuple 中取一个元素。

但如果 Tuple 是 packed model parameter，代码会把它标记为 `kOpaque`。这样可以避免大量参数提取操作参与融合，从而改变融合结果的参数顺序。

之后把 Tuple 表达式作为输入叶子访问。

### 5.8 访问不支持融合的表达式：第 231～241 行

遇到不支持的表达式时，当前节点被设置为 `kOpaque`。

代码不会把整个复杂表达式当成一个普通融合算子，而是通过 `PostOrderVisit` 找到其中的变量和常量，并为这些叶子建立依赖关系。

这样即使当前表达式不能参与融合，图中仍然可以保留它的输入依赖。

### 5.9 访问叶子表达式：第 244～272 行

叶子表达式包括：

- Tuple；
- Var；
- Constant；
- ShapeExpr；
- PrimExpr；
- StringImm；
- DataTypeImm。

Tuple 会递归访问每个字段。

GlobalVar、ExternFunc 和 OpNode 只是符号或引用，不作为真正的数据流值节点，因此跳过。

如果叶子已经在 `node_map` 中，就复用原节点；否则创建新节点。

常量的 pattern 设置为 `kOpaque`，因为当前实现不融合常量。

最后调用 `AddEdge`，建立：

```text
输入叶子 → 当前绑定变量
```

的有向边。

### 5.10 图辅助函数：第 280～339 行

`CreateNode`：

- 检查同一个 AST 对象不能重复创建节点；
- 用 Arena 分配节点；
- 将对象地址映射到节点。

`AddToPostDFSOrder`：

- 确认节点已经创建；
- 确认节点还没有加入顺序表；
- 设置节点引用和 index；
- 追加到 `post_dfs_order`。

`AddEdge`：

- 用 Arena 创建边；
- 设置边的终点和 pattern；
- 将边放进起点节点的 outputs 链表。

`MarkAsExternRef` 把节点标记为外部引用。

`SetNodePattern` 确保一个节点的 pattern 只设置一次，然后保存 pattern。

成员变量包括：

- `mod_`：当前 IRModule；
- `arena_`：Arena；
- `graph_`：正在构造的图；
- `initialized_nodes_`：已经设置 pattern 的节点；
- `input_params_`：packed 参数中的输入变量。

## 6. `FunctionCreator`：第 363～659 行

`FunctionCreator` 的职责是：

> 根据一个融合组中的 binding，创建新的 Relax 函数。

它继承 `ExprMutator`，因为它需要把原变量替换成新的函数参数和内部变量。

### 6.1 构造函数：第 365～366 行

构造函数接收：

- `lift_constant`：是否将常量提升成函数参数；
- `outer_bindings`：外部函数中的变量到实际值的映射。

如果外部变量最终绑定的是静态常量，则可以直接内联到 fused 函数中。

### 6.2 添加 binding：第 377～437 行

`AppendBinding` 把 binding 加入当前融合函数，并同时更新：

- 函数名字；
- 参数列表；
- 输出列表；
- 已定义变量集合。

对于 `call_tir` 和 `call_tir_inplace`，代码从第二个 Call 参数中取出输入 Tuple，逐个检查输入是否已经在当前函数内部定义。如果没有，就成为新函数参数。

对于普通 Call，代码根据被调用的 Op 或 GlobalVar 构造函数名，并递归检查每个调用参数。

如果参数是 Tuple，会展开 Tuple 字段，以便每个字段都能成为独立参数。

对于 TupleGetItem，代码记录被访问的 Tuple index。后面如果发现 Tuple 只是部分使用，就会将 Tuple 参数拆成字段参数。

每个变量定义完成后会被加入 `defined_vars_`。如果左值不是 `DataflowVar`，就会被视为融合函数输出。

### 6.3 记录输出：第 439～443 行

`AppendOutput` 确保一个变量只加入一次输出列表。

输出的顺序很重要，因为多输出 fused 函数最终会按照这个顺序返回 Tuple。

### 6.4 创建函数：第 453～541 行

这是 `FunctionCreator` 的核心。

第一步，开始一个新的 dataflow block。

第二步，处理部分使用的 Tuple 参数。

例如外部有：

```text
params = (w0, w1, w2)
x = params[1]
```

新函数可以不接收完整 `params`，而是直接接收 `params_1`。

代码会：

- 找到 Tuple 参数的位置；
- 找到被访问的 index；
- 创建对应类型的新字段参数；
- 建立旧 TupleGetItem 到新参数的映射；
- 用字段参数替换完整 Tuple 参数。

第三步，遍历所有 binding。

如果是被拆分 Tuple 对应的 TupleGetItem，就直接用新字段参数替换它。

如果是输出 binding，就用 `EmitOutput` 发射，并保存原变量到新变量的映射。

如果是内部 binding，就通过 `VisitBinding` 放入新 dataflow block。

第四步，结束 dataflow block。

如果没有输出，代码认为当前融合组是死代码，打印警告并放弃创建函数。

如果只有一个输出，函数体直接返回该表达式；如果有多个输出，函数体返回一个 Tuple。

之后调用 `Normalize` 规范化表达式和 `SeqExpr`，并给函数加上：

```text
attr::kPrimitive = true
```

这个属性告诉后续 Pass：该函数已经是融合结果，不要再次进行普通 Relax 算子融合。

代码还会通过 `FreeSymbolicVars` 查找动态形状中出现的自由符号。如果存在，就增加一个 `tir_vars` Shape 参数和对应的 ShapeExpr 实参。

最后通过 `SymbolicVarRenewMutator::Renew` 更新符号，避免符号变量发生作用域冲突。

### 6.5 查找输出：第 553～559 行

`GetOutputIndex` 在线性输出数组中查找变量：

- 找到就返回 index；
- 找不到就返回 `std::nullopt`。

### 6.6 检查定义并创建参数：第 565～607 行

`CheckDefAndUpdateParam` 首先检查表达式是否已经作为函数参数出现，避免重复添加。

如果表达式是一个未定义变量，代码沿着 `outer_bindings_` 继续追踪它的绑定值。

`visited` 集合用于防止变量绑定形成环。

如果最终绑定值是可以静态内联的常量，就记录到 `inlined_bindings_`，稍后访问该变量时直接替换为常量。

如果表达式没有在当前函数内部定义，并且不能被内联，就创建新的参数：

- `arguments_` 保存调用方传入的表达式；
- `params_` 保存新 fused 函数的参数变量。

这两个数组按位置对应。

### 6.7 表达式替换：第 608～621 行

`VisitExpr` 的替换顺序是：

1. 如果表达式是函数参数，用对应的新参数替换；
2. 如果变量可以内联，用静态值替换；
3. 否则递归访问表达式。

### 6.8 可内联常量判断：第 623～637 行

规则如下：

- Tuple：所有字段都必须可内联；
- Var：不可内联；
- Call：不可内联；
- PrimExpr：不能包含未定义符号；
- ShapeExpr：每个维度都不能包含未定义符号；
- 其他类型：不可内联。

成员变量主要包括：

- `defined_vars_`：已经在当前融合函数中定义的变量；
- `inlined_bindings_`：可以被常量替换的变量；
- `output_vars_`：函数输出；
- `outer_bindings_`：外部变量绑定；
- `lift_constant_`：是否提升常量；
- Tuple 参数位置和部分使用信息。

## 7. `OperatorFusor`：第 676～992 行

`OperatorFusor` 把图分区结果真正应用到 Relax IR 上。

它主要做三件事：

1. 收集每个 group 的 binding；
2. 创建每个 group 的 fused 函数；
3. 用 fused 函数调用替换原 binding。

### 7.1 构造函数：第 678～698 行

`GroupMap` 是：

```text
AST 对象地址 → Group*
```

构造函数保存输入模块、group 映射和常量提升选项。

另一个构造函数接收图和分区结果，并通过 `CreateGroupMap` 转成对象到 group 的映射。

### 7.2 `Transform`：第 703～724 行

如果没有指定入口函数，就处理模块中所有 GlobalVar；否则只处理指定函数。

只访问：

- Relax Function；
- 没有 `kPrimitive`；
- 没有 `kCodegen`。

处理每个函数前，代码调用 `AnalyzeVar2Value` 建立外部变量绑定关系。然后访问并重写函数，再通过 `UpdateFunction` 写回 IRModule。

### 7.3 `CreateGroupMap`：第 727～736 行

图的 `post_dfs_order` 与分区结果按 index 对齐。

代码为每个 AST 对象建立：

```text
对象 → 它所属的 union-find 根 group
```

后续遇到一个 Relax Var 时，就可以查询它属于哪个融合组。

### 7.4 重写 DataflowBlock：第 738～846 行

进入一个 dataflow block 后，执行五个阶段。

第一阶段，收集每个融合组的 binding。

第二阶段，收集每个 group 的边界，也就是哪些变量要作为 group 输出。

第三阶段，为每个 group 创建新的 fused 函数。

第四阶段，开始生成新的 binding block。

如果某个 group 只有一个节点并且没有特殊属性，直接正常重写 binding，不创建新函数。

对于多节点 group，只在该 group 最后一个原始 binding 的位置发射一次 fused call。该 group 前面的 binding 已经被移动到新函数内部，因此会被跳过。

如果一个融合函数有多个输出，fused call 返回 Tuple。代码会为每个输出创建对应的 `TupleGetItem`，并把旧变量映射到新表达式。

如果输出变量是 DataflowVar，就使用普通 `Emit`；如果输出变量需要离开 dataflow block，就使用 `EmitOutput`。

### 7.5 `CollectFuncBindings`：第 854～865 行

遍历 block 中的 binding：

- 单节点且无属性的 group 直接跳过；
- 多节点 group 创建 `FunctionCreator`；
- 将 binding 追加到对应的 creator。

### 7.6 `CollectFuncBoundary`：第 867～902 行

该函数分析 group 之间的依赖关系。

例如：

```text
group_A:
    a = conv2d(x)

group_B:
    b = relu(a)
```

`a` 是 group_A 的边界输出，也是 group_B 的输入。

如果当前 binding 使用了来自其他 group 的变量，代码会：

- 添加当前 group 到生产者 group 的依赖；
- 检查是否出现循环依赖；
- 如果生产者 group 会生成 fused 函数，则将该变量加入生产者函数的输出列表。

### 7.7 获取 group 和更新参数：第 909～937 行

`GetGroupFromBinding` 从 binding 的左值变量获取 group。

`GetGroupFromVar` 在 `obj2group_` 中查找变量并返回 union-find 根 group。

`UpdateArgs` 遍历保存的调用参数，并通过 `VisitExpr` 应用变量重映射。

### 7.8 group 拓扑排序：第 941～972 行

代码首先按照 group 分类 binding，同时保留原始出现顺序。

之后使用 DFS：

- 先访问依赖 group；
- 再访问当前 group；
- 用 `visited` 防止重复访问。

最终得到满足 group 依赖关系的 binding 顺序，保证生产者先于消费者发射。

## 8. 普通 `FuseOps` 入口：第 993～1010 行

`FuseOps` 的执行步骤是：

1. 创建 Arena；
2. 使用 `GraphCreator` 构造 indexed forward graph；
3. 使用 `GraphPartitioner` 根据优化级别、最大融合深度和算子 pattern 进行分区；
4. 使用 `OperatorFusor` 把分区结果应用到 IRModule。

这里传入 `max_function_args = 0`，表示普通 `FuseOps` 不限制 fused 函数参数数量。

`MakeGroupedFunctions` 是辅助入口。它不负责重新执行图分区，而是直接接收外部提供的 partition map，然后调用 `OperatorFusor`。

## 9. `PatternBasedPartitioner`：第 1016～1180 行

这个类负责基于 `DFPattern` 的融合。

它保存：

- pattern 名称；
- 主 DFPattern；
- 注解 pattern；
- 可选的 check 函数；
- 可选的 attrs getter；
- 表达式到变量的映射；
- 变量到 group 的映射。

### 9.1 访问 DataflowBlock

进入 dataflow block 时建立 `current_block_use_def_`，记录变量的使用和定义关系；离开 block 后清空。

这部分信息会传给 `PatternCheckContext`，供用户自定义检查函数使用。

### 9.2 初始化变量 group

每遇到一个变量，就创建一个初始 group，并把变量放进去。

初始状态下，每个变量都是独立 group。

### 9.3 匹配 Call

代码使用 `ExtractMatchedExpr` 将当前 Call 与主 pattern 匹配。

匹配成功后：

1. 创建 `PatternCheckContext`；
2. 如果存在用户 check 函数，执行检查；
3. 检查当前匹配是否与以前的匹配重叠；
4. 如果合法，设置 `kComposite` 属性；
5. 读取并设置自定义属性；
6. 把匹配到的相关变量合并到同一个 group。

如果已有的匹配子图不能被当前匹配完全包含，当前匹配会被丢弃，避免生成错误的融合子图。

### 9.4 处理 wildcard

不能无条件地把 wildcard 匹配到的变量合并进当前 group。

例如两个连续的 conv2d：

```text
conv2d_1 → conv2d_2
```

第二个 pattern 的 wildcard 可能匹配到第一个算子的输出，但这不代表第一个算子也应该被融合进第二个 group。

因此代码只对特定的 CallPattern 和 TupleGetItemPattern 执行 group 合并。

### 9.5 `PatternCheckContext`

上下文中保存：

- 当前匹配的 Call；
- 注解表达式；
- 匹配到的 binding；
- 当前 block 的 use-def 信息；
- 表达式到变量的映射。

用户可以使用这些信息判断匹配是否满足后端约束。

## 10. `CompositeFunctionAnnotator`：第 1182～1270 行

该类处理 `annotate_codegen = true` 的情况。

它把带有 `kComposite` 属性的 fused 函数包装成带有：

- `kCodegen`；
- `tvm::attr::kGlobalSymbol`；

的外层函数。

### 10.1 `Run`

遍历模块中的 GlobalVar，跳过已经是 composite 或 codegen 的函数。

如果函数体被修改，就重新构造并写回函数。

### 10.2 访问 Call

如果 Call 的目标函数具有 `kComposite` 属性，代码会：

1. 生成 codegen 名称；
2. 生成全局符号名；
3. 设置 `kCodegen` 和 `GlobalSymbol`；
4. 删除旧函数；
5. 添加新函数；
6. 把原调用替换成对新 GlobalVar 的调用。

如果该 GlobalVar 已经重映射过，则直接使用重映射结果。

### 10.3 访问 Function

对于 FuseOps 生成的 composite function，代码会构造一个外层函数：

```text
local_func = 内部 fused 函数
output = local_func(params)
return output
```

这样后端可以把外层函数作为 codegen 入口，而内部函数保留实际融合逻辑。

## 11. `FuseOpsByPattern`：第 1271～1321 行

该函数依次处理传入的 FusionPattern。

如果指定了入口函数名，代码只处理这些函数，并检查它们必须是 Relax Function。

如果没有指定入口函数，则遍历整个模块，并跳过：

- TIR PrimFunc；
- 已经是 primitive 的函数；
- 已经是 composite 的函数；
- 已经有 codegen 属性的函数。

然后对每个入口函数执行 `PatternBasedPartitioner::Run`。

代码还会检查同一个 AST 对象不能同时出现在多个 partition 中。如果出现，就说明 IRModule 可能不是 single-site assignment，会抛出 `ValueError`。

之后通过 `MakeGroupedFunctions` 生成 grouped functions。

这里的参数关系是：

```text
lift_constants = !bind_constants
```

因此：

- `bind_constants = true`：常量绑定在 fused 函数中，不作为外部参数；
- `bind_constants = false`：常量可以被提升为函数参数。

如果 `annotate_codegen` 为 true，最后执行 `CompositeFunctionAnnotator`。

## 12. `FusionPattern` 和 `PatternCheckContext`：第 1322～1355 行

`FusionPattern` 构造函数将以下内容保存到 `FusionPatternNode`：

- pattern 名称；
- 主 pattern；
- 注解 pattern；
- 可选的 check 函数；
- 可选的 attrs getter。

后面的 FFI 注册代码将 `relax.transform.FusionPattern` 暴露给 Python 和其他 FFI 客户端。

`PatternCheckContext` 构造函数保存：

- 当前匹配表达式；
- 注解表达式；
- 匹配到的 binding；
- 变量使用关系；
- 表达式到绑定变量的映射。

## 13. Pass 注册：第 1357～1387 行

`FuseOps(int fuse_opt_level)` 返回一个 ModulePass。

执行时：

- 如果 `fuse_opt_level == -1`，使用 PassContext 的优化级别；
- 读取 `relax.FuseOps.max_depth`；
- 调用真正执行融合的 `relax::FuseOps`。

随后通过 FFI 注册为：

```text
relax.transform.FuseOps
```

`FuseOpsByPattern` 同样被包装成 ModulePass，并注册为：

```text
relax.transform.FuseOpsByPattern
```

最后几行关闭 `transform`、`relax` 和 `tvm` 命名空间。

## 14. 本章总结

这份代码最重要的五个设计点是：

1. `GraphCreator` 只负责把 Relax IR 转成图，不负责决定是否融合。

2. `GraphPartitioner` 使用算子 pattern、post-dominator 和 union-find 决定哪些节点属于同一组。

3. `FunctionCreator` 负责把一组 binding 封装成新的 Relax 函数，同时处理参数提升、常量内联、Tuple 拆分、多输出和动态形状。

4. `OperatorFusor` 负责改写原始函数，把多条 binding 替换成一次 fused function call。

5. `PatternBasedPartitioner` 不依赖普通算子 pattern，而是使用用户指定的 DFPattern，并给生成函数添加 `kComposite` 等属性。

最终可以用一句话概括：

```text
先把 Relax IR 建模成数据流图，
再利用算子模式和依赖关系完成分组，
然后把每个分组封装成新的 Relax 函数，
最后用函数调用替换原来的多条算子 binding。
```

## 15. 建议的配套阅读

为了继续深入实现细节，建议按以下顺序阅读：

1. `src/relax/transform/fuse_ops.cc`
2. `src/relax/analysis/graph_partitioner.h`
3. `src/relax/analysis/graph_partitioner.cc`
4. Relax 的 `ExprVisitor` 和 `ExprMutator`
5. `BlockBuilder` 的 `Emit`、`EmitOutput` 和 `Normalize`
6. `FuseTIR` 相关实现

其中，`graph_partitioner.cc` 负责实现 `CheckPath`、`CommitFuse`、post-dominator tree 和具体的融合阶段；`fuse_ops.cc` 则负责把分组结果重新生成 Relax 函数并改写原始 IR。
