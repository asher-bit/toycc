# 第 28 课：TVM Relax 常量折叠源码详解

## 1. `src/relax/transform/fold_constant.cc` 源码详解

> 本章对应 Apache TVM `main` 分支中的 `src/relax/transform/fold_constant.cc`。
> 源码链接：[fold_constant.cc](https://github.com/apache/tvm/blob/main/src/relax/transform/fold_constant.cc)
>
> 说明：`main` 分支会持续变化，本文中的行号以撰写时版本为准。

## 1. 本章学习目标

读完本章后，应当能够理解：

- Relax 的常量折叠与普通编译期表达式重写有什么区别；
- 为什么 `FoldConstant` 需要构建并执行 TIR PrimFunc；
- 如何判断 `call_tir` 的输入、形状和输出类型是否足够静态；
- 单输出和多输出 TIR 函数是如何被编译期执行的；
- 为什么大尺寸创建算子不一定应该被折叠；
- `legalize`、`tensor_to_shape`、`shape_to_tensor` 在常量折叠中的作用；
- `ExprMutator` 的 post-order mutation 和变量绑定替换如何配合工作。

## 2. 这份 Pass 的整体职责

`fold_constant.cc` 实现的不是简单的“把两个整数相加”。它主要处理下面这类 Relax IR：

```text
data = Constant(...)
weight = Constant(...)
result = call_tir(prim_func, (data, weight), ...)
```

如果：

- TIR PrimFunc 可以在当前主机上构建；
- 所有输入都是常量 Tensor；
- 输出形状是静态的；
- 输出 dtype 已知；
- 折叠结果不会造成不必要的大常量；

那么 Pass 会在编译期执行这个 PrimFunc，并把结果替换成一个 Relax `Constant`。

整体流程可以概括为：

```text
Relax Function
    ↓
ConstantFolder：递归访问并重写表达式
    ↓
发现常量输入的 call_tir
    ↓
构建或获取缓存中的 CPU 版 TIR 函数
    ↓
分配输出 Tensor 并执行 CallPacked
    ↓
用 Relax Constant 替换原 Call
    ↓
RemoveAllUnused 删除被替换后无用的 binding
```

这说明它与 `FuseOps` 的目标不同：

- `FuseOps` 把多个运行时算子合并成一个函数；
- `FoldConstant` 把编译期可以计算的运行时算子直接计算掉。

## 3. 文件头和头文件：第 1～32 行

第 1～17 行是 Apache 许可证声明。

第 18～29 行引用了本 Pass 所需的组件：

- `tvm/ffi/cast.h`：FFI 对象类型转换；
- `tvm/ffi/extra/module.h`：运行时模块和函数获取；
- `reflection/registry.h`：注册 Relax Pass；
- `ir/function.h`：IR 函数相关类型；
- `relax/analysis.h`：变量绑定分析、删除无用代码等；
- `relax/expr_functor.h`：`ExprMutator` 和表达式访问；
- `relax/op_attr_types.h`：`FInferType`、`FLegalize` 等算子属性函数类型；
- `relax/transform.h`：Pass 构造接口；
- `relax/type.h`：TensorType、TupleType 等 Relax 类型；
- `runtime/logging.h`：构建失败时的警告；
- `tirx/function.h`：TIR PrimFunc；
- `tirx/op.h`：TIR 构建和相关算子接口。

第 31～32 行进入 `tvm::relax` 命名空间。

## 4. `ConstantFolder` 类：第 33～396 行

`ConstantFolder` 继承自 `ExprMutator`：

```cpp
class ConstantFolder : public ExprMutator
```

这表示它会遍历 Relax 表达式，并且可以返回一个不同的新表达式。

### 4.1 静态入口 `Fold`：第 35～39 行

`Fold` 接收一个 Relax Function 和当前 IRModule。

执行过程是：

1. 创建 `ConstantFolder`；
2. 用 folder 访问并重写函数；
3. 调用 `RemoveAllUnused`；
4. 将结果转换回 Relax Function。

为什么最后需要 `RemoveAllUnused`？

例如：

```text
lv0 = Constant(...)
lv1 = call_tir(..., lv0, ...)
```

如果 `lv1` 被折叠成新的 Constant，那么原来的 `lv0` 或 `lv1` binding 可能已经没有使用者了，需要清理掉。

### 4.2 构造函数：第 41～42 行

构造函数把 IRModule 传给基类 `ExprMutator`。

基类中的 `builder_`、变量作用域和 binding 查询功能都依赖这个上下文模块。

## 5. 识别静态输入：第 43～93 行

这部分的函数只负责“模式匹配”和“提取静态信息”，本身不执行 TIR。

### 5.1 `MatchConstShape`：第 51～66 行

它尝试从一个 Relax `Type` 中提取静态运行时 shape。

第 52～54 行要求类型必须是 `TensorType`。如果不是 TensorType，就返回空结果。

第 56～57 行要求 TensorType 的 shape 必须是 `ShapeExpr`。

这意味着下面这种动态 shape 不能直接得到静态 shape：

```text
Tensor((n, m), float32)
```

第 59～64 行遍历每个维度，并要求它必须是 `IntImm`。

只要某个维度不是编译期整数，就返回 `std::nullopt`。

如果所有维度都是常量整数，第 65 行把它们转换为运行时 `ffi::Shape` 并返回。

### 5.2 `MatchConstArrayArgs`：第 71～79 行

该函数检查一个参数数组是否全部由 Relax Constant 组成。

对每个参数：

- 转换为 `relax::ConstantNode`；
- 如果不是 Constant，匹配失败；
- 如果是 Constant，就提取其底层 `runtime::Tensor`。

成功后返回 Tensor 数组，供后续构造 TIR 调用参数。

### 5.3 `MatchPrimFunc`：第 85～93 行

这个函数尝试从一个操作数中找到 TIR PrimFunc。

具体步骤：

1. 要求操作数是 GlobalVar；
2. 从当前 IRModule 查找该 GlobalVar 对应的 BaseFunc；
3. 如果它是 TIR PrimFunc，就返回该 PrimFunc；
4. 否则返回空结果。

因此，只有 IRModule 中真实存在的 TIR PrimFunc 才能被编译期执行。

## 6. TIR 函数构建缓存：第 94～122 行

### 6.1 `GetCachedBuild`

常量折叠可能多次遇到同一个 PrimFunc。如果每次都重新用 LLVM 构建，会浪费大量编译时间，所以代码维护了 `func_build_cache_`。

第 101 行选择 LLVM CPU target：

```text
llvm
```

第 102～105 行先查缓存：

- 找到缓存就直接返回；
- 没找到才继续构建。

第 106 行默认构建结果为空。

第 112 行通过 FFI 获取全局函数：

```text
tirx.build
```

第 113 行给 PrimFunc 设置全局符号名 `tir_function`。

第 114 行调用构建函数，把 PrimFunc 和 LLVM target 传入，得到运行时模块。

第 115 行从运行时模块中取出 `tir_function`。

第 116～119 行捕获构建异常。

并不是所有 PrimFunc 都能被当前 CPU 的 LLVM 直接执行，例如某些函数可能只适用于 GPU。构建失败时，代码打印 warning，但不让整个常量折叠 Pass 失败。

第 120 行把成功结果或空结果都缓存起来。

这点很重要：失败结果也会缓存，避免后续不断重复尝试构建同一个不可折叠 PrimFunc。

## 7. 判断折叠是否值得：第 123～180 行

常量折叠不是“能算就一定算”。折叠会产生编译期成本和新的常量数据，因此需要考虑收益。

### 7.1 `ExprContainsTensor`：第 130～142 行

该函数判断表达式或其 Tuple 字段中是否包含 Tensor。

第 131～133 行直接检查表达式类型是否为 TensorType。

第 134～140 行递归检查 Tuple 的每个字段。

这个函数用于区分：

- 没有 Tensor 输入的创建算子，例如 `zeros`、`ones`、`full`；
- 使用已有 Tensor 输入的计算算子，例如 `add`、`reshape`、`matmul`。

### 7.2 `ShouldBeFolded`：第 143～180 行

该函数判断当前 Call 是否值得折叠。

第 147 行定义最大允许折叠元素数：

```text
1024
```

第 149～150 行：如果表达式不是 Call，直接认为可以继续处理。

第 152～155 行：如果返回类型不是 TensorType，或者没有可用 shape，也继续允许处理，因为当前函数无法据此判断输出大小。

第 157～168 行计算输出元素数量：

- 初始值为 1；
- 逐个乘以维度；
- 如果维度不是 IntImm，返回 true，交给后续逻辑处理；
- 如果维度小于等于 0，也返回 true；
- 使用除法检查避免整数乘法溢出。

第 170 行：如果输出元素数量不超过 1024，就允许折叠。

第 171～179 行处理大输出：

- 如果输出很大，但输入中存在 Tensor，仍然允许折叠；
- 如果输出很大且没有 Tensor 输入，则认为这是大型纯创建算子，不折叠。

原因是：运行时创建一个大 `zeros` 可能很便宜，而把它提前物化为 IRModule 中的常量会增大二进制和内存占用。

## 8. 编译期执行单输出 TIR：第 181～209 行

### `ConstEvaluateCallTIR`

这个函数执行“单 Tensor 输出”的 `call_tir`。

第 187 行从缓存获取已经构建好的运行时函数。如果拿不到，返回空结果。

第 190 行创建 packed 参数数组。长度比输入 Tensor 数量多 1，因为最后还要放输出 Tensor。

第 192 行指定 CPU device。

第 193 行按照静态 shape、dtype 和 CPU device 分配输出 Tensor。

第 197 行复制输入 Tensor 到 `temp_args`，确保传给 packed 调用的对象生命周期稳定。

第 199～201 行把所有输入 Tensor 放进 packed 参数数组。

第 203 行把输出 Tensor 放在最后。

第 207 行调用编译好的函数：

```text
CallPacked(inputs..., output)
```

TIR PrimFunc 直接写入预先分配的输出 Tensor。

第 208 行将填充好的输出 Tensor 包装成 Relax Constant。

## 9. 编译期执行多输出 TIR：第 210～247 行

### `ConstEvaluateCallTIRTuple`

这个函数处理返回 Tuple 的 TIR PrimFunc。

第 218 行读取输出数量。

第 220～228 行为每个输出分配 Tensor：

- 从对应字段类型中提取静态 shape；
- 如果 shape 不是静态的，失败；
- 如果 dtype 未知，失败；
- 按 shape、dtype 和 CPU device 分配输出。

第 229～238 行把所有输入 Tensor 和所有输出 Tensor 按顺序放入 packed 参数数组。

第 241 行执行 TIR 函数。

第 242～246 行把每个输出 Tensor 包装成 Constant，再组合成 Relax Tuple。

## 10. `VisitCallTIR`：第 248～272 行

这个函数负责识别并折叠 Relax 的 `call_tir`。

第 251 行要求 Call 至少有两个参数：

- 第一个是 TIR PrimFunc；
- 第二个是输入 Tuple。

第 252 行查找 TIR PrimFunc。

第 253～255 行确认第二个参数是 Tuple，并检查 Tuple 中所有字段是否都是常量 Tensor。

第 256 行要求 `ty_args` 恰好有一个元素，因为必须知道 Call 的输出类型。

第 257 行：如果 PrimFunc 或输入常量匹配失败，就返回空结果。

第 259～262 行处理 Tuple 输出。输出类型是 `TupleType` 时，调用前面介绍的多输出版本。

第 264～268 行处理单 Tensor 输出：

- 从输出类型提取静态 shape；
- 从 Call 类型中提取 Tensor dtype；
- 调用单输出执行函数。

第 271 行：shape 不静态或类型不符合要求时，返回空结果。

## 11. 重写 Call：第 274～382 行

### 11.1 前置说明：第 274～279 行

第 274 行将基类的 `VisitExpr_` 引入当前类。

第 275～278 行的 TODO 说明当前实现暂时不支持：

- 带 `ffi::Function` 的常量折叠；
- MatchCast 的常量折叠。

注释还说明，在当前设计下，`DecomposeOps()` 应该在本 Pass 后处理 `tensor_to_shape` 等操作。

### 11.2 `VisitExpr_(const CallNode*)`

第 281 行先调用 `VisitExprPostOrder_`。

这意味着先递归处理 Call 的输入，再处理当前 Call 本身，也就是 post-order mutation。

例如：

```text
z = add(const_a, const_b)
y = reshape(z, shape)
```

会先尝试折叠 `z`，再使用折叠后的结果处理 `y`。

第 284 行调用 `ShouldBeFolded`。如果当前 Call 不适合折叠，就直接返回递归重写后的 Call。

第 285～287 行获取：

- `relax.call_tir` Op；
- `FInferType` 属性表；
- `FLegalize` 属性表。

第 288～293 行检查被调用对象是否是 `OpNode`。如果不是 Op，例如是一个变量函数，就不做下面这些基于 Op 属性的特殊处理。

第 294 行将 OpNode 转为 `Op`。

第 296～298 行：如果当前 Op 是 `relax.call_tir`，直接调用 `VisitCallTIR` 尝试折叠。

### 11.3 ShapeExpr 传播：第 299～323 行

这一段处理下面的情况：

```text
lv = R.shape([16, 16])
gv = R.reshape(data, lv)
```

如果 `lv` 的 binding 已经是静态 ShapeExpr，那么在重写后续 Call 时，可以直接把变量替换成 ShapeExpr：

```text
gv = R.reshape(data, R.shape([16, 16]))
```

第 307～317 行遍历 Call 参数：

- 如果参数是 Var，就通过 `LookupBinding` 查找它的值；
- 如果值是 ShapeExpr，就直接放入新参数；
- 否则保留原参数。

第 318～321 行确定重建 Call 的返回类型。如果返回的是 PrimType 且没有类型推导函数，就保留原类型；否则使用 `Type::Missing()`，让后续机制处理类型推导。

第 322～323 行使用新参数重新构造 Call。

### 11.4 在 dataflow block 中尝试 legalize：第 324～335 行

只有位于 dataflow block 中时，代码才尝试进一步折叠算子。

第 327 行判断当前 Op 是否存在 `FLegalize`。

如果存在：

1. 规范化当前 Call；
2. 调用该 Op 的 legalize 函数；
3. 再次规范化 legalize 结果；
4. 如果结果变成 `call_tir`，再次调用 `VisitCallTIR` 尝试编译期执行。

这体现了 Relax 的典型流程：高层 Relax Op 先 legalize 成低层 `call_tir`，再由 ConstantFolder 执行。

### 11.5 特殊处理 `tensor_to_shape`：第 336～360 行

`relax.tensor_to_shape` 是复合操作，目前没有统一的 decomposition map，因此代码暂时专门处理它。

第 344 行要求输入数量为 1。

第 346～347 行要求输入是 Constant。

第 348 行取得底层 runtime Tensor。

第 349～352 行要求该 Tensor：

- 位于 CPU；
- 内存连续；
- `byte_offset == 0`；
- 是一维 Tensor。

第 353～354 行把数据解释为 `int64_t` 数组。

第 355～359 行逐个读取 shape 数值，并创建 `ShapeExpr` 返回。

也就是说，它把：

```text
Constant([16, 16]) → tensor_to_shape
```

转换成：

```text
ShapeExpr([16, 16])
```

### 11.6 特殊处理 `shape_to_tensor`：第 361～379 行

`shape_to_tensor` 通过 `ffi::Function` 实现，而不是普通 Relax Op，因此暂时单独处理。

第 364～366 行读取输入 ShapeExpr 及其维度。

第 368～373 行把每个维度转换为整数数组，并检查是否都是 int64。

第 374～378 行：如果所有维度都已知，就调用：

```text
relax.run.shape_to_tensor
```

得到 runtime Tensor，再包装成 Relax Constant。

如果以上条件不满足，第 381 行返回重写后的原 Call。

## 12. 重写变量：第 384～390 行

`VisitExpr_(const VarNode*)` 会查询变量绑定。

如果变量绑定到一个 Relax Constant，就直接返回这个 Constant。

否则交给基类继续处理。

这一步使得前面已经折叠出的常量能够沿着后续 binding 传播。

## 13. 构建缓存字段：第 392～396 行

`func_build_cache_` 是 PrimFunc 到构建结果的缓存。

它使用：

- `StructuralHash`：结构哈希；
- `StructuralEqual`：结构相等比较。

因此即使两个 PrimFunc 不是同一个 C++ 指针，只要结构相同，也可以复用构建结果。

## 14. Pass 注册：第 398～409 行

第 400 行定义 `FoldConstant()` Pass 工厂。

第 401～403 行创建 FunctionPass：

- 输入是单个 Relax Function；
- 可以访问当前 IRModule 和 PassContext；
- 调用 `ConstantFolder::Fold`；
- 返回重写后的 Function。

第 404 行创建名为：

```text
FoldConstant
```

的 FunctionPass。

第 406～409 行把它注册为：

```text
relax.transform.FoldConstant
```

最后几行关闭命名空间。

## 15. 这份 Pass 的关键设计点

### 15.1 它不是纯符号替换

普通的常量传播可能只需要改写 AST；这里的 `call_tir` 折叠需要：

1. 构建 PrimFunc；
2. 分配真实 runtime Tensor；
3. 调用生成的机器码；
4. 将结果重新包装为 Relax Constant。

因此这是“编译期执行”，而不只是“编译期推导”。

### 15.2 构建失败不会让 Pass 失败

GPU-only PrimFunc 或当前 LLVM 不支持的函数可能无法构建。代码会缓存失败结果并跳过折叠，让后续 Pass 继续工作。

### 15.3 大型纯创建算子不一定折叠

例如大型 `zeros`、`ones` 或 `arange`，运行时生成可能比把整个结果保存进模块更划算。

### 15.4 `FuseOps` 与 `FoldConstant` 的执行顺序

通常应当先让常量尽可能折叠，再对剩余运行时算子做融合；但实际顺序要结合 legalize、decompose 和 backend pipeline 决定。

### 15.5 当前限制

源码中的 TODO 明确表明当前实现暂不完整支持：

- `ffi::Function` 的通用常量折叠；
- MatchCast；
- 所有复合算子的统一 decomposition map。

## 16. 建议实验

### 实验一：折叠纯常量 `call_tir`

构造两个 Constant Tensor，将它们传给一个简单的加法 PrimFunc，运行 `FoldConstant`，观察 `call_tir` 是否变成 Constant。

### 实验二：测试大输出创建算子

分别测试输出元素数量小于和大于 1024 的 `zeros`，观察大尺寸纯创建算子是否被跳过。

### 实验三：测试 legalize 路径

选择一个高层 Relax Op，确认它经过 legalize 后是否变成 `call_tir`，以及 ConstantFolder 是否能进一步执行。

### 实验四：测试 shape 传播

构造：

```text
lv = shape([16, 16])
out = reshape(data, lv)
```

观察 `lv` 是否被直接替换为 ShapeExpr。

### 实验五：观察构建缓存

让同一个 PrimFunc 在多个常量折叠位置出现，给 `GetCachedBuild` 加日志，确认 PrimFunc 只被构建一次。

