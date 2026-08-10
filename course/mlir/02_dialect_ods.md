# MLIR 第 2 章：Dialect、ODS 与 TableGen

## 1. 本章目标

- 理解 Dialect 为什么是 MLIR 可扩展性的核心；
- 看懂一个 operation 从 `.td` 描述到 C++ 类的生成链；
- 区分 builder、parser、printer、verifier、trait 和 interface；
- 能从一个现有 Dialect 反向追到其定义。

## 2. Dialect 解决什么问题

Dialect 是一组相关 operation、type、attribute 和接口的命名空间及扩展边界。`arith` 表达算术，`tensor` 表达张量级操作，`linalg` 表达结构化线性代数，`llvm` 则接近 LLVM IR。不同 Dialect 可以在同一个模块中共存，Pass 决定什么时候把一种语义转换成另一种语义。

## 3. ODS/TableGen 的设计

Operation Definition Specification（ODS）把重复的 C++ 样板抽成声明式定义。一个简化示意：

```tablegen
def Toy_AddOp : Toy_Op<"add", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
```

这份描述可以生成 operation 类的声明、访问器、构造辅助函数、解析/打印相关代码和 verifier 框架。真实项目中的基类、约束和生成选项会更复杂，读源码时要以对应版本的 `.td` 文件为准。

```text
*.td
  ↓ mlir-tblgen
OpDeclarations.h.inc / OpDefinitions.cpp.inc / ...
  ↓ include / 编译
具体 Dialect 的 C++ operation 类
```

## 4. Builder、Parser、Printer、Verifier

- Builder：用 C++ 或 generated builder 构造 operation；
- Parser：把文本语法还原成 operation、operands、结果类型和属性；
- Printer：把 operation 打印为通用或自定义语法；
- Verifier：检查 operation 的局部语义约束；
- Trait：复用结构性性质，如无副作用、单 block region 或特定操作数关系；
- Interface：让通用 Pass 通过能力查询调用不同 Dialect 的实现。

一个 operation 看起来只有几行定义，实际上“能被工具链使用”依赖这些层一起工作。定义一个新 op 时，必须考虑它如何构造、如何打印、如何验证以及 lowering 时需要哪些接口。

## 5. C++ 侧的阅读方式

```cpp
class AddOp : public mlir::Op<AddOp, mlir::OpTrait::NOperands<2>::Impl,
                              mlir::OpTrait::OneResult> {
public:
  using Op::Op;
  mlir::Value lhs() { return getOperand(0); }
  mlir::Value rhs() { return getOperand(1); }
  static void build(mlir::OpBuilder &b, mlir::OperationState &state,
                    mlir::Value lhs, mlir::Value rhs) {
    state.addOperands({lhs, rhs});
    state.addTypes(lhs.getType());
  }
};
```

这段示例强调 operation 类如何访问通用存储；真实代码通常由 ODS 生成 `getOperationName`、访问器、builder 和约束逻辑。读生成类时，先回到 `.td` 找字段来源，再看 C++ 是否添加了自定义 verifier、parser 或 canonicalization pattern。

## 6. 源码阅读地图

- `mlir/include/mlir/IR/OpDefinition.h`：Op、trait、interface 等通用定义；
- `mlir/tools/mlir-tblgen/`：TableGen backend；
- `mlir/include/mlir/TableGen/`：读取 ODS 定义的工具接口；
- `mlir/include/mlir/Dialect/Arith/` 与 `mlir/lib/Dialect/Arith/`：成熟小 Dialect 示例；
- `mlir/include/mlir/Dialect/Func/`：函数、符号、region 约束示例；
- `mlir/test/TableGen/`、`mlir/test/Dialect/`：定义和行为测试。

## 7. 动手路线

```bash
mlir-tblgen --gen-op-decls ToyOps.td -I <mlir-source>/include
mlir-tblgen --gen-op-defs ToyOps.td -I <mlir-source>/include
mlir-opt --help | Select-String toy
```

先复制一个最小 toy dialect，只增加一个二元算术 op；确认能注册、解析、打印和验证，再增加自定义类型或 region。不要一开始就同时引入自定义 parser、interface 和复杂 lowering。

## 8. 练习

1. 为 `Toy_AddOp` 增加同类型约束；
2. 添加一个带属性的 `constant` op，并说明该属性为什么不是 SSA value；
3. 找到 `arith.addi` 的 ODS 定义，追踪它的 generated accessor；
4. 比较 trait 和 interface：前者更像可复用结构性质，后者更像跨 Dialect 的能力协议。

参考：[Defining Dialects](https://mlir.llvm.org/docs/DefiningDialects/)、[MLIR TableGen](https://mlir.llvm.org/docs/OpDefinitions/)。

