# MLIR 第 2 章：Dialect、ODS 与 TableGen——几行声明如何变成完整的一个 op

## 1. 本章目标

- 能解释 Dialect 为什么是 MLIR 可扩展性的核心，并说出 arith/tensor/linalg/llvm 各自表达什么；
- 能逐字段读懂一个 ODS 定义（arguments / results / assemblyFormat），并手算"一段文本如何按 assemblyFormat 被解析"；
- 能区分 builder、parser、printer、verifier、trait、interface 六个概念，各举一个对象；
- 能跑 `mlir-tblgen` 的两个 backend，并说明生成的 `.inc` 文件分别是什么；
- 能从一个现有方言（如 arith）反向追到它的 `.td` 定义。

前置：第 1 章的对象模型（Operation/Region/Block/Value）。工具：MLIR（`mlir-tblgen`、`mlir-opt`）。

## 2. 工作中的问题长什么样

ODS 方向的三个入门问题：

```text
"一个 op 就这么几行 .td，C++ 类从哪来的？"
"assemblyFormat 那一串字符串是什么语法？"
"trait 和 interface 听起来一样，到底差在哪？"
```

三个问题对应：**生成链**（声明 → mlir-tblgen → C++）、**自定义语法**（assemblyFormat 的 DSL）、**能力复用**（trait vs interface）。本章逐个建立，最后用一个最小 toy dialect 全部走一遍。

## 3. Dialect：扩展性的核心

**Dialect（方言）**是一组相关 operation、type、attribute 与接口的命名空间与扩展边界。四个例子各一句话：

- `arith`：标量算术（`addi`、`constant`）；
- `tensor`：张量级不可变操作（不关心内存布局）；
- `linalg`：结构化线性代数（把循环结构写进操作）；
- `llvm`：接近 LLVM IR 的方言（下降到 LLVM 前的最后一站）。

**不同方言可以在同一个模块共存**，Pass 决定什么时候把一种语义转换（convert）成另一种——第 1 章的"容器通用、语义外挂"在这里变成"一个模块里同时住着 arith 的加法和 linalg 的 GEMM，某个 pass 再把它们一起降成 llvm 方言"。

## 4. ODS：把重复 C++ 抽成声明

**ODS（Operation Definition Specification）**用 TableGen 声明 op，生成器产出 C++ 样板。一个简化定义：

```tablegen
def Toy_AddOp : Toy_Op<"add", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
```

逐字段拆解：

- `def Toy_AddOp : Toy_Op<"add", [Pure]>`：定义名为 `Toy_AddOp` 的 C++ 类，op 名是 `toy.add`（方言名 + 点 + 名），`[Pure]` 是 trait 列表——声明"无副作用"（第 5 节）；
- `arguments = (ins AnyType:$lhs, ...)`：两个操作数，类型约束是 `AnyType`（任意类型），名字是 `$lhs`/`$rhs`；
- `results = (outs AnyType:$result)`：一个结果；
- `assemblyFormat`：自定义文本语法（第 4.1 节）。

这三行声明能生成：operation 类的声明、访问器（`lhs()`/`rhs()`）、构造辅助函数、解析/打印代码、verifier 框架。**手写的部分只剩语义**——约束、规范化、特殊逻辑。真实项目里的基类与选项更复杂，读源码以对应版本 `.td` 为准。

### 4.1 assemblyFormat：手算"文本如何被解析"

`assemblyFormat` 是一个小 DSL，定义 op 在文本里的长相。取格式：

```text
"$lhs `,` $rhs attr-dict `:` type($result)"
```

逐段：`$lhs` 是第一个操作数；反引号里的 `,` 是字面逗号；`$rhs` 第二个操作数；`attr-dict` 是可选属性字典的位置；`` `:` `` 字面冒号；`type($result)` 表示"结果类型写在这里"。手算一条文本的解析过程：

```mlir
%0 = toy.add %a, %b : i32
       └┬┘    └┬┘ └┬┘   └┬┘
       $lhs    逗号 $rhs  type($result) → 结果类型 i32
```

parser 按格式从左到右匹配：看到 `%a` 填进 `$lhs`，看到逗号继续，`%b` 填进 `$rhs`，冒号后的 `i32` 成为结果类型——**格式串同时定义了打印器和解析器**，写一次两边都有。这也是为什么 MLIR 的文本比 LLVM IR 紧凑：每个方言自带"自己的方言语法"。

## 5. 六个概念逐个定义

| 概念 | 一句话定义 | 典型对象 |
|---|---|---|
| Builder | 用 C++ 或生成代码**构造** operation 的入口 | `build()` 方法、`OpBuilder` |
| Parser | 把文本还原成 operation/operands/type/attr | 由 assemblyFormat 生成 |
| Printer | 把 operation 打印成文本 | 由 assemblyFormat 生成 |
| Verifier | 检查 operation 的**局部语义约束** | `verify()`（类型关系、region 结构） |
| Trait | 可复用的**结构性质**（无副作用、可交换、单 region） | `Pure`、`Commutative` |
| Interface | 跨方言的**能力协议**（按能力查询，不依赖具体类型） | `InferTypeOpInterface` |

trait 与 interface 的区别一句话：**trait 描述"这个 op 是什么样"（结构性质），interface 描述"这个 op 会干什么"（能力，通用 pass 靠它调用不同方言的实现）**。比如一个 pass 想"让任何 op 推自己的结果类型"，它依赖的是 `InferTypeOpInterface` 这个协议，而不是某个具体方言的类。

## 6. C++ 侧的形态：一个手写的等价类

ODS 生成之前的等价手写代码长这样（对照第 4 节的 .td）：

```cpp
class AddOp : public mlir::Op<AddOp, mlir::OpTrait::NOperands<2>::Impl,
                              mlir::OpTrait::OneResult> {
public:
  using Op::Op;                        // 继承通用构造
  mlir::Value lhs() { return getOperand(0); }   // 访问器 = 取第 0 个操作数
  mlir::Value rhs() { return getOperand(1); }
  static void build(mlir::OpBuilder &b, mlir::OperationState &state,
                    mlir::Value lhs, mlir::Value rhs) {   // builder
    state.addOperands({lhs, rhs});
    state.addTypes(lhs.getType());
  }
};
```

逐行：`mlir::Op<...>` 是 CRTP 基类，把通用存储接口（operands/results/attrs）交给派生类；trait 以模板参数混入（`NOperands<2>` = 恰好两个操作数）；`build` 往 `OperationState` 里填操作数与类型——**ODS 干的事就是把这段样板从 .td 重新生成**，并附上 assemblyFormat 的 parser/printer。读生成类时先回 `.td` 找字段来源，再看 C++ 里手写了什么（自定义 verifier、parser、canonicalization pattern）。

## 7. 最小动手：一个 toy dialect 的完整闭环

【可运行代码】。最小 `.td` 文件 `ToyOps.td`：

```tablegen
include "mlir/IR/OpBase.td"

def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "::toy";
}

class Toy_Op<string mnemonic, list<Trait> traits = []>
    : Op<Toy_Dialect, mnemonic, traits>;

def Toy_AddOp : Toy_Op<"add", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}
```

生成两类产物：

```bash
mlir-tblgen --gen-op-decls ToyOps.td -I <mlir-source>/include   # 类声明(头文件片段 .h.inc)
mlir-tblgen --gen-op-defs ToyOps.td -I <mlir-source>/include   # 方法实现(源文件片段 .cpp.inc)
```

两个 backend 的分工：`--gen-op-decls` 生成 `Toy_AddOp` 的类骨架与访问器声明；`--gen-op-defs` 生成 `getOperationName`、parser/printer、verifier 框架的实现。加上十几行 dialect 注册代码（在 `MLIRContext` 上 `loadDialect`），`mlir-opt --help` 里就能看到 `toy` 方言，`mlir-opt` 能解析打印 `toy.add` 的文本。**动手路线：先只加一个二元 op，确认"注册 → 解析 → 打印 → 验证"四步全通，再加类型和 region**——不要一开始同时上自定义 parser、interface 与复杂 lowering。

## 8. 源码阅读地图

- `mlir/include/mlir/IR/OpDefinition.h`：Op、trait、interface 的通用定义；
- `mlir/tools/mlir-tblgen/`：TableGen backend 的实现；
- `mlir/include/mlir/TableGen/`：读取 ODS 定义的工具接口；
- `mlir/include/mlir/Dialect/Arith/` 与 `mlir/lib/Dialect/Arith/`：成熟小方言的完整样板（从 .td 到 C++）；
- `mlir/include/mlir/Dialect/Func/`：函数、符号、region 约束的示例；
- `mlir/test/TableGen/`、`mlir/test/Dialect/`：定义与行为测试。

反向追踪练习：从 `arith.addi` 的报错或文档出发 → 找到 `ArithOps.td` 里的定义 → 看它的 traits 与 assemblyFormat → 再对照生成的访问器。

## 9. 常见错误与归因

| 现象 | 根因 | 修正 |
|---|---|---|
| tblgen 报"unknown field" | `.td` 字段名/基类名与当前版本不符 | 对照同版本已有 op 的定义 |
| 文本解析失败 | assemblyFormat 与实际文本不一致 | 按第 4.1 节逐段对格式串 |
| verifier 不拦错 | 约束没写进 ODS（AnyType 太宽） | 收紧 `ins/outs` 的类型约束 + 自定义 verify |
| `mlir-opt` 不认方言 | dialect 没注册/没加载 | 检查注册代码与 `--help` 里的方言列表 |
| trait 当 interface 用 | 混淆结构性质与能力协议 | 按第 5 节定义重新归类 |

## 10. 本章检查点

完成以下四项才算通过本章：

1. 为 `Toy_AddOp` 把两个操作数收紧为同一类型（提示：用 `TypeConstraint` 或自定义 verify），并做一次负向实验验证 verifier 会拦；
2. 添加一个带属性的 `constant` op（`arguments = (ins APIntAttr:$value)`），并说明"属性为什么不是 SSA value"；
3. 找到 `arith.addi` 的 ODS 定义，摘出它的 traits 与 assemblyFormat，并与生成代码里的访问器对上；
4. 用一句话 + 一个对象各举例，区分 trait 与 interface。

## 11. 本章小结与下一步

本章完成了"从声明到可用的 op"的闭环：Dialect 是边界，ODS 是声明，tblgen 是生成器，assemblyFormat 是语法。下一章（MLIR 03：Pattern Rewrite 与 Dialect Conversion）回答"这些 op 怎么被改写"——第 1 章的 pass 概念在 MLIR 里变成 pattern 与 conversion 的框架。

**导航**：⬅ [上一章](01_ir_core.md)（Operation、Region、Block 与 Value）　｜　[下一章](03_rewrite_and_conversion.md)（Pattern Rewrite 与 Dialect Conversion）➡
