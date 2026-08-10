# 第 26 课：MLIR 深入——多层可扩展 IR，现代 AI 编译器的新底座

> 本课风格：从"为什么 LLVM 一层不够用"讲起，把方言、ODS、pattern rewrite、
> 下降链讲透，最后落到"自研芯片要不要用 MLIR"。
> 前置：第 14 课（IR 家族）、第 25 课（LLVM 深入）、附录 A（C++）。

---

## 1. 先回答一个问题：LLVM 一层不够用吗？

第 14 课说过：LLVM IR 太底层，表达不了"卷积/融合/布局"这些高层结构。

**更具体一点**：假设你想做"conv + relu 融合"这个优化。

- 在 **toycc/Relax** 层：图上有 conv 节点、relu 节点，融合很自然
- 在 **LLVM IR** 层：conv 早被下降成"循环 + 标量指令"了——
  你看到的是一堆 `load/fmul/fadd/store`，**根本看不出"这里原来是个 conv"**

所以高层优化必须在高层做。但高层 IR 一家一个样（TVM 一套、XLA 一套）——
**MLIR 的目的：给所有"层"提供一套统一的、可扩展的 IR 框架**，
大家共用基础设施（pass 管理、解析、打印、pattern 匹配），各自定义自己的"方言"。

---

## 2. MLIR 的核心：Operation 是万能的

第 14 课你见过 MLIR 的操作语法。这里往深一层。

MLIR 里**所有东西都是 Operation**——不管你是"卷积"还是"加法"还是"加载"，
都长一个样：

```
%结果 = "方言名.算子名"(%输入...) {属性} : (输入类型) -> (输出类型)
```

**这就是 MLIR 的秘密武器**：因为它统一了"操作"的表示，所以：
- **pass 可以一套框架写遍所有层**（遍历 Operation、改写 Operation）
- **高层和低层可以共存于同一个文件**（一个 module 里既有 toy.conv 又有 arith.addf）
- **可以平滑地一层层下降**，每一步都还是合法的 MLIR

**对照 toycc**：你的 `Node` 只有"算子 + 输入 + 属性"，
MLIR 的 `Operation` 多了：**类型系统、region（嵌套）、interface（接口）**。

> **原理深挖：为什么"所有东西都是 Operation"这么重要？**
>
> 这条设计看似平淡，实则是 MLIR 全部价值的锚点。想想 TVM 的 IR：
> 图 IR 是一套类，TIR 是另一套类，两套之间靠翻译函数对接——每次加一个
> 新算子，要改前端、图优化、TIR 生成三处。**维护成本随层数爆炸**。
>
> MLIR 反着做：**只定义一种"操作"，所有层都只是"操作 + 方言名"**。
> 于是原来散落在各层的公共能力（pass 遍历/改写、打印、解析、诊断）
> **全部只需要写一遍**，因为操作长得一样，遍历的逻辑就一样。
>
> 实际收益换算成工程节奏：
> - 新加一种算子 = 新定义`方言名.算子名`，其余 pass **原样复用**
> - 高层降低层 = 只改"匹配这个方言的那部分 pattern"，其余不动
> - 想接一个新设备后端 = 新增一个后端方言，前端和优化分享整套框架
>
> **一句话**：MLIR 是"少做脚手架、多做编译"的哲学——把**表示的统一性**
> 提升到战略地位，让"层多"不再是"代码多"。对你写自研芯片工具链的启发：
> 当你的 IR 层变得很多（图 → 调度 → 指令），**考虑统一成"操作 +
> 方言名"的表示**，而非一堆互不相同的类，能少写大量转换胶水。

### region：操作里能装"一块代码"

```mlir
scf.for %i = 0 to 10 {
  %x = arith.addf %a, %b : f32      ← for 这个操作"里面"装了一段代码
}
```

`scf.for` 这个操作有个 **region**（大括号里那块），里面又是操作。
**这就是"循环"在 MLIR 里的表示**——不是特殊语法，是"带 region 的操作"。
这设计让 MLIR 能表达任意控制流，且 pass 框架统一处理。

---

## 3. 方言（Dialect）：给一层抽象起个名字

一个方言 = **一组相关的算子 + 类型**，共享一个前缀：

```
toy.*      ← 教学方言(conv/transpose/mul)
arith.*    ← 算术(addf/muli)
linalg.*   ← 线性代数(matmul/conv, AI 编译常用)
memref.*   ← 内存(load/store/alloc)
scf.*      ← 结构化控制流(for/if/while)
affine.*   ← 仿射循环(多面体优化)
llvm.*     ← 一个"在 MLIR 里包了一层 LLVM IR"的方言(下降终点)
gpu.*      ← GPU 相关(launch/block/thread)
```

**关键认知**：真实 AI 编译器里，下降链大致是：

```
linalg(高层算子) → affine/scf(循环) → memref(内存) → llvm(出 LLVM IR)
```

每一跳都是一个 pass，把上一层方言的操作**部分**改写成下一层。
"部分"很重要——**渐进式下降**：一次 pass 不必全降完，
降不了的操作留着，等下一个 pass 接着降。这是 MLIR 和"一步到位 lowering"的本质区别。

---

## 4. ODS：用 TableGen 定义算子（不写重复的 C++）

手写一个算子的 C++ 类很啰嗦（名字/输入/输出/校验/打印/解析...）。
MLIR 用 **ODS（Operation Definition Specification）**，基于 TableGen：

```tablegen
def ConvOp : Toy_Op<"conv"> {
  let summary = "convolution operation";
  let arguments = (ins F64Tensor:$input, F64Tensor:$weight);
  let results = (outs F64Tensor:$output);
  let assemblyFormat = "$input `,` $weight attr-dict `:` type($input) `->` type($output)";
}
```

**你只描述"这算子长什么样"**，ODS 自动生成：
- C++ 类（getters/setters）
- 打印 / 解析代码
- 校验逻辑

**这就是第 25 课 TableGen 思想在"算子层"的复用**：
LLVM 用 TableGen 描述指令，MLIR 用 ODS 描述算子——同一套"声明式生成"哲学。

---

## 5. Pattern Rewrite：MLIR 的 pass 心脏

第 3 课你手写融合：遍历图、找模式、改写。MLIR 把这抽象成 **pattern rewrite**：

```cpp
// 把 "transpose(transpose(x))" 优化成 "x"
struct SimplifyRedundantTranspose : public OpRewritePattern<TransposeOp> {
  LogicalResult matchAndRewrite(TransposeOp op, PatternRewriter &rewriter) const override {
    // op 的输入本身也是一个 transpose?
    auto parent = op.getOperand().getDefiningOp<TransposeOp>();
    if (!parent) return failure();                 // 不匹配, 放弃
    rewriter.replaceOp(op, parent.getOperand());   // 用爷爷的输入替换掉这一对 transpose
    return success();
  }
};
```

**逐行讲**：
- `OpRewritePattern<TransposeOp>`：我只关心 transpose 这种操作
- `matchAndRewrite`：先匹配（是不是 transpose(transpose(x))），匹配上就改写
- `rewriter.replaceOp(op, ...)`：框架帮你安全地替换，自动处理所有引用

**这套机制统一了三种事**：
1. **优化**：transpose(transpose(x)) → x（化简）
2. **下降**：linalg.matmul → 循环（高层降低层）
3. **合法化**：框架只接受"合法结果"，保证每步 IR 都正确

**greedy driver**：你把一堆 pattern 扔进 `applyPatternsGreedily`，
框架反复应用直到没有可改——**这就是 MLIR 里"跑 pass"的实际形态**。

---

## 6. 一条完整的下降链（看懂真实编译器）

以 AI 编译器 IREE 的简化版为例，一个 conv 怎么从高层走到机器码：

```
tosa.conv2d            ← 最高层: 张量算子方言
  │  (pattern rewrite: conv → linalg)
linalg.conv_2d         ← 线性代数方言(还是"算子", 但带循环语义)
  │  (linalg → loops)
affine.for / scf.for   ← 循环方言(变成真的循环了)
  │  (bufferize: 张量 → 内存)
memref.load/store      ← 内存方言(循环里读写 buffer)
  │  (convert to llvm)
llvm.*                 ← LLVM 方言(MLIR 里的 LLVM IR)
  │  (translate to LLVM IR)
LLVM IR → 第 25 课的后端 → 机器码
```

**这一跳"bufferize"为什么必须存在**（新手最常跳过去不看的一步）：
下降到循环后，`linalg.conv_2d` 的输入还是一个"不可变张量值"
（SSA value，每用一次算一份）。机器上哪能这样——内存只能有一份，
循环必须读写同一块地址。所以 bufferize 做两件事：
1. 给每个张量**分配一块可写的 buffer**（`memref`）
2. 让循环里的 load/store **读写这块 buffer**，而不是"值"

这就是第 17 课 VM 里 `alloc_tensor` + `kill_tensor` 的 MLIR 版——
**内存生命周期管理从高层就决定好了**，不是到机器码才补。

**每一跳都是 pattern rewrite**，每次只降一点点，中间每一步都是合法 MLIR。
**这就是"多层 IR"的实际工作方式**——也是第 14 课那张"金字塔"的具体实现。

---

## 7. MLIR vs TVM vs 你的自研芯片

| | TVM | MLIR |
|---|---|---|
| 图 IR | Relax | 各种方言(tosa/linalg/...) |
| 循环 IR | TIR | affine/scf/linalg |
| 调度 | meta_schedule | 各项目自定义(或 poly) |
| 下降 | legalize/FuseTIR | pattern rewrite(渐进) |
| 出后端 | codegen → C/PTX/LLVM | llvm 方言 → LLVM IR |

**现实格局**：
- **TVM**：成熟、AI 优化强、自带调度系统——但 IR 是自家一套
- **MLIR**：基础设施统一、生态在涨（IREE/ torch-mlir / XLA 演进）——但调度要自己搭

**对你自研芯片的判断**：
- 想**最快出活**：基于 TVM 加后端（第 24 课五步）
- 想**押注未来生态**：基于 MLIR 建方言 + 下降链（本课），社区资源更多
- 两者不冲突——很多团队**MLIR 做前端/高层，TVM 或自家做调度/后端**

---

## 8. 本课小结

- MLIR = **可扩展的多层 IR 框架**，一切皆是 Operation
- **方言**给每层抽象起名字，**region** 让操作能装控制流
- **ODS/TableGen** 声明式定义算子，自动生成 C++
- **pattern rewrite + greedy driver** 是 pass 的统一写法，优化/下降/合法化三位一体
- **渐进式下降**：一层层降，每步都是合法 MLIR，降不了的留给下一跳
- 自研芯片：MLIR 做前端/高层，后端可接 LLVM（第 25 课）

**到这里，课程从"看懂 toycc"一路走到了"能用工业级基础设施搭自研芯片工具链"。**
剩下的就是动手——回到第 10 课的任务，或在公司代码里找一个真实的 pass 读。

---

## 9. 深层拓展 A：为什么 MLIR 的操作要"强制带 loc"？

你注意过 MLIR 每个操作都有 `loc(...)`（源位置）吗？而且**不能省**。

**原因**：MLIR 是给"基础设施"用的——当一个 AI 模型经过 5 层下降出 bug 时，
你必须能回答"这条机器指令，最初是模型里哪一行？"
强制带 loc，让任何一层报错都能**回溯到源代码**。
toycc 没做这个（教学简化），但真实编译器里这是救命的。

---

## 10. 深层拓展 B：Interface——MLIR 的"鸭子类型"

不同方言的算子，怎么共享同一个优化？比如"所有能内联的操作"。

MLIR 用 **Interface（接口）**：算子声明"我实现了 InlinerInterface"，
pass 就敢对它内联，**不用管它是哪个方言的**。
这就是 MLIR 能"跨方言写通用 pass"的机制——比继承更灵活。

---

## 11. 思考题

1. MLIR 的"渐进式下降"相比"一步到位 lowering"有什么好处？
2. 为什么 pattern rewrite 能同时干"优化"和"下降"两件看起来不同的事？
3. 用 ODS 定义算子，比手写 C++ 类省了什么？
4. 如果让你给自研 GPU 建一个 MLIR 方言，第一层你会放哪些算子？为什么？

> 答案：1) 每步都是合法 IR，可分步调试/测试；多个团队可各管一跳；降不了的能留着。
> 2) 因为"下降"本质就是"把高层操作改写成低层操作组合"——和优化是同一种改写机制。
> 3) 自动生成 C++ 类、打印/解析、校验，避免手写几百行模板代码，还不易写错。
> 4) 第一层放"张量算子"（conv/matmul/relu），因为前端进来就是这个层级，
>    且要保留足够高层语义才能做融合/布局这类图优化。

---

**导航**：⬅ [上一节](lesson25.md)（第 25 课 · LLVM 深入）　｜　[下一节](appendix_cpp.md)（附录 A · C++ 阅读手册）➡
