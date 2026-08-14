# LLVM 第 1 章：IR、SSA 与验证器——从对象模型到"为什么能这样优化"

## 1. 本章目标

- 能画出 `Module → Function → BasicBlock → Instruction` 的容器关系，并说出 `Value`/`Type` 在其中扮演的角色；
- 能读懂带分支、循环与 `phi` 的 LLVM IR，并手算"沿哪条边进入合流点会选哪个值"；
- 能解释 SSA 与支配关系的定义，并判断一个 def 对某个 use 是否可见；
- 能用手算 GEP 的地址偏移，说明"GEP 不算内存"为什么对优化重要；
- 能运行 `llvm-as / llvm-dis / opt -passes=verify / llc` 四个命令并解释各自输出。

前置：第 14 课"IR 家族"的概念地图（Relax/TIR/LLVM IR/PTX 各在哪一层）。工具：LLVM ≥ 15（命令以当前安装版本为准）。示例 IR 基于 opaque pointer 语法（LLVM 15+ 默认）。

## 2. 工作中的问题长什么样

编译器方向读 IR 时反复出现的三个问题：

```text
"这个 pass 为什么不敢删这条看起来没用的指令？"
"phi 节点执行几次？它是一条机器指令吗？"
"verifier 报 'instruction does not dominate all uses'，什么意思？"
```

三个问题的答案分别在：**SSA 的性质**（值怎么定义、怎么用）、**phi 的语义**（合流点的选择规则）、**支配关系**（def 对 use 的可见性）。本章把这三件事建成模型，verifier 只是把模型写成可执行检查。

## 3. 最小例子：sum 的逐行拆解

把下面的代码保存为 `sum.ll`：

```llvm
target triple = "x86_64-pc-linux-gnu"

define i32 @sum(i32 %a, i32 %b) {
entry:
  %s = add i32 %a, %b
  ret i32 %s
}
```

逐行看，每行标出它在 C++ 里的对象：

- `target triple`：描述目标平台的字符串（"架构-厂商-系统-ABI"）。它影响默认 ABI、指令选择与可用特性——对象是 `Module` 的一个属性；
- `define i32 @sum`：定义一个返回 `i32` 的函数。对象是 `llvm::Function`，它本身又是一个 `GlobalValue`（模块级符号，有自己的名字 `@sum`）；
- `%a`、`%b`：函数参数，也是 SSA 值。对象是 `Argument`（`Value` 的子类）；
- `add i32`：二元运算指令，**类型显式写在指令里**（`i32` 同时是操作数与结果的类型）。对象是 `BinaryOperator`（`Instruction` 的子类，而 `Instruction` 又是 `Value` 的子类——"指令即值"是 LLVM 的核心设计）；
- `%s`：这条指令产生的 SSA 名字。**LLVM 允许两种名字：命名值（`%s`）和编号值（`%0`、`%1`）**，编号只是打印器按出现顺序自动编的，语义上没有差别；
- `ret`：终结指令，结束当前基本块并返回。对象是 `ReturnInst`（`Instruction` 的子类）。

对象层级一张图：

```text
Module(模块)
 ├─ Function(函数: 参数表 + 基本块列表)
 │    ├─ BasicBlock(基本块: 指令链表 + 前驱/后继)
 │    │    ├─ Instruction(指令, 同时是 Value)
 │    │    └─ ...
 │    └─ ...
 └─ GlobalVariable(全局变量)
Value = 一切能出现在操作数位置的对象的基类(指令、参数、常量、全局变量...)
Type  = i32 / i64 / ptr / {..} / <..> 等类型对象, 挂在值身上
```

关于 opaque pointer 的一个常见混淆：**LLVM 新语法里 `ptr` 不再携带"指向什么类型"**，但"LLVM IR 有类型"依然成立——`load i32, ptr %p` 里的 `i32` 就是这次加载的类型。读代码时不要混为一谈：指针类型变"秃"了，内存访问的类型语义还在指令上。

## 4. 分支、phi 与 SSA 合流

### 4.1 phi 的语义：手算"选哪个值"

```llvm
define i32 @select_value(i1 %cond, i32 %x, i32 %y) {
entry:
  br i1 %cond, label %then, label %else

then:
  br label %join

else:
  br label %join

join:
  %v = phi i32 [ %x, %then ], [ %y, %else ]
  ret i32 %v
}
```

**phi 节点不是运行时执行的一条普通机器指令**。它的语义是："沿哪条前驱边进入本块，就选那条边对应的值"。手算两条路径：

```text
cond = true  → entry → then → join, 走 [ %x, %then ] 这条 → %v = %x
cond = false → entry → else → join, 走 [ %y, %else ] 这条 → %v = %y
```

两个硬性约束：**每个 incoming 值与一个前驱块一一对应**（`[ %x, %then ]` 表示"从 %then 来时取 %x"）；**phi 只能出现在基本块的开头**（合流发生在块入口，不在块中间）。降到机器码时 phi 被消除——后端的做法通常是让各前驱把值写进同一个物理寄存器/栈槽（`%x` 和 `%y` 的"写回位置"被统一），而不是生成一条"选值指令"。

### 4.2 循环需要 phi：一个非 SSA 的变量怎么表达

普通语言里 `i = i + 1` 反复赋值，SSA 里"每个名字只被定义一次"，循环计数长这样：

```llvm
define i32 @sum_to_n(i32 %n) {
entry:
  br label %loop

loop:                                     ; 循环头
  %i = phi i32 [ 0, %entry ], [ %i_next, %loop ]   ; ← 进入循环时取 0, 回边取 %i_next
  %acc = phi i32 [ 0, %entry ], [ %acc_next, %loop ]
  %done = icmp eq i32 %i, %n
  br i1 %done, label %exit, label %body

body:
  %acc_next = add i32 %acc, %i
  %i_next = add i32 %i, 1
  br label %loop

exit:
  ret i32 %acc
}
```

两个 phi 各有两个 incoming：一条来自 `%entry`（第一次进循环，初始值 0），一条来自 `%loop` 自己（回边，下一轮的值）。**"哪个值"由"从哪条边进来"决定**——这就是循环在 SSA 里的全部秘密：没有可变变量，只有"每个版本的 i 都有一个新名字，phi 在循环头把它们接起来"。

### 4.3 mem2reg：alloca 是怎么变成 phi 的

前端（如 Clang）生成 IR 时偷懒，用 `alloca/load/store` 模拟局部变量；`mem2reg` pass 把这种"内存版变量"提升成真正的 SSA。手算一个例子（左边是 mem2reg 前，右边是后）：

```llvm
; mem2reg 前                             ; mem2reg 后
entry:                                   entry:
  %p = alloca i32                          br i1 %cond, label %then, label %else
  store i32 42, ptr %p
  %x = load i32, ptr %p                 then:
  %cond = ...                              br label %join
  br i1 %cond, label %then, label %else
                                         else:
then:                                      br label %join
  store i32 7, ptr %p
  br label %join                         join:
                                           %v = phi i32 [ 42, %then ], [ 7, %else ]
else:                                      ret i32 %v
  br label %join

join:
  %v = load i32, ptr %p
  ret i32 %v
```

变换的逻辑：**"存进 %p 的值"变成"沿途携带的 SSA 值"**——entry 里的 store 42 变成 then 边上的 42，then 里的 store 7 变成 else 边上的 7，join 的 load 变成 phi 的选择。这就是"内存版变量 → SSA 值"的完整推导：每个 store 定义一个新版本，每个合流点用一个 phi 把版本接起来。跑 `opt -passes='mem2reg' -S` 可以亲眼看这个变换。

## 5. 支配关系：def 凭什么对 use 可见

**支配（dominance）**的定义：节点 d **支配**节点 u，当且仅当从入口到 u 的**每条**路径都经过 d。在 `select_value` 的 CFG 上验证两个判断：

```text
entry 支配 join 吗？  是——去 join 必经 entry(入口当然支配一切)
%x 的定义块 entry 支配 join 吗？  是——%x 是参数, 参数在所有块可见
then 支配 join 吗？  否——存在 entry→else→join 这条路径不经过 then
```

SSA 的**支配规则**：一个值的 def 必须支配它的每个 use。违反时 verifier 报 `instruction does not dominate all uses`——原因直观：如果 def 不支配 use，就存在一条路径"用到了还没算出来的值"，程序在这条路径上没有定义。于是三个问题的答案闭环：pass 不敢删一条"看起来没用"的指令，常常是因为它有 use 在别的块里；verifier 的报错就是这条规则的执行版。很多 pass 的第一步不是改指令，而是先向 `DominatorTree`、`LoopInfo` 要控制流事实——这两个是 LLVM 里最常用的两个分析对象。

## 6. GEP：地址计算与内存访问的分离

**getelementptr（GEP）**是"纯地址计算"指令：给定基址指针与索引，算出子元素的地址，**不读内存**。

```llvm
%p = getelementptr i32, ptr %base, i64 %index   ; 地址 = base + index × 4(按 i32 缩放)
%x = load i32, ptr %p, align 4                  ; 真正读内存的是 load
```

手算 GEP 的偏移——**每个索引按它前面的类型缩放**。一个结构体例子（`DataLayout` 规定 i64 对齐 8）：

```llvm
%struct.S = type { i32, i8, i64 }        ; 字段偏移: 0, 4, 8(8 字节对齐, i8 之后有 3 字节 padding)
%g = getelementptr %struct.S, ptr %base, i64 2, i32 1
; 第 1 个索引缩放 = sizeof(struct) = 16 → base + 2×16
; 第 2 个索引进字段 1 = i8, 偏移 4        → 最终地址 = base + 32 + 4 = base + 36
```

手算的关键：**第一个索引按整体类型缩放（16 字节），第二个索引按字段类型缩放（1 字节）**。GEP 与 load 分离的设计收益：别名分析可以只追踪"地址怎么算出来"（两个 GEP 是否可能指向同一地址），而不必追踪每一次内存读写；向量化、后端寻址模式匹配也都把 GEP 当成纯算术来变换。`align 4` 是访问对齐声明——load 的地址不满足对齐时，优化器不能再假设对齐，甚至程序行为未定义，这是"目标信息不是装饰"的第一个例子。

## 7. 目标信息：DataLayout 与合法性的三层检查

模块里的 `target triple` 与 `DataLayout` 告诉优化器：指针宽度、整数/浮点的 ABI 对齐、地址空间、端序、结构体布局。**"抽象 IR 上合法"不等于"目标上合法"**——合法性分三层检查：

```text
IR 语义正确性   →  verifier(SSA、支配、类型、块结构...)
目标相关合法性  →  DataLayout / TargetTransformInfo / target hooks(对齐、布局、代价)
机器级可实现性  →  TargetLowering / instruction selector(指令是否存在、寻址模式)
```

一个具体例子：x86-64 上 `load i32, ptr %p, align 4` 可以生成一条 `movl`；如果程序保证不了 4 字节对齐，就必须走未对齐路径（多条指令）。优化器每动一次访存，都要按 DataLayout 重新核对对齐与布局——忽略这层的变换会在后端"突然"产生错误代码。

## 8. 工具实验

对 `sum.ll` 依次执行（命令属于【可运行代码】）：

```bash
llvm-as sum.ll -o sum.bc                    # 文本 IR → bitcode(二进制 IR)
llvm-dis sum.bc -o -                        # bitcode → 文本, 观察往返后是否等价
opt -passes='verify' sum.bc -disable-output # 只跑 verifier, 检查 IR 不变量
llc sum.ll -o sum.s                         # IR → 目标汇编(需要目标后端可用)
```

再做一个**负向实验**：把 `select_value` 里 phi 的 `[ %y, %else ]` 删掉，只剩 `[ %x, %then ]`，跑 `opt -passes='verify'`。预期报错形如：`PHI node entries do not match predecessors!`——verifier 把"incoming 与前驱一一对应"这条约束变成了可执行检查。整理"报错信息 ↔ 对应的 IR 不变量"这张表，就是读 `Verifier.cpp` 的入门地图。

## 9. 源码阅读地图

按顺序打开，每层只记三件事（对象由谁拥有、迭代器遍历什么、修改后哪些分析失效）：

1. `llvm/include/llvm/IR/Value.h`：所有 SSA 值共享的基础接口（use 列表、类型、名字）；
2. `llvm/include/llvm/IR/Instruction.h`：指令是 Value，同时挂在某个基本块的指令链表里；
3. `llvm/include/llvm/IR/BasicBlock.h`：指令链表 + 前驱/后继（CFG 关系）；
4. `llvm/include/llvm/IR/Function.h`、`Module.h`：容器关系与全局符号表；
5. `llvm/lib/IR/Verifier.cpp`：LangRef 的约束如何变成 C++ 检查（对照第 8 节的负向实验读）；
6. `llvm/include/llvm/IR/DataLayout.h`：目标布局查询接口（对齐、偏移、端序）。

## 10. 常见错误与归因

| 现象 | 根因 | 修正 |
|---|---|---|
| verifier 报 `does not dominate all uses` | 值在不可见的分支被使用 | 在合流点补 phi，或调整代码位置 |
| verifier 报 `PHI node entries do not match predecessors` | phi 的 incoming 与前驱块不对应（改了 CFG 没改 phi） | 边与值一一对应地修 |
| 手写 IR 结果错 | GEP 索引缩放算错 / 当成了 load | 按第 6 节逐索引手算偏移 |
| opaque pointer 语法下报类型错 | 把"指针不带 pointee 类型"当成"IR 没类型" | 类型写在 load/store/GEP 上 |
| 优化后访存出错 | 忽略了 `align` 与 DataLayout | 核对对齐与结构体布局 |

## 11. 本章检查点

完成以下四项才算通过本章：

1. 给一个 `if/else` 的 C 函数手写 IR，画出 CFG，标出 phi 的每条 incoming 来自哪条边；
2. 在 `sum_to_n` 的 CFG 上判断：`body` 支配 `loop` 吗？`%acc_next` 的 def 支配它的 use 吗？（后者该在 verifier 报错，说出为什么）；
3. 手算 `getelementptr %struct.S, ptr %base, i64 1, i32 2` 的字节偏移（结构体同第 6 节）；
4. 用 `opt -passes='mem2reg' -S` 把一个带 `alloca` 的函数转成 SSA，指出 phi 从哪条 store 来。

## 12. 本章小结与下一步

读 LLVM 源码时，先把 IR 当成"带控制流约束的数据结构"，再把每个 Pass 看作对这些结构的**不变量维护**。只看指令名字而不看支配关系、类型、布局与分析失效规则，结论很容易错。下一章（LLVM 02：Analysis 与 Pass 体系）回答"这些分析结果如何被缓存、失效和复用"——本章的 DominatorTree/LoopInfo 在那里成为分析体系的第一个例子。

**导航**：⬅ 上一章：无（本专题第一章，先看 [专题目录](README.md)）　｜　[下一章](02_analysis_passes.md)（Analysis、Pass 与新 Pass Manager）➡
