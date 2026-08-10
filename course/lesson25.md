# 第 25 课：LLVM 深入——给自研芯片写汇编器的标准做法

> 本课风格：从"llvm-project 到底是什么"讲起，逐层拆到"怎么用它写你们芯片的后端"。
> 目的：LLVM 不只是"一个编译器"——它是一套**可复用的编译器基础设施**。
> 自研 GPU 芯片要不要写汇编器？用 LLVM MC 能省 80% 的活。
> 前置：第 14 课（IR 家族）、第 24 课（工具链全景）、附录 A（C++）。

> 想继续深入，请按章节学习：[LLVM 深入专题](llvm/README.md)：
> [IR/SSA](llvm/01_ir_ssa.md) → [Analysis/Pass](llvm/02_analysis_passes.md) →
> [写 Pass 与测试](llvm/03_write_pass_and_tests.md) → [后端/ABI/MC](llvm/04_backend_abi_mc.md)。

---

## 1. 先破除一个误解："LLVM"不是一个编译器

很多人以为 LLVM 是"一个 C 编译器"。不是。**llvm-project 是一个仓库，里面装了一整套编译器工具**：

```
llvm-project/
├── llvm/        ← 核心：LLVM IR + 优化器 + 后端框架 + MC 层
├── clang/       ← C/C++ 前端(把 C++ 转成 LLVM IR)
├── mlir/        ← MLIR(第 26 课)
├── lld/         ← 链接器
├── lldb/        ← 调试器
├── libcxx/      ← C++ 标准库
└── ... (flang/polly/openmp 等)
```

**对你的意义**：你不用"用 LLVM 编译 C"，你要**复用 LLVM 的零件**——
特别是 **MC 层（Machine Code）**，它就是"写汇编器/反汇编器/目标文件输出"的框架。

---

## 2. LLVM IR 再深一层：第 14 课没讲的三个关键

第 14 课讲了 LLVM IR 是 SSA + 强类型 + 虚拟寄存器。这里补三个"看懂真实 .ll 文件必备"的语法。

### 2.1 控制流：基本块 + phi 节点

LLVM IR 里**函数由基本块（basic block）组成**，块之间用跳转连接：

```llvm
define i32 @max(i32 %a, i32 %b) {
entry:                          ; ← 基本块标签
  %cmp = icmp sgt i32 %a, %b    ; 比较 a > b, 结果是 i1
  br i1 %cmp, label %then, label %else   ; 条件跳转

then:
  br label %merge               ; 无条件跳到 merge

else:
  br label %merge

merge:
  %r = phi i32 [ %a, %then ], [ %b, %else ]  ; phi: 看从哪个块来, 选哪个值
  ret i32 %r
}
```

**逐行讲**：
- `entry:` `then:` `else:` `merge:` 是**基本块**——内部顺序执行，无分支
- `icmp sgt` = 整数比较（signed greater than）
- `br i1 %cmp, label %then, label %else` = if/else 的底层形态
- **`phi` 节点是 SSA 的命门**：一个值"从 then 来就是 %a，从 else 来就是 %b"。
  SSA 要求每个变量只赋值一次，但 if/else 两个分支都可能给结果赋值——
  phi 就是"在汇合点，根据来路选一个"。

**对你的意义**：你的模型里的 `if/while`（第 17 课控制流）降到这一层，
就会变成基本块 + phi。看懂 phi，就看懂了所有带控制流的编译产物。

> **原理深挖：为什么 SSA 非要有 phi，别处能不能躲掉？**
>
> 直觉上"if/else 的结果"很好办：提前初始化一个变量，分支里写它就行。
> 但这**恰恰是 SSA 禁止的**——SSA 的核心是"每个值只被赋值一次"，
> 因为只有这样，优化器（第 5 课的死代码消除、重命名）才能确定
> 每个变量的**唯一来源**，从而大胆做 copy propagation、再排序。
>
> 如果不引入 phi，数据流分析就要在函数里建"哪个分支写了它"的
> 联合集合，复杂度爆炸。**phi 把这种"多来源"的语义，变成 IR 里
> 一个显式的节点**，让所有数据流优化都变得局部而简单。
>
> **换个角度记**：SSA + phi = "每个值一个主人，汇合处配一个仲裁"。
> 假如你写模型时用"提前初始化"的普通写法，编译器把它转成 SSA 时，
> 就会**自动插入 phi**。所以 phi 不是你手写的负担，而是 IR 的
> 一种"诚实记录"——它把"此处有多个可能的来源"这句真相，画在了图上。
> 看懂这一点，你读 .ll 文件时，对 phi 就不再是死记语法，而是理解
> "这里是 if/else 的汇合点，值有两个候选来源"。

### 2.2 内存访问：getelementptr（GEP）

LLVM 没有"数组下标"这种高层概念，访存全部通过 `getelementptr`（算地址）：

```llvm
%ptr = getelementptr inbounds [4 x i32], ptr %arr, i32 0, i32 %idx
%val = load i32, ptr %ptr
```

**逐行讲**：
- `getelementptr [4 x i32], ptr %arr, i32 0, i32 %idx` = "在 `%arr` 这个数组里，
  取第 0 维的第 `idx` 个元素的地址"
- GEP 是**纯地址计算，不读内存**；`load` 才真正读
- `inbounds` 是编译器承诺"不会越界"，越界是未定义行为（UB）

**关键认知**：GEP 的每个"下标"都有**类型**——LLVM 靠类型算出
"该往下跳多少字节"。这就是"强类型"的实际作用。

### 2.3 属性（attributes）：指令/函数上的"编译器提示"

```llvm
declare i32 @puts(ptr captures(none)) nounwind
```

- `captures(none)`：这个指针不会被函数"偷走"（逃逸分析用）
- `nounwind`：这个函数不会抛异常（省掉异常处理代码）

属性是**编译器之间的暗语**——前端告诉后端"可以放心优化"。

---

## 3. LLVM 的 pass 体系：和你 toycc 的 pass 一脉相承

第 3 课你写了 toycc 的融合 pass。LLVM 的 pass 体系是**同一思想、工业级实现**。

### 3.1 三种 pass 层级

| 层级 | 作用范围 | 例子 |
|---|---|---|
| **ModulePass** | 整个模块（所有函数） | 内联、全局常量合并 |
| **FunctionPass** | 单个函数 | 死代码消除、循环优化 |
| **BasicBlockPass** | 单个基本块 | 窥孔优化（极少用） |

**对应 toycc**：你的 `FusionPass.__call__(graph)` 就是一个 ModulePass。

### 3.2 新 pass 管理器（New Pass Manager）

LLVM 现在用的是"新 PM"。一个 FunctionPass 长这样（C++）：

```cpp
struct MyDCE : public PassInfoMixin<MyDCE> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    bool changed = false;
    for (BasicBlock &BB : F)
      for (Instruction &I : llvm::make_early_inc_range(BB))
        if (isInstructionTriviallyDead(&I)) {   // LLVM 自带的判断
          I.eraseFromParent();
          changed = true;
        }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};
```

**逐行讲**：
- `PassInfoMixin<MyDCE>`：注册 pass 的"身份证"模板
- `run(Function &F, ...)`：对一个函数跑这个 pass
- `PreservedAnalyses`：告诉框架"我改了什么，哪些之前的分析还作数"——
  这就是 toycc 没有的**增量分析缓存**机制
- `make_early_inc_range`：边删边遍历的安全迭代器

**重点**：`isInstructionTriviallyDead` 这种判断，LLVM **已经写好了**——
你写 pass 是**组装现成的分析**，不是从零写判断。这就是复用的力量。

### 3.3 分析 pass（Analysis） vs 变换 pass（Transform）

```
分析 pass: 只算不改 (例: 活跃变量分析、支配树分析)
变换 pass: 用分析结果去改 IR (例: 死代码消除 = 用活跃分析去删)
```

LLVM 强制分离两者——变换 pass **依赖**分析 pass，框架自动算好顺序和缓存。
toycc 里 `_is_constant` 那种"边查边改"是简化版，工业级要分开。

---

## 4. LLVM 后端：从 IR 到机器码，到底发生了什么

这是你最关心的部分——**给自研芯片写后端，就是写这段**。

### 4.1 后端流水线（一条指令的旅程）

```
LLVM IR
  │ Instruction Selection (指令选择)
  ▼
SelectionDAG / GlobalISel   ← 把 IR 指令映射成"目标指令的候选"
  │ Instruction Scheduling (指令调度)
  ▼
MachineInstr (带真实指令, 但寄存器还是虚拟的)
  │ Register Allocation (寄存器分配)
  ▼
带真实寄存器的 MachineInstr
  │ MC 层: 编码成字节 / 输出汇编文本 / 输出目标文件
  ▼
机器码 (你们芯片的指令)
```

**对照 toycc**：你的 `codegen` 直接"图 → C 文本"，跳过了中间所有这些。
真实后端要一层层来。

### 4.2 指令选择：TableGen 描述"这条 IR 用哪条目标指令"

LLVM 用 **TableGen（.td 文件）** 声明式地写指令。比如描述一个加法指令：

```tablegen
def ADDrr : Instruction {
  let AsmString = "add $rd, $rs1, $rs2";   // 汇编文本长什么样
  let Constraints = "$rd = $rs1";           // 约束(比如复用寄存器)
  dag OutOperandList = (outs GPR:$rd);      // 输出操作数
  dag InOperandList  = (ins GPR:$rs1, GPR:$rs2);  // 输入
}
```

**关键认知**：TableGen 是**"用数据描述指令"**——你填一张表（名字/操作数/编码），
LLVM 自动生成"指令选择 + 汇编打印 + 编码"三份 C++ 代码。
**自研芯片的第一步，就是把你们 ISA 手册翻译成一堆 .td 文件。**

### 4.3 寄存器分配：图着色（衔接第 6 课）

第 6 课内存规划用的"生命周期 + 复用"，LLVM 寄存器分配是它的**升级版**：

```
虚拟寄存器 (无限)  →  物理寄存器 (有限, 比如 32 个)
约束: 同一时刻"活着"的虚拟寄存器不能共用同一个物理寄存器
方法: 图着色 (寄存器=颜色, 冲突=连边, 不能同色)
装不下: spill (写回内存, 第 3 课"寄存器压力"的落地)
```

LLVM 内置了好几种分配器（greedy/fast/PBQP），你不用自己写图着色。

---

## 5. MC 层：写"你们芯片的汇编器"就是用它

### 5.1 MC 是什么

MC（Machine Code）层是 LLVM 最底层的库，提供：
- **指令的内存表示**（MCInst）
- **汇编器**：把汇编文本 → MCInst → 字节
- **反汇编器**：字节 → 汇编文本
- **目标文件输出**：ELF/COFF/Mach-O
- **重定位**（符号地址占位，链接时填）

**一句话**：MC 把"指令"这件事彻底数据化了。你只要用 TableGen 描述
你们芯片的每条指令（怎么编码成字节），MC 帮你把剩下的活全干了。

> **手算：MC 数据流——一行汇编变字节，中间多了个"对象"**
>
> ```
> 汇编文本            MCInst(结构化对象)         字节
> "add r1, r2, r3" →  {opcode: ADD,            → 00 21 00 13
>                      operands: [r1,r2,r3]}      (4 字节)
>    (解析器)          (编码器, 由 TableGen 生成)
> ```
>
> 为什么中间要有一个 MCInst 而不是直接文本→字节？三个原因：
> 1. **可逆**：字节 → MCInst → 文本（反汇编器免费，5.3 节）
> 2. **可分析**：调试器/模拟器都吃 MCInst（第 24 课 debugger 就能
>    直接复用 MC 的反汇编输出，不用自己再写一套）
> 3. **可重定位**：MCInst 里的符号（如 `call foo`）先留占位，
>    链接时填地址——这就是第 24 课 linker 能拼模块的基础
>
> 你的芯片加 LLVM 后端时，编码规则写在 TableGen（`let AsmString` 旁边
> 那个 `let Encoding = ...`），MC 自动生成编码器和解码器，
> **编码/解码不一致的 bug 在生成阶段就被消掉了**。

### 5.2 给自研芯片加 LLVM 后端的清单

这就是你入职后可能真要做的事（对照第 24 课"五步"）：

```
1. 写 TableGen 描述你们 ISA 的每条指令 (.td)
2. 定义寄存器类 (RegisterClass: 哪些寄存器能干什么)
3. 定义调用约定 (CallingConv: 参数/返回值怎么传)
4. 写指令选择模式 (SelectionDAG pattern 或 GlobalISel)
5. 实现 TargetLowering / TargetInstrInfo 等 C++ 钩子
6. 用 MC 自动生成汇编器/反汇编器
7. 跑 LLVM 的 lit 测试 (FileCheck 验证生成的汇编对不对)
```

**实际工作量**：一个"能跑简单算术"的最小后端，熟练的人 1~2 周；
一个"生产可用"的后端，数月。**但比从零写汇编器省几个数量级**。

### 5.3 反汇编器 = 免费的副产品

因为你用 TableGen 描述了"指令 ↔ 字节"的编码，MC **自动**给你反汇编器。
这对调试极其重要——生成的机器码对不对，反汇编一下就知道。

---

## 6. LLVM vs TVM vs 你的自研芯片：谁管哪段

```
模型 (PyTorch/ONNX)
  │ 前端 (TVM/Relax 或 MLIR)
  ▼
图优化 (融合/布局/量化)          ← TVM 的事 (第 3-6 课)
  │ FuseTIR / 下降
  ▼
TIR 调度 (分块/向量化/bind)      ← TVM 的事 (第 11 课)
  │ codegen
  ├─→ 出 PTX (用 NVIDIA 的)      ← 现有 GPU 路线
  └─→ 出你们芯片的指令            ← 你的事!
        │ 这里要 LLVM 后端 + MC
        ▼
      你们芯片的机器码
```

**判断题**（入职第一周就要想清楚的）：
- 如果芯片有**现成 C 编译器**：让 TVM 出 C 源码，走那条编译器（最省事）
- 如果芯片**没有任何编译器**：用 LLVM MC 写后端（本课）
- 如果芯片像 GPU：可能要自己设计一层"类 PTX" + 自己的 ptxas（第 22 课）

---

## 7. 本课小结

- llvm-project 是一整套基础设施：**IR + 优化器 + 后端框架 + MC**
- LLVM IR 深入：**基本块 + phi**（控制流）、**GEP**（访存）、**属性**（编译器提示）
- pass 体系：**分析/变换分离** + **新 pass 管理器**（PreservedAnalyses 缓存）
- 后端流水线：指令选择 → 调度 → 寄存器分配 → MC 编码
- **MC 层 = 写汇编器的标准框架**：TableGen 描述指令，自动生成汇编/反汇编/目标文件
- 自研芯片后端清单：ISA→.td → 寄存器类 → 调用约定 → 指令选择 → MC → lit 测试

**下一步**：第 26 课——MLIR 深入。LLVM 是"单层 IR + 后端框架"，
MLIR 是"多层可扩展 IR"——你第 14 课见过它的方言，这课把它讲到能上手。

---

## 8. 深层拓展 A：为什么 LLVM 后端这么"重"？

你可能会问：toycc 的 codegen 才 200 行，LLVM 后端为什么这么复杂？

**因为目标不同**：
- toycc 只生成"能跑的教学代码"，不用管寄存器分配、指令调度、编码
- LLVM 后端要生成**生产级机器码**：省寄存器、排指令、处理所有 ISA 细节

**但你可以"按需取用"**：只用 MC 层写汇编器，不用写完整指令选择，
就是一个"小而美"的开始。很多自研芯片团队就是这么起步的。

---

## 9. 深层拓展 B：SelectionDAG vs GlobalISel——两条指令选择路线

LLVM 有两套指令选择框架：
- **SelectionDAG**（老，主流）：把 IR 建成有向无环图，匹配模式
- **GlobalISel**（新）：更模块化、更快，但部分后端还没完全迁移

新后端（比如一些 RISC-V）直接用 GlobalISel。你做自研芯片时，
**推荐直接上 GlobalISel**——它是未来，且写起来更现代。

---

## 10. 思考题

1. 为什么 LLVM IR 用"基本块 + phi"而不是直接允许任意跳转？（提示：SSA）
2. `getelementptr` 和 `load` 的区别是什么？为什么 LLVM 要分开？
3. 为什么说"用 TableGen 描述指令"能同时得到汇编器和反汇编器？
4. 你们芯片如果只有 16 个寄存器，寄存器分配会比 32 个寄存器难在哪？

> 答案：1) phi 让 SSA 在汇合点也能"每个值只定义一次"，支配关系清晰，优化好写。
> 2) GEP 只算地址不读内存，load 才读——分开让编译器能单独优化地址计算。
> 3) 因为编码是双向的：汇编=编码，反汇编=解码，同一份 .td 驱动两个方向。
> 4) 寄存器少 → 更容易冲突 → spill 更多 → 要写回内存，性能掉；分配器要更聪明。

---

**导航**：⬅ [上一节](lesson24.md)（第 24 课 · 自研 GPU 工具链全景）　｜　[下一节](lesson26.md)（第 26 课 · MLIR 深入）➡
