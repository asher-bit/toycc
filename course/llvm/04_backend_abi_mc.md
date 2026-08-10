# LLVM 第 4 章：后端、ABI、TableGen 与 MC 层

## 1. 本章目标

- 把 LLVM IR 到汇编/目标文件的链路串起来；
- 知道 `TargetMachine`、指令选择、Machine IR、寄存器分配和 MC 层分别负责什么；
- 读懂一个目标后端的 `.td` 文件和关键 C++ 类；
- 理解 ABI、调用约定、栈帧和目标文件格式为什么必须进入编译器。

## 2. 从 IR 到机器码

可以先用这张图定位源码：

```text
LLVM IR
  ↓ TargetMachine / target options
指令选择：SelectionDAG 或 GlobalISel
  ↓
Machine IR（MIR）
  ↓ 调度、窥孔优化、寄存器分配、栈帧布局
  ↓
MachineInstr
  ↓ AsmPrinter / MCCodeEmitter
汇编文本或目标文件
```

LLVM 后端不是简单的“把 opcode 换成字符串”。它要决定如何满足目标 ISA 的操作数约束、寄存器类、指令延迟、内存寻址、调用约定和重定位要求。

## 3. 后端关键组件

| 组件 | 主要问题 |
|---|---|
| `TargetMachine` | 目标、CPU、特性和数据布局如何组合 |
| `TargetSubtargetInfo` | 某一 CPU/子架构启用哪些特性 |
| `TargetInstrInfo` | 指令描述、拷贝、分支和指令级辅助操作 |
| `TargetRegisterInfo` | 物理寄存器、寄存器类、保留寄存器和调用保存规则 |
| `TargetLowering` | 通用 IR 操作如何变成目标可实现的节点/指令 |
| `AsmPrinter` | MachineInstr 如何打印成汇编 |
| MC 层 | 汇编解析、编码、解码、对象文件和重定位 |

## 4. TableGen：把声明变成后端基础设施

目标后端通常在 `llvm/lib/Target/<Target>/` 下使用 TableGen 描述寄存器和指令：

```text
Target.td          目标通用信息
RegisterInfo.td    寄存器与寄存器类
InstrInfo.td       指令、操作数、编码、调度信息
CallingConv.td     调用约定规则
<Target>Gen*.inc   由 llvm-tblgen 生成的 C++ 片段
```

`.td` 文件不是最终执行逻辑，而是声明式输入。`llvm-tblgen` 根据不同 backend 生成寄存器编号、指令描述、匹配模式和调度相关数据。读源码时要同时看 `.td` 的定义与生成文件的使用点。

## 5. ABI 是后端的边界协议

调用一个函数时，调用者和被调用者必须在这些问题上达成一致：

- 参数和返回值放在哪些寄存器或栈槽；
- 哪些寄存器由 caller 保存、哪些由 callee 保存；
- 栈指针对齐多少，栈帧如何建立；
- 可变参数、结构体返回、向量参数如何传递；
- 符号如何命名、地址如何重定位。

因此一个“能执行算术指令”的后端，离能编译真实程序还很远。ABI 规则通常分散在 `CallingConv.td`、`TargetLowering`、寄存器信息和 prologue/epilogue 代码中。

## 6. SelectionDAG、GlobalISel 与 MIR

SelectionDAG 以 DAG 节点表达一部分指令选择和合法化过程；GlobalISel 以通用 Machine IR 为中心，通常拆成 IRTranslator、Legalizer、RegBankSelect 和 InstructionSelect 等阶段。两套路径在 LLVM 中长期共存，具体目标可能支持程度不同。

调试后端时，可尝试：

```bash
llc -march=<target> -stop-after=instruction-select input.ll -o output.mir
llc -run-pass=greedy input.mir -o -
llvm-mc -triple=<target> input.s -filetype=obj -o input.o
llvm-objdump -d input.o
llvm-readobj -h -r input.o
```

参数和 pass 名称随版本及目标变化，先查看 `llc --help`。MIR 是连接“后端算法”和“最终汇编”的重要观察窗口。

## 7. MC 层到底解决什么

MC 层处理汇编语法、指令编码/解码、符号、重定位和目标文件相关信息。`llvm-mc` 可以独立验证汇编器/编码器，`llvm-objdump` 和 `llvm-readobj` 可以反向检查生成结果。

对自研芯片而言，MC 层通常是一个很好的切入点：先让 `.s` 能解析、指令能编码、目标文件能生成，再逐步接入指令选择和完整 ABI。这样每一层都有独立的可执行测试。

## 8. 源码阅读地图与练习

建议选择 RISC-V 或目标项目中最接近的后端，按以下顺序读：

1. `llvm/lib/Target/<Target>/<Target>TargetMachine.cpp`；
2. `RegisterInfo.td`、`InstrInfo.td` 和生成文件的引用；
3. `TargetLowering.cpp` 与调用约定；
4. `ISelLowering` / GlobalISel 入口；
5. `AsmPrinter` 和 `MCTargetDesc`；
6. `llvm/test/CodeGen/<Target>/` 与 `llvm/test/MC/<Target>/`。

练习：

- 用 `llc` 比较不同 `-mcpu` 的指令选择；
- 修改一份 `.s`，用 `llvm-mc` 观察编码错误；
- 找一个带重定位的目标文件，解释 `llvm-readobj -r` 的输出；
- 画出一个函数从调用约定进入到返回指令的寄存器/栈变化。

参考：[Writing an LLVM Backend](https://llvm.org/docs/WritingAnLLVMBackend.html)、[LLVM Code Generator](https://llvm.org/docs/CodeGenerator.html)、[GlobalISel](https://llvm.org/docs/GlobalISel/index.html)。

