# LLVM 第 4 章：后端、ABI、TableGen 与 MC 层——从 IR 到目标文件的可执行链路

## 1. 本章目标

- 能画出 LLVM IR → 指令选择 → MIR → 寄存器分配 → 汇编/目标文件的完整链路，并说出每层由哪个组件负责；
- 能读懂一个 `.td` 指令定义，并手算一条 RISC-V 指令的机器编码；
- 能说清 ABI 必须约定的五件事，并用 x86-64 System V 的例子逐条对应；
- 能解释 SelectionDAG 与 GlobalISel 的分工，以及 GlobalISel 四个阶段各干什么；
- 能用 `llc` / `llvm-mc` / `llvm-objdump` / `llvm-readobj` 观察后端各层产物。

前置：第 1~3 章的 IR 与 pass 体系。工具：LLVM ≥ 15，含目标后端（示例用 x86-64 与 RISC-V，后者更简单、适合手算编码）。

## 2. 工作中的问题长什么样

后端方向的三类日常问题：

```text
"TableGen 报错位置看不懂，.td 文件到底生成了什么？"
"为什么同样的 IR 在两个 CPU 上生成的汇编不一样？"
"自研芯片的后端，第一刀该从哪里切？"
```

三个问题对应三个主题：**TableGen**（声明如何变成代码）、**目标信息**（TargetMachine/Subtarget 如何改变后端行为）、**分层切入**（MC 层为什么是好起点）。本章逐个建立。

## 3. 完整链路：每层一句话定义

```text
LLVM IR
  ↓ TargetMachine / target options(目标、CPU、特性、数据布局)
指令选择: SelectionDAG 或 GlobalISel
  ↓
Machine IR(MIR: 用物理寄存器编号与目标指令表达的"接近机器"的 IR)
  ↓ 调度、窥孔优化、寄存器分配、栈帧布局
MachineInstr(后端的基本指令对象)
  ↓ AsmPrinter / MCCodeEmitter
汇编文本或目标文件
```

LLVM 后端不是"把 opcode 换成字符串"：它要决定如何满足目标 ISA 的操作数约束、寄存器类、指令延迟、内存寻址、调用约定与重定位要求。链路里每一步的观察窗口是 MIR——`llc -stop-after=<阶段>` 可以在任意一层停住看中间状态。

## 4. 后端关键组件：每个都回答一个问题

| 组件 | 一句话定义 | 典型对象/文件 |
|---|---|---|
| `TargetMachine` | 一个编译目标的入口：目标、CPU、特性、DataLayout 的组合 | `X86TargetMachine.cpp` |
| `TargetSubtargetInfo` | 某一 CPU/子架构启用哪些特性（指令集扩展、调度模型） | `X86Subtarget.h` |
| `TargetInstrInfo` | 指令级辅助操作：拷贝、分支、折叠 | `X86InstrInfo.cpp` |
| `TargetRegisterInfo` | 物理寄存器、寄存器类、保留寄存器、调用保存规则 | `X86RegisterInfo.td/.cpp` |
| `TargetLowering` | 通用 IR 操作如何变成目标可实现的节点/指令 | `X86ISelLowering.cpp` |
| `AsmPrinter` | MachineInstr 如何打印成汇编文本 | `X86AsmPrinter.cpp` |
| MC 层 | 汇编解析、指令编码/解码、对象文件与重定位 | `X86MCTargetDesc.cpp` |

读后端代码时先拿这张表定位：一个"参数怎么传"的问题通常在 `TargetLowering` 与 `CallingConv.td`，"寄存器为什么被保留"在 `TargetRegisterInfo`，"汇编长什么样"在 `AsmPrinter`。

## 5. TableGen：声明式描述如何变成 C++

**TableGen** 是 LLVM 的声明式 DSL：目标用 `.td` 文件描述寄存器与指令，`llvm-tblgen` 按不同 backend 生成 C++ 片段。一个最小指令定义（RISC-V 风格示意）：

```tablegen
// InstrInfo.td
def ADD : RVInstR<0b0000000, 0b000, OPC_OP, "add">;
// 含义: funct7=0000000, funct3=000, opcode=OPC_OP, 汇编名 "add"
// RVInstR 基类把字段拼进 32 位编码: [funct7][rs2][rs1][funct3][rd][opcode]
```

`.td` 文件**不是最终执行逻辑，而是生成器的输入**。`llvm-tblgen` 的不同 backend 各生成一类东西：寄存器编号表、指令描述表、指令选择匹配表、调度数据、汇编器/反汇编器的编码信息。读源码要**同时看 `.td` 的定义与生成文件的使用点**——`.td` 里一行 `def ADD`，生成文件里可能对应十几处。

### 5.1 手算一条 RISC-V 指令编码

用定义里的字段格式手算 `add x1, x2, x3`（R 型：funct7=0000000，rs2=3，rs1=2，funct3=000，rd=1，opcode=0110011）：

```text
位布局: [31:25] funct7 | [24:20] rs2 | [19:15] rs1 | [14:12] funct3 | [11:7] rd | [6:0] opcode
填值:    0000000        00011       00010        000          00001       0110011
拼成 32 位: 0000000 00011 00010 000 00001 0110011
= 0x0031 00B3
```

验证：`llvm-mc -triple=riscv64 -show-encoding` 输入 `add x1, x2, x3`，输出应打印 `encoding: [0xb3,0x00,0x31,0x00]`（小端字节序）。**这条手算链就是"读懂 .td 编码字段"的验收标准**——编码字段与手算对不上时，一定是 `.td` 读错了。

## 6. ABI：后端的边界协议

**ABI（应用二进制接口）**是调用者与被调用者之间的协议。x86-64 System V 的最小集合（示例架构，其余架构各有规则）：

1. **参数与返回值放哪**：前 6 个整数参数放 `rdi, rsi, rdx, rcx, r8, r9`，更多进栈；整数返回值在 `rax`；
2. **寄存器由谁保存**：caller-saved（`rax, rcx, rdx, rsi, rdi, r8-r11`，被调用者可以随便改）vs callee-saved（`rbx, rbp, r12-r15`，被调用者若使用必须先保存、返回前恢复）；
3. **栈对齐与栈帧**：`call` 指令执行前栈指针必须 16 字节对齐；栈帧的建立是 `push rbp; mov rbp, rsp; sub rsp, N` 三件套（保存旧帧指针、建立新帧指针、为局部变量留空间）；
4. **特殊类型**：可变参数、结构体返回（可能经隐藏指针）、向量参数各有专门规则；
5. **符号与重定位**：符号如何命名（加不加下划线/前缀）、地址如何重定位。

手算一个调用：`int f(int a, int b)` 被 `f(1, 2)` 调用时，调用者把 `1` 放进 `rdi`、`2` 放进 `rsi`，`call f`；被调用者若要用 `rbx`，必须 `push rbx` 保存、`pop rbx` 恢复——**callee-saved 的意思是"被调用者承诺还给调用者原值"**。为什么 ABI 必须进编译器：前端/中端只关心语义，后端必须把"传参"翻译成"寄存器/栈的精确摆法"，前后端分离的前提就是这份协议。一个"能执行算术指令"的后端离能编译真实程序还很远，ABI 规则通常分散在 `CallingConv.td`、`TargetLowering`、寄存器信息与 prologue/epilogue 代码里。

## 7. SelectionDAG 与 GlobalISel：两条指令选择路径

**SelectionDAG**：把 IR 转成 DAG 节点，在 DAG 上做合法化与模式匹配。**GlobalISel**：直接以 Machine IR 为中心，拆成四个阶段：

| 阶段 | 一句话定义 |
|---|---|
| IRTranslator | LLVM IR → 通用 MIR（还是"通用指令"，没有目标细节） |
| Legalizer | 把目标不支持的指令/类型拆成支持的（如 64 位加法拆两条 32 位） |
| RegBankSelect | 给虚拟寄存器指定寄存器组（通用组 / FP 组） |
| InstructionSelect | 通用指令 → 目标指令（用 TableGen 生成的匹配表） |

两套路径长期共存：SelectionDAG 成熟稳定，GlobalISel 结构更清晰、更适合新目标。**具体目标支持哪条，查该目标的配置**——新写后端通常先上 GlobalISel 或先做 SelectionDAG 的最小集，取决于团队积累。

调试命令（【可运行代码】，pass 名随版本变化，先 `llc --help`）：

```bash
llc -march=riscv64 -stop-after=instruction-select input.ll -o out.mir   # 在指令选择后停住, 看 MIR
llc -run-pass=greedy out.mir -o -                                       # 只跑寄存器分配 pass
llvm-mc -triple=riscv64 -show-encoding input.s                          # 独立验证汇编器与编码
llvm-objdump -d input.o                                                 # 反汇编目标文件
llvm-readobj -h -r input.o                                              # 看 ELF 头与重定位表
```

五个命令覆盖链路的后半段：MIR 观察、单 pass 重放、汇编器独立验证、目标文件反向检查、重定位细节。

## 8. MC 层：自研芯片的最佳切入点

MC 层解决五件事：**汇编语法解析、指令编码、反汇编解码、符号、重定位与目标文件信息**。`llvm-mc` 可以脱离整个后端独立验证汇编器/编码器（第 5.1 节就是这么用的）。

对自研芯片，MC 层是推荐的第一刀，因为它把风险切成可独立测试的薄片：

```text
第 1 刀: .s 能解析、指令能编码/解码(llvm-mc 单测)
第 2 刀: 目标文件能生成、重定位能写对(llvm-readobj 检查)
第 3 刀: 接指令选择(IR → 你的指令)
第 4 刀: 接 ABI(调用约定、栈帧)
```

每层都有独立的可执行测试，任何一层出错都不会污染其他层——与第 3 章"四步拆开练"是同一个方法论，只是规模换成后端。

## 9. 源码阅读地图

选 RISC-V（编码简单、代码量小）或目标项目里最接近的后端，按顺序读：

1. `llvm/lib/Target/<Target>/<Target>TargetMachine.cpp`：目标如何组装；
2. `RegisterInfo.td`、`InstrInfo.td`：寄存器与指令声明（对照第 5 节手算）；
3. `<Target>ISelLowering.cpp`：TargetLowering 与调用约定；
4. ISel 入口（SelectionDAG 或 GlobalISel 的四个阶段）；
5. `AsmPrinter` 与 `MCTargetDesc`：打印与 MC 集成；
6. `llvm/test/CodeGen/<Target>/` 与 `llvm/test/MC/<Target>/`：每层产物的回归测试。

## 10. 常见错误与归因

| 现象 | 根因 | 修正 |
|---|---|---|
| tblgen 报错读不懂 | `.td` 语法/字段名错，报错指向声明位置 | 对照同目标的已有定义 |
| 编码与手算不符 | `.td` 编码字段填错 | 第 5.1 节手算 + `llvm-mc -show-encoding` |
| 生成的汇编参数错乱 | 调用约定（CallingConv/ABI）配错 | 对照 ABI 文档逐寄存器核对 |
| 栈对齐崩溃 | prologue 没遵守 16 字节对齐 | 看 MIR 里栈帧分配，用 `llc -stop-after` 检查 |
| 反汇编结果不对 | 编码与解码表不一致 | `llvm-mc` 汇编+反汇编往返测试 |

## 11. 本章检查点

完成以下四项才算通过本章：

1. 手算 `add x5, x6, x7` 的 RISC-V 机器码，并用 `llvm-mc -show-encoding` 验证；
2. 用 `llc -stop-after=instruction-select` 生成一段 MIR，指出其中一条指令的虚拟寄存器与目标指令名；
3. 按第 6 节五条，写出 x86-64 上 `long f(long a, long b, long c)` 调用的参数寄存器分配与返回值位置；
4. 为自研芯片列出 MC 层第一刀需要交付的三样东西（可解析的 .s、可编码的指令、可检查的目标文件）及各自的验证命令。

## 12. 本章小结与下一步

本章把后端拆成了可观察的层：TableGen 声明、两条 ISel 路径、MIR、寄存器分配、ABI 与 MC。LLVM 专题到这里闭环：IR → Pass → 写 pass → 后端。下一站是 MLIR 专题（MLIR 01：Operation/Region/Block 与 Value）——LLVM IR 的"单一路径"在 MLIR 里变成"可扩展的多方言"，第 2 章的 pass 体系在那里获得"pattern rewrite"的更高层形态。

**导航**：⬅ [上一章](03_write_pass_and_tests.md)（写 Pass、接入 opt 与测试）　｜　本专题完，返回 [专题目录](README.md) ➡
