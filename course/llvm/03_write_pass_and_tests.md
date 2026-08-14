# LLVM 第 3 章：写一个 Pass、接入 opt、编写测试——从零到可提交的最小闭环

## 1. 本章目标

- 能把一个只读诊断 pass 从源码一路做到 `opt -passes='count-add'` 可运行；
- 能解释插件注册回调的机制（`opt` 如何从字符串找到你的 pass）；
- 能用 FileCheck 写出有约束力的测试（CHECK-LABEL / CHECK-NEXT / CHECK-NOT / 变量捕获）；
- 能把诊断 pass 改造成真正修改 IR 的变换 pass，并正确处理"替换 uses、删除指令、保留分析"三件事；
- 能跑 `llvm-lit`，并用 `-verify-each` 定位是哪一步破坏了 IR。

前置：第 1 章的对象模型、第 2 章的 PassManager 契约。工具：LLVM 源码树或装了 dev 包的 LLVM（`llvm-config` 可用）。本章代码基于 LLVM 15+ 的插件 API，具体以当前 checkout 的 `PassPlugin.h` 为准。

## 2. 从最小需求开始：拆成四步

需求：统计每个函数里 `add` 指令的数量。先做**只读**版本，是为了把四件独立的事拆开练：

```text
① 遍历 IR(找到所有 add)
② 注册 pass(让 opt 按名字找到它)
③ 运行并验证输出(命令与结果)
④ 写测试(让正确性可持续)
```

拆开的原因：新手最常见的失败是把四件事糊在一起，报错时分不清是"遍历错了"还是"注册错了"。四步各自独立验证，哪步错改哪步。

## 3. Pass 主体：逐行对应对象

```cpp
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

struct CountAddPass : PassInfoMixin<CountAddPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    unsigned Count = 0;
    for (BasicBlock &BB : F)                    // Function 里的 BasicBlock 列表
      for (Instruction &I : BB)                 // BasicBlock 里的指令链表
        if (I.getOpcode() == Instruction::Add)  // 按 opcode 判断
          ++Count;

    errs() << "count-add: " << F.getName() << " = " << Count << "\n";
    return PreservedAnalyses::all();            // 只读, 什么都没改(第 2 章契约)
  }
};
```

逐行对象：`F` 是 `Function&`（函数级 pass 的粒度），`BB` 是 `BasicBlock&`，`I` 是 `Instruction&`。判断指令种类用 `getOpcode()` 与 `Instruction::Add` 枚举比较——`add` 在 LLVM 里是一个 opcode，`BinaryOperator` 的 `add/sub/mul/...` 都挂在同一张 opcode 表上。返回 `all()` 的理由在第 2 章：只读不写，全部分析仍然有效。

## 4. 插件注册：opt 怎么从字符串找到你的 pass

pass 类写完还不够——`opt -passes='count-add'` 里的字符串要能解析到 `CountAddPass` 这个类型。机制是**注册一个 pipeline 解析回调**：

```cpp
extern "C" PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "CountAdd", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, FunctionPassManager &FPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "count-add") {
            FPM.addPass(CountAddPass());
            return true;                    // 这个名字我们认领了
          }
          return false;                     // 不认识, 让别的注册者试
        });
    }};
}
```

机制拆解：`opt` 解析 pipeline 字符串时，把每个名字依次交给所有插件注册的回调；谁认识这个名字，谁就 `addPass` 并返回 `true`。**返回 `false` 表示"不是我家的名字"**，让后续插件继续试；所有回调都不认识时，opt 报"未知 pass"。三处版本敏感点：`LLVM_PLUGIN_API_VERSION`（插件 ABI 版本，不匹配直接拒绝加载）、`LLVM_VERSION_STRING`（可读版本号）、回调签名本身——**编译时以当前 checkout 的 `PassPlugin.h` 为准**，不要拼不同版本的示例。

## 5. 构建与运行

在 LLVM 源码树内用 CMake 函数：

```cmake
add_llvm_pass_plugin(CountAdd CountAdd.cpp)
```

独立项目则用 `llvm-config` 拿参数（【可运行代码】）：

```bash
g++ -shared -fPIC CountAdd.cpp -o CountAdd.so \
  $(llvm-config --cxxflags --ldflags) $(llvm-config --libs core passes)
```

三个参数各是什么：`--cxxflags` 是编译 LLVM 时的 C++ 标准与 include 路径；`--ldflags` 是链接选项（如 rpath）；`--libs core passes` 是需要的组件库。运行：

```bash
opt -load-pass-plugin ./CountAdd.so \
  -passes='count-add' input.ll -disable-output
# 预期输出: count-add: f = 2
```

平台差异：Windows 是 DLL、macOS 是 `.dylib`，扩展名与路径写法不同；先用 `opt --help` 确认当前版本支持 `-load-pass-plugin`。常见失败"找不到 `count-add`"的排查顺序：① 插件是否加载成功（有没有加载报错）；② 回调返回 true 了吗；③ 名字拼写与回调里 `Name ==` 是否一字不差。

## 6. FileCheck 测试：输入、命令、期望三合一

LLVM 的测试惯例是把**输入、执行命令、期望输出**放在同一个 `.ll` 文件里（【可运行代码】）：

```llvm
; RUN: opt -load-pass-plugin %shlibdir/CountAdd%shlibext \
; RUN:   -passes=count-add %s -disable-output 2>&1 | FileCheck %s

define i32 @f(i32 %x) {
entry:
  %a = add i32 %x, 1
  %b = add i32 %a, 2
  ret i32 %b
}

; CHECK: count-add: f = 2
```

三行语义：`RUN` 行是 lit 要执行的命令（`%s` 替换为本文件路径、`%shlibdir/%shlibext` 替换为插件目录与平台扩展名）；`| FileCheck %s` 把命令输出交给 FileCheck；`CHECK` 行声明"输出中必须按顺序出现这一行"。**只测"命令退出码为 0"是不够的**——pass 输出错了数字，命令照样返回 0，只有 CHECK 才把"输出内容"钉死。

FileCheck 的四个常用原语：

| 原语 | 语义 | 典型用法 |
|---|---|---|
| `CHECK:` | 输出中依次出现此行 | 基本断言 |
| `CHECK-LABEL:` | 匹配到函数/节的开头，把匹配范围分区 | 多个函数各自检查，互不串扰 |
| `CHECK-NEXT:` | 必须紧跟在上一匹配的下一行 | 钉死输出的**顺序**与**邻接** |
| `CHECK-NOT:` | 两个检查之间不得出现 | 负向断言（如"不该有这条指令"） |
| `[[VAR:pattern]]` | 捕获一段输出给后面引用 | 检查"同一个值出现两次"（如寄存器编号） |

一个顺序敏感的例子：如果 CHECK 写 `count-add: f = 2` 而实际输出是 `f = 3`，FileCheck 失败并打印差异位置——**测试失败信息本身会告诉你第几行不匹配**，这就是定位入口。

## 7. lit 与调试：负向实验最值钱

```bash
llvm-lit path/to/test.ll     # 跑单个测试或整个目录
llvm-lit -a path/to/test.ll  # 显示完整命令与输出(排查 RUN 行问题)
llvm-lit -j 1 ...            # 单线程, 日志顺序可读
```

两个排查专用选项：

- `-verify-each`：在 pipeline 每个 pass 之间插入 verifier——IR 在哪一步被破坏，错误就停在哪一步；
- `-debug-pass-manager`：打印实际运行了哪些 pass 与失效信息（第 2 章第 7 节）。

写测试的正确姿势：**先写负向测试**——故意让 pass 输出错误数字，确认测试真的会失败；再改回正确实现，确认测试通过。一个从来不会失败的测试和没有测试等价。

## 8. 从诊断 pass 变成变换 pass：`add x, 0` → `x`

真正的 pass 要改 IR。以简化 `add x, 0` 为例：

```cpp
struct SimplifyAddZeroPass : PassInfoMixin<SimplifyAddZeroPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    for (BasicBlock &BB : F)
      for (Instruction &I : llvm::make_early_inc_range(BB)) {
        auto *Add = dyn_cast<BinaryOperator>(&I);
        if (!Add || Add->getOpcode() != Instruction::Add)
          continue;
        Value *L = Add->getOperand(0), *R = Add->getOperand(1);
        if (auto *C = dyn_cast<ConstantInt>(R))
          if (C->isZero() && Add->getType() == L->getType()) {
            Add->replaceAllUsesWith(L);   // ① 所有用到 add 结果的地方改用 L
            Add->eraseFromParent();       // ② 从基本块链表中摘除并释放
          }
      }
    return PreservedAnalyses::none();     // ③ 保守声明(改了指令, 且没有精确分析)
  }
};
```

三个关键动作的对象语义：

**① `replaceAllUsesWith(L)`**：把"所有以这条 add 为操作数的指令"里的引用全部改指 L——对应第 1 章说的 use 列表。**漏掉这一步就删指令 = 悬空引用**，verifier 立刻报 `use of undefined value`；
**② `eraseFromParent()`**：把指令从所属 BasicBlock 的链表里摘除并释放。注意遍历时删元素会失效迭代器——示例用 `make_early_inc_range` 先自增再处理；
**③ 返回 `none()`**：改了指令，保守地让所有分析失效（第 2 章的契约；精确一点可以只保留 CFG 类分析）。

五条正确性检查（对照原版需求）：

1. 两个操作数类型是否相同（`add i64 %x, i32 0` 不存在，但指针/向量要小心）；
2. `nsw`/`nuw` 等语义标志：`add nsw x, 0` 变成 `x` 时丢掉的只是标志本身，但**如果优化依赖这些标志证明不溢出，就必须保留或传播**；
3. debug location 与 metadata：变换后指令的调试信息要么转移、要么说明丢弃；
4. 删除前所有用户是否已更新（① 是它的检查点）；
5. 返回的 `PreservedAnalyses` 是否诚实（③）。

这就是"能遍历 IR"与"能维护编译器正确性"的分界线：前者读数据，后者维护**不变量**。

## 9. 源码阅读地图

- `llvm/tools/opt/`：opt 如何建上下文、解析 pipeline、跑 pass；
- `llvm/lib/Passes/PassBuilder.cpp`：pipeline 字符串的解析与注册表；
- `llvm/include/llvm/Passes/PassPlugin.h`：动态插件入口的 ABI；
- `llvm/utils/FileCheck/`：FileCheck 工具的实现；
- `llvm/utils/llvm-lit/`：lit 启动器；
- 各子目录的 `test/`：真实 pass 的测试写法（`CHECK-LABEL` 的教科书用法都在这里）。

## 10. 常见错误与归因

| 现象 | 根因 | 修正 |
|---|---|---|
| opt 报未知 pass | 回调没注册/返回 false/名字拼写错 | 按第 5 节三步排查顺序 |
| 插件加载失败 | 插件 ABI 版本与 opt 不匹配 | 用同一 checkout 的 LLVM 编译插件 |
| 变换后 verifier 报 undefined value | 删指令前没 `replaceAllUsesWith` | 先替换 uses 再 erase |
| 遍历中崩溃 | 删除当前指令后迭代器失效 | `make_early_inc_range` |
| 测试永远通过 | 只断言退出码，没 CHECK 内容 | 第 6 节 FileCheck + 负向测试 |
| 输出顺序乱导致测试误报 | 检查跨函数串扰 | `CHECK-LABEL` 分区 |

## 11. 本章检查点

完成以下四项才算通过本章：

1. 把 `count-add` 从源码跑到 lit 测试全绿（含一次故意改错数字、确认测试失败的负向实验）；
2. 加一个 `CHECK-LABEL` 让同一测试覆盖两个函数，且两个函数各自的 add 数不同；
3. 完成 `add x, 0 → x` 的变换，并在变换前后都跑 `opt -passes='verify'`，说明返回 `none()` 的理由；
4. 用 `-verify-each` 复现一次"中间某步破坏 IR"的定位（可以故意写一个坏 pass）。

## 12. 本章小结与下一步

到这里，你拥有了一条完整闭环：写 pass → 注册 → 运行 → 测试 → 定位。下一章（LLVM 04：后端、ABI、TableGen 与 MC）离开中端，回答"经过 O2 的 IR 如何变成目标机器码"——第 1 章的目标信息、第 2 章的分析体系在那里汇入指令选择与寄存器分配。

**导航**：⬅ [上一章](02_analysis_passes.md)（Analysis、Pass 与新 Pass Manager）　｜　[下一章](04_backend_abi_mc.md)（后端、ABI、TableGen 与 MC）➡
