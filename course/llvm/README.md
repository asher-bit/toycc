# LLVM 深入专题

这一专题是[第 25 课 · LLVM 深入](../lesson25.md)的源码与实践分章。目标不是记住 LLVM 的名词，而是建立一条可以反复使用的阅读路径：

```text
LLVM IR / SSA
    ↓
分析结果与 Pass 管理
    ↓
自己写 Pass 并用 FileCheck 验证
    ↓
TargetMachine → MI → MC → 汇编/目标文件
```

## 学习顺序

1. [第 1 章：LLVM IR、SSA 与验证器](01_ir_ssa.md)
2. [第 2 章：Analysis、Pass 与新 Pass Manager](02_analysis_passes.md)
3. [第 3 章：写一个 Pass、接入 opt、编写测试](03_write_pass_and_tests.md)
4. [第 4 章：后端、ABI、TableGen 与 MC 层](04_backend_abi_mc.md)

## 建议的源码阅读顺序

先从 `llvm/include/llvm/IR/` 看数据结构，再看 `llvm/lib/IR/Verifier.cpp` 和 `llvm/include/llvm/IR/PassManager.h`。有了 IR 和 Pass 的基础后，再选择一个简单目标后端，沿着 `TargetMachine`、`TargetInstrInfo`、`TargetRegisterInfo`、`AsmPrinter` 追到底。

## 需要准备的工具

```text
opt llvm-as llvm-dis llc llvm-mc llvm-objdump llvm-readobj FileCheck llvm-lit
```

不同发行版的工具名称和可用参数可能略有差异，动手时以 `--help` 和当前 checkout 的 LLVM 版本为准。

## 官方入口

- [LLVM Language Reference](https://llvm.org/docs/LangRef.html)
- [Using the New Pass Manager](https://llvm.org/docs/NewPassManager.html)
- [Writing an LLVM Pass](https://llvm.org/docs/WritingAnLLVMPass.html)
- [Writing an LLVM Backend](https://llvm.org/docs/WritingAnLLVMBackend.html)
- [The LLVM Target-Independent Code Generator](https://llvm.org/docs/CodeGenerator.html)

