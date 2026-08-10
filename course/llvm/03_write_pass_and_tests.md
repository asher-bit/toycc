# LLVM 第 3 章：写一个 Pass、接入 `opt`、编写测试

## 1. 从一个最小需求开始

我们实现一个只读诊断 Pass：统计每个函数中 `add` 指令的数量。先做只读版本，是为了把“遍历 IR”“注册 Pass”“验证输出”“编写测试”拆开，后面再改成真正的变换。

## 2. Pass 主体

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
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        if (I.getOpcode() == Instruction::Add)
          ++Count;

    errs() << "count-add: " << F.getName() << " = " << Count << "\\n";
    return PreservedAnalyses::all();
  }
};
```

真实项目还需要把这个类型注册到插件入口。不同 LLVM 版本的 CMake 和插件宏可能不同，原则是：插件向 `PassBuilder` 注册一个 pipeline parsing callback，使 `-passes='count-add'` 能找到它。

```cpp
extern "C" PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "CountAdd", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, FunctionPassManager &FPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "count-add") {
            FPM.addPass(CountAddPass());
            return true;
          }
          return false;
        });
    }};
}
```

这段代码展示接口形状；编译时要以当前 checkout 的 `PassPlugin.h` 定义为准，不要把不同版本的示例拼在一起。

## 3. 构建与运行

在 LLVM 源码树内，通常用 `add_llvm_pass_plugin`；独立项目则使用 `llvm-config --cxxflags --ldflags --libs core passes` 获取编译和链接参数。运行形态类似：

```bash
opt -load-pass-plugin ./CountAdd.so \
  -passes='count-add' input.ll -disable-output
```

如果插件是 Windows DLL 或 macOS 动态库，扩展名和路径写法会不同。先用 `opt --help` 确认当前版本是否支持 `-load-pass-plugin`。

## 4. FileCheck 测试

LLVM 测试一般把输入、执行命令和期望输出放在同一个 `.ll` 文件中：

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

`RUN` 说明如何执行测试；`CHECK` 说明输出必须包含什么。不要只测试“命令退出码为 0”，那样 Pass 输出错误时测试仍可能通过。

## 5. `lit` 测试系统与定位手段

- `llvm-lit path/to/test.ll`：运行单个测试或目录；
- `-a`：显示完整命令和输出；
- `-j 1`：降低并发，便于观察日志；
- `-verify-each`：在 Pass 之间运行 verifier，快速定位哪一步破坏了 IR；
- `-debug-pass-manager`：查看 pipeline 实际运行了哪些 Pass。

当输出不稳定时，优先把检查条件写得具体但不过度依赖无关格式。需要检查结构时，可组合 `CHECK-LABEL`、`CHECK-NEXT`、`CHECK-NOT` 和变量捕获。

## 6. 从诊断 Pass 变成变换 Pass

例如将 `add x, 0` 替换为 `x`，需要考虑：

1. 两个操作数的类型是否相同；
2. 整数加法的语义属性（如 `nsw`、`nuw`）是否能安全丢弃；
3. 指令是否带 debug location、metadata 或其他语义信息；
4. 删除旧指令后，哪些用户要被更新；
5. 修改后应返回哪些 `PreservedAnalyses`，并在测试中检查 IR。

这也是从“能遍历 IR”到“能维护编译器正确性”的分界线。

## 7. 源码阅读地图

- `llvm/tools/opt/`：`opt` 工具如何建立 LLVM 上下文并运行 pipeline；
- `llvm/lib/Passes/PassBuilder.cpp`：Pass 名称与 pipeline 的解析；
- `llvm/include/llvm/Passes/PassPlugin.h`：动态 Pass 插件入口；
- `llvm/utils/FileCheck/`：FileCheck 工具；
- `llvm/utils/llvm-lit/`：lit 启动器；
- 各子目录下的 `test/`：真实 Pass 的输入与回归测试。

## 8. 练习

1. 加一个 `CHECK-LABEL`，让测试同时覆盖两个函数；
2. 写一个统计 `load` 数量的 Module/Function Pass；
3. 写一个安全的 `add x, 0` 简化，并在变换前后都运行 verifier；
4. 用 `llvm-reduce` 将一个失败测试缩减成最小输入。

参考：[Writing an LLVM Pass](https://llvm.org/docs/WritingAnLLVMPass.html)、[FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html)、[LLVM Testing Infrastructure Guide](https://llvm.org/docs/TestingGuide.html)。

