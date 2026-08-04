# toycc —— 迷你 AI 编译器 + 完整教学课程

一个几百行的可运行 AI 编译器骨架，以及配套的 20 课手把手课程。
**它模仿 TVM 的架构**，用来学"AI 编译器到底在做什么"，然后能读得懂真实源码、参与开发。

## 快速开始

```bash
python -m course.runner 1       # 第1课实验
python -m toycc.examples.demo   # 完整流水线一键演示
```

`demo` 会打印：10 个算子 → 融合/布局/常量折叠 → 6 个算子，内存省 70%，
生成的代码与参考执行结果完全一致（max|Δ|=0）。

## 项目结构

```
toycc/          迷你 AI 编译器（模仿 TVM 架构）
├── ir/         计算图 IR（对应 Relax/ONNX）
├── passes/     优化 pass（融合/布局/常量折叠/内存/DCE/量化）
├── codegen/    代码生成（C + Python 双后端）
├── runtime/    numpy 参考执行器（正确性裁判）
├── schedule.py TIR 调度模拟器
└── hardware.py 缓存/延迟硬件模型
course/         20 课 + 3 附录 + 词汇表（Markdown 教学课程）
```

## 课程内容

覆盖：IR/Pass/后端、算子融合、布局优化、常量折叠、内存规划、代码生成、
真实 TVM 源码精读、TIR 调度、优化全景、自动调度（meta_schedule）、
LLVM/MLIR/PTX、硬件基础、量化、模型导入、性能加速、工程开发流程。

详细学习路径见 `course/README.md`。

## 要求

- Python 3.10+（本机 3.14）
- 只需要 numpy
