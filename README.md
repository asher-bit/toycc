# toycc —— 迷你 AI 编译器 + 机器学习编译器/GPU 工具链新人手册

一个几百行的可运行 AI 编译器骨架，以及面向新员工的机器学习编译器与 GPU 工具链知识体系。
**它以 TVM 为起点，但不止于 TVM**，覆盖 LLVM、MLIR、CUDA、PTX、CUTLASS、Triton、硬件、Runtime、Driver 和性能工具。

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
course/         新人手册、知识地图、岗位路径、27 课主线和各技术专题（Markdown）
```

## 课程内容

覆盖：模型导入、IR/Pass/后端、TVM、LLVM、MLIR、GPU 硬件、CUDA、PTX、Kernel、
CUTLASS、Triton、Runtime/Driver、量化、性能分析、工程开发流程和多 GPU 编译方向。

详细学习路径见 `course/README.md`。

## 要求

- Python 3.10+（本机 3.14）
- 只需要 numpy
