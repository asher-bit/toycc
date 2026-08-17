# 编译器与 GPU 工具链自学课程（偏编译器方向）

一门以编译器为主线的自学课程：从计算图 IR 与 Pass 出发，贯穿机器学习编译器
（TVM / MLIR / LLVM）和 GPU 工具链（CUDA、PTX、CUTLASS、Triton、Runtime、
Driver、性能工具），并延伸到 LLM 推理与分布式。课程不绑定任何单一框架——
TVM、LLVM、MLIR、CUDA 各是整条工具链中的一层。

仓库还带一个几百行的迷你 AI 编译器 `toycc`，作为**动手教具**：IR、Pass、
代码生成、参考执行器等核心概念都可以先在它身上跑通，再对照真实框架源码。
**toycc 只承担课程前半段的"亲手写"环节，不是课程的中心。**

## 快速开始

```bash
python -m course.runner 1       # 第1课实验
python -m toycc.examples.demo   # 完整流水线一键演示
python -m toyisa.demo           # 迷你教学 ISA: 汇编→链接→双ISS→差分→覆盖率门禁
```

`demo` 会打印：10 个算子 → 融合/布局/常量折叠 → 6 个算子，内存省 47%，
生成的代码与参考执行结果完全一致（max|Δ|=0）。

## 项目结构

```
course/         课程主体：新人手册、知识地图、岗位路径、36 课主线和各技术专题（Markdown）
toycc/          配套迷你 AI 编译器（动手教具，模仿 TVM 架构）
├── ir/         计算图 IR（对应 Relax/ONNX）
├── passes/     优化 pass（融合/布局/常量折叠/内存/DCE/量化）
├── codegen/    代码生成（C + Python 双后端）
├── runtime/    numpy 参考执行器（正确性裁判）
├── schedule.py TIR 调度模拟器
└── hardware.py 缓存/延迟硬件模型
toyisa/         迷你教学 ISA（第 27~29 课的动手项目，流片前工具链第一件交付物）
```

## 课程内容

覆盖：模型导入、IR/Pass/后端、TVM、LLVM、MLIR、GPU 硬件、CUDA、PTX、Kernel、
CUTLASS、Triton、Runtime/Driver、量化、性能分析、工程开发流程、LLM 推理、
分布式与多 GPU 编译方向。

详细学习路径见 `course/README.md`。

## 要求

- Python 3.10+（本机 3.14）
- 只需要 numpy
