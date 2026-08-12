# GPU 第 3 章：GPU ISA、寄存器分配与 ABI

## 1. 本章目标

- 从“寄存器是硬件资源”过渡到“寄存器是编译器约束”；
- 解释 occupancy、spill、调用约定和 kernel 元数据的关系；
- 能阅读 PTX/SASS/MIR 中的资源使用信息；
- 为自研 GPU 列出 ISA、ABI 和 code object 的最小设计清单。

## 2. 三种寄存器视角

| 视角 | 你要问的问题 |
|---|---|
| 源码变量 | 这个值的生命周期和并行粒度是什么？ |
| 虚拟寄存器 | 编译器如何命名、合并、重用和降低它？ |
| 物理寄存器 | 每个线程占多少硬件寄存器，是否造成 spill 或降低 occupancy？ |

一个 kernel 的寄存器使用量按线程计算，但 SM 的寄存器文件按 block/warp 分配。寄存器多不一定坏：它可能减少访存和同步；但超过阈值会限制同时驻留的 warp，甚至将局部值 spill 到 local memory。

## 3. occupancy 不是性能本身

粗略地说，驻留 block 数受多个资源共同限制：

```text
resident_blocks = min(
  thread_limit / threads_per_block,
  register_file / registers_per_block,
  shared_memory_limit / shared_memory_per_block,
  architectural_limit
)
```

实际架构还有 warp、cluster、调度器和动态共享内存约束。occupancy 主要说明隐藏延迟的能力，不等于 IPC、带宽或最终吞吐。一个高 occupancy 的 kernel 可能因为访存不合并或指令依赖而很慢；一个中等 occupancy 的 GEMM 可能凭借高寄存器复用和 Tensor Core 吞吐更快。

## 4. spill 与调用边界

寄存器压力来源包括：过大的 unroll、过多临时值、复杂地址计算、函数内联和过大的 tile。spill 会把虚拟寄存器放入线程私有的 local memory；从性能报告看，它可能表现为额外的 local load/store，而不是明显的“栈错误”。

编译器工程师常用的手段：

- 限制或改变 unroll；
- 缩小 tile 或减少同时存活的累加器；
- 重新安排计算和加载的交错关系；
- 避免不必要的内联/临时对象；
- 用 `--maxrregcount` 做实验，但不要把它当万能修复。

## 5. GPU ABI 要约定什么

至少要明确：

- kernel 参数如何传递、参数布局和对齐；
- 标量、指针、向量和聚合类型的表示；
- 地址空间和指针宽度；
- 特殊寄存器、返回值和调用边界；
- 谁保存寄存器、如何处理栈/局部内存；
- kernel 的 block size、寄存器数、shared memory 和 barrier 元数据放在哪里；
- code object 如何携带架构、版本、重定位和调试信息。

自研 GPU 不应只从 opcode 表开始设计。ABI、加载器、调试器和 runtime 如果没有共同协议，编译器生成的“正确机器码”仍无法稳定执行。

## 6. 指令选择与调度

后端需要把高层操作映射到目标 ISA，同时满足：

- 操作数寄存器类和 bank 约束；
- 指令延迟与吞吐；
- load/use 距离和依赖链；
- barrier、memory fence 和异步拷贝的顺序；
- 特殊功能单元或 Tensor Core 的发射条件。

这就是为什么仅仅把 LLVM IR 中的 `fmul/fadd` 翻成两条 GPU 指令还不够：真正性能取决于调度、数据复用、寄存器和内存层次的联合决策。

## 7. 对照 LLVM/MLIR 和第 27~30 课

```text
MLIR GPU/NVVM/LLVM Dialect
 → LLVM NVPTX backend / Triton backend
 → PTX / cubin
 → 第29课的二进制与加载
 → 第30课的 driver/command submission
 → 第27课的 simulator / cycle model
```

如果要支持自研 GPU，建议先建立功能 ISA/模拟器和 ABI 测试，再逐步加入调度、寄存器分配和性能模型。每个阶段都要有“同一输入、同一输出、可比较”的差分测试。

## 8. 练习

1. 修改一个 GEMM tile，观察寄存器、occupancy 和时间的变化；
2. 用编译器报告定位一个 local spill；
3. 画出一个 kernel 参数从 host 到 device 入口的布局；
4. 为自研 GPU 写一页 ABI 草案：参数、寄存器、地址空间、code object、错误码。

参考：[PTX Machine Model](https://docs.nvidia.com/cuda/parallel-thread-execution/#ptx-machine-model)、[LLVM Code Generator](https://llvm.org/docs/CodeGenerator.html)、第 27~30 课。

