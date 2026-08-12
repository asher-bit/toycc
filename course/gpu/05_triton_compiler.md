# GPU 第 5 章：Triton——从 Python Kernel 到 GPU 代码

## 1. 本章目标

- 理解 Triton 的“程序实例”与 CUDA thread/block 的差异；
- 看懂 `tl.program_id`、`tl.arange`、mask、pointer arithmetic 和 `tl.load/store`；
- 了解 Triton IR 到 LLVM/PTX 的编译链；
- 知道 Triton autotune、layout 和调试的边界。

## 2. Triton 的定位

Triton 是面向并行编程的语言和编译器，目标是让开发者用 Python 风格的 DSL 编写高吞吐 DNN kernel。它不等于 Python wrapper，也不等于 CUDA runtime；运行时仍要负责编译产物、设备、stream 和 launch。

## 3. 一个最小 kernel

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements,
               BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

这里的 program instance 通常负责一块数据；它不是“一个 Python 函数调用只运行一次”。`BLOCK` 是编译期常量，`mask` 明确处理尾部越界。Triton 的抽象重点是 tile、布局和向量化访问，而不是手工写每个 CUDA thread 的全部细节。

## 4. Triton 与 CUDA 的对照

| CUDA | Triton 中的近似概念 | 关注点 |
|---|---|---|
| grid/block | program instances / launch grid | 一个实例处理多大 tile |
| threadIdx | `tl.arange` 等向量索引 | lane 到元素的映射 |
| predicate | mask | 边界和条件访问 |
| shared/register | 编译器决定的布局与缓存路径 | 数据复用和资源压力 |
| warp primitive | layout、dot、扩展 API | 硬件级协作是否可表达 |
| autotune config | `triton.Config` | tile、warp、stage 候选 |

这种对照不是一一对应。遇到性能问题时，最终仍要回到 PTX/SASS、寄存器、内存事务和 profiler。

## 5. 编译链

```text
Python AST / @triton.jit
    ↓
Triton IR（TTIR）
    ↓
Triton GPU IR（TTGIR：布局、warp、硬件相关信息）
    ↓
LLVM IR
    ↓
PTX / cubin
    ↓
CUDA driver/runtime launch
```

Triton 的关键价值在于把“线程到数据的映射”作为编译器可以推理和变换的对象。读 Triton 源码时，先找语义定义和 IR pass，再看 backend 如何选择目标指令，不要只停留在 Python decorator。

## 6. autotune 的正确用法

autotune 是在明确候选空间和测量协议后搜索配置，不是让编译器凭空变快。候选配置通常影响：BLOCK size、warps、stages、矩阵 tile、layout 和数据类型。必须绑定输入 shape、dtype、GPU、正确性校验和 cache key，否则很容易把偶然噪声当成最优。

## 7. 调试方法

```text
Python 结果错误
 → 检查 mask、stride、dtype、指针和边界
编译失败
 → 检查 constexpr、layout、unsupported op、目标架构
性能差
 → dump IR/PTX → 看寄存器/访存/warp → Nsight Compute
运行时错误
 → 检查 stream、生命周期、异步错误和 driver/toolkit 兼容性
```

建议为每个 kernel 同时保留 reference implementation、随机输入、边界 shape、非连续 stride 和误差阈值测试。

## 8. 源码阅读地图

- Triton Python API 与语言语义；
- Triton IR/TTIR/TTGIR 的定义与转换 pass；
- LLVM backend / NVIDIA backend；
- runtime、cache、driver launch；
- tutorials 和 regression tests。

从一个 tutorial 的 `@triton.jit` 函数开始，搜索其编译入口，再追到 IR dump、backend pipeline 和生成的 PTX。

## 9. 练习

1. 把 vector add 改成支持任意 stride；
2. 写 tiled matmul，记录不同 BLOCK/warps 的性能；
3. 用 mask 覆盖非 2 的幂次 shape；
4. 对比 Triton 和 CUDA 版本的 PTX、寄存器和 profiler 报告。

参考：[Triton Documentation](https://triton-lang.org/main/index.html)、[Triton Language API](https://triton-lang.org/main/python-api/triton.language.html)。

