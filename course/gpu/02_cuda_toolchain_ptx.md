# GPU 第 2 章：CUDA 工具链与 PTX——从 `.cu` 到 `cubin`

## 1. 本章目标

- 区分 host 编译、device 编译、PTX、cubin、fatbin 和模块加载；
- 看懂 `nvcc` 的主要阶段和 `-gencode` 的意义；
- 能用 `cuobjdump`、`nvdisasm`、`nvcc --ptx` 观察中间产物；
- 理解 PTX 是虚拟 ISA，不是最终硬件指令 SASS。

## 2. 典型产物链

```text
.cu
 ├─ host compiler → CPU object
 ├─ device front-end → device code
 ├─ PTX（可选中间形态）
 ├─ ptxas → cubin / SASS for a target SM
 └─ fatbin = 多个 cubin +/或 PTX 的容器
        ↓
CUDA Runtime / Driver API 加载 module
```

`nvcc` 是编排器，不等于一个单阶段编译器。它会拆分 host/device 代码并调用其他工具。排查编译问题时先判断错误属于 host C++、device front-end、PTX assembler、链接/打包还是运行时加载。

## 3. 常用观察命令

```bash
nvcc -arch=compute_XX -code=compute_XX -ptx kernel.cu -o kernel.ptx
nvcc -arch=compute_XX -code=sm_XX kernel.cu -cubin -o kernel.cubin
cuobjdump --dump-ptx app
cuobjdump --dump-sass app
nvdisasm kernel.cubin
```

目标架构参数会随 GPU 代际变化。`compute_XX` 通常表示虚拟计算能力，`sm_XX` 表示具体目标架构；一个 fatbin 可以携带多个目标，使同一应用覆盖多个 GPU，但会增加体积和构建复杂度。

## 4. PTX 的角色

PTX 是面向并行线程执行模型的虚拟指令集。它有寄存器、谓词、地址空间、内存语义、线程层次和 barrier 等概念；驱动或安装时的 JIT 可以把兼容的 PTX 翻译到目标 GPU 的机器 ISA。PTX 的合法性和最终性能不是一回事：PTX 能汇编，不代表能得到理想 SASS。

```ptx
.visible .entry add_one(
    .param .u64 p,
    .param .u32 n
) {
    .reg .pred %p;
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;
    // 这里只展示结构，具体地址和边界逻辑略去
    ret;
}
```

读 PTX 时依次看：寄存器声明、谓词控制、地址空间、内存指令的 `.scope/.sem`、同步指令和特殊寄存器。不要用 CPU 汇编的“一条指令一个线程”直觉解释 PTX。

## 5. PTX、SASS 与兼容性

| 层次 | 作用 | 常见问题 |
|---|---|---|
| CUDA C++ | 程序员表达 kernel | 模板、类型、同步和边界错误 |
| PTX | 虚拟 GPU ISA | 版本、地址空间、内存序和目标特性 |
| SASS | 某一 GPU 的真实机器指令 | 指令选择、调度、寄存器、stall |
| cubin | 面向目标架构的设备代码容器 | 架构不匹配、加载失败 |
| fatbin | 多架构代码/ PTX 的打包 | 体积、JIT、兼容策略 |

性能回归时不要只比较 CUDA 源码或 PTX；应同时记录目标 GPU、driver/toolkit、cubin 是否命中、是否发生 JIT、寄存器和 shared-memory 使用量。

## 6. 与第 29 课的连接

第 29 课介绍 ELF、cubin/fatbin、重定位和 JIT；本章关注“这些产物从哪来”。二者合起来才是完整链路：

```text
编译器生成 code object
 → 记录 kernel 参数/资源元数据
 → 链接/打包/重定位
 → Driver 选择 cubin 或 JIT PTX
 → Module loader 建立设备可执行对象
```

## 7. 源码/工具阅读地图

- NVIDIA CUDA：`nvcc`、`nvrtc`、`ptxas`、`cuobjdump`、`nvdisasm`；
- LLVM 生态：NVPTX backend、LLVM IR 到 PTX 的 lowering；
- MLIR 生态：GPU/NVGPU/NVVM/LLVM Dialect 的逐步下降；
- TVM/Triton：从高层调度或 DSL 生成目标相关 kernel；
- 自研芯片对照：把 PTX/cubin/fatbin 替换成你的虚拟 ISA、目标 ISA 和 code object。

## 8. 练习

1. 生成同一个 kernel 的 PTX 和 cubin，比较目标信息；
2. 用 `cuobjdump` 找到 kernel 入口和资源元数据；
3. 故意使用不支持的 `-arch`，区分编译期错误和运行时错误；
4. 解释为什么只携带 cubin 的程序和只携带 PTX 的程序兼容性不同。

参考：[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)、[NVCC](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html)、[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)。

