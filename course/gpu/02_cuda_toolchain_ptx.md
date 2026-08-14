# GPU 第 2 章：CUDA 工具链与 PTX——一段 `.cu` 如何变成 GPU 上可执行的代码

## 1. 本章目标

- 能画出 `.cu` 从源码到 GPU 可执行代码的完整产物链，并说清每一层的产物由哪个工具生成；
- 能解释 `compute_XX` 与 `sm_XX` 的区别，并手算出"一个只带某种 cubin 的程序在哪些 GPU 上能跑"；
- 能对照第 1 章的 `vector_add` 逐块读一段真实 PTX（入口、参数、寄存器、地址空间、谓词）；
- 能说出 PTX 是"虚拟 ISA"的三个具体证据；
- 能区分编译期错误、模块加载期错误和 JIT 错误的报错位置。

前置：第 1 章的索引公式与内存空间概念。工具要求：CUDA Toolkit ≥ 11.8（`nvcc`、`cuobjdump`、`nvdisasm` 均随 toolkit 安装），观察 PTX/SASS 不需要 GPU，跑加载实验需要 NVIDIA GPU。

## 2. 工作中的问题长什么样

工具链层的问题通常长这样：

```text
"同一个 .cu，为什么编译过了，在这台机器上一跑就报 no kernel image？"
"我看不懂 .ptx 文件，怎么知道编译器把我的循环翻译成什么样了？"
"fatbin 是什么？为什么它比我的源码大那么多？"
```

这三问分别对应：**兼容性账**（什么产物能在什么硬件上跑）、**中间表示阅读**（PTX 怎么读）、**打包策略**（fatbin 装了什么）。本章把这三个问题建成模型，底层格式（ELF、重定位、加载器如何消费 cubin）的细节在主教材第 29 课展开，本章只讲"这些产物从哪来、各是什么"。

## 3. 一张总图：产物链

`nvcc` 是一个**编排器（orchestrator）**，不是单阶段编译器。它把一次编译拆给多个工具，每个工具负责一段：

```text
vector_add.cu
   │  nvcc 拆分 host 代码与 device 代码
   ├─ host 部分 ──► 宿主编译器(如 gcc/cl) ──► CPU 目标文件
   │
   └─ device 部分
        │  device 前端
        ▼
      PTX(虚拟 GPU ISA, 文本)
        │  ptxas(PTX 汇编器, 按目标架构优化)
        ▼
      SASS(某一代 GPU 的真实机器指令, 只存在与 cubin 里)
        │  打包
        ▼
      cubin(单架构设备代码容器)
        │  与其它 cubin / PTX 一起打包
        ▼
      fatbin(多架构容器, 嵌进宿主可执行文件)
        │  运行时 cuModuleLoad
        ▼
      CUDA Runtime / Driver 选择 cubin 或 JIT PTX, 建立设备模块
```

排查编译问题时的第一步永远是**判断错误属于哪一层**：host C++ 语法错、device 前端错、`ptxas` 汇编错、链接打包错、还是运行时加载错。每一层的报错格式不同（见第 9 节），定位错层会让排查时间翻倍。

## 4. 最小例子：对同一个 kernel 观察四种产物

下面的命令都是【可运行代码】，在第 1 章的 `vector_add.cu` 上直接执行。`sm_86` 是 Ampere 架构示例值，换成目标 GPU 的实际架构。

### 4.1 只看 PTX：`nvcc --ptx`

```bash
nvcc -O3 --ptx -arch=compute_86 vector_add.cu -o vector_add.ptx
```

产物 `vector_add.ptx` 是文本文件，可以直接打开。预期开头两行是版本头：

```text
.version 8.1
.target sm_86
.address_size 64
```

`.version 8.1` 是 PTX ISA 版本号，`.target` 是目标架构，`.address_size 64` 声明指针是 64 位。这三行就是"这份 PTX 是用什么方言写的"——驱动做 JIT 时靠它判断自己的 ptxas 能否消费这份文件。

### 4.2 只要 cubin：`nvcc -cubin`

```bash
nvcc -O3 -cubin -arch=sm_86 vector_add.cu -o vector_add.cubin
```

`-cubin` 让 nvcc 把 device 部分一路编译到 SASS 并打包成 cubin。cubin 是二进制，不能直接看，要用 4.3 的工具拆。

### 4.3 从成品里拆产物：`cuobjdump`

对一个已经编译好的可执行文件（普通 `nvcc vector_add.cu -o vector_add`），`cuobjdump` 能把嵌在里面的 fatbin 拆开：

```bash
cuobjdump --dump-ptx  vector_add     # 列出 fatbin 里携带的所有 PTX
cuobjdump --dump-sass vector_add     # 列出 fatbin 里携带的所有 SASS
nvdisasm -c vector_add.cubin         # 反汇编单独的 cubin 文件
```

预期输出：`--dump-ptx` 能看到第 5 节那种 PTX 文本；`--dump-sass` 能看到第 6 节那种 SASS 指令流。

### 4.4 看资源账：`ptxas -v`

```bash
nvcc -O3 -arch=sm_86 --ptxas-options=-v vector_add.cu -o vector_add
```

预期输出形如（示例值，随代码与优化级别变化）：

```text
ptxas info    : Compiling entry function 'vector_add'
ptxas info    : Used 14 registers, 0 bytes smem, 0 bytes spill
```

三个数字分别回答三个问题：**用了多少寄存器**（14 个/线程——对照第 1 章"每 SM 65536 个寄存器"的账，这个 kernel 不会被寄存器限制）、**用了多少共享内存**（0，这个 kernel 没用 shared）、**spill 多少字节**（0，没有局部变量落到 local）。"spill 非零"就是第 1 章说的"local 变量的物理位置是显存"在编译报告里的实证。

## 5. 逐块读 PTX：以 vector_add 为例

下面是【简化示意代码】：结构与真实 PTX 一致，但为了聚焦核心元素，把真实输出里数百行的中间指令压缩成注释。真实输出请用 4.1 的命令自己生成对照。

```ptx
.version 8.1
.target sm_86
.address_size 64

.visible .entry vector_add(
    .param .u64 vector_add_param_0,   // 形参 a 的指针
    .param .u64 vector_add_param_1,   // 形参 b 的指针
    .param .u64 vector_add_param_2,   // 形参 out 的指针
    .param .u32 vector_add_param_3    // 形参 n
)
{
    .reg .pred  %p<2>;      // 谓词寄存器: %p0, %p1
    .reg .b32   %r<8>;      // 32 位通用寄存器: %r0..%r7
    .reg .b64   %rd<8>;     // 64 位通用寄存器: %rd0..%rd7

    ld.param.u64   %rd1, [vector_add_param_0];  // 取形参 a
    ld.param.u32   %r1,  [vector_add_param_3];  // 取形参 n
    // 计算全局索引 i = blockIdx.x * blockDim.x + threadIdx.x:
    // mov.u32 %r2, %ctaid.x;   mov.u32 %r3, %ntid.x;
    // mad.lo.s32 %r4, %r2, %r3, %tid.x;
    // 边界判断 i < n:  setp.lt.s32 %p0, %r4, %r1;  @%p0 bra BODY;
    // BODY: 从 a、b 各取一个 float, 相加, 写回 out(指令见下)
    //   ld.global.f32  %f1, [%rd2];   // 读 a[i]
    //   ld.global.f32  %f2, [%rd3];   // 读 b[i]
    //   add.f32        %f3, %f1, %f2;
    //   st.global.f32  [%rd4], %f3;   // 写 out[i]
    ret;
}
```

逐块拆解，每块回答"它是什么、对应源码里的什么"：

**`.visible .entry`**：`.entry` 声明这是一个 kernel 入口（不是普通子函数）；`.visible` 表示它对模块外部可见——host 侧按名字查找 kernel 时，找的就是这个符号。对应 C++ 里的 `__global__`。

**`.param`**：kernel 形参的声明。GPU 的参数传递不走栈：参数放在 `.param` 地址空间，硬件把少量参数放进常量内存的"参数区"，kernel 里用 `ld.param` 读取。这就是 host 侧 `<<<...>>>(d_a, d_b, d_out, n)` 的四个实参在 device 侧的形态。

**`.reg` 声明**：PTX 要求所有寄存器先声明再用。`%r<8>` 表示声明一组从 `%r0` 到 `%r7` 的寄存器。注意这些名字是**虚拟寄存器名**——它不承诺 `%r0` 会变成物理寄存器 R0，这是第 6 节"虚拟 ISA"的第一个证据。

**特殊寄存器**：`%tid.x`、`%ctaid.x`、`%ntid.x` 是硬件内建的特殊寄存器，与 C++ 内建变量的对应关系：

| PTX 特殊寄存器 | C++ 内建变量 | 含义 |
|---|---|---|
| `%tid.x` | `threadIdx.x` | 线程在 block 内的编号 |
| `%ntid.x` | `blockDim.x` | 每 block 的线程数 |
| `%ctaid.x` | `blockIdx.x` | block 在 grid 内的编号 |
| `%nctaid.x` | `gridDim.x` | grid 的 block 数 |

对照第 1 章的索引公式 `i = blockIdx.x * blockDim.x + threadIdx.x`，PTX 里的 `mad.lo.s32 %r4, %r2, %r3, %tid.x` 就是同一条公式的指令形态（`mad` = 乘加，`r4 = r2 * r3 + tid.x`）。

**谓词**：`setp.lt.s32 %p0, %r4, %r1` 计算"r4 < r1 吗"，把结果存进谓词寄存器 `%p0`；`@%p0 bra BODY` 表示"仅当 %p0 为真时跳转"。对照源码里的 `if (i < n)`——**第 1 章说的"边界判断"在 PTX 层就是这个谓词**。去掉它，SASS 里就不会生成条件跳转，越界访问随之而来。

**地址空间后缀**：`ld.global.f32` 里的 `.global` 就是第 1 章五种内存空间的指令层体现。读 PTX 时，指令后缀直接告诉你数据在哪：

| 指令后缀 | 内存空间 | 对应 C++ 声明 |
|---|---|---|
| `ld.global` / `st.global` | Global | 普通指针 |
| `ld.shared` / `st.shared` | Shared | `__shared__` |
| `ld.param` | Param（参数区） | kernel 形参 |
| `ld.local` / `st.local` | Local | 局部数组、spill |

读一段 PTX 的固定顺序：① 看 `.version/.target` 判断方言；② 找 `.entry` 和 `.param` 确定入口与参数；③ 看 `.reg` 声明了解用了多少虚拟寄存器；④ 逐条看指令的后缀判断数据在哪个地址空间；⑤ 看 `setp`/`@%p` 谓词判断控制流；⑥ 看 `ret` 确认出口。不要用"CPU 汇编一条指令一个线程"的直觉读 PTX——PTX 的一条指令描述的是**整个 warp 的行为**。

## 6. PTX 是虚拟 ISA，不是最终机器码

PTX 到 GPU 上实际执行的指令之间还有一步：`ptxas` 把 PTX 汇编成 **SASS**（某一代 GPU 的真实机器指令集）。SASS 只存在于 cubin 里，用 `nvdisasm` 看。下面是从 `vector_add` 反汇编出的【示意代码】（指令名真实、具体寄存器号随编译变化）：

```text
        S2R          R0, SR_TID.X          ; 读线程号
        IMAD         R2, R0, R4, R5        ; 计算全局索引
        LDG.E        R6, [R2.64]           ; 读 global 内存
        FADD         R7, R6, R8
        STG.E        [R4.64], R7           ; 写 global 内存
        EXIT
```

"虚拟"有三个具体证据：

1. **寄存器是虚拟的**：PTX 里的 `%r0..%r7` 只是名字。`ptxas` 会把它们映射到物理寄存器文件（R0..R255）并做寄存器分配——PTX 声明了 8 个 `%r`，SASS 里具体用哪几个物理寄存器、有没有复用，是 `ptxas` 决定的；
2. **指令会变**：PTX 的 `add.f32` 不一定对应一条 SASS 的 `FADD`。`ptxas` 会做指令选择（`mad` 可能合成 `FFMA`）、指令调度（重排顺序填延迟槽），PTX 的指令数和 SASS 的指令数通常不相等；
3. **性能不归 PTX 管**："PTX 能汇编"只保证合法，不保证快。同一个 PTX 在 sm_86 和 sm_90 上会得到不同的 SASS、不同的寄存器分配和不同的调度——这就是为什么第 4.4 节的资源账要在**目标架构上**看，而不是看 PTX 文本。

## 7. `compute_XX` vs `sm_XX`：兼容性手算

`-arch` 和 `-code` 两个参数控制编译目标：

- **`compute_XX`（虚拟架构）**：只规定 PTX 的方言版本。`-arch=compute_86` 表示"生成 8.x 版本的 PTX"；
- **`sm_XX`（真实架构）**：规定 `ptxas` 为哪一代物理 GPU 生成 SASS。`-code=sm_86` 表示"为 sm_86 生成 cubin"。

四个典型组合的含义：

| 编译方式 | 产物 | 运行时行为 |
|---|---|---|
| `-arch=compute_86 -code=compute_86` | 只带 PTX | 运行时由驱动的 ptxas 现场 JIT 成当前 GPU 的 SASS |
| `-arch=compute_86 -code=sm_86` | 只带 sm_86 的 cubin | 直接加载，无 JIT |
| `-arch=compute_86 -code=sm_86,sm_90`（多组 `-gencode`） | 两个 cubin | 运行时按设备挑一个 |
| `-gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86` | cubin + PTX | 有 cubin 用 cubin；没有就 JIT PTX |

兼容性规则（NVIDIA 文档的二进制兼容约定）：**同一 major 内向前兼容，跨 major 不兼容**。手算三个场景：

```text
程序只带 sm_86 cubin:
  sm_86 GPU  → 命中, 直接跑           ✓
  sm_89 GPU  → 同 major(8), 前向兼容   ✓
  sm_90 GPU  → 跨 major, 无可用镜像   ✗ 报 cudaErrorNoKernelImageForDevice(209)

程序只带 compute_86 PTX:
  任何 >= sm_86 的 GPU → 驱动 JIT 编译后运行 ✓(首次加载有 JIT 开销)
  低于 sm_86 的 GPU    → PTX 版本过高, 无法编译 ✗
```

两条结论：**"编译过了"不等于"在这台机器上能跑"**——编译期能过、加载期报错的根因几乎都是这个架构匹配问题；**JIT 是兼容性保险，不是免费的**——首次 `cuModuleLoad` 会现场编译，简单 kernel 也要几十到几百毫秒，驱动把结果缓存在 `~/.nv/ComputeCache`（Linux）或 `%APPDATA%\NVIDIA\ComputeCache`（Windows），第二次加载才变快。

fatbin 的体积账同理：fatbin = 每个目标架构一份 cubin + 可选 PTX，**体积随携带的架构数线性增长**。一个简单 kernel 的单份 cubin 只有几 KB 到几十 KB，但大型应用的 fatbin 带 5~6 个架构时会明显变大。选多少架构是"覆盖范围 vs 体积/构建时间"的权衡。

## 8. cubin 里除了指令还有什么

cubin 不只是指令流，还带着**资源元数据**：每个 kernel 的每线程寄存器数、共享内存字节数、参数布局、最大线程数等。运行时在 launch 之前会用这些元数据校验 launch 配置——第 1 章的 `cudaGetLastError()` 抓到的 `invalid configuration` 错误，就是这个校验失败。一条链路闭环：

```text
ptxas -v 打印的资源账(=cubin 元数据)
  → 运行时 launch 时逐项校验(线程数、shared 大小、参数)
  → 非法配置当场报错, 而不是等 kernel 跑挂
```

这也是第 4.4 节资源账的第二个用途：launch 报错时，把报错数字与 cubin 元数据对照，就能判断是配置超限还是代码本身用太多资源。

## 9. 常见错误与归因

| 现象 | 所属层 | 根因与定位 |
|---|---|---|
| `ptxas fatal: Unsupported .version 8.5` | 编译期 | toolkit 太旧，不认新 PTX 方言，升级 toolkit |
| `ptxas error: Entry function uses too much shared data` | 编译期 | shared 超静态上限（每 block 默认 48 KB），需 `cudaFuncSetAttribute` 申请 |
| `no kernel image is available for execution on the device` | 加载期 | fatbin 里没有匹配当前 GPU 的 cubin 也没有可 JIT 的 PTX；`cuobjdump --dump-elf` 看携带了哪些架构 |
| 首次运行慢、第二次快 | JIT | 只带了 PTX，首次加载触发 JIT；带对应 sm 的 cubin 可消除 |
| 二进制体积异常大 | 打包 | `-gencode` 列了过多架构；按实际部署机器收敛架构列表 |
| 改了源码但性能没变 | 观察 | 看的是旧 cubin；确认编译命令的 `-arch/-code` 与目标机器一致 |

## 10. 检查点

完成以下四项才算通过本章：

1. 画出 `.cu → PTX → SASS → cubin → fatbin → 模块加载` 的链条，在每一层标出生成它的工具；
2. 手算：一个 fatbin 只带 `sm_80` cubin + `compute_80` PTX，在 sm_80、sm_86、sm_70 三台 GPU 上分别走哪条路径（cubin / JIT / 失败）；
3. 用 4.1 的命令生成 `vector_add.ptx`，指出：`.version` 行、`.entry` 名、`.param` 个数、索引计算对应的那组指令；
4. 用 `--ptxas-options=-v` 拿到资源账，解释三个数字（registers / smem / spill）各自的含义和用途。

## 11. 下一步与扩展阅读

本章解决了"产物从哪来"。下一个自然问题是"产物内部长什么样、加载器如何消费它"——主教材第 29 课把 cubin 当作 ELF 变体拆开讲（重定位、kernel 元数据表、JIT 缓存）。第 3 章（GPU ISA、寄存器与 ABI）则沿本章第 6 节的 SASS 继续向下，讲寄存器分配与调用约定。

- 官方：[NVCC 文档](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html)、[PTX ISA 手册](https://docs.nvidia.com/cuda/parallel-thread-execution/)、[cuobjdump/nvdisasm 文档](https://docs.nvidia.com/cuda/cuda-binary-utilities/)；
- 与本课程的关系：自研 GPU 工具链中，PTX 的位置就是"稳定虚拟 ISA 层"——它对下隔离硬件代际、对上提供统一编译目标，是第 24 课"工具链全景"里稳定中间层的 NVIDIA 参考实现。

**导航**：⬅ [上一章](01_cuda_programming_model.md)（CUDA 编程模型）　｜　[下一章](03_gpu_isa_registers_abi.md)（GPU ISA、寄存器分配与 ABI）➡
