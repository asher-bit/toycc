# GPU 第 5 章：Triton——从 Python kernel 到 GPU 代码

## 1. 本章目标

- 能解释"program instance"与 CUDA thread 的区别，并手算一个 kernel 的 launch grid；
- 能逐行读懂一个带 mask 的 `tl.load`/`tl.store`，说清 `tl.arange`、指针运算、mask 各是什么对象；
- 能画出 Triton 的编译链（Python AST → TTIR → TTGIR → LLVM IR → PTX → cubin），并说出每层新增了什么信息；
- 能解释 `BLOCK`、`num_warps`、`num_stages` 三个旋钮各自影响什么，以及 autotune 为什么必须绑定 shape/dtype/arch；
- 能用"结果错 / 编译错 / 性能差 / 运行时错"四棵树做第一层定位。

前置：第 1 章的线程/内存模型、第 4 章的 tile 概念（本章的 `BLOCK` 就是第 4 章 threadblock tile 的语言原语）。跑实验需要 NVIDIA GPU + `pip install triton`；手算不需要 GPU。

## 2. 工作中的问题长什么样

DSL 方向的日常问题：

```text
"Triton 写的 matmul 和 CUTLASS 差多少？什么时候用哪个？"
"pid 是线程号吗？为什么没有 threadIdx 也能写 kernel？"
"autotune 跑了两遍，选出来的配置不一样，怎么回事？"
```

三问对应三个认知：**抽象层次**（Triton 把你从 thread 级抬到 block 级）、**编译链**（Python 语法只是入口，真正的程序是 IR）、**搜索方法**（autotune 是测量协议，不是魔法）。本章逐个建立。

## 3. Triton 是什么、不是什么

Triton 是**面向并行 kernel 的语言和编译器**：用 Python 风格 DSL 写 kernel，编译期走 AST → IR → PTX 的完整管线。两个"不是"：

- **不是 Python wrapper**：`@triton.jit` 装饰的函数不在 Python 解释器里执行，装饰器在编译期捕获它的 AST，翻译成 IR 再编译成 GPU 代码。所以函数体里只能用 Triton 语言允许的操作（`tl.*`），普通 Python 控制流会被当作编译期结构处理；
- **不是 CUDA runtime 的替代**：编译产物加载、设备选择、stream、launch 仍由 driver/runtime 完成，Triton 的运行时部分只是把"编译缓存 + 发射"包了一层。

对照第 4 章的决策表：**图级问题用图编译器，tile 级算法用 Triton/CUTLASS，指令级控制才手写 CUDA**。Triton 与 CUTLASS 的分工是：Triton 用"块级原语 + 编译器"快速迭代算法；CUTLASS 用"模板 + 手工布局"追极致；两者共享同一条底层链（PTX → ptxas → SASS）。

## 4. 最小 kernel 逐行拆

下面的代码属于【可运行代码】，保存为 `add.py`，需 `triton` 包与 NVIDIA GPU：

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements,
               BLOCK: tl.constexpr):          # ① 编译期常量
    pid = tl.program_id(axis=0)               # ② 程序实例编号
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # ③ 本实例负责的元素索引向量
    mask = offsets < n_elements               # ④ 尾部越界掩码
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)  # ⑤ 带掩码的块加载
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

逐行对应对象：

**① `BLOCK: tl.constexpr`（编译期常量）**：`BLOCK` 的值在编译时固定，参与编译缓存 key。它的含义与第 4 章 threadblock tile 相同——一个程序实例一次处理多大一块数据。**每次改 `BLOCK` 的值都会触发一次重新编译**（第 7 节的 autotune 编译时间账就来自这里）。

**② `tl.program_id(axis=0)`（程序实例编号）**：一个 **program instance** 是 launch 网格里的一个执行单元，物理上对应一个 CTA（即一个 block）。**它不是线程号**——Triton 的抽象里没有显式的 threadIdx，一个 instance 内部"有多少线程、每个线程拿哪些元素"由编译器的 layout 决定。这就是 Triton 与 CUDA 最本质的差别：CUDA 让你直接写每个线程的行为，Triton 让你写每个 block 的行为，线程级细节交给编译器。

**③ `tl.arange(0, BLOCK)`（索引向量）**：生成 `0..BLOCK-1` 的整数向量。`offsets = pid*BLOCK + tl.arange(...)` 是**向量化的索引公式**——对照第 1 章的 `i = blockIdx.x * blockDim.x + threadIdx.x`，两者结构相同，但 Triton 版里 `threadIdx` 不存在，向量的每个元素最终落到哪个线程由 layout 决定。`x_ptr + offsets` 是指针与向量的逐元素相加（pointer arithmetic），得到指向本块各元素的地址向量。

**④ mask（掩码）**：`offsets < n_elements` 是逐元素比较的布尔向量。它对应第 1 章 `if (i < n)` 的向量化版本。

**⑤ `tl.load(ptr, mask, other)`（块加载）**：一次语义上的"块加载"，`mask` 为假的元素不实际访问内存、返回 `other` 值（这里填 0.0）。编译器负责把它变成合并的访存（或 `cp.async`/TMA）。

### 4.1 手算一个 launch 网格

取 n_elements = 2500、BLOCK = 1024：

```text
grid = ceil(2500 / 1024) = 3 个 program instance
instance 0: offsets = 0..1023     → mask 全真
instance 1: offsets = 1024..2047  → mask 全真
instance 2: offsets = 2048..3071  → 2048..2499 有效(452 个), 2500..3071 越界(572 个) → mask 为假
```

最后一块越界 572 个元素，全被 mask 挡住——**漏写 mask 的后果与第 1 章漏写 `if (i<n)` 完全一样**：越界访问、随机错值。

## 5. Triton 与 CUDA 的对照

| CUDA | Triton 中的近似概念 | 差异说明 |
|---|---|---|
| grid/block | program instance / launch grid | 一个 instance 处理一个 tile，tile 大小是语言原语 |
| threadIdx | `tl.arange` 等向量索引 | 没有显式线程号，lane 映射由 layout 决定 |
| 分支/predicate | mask | 向量化的逐元素条件 |
| shared/register | 编译器决定的 layout 与缓存路径 | 手写 CUDA 里显式 `__shared__`，Triton 里是编译决策 |
| warp primitive | layout、`tl.dot`、扩展 API | 硬件级协作被抽象成布局变换 |
| launch 配置 | `triton.Config`（BLOCK/warps/stages） | 配置参与编译与 autotune |

这个对照**不是一一对应**，性能问题的终点不变：最终仍要回到 PTX/SASS、寄存器、内存事务和 profiler（第 2、3、7 章的工具链不变）。Triton 只是把"从算法到这些底层对象"的路径自动化了一部分。

## 6. 编译链：每层新增什么信息

```text
Python 源码(@triton.jit 捕获 AST)
    ↓ ① 翻译
TTIR(Triton IR: 语义层, 与硬件无关)
    ↓ ② 加布局
TTGIR(Triton GPU IR: layout、warp、共享内存等硬件信息)
    ↓ ③ 降级
LLVM IR
    ↓ ④ NVPTX backend
PTX → cubin
    ↓ ⑤
CUDA driver / runtime launch
```

逐层看"新增了什么"：

**① Python → TTIR**：`@triton.jit` 在编译期捕获函数 AST，翻译成 Triton IR。Triton IR 是基于 MLIR 的方言体系（MLIR 的"方言"概念见主教材第 26 课；这里只需知道：TTIR 是语义层，只描述"算什么"，不描述"谁算哪块"）。`tl.load`、`tl.dot` 在这里是带语义的 IR 操作。

**② TTIR → TTGIR**：这一跳加 **layout**。layout 回答"向量里的每个元素放在哪个线程的哪个寄存器"——blocked layout、shared layout 等。第 4 章 CuTe 的 Layout 是模板期类型，Triton 的 layout 是 IR 对象，同一个概念、两种载体。`tl.dot` 要求操作数处于 shared/MMA 专用 layout，不符就要插 `convert_layout`——**一次多余的 convert 就是一次 shared 往返**（写进再读出），这是 Triton kernel 里最常见的隐藏开销。

**③ TTGIR → LLVM IR**：把带 layout 的操作展开成标量/向量 LLVM IR，layout 在此层变成具体的 `extractelement`/`insertelement` 与地址计算。

**④⑤ LLVM IR → PTX → cubin → launch**：与第 2 章完全相同的 NVPTX 链，`ptxas` 的资源账、cubin 元数据、JIT 规则全部适用。所以 Triton 生成的代码最终也要接受第 2、3 章的检查：看 `ptxas -v`、看 SASS、算 occupancy。

一个工程细节：编译产物按 **cache key**（源码、函数签名、`constexpr` 值、目标架构、triton 版本等）缓存。同一 kernel 改 `BLOCK` 或改 `num_warps` 都是新 key、重新编译——autotune 的编译开销就是"候选配置数 × 单次编译时间"。

## 7. autotune：测量协议，不是魔法

**autotune（自动调优）**是"在候选空间里按测量协议搜索最好配置"的过程。写法：

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 128}, num_warps=8, num_stages=4),
        # ... 更多候选
    ],
    key=['M', 'N', 'K'],          # 按 shape 维度分别缓存最优配置
)
@triton.jit
def matmul_kernel(...):
    ...
```

三个旋钮各自影响什么（与第 3 章的账一一对应）：

| 旋钮 | 影响 | 对应的底层账 |
|---|---|---|
| `BLOCK`（tile 大小） | 数据复用、occupancy | 算术强度（第 4 章）与寄存器限制（第 3 章） |
| `num_warps`（每 instance 的 warp 数） | CTA 内并行度、每线程工作量 | 驻留 warp 数（第 3 章） |
| `num_stages`（软件流水级数） | 加载/计算重叠深度 | 共享内存用量 ≈ stages × 每级 tile 字节数 |

手算 `num_stages` 的 shared 账：BLOCK=128×128、K 块 BK=64 时，每级要缓存 A tile（128×64×2 B=16 KB）+ B tile（64×128×2 B=16 KB）= 32 KB；`num_stages=4` → 约 128 KB shared——对照第 3 章 Ampere 每 SM 164 KB 的上限，stages 再大就直接触顶。这就是"stages 调大不总是变快"的账。

autotune 的正确用法：**候选空间和测量协议都是显式对象**，不是"让编译器凭空变快"。必须绑定输入 shape、dtype、GPU 与正确性校验（`key` 参数就是干这个的）；否则搜索会"把偶然噪声当最优"——同一批候选在两次运行里选出不同配置，常见原因就是 key 没覆盖 shape，导致不同 shape 的最优配置互相污染。

## 8. 调试四棵树

```text
Python 结果错误
  → 检查 mask、stride、dtype、指针和边界；开 TRITON_INTERPRET=1 用解释器跑同一 kernel 定位
编译失败
  → 检查 constexpr、layout、不支持的 op、目标架构
性能差
  → TRITON_KERNEL_DUMP=1 dump 出 .ttgir/.llir/.ptx → 看寄存器/访存/warp → Nsight Compute
运行时错误
  → 检查 stream、生命周期、异步错误和 driver/toolkit 兼容性
```

`TRITON_INTERPRET=1` 让 kernel 在 CPU 解释器上执行（慢但可打印中间值），`TRITON_KERNEL_DUMP=1` 把每层 IR 落盘——前者查语义错，后者查编译与性能错。每个 kernel 建议同时保留：reference 实现、随机输入、边界 shape（非 2 的幂）、非连续 stride 和误差阈值测试——第 1 章"参考执行器当裁判"的方法论在这里原样适用。

## 9. 源码阅读地图

- Python API 与语言语义：`tl.*` 的定义（哪个操作长什么样、有什么约束）；
- TTIR/TTGIR：方言定义与 layout 变换 pass（读"布局怎么插入、怎么消"）；
- LLVM backend / NVIDIA backend：目标指令选择与资源分配；
- runtime/cache：编译缓存与 driver launch；
- tutorials 与 regression tests：最小可用示例与边界回归。

阅读顺序：从一个 tutorial 的 `@triton.jit` 函数开始 → 找编译入口 → 追 IR dump → backend pipeline → 生成的 PTX。不要停留在 Python 装饰器层——真正的程序是那几层 IR。

## 10. 常见错误与归因

| 现象 | 根因 | 定位手段 |
|---|---|---|
| 结果随机错、偶发 | mask 缺失或 stride 错（越界/错位访问） | `TRITON_INTERPRET=1` 对比中间值 |
| 改 `BLOCK` 后第一次运行很慢 | constexpr 变化触发重新编译 | 看编译缓存 key 是否变化 |
| dot 前后性能差很多 | 布局不匹配插入了 convert_layout（shared 往返） | dump TTGIR 找 convert_layout |
| 大 tensor 结果错 | 指针偏移用 int32，超过 2 GB 溢出 | 检查指针/索引 dtype，换 int64 |
| autotune 两次结果不同 | key 未覆盖 shape/dtype，最优配置互相污染 | 检查 `key=['M','N','K']` 是否够 |
| stages 调大反而慢 | shared 触顶，occupancy 掉 | 手算 stages × tile 字节 vs SM shared 上限 |

## 11. 检查点

完成以下四项才算通过本章：

1. 手算：n=5000、BLOCK=512 的 add_kernel，grid 是多少、最后一个 instance 的有效元素与越界元素各多少；
2. 画出 Triton 编译链五层，标出"layout 信息"在哪一层被加入；
3. 解释 `BLOCK`、`num_warps`、`num_stages` 各影响哪个底层账，并给出手算 stages 的 shared 字节公式；
4. 写一个"结果错"的三步排查顺序（用第 8 节的树，落到具体工具名）。

## 12. 下一步与扩展阅读

本章把第 4 章"手工分层"变成了"语言原语 + 编译器"。下一章（GPU 06：Runtime / Driver）回到系统软件层：编译产物如何被加载、内存如何分配、kernel 如何发射——Triton 与 CUDA 在这里汇合到同一条运行时链上。

- 官方：[Triton 文档](https://triton-lang.org/main/index.html)、[Triton Language API](https://triton-lang.org/main/python-api/triton.language.html)、[Triton 教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)；
- 与本课程的关系：Triton IR 是 MLIR 方言体系的真实应用（主教材第 26 课的"方言"在这里是生产代码）；Triton 的 layout 与 toycc 的 layout pass、CuTe 的 Layout 处理的是同一个"数据如何排布"的问题，只是层次不同。

**导航**：⬅ [上一章](04_cutlass_cute.md)（CUTLASS / CuTe）　｜　[下一章](06_runtime_driver.md)（Runtime / Driver）➡
