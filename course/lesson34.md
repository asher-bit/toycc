# 第 34 课：Triton 与 CUTLASS——手写 kernel 的两条现代路线

> 本课风格：代码对照 + 机制拆解。
> 目的：2026 年高性能部写 kernel 就两条主流路线：Triton（快迭代）
> 和 CUTLASS（极致性能）。搞懂它们各自"替你做了什么"，
> 以及和你学过的 TVM/meta_schedule 是什么分工。
> 前置：第 11 课（tile/调度）、第 13 课（autotune）、第 21~23 课（GPU/kernel）。

---

## 1. 为什么这两个是 2026 年的必会项

```
PyTorch 2.x 的默认编译器 Inductor:  torch.compile() → 生成 Triton kernel
NVIDIA 官方 kernel 库:              cuBLAS/cuDNN 的性能武器 = CUTLASS
推理框架(vLLM/TensorRT-LLM):        自定义算子 = Triton 或 CUTLASS 改写
```

**你不会写 Triton 就读不懂 Inductor 的产物；不懂 CUTLASS
就说不清"库为什么比编译器快"**。这两个是高性能部的"普通话"。

---

## 2. Triton：把 CUDA 的"线程级"抬到"块级"

### 2.1 一段 vector_add 的两版对照

```python
# CUDA C 版(第23课): 每个线程算 1 个元素, 你要管 threadIdx/合并访问
__global__ void add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

# Triton 版: 每个"程序实例"算一个块(BLOCK 个元素)
@triton.jit
def add(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                    # 我是第几个块
    offs = pid * BLOCK + tl.arange(0, BLOCK)  # 这个块负责的偏移向量
    mask = offs < n                            # 边界掩码
    a = tl.load(a_ptr + offs, mask=mask)       # 向量加载
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)   # 向量存储
```

### 2.2 逐行讲：Triton 替你做了三件事

- **`tl.arange` + `tl.load` 向量操作**：你写"这一块"，编译器展开成
  warp 内 32 线程的分工——**合并访问自动保证**（这块知识编译器代劳了）
- **`mask`**：边界处理不用手写 if——编译器生成 predication
- **`BLOCK: tl.constexpr`**：块大小是编译期参数——**autotune 直接搜它**
  （这正是 autotune 的搜索空间，Triton 内置 `@triton.autotune`）

**一句话**：CUDA 让你指挥每个线程；Triton 让你指挥每个块，
线程级的脏活（合并访问/发散/掩码/流水）编译器全包。
这就是为什么 PyTorch 选它当默认后端：**Python 前端 + 编译期块抽象 +
自动调优**，三个特性正好对上 AI 框架的需求。

### 2.3 下降链（对照你学过的 IR 课）

```
Triton Python DSL
  → Triton IR (MLIR 方言! 第26课的概念)
  → TTGIR (GPU 方言: 块→warp→线程的映射在这层定)
  → LLVM IR → PTX (第14/22课) → SASS (ptxas)
```

**Triton 是 MLIR 生态最成功的商用案例**——MLIR 的
方言/pattern rewrite/渐进式下降，在 Triton 源码里全部都有。

---

## 3. CUTLASS：把"人类最优经验"编码成模板层次

### 3.1 为什么库比编译器生成快

GEMM 的最优实现依赖一串**人工发现的诀窍**：三级 tile 尺寸、
双缓冲、软件流水、张量核指令选择、bank conflict 规避……
这些诀窍太"巧"，通用编译器（meta_schedule）搜不到全部。
CUTLASS 的做法：**把诀窍写成 C++ 模板参数，组合成库**。

### 3.2 层次化 tile：CUTLASS 的核心结构

```
Grid (整个 GEMM)
  └─ CTA tile (128×128×64)     ← 一个线程块算这一块
       └─ warp tile (64×64×64) ← 块内每个 warp 算一块
            └─ instruction tile (16×8×16) ← warp 内一条 mma.sync 算一块
```

每一层对应一级硬件：CTA→SM 共享内存、warp→寄存器、
instruction→张量核。**tile 尺寸是模板参数**：换芯片 = 换一组数字，
代码结构不变。这就是"加新后端五步"里 autotune 那步的工业版答案——
autotune 搜参数，CUTLASS 直接把参数做成模板。

### 3.3 读一份真实的模板实例化

```cpp
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                          // 数据类型 fp16
    cutlass::layout::RowMajor,                // A 布局
    cutlass::gemm::GemmShape<128, 128, 64>,   // CTA tile
    cutlass::gemm::GemmShape<64, 64, 64>,     // warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,      // mma 指令形状
    cutlass::epilogue::LinearCombination<...> // 输出融合(alpha/beta)
>;
```

读法：每个模板参数回答一个性能问题（tile 多大、
哪级缓存、什么指令、epilogue 融什么）。**性能工程师读 CUTLASS
参数表，就像你读调度参数**——同一套语言。

---

## 4. 三条路线的分工（什么时候用哪个）

| 路线 | 适用 | 代价 |
|---|---|---|
| **Triton** | 新算子快速实现、框架自动生成（Inductor）、研究迭代 | 极致性能有差距（差距口径见 FAQ） |
| **CUTLASS** | GEMM/attention 类重算子、要榨干张量核 | 学习曲线陡、模板代码长 |
| **编译器自动生成**（TVM/meta_schedule） | 长尾算子、自研芯片（没人手写） | 极致性能追不上手写库 |

**决策口诀**：新算子先 Triton 跑通 → 成为瓶颈就换 CUTLASS/手写 →
自研芯片没人生态就靠编译器+autotune 兜底（加新后端五步）。

---

## 5. 自研芯片视角

1. **要不要支持类 Triton 前端？** 要。tile 抽象已是 AI 编译器的
   共识层——你的编译器若能吃 Triton IR（或 MLIR 方言），
   PyTorch 生态的 kernel 就能低成本移植过来。**生态兼容比
   指令性能更能决定芯片生死**
2. **CUTLASS 的层次 tile 直接可搬**：你的芯片的 SM/warp/张量核
   层级对应一套 tile 参数——target 描述里的数字，
   就是填进这种模板的东西
3. **Triton 的 autotune 思路**：`@triton.autotune` 对 BLOCK/num_warps
   网格搜索 + benchmark 选优——这就是 autotune 的 mini 版，
   你的工具链可以直接照抄接口设计

---

## 6. FAQ

**Q：Triton 的性能天花板在哪？**
A：它封装了 warp 内细节，所以**warp specialization、TMA 的
精细控制、寄存器级调度**这些最新技巧表达不了（或很难）。
大 GEMM/attention 上典型差距 10~20%——差距正是来自这些不可表达特性；
追极致性能还是 CUTLASS/手写。

**Q：Triton 和 TVM 是什么关系？**
A：都是"AI 编译器"，哲学不同：TVM 是"搜索+生成"（ autotune
驱动），Triton 是"手写块级 + 编译器补线程级"。2026 年格局：
PyTorch 生态 Triton 占主导；TVM 在自研芯片/长尾算子上有优势
（后端可插拔）。两者都在用 MLIR。

**Q：CUTLASS 只能用于 NVIDIA 吗？**
A：CUTLASS 本体绑定 NV 张量核指令。但它的**层次 tile 架构思想**
已被各家复刻（AMD 的 CK、Intel 的 oneDNN 模板）——你自研芯片
的 kernel 库大概率也是"类 CUTLASS 的模板层次"。

**Q：我要学到什么程度？**
A：Triton 要**会写**（vector_add → fused_softmax → 简单 GEMM
三级），CUTLASS 要**会读**（看得懂模板参数表 + 会改 epilogue）。
高性能部面试常考："写一个 fused kernel 你怎么选 BLOCK"——决策链：先算该 kernel 的算术强度，据此定 tile 大小，再在 SRAM/寄存器约束内枚举 2~3 档各跑一次 autotune。

---

## 7. 本课小结

- Triton = 块级抽象 + Python 前端 + 内置 autotune；
  合并访问/掩码/流水编译器全包
- Triton 下降链 = MLIR 方言 → TTGIR → LLVM → PTX（MLIR 的商用范本）
- CUTLASS = 人类最优经验编码成 C++ 模板层次（CTA/warp/instruction 三级 tile）
- 分工：Triton 快迭代、CUTLASS 追极致、编译器+autotune 兜底长尾/自研
- 自研芯片：支持类 Triton 前端 = 生态入场券；CUTLASS 思想可原样搬

**下一步**：第 35 课——前沿专题速览（稀疏/MoE/speculative/MLPerf）。
主课到此收尾，下一课把会议室里还会冒出来的前沿名词一次收编。

---

## 深层拓展：两个现代编译器路线的三个问题

### A. Triton 为什么能赢过"手写 CUDA"

不是性能赢（CUTLASS 更快），是**迭代速度**赢：一个新算子
（如 RMSNorm、RoPE 变体）研究员下午写完 Triton 晚上就上卡测。
性能工程常常是**人力不是机器**：同一算子手写 CUDA 3 人日、Triton 0.5 人日，性能差 15%——多数算子这差价买人力划算。这是工具选型的
真实逻辑，和 16 课"迭代速度优先"同源。

### B. Inductor 的接入点长什么样

```
PyTorch 图 → TorchDynamo 抓取 → AOTAutograd → Inductor 图优化
  → 算子 lowering → Triton 代码生成 → JIT 编译 → 执行
```
你的编译器要进这个生态，有两个接入点：①替换 Triton codegen
（自研芯片后端，类 24 课五步）②在 Inductor 图优化层插 pass
（图级优化）。①是重活但一劳永逸，②是轻活但受制于人。

### C. 为什么"tile"是对的抽象

回头看：第 11 课 tile（局部性）、13 课 autotune（搜 tile 参数）、
19 课微内核（寄存器 tile）、21 课张量核（硬件 tile 指令）、
34 课 Triton/CUTLASS（tile 是编程模型）——**同一个概念
贯穿了整门课的五层**。AI 编译器 20 年的经验收敛成一句话：
**tile 是连接"数学（GEMM）"和"机器（内存层次）"的最小抽象**。
自研芯片的编程模型设计，先把 tile 这一层想明白。
---

**导航**：⬅ [上一节](lesson33.md)（第 33 课 · 生产级量化）　｜　[下一节](lesson35.md)（第 35 课 · 前沿专题速览）➡