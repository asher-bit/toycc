# 第 11 课：TIR 与调度——TVM 开发者 80% 的日常

> 本课风格：代码驱动 + 亲手"调度"一个循环 + 概念逐个拆透。
> 对应代码：`toycc/schedule.py`（一个极简调度模拟器）
> 准备：跑 `python -m course.runner 11` 对照看。
>
> **为什么这一课重要？** 前 10 课我们把"图优化"讲完了——那是 TVM 里
> 偏"架构"的部分。但 TVM 社区日常的 PR 和讨论，**大部分发生在 TIR 调度层**：
> "这个 kernel 怎么分块更快""向量化为什么没生效""线程怎么绑"。
> 不懂 TIR，你读不了最热门的讨论。

---

## 1. 为什么需要"第二层 IR"？——Relax 说"算什么"，TIR 说"怎么算"

前几课我们反复强调：Relax（高层图 IR）**刻意不存**布局/内存/循环。
但最终代码里**必须得有循环**。所以需要另一层 IR 专门描述"怎么算"——
这就是 **TIR（Tensor Intermediate Representation）**。

两层的分工：

| 层 | 回答的问题 | 例子 |
|---|---|---|
| Relax（图） | **算什么**（语义） | `conv2d → add(bias) → relu` |
| TIR（循环） | **怎么算**（如何执行） | 6 层循环、线程、内存访问顺序 |

**关键**：怎么算，存在巨大的自由度。同一个矩阵乘法，有无数种写法：
- 循环顺序：`i,j,k`？`k,i,j`？
- 要不要分块？分多大？
- 内层要不要向量化？
- 哪个循环上线程/并行？

**这些写法计算结果完全一样，但性能天差地别。** 选择"怎么算"的过程，
就叫**调度（schedule）**。TIR 层存在的意义，就是让编译器能**程序化地**
尝试各种调度，找到最快的那种。

> 打个比方：Relax 是"目的地和路线图"，TIR 是"具体每步怎么迈、迈多大"。
> 同一张路线图，走路、跑步、骑车，结果一样到，速度不一样。

---

## 2. TIR 长什么样？先看数据结构

TIR 程序是一棵"语句树"。核心节点（`Stmt` 和 `Expr`）：

```
TIR 程序
├── For 节点:  for (i, 0, 16) { ... }        ← 循环
├── BufferStore: A[i] = ...                  ← 写内存
├── BufferLoad:  x = A[i]                    ← 读内存
├── IfThenElse:  if (cond) { } else { }      ← 分支
├── LetStmt:     let acc = 0;                ← 局部变量
└── 各种算术表达式:  i*8+k, acc + a*b, ...
```

TVM 里你会看到类似这样的文本表示：

```
// 一个矩阵乘的 TIR
for (i, 0, 16) {
  for (j, 0, 16) {
    let acc = 0
    for (k, 0, 16) {
      acc += A[i*16 + k] * B[k*16 + j]
    }
    C[i*16 + j] = acc
  }
}
```

**和我们 `toycc/schedule.py` 的 `LoopNest` 一一对应**：`Loop` = `For` 节点，
`body` 字符串 = BufferStore 那行。toycc 用字符串模拟，TVM 用对象树。

---

## 3. 调度原语：一张"循环怎么改"的菜单

调度就是通过**一系列不改变语义的循环变换**，把"朴素写法"变成"高效写法"。
TVM 的调度原语（也叫 primitives），大部分都是对循环的操作：

| 原语 | 作用 | 对应 toycc/schedule.py |
|---|---|---|
| `split(i, f)` | 把循环拆成外层+内层 | `split(var, factor)` |
| `fuse(i, j)` | 把两个相邻循环合并 | `fuse(var1, var2)` |
| `reorder(...)` | 重排循环顺序 | `reorder(*vars)` |
| `tile(i, j, b1, b2)` | 2D 分块（= 两次 split + reorder） | `tile(...)` |
| `vectorize(i)` | 内层循环变成 SIMD | `vectorize(var)` |
| `parallel(i)` | 循环并行化（多线程） | `parallel(var)` |
| `bind(i, "blockIdx.x")` | 绑定到 GPU 线程 | `thread(var)` |
| `cache_read/cache_write` | 加缓存（共享内存/寄存器） | 未实现（见 FAQ） |

**逐个讲透**：

### split：把循环一分为二

```
原:  for i in [0, 16):   body(i)
拆:  for i_o in [0, 4):  for i_i in [0, 4):   body(i_o*4 + i_i)
```

为什么拆？为了后面的**重排**——内层能放进缓存/寄存器，外层能并行。

### reorder：重排循环顺序（性能差异的最大来源）

同一个三重循环，`i,j,k` 和 `k,i,j` 性能可以差 10 倍。为什么？
看内存访问顺序。如果 `C[i][j] = sum_k A[i][k]*B[k][j]`：
- `i,j,k` 顺序：内层 k 扫 A 的一行、B 的一列 → B 按列访问 → **缓存命中差**
- `i,k,j` 顺序：B 按行访问 → 缓存友好

调度器的核心工作，就是**找到内存访问最顺的循环顺序**。

### tile：2D 分块（缓存阻塞 cache blocking）

```
原:  for i in [0, 8):  for j in [0, 8):  ...
分块: for i_o in [0, 2):  for j_o in [0, 2):
        for i_i in [0, 4):  for j_i in [0, 4):
          body(i_o*4+i_i, j_o*4+j_i)
```

**为什么？** 数据从一个 8×8 的大区域读，变成"小块小块"地读——
每一小块都能完整待在 L1 缓存里，反复复用，不用反复去慢速内存取。
这是计算机体系结构里最经典的优化之一。

### vectorize：向量化（SIMD）

```
内层循环 j_i (4 个元素) → 一条 SIMD 指令:  v = ld(4个float); v = v*...; st(v)
```

现代 CPU 一次能对 8 个 float 做乘法。把内层循环**展开成一条指令**，
就省掉了 4 次"取数→算→存"的指令开销。这是 4 倍的速度提升来源之一。

### parallel / bind：并行化

- `parallel(i)`：CPU 上 `#pragma omp parallel for`，多核同时算不同 i
- `bind(i, "blockIdx.x")`：GPU 上把循环的每个迭代交给一个线程

---

## 4. 亲手调度一次：完整走查（跟着跑 `runner 11`）

我们的模拟器从"朴素 matmul"出发：

```c
for (i = 0; i < 8; i++) {
  for (j = 0; j < 8; j++) {
    for (k = 0; k < 8; k++) {
      acc[i,j] += A[i*8+k] * B[k*8+j]
    }
  }
}
```

四步调度（`toycc/schedule.py` 的 `matmul_scheduled`）：

**第 1 步 `tile(i, j, 4, 4)`**：把 8×8 的输出拆成 2×2 个 4×4 小块。
块内 4×4 = 16 个输出，正好填满寄存器/小缓存。

**第 2 步 `reorder(k, i_o, j_o, i_i, j_i)`**：把 k 提到最外层。
现在对于**每一个 k 值**，整块 4×4 都用 A 的这一列——A 的访问变成"一列反复用"，
完美复用缓存。

**第 3 步 `vectorize(j_i)`**：内层 4 个 j 用一条 SIMD 指令算。

**第 4 步 `parallel(k)`**：k 是外层循环，各 k 之间无依赖 → 交给不同线程/核。

**所有变换都满足语义等价**：每个 `acc[...] += ...` 计算的内容没变，
只是循环怎么组织变了。这就是调度的铁律——和第 3 课"融合必须语义等价"
是同一个原则在循环层的体现。

---

## 5. 调度 ≠ 手工写死：调度器是怎么"找到"好调度的？

有人会问：这些变换我都懂，为什么不直接写最快的循环？

**因为"最优调度"因硬件而异，而且参数空间巨大**：
- 分块大小：8? 16? 32? 64?
- 向量宽度：4? 8?（取决于 CPU 指令集）
- 线程数、是否拆层、缓存复用策略……

组合起来是**天文数字**。手工调一次要几个小时，换个硬件全部失效。
所以真实编译器提供两条路：

1. **手写调度（`te.schedule`）**：人指定变换序列，编译期应用
2. **自动调度（meta_schedule / autotuning）**：机器搜索，第 13 课专讲

TVM 的经典流程（`te` 时代）长这样：

```python
import tvm
from tvm import te

# 1. 描述计算(像我们的参考实现一样, 只描述算什么)
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

# 2. 创建调度, 施加变换(像我们的 schedule.py 一样)
s = te.create_schedule(C.op)
yo, yi = s[C].split(C.op.axis[0], 16)      # 第1维拆成16
s[C].reorder(yi, C.op.axis[1], yo)          # 重排
s[C].vectorize(C.op.axis[1])                # 向量化
s[C].parallel(yo)                           # 并行

# 3. 编译成模块
mod = tvm.build(s, [A, B, C], target="llvm")
```

**对照我们的 `toycc/schedule.py`**：`te.create_schedule` 就是我们构造
`LoopNest`；`split/reorder/vectorize/parallel` 是一模一样的名字。
你刚在 toycc 里手玩过的，就是 `te` 调度的缩小版。

---

## 6. 实验

```bash
python -m course.runner 11
```

对照"调度前/调度后"两段代码，数一数：循环从 3 层变成 6 层（分块把 2 维拆成 4 维），
内层加了 SIMD 注释，外层加了并行指令。**语义没变，写法大变。**

再自己玩：打开 `toycc/schedule.py`，把 `block` 从 4 改成 2 或 8，
看循环嵌套怎么变。

---

## 7. 真实 TVM 对照：TIR 调度在源码里的样子

TVM 里调度相关的核心文件（`src/tir/transforms/`）：

| 文件 | 干什么 |
|---|---|
| `split_loop.cc` | 实现 split |
| `loop_partition.cc` | 循环分区 |
| `unroll_loop.cc` | 循环展开（把循环体复制 N 份消除判断开销） |
| `vectorize_loop.cc` | 向量化 |
| `storage_rewrite.cc` | 存储重写（对应我们的内存规划） |
| `thread_storage_sync.cc` | 线程同步 |

这些 pass 的输入输出都是 TIR 程序。你读它们的方式，和第 8 课一样：
**结构扫读 → 入口精读 → 对照 toycc**。比如 `vectorize_loop.cc` 就是
"找到可向量化的内层循环，把 For 节点替换成向量指令"。

---

## 8. 深层理解：调度为什么能"白捡"性能？

很多人第一次听说调度，会觉得"这不是编译器作弊吗"。
不是作弊，它是在**利用硬件的物理特性**：

| 调度手段 | 利用的硬件特性 |
|---|---|
| reorder | 缓存按行读取、时间局部性 |
| tile | 缓存大小有限，分块让数据复用 |
| vectorize | CPU 的 SIMD 执行单元一次算多个 |
| parallel/bind | 多核/多线程并行单元 |
| cache_read | 共享内存/寄存器的低延迟 |

**性能的本质 = 让硬件尽量"顺路"地工作**：缓存命中、指令流水、并行单元。
调度就是把计算重新组织，让它顺路。这就是为什么"同样的算法，
调度不同，性能差 10 倍"。

---

## 9. FAQ

**Q：`cache_read` / `cache_write` 是什么？我没在 toycc 里实现。**
A：在 TIR 里"加缓存"= 在循环里引入一个中间缓冲（寄存器/共享内存），
先读进去再用，避免反复访问慢速内存。比如 matmul 把 B 的一块先拷进共享内存。
toycc 没实现是因为它要引入新的"缓冲分配"概念，超出模拟器范围。
理解成"手动控制缓存"即可。

**Q：调度改变循环顺序，怎么保证结果不变？**
A：只要满足**依赖关系**——内层循环使用的结果必须已经算好。
TVM 有**依赖分析**（`src/arith/` 里的分析器）自动检查；
违规的 reorder 会被拒绝。我们的 toycc 模拟器不检查（教学简化），
但你要知道真框架是检查的。

**Q：调度和"优化 pass"是什么关系？**
A：高层 pass（Relax 层）改图；调度（TIR 层）改循环。
一条编译流水线里，先 Relax 的 pass 把图优化好，再下降成 TIR，
然后在 TIR 层做调度，最后 codegen。第 9 课的 `FuseTIR` 就是
"融合后的 Relax 函数 → TIR"的下降步骤。

**Q：我该从哪开始学 te/meta_schedule 的实际 API？**
A：官方教程 `tutorials/language/schedule_primitives.py`（中文站有翻译），
把 split/fuse/reorder/vectorize/parallel 逐个在真实 API 上跑一遍。
你有 toycc 模拟器的底子，会非常快。

---

## 10. 本课小结

- 两层 IR：**Relax 说算什么，TIR 说怎么算**
- 调度 = 一系列不改变语义的循环变换
- 原语：split/fuse/reorder/tile/vectorize/parallel/bind/cache_*
- 性能的来源 = 让硬件"顺路"（缓存/SIMD/并行）
- 最优调度因硬件而异 → 手写难 → 引出自动调度（第 13 课）
- toycc 的 `schedule.py` = `te.schedule` 的教学缩小版

**下一步**：第 12 课，我们从"融合"这个单独优化，跳到"整个优化全景"——
编译器还有哪些 pass、`op_pattern` 系统、以及支配分析（读懂 fuse_ops
最后一块拼图）。

---

## 深层拓展：调度的三个"进阶细节"

### A. 为什么 reorder 和 tile 经常要一起做？

单独 tile 不够——tile 出来的"小块"如果还是按原来的顺序访问，
缓存行为没变。**reorder 决定"数据进缓存的顺序"，tile 决定"一次进多少"**，
两个配合才能让"同一个数据块在被踢出缓存前被用很多次"。
这就是第 15 课"时间局部性"在调度层的落地。

### B. `bind` 调度和第 21 课 GPU 的对应

`bind(loop, "blockIdx.x")` 不是语法糖——它把一个循环轴**映射到 GPU 的硬件线程层**。
你做的每一次 bind，都在回答第 22 课的问题："这个循环，是 block 级还是 thread 级？"
bind 错了，要么并行度不够，要么越界。**调度在 GPU 上 = 线程分配决策**。

### C. 调度的"合法性"谁来保证？

你随便 reorder，万一改错了依赖怎么办？TVM 的调度原语内部会做**依赖检查**——
比如两个循环有"先写后读"依赖，就不许交换。这就是第 12 课讲的支配/依赖分析
在调度层的应用：**调度不是自由的，是被依赖约束的**。

---

**导航**：⬅ [上一节](lesson10.md)（第 10 课 · 从看懂到上手）　｜　[下一节](lesson12.md)（第 12 课 · 优化全景）➡
