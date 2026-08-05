# 第 23 课：Kernel 开发与性能分析——工具链工程师的日常

> 本课风格：写一个 kernel → 测它 → 找瓶颈 → 改 → 再测，完整走一遍。
> 目的：把前面所有课的"概念"变成"手感"——看一个真实的 kernel 怎么被调优。
> 前置：第 21/22 课（GPU 架构与编译器）、第 20 课（roofline）。

---

## 1. 为什么工具链工程师要会写 kernel？

你可能会问：我是做编译器的，为什么要手写 kernel？

因为：**编译器的目标就是"生成高效的 kernel"**。如果你不知道什么是
"高效 kernel"，就不知道编译器该优化什么、优化到什么程度。
所以工具链工程师必须会写 kernel——它是"终点"的参考标准。

写一个 kernel 的完整流程：

```
1. 写"正确但慢"的版本 (朴素 kernel)
2. 测性能 (benchmark)
3. 用 profiler 找瓶颈 (是算力? 带宽? 分支?)
4. 针对性优化 (分块/向量化/合并访问/共享内存...)
5. 再测, 验证确实变快
6. 把"最优模式"固化进编译器/autotune 数据库
```

**这 6 步就是你的日常。** 下面拿一个真实的例子走一遍。

---

## 2. 实战：优化一个 vector-add kernel（最简例子）

我们选一个最简单的 `vector_add`：`c[i] = a[i] + b[i]`。

### 2.1 朴素版

```c
__global__ void add_naive(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

**问题**：每个线程读 `a[i]`、`b[i]`、写 `c[i]`。如果线程连续，
相邻线程读相邻地址——**这本身就是合并访问，已经是"最优"的**。
向量加这种简单算子，主要瓶颈是**带宽**，不是优化空间。

### 2.2 用 roofline 判断瓶颈

`vector_add` 的计算强度：

```
计算: 每个元素 1 次加法 (1 FLOP)
搬运: 读 a + 读 b + 写 c = 3 个元素 × 4 字节 = 12 字节
计算强度 = 1 / 12 ≈ 0.08 FLOP/Byte
```

**极低强度 → 完全带宽受限。** 这类算子优化不了"算"，只能优化"搬"——
所以 vector_add 几乎永远是带宽顶，你不用浪费时间去优化计算。

**这就是 roofline 的价值**：一眼看出"这个 kernel 的天花板在哪，
值不值得优化"。

### 2.3 测它（benchmark 方法论）

```python
import time
# warmup(冷启动会慢)
for _ in range(10): launch_add(a, b, c, n)
# 测 100 次取中位数
times = []
for _ in range(100):
    t = time.perf_counter()
    launch_add(a, b, c, n)
    torch.cuda.synchronize()     # 必须等 GPU 真的算完!
    times.append(time.perf_counter() - t)
times.sort()
print(f"中位数: {times[50]*1e3:.3f} ms")
```

**注意两个坑**：
1. **`torch.cuda.synchronize()`**：GPU 是异步的，不算完就返回，
   你必须同步才能测准
2. **warmup**：第一次调用有编译/驱动初始化开销，要预热

---

## 3. 进阶：优化一个 matmul kernel（有优化空间的例子）

`vector_add` 是"带宽顶死"的没意思例子。换一个能优化的：**matmul**。

### 3.1 朴素版

```c
__global__ void matmul_naive(float* A, float* B, float* C, int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
        float acc = 0;
        for (int k = 0; k < K; k++)
            acc += A[i*K+k] * B[k*N+j];
        C[i*N+j] = acc;
    }
}
```

**问题**：内层 `k` 循环里，`A[i*K+k]` 相邻线程（相邻 j）访问不同 k →
**不合并访问**；`B[k*N+j]` 相邻线程 j 相邻 → 合并。所以一半访问好、一半坏。

### 3.2 优化思路（对照第 11/21 课）

1. **分块（tile）**：把 A/B 的一块搬进共享内存，让 warp 反复用
2. **reorder**：把 k 提到外层，让每个小块复用 A 的一整行
3. **向量化**：内层用 SIMD
4. **寄存器分块**：内层维护 6×8 个累加器（第 20 课微内核）
5. **用张量核**：如果能映射到 mma 指令，性能上一个数量级

**每改一步，用 benchmark 验证**——这就是工具链工程师的"手感"：
不是猜哪个优化好，是**测出来的**。

### 3.3 用 profiler 找瓶颈

真实开发里，瓶颈不是靠猜，靠**profiler**（NVIDIA 用 nsight/ncu）。
它能告诉你：
- 占用率多少
- 哪个访存是瓶颈（global/shared/local）
- 哪个指令占的时间最多（指令级 profiling）
- 有没有分支发散

**你自研芯片时，也要做自己的 profiler**——这是工具链的一部分。
输出的指标就是你第 21 课学的那些（占用率/合并访问/发散）。

---

## 4. 从"手写 kernel"回到"编译器"

写完 kernel、测完性能、找到最优模式后，**编译器要"复制"这个最优**：

1. 把这个 kernel 的结构（分块/重排/向量化）写成 **TIR 调度模板**
2. 或者注册成 meta_schedule 的**搜索空间**（第 13 课）
3. 让 autotune 在你的芯片上搜出最佳参数
4. 固化进数据库，以后同类算子直接复用

**这就是工具链的闭环**：人类找到最优 kernel → 编译器把它变成可复用的调度 →
自动应用到所有模型。

> 第 13 课的 meta_schedule 不是凭空想的，它是"把人类调 kernel 的经验"
> 编码成搜索空间。你写的 kernel 越懂，你的搜索空间就越聪明。

---

## 5. 一个真实工作日的工具链任务清单

公司里最常做的几件事：

1. **修 bug**：某个算子算错了（数值验证失败）→ 找是 pass 还是 kernel 的错
2. **优化性能**：某个 kernel 比预期慢 → profiler 找瓶颈 → 改调度/kernel
3. **加新算子**：模型有新算子（如 GELU）→ 写 kernel + 注册进图 + 写测试
4. **适配新芯片**：芯片换代 → 改 target 描述 + 重跑 autotune
5. **加新优化**：写一个 pass（比如新的融合规则）→ 写测试 → 提 PR

**共同点**：都离不开"改代码 → 跑测试 → 看结果 → 再改"。
这就是第 17 课讲的循环。

---

## 6. FAQ

**Q：我应该手写 kernel 还是让编译器自动生成？**
A：**探索阶段手写，量产阶段自动生成**。手写用来"发现什么是最优"，
然后用 meta_schedule/autotune 把这个最优固化。永远不手写
"生产中每个 kernel"——那不现实。

**Q：profiler 显示占用率只有 20%，怎么提？**
A：看是寄存器多还是共享内存多。减少每个线程的资源（比如少用临时变量、
共享内存复用），或者把数据分块减小——这些都是编译器/target 描述里
能调的参数。

**Q：我做的是"编译器"不是"kernel 工程师"，为啥要学这个？**
A：因为编译器的产出就是 kernel。**你不理解 kernel 怎么快，就做不出
快的编译器。** 这两个角色不是分开的，是"同一个人做两层的事"。

**Q：张量核一定要手写吗？**
A：不要。直接调 CUTLASS/cuBLAS（用 BYOC 集成），或者用 autotune
搜索。张量核的极致优化是"库的活"，编译器负责"认出这里能用张量核"。

---

## 7. 本课小结

- 工具链工程师的日常 = **写 kernel → benchmark → profiler 找瓶颈 → 优化 → 再测**
- **roofline 判断"值不值得优化"**：vector_add 带宽顶死，matmul 有优化空间
- **benchmark 方法论**：warmup + 中位数 + 同步 + 同条件
- **profiler** 告诉你瓶颈在哪（占用率/访存/指令级）
- 最优 kernel 要**固化回编译器**（TIR 模板 / autotune 搜索空间）

**下一步**：第 24 课——自研 GPU 工具链全景。把 compiler / driver /
runtime / assembler / profiler / debugger 这些组件拼成一张完整的图，
然后告诉你"你的第一份任务"该从哪入手。

---

**导航**：⬅ [上一节](lesson22.md)（第 22 课 · GPU 编译器技术）　｜　[下一节](lesson24.md)（第 24 课 · 自研 GPU 工具链全景）➡
