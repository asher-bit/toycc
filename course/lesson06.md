# 第 6 课：内存规划——6 个张量塞进 2 个缓冲区，省 47%

> 本课风格：代码驱动 + 时间轴图 + 一步步"模拟分配"。
> 对应文件：`toycc/passes/memory.py`
> 准备：跑 `python -m course.runner 6` 对照看。

---

## 1. 先问一个反直觉的问题

一个 10 层模型有 9 个中间张量。最笨的分配方式：**每个张量独占一块内存，
谁也不许碰谁**。后果：内存峰值 = 所有中间张量大小之和。

但你真的需要同时装下所有中间张量吗？

**不需要。** 因为中间张量不是"同时活着"的：

```
conv1 算完 → 只有 conv1 活着
relu 算完 → conv1 不再需要了, 只有 relu 活着
...
```

**只要两个张量的"存活区间"不重叠，它们就能用同一块内存。**

这就是内存规划（memory planning）的全部秘密。它还有一个更著名的名字：
**buffer reuse（缓冲区复用）**。

---

## 2. 什么叫"活着"？——生命周期（liveness）

每个张量有两个关键时刻：

- **出生（born）**：被产生的时刻（拓扑序里的位置）
- **死亡（died）**：**最后一个消费者**执行完的时刻

一个张量从"出生"到"死亡"之间的区间，就是它的**生命周期**。

> **关键理解**：张量的死亡，不是在"没人再用它"的瞬间，
> 而是在"最后一个用它的人用完"的瞬间。之前它都得活着，因为后面还有人要读。

> **原理深挖：为什么"死亡时刻 = 最后一个消费者"这么自然，却又容易忽略？**
>
> 人的第一直觉是"谁读完它，它就死了"。不对——**要等"所有消费者都读完了"**。
>
> 手算一个例子：`conv1` 被 `bias_add1` 和 `dce_demo` 里的 `dead1`、`dead2` 都消费。
> 只有当 **最后那个消费者** 执行完，`conv1` 的内存才能释放。中途任何一个
> 消费者还需要它，它就得占着。**少算一个消费者 → 提前释放 → 后面读到脏数据**，
> 这是内存优化最经典的 bug。
>
> 所以编译器必须**精确统计每个张量的消费者**（就是 `consumers()` 这个函数）：
> 死亡 = 拓扑序最大的那个消费者的时刻。这也是"活跃变量分析"的雏形——
> 寄存器分配、内存规划用的都是同一套"最后使用"逻辑。
>
> 真实 TVM 的 `KillAfterLastUse` 就是精确做这件事：
> 遍历图，给每个张量找"最后一个消费者"，在那里插入释放点。

---

## 3. 用时间轴图看清我们的模型

优化后的图，拓扑序是这样的（`t` 表示时刻）：

```
t0: conv1_lt_0 = layout_transform(x)      # (1,8,8,3) 192
t1: conv1      = conv(...)                # (1,8,8,4) 256
t2: conv2      = conv(...)                # (1,4,4,8) 128
t3: flat_lt_0  = layout_transform(conv2)  # (1,8,4,4) 128
t4: flat       = reshape(...)             # (1,128)   128
t5: mm         = matmul(...)              # (1,16)    16
```

每个张量的生死：

```
张量            生→死      生命周期        元素数
conv1_lt_0      0→1        [0,1)           192
conv1           1→2        [1,2)           256   ← 最大的
conv2           2→3        [2,3)           128
flat_lt_0       3→4        [3,4)           128
flat            4→5        [4,5)           128
mm              5→6        [5,6)            16
```

**看出来了没？** 这些生命周期**一个挨一个**——每个张量刚被下一个
算子消费完就死。但注意交接时刻是**重叠一拍**的：t=1 时 conv1_lt_0
还在被 conv1 读（它 t=1 才死），conv1 的输出又正在写，所以那一刻
两个缓冲区必须同时存在。复用规则 `died_at < t`（死得够早）正是
为了躲开这个交接。

- 朴素方案：6 个张量各占一块 = 848 元素
- 复用方案：2 个缓冲区轮流接管 = 192 + 256 = 448 元素，省 47%

**内存省 47% 就是这么来的：让"死掉的"缓冲区被下一个张量接管。**
（不是省 70%——那得假设交接时刻不占内存，实际占。）

---

## 4. 逐行读 `toycc/passes/memory.py`

### 4.1 第一步：算每个张量的生死时刻

```python
born = {i: 0 for i in g.inputs}
died = {i: len(topo) for i in g.inputs}
for t, node in enumerate(topo):
    born[node.name] = t
for i in g.outputs:
    died[i] = len(topo)          # 图输出活到最后

for t, node in enumerate(topo):
    for inp in node.inputs:
        died[inp] = max(died.get(inp, -1), t)   # 被消费的时刻
```

**逐行**：

| 行 | 含义 |
|---|---|
| `born[i] = 0` / `died[i] = len(topo)` | 图输入一开始就活着，活到最后 |
| `born[node.name] = t` | 每个节点的输出在"它自己那一步"出生 |
| `died[i] = len(topo)`（对 outputs） | 图的最终输出活到结束 |
| `died[inp] = max(...)` | **每个被消费的张量，把"死亡时刻"更新为最后一次被消费的时刻** |

注意最后一步的 `max`：一个张量可能被多个算子读，`died` 要取**最大**
（最晚）那个——那才是它真正该死的时候。

> 这就是"last-use 分析"：找每个张量最后一次被用的位置。
> TVM 的 `KillAfterLastUse` pass 干的就是这件事。

### 4.2 第二步：分配 + 复用

```python
buffers: list[Buffer] = []
allocs: dict[str, AllocEntry] = {}
for t, node in enumerate(topo):
    name = node.name
    sz = size_of(name)
    reuse = next((b for b in buffers if b.died_at < t and b.size >= sz), None)
    if reuse is None:
        reuse = Buffer(index=len(buffers), size=sz)
        buffers.append(reuse)
    reuse.owner = name
    reuse.born_at = t
    reuse.died_at = died.get(name, t)
    allocs[name] = AllocEntry(name, shapes[name], sz, reuse.index, t, reuse.died_at)
```

**核心就一行**：

```python
reuse = next((b for b in buffers if b.died_at < t and b.size >= sz), None)
```

翻译成人话：**找一个"已经死了的"（`died_at < 当前时刻`）且"够大的"
（`size >= 我需要的`）缓冲区来用。**

- 找到 → 复用这块，更新 owner/生死
- 找不到 → 新开一块

`Buffer` 数据结构：

```python
@dataclass
class Buffer:
    index: int
    owner: str = ""      # 当前占用它的张量
    born_at: int = -1
    died_at: int = -1    # 死了才能被别人用
    size: int = 0
```

### 4.3 一步步"模拟分配"（跟着手算一遍）

```
t0: conv1_lt_0 需要 192
    buffers 空 → 新开 buf0, size=192
    buf0: owner=conv1_lt_0, died_at=1

t1: conv1 需要 256
    buf0 的 died_at=1, t=1, 复用要求 died_at < 1 → 不满足
    (conv1_lt_0 这一拍还在被 conv1 读, 不能抢它的缓冲)
    → 新开 buf1, size=256
    buf1: owner=conv1, died_at=2

t2: conv2 需要 128
    buf0 死了吗? died_at=1 < 2 ✓, 而且 192 >= 128 ✓ → 复用 buf0!
    buf0: owner=conv2, died_at=3

t3: flat_lt_0 需要 128
    buf1 死了吗? died_at=2 < 3 ✓, 256 >= 128 ✓ → 复用 buf1!
    buf1: owner=flat_lt_0, died_at=4

t4: flat 需要 128
    buf0 死了吗? died_at=3 < 4 ✓ → 复用 buf0

t5: mm 需要 16
    buf1 死了吗? died_at=4 < 5 ✓ → 复用 buf1
```

**结果：只有 2 个缓冲区**，完美循环交替使用。这就是你会在实验里看到的表：

```
张量            元素数   缓冲      生→死
conv1_lt_0       192    0  0→1
conv2            128    0  2→3      ← 和 conv1_lt_0 共用 buf0
flat             128    0  4→5      ← 继续用 buf0
conv1            256    1  1→2
flat_lt_0        128    1  3→4      ← 和 conv1 共用 buf1
mm                16    1  5→6      ← 继续用 buf1
```

---

## 5. 实验

```bash
python -m course.runner 6
```

看分配表 + 底部统计，重点理解这句：

```
朴素方案(每张量独占):  848 元素
复用方案(总分配):      448 元素 (2 个缓冲区)
```

**448 恰好是 conv1_lt_0 的 192 + conv1 的 256**——两个缓冲区，
一个装"正在被读的上一棒"，一个装"正在写的这一棒"，轮流交棒。
"总分配 = 所有缓冲区大小之和"就是内存规划的目标函数。

---

## 6. 真实 TVM 对照

TVM 把内存规划拆成两个 pass：

### 6.1 `KillAfterLastUse`（281 行）——找释放点

```cpp
static Result Collect(const Expr& expr) {
  CollectLastUsage visitor;
  visitor(expr);   // 遍历, 记录每个对象最后一次被使用的位置
  ...
  if (auto it = visitor.last_usage_of_.find(var); it != visitor.last_usage_of_.end()) {
    const auto* last_usage_point = it->second;
    bool is_output = last_usage_point == nullptr;     // 输出活到最后
    ...
    if (!is_output && !already_killed) {              // 不是输出且还没被释放
      if (visitor.storage_objects_.count(var))
        output[last_usage_point].storage.push_back(var);
      else if (... && stored_in_vm_register)
        output[last_usage_point].tensors.push_back(var);
      ...
    }
  }
}
```

然后在"最后一次使用之后"插入显式的释放指令：

```cpp
void VisitBinding(const Binding& binding) override {
  ...
  if (auto it = last_usage_.find(binding->var.get()); it != last_usage_.end()) {
    for (const auto& tensor_obj : it->second.tensors) {
      builder_->Emit(Call(Type::Missing(), relax.memory.kill_tensor, {tensor_obj}));
    }
  }
}
```

对应我们 `died[inp] = max(..., t)`：都是在**找最后一次使用点**。
TVM 把它变成显式的 `kill_tensor` 算子（运行时真的释放内存），
我们把"何时能复用"直接算进分配表——**同一件事，两种表达**。

### 6.2 `AllocateWorkspace`（215 行）——分配工作区

```cpp
// 给带 kWorkspaceSize 的外部函数追加 workspace 形参
if (auto workspace = func_node->GetAttr<int64_t>(attr::kWorkspaceSize)) {
  auto ty = TensorType(ShapeExpr({IntImm::Int32(max_workspace_size_)}), PrimType::UInt(8));
  Var workspace_param(name_sup_->FreshName("workspace"), ty);
  ...
}
// 在主函数开头分配一块 workspace_main
auto shape = ShapeExpr({IntImm::Int32(max_workspace_size_)});
auto workspace = MakeAllocTensor(shape, ty, IntImm::Int64(0));
workspace_var_main_ = builder_->Emit(workspace, "workspace_main");
```

这个 pass 解决"有些算子需要临时工作区"的问题：按**所有需求里的最大值**
分配一块，传给需要它的函数。和我们的 `max(size_of)` 峰值统计是一个思路：
**工作区大小 = 所有需求的最大值**。

对照表：

| 真实 TVM | toycc |
|---|---|
| `CollectLastUsage` 找最后一次使用 | `died[inp] = max(..., t)` |
| 插入 `kill_tensor` 显式释放 | `AllocEntry.died_at` 记录释放时机 |
| `AllocateWorkspace` 按最大需求分一块 | 峰值 = `max(活跃缓冲区之和)` |
| 释放由 VM 运行时真正执行 | 复用决策由 pass 直接算 |

---

## 7. 关键概念速记

- **出生/死亡**：出生=被产生；死亡=最后一个消费者用完
- **生命周期**：[出生, 死亡) 的区间
- **复用条件**：`died_at < 当前时刻` 且 `size >= 需要的`（死得够早、装得下）
- **峰值内存**：任意时刻活跃缓冲区大小之和的最大值 = 目标函数
- **图输出**：永远活到最后（因为外面的人可能还要读）

---

## 8. 课后答疑

**Q：为什么"小张量能复用大缓冲，大张量不能复用小缓冲"？**
A：复用大缓冲 = 占用其前 `sz` 个位置，浪费点空间但安全；
复用小缓冲 = 装不下，越界写坏别人的数据。所以规则要求 `size >= sz`。

**Q：如果两个大算子必须同时活着（比如它们的输入输出互相依赖）？**
A：那它们就都得独占，峰值 = 两者之和。内存规划**不能凭空造空间**，
只能消除"不重叠"的浪费。这也是为什么它省不了"本来就是同时的"内存。

**Q：为什么图输出要活到最后？**
A：因为图外面（调用者）还要读它。如果提前释放，外面读到的就是垃圾数据。
真实编译器里输出张量由运行时管理，甚至可以不参与复用。

**Q：这个分析和"垃圾回收"有什么区别？**
A：GC 是运行时自动找"没人用"的对象；我们是**编译期**静态分析出"何时死"，
然后**预先规划**复用。静态规划是编译器擅长的，零运行时开销。

---

## 9. 本课小结

- 中间张量**不是同时活着**的 → 生命周期不重叠就能共用内存
- 生死由"出生时刻 + 最后一个消费者"决定
- 复用条件两句话：**死得够早、装得下**
- 朴素 848 → 复用 448（2 个缓冲区），省 47%——总分配 = 各缓冲区大小之和
- TVM 对应：`KillAfterLastUse`（找释放点）+ `AllocateWorkspace`（分配）

**下一步**：第 7 课——终于到最后一步了：代码生成。看看一张图
是怎么变成"带循环的真实代码"的，以及那一堆吓人的 `n*192 + c*64 + ...`
下标到底是怎么来的。

---

## 10. 扩展阅读 A：分配策略不止一种——first-fit / best-fit

第 6 课我们用的策略是"第一个死的就用它"（first-fit）。但分配算法有多种，
各有取舍：

| 策略 | 规则 | 优点 | 缺点 |
|---|---|---|---|
| first-fit | 第一个"死得够早且够大"的就用 | 快（O(n)） | 可能把大缓冲留给了小张量，造成"碎片" |
| best-fit | 找"最接近需要大小"的复用 | 碎片少 | 慢一点（O(n) 也要遍历比较） |
| 最坏-fit | 找最大的复用 | 保持大缓冲 | 可能破坏大张量后续复用 |

我们 toycc 用 first-fit（教学最简单）。真实编译器（XLA/TVM）会用
更精细的**图着色 / 区间分配**算法——把"哪些张量能共用"建模成
"区间不重叠"问题，求最优解。

**关键概念：内存碎片**。如果先分配一个大张量（占住大块），
又分配几个小张量，大张量死后留下的大块被小张量"切碎"，
后续大张量反而找不到整块。好的分配算法会尽量减少这种碎片。

---

## 11. 扩展阅读 B：对齐（alignment）——分配里最容易忽略的约束

设备内存对"对齐"有硬性要求：float 数组通常要 4 字节对齐，
向量化（第 11 课）要求 8/16/32 字节对齐，否则 CPU 直接**崩溃**
（Segfault）或性能暴跌。

所以分配器不是"给 n 个元素"就行，而是：

```
实际分配 = align_up(n * 元素大小, 对齐字节数)
```

**副作用**：为了对齐，每个缓冲区会"多分配一点"（padding）。
这解释了为什么真实编译器算内存时，峰值不是简单相加——
还要加上对齐填充。toycc 没做对齐（教学简化），但你要知道
真实系统里"对齐"无处不在。

---

## 12. 扩展阅读 C：从复用到图着色——内存分配的理论化

内存复用问题在理论上很漂亮，值得了解一下（也是面试常聊的）：

把每个张量看成一个"区间"（生命周期）。问题变成：
**给区间着色，让重叠的区间颜色不同，用最少的颜色。**

这是**区间图着色问题**，有高效解法（贪心按结束时间排序即可，
区间图是完美图）。这个问题的答案（最少颜色数）就是
**最大重叠数** = 我们说的"峰值内存"。

**为什么值得知道**？
1. 它告诉你"峰值 = 最大活跃张量数"不是巧合，是图论定理
2. 讨论内存优化时，用"区间着色"这个术语能立刻建立专业形象
3. XLA 等编译器的 buffer assignment 就是这么建模的

**一句话总结**：内存规划 = 区间着色 = 找出最少颜色覆盖所有区间，
答案就是峰值。

---

**导航**：⬅ [上一节](lesson05.md)（第 5 课 · 常量折叠）　｜　[下一节](lesson07.md)（第 7 课 · 代码生成）➡
