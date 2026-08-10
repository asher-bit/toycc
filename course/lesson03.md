# 第 3 课：算子融合——最值钱的优化，也是最容易翻车的优化

> 本课风格：代码驱动 + 用具体数字演算翻车现场。
> 对应文件：`toycc/passes/fusion.py`（82 行）
> 准备：跑 `python -m course.runner 3`，对照看。

---

## 1. 为什么融合能变快？先算一笔"开销账"

在 GPU/CPU 上，执行一个算子不只花"计算时间"，还有固定开销：

```
读输入 → 启动核 → 计算 → 写回内存
↑这些步骤每次调用都要付一遍
```

现在有两个算子连着：

```
conv(x) ──[写中间结果]──> relu ──[写结果]
```

原样执行要付**两次**读写和启动。融合后：

```
conv+relu(x) ──────────────> 直接写结果
```

省掉的是一次"中间结果的写 + 读"。**内存带宽（搬数据）在推理里往往比
计算本身还贵**——融合是"白捡"的性能。

> 真实数字：TVM 里 `conv2d+bias+relu` 融合，在端侧模型上常有 2~4 倍加速。
> 这不是优化了算法，只是省掉了无谓的数据搬动。

> **手算：这笔账到底多大？**
>
> 拿 conv1 的输出举例：`(1, 8, 8, 4)` = 256 个 float32 = 1 KB。
> "写一次 + 读一次"中间结果 = 搬 2 KB。假设内存带宽 100 GB/s，
> 搬 2 KB 要 `2×1024 / 1e11 = 20 ns`。看起来微不足道？
>
> 但推理是**海量重复**的：一个 50 层网络、每秒 1000 次推理，
> 光"中间结果搬运"就累计可观。更关键的是——**这 20ns 是纯浪费**，
> 计算本身一点没少，只是数据白搬了一趟。融合就是把这个"白搬"清零。
>
> **为什么带宽比计算贵？** 因为硬件的算力增长远快于带宽增长（第 15 课）。
> 一个算子的计算强度（FLOP/字节）如果低于某个阈值，它就是"带宽受限"——
> 再强的算力也只能干等数据。relu/add 这种逐元素算子，计算强度极低，
> **天生带宽受限**，所以融合它们收益最大。这就是第 19 课 roofline 的预演。

---

## 2. 什么能融合？三条规则（先背下来）

我们的 toycc 定义了三条直觉规则（在 `fusion.py` 顶部）：

```python
# 可被吸收进"计算根算子"的点算子
_POINTWISE = {"relu", "sigmoid"}     # 逐元素算子
# 可当 bias 吸收进 conv/matmul 的加法
_BIASABLE = {"add"}
# 作为计算根、可以吸收下游的算子
_ROOTS = {"conv", "matmul"}
```

**规则 1（激活）**：`relu(conv(x))` → `conv(x, activation="relu")`
因为 relu 是逐元素的：输出的每个元素只依赖 conv 输出的对应位置，可原地做。

**规则 2（偏置）**：`conv(x) + bias` → `conv(x, w, bias)`
bias 按通道广播、且是常量，可以直接并进卷积里。

**规则 3（禁止）**：根算子有**多个消费者**时，不能随便内联。
后面细说，这是最容易踩的坑。

> **原理深挖：为什么"逐元素"就能融合，"带窗口/归约"就难？**
>
> 三条规则的本质是**依赖的"形状"**：
> - relu/add 是**逐元素**——输出 `[i]` 只依赖输入 `[i]`。融合进 conv 后，
>   只需在 conv 算出 `[i]` 后"顺手"过一遍 relu，不需要等整张图算完。
> - 而 conv/matmul 是**带窗口/归约**的——输出 `[i]` 依赖输入的**一整片**。
>   这种算子融合进另一个，就要协调"哪片数据先算"，复杂得多。
>
> 这就是为什么融合规则按"算子类型"分：逐元素最安全，归约最难。
> 真实 TVM 用 `op_pattern`（第 12 课）给每个算子标注这个"依赖形状"，
> 融合算法照着标注决定。你现在背的"三条规则"，就是 `op_pattern` 的简化版。

---

## 3. 逐行读 `FusionPass`

### 3.1 入口：`__call__`

```python
class FusionPass:
    def __call__(self, graph: Graph) -> Graph:
        # 用副本改写,保持 pass 无副作用(可组合/可重复跑)
        g = graph.clone()
        for node in list(g.nodes.values()):
            if node.op_type not in _ROOTS:
                continue
            self._absorb_followers(g, node)
        return g
```

**两个设计点**：

1. **`graph.clone()`**：在副本上改写。为什么？如果 pass 直接改原图，
   那"先融合再布局"和"先布局再融合"就没法自由组合了（原图被第一个 pass 毁了）。
   **pass 无副作用**，才能任意排列、重复执行。真实编译器里 pass 也是
   "输入不可变，输出是新对象"。

2. **只对"计算根"动手**：只有 `conv`/`matmul` 能当融合的"锚点"，
   从它出发往下吸收。`relu`/`add` 自己不能当根（它们没有可吸收的形态）。

### 3.2 核心循环：`_absorb_followers`

```python
def _absorb_followers(self, g: Graph, node):
    while True:                                   # 一直吸,直到吸不动
        cons = g.consumers(node.name)
        if len(cons) != 1:                        # 规则3:多个消费者不融
            break
        c = cons[0]

        # 情况 1:relu/sigmoid —— 变成父算子的 activation 属性
        if c.op_type in _POINTWISE:
            node.attrs.activation = c.op_type
            node.fused_ops = (node.fused_ops or []) + [c.op_type]
            self._absorb(g, node, c)
            continue

        # 情况 2:add —— 且另一个操作数是常量,当 bias 吸收
        if c.op_type in _BIASABLE and node.op_type in _ROOTS:
            other = [i for i in c.inputs if i != node.name]
            if len(other) == 1 and _is_constant(g, other[0]):
                node.attrs.bias = True
                node.inputs.append(other[0])
                node.fused_ops = (node.fused_ops or []) + [c.op_type]
                self._absorb(g, node, c)
                continue

        break   # 其余情况不融合
```

**这是个 while 循环**：每次吸收一个消费者，然后**再看新的消费者**，继续吸。
这就是"沿唯一消费者链一路吃下去"。

**情况 1 逐行**：
- `cons = g.consumers(node.name)`：看看谁在吃我的输出
- `len(cons) != 1` → break：**两个以上消费者不融**（为什么？见 FAQ）
- `node.attrs.activation = c.op_type`：把 relu 的"身份"记到父算子属性里，
  表示"我输出前要先过 relu"（参考执行器看到这个属性就会套 relu）
- `fused_ops` 追加记录：方便打印"这个核融合了哪些算子"
- `_absorb(g, node, c)`：真正把 c 从图里摘掉，下面讲

**情况 2 逐行**：
- 消费者是 `add`，且我自己是根（conv/matmul）
- `other = [i for i in c.inputs if i != node.name]`：add 有两个输入，
  一个是我的输出，另一个"另一个操作数"是谁？
- 要求：另一个操作数**恰好一个**，而且**是常量**（权重/bias 之类）
- 满足就 `node.inputs.append(other[0])`：把 bias 变成我的第三个输入
- `node.attrs.bias = True`：标记"我自带偏置"

**为什么 bias 必须是常量？** 因为把 bias 并进 conv 是"编译期决策"，
如果 bias 是运行时才给的值，你还得想办法传参，语义也变复杂。
只有常量才能安全、无条件地吸收。

### 3.3 `_absorb`：怎么把子节点"摘"掉

```python
@staticmethod
def _absorb(g: Graph, parent, child):
    g.rewire(child.name, parent.name)   # 所有用 child 的地方改指向 parent
    g.remove_node(child.name)           # 删掉 child 节点
```

**就两行**。回顾 `graph.py` 的 `rewire`：

```python
def rewire(self, old_name, new_name):
    for n in self.nodes.values():
        n.inputs = [new_name if i == old_name else i for i in n.inputs]
    self.outputs = [new_name if o == old_name else o for o in self.outputs]
```

把图里所有"指向 child 名字"的边，改成指向 parent。
然后删掉 child 节点。**吸收完成。**

> 这就是整个融合 pass 的全部：**改属性 + 重连边 + 删节点**。

### 3.4 `_is_constant`

```python
def _is_constant(graph, name):
    return name in graph.inputs
```

判断张量是不是常量：只要它是个图输入（权重、bias 都是），就算常量。
（我们 toycc 简化了——真实编译器会检查"是否真的绑定了常量值"。）

---

## 4. 用手算演算一次完整的融合

我们的模型前半段：

```
conv1(x, conv1_w) → bias_add1(conv1, bias1) → relu1(bias_add1) → conv2(...)
```

**`FusionPass` 对 conv1 的处理过程**：

```
第1轮循环:
  consumers(conv1) = [bias_add1]          ← 恰好1个,继续
  bias_add1 是 add, 另一个操作数 bias1 是常量
  → conv1.attrs.bias=True, conv1.inputs = [x, conv1_w, bias1]
  → rewire(bias_add1 → conv1): 所有用 bias_add1 的地方改成 conv1
     此时 relu1 的输入从 bias_add1 变成 conv1
  → 删掉 bias_add1
第2轮循环:
  consumers(conv1) = [relu1]              ← 又恰好1个,继续
  relu1 是 relu → conv1.attrs.activation="relu"
  → rewire(relu1 → conv1): conv2 的输入从 relu1 变成 conv1
  → 删掉 relu1
第3轮循环:
  consumers(conv1) = [conv2]              ← conv2 是根,不是可吸收的形态
  → break
```

最终：`conv1 = conv(x, conv1_w, bias1)` + `activation=relu`。

**见证图的变化**——融合前 3 个节点：

```
conv1 = conv(x, conv1_w)
bias_add1 = add(conv1, bias1)
relu1 = relu(bias_add1)
```

融合后 1 个节点：

```
conv1 = conv(x, conv1_w, bias1)  <- fused: add+relu
```

`<- fused: add+relu` 就是 `fused_ops` 打印出来的，告诉你这个核的来历。

---

## 5. 翻车现场：为什么"顺序"错了就会算错

实验里故意造了个**顺序不对**的图：

```
conv1(x, w) → r(relu) → out(add bias)      ← 注意: 先 relu, 再加 bias
```

我们的融合 pass 会照吸不误，把它并成 `conv(x, w, bias)` + `activation=relu`。
但数学上：

```
融合前:  out = relu(conv(x)) + bias     ← 先 relu 再加
融合后:  out = relu(conv(x) + bias)     ← 先加再 relu
```

**这俩不相等！** 因为 relu 不是线性函数，加法**不能穿过**它。

用具体数字验证（假设 `conv(x) = -3`，`bias = 2`）：

```
融合前:  relu(-3) + 2 = 0 + 2 = 2
融合后:  relu(-3 + 2) = relu(-1) = 0
2 ≠ 0    → 错!
```

实验跑出来的 `max|Δ| = 2.250e-01` 就是这种错误的宏观体现。

**这正是参考执行器存在的意义**：一个看着"合理"的融合，数值验证立刻翻车，
逼你去检查规则。你在经历 TVM 开发者的日常：**写 pass → 数值验证 → 发现
语义被破坏 → 修规则**。

> 现实中的融合 pass 有一堆这种"规则健全性"检查，防止各种顺序组合出错。
> 我们 toycc 的融合 pass 是"默认信任顺序"，所以它不对这样的图设防——
> 这恰好让你看到了"不设防"的后果。

---

## 6. 真实 TVM 对照：`src/relax/transform/fuse_ops.cc`

真实文件 1514 行，核心注释开宗明义：

```cpp
/*
  Note on Fusing algorithm:
  核心挑战: 处理菱形分支(diamond shape)。
            conv2d
            /  |  \
           /   |   \
         op    op   op
          \    |    /
           \   |   /
          elemwise add

  做法:
  - 建数据流 DAG, 做支配分析(dominator analysis)
  - 构造后支配树(post-dominator tree)
  - CheckPath: 检查源节点到后支配节点之间所有路径是否满足融合条件
  - CommitFuse: 把这段路径上的节点标成同一组
  - 用 Union-Find 并查集管理组
*/
```

**为什么真实 TVM 用"后支配树"而不是我们的贪心？**

看上面那个菱形：`conv2d` 的输出分三路，经过三个 op，汇合到 add。
- 贪心：每个 op 都是 add 的输入，逐个尝试"absorb"会很乱，还容易漏
- 后支配树：能算出"add 必然在 conv2d 之后执行"，把中间整段安全地圈成一组

我们的 toycc 用 `len(consumers) != 1` 直接放弃这种结构——**保守但不犯错**。
这是 toycc 和真框架最典型的一处差距，理解这个差距就理解了支配分析的动机。

入口函数 `FuseOps` 三步走：

```cpp
IRModule FuseOps(IRModule mod, int opt_level, size_t max_fuse_depth) {
  // Step 1. 建图
  IndexedForwardGraph graph = GraphCreator::Create(mod, &arena);
  // Step 2. 划分: 后支配树 + 并查集, 得到若干融合组
  std::vector<GraphPartitioner::Group*> groups =
      GraphPartitioner(...).Partition(graph);
  // Step 3. 按分组改写 IRModule
  return OperatorFusor(mod, graph, groups, /*lift_constants*/ true).Transform();
}
```

对照表：

| 真实 TVM (fuse_ops.cc) | toycc (fusion.py) |
|---|---|
| `GraphCreator::Create` 建图 | `Graph` + `consumers` |
| `GraphPartitioner` 后支配树分组 | `_absorb_followers` 贪心 |
| `OperatorFusor` 生成融合函数 | `_absorb`（rewire+删节点） |
| `FuseOps()` pass 工厂 | `register_pass("fusion")` |

pass 注册（真实代码）：

```cpp
Pass FuseOps(int fuse_opt_level) {
  auto pass_func = [=](IRModule m, PassContext pc) {
    int opt_level = fuse_opt_level == -1 ? pc->opt_level : fuse_opt_level;
    ...
    return relax::FuseOps(m, opt_level, max_fuse_depth);
  };
  return CreateModulePass(/*pass_function=*/pass_func,
                          /*opt_level=*/0,
                          /*name=*/"FuseOps",
                          /*required=*/{});
}
```

注意 `opt_level` 从 `PassContext` 取——真实编译器里 pass 会根据
优化级别（O0/O1/O2）决定跑不跑。我们 toycc 的 `run_passes` 用字符串名调度，
是它的极简版。

---

## 7. 实验

```bash
python -m course.runner 3
```

两段输出分别对应"正确融合"（Δ=0）和"错误融合"（Δ=2.25e-01）。
第二段就是上面第 5 节翻车现场的真实执行。

---

## 8. FAQ

**Q：为什么"多个消费者"就不能融合？**
A：看例子：`y1 = relu(conv(x))`，且 `y2 = add(conv(x), z)`。
如果把 relu 融进 conv，conv 的输出语义就变成了"已过 relu"，
y2 就会错误地多算一次 relu。要么不融，要么复制一份 conv——
复制又是另一种复杂优化（code duplication）。所以默认不融最安全。

**Q：融合会不会让单个核太大反而慢？**
A：会。极端情况下一个核塞几百个算子，寄存器不够用、缓存爆炸。
真实 TVM 有 `max_fuse_depth`（默认 256）限制融合深度——你会在
`FuseOps` 里看到这个配置。我们 toycc 没设上限（教学简化）。

**Q：`fused_ops` 这个记录除了打印还有什么用？**
A：调试（知道核从哪来）；给后端提示（如 GPU 上 relu 可以走硬件融合指令）；
某些 pass 需要知道"这是复合算子"来决定怎么处理。

**Q：融合和"算子合并"（如把 3 个 add 合成 1 个）是一回事吗？**
A：不一样。融合是把"根+逐元素"并成一个核；合并是代数化简
（如 `add(add(a,b),c)` → `add(a, b+c)`）。我们 toycc 只做了融合。

---

## 9. 本课小结

- 融合省的是**内存带宽 + 核启动开销**，不是计算量
- 三条规则：激活可融、常量 bias 可融、**多个消费者不融**
- 实现 = 改父算子属性 + `rewire` 重连边 + 删节点，循环直到吸不动
- **铁律：融合必须语义等价**——relu 不能穿过加法
- 参考执行器抓融合 bug（现场翻车 2.25e-01）
- 真实 TVM 用后支配树处理菱形分支，有 `max_fuse_depth` 上限

**下一步**：第 4 课，布局优化。这次主角从"算子怎么合并"变成
"同一个张量，换个内存排法就更快"。

---

## 10. 深层拓展 A：融合到底省了哪几笔开销？（逐笔算）

"融合省开销"太抽象。我们把一次算子执行的开销拆开看：

| 开销项 | 说明 | 融合的影响 |
|---|---|---|
| 读输入 | 从内存/缓存读数据 | 不变（还是要读） |
| **写中间结果** | 把输出写到内存 | **省掉**（直接接着算） |
| **读中间结果** | 下一个算子再读回来 | **省掉** |
| **核启动** | 建立 kernel、调度、同步 | **省掉一次** |
| **循环开销** | 每个循环的边界检查等 | 少一层边界 |

**最值钱的是"写+读中间结果"**。为什么？因为内存带宽（数据搬运速度）
在推理里通常比计算单元更稀缺——搬 1 个字节的时间 ≈ 算几十次浮点的时间。
融合把"搬一次来回"省了，等于把带宽瓶颈松了一口。

**工程直觉**：判断要不要融合，就看**中间结果的体积**。
中间结果越大（通道多、分辨率高），融合越赚。

### 寄存器压力：融合的"另一面"

融合不是无限制的。把太多算子塞进一个核，会导致**寄存器不够用**：

```
核里有 10 个算子, 每个需要 8 个中间寄存器 → 需要 80 个寄存器
但 GPU/CPU 只有 ~64 个可用 → 内存溢出(register spilling)
→ 中间值被迫写回内存 → 反而更慢!
```

这就是为什么 TVM 有 `max_fuse_depth`（默认 256）限制融合规模，
以及成本模型（第 13 课）会评估"融合多少最划算"。**不是融得越多越好**。

---

## 11. 深层拓展 B：多消费者场景到底有多常见？（菱形详解）

第 5 节说"多个消费者不融合"。但多消费者很常见，值得画清楚。

**场景 1：两个消费者（最常见的禁用原因）**

```
         conv(x)
        /       \
     relu       add(→z)
      |
     y1          y2
```

`relu` 要融进 `conv`？不行——`conv` 的输出还被 `add` 用着。
如果强行融，`add` 会拿到"已经过 relu"的错误数据。
**除非**：复制一份 `conv` 给 `add`——这叫 code duplication，
省了融合收益却多了计算量。编译器要权衡。

**场景 2：菱形分支（融合算法真正的难点）**

```
         conv
        /  |  \
      op   op  op
       \   |   /
        add        ← 汇合点
```

每个 `op` 都只有一个消费者（add），单独看都能融。
但三者都融进 add 后，add 变成一个"三输入的超大核"，可能太大。
**后支配树**（第 12 课）能判断"add 是必经汇合点，可以安全合并"，
这是 toycc 贪心做不到的。

**结论**：多消费者不是"禁止融合"的绝对铁律，而是"要小心权衡"的信号。
我们 toycc 选择保守（不融），真实编译器用分析来决定。

---

## 12. 深层拓展 C：融合后的"核"怎么表示？——从属性到子图

toycc 用 `node.attrs.activation` + `fused_ops` 列表表示"融合了谁"。
但真实编译器（Relay/Relax）用**子图（fused function）**：

```
融合前:  3 个独立的 relax.call_tir
融合后:  1 个 fused_function, 内部是 3 个原始算子的子图
```

**为什么用子图而不是属性？** 因为：
1. 融合可以嵌套/组合任意形状（不只是"根+激活"），属性表达不了
2. 子图可以**整体交给后端**（第 7 课的 `relax.ext.*`），后端看到完整的子图
3. pass 能进一步优化子图内部

我们 toycc 用属性是"教学简化"——够展示概念，但你要知道真框架
用的是子图。第 8 课读 `fuse_ops.cc` 的 `FunctionCreator` 时
你会看到它"为每个融合组创建新函数"——就是子图。

---

## 13. 思考题（加深版）

1. 融合省掉的"写+读中间结果"，为什么在带宽受限的硬件上尤其值钱？
2. `max_fuse_depth` 太小会怎样？太大又会怎样？
3. 为什么真实编译器用"子图"表示融合，而不是一个"融合属性"？

> 答案：
> 1. 带宽受限时，"搬数据"比"算数据"还贵，省一次搬运=省一笔大开销。
> 2. 太小：错过融合收益；太大：寄存器溢出、缓存爆炸，反而更慢。要平衡。
> 3. 子图能表达任意融合结构、能整体交给后端优化、可继续被 pass 处理；
>    单个属性表达不了这些。

---

**导航**：⬅ [上一节](lesson02.md)（第 2 课 · 参考执行器）　｜　[下一节](lesson04.md)（第 4 课 · 布局优化）➡
