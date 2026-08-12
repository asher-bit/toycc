# 第 1 课：计算图与 IR——编译器"看见"模型的方式

> 本课风格：**代码驱动，逐行拆解**。我们把 `toycc/ir/graph.py`（146 行）
> 从头读到尾，每读一段代码，就用手算的小例子验证它在干嘛。
>
> 准备：先跑一次 `python -m course.runner 1`，把输出放旁边对照。

---

## 1. 从一个"两行命令"的直觉开始

编译器最核心的需求：**能用程序读、改、遍历一份"计算的描述"**。

你想一想，如果模型是 PyTorch 的 `nn.Module`，编译器怎么去"改"它？
——很别扭，因为 `nn.Module` 是给"人类写"的，不是给"程序改"的。

所以编译器造了自己的数据结构，就是 **IR（Intermediate Representation，
中间表示）**。它只保留编译器需要的信息，丢掉别的东西。
这个 IR 长什么样？就是我们说的**计算图（computational graph）**。

**计算图 = 一张有向无环图（DAG）**
- 节点 = 一个算子（卷积、加法、relu...）
- 边 = 张量的流动（谁把结果喂给谁）

---

## 2. 逐行读 `toycc/ir/ops.py`：先知道"有哪些算子"

打开文件，先看最底下几行——**算子注册表**：

```python
register("conv", 2, _conv_shape, None)        # 权重是固定输入 #1
register("matmul", 2, _matmul_shape, None)
register("relu", 1, _pointwise_shape, None)
register("sigmoid", 1, _pointwise_shape, None)
register("add", 2, _add_shape, None)          # 广播加法(也当 bias 用)
register("mul", 2, _add_shape, None)
register("reshape", 1, _reshape_shape, None)  # 内存视图变换(不改数据)
register("layout_transform", 1, _layout_transform_shape, None)  # 布局搬移
```

### 逐参数拆解 `register`

```python
def register(name, num_inputs, infer_shape, ref):
    OPS[name] = OpInfo(name, num_inputs, 1, infer_shape, ref)
```

每个算子登记四样东西：

| 参数 | 含义 | 例子(conv) |
|---|---|---|
| `name` | 算子的唯一名字 | `"conv"` |
| `num_inputs` | 固定要几个输入 | `2`（数据 + 权重） |
| `infer_shape` | 怎么从输入形状推输出形状 | `_conv_shape` |
| `ref` | 用 numpy 怎么算（参考实现） | 第 2 课才填 |

**为什么要"注册表"而不是硬编码？** 因为后面所有 pass 都是"看算子名字 + 属性"
来决策的。把它们集中登记，新增一个算子只需要注册一次，全编译器都能用。
真实 TVM 里也有等价物：`Op::Get("relax.call_tir")`，只不过用 C++ 写、
登记了几千个算子。

> **原理深挖：为什么 `infer_shape` 和 `ref` 要分开成两个函数？**
>
> 注意 `register` 同时登记了 `infer_shape`（推输出形状）和 `ref`（真正计算）。
> 为什么不能合成一个"直接算"？因为**编译期只要形状，不要数值**。
>
> - **推形状**：编译器优化时只需要知道"输出是 (1,16) 还是 (1,8,8,4)"，
>   用来排内存、查布局——这时候**权重还没加载，数值不存在**，只能算形状。
> - **算数值**：那是运行时的事，或者常量折叠时的事。
>
> 把两者拆开，编译器就能在**没有真实数据**的情况下，先把整张图的
> 形状、内存、布局全规划好。这就是"编译期 vs 运行时"二分法（第 0 课）
> 在 IR 设计里的第一个具体体现。真实 TVM 里这叫 `FInferShape` / `FCompute` 分离。

### 属性（attrs）是什么？

再往上看 `OpAttrs`（dataclass）。它是**每个算子实例的配置**：

```python
@dataclass
class OpAttrs:
    kernel: tuple[int, int] | None = None   # conv 核大小 (kh, kw)
    stride: tuple[int, int] | None = None   # conv/maxpool 步长
    pad: tuple[int, int, int, int] | None = None
    groups: int = 1
    bias: bool = False
    activation: str | None = None           # 融合进本算子的激活名
    target_shape: tuple | None = None       # reshape 的目标形状
    layout: str | None = None               # "nchw"/"nhwc"/None
    from_layout / to_layout                  # layout_transform 专用
```

**关键理解：op_type 是"哪一类算子"，attrs 是"这个算子具体的参数"。**
`conv` 是一种算子，但 `kernel=3、stride=1、pad=1` 是这个卷积的配置。
真实 TVM 里 attrs 就是一堆 `IntImm("kernel_size", 3)` 这样的键值对。

---

## 3. 逐行读 `toycc/ir/graph.py`

### 3.1 Node：一个算子的"身份证"

```python
@dataclass
class Node:
    name: str                          # 在整张图里的唯一名字
    op_type: str                       # 哪种算子(必须是注册表里的)
    inputs: list[str] = field(default_factory=list)   # 依赖哪些张量
    attrs: OpAttrs = field(default_factory=OpAttrs)   # 具体参数
    fused_ops: Optional[list[str]] = None              # 融合来源(第3课)
```

一行行看：

- **`name`**：图里张量的名字就是"节点名"。`conv1` 这个节点，它的输出张量也叫 `conv1`。
  这是我们的设计约定：**张量名 = 产生它的节点名**。
- **`inputs`**：这算子的输入是哪些张量名。`inputs=["x", "conv1_w"]` 表示
  这个卷积吃 `x` 和权重 `conv1_w`。
- **`fused_ops`**：现在用不到，先记住它是第 3 课的"融合来源记录"。

**注意一个反直觉点**：`Node` 里**没有**存"输出张量的形状"。为什么？
形状是"推导"出来的（后面 pass 用），不需要手工存。这就是 IR "只存必要信息"的体现。

### 3.2 校验：`__post_init__`

```python
def __post_init__(self):
    if self.attrs is None:
        self.attrs = OpAttrs()
    if self.op_type not in OPS:
        raise KeyError(f"未知算子 {self.op_type!r},可用的有 {sorted(OPS)}")
```

创建一个 Node 时自动执行：① 如果没给 attrs 就补一个空的；② **如果算子名不在
注册表里，直接报错**。这保证了"图里出现的每个算子都是注册过的"——很早就把
拼写错误拦下来。真实编译器管这叫 **IR 合法化检查**。

### 3.3 Graph：整张图的容器

```python
class Graph:
    def __init__(self, name: str = "main"):
        self.name = name
        self.inputs: list[str] = []          # 图的输入张量名
        self.nodes: dict[str, Node] = {}     # name -> Node
        self.outputs: list[str] = []         # 图的输出张量名
        self.constants: dict[str, object] = {}  # 常量张量值
```

两个列表 + 两个字典，就这么简单：

| 字段 | 类型 | 存什么 |
|---|---|---|
| `inputs` | `list[str]` | 图的输入张量名（如 `x`、各权重） |
| `nodes` | `dict[str, Node]` | **所有算子**，用名字当 key |
| `outputs` | `list[str]` | 图的输出张量名 |
| `constants` | `dict` | 编译期已知的常量值（第 5 课用） |

**图为什么不用"边"的数据结构？** 因为每条边就是"某个 node 的 inputs 里出现
的那个名字"。`node.inputs` 里的名字，要么是图输入，要么是另一个 node 的名字。
所以**图的结构信息全在 inputs 列表里**，不需要单独的边表。这是最精简的实现。

### 3.4 建图 API：add_input / add_op / mark_output

```python
def add_input(self, name: str) -> Node:
    self.inputs.append(name)
    return self._fake_node(name, "placeholder")
```

`add_input` 只登记名字，**不创建真正的节点**——输入不是一个"计算"。
`_fake_node` 用 `Node.__new__(Node)` 绕过了校验（因为 `placeholder` 不在注册表），
只是给我们返回一个占位对象。

```python
def add_op(self, op_type, inputs, name=None, attrs=None, fused_ops=None) -> Node:
    info = OPS[op_type]
    if len(inputs) != info.num_inputs:
        raise ValueError(
            f"{op_type} 需要 {info.num_inputs} 个输入,给了 {len(inputs)}: {inputs}")
    if name is None:
        name = f"{op_type}_{len(self.nodes)}"
    node = Node(name, op_type, list(inputs),
                attrs=attrs or OpAttrs(), fused_ops=fused_ops)
    self.nodes[name] = node
    return node
```

`add_op` 是建图的主力，干四件事：
1. **查注册表**，拿到这个算子的声明信息 `info`
2. **校验输入个数**——`conv` 要求 2 个输入，你给 1 个就报错
3. 没给名字就自动起一个（`conv_3`、`relu_4`...）
4. 创建 Node 存进 `self.nodes`

> **小实验**：你可以在任意 Python 里试——
> `Graph().add_op("conv", ["x"], "c")` 会怎样？答：报错，因为 `conv` 要求
> 2 个输入。这种"尽早报错"是编译器很重要的设计习惯。

```python
def mark_output(self, name: str):
    if name not in self.nodes and name not in self.inputs:
        raise KeyError(f"输出 {name} 不存在")
    self.outputs.append(name)
```

标记"谁是最终输出"——也就是**图被消费的终点**。

### 3.5 图分析（这是全课最重要的三个函数）

#### ① consumers：谁在消费我这个张量？

```python
def consumers(self, name: str) -> list[Node]:
    return [n for n in self.nodes.values() if name in n.inputs]
```

**逐行**：
- `self.nodes.values()` 遍历所有算子
- `name in n.inputs` 判断"这个算子是否把 `name` 当输入"
- 筛选出来的就是 `name` 的消费者

**为什么重要？** 全课到处用它：
- 第 3 课融合：`len(consumers) != 1` 就不能融合
- 第 6 课内存：最后一个消费者决定张量何时"死亡"

> **手算例子**：图里有 `bias_add1 = add(conv1, bias1)`，
> 那 `consumers("conv1")` 返回 `[bias_add1]`。如果还有别的算子也吃 `conv1`，
> 就会返回好几个——这正是"能不能融合"的关键信息。

#### ② topo_order：按什么顺序执行？

```python
def topo_order(self) -> list[Node]:
    # 简单 Kahn 算法:入度 = 依赖的节点个数(图输入不是节点,不算)
    indeg = {n: sum(1 for i in node.inputs if i in self.nodes)
             for n, node in self.nodes.items()}
    ready = [n for n, d in indeg.items() if d == 0]
    order = []
    while ready:
        n = ready.pop(0)
        order.append(self.nodes[n])
        for c in self.nodes.values():
            if n in c.inputs:
                indeg[c.name] -= 1
                if indeg[c.name] == 0:
                    ready.append(c.name)
    if len(order) != len(self.nodes):
        raise RuntimeError("图中存在环,不是合法的 DAG")
    return order
```

这是**拓扑排序**，Kahn 算法。逐行：

1. **算入度**：每个节点依赖几个"别的节点"。
   ```python
   indeg = {n: sum(1 for i in node.inputs if i in self.nodes) ...}
   ```
   `node.inputs` 里有的是图输入（`x`、权重），不算节点，不计入度；
   有的是别的节点的名字，计入度。入度为 0 = 没有前置依赖，可以先执行。

2. **ready 队列**：先放所有入度为 0 的节点。

3. **主循环**：
   - 从 `ready` 弹一个出来放进结果
   - 它作为"已执行"，把依赖它的节点的入度减 1
   - 减到 0 就说明"我依赖的都执行完了"，可以进 ready

4. **防环**：如果最后 `order` 数量不等于节点总数，说明有环，报错。

> **为什么必须是无环的？** 如果 `a 依赖 b` 且 `b 依赖 a`，谁先执行？
> 死锁。拓扑排序就是来发现这种非法图的。

> **原理深挖：为什么"执行顺序"是编译器的命门？**
>
> 你可能会想：反正每个算子的输入输出都写明了，顺序有那么重要吗？重要，
> 而且是后面三件事的共同基础：
> - **代码生成（第 7 课）**：生成的 C 代码就是按拓扑序一行行写的。
>   顺序错了，`bias_add1` 还没算出来，`relu1` 就去用它——直接编译报错或算错。
> - **内存规划（第 6 课）**：要算"每个张量活到第几步"，前提是先有"第几步"
>   这个序列。没有拓扑序，"生命周期"无从谈起。
> - **参考执行（第 2 课）**：执行器就是按拓扑序挨个算。
>
> **一句话**：拓扑序把"一张图"压成"一条时间线"。后面所有"按时间分析"
> 的优化（内存、调度）都建立在这条时间线上。这就是为什么它在 `graph.py`
> 里，而不是某个 pass 里——它是图的基础能力。

> **手算例子**：对 `conv1 → bias_add1 → relu1` 这条链：
> `conv1` 入度 0 → 先执行；`bias_add1` 依赖 conv1，减到 0 → 再执行；
> `relu1` 同理。顺序就是链的顺序。

> **手算：拿真实模型完整走一遍 Kahn 算法**
>
> 取模型里的 6 个节点：`conv1(x, conv1_w)`, `bias_add1(conv1, bias1)`,
> `relu1(bias_add1)`, `conv2(relu1, conv2_w)`, `bias_add2(conv2, bias2)`,
> `relu2(bias_add2)`。注意 `x`/`conv1_w`/`bias1` 是图输入，不计入度。
>
> ```
> 初始入度: conv1=0, bias_add1=1, relu1=1, conv2=1, bias_add2=1, relu2=1
> ready=[conv1]
>
> 弹 conv1  → 顺序[conv1]
>   conv1 是 bias_add1 的输入 → bias_add1 入度 1→0 → ready=[bias_add1]
> 弹 bias_add1 → 顺序[conv1, bias_add1]
>   relu1 入度 1→0 → ready=[relu1]
> 弹 relu1 → 顺序[conv1, bias_add1, relu1]
>   conv2 入度 1→0 → ready=[conv2]
> 弹 conv2 → [conv1, bias_add1, relu1, conv2]
>   bias_add2 入度 1→0 → ready=[bias_add2]
> 弹 bias_add2 → [..., bias_add2]  → relu2 入度 1→0
> 弹 relu2 → 全部 6 个节点出列 ✓
> ```
>
> 每一轮都保证一件事：**出列的节点，它的输入一定已经算完了**。
> 这就是拓扑序的全部含义。真实 TVM 里 2000+ 个节点的图也是同一个算法，
> 只是把"遍历所有节点找消费者"换成邻接表，从 O(N²) 优化到 O(N)。

#### ③ dump：把图"画"给人看

```python
def dump(self) -> str:
    lines = [f"graph {self.name}:"]
    for i in self.inputs:
        lines.append(f"  input: {i}")
    for n in self.topo_order():
        lines.append(f"  {n}")       # 调用了 Node.__str__
    lines.append(f"  output: {', '.join(self.outputs)}")
    return "\n".join(lines)
```

`Node.__str__` 把节点拼成一行字：

```python
def __str__(self):
    if self.fused_ops:
        inner = "+".join(self.fused_ops)
        extra = f"  <- fused: {inner}"
    else:
        extra = ""
    return f"{self.name} = {self.op_type}({', '.join(self.inputs)}){extra}"
```

所以 `conv1 = conv(x, conv1_w)` 就是"conv1 这个算子，类型是 conv，
输入是 x 和 conv1_w"。`<- fused: add+relu` 是融合来源标记（第 3 课）。

---

## 4. 对照真实模型文件 `toycc/examples/model.py`

现在回头看 `build_model_with_weights()`，你应该能完全读懂了：

```python
g = Graph("tiny_cnn_const")
g.add_input("x")                          # 输入张量
for name in w:
    g.add_input(name)                     # 权重也当输入
    g.set_constant(name, w[name])         # 顺带把值存进常量表

c1 = g.add_op("conv", ["x", "conv1_w"], "conv1", OpAttrs(kernel=(3,3), stride=(1,1), pad=(1,1,1,1)))
b1 = g.add_op("add", ["conv1", "bias1"], "bias_add1")
r1 = g.add_op("relu", ["bias_add1"], "relu1")
...
g.mark_output("output")
```

**逐行对应**：
- `add_input("x")` → 图的输入
- `add_op("conv", ["x", "conv1_w"], ...)` → conv1 吃 x 和权重
- `set_constant` → 把权重数值存在图里，第 5 课常量折叠要用
- `mark_output("output")` → 声明最终输出

跑一下 `python -m course.runner 1`，对照输出，图上每个 `=` 左边是名字、
右边是 `算子类型(输入列表)`。这就是"编译器眼里的模型"。

---

## 5. 三个 pass 共用的一张图：为什么设计成这样

**一个很重要的思想**：`Graph` 只负责"存图 + 分析图"，**不负责优化**。
优化全在 `passes/` 里写。为什么分开？

- pass 可以任意组合、重复跑（`run_passes(g, ("fusion","layout",...))`）
- 每个 pass 只通过公开 API（`consumers`/`rewire`/`clone`）动图，互不干扰
- 想加新优化，只写新 pass，不碰 IR

这跟真实编译器一致：IR 是"稳定的地基"，pass 是"随便盖的楼"。
TVM 里 IR 用 C++ 对象（`Expr`/`IRModule`），pass 用 `ExprMutator` 改写，
职责划分和我们的 `Graph`/pass 一模一样。

---

## 6. 课后答疑

**Q：为什么 `Node` 不存形状？**
A：形状可以由 `infer_shapes`（第 6 课会看到）从输入一路推出来。
存了就重复了，而且可能和推导结果不一致。编译器里"信息单一来源"是重要原则。

**Q：为什么图输入和权重都放在 `inputs` 里？**
A：因为对 pass 来说，"这是个输入"就够了。是不是常量、值是多少，
看 `constants` 表。这样统一，pass 写起来简单。

**Q：`topo_order` 每次调用都重新算，慢吗？**
A：图很小没关系。真实编译器里拓扑序是**缓存的**，图变了才重算。
我们为了教学清晰，每次现算。

**Q：图和"张量"是什么关系？**
A：图节点是算子；算子的输出就是张量，张量的名字 = 节点名。
所以"张量 `conv1`"和"节点 `conv1`"是同一个东西的两个视角。

---

## 7. 本课小结

- IR = 计算图：节点(算子) + 边(张量名)，**不存布局/内存/循环**
- `Node`：名字、算子类型、输入、属性；校验保证只出现已注册算子
- `Graph`：inputs / nodes / outputs / constants 四个容器
- 三个分析函数是全场主角：`consumers`（谁吃我）、`topo_order`（谁先跑）、`dump`（画图）
- IR 是稳定的地基，pass 是随便盖的楼

**下一步**：第 2 课——光有图不会"跑"是没用的，我们给图装上一个
**能算出结果的执行器**，它将成为所有优化的"裁判"。

---

## 8. 扩展阅读 A：IR 设计的几个"隐藏选择"

我们的 `Graph` 看似随意，其实藏着几个**编译器设计的根本选择**。
搞懂它们，你就能看懂 ONNX/Relay/LLVM 的 IR 为什么长那样。

### 8.1 为什么叫"中间表示"？——它是夹在两端的中间产物

```
前端(模型格式)  ──>  IR  ──>  后端(机器码/代码)
   上层语言           中间层        下层语言
```

IR 的价值就是"翻译的中间站"：前端只要翻译到 IR，后端只要从 IR 翻译，
**两边互不相见**。新增一个前端（比如支持 JAX）只写前端；
新增一个后端（比如支持某款芯片）只写后端。这就是"中间表示"的架构意义。

### 8.2 为什么节点是"名字 → 定义"的映射，而不是树？

我们的 `nodes: dict[str, Node]` 把每个算子**按名字登记**，
这就是"表格（SSA 风格）"而非"嵌套树"。两种设计各有用途：

- **树**（如表达式树）：适合"一个表达式一个结果"的简单计算
- **表 + 名字**（如我们、Relay、SSA）：每个中间结果都有名字，
  **可以引用任意多次**（共享子图），pass 也好定位

真实 IR 里，TensorFlow 用"图+名字"，PyTorch eager 用"执行时建图"，
ONNX 用"图+名字"——都接近我们的设计。

### 8.3 不可变性与哈希一致（最重要的工业级细节）

我们的 `Node` 是可变对象（pass 能改 attrs）。工业级 IR（如 MLIR/Relay）
的节点通常是**不可变**的：改一次就新建一个对象。

**为什么？**
1. **安全**：pass 不会意外改坏别人的节点
2. **哈希一致**：同一个节点 = 同一个对象，可以用地址做 map 的 key
3. **结构共享**：改一个小地方，其它引用自动指向旧版，天然支持"回溯/对比"

MLIR 甚至更进一步：**哈希一致（hash-consing）**——两个结构相同的节点
只存一份。toycc 没做（教学简化），但这是你读工业源码时会撞见的设计。

### 8.4 形状系统：静态形状 vs 符号形状

我们的 `infer_shapes` 只处理具体数字（`(1,3,8,8)`）。真实编译器还有
**符号形状**：`(1, C, H, W)`，其中 `C/H/W` 是符号变量。

为什么需要？**batch size 运行时才定**。符号形状让编译器能写"通用"的图，
代价是很多优化（比如固定分块）做不了。Relax 支持符号形状，
这也是它比老 Relay 强的地方之一。

> **读代码时注意**：看到 `T.IntImm`（具体整数）vs `T.Var`（符号变量），
> 就是这两种形状的分界。你能在 TVM 源码里大量见到。

### 8.5 图分析的信息从哪来？——缓存 vs 现算

`consumers`/`topo_order` 每次现算。工业编译器会**缓存**分析结果
（比如一个 `Analyzer` 对象存好支配关系、形状、常量值），
图变化时用"失效标记"告诉分析器重算。这背后是工程问题：
图有百万节点时，每次全量算拓扑序太慢。toycc 不缓存是因为图太小。

---

## 9. 扩展阅读 B：一个"名字即张量"设计带来的连锁便利

我们约定"张量名 = 产生它的节点名"，这个简单约定带来巨大便利，
值得单独讲：

- **图本身就是"数据流"**：`node.inputs` 里的名字就是"边"
- **`consumers` 一行实现**：查"谁的名字出现在 inputs 里"
- **`rewire` 一行实现**：把所有出现旧名字的地方换成新名字
- **pass 改图极简单**：改名字列表、改 attrs，不需要维护复杂的边表

代价：**重命名要小心**。`rewire` 必须把所有引用一起换，
漏一个就出悬空引用。工业 IR（SSA）用 `Var` 对象而非字符串，
但本质相同：**名字是引用，引用必须一致**。

---

## 10. 思考题（加深版）

1. 如果两个算子结构完全相同（同样的 op 和输入），工业 IR 会怎么处理？
   提示：哈希一致 + CSE（第 12 课）。
2. 为什么 `topo_order` 报错"有环"是好事而不是坏消息？
3. `add_input` 返回的"假节点"有什么用？它和真节点有什么区别？

> 答案：1) 哈希一致让它们可能是同一个对象；CSE 再进一步消除重复计算。
> 2) 它提前发现非法模型，比运行到一半死循环好一万倍。
> 3) 用来当 `producer` 的返回、以及 pass 分析时统一处理"输入也是张量"；
>    它不是真计算，`nodes` 里没有它。

---

**导航**：⬅ [上一节](lesson00.md)（第 0 课 · 总览——一个模型是怎么被编译的）　｜　[下一节](lesson02.md)（第 2 课 · 参考执行器）➡
