# 第 4 课：布局优化——同一个张量，换个内存排法就更快

> 本课风格：代码驱动 + 用"地址表"把排布讲透 + 手算搬移代价。
> 对应文件：`toycc/passes/layout.py`
> 准备：跑 `python -m course.runner 4` 对照看。

---

## 1. 一个必须亲眼看见的事实：张量在内存里是"一维数组"

不管张量是 2 维、3 维还是 4 维，**它在内存里都是一个连续的一维数组**。
多维的"形状"只是我们给这块一维内存取的"切片方式"。

一个 `(2, 3)` 的矩阵，内存里就是 6 个 float 排成一排：

```
内存地址:  0    1    2    3    4    5
数值:    [1.0][2.0][3.0][4.0][5.0][6.0]
```

`matrix[1][0]` 对应地址 3。怎么算的？**行主序（row-major）**：

```
matrix[i][j] 的地址 = i * 3 + j
```

**"布局"（layout）就是：多维下标 → 一维地址 的换算规则。**

---

## 2. 同一个张量，两种排法，两种结果

一个 `(N=1, C=2, H=2, W=2)` 的张量，8 个元素。float32 下每个元素占 4 字节，
两种排法在内存里的**真实字节布局**完全不同——同一个元素跑到不同的地址去了。

**NCHW**（把通道放前面：先铺完通道 0 的整张图，再铺通道 1）：

```
地址:   0        4        8       12       16       20       ...
元素: [c0:h0w0][c0:h0w1][c0:h1w0][c0:h1w1][c1:h0w0][c1:h0w1]...
        ← 通道0的4个像素挨着 →         ↑ 然后整块跳到通道1
```

**NHWC**（把通道放最后：每个像素的 2 个通道挨在一起）：

```
地址:   0        4        8       12       16       20       ...
元素: [h0w0:c0][h0w0:c1][h0w1:c0][h0w1:c1][h1w0:c0][h1w0:c1]...
        ↑ 同一像素的通道紧挨, 才几字节
```

**两种排法逻辑上是同一个张量，但内存里的数值顺序完全不同。**

**你只需要看一件事就懂了两者差别**：卷积要算输出像素 `(h0w0)` 时，
必须同时读"通道 0 的 h0w0"和"通道 1 的 h0w0"两个值：

```
NCHW: 一个在地址 0, 一个在地址 16 → 隔了整张通道0的图 → 跨通道必跳
NHWC: 一个在地址 0, 一个在地址 4  → 紧挨着 → 一次取到两个通道
```

> 记住这句话：**"布局"不是抽象概念，它决定"同一个元素放在内存哪个地址"，
> 决定"卷积要的那几个值是不是挨在一起"。**

---

## 3. 为什么排法影响速度？（两个机制）

### 机制一：缓存局部性

CPU/GPU 读内存是**按块（cache line）**读的，一次读一整块（比如 64 字节）到缓存。

- **NCHW 卷积**：算输出 `(n, oc, oh, ow)` 时，要读输入 `(n, c, 窗口)` 的
  每个通道。通道在 NCHW 里是连续的吗？是**每个通道内**连续，但跨通道跳。
  每次"跳到下一个通道"都可能缓存未命中 → 反复从慢速内存读。
- **NHWC 卷积**：同一窗口内，所有通道的 8 个像素挨在一起——**一次 cache line
  就把一个窗口需要的通道全带来了**。连续、顺路。

### 机制二：SIMD 向量化

CPU 的 SIMD 指令一次能对 8 个 float 做运算。NHWC 里"同一像素的 4 个通道"
正好排在一起，一条 SIMD 指令就能把 4 个通道一起算了。NCHW 则散着，不好向量化。

> **一句话直觉**：卷积内层循环是对"通道"求和。数据排法决定了这个内层循环
> 能不能"顺路"读、能不能一条指令算多个。这就是布局影响性能的本质。

**为什么 GPU 上反而常用 NCHW？** 因为 cuDNN 等库为 NCHW 做了深度手工优化
（专门的 kernel、更好的并行分配），库的优化 > 排法本身的收益。所以：
**布局好坏，永远要绑定"你要跑的算子"来看**。没有绝对的优劣。

---

## 4. 布局优化的问题定义

如果所有算子都"随便排"，编译器就没法保证边界处数据排法对得上。
所以问题变成：

```
输入 x 是 NCHW (外部给的,改不了)
但 conv 想要 NHWC 才能快
```

两条路：

| 方案 | 代价 | 收益 |
|---|---|---|
| 把 x 转成 NHWC 再算 | 一次搬移（把 192 个元素按新顺序重排） | conv 快 |
| 保持 NCHW 直接算 | 0 搬移 | conv 慢 |

**关键洞察**：中间一堆逐元素算子（relu/add）**不挑布局**——它对每个元素
单独操作，排成 NCHW 还是 NHWC 结果一样。于是布局可以**沿图传播**：

```
conv 输出 NHWC → relu 继承 NHWC → add 继承 NHWC → 直到下一个"挑布局"的算子
```

**优化目标 = 只在"布局不匹配的边界"插搬移，其余靠传播免费继承。**

这就是 `layout.py` 全部要做的事。

> **原理深挖：为什么"继承"是免费的，而"搬移"是花钱的？**
>
> 继承只是**改表**：`layout[relu1] = layout[conv1]`——一行代码，零成本。
> 它的意思是"relu1 的输出和输入用同一个内存排法"，物理上什么都没动。
> 搬移则是**真的重新排数据**：把 256 个元素按新顺序写一遍，耗时耗内存。
>
> 所以布局优化的全部学问在于：**让"免费"的继承覆盖尽量多，让"付费"的
> 搬移只在边界发生一次**。中间一串逐元素算子全部继承同一排法，
> 直到下一个"挑布局"的算子才需要搬。这就是"沿图传播 + 只在边界转换"。
>
> **真实 TVM 里**：布局传播后插的是 `layout_transform` 算子。它的成本
> 就是"元素被重排写一遍"。业界甚至有专门优化 `layout_transform` 本身的 pass
> （因为模型一深，边界搬移反而可能成为新瓶颈——这是布局优化的"反噬"）。
> 你以后写布局 pass 时，记得**数一数到底插了几次搬移**——最少才是最好。

---

## 5. 逐行读 `toycc/passes/layout.py`

### 5.1 关键数据结构：一张"布局表"

```python
layout: dict[str, str] = {i: "nchw" for i in g.inputs}
```

`layout[tensor名] = "nchw"/"nhwc"`，记录"这个张量当前是什么排法"。
从输入开始：输入默认 NCHW。

### 5.2 规则表：每个算子"怎么处理布局"

```python
for node in g.topo_order():
    if node.op_type == "conv":
        if node.name in self.nhwc_conv:            # 目标: 这些 conv 走 NHWC
            for i in range(len(node.inputs)):      # 数据/权重/bias 全转 NHWC
                self._ensure_layout(g, layout, node, i, want="nhwc")
            node.attrs.layout = "nhwc"
            layout[node.name] = "nhwc"
        else:
            layout[node.name] = "nchw"

    elif node.op_type in _LAYOUT_AGNOSTIC:          # relu/add/mul/...
        layout[node.name] = layout.get(node.inputs[0], "nchw")   # 继承!

    elif node.op_type == "reshape":
        self._ensure_layout(g, layout, node, 0, want="nchw")
        layout[node.name] = "nchw"
```

**三种情况，三种策略**：

| 算子类型 | 策略 | 为什么 |
|---|---|---|
| conv（目标算子） | 全部输入转成 NHWC，输出标 NHWC | 它要快 |
| 逐元素（relu/add） | **继承**输入的布局 | 它不挑，白送 |
| reshape | 必须 NCHW | 见下面"为什么 reshape 挑布局" |

### 5.3 `_ensure_layout`：插搬移的核心

```python
@staticmethod
def _ensure_layout(g, layout, node, input_idx, want):
    src = node.inputs[input_idx]
    cur = layout.get(src)
    if cur is None or cur == want:      # 布局已经对了 → 什么都不用做
        return
    name = f"{node.name}_lt_{input_idx}"
    lt = g.add_op("layout_transform", [src], name=name,
                  attrs=OpAttrs(from_layout=cur, to_layout=want))
    node.inputs[input_idx] = name       # 让算子改吃"转换后的张量"
    layout[name] = want
```

**逐行**：
1. `cur = layout.get(src)`：看看这个输入现在是什么排法
2. 已经对上了 → 直接返回，**零开销**
3. 没对上 → 插入一个新节点 `layout_transform`（一个"搬数据"的算子），
   属性记录"从什么排法转成什么排法"
4. 让原来的算子**改吃转换后的张量**
5. 更新布局表：新节点输出是 `want` 排法

**这个 `layout_transform` 就是真实 TVM 里的 `permute_dims` / `layout_transform`**——
一个专门干"重新排列内存"的算子。它本身要花一次搬移，所以编译器要精打细算
少插它。

### 5.4 为什么 reshape 必须"挑布局"？

`reshape` 按**内存顺序**拍平。同样的数据：

```
NCHW 拍平:  n c h w  →  通道 c 最快变化
NHWC 拍平:  n h w c  →  像素 w 最快变化
```

两种拍平得到的**一维顺序不一样**，那 `reshape((N, C*H*W))` 的结果就不同。
所以 reshape 必须在"标准排法"下做，否则结果会乱。于是布局在 reshape 处
必须"对齐回 NCHW"。

---

## 6. 手算一次"插搬移"

假设 `x` 是 NCHW `(1, 3, 8, 8)`，conv1 要 NHWC：

```
节点: conv1 = conv(x, conv1_w, bias1)   ← 布局 pass 处理 conv1
检查 x:  layout[x] = "nchw", 想要 "nhwc"  → 不匹配!
插入:  conv1_lt_0 = layout_transform(x, nchw→nhwc)
改写:  conv1 的输入从 x 变成 conv1_lt_0
更新:  layout[conv1_lt_0] = "nhwc"
```

**layout_transform 到底做了什么运算？** 看参考执行器：

```python
def layout_transform(inputs, attrs):
    x = inputs[0]
    if attrs.from_layout == "nchw" and attrs.to_layout == "nhwc":
        return np.transpose(x, (0, 2, 3, 1))    # 轴重排: 内存顺序跟着变
    ...
```

`np.transpose(x, (0,2,3,1))` 把维度 `(n,c,h,w)` 重排成 `(n,h,w,c)`。
**逻辑张量没变，但内存里的顺序变了**——这就是"搬移"。

---

## 7. 布局传播的收益：数一数插了几个搬移

跑 `python -m course.runner 4`，数 `layout_transform`：

- conv1：数据、权重、偏置 → 3 个
- conv2：权重、偏置 → 2 个
- reshape 前：1 个
- 一共 **6 个**

但注意：其中 4 个是**转常量**（权重、偏置）。常量转换在编译期做一次，
运行时根本不付钱（第 5 课常量折叠会把它删掉）。真正运行时只有 2 次数据搬移：

```
x(NCHW) →[conv1_lt_0]→ conv1 → conv2 →[flat_lt_0]→ reshape → matmul
```

**如果不会传播**，你可能在每个算子前后都插（10+ 次）。
正确传播后运行时只付 2 次。这就是这个 pass 的全部价值。

---

## 8. 真实 TVM 对照：`convert_layout.cc`

`src/relax/transform/convert_layout.cc`（373 行）。头注释：

```cpp
/*
  \brief 把 conv2d 的布局转换。其它算子会跟着 conv 的转换自适应。
  每个 op 都注册了布局推导函数 FRelaxInferLayout:
  输入是(当前 call, 想要的布局, 之前变量的布局表),
  输出是(输入布局, 输出布局, 可能被改写的 attrs)。
  注意: 目前只支持轴交换(NCHW↔NHWC), 不支持打包布局(NCHW→NCHW4c)。
*/
```

注意三个词：

- **"其它算子跟着自适应"** = 我们的"布局无关继承"
- **`FRelaxInferLayout` 注册表** = 每个算子声明"我吃啥布局、吐啥布局"。
  我们的 `_LAYOUT_AGNOSTIC` 集合 + 规则表就是它的手写版。
  TVM 的好处：新算子注册一下就行，不用改 pass 代码。
- **"只支持轴交换"** = 和我们的 `np.transpose` 一个层次。
  打包布局（NCHW→NCHW4c，把通道拆成 4 个一组配合 SIMD）是进阶话题。

插搬移的 `RewriteExpr`：

```cpp
Expr RewriteExpr(const Expr& expr, const NLayout& to) {
  ...
  if (NLayoutEqual()(from, to) || ...) return expr;   // 布局一样 → 跳过(零开销)
  if (from...ndim() == to...ndim()) {
    SLayout axes = TransposeLike(...);                // 轴交换 → permute_dims
    return permute_dims(expr, LayoutToIntegers(axes));
  } else {
    // 维数不同 → layout_transform + index_map
    return Call(..., layout_transform_op_, {expr}, attrs, {});
  }
}
```

和我们的 `_ensure_layout` 完全同构：**布局相同跳过；轴交换用 permute_dims
（= 我们的 transpose）；维数变化用 layout_transform。**

---

## 9. 实验

```bash
python -m course.runner 4
```

对照第 5 节数一数插的搬移，再对照输出里 `layout_transform` 节点。

---

## 10. 课后答疑

**Q：为什么权重和 bias 也要转布局？**
A：conv 的 NHWC 卷积要求"通道在最后"。权重 `(OC,C,KH,KW)` 也得按 NHWC
重排成 `(OC,KH,KW,C)`，bias 从 `(1,OC,1,1)` 重排成 `(1,1,1,OC)`，
否则广播对不上、点积对不上。第 5 课你会看到：这些常量转换被折叠掉，
运行时 0 成本。

**Q：布局转换本身要搬数据，什么时候"转"是赚的？**
A：当"转一次的代价" < "用错布局跑 N 层算子的额外代价"时。
链越长、中间逐元素算子越多，越值得转（因为转一次，后面全免费）。
真实编译器用 cost model 算这个权衡，我们 toycc 用"拍脑袋决定哪些 conv 走 NHWC"。

**Q：为什么输入 x 不能也提前转好？**
A：x 是**运行时**才来的数据（每张图片都不一样），没法在编译期转。
所以 `conv1_lt_0`（转 x）是运行时真实的 2 次搬移之一。

**Q：会不会出现"转来转去"死循环？**
A：不会。因为 layout 表是单向推进的（拓扑序），每次 `_ensure_layout` 要么
跳过、要么把输入换成 transform 节点，图只会前进。

---

## 11. 本课小结

- 布局 = 多维下标 → 一维地址 的换算；内存里只有一维数组
- 布局影响性能的两机制：**缓存局部性**、**SIMD 向量化**
- 好坏必须绑定算子看：CPU 常选 NHWC，GPU(cuDNN) 常选 NCHW
- 逐元素算子"不挑布局" → 布局可**沿图传播**
- 优化 = 只在边界插 `layout_transform`，运行时只付 2 次数据搬移
- 权重/bias 的转换交给常量折叠（下一课）在编译期解决

**下一步**：第 5 课——常量折叠。刚才那些"给权重转布局"的操作，
凭什么能在编译期做完、运行时一分钱不花？

---

## 12. 扩展阅读 A：strides——布局的数学表达

"布局"除了用 NCHW/NHWC 这种名字，还能用更精确的数学表示：**strides**。

一个 `(2,3)` 矩阵的 strides 是 `(3, 1)`，意思是：
- 第 0 维每走 1，内存跳 3
- 第 1 维每走 1，内存跳 1

**布局 = 每个维度的 stride**。NCHW vs NHWC 就是 strides 的排列：

```
NCHW (1,3,8,8): strides = (192, 64, 8, 1)     # 通道最连续
NHWC (1,8,8,3): strides = (192, 24, 3, 1)     # 像素最连续
```

**这个视角带来三个理解**：

1. **布局转换 = 换 strides**。`transpose` 不改数据逻辑，改的是
   "每个维度的步长"（物理上要搬内存）。
2. **"逻辑形状"和"物理形状"可以分离**：`(1,3,8,8)` 逻辑上一直是
   这个形状；物理排布（strides）可以不同。
3. **strides 还能表达"视图"**：比如切片 `x[:, :, 1:5, :]` 逻辑形状变了，
   但可以共享底层内存、只改 strides——这是 numpy 里 reshape/transpose
   "零拷贝"的原理。

> 读 TVM 源码时你会看到 `strides` / `byte_offset` / `layout_decision`，
> 它们都是这个概念的工业实现。

---

## 13. 扩展阅读 B：打包布局（NCHW4c）——比轴交换更深入的排法

NCHW 和 NHWC 只是"轴交换"（轴重排）。工业界还有更强的：**打包布局**。

以 `NCHW4c` 为例，把通道拆成"4 个一组"：

```
逻辑: (N, C, H, W)
物理: (N, C/4, H, W, 4)     # 4 个通道一组, 组内连续

内存:  [c0 c1 c2 c3] [c4 c5 c6 c7] [c8 ...] ...
        组0(4个连续)  组1(4个连续)
```

**为什么这么排？**
- 一次 SIMD 指令正好处理 4 个 float（SSE=4、AVX=8、AVX512=16）
- 卷积内层对通道求和时，4 个通道一次向量化算完
- 相比 NHWC 的"所有通道连续"，打包布局**按向量宽度对齐**，配合硬件最顺

**常见变体**：`NCHW4c`、`NCHW8c`、`NCHW32c`（GPU 上）、`NHWC1c`。

**关键点**：打包布局改变了"物理维度数量"（4 维 → 5 维），
所以需要 `layout_transform` 而不是简单的 `permute_dims`。
还记得第 4 课 TVM `RewriteExpr` 里那个 `else` 分支吗？——"维数不同
用 layout_transform"。说的就是打包布局。

---

## 14. 扩展阅读 C：布局决策在编译器里怎么做？

toycc 是"拍脑袋"指定 `nhwc_conv = ("conv1", "conv2")`。真实编译器呢？

**TVM 的做法**：每个算子注册"布局偏好"和"布局推导函数"
（`FRelaxInferLayout`），布局 pass 遍历时：
1. 看每个算子声明它"吃哪个布局"（如 conv 偏好 NHWC）
2. 沿图传播，布局无关算子继承
3. 边界处评估"转换代价 vs 收益"，决定插不插 layout_transform

**收益评估很关键**：不是所有 conv 都该转 NHWC。
- 转换要搬一次数据（有成本）
- 但如果后面还有 10 个算子都能受益，一次转换赚了 10 次
- 成本模型（第 13 课）甚至能实测比较

**工程结论**：布局决策 = "插入转换的代价 vs 全局收益"的权衡。
toycc 用手写名单（教学简化），TVM 用注册表 + 评估。
理解这个差距，你就知道为什么真实编译器有那么多"启发式"。

---

**导航**：⬅ [上一节](lesson03.md)（第 3 课 · 算子融合）　｜　[下一节](lesson05.md)（第 5 课 · 常量折叠）➡
