# 第 2 课：参考执行器——编译器正确性的"裁判"

> 本课风格：代码驱动 + 一次完整的手算执行轨迹。
> 对应文件：`toycc/runtime/ref.py`
> 准备：跑 `python -m course.runner 2` 对照看。

---

## 1. 一个全课最重要的思想实验

设想你在写一个优化 pass：把 `conv(x) + relu` 融合成一个算子。

你改完了图，自我感觉良好。但你凭什么说它**算对了**？

你无法"看一眼"就知道——图里有几千个算子，人眼不可能逐个验证。
你需要一个**客观的裁判**：

> 一份"绝对正确、但是故意写得很朴素"的执行器。
> 任何优化后的图，都拿它算一遍，和优化前的参考结果逐元素对比。
> 一样 → 优化没问题；不一样 → 优化坏了，回去修。

这份执行器，就叫**参考执行器（reference evaluator）**。

它"正确"的秘诀就一条：**直接用数学定义写，不做任何优化。**

```python
def relu(inputs, attrs):
    return np.maximum(inputs[0], 0.0)      # 这就是 relu 的定义,不可能更正确了
```

> 类比：修车师傅修车前先量一遍"出厂参数"，修完再量一遍对比。
> 参考执行器就是那张"出厂参数表"。

---

## 2. 逐行读 `toycc/runtime/ref.py`

### 2.1 每个算子一个实现：直接用定义写

```python
class RefImpl:
    @staticmethod
    def conv(inputs, attrs: OpAttrs):
        x, w = inputs[:2]                       # 数据 + 权重
        bias = inputs[2] if len(inputs) > 2 else None   # 融合的 bias
        N, C, H, W = x.shape                    # 解出数据形状
        OC, _, KH, KW = w.shape                 # 解出权重形状
        SH, SW = attrs.stride
        PH, PB, PL, PR = attrs.pad
        xp = np.pad(x, ((0,0),(0,0),(PH,PB),(PL,PR)))   # 手动补 padding
        OH = (H + PH + PB - KH) // SH + 1       # 输出高
        OW = (W + PL + PR - KW) // SW + 1       # 输出宽
        out = np.zeros((N, OC, OH, OW))
        for n in range(N):                       # 逐个输出位置手算
            for oc in range(OC):
                for oh in range(OH):
                    for ow in range(OW):
                        h0, w0 = oh * SH, ow * SW
                        out[n, oc, oh, ow] = np.sum(
                            xp[n, :, h0:h0+KH, w0:w0+KW] * w[oc])
        if bias is not None:
            out += bias
        return out
```

**这段代码唯一的优点就是"忠实于定义"**：卷积 = 输出每个位置 = 输入对应
窗口和权重逐元素相乘求和。慢？当然慢——6 层循环。但**不需要快**，
它只在编译期被用来验证，不参与真正的推理。

**为什么先看这段？** 因为你马上会发现：后面所有 pass 的正确性，
都是拿这个"最笨的实现"当准绳的。笨，正是它值得信任的原因。

其他算子一样直白：

```python
@staticmethod
def relu(inputs, attrs):
    return np.maximum(inputs[0], 0.0)

@staticmethod
def matmul(inputs, attrs):
    a, b = inputs[:2]
    out = np.matmul(a, b)
    if len(inputs) > 2:                          # 融合了 bias
        out = out + inputs[2]
    return out
```

### 2.2 激活的后处理

```python
_ACTIVATIONS = {"relu": RefImpl.relu, "sigmoid": RefImpl.sigmoid}
```

融合进来的激活（第 3 课会产生 `conv(x, w, bias)` + `activation="relu"`），
执行时在算子输出上**再套一层**：

```python
if node.attrs.activation:
    out = _ACTIVATIONS[node.attrs.activation]([out], OpAttrs())
```

意思就是：`conv+relu` 这个融合算子执行时 = 先做 conv，再做 relu。
**融合只是把两个算子合并成一个节点，数学上还是两次运算的叠加。**

### 2.3 执行器主体：按拓扑序"跑图"

```python
class ReferenceEvaluator:
    def __init__(self, graph: Graph):
        self.graph = graph

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        vals: dict[str, np.ndarray] = dict(self.graph.constants)   # 常量先就位
        vals.update(feed)                                          # 输入覆盖进来
        for node in self.graph.topo_order():                       # 按拓扑序跑
            ins = [vals[i] for i in node.inputs]                   # 取输入值
            fn = getattr(RefImpl, node.op_type)                    # 找实现
            out = fn(ins, node.attrs)                              # 执行
            if node.attrs.activation:
                out = _ACTIVATIONS[node.attrs.activation]([out], OpAttrs())
            vals[node.name] = out                                  # 缓存结果
        return {o: vals[o] for o in self.graph.outputs}
```

**逐行**：

| 行 | 在干嘛 |
|---|---|
| `vals = dict(constants)` | 先放常量值（权重等，编译期已知） |
| `vals.update(feed)` | 再把运行时输入放进来 |
| `for node in topo_order()` | **按拓扑序**执行——保证依赖已算好 |
| `ins = [vals[i] for i in node.inputs]` | 收集本算子的输入值 |
| `getattr(RefImpl, node.op_type)` | 按算子类型找实现函数 |
| `fn(ins, node.attrs)` | 执行 |
| `vals[node.name] = out` | 结果存进"值字典"，后面的算子能取到 |

最后 `return {o: vals[o] for o in self.graph.outputs}` 只把**图输出**取出来。

**这就是"解释执行"**：走到哪算到哪，结果放一张大字典里。
你会在后面发现：常量折叠（第 5 课）就是把这个执行器**搬到编译期**跑了一遍。

> **原理深挖：为什么参考执行器要写得"又笨又慢"？**
>
> 注意这个执行器**完全不优化**：不融合、不规划内存、一个算子一个算子算，
> 中间结果全堆在 `vals` 字典里。在真实硬件上这么跑会慢得离谱。但它是故意的。
>
> 原因是**它要当"裁判"**。裁判的职责是"给出无可争议的正确答案"，
> 不是跑得快。如果参考实现也搞优化，那"优化错了"和"参考错了"就分不清了——
> 你拿一个"可能也有 bug 的东西"去验证另一个，等于没验证。
>
> 这是编译器开发的第一原则：**参考实现要"笨到人眼能看出它对"**。
> 它越简单、越接近数学定义，它当裁判就越可信。后面每个 pass 的验证，
> 都是"优化版 vs 这个笨版本"的对比。笨，是它的优点，不是缺点。

---

## 3. 一次完整的手算执行轨迹

用最小的图演示（就两个算子）：

```
graph mini:
  input: a          # a = [1, 2, 3]
  input: b          # b = [0.5]
  add1 = add(a, b)  # a + b
  r1   = relu(add1)
  output: r1
```

`run({a:[1,2,3], b:[0.5]})` 的执行过程：

```
步骤0  vals = {a:[1,2,3], b:[0.5]}          ← constants + feed
步骤1  拓扑序 = [add1, r1]
步骤2  处理 add1:
        ins = [vals[a], vals[b]] = [[1,2,3], [0.5]]
        out = [1.5, 2.5, 3.5]               ← numpy 广播
        vals[add1] = [1.5, 2.5, 3.5]
步骤3  处理 r1:
        ins = [vals[add1]] = [[1.5, 2.5, 3.5]]
        out = max(0, ...) = [1.5, 2.5, 3.5]
        vals[r1] = [1.5, 2.5, 3.5]
步骤4  返回 {r1: [1.5, 2.5, 3.5]}
```

注意 `add1` 已经"消费完"了 `a` 和 `b`，但它们的值还在字典里——
**参考执行器不管内存**（那是第 6 课的事），它只求算对。

---

## 4. "对错"怎么量化：max|Δ|

跑优化前后的图，各得到一个输出。怎么比较？

```python
diff = np.max(np.abs(out_ref - out_optimized))
# diff 是个数:
#   接近 0 (我们要求 < 1e-9)  → 一致
#   明显大于 0                → pass 改坏了,回去修
```

**为什么用 `max`（最大值）而不是 `mean`（平均值）？**

假设输出有 1000 个数，其中 999 个正确，1 个错得离谱（比如 1000 倍）：

```python
mean: (999*0 + 1*1000) / 1000 = 1.0    ← 看起来"还可以"?
max : 1000                             ← 瞬间暴露问题
```

`mean` 会把错误"平均掉"；`max` 抓住任何一个位置的错误。
所以编译器验证都用 `max|Δ|`（或等效的 `allclose`）。

**为什么允许 1e-9 而不是要求严格为 0？** 浮点运算有舍入误差：
`0.1 + 0.2 != 0.3`（在二进制浮点里），所以任何计算顺序改变都会引入
微小差别。我们的实验里，优化前后差异是 `5.55e-17`，这就是纯舍入噪声。

> **手算：TVM 的 rtol/atol 到底在比什么**
>
> `assert_allclose(ref, out, rtol=1e-4, atol=1e-4)` 的判据逐元素是：
> ```
> |ref - out| ≤ atol + rtol × |ref|
> ```
> 拆开看两个项的含义：
>
> - `atol`（绝对容差）：**接近 0 的数的判据**。`ref=1e-8` 时，
>   允许的绝对误差就是 `1e-4`——因为数据本身就是个小数字，
>   相对误差在它身上没意义（`1e-4 × 1e-8 = 1e-12` 太小，舍入就会超）。
> - `rtol`（相对容差）：**大数字的判据**。`ref=1000` 时，允许误差
>   `1e-4 × 1000 = 0.1`——1% 万分之一的相对差。对数值大的激活值，
>   绝对容差太苛刻，相对容差才反映"精度比例"。
>
> 手算两个极端：
> ```
> ref = 1e-8  → 容差 = 1e-4 + 1e-4×1e-8 ≈ 1e-4   ← atol 主导
> ref = 1000  → 容差 = 1e-4 + 1e-4×1000 = 0.1001  ← rtol 主导
> ```
>
> **为什么比纯 `max|Δ|` 更精细**：`max|Δ| < 1e-9` 对"数值巨大的层"
> 太严、对"接近 0 的层"可能太松；rtol/atol 分场景给判据。
> 我们的 toycc 用 `max|Δ|` 是为了教学直观——**阈值本身怎么定，
> 是数值验证里最需要工程经验的部分**（第 18 课量化时会再次碰到）。

---

## 5. 实验

```bash
python -m course.runner 2
```

看三行输出：

```
output: shape=(1, 16) mean=0.0475
固定随机种子 => 两次执行结果一致: True
```

- 第一行：参考执行器真的把 10 个算子跑完了，输出一个 `(1,16)` 的张量
- 第三行：**确定性**——同一份输入，跑两次结果完全一样。
  这看起来理所当然，但它是"能拿它当裁判"的前提：
  如果执行器本身随机，你都没法说清差异是谁造成的。

---

## 6. 真实 TVM 对照

TVM 没有单独一个叫"reference evaluator"的东西，但思路无处不在：

- **Python 前端**能直接解释执行高层算子——本质上就是我们的 `evaluate`
- 编译完，官方教程的标准动作：
  ```python
  tvm.testing.assert_allclose(ref_output, compiled_output, rtol=1e-4, atol=1e-4)
  ```
  这就是把"优化结果 vs 参考结果"逐元素对比，只不过用的是
  **相对误差+绝对误差**的组合，比我们的 `max|Δ|` 更精细。
- 很多 pass 自带**数值测试**：构造随机输入 → 跑 pass 前算参考 →
  跑 pass 后对比。你在 TVM 仓库的 `tests/python/relax/test_transform_*` 里
  会看到海量这种测试。

**另一个视角**：参考执行器不止用来验证，它还是**常量折叠**的引擎
（第 5 课）。一个执行器，两种用途——"验证"和"编译期求值"。

---

## 7. FAQ

**Q：参考执行器那么慢，为什么不直接拿它跑推理？**
A：能，但它不为性能而生。推理跑的是优化后的 C/CUDA 代码。
参考执行器只出现在：编译期验证、写 pass 时的快速试错。

**Q：如果参考执行器自己写错了呢？**
A：那就是"所有优化都'正确'地继承了错误"——最可怕的 bug。
所以参考实现必须**用定义本身**写（比如卷积就用嵌套循环），
越直白越难写错。这也是为什么我们有 `RefImpl` 和优化实现完全分离。

**Q：`5.55e-17` 是什么概念？**
A：大约等于 0.0000000000000000555。比"1 后面 16 个 0 分之一"还小。
如果输出是 0.0475 这个量级，这个差异就是"小数点后 16 位"——纯舍入噪声。

**Q：为什么 `feed` 里只给 `x`，权重哪来的？**
A：`run` 里第一行就把 `graph.constants` 并进去了。权重存在图的常量表里
（`build_model_with_weights` 里 `set_constant` 过），不需要每次传。

---

## 8. 本课小结

- 参考执行器 = **最朴素、只用定义写的正确实现** = 所有优化的裁判
- 执行方式：常量+输入进字典 → 按拓扑序逐个算子跑 → 缓存结果
- 判对错：`max|Δ| < 1e-9`（浮点舍入级别的误差可接受）
- 确定性是能当裁判的前提
- 它还是第 5 课"常量折叠"的引擎

**下一步**：第 3 课，我们的第一个真正优化——**算子融合**。
还记得第 0 课那张总图吗？10 个算子要并成 4 个。而且——这次我要
**故意写一个会算错的融合**，让你亲眼看到参考执行器是怎么把它抓出来的。

---

## 9. 深层拓展 A：浮点比较的科学（rtol / atol / NaN / 逐 bit）

我们用 `max|Δ| < 1e-9` 判对错。但工业验证要更精细，这里讲透。

### 9.1 绝对误差 vs 相对误差

**问题**：`1e-9` 的阈值对数值大的张量太严，对数值小的张量太松。

- 输出值是 `1e+12` 时，误差 `1e-3` 都算很小（相对 1e-9 的比例）
- 输出值是 `1e-12` 时，误差 `1e-15` 都算很大

所以工业验证用**两者结合**：

```python
np.allclose(ref, got, rtol=1e-4, atol=1e-5)
# 判定: |ref - got| <= atol + rtol * |got|
```

- `atol`：绝对误差兜底（防止 `ref=0` 时相对误差无穷大）
- `rtol`：相对误差主判据

这就是 TVM `tvm.testing.assert_allclose` 用的方式。

### 9.2 为什么结果顺序不同，浮点就会不同？

浮点运算是**不可结合**的：`(a + b) + c ≠ a + (b + c)`（在二进制里）。
因为每一步都四舍五入，舍入误差会累积。

```
a=1e20, b=-1e20, c=1
(a+b)+c = 0 + 1 = 1
a+(b+c) = 1e20 + (-1e20) = 0   ← 结果不同!
```

所以"融合/重排循环/换顺序"哪怕数学等价，浮点结果也会**略有差异**。
这不是 bug，是浮点的天性。这就是为什么我们允许 `1e-9` 而不是要求严格为 0。

### 9.3 特殊值：NaN 和 Inf 的陷阱

`0/0 = NaN`，`1/0 = Inf`。比较时要特别小心：

- `NaN != NaN`（NaN 和任何数都不相等，包括它自己！）
- `abs(NaN - NaN)` 也是 `NaN`，`NaN < 1e-9` 是 **False**

**陷阱**：如果输出里有 NaN，`max|Δ|` 会变成 NaN，`NaN < 1e-9` 是 False，
验证会"失败"。这其实是**好事**——NaN 说明哪里出问题了。
但你要知道 `np.isnan` 单独检查，别被 `max|Δ|` 骗了。

### 9.4 参考实现写成"最笨"的真正理由

你可能觉得"参考实现能不能写得聪明点、快一点"？**不能。**

参考实现的唯一品质要求是**正确**。聪明的实现会引入：
- 更多优化 → 更多出错机会
- 非显然的实现 → 难 audit
- 用更快的库（如 FFT 卷积）→ 数值行为和定义不同

所以参考实现 = 用定义本身写 = 一个"能让人一眼看出对"的实现。
**正确性优于一切，包括速度。**

---

## 10. 深层拓展 B：工业界到底怎么验证一个 pass？

TVM / LLVM / GCC 的验证体系，比"跑一次对比"复杂得多，但核心一致：

| 层级 | 干什么 | 例子 |
|---|---|---|
| 单元测试 | 每个 pass 配测试 | `tests/python/relax/test_transform_fusion.py` |
| 随机测试 | 随机输入反复跑 | 随机形状、随机数据，跑几十次 |
| 回归测试 | 防旧 bug 复发 | 每个修过的 bug 留一个测试 |
| 数值测试 | 参考 vs 优化对比 | `assert_allclose` |
| CI 自动化 | 每次提交全跑 | GitHub Actions / 本地跑 `tests` |
| 形式化验证（个别） | 数学证明 pass 正确 | 学术界有，工业界少 |

**你作为初学者最该养成的习惯**：改一个 pass → **先写测试 → 再跑全量测试**。
测试写得好，等于给你的 pass 上保险。第 10 课的任务会带你做这个。

---

## 11. 一个经典的"验证抓 bug"案例（真实发生）

假设有人写了一个融合 pass，把 `conv+bn+relu` 融成一个核。
融合规则里把 BN 的均值和方差当常量吸收进卷积。跑数值测试：

```
reference:  conv → bn → relu   的输出
optimized:  融合核             的输出
```

第一次跑：`max|Δ| = 3.2e-1` —— 不对。查原因：
BN 在**推理**和**训练**两个模式下的公式不同
（训练用 batch 均值方差，推理用缓存好的均值和方差）。
融合 pass 只实现了推理版本，但测试模型还在训练模式——**BN 的语义没对上**。

修复：在融合前强制模型转推理模式，或让融合 pass 检查 BN 的模式属性。
**结论**：验证不仅抓"数学错误"，还能抓"语义上下文没处理好"。
这就是参考执行器的价值——它是唯一客观的裁判。

---

**导航**：⬅ [上一节](lesson01.md)（第 1 课 · 计算图与 IR）　｜　[下一节](lesson03.md)（第 3 课 · 算子融合）➡
