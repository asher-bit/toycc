# 第 10 课：从"看懂"到"上手"——中途检查点 + 手感练习

> 课程刚走完一半（前 9 课 = 骨架 + 真实框架对照）。从现在起，你不再只是
> "读"课程，而是**在 toycc 上亲手改代码**，把"看懂的"变成"干过的"。
> 这一课就是你的"手感训练营"：做完本课任务，你才有底气继续啃后面的
> 调度（第 11 课）、硬件（第 15 课），并最终走进 GPU 专项（第 21~24 课）。

---

## 1. 先盘点：前 9 课你拥有了什么，还缺什么

| 能力 | 证据 |
|---|---|
| 看得懂 IR | 能读 `Graph`/`Node`，能理解 `relax.call_tir` 的文本表示 |
| 看得懂 pass | 融合/布局/折叠/内存，每个都能讲出"输入→输出→为什么" |
| 知道正确性怎么保证 | `max|Δ|`、参考执行器、`assert_allclose` |
| 看得懂 TVM 源码 | 会三遍读法，能定位 `fuse_ops.cc` 里任何类的职责 |
| 会跑真框架 | 装 tvm、`Sequential([...])`、`relax.build` |

**这是"能进编译器会议室的人"的最低配置。** 但你还缺**手感**——亲手改过代码、
跑过测试、碰过 bug。后半程的课（调度、量化、GPU 专项）都会默认你有这点手感，
所以本课的任务不是"可选项"，是**继续往下走的前置条件**。

---

## 2. 任务 A：给 toycc 加一个算子 `maxpool`（必做）

这是最重要的热身，也是"真实编译器新人入职第一周"的标准动作。
一个算子要"落地"，必须同时打通四条线——**这四条线本身就是编译器的新人课**：

```
注册表(IR) → 形状推导 → 参考实现(裁判) → 建图验证
```

### 2.1 注册表：`toycc/ir/ops.py`

```python
register("maxpool", 1, _maxpool_shape, None)   # (名字, 输入数, 形状推导, 属性推导)
```

对照 `conv` 的登记写法：`maxpool` 输入数 = 1（一个张量）；
`_maxpool_shape` 要新写（见下）；属性推导先给 `None`。

### 2.2 形状推导：`_maxpool_shape`

maxpool 输出形状：（忽略 padding 时）`(N, C, H/kh, W/kw)`。

```python
def _maxpool_shape(in_shapes, attrs):
    N, C, H, W = in_shapes[0]
    kh, kw = attrs.kernel
    return (N, C, H // kh, W // kw)
```

**这一步错了，图跑起来就是错的**——形状是布局变换、内存规划的地基。
所以**先写形状推导，再写计算**（`infer_shape` 和 `ref` 分开，就是为了让形状推导可单独验证）。

### 2.3 参考实现：`toycc/runtime/ref.py`

照 conv 的写法改成"取窗口最大值"：

```python
@staticmethod
def maxpool(inputs, attrs):
    x = inputs[0]
    kh, kw = attrs.kernel
    N, C, H, W = x.shape
    oh, ow = H // kh, W // kw
    out = np.zeros((N, C, oh, ow))
    for n in range(N):
        for c in range(C):
            for i in range(oh):
                for j in range(ow):
                    out[n, c, i, j] = x[n, c, i*kh:(i+1)*kh, j*kw:(j+1)*kw].max()
    return out
```

**为什么必须用"直接按定义"的写法？** 参考实现是要当裁判的，
它越笨、越接近数学定义，就越可信。不要在这里耍花招（比如用
`np.lib.stride_tricks` 写出"聪明"版本）——**裁判必须人眼可核对**。

### 2.4 建图 + 验证

在 `toycc/examples/model.py` 里把 maxpool 塞进模型（比如 conv1 之后），然后：

```bash
python -m toycc.examples.demo
```

**验证什么？三层：**
1. `max|Δ| ≈ 0` —— 数值对
2. `dump` 输出里 maxpool 的形状符合预期 —— 形状对
3. **融合规则有没有被误触发** —— 见 2.5

> 这三层就是编译器新人每天的验证循环：数值、形状、pass 行为。少验一层，
> 就可能在 1000 次推理后才炸。

> **为什么"会改代码"不是核心能力，"会验证"才是？**
>
> 你可能会觉得：任务 A 就是把 maxpool 登进去、写个实现——这有啥难的？
> 真正的难点在你**注意不到**的地方：
>
> - `RefImpl.maxpool` 写对了，但**建图时形状推断对了吗**？（`_maxpool_shape`）
> - maxpool 能不能被融合？**它和 conv 的依赖形状一样吗**？（2.5 的分析）
> - 输出对了，但**如果写法不同（比如 stride 处理），隐藏的形状/边界 case
>   会在第 1000 次推理时才炸**。
>
> **验证（`max|Δ|` / 单测 / 随机测试）就是替你覆盖这些"没想到"的地方。**
> 一个编译器工程师估值，不看"能写多少行"，看"写完敢不敢拍胸脯说它对"——
> 而这个底气全来自验证。这就是为什么本课每个任务都强制带验证：
> **你在练的不是"加算子"，是"让代码可见地、可证地对"。**
> 把这条习惯带进公司，你就比一半新人强。

### 2.5 思考：融合规则要不要为 maxpool 改？

maxpool 是**逐窗口**算子（输出一个点依赖输入一个窗口），不是逐元素。
对照我们的三条融合规则：

- 它不能当"根"（`_ROOTS` 只有 conv/matmul）——因为它没有"吸收下游"的形态
- 它**能被根吸收吗**？若 `maxpool(x)` 的输出喂给 conv，那 `conv(maxpool(x))`
  能不能写成一步？**数学上不行**（max 的窗口和 conv 的窗口不重叠，
  不能像 relu 那样逐元素剥离）。所以 `_POINTWISE` 不应该加 maxpool
- 但 `add(maxpool, bias)` 或 `relu(maxpool)` **可以**被 maxpool 吸收吗？
  理论上可行，但收益小、实现复杂——**不做是最优解**

**这就是改编译器的日常**：不是每加一个算子都要动融合规则。
先想"数学上允不允许、收益大不大"，再动手。真实 TVM 里对每个新算子
都要做同样的"合不合法、值不值得"分析（`op_pattern` 标注本身就是这分析）。

---

## 3. 任务 B：加一条融合规则（选做，强烈建议）

以 `mul`（乘常量，比如卷积输出乘一个缩放系数）为例：

**先做数学分析**（这是核心，代码反而是次要的）：

```
融合成: conv(x, w) * scale  ==  conv(x, w * scale) ?
答: 对。乘法对每个输出元素而言是逐元素的,和 conv 的线性累加可交换:
    (sum_k a_k w_k) * s == sum_k a_k (w_k * s)
```

所以正确做法是：**把 scale 乘进权重**，而不是给 conv 加一个 `mul` 属性。

```python
# _BIASABLE 里加 mul 是不对的——mul 不是"加",语义是"缩放":
# 要动的是权重常量,不是算子形态!
```

**这个任务的精髓**：融合规则不是"想融就融"，每一步都必须有
**数学等价性的证明**（前面把 `mul` 塞进 `_BIASABLE` 就是反例）。做一遍这个分析，
你就明白为什么真实编译器里每条融合规则都配着一篇注释解释"为什么等价"。

---

## 4. 任务 C：给"内存规划"加一种分配策略（进阶）

toycc 现在实现的是 first-fit（第一个死的就复用）。真实分配器还有 best-fit /
区间着色。任务：把 `memory.py` 的分配器改成 **best-fit**（选"死亡时间最接近"
的那个缓冲区），对比两种策略的峰值内存。

**验证**：两种策略的总分配应该一样（都是 448、2 个缓冲区）——因为
我们的图是"生命周期严格交替"的线形图，挑哪个死缓冲复用结果相同；
差异只在**有碎片时怎么挑**（见第 6 课扩展阅读 A）。改完后你看到的
是算法差异在真实代码里的模样。
做完说一下：**为什么现实里选哪种策略要看"碎片化"**——这正是
自研芯片工具链里要自己拍板的问题。

---

## 5. 装了 tvm 之后：写一个 TVM 自定义 pass

给 TVM 写一个最简单的自定义 pass（复制这段到 `tvm_demo.py` 里跑）：

```python
from tvm.ir import transform
from tvm.relax import expr, Function

@transform.function_pass(opt_level=0, name="MyFirstPass")
class MyFirstPass:
    def transform_function(self, func, mod, ctx):
        # 遍历函数体,打印每个 call 的算子
        def visitor(e):
            if isinstance(e, expr.Call):
                print("call:", e.op)
        expr.visit_functor(expr.PreOrderVisitor(visitor), func.body)
        return func
```

**这行代码在干嘛？** `function_pass` 是 TVM 的 pass 装饰器——它把你写的
"变换函数"包装成 pass 管线的标准件（pass 是无副作用的工序）：
输入一个 `func`，输出一个新 `func`，期间你可以自由打印、改写、替换。

**目标不是写出多牛的优化**，而是打通三个动作：
1. 注册一个 pass（`function_pass` 装饰器）
2. 遍历 IR 并打印（`PreOrderVisitor` —— 相当于我们 toycc 的 `consumers`/`topo_order` 遍历）
3. 跑测试看输出（`new_mod = MyFirstPass()(mod)`）

这就是 TVM 社区 "first-time contributor" 最常见的起点。

---

## 6. 参与真实社区：讨论和贡献（现在就能开始）

### 6.1 去哪讨论

| 地方 | 干什么 |
|---|---|
| GitHub Issues（apache/tvm） | 报 bug、讨论设计、认领任务 |
| GitHub Discussions / Discourse | 方案讨论、社区问答 |
| 官方文档 & RFC | 大功能设计文档（读 RFC 是学设计的最佳方式） |

**参与讨论的最低门槛**：读一个 issue，能复述"它报的是什么问题、和哪个
pass 有关、可能的修复方向"。用对照表去定位，你就已经有比很多人
清晰的视角。

### 6.2 怎么找"第一个任务"

TVM 有 `good-first-issue` 标签。打开：

```
https://github.com/apache/tvm/issues?q=is%3Aissue+is%3Aopen+label%3A%22good-first-issue%22
```

**挑任务的策略**：
1. 找和"pass/优化"相关的（你熟悉的领域）
2. 找带复现步骤的（能自己先复现）
3. 在 issue 下留言，维护者会给指引

### 6.3 贡献的完整流程

```
1. fork apache/tvm 到自己账号
2. git clone 到本地
3. 建分支:  git checkout -b fix-my-bug
4. 改代码 + 写测试(必须有!)
5. 本地跑相关测试
6. git push + 开 Pull Request
7. 通过 CI + 维护者 review + 修改意见
8. 合并!
```

**社区文化三条铁律**：测试先行（没写测试的改动基本不会被接受）、
小步提交（一次 PR 只做一件事）、尊重 review（维护者的意见是免费教学）。

---

## 7. 一份"四周上岗"计划（插在课程中间执行的）

| 周 | 目标 | 具体行动 |
|---|---|---|
| 第 1 周 | toycc 手感 | 做任务 A（maxpool）+ 任务 C（best-fit） |
| 第 2 周 | 真框架上手 | 装 tvm；跑通 tvm_demo；写 MyFirstPass |
| 第 3 周 | 读真实代码 | 挑 `relax/transform` 里一个 500 行内的 pass 精读，写 500 字笔记 |
| 第 4 周 | 第一次贡献 | 复现一个 good-first-issue；尝试提 PR（不行也先留讨论评论） |

做完这四周，再回课程主线（TIR 调度）。关键不是快，是**每个环节都
亲手做一遍**。

---

## 8. 面试/会议里会被问到的（提前打个底）

- "你了解哪些 pass？" → 融合/布局/折叠/内存，讲清楚动机和机制
- "pass 怎么保证正确性？" → 参考执行器 + `max|Δ|` / `assert_allclose`
- "布局为什么影响性能？" → 缓存局部性 + SIMD
- "TVM 的 IR 分层？" → Relax(高层图) → TIR(底层循环) → 后端
- "Relay 和 Relax 区别？" → 旧 vs 新高层 IR
- "你写过什么？" → toycc 的 maxpool、best-fit、tvm 的 MyFirstPass

---

## 9. 课程下半程地图

```
你已经在这 ──┐
              ▼
第 11~14 课   深入: 调度 / 优化全景 / 自动调度 / IR 家族
第 15~19 课   系统: 硬件 / 工程 / 导入 / 量化 / 性能
第 20 课     盘点: 知识地图(查缺补漏)
第 21~26 课  岗位: GPU 专项 + LLVM/MLIR
第 27~30 课  地基: 模拟器 / 并发原语 / 二进制 / 驱动(流片前四柱)
第 31~35 课  实战: 推理性能账 / 分布式 / 量化 / Triton+CUTLASS / 前沿
             (高性能部会议室里的日常话题)
```

**每个方向的下一步**（等你读完对应课后回来执行）：
- TIR 调度：读 TVM 官方教程 `tutorials/language/schedule_primitives.py`
- 量化：读 `relax/transform` 里量化相关 pass（第 18 课）
- 后端：读 `src/target/`、看一个 codegen 怎么实现（第 24 课）
- GPU：CUDA 编程手册 + 第 21~24 课实验
- 模拟器：给 toycc-ISA 写一个 30 行 ISS + 差分测试（第 27 课）
- 高性能部：口算三笔账——decode 上限、KV cache 容量、8 卡 allreduce 时间（第 31/32 课）

---

## 10. 扩展阅读 A：怎么读一个 PR（参与讨论的基本功）

看 PR 是学习 + 参与讨论的最好方式。一个 PR 通常包含：标题、描述、
改动文件、测试、CI 结果、review 对话。怎么高效读？

```
1. 先看标题 → 判断改的是哪个组件(transform/target/runtime?)
2. 读描述 → 它解决什么问题、为什么这么改
3. 只读核心 diff → 跳过格式改动(缩进/重命名), 看逻辑改动
4. 看测试 → 它怎么证明自己是对的(数值验证? 新用例?)
5. 看 review 对话 → 维护者提了什么意见, 为什么
```

**练习**：找一个已合并的 pass 相关 PR，照着这 5 步走一遍，写三句话总结。
这就是你参与讨论的"入场练习"。

---

## 11. 扩展阅读 B：四个讨论黑话快速上岗

| 黑话 | 意思 | 怎么用 |
|---|---|---|
| "这个能 `legalize` 吗" | 能不能下降到更底层的算子 | "conv2d 能 legalize 成 matmul" |
| "pass 的 `required` 是什么" | 前置依赖哪些 pass | "这 pass 依赖 FuseOps" |
| "这会不会破坏 `op_pattern`" | 会不会违反融合模式规则 | "新算子标 kBroadcast 就能融合" |
| "数值验证过了吗" | 有没有和参考结果对比 | "跑了 assert_allclose, 通过" |

掌握了这些黑话，你至少能听懂讨论在说什么。接下来就是敢开口提问。

---

## 12. 扩展阅读 C：从读到写的五个等级

| 等级 | 能力 | 证据 |
|---|---|---|
| L0 | 能看懂概念 | 讲得出 IR/Pass/后端 |
| L1 | 能读懂 toycc 全部代码 | 能加一个 maxpool 算子 |
| L2 | 能读懂 TVM 的一个 pass | 能说出 fuse_ops 的三步 |
| L3 | 能给 TVM 加一个小 pass | 写完能跑测试 |
| L4 | 能参与 PR review / 设计讨论 | 能评价"这个改动会不会破坏 X" |

完成本课任务 A（maxpool），你就到 L1；完成 MyFirstPass，你就到 L3。
**前半程课程的目标是把你带到 L3 的门口**；后半程（GPU 专项）的目标是
让你站上 L3、摸到 L4。

> 提醒：L0→L1 靠"写代码"；L1→L3 靠"读源码 + 写测试"。
> 每一步都要亲手做，别只"看"。

---

**导航**：⬅ [上一节](lesson09.md)（第 9 课 · 真实 TVM（下））　｜　[下一节](lesson11.md)（第 11 课 · TIR 与调度）➡
