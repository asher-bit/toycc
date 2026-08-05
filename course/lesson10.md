# 第 10 课：从"看懂"到"上手"——怎么真正参与编译器开发

> 这是最后一课，但也是你"开发者生涯"的第一课。
> 目标：给你一份可执行的路线图，从"学完了"变成"能动手、能讨论、能贡献"。

---

## 1. 先盘点：你现在拥有了什么

完成前 9 课，你拥有的能力清单：

| 能力 | 证据 |
|---|---|
| 看得懂 IR | 能读 `Graph`/`Node`，能理解 `relax.call_tir` 的文本表示 |
| 看得懂 pass | 融合/布局/折叠/内存，每个都能讲出"输入→输出→为什么" |
| 知道正确性怎么保证 | `max|Δ|`、参考执行器、`assert_allclose` |
| 看得懂 TVM 源码 | 会三遍读法，能定位 `fuse_ops.cc` 里任何类的职责 |
| 会跑真框架 | 装 tvm、`Sequential([...])`、`relax.build` |

**这已经是一个"能进编译器会议室的人"的最低配置。** 你现在缺的不是知识，
是**手感**——亲手改过代码、跑过测试、碰过 bug 的经验。

---

## 2. 路线一（今天就能做）：给 toycc 加功能——练手感

这是最重要的热身。挑一个做：

### 任务 A：加一个新算子 `maxpool`
1. 在 `toycc/ir/ops.py` 注册：`register("maxpool", 1, _maxpool_shape, None)`
2. 在 `toycc/runtime/ref.py` 写 `RefImpl.maxpool`（照 conv 写，改成取 max）
3. 在 `toycc/examples/model.py` 塞进模型
4. `python -m toycc.examples.demo` 看验证是否 `OK`
5. 思考：融合规则要不要改？（maxpool 能不能当根？）

**这一套动作 = 一个真实编译器新人入职第一周的工作。**
涉及注册表、参考实现、建图、验证——全部是你学过的。

### 任务 B：给融合 pass 加一条规则
比如让 `mul`（乘常量，如缩放系数）也能被吸收进 conv/matmul。
要改 `fusion.py` 的 `_BIASABLE`，还要想清楚：
`conv(x) * scale` 融合进 conv 的数学形式是什么？**会不会有语义问题？**

### 任务 C：实现"死代码消除"（DCE）
遍历图，删掉"没人消费的节点"（`consumers` 为空且不是输出）。
这是第 5 课提过的 pass，现在你自己写一个。

### 任务 D（进阶）：量化 pass
给图加一个 `QuantizePass`：把 `conv` 改成"先量化输入→int8 卷积→反量化"。
验证精度损失多少（`max|Δ|` 会变大，但应该在可接受范围）。

> 每个任务都配了"验证"——做完必须跑 demo，`max|Δ|` 说明一切。
> **这就是编译器开发的日常节奏：改代码 → 数值验证 → 改 bug → 再验证。**

---

## 3. 路线二（装好 tvm 后）：给 TVM 写一个小 pass

用官方 API 写一个最简单的自定义 pass：

```python
from tvm.ir import transform
from tvm.relax import expr, Function

@transform.function_pass(opt_level=0, name="MyFirstPass")
class MyFirstPass:
    def transform_function(self, func, mod, ctx):
        # 遍历函数体, 打印每个 call_tir 的算子
        def visitor(e):
            if isinstance(e, expr.Call):
                print("call:", e.op)
        expr.visit_functor(expr.PreOrderVisitor(visitor), func.body)
        return func

# 用 tvm_demo.py 里的 mod 试:
# new_mod = MyFirstPass()(mod)
```

**目标不是写出多牛的优化，而是**：
- 学会"注册一个 pass"
- 学会"遍历 IR 并打印"
- 学会"跑测试看输出"

这就是 TVM 社区里"first-time contributor"最常见的起点。

---

## 4. 路线三：参与真实社区——讨论和贡献

### 4.1 去哪讨论（术语也要会）

| 地方 | 干什么 |
|---|---|
| GitHub Issues（apache/tvm） | 报 bug、讨论设计、认领任务 |
| GitHub Discussions / Discourse 论坛 | 方案讨论、社区问答 |
| 官方文档 & RFC | 大功能设计文档（读 RFC 是学设计的最佳方式） |
| Discord/Slack（社区工作区） | 日常交流 |

**你能参与讨论的最低门槛**：读一个 issue，能复述"它报的是什么问题、
和哪个 pass 有关、可能的修复方向"。用本课对照表定位，你就已经有了
比很多人更清晰的视角。

### 4.2 怎么找"第一个任务"（First Issue）

TVM 有 `good-first-issue` 标签。打开：

```
https://github.com/apache/tvm/issues?q=is%3Aissue+is%3Aopen+label%3A%22good-first-issue%22
```

**挑任务的策略**：
1. 找和"pass/优化"相关的（你熟悉的领域）
2. 找带复现步骤的（能自己先复现）
3. 在 issue 下留言，社区维护者会给指引

### 4.3 贡献的完整流程（Git 工作流）

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

**记住社区文化**：
- **测试先行**：没写测试的改动基本不会被接受
- **小步提交**：一次 PR 只做一件事
- **尊重 review**：维护者的意见是免费教学
- **先看贡献文档**：`CONTRIBUTING.md` / `docs/dev`

---

## 5. 一份"四周上岗"计划

| 周 | 目标 | 具体行动 |
|---|---|---|
| 第 1 周 | toycc 手感 | 做路线一的任务 A + C（maxpool + DCE） |
| 第 2 周 | 真框架上手 | 装 tvm；跑通 tvm_demo；写 MyFirstPass |
| 第 3 周 | 读真实代码 | 挑 `relax/transform` 里一个 500 行内的 pass 精读，写一篇 500 字笔记 |
| 第 4 周 | 第一次贡献 | 复现一个 good-first-issue；尝试提交 PR（不行也先留讨论评论） |

> 调整建议：如果周一到周四时间有限，把计划拉长到 6~8 周。
> 关键不是快，是**每个环节都亲手做一遍**。

---

## 6. 面试/会议里会被问到的（提前打个底）

- "你了解哪些 pass？" → 融合/布局/折叠/内存，讲清楚动机和机制
- "pass 怎么保证正确性？" → 参考执行器 + `max|Δ|` / `assert_allclose`
- "布局为什么影响性能？" → 缓存局部性 + SIMD
- "TVM 的 IR 分层？" → Relax(高层图) → TIR(底层循环) → 后端
- "Relay 和 Relax 区别？" → 旧 vs 新高层 IR
- "你写过什么？" → toycc 的 maxpool、DCE、tvm 的 MyFirstPass

---

## 7. 长期进阶地图

```
你已经在这 ──┐
             ▼
toycc(骨架) → TVM Relax(pass 框架) → TIR 调度(手写优化)
                                      → 量化/剪枝/蒸馏
                                      → 后端移植(CUDA/端侧)
                                      → 分布式/大模型推理优化
```

**每个方向的下一步**：
- TIR 调度：读 TVM 官方教程 `tutorials/language/schedule_primitives.py`，学 `te` API
- 量化：读 `relax/transform` 里量化相关 pass + TVM 文档 quantization 篇
- 后端：读 `src/target/`、看一个 codegen（如 C codegen）怎么实现
- 大模型：TVM 的 LLM 相关 (`relax.frontend.llm`) 是当前最热的领域

---

## 8. 最后的话

你用了两周不到，把很多人学了半年还在"看不懂概念"的东西，
做成了一个**能跑、能验证、能对照真框架**的完整系统。这个基础比
大部分"看过教程"的人扎实——因为你**亲手写过、跑过、改错过**。

剩下的路，就是**继续写、继续读、继续交流**。编译器是个"越读越觉得自己
懂的少"的领域，但你已经拿到了地图。祝你上手指日可待。

> 把这条路线图收藏好。等你在 toycc 上做完 maxpool 和 DCE，
> 随时回来找我——我陪你做下一步。

---

## 9. 深层拓展 A：怎么"读一个 PR"（参与讨论的基本功）

看 PR 是学习 + 参与讨论的最好方式。一个 PR 通常包含：标题、描述、
改动文件、测试、CI 结果、review 对话。怎么高效读？

```
1. 先看标题 → 判断改的是哪个组件(transform/target/runtime?)
2. 读描述 → 它解决什么问题、为什么这么改
3. 只读核心 diff → 跳过格式改动(缩进/重命名), 看逻辑改动
4. 看测试 → 它怎么证明自己是对的(数值验证? 新用例?)
5. 看 review 对话 → 维护者提了什么意见, 为什么
```

**练习**：找一个已合并的 pass 相关 PR，照着这 5 步走一遍，
写三句话总结。这就是你参与讨论的"入场练习"。

---

## 10. 深层拓展 B：四个"讨论黑话"快速上岗

参与编译器讨论，这几个词先学会：

| 黑话 | 意思 | 怎么用 |
|---|---|---|
| "这个能 `legalize` 吗" | 能不能下降到更底层的算子 | "conv2d 能 legalize 成 matmul" |
| "pass 的 `required` 是什么" | 前置依赖哪些 pass | "这 pass 依赖 FuseOps" |
| "这会不会破坏 `op_pattern`" | 会不会违反融合模式规则 | "新算子标 kBroadcast 就能融合" |
| "数值验证过了吗" | 有没有和参考结果对比 | "跑了 assert_allclose, 通过" |
| "这里该 `lower` 吗" | 该下降吗（从高层到底层） | "先 legalize 再 lower 到 TIR" |

**掌握了这些黑话，你至少能听懂讨论在说什么**。接下来就是敢开口提问。

---

## 11. 深层拓展 C：从"读"到"写"的五个等级（自测你在哪）

| 等级 | 能力 | 证据 |
|---|---|---|
| L0 | 能看懂概念 | 讲得出 IR/Pass/后端 |
| L1 | 能读懂 toycc 全部代码 | 能加一个 maxpool 算子 |
| L2 | 能读懂 TVM 的一个 pass | 能说出 fuse_ops 的三步 |
| L3 | 能给 TVM 加一个小 pass | 写完能跑测试 |
| L4 | 能参与 PR review / 设计讨论 | 能评价"这个改动会不会破坏 X" |

完成第 10 课任务 A（maxpool），你就到 L1；
完成 MyFirstPass，你就到 L3。**本课程的目标是把你带到 L3 的门口。**

> 提醒：L0→L1 靠"写代码"；L1→L3 靠"读源码 + 写测试"。
> 每一步都要亲手做，别只"看"。

---

**导航**：⬅ [上一节](lesson09.md)（第 9 课 · 真实 TVM（下））　｜　[下一节](lesson11.md)（第 11 课 · TIR 与调度）➡
