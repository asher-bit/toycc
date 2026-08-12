# 第 8 课：真实 TVM ①——如何阅读一个工业级 Pass

> 前 7 课我们一直在 toycc 里看 IR、Pass 和代码生成。本课把视线转向真实 TVM，学习如何在一个更大的工程中定位入口、识别职责和建立局部理解。
> 本课先讲源码阅读方法，再用它分析 `fuse_ops.cc` 和 `fold_constant.cc`。
> 源码位置：`apache/tvm` 仓库 `src/relax/transform/`（当前 main 分支是 Relax 时代）
> 准备：跑 `python -m course.runner 8` 看对照清单。

> **完成标准**：你不需要记住每个 API，但要能说明一个 Pass 的入口、遍历对象、核心变换、分析依赖和测试方式。

本课是源码阅读方法和整体结构课。需要逐行精读时，进入本部分的专题页：

- [`fuse_ops.cc` 源码详解](tvm/fuse_ops.md)
- [`fold_constant.cc` 源码详解](tvm/fold_constant.md)
- [经典 Relax Pass 学习路线](tvm/pass_roadmap.md)

---

## 1. 为什么要"读源码"？——你现在的能力已经够了

经过前 7 课，你脑子里已经有了一张完整的地图：

```
IR(计算图) → Pass(融合/布局/折叠/内存) → 代码生成
```

真实 TVM 就是这张地图的"工业放大版"：
- 图更大、算子更多、更健壮
- 但骨架完全一样：IR、Pass、后端

所以你现在读 TVM 源码，**不是从零开始**，而是"用已知参照物找对应物"。
看到陌生代码，你该做的第一反应是：**"它对应 toycc 里的哪个函数？"**
对得上，就看细节；对不上，跳过。

---

## 2. 读大文件的方法论（三遍阅读法）

一个 1500 行的 `.cc`，别从头到尾读。按三遍走：

### 第一遍：结构扫读（5 分钟）
- 读**文件头注释**（`/* Note on ... */`）——作者把"为什么这么设计"写在最上面，
  这是最值钱的 20 行
- 用 `grep` / 搜索列出所有**类名和函数名**，画一个"谁调谁"的草图

### 第二遍：入口精读（10 分钟）
- 找 pass 工厂（如 `Pass FuseOps(...)`）：注册名、opt_level、`required` 依赖
- 找顶层函数（如 `FuseOps(mod, ...)`）：它分哪几步

### 第三遍：按需跳读（想多深就多深）
- 只读你关心的实现类（`ExprMutator`/`ExprVisitor` 子类）
- 带着问题读："它怎么判断能不能融合？""它怎么改写图？"

**核心技巧：永远带着 toycc 当参照物。** 你不需要理解每个 C++ 语法，
只需要理解"这一步对应 toycc 的哪一步"。

> **原理深挖：为什么"三遍读法"是对的？——这是记忆科学的结论**
>
> 人脑一次性能记住的"新奇点"只有 4~7 个。你从头到尾读一遍 1500 行，
> 第二行就把第一行盖掉了——**学完等于没学**。
>
> 三遍读法的本质是**分层压缩**：
> - 第一遍扫结构：把 1500 行压成"10 个函数名 + 依赖关系"（一张小地图）
> - 第二遍读入口：把"谁调谁"压成"三步流程"
> - 第三遍跳读：只在**带着具体问题**时深入细节
>
> 每一遍都在"上一遍的地图上"贴局部细节，而不是在情绪的混沌里从头再来。
> 这就是为什么：**先有地图，再进房间**——而不是一头扎进房间被家具淹死。
>
> 这套方法不止用于读源码：做 code review、排查 deadline 前的海量日志、
> 入职第一周读公司代码，全是同一套"先结构后细节"。它是一个**通用技能**，
> 不只属于编译器。**

---

## 3. 用方法论走一遍 `fuse_ops.cc`（1514 行）

### 第一遍：结构扫读

**文件头注释**（核心！）：

```cpp
/*
  Note on Fusing algorithm:

  The main challenge of general fusor is to handle possible diamond shape branches,
  in the following graph, conv2d can be fused to elemwise add.

            conv2d
            /  |  \
           /   |   \
         op    op   op
          \    |    /
           \   |   /
          elemwise add
               |

  - Construct a DAG of dataflow graph for dominator analysis
  - Construct a post-dominator tree which gives immediate post dominator of each node.
  - Run fusion algorithm with the given post-dominator information.
  ...
  - CheckPath: check all the path between a node and its immediate post-dominator
               satisfies the fuse condition.
  ...
  - CommitFuse: mark all the nodes between source and post-dominator as the same group.
  - We use an Union-Find data structure to manage the groups.
*/
```

**这段注释在说什么？** 翻译成人话：

- **问题**：菱形分支（一个输出分成多路，最后汇合）。贪心会抓瞎。
- **解法**：
  1. 建数据流 DAG
  2. 做支配分析，得到"每个节点的后支配者"（post dominator）——
     "在它之后**必然**执行的节点"（顺着所有路都经过它）
  3. 对每条路径检查融合条件是否满足（`CheckPath`）
  4. 满足就把整段标成一组（`CommitFuse`）
  5. 用并查集（Union-Find）管理分组

**为什么这个设计高明？** 第 3 课我们用 `len(consumers) != 1` 直接放弃菱形，
保守但错过优化。TVM 用后支配树把"整段路径是否安全"判断出来，
既能安全又能抓住优化机会。**这就是 toycc 和真框架的分水岭之一。**

**类名清单**（用 grep 找出来）：

```
GraphCreator                建图 (ExprVisitor)
FunctionCreator             为融合组构造新 Relax 函数 (ExprMutator)
OperatorFusor               融合改写主体 (ExprMutator)
PatternBasedPartitioner     按 DPL 数据流 pattern 分组
CompositeFunctionAnnotator  给融合函数打 kCodegen/kComposite 标注
FuseOps / FuseOpsByPattern  顶层函数 + Pass 工厂
```

### 第二遍：入口精读

```cpp
IRModule FuseOps(IRModule mod, int opt_level, size_t max_fuse_depth) {
  support::Arena arena;

  // Step 1. Create the indexed-forward graph according to the input IRModule.
  IndexedForwardGraph graph = GraphCreator::Create(mod, &arena);

  // Step 2. Partition the graph by applying the fusion algorithm.
  std::vector<GraphPartitioner::Group*> groups =
      GraphPartitioner(&arena, opt_level, max_fuse_depth, /*max_function_args=*/0).Partition(graph);

  // Step 3. Transform the IRModule by fusing the operators in accordance with the graph partition
  // results.
  return OperatorFusor(mod, graph, groups, /*lift_constants*/ true).Transform();
}
```

**三步，每一步都有 toycc 对应物**：

| TVM 步骤 | toycc 对应 |
|---|---|
| Step 1 建图 `GraphCreator::Create` | `Graph` + `consumers` |
| Step 2 分组 `GraphPartitioner::Partition` | `FusionPass._absorb_followers`（贪心） |
| Step 3 改写 `OperatorFusor::Transform` | `FusionPass._absorb`（rewire + 删节点） |

注意 `max_fuse_depth` 参数——**融合深度上限**。这就是第 3 课 FAQ 里说的
"防止单核过大"。TVM 有，toycc 没做。

**Pass 工厂**（看它是怎么注册进 pass 体系的）：

```cpp
Pass FuseOps(int fuse_opt_level) {
  auto pass_func =  //
      [=](IRModule m, PassContext pc) {
        int opt_level = fuse_opt_level == -1 ? pc->opt_level : fuse_opt_level;
        auto max_fuse_depth = pc->GetConfig<int64_t>("relax.FuseOps.max_depth", kMaxFusedOps);
        return relax::FuseOps(m, opt_level, static_cast<size_t>(max_fuse_depth.value()));
      };
  return CreateModulePass(/*pass_function=*/pass_func,  //
                          /*opt_level=*/0,              //
                          /*name=*/"FuseOps",           //
                          /*required=*/{});
}
```

**逐参数**：
- `pass_func`：一个 lambda，输入 IRModule + PassContext，输出新 IRModule
- `opt_level`：从 **PassContext** 读（全局优化级别 O0/O1/O2）——传 -1 表示"跟随全局"
- `pc->GetConfig("relax.FuseOps.max_depth", 256)`：**读配置项**，可被用户覆盖
- `name="FuseOps"`：注册名，Python 里就是 `relax.transform.FuseOps()`
- `required={}`：这个 pass 依赖哪些前置 pass（这里是空的）

对照 `toycc/passes/pipeline.py`：

```python
register_pass("fusion")(_F)     # 注册名 "fusion"
def run_passes(graph, passes):
    for name in passes:          # 相当于 Sequential
        g = _PASSES[name]()(g)   # 逐个执行
```

**概念完全同构**：注册名、调度器、配置。toycc 少了 opt_level、required、config。

### 第三遍：跳读（可选，想深入再看）

- `PatternBasedPartitioner`：用 TVM 的 DPL（数据流 pattern 语言）描述
  "conv2d+relu" 这样的模板，自动生成融合规则。这是"手写规则"的工业升级。
- `CompositeFunctionAnnotator`：给融合函数打 `kComposite` 标注，
  这样第 7 课讲的外部后端（`relax.ext.*`）能认出"这是谁融合的"。

---

## 4. 再用方法论走一遍 `fold_constant.cc`（447 行）

### 第一遍：结构扫读

类/函数清单：

```
ConstantFolder          折叠主体 (ExprMutator)
MatchConstShape         检查输出形状是常量
MatchPrimFunc           检查是纯函数、无副作用
GetCachedBuild          用 LLVM 编译 PrimFunc
ShouldBeFolded          值不值得折叠
ConstEvaluateCallTIR    对 call_tir 求值
VisitCallTIR / VisitExpr_(CallNode) / VisitExpr_(VarNode)  改写入口
FoldConstant()          Pass 工厂
```

### 第二遍：入口 + 关键决策

**`ShouldBeFolded`（第 5 课讲过，再看一遍"为什么"）**：

```cpp
bool ShouldBeFolded(Expr expr) {
  static constexpr int64_t kMaxFoldElements = 1024;
  ...
  if (num_elements <= kMaxFoldElements) return true;
  // Large output. Only skip if there are no tensor inputs,
  // i.e., this is a pure creation op.
  for (const auto& arg : call->args)
    if (ExprContainsTensor(arg)) return true;
  return false;
}
```

**主改写逻辑**：

```cpp
Expr VisitExpr_(const CallNode* call) final {
  // post-order mutation
  Call post_call = VisitExprPostOrder_(call).as_or_throw<Call>();

  if (!ShouldBeFolded(post_call)) return post_call;    // 不值得 → 不折

  static const Op& call_tir_op = Op::Get("relax.call_tir");
  ...
  if (op.same_as(call_tir_op)) {                       // 已经是 call_tir
    return VisitCallTIR(post_call).value_or(post_call);
  }
  ...
  if (builder_->CurrentBlockIsDataFlow()) {            // dataflow 块内
    if (legalize_map.count(op)) {
      // 先 legalize 成 call_tir, 再折叠
      Expr legalized_expr = builder_->Normalize(legalize_map[op](...));
      if (call && call->op.same_as(call_tir_op)) {
        return VisitCallTIR(ffi::GetRef<Call>(call)).value_or(post_call);
      }
    }
  }
  ...
}
```

**这段代码在问三个问题**：
1. 值得折吗？（`ShouldBeFolded`）
2. 它是 `call_tir` 吗？（已经是底层函数了 → 直接执行）
3. 它是高层算子吗？（先 `legalize` 下降成 `call_tir`，再执行）

`legalize_map` 是"高层算子 → 底层 call_tir"的翻译表——
TVM 里叫 **legalization（合法化/下降）**，这是 toycc 没做的一层：
我们把图直接执行了，TVM 得先把"逻辑算子"翻译成"可编译的函数"再折。

**`ConstEvaluateCallTIR` 的求值**（第 5 课提过）：

```cpp
// 1. GetCachedBuild(prim_func) → 用 LLVM 编译成可执行函数
// 2. 分配输出, 调用, 拿结果
// 3. 包成 Constant 放回 IR
```

这就是"编译期执行"：不是 numpy，而是真的编译+跑一遍。思想同构，引擎不同。

### 第三遍：细节

`MatchPrimFunc` 检查：函数必须是纯函数、输出形状必须是常量（不能是符号形状）。
**为什么？** 折叠是"把结果焊进 IR"，如果输出形状都不确定，没法焊。
如果函数有副作用（打印/计数/读文件），提前执行会改变语义。
toycc 的算子都是纯的、形状都是确定的，所以没做这些检查——但你要知道
生产级编译器必须做。

---

## 5. 动手环节

```bash
python -m course.runner 8
```

然后打开浏览器（或本地 clone 的 tvm 仓库）：

```
https://github.com/apache/tvm/blob/main/src/relax/transform/fuse_ops.cc
https://github.com/apache/tvm/blob/main/src/relax/transform/fold_constant.cc
```

**建议动作清单**：
1. 在 `fuse_ops.cc` 里搜 `Note on Fusing algorithm`，读那段注释
2. 搜 `Pass FuseOps(`，对照本课的逐参数分析
3. 在 `fold_constant.cc` 里搜 `ShouldBeFolded` 和 `ConstEvaluateCallTIR`
4. 回到 toycc，把"我们缺什么"写成一个清单（后支配树、legalize、配置、required...）

---

## 6. 完整对照表（本课核心产出）

| 概念 | toycc | TVM Relax |
|---|---|---|
| 图结构 | `Graph`/`Node` | `IRModule`/`Expr`(call) |
| 融合分组 | 贪心吸唯一消费者链 | 后支配树 + 并查集 |
| 融合改写 | `_absorb` rewire+删节点 | `OperatorFusor` 生成 fused function |
| 融合深度上限 | 无 | `max_fuse_depth`（默认 256） |
| 折叠求值 | numpy `RefImpl` | LLVM 编译执行 |
| 折叠权衡 | 无 | `ShouldBeFolded`（1024 元素上限） |
| 纯函数检查 | 无（都是纯的） | `MatchPrimFunc` |
| 高层算子下降 | 直接执行 | `legalize_map` → call_tir |
| pass 注册 | `register_pass("fusion")` | `CreateModulePass(name="FuseOps")` |
| pass 配置 | 无 | `PassContext` + `pc->GetConfig` |

**这张表就是你"读得懂 TVM"的通行证。** 以后在任何编译器源码里看到
陌生概念，先在表里找"对应物"。

---

## 7. 课后答疑

**Q：为什么不直接学 TVM，还要先写 toycc？**
A：toycc 是"无装饰的地图"——每行代码都能对应一个概念。直接啃 TVM，
你会被 C++ 模板、FFI、几千行基础设施淹没，分不清主次。
先有骨架，再填血肉，效率高得多。

**Q：`ExprMutator`/`ExprVisitor` 是什么？**
A：TVM IR 的两种遍历器：`Visitor` 只读不改，`Mutator` 遍历时能改写并返回新对象。
toycc 里我们手写 `for node in topo_order()` 遍历——概念一样，TVM 封装成了基类。

**Q：`IRModule` 和我们的 `Graph` 什么关系？**
A：`IRModule` 是"一组函数"的集合（一个模型可以拆成多个子函数），
每个函数内部才是类似我们 `Graph` 的结构（一串 binding）。
我们 toycc 一个 `Graph` = 一个函数。

**Q：TVM 的 pass 在 Python 里怎么调用？**
A：`relax.transform.FuseOps()` 返回一个 Pass 对象，放进
`tvm.ir.transform.Sequential([...])` 里串起来。第 9 课会完整跑一遍。

---

## 8. 本课小结

- 读大源码三遍法：**结构扫读 → 入口精读 → 按需跳读**
- 永远带参照物：每个类/函数都问"对应 toycc 的什么"
- `fuse_ops.cc`：后支配树处理菱形分支，三步（建图→分组→改写）
- `fold_constant.cc`：值得吗 + 纯函数吗 + 先下降再执行
- 把"对照表"当通行证，以后的源码都用它定位

**下一步**：第 9 课——把 toycc 的整条管线在**真实 TVM 里跑一遍**，
然后我们聊聊：怎么真正开始参与编译器开发。

---

## 9. 扩展阅读 A：TVM 源码树怎么逛？（导航地图）

第一次打开 `github.com/apache/tvm` 会懵。给你一张导航地图：

```
apache/tvm
├── src/                       ← C++ 核心(绝大多数 pass 在这)
│   ├── relax/transform/       ← 高层图 pass(本课主角)
│   ├── tir/                   ← TIR + 调度 + 各种 lowering
│   ├── arith/                 ← 算术/依赖/化简分析
│   ├── target/                ← 各硬件后端
│   ├── topi/                  ← 算子的 TVM 实现(conv/matmul 的 te 定义)
│   └── runtime/               ← 运行时(VM、内存管理)
├── python/                    ← Python 绑定(API 定义、前端)
│   └── tvm/
│       ├── relax/transform/   ← pass 的 Python 壳(和 C++ 一一对应)
│       ├── ir/                ← IR 对象、pass infra 的 Python 侧
│       └── meta_schedule/     ← 自动调度(第 13 课)
├── tests/python/              ← 测试(读它=学用法)
├── apps/                      ← 集成示例(怎么用 tvm 部署)
└── 3rdparty/                  ← 第三方依赖(cutlass、tensorrt 等)
```

**逛法建议**：
- 想看"某个 pass 怎么用" → 找 `tests/python/relax/test_transform_<名字>.py`
- 想看"某个 pass 怎么实现" → 找 `src/relax/transform/<名字>.cc`
- 想找"Python API 在哪" → `python/tvm/relax/transform/transform.py`

---

## 10. 扩展阅读 B：FFI——Python 怎么调 C++（TVM 的连接层）

你在 Python 里写 `relax.transform.FuseOps()`，但实现是 C++ 的 `fuse_ops.cc`。
怎么连起来的？靠 **FFI（Foreign Function Interface）**。

```
Python:  relax.transform.FuseOps()
            ↓ 注册名 "relax.FuseOps" 查表
FFI:     tvm.ffi.Function::GetGlobal("relax.FuseOps")
            ↓ 调 C++ 函数
C++:     Pass FuseOps(int) {...}
```

TVM 的 FFI 是一个**全局注册表**：C++ 侧 `TVM_REGISTER_GLOBAL("relax.FuseOps")`
注册函数，Python 侧 `tvm.get_global_func(...)` 查到并调用。
**所有 pass 都通过这个表注册**——这就是第 3 课讲的"注册表"思想的
工业实现。

**为什么知道 FFI 很重要**？因为读 TVM 时你会看到：
- C++ 定义 → `TVM_REGISTER_GLOBAL`
- Python 侧 `@register_func` / `tvm.ir.transform` 包装
- 新增一个 pass 要"两边登记"

理解 FFI，你就理解了"为什么一个 pass 在 Python 和 C++ 都出现"。

---

## 11. 扩展阅读 C：在 1500 行文件里快速定位（grep 方法）

读大文件的三遍法里，"结构扫读"要靠搜索。给你一套实用命令：

```bash
# 找类/函数定义
grep -n "^class \|^IRModule \|^Pass \|::\w*(" fuse_ops.cc

# 找关键注释
grep -n "Note on" fuse_ops.cc

# 找 pass 工厂
grep -n "CreateModulePass\|CreateFunctionPass" fuse_ops.cc

# 找配置项
grep -n "GetConfig\|max_fuse_depth" fuse_ops.cc
```

在 GitHub 网页上：点文件 → `Ctrl+F` 搜索 → 用右上角
"符号列表"（blob 视图的 jump to）直接跳类定义。

**原则**：先找"骨架符号"（Pass 工厂、入口函数、头注释），
再决定读哪段。**不要从头读到尾**——那是新手最容易犯的错，
也是读懂大文件的最大障碍。

---

**导航**：⬅ [上一节](lesson07.md)（第 7 课 · 代码生成）　｜　[下一节](lesson09.md)（第 9 课 · 真实 TVM（下））➡
