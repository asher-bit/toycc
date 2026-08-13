# 第 9 课：真实 TVM ②——把源码理解变成一次真实运行

> 第 8 课解决了“如何读”，本课继续解决“如何验证自己读对了”。我们先把剩余源码放回 TVM 的整体管线，再运行一条最小流程，把 toycc 的概念和真实框架对应起来。
>
> **完成标准**：你应该能把一个 Relax Pass 放进编译流程，观察输入和输出 IR，并说明它与 toycc 对应实现的相同点和不同点。

完整的 Relax Pass 阅读顺序和后续源码清单见：[真实 TVM 源码精读专题](tvm/pass_roadmap.md)。

---

## 1. 先把剩下 4 个源码文件扫一遍（用"三遍法"）

### 1.1 `convert_layout.cc`（373 行）——布局

对照 `toycc/passes/layout.py`。

**结构扫读**：

```
LayoutConvertMutator    改写主体 (ExprMutator)
  LayoutToIntegers      布局 → 轴编号数组
  LayoutIndexMap        布局 → index_map(用于 layout_transform)
  RewriteExpr           插 permute_dims / layout_transform
  RewriteArgs           重写算子参数
  GetInferLayoutInfo    查 FRelaxInferLayout 注册表
ConvertLayoutPass / ConvertLayout()
```

**入口精读**——头注释：

```cpp
/*
  每个 op 都注册了布局推导函数 FRelaxInferLayout:
  输入是(当前 call, 想要的布局, 之前变量的布局表),
  输出是(输入布局, 输出布局, 可能被改写的 attrs)。
  注意: 目前只支持轴交换(NCHW↔NHWC), 不支持打包布局(NCHW→NCHW4c)。
*/
```

**关键点**：TVM 用**注册表**（`FRelaxInferLayout`）而不是手写规则。
每个算子声明"我吃啥布局、吐啥布局"，pass 遍历时查表。好处：
新增算子只需注册，不用改 pass。对照 toycc 的 `_LAYOUT_AGNOSTIC` 集合 +
手写规则——我们是在"纸面上注册"。

**插转换的核心 `RewriteExpr`**：

```cpp
Expr RewriteExpr(const Expr& expr, const NLayout& to) {
  ...
  if (NLayoutEqual()(from, to) || ...) return expr;   // 布局一样 → 跳过
  ...
  if (from.LeafValue()->layout.ndim() == to.LeafValue()->layout.ndim()) {
    SLayout axes = TransposeLike(...);
    return permute_dims(expr, LayoutToIntegers(axes));   // 轴交换
  } else {
    auto index_map = LayoutIndexMap(...);
    ...
    return Call(Type::Missing(), layout_transform_op_, {expr}, attrs, {});
  }
}
```

**逐行**：
- `NLayoutEqual()(from, to)`：布局相同 → 直接返回（零开销，同我们的 `cur == want`）
- `ndim` 相同 → 轴交换 → 用 `permute_dims`（= 我们的 transpose）
- `ndim` 不同（如 NCHW→NCHW4c 打包布局）→ 用 `layout_transform` + `index_map`

对照 toycc `_ensure_layout`：**布局相同跳过 / 不匹配插 transform**，完全同构。

### 1.2 `kill_after_last_use.cc`（281 行）——内存(释放点)

对照 `toycc/passes/memory.py`。

```
CollectLastUsage        收集"最后一次使用" (ExprVisitor)
  static Result Collect(expr)
KillInserter            在 last-use 后插 kill 指令 (ExprMutator)
UnusedTrivialBindingRemover  清理无用绑定
KillAfterLastUse / Pass 工厂
```

**核心**：

```cpp
static Result Collect(const Expr& expr) {
  CollectLastUsage visitor;
  visitor(expr);
  Result output;
  for (const auto* var : visitor.binding_order_) {
    if (auto it = visitor.last_usage_of_.find(var); it != visitor.last_usage_of_.end()) {
      const auto* last_usage_point = it->second;
      bool is_output = last_usage_point == nullptr;      // 输出活到最后
      ...
      if (!is_output && !already_killed) {
        if (visitor.storage_objects_.count(var))
          output[last_usage_point].storage.push_back(var);
        else if (var->ty.as<TensorTypeNode>() && stored_in_vm_register)
          output[last_usage_point].tensors.push_back(var);
        else if (stored_in_vm_register)
          output[last_usage_point].objects.push_back(var);
      }
    }
  }
  return output;
}
```

**三个要点**：
1. `is_output = last_usage_point == nullptr`：**输出活到最后**（同我们
   `died[i] = len(topo)`）
2. 结果按"最后一次使用点"组织成 `Result` 表
3. 把张量/存储/对象**分类**对待（storage 是内存块，tensor 是值，
   object 是 VM 对象）——我们 toycc 只处理张量

插入释放：

```cpp
void VisitBinding(const Binding& binding) override {
  ...
  if (auto it = last_usage_.find(binding->var.get()); it != last_usage_.end()) {
    static const Op& mem_kill_tensor = Op::Get("relax.memory.kill_tensor");
    for (const auto& tensor_obj : it->second.tensors) {
      builder_->Emit(Call(Type::Missing(), mem_kill_tensor, {tensor_obj}));
    }
  }
}
```

**对比**：我们把"何时能复用"算进分配表；TVM 生成显式 `kill_tensor` 算子，
由**运行时**真正释放。一个"静态规划"，一个"显式指令"——本质都是 last-use 分析。

### 1.3 `allocate_workspace.cc`（215 行）——工作区

**核心**：

```cpp
// 给带 kWorkspaceSize 的外部函数追加 workspace 形参
if (auto workspace = func_node->GetAttr<int64_t>(attr::kWorkspaceSize)) {
  auto ty = TensorType(ShapeExpr({IntImm::Int32(max_workspace_size_)}), PrimType::UInt(8));
  Var workspace_param(name_sup_->FreshName("workspace"), ty);
  ...
}
// 在主函数开头分配一块 workspace_main
auto workspace = MakeAllocTensor(shape, ty, IntImm::Int64(0));
workspace_var_main_ = builder_->Emit(workspace, "workspace_main");
```

**在解决什么问题**？有些算子（如某些 kernel）需要一块**临时工作区**
（scratch space）。这个 pass：
1. 读每个外部函数的 `kWorkspaceSize` 属性，知道它要多大
2. 按**最大值**分配一块
3. 给需要它的函数追加 workspace 参数

对应我们 `report()` 里的峰值统计：**工作区大小 = 所有需求的最大值**。

### 1.4 `run_codegen.cc`（244 行）——代码生成入口

`InvokeCodegen` 会把带 `kCodegen` 属性的函数分桶。补一个细节——**用 ExternFunc 替换**：

```cpp
Expr VisitExpr_(const FunctionNode* func_node) override {
  Function func = ffi::GetRef<Function>(func_node);
  auto opt_codegen = func->GetAttr<ffi::String>(attr::kCodegen);
  if (opt_codegen) {
    auto ext_symbol = GetExtSymbol(func);
    // 给常量起唯一名字(供外部后端引用)
    ...
    return ExternFunc(GetExtSymbol(func));   // 替换成外部符号
  }
  return ExprMutator::VisitExpr_(func_node);
}
```

**"用 ExternFunc 替换"意味着**：融合后的函数被交给 `relax.ext.<target>`
后端（CUTLASS/TensorRT...）编译成库，图里只留一个外部符号调用。
这就是"融合的成果交给外部专家后端"的完整闭环。

---

## 2. 安装 TVM

**方式 A：pip（最快）**

```bash
python -m pip install apache-tvm
```

> `apache-tvm` 是官方包名。注意：如果你的 Python 太新（3.14 很可能）没有
> 对应轮子，会报错——那就用方式 B。

**方式 B：Docker 官方镜像（学习推荐）**

```bash
docker run -it --rm tlcpack/ci-cpu:latest bash
```

进去后 `python -c "import tvm; print(tvm.__version__)"` 验证。
官方文档：`https://tvm.apache.org/docs/install/index.html`

**判断装没装成功**：

```bash
python -c "import tvm; print(tvm.__version__)"
```

---

## 3. 在真实 TVM 里复刻 toycc 的整条管线

存成 `course/tvm_demo.py`：

```python
"""tvm_demo.py: 用真实 TVM Relax 复刻 toycc 的 pass 管线。"""
import numpy as np
import tvm
from tvm import relax


def build_small_model():
    """构造 1x3x8x8 的小模型, 返回 (IRModule, params)。
    三种方式任选:
      A. tvm.relax.testing.nn.Module 前端(需 torch)
      B. ONNX 导入:   tvm.relax.frontend.onnx.from_onnx(onnx_model)
      C. 手动拼 relax.expr
    本骨架以 A 为例, 按官方 tutorial 补全。
    """
    from tvm.relax.testing import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1, strides=1)
            self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1, strides=2)
            self.fc = nn.Linear(8 * 4 * 4, 16)

        def forward(self, x):
            x = tvm.relu(self.conv1(x))
            x = tvm.relu(self.conv2(x))
            x = tvm.reshape(x, (1, -1))
            x = tvm.relu(self.fc(x))
            return x

    return nn.Module.from_net(Net(), {"x": tvm.tir.Tensor((1, 3, 8, 8), "float32")})


def main():
    mod, params = build_small_model()
    print("=== 初始 IRModule ===")
    print(mod)

    # 串行 pass 管线(对照 toycc 的 ("fusion","layout","constfold"))
    pipeline = tvm.ir.transform.Sequential([
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),
        relax.transform.FoldConstant(),
        relax.transform.ConvertLayout({"nn.conv2d": ["NHWC", "OHWI"]}),
        relax.transform.LegalizeOps(),
        relax.transform.KillAfterLastUse(),
        relax.transform.AllocateWorkspace(),
    ])

    optimized = pipeline(mod)
    print("\n=== 优化后 IRModule ===")
    print(optimized)

    # 正确性验证
    ex = relax.build(optimized, target="llvm")
    vm = tvm.runtime.vm.VirtualMachine(ex, tvm.cpu())
    x = np.random.default_rng(1).standard_normal((1, 3, 8, 8)).astype("float32")
    out = vm["main"](tvm.nd.array(x)).numpy()
    print("\n输出 shape:", out.shape)


if __name__ == "__main__":
    main()
```

**跑起来看什么**：
1. 每个 `Sequential` 阶段后 `print(mod)`，看 `IRModule` 文本怎么变
2. 融合后：`call_tir` 变少，PrimFunc 里合并了 bias 和 relu
3. `ConvertLayout` 后：出现 `permute_dims`（= 我们的 layout_transform）
4. 对照 toycc 每个 pass 打印的图

> 如果某个 pass 名在你的版本不存在，用 `relax.transform.<Tab>` 补全，
> 或打开 `python/tvm/relax/transform/transform.py` 看注册列表——
> 这本身就是很好的"读源码"练习。

> **原理深挖：为什么"在真实框架里复刻"是学习编译器的捷径？**
>
> 你可能会想：toycc 我懂了，为什么还要费劲去真实 TVM 里重跑一遍？
> 因为**玩具和真实的差距，只有亲手跨一次才能量出来**。
>
> - toycc 的 `FusionPass` 只有 82 行；真实 `FuseOps` 是 1514 行。
>   差的不是代码量，是**工业需求的增量**：多消费者菱形、子图划分、
>   op_pattern 标注、缓存……每一条增量都是你没想到过的 edge case。
> - "对照表法"的本质是：**用你知道的（toycc）去锚定你不知道的（TVM）**。
>   你不需要一次懂 1514 行，只需要在每一行回答一次
>   "这对应 toycc 的哪一步"——锚点越清晰，细节越容易贴上去。
>
> 这就是为什么这门课坚持"toycc + 真实阅读"双线：
> **玩具给你主权（能改），真实给你坐标（知道自己站在哪）**。
> 缺玩具，你不知道自己在干什么；缺真实，你学的只是"一个玩具"。

---

## 4. 通关清单（全部做到 = 可以开始参与开发）

- [ ] 不看注释，能讲出 toycc 每个 pass 的输入/输出/为什么
- [ ] 能说出 `FuseOps` 里 `GraphPartitioner` 和 toycc 贪心的差距
- [ ] 能解释 `ConvertLayout` 为什么只在边界插 `permute_dims`
- [ ] 装好 tvm，跑通 `tvm_demo.py`，指出它对应 toycc 的哪个函数
- [ ] 打开任意 `src/relax/transform/*.cc`，10 分钟内说出它干嘛的
- [ ] 能画出"IR → 融合 → 布局 → 折叠 → 内存 → 后端"的完整地图

---

## 5. 课后答疑

**Q：为什么 TVM 有这么多 pass？我数了下 transform 目录有几十个。**
A：生产编译器按"每件事一个 pass"组织：融合、布局、折叠、dead code、
inline、legalize、调参……几十个很正常。我们的 toycc 只做了其中 4 个核心的。
**读了本课，你已经掌握这几十个里最核心的那批。**

**Q：Relax 和 Relay 是什么关系？**
A：Relay 是 TVM 旧的高层 IR，2023 年后被 Relax 取代。Relax 面向动态形状、
分布式、端侧部署。你现在读的源码都是 Relax 时代的。**骨架一样，细节升级。**

**Q：装了 tvm 之后，我该从哪里开始"动手"？**
A：接下来会给一份完整的"上岗路线图"：改 toycc → 给 TVM 写小 pass →
跑官方测试 → 找 first issue。别急着啃大功能，从小 pass 开始。

---

## 6. 本课小结

- 4 个源码文件对应 toycc 的 4 件事：布局/内存/工作区/后端
- 安装：`pip install apache-tvm` 或 Docker 官方镜像
- `Sequential([...])` = toycc 的 `run_passes(("fusion",...))`
- 通关清单 6 条全做到，就可以开始参与真实开发了

**下一步**：第 10 课会把重点从“理解”转到“动手”：你将给 toycc 加功能，练习测试和调试，再把这套工作方式迁移到真实 TVM 的贡献流程。

---

## 7. 扩展阅读 A：Pass 基础设施的“三张通行证”

TVM 的 pass 体系有**三种 pass 类型**，对应三种作用范围：

| 类型 | 作用范围 | 创建函数 | toycc 对应 |
|---|---|---|---|
| `ModulePass` | 整个 IRModule（一组函数） | `CreateModulePass` | `run_passes` |
| `FunctionPass` | 每个 Relax Function | `CreateFunctionPass` | 作用在单个 `Graph` |
| `DataflowBlockPass` | 每个数据流块 | `CreateDataflowBlockPass` | `LayoutPass` 作用范围 |

**怎么选**？看你要改什么：
- 要加/删整个函数 → ModulePass（如 `AllocateWorkspace`）
- 要改每个函数内部 → FunctionPass（如 `FoldConstant`）
- 只改某个 dataflow 块 → DataflowBlockPass（如 `ConvertLayout`）

**为什么分这么细？** 范围越小，pass 越快、越安全、越能并行。
这是编译器"职责清晰"的设计哲学。

---

## 8. 扩展阅读 B：Sequential 的隐藏能力——依赖、配置、调试

`relax.transform.Sequential([...])` 不是简单循环。它有隐藏机制：

1. **依赖检查**：每个 pass 声明 `required`（前置 pass），
   Sequential 会检查顺序是否满足，不满足就报错提醒你。
2. **配置传递**：`PassContext(opt_level=3, config={...})` 全局配置，
   所有 pass 从它读参数（第 8 课 `pc->GetConfig`）。
3. **调试输出**：`PassContext(config={"relax.transform.print_all": True})`
   可以打印每个 pass 前后的 IR——这是调试 pass 的利器！
4. **跳过机制**：`opt_level` 低的 pass 在 `opt_level` 高的配置下
   可能被跳过（太弱的优化不值得跑）。

**实际调试经验**：觉得某个 pass 没生效？
打开 print_all，看它跑之前/之后的 IR，一对比就知道改了没、改对没。
这套"前后对比"就是编译器开发者的日常 Debug 方式。

---

## 9. 扩展阅读 C：Relax 的“三阶段”编译流程

toycc 是"图优化 → codegen"两步。真实 Relax 是三个阶段：

```
阶段一: 图优化     (Relax 层)
  各种 transform + legalize → 干净的、底层的 Relax
阶段二: 下降到 TIR (FuseTIR / call_tir_rewrite)
  融合后的 Relax 函数 → TIR PrimFunc
阶段三: TIR 优化 + codegen
  调度 → 目标代码 → 可执行模块
```

**第 9 课的通关清单里有一条"画出完整地图"**——现在补全它：

```
模型
 → 图优化(融合/布局/折叠/内存)     [本课 + 3/4/5/6]
 → Legalize(高层 op 降为 call_tir) [第8课 fold_constant 提到]
 → FuseTIR(Relax → TIR)           [阶段二]
 → TIR 调度(split/vectorize...)   [第11课]
 → 自动调度(meta_schedule)        [第13课]
 → codegen(LLVM/CUDA/C)           [第7课]
 → 运行时(VM 执行)
```

**能画出这张图，你对 TVM 的理解就已经超过大多数"教程读者"。**
剩下的就是第 10 课：把理解变成代码。

---

**导航**：⬅ [上一节](lesson08.md)（第 8 课 · 真实 TVM（上））　｜　[下一节](lesson10.md)（第 10 课 · 从看懂到上手）➡
