# 第 17 课：模型怎么进编译器——前端、ONNX、下降、动态形状、运行时

> 本课风格：概念 + 真实示例 + "从模型到跑起来"的完整链条。
> 目的：搞懂"模型文件 → 可执行产物 → 运行时执行"这整条链，
> 以及为什么动态形状/控制流是 AI 编译器当前的最大战场。
> 前置：第 1 课（IR）、第 14 课（IR 家族）。

---

## 1. 一个完整的"模型进编译器"流程

```
PyTorch 模型 / ONNX 文件
   │ 前端(Frontend)        ← 翻译成图 IR
   ▼
Relax 图
   │ 图优化 pass            ← 第 3/4/5/6/12 课
   ▼
优化后的图
   │ legalize(下降成底层算子)
   ▼
call_tir + PrimFunc        ← 第 8 课
   │ FuseTIR / lower
   ▼
TIR → codegen              ← 第 11/7 课
   ▼
可执行产物(lib 或源码)
   │ 运行时(VM/Runtime)    ← 本课第 5 节
   ▼
真正跑出结果
```

中间的图优化到 codegen 你已经熟悉，本课补**两头**：最前端（模型怎么进来）和
最末端（编译产物怎么执行）。

---

## 2. 前端：PyTorch / ONNX 模型怎么变成图

### 2.1 前端要解决的三件事

1. **翻译**：把框架的算子变成编译器的算子（`conv2d → nn.conv2d`）
2. **形状推导**：算出所有中间张量的形状（toycc 的 `infer_shapes`）
3. **整理**：把权重变成常量、去掉框架特有的包装（`nn.Module` 的 hook 等）

> **原理深挖：为什么"最枯燥"的前端，反而是编译器团队的隐形护城河？**
>
> 前端工作看起来是体力活——一个算子一个算子地翻译。但它藏着三笔真本事：
>
> 1. **语义的精确性**：`conv2d` 在 PyTorch 里可能有 `groups`、`padding_mode`、
>    `dilation` 一堆参数，翻译错一个，模型"能跑但算错"——这是最危险的 bug
>    （不是崩溃，是悄悄错）。**前端的价值 = 语义翻译 100% 正确**。
> 2. **覆盖度决定你的编译器"能接什么活"**：真实团队的模型五花八门，
>    算子支持不全，编译就失败。前端每支持一个算子，就是给编译器
>    **解锁一类模型**。这是"往里加算子"的工作，也是新人最容易上手
>    被信任的任务（权限低、影响清晰、可测试）。
> 3. **形状推导 = 整个编译器的地基**：输不出形状，内存规划、
>    布局全卡壳。所以前端做不好，中端没人敢用它的输出。
>
> 给你的建议：**入职第一周的高价值任务，就是"支持一个新算子"**——
> 这是前端工作，也是最容易从"看懂"迈向"贡献"的入口。

### 2.2 ONNX 格式简介（你要能看懂）

ONNX（Open Neural Network Exchange）是**模型交换的标准格式**。
一个 `.onnx` 文件核心是"计算图"：

```text
graph "main"
  input:   x: (1,3,8,8) f32
  initializer:  conv1_w: (4,3,3,3) f32     ← 权重(initializer = 常量)
  nodes:
    node1: conv  (x, conv1_w) → conv1
    node2: relu  (conv1)      → r1
    ...
  output: out: (1,16) f32
  opset: 13
```

**对照 toycc**：`input` = `add_input`，`initializer` = 权重（常量），
`node` = `add_op`，`output` = `mark_output`。
**你学过的图 IR 就是 ONNX 的"同一类东西"**，只是 ONNX 有标准化格式
（protobuf）和 `opset`（算子版本号）。

### 2.3 前端导入后必须处理的"脏活"

导入远不是"拷贝算子"那么简单：

| 问题 | 怎么办 |
|---|---|
| 不支持/未知算子 | 报错 or 用"复合算子"模拟 |
| 动态形状 | 保留符号形状，或"特化"成固定形状 |
| 框架特有的算子（如 `nn.Dropout` 推理模式） | 剪掉或替换 |
| 冗余结构（`Identity`、无用的 reshape） | 简化 pass 清理 |
| 权重格式 | 转成编译器自己的布局 |

> 这就是为什么"导入模型"会有那么多坑、为什么有专门的
> `tvm.relax.frontend.onnx` / `torch` 模块。**前端是编译器里最枯燥
> 但也最不能出错的活**。

---

## 3. Legalize（合法化/下降）：从"高层语义"到"可编译算子"

### 3.1 为什么需要下降

高层算子（如 `LayerNorm`、`Softmax`、`ConvTranspose`）在编译器里
往往没有"原生实现"。要执行它们，必须**拆成底层能实现的组合**：

```
LayerNorm(x, gamma, beta)
  →  mean = reduce_mean(x)
  →  var  = reduce_mean((x-mean)^2)
  →  y = (x-mean)/sqrt(var+eps) * gamma + beta
```

这个过程叫 **legalize（合法化/下降）**。TVM 的 `legalize_ops.cc` 里就有 `legalize_map`——它就是"高层算子 → 底层 call_tir"的翻译表。

> **手算：证明 LayerNorm 下降公式"确实是它"**
>
> LayerNorm 的定义（对每个向量 x 归一化）：
> ```
> y_i = (x_i - mean(x)) / sqrt(var(x) + eps) * gamma_i + beta_i
> ```
> 下降后发现它其实是**五个底层算子的组合**：
> ```
> m  = reduce_mean(x)              # ① 均值
> d  = x - m                       # ② 去均值
> v  = reduce_mean(d * d)          # ③ 方差
> y1 = d / sqrt(v + eps)           # ④ 归一化(除标准差)
> y  = y1 * gamma + beta           # ⑤ 仿射(scale+shift)
> ```
>
> 逐行核对：
> - ③ 的 `v = mean((x-m)²)` 正是方差的定义（这里用总体方差，没除 N-1）
> - ④ 把 x 变成均值为 0、方差为 1 的分布：任何 y1 的均值 = 0、方差 = 1
>   （手算验证：`mean(y1) = mean(d)/sqrt(v) = 0` ✓；`var(y1) = var(d)/v = 1` ✓）
> - ⑤ 通过 gamma/beta 把"标准分布"映射回想要的范围
>
> **两个工程细节藏在里面**：
> 1. `eps`（如 1e-5）不是数学需要，是**数值安全**需要——防止某通道
>    方差恰好为 0（全等值输入）时除零。
> 2. 为什么编译器**必须**在高层做这个下降，而不等到底层？
>    因为"reduce_mean + 逐元素"这个组合在底层的循环形态完全取决于
>    形状——**高层降一次，底层优化（调度）才能针对正确形态**。
>    这就是"下降时机讲究"的机制层面的原因。

### 3.2 常见下降例子

| 高层算子 | 下降成 |
|---|---|
| `ConvTranspose` | 转置 + Conv（或 im2col 的转置） |
| `LayerNorm` | 一组 mean/var/逐元素算子 |
| `Softmax` | exp + sum + div |
| `Gemm` | `matmul` + `add` |
| `Attention` | matmul×3 + softmax + matmul（不优化版） |

### 3.3 下降的"时机"很讲究

- 图优化 pass 跑在**高层**（还能识别"这是 LayerNorm"）
- 下降到低层后，很多高层优化就看不出来了
- 所以顺序通常是：**高层优化 → legalize → 低层优化**

> 真实 TVM 的 pipeline 也是这个顺序：`LegalizeOps` 在 `FuseOps`
> 之前——先合法化，再在合法化后的图上做融合（Relax 的 FuseOps
> 跑在 call_tir 图上，FuseTIR 再在 TIR 层融合）。

---

## 4. 动态形状与控制流——AI 编译器的"当代主战场"

### 4.1 静态 vs 动态形状

```
静态:  x: (1, 3, 224, 224)      ← 所有形状编译期已知, 好优化
动态:  x: (N, 3, 224, 224)      ← N 运行时才知道
       甚至: x: (N, S, S)        ← 序列长度也变(LLM 的 token 数)
```

**为什么动态形状难**：
1. **内存规划失效**：不知道张量多大，无法提前分配固定缓冲
2. **调度难定**：分块大小取决于尺寸，动态形状无法静态选择
3. **多分支**：不同尺寸可能走不同优化路径

**编译器怎么应对**：
- **形状特化**：编译时"绑定"一批常见形状，各出一份专用 kernel
- **动态调度**：运行时查形状，选对应 kernel（分派表）
- **Relax 的设计**：原生支持符号形状（`T.Var`），让优化"部分做得了"

### 4.2 控制流：模型里也有 if/while

不是所有模型都是直线。语言模型、动态网络有：

```text
if (is_training) { ... } else { ... }      ← 分支
while (cur_pos < seq_len) { ... }          ← 循环(自回归生成)
```

图 IR 要表达控制流，就超越了"纯 DAG"。Relax 用 `If` / `While` 节点，
LLVM 用 basic block + branch，MLIR 用 region。**控制流使 IR 从 DAG
变成 CFG（控制流图）**——支配分析（dominance analysis）就是为 CFG 服务的。

> 自回归大模型推理 = 一个 while 循环，每步生成一个 token。
> 这就是为什么"大模型推理优化"很大程度是"把循环每一步做到极致"。

---

## 5. 运行时（Runtime / VM）：编译产物怎么"跑起来"

### 5.1 编译产物是什么

编译完你得到一个**函数库**（或源码）：每个融合核是一个可调用函数。
但没有"main"——谁调用它们、按什么顺序、怎么管理内存？

**运行时（runtime）** 负责：
1. **调度**：按图的顺序调用每个核
2. **内存管理**：给中间张量分配/释放缓冲
3. **设备管理**：数据搬入/搬出 GPU，同步
4. **参数绑定**：把训练好的权重装进常量区

### 5.2 TVM 的两种运行时

| | Virtual Machine (VM) | Graph Executor |
|---|---|---|
| 适合 | Relax（新） | Relay（旧） |
| 执行方式 | 解释"字节码"（指令序列） | 解释图 |
| 支持动态形状 | 是 | 弱 |
| 灵活性 | 高 | 低 |

**VM 概念**：把优化后的图编译成一条**指令序列**（如 `call_tir(fn_3)`、
`alloc_tensor`、`kill_tensor`），运行时像解释器一样逐条执行。
这有点像把图"编译成字节码再解释"——兼顾灵活和快。

> **更深一层：看一段"编译后的图"长什么样**
>
> toycc 的示例模型（融合后 4 个核）被 Relax 编译成 VM 后，
> 指令流大致是：
>
> ```
> alloc_tensor(buf0, [1,8,8,4])     # 按第6课内存规划,复用同一块缓冲
> alloc_tensor(buf1, [1,4,4,8])
> call_tir(fn_conv1, [x, conv1_w, conv1_b], buf0)   # 核1: conv+bias+relu
> call_tir(fn_conv2, [buf0, conv2_w, conv2_b], buf1) # 核2
> kill_tensor(buf0)                  # buf0 的消费者跑完了, 可以回收
> call_tir(fn_mm, [buf1, w_mm, bias3], buf2)
> kill_tensor(buf1)
> ret(buf2)
> ```
>
> 这段指令揭示了三件"运行时在替编译器兑现承诺"的事：
>
> 1. **内存规划在这里落地**：`alloc/kill` 成对出现——
>    缓冲的复用策略被编进了指令流，而不是运行时临时想。
> 2. **融合的核是黑盒**：`call_tir(fn_conv1, ...)` 一个函数
>    一次调用——中间 tensors 根本没出现（它们没逃出核）。
> 3. **指令是"解释执行"的**：所以 VM 天然支持动态形状/控制流——
>    遇到 `If`/`While` 变成跳转指令，形状可以在运行时才知道。
>
> 对比 toycc：`ReferenceEvaluator` 就是"解释执行"的最小版——
> 它按拓扑序逐算子调用。VM 只是把这一步**预先编译成稳定指令流**，
> 并在每步带上内存/设备信息。**从 toycc 到 VM 的差距 = 运行时工程化**。

### 5.3 部署到真实设备

```
编译 + 调优(meta_schedule) → 导出(export_library) → 目标设备加载 → 推理
```

- `tvm.relax.build`：编译成可执行模块
- `export_library("model.so")`：导出成库文件
- 目标设备：`RPC`（远程调用，把编译产物发到真实设备跑）
- 数据格式：NDArray / DLTensor（跨框架的内存描述）

---

## 6. 概念对应表

| 真实 TVM | 概念 | toycc 对应 |
|---|---|---|
| `relax.frontend.onnx.from_onnx` | 前端导入 | `build_model`（手写图） |
| `relax.transform.LegalizeOps` | 算子下降 | （未实现） |
| `T.Var` 符号形状 | 动态形状 | `infer_shapes`（静态） |
| `relax.If` / `While` | 控制流 | `Graph`（纯 DAG） |
| `relax.vm.VirtualMachine` | 运行时执行 | `ReferenceEvaluator` |
| `relax.build` + `export_library` | 部署 | `emit_c` / `emit_python` |
| `tvm.rpc` | 远程设备部署 | 无 |

---

## 7. 课后答疑

**Q：为什么 ONNX 有 `opset` 这个东西？**
A：算子语义会演进（比如 conv 加了参数）。`opset` = "算子集版本"，
告诉编译器"这个模型用的是哪一版语义"，避免误判。

**Q：法律化（legalize）和下降（lower）是一回事吗？**
A：中文常混用。细分：legalize 是"把合法但不可执行的算子转成可执行的"；
lower 是"整体移到更低层 IR"。实践中常指同一件事。

**Q：为什么动态形状这么难，还非要支持？**
A：因为现实（NLP/视频）的 batch/序列长度就是动态的。为了性能可以
"特化成几个固定形状"，但完全拒绝动态形状 = 拒绝大多数真实场景。
Relax 就是为此而生。

**Q：运行时是自己写的吗？**
A：编译器自带运行时（TVM 的 `src/runtime/`），编译产物要链接它。
你写的 pass 不碰运行时，但要知道"产物最终靠它执行"。

---

## 8. 本课小结

- 完整链路：模型 → 前端 → 图优化 → **legalize** → TIR → codegen → **运行时**
- ONNX = 标准化图格式（input/initializer/node/output/opset）
- Legalize = 高层算子拆成可实现的组合（LayerNorm → 一堆算子）
- **动态形状 + 控制流 = 现代 AI 编译的主战场**（LLM 推理的根基）
- 运行时 = 调度核 + 管内存 + 管设备；Relax 用 VirtualMachine 解释指令
- 部署 = build → export → RPC 到目标设备

**下一步**：第 18 课——量化与数值精度：把模型从 fp32 换到 int8/fp16
要跨过的精度坎。之后第 19 课会讲性能的度量（roofline/benchmark）
以及卷积/矩阵乘的经典加速技术（im2col / Winograd / GEMM 微内核）。

---

## 本课检查点

完成以下四项才算通过本课（每题都能用 `python -m course.runner 17` 验证）：

1. 写出 ONNX 模型的四个要素（input / initializer / node / output），并说清"权重是 initializer 而不是 input"对常量折叠意味着什么；
2. 把 LayerNorm 拆成原始算子链（mean/sub/pow/mean/add/sqrt/div/mul/add），写出其中每个算子的输入；
3. 解释符号形状（`(1, C, H, W)`）与静态形状在编译上的差别：哪些优化能做、哪些不能做；
4. 用一句话回答"LLM 推理为什么把图从 DAG 变成 CFG"（while 循环 = 每步生成一个 token 的循环）。

---

## 扩展阅读：前端的三个常见坑

### A. 为什么 ONNX 转换经常"差一点点"？

ONNX 是"模型交换标准"，但**每个框架对同一个算子的定义有微妙差别**。
比如 PyTorch 的 `pad` 和 ONNX 的 `Pad`，默认参数、边界处理可能不同。
所以前端转换**永远要跑数值验证**——不能假设"转过去就对了"。

### B. legalize 的"度"怎么把握？

下降（legalize）把高层算子拆成低层组合。但**拆多细是个权衡**：
拆太细，后续优化看不出"这原本是 LayerNorm"（失去融合机会）；
拆太粗，后端不认识。**工程上通常"降到后端认识的最低层"为止**——
这就是为什么 TVM 的 legalize 是可配置的。

### C. 动态形状为什么这么难？

静态形状下，编译器知道每个张量多大，可以算内存、选调度。
**动态形状把这些全变成"运行时才知道"**——内存分配要留 buffer、
调度参数要按范围 tune。这就是为什么 LLM（序列长度动态）是
编译器最难啃的场景之一。

---

**导航**：⬅ [上一节](lesson16.md)（第 16 课 · 工程开发流程）　｜　[下一节](lesson18.md)（第 18 课 · 量化与数值精度）➡
