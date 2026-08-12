# 第 7 课：代码生成——把一张图变成真的能跑的代码

> 本课风格：代码驱动 + 手算下标 + 读一段生成的 C 代码。
> 对应文件：`toycc/codegen/cgen.py`
> 准备：跑 `python -m course.runner 7` 对照看。

---

## 1. 前面所有课都在准备，这一课是"出货"

前面几课我们一直在改图：融合、布局、折叠、内存规划。
但**图终究不是程序**。这一步，要把图翻译成真正的代码——这就是
**代码生成（codegen）**。

它要解决三件事：

1. **算子 → 循环**：`conv` 变成一个 6 层嵌套 for 循环
2. **张量 → 内存**：每个张量对应哪个缓冲区（用第 6 课的内存表）
3. **常量 → 静态数组**：权重直接烙进代码

我们的 toycc 做了一个真框架也在做的事：**一个图，多个后端**。
同一个"循环生成逻辑"，同时产出 C 和 Python 两种代码。

- `out.c` → 可以在装了 gcc 的机器上编译成原生程序
- `out.py` → 纯 Python 循环复刻同一套计算，用来**验证**（本机没 gcc）

这就像 TVM：一套 IR，可以出 CUDA 代码、x86 代码、TensorRT 调用……

> **原理深挖：为什么"一个图、多个后端"是编译器最基本的架构？**
>
> 如果你只为一个硬件写编译器，代码生成就是一锤子买卖。但现实是：
> **同一个模型要部署到无数种硬件**（手机、A100、你家自研 GPU）。
> 为每种硬件写一套完整编译器不现实。
>
> 所以架构是"**分叉**"的：前中端的优化（融合/布局/折叠）跟硬件无关，写一遍；
> **只有最后一步 codegen 取决于硬件**。于是管线长这样：
>
> ```
> 模型 → 图优化(通用) → [codegen]→ 硬件A
>                     → [codegen]→ 硬件B
>                     → [codegen]→ 你自研的GPU
> ```
>
> 这就是为什么第 24 课"给 TVM 加新后端"只讲 codegen 那一步——
> **你只写"从通用 IR 到你们芯片"这一段，前面所有优化都白捡**。
> 这也是"编译器=基础设施"的意义：一次投入，多处复用。
>
> 对应到 TVM：`target="cuda"` / `"llvm"` / `"my_chip"` 只是换了 codegen，
> 之前的 Relax/TIR 优化完全不变。你第 7 课在这里亲手验证了"同一逻辑
> 双后端"——它就是真实架构的微缩版。

---

## 2. 前置知识：多维下标怎么展平成"一维偏移"

这是理解 codegen 最重要、也最容易绕晕的部分，我们先花 10 分钟彻底搞懂。

任何张量在内存里都是一维数组。`x` 是 `(1,3,8,8)`，就是一排 192 个 float。
那 `x[n][c][h][w]` 对应第几个？公式是 **行主序**：

```
offset = n*3*8*8 + c*8*8 + h*8 + w
       = n*192   + c*64  + h*8 + w
```

**规律：第 k 维的下标，乘以它右边所有维度的乘积（步长）。**

```
维度:   n(1)  c(3)  h(8)  w(8)
步长:    192   64    8    1     ← 步长 = 右边维度乘积
```

验证一下：`x[0][2][5][6]`：

```
offset = 0*192 + 2*64 + 5*8 + 6 = 0 + 128 + 40 + 6 = 174
```

**这就是 `codegen/cgen.py` 里 `flat_idx` 在干的事**。看它怎么写的：

```python
def flat_idx(shape, idxs):
    expr = str(idxs[-1])                 # 最后一位下标
    stride = 1
    for k in range(len(shape) - 2, -1, -1):
        stride *= shape[k + 1]           # 步长 = 右边所有维度乘积
        expr = f"({idxs[k]} * {stride} + {expr})"
    return expr
```

用 `(1,3,8,8)` 和下标 `[n,c,h,w]` 走一遍：

```
expr = "w", stride = 1
k=2: stride *= shape[3]=8 → 8;  expr = (h*8 + w)
k=1: stride *= shape[2]=8 → 64; expr = (c*64 + (h*8+w))
k=0: stride *= shape[1]=3 → 192;expr = (n*192 + (c*64+(h*8+w)))
```

得到 `n*192 + c*64 + h*8 + w`——和我们手算一致。

> **为什么当初这里容易写错？** 反直觉点在于：`shape[k+1]` 用的是
> **当前位置右边**的维度，而不是当前位置本身。`c` 的步长 64 来自
> 它右边的 `8*8`，不是它自己那维的 3。我们最初写错过一次，数值验证立刻抓出来。

---

## 3. 逐行读 `cgen.py`：一个能同时写两种语言的工具

### 3.1 CodeBuilder：统一生成 C 和 Python 的"小抄"

```python
class CodeBuilder:
    def __init__(self, lang: str):
        self.lang = lang
        self.lines = []
        self._ind = ""

    def push_loop(self, var, rng):
        if self.lang == "c":
            self.lines.append(f"{self._ind}for ({var} = 0; {var} < {rng}; {var}++) {{")
        else:
            self.lines.append(f"{self._ind}for {var} in range({rng}):")
        self._ind += "    "

    def pop(self):
        self._ind = self._ind[:-4]
        if self.lang == "c":
            self.lines.append(f"{self._ind}}}")

    def assign(self, lhs, rhs):
        if self.lang == "c":
            self.lines.append(f"{self._ind}{lhs} = {rhs};")
        else:
            self.lines.append(f"{self._ind}{lhs} = {rhs}")
```

**设计核心**：循环和赋值在两种语言里结构完全一样，只是语法不同。
`CodeBuilder` 把这些语法差异封装起来，让后面的算子代码**只写一次**、
自动适配两种语言。这就是"多后端"的最小实现。

### 3.2 TensorInfo：张量 → 运行时名字

```python
class TensorInfo:
    def __init__(self, graph, allocs, shapes):
        self.info = {}
        # 常量优先: 折叠出的常量张量也是图输入, 但它是编译期已知的
        for name in graph.constants:
            self.info[name] = ("const", f"K_{name}")     # 静态数组
        for name in graph.inputs:
            self.info.setdefault(name, ("input", f"in_{name}"))  # 运行时输入
        for name, alloc in allocs.items():
            self.info.setdefault(name, ("buf", f"buf{alloc.buf}"))  # 缓冲区
```

每个张量被安排成三类名字之一：

| 种类 | 运行时名字 | 例子 |
|---|---|---|
| 常量 | `K_<名字>` | `K_conv1_lt_1` → `static const float` 数组 |
| 输入 | `in_<名字>` | `in_x` → 运行时喂入 |
| 缓冲区 | `buf<序号>` | `buf0` → 第 6 课分配的内存 |

**注意"常量优先"**：折叠后 `conv1_lt_1` 既是图输入（名字被引用）又是有值的
常量——必须按常量处理，否则会被当成运行时输入乱填数据。
这是我们在开发中踩过的坑，加了这行 `for constants` 才修对。

### 3.3 算子 → 循环：以卷积为例

`_emit_node` 里，用**同一套遍历逻辑**描述卷积，输出两种语言：

```python
b.comment("conv" + (" (nhwc)" if nhwc else ""))
for v, rng in [("n", N), ("oh", OH), ("ow", OW), ("oc", OC)]:
    b.push_loop(v, rng)                      # 输出位置的4层循环
b.stmt("acc = 0.0" if b.lang == "py" else "float acc = 0.0f")
korder = [("kh", KH), ("kw", KW), ("c", C)] if nhwc else [("c", C), ("kh", KH), ("kw", KW)]
for v, rng in korder:
    b.push_loop(v, rng)                      # 归约的3层循环
# 边界判断 + 累加
if b.lang == "c":
    b.raw(f"{b._ind}if ({ih} >= 0 && {ih} < {H} && ...) {{ acc += ...; }}")
else:
    b.raw(f"{b._ind}if 0 <= {ih} < {H} and 0 <= {iw} < {W}: acc += ...")
for _ in korder:
    b.pop()
...
b.assign(f"{out}[{flat_idx(out_shape, oi)}]", "acc")   # 写结果
```

对照真实生成的 C（`out.c` 里 conv1 部分）：

```c
for (n = 0; n < 1; n++) {
  for (oh = 0; oh < 8; oh++) {
    for (ow = 0; ow < 8; ow++) {
      for (oc = 0; oc < 4; oc++) {
        float acc = 0.0f;
        for (c = 0; c < 3; c++) {
          for (kh = 0; kh < 3; kh++) {
            for (kw = 0; kw < 3; kw++) {
              if ((oh*1-1+kh) >= 0 && (oh*1-1+kh) < 8 &&
                  (ow*1-1+kw) >= 0 && (ow*1-1+kw) < 8) {
                acc += buf0[...] * K_conv1_lt_1[...];
              }
            }
          }
        }
        acc += K_conv1_lt_2[...];     // bias
        acc = fmaxf(acc, 0.0f);       // relu (融合进来了!)
        buf1[...] = acc;
      }
    }
  }
}
```

**注意几个细节**：

- `acc` 累加器：卷积 = 逐元素相乘求和
- `if (边界)`：padding 处理——越界的位置跳过（相当于补 0）
- `acc += K_...`（bias）+ `fmaxf(acc, 0)`（relu）：**这就是融合后的效果**——
  一个核里做完 conv+bias+relu，不写中间结果
- 所有多维访问都展平成一维偏移（`flat_idx`）

---

## 4. 对照 `out.py`：同一套计算的 Python 版

生成的同时，我们产出一份 Python 镜像。它和 C 结构一模一样：

```python
for n in range(1):
    for oh in range(8):
        for ow in range(8):
            for oc in range(4):
                acc = 0.0
                for c in range(3):
                    for kh in range(3):
                        for kw in range(3):
                            if 0 <= (oh * 1 - 1 + kh) < 8 and 0 <= (ow * 1 - 1 + kw) < 8:
                                acc += buf0[(n * 192 + ((oh*1-1+kh) * 24 + ((ow*1-1+kw) * 3 + c)))] * K_conv1_lt_1[(oc * 27 + (kh * 9 + (kw * 3 + c)))]
                acc += K_conv1_lt_2[(0 * 4 + (0 * 4 + (0 * 4 + oc)))]
                acc = max(acc, 0.0)
                buf1[(n * 256 + (oh * 32 + (ow * 4 + oc)))] = acc
```

**为什么要有两份？** 没有 gcc 的机器上，Python 版能当场跑、当场验证
（`python -m course.runner 7` 会对比 `max|Δ|`）。这就是"一个 IR 多后端"
思想的教学版：C 是生产，Python 是验证。

---

## 5. 验证闭环：生成的代码算对了

`runner 7` 最后做的事：

```python
got = gen.run(f["x"]).reshape(out0["output"].shape)
print(f"生成的 python 后端执行结果与参考 max|Δ| = {np.max(np.abs(got - out0['output'])):.2e}")
```

跑出来的结果 `5.55e-17`——和参考执行器（第 2 课）完全一致。
**整条流水线闭环了**：原始图 → 4 个 pass → 生成的代码 → 结果和没优化时一样。

这行数字是全课程的"毕业证书"：改了一路的图，最后没算错。

---

## 6. 真实 TVM 对照：`run_codegen.cc`

TVM 的 codegen 走的是 **BYOC（Bring Your Own Codegen）**架构：

```cpp
ffi::Array<ffi::Module> InvokeCodegen(IRModule mod, ...) {
  // 1. 按 kCodegen 属性把函数分桶
  for (const auto& entry : mod->functions) {
    PostOrderVisit(entry.second, [&target_functions](Expr e) {
      if (auto target_opt = f->GetAttr<ffi::String>(attr::kCodegen)) {
        target_functions[target.value()].push_back(f);
      }
    });
  }
  // 2. 按 target 查注册的 codegen, 调用它
  for (const auto& [target, functions] : target_functions) {
    ffi::String codegen_name = "relax.ext." + target;   // 如 relax.ext.cutlass
    const auto codegen = tvm::ffi::Function::GetGlobal(codegen_name);
    ffi::Array<ffi::Module> compiled = (*codegen)(functions, options, ...);
  }
}
```

**对照**：

| 真实 TVM | toycc |
|---|---|
| 按 `kCodegen` 属性分桶 | `_emit_node` 按 `op_type` 分发 |
| `relax.ext.<target>` 查后端 | `CodeBuilder(lang)` 选 C 或 Python |
| 常量起唯一名字 | `K_<常量名>` |
| 融合函数 → `ExternFunc`（调外部库） | 融合节点 → 一段内联循环 |
| 末尾跑 DCE 清理 | 无（简化） |

真实 TVM 内置 `relax.ext.cutlass`、`relax.ext.tensorrt` 等大厂后端——
**"一个 IR，出哪个后端的活由 target 决定"**，和我们 `CodeBuilder("c")`
vs `CodeBuilder("py")` 是同一哲学。

---

## 7. 实验

```bash
python -m course.runner 7
```

再看一眼生成的 `out.c` 开头几行（常量数组 + 缓冲区声明），
和上面第 3.3 节的 C 片段对一对。

---

## 8. 课后答疑

**Q：为什么权重是 `static const float K_xxx[]`？**
A：权重编译期确定、永不变。烙成只读静态数组：零加载、可被编译器优化
（甚至当立即数）、不占运行时内存。

**Q：`flat_idx` 生成那么长的表达式，会不会很慢？**
A：那是**生成阶段**拼的字符串，无所谓。真正的 C 编译器会把
`n*192 + c*64 + h*8 + w` 化简成高效的地址计算。

**Q：为什么代码里 `acc` 的边界检查用 `if` 而不是显式补 0 的 padding？**
A：两种做法都行。用 `if` 跳过的代码更小；显式 pad 数组访问更快但多占内存。
真实编译器会根据目标选择，甚至用 `max`/`min` 钳制下标（clamping）。

**Q：TVM 生成的代码会比我们优化得多吗？**
A：会。TVM 有专门的 TIR 层（循环分块、向量化、tiling），
然后才到 codegen。我们的 toycc 直接跳到 codegen，跳过了 TIR 调度——
这是 toycc 和真框架最大的省略，也是你后续深入的方向。

---

## 9. 本课小结

- 下标展平：`offset = 各维下标 × 右边维度的乘积`
- `CodeBuilder` 封装两种语言语法 → 一个算子逻辑、两套输出
- 融合的成果在代码里可见：bias + relu 都进了同一个核
- Python 镜像 = 无 gcc 环境下的验证手段（多后端思想的数学版）
- `max|Δ| = 5.55e-17` = 全流程闭环成功
- TVM 有专门的 TIR 调度层，比我们的直出循环优化得多

**下一步**：第 8 课。你已经用 toycc 完整走了一遍编译器的骨架。
现在，把这份理解带到真实世界——我们打开 TVM 源码，看看它 1500 行的
`fuse_ops.cc` 到底是不是你想的那样。

---

## 10. 扩展阅读 A：完整代码生成流水线（TVM 是怎么一步步到机器码的）

toycc 直接从图生成 C。真实编译器（TVM）有一整条"下降"链：

```
Relax(图) 
  → TIR(循环, 带调度)          [第9课 FuseTIR / 第11课 调度]
  → 目标相关 TIR(带线程/向量化)  [调度后的 TIR]
  → 目标代码                     [CodeGenLLVM / CodeGenCUDA / CodeGenC]
  → 机器码 / 源码               [LLVM JIT / 编译器]
```

关键点：**codegen 不是一步到位的**。中间每一层都在"更接近硬件"：
- 图层：不知道线程是什么
- TIR：知道了循环和 buffer
- 调度后：知道了线程、向量宽度
- 最终：机器码

**toycc 的省略**：toycc 直接"图 → C"，跳过了 TIR 和调度层。
这是教学简化（第 11 课专门补了 TIR 调度），真实系统里
每一层都有大量 pass 在做准备。

---

## 11. 扩展阅读 B：代码生成器内部——符号表 + 表达式递归下降

我们 `cgen.py` 的 `_emit_node` 是"看图发话"。真实 codegen
（比如 LLVM IR 生成器）是**递归下降**式的：

```
CodeGenVisitor:
  VisitFor(循环节点)    → 生成 label + 跳转
  VisitBufferStore(写)  → 生成 store 指令
  VisitAdd(加法)        → 生成 add 指令
  ...每个 TIR 节点一个处理函数
```

它维护一张**符号表**：`变量名 → LLVM 寄存器`。遇到表达式就递归
生成子表达式，把结果登记进寄存器。

**这是编译器的"翻译器"设计模式**：把 IR 树遍历一遍，每个节点
翻译成目标语言的对应结构。toycc 用字符串拼，真实 codegen 用
目标 AST（LLVM IR / CUDA 源码）——**思想一模一样：遍历 → 翻译**。

---

## 12. 扩展阅读 C：ABI、内存布局、函数调用——生成代码的接缝

生成的 kernel 要能嵌入运行时，必须遵守"接缝"约定（ABI）：

1. **参数约定**：输入张量怎么传给 kernel？
   - 传指针 + 形状？还是打包成结构体？
2. **内存分配**：kernel 里临时内存谁分配？运行时还是 kernel 自己？
3. **返回值**：结果写到哪里？输出 buffer 谁提供？
4. **对齐**：float 数组 4 字节对齐（第 6 课深拓展 B）

toycc 的 `run(float* out)` 就是一个简化 ABI：输入全用全局变量、
输出传指针。真实运行时（`relax.vm`）有完整的调用约定。

**为什么值得知道**：看 TVM 生成的代码（`print(mod)` 或 codegen 输出）时，
你会看到 `T.allocate`、`T.call_extern`、packed func 等"接缝代码"——
它们就是 ABI 的体现。看懂这些，你就看懂了"编译产物怎么和运行时对接"。

---

## 13. 思考题（加深版）

1. 为什么 codegen 之前一定要有"调度"？（提示：循环顺序谁决定？）
2. `CodeGenVisitor` 的"遍历→翻译"和 toycc 的"看图发话"区别在哪？
3. 如果生成代码里忘了 `#include <math.h>` 会怎样？这算 ABI 问题吗？

> 答案：
> 1. 因为"怎么写循环"（分块/向量/线程）在 codegen 之前就该定好；
>    codegen 只是按定好的循环结构发指令。
> 2. toycc 是"遍历节点直接拼字符串"；真实 codegen 是"递归下降每个
>    IR 节点到目标 AST"，能处理任意复杂结构。
> 3. 编译报错/链接失败——算"接缝"问题的一种：生成的代码依赖外部符号
>    和声明，约定没对上就崩。

---

**导航**：⬅ [上一节](lesson06.md)（第 6 课 · 内存规划）　｜　[下一节](lesson08.md)（第 8 课 · 真实 TVM（上））➡
