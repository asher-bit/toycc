# 第 14 课：真实的 IR 家族——LLVM IR、MLIR、TIR、PTX

> 本课风格：**每种 IR 一个真实示例 + 逐行讲解 + 定位它在编译流程里的位置**。
> 目的：你在 GitHub/PR/论文里看到的 `.ll`、`.mlir`、TIR、`.ptx` 都是什么，
> 和 toycc 的关系是什么，一次性建立全局认知。
>
> 说明：以下 LLVM IR / MLIR / PTX 示例取自官方文档（LLVM LangRef、MLIR Toy 教程、
> LLVM NVPTX 指南），保证准确。

---

## 0. 一张总图：这些 IR 在编译流程里各占什么位置

```
 高抽象 ────────────────────────────────────────────── 低抽象

 模型(PyTorch/ONNX)          ← 第 1 课 toycc 的计算图
      │ 前端
     Relax (TVM 图 IR)       ← 融合/布局/折叠 pass 干活的层
      │ FuseTIR / lower
     TIR (TVM 循环 IR)       ← 调度(分块/向量化)干活的层 ← 第 11 课
      │ codegen
     MLIR / LLVM IR          ← 通用编译器 IR(TVM 可以出口到 LLVM)
      │ LLVM 优化 + 后端
     NVPTX / PTX             ← GPU 目标汇编
      │ ptxas
     SASS(真实机器码)

 这个"金字塔"每往下一层, 信息越接近硬件, 可移植性越低。
 上层 IR 能跨硬件, 下层 IR 绑定硬件。
```

**核心认知**：现代编译器大多是**分层 IR** 架构。每一层解决特定问题，
各层之间用 pass 下降。你在 toycc 里只做了"图 + 直接 codegen"两层；
真实世界有 4~6 层。

> **原理深挖：为什么"分层"而不是"一层管到底"？**
>
> 想像一下只有一层 IR——为了表达"模型级"的算子（conv），它得既存图结构、
> 又存循环、又存寄存器。这层 IR 会**巨大而臃肿**：做图优化的人要盯着
> 几百个循环细节，做后端的人要被一堆高层语义干扰。
>
> 分层的本质是**关注点分离**：
> - 高层 IR **只问"算什么"**，扔掉布局/循环/内存——图优化（融合/布局）
>   在干净的语义上做，不被执行细节干扰。
> - 低层 IR **只问"怎么算"**，不再有"卷积"这种算子——后端在干净的
>   指令/循环上做，不被高层语义干扰。
>
> 每层的中间产物，就是一次"抽象边界"。**层与层之间的下降（lowering）
> 就是那道边界上的转换**。这也是第 0 课那张"金字塔"的理论基础——
> 每个新概念（动态形状、分布式）都能落在"离它最近的某层"处理，
> 而不是污染所有层。
>
> **对你的意义**：自研芯片时，"你加一层还是两层 IR"直接决定系统复杂度。
> 大部分自研芯片团队的错误，是**想一层直接生成机器码**——结果图优化做不了、
> 后端又太杂。参考本课的分层，至少留"图层 + 循环层"两层。

---

## 1. LLVM IR——"万能汇编"（通用编译器的中间层）

### 1.1 它是什么

LLVM IR 是 LLVM 项目的中间表示。设计目标是**既能表达几乎所有高级语言，
又足够底层好优化**。Clang（C/C++）、Rust、Swift 的前端都产它，
后端（x86/ARM/RISC-V/GPU...）都吃它。

**特点**：
- SSA（静态单赋值）：每个变量只赋值一次
- 强类型：所有操作都带类型（`i32`、`float`、`ptr`）
- 无限"虚拟寄存器"：`%name` 是虚拟寄存器，后端再分配成真寄存器

### 1.2 真实示例（来自 LLVM LangRef）

把变量 `%X` 乘 8，三种写法（教学例子）：

```llvm
; 简单写法
%result = mul i32 %X, 8

; 强度消减后(乘8 = 左移3)
%result = shl i32 %X, 3

; 最笨写法(用三次加法)
%0 = add i32 %X, %X           ; %0 = 2*X
%1 = add i32 %0, %0           ; %1 = 4*X
%result = add i32 %1, %1      ; %result = 8*X
```

**逐行讲**：
- `;` 是注释
- `%X`、`%result` 是**虚拟寄存器**（SSA 值）
- `mul i32 %X, 8` = "把 `%X` 和 8 做 i32 乘法"
- 注意第三个例子：`%0`、`%1` 是**无名临时值**，LLVM 自动编号——
  这正体现了 SSA"每个值定义一次、之后只被引用"

一个完整的小模块（hello world）：

```llvm
; 声明字符串常量
@.str = private unnamed_addr constant [13 x i8] c"hello world\0A\00"

; 声明外部函数 puts
declare i32 @puts(ptr captures(none)) nounwind

; 定义 main 函数
define i32 @main() {
  call i32 @puts(ptr @.str)    ; 调用 puts
  ret i32 0
}
```

**逐行讲**：
- `@` 开头的全局符号，`%` 开头的局部值
- `[13 x i8]` = 13 字节的数组类型
- `define` = 定义函数，`declare` = 只声明（链接时再找）
- 注意最近的 LLVM 用**不透明指针** `ptr`（旧版本写 `i8*`）

### 1.3 LLVM IR 对应的 toycc 概念

| LLVM IR | toycc |
|---|---|
| `%result = mul i32 %X, 8` | `Node("mul", inputs=[x, w], attrs={dtype:i32})` |
| `define i32 @main() {...}` | `Graph` + `mark_output` |
| 虚拟寄存器（SSA） | 张量名 = 节点名（同是 SSA 风格） |
| `verify` pass 检查合法性 | `Node.__post_init__` 校验算子存在 |

**关键区别**：LLVM IR 是**低层**的（标量指令、显式类型、显式控制流）；
toycc 图是**高层**的（张量算子、无控制流）。LLVM IR 里没有"卷积"这种算子，
它已经被下降成循环+标量指令了。

---

## 2. MLIR——"可扩展的多层 IR"（能同时容纳高层和低层）

### 2.1 它是什么

LLVM IR 的缺陷：**太底层**。想把"卷积+融合+布局"这些**高层结构**直接表达、
直接优化，LLVM IR 做不到（你得先把它砸成循环）。MLIR 的设计就是
**让每一层抽象都有自己的"方言（dialect）"**。

核心概念 **Dialect（方言）**：一个命名空间 + 一组算子。
```
toy.transpose   ← toy 方言的转置算子
arith.addf      ← 算术方言的浮点加法
affine.for      ← 仿射方言的循环
memref.load     ← 内存方言的加载
```

高层方言优化完，再一层层**下降（lowering）**到低层方言，最后到 LLVM IR。

### 2.2 真实示例（来自 MLIR Toy 教程）

**第一个：通用格式的单个操作**（官方逐字段拆解过的）：

```mlir
%t_tensor = "toy.transpose"(%tensor) {inplace = true} : (tensor<2x3xf64>) -> tensor<3x2xf64>
```

**逐字段讲**（每个字段都是 MLIR 的"通用操作语法"）：

| 片段 | 含义 |
|---|---|
| `%t_tensor` | 结果的名字（SSA 值） |
| `"toy.transpose"` | 操作名 = `方言名.算子名` |
| `(%tensor)` | 输入操作数列表 |
| `{inplace = true}` | 属性字典（**总是常量**，如 inplace 标志） |
| `(tensor<2x3xf64>) -> tensor<3x2xf64>` | 输入类型 → 输出类型 |
| `loc(...)` | 源位置（MLIR **强制**要求，便于调试） |

**关键认知**：MLIR 里"操作"是**可扩展的统一单元**。无论多高级的算子
还是多低级的指令，都是同一个 `Operation` 结构——这就是它能容纳多层的秘密。

**第二个：完整函数（自定义格式，可读版）**：

```mlir
module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
    %2 = toy.mul %0, %1 : tensor<*xf64>
    toy.return %2 : tensor<*xf64>
  }
}
```

**逐行讲**：
- `module {...}`：MLIR 的顶层容器（类似 LLVM Module / 我们的 Graph）
- `toy.func @名字(参数类型) -> 返回类型 {...}`：一个函数
- `tensor<*xf64>`：**动态形状**张量（`*` = 任意维度）——MLIR 原生支持
- `%2 = toy.mul %0, %1`：两个张量相乘

**注意**：`tensor<*xf64>` 用 `*` 表示动态形状——这就是第 1 课深拓展讲的
"符号形状"。MLIR 把它当成**一等公民**。

### 2.3 MLIR 和 toycc / TVM 的关系

| MLIR | toycc | TVM |
|---|---|---|
| `module` | `Graph` | `IRModule` |
| 方言（dialect） | 算子注册表 `OPS` | `Op` 注册表 |
| 操作（operation） | `Node` | `Expr`/`Call` |
| 下降（lowering） | 无（直接 codegen） | `legalize`/`FuseTIR` |
| 可扩展算子 | 需改注册表 | 需改注册表 |

**一句话**：TVM 的 TIR 的设计理念和 MLIR 同源（TVM 作者之一陈天奇
也是 MLIR 的设计参与者）。学会一个，另一个的架构一眼就懂。

---

## 3. TIR——TVM 的循环 IR（你已经在第 11 课学过的）

### 3.1 它是什么

TIR 描述"**怎么算**"：循环、buffer、线程、内存访问。它是
TVM 调度（第 11 课）和大部分后端工作的战场。

### 3.2 真实示例（TVMScript，TVM 的可读 TIR 语言）

TVMScript 写一个 matmul PrimFunc：

```python
@T.prim_func
def matmul(A: T.Buffer((M, K), "float32"),
           B: T.Buffer((K, N), "float32"),
           C: T.Buffer((M, N), "float32")):
    for i, j, k in T.grid(M, N, K):      # 三重循环
        with T.block("C"):               # 一个计算块
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])  # S=串行 S=串行 R=归约
            C[vi, vj] += A[vi, vk] * B[vk, vj]
```

解析后的文本形式大致长这样：

```
@main = tir.prim_func(A: Buffer[(M, K), "float32"], B: ..., C: ...) -> () {
  for (i: int32, 0, M) {
    for (j: int32, 0, N) {
      for (k: int32, 0, K) {
        C[i, j] = C[i, j] + (A[i, k] * B[k, j])
      }
    }
  }
}
```

**逐行讲**：
- `T.Buffer((M,K), "float32")`：声明一块内存（形状 + 类型）
- `T.grid(M, N, K)`：三重循环的糖衣
- `T.block("C")`：**计算块**——TIR 的原子计算单元（调度操作的对象）
- `T.axis.remap("SSR", ...)`：告诉编译器哪个轴是归约轴（R），
  归约轴能否并行/重排就靠这个判断

**对应关系**：
| TIR | toycc |
|---|---|
| `T.Buffer` | 内存规划里的缓冲区 |
| `T.grid`/`for` | 第 11 课 `LoopNest` 的循环 |
| `T.block` | 一个融合核的计算单元 |
| 调度（split/vectorize...） | `toycc/schedule.py` |

> TVM 的 `relax.build` 全流程就是：Relax 图 → `FuseTIR` → TIR →
> 调度 → codegen。你已经在第 9、11 课分别见过两端了。

---

## 4. PTX / NVPTX——GPU 的"汇编"

### 4.1 它是什么

PTX（Parallel Thread Execution）是 NVIDIA GPU 的**虚拟汇编**：
CUDA/LLVM 编译到 PTX，再由驱动（ptxas）转成各代 GPU 的真机器码（SASS）。
NVPTX 是 LLVM 里"从 LLVM IR 生成 PTX"的后端。

### 4.2 真实示例（来自 LLVM NVPTX 指南）

**第一步：LLVM IR 层面——标记 kernel 和设备函数**：

```llvm
; 设备函数: 只能被 GPU 代码调用
define float @my_fmad(float %x, float %y, float %z) {
  %mul = fmul float %x, %y
  %add = fadd float %mul, %z
  ret float %add
}

; kernel 函数: 可被 CPU(host) 调用, 用 ptx_kernel 标记
define ptx_kernel void @my_kernel(ptr %ptr) {
  %val = load float, ptr %ptr
  %ret = call float @my_fmad(float %val, float %val, float %val)
  store float %ret, ptr %ptr
  ret void
}
```

**逐行讲**：
- `ptx_kernel` 调用约定 = "这是 GPU kernel，host 能调用"
- 没有 `ptx_kernel` 的就是设备函数（只有 GPU 内部能调）
- `fmul`/`fadd` 是浮点乘/加；LLVM 会把它合并成 FMA 指令（`fma`）
  如果硬件支持——这就是"融合"在指令层的体现！

**第二步：地址空间——GPU 内存的分级**（NVPTX 指南的表格）：

| Address Space | 内存空间 | 含义 |
|---|---|---|
| 0 | Generic | 通用 |
| 1 | Global | 全局内存（host 可见，最慢） |
| 3 | Shared | 共享内存（块内线程共享，快） |
| 4 | Constant | 常量内存 |
| 5 | Local | 本地（线程私有） |

在 LLVM IR 里这样声明一块全局内存数组：

```llvm
@g = internal addrspace(1) global [4 x i32] [ i32 0, i32 1, i32 2, i32 3 ]
```

**这就是第 4 课"布局/内存分级"在 GPU 上的正式表达**：
不同地址空间访问速度差一个数量级，编译器必须决定数据放哪。

**第三步：线程坐标**（NVPTX 指南的映射表）：

| CUDA 内建 | PTX 特殊寄存器 | 含义 |
|---|---|---|
| `threadIdx` | `%tid.x` | 线程在块内的编号 |
| `blockIdx` | `%ctaid.x` | 块在网格中的编号 |
| `blockDim` | `%ntid.x` | 每块多少线程 |
| `gridDim` | `%nctaid.x` | 网格多少块 |

LLVM IR 里读线程号：

```llvm
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
```

### 4.3 一段真实的 PTX 长什么样

NVPTX 后端把这些 LLVM IR 转成 PTX，大致形态：

```ptx
// PTX 汇编片段(示意)
.reg .b32 %r<4>;          // 声明 4 个 32 位寄存器
.reg .f32 %f<2>;
ld.global.f32 %f1, [gbl]; // 从全局内存读
fma.rn.f32 %f2, %f1, %f1, %f1;  // 乘加融合指令
st.global.f32 [gbl], %f2; // 写回全局内存
```

**逐行讲**：
- `.reg` 声明**真寄存器**（这里没有"虚拟寄存器"了，是真实分配的）
- `ld.global` / `st.global`：带**地址空间前缀**的访存指令
- `fma.rn.f32`：**FMA 指令**——一条指令做 `a*b+c`，这就是
  LLVM 把 `fmul+fadd` 融合成的结果（第 3 课"融合"在指令层的版本！）

### 4.4 对应关系

| PTX/NVPTX | toycc |
|---|---|
| 地址空间（global/shared/local） | 第 4/6 课的布局与内存分级 |
| 真寄存器 | 第 3 课深拓展"寄存器压力" |
| 线程坐标（tid/ctaid） | 第 11 课 `bind` 调度 |
| `fma` 指令融合 | 第 3 课算子融合的指令层版本 |
| PTX → SASS | 第 7 课 codegen → 机器码 |

---

## 5. 全家福对比表（本课核心产出）

| IR | 抽象层级 | 谁产出 | 谁消费 | 关键特性 | toycc 对应 |
|---|---|---|---|---|---|
| **Relax** | 高（张量图） | 前端 | 图优化 pass | 动态形状、分布式 | `Graph` |
| **TIR** | 中（循环） | FuseTIR | 调度 + codegen | 计算块、buffer | `LoopNest` |
| **MLIR** | 中高（可扩展多层） | 各前端 | 各类 pass + 下降 | 方言、可扩展 | `Graph`+`OPS` |
| **LLVM IR** | 低（标量指令） | Clang 等 | LLVM pass + 后端 | SSA、强类型 | Node 的标量版 |
| **PTX/NVPTX** | 很低（GPU 汇编） | LLVM 后端 | ptxas → SASS | 地址空间、线程模型 | 生成的 C 的 GPU 版 |

**记忆法**：每一层 IR = 一层"信息增量"。高层只管"算什么"，
越往下越补上"怎么在具体硬件上算"。toycc 用 2 层（图 + C），
真实 GPU 编译用 5~6 层。

---

## 6. 实验与延伸阅读

```bash
python -m course.runner 14
```

**如果你想亲手看这些 IR**（需要装工具）：
- LLVM IR：`echo 'int main(){return 0;}' | clang -S -emit-llvm -x c - -o -`
- MLIR：装 MLIR 后 `mlir-opt` 跑 `.mlir` 文件
- TIR：跑第 9 课 `tvm_demo.py`，打印优化后的 `IRModule`
- PTX：`nvcc -ptx kernel.cu` 或 TVM `target="nvidia"` build 后看

> 不装工具也能学的建议：去 GitHub 搜 `*.mlir`、`*.ll`、`*.ptx`，
> 看真实项目里的 IR 文件长什么样。看得越多，越熟。

---

## 7. 课后答疑

**Q：为什么编译器要用这么多层 IR？一层不行吗？**
A：一层"能表达一切"的 IR 要么太高（表达不了硬件细节）、要么太低
（高层结构被砸碎了没法优化）。分层让每层在"适合自己的抽象"上干活，
层间用 pass 下降。这是现代编译器的主流设计（LLVM 也是一层层降）。

**Q：MLIR 和 TVM 是竞争关系吗？**
A：部分重叠，但定位不同：MLIR 是通用"多层 IR 基础设施"，TVM 是
专注 AI 的编译器（它的 TIR 层和 MLIR 理念同源）。业界也常把
MLIR 用于 AI 编译（比如 IREE、XLA 演进方向）。

**Q：为什么 LLVM IR 里没有"卷积"这种算子？**
A：LLVM IR 目标是"表达所有语言的通用指令"——如果为卷积加指令，
就得为几千个算子加，LLVM 会无限膨胀。高层算子应该留在高层 IR
（Relax/MLIR 的专用方言），下降到 LLVM 时已经变成循环了。

**Q：我看 PR 时遇到 `.td`（TableGen）文件是什么？**
A：TableGen 是 LLVM/MLIR 的**声明式描述语言**——用表格描述指令/方言，
自动生成 C++ 代码。你在第 14 课 MLIR 部分看到的 `def ConstantOp` 就是。
读它不用懂全部，会看"这个 op 有几个输入输出、什么名字"就够。

---

## 8. 本课小结

- 编译器 = **分层 IR 金字塔**，每层管一段抽象，层间用 pass 下降
- **LLVM IR**：SSA + 强类型的低层指令，`%`虚拟寄存器
- **MLIR**：可扩展的多层 IR，方言机制让高层和低层并存
- **TIR**：TVM 的循环 IR，`T.block` + 调度（第 11 课）
- **PTX/NVPTX**：GPU 汇编，地址空间 + 线程模型 + `fma` 指令融合
- 五层对比表就是你的"IR 认知地图"

**下一步**：第 15 课。这些 IR 最终要跑在**硬件**上——缓存、寄存器、
内存层次。不懂硬件，编译优化就是无源之水。我们用模拟器把这部分讲透。

---

**导航**：⬅ [上一节](lesson13.md)（第 13 课 · meta_schedule 与自动调度）　｜　[下一节](lesson15.md)（第 15 课 · 硬件必修课）➡
