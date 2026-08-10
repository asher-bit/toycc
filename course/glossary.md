# 词汇表

按出现顺序排列，方便随时查。

## IR 与图

- **IR (Intermediate Representation)**：编译器内部用来描述计算的数据结构。本课特指"计算图"。
- **计算图 (Computational Graph)**：由"输入张量、算子节点、边（张量流动）"组成的有向无环图（DAG）。
- **节点 (Node)**：图里的一个算子实例，含名字、算子类型、输入、属性。
- **拓扑序 (Topological Order)**：一种"每个节点都排在消费者前面"的节点排列，编译器和执行器按它遍历。
- **消费者 (Consumer)**：读某个张量作为输入的算子。`x` 有几个消费者 = 有几个算子读 `x`。
- **生产者 (Producer)**：产生某个张量的算子。
- **形状推导 (Shape Inference)**：从输入形状出发，逐个算子推算出所有中间张量的形状。
- **算子注册表 (Op Registry)**：声明"每个算子叫什么、吃几个输入、怎么推形状、怎么执行"的登记表。
- **属性 (Attrs)**：算子除输入外的参数，如卷积的 kernel/stride/pad。

## Pass 与优化

- **Pass**：读一张图，改写成"语义等价但更优"的另一张图的优化过程。
- **Pass 管线 (Pipeline)**：多个 pass 按依赖顺序串起来，依次对图加工。
- **语义等价 (Semantic Equivalence)**：改写前后计算结果必须完全一致。编译器优化的铁律。
- **算子融合 (Operator Fusion)**：把"计算根算子 + 逐元素/偏置算子"合并成一个核，省内存带宽和启动开销。
- **布局 (Layout)**：张量各维度在内存中的排列顺序，如 NCHW / NHWC。
- **布局传播 (Layout Propagation)**：布局无关的算子继承输入布局，让布局沿图传递，减少搬移。
- **layout_transform / permute_dims**：把张量从一种布局搬成另一种布局的数据搬移算子。
- **常量折叠 (Constant Folding)**：把"输入全是常量"的算子挪到编译期算完，结果焊进常量。
- **编译期 vs 运行时**：编译期只做一次、可慢；运行时每次推理都执行、必须快。
- **死代码消除 (DCE)**：删掉"没人消费"的节点/常量。
- **后支配树 (Post-dominator Tree)**：刻画"谁必然在谁之后执行"的分析结构，融合算法靠它处理菱形分支。
- **并查集 (Union-Find)**：一种分组数据结构，融合时把同组节点快速合并。
- **op_pattern**：算子的计算模式标注（如 elementwise/broadcast/reduce/opaque），融合规则依赖它。

## 内存与后端

- **生命周期 (Liveness)**：张量从出生（被产生）到死亡（最后一个消费者执行完）的区间。
- **峰值内存 (Peak Memory)**：任意时刻活跃张量大小之和的最大值，决定设备内存够不够。
- **缓冲区复用 (Buffer Reuse)**：两个生命周期不重叠的张量共用同一块内存。
- **Workspace**：为某个算子预留的临时工作区。
- **代码生成 (Codegen)**：把最终图翻译成可执行代码（C/CUDA/...）。
- **后端 (Backend)**：目标平台/语言（x86、CUDA、TensorRT、Python…）。"一个 IR 多个后端"是编译器核心思想。
- **BYOC**：Bring Your Own Codegen，TVM 让第三方后端注册进来生成代码的架构。
- **ExternFunc**：IR 里指代"外部已编译符号"的节点，codegen 后替换掉原函数。

## TVM 专用

- **Relax**：TVM 现役的高层图 IR（取代了老一代 Relay）。
- **Relay**：TVM 旧的高层 IR，2023 年后被 Relax 取代。
- **TIR**：TVM 的底层 IR，描述循环/线程/内存访问（对应我们 codegen 前的"循环结构"）。
- **PrimFunc**：TIR 里一个"原语函数"，一个融合核就是一个 PrimFunc。
- **call_tir**：Relax 中"调用一个 TIR 原语函数"的算子。
- **IRModule**：一组函数的集合（一个或几个 Relax Function / PrimFunc）。
- **PassContext**：pass 执行时的全局上下文，opt_level、配置都从这取。
- **meta_schedule / autotuning**：自动搜索调度（分块/向量化参数）的子系统。

## 调度（第 11 课）

- **调度 (Schedule)**：为算子选择"循环怎么写"的过程；Relax 决定算什么，TIR 调度决定怎么算。
- **split**：把一个循环拆成外层+内层两个循环。
- **fuse**：把两个相邻循环合并成一个。
- **reorder**：重排循环的执行顺序（影响缓存命中，性能差异巨大）。
- **tile（分块）**：2D 分块 = 两次 split + 重排，让数据小块待进缓存。
- **vectorize（向量化）**：把内层循环变成一条 SIMD 指令。
- **parallel / bind**：循环并行化（CPU 多线程 / GPU 绑定线程）。
- **cache_read / cache_write**：在循环里引入中间缓冲（寄存器/共享内存）。
- **循环不变量 (loop-invariant)**：循环内不变的值，可提到循环外（LICM）。

## 优化全景（第 12 课）

- **DCE（死代码消除）**：删掉计算结果没人消费的节点；用"不动点"反复删。
- **CSE（公共子表达式消除）**：同一表达式只算一次，结果复用。
- **代数化简 (Simplify)**：用数学恒等式消掉计算（如 `x+0→x`）。
- **内联 (Inlining)**：把函数体展开到调用处。
- **合并 (Combine)**：把并联的算子（如多个 matmul）并成一个。
- **不动点 (Fixpoint)**：pass 反复执行直到图不再变化。
- **op_pattern**：算子的计算模式标注：`kElemWise`（逐元素）、`kBroadcast`（逐元素+广播）、`kInjective`（纯搬移）、`kCommReduce`（归约）、`kOpaque`（计算密集）。融合规则按模式匹配。
- **支配 (Dominate)**：从入口到 Y 的所有路径都经过 X，则 X 支配 Y。
- **立即支配者 (idom)** / **支配树**：每个节点唯一的最深支配者，连起来成树。
- **后支配 (Post-dominate)**：从 Y 到出口的所有路径都经过 X，则 X 后支配 Y；融合算法靠它处理菱形分支。
- **依赖分析**：判断循环/算子之间谁先谁后的分析，调度合法性检查的基础。

## autotuning（第 13 课）

- **搜索空间 (SearchSpace)**：所有合法调度的参数集合。
- **成本模型 (CostModel)**：预测未测调度耗时的机器学习模型。
- **Runner**：在真机/远程设备上实测耗时的执行器。
- **Database**：存"调度→耗时"记录，可复用、可导出。
- **AutoTVM → Ansor → meta_schedule**：TVM 自动调度的三代实现，现在是 meta_schedule。

## IR 家族（第 14 课）

- **LLVM IR**：通用编译器的低层 IR。SSA + 虚拟寄存器（`%x`）+ 强类型（`i32`）。
- **MLIR**：可扩展的多层 IR。核心是**方言（Dialect）**，每层抽象一个命名空间+算子集。
- **Operation / 方言 / 下降 (lowering)**：MLIR 三件套——操作是统一单元，方言分组，层间下降。
- **TVMScript / T.block**：TVM 的可读 TIR 语言；计算块是 TIR 的原子单元。
- **PTX / NVPTX**：NVIDIA GPU 的虚拟汇编；NVPTX 是 LLVM 的 PTX 后端。
- **地址空间 (address space)**：GPU 内存分级（global/shared/local/constant）。
- **FMA**：一条指令做 `a*b+c` 的乘加融合，指令层的"融合"。
- **虚拟寄存器 → 真寄存器**：LLVM IR 的 `%x` 是无限的虚拟寄存器，后端寄存器分配器映射到有限的真寄存器（溢出时写回内存）。

## 硬件基础（第 15 课）

- **内存层次 (memory hierarchy)**：寄存器 < L1 < L2 < L3 < DRAM，速度差上百倍。
- **缓存行 (cache line)**：缓存按"行"读取（典型 64 字节）；读一行只用 1 个元素=浪费。
- **时间/空间局部性**：刚用过的还会用 / 旁边的快用了；编译器靠布局和分块制造局部性。
- **冲突缺失 (conflict miss)**：两块数据映射到缓存同一位置互相踢。
- **寄存器压力 (register pressure)**：同时活着的值个数；太多会溢出（spill）到内存。
- **SIMD / 向量化**：一条指令算多个数；依赖数据连续。
- **延迟 vs 带宽**：一次等多久 vs 每秒搬多少；现代瓶颈多是带宽。
- **warp / 线程块 / occupancy**：GPU 概念——32 线程锁步执行；块内共享内存；占用率。
- **ILP / 流水线**：指令级并行，减少依赖、循环展开来利用。

## GPU 芯片与编译器（第 21-24 课）

- **SM (Streaming Multiprocessor)**：GPU 的计算集群，含 CUDA 核、寄存器文件、共享内存、张量核、warp 调度器。
- **warp**：GPU 的执行单位，32 个线程一组，锁步执行（SIMT）。
- **SIMT**：单指令多线程——一个 warp 的 32 个线程执行同一条指令。
- **合并访问 (coalescing)**：warp 的 32 个线程读连续地址，合并成一次内存事务。
- **bank conflict（内存库冲突）**：32 线程访问共享内存撞到同一个 bank，硬件串行拆多次；垫 padding 解决。
- **分支发散 (branch divergence)**：warp 内线程走不同分支 → 串行执行 → 性能掉一半。
- **predication**：把分支变成"带开关的指令"，避免真分支。
- **共享内存 (shared memory)**：SM 内显式分配的快速内存，编译器用它做分块复用。
- **张量核 (Tensor Core)**：专门做矩阵乘加（MMA）的硬件单元。
- **占用率 (occupancy)**：活跃 warp / 最大 warp，决定延迟隐藏能力。
- **寄存器溢出 (spill)**：寄存器装不下，值被迫写回内存。
- **PTX / SASS**：GPU 的虚拟汇编 / 真实机器码，ptxas 负责从 PTX 到 SASS。
- **ISA**：Instruction Set Architecture，指令集架构。
- **工具链**：compiler + assembler + linker + driver + runtime + profiler + debugger。
- **BYOC**：Bring Your Own Codegen，外部后端接入机制。
- **target description**：告诉编译器芯片特性（寄存器/共享内存/线程数）的描述表。

## LLVM / MLIR 深入（第 25-26 课）

- **基本块 (basic block)**：LLVM IR 里"内部顺序执行、无分支"的代码块，块间用跳转连接。
- **phi 节点**：SSA 在汇合点的取值选择——"从 then 块来就是 a，从 else 块来就是 b"。
- **getelementptr (GEP)**：LLVM 的纯地址计算指令，只算地址不读内存。
- **Pass 管理器 (New PM)**：LLVM 调度 pass 的框架；`PreservedAnalyses` 告诉它哪些分析还作数。
- **指令选择 (instruction selection)**：把 LLVM IR 映射成目标指令候选的过程。
- **寄存器分配 (register allocation)**：虚拟寄存器→物理寄存器，图着色，装不下就 spill。
- **MC 层**：LLVM 最底层库，提供指令内存表示、汇编器、反汇编器、目标文件输出。
- **TableGen (.td)**：声明式描述指令/算子的语言，自动生成 C++ 代码。
- **方言 (dialect)**：MLIR 里"一层抽象"的命名空间（linalg/affine/scf/memref/llvm 等）。
- **Operation**：MLIR 的统一单元——所有算子/指令都是 Operation，高层低层可共存。
- **region**：MLIR 操作里嵌套的代码块（for/if 就是带 region 的操作）。
- **ODS**：Operation Definition Specification，用 TableGen 定义 MLIR 算子。
- **pattern rewrite**：MLIR 的 pass 核心机制，优化/下降/合法化三位一体。
- **渐进式下降 (progressive lowering)**：一层层降，每步都是合法 IR，降不了的留给下一跳。
- **greedy driver**：反复应用 pattern 直到没有可改的驱动器。

## 量化与精度（第 18 课）

- **量化 (Quantization)**：把 float 转成 int8 等低精度，内存/带宽减 4 倍 + int8 加速单元。
- **PTQ / QAT**：训练后量化（快，编译器主场）/ 量化感知训练（准，要重训）。
- **scale / zero_point**：量化的两个参数（对称量化只有 scale）。
- **per-tensor / per-channel**：scale 粒度；per-channel 权重精度高得多。
- **校准 (calibration)**：跑一批数据统计每层 scale 的过程。
- **fp16 / bf16 / tf32**：低精度浮点格式（bf16 范围同 fp32，训练不溢出）。
- **混合精度 (mixed precision)**：敏感层保留高精度，其余量化。

## 模型导入与下降（第 17 课）

- **前端 (Frontend)**：把 PyTorch/ONNX 模型翻译成编译器图 IR。
- **ONNX**：模型交换标准格式；input / initializer（权重）/ node / output / opset。
- **legalize / 下降 (lowering)**：把高层算子拆成可实现的底层算子组合（LayerNorm → mean/var/...）。
- **动态形状 (dynamic shape)**：形状运行时才定（如 batch/序列长），难优化 → 形状特化/动态分派。
- **控制流 (control flow)**：模型里的 if/while，使 IR 从 DAG 变成 CFG。
- **运行时 (Runtime) / VirtualMachine**：调度核、管内存、管设备；Relax 用 VM 解释指令序列。

## 性能与算子（第 19 课）

- **Roofline 模型**：性能上限 = min(算力墙, 带宽墙 × 计算强度)。
- **计算强度 (arithmetic intensity)**：总 FLOP / 总搬运字节；判断 compute-bound vs memory-bound。
- **im2col**：把卷积转成矩阵乘法（重复拷贝换 GEMM 优化复用）。
- **Winograd**：小卷积（3×3）用加法换乘法，省 50%+ 乘法。
- **GEMM 微内核**：分块 + 寄存器累加 + 向量化 + 双缓冲的极致优化。
- **benchmark 方法论**：warmup + 多次取中位数 + 同条件对比 + 区分延迟/吞吐。
