# 第 31 课：LLM 推理性能工程——KV cache、prefill/decode、吞吐与延迟要分开算

> 本课风格：手算为主——每一笔账都能自己在纸上复算。
> 目的：进高性能部会议室，第一句话往往是"这个模型在这张卡上跑多快"——
> 本课教你把这句话拆成三张可计算的账：算力账、带宽账、显存账。
> 前置：第 19 课（roofline/benchmark）、第 21 课（GPU 架构）、第 15 课（内存层次）。

---

## 1. 高性能部的日常问题长什么样

```
"7B 模型在 A100 上 decode 能到多少 tok/s？"
"70B 为什么一张卡装不下？"
"为什么 prefill 快、decode 慢？"
"batch 翻倍，吞吐为什么没翻倍？"
```

这些问题的答案都不是"测一下"，而是**一算就算出来**。本课给你
三张账：**算力账（FLOP）、带宽账（字节/秒）、显存账（字节）**。
LLM 推理的所有性能讨论，都是这三张账的组合。

---

## 2. 推理两阶段：prefill vs decode，瓶颈完全不同

自回归 LLM 的推理分两步：

```
prefill:  输入 prompt(N 个 token) 一次并行算完  → 大 GEMM, 算力受限(compute-bound)
decode:   每次只生成 1 个 token, 循环 N 次        → 小 GEMV, 带宽受限(memory-bound)
```

> **手算 1：decode 为什么注定带宽受限**
>
> 7B 模型，fp16 权重 = 7B × 2B = **14 GB**。每生成 1 个 token，
> 都要把**全部权重读一遍**（前向一次）。A100 HBM 带宽 ≈ 2 TB/s：
>
> ```
> 每 token 最快 = 14 GB / 2 TB/s = 7 ms/token → 上限约 140 tok/s
> 算力需求呢? 7B 参数 × 2 FLOP × 1 token = 14 GFLOP
> A100 算力 312 TFLOPS(fp16) → 只需 0.045 ms
> ```
>
> 比值：带宽时间 7ms vs 算力时间 0.045ms——**155 倍的带宽瓶颈**。
> 这就是 roofline 的直接应用：decode 的算术强度 ≈ 1 FLOP/2B，
> 远低于 A100 的拐点（~156 FLOP/B）。**decode 的全部优化，
> 都是围绕"怎么少读、快读"展开的**——量化、KV 压缩、
> speculative decoding 都是这条逻辑的分支。

> **手算 2：prefill 为什么相反**
>
> prompt 512 个 token 一次算：FLOP = 7B × 2 × 512 ≈ 7 TFLOP，
> 字节还是那 14 GB 权重（但摊到 512 token 上）：
>
> ```
> 算力时间 ≈ 7 TFLOP / 312 TFLOPS = 22 ms
> 带宽时间 ≈ 14 GB / 2 TB/s = 7 ms
> 算术强度 = 512 × (1 FLOP/2B) = 256 FLOP/B  → 越过拐点, 算力受限 ✓
> ```
>
> **关键认知**：同一个模型、同一张卡，prefill 和 decode 的瓶颈
> **完全相反**——所以 serving 系统永远把两个指标分开报：
> TTFT（首 token 时间 ≈ prefill）和 TPOT（每 token 时间 ≈ decode）。

---

## 3. 显存账：KV cache 为什么是"第一约束"

### 3.1 KV cache 是什么

decode 每步要用到**所有历史 token 的 key/value**（注意力），
所以每步把算好的 K/V 存下来——这就是 KV cache。它随
`batch × 序列长度 × 层数` 线性增长。

### 3.2 手算：KV cache 到底多大

```
公式: KV 字节 = 2(K和V) × 层数 × 头维度×头数(=隐层 d_model) × 序列长 × batch × 精度

7B 模型(L=32 层, d=4096), fp16, batch=1, 序列 4096:
  = 2 × 32 × 4096 × 4096 × 1 × 2B ≈ 2.1 GB
batch=32: 2.1 × 32 ≈ 67 GB   ← 比权重(14GB)大近 5 倍!

70B(L=80, d=8192), fp16, batch=8, 序列 8192:
  = 2 × 80 × 8192 × 8192 × 8 × 2B ≈ 172 GB   ← 单卡 H100 80GB 装不下
```

**结论**：大模型推理的显存瓶颈往往**不是权重，是 KV cache**。
所以"这张卡能跑多大 batch/多长上下文"是背下来的公式。
这也是 PagedAttention/KV 量化/投机解码存在的理由。

### 3.3 PagedAttention：KV cache 的"虚拟内存"

naive 方案给每个请求**预分配最大长度**的连续显存——和"给每个
进程预分配最大内存"一样浪费（内部碎片 + 过度预留）。vLLM 的
PagedAttention 把 KV cache 切成**固定大小的页（如 16 token/页）**，
用**页表**映射逻辑块到物理块：

```
请求 A 的逻辑序列: [tok0..tok15][tok16..tok31]...   (逻辑页)
        页表映射:      页3         页7               (物理页, 可不连续)
```

**这就是 GPU MMU 页表思想原样搬到应用层**——按需分配、
碎片消失、可共享（beam search/并行采样共用前缀页，copy-on-write）。
显存利用率从 ~40% 提到 ~90%+，吞吐直接翻倍。

---

## 4. FlashAttention：把 N² 的中间结果"算完就扔"

### 4.1 naive attention 的内存账

```
Q×Kᵀ → S (N×N 得分矩阵) → softmax → ×V → 输出
         ↑ 这个 N×N 矩阵要写进显存再读回来!
序列 8K: N×N = 64M 元素 × 2B = 128 MB 的中间张量
```

N×N 矩阵是**纯中间结果**——算完 softmax 就没用了，却要付两次
显存往返（写 + 读）。手算它的代价（A100，N=8192）：

```
写 128MB + 读 128MB = 256 MB / 2 TB/s ≈ 0.13 ms
这还只是 softmax 一次; 再乘 V 又一轮 → attention 的时间
大头不是乘法是搬运 → memory-bound, 又是 roofline 的形状
```

### 4.2 FlashAttention 的思想：分块 + 在线 softmax

```
把 Q/K/V 切成块(tile), 在 SRAM(共享内存)里:
  for 每个 K/V 块:
    算这一块的 S 块(在 SRAM, 不落显存)
    在线更新 softmax 的 max 和分母(数学上可增量)
    累加到输出块
N×N 矩阵从未落显存 → IO 从 O(N²) 降到 O(N²d/M)(M=SRAM 大小)
```

**这正是 tile + roofline 在真实算子上的教科书
应用**：把"带宽受限"的算子通过分块复用变成接近算力受限。
FlashAttention-2/3 再加 warp 级分工（warp 原语）和
异步拷贝（TMA）——**你学过的每一块拼图都在里面**。

---

## 5. 编译器/工具链的接入点

推理性能工程不是只有 kernel，编译器在四个地方起作用：

1. **GEMV vs GEMM 切换**：decode 时 batch=1 是 GEMV（带宽受限），
   continuous batching 把多请求拼成 GEMM——**算子形态由 batch 决定**，
   编译器要按形状分发不同 kernel
2. **continuous batching 是调度问题**：请求来了就插进当前 batch——
   "搜索/调度"思想的服务化版本（目标是 GPU 不空转）
3. **量化 kernel**：W4A16 的 dequant 融合进 GEMM
4. **图优化**：把 RMSNorm/SwiGLU/RoPE 融进 GEMM 的 epilogue——
    算子融合在 LLM 上的直接应用

---

## 6. FAQ

**Q：为什么 batch 翻倍，吞吐没翻倍？**
A：decode 带宽受限时，batch 翻倍只让"同一次权重读取喂 2 个请求"——
权重带宽摊薄了，吞吐确实涨；但 KV cache 读取随 batch 线性涨，
**KV 带宽成为新瓶颈**时就涨不动了。拐点 = 权重时间 vs KV 时间相等处，
也是一笔账。

**Q：H100 比 A100 decode 快多少，怎么估？**
A：带宽比：3.35/2.0 ≈ 1.7×——decode 的提速基本就是 HBM 带宽比
（量化后再乘量化收益）。这就是为什么推理卡选型先看 HBM 规格。

**Q：TTFT 长好还是 TPOT 短好？**
A：看产品形态：聊天场景 TTFT 敏感（用户在等第一个字），
长文生成 TPOT 敏感。serving 系统的所有权衡（分chunk prefill、
抢占调度）都是在这两个指标间选位置。

**Q：这些账需要背吗？**
A：公式结构要背（三行），数字不用——会查表（HBM 带宽、算力）
现场算。会议室里"7B fp16 decode 上限 140 tok/s"这种数，
都是从这两行公式现算的。

---

## 7. 本课小结

- 推理性能 = **算力账 + 带宽账 + 显存账**，三张账分开算
- prefill 算力受限（GEMM），decode 带宽受限（GEMV，上限 = 权重字节/带宽）
- KV cache 字节 = 2×层数×d×序列×batch×精度——**大模型显存第一约束**
- PagedAttention = KV cache 的虚拟内存（页表思想）
- FlashAttention = tile + 在线 softmax，把 N² 中间结果消灭在 SRAM 里
- TTFT/TPOT 分开报，continuous batching 是调度问题

**下一步**：第 32 课——分布式并行与通信。单卡的账算完了，
下一步是"一张卡装不下"的账：8 卡怎么分、怎么通信、
allreduce 的带宽模型长什么样。

---

## 深层拓展：推理性能的三个进阶问题

### A. 为什么 decode 合并 batch 也不"省带宽"

batch 合并摊薄的是**权重**读取（共享一份），但每请求有**自己的
KV cache**——batch 32 就是 32 份 KV 读取。所以当序列长、batch 大时，
KV 带宽超过权重带宽，decode 从"权重受限"变"KV 受限"——
这个交叉点决定了"该用大 batch 还是小 batch"。

### B. attention 的 roofline 点怎么算

FlashAttention 的 IO ≈ O(N²d²/M)。把 M（SRAM 227KB@H100）和
d=128 代入：tile 选多大让 SRAM 装下 K/V 块 + 输出块 + 统计量——
这是一道真实的"调度参数"题，答案决定 kernel 快一倍还是慢一倍。
第 13 课 autotune 搜的就是这类参数。

### C. 从账到决策：一份真实的容量规划

```
需求: 70B 模型, 4K 上下文, 100 并发
权重: W4A16 → 35GB
KV:   GQA 头维度 1024 (Llama-2-70B 实际配置):
      2 × 80 × 1024 × 4096 × 100 × 1B(KV int8) ≈ 67GB
      无 GQA(d_kv = 8192) 时同式 ×8 = 537GB
合计 ≈ 102GB → 2×H100(80GB) TP=2 起步
```

注意 d 用的是 KV 头的维度，不是隐层维度：GQA（分组查询注意力）
把 KV 头砍到 8 个（1024 维），KV cache 直接省 8 倍——这也是
"长上下文模型都用 GQA"的原因。无 GQA 的 70B 要 537GB KV，
一张卡连 KV 都装不下。
---

**导航**：⬅ [上一节](lesson30.md)（第 30 课 · 驱动与命令提交）　｜　[下一节](lesson32.md)（第 32 课 · 分布式并行与通信）➡