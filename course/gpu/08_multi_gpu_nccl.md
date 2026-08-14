# GPU 第 8 章：多 GPU、NCCL 与通信计算重叠

## 1. 本章目标

- 能说清 AllReduce / ReduceScatter / AllGather / Broadcast / AlltoAll 五种集合通信的数据语义，并手算各自每卡的通信量；
- 能手算 ring allreduce 的"每卡发送字节数"公式，并用它解释一次 70B 模型梯度同步的 272 ms 从哪来；
- 能说清 NCCL 的对象模型（rank / uniqueId / comm / stream），知道它不是什么；
- 能用"带宽层级"判断一个通信瓶颈该往哪优化；
- 能手算通信计算重叠的收益上限，并列出实现重叠的五个条件。

前置：第 6 章的 stream/event、第 7 章的时间线读法。跑实验需要多张 NVIDIA GPU 与 NCCL；手算不需要。

## 2. 工作中的问题长什么样

多 GPU 方向的日常问题：

```text
"8 卡训练怎么只有 4 倍速度？"
"梯度同步一次为什么花 272 ms？"
"什么时候用 AllReduce，什么时候拆成 ReduceScatter + AllGather？"
```

三问对应三个账：**通信量账**（要搬多少字节）、**带宽账**（搬这些字节要多久）、**重叠账**（通信时间能不能藏进计算里）。本章建立这三本账，性能分析和编译优化都以它们为底。

## 3. 五种集合通信：语义与每卡通信量

集合通信操作是**有数据依赖的 kernel/传输序列**，不是"调一个 API 免费完成"。算法、消息大小、rank 数、拓扑与 stream 都影响代价。五种操作的语义：

| 操作 | 数据语义（一句话） | 常见场景 |
|---|---|---|
| AllReduce | 所有 rank 的对应元素做规约（如求和），**每个 rank 都拿到完整结果** | 数据并行梯度同步 |
| ReduceScatter | 先规约，再按块**分散**——每个 rank 只拿结果的一段 | 分片优化器、张量并行反向 |
| AllGather | 每个 rank 的片段**汇集**到所有 rank——人人拿全量 | 参数/激活重建 |
| Broadcast | 一个 rank 的数据复制给所有 rank | 初始化、权重下发 |
| AlltoAll | 每个 rank 向**每个** rank 发一段不同的数据 | MoE 专家分发、复杂 sharding |

### 3.1 ring allreduce 的通信量账

AllReduce 的经典实现是 ring 算法：N 个 rank 围成环，规约分两步（先 N−1 步环上归约，再 N−1 步广播结果）。关键结论是**每个 rank 发送的字节数**：

```text
每卡发送 = 2 × (N−1)/N × S        (S = 待规约数据总量)
```

推导骨架：把 S 切成 N 段，归约阶段每卡发 (N−1) 段、每段 S/N；广播阶段同样 (N−1) 段。手算 N=4、S=100 MB：

```text
每卡发送 = 2 × 3/4 × 100 MB = 150 MB
```

### 3.2 一次真实梯度同步：272 ms 从哪来

70B 模型 fp16 梯度约 140 GB，8 卡 NVLink 有效带宽约 900 GB/s：

```text
每卡发送 = 2 × 7/8 × 140 GB = 245 GB
时间 = 245 GB / 900 GB/s ≈ 272 ms
```

三个数字一条链，每一步都可复算。**注意公式的性质：时间几乎不随卡数增长**（N 增大时 (N−1)/N → 1，每卡要发约 2S，但带宽也在更多环段上分摊）——所以"加卡减少通信时间"通常是错的，加卡减少的是**每卡的计算量**，通信时间近似不变。这也是大规模训练里"通信成为第一瓶颈"的账本依据。

### 3.3 ReduceScatter + AllGather 与 AllReduce 的等价

对"每个 rank 最终都要完整结果"的场景，两者数据语义等价，但中间态不同：AllReduce 直接给完整结果；ReduceScatter 先让每卡只拿一段（通信量与 AllReduce 相同），之后需要时再 AllGather。**张量并行反向**恰好只需要"每卡一段"（梯度按列切分），所以用 ReduceScatter 省掉最后的广播段——选哪种操作，由"结果最后归谁"决定，不是口味问题。

## 4. NCCL 的对象模型

```text
rank / device(每张 GPU 一个编号)
   ↓ 唯一 ID 协商(ncclUniqueId)
ncclComm(communicator: 一组 rank 的通信上下文)
   ↓
collective(buffer, count, datatype, op, comm, stream)
```

逐个定义：**rank** 是参与通信的每张 GPU 的编号（0..N−1）；**ncclUniqueId** 是建立一组通信前各 rank 用来"对暗号"的唯一标识；**ncclComm（communicator）** 是初始化后持有拓扑、算法、缓冲等状态的通信上下文，所有 collective 都挂在它下面；**stream** 决定通信与计算如何在设备时间线上排队（第 6 章的对象在这里复用）。【示意代码】（省略错误检查；真实代码必须检查每个返回码）：

```cpp
ncclUniqueId id;                       // rank 0 生成并广播给其他 rank
ncclComm_t comm;
ncclCommInitRank(&comm, nranks, id, my_rank);      // 建立 communicator
ncclAllReduce(sendbuf, recvbuf, count, ncclFloat,  // 规约
              ncclSum, comm, stream);              // 挂在 stream 上异步执行
```

**NCCL 不是分布式训练框架**：它只解决"集合通信怎么最快完成"；切分策略、梯度累积、重计算、容错是框架的事。调用时的四个明确对象——buffer、count、dtype、stream——任何一个配错，错误都以"异步、迟到"的方式出现（第 6 章 sticky error 的规则原样适用）。

## 5. 带宽层级：通信为什么必须被重叠或压缩

把第 1 章的带宽层级扩展到多卡：

```text
HBM(卡内显存)   ≈ 3 TB/s       ← 计算侧
NVLink(卡间)    ≈ 900 GB/s     ← 节点内通信, 比 HBM(3.35 TB/s) 低 3~4 倍
InfiniBand(跨机) ≈ 50 GB/s     ← 跨节点通信, 比 NVLink 再低 ~18 倍
```

（近似值，随代际变化。）两个推论：**通信带宽比计算侧低一到两个数量级**，所以任何"把通信裸露在关键路径上"的设计都会立刻触顶；**跨机通信比卡间贵一个数量级**，所以并行策略的排布规则是：通信重的切法（张量并行）放节点内，通信轻的切法（数据/流水并行）放跨节点。这与主教材第 32 课的 3D 并行摆法同源。

通信时间的通用模型：

```text
通信时间 ≈ latency + payload / effective_bandwidth
```

**小消息 latency 主导**（一次往返的固定开销，几微秒到几十微秒量级），**大消息带宽主导**。判断"现在卡在哪一项"的方法是扫不同大小的消息测时间：时间不随大小增长 = latency 主导（优化方向：减少通信次数、合并小消息）；时间线性增长 = 带宽主导（优化方向：压缩数据、换更宽的链路、重叠）。

## 6. 通信计算重叠：收益账与五个条件

理想时间线：把计算与通信分块交错，通信藏进计算里：

```text
GPU 0: compute chunk 0 ── compute chunk 1 ──
          comm chunk 0 ── comm chunk 1 ──
GPU 1: compute chunk 0 ── compute chunk 1 ──
```

收益上限的手算（用第 3.2 节的数字）：一段反向计算 800 ms、梯度同步 272 ms：

```text
不重叠: 800 + 272 = 1072 ms
完美重叠: max(800, 272) = 800 ms → 省 272 ms, 快 25%
```

**重叠的收益上限 = 串行总时间 − max(计算, 通信)**——重叠只能"藏"时间，不能"消"时间。实现重叠需要五个条件，缺一不可：

1. 计算与通信用**不同的 stream**（同 stream 天然串行，第 6 章）；
2. 生产者/消费者用 **event 或框架依赖**正确连接（先算完的块才能发）；
3. **buffer 生命周期覆盖异步通信**（通信没完成 buffer 就复用/释放 = 数据竞争）；
4. 通信没有占满与计算共享的资源（NVLink 与 HBM 独立，但 SM 与部分内存子系统共享）；
5. **分块粒度**足够大以摊薄 launch/同步开销，又不至于过大失去交错机会。

最后一条纪律：**"开了两个 stream"不等于重叠**——必须回到 Nsight Systems 时间线（第 7 章）验证 comm 块真的落在 compute 块下方，而不是串成一排。

## 7. 通信进入编译器：图优化器需要知道什么

多 GPU 不是"在模型末尾加一个 NCCL 调用"，而是 IR、调度、runtime、profiler 的跨层问题。编译器/图优化器在处理 collective 时需要六类信息：

| 信息 | 在 IR 里对应什么 |
|---|---|
| collective 的输入输出 shape/dtype/layout | 通信算子节点的类型与张量属性 |
| rank/world size、拓扑、设备映射 | 编译目标的设备表 |
| 通信可否重排、分块、与计算重叠 | 通信算子的调度约束（可否移动、能否拆分） |
| 哪些 buffer 可复用、哪些必须活到通信完成 | 内存规划的生命周期分析 |
| 通信失败/超时如何传播 | 错误语义（异步错误沿 stream 上报） |
| sharding 改变后如何重新生成 collective | 切分策略到通信算子的重写规则 |

这六类与 toycc 的 pass 一一对应：shape 推导、内存规划、调度约束、错误处理——多 GPU 编译器不引入新原理，只是把单卡 pass 的对象从"算子"扩展成"collective 算子"。

## 8. 常见错误与归因

| 现象 | 根因 | 定位手段 |
|---|---|---|
| 8 卡只有 4 倍速度 | 通信裸露、负载不均、流水气泡 | 时间线找未重叠的 comm 块 |
| 通信时间几乎不随消息变小 | latency 主导，小消息太多次 | 扫消息大小的通信时间曲线 |
| 开了双 stream 仍无重叠 | NULL stream 隐式同步 / 缺 event 依赖 | Nsight Systems 看 comm 块位置 |
| 结果偶发错 | buffer 提前复用/释放，与异步通信竞争 | 检查 buffer 生命周期与 event |
| NCCL 报错难定位 | 异步错误 + communicator abort | 分阶段同步查错（第 6 章纪律） |
| 跨机比卡内慢一个数量级 | 跨了 IB 链路（带宽层级账） | 核对并行切法的节点内/跨节点排布 |

## 9. 检查点

完成以下四项才算通过本章：

1. 手算：8 卡、梯度 80 GB 的 ring allreduce，每卡发送多少字节；
2. 用"latency + payload/bandwidth"模型，设计一个区分"latency 主导 vs 带宽主导"的测量实验（说明测什么、看什么）；
3. 手算：计算 600 ms、通信 150 ms 时，重叠后的时间下限与最大收益；
4. 说出"AllReduce 换 ReduceScatter"在什么场景下是等价且更省的，依据是什么。

## 10. 下一步与扩展阅读

本章建立了多卡的三本账。下一章（GPU 09：端到端案例）把第 1~8 章全部串起来：一个算子从模型里出发，经过 IR、编译器、kernel、运行时、通信与 profiler，走到多卡执行。分布式并行的切分策略（DP/TP/PP/EP）与通信账的完整推导见主教材第 32 课。

- 官方：[NCCL 文档](https://docs.nvidia.com/deeplearning/nccl/)、[NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage.html)；
- 与本课程的关系：NCCL 的 communicator/stream 模型就是第 6 章 runtime 对象的"多卡扩展"；ring 算法的每卡通信量账是主教材第 32 课带宽模型的同一推导。

**导航**：⬅ [上一章](07_profiling_performance.md)（性能分析与 Nsight）　｜　[下一章](09_end_to_end.md)（端到端案例）➡
