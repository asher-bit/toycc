# GPU 第 8 章：多 GPU、NCCL 与通信计算重叠

## 1. 本章目标

- 区分数据并行、张量并行、流水并行和参数/序列切分；
- 理解 AllReduce、AllGather、ReduceScatter、Broadcast 的数据语义；
- 知道 NCCL communicator、rank、stream 和 topology 的关系；
- 能判断多 GPU 训练/推理瓶颈来自计算、通信、同步还是拓扑。

## 2. 集合通信的基本语义

| 操作 | 直观含义 | 常见场景 |
|---|---|---|
| AllReduce | 所有 rank 得到规约结果 | 数据并行梯度同步 |
| ReduceScatter | 规约后每个 rank 得到一段 | 分片优化器/张量并行 |
| AllGather | 每个 rank 收集所有片段 | 参数/激活重建 |
| Broadcast | 一个 rank 广播给所有 rank | 初始化、权重同步 |
| AlltoAll | 每个 rank 向所有 rank 交换不同片段 | MoE、复杂 sharding |

通信操作是有数据依赖的 kernel/传输序列，不是“调用一个 API 就免费完成”。算法、消息大小、rank 数量、GPU 拓扑、NVLink/PCIe/网络和 stream 都影响结果。

## 3. NCCL 的对象模型

```text
rank / device
   ↓
ncclUniqueId 或 communicator 初始化
   ↓
ncclComm
   ↓
collective(buffer, count, datatype, op, comm, stream)
```

NCCL 是面向多 GPU 集合通信的库，不是完整的分布式训练框架。调用前需要建立 communicator，调用时明确 buffer、count、dtype、stream 和 collective 语义；错误处理还要考虑异步错误和 communicator abort。

## 4. 通信与计算重叠

理想时间线：

```text
GPU 0: compute chunk 0 ── compute chunk 1 ──
          communication chunk 0 ── communication chunk 1 ──
GPU 1: compute chunk 0 ── compute chunk 1 ──
```

要实现重叠，至少需要：

- 计算和通信使用合适的 stream；
- 生产者/消费者通过 event 或框架依赖正确连接；
- buffer 生命周期覆盖异步通信；
- 通信没有占满与计算共享的资源；
- 分块粒度足够大，又不会增加过多同步和 launch 开销。

“两个 stream”不保证重叠；需要通过 Nsight Systems 时间线验证。

## 5. 通信编译的关注点

编译器/图优化器需要知道：

- collective 的输入输出 shape、dtype 和布局；
- rank/world size、拓扑和设备映射；
- 通信是否可重排、可分块、可与计算重叠；
- 哪些 buffer 可复用，哪些必须保持到通信完成；
- 通信失败、超时和异步错误如何传播；
- sharding 改变后如何生成对应的 collective。

这也是为什么多 GPU 不是在模型末尾“加一个 NCCL 调用”，而是 IR、调度、runtime 和 profiler 的跨层问题。

## 6. 性能分析

先用模型公式估算通信量，再用实际 trace 验证：

```text
通信时间 ≈ latency + payload / effective_bandwidth
端到端时间 ≈ max(计算关键路径, 通信关键路径) + 未重叠部分
```

实际还需考虑 ring/tree 算法、分片、协议、拓扑、PCIe/NVLink/网络、并发 communicator 和 GPU 争用。不要用单卡 kernel 的 roofline 解释一个多卡通信瓶颈。

## 7. 练习

1. 画出 2、4、8 GPU AllReduce 的数据流；
2. 比较 AllReduce 与 ReduceScatter + AllGather 的数据量和用途；
3. 用两个 stream 设计 chunked compute/communication pipeline，并用 timeline 验证；
4. 记录 rank 映射、拓扑、消息大小、算法和带宽，写一份通信性能报告。

参考：[NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)、[Using NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage.html)。

