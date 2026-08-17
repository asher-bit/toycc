# exp09: 第 9 章实验 —— 端到端排查清单

把 exp01 + exp02 + profiler 串成第 9 章的"假设 → 证据 → 行动 → 验证"循环。
以"fused GEMM 端到端只有 60%"为靶子，逐层走：

## 层 0：图上有没有可融合的机会

```text
问题: bias+relu 是三个 kernel 还是融进 epilogue?
证据: 你的图(或框架 dump)上 matmul/bias/relu 是三个节点
行动: 按第 4 章 epilogue 账(每 tile 省 128KB 往返)融合
验证: 端到端时间下降? 记下百分比
```

## 层 2：资源账有没有溢出

```bash
nvcc -O3 -arch=sm_86 --ptxas-options=-v fused_gemm.cu -o fused_gemm
# 看 spill 是否非零(第 3 章第 5 节); 非零 → 缩小 tile / 调 unroll
```

## 层 4：访存与 stall

```bash
ncu --set full ./fused_gemm
# Speed of Light: SM 利用率 vs DRAM 吞吐谁先到顶?
# Warp State: stall 主因是 memory 还是 wait? → 第 7 章指标到行动表
```

## 层 5：时间线

```bash
nsys profile --stats=true ./fused_gemm
# kernel 块之间的大段空白 = 同步/launch 开销(第 6 章)
```

## 复盘输出

按第 9 章第 6 节模板写一页：问题 / 基线 / 定位(哪一层) / 证据(哪个工具的数字) /
修复(改了什么) / 验证(数字) / 风险。**每轮只改一个变量。**
