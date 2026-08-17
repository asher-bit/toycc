# GPU 专题真机实验目录

每章一个可运行的实验。**全部需要 NVIDIA GPU + CUDA Toolkit（WSL2 或 Linux）**；
本目录不替代章节里的手算——先手算，再上机验证。

```text
exp01_vector_add.cu      第 1 章: 合并访问与有效带宽测量
exp02_toolchain.md       第 2 章: nvcc/ptxas/cuobjdump 产物链命令集
exp04_gemm_triton.py     第 4/5 章: Triton GEMM(可对比 BLOCK/warps/stages)
exp06_module_loader.cu   第 6 章: Driver API 最小 module loader
exp08_nccl_allreduce.cu  第 8 章: 2 GPU ring 通信量验证(需 2 卡)
exp09_end_to_end.md      第 9 章: 端到端排查清单(实验+profiler 组合)
```

## 通用纪律（第 7 章六步协议）

1. 锁时钟/固定功耗模式，独占 GPU；
2. 每个实验先 warmup 再计时；
3. 用 cudaEvent 计时，报 median 不报 mean；
4. 每次只改一个变量；
5. 记录 GPU 型号、driver/toolkit 版本、编译参数——写进你的复盘。

## 版本边界

命令与指标基于 CUDA 11.x~12.x、Ampere 及以上架构；具体以本机
`nvidia-smi`、`nvcc --help` 与 `--ptxas-options=-v` 的实际输出为准。
