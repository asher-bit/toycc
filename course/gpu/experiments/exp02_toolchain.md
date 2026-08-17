# exp02: 第 2 章实验 —— 观察同一 kernel 的四种产物

【可运行代码】在 WSL2/Linux 的 CUDA 环境执行。先把 `exp01_vector_add.cu`
复制到本目录，然后按顺序跑下面命令，边跑边对照第 2 章 4.1~4.4 节。

## 1. 只看 PTX

```bash
nvcc -O3 --ptx -arch=compute_86 exp01_vector_add.cu -o exp01.ptx
head -5 exp01.ptx          # 预期: .version / .target / .address_size 三行头
grep -n "add_coalesced" exp01.ptx   # 找到 kernel 入口, 对照第 2 章第 5 节的逐块读法
```

## 2. 只出 cubin

```bash
nvcc -O3 -cubin -arch=sm_86 exp01_vector_add.cu -o exp01.cubin
ls -l exp01.cubin          # 二进制容器, 下一步拆开看
```

## 3. 从成品里拆产物

```bash
nvcc -O3 -arch=sm_86 exp01_vector_add.cu -o exp01
cuobjdump --dump-ptx  exp01 | head -20    # fatbin 里带的 PTX
cuobjdump --dump-sass exp01 | head -30    # fatbin 里带的 SASS(找 LDG.E/STG.E)
nvdisasm -c exp01.cubin  | head -20      # 单独反汇编 cubin
```

## 4. 看资源账

```bash
nvcc -O3 -arch=sm_86 --ptxas-options=-v exp01_vector_add.cu -o exp01
# 预期输出三个数字: registers / smem / spill(第 2 章 4.4 节解释每个的用途)
```

## 5. 兼容性负向实验

```bash
# 只带 sm_86 cubin, 在 sm_70 的卡上运行 → 预期报 209 cudaErrorNoKernelImageForDevice
nvcc -O3 -arch=sm_86 -code=sm_86 exp01_vector_add.cu -o exp01_sm86
# 只带 PTX, 在新卡上运行 → 首次加载会 JIT(观察第一次启动的延迟, 第 2 章第 7 节)
nvcc -O3 -arch=compute_86 -code=compute_86 exp01_vector_add.cu -o exp01_ptx_only
```

记录每个命令的实际输出, 与第 2 章的手算对照——这就是"产物链"的完整证据。
