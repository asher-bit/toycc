# GPU 第 1 章：CUDA 编程模型——一个 kernel 的线程、内存与同步如何映射到硬件

## 1. 本章目标

- 能把任意一个 kernel 的索引公式 `i = blockIdx.x * blockDim.x + threadIdx.x` 手算成一张"哪些线程算哪些元素"的映射表；
- 能说清 grid / block / thread / warp 四个对象各是什么、在硬件上落在哪里；
- 能用一个 warp 的访存模式手算出内存事务数，判断访问是否合并；
- 能解释 `__syncthreads()` 的死锁条件，以及 atomic / fence 各自解决什么不同的问题；
- 能用算术强度给一个 kernel 判断瓶颈在带宽还是在算力。

前置：需要知道"GPU 由多个 SM 组成、每个 SM 上有寄存器和共享内存"这个事实。如果这些词还陌生，先花五分钟读一遍主教材第 21 课的硬件地图，再回到本章。本章的示例代码在 CUDA Toolkit ≥ 11.8、任意 compute capability ≥ 7.0 的 NVIDIA GPU 上可运行；没有 GPU 也能完成全部手算。

## 2. 工作中的问题长什么样

新人接手 kernel 工作后，最常见的三个提问：

```text
"这个 kernel 为什么只跑到了 40% 带宽？"
"为什么结果偶尔对、偶尔错？"
"为什么我把 block 数调大，反而更慢了？"
```

这三个问题的答案分别落在四个不同的机制上：**线程映射**（谁算哪块数据）、**内存访问**（一次访存花几笔事务）、**同步**（谁能看见谁的写入）、**launch 开销**（发射本身花多少时间）。本章把前三个机制建立成可以手算的模型，launch 开销的完整账留给第 6 章（Runtime/Driver）。

## 3. 最小例子：一个完整的 vector_add

下面的代码属于【可运行代码】，保存为 `vector_add.cu`，用 `nvcc -O3 -lineinfo -arch=sm_86 vector_add.cu -o vector_add` 编译（`sm_86` 换成目标 GPU 的实际架构），预期输出一行 `max error = 0`：

```cuda
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do {                                    \
    cudaError_t e = (call);                                 \
    if (e != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",        \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1);                                            \
    }                                                       \
} while (0)

__global__ void vector_add(const float* a, const float* b,
                           float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 20;                    // 1M 个 float = 4 MB
  const size_t bytes = n * sizeof(float);

  float *h_a, *h_b, *h_out;                 // host 内存
  float *d_a, *d_b, *d_out;                 // device 内存
  h_a    = (float*)malloc(bytes);
  h_b    = (float*)malloc(bytes);
  h_out  = (float*)malloc(bytes);
  for (int i = 0; i < n; i++) { h_a[i] = i; h_b[i] = 2 * i; }

  CHECK(cudaMalloc(&d_a, bytes));
  CHECK(cudaMalloc(&d_b, bytes));
  CHECK(cudaMalloc(&d_out, bytes));
  CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

  int threads = 256;
  int blocks  = (n + threads - 1) / threads;   // 向上取整
  vector_add<<<blocks, threads>>>(d_a, d_b, d_out, n);
  CHECK(cudaGetLastError());                   // 捕获 launch 错误

  CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
  float max_err = 0.0f;
  for (int i = 0; i < n; i++) {
    float expect = 3.0f * i;
    float got = h_out[i];
    float err = fabsf(expect - got);
    if (err > max_err) max_err = err;
  }
  printf("max error = %g\n", max_err);

  cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
  free(h_a); free(h_b); free(h_out);
  return 0;
}
```

### 3.1 kernel 的三个标志

一个函数要成为 kernel，看三处：

- **`__global__`**：声明"这是一个从 host 发射、在 device 上执行的入口函数"。在编译产物里它对应一个带名字的 GPU 入口符号，运行时就是靠这个名字在模块里查找 kernel（第 6 章展开）。
- **`<<<blocks, threads>>>`**：launch 配置。左边的数给 grid 多少个 block，右边的数给每个 block 多少个 thread。它们是 host 侧的普通整数，被运行时打包进 launch 指令传给硬件。
- **索引公式**：kernel 内部不知道自己是谁，只能通过内建变量 `blockIdx`、`blockDim`、`threadIdx` 计算自己的全局编号。三者都是 `uint3` 结构（有 `.x/.y/.z` 三个分量），本章只用 `.x`。

### 3.2 索引公式的手算表

取 n = 1000、threads = 256 走一遍：

```text
blocks = ceil(1000 / 256) = 4          ← 4 个 block × 256 线程 = 1024 个线程
但只有 1000 个元素 → 有 24 个线程"越界"

blockIdx.x=0: i = 0*256 + threadIdx.x   → 覆盖元素   0..255
blockIdx.x=1: i = 1*256 + threadIdx.x   → 覆盖元素 256..511
blockIdx.x=2: i = 2*256 + threadIdx.x   → 覆盖元素 512..767
blockIdx.x=3: i = 3*256 + threadIdx.x   → 覆盖元素 768..1023 ← 最后 24 个越界
```

所以 `if (i < n)` 不是防御性洁癖：**当 n 不是 threads 的整数倍时，最后一块必然有一批线程指向不存在的元素**。去掉这个判断，`out[1000]` 会写到分配范围之外——结果随机错、可能带乱相邻内存，`compute-sanitizer` 会精确报出越界位置。

### 3.3 host 侧的四段固定流程

`main` 里的结构在几乎所有 CUDA 程序里重复出现，值得逐段命名：

1. **分配**：`cudaMalloc` 在 GPU 显存上分配，返回的是 device 指针，host 不能直接解引用它；
2. **搬运**：`cudaMemcpy` 把数据从 host 拷进 device（方向 `cudaMemcpyHostToDevice`），算完再拷回来；
3. **发射**：`<<<...>>>` 只是"下达指令"，kernel 不一定已经执行——想要结果，要么后面跟 `cudaMemcpy`（隐式同步），要么显式 `cudaDeviceSynchronize()`；
4. **校验**：`cudaGetLastError()` 抓到的是 launch 阶段的错误（如 grid 过大、kernel 名写错），它抓不到 kernel 内部的内存错误，后者属于 sanitizer 的职责。

## 4. 线程层次：thread / block / grid 三个对象

**thread（线程）**是执行一个 kernel 实例的最小软件单位。它拥有一份自己的 `threadIdx`、自己的局部变量，对应硬件上一个 warp 里的一个 lane。

**block（线程块）**是一组可以互相协作的线程。同一个 block 内的线程保证被调度到同一个 SM 上，因此它们能共享那片 SM 的共享内存，也能用 `__syncthreads()` 互相等待。**不同 block 之间没有任何顺序保证**：调度器可以任意顺序、任意时刻把 block 派到空闲的 SM 上，所以任何依赖"block 0 先于 block 1 执行"的代码都是错的。

**grid（网格）**是一次 launch 的全部 block。`gridDim` 就是 grid 里 block 的个数。

三个层次与硬件的对应关系：

| 层次 | 可见范围 | 同步手段 | 硬件落点 |
|---|---|---|---|
| thread | 自己的寄存器、局部变量 | 无 | SM 内 warp 的一个 lane |
| block | 同 block 线程 + 共享内存 | `__syncthreads()` | 一个 SM |
| grid | 全局内存 | 只有 kernel 边界（一次 launch 结束） | 整个 GPU |

这张表回答一个常见困惑：为什么不能在 kernel 中间"等所有 block 都算完再继续"？因为 block 之间唯一的同步点就是 kernel 结束——中间想同步，只能拆成两个 kernel（第 6 章会把这个拆分的开销算成数字）。

## 5. warp 与 SIMT：为什么"每个线程一份代码"是假象

**warp（线程束）**是 GPU 调度与执行的基本单位：32 个连续编号的线程组成一个 warp，硬件以 warp 为粒度发射指令。

**SIMT（单指令多线程）**是这种执行模型的名字：一个 warp 的 32 个线程共享一条指令流，同时执行同一条指令，但各自操作自己的数据。它和 CPU 的 SIMD 的区别在于：SIMD 是程序员显式写向量指令，SIMT 是程序员写"每线程一份标量代码"，硬件把它们组织成锁步执行——源码里看起来是 32 份独立控制流，硬件上是一条指令 32 个数据。

### 5.1 分支发散：一个可以手算的吞吐损失

当 warp 内线程走不同分支时，硬件必须把两条路径**串行执行**，用活动掩码（active mask）屏蔽不参与当前路径的线程：

```cuda
if ((threadIdx.x & 1) == 0) {
  x = a + b;        // 偶数线程走这里
} else {
  x = a * b;        // 奇数线程走这里
}
```

手算这笔账：

```text
无发散:  32 个线程全走同一条路径 → 1 条计算指令, 32 个线程都在干活
对半发散: 先执行偶数路径(16 活跃, 16 屏蔽) → 1 条指令, 只干了 16 个线程的活
         再执行奇数路径(16 活跃, 16 屏蔽) → 又 1 条指令
结果: 同样 32 个结果, 用了 2 条指令的发射槽 → 这一段的有效利用率 = 50%
```

**发散本身不一定错，但发散必然消耗额外的发射槽**。这里的关键判断是"发散"发生在 warp 内部：如果整个 warp 32 个线程走同一分支（比如按 `blockIdx` 分支），硬件没有任何额外代价。

### 5.2 `__syncthreads()` 为什么会在分支里死锁

`__syncthreads()`（block 内 barrier）的语义是：**本 block 的所有线程都到达这一行之后，大家才继续**。它和发散的关系是：

```cuda
if (threadIdx.x < 16) {
  __syncthreads();    // 只有 16 个线程能到达
}
// 其余 16 个线程直接跳过了 barrier → 没有任何时刻"全员到齐" → 卡死
```

死锁的根因不是 barrier 本身，而是"**到达条件与线程分支绑定**"。只要 `__syncthreads()` 在 block 内所有线程的必经路径上（不管在哪个分支），就是安全的；一旦它的执行与否依赖发散的条件，block 就永远等不到全员到达。

## 6. 内存空间：五种空间逐个定义

每个内存空间都用四个字段描述：定义、硬件位置、可见性/生命周期、量级延迟。延迟数字是**近似值**，用于建立相对感觉，具体值随架构变化（出处：NVIDIA CUDA Best Practices Guide 与各代架构白皮书）。

| 空间 | 一句话定义 | 硬件位置 | 可见性 / 生命周期 | 延迟量级 |
|---|---|---|---|---|
| Register | 每线程私有的最快存储，编译器把局部变量分到这里 | SM 寄存器文件（每 SM 65536 个 32 位寄存器 = 256 KB） | 仅本线程 / 线程存活 | ~1 cycle |
| Local | 寄存器不够用时的"溢出区"，逻辑上每线程私有 | **物理上在 device 内存（DRAM）**，靠 L1/L2 缓存 | 仅本线程 / 线程存活 | 与全局内存同级 |
| Shared | block 内共享的可编程 SRAM | SM 片上 SRAM（A100 上每 SM 最大 164 KB） | 同 block / block 存活 | ~20-30 cycles |
| Global | 全体线程都能读写的设备内存 | GPU DRAM（HBM） | 全 grid / 分配者释放为止 | ~400-600 cycles |
| Constant | 只读数据空间，warp 内所有线程读同一地址时硬件广播 | 常驻显存 + 专用只读缓存 | 全 grid / 分配者释放为止 | 广播命中时接近 L1 |

这张表本身就是一个常见的错误归因工具：

- 一个变量"声明在 kernel 里"不保证它在寄存器里——寄存器不足时它会悄悄变成 local，而 **local 的物理位置是显存**。一个"局部数组写得多"的 kernel 可能比看起来慢几十倍；
- Shared 不是自动的：数据要显式 `__shared__` 声明、显式从 global 搬进去，编译器不会替你做；
- 判断一个值在哪，最可靠的方法是看编译报告：`nvcc --ptxas-options=-v` 会打印每个 kernel 用了多少寄存器、多少 shared、多少 local（local 非零就是有 spill）。

## 7. 全局内存的合并访问：手算事务数

**合并访问（coalesced access）**的定义：一个 warp 的一次访存，落进尽量少的内存事务里。

硬件侧的事实（Volta 及以后架构）：全局内存以 **32 字节的 sector** 为事务单位，L2 缓存行是 128 字节（= 4 个 sector）。一次 warp 访存的实际代价 = 它碰到的 sector 总数。所以判断一个访存模式贵不贵，就是把"每个线程要的地址"数成"覆盖了几个 sector"。

### 7.1 例 1：连续读 float，1 笔事务

warp 的 32 个线程读连续的 32 个 float（每个 4 B）：

```text
线程 0..31 的地址: base + 0, 4, 8, ..., 124
覆盖字节范围: 128 B = 4 个 sector = 1 条 cache line
事务数: 1(条 cache line 内的 4 个 sector)   ← 理想情况
```

### 7.2 例 2：stride 访问，32 笔事务

同样 32 个线程，但间隔 128 B 取一个 float（比如 `a[i * 32]`）：

```text
线程 k 的地址: base + k*128
每个地址落在不同的 cache line 里 → 每笔事务只用了 32 B 中的 4 B
事务数: 32
有效利用: 4 B / 32 B = 12.5%
```

两次访存请求的数据总量一模一样（都是 128 B），但事务数差了 32 倍——**有效带宽掉到 1/32**。这就是"为什么把循环换个写法速度差几十倍"的最常见答案。

### 7.3 例 3：二维数组按列访问

行宽 128 B 的 `float M[32][32]`，线程 k 读 `M[k][0]`（第 k 行第 0 列）：

```text
线程 k 的地址: base + k*128
与例 2 完全同构 → 32 笔事务
```

按列访问慢，不是因为"列不连续"这个说法本身，而是因为行宽把相邻线程的地址拉开了 128 B。反过来，把数据转置后按行读，就回到例 1 的 1 笔事务。

三例合并成一张表：

| 模式 | 每线程 4 B | 覆盖 sector | 有效利用 |
|---|---|---|---|
| 连续 | 地址连续 | 4（1 条 line） | 100% |
| stride 128B | 地址间隔 128B | 32 | 12.5% |
| 按列访问行宽 128B | 地址间隔 128B | 32 | 12.5% |

## 8. 共享内存与 bank conflict：数出来的冲突

**共享内存 bank** 是共享内存的组织方式：把 shared 内存按地址模 32 划分成 32 个 bank（每个 bank 4 字节宽）。一个 warp 的 32 个线程访问 **32 个不同 bank** 时，一次访问完成；多个线程访问**同一个 bank 的不同地址**时，硬件要把这次访问拆成多拍——这个拆分就是 **bank conflict**。

手算一个例子。声明 `__shared__ float s[32][32]`，线程 k 访问 `s[0][k]`：

```text
s[0][k] 的地址偏移 = 4*k 字节 → bank = (4k / 4) mod 32 = k mod 32
线程 0..31 → bank 0..31 各一个 → 无冲突, 1 拍完成 ✓
```

再看线程 k 访问 `s[k][0]`（按"列"）：

```text
s[k][0] 的偏移 = 128*k 字节 → bank = (128k / 4) mod 32 = (32k) mod 32 = 0
32 个线程全部落在 bank 0 的不同地址 → 32 路冲突, 32 拍 ❌
```

修复手法是给行加 padding：`__shared__ float s[32][33]` 之后，`s[k][0]` 的偏移变成 `132*k`，`bank = (33k) mod 32 = k mod 32`——又回到每个线程一个 bank。**多申请的一列 float 就是 padding 的全部成本**（每行 4 B）。CUTLASS/CuTe/Triton 的 layout 抽象（第 4、5 章展开）本质都是在帮程序员系统化地表达这种"线程—数据"映射，避免手写 padding 时漏算。

一个特例：所有线程读**同一个地址**时，现代架构走广播（broadcast），不算冲突——所以"相同地址必然冲突"是错的，冲突指的是"同 bank、不同地址"。

## 9. 同步三件套：barrier / atomic / fence 各管一件事

这三个词常被混用，其实各管一个不同的问题：

**`__syncthreads()`（barrier）**管的是**到达**：等 block 内所有线程都执行到这一行。它保证"都到了"，不保证"之前写的内存别人一定读得到新值"。

**atomic 操作**管的是**竞争**：读-改-写作为一个不可分割的整体（如 `atomicAdd`）。没有它，32 个线程同时对 `counter += 1` 会互相覆盖——每个线程读到旧值再加 1 写回，最终结果可能只加了几次而不是 32 次。代价是这些操作在硬件上排队执行，争用越激烈越慢。

**fence / 内存序**管的是**可见性**：约束内存操作的观察顺序。弱内存序的 GPU 允许不同地址的读写被重排，`__threadfence()` 等 fence 指令告诉硬件"之前的写必须先被看见"。它不负责让线程互相等待——barrier 和 fence 谁也不能代替谁。完整的并发账（什么时候必须配 fence、原子操作配什么序）在主教材第 28 课展开。

检查一段并发代码，用四个问题代替背 API 名字：

```text
1. 谁写这个数据？         （写者是谁、在哪个 block/warp）
2. 谁读这个数据？         （读者是谁）
3. 读的时刻怎么写者同步？ （barrier / kernel 边界 / event）
4. 如何证明读到的值已可见？（内存序 / fence / 架构保证）
```

## 10. 一次真实观测：用算术强度给 vector_add 定位

本节的数字全部可以在纸上复算。**算术强度（arithmetic intensity）** = 计算量 ÷ 数据搬运量，单位 FLOP/B。

vector_add 每处理一个元素：

```text
计算: 1 次加法 = 1 FLOP
搬运: 读 a(4 B) + 读 b(4 B) + 写 out(4 B) = 12 B
算术强度 = 1 / 12 ≈ 0.08 FLOP/B
```

再算 A100 的 roofline 拐点（算术强度多大时算力才是瓶颈）：

```text
算力墙: FP32 约 19.5 TFLOPS
带宽墙: HBM2e 约 1.55 TB/s
拐点 = 19.5e12 / 1.55e12 ≈ 12.6 FLOP/B
```

vector_add 的 0.08 FLOP/B 比拐点低 150 多倍 → **带宽受限**。所以这个 kernel 的优化方向不是"算得更快"，而是"搬得更少"：合并访问（本章第 7 节）、少读写、或把多个 kernel 合成一个（编译器里叫融合，toycc 的 fusion pass 就是同一思想在计算图层的实现）。反过来，矩阵乘的算术强度随 K 增长（32×32 的强度约 5.3 FLOP/B，主教材第 19 课有完整手算），优化重点才转向计算。

观测的正确姿势：写一个 benchmark，warmup 一轮后取多次中位数，用 `cudaEvent` 计时，然后算两个数——

```text
实测带宽 = 搬运字节数 / 实测时间
健康度   = 实测带宽 / 理论带宽
```

健康度远低于 1 时，第 7 节的事务数手算表就是排查清单的第一页。wall time 本身不回答任何问题，"有效带宽/理论带宽"的比值才是这个 kernel 的健康度。

## 11. 常见错误与归因

| 现象 | 根因 | 定位手段 |
|---|---|---|
| 结果随机错、时好时坏 | 越界访问（`if (i < n)` 缺失）或未初始化内存 | compute-sanitizer 精确报越界位置 |
| `cudaMemcpy` 报 invalid argument | 指针为 NULL（cudaMalloc 失败没检查）或长度算错 | CHECK 宏逐个包住 cuda 调用 |
| host 读到的还是旧值 | kernel 异步执行，没同步就拷贝回来 | 紧跟 `cudaMemcpy` 或显式 `cudaDeviceSynchronize` |
| kernel 挂死 | `__syncthreads()` 在发散分支里 | 检查 barrier 是否在全员必经路径 |
| 带宽掉 30 倍 | 按列 / stride 访问全局内存 | 第 7 节的事务数手算 |
| launch 报"invalid configuration" | grid/block 超限（如每 block 超过 1024 线程） | 查 `cudaGetLastError()` 字符串 |

## 12. 本章检查点

完成以下四项才算通过本章：

1. 取 n = 5000、threads = 128，写出 blocks 的值、画出最后一块的线程覆盖范围，并指出哪些线程越界；
2. 给访存模式"线程 k 读 `a[2*k]`（stride 2 float）"数出它覆盖的 sector 数和有效利用率；
3. 手算 vector_add 的算术强度，并说出一句话的优化结论；
4. 用并发四问（谁写/谁读/何时同步/如何证明可见）解释一个 block 级归约（reduction）里每次 `__syncthreads()` 的必要性。

## 13. 下一步与扩展阅读

本章建立了"线程映射 + 内存访问 + 同步"的手算模型。下一章（GPU 02：CUDA 工具链与 PTX）回答本章遗留的问题：`vector_add.cu` 经过 `nvcc` 之后变成了什么、kernel 参数如何传到 device、PTX 与 SASS 的区别。

- 官方：[CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 第 2 章（编程模型）、第 5 章（性能指南）；本仓库附录 C 有对应的中文导读；
- 手册：[CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) 第 9 章（合并访问）；
- 与本课程的关系：toycc 是计算图层的编译器（IR → Pass → 代码生成），本章处在它下面一层——硬件编程模型。toycc 的 codegen 生成的是 C/Python 代码，CUDA 的 codegen 生成的是 PTX/SASS，两者共享"后端"这个位置。

**导航**：⬅ 上一章：无（本专题第一章）　｜　[下一章](02_cuda_toolchain_ptx.md)（CUDA 工具链与 PTX）➡
