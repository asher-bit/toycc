# 2.2 CUDA Python 入门

本章介绍在 Python 中进行 CUDA kernel 编程。CUDA Python 生态是一个庞大且活跃演进的工具与库生态。本章先介绍其中一些组件，然后用其中几个来示例在 Python 中编写与执行 GPU 代码的方法。

在 Python 中利用 GPU 计算的方式非常多，其中很多并不需要显式编写 GPU kernel。CUDA Python 生态的一些组件提供的函数会把其操作放到 GPU 上完成，开发者无需做任何专门的 GPU 控制或编码。NVIDIA 加速计算中心有一份 Accelerated Python User's Guide，介绍并讨论了让 Python 中 GPU 加速计算得以实现的大量不同库与工具。对于想尽可能快速且轻松地在 Python 中用上 GPU、并且不一定直接写 GPU 代码的使用者，该资源是个不错的起点。

而本章聚焦于对 GPU 的直接控制，以及用 Python 编写在 GPU 上执行的 kernel。本章重点是 Python 中的 CUDA 单指令多线程（SIMT）编程。

## 2.2.1 CUDA Python 生态

CUDA Python 是一组让 Python 中 GPU 计算成为可能的工具与库生态。下面的列表介绍 CUDA Python 的主要部件，并非全部都是本章内容所必需。该列表改编自 CUDA Python github 仓库里的完整清单。

**主组件**——用于 GPU 控制和运行库提供的 GPU 代码

- **cuda.core**——面向 CUDA 控制（如内存与设备管理）的 Python 风格接口。它为 Python 提供了 CUDA Runtime 为 CUDA C++ 所提供的等价物。
- **cuda.compute**——一个 Python 模块，提供由 CUDA Core Compute Library（CCCL）提供的 GPU 加速函数。
- **CuPy**——一个 Python 库，提供 NumPy 例程的 GPU 加速版本，也提供 ndarray 数据容器的 GPU 加速版本。

**编写 Kernel 所用组件**

- **cuda.lang**——一种 Python 领域特定语言（DSL），用 Python 语言的一个子集按 SIMT 编程模型编写 CUDA kernel 与 device 函数。
- **cuda.coop**——一个 Python 模块，提供可由 device 调用（用 cuda.lang）的 CUDA Core Compute Library（CCCL）原语。
- **cuda.tile**——一种 Python 领域特定语言（DSL），按 Tile 编程模型编写 CUDA kernel 与 device 函数。

**其它组件**

- **cuda.pathfinder**——用于在 Python 环境中定位已安装的 CUDA 组件的工具。
- **cuda.bindings**——CUDA 库与工具的低层 Python 绑定，包括 CUDA Driver API、CUDA Runtime API、NVRTC、NVVM 等。cuda.bindings 通过 CUDA Driver 和 CUDA Runtime 组件提供了与 cuda.core 相同的功能。然而 cuda.bindings 是把这些作为 C 语言 API 上的 Python 包装提供的，不是原生 Python 风格接口。

### 2.2.1.1 在 Python 中使用 CUDA 库

CUDA C++ 有一套丰富的库生态，允许在无需直接编写 kernel 或 GPU 代码的情况下进行 GPU 加速。CUDA C++ 在 2006 年问世时库还很少，开发者大多需要自己写 GPU kernel 代码。此后大量库陆续被开发出来，让 C++ 开发者不用怎么写 GPU 代码（甚至完全不写）就能用上 GPU 计算。

CUDA Python 生态则从另一个方向演进：在能用 Python 语法和语义直接写自定义 kernel 的能力出现之前，CuPy 等 Python 库就向 Python 开发者提供了计算与算法的 GPU 加速实现。这些库中很多都为 CUDA C++ 写的 GPU 代码提供了 Python 绑定。

在现代 CUDA 时代，若 GPU 加速库能提供你所需的足够表达力，几乎总是建议优先使用它们。这些库中很多提供了由 GPU 计算专家调优过的实现。当库不可用或不够用时，与 C++ 一样，在 Python 中也可以直接写 GPU kernel 与 device 函数。

### 2.2.1.2 本章范围

虽然开发者应当优先使用库，但本章余下部分介绍当开发者需要在 Python 中写自定义 GPU 代码时如何进行。本章覆盖在 GPU 上编写并运行自定义 Python 代码，与 2.1 节对 C++ 的做法一致——先讲如何指定一个 GPU kernel，然后如何用 cuPy 提供的 GPU 加速 ndarray 在 GPU 上分配内存并通过 CPU 与 GPU 之间通信。

### 2.2.1.3 环境准备

总体上，CUDA Python 生态大部分组件都能在 PyPi 上找到，可用 pip 或任何流行的 Python 包管理器安装。所有包都要求系统装有最新的 NVIDIA 驱动。用 CUDA Python 编写或运行应用一般并不需要 CUDA Toolkit。

不同平台安装与配置 CUDA Python 的信息见 NVIDIA Developer Zone 上的 CUDA Python 页面。

### 2.2.1.4 运行 CUDA Python 应用

CUDA Python 应用——无论使用 CUDA 加速库还是用户自写 GPU 代码——都和常规 Python 应用一样运行。本节中所有示例都从命令行调用 `python3` 来执行名为 `cuda-python-app.py` 的程序，如下所示。

```bash
$ python3 cuda-python-app.py
```

## 2.2.2 Python 中的 SIMT Kernel

如"CUDA 编程模型引言"中所述，在 GPU 上执行、可被主机调用的函数称为 kernel。CUDA 提供两种不同模型：单指令多线程（SIMT）与 CUDA tile。SIMT kernel 被写成由许多并行线程同时运行。该概念在 CUDA Python 和 CUDA C++ 中相同。本章用 SIMT kernel 来介绍 CUDA Python。

### 2.2.2.1 指定 Kernel

在 CUDA Python 中指定 kernel 之前，要先导入 `numba.cuda` 包。通常如下所示。

```python
from numba import cuda
```

这会导入 `numba.cuda` 包，并允许使用该包所提供的 `cuda` 命名空间的组件。

要把一个函数指定为 CUDA Python kernel，在函数定义上一行放上装饰器 `@cuda.jit`，如下所示。

```python
from numba import cuda

@cuda.jit
def function(input_array, output_array):
    ...
```

这会让 kernel 在第一次发射时为当前活跃 GPU 做 JIT 编译。当未指定特定 GPU 时，使用默认 CUDA 设备——本节示例都是这样。

### 2.2.2.2 发射 Kernel

执行一个 kernel 的线程数在 kernel 发射时指定，称为**执行配置**。每次 kernel 调用可以有自己独有的执行配置，例如不同的 block 大小或线程块数。

#### 2.2.2.2.1 Kernel 发射

发射 kernel 时，执行配置写在 kernel 名之后的方括号 `[ ]` 里、函数参数之前。参数顺序与 2.1.2.2.1 节中 C++ 三尖括号记法相同，具体为：

```python
kernel_name[number_of_thread_blocks, threads_per_block](arguments, ...)
```

下面代码片段展示如何在 Python 源文件中定义并调用一个 kernel。

```python
from numba import cuda

@cuda.jit
def my_kernel(input, output):
    ...

## launch the kernel
my_kernel[num_thread_blocks, threads_per_block](in_array, out_array)
```

每 block 的线程数有上限，因为一个 block 的所有线程都驻留在同一 SM 上、必须共享 SM 的资源。当前 GPU 上一个线程块最多可含 1024 个线程。如果资源允许，一个 SM 上可以同时调度多个线程块。

#### 2.2.2.2.2 多维 grid 与线程块

CUDA 中线程块和线程块 grid 可以是一维、二维或三维的。当 grid 或线程块是一维时，发射配置中可以用整数。当线程块或 grid 是二维或三维时，使用二维或三维元组，如下所示为 2D grid 与线程块发射，`gridX` 和 `gridY` 是 grid 的 x、y 维度，`blockX` 和 `blockY` 是 grid 中每个线程块的 x、y 维度。

```python
from numba import cuda

@cuda.jit
def function(input, output):
    ...

## launch the kernel
function[(gridX, gridY), (blockX, blockY)](in_array, out_array)
```

### 2.2.2.3 线程与 grid 索引内建变量

1.2.2.1 节介绍了线程与 grid，2.2.2.2 节展示了如何为 kernel 发射指定 grid 与线程块大小。在 kernel 内部，每个线程可以访问执行配置的参数，以及该线程的索引和其线程块在 grid 内的索引。

kernel 函数内可以访问以下变量来确定线程身份：

- `cuda.threadIdx.[xyz]`：给出线程在自己所在线程块内的索引。线程块内每个线程都有不同索引。
- `cuda.blockDim.[xyz]`：给出线程块的维度，这是 kernel 发射时在执行配置中指定的。
- `cuda.blockIdx.[xyz]`：给出线程块在 grid 内的索引。每个线程块都有不同索引。
- `cuda.gridDim.[xyz]`：给出 grid 的维度，这是 kernel 发射时在执行配置中指定的。

这些变量每个都是三分量向量，含 `.x`、`.y`、`.z` 三个成员。若 kernel 发射时执行配置中未指定某维度，维度默认为 1，索引默认为 0。

`cuda.threadIdx` 和 `cuda.blockIdx` 从 0 开始编号。也就是说，`cuda.threadIdx.x` 取值从 0 到 `cuda.blockDim.x - 1`（含）。`.y` 和 `.z` 在各自维度上同理。

下面是一个简单向量加 kernel 的代码，逐元素相加两个向量。该函数接收三个数组 A、B、C，实现逐元素向量加 `C = A + B`。

```python
# C = A + B vector addition
@cuda.jit
def vecadd(A, B, C):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    C[idx] = A[idx] + B[idx]
```

kernel 先计算该线程在 grid 中的唯一索引。该 kernel 假设它在一维 grid 中以一维线程块发射。`idx` 是从 0 到 N-1 的唯一索引，其中 N 是 grid 中线程总数，即 `N = cuda.gridDim.x * cuda.blockDim.x`。

上面代码块中计算线程索引的模式非常常见，Numba 为此操作提供了一个简写语法：`cuda.grid(n)`，其中 n 是维度数。在上例中，下面这一行

```python
idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
```

可以被更简洁的

```python
idx = cuda.grid(1)
```

替换。该 kernel 一个值得注意的方面是它没有检查对 A、B、C 的越界访问。本章假设它们是 cuPy 创建的 ndarray，cuPy 将在 2.2.3.3 节介绍。使用 cuPy ndarray 时，边界检查由数组类型隐式实现。

## 2.2.3 GPU 计算中的内存

> **注意**
>
> cuPy 等 Python 包通过直接使用 CUDA C++ API 来做 GPU 内存管理，这些 API 见 2.1.3.2 节。多个 Python 包提供了用于控制 GPU 内存分配的包装和工具。本指南只覆盖 cuPy。多数包概念相似，而且除了特别指出的场合外，多数与其 C++ 对应物行为相似。

如 1.2.3 节所述，GPU 有直连的 DRAM。kernel 中要使用的数据数组一般需要在被 kernel 访问前就放到 GPU 的 DRAM 中。在 Python 中，控制数据在内存中的位置——即在 CPU 和 GPU 之间搬运数据——是程序员的责任。这与 2.1.3.2 节介绍的 C++ 中显式内存管理情形相同。

### 2.2.3.1 在 GPU 上实例化数组

CuPy 提供在 GPU 上创建指定类型和维度的 ndarray 对象的函数，也提供在 CPU 和 GPU 之间复制数据的函数。cuPy 中很多函数的签名与 NumPy 中创建 ndarray 的函数相似。下面是几个用 CuPy 在 GPU 内存中创建并填充数组的例子。

```python
import cupy as cp
import numpy as np

## create a matrix of zeros on the GPU
## when a datatype is not specified, float32 is used by default
A_device = cp.zeros((1024, 1024))

## create an array of 2^20 random doubles on the GPU
B_device = cp.random.random((2**20), dtype=np.double)

## create an array of zeros with the same shape and datatype as an existing array
C_device = cp.zeros_like(A)
```

### 2.2.3.2 在主机内存与 GPU 内存之间复制数组

CuPy 也可用于把数据从位于 CPU 内存中的 NumPy ndarray 复制到位于 GPU 内存中的 CuPy 数组。

```python
import cupy as cp
import numpy as np

## Create an array in host memory
A_host = np.zeros((1024, 1024))
## Copy the array to the GPU
A_device = cp.array(A_host)

## Create an array in GPU memory
B_device = cp.random.random((1024, 1024))
## copy the array to host memory
B_host = cp.asnumpy(B_device)
```

### 2.2.3.3 ndarray 对象类型

上一节展示的 ndarray 对象要么在主机内存，要么在 GPU 内存，不会同时在两处。把位于主机的数组作为参数传给 kernel 会导致错误。把位于 GPU 内存的数组传给普通 Python 函数（即非 kernel）也会导致错误。CuPy 不会隐式地在 CPU 和 GPU 之间复制，因为这种复制可能很昂贵，过多数据复制会损害性能。因此 CuPy 要求程序员有意识地选择何时在 CPU 和 GPU 间复制数据。

在 GPU kernel 中使用 ndarray 类型的一个好处是数组自带其各维度的尺寸。如 2.2.2.3 节所示，边界检查自动完成，当所需线程总数略小于执行块或 grid 总尺寸时，kernel 代码不需要检查越界访问。

## 2.2.4 同步 CPU 与 GPU

与 C++ 一样，CUDA Python 中 kernel 发射相对于主机线程是异步的。即 kernel 发射后主机代码在 CPU 上继续执行，并不保证 kernel 已经完成甚至启动。要保证 GPU kernel 已执行完成，主机线程必须以某种形式与 GPU 同步。

最简单的同步形式是同步整张 GPU。这种设备级同步是 CUDA 驱动提供的操作，由 cuPy 和 numba.cuda 都以 `synchronize()` 方法暴露给 Python。

```python
import cupy as cp
from numba import cuda

# ...

## Wait on host thread for all pending GPU work to complete
## this uses the interface provided by cupy
cp.synchronize()

## Wait on host thread for all pending GPU work to complete
## this uses the interface provided by numba.cuda
cuda.synchronize()
```

设备级同步会阻塞主机线程直到 GPU 上此前发出的所有工作都完成。更细粒度的同步可用 CUDA stream，见 2.5 节。在 Python 中使用 stream 时，推荐做法是用 cuda.core 创建 CUDA stream，并只在需要时对特定 stream 同步。

## 2.2.5 串起来

下面源码清单给出了最常见的第一个 GPU kernel——并行向量加的 Python 版本。

```python
import numpy as np
from numba import cuda
import cupy as cp


## Defines a CUDA kernel to perform C = A + B vector addition
@cuda.jit
def vecadd(A, B, C):
    work_index = cuda.grid(1)
    C[work_index] = A[work_index] + B[work_index]


# note that vector size is not a power of 2 nor a multiple of the block_size defined below
vector_size = 2**24 + 11

device = cp.cuda.Device()
## Create device arrays of uniform random float32 values as input, and an array of zeros
## as the result vector
a = cp.random.uniform(-1, 1, vector_size)
b = cp.random.uniform(-1, 1, vector_size)
c = cp.zeros_like(a)

block_size = 256
grid_size = int(np.ceil(vector_size/block_size))
vecadd[grid_size, block_size](a, b, c)

## synchronize the CPU thread and the GPU to ensure that the kernel has completed
## this is included to illustrate good practices, even though the copy below would implicitly wait for
## the kernel to complete
device.synchronize()

## Copy all 3 arrays to the CPU as ndarrays
a_np = cp.asnumpy(a)
b_np = cp.asnumpy(b)
c_np = cp.asnumpy(c)

## Perform the copy on the CPU to verify the answer
expected = a_np + b_np

## Test that the answer is correct, within floating point epsilon
np.testing.assert_array_almost_equal(c_np, expected)

## The assert will print diagnostics and abort
## so this only prints if the assertion passes
print("Test succeeded")
```

本例中 A、B 输入数组由 CuPy 在 GPU 上创建并初始化为随机值。它们在代码末尾被复制到 CPU，仅是为了让 CPU 也做一次向量加，并验证 CPU 与 GPU 答案一致。

## 2.2.6 CUDA Python 中的错误检查

任何影响 GPU 的操作——从内存分配和复制到 kernel 发射——都可能引发错误条件。如 2.1.7 节对 C++ 所示，确保在与 GPU 的交互过程中没有发生错误是最佳实践。

在 Python 中，CUDA 错误会抛出异常，若不被捕获就会终止程序。可以用普通 Python 语法捕获异常。下例与上面的向量加相同，但故意加了一个错误：每 block 的线程数 2048 大于任何当前 GPU 能运行的上限。这会让 kernel 无法发射，并抛出一个异常，由这段代码捕获。

```python
import numpy as np
from numba import cuda
import cupy as cp


## Defines a CUDA kernel to perform C = A + B vector addition
@cuda.jit
def vecadd(A, B, C):
    work_index = cuda.grid(1)
    C[work_index] = A[work_index] + B[work_index]


try:
    vector_size = 2**24 + 11

    device = cp.cuda.Device()
    a = cp.random.uniform(-1, 1, vector_size)
    b = cp.random.uniform(-1, 1, vector_size)
    c = cp.zeros_like(a)

    ## this block size is too large for any current GPUs
    block_size = 2048
    grid_size = int(np.ceil(vector_size/block_size))
    # Error: launching kernel with invalid block size
    vecadd[grid_size, block_size](a, b, c)

    device.synchronize()
    print("Test did not encounter any errors")

except Exception as e:
    print(f"Exception occurred: {e}")
```

运行这段代码会让错误被捕获并显示，如下所示：

```bash
$ python3 vecadd_error.py
Exception occurred: CUDA_ERROR_INVALID_VALUE: This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.
```

程序在捕获异常后正常退出。如果不用 `try:` 和 `except:` 直接运行这段代码，它会异常退出并向控制台 dump 一份 traceback，应显示同样的错误。

---

[← 上一章 2.1 CUDA C++ 入门](04_intro_cuda_cpp.md) ｜ [返回附录 C 首页](README.md)