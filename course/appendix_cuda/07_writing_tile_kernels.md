# 2.4 编写 Tile Kernel

CUDA Tile 提供一种与前几章所述单指令多线程（SIMT）模型不同的编写 GPU kernel 代码方式。Tile 编程允许程序员以不同方式表达并行性，把最低层并行性交给编译器和内建操作。由此，tile 提供了一种更简单的方式来访问 NVIDIA GPU 近期的性能特性，例如张量内存加速器（TMA）单元和 Tensor Core。

CUDA Tile 编程在 Python 中通过 cuTile Python 包 `cuda.tile` 可用。

CUDA Tile C++ 自 CUDA Toolkit 13.3 版起可用。

围绕 tile kernel 的应用代码——如分配设备内存、在主机与设备间搬运数据、编排 kernel 发射——与前几章为 SIMT kernel 所述完全相同。tile kernel 操作用标准 CUDA API 分配的全局内存，其结果复制回主机的方式也相同。唯一变化的是程序员在 kernel 内部写的代码。

在 SIMT kernel 中，程序员以单个线程为单位思考：计算全局线程索引、装载该线程的元素、对它们执行操作、存储结果。在 tile kernel 中，程序员以整个 block 为单位思考：装载含许多元素的 tile、对整个 tile 执行操作、存储结果。编译器负责把 tile 操作映射到每个 block 的硬件线程——这是 SIMT 程序员显式处理的事务。

本章专门聚焦这一差异：如何编写 kernel 入口点及其内部的 tile 操作。每个模式都同时用 CuTile Python（`cuda.tile`）和 CUDA Tile C++（`cuda::tiles`）示范——两者共享同一编译器后端（CUDA Tile IR），因此执行语义相同。

按约定，两种语言都把 tile API 别名为 `ct`。

- Python 中 `import cuda.tile as ct`
- C++ 中 `namespace ct = cuda::tiles`

Python 中 tile API 位于 `cuda.tiles` 模块，如上所示导入。

C++ 中 tile API 位于 `cuda::tiles` 命名空间，由 `cuda_tile.h` 头文件暴露。

```cpp
#include "cuda_tile.h"
namespace ct = cuda::tiles;
```

下面片段中的 `ct.` / `ct::` 前缀指代你所读语言的 tile API。

## 2.4.1 Kernel 与函数声明

tile kernel 是 GPU 入口点，在发射 grid 中每个 block 执行一次。tile function 可从 tile kernel 或另一个 tile function 调用，但自身不是入口点。与 SIMT kernel 一样，tile kernel 不能直接从主机代码调用——必须被发射。

CUDA Tile C++ 中：

- `__tile_global__` 是 tile 版的 `__global__`，标记一个 tile kernel 入口点
- `__tile__` 是 tile 版的 `__device__`，指示一个应被编译为 GPU 版且可从其它 `__tile__` 或 `__tile_global__` 函数调用的函数

数组和标量参数的传递方式与 SIMT kernel 相同。Tile 代码与 SIMT 代码可以共存：单个 `.cu` 文件可同时定义 `__tile_global__` 和 `__global__` kernel，单个主机程序可发射两者。

> **注意**
>
> 目前 `__tile__` 函数不能从 `__global__` 或 `__device__` 函数调用。类似地，`__device__` 函数不能从 `__tile_global__` 或 `__tile__` 函数调用。此限制可能在未来 CUDA 版本中解除。

cuTile Python 中：

- `@ct.kernel` 装饰器把一个函数标记为 tile kernel 入口点
- `@ct.function` 装饰器把一个函数标记为可从 tile kernel 或另一个 tile function 调用

实践中，从 kernel 调用的任何函数都会被自动编译为 tile 代码，因此 `@ct.function` 装饰器是可选的。数组参数接受任何暴露 DLPack 或 CUDA Array Interface 的设备驻留数组，例如 PyTorch 张量和 CuPy 数组。标量参数直接传递。

**C++**

```cpp
#include "cuda_tile.h"

// Tile kernel entry point. Cannot be called directly; must be launched.
__tile_global__ void my_kernel(float* a, float* b, float* c) {
    ...
}

// Tile function. Callable from tile kernels and tile functions.
__tile__ float helper(float x, float y) {
    return x + y;
}
```

**Python**

（Python 中用 `@ct.kernel` 和 `@ct.function` 装饰器，逻辑同上。）

## 2.4.2 发射 Kernel

tile kernel 在 tile block 组成的 grid 上发射，正如 SIMT kernel 在线程块组成的 grid 上发射。程序员指定 grid 形状，最多三维。从程序员视角，每个 tile block 由单个逻辑线程执行。block 内的并行性由编译器管理。

C++ 中，tile kernel 复用 SIMT 中熟悉的三尖括号发射语法。第一个尖括号参数是 grid 形状（tile block 数）。第二个参数是 SIMT 中的每 block 线程数；对 tile kernel，编译器内部决定线程数，第二个参数必须为 1。tile kernel 也是普通 CUDA kernel，故可通过运行时已有的 `cudaLaunchKernel` 和 `cudaLaunchKernelEx` API 以同样的 grid、1 配置发射。这在把 tile kernel 集成到已用那些 API 驱动发射的代码库中时有用。

Python 中，`ct.launch` 取四个位置参数：一个 CUDA stream、一个指定每维 tile block 数的 grid 元组、kernel 对象，以及 kernel 参数元组。

**C++**

```cpp
my_kernel<<<dim3(num_blocks_x, num_blocks_y), 1>>>(a, b, c);  // second arg must be 1
```

**Python**

（Python 中用 `ct.launch(stream, grid, kernel, args)`。）

### 2.4.2.1 Grid 尺寸模式

一种常见模式是发射足够多的 block 覆盖整个数组，包括最后一个可能在一维或多维上超出数组大小的 block。

**C++**

```cpp
int num_blocks = (N + tile_size - 1) / tile_size;   // ceil division -> covers partial tail
kernel<<<num_blocks, 1>>>(in, out, N);
```

**Python**

（Python 中同样按上取整计算 grid 尺寸。）

处理数组大小不被 tile 大小整除的情形在 2.4.6 节的几个小节中讨论。

## 2.4.3 查询 Block 位置

每个 block 需要知道自己在 grid 中的位置，以便确定处理哪部分数据。SIMT 中程序员组合 `blockIdx` 与 `threadIdx` 计算全局线程索引。Tile 代码中只需 block 索引。编译器处理 block 内所有线程级索引。

C++ 中，`ct::bid()` 返回含三维 block 索引的 `uint3`。`ct::num_blocks()` 返回含每维 block 总数（由 kernel 发射参数决定）的 `dim3`。各分量通过 `.x`、`.y`、`.z` 访问。

Python 中，`ct.bid(axis)` 返回当前 block 沿指定轴（0、1 或 2）的索引，作为 int32 标量。`ct.num_blocks(axis)` 返回沿该轴的 block 总数——对边界检查和循环计数有用。

**C++**

```cpp
#include "cuda_tile.h"

__tile_global__ void my_kernel(float* a, float* b, float* c) {
    namespace ct = cuda::tiles;
    int bid_x = ct::bid().x;          // block index along .x
    int bid_y = ct::bid().y;          // block index along .y
    int num_x = ct::num_blocks().x;   // total blocks along .x
}
```

**Python**

（Python 中用 `ct.bid(0)`、`ct.num_blocks(0)`。）

## 2.4.4 创建 Tile

block 身份确定后，下一个问题是 tile kernel 实际操作什么。那就是 tile：固定大小、标量元素的多维数组，其形状和元素类型在编译期已知。tile 的每一维必须是 2 的幂。Tile 具有值语义：拷贝一个 tile 会拷贝其元素，两份拷贝完全独立。尽管如此，拷贝很便宜，因为编译器控制 tile 在硬件内部如何表示。程序员不为 tile 分配或释放内存。

实践中，tile 通过从数组加载数据（见"Tile 空间 Load 和 Store"）或用产生以指定模式填充的 tile 的工厂函数来创建。

C++ 中 tile 类型是显式的：`ct::tile<T, ct::shape<dims...>>`，其中 `T` 是元素类型，`ct::shape<dims...>` 把维度作为模板参数编码（整数值是每轴的编译期尺寸）。例如 `ct::tile<float, ct::shape<8>>` 是 8 个 float 的一维 tile，`ct::tile<float, ct::shape<4, 4>>` 是 4×4 float tile。因为形状是类型的一部分，所以总在编译期已知。

工厂函数取完整 tile 类型（下面记为 Tile）作为模板参数：

- `ct::zeros<Tile>()` 和 `ct::ones<Tile>()`——填充零或一的 tile。
- `ct::full<Tile>(val)`——每个元素都为 `val` 的 tile。
- `ct::iota<Tile>()`——含 `(0, 1, ..., N-1)` 的 tile，N 是 tile 的大小。

本章 C++ 示例用 `using` 别名（如 `using f32x4x4 = ct::tile<float, ct::shape<4, 4>>`）使调用处 tile 类型可读。

Python 中，tile 工厂的 shape 元组和 dtype 参数都是编译期值。Python 字面量（如 `(64, 64)` 和 `ct.float32`）自然满足。也可用 Constant 标注的 kernel 参数提供，如下面 Python `Constant[T]` 所示。结果 tile 暴露 `.shape`、`.dtype`、`.ndim` 属性，反映其编译期属性。

工厂函数为：

- `ct.zeros(shape, dtype)` 和 `ct.ones(shape, dtype)`——填充零或一的 tile。
- `ct.full(shape, fill_value, dtype)`——任意常量值的 tile。
- `ct.arange(size, dtype=...)`——含 `[0, 1, ..., size-1]` 的一维 tile。

**C++**

```cpp
#include "cuda_tile.h"

__tile__ void factories() {
    namespace ct = cuda::tiles;

    using i32x8   = ct::tile<int,   ct::shape<8>>;      // 1-D: 8 ints
    using f32x4x4 = ct::tile<float, ct::shape<4, 4>>;   // 2-D: 4x4 floats

    auto z      = ct::zeros<f32x4x4>();       // all zeros
    auto o      = ct::ones<f32x4x4>();        // all ones
    auto filled = ct::full<f32x4x4>(3.14f);   // all 3.14
    auto seq    = ct::iota<i32x8>();          // {0, 1, 2, 3, 4, 5, 6, 7}
}
```

**Python**

（Python 中用 `ct.zeros(shape, dtype)` 等。）

## 2.4.5 编译期常量

tile 编译器为每种 tile 形状、数据类型和其它结构参数的组合生成专用机器码。因此影响生成代码的值必须在编译期已知。即 tile 的形状和数据类型必须在编译期已知。"创建 Tile" 用字面量指定 tile 形状和数据类型：`ct.zeros((64, 64), dtype=ct.float32)` 和 `ct::tile<int, ct::shape<8>>`。

形状也可通过 kernel 接口作为编译期已知值传递，如下面几节所示。

### 2.4.5.1 Python Constant[T]

kernel 参数上的 `ct.Constant[T]` 类型提示把它标记为常量内嵌。这意味着 kernel 内对该参数的每次使用都如同在其位置写入了字面值。类型参数可选，不带类型参数的 `ct.Constant` 内嵌任意类型的常量。`ct.Constant` 最常用于整数——`ct.Constant[int]`——驱动 tile 形状和循环边界的参数。

**Python**

```python
import cuda.tile as ct

@ct.kernel
def my_kernel(TILE: ct.Constant[int]):
    # TILE is constant-embedded: wherever TILE appears, the compiler sees its
    # literal value (e.g., 128) and generates specialized code. Here TILE drives
    # the shape of a factory-built tile.
    zeros = ct.zeros((TILE,), dtype=ct.float32)
```

### 2.4.5.2 C++ integral_constant 与 _ic 字面量

CUDA Tile C++ 中编译期值通过 `ct::integral_constant` 表达，这是一个数值编码在类型本身中的类型。`ct::literals` 命名空间的 `_ic` 字面量提供简洁简写：`0_ic` 产生 `ct::integral_constant<0>` 值。

接受编译期值的 API 既接受非类型模板参数（NTTP）形式，也接受 `_ic` 字面量形式。例如 `ct::cat` 沿给定维度连接两个 tile，该维度必须在编译期已知。下面两行以同一编译期轴调用 `ct::cat`，仅在编译期值写在哪里不同：

**C++**

```cpp
#include "cuda_tile.h"

__tile__ void concat_demo() {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    using T = ct::tile<int, ct::shape<4, 8>>;
    T lhs = ct::full<T>(0);
    T rhs = ct::full<T>(1);

    auto a = ct::cat<0>(lhs, rhs);     // NTTP form
    auto b = ct::cat(lhs, rhs, 0_ic);  // _ic form
}
```

`_ic` 字面量还有一个常见出现处。`ct::extents` 和 `ct::shape` 各自都有 NTTP 形式（如 `ct::extents<std::uint32_t, 4, 8>`）和大括号形式。与 NTTP 形式不同，大括号形式接受运行期值——所以当一维或多维仅在发射时已知时，用大括号形式：编译期维度用 `_ic` 字面量、运行期维度用普通变量。tile 空间 API 如 `ct::tensor_span` 和 `ct::partition_view`（见"Tile 空间 Load 和 Store"）用此形式包裹此类数组：

**C++**

```cpp
auto shape2d = ct::extents{8_ic, length};  // 8 is compile-time; length is runtime
```

`_ic` 字面量是编译期值在值形式 API 参数要求时的统一简写，如 `ct::cat` 的维度或 extents / shape 的分量。

## 2.4.6 加载与存储 Tile

如 1.2.2.3.1 节首次介绍，CUDA Tile 编程模型中有两个关键内存对象：tile 和数组。数组是全局内存中的多维元素容器，对 tile kernel 的所有 block 可见。tile 也是多维元素容器，但局部于单个 tile 代码 block。tile 常是数组元素的子集。本节讨论从数组加载到 tile 以便在 tile kernel 中使用，以及把 tile 存回数组。

后面小节覆盖两种加载与存储 tile 的方法：

- "Tile 空间 Load 和 Store" 覆盖用 tile 空间索引的 load 和 store，使用 view 对象规定数组元素如何映射到 tile 的可预测模式
- "Gather 和 Scatter" 覆盖用索引 tile 或指针 tile 指示数组中哪个元素是 tile 元素在加载或存储时的源或目标的 load 和 store

性能提示：tile 空间 load 可由编译器在支持的硬件上下降为张量内存加速器（TMA），比逐元素 gather 快得多。（C++ 侧另见 2.4.12 节。）

程序员必须决定 load 时越界元素取什么值。Python 中以及 C++ 中使用 masked 变体时越界写被静默丢弃。

### 2.4.6.1 Tile 空间 Load 与 Store

tile 空间 load 创建一个 view 对象，指定数组如何被划分为 grid 的 tile 大小区域。这种映射称为 tile 空间，tile kernel 可以用 tile 空间索引一次加载或存储一个区域。

tile 空间 load 的核心是数组的 tiled view，指定数组元素如何映射到指定大小的 tile。图 19 所示的 tiled view 是 partition view——一种以指定大小的不重叠 tile、tile 间无间隙的 tile 空间。

![图 19 partition view 的 tile 空间索引](images/figure19-tile-space-indexing.png)

> 图 19 partition view 的 tile 空间索引。形状 (10, 16) 的二维数组被划分为形状 (2, 4) 的 tile，产生形状 (5, 4) 的 tile 网格。每格显示其 tile 空间索引 (i, j)。tile 空间索引 (1, 2) 处的高亮区域覆盖元素索引 (2, 8) 到 (3, 11)。

当数组维度不能被 tile 完美整除时，在一维或多维上越过数组边界的 tile 会被部分填充。程序员可指定加载这些 tile 时的行为，在 2.4.6.1.3 节介绍。

> **注意**
>
> 这里示例和描述用 partition view 来图示 tile 空间 load 和 store，因为这是 CUDA Tile 代码中首个支持的 view 类型。其它 view 类型预计在后续 CUDA Tile 版本中加入。

#### 2.4.6.1.1 Partition View Load 与 Store

结构化 tile 空间 load 是在全局内存和 tile 之间搬运数据的首选方式。kernel 必须先创建一个定义 tile 空间的 view 对象，然后按 tile 空间索引一次加载或存储一个 tile。

C++ 中 partition view 分两步构造：

- `ct::tensor_span`——把裸指针与 `ct::extents` 配对，给指针多维结构。
- `ct::partition_view`——把 span 划分为固定大小 tile 的 grid，暴露在 tile 空间坐标中操作的 `.load(idx...)` / `.store(tile, idx...)` 方法。

Python 中，`Array.tiled_view(tile_shape)` 返回把数组划分为给定形状 tile 的 `TiledView`。该 view 暴露取 tile 空间索引的 `.load(index)` / `.store(index, tile)` 方法，直接对应 C++ `partition_view`。

> **注意**
>
> 本章 C++ 示例代码用 `__restrict__` 标注指针参数并在 kernel 顶部附近调用 `ct::assume_aligned(ptr, 16_ic)`。这些是 2.4.12 节中详述的重要性能标注。数字字面量上的 `_ic` 后缀（如 `128_ic`、`8_ic`）把它们标记为编译期常量，如"编译期常量"所介绍。

**C++**

```cpp
__tile_global__ void vec_add(float* __restrict__ a, float* __restrict__ b, float* __restrict__ out) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    a   = ct::assume_aligned(a,   16_ic);
    b   = ct::assume_aligned(b,   16_ic);
    out = ct::assume_aligned(out, 16_ic);

    // Step 1: attach a shape to each raw pointer. 128_ic marks 128 as a compile-time constant.
    auto aSpan = ct::tensor_span{a,   ct::extents{128_ic}};
    auto bSpan = ct::tensor_span{b,   ct::extents{128_ic}};
    auto oSpan = ct::tensor_span{out, ct::extents{128_ic}};

    // Step 2: partition each span into a tile space of fixed 8-element tiles.
    auto aView = ct::partition_view{aSpan, ct::shape{8_ic}};
    auto bView = ct::partition_view{bSpan, ct::shape{8_ic}};
    auto oView = ct::partition_view{oSpan, ct::shape{8_ic}};

    int  bx    = ct::bid().x;             // this block's tile-space index along .x
    auto aTile = aView.load(bx);          // pick the bx-th tile of a
    auto bTile = bView.load(bx);
    oView.store(aTile + bTile, bx);       // write the tile back at the bx-th position of out
}
```

**Python**

（Python 中用 `Array.tiled_view(shape)` 构造 view，逻辑同上。）

#### 2.4.6.1.2 Python 一次调用 Load 与 Store

Python 另提供一次调用形式，在每次 load 和 store 时内联取 tile 形状，无需显式 view 对象。`ct.load(array, index, shape)` 在给定 tile 空间索引读取给定形状的 tile。`ct.store(array, index, tile)` 是对应写。

`ct.load`/`ct.store` 和 `Array.tiled_view` 表达同一 tile 空间访问模式。区别在于 tile 形状存在哪。用 `Array.tiled_view` 时 tile 形状一次性绑定到 view 对象。用 `ct.load`/`ct.store` 时 tile 形状在每次调用内联提供。当同一划分在多次 load 和 store 中复用时优先用 `tiled_view`。当单次一次性 load 更简洁时用 `ct.load`/`ct.store`。

**Python**

```python
@ct.kernel
def vec_add(a, b, c, TILE: ct.Constant[int]):
    bid = ct.bid(0)                                    # this block's tile-space index along axis 0
    a_tile = ct.load(a, index=(bid,), shape=(TILE,))   # (index, shape) = pick the bid-th TILE-sized region of a
    b_tile = ct.load(b, index=(bid,), shape=(TILE,))
    ct.store(c, index=(bid,), tile=a_tile + b_tile)    # write the tile back to the bid-th region of c
```

#### 2.4.6.1.3 Tile 空间边界处理

C++ 中 `partition_view` 提供 unmasked 与 masked 变体：

- `.load(idx...)` / `.store(tile, idx...)` 假设 tile 完全在界内。部分越界访问是未定义行为。
- `.load_masked(idx...)` / `.store_masked(tile, idx...)` 安全处理部分边缘 tile。
  - `.load_masked()` 默认以零填充越界位置；可选择其它填充模式（如对 float tile 用 NaN）。
  - `.store_masked()` 静默丢弃越界写。

当数组被 tile 大小完美整除时优先用 unmasked load 和 store 变体。当必须处理边界条件时，即便 tile 完全填充也可用 masked 变体。

这也是本指南中首个数组维度为运行期值的 C++ 示例。`ct::extents{N}` 接受运行期维度，`ct::extents` 支持编译期（`_ic`）和运行期值的任意组合，因此 span 和 partition view 可包裹仅在 kernel 发射时已知大小的数组。

Python 中，`ct.load` 接受 `padding_mode` 参数控制越界元素取何值。两种常用模式：

- `PaddingMode.ZERO`——越界元素以零填充。
- `PaddingMode.UNDETERMINED`（默认）——越界元素值由实现决定。当程序员知道 tile 完全在界内时适用。

对于 store，`ct.store` 总是静默丢弃越界位置的写，不需要 `padding_mode` 参数。同样规则适用于 `tiled_view`——其在 view 创建时固定 `padding_mode`。

**C++**

```cpp
__tile_global__ void edge_safe(float* __restrict__ in, float* __restrict__ out, int N) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    in  = ct::assume_aligned(in,  16_ic);
    out = ct::assume_aligned(out, 16_ic);

    // ct::extents{N} uses a runtime dimension; 128_ic stays compile-time.
    auto inView  = ct::partition_view{ct::tensor_span{in,  ct::extents{N}}, ct::shape{128_ic}};
    auto outView = ct::partition_view{ct::tensor_span{out, ct::extents{N}}, ct::shape{128_ic}};

    int  bx   = ct::bid().x;
    auto tile = inView.load_masked(bx);    // masked load: OOB lanes default to 0
    outView.store_masked(tile, bx);        // masked store: OOB writes silently discarded
}
```

**Python**

（Python 中用 `PaddingMode.ZERO` 做 load、`ct.store` 自动丢弃越界写。）

C++ kernel 内 `.load_masked()` 和 `.store_masked()` 处理部分边缘 tile。Python kernel 内 load 上 `PaddingMode.ZERO` 保证部分边缘 tile 以零填充，`ct.store` 静默丢弃越过数组边界的写。完整填充模式、masking 选项和填充值集合见各语言 API 参考（CUDA Tile C++ view 填充、cuTile Python 填充模式）。

对完全在数组之外的 tile 做加载或存储是未定义的。这里讨论的边界处理只适用于在一维或多维上部分越界的 tile。

### 2.4.6.2 Gather 与 Scatter

"Tile 空间 Load 和 Store" 中的 tile 空间 load 用 partition view，它定义数组的规则、块对齐划分。当访问模式不规则或依赖数据——如查找表或置换——gather 和 scatter 操作允许通过任意索引或地址从数组的不均匀、不连续元素加载和存储 tile。

gather 和 scatter 操作在 C++ 和 Python 中略有不同：

- Python 用传递给 `ct.gather()` / `ct.scatter()` 的整数索引 tile，带内建边界检查。
- C++ 用传递给 `ct::load()` / `ct::store()` 的指针 tile，masked 变体 `ct::load_masked()` 和 `ct::store_masked()` 接受布尔 mask tile 处理数组边界处的 tile。

C++ 中 gather 和 scatter 通过形成一个指针 tile——每元素一个指针——并把指针 tile 传给 `ct::load()` 或 `ct::store()` 工作。标量指针与整数 tile 之间的算术按元素执行，产生指针 tile。这是 C++ 中构造 gather/scatter 索引 tile 的标准惯用法。

Python 中，`ct.gather` 加载索引 tile 中每个索引处的元素。边界检查默认开启：越界索引返回填充值（默认零，可通过 `padding_value=` 配置），可用 `check_bounds=False` 禁用。`ct.scatter` 每个索引存一个值；越界写静默丢弃。

**C++**

```cpp
__tile_global__ void vec_add_gather(int* __restrict__ a, int* __restrict__ b, int* __restrict__ out) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using i32x8 = ct::tile<int, ct::shape<8>>;

    a   = ct::assume_aligned(a,   16_ic);
    b   = ct::assume_aligned(b,   16_ic);
    out = ct::assume_aligned(out, 16_ic);

    int bx       = ct::bid().x;
    auto offsets = 8 * bx + ct::iota<i32x8>();   // element-level offsets, one per lane

    // scalar pointer + int tile = tile of pointers (one pointer per offset).
    auto aPtrs = a + offsets;
    auto bPtrs = b + offsets;

    auto aTile = ct::load(aPtrs);                // gather: one load per pointer
    auto bTile = ct::load(bPtrs);
    ct::store(out + offsets, aTile + bTile);     // scatter: one store per pointer
}
```

**Python**

（Python 中用 `ct.gather` / `ct.scatter` 加整数索引 tile。）

#### 2.4.6.2.1 Gather 与 Scatter 边界处理

"Gather 和 Scatter" 中 gather/scatter 操作的边界处理遵循不同规则。

Python 中 `ct.gather` 和 `ct.scatter` 默认边界安全。越界读返回填充值（默认零），越界写静默丢弃。当能证明每个索引都在范围内时边界检查可禁用；这样做使越界访问成为未定义行为。见 API 参考中的可选 mask 和填充值调节（CUDA Tile C++ load 操作、cuTile Python load/store 操作）。

C++ 中边界检查不是自动的。程序员构造布尔 mask（如把偏移与数组长度比较）并传给 `ct::load_masked` 或 `ct::store_masked`：

**C++**

```cpp
__tile_global__ void gather_safe(int* __restrict__ arr, int* __restrict__ out, int N) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using i32x8 = ct::tile<int, ct::shape<8>>;

    arr = ct::assume_aligned(arr, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    int bx       = ct::bid().x;
    auto offsets = 8 * bx + ct::iota<i32x8>();   // element-level offsets, one per lane
    auto mask    = offsets < N;                  // boolean tile: true where the offset is in-bounds

    auto ptrs = arr + offsets;                   // tile of pointers, one per offset
    auto tile = ct::load_masked(ptrs, mask, 0);  // masked lanes get the pad value 0
    ct::store_masked(out + offsets, tile, mask); // masked lanes are skipped on the store
}
```

## 2.4.7 控制流

从程序员视角，tile kernel 每 block 走单一控制流路径。条件和循环边界中的标量值驱动控制流，体中的 tile 操作由编译器分配到硬件线程。

并非每种控制流结构都支持。例如 tile 代码中不允许从循环内部返回。完整限制列表见各语言 API 参考（CUDA Tile C++ 一般原则、cuTile Python 控制流）。

### 2.4.7.1 循环

一种常见模式是遍历数组中的 tile，依次处理每个。

C++ 中 `ct::irange` 是一个前向范围，表示从下界到但不包括上界、以可选步长分隔的递增整数序列。用 `ct::irange` 给编译器提供关于迭代边界的结构化信息，可用于更好地优化生成代码。要应用该优化，循环变量必须通过对 `ct::irange` 的 range-for 表达式绑定。

Python 中 tile 代码支持内建 `range()`、`for`、`while` 和嵌套循环。

step 参数必须严格为正；不支持负步长 range。

下面单 block kernel 对 1D 数组所有 tile 求和：

**C++**

```cpp
__tile_global__ void tile_sum(float* __restrict__ arr, float* __restrict__ out, int num_tiles) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using f32x8 = ct::tile<float, ct::shape<8>>;

    arr = ct::assume_aligned(arr, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto inView  = ct::partition_view{ct::tensor_span{arr, ct::extents{8 * num_tiles}},
                                      ct::shape{8_ic}};
    auto outView = ct::partition_view{ct::tensor_span{out, ct::extents{8_ic}},
                                      ct::shape{8_ic}};

    auto acc = ct::full<f32x8>(0.0f);
    // range-for over ct::irange gives the compiler structured iteration bounds.
    for (auto k : ct::irange(0, num_tiles)) {
        auto tile = inView.load(k);
        acc = acc + tile;                               // accumulate the k-th tile into acc
    }
    outView.store(acc, 0);                              // write the final result as the 0-th tile of out
}
```

**Python**

（Python 中用内建 `range` 和 `for`。）

### 2.4.7.2 条件

标准 if/else 条件正常工作。因为每个 block 走单一控制流路径，warp 内分支发散的考量不适用于 tile kernel。

**C++**

```cpp
__tile_global__ void conditional_load(float* __restrict__ arr, float* __restrict__ out, int N) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using f32x8 = ct::tile<float, ct::shape<8>>;

    arr = ct::assume_aligned(arr, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto inView  = ct::partition_view{ct::tensor_span{arr, ct::extents{N}}, ct::shape{8_ic}};
    auto outView = ct::partition_view{ct::tensor_span{out, ct::extents{N}}, ct::shape{8_ic}};

    int bx   = ct::bid().x;
    int nb_x = ct::num_blocks().x;

    auto tile = ct::full<f32x8>(0.0f);    // default for the last-block branch
    // Scalar condition -> one control-flow path per block; no divergence to reason about.
    if (bx < nb_x - 1) {
        tile = inView.load(bx);           // all blocks except the last
    }
    outView.store_masked(tile, bx);       // masked to handle a potentially partial final tile
}
```

**Python**

（Python 中 if/else 同样工作。）

## 2.4.8 逐元素算术与广播

Tile 支持标准逐元素算术。当两个操作数形状兼容但不同时，较小的在操作执行前广播以匹配。

### 2.4.8.1 广播

广播遵循 NumPy 语义：标量在 tile 上复制，单例维度（长度 1）被拉伸以匹配另一操作数的对应维度，较低 rank 的操作数通过把缺失的前导维度视为单例对齐到较高 rank 操作数的尾随维度。若两个对应维度都非单例且不等，则操作非法。

下面示例在一次加法中同时用了单例拉伸和 rank 提升：形状 8x2 的 rank-2 tile 被提升为 1x8x2，然后与形状 4x1x2 的 rank-3 tile 广播到共同形状 4x8x2。

**C++**

```cpp
auto x = ct::iota<ct::tile<int, ct::shape<8, 2>>>();      // 8x2   (rank 2)
auto y = ct::iota<ct::tile<int, ct::shape<4, 1, 2>>>();   // 4x1x2 (rank 3)
auto z = x + y;                                           // x promoted to 1x8x2, then broadcasts to 4x8x2
```

**Python**

（Python 中广播语义相同。）

### 2.4.8.2 算术运算符

所有支持的算术运算符对 tile 逐元素应用，产生广播形状的新 tile。标量与 tile 组合在每元素广播。当操作数类型不同时，偏向保留更多信息的类型：

- Tile 与 tile 组合：结果为更高精度或更大范围的类型 tile。例如：
  - int + float 产生 float
  - int16 + int32 产生 int32
- 标量与 tile 组合：当标量类型可精确表示为 tile 元素类型时（如整数字面量 2 与 int tile 组合，或 2.0f 与 float tile 组合），操作在 tile 元素类型中进行。当标量需收窄以放入 tile 元素类型时（如字面量 2.5 与 int tile 组合），两种语言不同：
  - Python 把结果提升到能容纳两者的类型
  - C++ 拒绝该表达式为非法

下面片段图示标量-tile 分歧情形：

**C++**

```cpp
using i32x8 = ct::tile<int, ct::shape<8>>;
i32x8 x = ct::full<i32x8>(3);

x + 2;       // OK - int literal matches int tile element type
x + 2.5;     // ill-formed - 2.5 would narrow to int
```

**Python**

（Python 中 `x + 2.5` 会提升结果到 float。）

实践中，尽可能用 tile 元素类型写标量字面量，需要不同精度时显式转换。同样规则在 kernel 内对加载的 tile 操作数也适用：

**C++**

```cpp
__tile_global__ void elementwise(float* __restrict__ a, float* __restrict__ b, float* __restrict__ out, int N) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    a   = ct::assume_aligned(a,   16_ic);
    b   = ct::assume_aligned(b,   16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto aView = ct::partition_view{ct::tensor_span{a,   ct::extents{N}}, ct::shape{8_ic}};
    auto bView = ct::partition_view{ct::tensor_span{b,   ct::extents{N}}, ct::shape{8_ic}};
    auto cView = ct::partition_view{ct::tensor_span{out, ct::extents{N}}, ct::shape{8_ic}};

    int  bx = ct::bid().x;
    auto x  = aView.load(bx);
    auto y  = bView.load(bx);
    // 2.0f matches the float tiles' element type, so no narrowing conversion is required.
    // The scalar is broadcast across every element; + then runs elementwise.
    auto z  = 2.0f * x + y;
    cView.store(z, bx);
}
```

**Python**

（Python 中同样写法。）

需要显式控制舍入模式或次正规处理时，CUDA Tile API 提供接受这些为参数的数学函数（如 `ct.add`、`ct::add`）。

## 2.4.9 Tile 原语

工厂函数（"创建 Tile"）、load 和 store（"Tile 空间 Load 和 Store"）和逐元素算术（"逐元素算术与广播"）都是 tile 原语，即语言的一部分。程序员在 tile 粒度写它们，编译器把它们映射到硬件——包括可用的 Tensor Core。本节覆盖 CUDA tile 中可用的其它原语。

### 2.4.9.1 矩阵乘

两个 tile 的矩阵乘是实现两个数组间矩阵乘的基本操作。CUDA Tile 提供两种 tile 间矩阵乘形式：纯矩阵乘（matmul）`a @ b`，以及矩阵乘累加（mma）`a @ b + acc`。mma 中累加器从一个 K-tile 携带部分积到下一个。这对 tiled 矩阵乘的内循环有用。matmul 和 mma 都支持 2D 矩阵乘和 3D 批乘，以及操作数与累加器数据类型（精度）混合。rank 和元素类型约束见各操作 API 参考（CUDA Tile C++ 矩阵乘、cuTile Python matmul）。

下面 kernel 中用的常见模式是：无论输入精度如何都用 FP32 累加，store 时转回输出元素类型。Python 中是带 FP32 类型 acc 的 `ct.mma(a, b, acc)`。C++ 中是带显式 FP32 累加器类型的 `ct::mma(a, b, acc)`。K 循环迭代 `ceil(K / tk)` 次以覆盖 A 的右边和 B 的底边；部分 K-tile 在 load 时零填充（Python 中 `PaddingMode.ZERO`，C++ 中 `.load_masked()`），C 侧的部分 M/N 边缘 tile 由 store 侧 OOB 丢弃处理（Python 中 `ct.store`，C++ 中 `.store_masked()`）。

**C++**

```cpp
__tile_global__ void gemm(const __half* __restrict__ A, const __half* __restrict__ B, float* __restrict__ C,
                          std::size_t M, std::size_t K, std::size_t N) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using f32_acc = ct::tile<float, ct::shape<32, 32>>;

    A = ct::assume_aligned(A, 16_ic);
    B = ct::assume_aligned(B, 16_ic);
    C = ct::assume_aligned(C, 16_ic);

    constexpr auto tm = 32_ic;
    constexpr auto tn = 32_ic;
    constexpr auto tk = 16_ic;

    auto aView = ct::partition_view{ct::tensor_span{A, ct::extents{M, K}}, ct::shape{tm, tk}};
    auto bView = ct::partition_view{ct::tensor_span{B, ct::extents{K, N}}, ct::shape{tk, tn}};
    auto cView = ct::partition_view{ct::tensor_span{C, ct::extents{M, N}}, ct::shape{tm, tn}};

    auto [bx, by, bz] = ct::bid();
    auto acc = ct::full<f32_acc>(0.0f);                 // FP32 accumulator

    std::size_t num_k = (K + tk - 1) / tk;
    for (auto k : ct::irange(std::size_t{0}, num_k)) {
        acc = ct::mma(aView.load_masked(bx, k),         // zero-pad partial K-tile
                      bView.load_masked(k, by),
                      acc);                             // acc += a @ b
    }
    cView.store_masked(acc, bx, by);                    // drop OOB edge lanes
}
```

**Python**

（Python 中用 `ct.mma(a_tile, b_tile, acc)`。）

### 2.4.9.2 归约与扫描

归约是把 tile 收缩为标量或一行标量的工具。计算 softmax 的分母、layer norm 的均值与方差、attention 评分中的最大值都涉及归约操作。

一点值得先内化的是结果形状。Python 默认丢弃被归约的轴（传 `keepdims=True` 保留为长度 1）；C++ 总保留它，保持 tile 的 rank。下面两个片段都沿 axis 1 归约 2×4 tile；输出形状是可见的差异。

**C++**

```cpp
using namespace ct::literals;
using i32x2x4 = ct::tile<int, ct::shape<2, 4>>;

auto x = ct::iota<i32x2x4>();                         // [[0,1,2,3],[4,5,6,7]]
auto row_sums = ct::sum(x, 1_ic);                     // shape (2, 1) - axis kept
// row_sums == [[6], [22]]
```

**Python**

（Python 中 `ct.sum(x, axis=1)` 默认返回 shape (2,)，`keepdims=True` 时返回 (2,1)。）

扫描是运行对应物，沿一轴产生累积结果。例如前缀和（cumsum）产生与输入同维的输出，给定索引处的值是沿指定轴直到并包括该索引的所有元素之和。完整集合见各语言 API 参考（CUDA Tile C++ 归约与扫描、cuTile Python 归约与扫描）。

### 2.4.9.3 转置与置换

两个相关原语在不触及数据的情况下重排 tile 的轴：转置交换前两个轴，置换做任意重排。它们出现在 tile 逻辑布局需在操作间改变的场合——如实例化 matmul 操作数的转置、attention block 中交换行列、广播前对齐轴。

Python 中，`ct.transpose(x)` 对 rank-2 tile 交换两轴；对更高 rank tile 取显式 `axis0` / `axis1` 参数。`ct.permute(x, axes)` 取轴索引元组。C++ 中 `ct::transpose(x)` 交换前两维（尾随维度保留），`ct::permute(x, map)` 取描述新顺序的 `ct::dimension_map`。

**C++**

```cpp
using namespace ct::literals;
using t2d = ct::tile<int, ct::shape<2, 4>>;
using t3d = ct::tile<int, ct::shape<2, 2, 2>>;

auto tx = ct::iota<t2d>();
auto ty = ct::transpose(tx);                                     // shape (4, 2)

auto tz = ct::iota<t3d>();
auto tw = ct::permute(tz, ct::dimension_map{2_ic, 0_ic, 1_ic});  // axes (0,1,2) -> (2,0,1)
```

**Python**

（Python 中用 `ct.transpose(x)` 和 `ct.permute(x, axes)`。）

### 2.4.9.4 逐元素选择

逐元素选择是条件的 tile 形式：给定布尔 tile 和两个操作数 tile，每个输出元素根据对应布尔值从一个或另一个中选取。条件广播到操作数形状；操作数类型必须兼容（精确规则见各语言 API 参考：CUDA Tile C++ select、cuTile Python selection）。Python 写作 `ct.where(cond, x, y)`；C++ 写作 `ct::select(cond, lhs, rhs)`。

**C++**

```cpp
using namespace ct::literals;
auto cond = ct::iota<ct::tile<int, ct::shape<4>>>() < 2;   // {T, T, F, F}
auto t    = ct::full<ct::tile<float, ct::shape<4>>>( 1.0f);
auto f    = ct::full<ct::tile<float, ct::shape<4>>>(-1.0f);
auto r    = ct::select(cond, t, f);                        // {1, 1, -1, -1}
```

**Python**

（Python 中用 `ct.where(cond, x, y)`。）

### 2.4.9.5 数学函数

常见逐元素数学操作在 tile 代码中以 `ct` 命名空间中的函数可用：

- `add`、`sub`、`mul`
- `truediv`、`floordiv`、`cdiv`
- `mod`
- `pow`
- `exp`、`exp2`、`log`、`log2`
- `sqrt`、`rsqrt`
- `sin`、`cos`、`tan`
- `sinh`、`cosh`、`tanh`
- `minimum`、`maximum`
- `negative`
- `floor`、`ceil`

每个函数对输入 tile 逐元素应用其操作，返回同形状 tile。这些操作对 tile 代码中的标量也工作。

精确细节和完整支持的逐元素操作列表见 API 参考：

- cuTile Python 数学操作
- CUDA Tile C++ 数学操作

## 2.4.10 原子内存操作

tile 代码中有两种情形需使用内存原子：

- 跨 block 争用：每个 block 产生部分结果，用原子操作在全局内存位置与其它 block 的部分结果合并。
- block 内争用：tile 的多个元素写入同一内存位置。

对 tile 的原子操作执行每元素一次原子更新。每元素操作是原子的，但整个调用不是。每元素原子操作的顺序未指定。

Python 中原子操作用对数组的索引寻址目标，与 `ct.gather` 和 `ct.scatter` 用相同约定。可选参数控制边界检查、内存序和线程作用域。默认（边界检查开、`ACQ_REL`、设备作用域）让普通调用只传数组、索引和更新值即可。`TiledView` 也以实例方法暴露同样的原子操作（如 `TiledView.atomic_add(index, update)`）；这些用 tile 空间索引寻址目标、不返回值，并下降为 PTX 中的原子归约。当不需要先前值时，`TiledView` 形式因更好性能而优先。

C++ 中原子操作取指针和对应值：单一位置用裸指针和标量，或指针 tile 和值 tile。内存序是调用处的编译期类型标签，如 `ct::memory_order_relaxed_t{}`。线程作用域是同形式类型标签，省略时默认全系统可见。

### 2.4.10.1 跨 block 争用

下面代码示例中发生跨 block 争用，因为不同 block 写同一内存位置 `out`。没有原子操作，并行运行的 block 会导致错误答案。本例用设备线程作用域（C++ 中 `ct::thread_scope_device_t{}`，Python 中线程作用域默认为设备级），因为内存操作的结果必须对设备上所有运行的 block 可见。Python kernel 用 `TiledView.atomic_add`，因为每个 block 的部分和累加到 `out[0]` 后立即丢弃。

**C++**

```cpp
__tile_global__ void block_sum(int* __restrict__ arr, int* __restrict__ out, std::size_t N) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    constexpr auto TILE = 16_ic;

    arr = ct::assume_aligned(arr, 16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto aView = ct::partition_view{ct::tensor_span{arr, ct::extents{N}},
                                    ct::shape{TILE}};
    int bid = ct::bid().x;
    auto tile    = aView.load_masked(bid);        // partial final tile -> OOB lanes default to 0
    auto partial = ct::sum(tile, 0_ic);           // reduce to a 1-element tile

    ct::atomic_add(out, (int)partial,             // accumulate the scalar into out[0]
                   ct::memory_order_relaxed_t{},  // single-location accumulator -> relaxed suffices
                   ct::thread_scope_device_t{});  // visible across the device
}
```

**Python**

（Python 中用 `TiledView.atomic_add(index, update)`。）

### 2.4.10.2 block 内争用

下面代码片段中发生 block 内争用，因为 tile 的所有值被原子加到单一内存位置。

本例中，`ptrs` tile 的每个元素指向同一内存位置 `slot`。`ct::iota<i32x16>()` 创建的 tile 的每个元素被原子加到该内存位置中存储的值。从 tile 到单一内存地址的多个原子操作的执行顺序未指定。用 block 线程作用域 `ct::thread_scope_block_t{}` 指定原子操作结果只需在本线程块内可见。

**C++**

```cpp
using i32x16 = ct::tile<int, ct::shape<16>>;

int* slot = /* pointer to the contended location */;

// 16 lanes all aim at the same address. Add is commutative, so the
// unspecified ordering doesn't affect this sum; block scope suffices
// since contention stays within one block.
auto ptrs = ct::full<ct::tile<int*, ct::shape<16>>>(slot);
ct::atomic_add(ptrs, ct::iota<i32x16>(),
            ct::memory_order_relaxed_t{},
            ct::thread_scope_block_t{});
```

> **注意**
>
> 此处仅为图示。要在 block 内把 tile 求和为标量，2.4.9.2 节所示的 tile 归约操作是首选方法。

### 2.4.10.3 支持的原子操作

Tile 代码支持多种原子内存操作，区别在于写入值如何与内存中已有值组合：

- `atomic_and`——在传入值与内存中值之间逐元素执行原子按位与
- `atomic_or`——在传入值与内存中值之间逐元素执行原子按位或
- `atomic_xor`——在传入值与内存中值之间逐元素执行原子按位异或
- `atomic_max`——在传入值与内存中值之间逐元素比较，把较大值存入内存
- `atomic_min`——在传入值与内存中值之间逐元素比较，把较小值存入内存
- `atomic_add`——把传入值加到内存中值并把结果存入内存
- `atomic_xchng`——把传入值写入内存并返回写之前内存中的值
- `atomic_cas`——在内存中值与作为参数传入的期望值之间逐元素比较。若匹配，内存中值被替换为期望值

所有支持的原子内存操作的完整文档见 CUDA Tile C++ API 参考或 cuTile Python API 参考的内存操作章节。

## 2.4.11 优化提示

优化提示是附加到源结构（tile kernel 函数、load/store 调用点等）上、引导编译器代码生成的元数据。提示不改变程序语义：kernel 加不加提示都编译运行一致，因此可自由添加、删除或调试而不影响正确性。编译器也可忽略任何提示。

提示共享两个一般属性：

- 提示按结构划分。提示作用于它附加的具体 kernel 函数或调用表达式，不作用于周围代码。
- 提示可按架构指定。每个提示可对不同 GPU 架构设不同值，或设单一值应用于每个目标。

两种语言暴露提示的方式不同：

- C++ 用置于相关声明或语句上的 C++ 属性。
- Python 用 kernel 装饰器和单个内存操作调用点上的关键字参数。

提示种类集合、每种提示实际控制什么在两语言间共享，记录在"提示种类"中。

### 2.4.11.1 C++ —— cutile::hint 属性

C++ 中提示用 C++ 属性 `cutile::hint` 表达：

```cpp
[[ cutile::hint(arch, kind1=value1, kind2=value2, ...) ]]
```

第一个参数是目标架构，按与 `__CUDA_ARCH__` 宏相同的约定以整数编码（如 `900` 对应 sm_90，`1000` 对应 sm_100）。特殊值 `0` 表示架构无关提示，应用于每个目标架构。其余参数是 `kind=value` 对，指定提示种类及其值。

`cutile::hint` 属性应用于其前置的结构：

- 对 tile kernel 函数，属性置于函数声明上。
- 对 `ct::load`、`ct::store` 等 `ct::partition_view` load/store 内存操作，属性置于含调用的表达式语句上。

其它放置有限制；完整规则见 CUDA Tile C++ hint 规范。

下面 kernel 示例两种放置：kernel 级提示为 sm_90 和 sm_100 设不同 `num_cta_in_cga`，以及表达式语句提示标记特定 load 为带宽密集型。

**C++**

```cpp
[[ cutile::hint(900,  num_cta_in_cga=4),    // sm_90:  prefer 4 CTAs per cluster
   cutile::hint(1000, num_cta_in_cga=8) ]]  // sm_100: prefer 8 CTAs per cluster
__tile_global__ void optimization_hints(float* __restrict__ in,
                                        float* __restrict__ out) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    in  = ct::assume_aligned(in,  16_ic);
    out = ct::assume_aligned(out, 16_ic);

    auto inSpan  = ct::tensor_span{in,  ct::extents{128_ic}};
    auto outSpan = ct::tensor_span{out, ct::extents{128_ic}};
    auto inView  = ct::partition_view{inSpan,  ct::shape{8_ic}};
    auto outView = ct::partition_view{outSpan, ct::shape{8_ic}};

    int bx = ct::bid().x;

    // Expression-statement hint: tag this particular load as bandwidth-heavy.
    ct::tile<float, ct::shape<8>> tile;
    [[ cutile::hint(0, latency=8) ]]
    tile = inView.load(bx);

    outView.store(tile, bx);
}
```

当同种多个提示作用于同一结构时，架构特定提示覆盖架构无关提示。

### 2.4.11.2 Python —— 装饰器参数与调用点关键字

Python 以两种方式暴露提示：

- kernel 级提示是 `@ct.kernel(...)` 装饰器的关键字参数。编译 kernel 对象还有 `.replace_hints(**hints)` 方法返回带覆盖提示的新 kernel；新 kernel 有自己的 JIT 缓存，使 `replace_hints` 成为自动调优循环的天然构建块。
- 每次调用提示是内存操作调用点上的关键字参数：`ct.load` / `ct.store`、`TiledView.load` / `TiledView.store`、`ct.gather` / `ct.scatter`。

按架构值用 `cuda.tile.ByTarget(*, default=..., sm_XXX=..., sm_YYY=...)` 包裹。架构键必须是形如 `"sm_<major><minor>"` 的字符串（如 `"sm_100"` 或 `"sm_120"`）。普通（非 ByTarget）值应用于每个目标——是 C++ 中 arch=0 架构无关提示的 Python 等价物。

下面 kernel 是上面 C++ 示例的直接 Python 对应：`ByTarget` 携带 kernel 级提示，`latency=8` 关键字携带每次调用提示，`replace_hints` 无需编辑源码就产出重新调优的 kernel。

**Python**

```python
@ct.kernel(num_ctas=ByTarget(sm_90=4, sm_100=8))
def optimization_hints(in_, out, TILE: ct.Constant[int]):
    bid = ct.bid(0)

    # Per-call hint: this particular load is bandwidth-heavy.
    tile = ct.load(in_, index=(bid,), shape=(TILE,), latency=8)

    ct.store(out, index=(bid,), tile=tile)


# Autotuning: produce a new kernel with overridden hints without editing the
# source. The new kernel has its own JIT cache.
tuned_kernel = optimization_hints.replace_hints(num_ctas=8)
```

### 2.4.11.3 提示种类

以下提示在两语言间共享。每种提示内，C++ 名和 Python 名条目是同一底层提示的不同拼写；其余都相同：提示作用于何处、其值、其含义。

#### 2.4.11.3.1 每集群 CTA

- C++ 名：`num_cta_in_cga`（kernel 属性）。
- Python 名：`num_ctas`（`@ct.kernel` 装饰器参数）。
- 允许值：1、2、4、8、16。在 sm_80 上只 1 适用。
- 含义：发射 kernel 时编译器应偏好每 cooperative group array（CGA）多少个 cooperative thread array（CTA）。

#### 2.4.11.3.2 占用率

- C++ 名：`occupancy`（kernel 属性）。
- Python 名：`occupancy`（`@ct.kernel` 装饰器参数）。
- 允许值：闭区间 [1, 32] 内任意整数。
- 含义：每流多处理器（SM）活跃 CTA 的目标数。编译器把该值视为建议，会在代码生成时尽量满足。

#### 2.4.11.3.3 内存访问延迟

- C++ 名：`latency`（含调用的表达式语句上的属性）。
- Python 名：`latency`（调用点上的关键字参数）。
- 适用于：tile 空间 load 和 store（C++ 中 `ct::partition_view`；Python 中 `Array.tiled_view` 和 `ct.load` / `ct.store`）以及 gather/scatter（C++ 中带指针 tile 的 `ct::load` / `ct::store`；Python 中 `ct.gather` / `ct.scatter`）。
- 允许值：闭区间 [1, 10] 内任意整数，1 表示轻 DRAM 流量，10 表示重流量。较大值通常使编译器调度更大预取深度。

#### 2.4.11.3.4 允许 TMA

- C++ 名：`allow_tma`（含调用的表达式语句上的属性）。
- Python 名：`allow_tma`（调用点上的关键字参数）。
- 适用于：仅 tile 空间 load 和 store（C++ 中 `ct::partition_view`；Python 中 `Array.tiled_view` 和 `ct.load` / `ct.store`）。Gather 和 scatter 操作不接受此提示。
- 允许值：C++ 中 true / false，Python 中 True / False。TMA 默认允许；把提示设为 false/False 指示编译器在支持的硬件上不要把此特定 load 或 store 下降为 TMA。

## 2.4.12 C++ 性能提示

本指南中 C++ kernel 都用同一小组标注和惯用法。本节解释它们做什么、为何重要。

### 2.4.12.1 为内存中的数组使用 __restrict__ 指针

`__restrict__` 关键字告诉编译器通过某指针访问的内存区域在该指针生命期内只会通过该指针访问。见 5.4.1.4 节。

Tile C++ 中，使用满足这些条件的内存数组并用 `__restrict__` 标注其指针对良好内存操作性能至关重要。

为理解为何，考虑一个用非 `__restrict__` 指针数组做逐元素拷贝的例子：

**C++**

```cpp
__tile_global__ void tile_elementwise_copy(float* out, float const* in) {
    namespace ct = cuda::tiles;

    using f32x64 = ct::tile<float, ct::shape<64>>;
    using i32x64 = ct::tile<int, ct::shape<64>>;

    auto inPtrs  = in  + 64 * ct::bid().x + ct::iota<i32x64>();
    auto outPtrs = out + 64 * ct::bid().x + ct::iota<i32x64>();

    auto data = ct::load(inPtrs);   // (1)
    ct::store(outPtrs, data);       // (2)
}
```

编译器如何并行化 tile 操作通常在 CUDA Tile 程序中可忽略。但这里我们考虑它以理解为什么使用不重叠数组使编译器能生成更好性能代码。

考虑编译器如何并行化 load 和 store tile 操作。若输入和输出数组不重叠，load 可被并行化为一组独立内存读操作。类似地，store 可被并行化为多个内存写操作，每个只依赖它所写数据元素的 load 操作。

但若输入和输出数组可能重叠，编译器必须确保整个 tile 的所有内存 load 操作在发出任何内存 store 操作之前已完成，以保证正确程序语义。否则，store 操作可能在某元素被 load 读之前执行并覆盖它，导致不正确的程序执行。这限制了编译器交错读写的能力，因为所有读必须在任何写发出前完成。

简言之，当编译器不能保证数组不重叠时，必须生成更保守的代码。这就是使用不重叠数组并用 `__restrict__` 关键字告知编译器为何有助于获得最佳性能。

当内存区域可被另一指针访问时给指针标 `__restrict__` 会导致未定义行为。

### 2.4.12.2 把数组指针标记为 16 字节对齐

用 `ct::assume_aligned` 把数组指针标记为 16 字节对齐：

**C++**

```cpp
__tile_global__ void foo(float* __restrict__ in) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    in = ct::assume_aligned(in, 16_ic);

    ct::tensor_span t{in, ct::extents{256_ic, 256_ic}};
    ct::partition_view{t, ct::shape{4_ic, 4_ic}};

    // ...
}
```

此对齐保证是 `ct::partition_view` 使用张量内存加速器（TMA）所必需的。使用此技术时必须在运行期提供 16 字节对齐指针，否则行为未定义。

`cudaMalloc` 等 CUDA 内存分配器返回的指针保证至少 16 字节对齐。

### 2.4.12.3 优先用 ct::partition_view 做内存访问

优先用 `ct::partition_view` 而非 gather 和 scatter 形式 `ct::load` 和 `ct::store` 做结构化内存访问。基于 view 的形式可在支持的硬件上下降为张量内存加速器（TMA），比逐元素 gather 快得多。gather/scatter 上下文见"Gather 和 Scatter"。

### 2.4.12.4 对有界循环用 ct::irange

迭代固定范围时用 `ct::irange` 而非普通 for 循环。结构化形式让编译器应用流水线和向量化等优化——这些在循环边界和步长是不透明整数表达式时不可用（见"控制流"）：

**C++**

```cpp
for (auto idx : ct::irange(lowerBound, upperBound, step)) {
    // ...
}
```

---

[← 上一章 2.3 编写 SIMT Kernel](06_writing_simt_kernels.md) ｜ [返回附录 C 首页](README.md)