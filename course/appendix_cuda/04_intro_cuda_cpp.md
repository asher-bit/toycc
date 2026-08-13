# 2.1 CUDA C++ 入门

本章通过 C++ 中如何暴露 CUDA 编程模型的基本概念来对其进行介绍。

本编程指南以 CUDA 运行时 API 为主。CUDA 运行时 API 是 C++ 中使用 CUDA 最常见的方式，它构建于更底层的 CUDA 驱动 API 之上。

"CUDA 运行时 API 与 CUDA 驱动 API"一节讨论两种 API 的差异，"CUDA 驱动 API"一节讨论混合使用两种 API 的写法。

本指南假设 CUDA Toolkit 与 NVIDIA 驱动已安装，且存在一张受支持的 NVIDIA GPU。安装必要 CUDA 组件的步骤见 The CUDA Quickstart Guide。

## 2.1.1 用 NVCC 编译

用 C++ 编写的 GPU 代码由 NVIDIA Cuda Compiler——nvcc——编译。nvcc 是一个编译驱动，用以简化 C++ 或 PTX 代码的编译过程：它提供简单且熟悉的命令行选项，并通过调用一组实现不同编译阶段的工具来执行这些选项。

本指南将给出可以在任何安装了 CUDA Toolkit 的 Linux 系统、Windows 命令行或 PowerShell、或装有 CUDA Toolkit 的 Windows Subsystem for Linux 上使用的 nvcc 命令行。本指南的 nvcc 章节覆盖 nvcc 的常见用法，完整文档由 nvcc 用户手册提供。

## 2.1.2 Kernel

如"CUDA 编程模型引言"中所述，在 GPU 上执行、可被主机调用的函数称为 kernel。kernel 被写成由许多并行线程同时运行。

### 2.1.2.1 指定 Kernel

kernel 的代码用 `__global__` �声明说明符指定。它告诉编译器：这个函数会被编译成可由 kernel 发射调用的、在 GPU 上运行的版本。kernel 发射是从（通常）CPU 启动一个 kernel 运行的操作。kernel 是返回类型为 `void` 的函数。

```cpp
// Kernel definition
__global__ void vecAdd(float* A, float* B, float* C)
{

}
```

### 2.1.2.2 发射 Kernel

并行执行 kernel 的线程数在 kernel 发射时指定，称为**执行配置**。同一 kernel 的不同调用可使用不同的执行配置，例如不同的线程数或线程块数。

从 CPU 代码发射 kernel 有两种方式：三尖括号记法与 `cudaLaunchKernelEx`。这里先介绍最常用的三尖括号记法。用 `cudaLaunchKernelEx` 发射 kernel 的示例在 3.1.1 节详细展示与讨论。

#### 2.1.2.2.1 三尖括号记法

三尖括号记法是 CUDA C++ 的一种语言扩展，用于发射 kernel。称为"三尖括号"是因为它用三个尖括号字符把 kernel 发射的执行配置包起来，即 `<<< >>>`。执行配置参数以逗号分隔的列表写在尖括号内，类似函数调用的参数。下面是 `vecAdd` kernel 的发射语法。

```cpp
__global__ void vecAdd(float* A, float* B, float* C)
{

}

int main()
{
    ...
    // Kernel invocation
    vecAdd<<<1, 256>>>(A, B, C);
    ...
}
```

三尖括号记法的前两个参数分别是 grid 维度和线程块维度。使用一维线程块或 grid 时，可以用整数指定维度。

上面代码发射一个含 256 个线程的线程块。每个线程会执行完全相同的 kernel 代码。在"线程与 grid 索引内建变量"中，我们将展示每个线程如何用自己在线程块和 grid 内的索引来改变它操作的数据。

每 block 的线程数有上限，因为一个 block 的所有线程都驻留在同一 SM 上、必须共享 SM 资源。当前 GPU 上一个线程块最多可含 1024 个线程。如果资源允许，可以在一个 SM 上同时调度多个线程块。

kernel 发射相对于主机线程是异步的。也就是说，kernel 会被安排在 GPU 上执行，但主机代码不会等待 kernel 在 GPU 上完成（甚至启动）后才继续。必须使用某种 CPU 与 GPU 间的同步来确定 kernel 已完成。最基础的版本——完全同步整张 GPU——见"同步 CPU 与 GPU"。更精细的同步方法在"异步执行"中介绍。

使用二维或三维 grid 或线程块时，用 CUDA 类型 `dim3` 作为 grid 与线程块维度参数。下面代码片段用 16×16 的线程块 grid、每个线程块 8×8 来发射 `MatAdd` kernel。

```cpp
int main()
{
    ...
    dim3 grid(16,16);
    dim3 block(8,8);
    MatAdd<<<grid, block>>>(A, B, C);
    ...
}
```

### 2.1.2.3 线程与 grid 索引内建变量

在 kernel 代码内，CUDA 提供**内建变量**来访问执行配置参数以及线程或块的索引。

`threadIdx` 给出线程在自己所在线程块内的索引。线程块内每个线程都有不同索引。

`blockDim` 给出线程块的维度，这是 kernel 发射时在执行配置中指定的。

`blockIdx` 给出线程块在 grid 内的索引。每个线程块都有不同索引。

`gridDim` 给出 grid 的维度，这是 kernel 发射时在执行配置中指定的。

这些内建变量每个都是三分量向量，含 `.x`、`.y`、`.z` 三个成员。未被发射配置指定的维度默认为 1。`threadIdx` 与 `blockIdx` 从 0 开始编号。也就是说，`threadIdx.x` 取值从 0 到 `blockDim.x - 1`（含）。`.y` 和 `.z` 在各自维度上同理。

类似地，`blockIdx.x` 取值从 0 到 `gridDim.x - 1`（含），`.y`、`.z` 各自维度同理。

这些让单个线程能识别自己应执行哪份工作。回到 `vecAdd` kernel：该 kernel 接收三个参数，每个都是 float 向量。kernel 对 A 与 B 做逐元素相加并把结果存到 C。kernel 被并行化到每个线程做一次加法。哪个元素由它的线程与 grid 索引决定。

```cpp
__global__ void vecAdd(float* A, float* B, float* C)
{
   // calculate which element this thread is responsible for computing
   int workIndex = threadIdx.x + blockDim.x * blockIdx.x;

   // Perform computation
   C[workIndex] = A[workIndex] + B[workIndex];
}

int main()
{
    ...
    // A, B, and C are vectors of 1024 elements
    vecAdd<<<4, 256>>>(A, B, C);
    ...
}
```

本例用 4 个含 256 线程的线程块相加 1024 元素的向量。在第一个线程块中 `blockIdx.x` 为 0，每个线程的 `workIndex` 就是它的 `threadIdx.x`。第二个线程块中 `blockIdx.x` 为 1，所以 `blockDim.x * blockIdx.x` 等于 `blockDim.x`，本例即 256。第二个线程块每个线程的 `workIndex` 是它的 `threadIdx.x + 256`。第三个线程块的 `workIndex` 是 `threadIdx.x + 512`。

这种 `workIndex` 计算在一维并行化中非常常见。扩展到二维或三维时通常每一维都按相同模式。

#### 2.1.2.3.1 边界检查

上面的例子假设向量长度是线程块大小（本例 256）的整数倍。要让 kernel 处理任意向量长度，可以加上对内存访问不越过数组边界的检查，如下所示，然后再多发射一个会有部分线程不活跃的线程块。

```cpp
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
     // calculate which element this thread is responsible for computing
     int workIndex = threadIdx.x + blockDim.x * blockIdx.x;

     if(workIndex < vectorLength)
     {
         // Perform computation
         C[workIndex] = A[workIndex] + B[workIndex];
     }
}
```

有了上面的 kernel 代码，可以发射比所需更多的线程而不会导致数组越界访问。当 `workIndex` 超过 `vectorLength`，线程退出不做任何工作。在一个线程块里多发射一些不干活的线程不会带来大开销，但应避免发射整块都没有线程干活的线程块。该 kernel 现在能处理不是 block 大小整数倍的向量长度。

所需线程块数可以算作"所需线程数（本例为向量长度）除以每 block 线程数"的上取整。即所需线程数除以每 block 线程数的整数除法向上取整。一种常见的单整数除法写法如下。在整数除法前加上 `threads - 1`，使它表现得像上取整函数：仅当向量长度不能被每 block 线程数整除时才多加一个线程块。

```cpp
// vectorLength is an integer storing number of elements in the vector
int threads = 256;
int blocks = (vectorLength + threads-1)/threads;
vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```

CUDA Core Compute Library（CCCL）提供便利工具 `cuda::ceil_div` 用于做这种上取整除法以计算 kernel 发射所需的 block 数。该工具包含头文件 `<cuda/cmath>` 即可使用。

```cpp
// vectorLength is an integer storing number of elements in the vector
int threads = 256;
int blocks = cuda::ceil_div(vectorLength, threads);
vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```

此处选 256 个线程每 block 是任意的，但通常是一个好的起点值。

## 2.1.3 GPU 计算中的内存

要能使用上面的 `vecAdd` kernel，数组 A、B、C 必须位于 GPU 可访问的内存中。有几种方式做到这点，这里示例其中两种。其它方式将在后面关于统一内存的章节介绍。GPU 代码可用的内存空间在"GPU 内存"中介绍，并在"GPU 设备内存空间"中更详细地覆盖。

### 2.1.3.1 统一内存

统一内存是 CUDA 运行时的一个特性，它让 NVIDIA 驱动管理主机与设备之间的数据搬运。内存通过 `cudaMallocManaged` API 分配，或通过用 `__managed__` 说明符声明变量来分配。NVIDIA 驱动会保证 GPU 或 CPU 任一访问该内存时它都是可访问的。

下面的代码展示一个完整的函数，发射 `vecAdd` kernel 并对要用于 GPU 的输入和输出向量使用统一内存。`cudaMallocManaged` 分配的缓冲区既可被 CPU 也可被 GPU 访问。这些缓冲区用 `cudaFree` 释放。

```cpp
void unifiedMemExample(int vectorLength)
{
    // Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Use unified memory to allocate buffers
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // Wait for the kernel to complete execution
    cudaDeviceSynchronize();

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);
}
```

统一内存受 CUDA 支持的所有操作系统和 GPU 支持，但底层机制和性能因系统架构而异。"统一内存"一节给出更多细节。在某些 Linux 系统上（例如带地址翻译服务或异构内存管理的系统），所有系统内存自动就是统一内存，无需使用 `cudaMallocManaged` 或 `__managed__` 说明符。

### 2.1.3.2 显式内存管理

显式管理内存空间之间的内存分配与数据迁移能帮助改善应用性能，但代码会更冗长。下面代码用 `cudaMalloc` 在 GPU 上显式分配内存。GPU 上的内存用与上例统一内存相同的 `cudaFree` API 释放。

```cpp
void explicitMemExample(int vectorLength)
{
    // Pointers for host memory
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Pointers for device memory
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    //Allocate Host Memory using cudaMallocHost API. This is best practice
    // when buffers will be used for copies between CPU and GPU memory
    cudaMallocHost(&A, vectorLength*sizeof(float));
    cudaMallocHost(&B, vectorLength*sizeof(float));
    cudaMallocHost(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // start-allocate-and-copy
    // Allocate memory on the GPU
    cudaMalloc(&devA, vectorLength*sizeof(float));
    cudaMalloc(&devB, vectorLength*sizeof(float));
    cudaMalloc(&devC, vectorLength*sizeof(float));

    // Copy data to the GPU
    cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, vectorLength*sizeof(float));
    // end-allocate-and-copy

    // Launch the kernel
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    // wait for kernel execution to complete
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault);

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Explicit Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
    }

    // clean up
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparisonResult);
}
```

CUDA API `cudaMemcpy` 用于把数据从位于 CPU 的缓冲区复制到位于 GPU 的缓冲区。除目标指针、源指针和字节数外，`cudaMemcpy` 的最后一个参数是 `cudaMemcpyKind_t`。可取值包括：

- `cudaMemcpyHostToDevice`：从 CPU 到 GPU 的复制
- `cudaMemcpyDeviceToHost`：从 GPU 到 CPU 的复制
- `cudaMemcpyDeviceToDevice`：GPU 内或 GPU 之间的复制

本例向 `cudaMemcpy` 传 `cudaMemcpyDefault` 作为最后一个参数。这使 CUDA 根据源与目标指针的取值来决定执行的复制类型。

`cudaMemcpy` API 是同步的。也就是说，它在复制完成前不会返回。异步复制在"在 CUDA Stream 中启动内存搬运"中介绍。

代码用 `cudaMallocHost` 在 CPU 上分配内存。这会在主机上分配**页锁定内存**，能改善复制性能，并且是异步内存搬运所必需的。通常，对用于与 GPU 来回传数据的 CPU 缓冲区使用页锁定内存是好做法。如果过多主机内存被页锁定，某些系统性能可能下降。最佳实践是只对用于向 GPU 发送或从 GPU 接收数据的缓冲区做页锁定。

### 2.1.3.3 内存管理与应用性能

如上例所示，显式内存管理更冗长，要求程序员指定主机与设备之间的复制。这正是显式内存管理的优点也是缺点：它对何时在主机与设备之间复制数据、内存驻留哪里、究竟何处分配什么内存给予更多控制。显式内存管理可通过控制内存搬运以及与其它计算重叠来提供性能机会。

使用统一内存时也有 CUDA API（在"内存建议与预取"中介绍）向管理内存的 NVIDIA 驱动提供提示，使统一内存也能获得部分显式内存管理带来的性能收益。

## 2.1.4 同步 CPU 与 GPU

如"发射 Kernel"中所述，kernel 发射相对于调用它的 CPU 线程是异步的。即 CPU 线程在 kernel 完成前——甚至可能在它启动前——就继续执行控制流。为了在主机代码继续之前保证 kernel 已完成执行，需要某种同步机制。

同步 GPU 与主机线程最简单的方式是使用 `cudaDeviceSynchronize`，它会阻塞主机线程直到 GPU 上此前发出的所有工作都完成。本章示例中这已足够，因为 GPU 上只执行单次操作。在更大应用中，GPU 上可能有多个 stream 同时执行工作，`cudaDeviceSynchronize` 会等待所有 stream 的工作完成。这类应用中建议使用 Stream 同步 API 只与特定 stream 同步，或使用 CUDA Event。这些在"异步执行"一章详细讲解。

## 2.1.5 串起来

下面的清单给出本章引入的简单向量加 kernel 的完整代码，连同全部主机代码和用于验证答案正确的工具函数。这些示例默认用 1024 的向量长度，但接受可执行文件的命令行参数来指定不同的向量长度。

### 统一内存版

```cpp
#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float* A, int length)
{
    std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C,  int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

//unified-memory-begin
void unifiedMemExample(int vectorLength)
{
    // Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Use unified memory to allocate buffers
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // Wait for the kernel to complete execution
    cudaDeviceSynchronize();

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);
}
//unified-memory-end


int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }
    unifiedMemExample(vectorLength);
    return 0;
}
```

### 显式内存管理版

可以用 `nvcc` 如下构建和运行：

```bash
$ nvcc vecAdd_unifiedMemory.cu -o vecAdd_unifiedMemory
$ ./vecAdd_unifiedMemory
Unified Memory: CPU and GPU answers match
$ ./vecAdd_unifiedMemory 4096
Unified Memory: CPU and GPU answers match
$ nvcc vecAdd_explicitMemory.cu -o vecAdd_explicitMemory
$ ./vecAdd_explicitMemory
Explicit Memory: CPU and GPU answers match
$ ./vecAdd_explicitMemory 4096
Explicit Memory: CPU and GPU answers match
```

在这些示例中，所有线程都做相互独立的工作，并不需要彼此协调或同步。但线程常常需要彼此合作和通信才能完成工作。线程块内的线程可以通过**共享内存**共享数据，并同步以协调内存访问。

线程块层级最基础的同步机制是 `__syncthreads()` 内建函数，它充当一个屏障：线程块内所有线程都必须在此等待，之后才允许任何线程继续。"共享内存"一节给出使用共享内存的示例。

为高效协作，共享内存预期是靠近每个处理器核心的低延迟内存（很像 L1 缓存），`__syncthreads()` 预期是轻量的。`__syncthreads()` 只同步单个线程块内的线程。

线程块之间的同步仅在特定情形下支持。例如，线程块集群允许集群内的块相互同步，Cooperative Groups API 提供创建跨线程块同步域的机制。

通常当同步保持在单个线程块内时获得最佳性能。线程块仍可通过原子内存函数对共同结果进行协作，原子函数将在后续章节介绍。

3.2.4 节覆盖为最大化性能和资源利用率提供极细粒度控制的 CUDA 同步原语。

## 2.1.6 运行时初始化

CUDA 运行时为系统中每个设备创建一个 CUDA context。此 context 是该设备的主 context，在第一个需要该设备上有活跃 context 的运行时函数处被初始化。此 context 在应用的所有主机线程间共享。作为 context 创建的一部分，必要时设备代码会被即时编译并加载到设备内存。这些都在背后透明发生。CUDA 运行时创建的主 context 可以从驱动 API 访问以实现互操作，详见"运行时 API 与驱动 API 间的互操作"。

自 CUDA 12.0 起，`cudaInitDevice` 与 `cudaSetDevice` 调用会初始化运行时和与指定设备相关联的主 context。若这些调用之前出现运行时 API 请求，运行时会隐式使用设备 0 并按需自初始化来处理这些请求。这在为运行时函数调用计时以及在解读第一次进入运行时的返回错误码时很重要。CUDA 12.0 之前，`cudaSetDevice` 不会初始化运行时。

`cudaDeviceReset` 销毁当前设备的主 context。如果主 context 被销毁后再调用 CUDA 运行时 API，会为该设备创建一个新的主 context。

> **注意**
>
> CUDA 接口使用在主机程序启动时初始化、在主机程序终止时销毁的全局状态。在程序启动或终止期——main 之后——使用这些接口（隐式或显式）会导致未定义行为。

自 CUDA 12.0 起，`cudaSetDevice` 在为该主机线程切换当前设备之后，若运行时尚未初始化，会显式初始化运行时。此前版本的 CUDA 把新设备上的运行时初始化推迟到 `cudaSetDevice` 之后的第一次运行时调用。因此，检查 `cudaSetDevice` 的返回值以发现初始化错误非常重要。

参考手册中错误处理与版本管理章节的运行时函数不会初始化运行时。

## 2.1.7 CUDA 中的错误检查

每个 CUDA API 都返回一个枚举类型 `cudaError_t` 的值。示例代码里这些错误常常没被检查。在生产应用中，最好对每个 CUDA API 调用的返回值都进行检查和管理。没有错误时返回 `cudaSuccess`。很多应用选择实现一个工具宏，如下所示：

```cpp
#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,                           \
                cudaGetErrorString(result));      \
    }                                             \
} while(0)
```

此宏使用 `cudaGetErrorString` API，它返回描述特定 `cudaError_t` 值含义的可读字符串。用上面的宏，应用把 CUDA 运行时 API 调用放进 `CUDA_CHECK(expression)` 宏中调用，如下所示：

```cpp
    CUDA_CHECK(cudaMalloc(&devA, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devB, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devC, vectorLength*sizeof(float)));
```

如果这些调用中任一检测到错误，会用此宏打印到 stderr。此宏常见于较小项目，在更大应用中可改造为日志系统或其它错误处理机制。

> **注意**
>
> 值得注意的是，任何 CUDA API 调用返回的错误状态也可能指示一个此前发出的异步操作的错误。"异步错误处理"一节更详细地讲解。

### 2.1.7.1 错误状态

CUDA 运行时为每个主机线程维护一个 `cudaError_t` 状态。默认值为 `cudaSuccess`，并在发生错误时被覆盖。`cudaGetLastError` 返回当前错误状态，然后把它重置为 `cudaSuccess`。另一种选择，`cudaPeekAtLastError` 返回错误状态但不重置。

用三尖括号记法的 kernel 发射不返回 `cudaError_t`。最好在 kernel 发射之后立即检查错误状态，以发现 kernel 发射本身的即时错误或 kernel 发射之前的异步错误。kernel 发射之后立即检查错误状态得到 `cudaSuccess`，并不代表 kernel 已成功执行甚至启动。它只验证传给运行时的 kernel 发射参数和执行配置未触发任何错误，且错误状态不是 kernel 启动之前的先前错误或异步错误。

### 2.1.7.2 异步错误

CUDA kernel 发射和许多运行时 API 是异步的。异步 CUDA 运行时 API 将在"异步执行"中详细讨论。CUDA 错误状态在每次发生错误时被设置和覆盖。这意味着异步操作执行期间发生的错误只会在错误状态下次被检查时才被报告。如前所述，这可能是 `cudaGetLastError`、`cudaPeekAtLastError`，也可能是任何返回 `cudaError_t` 的 CUDA API。

当 CUDA 运行时 API 函数返回错误时，错误状态不会被清除。这意味着来自异步错误（例如 kernel 的非法内存访问）的错误码会被每个 CUDA 运行时 API 一直返回，直到通过调用 `cudaGetLastError` 清除错误状态。

```cpp
    vecAdd<<<blocks, threads>>>(devA, devB, devC);
    // check error state after kernel launch
    CUDA_CHECK(cudaGetLastError());
    // wait for kernel execution to complete
    // The CUDA_CHECK will report errors that occurred during execution of the kernel
    CUDA_CHECK(cudaDeviceSynchronize());
```

> **注意**
>
> `cudaError_t` 值 `cudaErrorNotReady`（可能由 `cudaStreamQuery` 与 `cudaEventQuery` 返回）不算作错误，不会被 `cudaPeekAtLastError` 或 `cudaGetLastError` 报告。

### 2.1.7.3 CUDA_LOG_FILE

另一种识别 CUDA 错误的好办法是使用 `CUDA_LOG_FILE` 环境变量。设置该环境变量时，CUDA 驱动会把遇到的错误消息写出到环境变量所指定路径的文件里。例如下面这段不正确的 CUDA 代码，尝试发射的线程块比任何架构所支持的最大值都大。

```cpp
__global__ void k()
{ }

int main()
{
    k<<<8192, 4096>>>(); // Invalid block size
    CUDA_CHECK(cudaGetLastError());
    return 0;
}
```

构建运行后，kernel 发射之后的检查会用 2.1.7 节所示宏检测并报告该错误：

```bash
$ nvcc errorLogIllustration.cu -o errlog
$ ./errlog
CUDA Runtime Error: /home/cuda/intro-cpp/errorLogIllustration.cu:24:1 = invalid argument
```

但当应用在 `CUDA_LOG_FILE` 设置为某个文本文件的情况下运行时，该文件中会对错误有更多信息：

```bash
$ env CUDA_LOG_FILE=cudaLog.txt ./errlog
CUDA Runtime Error: /home/cuda/intro-cpp/errorLogIllustration.cu:24:1 = invalid argument
$ cat cudaLog.txt
[12:46:23.854][137216133754880][CUDA][E] One or more of block dimensions of (4096,1,1) exceeds corresponding maximum value of (1024,1024,64)
[12:46:23.854][137216133754880][CUDA][E] Returning 1 (CUDA_ERROR_INVALID_VALUE) from cuLaunchKernel
```

把 `CUDA_LOG_FILE` 设为 `stdout` 或 `stderr` 会分别打印到标准输出和标准错误。用 `CUDA_LOG_FILE` 环境变量，即使应用未对 CUDA 返回值做正确错误检查，也能捕获并识别 CUDA 错误。此法对调试极具威力，但仅靠环境变量本身不能让应用在运行时处理并从 CUDA 错误中恢复。CUDA 的错误日志管理功能还允许向驱动注册一个回调函数，每当检测到错误就调用它。这可用于在运行时捕获并处理错误，也可用于把 CUDA 错误日志无缝集成进应用已有的日志系统。

4.8 节给出 CUDA 错误日志管理功能的更多示例。错误日志管理与 `CUDA_LOG_FILE` 自 NVIDIA 驱动 r570 及以后版本可用。

## 2.1.8 设备函数与主机函数

`__global__` 说明符用于指示 kernel 的入口。即一个将被调用来在 GPU 上并行执行的函数。kernel 通常从主机发射，但也可以在另一个 kernel 内通过动态并行来发射 kernel。

`__device__` 说明符指示一个函数应被编译为 GPU 版本，并可以被其它 `__device__` 或 `__global__` 函数调用。一个函数（含类成员函数、函数对象、lambda）可以同时被指定为 `__device__` 和 `__host__`，如下例所示。

## 2.1.9 变量说明符

CUDA 说明符可用于静态变量声明以控制其放置位置。

- `__device__` 指定变量存储在全局内存
- `__constant__` 指定变量存储在常量内存
- `__managed__` 指定变量作为统一内存存储
- `__shared__` 指定变量存储在共享内存

在 `__device__` 或 `__global__` 函数内不带说明符声明的变量，被尽可能分配到寄存器，必要时分配到 local memory。在 `__device__` 或 `__global__` 函数外不带说明符声明的变量会被分配在系统内存。

### 2.1.9.1 检测设备编译

当一个函数被指定为 `__host__ __device__`，编译器被指示为此函数同时生成 GPU 与 CPU 两份代码。在这种函数中，可能希望用预处理器只为 GPU 或 CPU 副本指定代码。最常见做法是检查 `__CUDA_ARCH__` 是否被定义，如下例所示。

## 2.1.10 线程块集群

从计算能力 9.0 起，CUDA 编程模型包含一个可选的层级——**线程块集群（thread block cluster）**——由多个线程块组成。类似线程块里的线程会被保证同调度到一个流多处理器上，集群里的线程块也被保证同调度到 GPU 的一个 GPU 处理集群（GPC）上。

类似线程块，集群也被组织成一维、二维或三维的线程块集群 grid，如图 5 所示。

集群中线程块数量可由用户定义。CUDA 中最多 8 个线程块的集群大小作为可移植的集群大小受支持。注意，在 GPU 硬件或 MIG 配置过小、不足以支持 8 个多处理器时，最大集群大小会相应减小。识别这些较小配置以及支持大于 8 的线程块集群大小的较大配置是架构相关的，可通过 `cudaOccupancyMaxPotentialClusterSize` API 查询。

集群里的所有线程块都被保证同时调度到单个 GPC 上，并允许集群里的线程块用 cooperative groups API 的 `cluster.sync()` 进行硬件支持的同步。Cluster group 还提供成员函数以按线程数或线程块数查询集群大小（分别用 `num_threads()` 与 `num_blocks()` API）。线程或块在集群组内的排名可分别通过 `dim_threads()` 与 `dim_blocks()` API 查询。

属于某集群的线程块可以访问**分布式共享内存**——即集群里所有线程块共享内存的合并。集群里线程块可以对分布式共享内存中任意地址执行读、写和原子操作。"分布式共享内存"一节给出在分布式共享内存中做直方图的例子。

> **注意**
>
> 在用 cluster 支持发射的 kernel 中，`gridDim` 变量出于兼容仍表示按线程块数计的尺寸。线程块在集群中的排名可通过 Cooperative Groups API 查询。

### 2.1.10.1 在三尖括号记法中用集群发射

线程块集群可以通过编译期 kernel 属性 `__cluster_dims__(X,Y,Z)` 或用 CUDA kernel 发射 API `cudaLaunchKernelEx` 来启用。下面例子展示用编译期 kernel 属性发射集群。用 kernel 属性指定的集群大小在编译期固定，之后该 kernel 可用经典 `<<< , >>>` 发射。如果一个 kernel 用了编译期集群大小，发射该 kernel 时不能修改集群大小。

```cpp
// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    // Kernel invocation with compile time cluster size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension must be a multiple of cluster size.
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```

---

[← 上一章 1.3 CUDA 平台](03_platform.md) ｜ [返回附录 C 首页](README.md)