# 2.5 异步执行

## 2.5.1 什么是异步并发执行？

CUDA 允许下面多种任务的并发或重叠执行，具体包括：

- 主机上的计算
- 设备上的计算
- 从主机到设备的内存搬运
- 从设备到主机的内存搬运
- 给定设备内存内的内存搬运
- 设备之间的内存搬运

并发通过异步接口表达：一个派发函数调用或 kernel 发射立即返回。异步调用通常在派发的操作完成前返回，甚至可能在异步操作启动前返回。随后应用可与原派发操作同时执行其它任务。当最初派发操作的最终结果被需要时，应用必须执行某种形式的同步以确保所讨论操作已完成。并发执行模式的一个典型示例是把主机与设备间的内存搬运与计算重叠，从而减少或消除其开销。

![图 20 用 CUDA stream 做异步并发执行](images/figure20-async-concurrent-execution.png)

> 图 20 用 CUDA stream 做异步并发执行。

总体而言，异步接口通常提供三种主要同步方式与派发操作同步：

- **阻塞方式**：应用调用一个会阻塞、或等待到操作完成的函数
- **非阻塞方式**或**轮询方式**：应用调用一个立即返回并提供操作状态信息的函数
- **回调方式**：在操作完成时执行一个预先注册的函数

虽然编程接口是异步的，实际能并发执行各种操作的能力取决于 CUDA 版本和所用硬件的计算能力——这些细节留给本指南后续章节（见"计算能力"）。

在"同步 CPU 与 GPU"中介绍了 CUDA 运行时函数 `cudaDeviceSynchronize()`，这是一个等待所有此前发出工作完成的阻塞调用。需要 `cudaDeviceSynchronize()` 调用的原因是 kernel 发射是异步的、立即返回。CUDA 同时提供阻塞和非阻塞两种同步 API，甚至支持使用主机侧回调函数。

CUDA 异步执行的核心 API 组件是 CUDA stream 与 CUDA event。本节余下部分解释如何用这些元素在 CUDA 中表达异步执行。

一个相关主题是 CUDA Graphs——它允许预先定义一个由异步操作组成的图，然后以最小开销重复执行。我们在 2.5.9.2 节"用 Stream Capture 介绍 CUDA Graphs"中做入门级覆盖，更全面的讨论见 4.1 节"CUDA Graphs"。

## 2.5.2 CUDA Stream

最基本层面上，CUDA stream 是一种让程序员表达一系列操作的抽象。stream 像一个工作队列，程序可以向其中添加操作——如内存复制或 kernel 发射——按序执行。某 stream 队列前端的操作被执行然后出队，让下一个排队操作来到前端被考虑执行。stream 中操作的执行顺序是顺序的，按入队顺序执行。

应用可同时使用多个 stream。此情形下，运行时根据 GPU 资源状态从有可做工作的 stream 中选择任务执行。可为 stream 指定优先级，作为给运行时影响调度的提示，但不保证特定执行顺序。

stream 中操作的 API 函数调用和 kernel 发射对主机线程是异步的。应用可通过等待 stream 任务为空与某 stream 同步，也可在设备层级同步。

CUDA 有默认 stream，未指定 stream 的操作和 kernel 发射排入此默认 stream。未显式指定 stream 的代码示例隐式用此默认 stream。默认 stream 有一些特定语义，在"阻塞与非阻塞 stream 及默认 stream"小节讨论。

### 2.5.2.1 创建与销毁 CUDA Stream

CUDA stream 可用 `cudaStreamCreate()` 函数创建。该函数调用初始化 stream 句柄，可用在后续函数调用中标识该 stream。

```cpp
cudaStream_t stream;        // Stream handle
cudaStreamCreate(&stream);  // Create a new stream

// stream based operations ...

cudaStreamDestroy(stream);  // Destroy the stream
```

如果应用调用 `cudaStreamDestroy()` 时设备仍在 `stream` 中工作，该 stream 会在销毁前完成其中所有工作。

### 2.5.2.2 在 CUDA Stream 中发射 Kernel

发射 kernel 的常用三尖括号语法也可用于把 kernel 发射到特定 stream。stream 作为 kernel 发射的额外参数指定。下面示例把名为 `kernel` 的 kernel 发射到句柄 `stream` 的 stream 中，`stream` 类型为 `cudaStream_t`，假设此前已被创建：

```cpp
kernel<<<grid, block, shared_mem_size, stream>>>(...);
```

kernel 发射是异步的，函数调用立即返回。假设 kernel 发射成功，kernel 会在 stream `stream` 中执行，应用可在 kernel 执行时在 CPU 上或 GPU 上其它 stream 中执行其它任务。

### 2.5.2.3 在 CUDA Stream 中启动内存搬运

要在 stream 中启动内存搬运，可用函数 `cudaMemcpyAsync()`。此函数类似 `cudaMemcpy()`，但取一个额外参数指定用于内存搬运的 stream。下面代码块中的函数调用在 stream `stream` 中把主机内存 `src` 指向的 `size` 字节复制到设备内存 `dst` 指向的位置。

```cpp
// Copy `size` bytes from `src` to `dst` in stream `stream`
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
```

像其它异步函数调用一样，此函数调用立即返回，而 `cudaMemcpy()` 函数会阻塞到内存搬运完成。为安全访问搬运结果，应用必须用某种同步方式确定操作已完成。

其它 CUDA 内存搬运函数如 `cudaMemcpy2D()` 也有异步变体。

> **注意**
>
> 让涉及 CPU 内存的内存复制异步执行，主机缓冲区必须被固定且页锁定。`cudaMemcpyAsync()` 在使用未固定、未页锁定的主机内存时仍能正确工作，但会回退到同步行为，不会与其它工作重叠。这会抑制使用异步内存搬运的性能收益。建议程序用 `cudaMallocHost()` 分配用于向 GPU 发送或从 GPU 接收数据的缓冲区。

### 2.5.2.4 Stream 同步

与 stream 同步最简单的方式是等待 stream 任务为空。可用 `cudaStreamSynchronize()` 函数或 `cudaStreamQuery()` 函数两种方式。

`cudaStreamSynchronize()` 函数会阻塞到 stream 中所有工作完成。

```cpp
// Wait for the stream to be empty of tasks
cudaStreamSynchronize(stream);

// At this point the stream is done
// and we can access the results of stream operations safely
```

如果不想阻塞，只想快速看 stream 是否为空，可用 `cudaStreamQuery()` 函数。

```cpp
// Have a peek at the stream
// returns cudaSuccess if the stream is empty
// returns cudaErrorNotReady if the stream is not empty
cudaError_t status = cudaStreamQuery(stream);

switch (status) {
    case cudaSuccess:
        // The stream is empty
        std::cout << "The stream is empty" << std::endl;
        break;
    case cudaErrorNotReady:
        // The stream is not empty
        std::cout << "The stream is not empty" << std::endl;
        break;
    default:
        // An error occurred - we should handle this
        break;
};
```

## 2.5.3 CUDA Event

CUDA event 是一种向 CUDA stream 中插入标记的机制。它们本质上是可用于跟踪 stream 中任务进展的示踪粒子。设想向 stream 发射两个 kernel。没有这种跟踪 event，我们只能判断 stream 是否为空。如果有一个操作依赖第一个 kernel 的输出，直到我们知道 stream 为空——此时两个 kernel 都已完成——我们才能安全启动该操作。

用 CUDA event 可以做得更好。通过在第一个 kernel 之后、第二个 kernel 之前向 stream 中入队一个 event，我们可以等待此 event 来到 stream 前端。然后我们可以安全启动依赖操作，知道第一个 kernel 已完成、第二个 kernel 尚未启动。这样用 CUDA event 可以在操作和 stream 之间建立依赖图。这种图类比直接对应后面 CUDA Graphs 的讨论。

CUDA event 还保留时间信息，可用于为 kernel 发射和内存搬运计时。

### 2.5.3.1 创建与销毁 CUDA Event

CUDA event 可用 `cudaEventCreate()` 和 `cudaEventDestroy()` 函数创建和销毁。

```cpp
cudaEvent_t event;

// Create the event
cudaEventCreate(&event);

// do some work involving the event

// Once the work is done and the event is no longer needed
// we can destroy the event
cudaEventDestroy(event);
```

应用负责在不再需要 event 时销毁它们。

### 2.5.3.2 向 CUDA Stream 中插入 Event

CUDA event 可用 `cudaEventRecord()` 函数插入到 stream 中。

```cpp
cudaEvent_t event;
cudaStream_t stream;

// Create the event
cudaEventCreate(&event);

// Insert the event into the stream
cudaEventRecord(event, stream);
```

### 2.5.3.3 在 CUDA Stream 中为操作计时

CUDA event 可用于为包括 kernel 在内的各种 stream 操作计时。event 来到 stream 前端时记录一个时间戳。通过用两个 event 包围 stream 中的 kernel，可以得到 kernel 执行时长的精确计时，如下面代码片段所示：

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaEvent_t start;
cudaEvent_t stop;

// create the events
cudaEventCreate(&start);
cudaEventCreate(&stop);

 // record the start event
cudaEventRecord(start, stream);

// launch the kernel
kernel<<<grid, block, 0, stream>>>(...);

// record the stop event
cudaEventRecord(stop, stream);

// wait for the stream to complete
// both events will have been triggered
cudaStreamSynchronize(stream);

// get the timing
float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
std::cout << "Kernel execution time: " << elapsedTime << " ms" << std::endl;

// clean up
cudaEventDestroy(start);
cudaEventDestroy(stop);
cudaStreamDestroy(stream);
```

### 2.5.3.4 检查 CUDA Event 状态

与检查 stream 状态一样，可阻塞或非阻塞地检查 event 状态。

`cudaEventSynchronize()` 函数会阻塞到 event 完成。下面代码片段中我们向 stream 发射 kernel、然后 event、然后第二个 kernel。可用 `cudaEventSynchronize()` 函数等待第一个 kernel 后的 event 完成，原则上立即启动依赖任务，可能早于 kernel2 完成。

```cpp
cudaEvent_t event;
cudaStream_t stream;

// create the stream
cudaStreamCreate(&stream);

// create the event
cudaEventCreate(&event);

// launch a kernel into the stream
kernel<<<grid, block, 0, stream>>>(...);

// Record the event
cudaEventRecord(event, stream);

// launch a kernel into the stream
kernel2<<<grid, block, 0, stream>>>(...);

// Wait for the event to complete
// Kernel 1 will be  guaranteed to have completed
// and we can launch the dependent task.
cudaEventSynchronize(event);
dependentCPUtask();

// Wait for the stream to be empty
// Kernel 2 is guaranteed to have completed
cudaStreamSynchronize(stream);

// destroy the event
cudaEventDestroy(event);

// destroy the stream
cudaStreamDestroy(stream);
```

CUDA event 可用 `cudaEventQuery()` 函数以非阻塞方式检查完成。下例中我们向 stream 发射 2 个 kernel。第一个 kernel `kernel1` 产生一些想复制到主机的数据，但我们还有 CPU 侧工作要做。下面代码中我们把 `kernel1`、然后一个 event（`event`）、然后 `kernel2` 入队到 stream `stream1`。然后进入 CPU 工作循环，但偶尔瞄一眼看 event 是否完成——表示 kernel1 完成。如是，向 stream `stream2` 启动一个 device 到 host 复制。此法允许 CPU 工作与 GPU kernel 执行以及 device 到 host 复制重叠。

```cpp
cudaEvent_t event;
cudaStream_t stream1;
cudaStream_t stream2;

size_t size = LARGE_NUMBER;
float* d_data;
float* h_data;

// Create some data
cudaMalloc(&d_data, size);
cudaMallocHost(&h_data, size);

// create the streams
cudaStreamCreate(&stream1);   // Processing stream
cudaStreamCreate(&stream2);   // Copying stream
bool copyStarted = false;

//  create the event
cudaEventCreate(&event);

// launch kernel1 into the stream
kernel1<<<grid, block, 0, stream1>>>(d_data, size);
// enqueue an event following kernel1
cudaEventRecord(event, stream1);

// launch kernel2 into the stream
kernel2<<<grid, block, 0, stream1>>>();

// while the kernels are running do some work on the CPU
// but check if kernel1 has completed because then we will start
// a device to host copy in stream2
while ( not allCPUWorkDone() || not copyStarted ) {
    doNextChunkOfCPUWork();

    // peek to see if kernel 1 has completed
    // if so enqueue a non-blocking copy into stream2
    if ( not copyStarted ) {
        if( cudaEventQuery(event) == cudaSuccess ) {
            cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream2);
            copyStarted = true;
        }
    }
}

// wait for both streams to be done
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);

// destroy the event
cudaEventDestroy(event);

// destroy the streams and free the data
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
cudaFree(d_data);
free(h_data);
```

## 2.5.4 来自 Stream 的回调函数

CUDA 提供一种从 stream 内在主机上启动函数的机制。目前有两个函数可用：`cudaLaunchHostFunc()` 和 `cudaAddCallback()`。但 `cudaAddCallback()` 计划弃用，应用应使用 `cudaLaunchHostFunc()`。

**使用 cudaLaunchHostFunc()**

`cudaLaunchHostFunc()` 函数的签名如下：

```cpp
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*func)(void *), void *data);
```

其中：

- `stream`：把回调函数启动到的 stream。
- `func`：要启动的回调函数。
- `data`：指向要传给回调函数的数据的指针。

主机函数本身是一个签名简单的 C 函数：

```cpp
void hostFunction(void *data);
```

`data` 参数指向用户自定义数据结构，函数可解读它。使用这种回调函数时有几点需要注意。特别地，主机函数不能调用任何 CUDA API。

用于统一内存时，提供以下执行保证：

- 该函数执行期间该 stream 被视为空闲。因此，例如该函数总可使用附加到它所入队 stream 的内存。
- 该函数执行的开始与同步同一 stream 中紧接该函数之前记录的 event 效果相同。因此它同步了在该函数之前已"汇合"的 stream。
- 向任何 stream 添加设备工作不会使该 stream 在所有前导主机函数和 stream 回调执行完之前变活跃。因此，例如函数可使用全局附加内存，即使已向另一 stream 添加工作——只要该工作已用 event 排序在该函数调用之后。
- 除上所述外，函数完成不导致 stream 变活跃。若函数后无设备工作，stream 保持空闲；连续主机函数或无设备工作间隔的 stream 回调之间也保持空闲。因此，例如 stream 同步可由 stream 末端的主机函数发信号完成。

### 2.5.4.1 使用 cudaStreamAddCallback()

> **注意**
>
> `cudaStreamAddCallback()` 函数计划弃用并移除，此处为完整性而讨论，因为它仍可能出现在现有代码中。应用应使用或改用 `cudaLaunchHostFunc()`。

`cudaStreamAddCallback()` 函数的签名如下：

```cpp
cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags);
```

其中：

- `stream`：把回调函数启动到的 stream。
- `callback`：要启动的回调函数。
- `userData`：指向要传给回调函数的数据的指针。
- `flags`：目前此参数必须为 0，以向前兼容。

回调函数的签名与我们用 `cudaLaunchHostFunc()` 函数时略有不同。此情形下回调函数是签名如下的 C 函数：

```cpp
void callbackFunction(cudaStream_t stream, cudaError_t status, void *userData);
```

其中函数被传入：

- `stream`：启动回调函数的 stream 句柄。
- `status`：触发回调的 stream 操作状态。
- `userData`：指向传给回调函数的数据的指针。

特别地 `status` 参数会含该 stream 当前错误状态，可能由先前操作设置。与 `cudaLaunchHostFunc()` 函数情形类似，stream 在主机函数完成前不会变活跃和推进到任务，且回调函数内不可调用任何 CUDA 函数。

### 2.5.4.2 异步错误处理

在 CUDA stream 中，错误可源于 stream 中任何操作，包括 kernel 发射和内存搬运。这些错误运行时可能不会传回给用户，直到 stream 被同步——例如等待 event 或调用 `cudaStreamSynchronize()`。有两种方法发现 stream 中可能已发生的错误。

- 使用函数 `cudaGetLastError()`——此函数返回并清除当前 context 中任何 stream 遇到的最后一个错误。若两次调用之间没有其它错误发生，紧接的第二次 `cudaGetLastError()` 会返回 `cudaSuccess`。
- 使用函数 `cudaPeekAtLastError()`——此函数返回当前 context 中最后一个错误，但不清除。

两个函数都以 `cudaError_t` 类型的值返回错误。可用 `cudaGetErrorName()` 和 `cudaGetErrorString()` 函数生成错误的可打印名称。

下面是使用这些函数的示例：

**清单 1 使用 cudaGetLastError() 和 cudaPeekAtLastError() 的示例**

```cpp
// Some work occurs in streams.
cudaStreamSynchronize(stream);

// Look at the last error but do not clear it
cudaError_t err = cudaPeekAtLastError();
if (err != cudaSuccess) {
    printf("Error with name: %s\n", cudaGetErrorName(err));
    printf("Error description: %s\n", cudaGetErrorString(err));
}

// Look at the last error and clear it
cudaError_t err2 = cudaGetLastError();
if (err2 != cudaSuccess) {
    printf("Error with name: %s\n", cudaGetErrorName(err2));
    printf("Error description: %s\n", cudaGetErrorString(err2));
}

if (err2 != err) {
    printf("As expected, cudaPeekAtLastError() did not clear the error\n");
}

// Check again
cudaError_t err3 = cudaGetLastError();
if (err3 == cudaSuccess) {
    printf("As expected, cudaGetLastError() cleared the error\n");
}
```

> **提示**
>
> 当错误出现在某次同步处——尤其是在含许多操作的 stream 中——常难以准确定位错误在 stream 中何处发生。调试此情形的有用技巧是把环境变量 `CUDA_LAUNCH_BLOCKING=1` 设上再运行应用。此环境变量效果是每次单一 kernel 发射后都同步。这可帮助追查是哪个 kernel 或搬运引发错误。同步可能很贵；设此环境变量时应用可能显著慢下来。

## 2.5.5 CUDA Stream 顺序

讲过 stream、event 和回调函数的基本机制后，重要的是考虑 stream 中异步操作的顺序语义。这些语义让应用程序员能以安全方式思考 stream 中操作的顺序。有些特殊场合这些语义可能为性能优化而放松，例如可编程依赖 kernel 发射——通过使用特殊属性和 kernel 发射机制允许两个 kernel 重叠；或用异步批量内存复制函数批量化内存搬运——此时运行时可并发执行不重叠的批量复制。

最重要的是，CUDA stream 是所谓的**有序流（in-order stream）**。这意味着 stream 中操作的执行顺序与这些操作入队顺序相同。stream 中一个操作不能越过其它操作。内存操作（如复制）被运行时跟踪，总会先于下一个操作完成，以让依赖 kernel 安全访问被搬运的数据。

## 2.5.6 阻塞与非阻塞 stream 及默认 stream

CUDA 有两类 stream：阻塞和非阻塞。名称可能略有误导，因为阻塞和非阻塞语义仅指 stream 如何与默认 stream 同步。默认情况下用 `cudaStreamCreate()` 创建的 stream 是阻塞 stream。要创建非阻塞 stream，必须用 `cudaStreamCreateWithFlags()` 函数并加 `cudaStreamNonBlocking` 标志：

```cpp
cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
```

非阻塞 stream 可用 `cudaStreamDestroy()` 以通常方式销毁。

### 2.5.6.1 遗留默认 stream

阻塞与非阻塞 stream 的关键区别在于它们如何与默认 stream 同步。CUDA 提供一个**遗留默认 stream**（也称 NULL stream 或 stream ID 为 0 的 stream），未在 kernel 发射或阻塞 `cudaMemcpy()` 调用中指定 stream 时使用。此默认 stream 在所有主机线程间共享，是阻塞 stream。向此默认 stream 发射操作时，它会与所有其它阻塞 stream 同步——即等所有其它阻塞 stream 完成后才能执行。

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

kernel1<<<grid, block, 0, stream1>>>(...);
kernel2<<<grid, block>>>(...);
kernel3<<<grid, block, 0, stream2>>>(...);

cudaDeviceSynchronize();
```

上面代码片段中默认 stream 行为意味着 kernel2 会等 kernel1 完成，kernel3 会等 kernel2 完成——即便原则上三个 kernel 都能并发执行。通过创建非阻塞 stream 可避开此同步行为。下面代码片段中我们创建两个非阻塞 stream。默认 stream 不再与这些 stream 同步，原则上三个 kernel 都能并发执行。因此我们不能假设 kernel 间任意执行顺序，应执行显式同步（如用相当重的 `cudaDeviceSynchronize()` 调用）以确保 kernel 已完成。

```cpp
cudaStream_t stream1, stream2;
cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

kernel1<<<grid, block, 0, stream1>>>(...);
kernel2<<<grid, block>>>(...);
kernel3<<<grid, block, 0, stream2>>>(...);

cudaDeviceSynchronize();
```

### 2.5.6.2 每线程默认 stream

从 CUDA-7 起，CUDA 允许每个主机线程有自己的独立默认 stream，而非共享的遗留默认 stream。要启用此行为，必须使用 nvcc 编译选项 `--default-stream per-thread` 或定义 `CUDA_API_PER_THREAD_DEFAULT_STREAM` 预处理宏。启用此行为时，每个主机线程会有自己的独立默认 stream，不会像遗留默认 stream 那样与其它 stream 同步。此情形下前述遗留默认 stream 示例会展现与非阻塞 stream 示例相同的同步行为。

## 2.5.7 显式同步

有多种方式显式地将多个 stream 互相同步。

- `cudaDeviceSynchronize()` 等待所有主机线程的所有 stream 的所有前导命令完成。
- `cudaStreamSynchronize()` 取一个 stream 作参数并等待给定 stream 的所有前导命令完成。可用于把主机与特定 stream 同步，让其它 stream 继续在设备上执行。
- `cudaStreamWaitEvent()` 取一个 stream 和一个 event 作参数（event 描述见"CUDA Event"），使调用 `cudaStreamWaitEvent()` 之后添加到给定 stream 的所有命令延迟执行，直到给定 event 完成。
- `cudaStreamQuery()` 让应用知道 stream 中所有前导命令是否已完成。

## 2.5.8 隐式同步

如果从不同 stream 来的两个操作之间提交了对 NULL stream 的任何 CUDA 操作，两个 stream 的操作就不能并发运行——除非这些 stream 是非阻塞 stream（用 `cudaStreamNonBlocking` 标志创建）。

应用应遵循以下指导原则以提升 kernel 并发执行潜力：

- 所有独立操作应在依赖操作之前发出
- 任何形式的同步应尽可能延迟

## 2.5.9 杂项与进阶主题

### 2.5.9.1 stream 优先级

如前所述，开发者可给 CUDA stream 指定优先级。优先级 stream 需用 `cudaStreamCreateWithPriority()` 函数创建。该函数取两个参数：stream 句柄和优先级。一般方案是数值越低优先级越高。给定设备和 context 的可用优先级范围可用 `cudaDeviceGetStreamPriorityRange()` 函数查询。stream 的默认优先级是 0。

```cpp
int minPriority, maxPriority;

// Query the priority range for the device
cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority);

// Create two streams with different priorities
// cudaStreamDefault indicates the stream should be created with default flags
// in other words they will be blocking streams with respect to the legacy default stream
// One could also use the option `cudaStreamNonBlocking` here to create a non-blocking streams
cudaStream_t stream1, stream2;
cudaStreamCreateWithPriority(&stream1, cudaStreamDefault, minPriority);  // Lowest priority
cudaStreamCreateWithPriority(&stream2, cudaStreamDefault, maxPriority);  // Highest priority
```

我们应注意 stream 优先级只是给运行时的提示，主要应用于 kernel 发射，对内存搬运可能不被尊重。stream 优先级不会抢占已在执行的工作，也不保证任何特定执行顺序。

### 2.5.9.2 用 stream capture 介绍 CUDA Graphs

CUDA stream 让程序指定一系列有序的操作——kernel 或内存复制。通过用多个 stream 和 `cudaStreamWaitEvent` 的跨 stream 依赖，应用可指定一整张有向无环图（DAG）的操作。一些应用可能有一段操作序列或操作 DAG 需要在整个执行过程中运行多次。

针对此情形，CUDA 提供 CUDA Graphs 特性。本节介绍 CUDA Graphs 及一种称为 stream capture 的创建机制。CUDA Graphs 的更详细讨论见"CUDA Graphs"。捕获或创建图可帮助减少从主机线程重复调用同一链 API 的延迟和 CPU 开销。取而代之，指定图操作的 API 可只调用一次，然后结果图被执行多次。

CUDA Graphs 按如下方式工作：

1. 图被应用**捕获**。这一步在图首次执行时做一次。图也可用 CUDA Graph API 手动组合。
2. 图被**实例化**。这一步在图捕获后做一次。此步可建立执行图所需的各种运行时结构，以让其组件启动尽可能快。
3. 在余下步骤中，预实例化的图被执行所需多次。由于执行图操作所需的所有运行时结构都已就位，图执行的 CPU 开销被最小化。

**清单 2 用 CUDA Graphs 捕获、实例化和执行一个简单线性图的各阶段（自 CUDA Developer Technical Blog，A. Gray, 2019）**

```cpp
#define N 500000 // tuned such that kernel takes a few microseconds

// A very lightweight kernel
__global__ void shortKernel(float * out_d, float * in_d){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N) out_d[idx]=1.23*in_d[idx];
}

bool graphCreated=false;
cudaGraph_t graph;
cudaGraphExec_t instance;

// The graph will be executed NSTEP times
for(int istep=0; istep<NSTEP; istep++){
    if(!graphCreated){
        // Capture the graph
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        // Launch NKERNEL kernels
        for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
            shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
        }

        // End the capture
        cudaStreamEndCapture(stream, &graph);

        // Instantiate the graph
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
        graphCreated=true;
    }

    // Launch the graph
    cudaGraphLaunch(instance, stream);

    // Synchronize the stream
    cudaStreamSynchronize(stream);
}
```

CUDA Graph 的更多细节见"CUDA Graphs"。

## 2.5.10 异步执行总结

本节要点：

- 异步 API 让我们表达任务并发执行，提供表达各种操作重叠的方式。实际达到的并发性取决于可用硬件资源和计算能力。
- CUDA 异步执行的关键抽象是 stream、event 和回调函数。
- 同步可在 event、stream 和设备层级进行。
- 默认 stream 是一个与所有其它阻塞 stream 同步的阻塞 stream，但不与非阻塞 stream 同步。
- 默认 stream 行为可通过 `--default-stream per-thread` 编译选项或 `CUDA_API_PER_THREAD_DEFAULT_STREAM` 预处理宏，使用每线程默认 stream 来避开。
- stream 可用不同优先级创建——这是给运行时的提示，对内存搬运可能不被尊重。
- CUDA 提供 API 函数以减少或重叠 kernel 发射和内存搬运开销，例如 CUDA Graphs、批量内存搬运和可编程依赖 kernel 发射。

---

[← 上一章 2.4 编写 Tile Kernel](07_writing_tile_kernels.md) ｜ [返回附录 C 首页](README.md)