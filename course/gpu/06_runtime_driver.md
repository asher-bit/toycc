# GPU 第 6 章：Runtime / Driver——模块、内存、Stream、Event 与提交

## 1. 本章目标

- 区分 CUDA Runtime API 与 CUDA Driver API；
- 解释 module、function、context、stream、event 和 memory allocation 的生命周期；
- 理解异步 launch、错误延迟和同步边界；
- 把第 29、30 课的二进制与命令提交串成可排查链路。

## 2. 编译器产物如何真正执行

```text
host application
  → device/context 初始化
  → module load（cubin/PTX/fatbin）
  → function lookup
  → device memory / stream / event
  → kernel launch
  → asynchronous execution
  → event/synchronize/callback
  → error query / profiling
```

编译成功只代表有一个可加载候选。加载时还要检查目标架构、driver 支持、符号、参数布局、资源元数据和地址有效性。

## 3. Runtime API 与 Driver API

Runtime API 更高层，负责常见设备、内存、stream 和 kernel launch 用法；Driver API 更接近 context、module、function、virtual memory 和显式加载。很多框架在上层用 Runtime，在需要模块管理、JIT、插件或细粒度控制时使用 Driver。

```cpp
// 示意：Driver API 的对象关系
CUcontext ctx;
CUmodule module;
CUfunction fn;
cuInit(0);
cuCtxCreate(&ctx, 0, device);
cuModuleLoad(&module, "kernel.cubin");
cuModuleGetFunction(&fn, module, "add_kernel");
cuLaunchKernel(fn, grid_x, 1, 1, block_x, 1, 1,
               shared_bytes, stream, args, nullptr);
```

示例省略错误检查和版本差异；真实代码必须检查每个返回码，并理解哪些调用是同步、哪些只把工作放入 stream。

## 4. Stream 与 Event

同一 stream 内通常保持提交顺序；不同 stream 之间没有天然的先后关系。event 可以记录一个 stream 中的点，并作为其他 stream 的等待依赖，也可以用于测量 GPU 时间。

```text
stream A: H2D → kernel A → event E
stream B: wait E → kernel B
```

host 返回不等于 device 已完成。一个异步 kernel 的错误可能在后续同步或下一次 API 调用时才暴露，因此排查时要在明确边界调用同步和错误查询。

## 5. 内存管理

需要区分：

- device allocation 与 host pinned memory；
- synchronous copy 与 async copy；
- memory pool 与直接分配；
- virtual address 与 physical allocation；
- unified memory 的迁移/预取与显式 H2D/D2H；
- stream-ordered allocation 的生命周期。

性能问题常常不是 kernel 本身，而是频繁分配、隐式同步、page fault、错误 stream 或 host memory 未固定造成的。

## 6. 错误排查矩阵

| 现象 | 首先检查 |
|---|---|
| module load failed | driver/toolkit、架构、PTX/cubin、符号、依赖 |
| invalid argument | 参数类型、指针、对齐、grid/block、shared bytes |
| launch 后结果错 | 边界、同步、stream、生命周期、data race |
| launch 返回很快但总体慢 | 隐式同步、H2D/D2H、频繁 launch、JIT |
| 错误在很晚才出现 | 异步错误，增加明确同步与错误查询 |
| 多 stream 结果不稳定 | 缺少 event 依赖、复用 buffer、host 生命周期 |

## 7. 与第 27~30 课的连接

- 第 27 课：用 ISS/cycle model 验证指令和硬件时序；
- 第 28 课：明确原子、fence、barrier 和内存序；
- 第 29 课：生成、重定位、打包和加载 code object；
- 第 30 课：context、stream、event、MMU、command buffer 和 launch；
- 本章：从应用 API 角度把这些组件串成一次真实执行。

## 8. 练习

1. 写一个带 stream/event 的异步 copy + kernel pipeline；
2. 故意提前释放 device buffer，观察错误如何出现；
3. 用 Runtime API 和 Driver API 分别加载同一 kernel；
4. 画出一个模块从 `cubin` 文件到 `cuLaunchKernel` 的对象生命周期。

参考：[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)、[CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/)。

