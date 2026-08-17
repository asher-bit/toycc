// exp06: 第 6 章实验 —— Driver API 最小 module loader
// 【可运行代码】需要 NVIDIA GPU + CUDA Toolkit >= 11.8
// 编译: nvcc -O3 -arch=sm_86 exp06_module_loader.cu -o exp06 -lcuda
// 运行: ./exp06 exp01.cubin   (先按 exp02 生成 exp01.cubin)
// 对照第 6 章: cuInit → cuCtxCreate → cuModuleLoad → cuModuleGetFunction
//             → cuLaunchKernel → 同步 → 查错 的完整对象链
#include <cstdio>
#include <cstdlib>
#include <cuda.h>

#define CHECK(call) do {                                          \
    CUresult r = (call);                                          \
    if (r != CUDA_SUCCESS) {                                      \
        const char* s; cuGetErrorString(r, &s);                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                __FILE__, __LINE__, s);                           \
        exit(1);                                                  \
    }                                                             \
} while (0)

// 由 Driver API 加载的 kernel(源码被 nvcc 编进 cubin, 这里只是签名参考)
extern "C" __global__ void add_coalesced(const float* a, const float* b,
                                         float* out, int n);

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s kernel.cubin\n", argv[0]); return 1; }

  CUcontext ctx = nullptr;
  CUmodule mod = nullptr;
  CUfunction fn = nullptr;
  CHECK(cuInit(0));                                    // ① 初始化 driver
  CHECK(cuCtxCreate(&ctx, 0, 0));                      // ② context(错误状态也挂这里)
  CHECK(cuModuleLoad(&mod, argv[1]));                  // ③ 模块加载(架构校验在此)
  CHECK(cuModuleGetFunction(&fn, mod, "add_coalesced"));  // ④ 按名字查 kernel 入口

  const int n = 1 << 20;
  const size_t bytes = n * sizeof(float);
  CUdeviceptr d_a, d_b, d_out;
  CHECK(cuMemAlloc(&d_a, bytes));
  CHECK(cuMemAlloc(&d_b, bytes));
  CHECK(cuMemAlloc(&d_out, bytes));
  CHECK(cuMemsetD8(d_a, 0, bytes));
  CHECK(cuMemsetD8(d_b, 0, bytes));

  void* args[] = { &d_a, &d_b, &d_out, (void*)&n };
  CHECK(cuLaunchKernel(fn,                             // ⑤ 发射(异步!)
                       1024, 1, 1,                    // grid
                       256, 1, 1,                     // block
                       0, nullptr, args, nullptr));   // shared / stream / 参数
  CHECK(cuCtxSynchronize());                           // ⑥ 同步边界
  // 异步错误在同步点才暴露: 正确做法是检查 cuCtxSynchronize 的返回值,
  // 出错时用 cuGetErrorString 打印(第 6 章 sticky error)

  printf("launch 完成(异步错误在同步点才暴露, 见第 6 章 sticky error)\n");
  cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_out);
  cuModuleUnload(mod);
  cuCtxDestroy(ctx);
  return 0;
}
