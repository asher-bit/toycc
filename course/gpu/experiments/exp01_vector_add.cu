// exp01: 第 1 章实验 —— 合并访问与有效带宽
// 【可运行代码】需要 NVIDIA GPU + CUDA Toolkit >= 11.8
// 编译: nvcc -O3 -arch=sm_86 exp01_vector_add.cu -o exp01
// (sm_86 换成 nvidia-smi 里你的架构)
// 运行: ./exp01
// 预期观察: 连续访问版有效带宽远高于 stride 版(对应第 1 章 7.2 节的事务数手算)
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

// 连续访问: 线程 k 读写第 k 个元素(第 1 章 7.1 节: 1 条 cache line)
__global__ void add_coalesced(const float* a, const float* b,
                              float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

// stride 访问: 线程 k 读写第 k*32 个元素(第 1 章 7.2 节: 32 笔事务)
__global__ void add_strided(const float* a, const float* b,
                            float* out, int n) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 32;
  if (i < n) out[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 22;                     // 4M 元素(16 MB)
  const size_t bytes = n * sizeof(float);
  float *d_a, *d_b, *d_out;
  CHECK(cudaMalloc(&d_a, bytes));
  CHECK(cudaMalloc(&d_b, bytes));
  CHECK(cudaMalloc(&d_out, bytes));
  CHECK(cudaMemset(d_a, 0, bytes));
  CHECK(cudaMemset(d_b, 0, bytes));

  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);

  auto bench = [&](const char* name, auto kernel) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    for (int i = 0; i < 5; i++) kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
    CHECK(cudaDeviceSynchronize());          // warmup
    cudaEventRecord(start);
    for (int i = 0; i < 20; i++) kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    // 有效带宽 = 搬运字节(读 a + 读 b + 写 out) / 时间
    double gb = 3.0 * n * sizeof(float) / 1e9;
    printf("%-14s %8.3f ms  有效带宽 %7.1f GB/s\n", name, ms / 20, gb / (ms / 20 / 1e3));
  };

  bench("coalesced", add_coalesced);
  bench("strided  ", add_strided);
  printf("\n对照第 1 章 7.2 节: stride 版每笔事务只用 4B/32B, 有效带宽应低一个数量级。\n");
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
  return 0;
}
