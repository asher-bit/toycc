// exp08: 第 8 章实验 —— 2 GPU ring 通信量验证
// 【可运行代码】需要 2 张 NVIDIA GPU + NCCL(编译时 -lnccl)
// 编译: nvcc -O3 exp08_nccl_allreduce.cu -o exp08 -lnccl
// 运行: ./exp08 <字节数, 如 1073741824=1GB>
// 对照第 8 章 3.1 节: 每卡发送 = 2×(N-1)/N×S 字节, 时间 = 发送量/带宽
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nccl.h>

#define CHECK_CUDA(call) do {                                     \
    cudaError_t e = (call);                                       \
    if (e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s\n", cudaGetErrorString(e));\
        exit(1);                                                  \
    }                                                             \
} while (0)

#define CHECK_NCCL(call) do {                                     \
    ncclResult_t r = (call);                                      \
    if (r != ncclSuccess) {                                       \
        fprintf(stderr, "NCCL error %s\n", ncclGetErrorString(r));\
        exit(1);                                                  \
    }                                                             \
} while (0)

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s bytes\n", argv[0]); return 1; }
  size_t bytes = strtoull(argv[1], nullptr, 10);

  int n_dev = 0, dev0 = 0, dev1 = 1;
  CHECK_CUDA(cudaGetDeviceCount(&n_dev));
  if (n_dev < 2) { fprintf(stderr, "需要 2 张 GPU(当前 %d 张)\n", n_dev); return 1; }
  CHECK_CUDA(cudaSetDevice(dev0));

  ncclComm_t comm;
  CHECK_NCCL(ncclCommInitAll(&comm, 2, nullptr));    // 2 个 rank 的 communicator

  float *send[2], *recv[2];
  cudaStream_t s[2];
  for (int i = 0; i < 2; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamCreate(&s[i]));
    CHECK_CUDA(cudaMalloc(&send[i], bytes));
    CHECK_CUDA(cudaMalloc(&recv[i], bytes));
    CHECK_CUDA(cudaMemset(send[i], 1, bytes));
  }

  // warmup
  CHECK_NCCL(ncclAllReduce(send[0], recv[0], bytes / sizeof(float),
                           ncclFloat, ncclSum, comm, s[0]));
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  cudaEventRecord(start, s[0]);
  for (int i = 0; i < 10; i++)
    CHECK_NCCL(ncclAllReduce(send[0], recv[0], bytes / sizeof(float),
                             ncclFloat, ncclSum, comm, s[0]));
  cudaEventRecord(stop, s[0]);
  cudaEventSynchronize(stop);
  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);

  double per_rank = 2.0 * (2 - 1) / 2 * bytes;        // 第 8 章公式: N=2
  double bw = per_rank / (ms / 10 / 1e3) / 1e9;
  printf("allreduce %zu bytes: %8.3f ms/次, 每卡发送 %.2f GB, 有效带宽 %.1f GB/s\n",
         bytes, ms / 10, per_rank / 1e9, bw);
  printf("对照第 8 章 3.1 节: N=2 时每卡发送 = S 字节, 与手算一致;\n"
         "再跑 nvidia-smi topo -m 看这两张卡的 NVLink 拓扑。\n");
  ncclCommDestroy(comm);
  return 0;
}
