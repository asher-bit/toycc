# CUDA 编程指南附录图片占位

本目录用于存放本附录（`appendix_cuda/`）各章引用的官方图片。

由于 NVIDIA 《CUDA Programming Guide》原图无法由翻译者保证版权与 CDN 稳定性，本仓库不直接内嵌官方原图。译文中按下面的文件名占位，请按需自行从 NVIDIA 官方页面（https://docs.nvidia.com/cuda/cuda-programming-guide/ 下载原图并按以下命名放入本目录）：

| 占位文件名 | 来源 |
|---|---|
| `figure1-cpu-vs-gpu-transistors.png` | 1.1.2 节"图 1 GPU 把更多晶体管用于数据处理" |
| `figure2-cuda-cpu-gpu.png` | 1.2.2 节"图 2 CUDA 编程模型视角下的 CPU 与 GPU 组件及其连接" |
| `figure3-grid-of-thread-blocks.png` | 1.2.2.1 节"图 3 线程块网格" |
| `figure4-thread-blocks-on-sms.png` | 1.2.2.1 节"图 4 线程块在 SM 上调度" |
| `figure5-clusters.png` | 1.2.2.1.1 节"图 5 集群中的线程块在 SM 上调度" |
| `figure6-clusters-on-gpc.png` | 1.2.2.1.1 节"图 6 集群内的线程块在 GPC 内的 SM 上调度" |
| `figure7-warp-lanes-masked.png` | 1.2.2.2 节"图 7 不活跃时 warp lane 被屏蔽" |
| `figure8-simt-vs-tile.png` | 1.2.2.3 节"图 8 SIMT 和 tile 编程模型下程序员的视角" |
| `figure9-tile-space-data-movement.png` | 1.2.2.3.2 节"图 9 tile 空间与数据搬运" |
| `figure10-fatbin-container.png` | 1.3.4 节"图 10 可执行或库中的 fatbin 容器可含多版本 GPU 代码" |
| `figure11-grid-of-thread-blocks.png` | 2.3.2 节"图 11 线程块网格" |
| `figure12-coalesced-memory-access.png` | 2.3.4.1 节"图 12 合并的内存访问" |
| `figure13-uncoalesced-memory-access.png` | 2.3.4.1 节"图 13 非合并的内存访问" |
| `figure14-matrix-transpose-global-memory.png` | 2.3.4.1.1 节"图 14 用全局内存做矩阵转置" |
| `figure15-strided-shared-memory-accesses.png` | 2.3.4.2 节"图 15 32 位 bank 大小模式下的跨步共享内存访问" |
| `figure16-irregular-shared-memory-accesses.png` | 2.3.4.2 节"图 16 不规则的共享内存访问" |
| `figure17-bank-structure-32x32.png` | 2.3.4.2.2 节"图 17 32×32 共享内存数组中的 bank 结构" |
| `figure18-bank-structure-32x33.png` | 2.3.4.2.2 节"图 18 32×33 共享内存数组中的 bank 结构" |
| `figure19-tile-space-indexing.png` | 2.4.6.1 节"图 19 partition view 的 tile 空间索引" |
| `figure20-async-concurrent-execution.png` | 2.5.1 节"图 20 用 CUDA stream 做异步并发执行" |

放入文件后图片即自动渲染。如不补图，原文引用会显示为占位 alt 文本，不影响译文的可读性。