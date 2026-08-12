# 附录 B：开发环境搭建——Windows 新手从零到能跑 TVM

> 你现在的环境：**Windows，没用过 Linux/WSL/终端**。
> 参与 TVM/LLVM 开发，生态几乎都在 Linux 上（编译、测试、CI 全在 Linux）。
> 所以第一步：在 Windows 里装一个 **WSL2（Windows 自带的 Linux）**。
>
> 本附录是一份**保姆级**步骤，跟着做就行。全程大概 1~2 小时。

---

## 1. 先明白：你为什么要装 Linux

| 你在 Windows 上的现状 | 参与开发需要的 |
|---|---|
| 没有 gcc/clang（第 7 课演示就吃了这个亏） | Linux 自带 gcc，`gcc` 一条命令就有 |
| pip 装 `apache-tvm` 大概率没轮子（Python 3.14 太新） | WSL 里能用官方 Docker 镜像 |
| 各种 cmake/编译依赖难装 | WSL 里 `apt` 一条命令装完 |

**WSL2 = 微软官方支持的在 Windows 里跑 Linux**，不需要双系统。
装好后你有一个"Linux 终端"（像在 Ubuntu 服务器上一样操作）。

---

## 2. 安装 WSL2（大约 20 分钟）

**前提**：Windows 10 2004+ 或 Windows 11。

### 第 1 步：一条命令装 WSL

打开 **PowerShell（管理员）**，运行：

```powershell
wsl --install
```

它会自动：
- 启用 WSL 功能
- 安装最新版 WSL2
- 装 Ubuntu（默认发行版）

**装完重启电脑**。

### 第 2 步：验证 + 设置

重启后打开"开始菜单"找到 **Ubuntu**（或 Windows Terminal），
第一次打开会让你**创建 Linux 用户名和密码**（记好，sudo 要用）。

进去后运行：

```bash
# 确认 WSL 版本是 2 (关键!)
wsl --version

# 更新软件源
sudo apt update
sudo apt upgrade -y
```

> 如果 `wsl --install` 失败（老系统），去微软文档手动开：
> 控制面板 → 启用或关闭 Windows 功能 → 勾选"适用于 Linux 的 Windows 子系统"+
> "虚拟机平台" → 重启 → 装 Ubuntu 应用。
> 文档：`https://learn.microsoft.com/zh-cn/windows/wsl/install`

---

## 3. 在 WSL 里装"编译器开发工具链"（约 20 分钟）

在 WSL 终端里依次运行：

```bash
# 1. 基础编译工具
sudo apt install -y build-essential cmake ninja-build git python3 python3-pip

# 2. LLVM (TVM 的 llvm target 需要; 也可以后装)
sudo apt install -y llvm clang

# 3. 验证
gcc --version      # 应该有输出
cmake --version
git --version
python3 --version
```

**验证通过 = 你的"开发环境"已经达标**。

---

## 4. 装 TVM：两种方式（推荐 B）

### 方式 A：pip 装（最快，先跑起来）

```bash
python3 -m pip install apache-tvm
```

> 优点：一条命令。缺点：`main` 分支的新特性没有；
> 如果 Python 版本不匹配会失败。

### 方式 B：Docker 官方镜像（学习最推荐，和 CI 一致）

先装 Docker Desktop for Windows（勾选"Use WSL 2 based engine"），
然后：

```bash
# 拉取官方 CPU 开发镜像(自带编译好的 tvm)
docker run -it --rm tlcpack/ci-cpu:latest bash

# 进去后验证
python3 -c "import tvm; print(tvm.__version__)"
```

**为什么推荐 Docker**：TVM 官方 CI 就是用这些镜像跑测试的。
你在镜像里跑，环境和 CI 完全一致——**复现问题、跑测试不会因环境差异翻车**。

### 方式 C：从源码编译（进阶，参与开发时必须会）

第 16 课详讲。先知道流程即可：

```bash
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
mkdir build && cd build
cp ../cmake/config.cmake .    # 复制默认配置
# 编辑 config.cmake, 打开 USE_LLVM=ON (去掉注释)
cmake .. && make -j$(nproc)    # 编译(第一次要十几分钟)
cd ../python && pip install -e .
```

---

## 5. Windows / WSL 协同的常用技巧

| 想做什么 | 怎么做 |
|---|---|
| Windows 和 Linux 互相访问文件 | WSL 里 `/mnt/c/...` 就是你的 C 盘 |
| 用 VS Code 编辑 WSL 里的代码 | 装 VS Code + "WSL" 扩展，`code .` 直接打开 |
| 终端美化 | 用 Windows Terminal（微软商店装） |
| 从 Windows 运行 WSL 命令 | PowerShell 里直接输 `wsl <命令>` |
| 图形界面 | WSLg 已内置（新版 WSL 支持 GUI 应用） |

**强烈建议**：以后代码都放 WSL 的文件系统里（`~/project`），
不要放 `/mnt/c/...`（跨文件系统访问慢）。

---

## 6. 你的第一个"开发环境验证"清单

装完后逐条打勾：

- [ ] `wsl --version` 显示 2.x
- [ ] `gcc --version` 有输出
- [ ] `python3 -c "import tvm"` 成功（方式 A 或 B）
- [ ] 能在 WSL 里运行 `python -m toycc.examples.demo`（把 toycc 拷进来）
- [ ] 会用 `git clone` 拉一个仓库

> 到这里，你就有资格进入第 16 课：真实的工程开发流程。

---

## 7. 课后答疑

**Q：必须用 Linux 吗？Windows 不行吗？**
A：能装但坑极多（编译依赖、路径、线程库）。除非你只想"跑一下 tvm"
（那 pip 装也行），否则**参与开发请用 WSL2/Docker**。这不是懒，是
社区共识：开发环境 = Linux。

**Q：WSL2 和双系统哪个好？**
A：WSL2 适合开发（文件共享、不重启、够快）；双系统性能最好但麻烦。
初学 WSL2 完全够。

**Q：装 Docker 很占空间吧？**
A：镜像几个 GB，但对学习值得。而且它能保证"环境永远能用、坏了删了重来"。

**Q：我电脑配置一般，跑得动吗？**
A：纯 CPU 编译学习完全够（tvm 的 CPU 版本不需要显卡）。
大模型推理优化才需要 GPU，那是后话。

---

## 8. 小结

- WSL2 是 Windows 上参与 Linux 生态开发的标准姿势
- 装完三步：`wsl --install` → 装工具链 → 装 tvm（pip 或 Docker）
- 用官方 Docker 镜像 = 和 CI 环境一致，问题不折腾
- 通关清单 5 条全打勾，就可以进第 16 课
