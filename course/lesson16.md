# 第 16 课：真实工程开发流程——从"会概念"到"会上手干活"

> 目标：把"读代码"变成"改代码、跑测试、调 bug、提 PR"的完整流程。
> 前置：附录 A（C++ 阅读）、附录 B（WSL2 环境）、第 10 课（git 流程）。
> 环境：WSL2 里的 Linux（你已装好）。
>
> 这课是"上班第一天"的完整预演。

---

## 1. 一天的开发日常是什么样

在真实编译器团队（TVM/LLVM/MLIR），一天大概是：

```
上午:  git pull → 看新代码/PR → 复现一个 bug
       (改代码 → 跑相关测试 → 改到通过)
下午:  写新 pass 或修 bug → 写测试 → 跑 CI → 提 PR
       (等 review → 根据意见修改 → 合并)
```

**核心循环**（你从第 3 课就知道的）：

```
改代码 → 跑测试 → 看失败原因 → 再改 → 通过 → 提交
```

> **原理深挖：为什么"跑测试"要快，比什么都重要？**
>
> 这条循环的命运取决于一个参数：**从"改完"到"看到失败原因"的时间**。
>
> - 如果一次测试要 30 分钟：你一天顶多迭代 5 次，改错地方很容易
>   "在错误的道路上走到天黑才回头"。
> - 如果一次只要 2 秒：你可以疯狂试探——改一个数、跑一下、再改，
>   心里随时知道对错。这是**调试时最有用的循环**。
>
> 所以真实工程里处处在优化"反馈速度"：
> - 拆分测试目录，`-k fuse` 只跑相关的那一个，而不是跑全量
> - 用缓存/增量编译，第二次编译只重编改动部分
> - 环境问题用 Docker 固定，避免"换个机器就变成玄学"
>
> **对你入职的意义**：第一天就把"最快失败循环"建立起来——
> 找到"改哪一行最快能验证"的命令，记下来。这决定了你每天能学多少、
> 能试探多少边界。**慢测试是代码走样的温床，快测试是迭代的自由。**

本课把这条循环里的每一步"工具化"。

---

## 2. 从源码构建 TVM（参与开发必须会）

pip 装的 tvm 只能"用"，改源码必须**从源码编译**。

### 2.1 克隆 + 配置

```bash
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
git checkout main
mkdir build && cd build
cp ../cmake/config.cmake .
```

### 2.2 打开配置，启用 LLVM

```bash
nano config.cmake    # 或 vim
# 找到这一行, 去掉前面的 # :
# set(USE_LLVM OFF)  →  set(USE_LLVM ON)
# 保存退出 (nano: Ctrl+O 保存, Ctrl+X 退出)
```

> `USE_LLVM=ON` 让 TVM 能出口到 LLVM IR 并利用 LLVM 做目标代码生成。
> 这是调试和跑 CPU 模型的关键。

### 2.3 编译 + 装 Python 包

```bash
cmake .. && make -j$(nproc)
# 编译完(第一次 10~30 分钟), 装 python 绑定
cd ../python
pip install -e .
```

### 2.4 验证

```bash
python3 -c "import tvm; print(tvm.__version__)"
python3 -c "from tvm import relax; print(relax.__name__)"   # 确认 relax 在
```

**常见问题**：
- `make` 报错缺依赖 → `sudo apt install -y <缺的东西>`，重新 `make`
- 内存不够（编译会吃很多）→ 加 `-j$(nproc-2)` 减少并行度
- 改了 C++ 源码 → 回到 `build` 目录 `make -j$(nproc)`，python 端自动生效

> **原理深挖：为什么编译器项目不能"pip 一键装"？**
>
> pip 装的 TVM 是**预编译产物**——别人在某个硬件上编好的，你只是"下载使用"。
> 这对"用"够了，但对"改"是致命的：你改了 `fuse_ops.cc`，pip 包里的
> 二进制还是老代码，改了等于没改。
>
> 所以开发模式必须**从源码构建**，它的本质是一个"你自己的产物"流程：
>
> ```
> 源码(C++) ──编译(cmake+make)──> 动态库 libtvm.so ──> Python 绑定(import tvm)
>     改一行代码 ── 重新 make ── 动态库更新 ── Python 端立刻生效
> ```
>
> 这也解释了为什么 TVM 有 100+ 个 `USE_*` 配置开关（LLVM / CUDA / OpenCL...）：
> **同一个源码，不同环境编出不同的库**。`USE_LLVM=ON` 就是把 LLVM 那个
> 巨大依赖编进去——编译器的工作方式就是"把要做后端（第 25 课 LLVM）时
> 才需要的东西，在编译期决定好"。这和你在第 5 课学的"编译期 vs 运行时"
> 是同一件事，只不过发生在**构建系统**里。

---

## 3. 跑测试——你的安全网

### 3.1 TVM 的测试体系

```
tests/python/relax/        ← Relax 相关(pass 测试在这)
tests/python/tir/          ← TIR 相关
tests/python/ir/           ← IR 基础设施
tests/python/topi/         ← 算子实现
```

### 3.2 常用命令

```bash
# 跑单个测试文件
pytest tests/python/relax/test_transform_fuse_ops.py

# 跑指定测试(用 -k 过滤名字)
pytest tests/python/relax/test_transform_fuse_ops.py -k "conv2d"

# 跑整个目录(慢, 慎用)
pytest tests/python/relax/transform/

# 跑出错就停, 打印详细
pytest -x -v tests/python/relax/test_transform_fuse_ops.py
```

### 3.3 测试长什么样（读法）

TVM 测试通常是"构造小模型 → 跑 pass → 对比期望结果"：

```python
def test_conv_fusion():
    mod = build_conv_relu_model()            # 构造 conv+relu 的 IRModule
    mod = tvm.ir.transform.Sequential([
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(),
    ])(mod)
    # 期望: 融合后 call_tir 的数量变少
    assert count_fused_calls(mod) == 1       # 断言!
```

**你写测试时的铁律**：
1. 先写**会失败的**测试（确认它真的能测出问题）
2. 跑通你的修复
3. 留一个**回归测试**（防止将来别人改坏）

### 3.4 MLIR / LLVM 的测试（lit）

如果你走 MLIR 方向，测试是 `lit`（文本匹配）：

```mlir
// RUN: mlir-opt %s -my-pass | FileCheck %s
// CHECK: toy.transpose
%0 = "toy.transpose"(%t) : ...
```

`// RUN:` 定义命令，`// CHECK:` 定义期望输出。**看到这两个注释 = 这是测试文件**。

---

## 4. 调试 pass——最常用的三招

### 4.1 第一招：打印 IR（百试百灵）

```python
# 每个 pass 前后打印 IR, 看它改了什么
from tvm.ir import transform
with transform.PassContext(config={
    "relax.transform.print_all": True,     # 打印每个 pass 前后的 IR
}):
    optimized = pipeline(mod)
```

**这就是第 9 课深拓展 B 说的"print_all"**。看到 pass 前后 IR 的差异，
你就知道 pass 生效没、改对没。

### 4.2 第二招：小步隔离

"pass 组合拳"出问题，不知道是哪个 pass 干的 → 一个个跑：

```python
g1 = relax.transform.FuseOps()(mod)
print(g1)               # 先只跑这一个, 看对不对
g2 = relax.transform.FoldConstant()(g1)
print(g2)               # 再叠下一个
```

**二分法**：`Sequential([a,b,c,d])` 出 bug → 试 `[a]`、`[a,b]`、
`[a,b,c]`，定位是哪一步引入的错误。

### 4.3 第三招：C++ 崩溃用 gdb

Python 报错 + C++ 段错误（Segmentation fault）时：

```bash
# 在 gdb 下跑 python 脚本, 崩溃时看调用栈
gdb --args python3 my_script.py
# gdb 里: run      ← 运行
#          bt       ← 崩溃时打印 backtrace(调用栈)
#          up/down  ← 在栈里上下移动看帧
```

**调用栈（backtrace）怎么读**：从下往上，最下面是"谁发起的"，
最上面是"崩在哪一行"。通常崩在 `as<>()` 强转失败（类型不对）或
空指针解引用。

> **手算：拿一段真实风格的回溯练读法**
>
> ```
> #0  0x... in tvm::relax::FuseOpsImpl::GetOpPattern(OpPatternKind*) 
> #1  0x... in tvm::relax::FuseOpsImpl::FuseOps() 
> #2  0x... in tvm::relax::transform::FuseOps(IRModule) 
> #3  0x... in tvm::runtime::TypedPackedFunc<...>::Call()   ← Python 入口
> #4  0x... in PyCFunc_  ← 从 Python 调进 C++ 的桥
> ```
>
> 读法三步：
> 1. **看 #0（最上层）**：崩在 `GetOpPattern`——说明"取融合模式"这一步
>    拿到一个空/非法对象（多半是 `as<>()` 强转失败：Add 被当成别的算子）。
> 2. **往下走 #1**：`FuseOps()` 调用它——确认是融合 pass 主流程。
> 3. **看 #4（最底层）**：从 Python 进来——确认不是 C++ 端独立崩溃。
>
> 结论就一句话：**融合 pass 里对某个算子的"类型假设"错了**。修复方向：
> 在 `GetOpPattern` 里对未知算子类型加判断（对照第 1 课 `__post_init__`
> 的"尽早报错"哲学——C++ 侧同样要"谁知道就校验谁"）。

### 4.4 数值不对 → 用参考执行器

第 2 课的思想：写个 numpy 参考实现对比。TVM 里：

```python
tvm.testing.assert_allclose(compiled_out, reference_out, rtol=1e-4, atol=1e-5)
```

---

## 5. git 协作——从"会用"到"专业"

你 git 熟练，这里只补充**开源编译器协作的特殊约定**：

### 5.1 提交信息风格（看项目 CONTRIBUTING）

```
[Relax][Transform] Fuse ops more aggressively (#12345)
 ^       ^            ^
组件    子模块      一句话说明 (+PR号)
```

提交信息要**解释为什么**，不是复述代码干了啥。

### 5.2 PR 检查清单

- [ ] 代码有测试覆盖（没有测试的改动大概率被拒）
- [ ] 跑过相关测试（`pytest tests/python/relax/...`）
- [ ] 跑过格式检查（TVM 用 `pre-commit`，改完跑 `pre-commit run --all-files`）
- [ ] 一个 PR 只做一件事
- [ ] 写了清晰的 PR 描述（问题、方案、验证）

### 5.3 处理 CI 失败

PR 提交后 CI 跑全量测试，红了 → 点进去看哪一步挂了：
- **lint 失败** → 格式问题，`pre-commit run --all-files` 修
- **某个测试失败** → 本机 `pytest <那个测试>` 复现，修好重推
- **超时/环境问题** → 可能是 CI 偶发，`@tvm-bot rerun`（或注释触发重跑）

---

## 6. 一个完整的"第一个任务"演练（跟着做）

假设任务：**"给融合 pass 加一条新规则：允许 `relu(sigmoid(x))` 也融合进 conv"**。

### 步骤 1：定位代码

```bash
grep -rn "sigmoid" src/relax/transform/fuse_ops.cc   # 找到处理激活的地方
```

### 步骤 2：写一个会失败的测试

```python
# tests/python/relax/test_fusion_sigmoid.py
def test_conv_sigmoid_fusion():
    ...构造 conv → sigmoid 的图...
    ...跑 FuseOps...
    assert 融合后是一个 call_tir     # 先跑, 应该失败
```

### 步骤 3：实现

在 `fuse_ops.cc` 的融合规则里，把 sigmoid 加入"可吸收的逐元素算子"集合，
然后重新 `make`。

### 步骤 4：跑测试直到通过

```bash
pytest tests/python/relax/test_fusion_sigmoid.py -v
```

### 步骤 5：跑相关回归（防止改坏别的）

```bash
pytest tests/python/relax/ -k "fuse or fusion"
```

### 步骤 6：提交 + PR

```bash
git add -A && git commit -m "[Relax][Transform] support sigmoid in fusion"
git push origin my-fusion-sigmoid-branch
# 去 GitHub 开 PR
```

**这就是真实任务的完整生命周期**。你第一次可能花一天，
熟练后半小时一个。

---

## 7. 调试的进阶工具（遇到再学，不必现在全会）

| 工具 | 干什么 | 什么时候用 |
|---|---|---|
| `perf` | 性能分析（热点在哪） | 优化 kernel 时 |
| `valgrind` | 内存错误检测 | C++ 崩溃但 gdb 没头绪时 |
| `nsight` / `ncu` | GPU 分析 | GPU kernel 优化时 |
| `gdb` | 断点/回溯 | C++ 崩溃时 |
| `pdb` | Python 断点 | Python 端逻辑问题时 |
| `print` / `LOG` | 最朴素但最常用 | 任何时候 |

**原则**：先用 `print` 快速定位，再用专业工具深挖。
不要一上来就上 gdb/perf（工具本身有学习成本）。

---

## 8. 公司里常见的"开会/讨论"语言（再补一批）

| 你听到 | 意思是 |
|---|---|
| "这 pr 能 merge 吗" | 能不能合并（CI 过了 + review 通过） |
| "跑一下 sanity" | 快速跑一遍基本测试确认没崩 |
| "regression" | 回归（改坏了过去的功能） |
| "这个改动要不要 rebase" | 要不要把你的分支基于最新 main 重放 |
| "LGTM" | Looks Good To Me（review 通过） |
| "blocked by ..." | 被某件事卡住了 |
| "nit" | 小问题（拼写/格式），不改也行 |
| "candidate / RFC" | 设计方案征求意见 |

---

## 9. 实验（本课没有自动实验，但给你一个自测）

```bash
python -m course.runner 16
```

看到的是本课的要点清单。真正的"实验"是下面这个动手自测：

**自测：在 WSL 里完成一次最小闭环**

```bash
# 1. 环境就绪(附录 B 通关)
wsl --version && gcc --version

# 2. 能跑 toycc
cd ~ && git clone <你的 toycc 路径> && python3 -m toycc.examples.demo

# 3. 装 tvm 成功
python3 -c "import tvm"

# 4. 从源码编译(进阶)
git clone --recursive https://github.com/apache/tvm tvm && cd tvm/build ...

# 5. 跑一个测试
pytest tests/python/relax/ -k "fuse" -x
```

**通关标准**：能说出"跑一个测试、看 CI 失败原因、改代码重推"
这三步各自用什么命令。

---

## 10. FAQ

**Q：编译一次 tvm 要多久？每次改代码都要重编吗？**
A：第一次 10~30 分钟；之后增量编译只重编改动的部分，几十秒。
改 Python 侧不需要重编；改 C++ 才需要 `make`。

**Q：我该先学 C++ 还是先跑流程？**
A：**先跑流程**（build/测试/跑通），过程中遇到 C++ 代码用附录 A 查。
"先有环境能跑，再学语言"比"学完语言再动手"效率高得多。

**Q：写测试重要吗？我不想写。**
A：在开源编译器里**没有测试 = 不会合并**。测试是"你改动正确性"的证据，
也是维护者的第一检查项。把"写测试"当"给自己上保险"，不是负担。

**Q：我的 toycc 练习和真实 TVM 差距太大，怎么过渡？**
A：先在 toycc 上把"改代码→跑验证"的循环做熟（第 10 课任务 A），
再到真实 TVM 做同一件事（MyFirstPass）。逻辑一样，只是工具变了。
**过渡的关键不是知识，是手感。**

---

## 11. 本课小结

- 开发日常 = **改代码 → 跑测试 → 调 bug → 提 PR** 的循环
- 从源码编译：clone → 配置(USE_LLVM) → cmake → make → pip -e
- 测试：`pytest`（TVM） / `lit`（MLIR/LLVM），先写会失败的测试
- 调试三招：**print IR → 小步隔离 → gdb 回溯**
- git 协作：提交信息规范、PR 清单、CI 失败处理
- 完成"第一个任务演练"六步 = 你已经能干活了

**下一步**：第 17 课——模型是怎么"进"编译器的：前端导入、ONNX、
legalize 下降、动态形状、运行时 VM。等学到第 20 课，再回来按
知识地图选方向深入。
需要我陪你走哪一步？可以一起从"WSL 环境搭建"或"第一个 TVM 任务"开始。

---

## 深层拓展：工程里的三个"软技能"

### A. 怎么"读"一个你不懂的报错？

报错信息永远是**从下往上读**：最下面一行是"哪里炸了"，往上是"谁调用的"。
新手常犯的错误是从第一行开始读，被一堆模板/栈帧淹没。
**先找最后一行的"error:"或"assert failed"**，再回头看调用链。

### B. CI 挂了，怎么定位？

CI 是"自动化测试"——挂了说明你的改动破坏了某个测试。步骤：
1. 在本地**复现**（跑同一个测试命令）
2. 如果本地不挂，多半是**环境差异**（版本/路径/依赖）
3. 二分法：回退你的一半改动，看还挂不挂

CI 不是敌人，是"帮你抓 bug 的同事"。

### C. 什么时候该问人，什么时候该自己扛？

经验法则：**卡住 30 分钟就求助**。但求助时不要说"我不会"，
要说"我试了 A、B、C，卡在 D"——这既是尊重别人时间，
也是梳理自己思路的过程。**会提问，是工程师的核心竞争力**。

---

**导航**：⬅ [上一节](lesson15.md)（第 15 课 · 硬件必修课）　｜　[下一节](lesson17.md)（第 17 课 · 模型导入）➡
