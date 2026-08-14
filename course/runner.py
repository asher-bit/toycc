"""课程实验运行器。

用法:
    python -m course.runner              # 列出所有课
    python -m course.runner <编号>       # 运行第 <编号> 课的引导实验
    python -m course.runner all          # 跑完全部可运行的实验

每个实验都在打印"发生了什么"的同时，把关键对象 dump 出来，
好让你对着 course/lessonXX.md 的讲解边看边验证。
"""
from __future__ import annotations

import importlib
import sys

import numpy as np

# Windows 控制台默认 GBK: 强制 UTF-8 输出, 避免特殊字符(³/Δ 等)编码崩溃
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")
from toycc.examples.model import build_model, default_weights  # noqa: E402


def feed():
    f = default_weights()
    f["x"] = np.random.default_rng(1).standard_normal((1, 3, 8, 8))
    return f


def first_out(res):
    """pass 会重命名输出(如 output->mm),统一按位置取第一个输出。"""
    return list(res.values())[0]


# ---------------- 各课实验 ----------------

def lesson01():
    print("=== 第1课:计算图与 IR ===")
    from toycc.examples.model import build_model
    g = build_model()
    print(g.dump())
    print("\n-- 拓扑序(执行顺序) --")
    for n in g.topo_order():
        print(f"  {n.name}")
    print("\n-- conv1 的消费者是谁? --")
    for c in g.consumers("conv1"):
        print(f"  {c}")
    print("\n观察: IR 只描述计算依赖,不关心数据怎么在内存排布。")


def lesson02():
    print("=== 第2课:参考执行器与正确性 ===")
    from toycc.runtime import evaluate
    g = build_model()
    out = evaluate(g, feed())
    for k, v in out.items():
        print(f"  {k}: shape={v.shape} mean={v.mean():.4f}")
    out2 = evaluate(g, feed())
    same = np.allclose(out["output"], out2["output"])
    print(f"\n固定随机种子 => 两次执行结果一致: {same}")
    print("观察: 参考执行器是'金标准',后面每个 pass 都要跟它对比。")


def lesson03():
    print("=== 第3课:算子融合 ===")
    from toycc.passes import run_passes
    from toycc.runtime import evaluate

    g0 = build_model()
    out0 = evaluate(g0, feed())
    g1 = run_passes(g0, ("fusion",), verbose=False)
    print("\n-- 融合后 --")
    print(g1.dump())
    diff = np.max(np.abs(out0["output"] - first_out(evaluate(g1, feed()))))
    print(f"\n融合前后 max|Δ| = {diff:.2e} (应接近 0)")

    print("\n-- 实验:改变运算顺序会破坏语义 --")
    from toycc.ir import Graph, OpAttrs
    gb = Graph("broken")
    gb.add_input("x")
    w = default_weights()
    for name in w:
        gb.add_input(name)
        gb.set_constant(name, w[name])
    c = gb.add_op("conv", ["x", "conv1_w"], "conv1", OpAttrs(
        kernel=(3, 3), stride=(1, 1), pad=(1, 1, 1, 1)))
    r = gb.add_op("relu", ["conv1"], "r")          # 注意顺序:先 relu
    _ = gb.add_op("add", ["r", "bias1"], "out")     # 再加 bias
    gb.mark_output("out")
    print("  这个图的顺序是 conv→relu→add(bias)")
    fused = run_passes(gb, ("fusion",), verbose=False)
    print(f"  融合后: {fused.dump()}")
    ref = first_out(evaluate(gb, feed()))
    got = first_out(evaluate(fused, feed()))
    print(f"  融合后 max|Δ| = {np.max(np.abs(ref - got)):.3e}")
    print("  结论: relu(conv)+bias != relu(conv+bias), 融合必须语义等价。")


def lesson04():
    print("=== 第4课:布局优化 ===")
    from toycc.passes import run_passes
    from toycc.runtime import evaluate
    g0 = build_model()
    out0 = evaluate(g0, feed())
    g1 = run_passes(g0, ("fusion",), verbose=False)
    g2 = run_passes(g1, ("layout",), verbose=False)
    print(g2.dump())
    diff = np.max(np.abs(out0["output"] - first_out(evaluate(g2, feed()))))
    print(f"\n布局变换后 max|Δ| = {diff:.2e}")
    n_lt = sum(1 for n in g2.nodes.values() if n.op_type == "layout_transform")
    print(f"布局 pass 插入 {n_lt} 个 transform(含权重/偏置的; 常量折叠后会消失)")
    print("运行时真正的数据搬移只有 2 次边界(见第5课)")


def lesson05():
    print("=== 第5课:常量折叠 ===")
    from toycc.passes import run_passes
    from toycc.runtime import evaluate
    g0 = build_model()
    out0 = evaluate(g0, feed())
    g3 = run_passes(g0, ("fusion", "layout", "constfold"), verbose=False)
    print(g3.dump())
    n_runtime = [n for n in g3.nodes.values()
                 if n.op_type != "layout_transform"
                 or n.inputs[0] in g3.inputs]
    print(f"\n运行时真正要执行的算子(含数据搬移): {len(g3.nodes)} 个")
    print(f"权重/偏置的布局转换已被折叠成常量(见上方 input 列表多出的 *_lt_* 常量)")
    diff = np.max(np.abs(out0["output"] - first_out(evaluate(g3, feed()))))
    print(f"折叠后 max|Δ| = {diff:.2e}")


def lesson06():
    print("=== 第6课:内存规划 ===")
    from toycc.passes import run_passes
    from toycc.passes.memory import MemoryPlanningPass, report
    g0 = build_model()
    g3 = run_passes(g0, ("fusion", "layout", "constfold"), verbose=False)
    _, allocs = MemoryPlanningPass({"x": (1, 3, 8, 8)})(g3)
    print(report(g3, allocs, {"x": (1, 3, 8, 8)}))


def lesson07():
    print("=== 第7课:代码生成 ===")
    from toycc.passes import run_passes
    from toycc.passes.memory import MemoryPlanningPass, infer_shapes
    from toycc.codegen import emit_c, emit_python
    from toycc.runtime import evaluate

    g0 = build_model()
    f = feed()
    out0 = evaluate(g0, f)
    g3 = run_passes(g0, ("fusion", "layout", "constfold"), verbose=False)
    _, allocs = MemoryPlanningPass({"x": (1, 3, 8, 8)})(g3)
    shapes = infer_shapes(g3, {"x": (1, 3, 8, 8)})
    amap = {a.tensor: a for a in allocs}

    c_code = emit_c(g3, amap, shapes)
    py_code = emit_python(g3, amap, shapes)
    open("out.c", "w", encoding="utf-8").write(c_code)
    open("out.py", "w", encoding="utf-8").write(py_code)

    spec = importlib.util.spec_from_file_location("gen_out", "out.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    got = gen.run(f["x"]).reshape(out0["output"].shape)
    print(f"  生成 out.c / out.py")
    print(f"  生成的 python 后端执行结果与参考 max|Δ| = "
          f"{np.max(np.abs(got - out0['output'])):.2e}")
    print("  第1行 C 代码:")
    print("\n".join(c_code.splitlines()[:3]))


def lesson08():
    print("=== 第8课:真实 TVM (Relax) 源码导读 ===")
    print("""
本机没装 tvm,这节课是"读源码 + 对照 toycc"。对应文件:
  src/relax/transform/fuse_ops.cc       -> 对照 toycc/passes/fusion.py
  src/relax/transform/fold_constant.cc  -> 对照 toycc/passes/constfold.py
读法建议:
  1. 先看注释里的 Note on Fusing algorithm(后支配树 + 并查集分组)
  2. 再看 FuseOps() 三步:建图 -> 划分 -> 改写
  3. 回到 toycc 的 FusionPass, 找共同点与差异(没做后支配分析)
装好 tvm 后再回来跑 lesson09 的可运行示例。
""")


def lesson09():
    print("=== 第9课:真实 TVM: 跑一次 Relax pass 管线 ===")
    print("""
装了 tvm 之后,这段代码等价于 toycc 的 demo(先装: pip install apache-tvm):

    import tvm
    from tvm import relax

    # 1. 构造一个极小 IRModule: conv2d -> bias_add -> relu
    # 2. 串行跑 pass:
    mod = tvm.ir.transform.Sequential([
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),
        relax.transform.FoldConstant(),
        relax.transform.LegalizeOps(),
    ])(mod)

    # 观察 mod 的文本表示在每一步之后怎么变化(打印即可)
    print(mod)

对应关系:
  Sequential      <=> toycc/passes/pipeline.py::run_passes
  FuseOps         <=> FusionPass
  FoldConstant    <=> ConstantFoldPass
  AnnotateTIROpPattern -> FuseTIR: 把融合后的 Relax 函数落成 TIR PrimFunc

完整可运行示例和通关清单见 course/lesson09.md。
""")


def lesson11():
    print("=== 第11课:TIR 与调度 ===")
    from toycc.schedule import matmul_scheduled
    nest, t = matmul_scheduled(M=8, N=8, K=8, block=4)
    print("-- 调度前:朴素三重循环 --")
    print(nest.render())
    print("\n-- 调度后:tile + reorder(k 外提) + vectorize(j_i) + parallel --")
    print(t.render())
    print("""
观察四个变换做了什么:
  tile(i,j,4,4)   : 2x2 的 4x4 小块, 缓存友好
  reorder(k,...)  : 把 k 提到最外层, 每个小块复用整列 A
  vectorize(j_i)  : 内层 4 个 j 用一条 SIMD 指令
  parallel(k)     : 各块并行
所有变换都"语义等价"——只是换了循环怎么写。
""")


def lesson12():
    print("=== 第12课:优化全景(DCE + op_pattern) ===")
    from toycc.ir import Graph, OpAttrs
    from toycc.passes import DCEPass, run_passes
    from toycc.examples.model import default_weights

    # 造一个带"死代码"的图
    g = Graph("dce_demo")
    g.add_input("x")
    w = default_weights()
    for name in w:
        g.add_input(name)
        g.set_constant(name, w[name])
    g.add_op("conv", ["x", "conv1_w"], "conv1", OpAttrs(
        kernel=(3, 3), stride=(1, 1), pad=(1, 1, 1, 1)))
    g.add_op("relu", ["conv1"], "dead1")        # ← 死代码:没人消费
    g.add_op("sigmoid", ["conv1"], "dead2")     # ← 死代码
    g.add_op("relu", ["conv1"], "out")
    g.mark_output("out")
    print("-- DCE 前 --")
    print(g.dump())
    d = DCEPass()(g)
    print("\n-- DCE 后 (dead1/dead2 被删除) --")
    print(d.dump())

    # op_pattern 分类示例
    print("\n-- op_pattern 分类(AnnotateTIROpPattern 的事) --")
    for op, pat in sorted(PATTERN_OF.items()):
        print(f"  {op:<16} -> {pat}")


PATTERN_OF = {
    "conv":     "kOpaque",          # 计算密集, 不能简单逐元素
    "matmul":   "kOpaque",
    "relu":     "kBroadcast",       # 逐元素, 可广播
    "sigmoid":  "kBroadcast",
    "add":      "kBroadcast",
    "mul":      "kBroadcast",
    "reshape":  "kInjective",       # 一一对应, 无计算
    "layout_transform": "kInjective",
}


def lesson13():
    print("=== 第13课:meta_schedule 与 autotuning ===")
    import random
    rng = random.Random(42)
    # 搜索空间: 4 个候选调度(分块, 线程)
    space = [(16, 128), (32, 128), (32, 256), (64, 256)]
    # "真机"耗时表: Runner 实测的结果(这里模拟)
    true_time = {(16, 128): 1.8, (32, 128): 1.2, (32, 256): 1.5, (64, 256): 2.4}

    # tune 主循环: 采样 → 测量 → 记 Database → 指导下一轮
    db = {}
    for rnd in range(2):
        for c in rng.sample(space, 2):        # 每轮采样 2 个候选
            if c not in db:
                db[c] = true_time[c]          # 真实世界 = Runner 上机实测
        best = min(db, key=db.get)
        print(f"  轮{rnd+1}: 已测 {sorted(db.keys())} 当前最优 {best} = {db[best]} ms")
    print(f"  Database 共 {len(db)} 条记录; 最终最优配置 = {best} ({db[best]} ms)")
    assert best == (32, 128)

    # 四个组件的分工
    for comp, job in (("SearchSpace", "所有合法调度的集合"),
                      ("Runner", "真机实测耗时"),
                      ("CostModel", "预测未测调度的快慢"),
                      ("Database", "存 调度→耗时, 供复现/导出")):
        print(f"  {comp:<12} → {job}")
    print("  观察: autotune = 采样×测量×学习的循环; 换硬件 = 换 true_time 表, 必须重搜。")


def lesson14():
    print("=== 第14课:IR 家族(LLVM/MLIR/TIR/PTX) ===")
    # 五层 IR: 各表达什么 + 对应对象
    layers = [
        ("Relax", "张量图 + 动态形状", "toycc 的 Graph"),
        ("TIR", "循环 + 计算块 + buffer", "toycc 的 LoopNest/schedule"),
        ("MLIR", "可扩展多层, 方言机制", "Operation/Region/Block"),
        ("LLVM IR", "SSA + 虚拟寄存器, 强类型", "BasicBlock + phi"),
        ("PTX", "GPU 虚拟汇编", "地址空间 + 线程 + fma"),
    ]
    for name, what, obj in layers:
        print(f"  {name:<8} → {what}; 对应对象: {obj}")
    assert len(layers) == 5

    # 下降链: 每层从"算什么"走向"怎么算"
    chain = ["Relax(算什么)", "TIR(怎么循环)", "LLVM IR(标量指令)", "PTX(哪条 GPU 指令)"]
    print("  下降链: " + " → ".join(chain))
    assert len(chain) == 4

    # 两个结构问题的答案
    print("  循环是显式对象的层: TIR(For 节点) / MLIR(scf.for + iter_args)")
    print("  合流的两种写法: LLVM phi(按前驱边选值) vs MLIR block argument(入口取值)")
    print("  观察: 五层 IR 共享 SSA 与「层次越低越具体」的下降逻辑。")


def lesson10():
    print("=== 第10课:从看懂到上手 ===")
    print("""
动手路线(今天就能做):
  A. 给 toycc 加 maxpool 算子: 注册 → 参考实现 → 塞进模型 → 跑 demo 验证
  B. 给 fusion 加一条规则(如 mul 缩放)
  C. 实现死代码消除(DCE): 删掉 consumers 为空的节点
  D. 进阶: 量化 pass

流程(编译器开发的日常):
  改代码 → python -m toycc.examples.demo → 看 max|Δ| → 改 bug → 再验证

装好 tvm 之后:
  1. 写一个 function_pass 遍历 IR 并打印(MyFirstPass)
  2. 跑通 course/lesson09.md 里的 tvm_demo.py
  3. 去 github.com/apache/tvm 找 good-first-issue

完整细节见 course/lesson10.md。
""")


def lesson15():
    print("=== 第15课:硬件必修课(缓存/寄存器/内存层次) ===")
    from toycc.hardware import compare_schedules, register_pressure_demo, LATENCY
    print("-- 内存层次延迟参考(周期) --")
    for k, v in LATENCY.items():
        print(f"  {k:<10} {v}")
    print("\n-- 同一 matmul, 不同循环顺序的缓存表现 --")
    print(compare_schedules(32, 32, 32, block=4))
    print("\n" + register_pressure_demo())


def lesson20():
    print("=== 第20课:知识地图(自测清单) ===")
    print("""
核心三支柱:  IR / Pass / 后端           [第1-7,14课]
正确性:      参考执行器 + max|Δ|         [第2课]
硬件:        缓存/寄存器/SIMD/GPU         [第15课]
调度:        原语 + 为什么影响性能        [第11课]
自动调度:    meta_schedule               [第13课]
读源码:      三遍读法 + 对照表 + FFI      [第8,9课]
参与开发:    git/PR/CI/讨论黑话          [第10,17课]

五个进阶方向: TVM / MLIR / LLVM / 性能工程 / 部署
完整地图和六阶段计划见 course/lesson20.md。
""")


def lesson16():
    print("=== 第16课:真实工程开发流程 ===")
    print("""
核心循环: 改代码 → 跑测试 → 调bug → 提PR
  build :  git clone --recursive apache/tvm → cmake → make -j
  test  :  pytest tests/python/relax/...       (MLIR/LLVM 用 lit)
  debug :  ① PassContext print_all 打IR  ② 小步隔离 ③ gdb bt
  git   :  提交规范 + PR清单 + CI失败处理(重跑/修测试)

动手自测(在 WSL 里):
  1. wsl --version && gcc --version         # 环境OK
  2. python3 -c "import tvm"                # tvm装好
  3. 跑通 toycc demo
  4. 从源码编译 tvm (进阶)
  5. pytest tests/python/relax/ -k "fuse" -x  # 跑一个测试

细节(含"第一个任务六步演练")见 course/lesson16.md
前置: 附录A(C++) 附录B(WSL2环境)
""")


def lesson18():
    print("=== 第18课:量化与数值精度 ===")
    import numpy as np
    from toycc.examples.model import build_model, default_weights
    from toycc.passes.quantize import report
    g = build_model()
    feed = default_weights()
    feed["x"] = np.random.default_rng(1).standard_normal((1, 3, 8, 8))
    for bits, pc in [(8, False), (8, True), (4, True)]:
        print(report(g, feed, "conv1", bits, pc))
        print()
    print("结论: per-channel 优于 per-tensor; int4 误差大 10 倍。")


def lesson17():
    print("=== 第17课:模型怎么进编译器(前端/下降/动态形状/运行时) ===")
    from toycc.ir import Graph
    # 实验1: ONNX 风格导入 —— node 列表 → 计算图
    g = Graph("imported")
    g.add_input("x")
    g.add_op("relu", ["x"], "r1")
    g.add_op("sigmoid", ["r1"], "out")
    g.mark_output("out")
    print("-- ONNX 风格 node 列表(名字, 算子, 输入) → toycc 计算图 --")
    print(g.dump())

    # 实验2: legalize 下降 —— LayerNorm 拆成原始算子链
    steps = ["mean(x)", "sub(x, mean)", "pow(diff, 2)", "mean(pow2)",
             "add(var, eps)", "sqrt", "div(diff, std)", "mul(norm, gamma)",
             "add(scaled, beta)"]
    print(f"-- legalize: LayerNorm(1 个高层算子) → {len(steps)} 个原始算子 --")
    for s in steps:
        print(f"  {s}")
    assert len(steps) == 9

    # 实验3: 动态形状 = 符号变量
    print("-- 符号形状: (1, 'batch', 'seq', 4096) 编译期保留符号, 运行时绑定 --")
    print("  观察: 前端导入=名字表翻译; legalize=高层算子分解; 符号形状=编译期不展开。")


def lesson19():
    print("=== 第19课:性能度量与算子加速 ===")
    # 手算1: matmul 32³ 的算术强度
    m = n = k = 32
    flop = 2 * m * n * k
    byte = 3 * m * n * 4                     # A、B 读 + C 读写(无复用近似)
    ai = flop / byte
    print(f"  matmul 32x32x32: {flop} FLOP / {byte} B = {ai:.2f} FLOP/B (≈5.3)")
    assert abs(ai - 5.33) < 0.05

    # 手算2: A100 roofline 拐点
    knee = 19.5e12 / 1.55e12
    print(f"  A100 拐点 = 19.5 TFLOPS / 1.55 TB/s ≈ {knee:.1f} FLOP/B")

    # 手算3: 两个 kernel 各在哪一侧
    vadd = 1 / 12
    print(f"  vector_add {vadd:.3f} FLOP/B << 拐点 → 带宽受限; matmul 接近拐点侧 → 有算力空间")
    assert vadd * 100 < knee

    # 手算4: im2col 膨胀与 Winograd 乘法账
    print(f"  im2col: 每个输入像素被复制 {3*3} 份(3×3 卷积核) → 内存膨胀换 GEMM 复用")
    wino = 36 / 16
    print(f"  Winograd F(2×2,3×3): 36 次乘法 → 16 次 = {wino:.2f} 倍(加法换乘法)")
    assert abs(wino - 2.25) < 1e-9

    # 手算5: benchmark 为什么用 median
    import statistics
    samples = [100, 101, 99, 102, 98, 1000]  # 混入一次偶发抖动
    print(f"  抖动样本 {samples}: mean={statistics.mean(samples):.0f}"
          f" vs median={statistics.median(samples):.0f} → median 不被拉偏")


def lesson21():
    print("=== 第21课:GPU 芯片架构 ===")
    from toycc.hardware import LATENCY
    # 手算1: 寄存器文件
    regs = 65536 * 4
    print(f"  寄存器文件 = 65536 寄存器 × 4B = {regs//1024} KB")
    assert regs == 256 * 1024
    # 手算2: 满 occupancy 的每线程寄存器
    print(f"  2048 线程满占时每线程 = {65536 // 2048} 个寄存器")

    # 手算3: 一个 GEMM 的 occupancy 账
    for rpt in (64, 128):
        blocks = 65536 // (rpt * 256)
        occ = min(blocks * 8, 64) / 64
        print(f"  256线程/块 × {rpt:3d} 寄存器/线程 → {blocks} blocks → occupancy {occ:.0%}")
    assert 65536 // (128 * 256) == 2

    # 手算4: 合并访问的事务账
    print(f"  warp 连续读 float: 32×4B = 128B = 1 条 cache line(4 个 32B sector)")
    print(f"  warp stride 读:     每线程各落 1 条 line → 32 笔事务, 有效利用 12.5%")

    # 手算5: 分支发散
    print(f"  if/else 对半发散: 2 条指令干 32 线程的活 → 该段利用率 50%")

    print("  内存层次(延迟, 近似):")
    for k, v in LATENCY.items():
        print(f"  {k:<10} {v} 周期")
    print("  观察: 三个核心矛盾 —— 合并访问/占用率/资源限制, 编译器在其中做平衡。")


def lesson22():
    print("=== 第22课:GPU 编译器特有技术 ===")
    # 手算1: 谓词化 —— 发散分支 → 带开关的指令
    print("""
  发散版(两条路径串行, 各 50% 活跃):
    if (tid & 1) x = a + b; else x = a * b;
  谓词化(两条指令各带谓词, 无跳转):
    @p0  FADD x, a, b     ← 谓词为真的 lane 执行
    @!p0 FMUL x, a, b     ← 其余 lane 执行
  """)
    # 手算2: 指令调度 —— 依赖链 vs 独立链
    dep = 3 * 4                              # 3 条串行 FMA × 4 周期延迟
    ind = 4
    print(f"  依赖链: 3 条串行 FMA × 4 周期 = {dep} 周期(全部干等)")
    print(f"  独立链: 4 条互不依赖的 FMA 互相填等待 ≈ {ind} 周期 → 调度重排的价值")
    # 手算3: PTX → SASS 两层
    print("""
  PTX(虚拟 ISA) → ptxas(寄存器分配 + 指令调度) → SASS(真实机器码)
  虚拟寄存器 %r0..%rN → 物理寄存器 R0..R255; 两条链的指令数通常不等
  """)
    # 手算4: occupancy 直读
    for warps in (16, 32, 64):
        print(f"  {warps:2d} warps/SM → occupancy {warps/64:.0%} (上限 64)")
    print("  观察: 编译器的四个独有矛盾 —— SIMT 锁步 / 编译期资源分配 / 显式分层 / 顺序调度。")


def lesson23():
    print("=== 第23课:Kernel 开发与性能分析 ===")
    # 手算1: roofline 判断值不值得优化
    vadd, knee = 1 / 12, 19.5e12 / 1.55e12
    mm = 2 * 32**3 / (3 * 32 * 32 * 4)
    print(f"  vector_add {vadd:.3f} FLOP/B vs 拐点 {knee:.1f} → 带宽顶死, 优化搬运")
    print(f"  matmul 32x32x32 {mm:.2f} FLOP/B → 有优化空间, 优化复用与算力")

    # 手算2: 健康度 = 实测带宽 / 理论带宽
    measured, theory = 0.6e12, 1.55e12
    print(f"  实测 0.6 TB/s / 理论 1.55 TB/s = 健康度 {measured/theory:.0%}"
          f" → 事务数手算是排查第一页")
    assert abs(measured / theory - 0.39) < 0.05

    # 手算3: 日常工作循环
    print("  循环: 写 kernel → benchmark(warmup + median) → profiler 找瓶颈"
          " → 一次只改一个变量 → 再测")

    # 手算4: 最优 kernel 固化
    print("  固化: 最优配置 → TIR 模板 / autotune 搜索空间 → 换硬件时重搜")


def lesson24():
    print("=== 第24课:自研 GPU 工具链全景 ===")
    # 实验: 给 TVM 加新后端的五步 —— 每步的可检查产物
    steps = [
        ("1. target 描述", "芯片硬件特性表(target 字符串 + 参数)"),
        ("2. codegen", "TIR → 目标指令(relax.ext.<target>)"),
        ("3. runtime 驱动", "VM 在芯片上跑通最小算子(端到端)"),
        ("4. meta_schedule", "搜索空间 + Runner 指向模拟器"),
        ("5. 测试 + CI", "差分测试 + 覆盖率门禁"),
    ]
    for name, artifact in steps:
        print(f"  {name:<18} → 产物: {artifact}")
    assert len(steps) == 5

    # 工具链七件套互相喂
    chain = ["编译器", "汇编器", "链接器", "驱动", "运行时", "profiler", "调试器"]
    print(f"  七件套: {' → '.join(chain)}")
    assert len(chain) == 7
    print("  观察: 每步都有独立可检查产物 —— 后四步由第 27~30 课逐个展开。")


def lesson25():
    print("=== 第25课:LLVM 深入 ===")
    # 手算1: phi 语义
    print("""
  phi 语义: 沿哪条前驱边进块, 就选哪条边携带的值
    %v = phi i32 [ %x, %then ], [ %y, %else ]
    cond=true → %v = %x;  cond=false → %v = %y
  """)
    # 手算2: GEP 是纯地址计算
    base, idx = 0x1000, 3
    addr = base + idx * 4
    print(f"  getelementptr i32, ptr base, i64 {idx} → 0x{base:x} + {idx}×4 = 0x{addr:x}"
          f" (只算地址, 不读内存)")
    assert addr == 0x100C

    # 手算3: 分析失效契约
    print("""
  pass 改了 CFG 却谎报 preserve → 后续 pass 用过期的支配树 → 错误优化
  契约: 改了什么就作废什么; 只读 pass 才能返回 PreservedAnalyses::all()
  """)
    # 手算4: 后端流水线
    print("  指令选择(TableGen 匹配) → 指令调度(填延迟) → 寄存器分配(图着色) → MC 编码")
    print("  观察: 四个概念各有一个 C++ 对象 —— BasicBlock / DominatorTree / PassManager / MC。")


def lesson26():
    print("=== 第26课:MLIR 深入 ===")
    # 手算1: 下降链每层一句话
    chain = [("tosa.conv", "算子语义"), ("linalg", "结构化循环"),
             ("affine/scf", "显式循环/控制流"), ("memref", "内存"),
             ("llvm", "接近 LLVM IR")]
    for d, what in chain:
        print(f"  {d:<12} → {what}")
    assert len(chain) == 5

    # 手算2: ODS 一行 → 生成四样
    print("""
  ODS 一行 def → 生成: 类声明 / 访问器 / parser+printer(assemblyFormat) / verifier 框架
  """)
    # 手算3: pattern rewrite 两步
    print("""
  pattern: addi(x, 0) → x
  match: 右操作数是常量 0?  →  failure(不接手) / rewrite(replaceOp 重连 uses)
  """)
    # 手算4: bufferization in-place 决策
    print("""
  y = add(x, c): x 无其他使用者 → 复用 x 的 buffer(in-place)
                 x 有其他使用者 → 分配新 buffer(out-of-place, 别名不允许覆盖)
  """)
    print("  观察: 方言/region/ODS/rewrite/bufferize —— 五个词各对应一个 IR 对象。")


def lesson30():
    print("=== 第30课:驱动与命令提交 ===")
    # 手算1: 每 launch 的固定开销
    t_user, t_syscall, t_doorbell = 1.0, 3.5, 0.5   # μs(近似值, 合计 3~5μs 区间内)
    launch = t_user + t_syscall + t_doorbell
    print(f"  launch 开销 = {t_user} + {t_syscall} + {t_doorbell} = {launch:.1f} μs")
    assert abs(launch - 5.0) < 1e-9

    # 手算2: 小 kernel 的吞吐利用率
    kernel = 2.0                                   # μs
    eff = kernel / (kernel + launch)
    print(f"  2μs kernel 吞吐利用率 = {kernel}/({kernel}+{launch}) = {eff:.0%}")
    print(f"  (70% 时间在发射, 不在计算)")
    assert abs(eff - 2 / 7.0) < 1e-9

    # 盈亏平衡点: kernel 多短时 launch 开销占一半?
    print(f"  开销占 50% 的 kernel 长度 = {launch:.1f} μs (t/(t+{launch})=0.5 → t={launch})")

    # 手算3: 1000 次小 launch vs 合并成一次
    many = 1000 * (kernel + launch) / 1000         # ms
    one = (1000 * kernel + launch) / 1000          # ms
    print(f"  1000 次 2μs launch 总耗时 {many:.1f} ms; 合并成一次 {one:.1f} ms")
    assert many > one * 3

    # 手算4: GPU MMU 页表
    page, pte = 4096, 8
    pages = 8 * 1024**3 // page                    # 8GB 显存, 4KB 页
    print(f"  8GB 显存 / 4KB 页 = {pages} 页 × {pte}B/PTE = {pages*pte/1e6:.0f} MB 页表")
    print("  观察: 命令提交的固定账、合并的收益、MMU 的代价, 三者都可复算。")


def lesson29():
    print("=== 第29课:二进制格式与模块加载 ===")
    # 手算1: 重定位 —— 链接器如何修正一个悬空地址
    text_base, text_size = 0x100, 0x40             # .text 在对象文件内
    data_base, data_size = 0x140, 0x20             # .data
    reloc = (0x108, "x", 0)                        # 在 .text 内引用符号 x, addend=0
    sym_off = 0x8                                   # x 在 .data 内的偏移
    link_text, link_data = 0x4000, 0x4000 + text_size
    s_addr = link_data + sym_off                    # 符号最终地址
    patch_at = link_text + (reloc[0] - text_base)   # 重定位写入点
    patch_val = s_addr + reloc[2]                   # S + A
    print(f"  重定位: 在 0x{patch_at:x} 写入符号 x 的地址 0x{patch_val:x} (=S+A)")
    assert patch_at == 0x4008 and patch_val == 0x4048

    # 手算2: cubin 元数据校验 launch 配置
    meta = {"maxThreads": 1024, "sharedBytes": 48 * 1024}

    def check(threads, shared):
        return threads <= meta["maxThreads"] and shared <= meta["sharedBytes"]

    ok1, ok2 = check(256, 48 * 1024), check(256, 64 * 1024)
    print(f"  launch 256线程/48KB shared: {'通过' if ok1 else '拒绝'} (元数据校验)")
    print(f"  launch 256线程/64KB shared: {'通过' if ok2 else '拒绝'} (超 static 上限)")
    assert ok1 and not ok2
    print("  观察: 加载器在 launch 前拿 cubin 元数据逐项校验, 非法配置当场拦下。")


def lesson28():
    print("=== 第28课:GPU内存模型与并发原语 ===")
    # 实验1: 丢更新 —— "读-改-写"三步被插入
    counter = 0
    read_a, read_b = counter, counter              # 两个线程都读到旧值 0
    counter = read_a + 1                           # 线程A写回 1
    counter = read_b + 1                           # 线程B写回 1 ← 覆盖, A 的更新丢失
    print(f"  无原子: 两线程各加1, 最终 = {counter} (丢 1 次更新)")
    assert counter == 1

    # 实验2: 32 线程各加 1 的最坏情况(全部读到 0 再写回)
    n = 32
    reads = [0] * n
    final = max(r + 1 for r in reads)
    lost = n - 1
    print(f"  无原子: {n} 线程各加1, 最坏最终 = {final}, 丢 {lost} 次更新")
    assert lost == n - 1

    # 实验3: 原子指令 —— 一步不可分割
    counter = 0
    for _ in range(n):
        counter += 1                               # 等价于 atom.add 的语义
    print(f"  原子:   {n} 线程各加1, 最终 = {counter} (不可分割, 不丢)")
    assert counter == n
    print("  观察: 原子保证不丢, 但冲突时在硬件上串行化 —— 正确性与吞吐的交换。")


def lesson27():
    print("=== 第27课:模拟器 ===")
    print("""
 模拟器 = ISA 规范的可执行版本, 工具链第一件交付物

 三层模型:
   功能模型(ISS)  指令精确   ~1e8 条/秒   回答"对不对"
   周期模型       周期近似   ~1e6 条/秒   回答"快不快"
   RTL 仿真       信号精确   ~1e4 条/秒   回答"电路对不对"

 验证: 差分测试(独立参考实现 + 覆盖率门禁)
 调试: 编译器输出 → 加载器 → 模拟器 ← 调试器挂断点
 详细见 course/lesson27.md
""")


def lesson35():
    print("=== 第35课:前沿专题速览 ===")
    # 手算1: 投机解码的期望产出
    for alpha in (0.5, 0.7, 0.8):
        total = 1 / (1 - alpha)
        print(f"  投机解码 α={alpha}: 期望产出 {total:.2f} 倍(净增 {total-1:.2f} 倍)")
    assert abs(1 / (1 - 0.7) - 3.33) < 0.01

    # 手算2: 2:4 结构化稀疏
    print(f"  2:4 稀疏: 权重 140GB → {140/2:.0f} GB(字节账); 每 4 权重只算 2 个(算力账)")

    # 手算3: MoE 容量/计算解耦
    print(f"  MoE: 8×7B=56B 参数, 每 token 激活 2 专家 = {2*7:.0f}B 计算 → 解耦")

    # 手算4: 框架接入三路径成本
    for name, cost in (("B 算子库", "2 周"), ("C ONNX", "折中"), ("A Inductor 后端", "1~2 人年")):
        print(f"  接入路径 {name}: {cost}")
    print("  观察: 每条前沿名词都能落回 算力/带宽/显存/接入成本 四本账。")


def lesson31():
    print("=== 第31课:LLM推理性能工程 ===")
    # 手算1: decode 上限
    w_gb, bw = 14.0, 2.0                           # 7B fp16 权重(GB), 2TB/s = 2GB/ms
    ms = w_gb / bw                                # GB / (GB/ms) = ms
    tok = 1000 / ms
    print(f"  decode: {w_gb}GB / {bw}TB/s = {ms:.1f} ms/token → 上限 ~{tok:.0f} tok/s")
    assert abs(ms - 7.0) < 1e-9

    # 手算2: prefill 512 token
    flop = 7e9 * 2 * 512
    tf = 312.0
    ms_c = flop / 1e12 / tf * 1000
    print(f"  prefill: {flop/1e12:.1f} TFLOP / {tf} TFLOPS = {ms_c:.0f} ms(算力)"
          f" vs {ms:.0f} ms(带宽) → 算力受限")
    assert ms_c > ms

    # 手算3: KV cache
    kv = 2 * 32 * 4096 * 4096 * 1 * 2
    print(f"  KV: 7B batch=1 序列4096 = {kv/1e9:.1f} GB; batch=32 = {kv*32/1e9:.0f} GB"
          f" (权重的 {kv*32/1e9/14:.0f} 倍)")
    assert abs(kv / 1e9 - 2.15) < 0.05

    # 手算4: FlashAttention 消灭的中间结果
    mid = 8192 * 8192 * 2
    print(f"  naive attention 8K 序列中间矩阵 = {mid/1e6:.0f} MB (FlashAttention 消灭它)")
    print("  观察: 三张账(算力/带宽/显存)全部可复算, 这就是会议室语言的底稿。")


def lesson32():
    print("=== 第32课:分布式并行与通信 ===")
    # 手算1: ring allreduce 每卡发送量
    n, s, bw = 8, 140.0, 900.0                    # 8卡, 140GB 梯度, 900GB/s
    per_rank = 2 * (n - 1) / n * s
    ms = per_rank / bw * 1000
    print(f"  ring allreduce: 每卡 2×7/8×{s:.0f} = {per_rank:.0f} GB"
          f" / {bw}GB/s = {ms:.0f} ms")
    assert abs(per_rank - 245.0) < 1e-9

    # 手算2: 流水线气泡
    p = 4
    for m in (1, 8, 32):
        bubble = (p - 1) / (m + p - 1)
        print(f"  PP: {p}卡 micro-batch={m:2d} → 气泡 {bubble:.0%}")
    assert abs((4 - 1) / (32 + 4 - 1) - 0.086) < 0.01

    # 手算3: 加速比上限与重叠收益
    calc, comm = 800.0, 272.0
    cap = calc / (calc + comm)
    print(f"  加速比上限 = {calc:.0f}/({calc:.0f}+{comm:.0f}) = {cap:.0%};"
          f" 完美重叠 = max({calc:.0f},{comm:.0f}) = {max(calc, comm):.0f} ms")
    print("  观察: 通信账决定切法与摆法 —— TP 节点内, PP/DP 跨节点。")


def lesson33():
    print("=== 第33课:生产级量化 ===")
    # 手算1: W4A16 decode 收益
    fp16_ms = 140 / 2.0                            # GB / (GB/ms) = ms
    w4_ms = 35 / 2.0
    print(f"  decode: fp16 {fp16_ms:.0f} ms/token(~{1000/fp16_ms:.0f} tok/s)"
          f" → W4A16 {w4_ms:.1f} ms(~{1000/w4_ms:.0f} tok/s), {fp16_ms/w4_ms:.1f} 倍")
    assert abs(fp16_ms / w4_ms - 4.0) < 1e-9

    # 手算2: per-group scale 的额外字节
    extra = 7e9 / 128 * 2
    print(f"  per-group(128) scale 额外 = {extra/1e9:.2f} GB = 35GB 的 {extra/1e9/35:.2%}")

    # 手算3: dequant 落显存的反面教材
    fused, unfused = 35.0, 35.0 + 140.0 + 140.0
    print(f"  流量: 融合 {fused:.0f} GB vs 不融合 {unfused:.0f} GB ({unfused/fused:.0f} 倍)")
    assert abs(unfused / fused - 9.0) < 1e-9
    print("  观察: 量化收益 = 位宽压缩比; dequant 必须融进 GEMM, 否则倒亏。")


def lesson34():
    print("=== 第34课:Triton 与 CUTLASS ===")
    # 手算: num_stages 的 shared 内存账(Ampere 每 SM 164KB)
    bk = 64
    tile = 128 * bk * 2 * 2                        # A tile + B tile, fp16
    limit = 164 * 1024
    for stages in (2, 4, 5, 6):
        used = stages * tile
        ok = "可" if used <= limit else "不可"
        print(f"  BLOCK=128×128, BK={bk}: stages={stages} → shared {used//1024:3d} KB"
              f" [{ok} (上限 {limit//1024} KB)]")
    assert 6 * tile > limit >= 4 * tile

    print("""
  分工: Triton 快迭代 | CUTLASS 追极致 | 编译器+autotune 兜底长尾/自研
  观察: stages 触顶 shared → occupancy 掉(第3课的账在 Triton 里变成 Config)。""")


LESSONS = {
    1: lesson01, 2: lesson02, 3: lesson03, 4: lesson04,
    5: lesson05, 6: lesson06, 7: lesson07, 8: lesson08, 9: lesson09,
    10: lesson10, 11: lesson11, 12: lesson12, 13: lesson13,
    14: lesson14, 15: lesson15, 16: lesson16, 17: lesson17,
    18: lesson18, 19: lesson19, 20: lesson20, 21: lesson21,
    22: lesson22, 23: lesson23, 24: lesson24, 25: lesson25,
    26: lesson26, 27: lesson27, 28: lesson28, 29: lesson29,
    30: lesson30, 31: lesson31, 32: lesson32, 33: lesson33,
    34: lesson34, 35: lesson35,
}


def main():
    if len(sys.argv) < 2:
        print("课程实验列表:")
        for k, fn in sorted(LESSONS.items()):
            print(f"  {k}: {fn.__doc__}")
        print("\n用法: python -m course.runner <编号>")
        return
    arg = sys.argv[1]
    if arg == "all":
        for k in sorted(LESSONS):
            fn = LESSONS[k]
            print("\n" + "=" * 60)
            fn()
        return
    try:
        k = int(arg)
    except ValueError:
        print(f"未知课程 {arg!r}")
        return
    LESSONS[k]()


if __name__ == "__main__":
    main()
