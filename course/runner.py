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
    print("""
本机没装 tvm,这节课是概念 + 流程 + 可运行骨架(装好 tvm 后执行)。

核心循环(autotune 的本质):
  for 轮次:
    1. 从搜索空间采样一组调度(分块/向量化/线程参数)
    2. 编译成可执行代码
    3. 在目标硬件上实测耗时(测量)
    4. 用结果训练成本模型(预测"没测过的调度"快不快)
    5. 下一轮采样时避开慢的, 聚焦快的

四个组件:
  SearchSpace    所有合法调度的集合(参数组合)
  CostModel      预测某个调度有多快(用已测结果学习)
  Runner         在真机上测量耗时的执行器
  Database       存"调度 → 耗时"记录, 供复现/导出

为什么需要 autotune:
  调度参数空间巨大(分块大小/向量宽度/线程数...),
  手写既难又容易过时(换硬件就失效)。

装好 tvm 后跑(骨架, 参考官方 meta_schedule 教程):
    from tvm import meta_schedule as ms
    # 1. 建工作负载(一个 matmul 的 te 或 TIR 函数)
    # 2. 定义 search space: ms.SearchGenerator(...)
    # 3. 定义 runner: ms.runner.LocalRunner(...)
    # 4. tune:  generate -> measure -> evolve 多轮
    # 5. 把最好调度 export 成 schedule, 固化到代码里
    # 完整可运行例子见 course/lesson13.md
""")


def lesson14():
    print("=== 第14课:IR 家族(LLVM/MLIR/TIR/PTX) ===")
    print("""
五种 IR 的层级关系:
  模型(ONNX/PyTorch) → Relax(图) → TIR(循环) → LLVM IR(标量) → PTX(GPU汇编)
每层回答: 算什么(高层) → 怎么算(低层)

速记:
  Relax  : 张量图, 动态形状         =  toycc 的 Graph
  TIR    : 循环 + 计算块 + buffer   =  toycc 的 LoopNest
  MLIR   : 可扩展多层, 方言机制
  LLVM IR: SSA + 虚拟寄存器, 强类型
  PTX    : GPU 汇编, 地址空间 + 线程 + fma

完整示例(官方权威)见 course/lesson14.md。
""")


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


def lesson16():
    print("=== 第16课:知识地图(自测清单) ===")
    print("""
核心三支柱:  IR / Pass / 后端           [第1-7,14课]
正确性:      参考执行器 + max|Δ|         [第2课]
硬件:        缓存/寄存器/SIMD/GPU         [第15课]
调度:        原语 + 为什么影响性能        [第11课]
自动调度:    meta_schedule               [第13课]
读源码:      三遍读法 + 对照表 + FFI      [第8,9课]
参与开发:    git/PR/CI/讨论黑话          [第10,17课]

五个进阶方向: TVM / MLIR / LLVM / 性能工程 / 部署
完整地图和六阶段计划见 course/lesson16.md。
""")


def lesson17():
    print("=== 第17课:真实工程开发流程 ===")
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

细节(含"第一个任务六步演练")见 course/lesson17.md
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


def lesson19():
    print("=== 第19课:模型怎么进编译器(前端/下降/动态形状/运行时) ===")
    print("""
链条: 模型 → 前端导入 → 图优化 → legalize(下降) → TIR → codegen → 运行时
关键点:
  ONNX   : input / initializer(权重) / node / output / opset
  legalize: LayerNorm → mean/var/逐元素 的组合
  动态形状: T.Var 符号形状 → 形状特化 / 动态分派
  控制流 : if/while → 图从 DAG 变成 CFG (LLM 推理= while 循环)
  运行时 : VirtualMachine 解释指令序列, 管内存/设备/参数绑定
完整内容见 course/lesson19.md。
""")


def lesson20():
    print("=== 第20课:性能度量与算子加速 ===")
    print("""
Roofline: 性能上限 = min(算力墙, 带宽墙 × 计算强度)
  计算强度 = 总FLOP / 总搬运字节
  低强度→带宽受限(优化搬运); 高强度→算力受限(优化计算)

Benchmark 方法论: warmup → 测N次取中位数 → 注明测的是什么

算子加速:
  im2col   : conv → GEMM(重复拷贝换复用优化)
  Winograd : 小卷积(3x3) 加法换乘法, 省 50%+ 乘法
  GEMM微内核: 分块 + 寄存器累加(6x8) + 向量化

手算例子: 32x32 matmul 强度≈5.3 FLOP/Byte → 通常带宽受限
完整内容见 course/lesson20.md。
""")


LESSONS = {
    1: lesson01, 2: lesson02, 3: lesson03, 4: lesson04,
    5: lesson05, 6: lesson06, 7: lesson07, 8: lesson08, 9: lesson09,
    10: lesson10, 11: lesson11, 12: lesson12, 13: lesson13,
    14: lesson14, 15: lesson15, 16: lesson16, 17: lesson17,
    18: lesson18, 19: lesson19, 20: lesson20,
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
