"""demo 入口:完整 pass 管线,验证每一步优化前后结果一致。

用法:
    python -m toycc.examples.demo
"""
from __future__ import annotations

import numpy as np

from toycc.examples.model import build_model, default_weights
from toycc.passes import run_passes
from toycc.runtime import evaluate


def make_feed():
    feed = default_weights()
    feed["x"] = np.random.default_rng(1).standard_normal((1, 3, 8, 8))
    return feed


def compare(label, a, b):
    if len(a) != len(b):
        print(f"  {label}: 输出数量不一致!")
        return
    ok = True
    for i in range(len(a)):
        v0, v1 = list(a.values())[i], list(b.values())[i]
        diff = np.max(np.abs(v0 - v1))
        if diff >= 1e-9:
            ok = False
        print(f"  {label} out[{i}]: max|Δ| = {diff:.2e}  "
              f"[{'OK' if diff < 1e-9 else 'FAIL'}]")
    return ok


def main():
    g0 = build_model()
    feed = make_feed()
    out0 = evaluate(g0, feed)

    print("=== 初始图 ({} 个算子) ===".format(len(g0.nodes)))
    print(g0.dump())

    # ---- pass 1:算子融合 ----
    g1 = run_passes(g0, ("fusion",), verbose=True)
    print("\n=== 融合后 ({} 个算子) ===".format(len(g1.nodes)))
    print(g1.dump())
    out1 = evaluate(g1, feed)
    print("\n=== 融合正确性 ===")
    compare("fusion", out0, out1)

    # ---- pass 2:布局规划 ----
    g2 = run_passes(g1, ("layout",), verbose=True)
    print("\n=== 布局规划后 ({} 个算子) ===".format(len(g2.nodes)))
    print(g2.dump())
    out2 = evaluate(g2, feed)
    print("\n=== 布局正确性 ===")
    compare("layout", out0, out2)

    # ---- pass 3:常量折叠(把权重/偏置的布局转换在编译期算掉) ----
    g3 = run_passes(g2, ("constfold",), verbose=True)
    print("\n=== 常量折叠后 ({} 个算子) ===".format(len(g3.nodes)))
    print(g3.dump())
    out3 = evaluate(g3, feed)
    print("\n=== 常量折叠正确性 ===")
    compare("constfold", out0, out3)

    # ---- pass 4:内存规划(只读分析,不改图) ----
    from toycc.passes.memory import MemoryPlanningPass, report
    input_shapes = {"x": feed["x"].shape}
    _, allocs = MemoryPlanningPass(input_shapes)(g3)
    print("\n=== 内存规划 ===")
    print(report(g3, allocs, input_shapes))

    # ---- 代码生成:产出 C 与 python 两个后端 ----
    from toycc.passes.memory import infer_shapes
    from toycc.codegen import emit_c, emit_python
    shapes = infer_shapes(g3, input_shapes)
    c_code = emit_c(g3, {a.tensor: a for a in allocs}, shapes)
    py_code = emit_python(g3, {a.tensor: a for a in allocs}, shapes)

    with open("out.c", "w", encoding="utf-8") as f:
        f.write(c_code)
    with open("out.py", "w", encoding="utf-8") as f:
        f.write(py_code)

    # ---- 验证生成代码:python 后端跑一遍,和参考结果对比 ----
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("gen_out", "out.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    out_gen = {"output": gen.run(feed["x"]).reshape(out0["output"].shape)}
    print("\n=== 生成代码验证(python 后端) ===")
    compare("codegen", out0, out_gen)

    print(f"\n=== 总结 ===")
    print(f"  初始图: {len(g0.nodes)} 个算子")
    print(f"  最终图: {len(g3.nodes)} 个算子 (含 2 次必要布局搬移)")
    print(f"  生成的 C 源码: out.c (可在装有 gcc 的机器上编译)")
    print(f"  生成代码与参考执行完全一致 [OK]" if all(
        np.max(np.abs(out0[list(out0)[i]] - out_gen["output"])) < 1e-9 for i in range(len(out0)))
          else "  生成代码验证失败!")


if __name__ == "__main__":
    main()
