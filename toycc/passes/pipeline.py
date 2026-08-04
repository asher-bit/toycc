"""Pass 管线(简易 PassManager)。

真实对照:TVM 的 relay.transform.Sequential / PassContext ——
pass 之间按依赖排序、可重复执行、可在 pass 间打印中间结果做调试。
"""
from __future__ import annotations

from toycc.ir import Graph

_PASSES = {}


def register_pass(name):
    def deco(cls):
        _PASSES[name] = cls
        return cls
    return deco


def run_passes(graph: Graph, passes: tuple[str, ...], verbose=True) -> Graph:
    g = graph.clone()
    for name in passes:
        if name not in _PASSES:
            raise ValueError(f"未知 pass {name!r},可用的有 {sorted(_PASSES)}")
        g = _PASSES[name]()(g)
        if verbose:
            print(f"[pass] {name}: {len(g.nodes)} 个节点")
    return g


# 注册内置 pass(放底部避免循环 import)
from toycc.passes.fusion import FusionPass as _F  # noqa: E402
from toycc.passes.layout import LayoutPass as _L  # noqa: E402
from toycc.passes.constfold import ConstantFoldPass as _C  # noqa: E402
from toycc.passes.dce import DCEPass as _D  # noqa: E402

register_pass("fusion")(_F)
register_pass("layout")(_L)
register_pass("constfold")(_C)
register_pass("dce")(_D)
