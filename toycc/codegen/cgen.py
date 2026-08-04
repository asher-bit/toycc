"""代码生成:把优化后的图 + 内存分配方案,生成后端代码。

真实对照:
- 这一步对应 TVM 的 CodeGen(TIR -> CUDA/x86 源码)、MLIR 的 LLVM dialect 后端。
- 每个融合后的"根算子"就是一个 kernel(函数)。
- 常量被"烙进"静态数组(编译期已折叠),运行时只有 x 是输入。
- 我们同时产出两种后端:
    C        -> 可在真实机器上 gcc 编译
    python   -> 同构镜像,用来在本机(没有 gcc 时)验证生成代码正确性
  这正是"一个 IR,多个后端"的思路。
"""
from __future__ import annotations

import numpy as np

from toycc.ir import Graph


def flat_idx(shape: tuple, idxs: list) -> str:
    """把 N 维坐标展开成一维偏移(行主序)。
    第 k 维的步长 = 它右边所有维度的乘积。"""
    expr = str(idxs[-1])
    stride = 1
    for k in range(len(shape) - 2, -1, -1):
        stride *= shape[k + 1]
        expr = f"({idxs[k]} * {stride} + {expr})"
    return expr


class CodeBuilder:
    """同时生成 C 与 Python 的小工具:管理缩进、for 循环、赋值、表达式。"""
    def __init__(self, lang: str):
        self.lang = lang          # "c" / "py"
        self.lines: list[str] = []
        self._ind = ""

    def comment(self, text):
        if self.lang == "c":
            self.lines.append(f"{self._ind}// {text}")
        else:
            self.lines.append(f"{self._ind}# {text}")

    def push_loop(self, var: str, rng: int):
        if self.lang == "c":
            self.lines.append(f"{self._ind}for ({var} = 0; {var} < {rng}; {var}++) {{")
        else:
            self.lines.append(f"{self._ind}for {var} in range({rng}):")
        self._ind += "    "

    def pop(self):
        self._ind = self._ind[:-4]
        if self.lang == "c":
            self.lines.append(f"{self._ind}}}")

    def assign(self, lhs, rhs):
        if self.lang == "c":
            self.lines.append(f"{self._ind}{lhs} = {rhs};")
        else:
            self.lines.append(f"{self._ind}{lhs} = {rhs}")

    def stmt(self, code):
        if self.lang == "c":
            self.lines.append(f"{self._ind}{code};")
        else:
            self.lines.append(f"{self._ind}{code}")

    def raw(self, code):
        self.lines.append(code)


def _act(lang, act, acc):
    if lang == "c":
        if act == "relu":
            return f"{acc} = fmaxf({acc}, 0.0f)"
        if act == "sigmoid":
            return f"{acc} = 1.0f / (1.0f + expf(-{acc}))"
    else:
        if act == "relu":
            return f"{acc} = max({acc}, 0.0)"
        if act == "sigmoid":
            return f"{acc} = 1.0 / (1.0 + math.exp(-{acc}))"
    return None


class TensorInfo:
    """给图里每个张量安排运行时名字:buf* / in_* / K_*。"""
    def __init__(self, graph: Graph, allocs: dict[str, object], shapes: dict):
        self.graph = graph
        self.shapes = shapes
        self.info: dict[str, tuple[str, str]] = {}
        # 常量优先:折叠出的"常量张量"同时也是图输入,但它是编译期已知的
        for name in graph.constants:
            self.info[name] = ("const", f"K_{name}")
        for name in graph.inputs:
            self.info.setdefault(name, ("input", f"in_{name}"))
        for name, alloc in allocs.items():
            self.info.setdefault(name, ("buf", f"buf{alloc.buf}"))

    def name(self, tensor: str) -> str:
        return self.info.get(tensor, ("buf", f"buf_{tensor}"))[1]

    def ref(self, tensor: str, idxs: list) -> str:
        """生成读张量的表达式:C/py 通用(方括号索引)。"""
        return f"{self.name(tensor)}[{flat_idx(self.shapes[tensor], idxs)}]"


def _emit_node(b: CodeBuilder, node, shapes, ti: TensorInfo):
    """把单个节点降级成循环代码(两个后端共用一套遍历逻辑)。"""
    a = node.attrs
    out = ti.name(node.name)
    out_shape = shapes[node.name]

    if node.op_type == "layout_transform":
        # 循环范围以"输入"的形状为准。索引变量直接用字面量,避免解包错位。
        ishape = shapes[node.inputs[0]]
        if a.from_layout == "nchw":
            N, C, H, W = ishape
            order = [("n", N), ("h", H), ("w", W), ("c", C)]
            oi, ii = ["n", "h", "w", "c"], ["n", "c", "h", "w"]
        else:
            N, H, W, C = ishape
            order = [("n", N), ("c", C), ("h", H), ("w", W)]
            oi, ii = ["n", "c", "h", "w"], ["n", "h", "w", "c"]
        for v, rng in order:
            b.push_loop(v, rng)
        b.assign(f"{out}[{flat_idx(out_shape, oi)}]", ti.ref(node.inputs[0], ii))
        for _ in order:
            b.pop()
        return

    if node.op_type == "reshape":
        n = int(np.prod(shapes[node.inputs[0]]))
        b.push_loop("i", n)
        b.assign(f"{out}[i]", f"{ti.name(node.inputs[0])}[i]")
        b.pop()
        return

    if node.op_type == "matmul":
        (N, K), (_, M) = shapes[node.inputs[0]], shapes[node.inputs[1]]
        bias = node.inputs[2] if len(node.inputs) > 2 else None
        b.comment("matmul" if a.bias else "matmul")
        b.push_loop("i", N)
        b.push_loop("j", M)
        b.stmt("acc = 0.0" if b.lang == "py" else "float acc = 0.0f")
        b.push_loop("k", K)
        b.stmt(f"acc += {ti.ref(node.inputs[0], ['i', 'k'])} * {ti.ref(node.inputs[1], ['k', 'j'])}")
        b.pop()
        if bias is not None:
            bshape = shapes[bias]
            if len(bshape) == 1:
                b.stmt(f"acc += {ti.ref(bias, ['j'])}")
            elif len(bshape) == 2:
                b.stmt(f"acc += {ti.ref(bias, ['0', 'j'])}")
            else:
                b.stmt(f"acc += {ti.ref(bias, ['0', 'j', '0', '0'])}")
        act = _act(b.lang, a.activation, "acc")
        if act:
            b.stmt(act)
        b.assign(f"{out}[{flat_idx(out_shape, ['i', 'j'])}]", "acc")
        b.pop()
        b.pop()
        return

    if node.op_type == "conv":
        nhwc = a.layout == "nhwc"
        x, w = node.inputs[0], node.inputs[1]
        bias = node.inputs[2] if len(node.inputs) > 2 else None
        xsh, wsh = shapes[x], shapes[w]
        if nhwc:
            N, H, W, C = xsh
            OC, KH, KW, _ = wsh
        else:
            N, C, H, W = xsh
            OC, _, KH, KW = wsh
        SH, SW = a.stride
        PH, PB, PL, PR = a.pad
        OH = (H + PH + PB - KH) // SH + 1
        OW = (W + PL + PR - KW) // SW + 1

        b.comment("conv" + (" (nhwc)" if nhwc else ""))
        for v, rng in [("n", N), ("oh", OH), ("ow", OW), ("oc", OC)]:
            b.push_loop(v, rng)
        b.stmt("acc = 0.0" if b.lang == "py" else "float acc = 0.0f")
        korder = [("kh", KH), ("kw", KW), ("c", C)] if nhwc else \
                 [("c", C), ("kh", KH), ("kw", KW)]
        for v, rng in korder:
            b.push_loop(v, rng)
        ih = f"(oh * {SH} - {PH} + kh)"
        iw = f"(ow * {SW} - {PL} + kw)"
        if nhwc:
            xi = ti.ref(x, ["n", ih, iw, "c"])
            wi = ti.ref(w, ["oc", "kh", "kw", "c"])
        else:
            xi = ti.ref(x, ["n", "c", ih, iw])
            wi = ti.ref(w, ["oc", "c", "kh", "kw"])
        if b.lang == "c":
            b.raw(f"{b._ind}if ({ih} >= 0 && {ih} < {H} && {iw} >= 0 && {iw} < {W}) "
                  f"{{ acc += {xi} * {wi}; }}")
        else:
            b.raw(f"{b._ind}if 0 <= {ih} < {H} and 0 <= {iw} < {W}: acc += {xi} * {wi}")
        for _ in korder:
            b.pop()
        if bias is not None:
            if nhwc:
                b.stmt(f"acc += {ti.ref(bias, ['0', '0', '0', 'oc'])}")
            else:
                b.stmt(f"acc += {ti.ref(bias, ['0', 'oc', '0', '0'])}")
        act = _act(b.lang, a.activation, "acc")
        if act:
            b.stmt(act)
        oi = ["n", "oh", "ow", "oc"] if nhwc else ["n", "oc", "oh", "ow"]
        b.assign(f"{out}[{flat_idx(out_shape, oi)}]", "acc")
        for _ in [("n", N), ("oh", OH), ("ow", OW), ("oc", OC)]:
            b.pop()
        return

    raise NotImplementedError(f"codegen 不支持算子 {node.op_type}")


def emit_c(graph: Graph, allocs, shapes) -> str:
    ti = TensorInfo(graph, allocs, shapes)
    nbufs = max((a.buf for a in allocs.values()), default=-1) + 1

    L = ["// 由 toycc 生成的 C 代码(教学示例)",
         "#include <stdio.h>",
         "#include <math.h>",
         ""]

    for name, val in graph.constants.items():
        if ti.info.get(name, ("", ""))[0] != "const":
            continue
        flat = np.asarray(val).reshape(-1)
        arr = ", ".join(f"{v:.8e}f" for v in flat)
        L.append(f"static const float {ti.name(name)}[] = {{ {arr} }};")
    L.append("")

    for b in range(nbufs):
        sz = max((a.size for a in allocs.values() if a.buf == b), default=1)
        L.append(f"static float buf{b}[{sz}];")
    L.append("")

    for name in graph.inputs:
        if ti.info.get(name, ("", ""))[0] == "input":
            n = int(np.prod(shapes[name]))
            L.append(f"static float {ti.name(name)}[{n}];")
    L.append("")

    L.append("void run(float* out) {")
    for node in graph.topo_order():
        b = CodeBuilder("c")
        _emit_node(b, node, shapes, ti)
        L.append("    {")
        L += ["    " + ln for ln in b.lines]
        L.append("    }")
    nout = int(np.prod(shapes[graph.outputs[0]]))
    L.append(f"    for (int i = 0; i < {nout}; i++) out[i] = {ti.name(graph.outputs[0])}[i];")
    L.append("}")
    L.append("")
    return "\n".join(L)


def emit_python(graph: Graph, allocs, shapes) -> str:
    ti = TensorInfo(graph, allocs, shapes)
    nbufs = max((a.buf for a in allocs.values()), default=-1) + 1

    L = ["import math",
         "import numpy as np",
         ""]
    for b in range(nbufs):
        sz = max((a.size for a in allocs.values() if a.buf == b), default=1)
        L.append(f"buf{b} = np.zeros({sz})")
    L.append("")
    for name, val in graph.constants.items():
        if ti.info.get(name, ("", ""))[0] == "const":
            L.append(f"{ti.name(name)} = np.array({np.asarray(val).reshape(-1).tolist()})")
    L.append("")

    L.append("def run(x):")
    L.append("    global " + ", ".join(f"buf{i}" for i in range(nbufs)))
    for name in graph.inputs:
        if ti.info.get(name, ("", ""))[0] == "input":
            L.append(f"    {ti.name(name)} = np.asarray(x, dtype=float).reshape(-1)")
    for node in graph.topo_order():
        b = CodeBuilder("py")
        _emit_node(b, node, shapes, ti)
        for ln in b.lines:
            L.append("    " + ln if ln else ln)
    nout = int(np.prod(shapes[graph.outputs[0]]))
    L.append(f"    return {ti.name(graph.outputs[0])}[:{nout}]")
    L.append("")
    return "\n".join(L)
