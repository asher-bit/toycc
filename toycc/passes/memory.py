"""内存规划 pass(liveness 分析 + 缓冲区复用)。

真实对照:
- TVM 的 StorageRewrite / plan_storage:给每个张量分配存储,生命周期结束的
  缓冲区可以被后续张量复用。
- XLA 的 BufferAssignment、Graphcore/PyTorch 的 memory planning 是同一件事。
- 关键:决定内存的是"张量的活跃区间(liveness)",不是"张量个数"。

这里我们输出一张"分配表":每个中间张量在哪个缓冲区、什么时候出生/死亡,
以及相比"每个张量独占一块内存"省下了多少。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from toycc.ir import Graph
from toycc.ir.ops import OPS


@dataclass
class Buffer:
    index: int
    owner: str = ""          # 当前占用它的张量名
    born_at: int = -1        # 出生时刻(topology 序号)
    died_at: int = -1        # 死亡时刻(最后一个消费者的序号)
    size: int = 0


@dataclass
class AllocEntry:
    tensor: str
    shape: tuple
    size: int
    buf: int
    born_at: int
    died_at: int


def infer_shapes(graph: Graph, input_shapes: dict[str, tuple]) -> dict[str, tuple]:
    """形状推导:沿拓扑序,用各算子的 infer_shape 算出每个张量的形状。
    真实编译器里这就是 shape inference / 常量 shape propagation。"""
    shapes = dict(input_shapes)
    for name, val in graph.constants.items():
        shapes.setdefault(name, tuple(val.shape))
    for node in graph.topo_order():
        ins = [shapes[i] for i in node.inputs]
        shapes[node.name] = OPS[node.op_type].infer_shape(ins, node.attrs)
    return shapes


class MemoryPlanningPass:
    def __init__(self, input_shapes: dict[str, tuple]):
        self.input_shapes = input_shapes

    def __call__(self, graph: Graph) -> tuple[Graph, list[AllocEntry]]:
        g = graph.clone()
        shapes = infer_shapes(g, self.input_shapes)
        topo = g.topo_order()

        # ---- 1. 计算每个张量的生命周期(出生=产出,死亡=最后一个消费者) ----
        born = {i: 0 for i in g.inputs}          # 输入一开始就活着
        died = {i: len(topo) for i in g.inputs}  # 输出结束才死
        for t, node in enumerate(topo):
            born[node.name] = t
        for i in g.outputs:
            died[i] = len(topo)  # 图输出活到最后

        for t, node in enumerate(topo):
            for inp in node.inputs:
                died[inp] = max(died.get(inp, -1), t)  # 被消费的时刻

        # ---- 2. 分配缓冲区:第一个"已死"的缓冲区可以复用 ----
        def size_of(name):
            s = shapes.get(name)
            if s is None:
                return 0
            n = 1
            for d in s:
                n *= d
            return n

        buffers: list[Buffer] = []
        allocs: dict[str, AllocEntry] = {}
        # 给每个节点的"输出"分配缓冲区(输入来自外部 feed,不占内部内存)
        for t, node in enumerate(topo):
            name = node.name
            sz = size_of(name)
            reuse = next((b for b in buffers if b.died_at < t and b.size >= sz), None)
            if reuse is None:
                reuse = Buffer(index=len(buffers), size=sz)
                buffers.append(reuse)
            reuse.owner = name
            reuse.born_at = t
            reuse.died_at = died.get(name, t)
            allocs[name] = AllocEntry(name, shapes[name], sz, reuse.index,
                                      t, reuse.died_at)
        return g, [a for _, a in sorted(allocs.items(), key=lambda kv: kv[1].buf)]


def report(graph: Graph, entries: list[AllocEntry], input_shapes: dict[str, tuple]) -> str:
    """打印分配表 + 内存节省统计。"""
    shapes = infer_shapes(graph, input_shapes)
    topo = graph.topo_order()
    lines = []
    lines.append(f"{'张量':<14}{'形状':<16}{'元素数':>8}{'缓冲':>5}{'生→死':>9}")
    n_naive, n_bufs, max_buf = 0, 0, 0
    for e in entries:
        lines.append(f"{e.tensor:<14}{str(e.shape):<16}{e.size:>8}{e.buf:>5}"
                     f"  {e.born_at}→{e.died_at}")
    for e in entries:
        n_naive += e.size
        if e.buf >= max_buf:
            max_buf = e.buf
    n_bufs = max_buf + 1
    # 统计实际峰值内存:每时刻所有活跃缓冲区大小之和
    peak = 0
    for t in range(len(topo)):
        used = sum(e.size for e in entries if e.born_at <= t < e.died_at)
        peak = max(peak, used)
    lines.append("-" * 50)
    lines.append(f"朴素方案(每张量独占): {n_naive:>8} 元素")
    lines.append(f"复用方案(峰值):       {peak:>8} 元素")
    lines.append(f"缓冲区个数: {n_bufs} 个(复用率 "
                 f"{len(entries)} 个张量)")
    lines.append(f"内存节省: {(1 - peak/n_naive)*100:.0f}%")
    return "\n".join(lines)
