"""计算图 IR(有向无环图 DAG)。

真实对照:
- 这就是 Relay / ONNX / TensorFlow 里的"图 IR"。
- 每个 Node 相当于一个 Relay Call / ONNX 算子的实例。
- 图本身只描述"计算依赖",不关心数据怎么排布、怎么在内存里放 ——
  那些是后续 pass(布局、内存规划)的事。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from toycc.ir.ops import OPS, OpAttrs


@dataclass
class Node:
    name: str
    op_type: str
    inputs: list[str] = field(default_factory=list)
    attrs: OpAttrs = field(default_factory=OpAttrs)
    # 融合信息:如果本节点是"融合后"的复合算子,这里记录子图
    fused_ops: Optional[list[str]] = None

    def __post_init__(self):
        if self.attrs is None:
            self.attrs = OpAttrs()
        if self.op_type not in OPS:
            raise KeyError(f"未知算子 {self.op_type!r},可用的有 {sorted(OPS)}")

    def __str__(self):
        if self.fused_ops:
            inner = "+".join(self.fused_ops)
            extra = f"  <- fused: {inner}"
        else:
            extra = ""
        return f"{self.name} = {self.op_type}({', '.join(self.inputs)}){extra}"


class Graph:
    def __init__(self, name: str = "main"):
        self.name = name
        self.inputs: list[str] = []          # 图的输入张量名
        self.nodes: dict[str, Node] = {}     # name -> Node
        self.outputs: list[str] = []         # 图的输出张量名
        self.constants: dict[str, object] = {}  # 常量张量值(编译期已知)

    def set_constant(self, name: str, value):
        self.constants[name] = value
        return self

    def add_input(self, name: str) -> Node:
        self.inputs.append(name)
        return self._fake_node(name, "placeholder")

    def add_op(self, op_type, inputs, name=None, attrs=None, fused_ops=None) -> Node:
        info = OPS[op_type]
        if len(inputs) != info.num_inputs:
            raise ValueError(
                f"{op_type} 需要 {info.num_inputs} 个输入,给了 {len(inputs)}: {inputs}")
        if name is None:
            name = f"{op_type}_{len(self.nodes)}"
        node = Node(name, op_type, list(inputs),
                    attrs=attrs or OpAttrs(), fused_ops=fused_ops)
        self.nodes[name] = node
        return node

    def mark_output(self, name: str):
        if name not in self.nodes and name not in self.inputs:
            raise KeyError(f"输出 {name} 不存在")
        self.outputs.append(name)

    def rewire(self, old_name: str, new_name: str):
        """把图里所有指向 old_name 的边改指向 new_name。
        等价于 TVM pass 里把子图的输出内联到父算子。"""
        for n in self.nodes.values():
            n.inputs = [new_name if i == old_name else i for i in n.inputs]
        self.outputs = [new_name if o == old_name else o for o in self.outputs]

    def remove_node(self, name: str):
        if name in self.nodes:
            del self.nodes[name]

    def _fake_node(self, name, op_type):
        # 图输入(placeholder)不真正存成 Node,只占个名字。
        node = Node.__new__(Node)          # 跳过 __post_init__ 的算子校验
        node.name = name
        node.op_type = op_type
        node.inputs = []
        node.attrs = OpAttrs()
        node.fused_ops = None
        return node

    # ---------------- 图分析 ----------------

    def consumers(self, name: str) -> list[Node]:
        """谁消费了张量 name —— 这就是做融合时判断"能不能合并"的关键。"""
        return [n for n in self.nodes.values() if name in n.inputs]

    def producer(self, name: str) -> Optional[Node]:
        """张量 name 由哪个节点产出。"""
        return self.nodes.get(name) or (
            Node(name, "placeholder") if name in self.inputs else None)

    def topo_order(self) -> list[Node]:
        """拓扑序:保证每个节点在其消费者之前执行(参考执行用)。"""
        # 简单 Kahn 算法:入度 = 依赖的节点个数(图输入不是节点,不算)
        indeg = {n: sum(1 for i in node.inputs if i in self.nodes)
                 for n, node in self.nodes.items()}
        ready = [n for n, d in indeg.items() if d == 0]
        order = []
        while ready:
            n = ready.pop(0)
            order.append(self.nodes[n])
            for c in self.nodes.values():
                if n in c.inputs:
                    indeg[c.name] -= 1
                    if indeg[c.name] == 0:
                        ready.append(c.name)
        if len(order) != len(self.nodes):
            raise RuntimeError("图中存在环,不是合法的 DAG")
        return order

    def clone(self) -> "Graph":
        g = Graph(self.name)
        g.inputs = list(self.inputs)
        g.outputs = list(self.outputs)
        g.constants = dict(self.constants)
        for n in self.nodes.values():
            g.nodes[n.name] = Node(n.name, n.op_type, list(n.inputs),
                                   n.attrs.copy(), list(n.fused_ops) if n.fused_ops else None)
        return g

    # ---------------- 展示 ----------------

    def dump(self) -> str:
        lines = [f"graph {self.name}:"]
        for i in self.inputs:
            lines.append(f"  input: {i}")
        for n in self.topo_order():
            lines.append(f"  {n}")
        lines.append(f"  output: {', '.join(self.outputs)}")
        return "\n".join(lines)

    def __str__(self):
        return self.dump()
