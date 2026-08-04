"""Pass 基础设施 + 算子融合 pass。

真实对照(TVM):
- `PassManager` 对应 tvm.relay.transform 里的 pass pipeline。
- 融合 pass 对应 tvm.relay.transform.FuseOps / ElemwiseFuse(生成 FusedOp)。
- 这里的"把子算子吸收进父算子"对应 Relay 里 FusionGroup 把多个 Call
  并成一个 `FusedFunction`。
- 能否融合的判断 = "融合代价/规则",真实编译器里有复杂的 cost model,
  我们先用几条直觉规则。
"""
from __future__ import annotations

from toycc.ir import Graph, OpAttrs

# 可被吸收进"计算根算子"的点算子
_POINTWISE = {"relu", "sigmoid"}
# 可当 bias 吸收进 conv/matmul 的加法
_BIASABLE = {"add"}
# 作为计算根、可以吸收下游的算子
_ROOTS = {"conv", "matmul"}


def _is_constant(graph: Graph, name: str) -> bool:
    """该张量是不是常量(图输入/权重)。bias 必须是常量才能安全吸收。"""
    return name in graph.inputs


class FusionPass:
    """贪婪融合:对每个"根算子",沿唯一消费者链不断吸收可融合的点算子。"""

    def __call__(self, graph: Graph) -> Graph:
        # 用副本改写,保持 pass 无副作用(可组合/可重复跑)
        g = graph.clone()
        for node in list(g.nodes.values()):
            if node.op_type not in _ROOTS:
                continue
            self._absorb_followers(g, node)
        return g

    def _absorb_followers(self, g: Graph, node):
        """把 node 之后一串可融合的算子一个个吸进来。"""
        while True:
            cons = g.consumers(node.name)
            if len(cons) != 1:          # 多个消费者:不能简单内联
                break
            c = cons[0]

            # 情况 1:relu/sigmoid —— 变成父算子的 activation 属性
            if c.op_type in _POINTWISE:
                node.attrs.activation = c.op_type
                node.fused_ops = (node.fused_ops or []) + [c.op_type]
                self._absorb(g, node, c)
                continue

            # 情况 2:add —— 且另一个操作数是常量,当 bias 吸收
            if c.op_type in _BIASABLE and node.op_type in _ROOTS:
                other = [i for i in c.inputs if i != node.name]
                if len(other) == 1 and _is_constant(g, other[0]):
                    node.attrs.bias = True
                    node.inputs.append(other[0])      # bias 成为根的第三个输入
                    node.fused_ops = (node.fused_ops or []) + [c.op_type]
                    self._absorb(g, node, c)
                    continue

            break  # 其余情况不融合

    @staticmethod
    def _absorb(g: Graph, parent, child):
        """把 child 节点并入 parent:重连所有用到 child 的边,再删掉 child。"""
        g.rewire(child.name, parent.name)
        g.remove_node(child.name)


def pass_pipeline(graph: Graph, passes=("fusion",)) -> Graph:
    """简易 PassManager。真实版会带依赖分析/调试输出,这里够用。"""
    result = graph
    for p in passes:
        if p == "fusion":
            result = FusionPass()(result)
        else:
            raise ValueError(f"未知 pass {p}")
    return result
