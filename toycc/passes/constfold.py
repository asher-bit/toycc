"""常量折叠 pass。

真实对照:TVM relay.transform.FoldConstant / ONNX optimize model 的
constant folding / XLA 的 constant folding。

这里它负责把"只吃常量"的 layout_transform 预先算掉 ——
例如把 conv 的权重从 NCHW 转成 NHWC。这非常关键:
布局优化若要求权重换布局,换布局的开销应该发生在编译期(离线一次),
而不是每次推理运行时都付一遍。
"""
from __future__ import annotations

import numpy as np

from toycc.ir import Graph
from toycc.runtime.ref import RefImpl


class ConstantFoldPass:
    def __call__(self, graph: Graph) -> Graph:
        g = graph.clone()
        for node in list(g.nodes.values()):
            # 所有输入都是常量,就可以在编译期把算子算出来
            if all(i in g.constants for i in node.inputs):
                ins = [g.constants[i] for i in node.inputs]
                fn = getattr(RefImpl, node.op_type)
                value = fn(ins, node.attrs)
                # 结果直接变成常量,节点消失,消费者照常引用这个名字
                g.set_constant(node.name, np.asarray(value))
                g.inputs.append(node.name)
                g.remove_node(node.name)
        return g
