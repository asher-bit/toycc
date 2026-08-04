"""死代码消除(DCE) pass。

真实对照:TVM `relax.transform.DeadCodeElimination` / LLVM `-dce`。

死代码 = 计算结果没人消费、也不是图输出的节点。
它白白浪费编译产物大小,删掉不影响任何结果。

关键:删除一个节点后,它的输入可能也不再被消费了 —— 所以要
"反复删到没有可删的"(不动点 / fixpoint),用工作队列实现。

一个变体是 "CSE 逆过程" 的配合:折叠/内联产生的新常量如果没人用,
DCE 顺手清掉(第5课 FAQ 提过)。
"""
from __future__ import annotations

from toycc.ir import Graph


class DCEPass:
    def __call__(self, graph: Graph) -> Graph:
        g = graph.clone()
        # 工作队列:所有节点先都"待检查"
        queue = [n.name for n in g.nodes.values()]
        while queue:
            name = queue.pop()
            if name in g.outputs:
                continue
            node = g.nodes.get(name)
            if node is None:
                continue
            # 没消费者 = 死代码
            if not g.consumers(name):
                # 删掉它, 它的输入可能因此变成死代码, 继续查
                g.remove_node(name)
                queue.extend(node.inputs)
        return g
