"""布局规划 pass。

真实对照(TVM/MLIR):
- conv 在 CPU 上常用 NHWC(通道在最后一维,利于 SIMD),GPU 上常用 NCHW。
- `LayoutPass` 对应 TVM 的 Layout 规划 + 插入 `layout_transform`(数据搬移)。
- "布局无关"算子(relu/add/reshape...)会继承输入的布局 —— 这就是"布局传播"。
- 布局不匹配的边界才插入 transform,避免冗余搬移 —— 这是本 pass 的核心收益。

我们的策略:把 conv1/conv2 改成 NHWC。由于 relu/add 继承布局,
conv1->bias->relu->conv2 整条链都是 NHWC,只在两处边界插 transform:
    x(nchw) --[nchw->nhwc]--> conv1 ... conv2 --[nhwc->nchw]--> reshape
"""
from __future__ import annotations

from toycc.ir import Graph, OpAttrs

# 布局无关的算子:输入什么布局,输出就什么布局
_LAYOUT_AGNOSTIC = {"relu", "sigmoid", "add", "mul", "layout_transform"}


class LayoutPass:
    def __init__(self, nhwc_conv: tuple[str, ...] = ("conv1", "conv2")):
        self.nhwc_conv = set(nhwc_conv)

    def __call__(self, graph: Graph) -> Graph:
        g = graph.clone()
        layout: dict[str, str] = {i: "nchw" for i in g.inputs}

        for node in g.topo_order():
            if node.op_type == "conv":
                if node.name in self.nhwc_conv:
                    # 数据、权重、bias 全部转成 nhwc(常量会在常量折叠阶段被预先算掉)
                    for i in range(len(node.inputs)):
                        self._ensure_layout(g, layout, node, i, want="nhwc")
                    node.attrs.layout = "nhwc"
                    layout[node.name] = "nhwc"
                else:
                    layout[node.name] = "nchw"

            elif node.op_type in _LAYOUT_AGNOSTIC:
                layout[node.name] = layout.get(node.inputs[0], "nchw")

            elif node.op_type == "reshape":
                # reshape 按内存顺序拍平,要求输入是标准 nchw 排布
                self._ensure_layout(g, layout, node, input_idx=0, want="nchw")
                layout[node.name] = "nchw"

            elif node.op_type == "matmul":
                layout[node.name] = "nchw"

            else:
                layout[node.name] = "nchw"

        return g

    @staticmethod
    def _ensure_layout(g: Graph, layout, node, input_idx: int, want: str):
        """如果节点第 input_idx 个输入的布局不是 want,就在它前面插一个 transform。
        插入后更新 layout 表 —— 这等价于 TVM 里的 InsertLayoutTransform。"""
        src = node.inputs[input_idx]
        cur = layout.get(src)
        if cur is None or cur == want:
            return
        name = f"{node.name}_lt_{input_idx}"
        lt = g.add_op(
            "layout_transform", [src], name=name,
            attrs=OpAttrs(from_layout=cur, to_layout=want))
        node.inputs[input_idx] = name
        layout[name] = want
