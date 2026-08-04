"""numpy 参考执行器。

真实对照:这是"朴素解释器" —— 不优化,只保证算对。
TVM 里的 python 前端、ONNX Runtime 的 reference evaluator、PyTorch eager mode
本质上都干这件事。我们所有优化 pass 的正确性,都以它为准。
"""
from __future__ import annotations

import numpy as np

from toycc.ir.graph import Graph
from toycc.ir.ops import OpAttrs


class RefImpl:
    """每个算子一个 numpy 实现。"""

    @staticmethod
    def conv(inputs, attrs: OpAttrs):
        x, w = inputs[:2]  # 融合 bias 后 inputs 可能是 3 个
        bias = inputs[2] if len(inputs) > 2 else None

        if attrs.layout == "nhwc":
            return RefImpl._conv_nhwc(x, w, bias, attrs)

        # ---- NCHW 路径 ----
        N, C, H, W = x.shape
        OC, _, KH, KW = w.shape
        SH, SW = attrs.stride
        PH, PB, PL, PR = attrs.pad
        xp = np.pad(x, ((0, 0), (0, 0), (PH, PB), (PL, PR)))
        OH = (H + PH + PB - KH) // SH + 1
        OW = (W + PL + PR - KW) // SW + 1
        out = np.zeros((N, OC, OH, OW))
        for n in range(N):
            for oc in range(OC):
                for oh in range(OH):
                    for ow in range(OW):
                        h0, w0 = oh * SH, ow * SW
                        out[n, oc, oh, ow] = np.sum(
                            xp[n, :, h0:h0+KH, w0:w0+KW] * w[oc])
        if bias is not None:
            out += bias  # numpy 广播:(1,OC,1,1) 或 (OC,)
        return out

    @staticmethod
    def _conv_nhwc(x, w, bias, attrs):
        """NHWC 卷积:输入 x:(N,H,W,C),权重/偏置已被布局 pass 预转换好。
        (在真实编译器里,这个预转换由"常量折叠"在编译期完成,不占推理开销)"""
        N, H, W, C = x.shape
        OC, KH, KW, _ = w.shape
        SH, SW = attrs.stride
        PH, PB, PL, PR = attrs.pad
        xp = np.pad(x, ((0, 0), (PH, PB), (PL, PR), (0, 0)))
        OH = (H + PH + PB - KH) // SH + 1
        OW = (W + PL + PR - KW) // SW + 1
        out = np.zeros((N, OH, OW, OC))
        for n in range(N):
            for oh in range(OH):
                for ow in range(OW):
                    for oc in range(OC):
                        h0, w0 = oh * SH, ow * SW
                        out[n, oh, ow, oc] = np.sum(
                            xp[n, h0:h0+KH, w0:w0+KW, :] * w[oc])
        if bias is not None:
            out += bias   # (1,1,1,OC) 广播
        return out

    @staticmethod
    def layout_transform(inputs, attrs):
        x = inputs[0]
        if attrs.from_layout == "nchw" and attrs.to_layout == "nhwc":
            return np.transpose(x, (0, 2, 3, 1))
        if attrs.from_layout == "nhwc" and attrs.to_layout == "nchw":
            return np.transpose(x, (0, 3, 1, 2))
        return x

    @staticmethod
    def matmul(inputs, attrs):
        a, b = inputs[:2]
        out = np.matmul(a, b)
        if len(inputs) > 2:
            out = out + inputs[2]
        return out

    @staticmethod
    def relu(inputs, attrs):
        return np.maximum(inputs[0], 0.0)

    @staticmethod
    def sigmoid(inputs, attrs):
        return 1.0 / (1.0 + np.exp(-inputs[0]))

    @staticmethod
    def add(inputs, attrs):
        return inputs[0] + inputs[1]

    @staticmethod
    def mul(inputs, attrs):
        return inputs[0] * inputs[1]

    @staticmethod
    def reshape(inputs, attrs):
        return inputs[0].reshape(attrs.target_shape)


# 融合激活:在算子输出上再套一个激活。这正是"conv+relu 融合"的数学含义。
_ACTIVATIONS = {"relu": RefImpl.relu, "sigmoid": RefImpl.sigmoid}


class ReferenceEvaluator:
    def __init__(self, graph: Graph):
        self.graph = graph

    def run(self, feed: dict[str, np.ndarray],
            overrides: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """按拓扑序逐个执行。feed: {输入名: ndarray},常量自动并入。
        overrides: {节点名: ndarray} —— 用给定值替换该节点的计算结果
        (量化/精度模拟用)。"""
        vals: dict[str, np.ndarray] = dict(self.graph.constants)
        vals.update(feed)
        overrides = overrides or {}
        for node in self.graph.topo_order():
            if node.name in overrides:
                vals[node.name] = overrides[node.name]
                continue
            ins = [vals[i] for i in node.inputs]
            fn = getattr(RefImpl, node.op_type)
            out = fn(ins, node.attrs)
            # 处理融合进来的激活:conv+relu / matmul+relu ...
            if node.attrs.activation:
                out = _ACTIVATIONS[node.attrs.activation]([out], OpAttrs())
            vals[node.name] = out
        return {o: vals[o] for o in self.graph.outputs}


def evaluate(graph: Graph, feed,
             overrides: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    return ReferenceEvaluator(graph).run(feed, overrides=overrides)


def evaluate_all(graph: Graph, feed) -> dict[str, np.ndarray]:
    """执行全部节点, 返回所有中间值(量化模拟要用)。"""
    vals: dict[str, np.ndarray] = dict(graph.constants)
    vals.update(feed)
    ev = ReferenceEvaluator(graph)
    for node in graph.topo_order():
        ins = [vals[i] for i in node.inputs]
        fn = getattr(RefImpl, node.op_type)
        out = fn(ins, node.attrs)
        if node.attrs.activation:
            out = _ACTIVATIONS[node.attrs.activation]([out], OpAttrs())
        vals[node.name] = out
    return vals
