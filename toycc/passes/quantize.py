"""量化模拟器(教学用):把某个算子换成 int8 版本,量化精度损失。

真实对照:
- 量化(quantization): 把 float 权重/激活转成 int8/fp16 等低精度,
  换来 4 倍内存减半 + 专用 int8 加速单元(如 NVIDIA Tensor Core)。
- 编译器里的量化 pass(TVM/ONNX Runtime)会: 插 Quantize/Dequantize 算子、
  把 conv/matmul 替换成 int8 实现、做 scale 折叠。
- 这里的 QuantizeSimulator 不做图改写, 而是"拦截"目标算子的结果,
  用 int8 版本重算, 再继续跑完整个图 —— 用 max|Δ| 量化损失。

量化数学(对称 int8):
  q = clamp(round(x / scale), -127, 127)
  scale = max|张量| / 127
  反量化: x ≈ q * scale
"""
from __future__ import annotations

import numpy as np

from toycc.ir import Graph
from toycc.runtime.ref import evaluate, evaluate_all


def quantize(x, scale, bits=8):
    """对称量化: float -> int8。scale 可为标量(per-tensor)或向量(per-channel)。"""
    qmax = 2 ** (bits - 1) - 1
    return np.clip(np.round(x / scale), -qmax, qmax).astype(np.int8)


def dequantize(q, scale, bits=8):
    return q.astype(np.float32) * scale


def _conv_int8(x_int8, w_int8, attrs, bias=None):
    """int8 卷积: 累加在 int32, 避免溢出(教学版直接用 numpy)。"""
    N, C, H, W = x_int8.shape
    OC, _, KH, KW = w_int8.shape
    SH, SW = attrs.stride
    PH, PB, PL, PR = attrs.pad
    xp = np.pad(x_int8, ((0, 0), (0, 0), (PH, PB), (PL, PR)))
    OH = (H + PH + PB - KH) // SH + 1
    OW = (W + PL + PR - KW) // SW + 1
    out = np.zeros((N, OC, OH, OW), dtype=np.int32)
    for n in range(N):
        for oc in range(OC):
            for oh in range(OH):
                for ow in range(OW):
                    h0, w0 = oh * SH, ow * SW
                    out[n, oc, oh, ow] = np.sum(
                        xp[n, :, h0:h0 + KH, w0:w0 + KW].astype(np.int32)
                        * w_int8[oc].astype(np.int32))
    if bias is not None:
        out = out + bias.astype(np.int32)
    return out


class QuantizeSimulator:
    """把图中指定算子模拟成 int8 量化, 报告精度损失。"""

    def __init__(self, target: str = "conv1", bits: int = 8, per_channel: bool = False):
        self.target = target
        self.bits = bits
        self.per_channel = per_channel

    def run(self, graph: Graph, feed: dict[str, np.ndarray]) -> dict:
        # 1. 先跑 fp32, 拿金标准和所有中间值
        gold = evaluate(graph, feed)
        vals = evaluate_all(graph, feed)

        node = graph.nodes[self.target]
        x = vals[node.inputs[0]]
        w = graph.constants[node.inputs[1]]
        bias = graph.constants[node.inputs[2]] if len(node.inputs) > 2 else None

        # 2. 算 scale(用实际数据的 max|.|) 并量化
        qmax = 2 ** (self.bits - 1) - 1
        if self.per_channel:
            scale_x = float(np.max(np.abs(x))) / qmax
            scale_w = np.max(np.abs(w), axis=(1, 2, 3)).reshape(-1, 1, 1, 1) / qmax  # (OC,1,1,1)
        else:
            scale_x = float(np.max(np.abs(x))) / qmax
            scale_w = float(np.max(np.abs(w))) / qmax

        x_int8 = quantize(x, scale_x, self.bits)
        w_int8 = quantize(w, scale_w, self.bits)

        # 3. int8 卷积(累加 int32) + 反量化
        y_int32 = _conv_int8(x_int8, w_int8, node.attrs, bias)
        if self.per_channel:
            # y_int32: (N,OC,OH,OW) × (scale_x * scale_w[oc])
            y = y_int32 * (scale_x * scale_w).reshape(1, -1, 1, 1)
        else:
            y = y_int32 * (scale_x * scale_w)
        if node.attrs.activation == "relu":
            y = np.maximum(y, 0.0)

        # 4. 用 int8 结果覆盖该节点, 跑完剩余图
        q_out = evaluate(graph, feed, overrides={self.target: y})

        # 5. 报告
        ref_out = list(gold.values())[0]
        got_out = list(q_out.values())[0]
        err = float(np.max(np.abs(ref_out - got_out)))
        bits = self.bits
        return {
            "bits": bits,
            "per_channel": self.per_channel,
            "scale_x": scale_x,
            "scale_w": scale_w,
            "int8_conv_range": (float(y_int32.min()), float(y_int32.max())),
            "max_abs_err": err,
            "gold_max": float(np.max(np.abs(ref_out))),
            "relative_err": err / (float(np.max(np.abs(ref_out))) + 1e-12),
            "gold": ref_out,
            "quantized": got_out,
        }


def report(graph, feed, target="conv1", bits=8, per_channel=False) -> str:
    r = QuantizeSimulator(target, bits, per_channel).run(graph, feed)
    lines = []
    lines.append(f"量化目标: {target}  ({bits} bit, "
                 f"{'per-channel 权重' if per_channel else 'per-tensor'})")
    lines.append(f"  输入 scale   = {r['scale_x']:.4e}")
    if per_channel:
        sw = np.asarray(r["scale_w"]).reshape(-1)
        lines.append(f"  权重 scale   = per-channel, 范围 "
                     f"[{np.min(sw):.2e}, {np.max(sw):.2e}]")
    else:
        lines.append(f"  权重 scale   = {r['scale_w']:.4e}")
    lines.append(f"  int8 卷积累加范围 = [{r['int8_conv_range'][0]}, "
                 f"{r['int8_conv_range'][1]}]  (int32 能安全装下)")
    lines.append(f"  输出 max|Δ|  = {r['max_abs_err']:.4e}  "
                 f"(相对误差 {r['relative_err']:.1%})")
    lines.append(f"  (gold 最大幅值 = {r['gold_max']:.4e})")
    return "\n".join(lines)
