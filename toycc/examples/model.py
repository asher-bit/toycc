"""示例模型:一个迷你 CNN 前段 + MLP 尾巴。

结构(真实 CNN 的顺序:卷积 → 加 bias → 激活):
    x:(1,3,8,8)
      └─ conv(3→4,k3) ─ add(bias1) ─ relu ─ conv(4→8,k3) ─ add(bias2) ─ relu
         ─ reshape ─ matmul ─ add(bias3) ─ relu ─> output

【教学点】一开始我把顺序写成 conv→relu→add,融合成 conv+bias+relu 后
语义变了(relu(conv)+bias ≠ relu(conv+bias)),验证就 FAIL。
融合必须保持语义等价 —— 这正是编译器 pass 正确性验证的意义。

我们会在 pass 阶段把这 11 个算子融合成 3 个核(kernel)。
"""
from __future__ import annotations

import numpy as np

from toycc.ir import Graph, OpAttrs


def build_model() -> Graph:
    return build_model_with_weights()[0]


def default_weights() -> dict[str, np.ndarray]:
    """生成固定随机种子权重,保证每次跑出来可复现。"""
    rng = np.random.default_rng(0)
    return {
        "conv1_w": rng.standard_normal((4, 3, 3, 3)) * 0.1,
        "bias1":   rng.standard_normal((1, 4, 1, 1)) * 0.1,  # 已按通道广播对齐
        "conv2_w": rng.standard_normal((8, 4, 3, 3)) * 0.1,
        "bias2":   rng.standard_normal((1, 8, 1, 1)) * 0.1,
        "w_mm":    rng.standard_normal((128, 16)) * 0.1,
        "bias3":   rng.standard_normal((1, 16)) * 0.1,
    }


def build_model_with_weights() -> tuple[Graph, dict[str, np.ndarray]]:
    """把权重以"常量输入"的形式接进图,并把值存进 graph.constants
    (真实编译器里权重是 Attach 到 IR 的常量,供"常量折叠"pass 使用)。"""
    w = default_weights()
    g = Graph("tiny_cnn_const")
    g.add_input("x")
    for name in w:
        g.add_input(name)
        g.set_constant(name, w[name])

    c1 = g.add_op("conv", ["x", "conv1_w"], "conv1", OpAttrs(
        kernel=(3, 3), stride=(1, 1), pad=(1, 1, 1, 1)))
    b1 = g.add_op("add", ["conv1", "bias1"], "bias_add1")
    r1 = g.add_op("relu", ["bias_add1"], "relu1")

    c2 = g.add_op("conv", ["relu1", "conv2_w"], "conv2", OpAttrs(
        kernel=(3, 3), stride=(2, 2), pad=(1, 1, 1, 1)))
    b2 = g.add_op("add", ["conv2", "bias2"], "bias_add2")
    r2 = g.add_op("relu", ["bias_add2"], "relu2")

    f = g.add_op("reshape", ["relu2"], "flat", OpAttrs(target_shape=(-1, 128)))
    mm = g.add_op("matmul", ["flat", "w_mm"], "mm", OpAttrs())
    b3 = g.add_op("add", ["mm", "bias3"], "bias_add3")
    out = g.add_op("relu", ["bias_add3"], "output")

    g.mark_output("output")
    return g, w
