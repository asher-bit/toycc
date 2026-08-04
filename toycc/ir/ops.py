"""算子定义层。

真实对照:TVM 里每个算子对应一个 `Op` 注册表(OpRegistry),
它描述了算子的输入数量、属性名、以及对应的调度(TE/TIR 模板)。
这里我们做同样的抽象:注册表 + 属性校验。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

# 每个算子需要一个 compute 函数,给出输出 shape 和引用实现。
# 后面写 pass(融合/布局)时只依赖"算子身份 + 属性",不依赖实现,
# 这正是 TVM 里 IR(Relay) 与后端(TE/TIR)解耦的关键思想。

# ---------------- 算子属性 ----------------

@dataclass
class OpAttrs:
    """算子的属性。不同算子只取自己需要的字段,多余字段置 None。"""
    kernel: tuple[int, int] | None = None   # conv 核大小 (kh, kw)
    stride: tuple[int, int] | None = None   # conv/maxpool 步长
    pad: tuple[int, int, int, int] | None = None  # 上/下/左/右
    groups: int = 1
    bias: bool = False
    activation: str | None = None           # 融合进本算子的激活名(如 relu)
    target_shape: tuple | None = None       # reshape 的目标形状
    layout: str | None = None               # 算子的数据布局: "nchw" / "nhwc" / None
    from_layout: str | None = None          # layout_transform 专用
    to_layout: str | None = None

    def copy(self) -> "OpAttrs":
        return OpAttrs(
            kernel=self.kernel,
            stride=self.stride,
            pad=self.pad,
            groups=self.groups,
            bias=self.bias,
            activation=self.activation,
            target_shape=self.target_shape,
            layout=self.layout,
            from_layout=self.from_layout,
            to_layout=self.to_layout,
        )


# ---------------- 算子注册表 ----------------

@dataclass(frozen=True)
class OpInfo:
    name: str
    num_inputs: int          # 固定输入个数(weight/bias 算固定输入)
    num_outputs: int
    infer_shape: Any         # 函数 (shapes: list[tuple], attrs) -> tuple
    ref: Any                 # 函数 (inputs: list[ndarray], attrs) -> ndarray


# 占位:引用实现放在 runtime/ref.py,这里延迟 import 避免循环依赖。
_REF = None

def _ref():
    global _REF
    if _REF is None:
        from toycc.runtime.ref import RefImpl  # noqa: PLC0415
        _REF = RefImpl
    return _REF


OPS: dict[str, OpInfo] = {}


def register(name, num_inputs, infer_shape, ref):
    """注册一个算子,等价于 TVM 的 `register_op` / ONNX 的算子库。"""
    OPS[name] = OpInfo(name, num_inputs, 1, infer_shape, ref)
    return OPS[name]


# ---------------- shape 推导 ----------------

def _conv_shape(input_shapes, attrs):
    kh, kw = attrs.kernel
    sh, sw = attrs.stride
    ph, pb, pl, pr = attrs.pad
    if attrs.layout == "nhwc":
        N, H, W, _ = input_shapes[0]
        OC = input_shapes[1][0]
        OH = (H + ph + pb - kh) // sh + 1
        OW = (W + pl + pr - kw) // sw + 1
        return (N, OH, OW, OC)
    N, C, H, W = input_shapes[0]
    OC = input_shapes[1][0]
    OH = (H + ph + pb - kh) // sh + 1
    OW = (W + pl + pr - kw) // sw + 1
    return (N, OC, OH, OW)


def _matmul_shape(input_shapes, attrs):
    return (input_shapes[0][0], input_shapes[1][1])


def _pointwise_shape(input_shapes, attrs):
    return input_shapes[0]


def _add_shape(input_shapes, attrs):
    a, b = input_shapes
    if a == b:
        return a
    # broadcast:b 是 (1,) 或 (OC,),输出取 a
    return a


def _reshape_shape(input_shapes, attrs):
    target = attrs.target_shape
    total = 1
    for d in input_shapes[0]:
        total *= d
    # 允许 -1 推断
    out = list(target)
    neg = out.count(-1)
    if neg > 1:
        raise ValueError("reshape 最多一个 -1")
    if neg == 1:
        known = 1
        for d in out:
            if d != -1:
                known *= d
        out[out.index(-1)] = total // known
    if total != 1:
        prod = 1
        for d in out:
            prod *= d
        if prod != total:
            raise ValueError(f"reshape {input_shapes[0]} -> {target} 元素数不匹配")
    return tuple(out)


def _layout_transform_shape(input_shapes, attrs):
    """layout_transform 不改逻辑形状,只改内存排布(物理形状)。"""
    if attrs.from_layout == "nchw" and attrs.to_layout == "nhwc":
        N, C, H, W = input_shapes[0]
        return (N, H, W, C)
    if attrs.from_layout == "nhwc" and attrs.to_layout == "nchw":
        N, H, W, C = input_shapes[0]
        return (N, C, H, W)
    return input_shapes[0]


# ---------------- 注册内置算子 ----------------

register("conv", 2, _conv_shape, None)       # 权重是固定输入 #1
register("matmul", 2, _matmul_shape, None)
register("relu", 1, _pointwise_shape, None)
register("sigmoid", 1, _pointwise_shape, None)
register("add", 2, _add_shape, None)          # 广播加法(也当 bias 用)
register("mul", 2, _add_shape, None)
register("reshape", 1, _reshape_shape, None)  # 内存视图变换(不改数据)
register("layout_transform", 1, _layout_transform_shape, None)  # 布局搬移
