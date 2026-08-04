"""TIR 调度模拟器(教学用)。

真实对照:TVM 里 TIR 调度是 `te.schedule` / `tirx.Schedule` 干的事——
在不改变计算语义的前提下,变换循环的:
  split(拆分) / fuse(合并) / reorder(重排) / tile(分块) /
  vectorize(向量化) / parallel(并行) / bind(绑定线程)

我们做一个极简版:把"一层层循环 + 一个主体表达式"表示成数据结构,
然后施加这些调度原语,最终渲染成 C 风格的代码让你亲眼看到"前后长什么样"。

关键思想:调度的对象是"循环",不是"图"。图(Relax)决定算什么,
调度(TIR)决定怎么算。调度必须保持语义等价——循环变换不改结果。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Loop:
    var: str                       # 循环变量名,如 "i"
    extent: int                    # 循环次数,如 16
    kind: str = "seq"              # seq / vector / parallel / thread
    comment: str = ""              # 附加说明(合并/拆分来源等)


@dataclass
class LoopNest:
    loops: list[Loop] = field(default_factory=list)
    body: str = ""                 # 主体,用 {var} 占位,如 "C[{i}*N+{j}] = acc"

    def render(self, lang="c") -> str:
        """把循环嵌套渲染成代码(缩进 + 可选的注释)。"""
        lines: list[str] = []
        ind = ""

        def emit(txt):
            lines.append(ind + txt)

        for L in self.loops:
            if L.kind == "seq":
                emit(f"for ({L.var} = 0; {L.var} < {L.extent}; {L.var}++) {{"
                     + (f"  // {L.comment}" if L.comment else ""))
            elif L.kind == "vector":
                emit(f"// [{L.var}] 向量化: 一次算 {L.extent} 个元素 (SIMD)"
                     + (f"  // {L.comment}" if L.comment else ""))
            elif L.kind == "parallel":
                emit(f"#pragma omp parallel for")
                emit(f"for ({L.var} = 0; {L.var} < {L.extent}; {L.var}++) {{")
            elif L.kind == "thread":
                emit(f"// 线程绑定: 一个线程负责 {L.var} 的一小段")
                emit(f"for ({L.var} = blockIdx*blockDim; {L.var} < {L.extent}; "
                     f"{L.var} += gridDim*blockDim) {{")
            ind += "  "
        emit(self.body.format(**{L.var: L.var for L in self.loops}))
        for _ in self.loops:
            ind = ind[:-2]
            if ind == "":
                lines.append("}")
            else:
                lines.append(ind + "}")
        return "\n".join(lines)

    # ---------------- 调度原语 ----------------

    def index(self, var: str) -> int:
        for i, L in enumerate(self.loops):
            if L.var == var:
                return i
        raise KeyError(f"循环 {var} 不存在")

    def split(self, var: str, factor: int) -> "LoopNest":
        """把循环 [0, extent) 拆成 外层[0, extent//factor) × 内层[0, factor)。
        变量重命名为 {var}_o / {var}_i,主体里的 {var} 替换成 {var}_o*factor+{var}_i。"""
        i = self.index(var)
        L = self.loops[i]
        outer = L.extent // factor
        inner = factor
        new_loops = list(self.loops)
        # 旧变量的引用换成语义等价的表达式
        body = self.body.replace("{" + var + "}", "{" + var + "_o}*" + str(factor) + "+{" + var + "_i}")
        new_loops[i:i+1] = [
            Loop(var + "_o", outer, L.kind, f"split({var},{factor}) 外层"),
            Loop(var + "_i", inner, "seq", f"split({var},{factor}) 内层"),
        ]
        out = LoopNest(new_loops, body)
        return out

    def fuse(self, var1: str, var2: str) -> "LoopNest":
        """合并两个相邻循环为一个循环。主体里 var2 换成 {var2}=row/... 略复杂,
        这里用 {var1}*extent2+{var2} 的约定,变量重命名 var1 为新循环。"""
        i, j = self.index(var1), self.index(var2)
        if j != i + 1:
            raise ValueError("fuse 只支持相邻的两个循环")
        e1, e2 = self.loops[i].extent, self.loops[j].extent
        new = Loop(var1, e1 * e2, self.loops[i].kind, f"fuse({var1},{var2})")
        body = self.body
        # 原 var2 语义由 var1 的高低位决定: var2 = var1 % e2
        body = body.replace("{" + var2 + "}", "{" + var1 + "} % " + str(e2))
        new_loops = list(self.loops)
        new_loops[i:j+1] = [new]
        return LoopNest(new_loops, body)

    def reorder(self, *vars) -> "LoopNest":
        """把指定循环按给定顺序重排(必须是当前循环集合的一个排列)。"""
        order = {v: k for k, v in enumerate(vars)}
        if sorted(vars) != sorted(L.var for L in self.loops):
            raise ValueError("reorder 必须包含所有循环且不重复")
        new_loops = sorted(self.loops, key=lambda L: order[L.var])
        return LoopNest(new_loops, self.body)

    def tile(self, var1: str, var2: str, b1: int, b2: int) -> "LoopNest":
        """2D 分块:把 (i,j) 拆成 (io,jo,ii,ji), 缓存友好。
        主体替换: i = io*b1+ii, j = jo*b2+ji。"""
        body = self.body
        body = body.replace("{" + var1 + "}", "{" + var1 + "_o}*" + str(b1) + "+{" + var1 + "_i}")
        body = body.replace("{" + var2 + "}", "{" + var2 + "_o}*" + str(b2) + "+{" + var2 + "_i}")

        loops = list(self.loops)
        i = self.index(var1)
        i_extent = loops[i].extent
        loops[i:i+1] = [Loop(var1 + "_o", i_extent // b1, "seq", "tile 外层"),
                        Loop(var1 + "_i", b1, "seq", "tile 内层")]
        # 在"修改后的列表"里找 var2,不能用原索引
        j = next(k for k, L in enumerate(loops) if L.var == var2)
        j_extent = loops[j].extent
        loops[j:j+1] = [Loop(var2 + "_o", j_extent // b2, "seq", "tile 外层"),
                        Loop(var2 + "_i", b2, "seq", "tile 内层")]
        return LoopNest(loops, body)

    def vectorize(self, var: str) -> "LoopNest":
        i = self.index(var)
        self.loops[i].kind = "vector"
        return self

    def parallel(self, var: str) -> "LoopNest":
        i = self.index(var)
        self.loops[i].kind = "parallel"
        return self


def matmul_nest(M, N, K) -> LoopNest:
    """一个经典的 matmul 循环嵌套,作为调度的"原料"。"""
    return LoopNest([
        Loop("i", M),
        Loop("j", N),
        Loop("k", K),
    ], body=("acc[{i},{j}] += A[{i}*{K}+{k}] * B[{k}*{N}+{j}]"
             .replace("{K}", str(K)).replace("{N}", str(N))))


def matmul_scheduled(M=8, N=8, K=8, block=4):
    """演示一次完整调度:分块 + 重排 + 向量化 + 并行。
    返回 (初始嵌套, 调度后嵌套)。"""
    nest = matmul_nest(M, N, K)          # 展示用(保持原样)
    t = matmul_nest(M, N, K)             # 调度的"原料"(独立实例,避免共享污染)
    # 1. tile i,j by block
    t = t.tile("i", "j", block, block)
    # 2. 把 k 提到最外层(减少缓存行冲突, 经典 matmul 优化)
    t = t.reorder("k", "i_o", "j_o", "i_i", "j_i")
    # 3. 内层 j_i 向量化
    t = t.vectorize("j_i")
    # 4. 最外层(块循环)并行
    t = t.parallel("k")
    return nest, t
