"""硬件模型模拟器(教学用):缓存、延迟、寄存器压力。

真实对照:编译器在写调度/内存优化时,脑子里要有这张硬件模型:
  - 寄存器(0 周期, 容量最小)
  - L1 缓存(~4 周期)
  - L2 缓存(~12 周期)
  - L3 缓存(~40 周期)
  - DRAM(~200 周期)   ← 一次内存访问比一次计算贵几十上百倍

本模块提供:
  1. Cache: 组相联 + LRU 的缓存模拟器
  2. matmul_misses: 按不同循环顺序跑一遍 matmul 的地址流, 统计缓存缺失
  3. compare_schedules: 对比 ijk / ikj / tiled 三种写法的性能
  4. LATENCY: 内存层次延迟参考表
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 内存层次延迟参考(典型值, 单位: CPU 周期)
LATENCY = {
    "register": 0,
    "L1": 4,
    "L2": 12,
    "L3": 40,
    "DRAM": 200,
}

# 典型容量(仅供参考)
CAPACITY = {
    "register": 64,      # 个(通用寄存器数)
    "L1": 32 * 1024,     # 字节
    "L2": 256 * 1024,
    "L3": 8 * 1024 * 1024,
    "DRAM": 16 * 1024 * 1024 * 1024,
}


@dataclass
class Cache:
    """组相联 + LRU 的缓存。

    参数:
      num_sets: 组数
      ways:     每组几路(相联度)
      line_size: 每行几个"字"(元素)。float 是 1 字=4 字节。
    """
    num_sets: int
    ways: int
    line_size: int = 4   # 每行 4 个字 (16 字节)
    # 每行记录 {tag: [最后一个使用的序号, ...]} 简化为: sets[set_idx] = {tag: last_use}
    sets: list = field(default_factory=list)
    hits: int = 0
    misses: int = 0

    def __post_init__(self):
        self.sets = [dict() for _ in range(self.num_sets)]

    def access(self, addr: int) -> bool:
        """访问地址 addr(字地址), 返回是否命中。"""
        line = addr // self.line_size
        set_idx = line % self.num_sets
        tag = line // self.num_sets
        st = self.sets[set_idx]
        if tag in st:
            self.hits += 1
            st[tag] = self.hits + self.misses  # LRU 时间戳
            return True
        # 未命中: 插入(满则踢 LRU)
        if len(st) < self.ways:
            st[tag] = self.hits + self.misses
        else:
            victim = min(st, key=st.get)       # 最久没用过
            del st[victim]
            st[tag] = self.hits + self.misses
        self.misses += 1
        return False

    def stats(self) -> tuple[int, int, float]:
        total = self.hits + self.misses
        return self.hits, self.misses, (self.hits / total if total else 0.0)


def _matmul_addresses(M, N, K, order: str, block: int = 0):
    """按指定循环顺序生成 matmul 的地址访问流。

    A: (M,K) 行主序, 地址 = i*K + k
    B: (K,N) 行主序, 地址 = k*N + j
    C: (M,N) 行主序, 地址 = i*N + j
    order: "ijk" | "ikj" | "kij" | "tiled"
    """
    out = []
    if order == "ijk":
        for i in range(M):
            for j in range(N):
                for k in range(K):
                    out.append(("A", i * K + k))
                    out.append(("B", k * N + j))
                out.append(("C", i * N + j))
    elif order == "ikj":
        for i in range(M):
            for k in range(K):
                a = i * K + k
                for j in range(N):
                    out.append(("A", a))
                    out.append(("B", k * N + j))
                out.append(("C", i * N + j))
    elif order == "kij":
        for k in range(K):
            for i in range(M):
                a = i * K + k
                for j in range(N):
                    out.append(("A", a))
                    out.append(("B", k * N + j))
                out.append(("C", i * N + j))
    elif order == "tiled":
        # 经典 3 维分块: A/B/C 各取 block×block 的小块, 全部能装进缓存
        for i0 in range(0, M, block):
            for k0 in range(0, K, block):
                for j0 in range(0, N, block):
                    for i in range(i0, min(i0 + block, M)):
                        for k in range(k0, min(k0 + block, K)):
                            a = i * K + k
                            for j in range(j0, min(j0 + block, N)):
                                out.append(("A", a))
                                out.append(("B", k * N + j))
                            out.append(("C", i * N + j))
    else:
        raise ValueError(order)
    return out


def _make_cache(M, N, K, order):
    """挑一个小缓存: 装不下整片矩阵, 让不同循环顺序的差异显出来。"""
    return Cache(num_sets=8, ways=2, line_size=4)


def matmul_misses(M, N, K, order: str, block: int = 4):
    """统计某种循环顺序下, L1 缓存的缺失率。返回 (访问数, 缺失数, 命中率)。"""
    cache = _make_cache(M, N, K, order)
    for _, addr in _matmul_addresses(M, N, K, order, block):
        cache.access(addr)
    return cache.stats()


def estimate_cycles(accesses: int, misses: int) -> int:
    """粗略延迟估计: 每次访问 1 周期 + 每次缺失额外付一次 DRAM 延迟。"""
    return accesses + misses * LATENCY["DRAM"]


def compare_schedules(M=16, N=16, K=16, block=4) -> str:
    """对比不同循环顺序的缓存表现, 打印成表格。"""
    rows = []
    for order in ("ijk", "ikj", "kij", "tiled"):
        hit_cnt, miss_cnt, hit_rate = matmul_misses(M, N, K, order, block)
        acc = hit_cnt + miss_cnt
        cyc = estimate_cycles(acc, miss_cnt)
        rows.append((order, acc, miss_cnt, hit_rate, cyc))
    lines = []
    lines.append(f"{'顺序':<6}{'访问数':>8}{'缺失数':>8}{'命中率':>8}{'延迟(周期)':>12}")
    for order, acc, miss, hit, cyc in rows:
        lines.append(f"{order:<6}{acc:>8}{miss:>8}{hit:>7.1%}{cyc:>12}")
    best = min(rows, key=lambda r: r[4])
    lines.append(f"\n最快: {best[0]} (延迟 {best[4]} 周期, 命中率 {best[3]:.1%})")
    return "\n".join(lines)


def register_pressure_demo():
    """演示"同时活着多少个值"= 寄存器压力。"""
    lines = []
    lines.append("寄存器压力: 一个核里同时活着的中间值个数")
    lines.append("  朴素 conv(6 层循环): 每步读1个输入、算1次, 活值约 3~4 个")
    lines.append("  融合 conv+bias+relu: 活值约 5~6 个(acc + 3 输入 + bias)")
    lines.append("  融合 50 个算子: 每个算子各带操作数, 活值几十个 → x86 只有 16 个通用寄存器")
    lines.append("  装不下 → 溢出(spill): 值被迫写回内存, 每次用再读回来, 反而变慢")
    return "\n".join(lines)
