"""toyisa 随机程序生成器 —— 差分测试的输入源。

对应第 27 课的覆盖率门禁: 随机生成测试程序, 统计指令覆盖。
约束(保证可终止、可比较):
  - 控制流只向前跳(beq/j/jal 目标都在后方) → 必然终止;
  - 访存只落在数据区 [DATA_BASE, DATA_BASE+64), 4 字节对齐;
  - jal/jr 成对出现: jal 跳到后方子块, 子块 jr 返回。
"""
from __future__ import annotations

import random

from toyisa import isa

DATA_BASE = 0x400        # 数据区基址(movi 是 12 位符号扩展, 必须 < 0x800)


def _rand_alu(rng):
    op = rng.choice(["add", "sub", "mul"])
    rd = rng.randint(1, 15)
    rs1, rs2 = rng.randint(0, 15), rng.randint(0, 15)
    return f"{op} r{rd}, r{rs1}, r{rs2}"


def _rand_mem(rng):
    kind = rng.choice(["ld", "st", "ld", "movi+st"])
    base_reg = rng.randint(1, 15)
    addr = DATA_BASE + rng.randint(0, 15) * 4
    lines = [f"movi r{base_reg}, {addr}"]
    if kind == "ld":
        lines.append(f"ld r{rng.randint(1, 15)}, 0(r{base_reg})")
    elif kind == "st":
        val_reg = rng.randint(1, 15)
        while val_reg == base_reg:          # 值寄存器不能覆盖基址寄存器
            val_reg = rng.randint(1, 15)
        lines.append(f"movi r{val_reg}, {rng.randint(0, 100)}")
        lines.append(f"st r{val_reg}, 0(r{base_reg})")
    else:
        lines.append(f"movi r{base_reg}, {addr}")
        lines.append(f"st r{rng.randint(1, 15)}, 0(r{base_reg})")
    return lines


def gen_program(rng: random.Random, n_ops: int = 6) -> str:
    """生成一个可终止的随机程序, 返回汇编文本。"""
    lines = [".data"]
    for i in range(16):
        lines.append(f"d{i}: .word {rng.randint(0, 99)}")
    lines.append(".text")
    lines.append("start:")

    for i in range(n_ops):
        r = rng.random()
        if r < 0.35:
            lines.append(_rand_alu(rng))
        elif r < 0.55:
            lines.extend(_rand_mem(rng))
        elif r < 0.70:
            lines.append(f"movi r{rng.randint(1, 15)}, {rng.randint(-100, 100)}")
        elif r < 0.80:
            # 向前条件跳: 跳过下一条
            lines.append(f"beq r{rng.randint(0, 15)}, r{rng.randint(0, 15)}, L{i}")
            lines.append(f"nop")
            lines.append(f"L{i}: nop")
        elif r < 0.90:
            lines.append(f"j L{i}")
            lines.append(f"nop")
            lines.append(f"L{i}: nop")
        else:
            # jal/jr 成对: 向前调用, 子块返回
            lines.append(f"jal r15, L{i}")
            lines.append(f"j L{i}b")
            lines.append(f"L{i}: nop")
            lines.append(f"jr r15")
            lines.append(f"L{i}b: nop")

    lines.append("halt")
    return "\n".join(lines)


def opcode_usage(programs: list[str]) -> dict:
    """统计一批程序文本里用到的指令名集合。"""
    used = set()
    for p in programs:
        for line in p.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith(".") or line.endswith(":"):
                continue
            name = line.replace(",", " ").split()[0]
            if name in isa.ISA:
                used.add(name)
    return {name: name in used for name in isa.ISA}
