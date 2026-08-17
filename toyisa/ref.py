"""toyisa 参考执行器 #2 —— 函数式实现, 与 iss.py 独立编写。

对应第 2 课 / 第 27 课的差分测试原则:
  "两份独立实现"是关键——同一个人写的两份代码会犯同样的错。
本文件故意用与 iss.py 不同的结构(函数式 + 字典内存 + 独立的解码),
语义必须与 iss.py 完全一致, 但实现路径不同。
"""
from __future__ import annotations

OP = {0x00: "halt", 0x01: "nop", 0x02: "add", 0x03: "sub", 0x04: "mul",
      0x05: "movi", 0x06: "ld", 0x07: "st", 0x08: "beq", 0x09: "j",
      0x0A: "jal", 0x0B: "jr"}

MEM_SIZE = 64 * 1024


class RefError(Exception):
    pass


def _sx(v: int, bits: int) -> int:
    sign_bit = 1 << (bits - 1)
    return (v & (sign_bit - 1)) - (v & sign_bit)


def _ld(mem: bytearray, addr: int) -> int:
    if addr % 4 or not (0 <= addr <= MEM_SIZE - 4):
        raise RefError(f"bad load addr 0x{addr:x}")
    return int.from_bytes(mem[addr:addr + 4], "little")


def _st(mem: bytearray, addr: int, val: int):
    if addr % 4 or not (0 <= addr <= MEM_SIZE - 4):
        raise RefError(f"bad store addr 0x{addr:x}")
    mem[addr:addr + 4] = (val & 0xFFFFFFFF).to_bytes(4, "little")


def run_ref(image: bytearray | bytes, entry: int = 0, max_steps: int = 100000):
    """纯函数执行: 返回 (regs, mem, steps, halted)。"""
    mem = bytearray(image)
    regs = [0] * 16
    pc = entry
    steps = 0
    while True:
        if not (0 <= pc <= MEM_SIZE - 4):
            raise RefError(f"pc out of range 0x{pc:x}")
        w = _ld(mem, pc)
        op = OP.get((w >> 24) & 0xFF)
        if op is None:
            raise RefError(f"illegal opcode 0x{(w >> 24) & 0xFF:02x} @ 0x{pc:x}")
        rd = (w >> 20) & 0xF
        rs1 = (w >> 16) & 0xF
        rs2 = (w >> 12) & 0xF
        imm12 = _sx(w & 0xFFF, 12)
        imm20 = _sx(w & 0xFFFFF, 20)

        if op == "halt":
            return regs, bytes(mem), steps, True
        if op == "nop":
            pc += 4
        elif op == "add":
            regs[rd] = (regs[rs1] + regs[rs2]) & 0xFFFFFFFF
            pc += 4
        elif op == "sub":
            regs[rd] = (regs[rs1] - regs[rs2]) & 0xFFFFFFFF
            pc += 4
        elif op == "mul":
            regs[rd] = (regs[rs1] * regs[rs2]) & 0xFFFFFFFF
            pc += 4
        elif op == "movi":
            regs[rd] = imm12 & 0xFFFFFFFF
            pc += 4
        elif op == "ld":
            regs[rd] = _ld(mem, regs[rs1] + imm12)
            pc += 4
        elif op == "st":
            _st(mem, regs[rs1] + imm12, regs[rs2])
            pc += 4
        elif op == "beq":
            pc = pc + 4 + (imm12 * 4 if regs[rs1] == regs[rs2] else 0)
        elif op == "j":
            pc = pc + 4 + imm20 * 4
        elif op == "jal":
            regs[rd] = (pc + 4) & 0xFFFFFFFF
            pc = pc + 4 + imm20 * 4
        elif op == "jr":
            pc = regs[rs1]
        steps += 1
        if steps > max_steps:
            raise RefError(f"step limit @ 0x{pc:x}")
