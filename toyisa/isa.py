"""toyisa —— 迷你教学 ISA 的规范与编码。

设计目标(与第 27~29 课对应):
  - 12 条指令, 32 位定长, 手算可编码(对应第 27 课的 ISS 概念);
  - 16 个寄存器 r0..r15, r0 恒为 0(对应真实 ISA 的 zero 寄存器);
  - ld/st 要求 4 字节对齐, 违反即异常(对应第 27 课的"异常路径 + 提交点");
  - halt 结束程序, 非法指令/越界访问都产生异常。

字段布局:
  R/I/B 型: [31:24] opcode | [23:20] rd | [19:16] rs1 | [15:12] rs2 | [11:0] imm12
  J 型:     [31:24] opcode | [23:20] rd | [19:0] imm20(有符号, 字偏移)
"""
from __future__ import annotations

# ---------- 指令表 ----------
# name: (opcode, 类型, 语义说明)
# 类型: R(三寄存器) / I(立即数) / M(访存) / B(分支) / J(跳转) / S(存储) / X(无操作数)
ISA = {
    "halt": (0x00, "X", "结束程序(正常退出)"),
    "nop":  (0x01, "X", "空操作"),
    "add":  (0x02, "R", "rd = rs1 + rs2"),
    "sub":  (0x03, "R", "rd = rs1 - rs2"),
    "mul":  (0x04, "R", "rd = (rs1 * rs2) 低 32 位"),
    "movi": (0x05, "I", "rd = signext(imm12)"),
    "ld":   (0x06, "M", "rd = mem[rs1 + signext(imm12)] (4 字节对齐)"),
    "st":   (0x07, "S", "mem[rs1 + signext(imm12)] = rs2 (4 字节对齐)"),
    "beq":  (0x08, "B", "rs1 == rs2 时 pc += signext(imm12) (字偏移)"),
    "j":    (0x09, "J", "pc += signext(imm20) (字偏移)"),
    "jal":  (0x0A, "J", "rd = pc + 1; pc += signext(imm20) (字偏移)"),
    "jr":   (0x0B, "R", "pc = rs1 (rd/rs2/imm 忽略)"),
}

OPCODE_OF = {name: spec[0] for name, spec in ISA.items()}
NAME_OF = {spec[0]: name for name, spec in ISA.items()}
TYPE_OF = {name: spec[1] for name, spec in ISA.items()}

NREGS = 16
MEM_SIZE = 64 * 1024      # 64 KiB 教学内存


def signext(value: int, bits: int) -> int:
    """按 bits 位做符号扩展。"""
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)


def encode(name: str, rd: int = 0, rs1: int = 0, rs2: int = 0, imm: int = 0) -> int:
    """把一条指令编码成 32 位字。imm 对 I/B 型取低 12 位, 对 J 型取低 20 位。"""
    if name not in ISA:
        raise KeyError(f"未知指令 {name!r}")
    opcode = OPCODE_OF[name]
    if ISA[name][1] == "J":
        return ((opcode & 0xFF) << 24) | ((rd & 0xF) << 20) | (imm & 0xFFFFF)
    return ((opcode & 0xFF) << 24) | ((rd & 0xF) << 20) | ((rs1 & 0xF) << 16) \
        | ((rs2 & 0xF) << 12) | (imm & 0xFFF)


def decode(word: int) -> dict:
    """把 32 位字解码成字段字典(imm 不做符号扩展, 由使用者按类型扩展)。"""
    opcode = (word >> 24) & 0xFF
    rd = (word >> 20) & 0xF
    rs1 = (word >> 16) & 0xF
    rs2 = (word >> 12) & 0xF
    imm12 = word & 0xFFF
    imm20 = word & 0xFFFFF
    return {"opcode": opcode, "rd": rd, "rs1": rs1, "rs2": rs2,
            "imm12": imm12, "imm20": imm20}


def fmt(name: str, rd: int, rs1: int, rs2: int, imm: int) -> str:
    """把一条指令打印成汇编文本形态(用于反汇编)。"""
    kind = ISA[name][1]
    if kind in ("R", "S"):
        if name == "st":
            return f"st r{rs2}, {imm}(r{rs1})"
        return f"{name} r{rd}, r{rs1}, r{rs2}"
    if kind == "I":
        return f"{name} r{rd}, {imm}"
    if kind == "M":
        return f"{name} r{rd}, {imm}(r{rs1})"
    if kind == "B":
        return f"{name} r{rs1}, r{rs2}, {imm}"
    if kind == "J":
        return f"{name} r{rd}, {imm}"
    return name
