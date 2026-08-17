"""toyisa 两遍汇编器 + 目标文件格式。

对应第 29 课: 汇编 → 目标文件(段 + 符号表 + 重定位记录)。
两遍:
  第 1 遍: 布局(每段多大)、收集符号(名字 → 段内偏移)
  第 2 遍: 编码指令、为"引用符号的立即数"生成重定位记录

语法:
  .data / .text 切换段;  label: 定义符号;  .word 4字节常量
  add r1, r2, r3 | movi r1, -5 | ld r2, x | ld r2, 4(r1)
  st r2, x | beq r1, r2, label | j label | jal r15, label | jr r15
  halt | nop
"""
from __future__ import annotations

from dataclasses import dataclass, field

from toyisa import isa

# 重定位类型
R_IMM12 = "R_IMM12"     # 把 (符号地址 + addend) 的绝对值写进 imm12
R_IMM20 = "R_IMM20"     # 把 (符号地址 + addend) 的绝对值写进 imm20
R_PC12 = "R_PC12"       # 把 (符号地址 - 本指令地址) 的字偏移写进 imm12
R_PC20 = "R_PC20"       # 把 (符号地址 - 本指令地址) 的字偏移写进 imm20


@dataclass
class Section:
    name: str
    data: bytearray = field(default_factory=bytearray)


@dataclass
class Symbol:
    name: str
    section: str
    offset: int            # 段内字节偏移


@dataclass
class Relocation:
    section: str
    offset: int            # 段内字节偏移(要修补的指令所在位置)
    kind: str
    symbol: str
    addend: int = 0


@dataclass
class ObjectFile:
    sections: dict = field(default_factory=dict)
    symbols: dict = field(default_factory=dict)
    relocs: list = field(default_factory=list)

    def section(self, name: str) -> Section:
        if name not in self.sections:
            self.sections[name] = Section(name)
        return self.sections[name]


REG_RE = None  # 运行时填充


def _parse_reg(tok: str) -> int:
    if not (tok.startswith("r") and tok[1:].isdigit()):
        raise ValueError(f"非法寄存器 {tok!r}")
    n = int(tok[1:])
    if not (0 <= n < isa.NREGS):
        raise ValueError(f"寄存器号越界 {tok!r} (0..{isa.NREGS - 1})")
    return n


def _parse_imm(tok: str) -> int:
    v = int(tok, 0)          # 支持十进制与 0x 十六进制
    if not (-(1 << 31) <= v < (1 << 31)):
        raise ValueError(f"立即数越界 {tok!r}")
    return v


def assemble(text: str) -> ObjectFile:
    obj = ObjectFile()
    cur_section = None
    pending = []             # (行号, 段, 段内偏移, 指令名, 操作数)
    layout_pos = {}          # 段名 -> 当前字节偏移

    def get_section(name):
        if name not in ("text", "data"):
            raise ValueError(f"未知段 {name!r} (只支持 .text/.data)")
        return obj.section(name)

    # ---- 第 1 遍: 布局 + 符号表 ----
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        # 处理行首的 label: 前缀(支持 label: 与指令/伪指令同行)
        while ":" in line:
            head, rest = line.split(":", 1)
            head = head.strip()
            if head:
                if cur_section is None:
                    raise ValueError(f"行 {lineno}: 符号 {head!r} 必须在段内")
                if head in obj.symbols:
                    raise ValueError(f"行 {lineno}: 重复符号 {head!r}")
                obj.symbols[head] = Symbol(head, cur_section,
                                           layout_pos.get(cur_section, 0))
            line = rest.strip()
            if not line:
                break
        if not line:
            continue
        if line.startswith("."):
            parts = line.split()
            if parts[0] in (".text", ".data"):
                cur_section = parts[0][1:]
                layout_pos.setdefault(cur_section, 0)
                continue
            if parts[0] == ".word":
                if cur_section is None:
                    raise ValueError(f"行 {lineno}: .word 必须在段内")
                get_section(cur_section).data.extend(b"\x00" * 4)
                # 值在第 2 遍填
                pending.append((lineno, cur_section, layout_pos[cur_section],
                                ".word", parts[1:]))
                layout_pos[cur_section] += 4
                continue
            raise ValueError(f"行 {lineno}: 未知伪指令 {parts[0]!r}")
        # 指令行
        if cur_section is None:
            raise ValueError(f"行 {lineno}: 指令必须在段内")
        toks = line.replace(",", " ").split()
        name = toks[0]
        if name not in isa.ISA:
            raise ValueError(f"行 {lineno}: 未知指令 {name!r}")
        pos = layout_pos.get(cur_section, 0)
        pending.append((lineno, cur_section, pos, name, toks[1:]))
        layout_pos[cur_section] = pos + 4

    # ---- 第 2 遍: 编码 + 重定位 ----
    for lineno, section, pos, name, ops in pending:
        sec = get_section(section)
        if name == ".word":
            if len(ops) != 1:
                raise ValueError(f"行 {lineno}: .word 需要一个值")
            v = _parse_imm(ops[0])
            sec.data[pos:pos + 4] = (v & 0xFFFFFFFF).to_bytes(4, "little")
            continue
        word = _encode_one(obj, section, pos, name, ops, lineno)
        sec.data[pos:pos + 4] = word.to_bytes(4, "little")
    return obj


def _encode_one(obj: ObjectFile, section: str, pos: int, name: str,
                ops: list, lineno: int) -> int:
    kind = isa.ISA[name][1]

    def sym_ref(tok: str):
        """符号引用: 直接返回 (name, addend), 或 (None, 立即数)。"""
        try:
            return None, _parse_imm(tok)
        except ValueError:
            if tok not in obj.symbols:
                raise ValueError(f"行 {lineno}: 未定义符号 {tok!r}")
            return tok, 0

    if kind == "X":
        return isa.encode(name)
    if kind == "R":
        if name == "jr":                    # jr rs1(单操作数)
            return isa.encode(name, rs1=_parse_reg(ops[0]))
        rd, rs1, rs2 = (_parse_reg(t) for t in ops)
        return isa.encode(name, rd, rs1, rs2)
    if kind == "I":
        if len(ops) != 2:
            raise ValueError(f"行 {lineno}: {name} 需要 2 个操作数")
        rd = _parse_reg(ops[0])
        sym, addend = sym_ref(ops[1])
        imm = addend if sym is None else 0
        if sym is not None:
            obj.relocs.append(Relocation(section, pos, R_IMM12, sym, addend))
        return isa.encode(name, rd=rd, imm=imm)
    if kind == "M":
        # ld rd, imm(rs1) 或 ld rd, symbol
        rd = _parse_reg(ops[0])
        tok = ops[1]
        if "(" in tok:
            imm_s, rs_s = tok.rstrip(")").split("(")
            return isa.encode(name, rd=rd, rs1=_parse_reg(rs_s),
                              imm=_parse_imm(imm_s))
        sym, addend = sym_ref(tok)
        if sym is None:
            raise ValueError(f"行 {lineno}: ld 的目标必须是 imm(rx) 或符号")
        obj.relocs.append(Relocation(section, pos, R_IMM12, sym, addend))
        return isa.encode(name, rd=rd, imm=0)
    if kind == "S":
        # st rs2, imm(rs1) 或 st rs2, symbol
        rs2 = _parse_reg(ops[0])
        tok = ops[1]
        if "(" in tok:
            imm_s, rs_s = tok.rstrip(")").split("(")
            return isa.encode(name, rs1=_parse_reg(rs_s), rs2=rs2,
                              imm=_parse_imm(imm_s))
        sym, addend = sym_ref(tok)
        if sym is None:
            raise ValueError(f"行 {lineno}: st 的目标必须是 imm(rx) 或符号")
        obj.relocs.append(Relocation(section, pos, R_IMM12, sym, addend))
        return isa.encode(name, rs2=rs2, imm=0)
    if kind == "B":
        rs1, rs2 = _parse_reg(ops[0]), _parse_reg(ops[1])
        sym, addend = sym_ref(ops[2])
        if sym is not None:
            obj.relocs.append(Relocation(section, pos, R_PC12, sym, addend))
            imm = 0
        else:
            imm = addend
        return isa.encode(name, rs1=rs1, rs2=rs2, imm=imm)
    if kind == "J":
        if name == "j":
            rd, target = 0, ops[0]
        else:  # jal rd, target
            rd, target = _parse_reg(ops[0]), ops[1]
        sym, addend = sym_ref(target)
        if sym is not None:
            obj.relocs.append(Relocation(section, pos, R_PC20, sym, addend))
            imm = 0
        else:
            imm = addend
        return isa.encode(name, rd=rd, imm=imm)
    raise ValueError(f"行 {lineno}: 无法编码 {name}")


def disasm(words: list, start_addr: int = 0) -> list[str]:
    """把一段字流反汇编成文本(立即数按类型符号扩展显示)。"""
    out = []
    for i, w in enumerate(words):
        f = isa.decode(w)
        name = isa.NAME_OF.get(f["opcode"], "??")
        kind = isa.ISA.get(name, ("", "X"))[1]
        if kind in ("I", "M", "S"):
            imm = isa.signext(f["imm12"], 12)
        elif kind == "B":
            imm = isa.signext(f["imm12"], 12)
        elif kind == "J":
            imm = isa.signext(f["imm20"], 20)
        else:
            imm = f["imm12"]
        out.append(f"0x{start_addr + i * 4:04x}: "
                   f"{isa.fmt(name, f['rd'], f['rs1'], f['rs2'], imm)}")
    return out
