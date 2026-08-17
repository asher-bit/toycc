"""toyisa 链接器 —— 对应第 29 课的重定位手算。

把目标文件摆进内存并修补重定位记录:
  - .text 放在地址 0x0000, .data 紧随其后(4 字节对齐)
  - R_IMM12/R_IMM20: 写入 符号地址 + addend(绝对地址)
  - R_PC12/R_PC20:   写入 符号地址 - 本指令地址(字偏移)

手算对照(第 29 课的账): 修补值 = S + A; 位置 = 段基址 + 记录偏移。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from toyisa import isa
from toyisa.assembler import (ObjectFile, R_IMM12, R_IMM20, R_PC12, R_PC20)


@dataclass
class Image:
    """链接后的内存映像。"""
    mem: bytearray = field(default_factory=lambda: bytearray(isa.MEM_SIZE))
    entry: int = 0
    layout: dict = field(default_factory=dict)   # 符号名 -> 最终地址


def link(obj: ObjectFile, entry: str = "start") -> Image:
    img = Image()
    # 布局: .text 在 0, .data 对齐到下一个 4 字节边界
    text = obj.sections.get("text")
    data = obj.sections.get("data")
    base = {"text": 0}
    if text is not None:
        img.mem[0:len(text.data)] = text.data
        base["data"] = (len(text.data) + 3) & ~3
    else:
        base["data"] = 0
    if data is not None:
        img.mem[base["data"]:base["data"] + len(data.data)] = data.data

    # 符号最终地址 = 段基址 + 段内偏移
    for name, sym in obj.symbols.items():
        if sym.section not in base:
            raise ValueError(f"符号 {name} 属于未知段 {sym.section}")
        img.layout[name] = base[sym.section] + sym.offset

    # 修补重定位
    for rel in obj.relocs:
        if rel.symbol not in img.layout:
            raise ValueError(f"重定位引用未定义符号 {rel.symbol!r}")
        sym_addr = img.layout[rel.symbol]
        patch_at = base[rel.section] + rel.offset
        word = int.from_bytes(img.mem[patch_at:patch_at + 4], "little")
        if rel.kind in (R_IMM12, R_PC12):
            if rel.kind == R_IMM12:
                value = sym_addr + rel.addend
                if not (-(1 << 11) <= value < (1 << 11)):
                    raise ValueError(
                        f"地址 0x{value:x} 放不进 12 位立即数 (符号 {rel.symbol})")
                word = (word & ~0xFFF) | (value & 0xFFF)
            else:  # R_PC12: 字偏移 = (目标地址 - 下一条指令地址) / 4
                value = (sym_addr + rel.addend - (patch_at + 4)) // 4
                if not (-(1 << 11) <= value < (1 << 11)):
                    raise ValueError(
                        f"跳转偏移 {value} 超出 12 位 (符号 {rel.symbol})")
                word = (word & ~0xFFF) | (value & 0xFFF)
        else:
            if rel.kind == R_IMM20:
                value = sym_addr + rel.addend
                if not (-(1 << 19) <= value < (1 << 19)):
                    raise ValueError(
                        f"地址 0x{value:x} 放不进 20 位立即数 (符号 {rel.symbol})")
                word = (word & ~0xFFFFF) | (value & 0xFFFFF)
            else:  # R_PC20
                value = (sym_addr + rel.addend - (patch_at + 4)) // 4
                if not (-(1 << 19) <= value < (1 << 19)):
                    raise ValueError(
                        f"跳转偏移 {value} 超出 20 位 (符号 {rel.symbol})")
                word = (word & ~0xFFFFF) | (value & 0xFFFFF)
        img.mem[patch_at:patch_at + 4] = word.to_bytes(4, "little")

    img.entry = img.layout.get(entry, 0)
    return img


def dump_image(img: Image, words: int = 0) -> str:
    """打印内存映像的 .text 区反汇编 + 数据区。"""
    from toyisa.assembler import disasm
    text_end = max(img.layout.values()) if img.layout else 0
    n = max(words, (text_end + 3) // 4)
    ws = [int.from_bytes(img.mem[i * 4:i * 4 + 4], "little") for i in range(n)]
    return "\n".join(disasm(ws))
