"""toyisa ISS #1 —— 类式实现, 带异常与提交点语义。

对应第 27 课的三个要点:
  1. 只回答"对不对", 不管时序;
  2. 异常路径必须模拟(非法指令 / 未对齐访存 / 越界);
  3. 提交点: 先算完、检查通过, 才写寄存器/内存——异常时状态不脏。
"""
from __future__ import annotations

from toyisa import isa


class ISSException(Exception):
    def __init__(self, kind: str, pc: int, detail: str = ""):
        self.kind = kind
        self.pc = pc
        self.detail = detail
        super().__init__(f"{kind} @ pc=0x{pc:04x} {detail}")


class ISS:
    def __init__(self, mem: bytearray | bytes, entry: int = 0):
        self.mem = bytearray(mem)
        self.regs = [0] * isa.NREGS
        self.pc = entry
        self.trace: list[str] = []
        self.halted = False

    def read_word(self, addr: int) -> int:
        if not (0 <= addr <= isa.MEM_SIZE - 4):
            raise ISSException("out-of-range", self.pc, f"addr=0x{addr:x}")
        return int.from_bytes(self.mem[addr:addr + 4], "little")

    def write_word(self, addr: int, value: int):
        if not (0 <= addr <= isa.MEM_SIZE - 4):
            raise ISSException("out-of-range", self.pc, f"addr=0x{addr:x}")
        self.mem[addr:addr + 4] = (value & 0xFFFFFFFF).to_bytes(4, "little")

    def step(self):
        if self.halted:
            return
        word = self.read_word(self.pc)
        f = isa.decode(word)
        name = isa.NAME_OF.get(f["opcode"])
        if name is None:
            raise ISSException("illegal-instruction", self.pc,
                              f"opcode=0x{f['opcode']:02x}")
        kind = isa.ISA[name][1]
        if kind == "J":
            imm = isa.signext(f["imm20"], 20)
        elif kind in ("I", "M", "S", "B"):
            imm = isa.signext(f["imm12"], 12)
        else:
            imm = f["imm12"]
        self.trace.append(
            f"0x{self.pc:04x}: "
            f"{isa.fmt(name, f['rd'], f['rs1'], f['rs2'], imm)}")
        self._execute(name, f)

    def _execute(self, name: str, f: dict):
        pc = self.pc
        kind = isa.ISA[name][1]
        r = self.regs

        if name == "halt":
            self.halted = True
            return
        if name == "nop":
            self.pc = pc + 4
            return
        if name in ("add", "sub", "mul"):
            a = r[f["rs1"]]
            b = r[f["rs2"]]
            if name == "add":
                out = a + b
            elif name == "sub":
                out = a - b
            else:
                out = a * b
            # 提交点: 算术无异常, 直接写回
            r[f["rd"]] = out & 0xFFFFFFFF
            self.pc = pc + 4
            return
        if name == "movi":
            r[f["rd"]] = isa.signext(f["imm12"], 12) & 0xFFFFFFFF
            self.pc = pc + 4
            return
        if name in ("ld", "st"):
            addr = r[f["rs1"]] + isa.signext(f["imm12"], 12)
            # 先全部检查, 通过才动手 —— 提交点语义
            if addr % 4 != 0:
                raise ISSException("misaligned-access", pc, f"addr=0x{addr:x}")
            if not (0 <= addr <= isa.MEM_SIZE - 4):
                raise ISSException("out-of-range", pc, f"addr=0x{addr:x}")
            if name == "ld":
                r[f["rd"]] = self.read_word(addr)
            else:
                self.write_word(addr, r[f["rs2"]])
            self.pc = pc + 4
            return
        if name == "beq":
            if r[f["rs1"]] == r[f["rs2"]]:
                self.pc = pc + 4 + isa.signext(f["imm12"], 12) * 4
            else:
                self.pc = pc + 4
            return
        if name == "j":
            self.pc = pc + 4 + isa.signext(f["imm20"], 20) * 4
            return
        if name == "jal":
            r[f["rd"]] = (pc + 4) & 0xFFFFFFFF
            self.pc = pc + 4 + isa.signext(f["imm20"], 20) * 4
            return
        if name == "jr":
            self.pc = r[f["rs1"]]
            return
        raise ISSException("illegal-instruction", pc, f"opcode=0x{f['opcode']:02x}")

    def run(self, max_steps: int = 100000):
        steps = 0
        while not self.halted:
            self.step()
            steps += 1
            if steps > max_steps:
                raise ISSException("step-limit", self.pc, f"max={max_steps}")
        return steps

    def state(self):
        return {"regs": list(self.regs), "mem": bytes(self.mem), "pc": self.pc}
