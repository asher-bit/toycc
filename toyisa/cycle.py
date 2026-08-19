"""toyisa 周期模拟器 —— 五级顺序流水线的逐周期模型。

对应 lesson27 第 3 节(周期模型)与扩展阅读 B(悲观校准)。
与 iss.py 的分工: ISS 回答"对不对", 本模块回答"快不快"。

模型约定(所有数字手算可验):
  五级流水: IF(取指) → ID(译码/读寄存器) → EX(执行) → MEM(访存) → WB(写回)
  单发射: 每周期最多发射 1 条指令(结构冒险由 prev_issue+1 隐式处理)
  无前向(no forwarding): 源寄存器必须等生产者 WB 完成才能被 ID 读到
    → 写回周期 wb, 消费者发射周期 >= wb; 背靠背相关 ALU 停 3 拍
  访存延迟 MEM_LATENCY=4: ld 的 WB = issue + 3 + 4 = issue + 7
    → load-use 比 ALU 相关多停 3 拍, 分类计入 mem_stall
  分支在 ID 解析(需要操作数就绪), 跳转发生 → 冲刷 IF 已取的 1 条
    → 下一条发射 >= 本分支 issue + 2, 分类计入 branch_stall
  功能语义与 iss.py 完全一致(功能优先: 分支方向/访存地址用"当前提交值"算)

输出: (总周期, 指令数, IPC, stall 分类 {data/mem/branch}, 每条指令的发射周期)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from toyisa import isa

MEM_LATENCY = 4        # MEM 段占用的周期数(教学值, 对应 lesson15 的延迟账)


@dataclass
class CycleReport:
    cycles: int = 0
    instructions: int = 0
    ipc: float = 0.0
    stalls: dict = field(default_factory=lambda: {"data": 0, "mem": 0,
                                                  "branch": 0})
    issues: list = field(default_factory=list)   # (pc, 指令文本, 发射周期)
    regs: list = field(default_factory=list)     # 终态寄存器(与 ISS 对齐)
    mem: bytes = b""                             # 终态内存(与 ISS 对齐)


def _read_word(mem: bytearray, addr: int) -> int:
    return int.from_bytes(mem[addr:addr + 4], "little")


def _write_word(mem: bytearray, addr: int, value: int):
    mem[addr:addr + 4] = (value & 0xFFFFFFFF).to_bytes(4, "little")


def _writes_rd(name: str) -> bool:
    return name in ("add", "sub", "mul", "movi", "ld", "jal")


def _sources(name: str, f: dict) -> list:
    """该指令在 ID 要读的源寄存器(统一按"ID 前就绪"的保守规则)。"""
    if name in ("add", "sub", "mul", "beq"):
        return [f["rs1"], f["rs2"]]
    if name in ("ld", "jr"):
        return [f["rs1"]]
    if name == "st":
        return [f["rs1"], f["rs2"]]           # 地址 + 数据
    return []


def run_cycle(image: bytearray | bytes, entry: int = 0,
              mem_latency: int = MEM_LATENCY, max_steps: int = 100000):
    """逐周期执行: 返回 CycleReport。功能状态与 ISS 同步推进。"""
    mem = bytearray(image)
    regs = [0] * isa.NREGS
    pc = entry

    # 计分板: 每个寄存器最后被写的"就绪周期"与生产者种类
    ready = [0] * isa.NREGS
    producer = [""] * isa.NREGS

    rep = CycleReport()
    prev_issue = -1

    for _ in range(max_steps):
        word = _read_word(mem, pc)
        f = isa.decode(word)
        name = isa.NAME_OF.get(f["opcode"])
        if name is None:
            raise ValueError(f"illegal opcode 0x{f['opcode']:02x} @ 0x{pc:04x}")

        # ---- ① 计算发射周期 ----
        t = prev_issue + 1                       # 结构冒险(单发射)

        for src in _sources(name, f):
            if ready[src] > t:                  # 数据冒险: 等写回
                stall = ready[src] - t
                t = ready[src]
                rep.stalls["mem" if producer[src] == "ld" else "data"] += stall

        # ---- ② 分支: 在 ID 解析(此刻源寄存器已就绪) ----
        taken = False
        if name in ("j", "jal", "jr"):
            taken = True
        elif name == "beq":
            taken = regs[f["rs1"]] == regs[f["rs2"]]

        issue = t
        if taken:
            # 冲刷 IF 已取的一条 → 惩罚落在"下一条", 不是本分支
            prev_issue = issue + 1
            rep.stalls["branch"] += 1
        else:
            prev_issue = issue

        # ---- ③ 更新计分板(本条将写哪个寄存器) ----
        if _writes_rd(name):
            wb = issue + 3 + mem_latency if name == "ld" else issue + 4
            ready[f["rd"]] = wb
            producer[f["rd"]] = name

        # ---- ④ 功能执行(与 ISS 语义一致, 推进提交状态) ----
        self_pc = pc
        if name == "halt":
            rep.issues.append((pc, "halt", issue))
            rep.instructions += 1
            rep.cycles = issue + 4               # halt 也走完流水线
            rep.ipc = rep.instructions / rep.cycles
            rep.regs = list(regs)
            rep.mem = bytes(mem)
            return rep
        if name == "nop":
            pass
        elif name in ("add", "sub", "mul"):
            a, b = regs[f["rs1"]], regs[f["rs2"]]
            v = (a + b if name == "add" else a - b if name == "sub"
                 else a * b)
            regs[f["rd"]] = v & 0xFFFFFFFF
        elif name == "movi":
            regs[f["rd"]] = isa.signext(f["imm12"], 12) & 0xFFFFFFFF
        elif name in ("ld", "st"):
            addr = regs[f["rs1"]] + isa.signext(f["imm12"], 12)
            if name == "ld":
                regs[f["rd"]] = _read_word(mem, addr)
            else:
                _write_word(mem, addr, regs[f["rs2"]])
        elif name == "beq":
            pass                                  # 分支本身不改寄存器
        elif name == "j":
            pass
        elif name == "jal":
            regs[f["rd"]] = (self_pc + 4) & 0xFFFFFFFF
        elif name == "jr":
            pass

        # ---- ⑤ 推进 PC(用提交状态, 与 ISS 相同) ----
        if name == "beq":
            pc = pc + 4 + (isa.signext(f["imm12"], 12) * 4
                           if regs[f["rs1"]] == regs[f["rs2"]] else 0)
        elif name == "j":
            pc = pc + 4 + isa.signext(f["imm20"], 20) * 4
        elif name == "jal":
            pc = pc + 4 + isa.signext(f["imm20"], 20) * 4
        elif name == "jr":
            pc = regs[f["rs1"]]
        else:
            pc = pc + 4

        rep.issues.append((self_pc,
                           isa.fmt(name, f["rd"], f["rs1"], f["rs2"],
                                   isa.signext(f["imm12"], 12)), issue))
        rep.instructions += 1
    raise ValueError(f"step limit exceeded @ 0x{pc:04x}")


def format_report(rep: CycleReport) -> str:
    lines = [
        f"周期模型: {rep.instructions} 条指令, {rep.cycles} 周期, "
        f"IPC = {rep.ipc:.2f}",
        f"  stall: data={rep.stalls['data']} mem={rep.stalls['mem']} "
        f"branch={rep.stalls['branch']}",
    ]
    for pc, text, issue in rep.issues:
        lines.append(f"  c{issue:3d} @0x{pc:04x}: {text}")
    return "\n".join(lines)
