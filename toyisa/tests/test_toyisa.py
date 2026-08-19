"""toyisa 回归测试。

覆盖: 编码手算 / 汇编器符号与重定位 / 链接 S+A / ISS 语义 /
      异常与提交点 / 双实现差分 / 覆盖率门禁。
"""
from __future__ import annotations

import random

import pytest

from toyisa import diff, gen, isa
from toyisa.assembler import assemble
from toyisa.iss import ISS, ISSException
from toyisa.loader import link
from toyisa.ref import run_ref


# ---------- 1. 编码手算 ----------

def test_encode_add_hand_computed():
    # add r1, r2, r3 → opcode 0x02 | rd=1 | rs1=2 | rs2=3
    w = isa.encode("add", rd=1, rs1=2, rs2=3)
    assert w == 0x02123000
    f = isa.decode(w)
    assert (f["opcode"], f["rd"], f["rs1"], f["rs2"]) == (0x02, 1, 2, 3)


def test_encode_movi_negative_signext():
    # movi r1, -5 → imm12 = 0xFFB
    w = isa.encode("movi", rd=1, imm=-5)
    assert w & 0xFFF == 0xFFB
    f = isa.decode(w)
    assert isa.signext(f["imm12"], 12) == -5


def test_encode_j_imm20():
    w = isa.encode("j", imm=7)
    f = isa.decode(w)
    assert f["opcode"] == 0x09 and f["imm20"] == 7


# ---------- 2. 汇编器 ----------

def test_assemble_symbols_and_relocs():
    obj = assemble("""
.data
x:  .word 5
.text
start:
    ld r2, x
    beq r1, r2, end
end:
    halt
""")
    assert "x" in obj.symbols and "start" in obj.symbols
    kinds = [r.kind for r in obj.relocs]
    assert "R_IMM12" in kinds and "R_PC12" in kinds


def test_assemble_forward_reference():
    obj = assemble("""
.text
start:
    j end
    nop
end:
    halt
""")
    assert obj.relocs[0].symbol == "end"


def test_assemble_unknown_symbol_raises():
    with pytest.raises(ValueError):
        assemble(".text\nstart:\n    j nowhere\n    halt\n")


# ---------- 3. 链接器(第 29 课 S+A) ----------

def test_link_relocation_arithmetic():
    obj = assemble("""
.data
x:  .word 9
.text
start:
    ld r2, x
    halt
""")
    img = link(obj)
    # ld 指令在 0x0000; 修补值 = S + A = x 地址 + 0, 写进 imm12
    w = int.from_bytes(img.mem[0:4], "little")
    assert (w & 0xFFF) == img.layout["x"]
    # .data 紧随 .text(8 字节)之后, 4 字节对齐 → x 在 0x0008
    assert img.layout["x"] == 8


def test_link_pcrel_branch():
    obj = assemble("""
.text
start:
    movi r1, 1
    movi r2, 1
    beq r1, r2, end
    nop
end:
    halt
""")
    img = link(obj)
    w = int.from_bytes(img.mem[8:12], "little")
    # 分支在 0x0008, 目标 0x0010, 下一条指令 0x000C → 字偏移 = 1
    assert isa.signext(w & 0xFFF, 12) == 1


# ---------- 4. ISS 语义 ----------

def test_iss_sum_program():
    obj = assemble("""
.data
x:  .word 5
y:  .word 7
z:  .word 0
.text
start:
    movi r1, 0
    ld   r2, x
    ld   r3, y
    add  r1, r2, r3
    st   r1, z
    halt
""")
    img = link(obj)
    iss = ISS(img.mem, entry=img.entry)
    iss.run()
    assert iss.regs[1] == 12
    z = int.from_bytes(iss.mem[img.layout["z"]:img.layout["z"] + 4], "little")
    assert z == 12


def test_iss_jal_jr():
    obj = assemble("""
.text
start:
    jal r15, sub
    halt
sub:
    movi r1, 42
    jr r15
""")
    img = link(obj)
    iss = ISS(img.mem, entry=img.entry)
    iss.run()
    assert iss.regs[1] == 42
    assert iss.halted


def test_iss_beq_and_loop():
    # 用 beq + j 数到 3: r1: 0,1,2,3 到 3 就停
    obj = assemble("""
.text
start:
    movi r1, 0
loop:
    movi r2, 3
    beq r1, r2, done
    movi r3, 1
    add  r1, r1, r3
    j loop
done:
    halt
""")
    img = link(obj)
    iss = ISS(img.mem, entry=img.entry)
    iss.run()
    assert iss.regs[1] == 3


# ---------- 5. 异常与提交点(第 27 课) ----------

def test_misaligned_access_commit_point():
    obj = assemble("""
.text
start:
    movi r1, 6
    ld   r2, 0(r1)
    halt
""")
    img = link(obj)
    iss = ISS(img.mem, entry=img.entry)
    with pytest.raises(ISSException) as ei:
        iss.run()
    assert ei.value.kind == "misaligned-access"
    assert iss.regs[2] == 0            # 提交点: r2 未被写脏


def test_illegal_instruction():
    img = link(assemble(".text\nstart:\n    halt\n"))
    img.mem[0:4] = (0xFF << 24).to_bytes(4, "little")
    iss = ISS(img.mem)
    with pytest.raises(ISSException) as ei:
        iss.run()
    assert ei.value.kind == "illegal-instruction"


# ---------- 6. 双实现差分 ----------

def test_diff_single_sum():
    ok, s1, s2, steps = diff.run_diff_pair("""
.data
x:  .word 5
y:  .word 7
z:  .word 0
.text
start:
    ld r2, x
    ld r3, y
    add r1, r2, r3
    st  r1, z
    halt
""")
    assert ok and s1["regs"][1] == 12 and steps == 5


def test_diff_fuzz_200():
    report = diff.fuzz(n_programs=200, seed=7)
    assert report["fails"] == 0, report["fail_details"]


def test_coverage_gate():
    report = diff.fuzz(n_programs=300, seed=42)
    assert report["covered_count"] == report["total_ops"] == 12
    assert diff.gate(report)


# ---------- 7. 生成器终止性 ----------

def test_generator_terminates():
    rng = random.Random(1)
    for _ in range(50):
        text = gen.gen_program(rng)
        obj = assemble(text)
        img = link(obj)
        iss = ISS(img.mem, entry=img.entry)
        iss.run()                       # 不抛 step-limit 即通过
        rr, rm, steps, halted = run_ref(img.mem, entry=img.entry)
        assert halted and rr == iss.regs


# ---------- 8. 周期模型(lesson27 §3 的手算) ----------

from toyisa.cycle import run_cycle   # noqa: E402


def _run_cycles(asm_text):
    img = link(assemble(asm_text))
    return run_cycle(img.mem, entry=img.entry)


def test_cycle_nop_pipeline_fill():
    # 3 条 nop + halt: 全部背靠背发射 → halt 在 c3, 流水线排空 +4
    rep = _run_cycles(".text\nstart:\n    nop\n    nop\n    nop\n    halt\n")
    assert rep.instructions == 4
    assert rep.cycles == 7                      # 3 + 4
    assert abs(rep.ipc - 4 / 7) < 1e-9
    assert rep.stalls == {"data": 0, "mem": 0, "branch": 0}


def test_cycle_dependent_add_no_forwarding():
    # movi(c0, wb4) → add 必须等 r1 就绪: 发射被推到 c4 → 停 3 拍
    rep = _run_cycles("""
.text
start:
    movi r1, 1
    add  r1, r1, r1
    halt
""")
    assert rep.issues[0][2] == 0
    assert rep.issues[1][2] == 4
    assert rep.stalls["data"] == 3
    assert rep.cycles == 9                      # halt 在 c5 + 4


def test_cycle_load_use_penalty():
    # movi(c0,wb4) → ld 等 r1: c4, wb=4+3+4=11 → add 等 r2: c11
    rep = _run_cycles("""
.text
start:
    movi r1, 0
    ld   r2, 0(r1)
    add  r3, r2, r2
    halt
""")
    assert rep.issues[1][2] == 4
    assert rep.issues[2][2] == 11
    assert rep.stalls["data"] == 3              # movi→ld
    assert rep.stalls["mem"] == 6               # ld→add: 11-5
    assert rep.cycles == 16                     # halt 在 c12 + 4


def test_cycle_taken_branch_flush():
    # beq(r1==r2==0, 跳转) → 冲刷 IF 已取的一条 → 下一条 c2, branch+1
    rep = _run_cycles("""
.text
start:
    beq r1, r2, L
L:  nop
    halt
""")
    assert rep.issues[0][2] == 0
    assert rep.issues[1][2] == 2
    assert rep.stalls["branch"] == 1
    assert rep.cycles == 7                      # halt 在 c3 + 4


def test_cycle_state_matches_iss():
    # 周期模型与 ISS 的功能语义必须一致(只差"快不快")
    text = """
.data
x:  .word 5
y:  .word 7
z:  .word 0
.text
start:
    ld r2, x
    ld r3, y
    add r1, r2, r3
    st  r1, z
    halt
"""
    img = link(assemble(text))
    iss = ISS(img.mem, entry=img.entry)
    iss.run()
    rep = run_cycle(img.mem, entry=img.entry)
    assert rep.regs == iss.regs
    assert rep.mem == bytes(iss.mem)
