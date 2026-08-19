"""toyisa 一键演示: 汇编 → 链接 → 双 ISS 执行 → 差分 → 覆盖率门禁。

用法: python -m toyisa.demo
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):   # Windows GBK 控制台 → UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from toyisa import diff
from toyisa.assembler import assemble, disasm
from toyisa.iss import ISS, ISSException
from toyisa.loader import dump_image, link
from toyisa.ref import run_ref

SUM_PROG = """
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
"""


def demo_sum():
    print("== 1. 汇编 + 链接: 求 x + y 写回 z ==")
    obj = assemble(SUM_PROG)
    img = link(obj)
    print("符号表:", {k: hex(v) for k, v in img.layout.items()})
    print("重定位数:", len(obj.relocs))
    print("链接后映像:")
    print(dump_image(img))
    iss = ISS(img.mem, entry=img.entry)
    iss.run()
    print("ISS 执行轨迹:")
    for t in iss.trace:
        print("  ", t)
    print("终态寄存器: r1 =", iss.regs[1], " r2 =", iss.regs[2],
          " r3 =", iss.regs[3])
    z = int.from_bytes(iss.mem[img.layout["z"]:img.layout["z"] + 4], "little")
    print(f"内存 z = {z} (期望 12)")
    assert z == 12 and iss.regs[1] == 12

    # 参考执行器交叉验证
    rr, rm, steps, halted = run_ref(img.mem, entry=img.entry)
    assert rr == iss.regs and rm == bytes(iss.mem)
    print(f"参考执行器一致 ✓ (共 {steps} 步)")


def demo_exceptions():
    print("\n== 2. 异常路径 + 提交点 ==")
    # 2.1 未对齐访问: ld 前 r1 存了 6, 6 % 4 != 0
    bad = """
.text
start:
    movi r1, 6
    ld   r2, 0(r1)
    halt
"""
    obj = assemble(bad)
    img = link(obj)
    iss = ISS(img.mem, entry=img.entry)
    try:
        iss.run()
    except ISSException as e:
        print(f"  未对齐访问 → 异常: {e}")
        # 提交点: r2 没被写脏
        print(f"  提交点检查: 异常后 r2 = {iss.regs[2]} (仍为 0, 未被写脏) ✓")
        assert iss.regs[2] == 0

    # 2.2 非法指令: 手工造一个 opcode=0xFF 的字
    from toyisa import isa as isa_mod
    img2 = link(assemble(".text\nstart:\n    halt\n"))
    img2.mem[0:4] = (0xFF << 24).to_bytes(4, "little")
    iss2 = ISS(img2.mem)
    try:
        iss2.run()
    except ISSException as e:
        print(f"  非法指令 → 异常: {e.kind} (opcode=0xff)")
        assert e.kind == "illegal-instruction"


def demo_diff():
    print("\n== 3. 差分测试 + 覆盖率门禁 ==")
    report = diff.fuzz(n_programs=300, seed=42)
    print(diff.format_report(report))
    assert diff.gate(report), "覆盖率门禁不通过"


def demo_cycle():
    print("\n== 4. 周期模型: 同一程序, ISS 回答对不对, cycle 回答快不快 ==")
    from toyisa.assembler import assemble
    from toyisa.cycle import format_report, run_cycle
    from toyisa.iss import ISS
    from toyisa.loader import link

    obj = assemble(SUM_PROG)
    img = link(obj)

    iss = ISS(img.mem, entry=img.entry)
    iss.run()
    rep = run_cycle(img.mem, entry=img.entry)
    print(format_report(rep))
    print(f"  功能一致性: 周期模型终态 == ISS 终态 → "
          f"{rep.regs == iss.regs and rep.mem == bytes(iss.mem)}")
    assert rep.regs == iss.regs and rep.mem == bytes(iss.mem)

    # 对比: 去掉访存延迟后同一程序的周期数(演示"参数化"如何改变结论)
    rep0 = run_cycle(img.mem, entry=img.entry, mem_latency=1)
    print(f"  参数化实验: mem_latency 4→1, 周期 {rep.cycles} → {rep0.cycles}"
          f" (性能模型 = 参数化的产物, 换参数换结论)")
    print("  对照 lesson27 第 3 节与 sim 专题: 手算 stall 分类与悲观校准。")


def main():
    demo_sum()
    demo_exceptions()
    demo_diff()
    demo_cycle()
    print("\n全部通过 ✓ —— 对照 course/lesson27.md 与 course/lesson29.md 阅读。")


if __name__ == "__main__":
    main()
