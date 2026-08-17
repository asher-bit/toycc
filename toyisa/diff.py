"""toyisa 差分测试 + 覆盖率门禁 —— 对应第 27 课的方法论。

流程: 随机生成程序 → 汇编 → 链接 → 两个独立 ISS 各跑一遍
     → 比较寄存器/内存终态 → 统计指令覆盖率。
门禁: 全部程序两实现一致, 且 12 条指令 100% 覆盖。
"""
from __future__ import annotations

import random

from toyisa import gen, isa
from toyisa.assembler import assemble
from toyisa.iss import ISS
from toyisa.loader import link
from toyisa.ref import run_ref


def run_diff_pair(asm_text: str):
    """对一份程序跑差分: 返回 (一致?, ISS终态, ref终态, 步骤数)。"""
    obj = assemble(asm_text)
    img = link(obj)
    iss = ISS(img.mem, entry=img.entry)
    steps = iss.run()
    r_regs, r_mem, r_steps, halted = run_ref(img.mem, entry=img.entry)
    assert halted, "参考执行器未正常结束"
    mem_eq = bytes(iss.mem) == r_mem
    regs_eq = iss.regs == r_regs
    return (mem_eq and regs_eq), iss.state(), \
        {"regs": r_regs, "mem": r_mem}, steps


def fuzz(n_programs: int = 300, n_ops: int = 6, seed: int = 42):
    """跑 n 份随机程序的差分测试, 返回报告 dict。"""
    rng = random.Random(seed)
    texts = [gen.gen_program(rng, n_ops) for _ in range(n_programs)]
    fails = []
    for i, t in enumerate(texts):
        ok, s1, s2, steps = run_diff_pair(t)
        if not ok:
            fails.append((i, t, s1, s2))
    usage = gen.opcode_usage(texts)
    covered = sum(1 for v in usage.values() if v)
    report = {
        "programs": n_programs,
        "fails": len(fails),
        "coverage": {name: v for name, v in usage.items()},
        "covered_count": covered,
        "total_ops": len(isa.ISA),
        "fail_details": fails[:3],
    }
    return report


def gate(report: dict) -> bool:
    """覆盖率门禁: 无失败 + 全指令覆盖。"""
    return report["fails"] == 0 and report["covered_count"] == report["total_ops"]


def format_report(report: dict) -> str:
    lines = [
        f"差分测试: {report['programs']} 份随机程序",
        f"  不一致: {report['fails']}",
        f"  指令覆盖: {report['covered_count']}/{report['total_ops']}"
        f" ({report['covered_count'] / report['total_ops']:.0%})",
    ]
    for name, used in report["coverage"].items():
        lines.append(f"    {'[x]' if used else '[ ]'} {name:<6} {isa.ISA[name][2]}")
    lines.append(f"  门禁: {'通过' if gate(report) else '不通过'}")
    return "\n".join(lines)
