"""
P7 -- does the tools/training GPU split in run_rl.sh do the right thing?

The bug it fixes is silent on 2 nodes and fatal on 1: the Toolshed placement
group and trainer.n_gpus_per_node were both handed GPUS_PER_NODE, so a single
machine had the same cards booked twice.

Same discipline as p7_guard_check.py: extract the block from the real script
and run it, rather than testing a copy of it.
"""
import re, subprocess, sys

SH = __import__("os").environ.get("SPACETOOLS_RL", "../../../SpaceTools-RL") + "/examples/toolshed/run_rl.sh"
src = open(SH).read()

m = re.search(r"\n# --- Toolshed / training GPU split -+\n.*?\nfi\n", src, re.S)
if not m:
    sys.exit("FAIL: split block not found in run_rl.sh")
block = m.group(0)


def run(num_nodes, gpus_per_node, tool_gpus=None, train_gpus=None):
    env = f"NUM_NODES={num_nodes}\nGPUS_PER_NODE={gpus_per_node}\n"
    if tool_gpus is not None:
        env += f"TOOL_GPUS={tool_gpus}\n"
    if train_gpus is not None:
        env += f"TRAIN_GPUS={train_gpus}\n"
    script = env + block + '\necho "$TOOL_GPUS $TRAIN_GPUS"\n'
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    if p.returncode != 0:
        return None, p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "exit"
    return tuple(int(x) for x in p.stdout.split()[-2:]), None


cases = [
    # label,                     nodes gpn  TOOL TRAIN   expect
    ("2 节点 8 卡(论文原配置,行为必须不变)", 2, 8, None, None, (8, 8)),
    ("1 节点 8 卡(旧代码在这里死锁)",        1, 8, None, None, (2, 6)),
    ("1 节点 6 卡(推荐配置)",                1, 6, None, None, (2, 4)),
    ("1 节点 4 卡(最小配置)",                1, 4, None, None, (2, 2)),
    ("1 节点 6 卡,显式 3/3",                 1, 6, 3, 3,       (3, 3)),
    ("1 节点 4 卡,要 3+3 > 4 -> 必须报错",   1, 4, 3, 3,       None),
    ("1 节点 2 卡,工具吃光 -> 必须报错",     1, 2, 2, 0,       None),
]

ok = True
for label, n, g, t, tr, expect in cases:
    got, err = run(n, g, t, tr)
    good = (got == expect)
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] {label}")
    print(f"         -> {'ERROR: ' + err if got is None else f'tools {got[0]} / training {got[1]}'}")

print("\n" + "=" * 62)
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
