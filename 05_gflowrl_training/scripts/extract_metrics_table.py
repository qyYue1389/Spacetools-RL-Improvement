#!/usr/bin/env python3
"""把训练日志里每一步的关键指标抽成一张表 + CSV。
按需运行,不是守护进程:每次重新解析整个日志,幂等,不会掉队也没有进程要看管。
    python3 /root/p7tab.py [日志路径]
"""
import re, sys
LOG = sys.argv[1] if len(sys.argv) > 1 else "/root/logs/full_train.log"
CSV = "/root/logs/p7_metrics.csv"
COLS = [("step", "step", "{}"),
        ("timing_s/step", "wall_s", "{:.0f}"),
        ("reward/degenerate_group_frac", "degen", "{:.3f}"),
        ("gflowrl/clip_saturation", "sat", "{:.3f}"),
        ("gflowrl/sat_at_beta_2", "sat_b2", "{:.3f}"),
        ("gflowrl/sat_at_beta_4", "sat_b4", "{:.3f}"),
        ("gflowrl/sat_at_beta_16", "sat_b16", "{:.3f}"),
        ("gflowrl/reward_term_abs_mean", "reward", "{:.3f}"),
        ("gflowrl/drift_term_abs_mean", "drift", "{:.3f}"),
        ("rew_over_drift", "rew/drift", "{:.2f}"),
        ("gflowrl/sign_agree_nondegenerate", "sign_ok", "{:.3f}"),
        ("gflowrl/d_seq_abs_mean", "d_seq", "{:.3f}"),
        ("actor/grad_norm", "grad_norm", "{:.0f}"),
        ("critic/score/mean", "score", "{:.3f}"),
        ("perf/max_memory_allocated_gb", "mem_gb", "{:.1f}")]

rows = {}
pat = re.compile(r"step:(\d+) - (.*)")
for line in open(LOG, errors="replace"):
    m = pat.search(line)
    if not m:
        continue
    d = {}
    for tok in m.group(2).split(" - "):
        k, _, v = tok.partition(":")
        try:
            d[k.strip()] = float(v)
        except ValueError:
            pass
    d["step"] = float(m.group(1))
    if d.get("gflowrl/drift_term_abs_mean"):
        d["rew_over_drift"] = d.get("gflowrl/reward_term_abs_mean", 0.0) / d["gflowrl/drift_term_abs_mean"]
    rows[int(m.group(1))] = d

if not rows:
    print("还没有完成的步。日志:", LOG)
    raise SystemExit(0)

hdr = [h for _, h, _ in COLS]
w = [max(len(h), 7) for h in hdr]
print("  ".join(h.rjust(x) for h, x in zip(hdr, w)))
with open(CSV, "w") as f:
    f.write(",".join(hdr) + "\n")
    for s in sorted(rows):
        d = rows[s]
        cells, raw = [], []
        for key, _, fmt in COLS:
            v = d.get(key)
            cells.append(("-" if v is None else fmt.format(v)))
            raw.append("" if v is None else repr(v))
        print("  ".join(c.rjust(x) for c, x in zip(cells, w)))
        f.write(",".join(raw) + "\n")
print(f"\n{len(rows)} 步 -> {CSV}")
print("盯这三样:sat 贴 1.0 = 幅度信息没了 · rew/drift 掉到 <1 = 奖励被漂移淹没 · mem_gb 逼近 46 = 要 OOM")
