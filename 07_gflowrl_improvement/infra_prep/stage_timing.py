#!/usr/bin/env python3
"""Per-stage wall time of a verl training log, and what is left unaccounted.

verl already times every stage of a step (ray_trainer.fit: gen, reward, old_log_prob, ref,
adv, update_actor, save_checkpoint, update_weights, dump_rollout_generations, testing, ...)
and prints them as `timing_s/<stage>:<sec>` on the `step:N - ...` line.  The P7 training
report only copied gen / ref / update_actor / save_checkpoint, which left 129 s of the 981 s
step "unaccounted".  This reads every timing_s/* key, so the gap can be attributed.

    python3 stage_timing.py full_train.log            # table: median / mean / share of step
    python3 stage_timing.py full_train.log --csv out.csv

The full P7 log is in the HF repo:  provenance/ (or bundle/p7_bundle.tar.gz)
    hf download qzpm55555/spacetools-p7-gflowrl-cprime-8xa40 --include "provenance/*" --local-dir p7_hf
"""
import argparse, re, statistics as st, sys

# Stages that sit directly under timing_s/step in ray_trainer.fit (not nested in another one).
TOP = ["gen", "reward", "old_log_prob", "ref", "values", "adv", "update_critic", "update_actor",
       "save_checkpoint", "update_weights", "dump_rollout_generations", "testing"]
NESTED = {"gen_max"}   # inside gen

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--csv")
a = ap.parse_args()

pat = re.compile(r"step:(\d+) - (.*)")
rows = {}
for line in open(a.log, errors="replace"):
    m = pat.search(line)
    if not m:
        continue
    d = {}
    for tok in m.group(2).split(" - "):
        k, _, v = tok.partition(":")
        k = k.strip()
        if k.startswith("timing_s/"):
            try:
                d[k[len("timing_s/"):]] = float(v)
            except ValueError:
                pass
    if "step" in d:
        rows[int(m.group(1))] = d          # last occurrence of a step wins (resume re-logs)

if not rows:
    sys.exit("no `step:N - ... timing_s/step:...` lines found")

keys = sorted({k for d in rows.values() for k in d} - {"step"})
unknown = [k for k in keys if k not in TOP and k not in NESTED]
for d in rows.values():
    d["_unaccounted"] = d["step"] - sum(d.get(k, 0.0) for k in TOP + unknown)

step_med = st.median(d["step"] for d in rows.values())
print(f"{len(rows)} steps · median step {step_med:.0f} s")
print(f"{'stage':28s} {'median':>8s} {'mean':>8s} {'share':>7s}")
for k in TOP + unknown + sorted(NESTED & set(keys)) + ["_unaccounted"]:
    vals = [d[k] for d in rows.values() if k in d]
    if not vals:
        continue
    med = st.median(vals)
    tag = " (nested in gen)" if k in NESTED else (" (not in TOP list; counted)" if k in unknown else "")
    print(f"{k:28s} {med:8.1f} {st.mean(vals):8.1f} {med / step_med:7.1%}{tag}")

if a.csv:
    cols = ["step_idx", "step"] + TOP + unknown + ["_unaccounted"]
    with open(a.csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for s in sorted(rows):
            f.write(",".join([str(s)] + [f"{rows[s].get(c, ''):.2f}" if c in rows[s] else "" for c in cols[1:]]) + "\n")
    print("wrote", a.csv)
