#!/usr/bin/env python3
"""Summarise the per-call tool timing written with TOOL_TIMING_DIR (patched tool_agent_loop.py).

    python3 tool_timing_summary.py $TOOL_TIMING_DIR [--gap 30]

Per tool: calls, error share, latency, and its split
    exec_wait  queued for a thread of the agent-loop worker's default executor
    remote     inside the Toolshed call (router + tool-actor queue + compute)
    post       on the event loop after the call (ray.put of images/variables, ...)
plus in-flight calls of that tool at submit.  How to read it:
    remote p50 >> remote min      -> the tool's actors are queueing: add replicas / GPU share
    remote p50 ~= remote min      -> compute-bound: batching or a faster GPU, not more replicas
    exec_wait large, calls >= executor threads -> the thread pool in the agent loop is the cap
Calls are grouped into rollout phases (one per training step) by gaps longer than --gap seconds.
"""
import argparse, glob, json, os, statistics as st
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--gap", type=float, default=30.0)
a = ap.parse_args()

recs = [json.loads(l) for f in glob.glob(os.path.join(a.dir, "*.jsonl")) for l in open(f) if l.strip()]
if not recs:
    raise SystemExit("no records")
recs.sort(key=lambda r: r["t_submit"])

def q(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")

print(f"{len(recs)} calls from {len({r['pid'] for r in recs})} agent-loop processes, "
      f"executor threads per process: {sorted({r['executor_workers'] for r in recs}, key=str)}")
hdr = f"{'tool':28s} {'calls':>6s} {'err%':>5s} | {'lat p50':>7s} {'p90':>6s} {'max':>6s} | {'wait p50':>8s} {'p90':>6s} | {'remote min':>10s} {'p50':>6s} {'p90':>6s} | {'post p50':>8s} | {'inflight p50':>12s} {'max':>4s} | {'lat share':>9s}"
print(hdr)
tot = sum(r["latency"] for r in recs)
by = defaultdict(list)
for r in recs:
    by[r["tool"]].append(r)
for t, rs in sorted(by.items(), key=lambda kv: -sum(r["latency"] for r in kv[1])):
    lat = [r["latency"] for r in rs]
    wait = [r["exec_wait"] for r in rs]
    rem = [r["remote"] for r in rs]
    post = [r["latency"] - r["exec_wait"] - r["remote"] for r in rs if r["exec_wait"] is not None and r["remote"] is not None]
    inf = [r["inflight_tool"] for r in rs]
    print(f"{t:28s} {len(rs):6d} {100 * sum(not r['ok'] for r in rs) / len(rs):5.1f} | "
          f"{q(lat, .5):7.2f} {q(lat, .9):6.2f} {max(lat):6.2f} | {q(wait, .5):8.2f} {q(wait, .9):6.2f} | "
          f"{q(rem, 0):10.2f} {q(rem, .5):6.2f} {q(rem, .9):6.2f} | {q(post, .5):8.2f} | "
          f"{q(inf, .5):12.0f} {max(inf):4d} | {sum(lat) / tot:9.1%}")

sat = [r for r in recs if r["executor_workers"] and r["inflight_all"] > r["executor_workers"]]
print(f"\ncalls submitted while that process had more calls in flight than executor threads: "
      f"{len(sat)}/{len(recs)} ({len(sat) / len(recs):.1%})")

phases, cur = [], [recs[0]]
for r in recs[1:]:
    if r["t_submit"] - (cur[-1]["t_submit"] + cur[-1]["latency"]) > a.gap:
        phases.append(cur); cur = []
    cur.append(r)
phases.append(cur)
print(f"\n{len(phases)} rollout phases (split on gaps > {a.gap:.0f} s)")
print(f"{'phase':>5s} {'calls':>6s} {'span s':>7s}  top tools by summed latency")
for i, ph in enumerate(phases):
    span = max(r["t_submit"] + r["latency"] for r in ph) - ph[0]["t_submit"]
    s = defaultdict(float)
    for r in ph:
        s[r["tool"]] += r["latency"]
    top = ", ".join(f"{k} {v / sum(s.values()):.0%}" for k, v in sorted(s.items(), key=lambda kv: -kv[1])[:3])
    print(f"{i:5d} {len(ph):6d} {span:7.1f}  {top}")
