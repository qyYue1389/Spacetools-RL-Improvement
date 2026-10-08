#!/usr/bin/env python3
"""P6 §2b: consistency (the fourth error category) — reads dumps only, needs neither a GPU nor a conda environment.

The input is the pass@k dump from experiment C (`p6/passk/<bench>/*.jsonl`, 5 rows per sample).
It does three things:

  1. split each sample's 5 runs by right/wrong into all-right / all-wrong / split;
  2. **cut the split samples once more by "are the <tool_call>s of the five runs verbatim identical"** —
     identical calls ⇒ the five runs face the same evidence, so a flip can only happen at the decision layer;
     different calls ⇒ each run asks something different, the five runs share no common evidence, nothing to vote on;
  3. report the number of samples where "the majority of the five runs is correct", and run a **per-sample paired test** against each greedy run
     (two-sided exact McNemar). **Do not compare ranges only** — one group of 5 samples gives a single vote value,
     comparing its range with the multiple greedy values compares a side with spread against a side without spread.
     **Note**: point prediction (Vacant) has no "majority" to vote on; that row is only descriptive statistics, not a strategy.

Usage (run from the repo root; greedy references are found by relative path, a missing one skips that comparison):

    python3 tools/p6/p6_consistency.py --root p6/passk

One trap: the `answer` field in the dump is **a copy of the GT**, not the model's answer
(every record has `answer == gts`). Right/wrong can only be read from `acc`. This script asserts this first.
"""
import argparse, json, glob, os, re, collections, statistics as st
from math import comb

TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
BINARY = {"yes", "no", "true", "false", "a", "b"}

# Dump directories of the greedy reference runs. **Must compare per-sample paired, not ranges only** —
# one group of 5 samples gives a single vote value; comparing its range with the multiple greedy values
# compares a side with spread against a side without spread. See P6_NOTES.md §2b.
GREEDY_DIRS = {
    "blinkdepth": ["p4/dumps/run1/blinkdepth", "p4/dumps/run4/blinkdepth",
                   "p6/gmu025/run5/blinkdepth", "p6/gmu025/run6/blinkdepth"],
    "robospatial": ["p4/dumps/run1/robospatial", "p4/dumps/run2/robospatial",
                    "p6/gmu025/run3/robospatial", "p6/gmu025/run8/robospatial"],
}


def mcnemar_exact(b, c):
    """Two-sided exact McNemar. b/c are the discordant counts on each side."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(path):
    rows = []
    for f in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            rows.extend(json.loads(l) for l in fh if l.strip())
    return rows


def split_key(gt):
    """robospatial's parquet mixes VQA and Vacant; split by the shape of the GT."""
    return "VQA" if str(gt).strip().lower() in BINARY else "Vacant"


def report(label, samples, bench, keep):
    """samples: {index: [row, ...]}; keep is this group's index set, used to filter the greedy references."""
    n = len(samples)
    tab = collections.Counter()
    for rows in samples.values():
        corr = [r["acc"] >= 0.5 for r in rows]
        calls = {tuple(c.strip() for c in TOOL_CALL.findall(r["output"])) for r in rows}
        state = "all-right" if all(corr) else ("all-wrong" if not any(corr) else "split")
        tab[(state, len(calls) == 1)] += 1

    avg1 = st.mean(st.mean(r["acc"] >= 0.5 for r in rows) for rows in samples.values())
    p5 = st.mean(any(r["acc"] >= 0.5 for r in rows) for rows in samples.values())
    maj_ok = {i: sum(r["acc"] >= 0.5 for r in rows) >= 3
              for i, rows in samples.items()}
    maj = sum(maj_ok.values())

    split = tab[("split", True)] + tab[("split", False)]
    print(f"\n=== {label}  n={n}")
    print(f"  avg@1 {avg1*100:6.2f}%   pass@5 {p5*100:6.2f}%   majority correct {maj}/{n} "
          f"({maj/n*100:.2f}%)")
    print(f"  all-right {tab[('all-right',True)]+tab[('all-right',False)]:4}  "
          f"all-wrong {tab[('all-wrong',True)]+tab[('all-wrong',False)]:4}  "
          f"split {split:4} ({split/n*100:.1f}%)")
    print(f"  among split samples: five tool calls **verbatim identical** {tab[('split',True)]:4}  "
          f"(pure decision flip)   calls differ {tab[('split',False)]:4}")

    dirs = [d for d in GREEDY_DIRS.get(bench, []) if os.path.isdir(d)]
    if dirs:
        print("  Per-sample paired (majority vote@5 vs each greedy run, two-sided exact McNemar):")
        sig = 0
        nets = []
        for d in dirs:
            g = {r["index"]: (r["acc"] >= 0.5) for r in load(d)}
            idx = [i for i in keep if i in g]
            b = sum(1 for i in idx if maj_ok[i] and not g[i])
            c = sum(1 for i in idx if g[i] and not maj_ok[i])
            p = mcnemar_exact(b, c)
            sig += p < 0.05
            nets.append(b - c)
            print(f"    vs {os.path.basename(os.path.dirname(d)):6}"
                  f"(={sum(g[i] for i in idx):4}) vote-only correct {b:3} · greedy-only correct {c:3}"
                  f"   net {b-c:+3}   p={p:.4f}{'  *' if p < 0.05 else ''}")
        if sig == len(dirs):
            verdict = "all significant"
        elif all(x > 0 for x in nets):
            verdict = "**direction consistently positive but not significant; not a conclusion**"
        elif all(x <= 0 for x in nets):
            verdict = "**no gain** (net difference ≤ 0)"
        else:
            verdict = "**direction inconsistent, no effect**"
        print(f"  => {sig}/{len(dirs)} comparisons significant. {verdict}")
        print("     Note: all comparisons **share the same vote arm** (only one group of 5 samples was run), "
              "they are not independent replicates; settling it needs a second group of samples.")

    if label.endswith("Vacant"):
        print("  ⚠ point prediction has no \"majority\" to vote on; the \"majority correct\" above is only descriptive statistics, not an implementable strategy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="p6/passk")
    args = ap.parse_args()

    for bench in sorted(os.listdir(args.root)):
        path = os.path.join(args.root, bench)
        if not os.path.isdir(path):
            continue
        rows = load(path)
        if not rows:
            continue

        bad = [r for r in rows if str(r["answer"]) != str(r["gts"])]
        assert not bad, (f"{bench}: `answer` no longer equals `gts` ({len(bad)} rows). "
                         "This script assumes answer is a copy of the GT; if upstream changed its meaning, confirm before reading further.")

        by_idx = collections.defaultdict(list)
        for r in rows:
            by_idx[r["index"]].append(r)
        ks = {len(v) for v in by_idx.values()}
        if ks != {5}:
            print(f"[warn] {bench}: runs per sample = {ks}, not 5")

        if bench == "boppose":
            # continuous metric, right/wrong binning is meaningless; report only avg@1 and best@5, and warn about selection bias
            avg = st.mean(st.mean(r["acc"] for r in v) for v in by_idx.values())
            best = st.mean(max(r["acc"] for r in v) for v in by_idx.values())
            print(f"\n=== boppose  n={len(by_idx)} (continuous IoU, no right/wrong binning)")
            print(f"  avg@1 {avg*100:.2f}   best@5 {best*100:.2f}")
            print("  ⚠ best@5 has selection bias: even if the scores were pure noise, max would be above the mean. **Cannot be counted as a gain**")
            continue

        groups = collections.defaultdict(dict)
        for i, v in by_idx.items():
            gt = v[0]["gts"]
            sub = split_key(gt)
            # only robospatial needs further splitting; single-type benchmarks get no suffix
            groups[sub][i] = v
        if len(groups) == 1:
            s = next(iter(groups.values()))
            report(bench, s, bench, set(s))
        else:
            for sub in ("VQA", "Vacant"):
                if sub in groups:
                    report(f"{bench}:{sub}", groups[sub], bench, set(groups[sub]))


if __name__ == "__main__":
    main()
