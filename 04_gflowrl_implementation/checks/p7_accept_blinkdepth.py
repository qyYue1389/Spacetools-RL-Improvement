#!/usr/bin/env python3
"""Migration acceptance for blinkdepth: stable upper bound + hard core.

The old, fragile criterion was "does this single run land in 107-109?".  That
band came from n=3; the actual bf16 range across runs is wider.  What is stable
across two hardware platforms is:

  * the union upper bound  (a sample counts as reachable if ANY run got it right)
  * the hard core          (samples EVERY run gets wrong)

So this compares the new run against the pooled baselines on those two, and
reports the single-run value only as context.

Usage:
  python3 tools/p7/p7_accept_blinkdepth.py NEW.jsonl BASE.jsonl [BASE.jsonl ...]
"""

import json
import sys


def load(path, threshold=0.5):
    """Return {index: correct} for one dump."""
    out = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            idx = d.get("index")
            if idx is None:
                raise SystemExit(f"{path}: no 'index' field; cannot align samples")
            out[int(idx)] = float(d["score"]) >= threshold
    return out


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    new_path, base_paths = sys.argv[1], sys.argv[2:]

    new = load(new_path)
    bases = [load(p) for p in base_paths]

    keys = set(new)
    for b in bases:
        if set(b) != keys:
            raise SystemExit("dumps do not cover the same sample indices")
    n = len(keys)

    print(f"n = {n} samples\n")
    print("single-run correct counts")
    print(f"  NEW   {new_path}")
    print(f"        {sum(new.values())}/{n} = {100 * sum(new.values()) / n:.2f}%")
    for p, b in zip(base_paths, bases):
        print(f"  base  {p}")
        print(f"        {sum(b.values())}/{n} = {100 * sum(b.values()) / n:.2f}%")

    base_ub = {k for k in keys if any(b[k] for b in bases)}
    base_hc = {k for k in keys if not any(b[k] for b in bases)}
    all_runs = bases + [new]
    pooled_ub = {k for k in keys if any(r[k] for r in all_runs)}
    pooled_hc = {k for k in keys if not any(r[k] for r in all_runs)}

    print(f"\nupper bound (union of correct)")
    print(f"  baselines only ({len(bases)} runs)   {len(base_ub)}/{n} = {100 * len(base_ub) / n:.2f}%")
    print(f"  + NEW          ({len(all_runs)} runs)   {len(pooled_ub)}/{n} = {100 * len(pooled_ub) / n:.2f}%")
    print(f"  NEW adds {len(pooled_ub) - len(base_ub)} sample(s) the baselines never got right")

    print(f"\nhard core (wrong in every run)")
    print(f"  baselines only   {len(base_hc)}   {sorted(base_hc)}")
    print(f"  + NEW            {len(pooled_hc)}   {sorted(pooled_hc)}")
    broke = sorted(base_hc - pooled_hc)
    print(f"  NEW solved {len(broke)} of the baseline hard core: {broke}")

    # samples the new run gets wrong that EVERY baseline got right
    always_right = {k for k in keys if all(b[k] for b in bases)}
    regressed = sorted(k for k in always_right if not new[k])
    print(f"\nsamples every baseline got right but NEW got wrong: {len(regressed)} {regressed}")
    gained = sorted(k for k in base_hc if new[k])
    print(f"samples every baseline got wrong but NEW got right:  {len(gained)} {gained}")


if __name__ == "__main__":
    main()
