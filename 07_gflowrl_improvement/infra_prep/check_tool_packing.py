#!/usr/bin/env python3
"""Pre-flight: do the Toolshed tool actors fit on the tool GPUs, card by card?  (no GPU needed)

    python3 check_tool_packing.py examples/toolshed/run_rl.sh 4                 # TOOL_GPUS=4
    python3 check_tool_packing.py examples/toolshed/run_rl.sh 4 --card-gib 46   # + memory check
    python3 check_tool_packing.py --selftest

run_rl.sh already asserts  sum(num_actors * num_gpus) <= TOOL_GPUS, but only while training is
starting, and a sum that fits is not enough:
  * Ray needs the fractions on EACH card to add up to <= 1.0.  An actor that fits on no card
    waits forever: the placement group goes ready and the tool never answers.
  * num_gpus is a logical reservation, not a memory quota.  A fraction sized for an 80 GB card
    is too small on a 46 GB one (vlm: measured 32.11 GiB = 0.70 of an A40; upstream says 0.6).

The tool table is not copied here.  The TOOL_CONFIGS block, including its scale-to-fit logic,
is cut out of the run script and executed, so this checks what that script will really launch.
Which card Ray gives a fractional actor, and in what order Toolshed creates them, is not pinned
down, so the config's own order is tried first and then shuffled orders under two policies.

Verdict:  PACK_OK               every order tried fits
          PACK_ORDER_DEPENDENT  the config's own order fits, but some orders strand an actor.
                                Not a blocker: the paper's 2-node layout (7.8 on 8) and the P7
                                layout (3.7 on 4, ran 85 steps) both land here.  It means there
                                is no slack; if a tool never answers after start-up, look here.
          PACK_FAIL             sum too large, the config's own order does not fit, or a
                                fraction reserves less memory than the actor's measured peak
Exit code 1 for PACK_FAIL, 0 otherwise.
"""
import argparse, random, sys

PEAK_GIB = {"vlm": 32.11}   # measured single-instance peak (full report §8.1); the rest: not measured


def load_configs(path, tool_gpus, version):
    lines = open(path).read().splitlines()
    starts = [i for i, l in enumerate(lines) if l.startswith("TOOL_CONFIGS = {")]
    if not starts:
        sys.exit(f"no 'TOOL_CONFIGS = {{' block in {path}")
    s = starts[min({"v1": 0, "v2": 1}[version], len(starts) - 1)]
    e = next((i for i in range(s, len(lines)) if lines[i].startswith("pg = ray.util.placement_group")), None)
    if e is None:
        sys.exit(f"no 'pg = ray.util.placement_group' line after TOOL_CONFIGS in {path}")
    ns = {"roborefer_model": "", "depth_checkpoint": "", "tool_gpus": float(tool_gpus)}
    exec("\n".join(lines[s:e]), ns)          # the script's own assert fires here if the sum is too large
    return ns["TOOL_CONFIGS"]


def place(actors, n_cards, best_fit):
    """Put (name, fraction) actors on cards in order.  Returns (cards, None) or (None, stranded actor)."""
    used, cards = [0.0] * n_cards, [[] for _ in range(n_cards)]
    for name, f in actors:
        fits = [i for i in range(n_cards) if used[i] + f <= 1 + 1e-9]
        if not fits:
            return None, (name, f)
        i = max(fits, key=lambda j: used[j]) if best_fit else fits[0]
        used[i] += f
        cards[i].append((name, f))
    return cards, None


def check(configs, tool_gpus, card_gib=None, peaks=PEAK_GIB, shuffles=500):
    """Returns (verdict, report lines)."""
    out, fail = [], []
    gpu = [(k, c["resources"]["num_gpus"]) for k, c in configs.items()]
    for k, c in configs.items():
        f, n = c["resources"]["num_gpus"], c["num_actors"]
        out.append(f"  {k:<16} {n} x {f:<4} = {n * f:.1f}" if f else f"  {k:<16} {n} x 0    (CPU only)")
    actors = [(k, f) for k, f in gpu for _ in range(configs[k]["num_actors"]) if f > 0]
    demand = sum(f for _, f in actors)
    out.append(f"  demand {demand:.2f} of {tool_gpus:g} logical GPUs")
    if demand > tool_gpus + 1e-9:
        fail.append(f"sum {demand:.2f} > TOOL_GPUS {tool_gpus:g}")

    n_cards = int(tool_gpus)
    cards, stuck = place(actors, n_cards, best_fit=False)
    if cards is None or place(actors, n_cards, best_fit=True)[0] is None:
        stuck = stuck or place(actors, n_cards, best_fit=True)[1]
        fail.append(f"config order strands {stuck[0]} ({stuck[1]}): no card has that much left")
    else:
        out.append("== layout in config order")
        for i, c in enumerate(cards):
            out.append(f"  card {i}: " + (" + ".join(f"{k} {f}" for k, f in c) or "(empty)") + f" = {sum(f for _, f in c):.1f}")

    rng, bad, example = random.Random(0), 0, None
    for _ in range(shuffles):
        order = actors[:]
        rng.shuffle(order)
        for best_fit in (False, True):
            _, s = place(order, n_cards, best_fit)
            if s:
                bad, example = bad + 1, example or s
    out.append(f"== creation-order stress: {bad} of {2 * shuffles} shuffled runs strand an actor"
               + (f", e.g. {example[0]} ({example[1]})" if example else ""))

    if card_gib:
        for k, f in gpu:
            if k in peaks and f * card_gib < peaks[k]:
                fail.append(f"{k}: num_gpus={f} reserves {f * card_gib:.1f} GiB of a {card_gib:g} GiB card, "
                            f"measured peak {peaks[k]} GiB -> needs >= {peaks[k] / card_gib:.2f}")
        seen = [k for k, f in gpu if k in peaks]
        out.append(f"== memory on a {card_gib:g} GiB card: checked {', '.join(seen) or 'nothing'}; "
                   f"no measured peak for {', '.join(k for k, f in gpu if f and k not in peaks) or 'nothing'}")
    out += [f"  FAIL: {m}" for m in fail]
    return ("PACK_FAIL" if fail else "PACK_ORDER_DEPENDENT" if bad else "PACK_OK"), out


def selftest():
    def v(fracs, n, **kw):
        return check({f"t{i}": {"num_actors": 1, "resources": {"num_gpus": f}} for i, f in enumerate(fracs)}, n, **kw)[0]
    assert v([0.5] * 4, 2) == "PACK_OK"
    assert v([0.7] * 3, 2) == "PACK_FAIL"                      # sum 2.1 > 2
    assert v([0.6] * 3, 2) == "PACK_FAIL"                      # sum 1.8 <= 2, but no two share a card
    p7 = [0.6] * 3 + [0.7] + [0.2] * 4 + [0.1] * 4             # the P7 training layout: 3.7 on 4 cards
    assert v(p7, 4) == "PACK_ORDER_DEPENDENT"                  # small actors first -> vlm fits nowhere
    vlm = lambda f: {"vlm": {"num_actors": 1, "resources": {"num_gpus": f}}}
    assert check(vlm(0.6), 1, card_gib=46)[0] == "PACK_FAIL"   # 27.6 GiB reserved < 32.11 GiB peak
    assert check(vlm(0.7), 1, card_gib=46)[0] == "PACK_OK"
    print("SELFTEST PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_script", nargs="?", help="run_rl.sh (or any script with the same TOOL_CONFIGS block)")
    ap.add_argument("tool_gpus", nargs="?", type=float, help="TOOL_GPUS the run will use")
    ap.add_argument("--version", default="v1", choices=["v1", "v2"])
    ap.add_argument("--card-gib", type=float, help="memory of one tool card; enables the memory check")
    ap.add_argument("--peak-gib", action="append", default=[], metavar="TOOL=GIB", help="add a measured peak")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    if a.run_script is None or a.tool_gpus is None:
        ap.error("need RUN_SCRIPT and TOOL_GPUS (or --selftest)")
    peaks = {**PEAK_GIB, **{k: float(g) for k, g in (p.split("=") for p in a.peak_gib)}}
    print(f"== tool actors ({a.run_script}, {a.version}, TOOL_GPUS={a.tool_gpus:g})")
    try:
        configs = load_configs(a.run_script, a.tool_gpus, a.version)
    except AssertionError as e:
        print(f"  FAIL: {e}\nVERDICT: PACK_FAIL")
        sys.exit(1)
    verdict, lines = check(configs, a.tool_gpus, a.card_gib, peaks)
    print("\n".join(lines) + f"\nVERDICT: {verdict}")
    sys.exit(verdict == "PACK_FAIL")
