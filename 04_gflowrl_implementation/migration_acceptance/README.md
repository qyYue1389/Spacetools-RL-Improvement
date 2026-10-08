# Migration acceptance check for the P7 GPU session (2026-09-02)

`run12` = full `blinkdepth` on 4×A100-SXM4-40GB, **diagnostic switch not enabled**
(`dump_token_diagnostics` is off by default; the field list is verbatim identical to P4 — see `records/P7_GPU_RESULTS.md` §1.2).

Read per the new definition: **do not use** the old doc's narrow 107–109 band that came from n=3; look at the **stable upper bound and hard core**.

    single run   107/124 = 86.29%     bf16 actual range over five runs 106–109  -> inside the band
    hard core    12 [20,22,37,43,55,61,76,84,93,94,114,119]  matches the doc's record one by one
    regression   samples every baseline got right but run12 got wrong  0

⚠ The upper bound of the four baselines combined is 111/124, hard core 13; **after adding run12 it becomes 112/12**.
The upper bound went up by one again — no saturation seen, **so it cannot be used as an invariant** (`P7_HANDOFF.md` §3.3 item 1).
The acceptance check rests on "single run inside the band + hard core matches + zero regressions", not on the upper bound hitting the paper value.

Recompute: `python3 tools/p7/p7_accept_blinkdepth.py p7/accept/run12/blinkdepth/0.jsonl \
    p4/dumps/run1/blinkdepth/0.jsonl p4/dumps/run4/blinkdepth/0.jsonl \
    p6/gmu025/run5/blinkdepth/0.jsonl p6/gmu025/run6/blinkdepth/0.jsonl`
