# SFT eval raw artifacts · 2026-09-11/12

Eval of `qyYue1389/spacetools-sft-v1-4xa6000` @ `91fd4bdf6d2ffe3b355cc650e71925c6e91a15c5`
on 4× RTX A6000. For the analysis and conclusions see `03_sft_eval/sft_eval_report.md`.

The machine has been terminated; this is everything that could be kept.

---

## rollouts/ — trajectories, the most valuable part

```
reflocation.jsonl         100 rows
refplacement.jsonl        100 rows
refunseen.jsonl            77 rows
robospatial_run1.jsonl    350 rows   ← first run
robospatial_run2.jsonl    350 rows   ← second run, same checkpoint, same config
```

Each row is the full record of one sample, JSON:

| Field | Content |
|---|---|
| `index` | Sample index, alignable across the two runs |
| `input` | Full prompt: system prompt + schemas of the seven tools + image (base64) + question, about 10 KB |
| `output` | **The model's generation**, including all `<think>` / `<tool_call>` / `<tool_response>` / `<answer>` turns |
| `gts` | Ground truth. For VQA it is `Yes`/`No`; for Vacant it is 10 points; for some RefSpatial it is a base64 PNG |
| `acc` `score` `reward` | Scoring result; for this task all three are the same |

**The two robospatial files can be compared per-sample**; this is the data source for quantifying non-determinism:
between the two runs 29 samples flip, 199/350 generations differ byte-for-byte, while the tool responses have zero differences.

### How to split VQA / Vacant

By the shape of `gts`, no extra metadata needed:

```python
import re, json
def is_vacant(r):                       # list of points → Vacant (122 questions)
    return bool(re.search(r"\(\s*-?[0-9.]*\s*,", r["gts"]))
# the rest are VQA (228 questions), gts is Yes / No
```

### Scoring functions

`verl/utils/reward_score/robos_all.py`:

- Point questions: `compute_points_score(..., point_evaluation_method="convex_hull")`
  — **whether the predicted point falls inside the convex hull of the 10 GT points, binary**.
  Note it is not "distance to the nearest GT point": a point can be very close to some GT point yet outside the convex hull and score 0.
  When analyzing wrong answers, compute the distance to the **convex hull boundary**, using the repo's own `_convex_hull` /
  `_is_point_in_convex_polygon` (verified to agree with the on-disk acc 122/122).
- Yes/No questions: `compute_yesno_score`

---

## logs/ — eval logs

```
run1_all4benchmarks.log    15.8 MB   full run of all four benchmarks (32 min 41 s)
run2_robospatial.log        8.7 MB   robospatial rerun
blinkdepth_attempt1.log             two blinkdepth attempts, both died on the verl side
blinkdepth_attempt2.log             (GPU budget problem, see report §6.2)
perbench_<key>.log                  logs written separately per benchmark by verl
```

The logs contain lots of ANSI codes and progress bars. Useful greps:

```bash
grep -oE "val-core/[A-Za-z0-9/-]+/acc/mean@1:[0-9.]+" run1_all4benchmarks.log | sort -u
grep -c OutOfMemoryError run1_all4benchmarks.log
grep -cE "Error:|ERROR:toolshed" run1_all4benchmarks.log    # substring match, also catches TypeError:
grep -c "Version mismatch" run1_all4benchmarks.log
```

---

## verify/ — acceptance-check artifacts

```
VERIFY_run_initial.log              first four-item acceptance check
VERIFY_run_after_patch.log          acceptance check after the Ray Python patch (seven tools 7/7)
<tool>.log × 7                      smoke-test output of each of the seven tools (real weights loaded, real results)
chain_raw.log                       raw output of the Ray cross-conda-env tool-chain test
chain_rerun_after_cleanup.log       rerun after clearing leftover ray address files
ray_python_patch.log                applying the minor-version patch and read-back verification
```

---

## gpu/ — GPU traces, not reproducible

```
gputrace_1hz.log        1 Hz × 75 samples during the eval run, per-GPU used_MiB + util%
                        with per-process attribution at the end (which actor on which GPU, using how much)
gputrace_summary.log    per-GPU peak/mean summary
```

**Checking nvidia-smi after the run only shows 0 MiB**, so this can only be sampled while running.
Recommend making this step routine for future eval / RL.

---

## config/ — provenance

```
toolshed_config.yaml    the seven-tool config actually in effect this time (produced by generate_toolshed_config.py)
run_eval.sh.asrun       the version of run_eval.sh used for the run
WEIGHTS_PINS.txt        pinned revisions of the seven model weights
pkg_MANIFEST.txt        environment package manifest (incl. Python version table of the five envs, measured GPU budget)
SHA256SUMS.remote       hashes of the environment package files
POSTRESTORE.sh          the three things to fix up after restore (pushed to the HF repo)
VERIFY.sh.new           acceptance-check script after changing to four items (pushed to the HF repo)
```

---

## scripts/ and analysis/ — analysis scripts and their outputs, reusable

| Script | What it does |
|---|---|
| `audit2.sh` | Per-sample check of tool calls and error text (second step of interpretation; looking only at logs is not enough) |
| `health.sh` | Five health metrics + per-sample distribution of tool-call counts |
| `nondet.sh` | Splits the divergence between the two runs into policy side / tool side |
| `stable.sh` | Always-correct / always-wrong / flip decomposition, gives expected value and probability of passing the criterion |
| `hull.sh` | Recomputes Vacant with the convex-hull criterion using the repo's own geometry functions |
| `vacant.sh` | Deviation distribution of Vacant wrong answers |
| `split.sh` | Splits VQA / Vacant by GT shape |
| `material.sh` | VQA confusion matrix + export of representative trajectories |
| `compare.sh` | Per-sample comparison of the two runs |
| `gputrace.sh` | 1 Hz GPU sampling |

Paths in the scripts are absolute paths on that machine; change them when reusing.
`analysis/` holds their outputs from that time; you can check them directly against the numbers in the report.
