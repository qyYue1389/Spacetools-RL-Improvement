# P1_prep: preparation and conclusions for P1 / P2 / P3 before the GPU session (no GPU)

2026-09-25. Corresponds to the items marked ✅ in the sections of the Design Doc.

## Files

| File | Purpose |
|---|---|
| `p1a_ray_trainer.diff` | P1-a patch (based on SpaceTools-RL `c6fef78a` as used in training; redone on 2026-09-30 from the old 9-14 base, the old version lacked the β-decomposition logs): `compute_gflowrl_flow_gap(filter_degenerate=...)` sets g̃ to 0 for groups whose within-group reward range is 0; driver guard (when filter is on, requires `ppo_mini_batch_size == train_batch_size` and `ppo_epochs == 1`); new logs `gflowrl/kept_groups`, `kept_rollout_frac`, `clip_saturation_kept`, `g_abs_p50/p90/p99` |
| `p1a_run_rl_gflowrl.diff` | New switch `GF_FILTER_DEGEN` (default false) |
| `patched/` | The two patched files (the SpaceTools-RL repo itself is untouched) |
| `p1a_check.py` | Self-check, runs on CPU, needs torch: `python p1a_check.py patched/ray_trainer.py <original ray_trainer.py> <core_algos.py>` |
| `p2_query_type.py` | P2 investigation before the GPU session: the query phrasing each of the three arms gives roborefer on Vacant (anchor object only / with a location description) and the respective hit rate, point override and accuracy |
| `robospatial_vacant.parquet` | In-training validation set: the 122 Vacant questions from the robospatial eval (same schema as the original parquet). `data.val_files=<it> trainer.test_freq=10 trainer.val_before_train=True` |
| `p1_monitor.py` | In-training monitoring: share of queries that ask only for the anchor object (primary criterion), point-override rate (± per-prompt bootstrap SE), 0.05 grid, pass-through / point-override accuracy, tool-point hit rate; `--upper` forces pass-through |

## Self-check results (p1a_check.py all PASS)

1. With filter off, advantages match the original function bit-for-bit.
2. With filter on, degenerate groups have g̃ = 0 and the other rows are unchanged; kept_groups / kept_rollout_frac are correct.
3. On-policy (dp_actor sets `old_log_prob = log_prob.detach()`), the gradient is bit-for-bit equal to "a real loss mask with all sequences kept in the denominator" (max|diff| = 0).
4. Counterexample: off-policy, degenerate groups receive a proximal-term gradient → the guard is necessary.
5. In addition, `04_gflowrl_implementation/checks/run_checks.sh` pointed at the full patched tree (after the 2026-09-30 redo): all six self-checks PASS.

## How to turn it on in training

```bash
GF_FILTER_DEGEN=true bash examples/toolshed/run_rl_gflowrl.sh \
    trainer.rollout_data_dir=$OUT/rollouts            # without this the point-override rate can't be read
# P1-c arm B: also add  actor_rollout_ref.actor.grad_clip=<L̄>   (under AdamW ≡ loss×1/L̄)
# P1-c arm C: GF_EPS_LOW=0.28 GF_EPS_HIGH=0.20
python p1_monitor.py --window 10 $OUT/rollouts
python p1_monitor.py --window 1 $OUT/val_outputs   # in-training validation: number of object-only questions out of 122 each time
```

Note: `_dump_generations` saves the images of the 320 samples as PNG at every step. A switch has been added in `../infra_prep/patched/ray_trainer.py` (the P1-a version in this directory doesn't have it): add `+trainer.dump_images=false` to the training command and only JSONL is written, see `../infra_prep/README.md` §6.

## Calibration of the monitoring script

On P0's 9 robospatial dumps it reproduces the P0 table exactly: pass-through 72.7 / 100.3 / 77.0, point override 49.3 / 21.7 / 44.0, grid 18.7 / 7.0 / 18.3, P4 point-override accuracy 27.7%.

## Key findings

- **The root of the Vacant gap is the query phrasing.** When asking only for the anchor object ("cup") the tool point hits 0–3% and the model can only override the point; with a location description it hits 56%. Object-only questions per run: SFT 42.0 · C′ 34.3 · GRPO 13.3 / 122; within each category the three arms are almost the same; the 13 questions P4 gets right and C′ gets wrong all follow this pattern. In the SFT data, 410 / 410 demonstrations of this RoboSpatial template ask only for the object. Primary-criterion threshold: a drop of ≥ 12 pp over a 20-step window.

- **The root of P2 is tool-call quality.** Tool point (first point of the last roborefer call) falls inside GT: SFT 46.3, GRPO 60.3, C′ 49.3 / 122 (mean of three runs; GRPO vs C′ stable samples 13 : 4, p = 0.049). On point-override samples the tool point hits only 10–19%, and the overridden answer 19–28%.
- **Forced pass-through is not an upper bound:** Vacant SFT 50.7 → 46.3, GRPO 62.3 → 60.3, C′ 53.7 → 49.3.
- **P1-c:** based on P7 grad_norm, with L̄ = 334, 9 / 85 steps (10.6%) would still be clipped; 250 → 21%, 200 → 40%.
- **P3:** in the SFT data, 977 RoboSpatial yes/no samples, 0 of which call depth_estimator; depth_estimator only appears in RefSpatial depth (A/B templates) and bopask. On the 29 front/behind questions the three-run means are SFT 17.3 / GRPO 19.0 / C′ 19.3; the old 72.4% → 62.1% was single-run noise.
- **Thresholds:** simulated on P0 data, the SD of the two-window difference reading only RoboSpatial vacant is ≈ 9 pp; merged with RefSpatial pointing ≈ 5 pp.
