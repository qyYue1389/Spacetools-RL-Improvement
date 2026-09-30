# 05 · GFlowRL training / GFlowRL 训练

One run of the C′ arm: 85 steps (1 epoch, as in upstream `run_rl.sh`), `rollout.n` = G = 5, β = 8, ε = 0.2 / 0.28, on 8× A40 (4 GPUs for tools, 4 for training), 21 h 57 min, exit code 0. Checkpoints 30 / 60 / 85 are on [`qzpm55555/spacetools-p7-gflowrl-cprime-8xa40`](https://huggingface.co/qzpm55555/spacetools-p7-gflowrl-cprime-8xa40).
C′ 臂的一次完整训练。

| Path | What |
|---|---|
| `gflowrl_training_report_8xA40.md` | Training report: machine, hyper-parameters, optimizations used, metrics, the broken-PCIe-P2P incident / 训练报告 |
| `training_metrics_85steps.csv` | One row per step: wall time, degenerate-group share, clip saturation (also recomputed at β = 2, 4, 16), reward vs drift terms of the flow gap, grad norm, score, memory |
| `config/full_train.sh.asrun` | The exact launch script (env vars + command) |
| `config/run_rl.sh.asrun`, `run_rl_gflowrl.sh.asrun`, `rl_toolshed_config.yaml` | Launchers and tool config as run |
| `config/ENVIRONMENT.txt` | Machine / driver / env snapshot |
| `scripts/extract_metrics_table.py` | Parse the training log into the metrics table / CSV |
| `scripts/ckpt_janitor.py` | Keep disk usage bounded while snapshotting milestone checkpoints |
| `scripts/merge_and_upload_ckpts.sh` | Merge FSDP shards to HF format and upload |
| `scripts/make_full_train_script.py` | Derives the full-run script from the 3-step smoke script |
| `scripts/debug_probes/` | Minimal reproductions used to debug the NCCL hang and the exit-code bug |
