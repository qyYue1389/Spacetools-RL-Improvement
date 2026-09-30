# 02 · SFT training (SpaceTools Step 3) / SFT 训练

Qwen2.5-VL-3B-Instruct, full fine-tuning with LLaMA-Factory ([SpaceTools-SFT](https://github.com/ChicyChen/SpaceTools-SFT)), 3000 steps on 4× RTX A6000 in 7 h 45 min, global batch 8 (= paper). The result is the start point π_ref for RL, published as [`qzpm55555/spacetools-sft-v1-4xa6000`](https://huggingface.co/qzpm55555/spacetools-sft-v1-4xa6000).

| Path | What |
|---|---|
| `run_sft.sh` | The script as run; drop it into `SpaceTools-SFT/scripts/spacetools/` / 实际运行的脚本 |
| `run_sft.sh.diff_vs_upstream` | Our changes: per-device batch / grad-accum derived from GPU count (global batch stays 8), ZeRO-2, `save_only_model`, `eval_steps` 500, `use_reentrant_gc: false`, faster download |
| `FINALIZE.sh` | After training: verify the run, collect evidence, prune, upload to HF / 训练后验收、收集证据、上传 |
| `smoke_check.py` | Format smoke test of the checkpoint vs the base model (did SFT take effect; not a capability score) |
| `sft_training_report_4xA6000.md` | Execution report: memory headroom, speed, findings / 执行报告 |
| `tool_schema_check/` | Confirms the tool schema in the SFT data matches the one used at eval time |

```bash
cd SpaceTools-SFT && cp /path/to/this/repo/02_sft_training/run_sft.sh scripts/spacetools/run_sft.sh
bash scripts/spacetools/run_sft.sh          # prints "GPU 数 4 · per_device=2 · ga=1 · 全局 batch 8 ✓"
bash /path/to/this/repo/02_sft_training/FINALIZE.sh /workspace/experiments/<run>
```

Background notes: `../docs/sft_training_notes.md`, `../docs/why_not_train_per_benchmark.md`.
