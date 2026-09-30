# tools

| File | What |
|---|---|
| `unpack_dumps.sh` | Unpack every `*.jsonl.gz` dump next to itself (run once after cloning) |
| `parse_dump.py` | verl validation dump → per-sample trajectory records (`parsed/`), incl. tool chain signature, per-turn tool calls, correctness |
| `gputrace.sh` | 1 Hz `nvidia-smi` trace used during eval / training |

```bash
python3 tools/parse_dump.py 06_gflowrl_eval/dumps/runB-rerun4/cvb3ddepth/0.jsonl --emit /tmp/parsed
```
