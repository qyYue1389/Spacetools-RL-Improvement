#!/bin/bash
# =============================================================================
# FINALIZE.sh — wrap-up after SFT training finishes (4× A6000).
#
# Why it's needed: once the pod is destroyed, everything that can **prove this training run was done right** is gone,
# and this checkpoint is π_ref, the shared starting point for the later two-arm RL comparison — if its definition is in doubt, all of Route C is void.
# The ckpt itself is only one of those things.
#
# Usage:
#     bash FINALIZE.sh                              # acceptance check + collect evidence only (no delete, no upload)
#     PRUNE=1 bash FINALIZE.sh                      # additionally delete intermediate checkpoints
#     HF_REPO=username/spacetools-sft-v1-4xa6000 \
#     HF_TOKEN=hf_xxx PRUNE=1 bash FINALIZE.sh      # full set: acceptance check → collect → clean up → upload
#
#     bash FINALIZE.sh /path/to/experiments/sft_v1_20260908_0130   # specify the experiment directory manually
#
# By default nothing destructive is done. Deleting and uploading both need explicit switches.
# =============================================================================

set -uo pipefail

# ---- Environment ------------------------------------------------------------
# ⚠️ Do not `export PATH=/opt/conda-st/bin:$PATH` — if the caller has already activated spacetools-sft,
# this puts base's bin first, and the subsequent `conda activate` becomes a no-op because it's "already activated",
# so python3 resolves to base. The consequences are asymmetric: sections 1–4 use only stdlib and go all green under base anyway;
# only section 5's import torch and section 8's huggingface_hub fail —
# giving the combination "acceptance check all passed + evidence incomplete", which is the easiest to misjudge.
eval "$(/opt/conda-st/bin/conda shell.bash hook)" 2>/dev/null
conda activate spacetools-sft 2>/dev/null
# Assertion: environment variables lie; whether it can import is what counts
PY3="$(command -v python3)"
case "$PY3" in
  /opt/conda-st/envs/spacetools-sft/bin/python3) ;;
  *) echo "✗ python3 resolves to $PY3, not spacetools-sft's. PATH is polluted, refusing to start."; exit 1 ;;
esac
python3 -c "import torch, huggingface_hub" 2>/dev/null || {
    echo "✗ $PY3 cannot import torch/huggingface_hub, environment not activated properly, refusing to start."; exit 1; }

REPO_DIR="${REPO_DIR:-/workspace/SpaceTools-SFT}"   # LLaMA-Factory fork (used to get git info)
# ⚠️ REPO_DIR in run_sft.sh is LLAMA_DIR/.. = /workspace, not the fork directory itself.
# The experiment directory defaults to /workspace/experiments/, and this time we explicitly passed OUTPUT_DIR=.../full.
EXP_ROOT="${EXP_ROOT:-/workspace}"
HANDOFF="${HANDOFF:-/workspace/runpod-handoff}"     # where the training logs actually are
PRUNE="${PRUNE:-0}"
KEEP_MID="${KEEP_MID:-1500}"     # the one step kept besides the final ckpt (for sanity comparison); set 0 to keep none

fail=0
warn=0
ok()   { echo "  ✓ $*"; }
bad()  { echo "  ✗ $*"; fail=$((fail+1)); }
note() { echo "  ⚠ $*"; warn=$((warn+1)); }

# ---- Locate the experiment directory -----------------------------------------
if [ $# -ge 1 ]; then
    EXP="$1"
else
    EXP=$(ls -1dt "$EXP_ROOT"/experiments/sft_v1_* "$EXP_ROOT"/experiments/full 2>/dev/null | head -1)
fi
[ -n "${EXP:-}" ] && [ -d "$EXP" ] || { echo "✗ experiment directory not found. Specify it with 'bash FINALIZE.sh <dir>'"; exit 1; }

CKPT="$EXP/sft_checkpoint"
DATA="$EXP/sft_data"
STAMP=$(date +%Y%m%d_%H%M%S)
EVID="$EXP/EVIDENCE_$STAMP"

echo "Experiment directory: $EXP"
echo "checkpoint: $CKPT"
echo

# =============================================================================
# 1. Batch-definition acceptance check — this is the most critical item
#
#    train_samples_per_second × train_runtime = total number of samples fed in during this training run
#                                             = max_steps × global batch 8
#
#    ⚠️ Don't look at epoch. run_sft.sh's DATASET_SPEC registers the dataset 3 times
#    (DATASET_SPEC="${NAME},${NAME},${NAME}"), so HF's epoch is computed against 7020×3=21060,
#    it will always look off; that's a false alarm.
# =============================================================================
echo "=== 1. Global batch definition ==="
python3 - "$EXP" "$CKPT" <<'PY'
import json, os, sys, re
exp, ckpt = sys.argv[1], sys.argv[2]

# Read from the actually generated config, don't use assumed values
cfg = os.path.join(exp, "sft_config.yaml")
vals = {}
if os.path.exists(cfg):
    for line in open(cfg):
        m = re.match(r'\s*(max_steps|per_device_train_batch_size|gradient_accumulation_steps|learning_rate|deepspeed|cutoff_len):\s*(\S+)', line)
        if m:
            vals[m.group(1)] = m.group(2)
else:
    print("  ✗ sft_config.yaml not found"); sys.exit(1)

steps = int(vals.get("max_steps", 0))
pd    = int(vals.get("per_device_train_batch_size", 0))
ga    = int(vals.get("gradient_accumulation_steps", 0))
print(f"  config: max_steps={steps} per_device={pd} ga={ga} lr={vals.get('learning_rate')} "
      f"deepspeed={os.path.basename(vals.get('deepspeed',''))} cutoff_len={vals.get('cutoff_len')}")

res = os.path.join(ckpt, "all_results.json")
if not os.path.exists(res):
    print("  ✗ no all_results.json — training did not finish normally"); sys.exit(1)
r = json.load(open(res))
rt  = r.get("train_runtime")
sps = r.get("train_samples_per_second")
if rt is None or sps is None:
    print(f"  ✗ all_results.json is missing train_runtime / train_samples_per_second: {list(r)}"); sys.exit(1)

seen = rt * sps
want = steps * 8
err  = abs(seen - want) / want
print(f"  train_runtime           {rt:.1f} s  ({rt/3600:.2f} h)")
print(f"  train_samples_per_second {sps:.4f}")
print(f"  samples seen = {seen:.0f}   expected = {steps} × 8 = {want}   deviation {err*100:.2f}%")
if err < 0.01:
    print(f"  ✓ global batch is indeed 8, consistent with the paper's definition")
else:
    print(f"  ✗ doesn't match. Each step actually fed {seen/steps:.2f} samples, not 8 — this ckpt cannot be used as π_ref")
    sys.exit(1)
print(f"  train_loss              {r.get('train_loss')}")
print(f"  (epoch field = {r.get('epoch')}, counted against 7020×3=21060, ignore it)")
PY
[ $? -eq 0 ] || fail=$((fail+1))

# =============================================================================
# 2. Health of the training run
# =============================================================================
echo
echo "=== 2. loss trajectory ==="
python3 - "$CKPT" <<'PY'
import json, os, sys, math
ckpt = sys.argv[1]
p = os.path.join(ckpt, "trainer_state.json")
if not os.path.exists(p):
    print("  ✗ no trainer_state.json"); sys.exit(1)
st = json.load(open(p))
hist = [h for h in st.get("log_history", []) if "loss" in h]
ev   = [h for h in st.get("log_history", []) if "eval_loss" in h]
if not hist:
    print("  ✗ no loss in log_history"); sys.exit(1)
first, last = hist[0], hist[-1]
print(f"  {len(hist)} logged points, last step global_step={st.get('global_step')}/{st.get('max_steps')}")
print(f"  loss  step {first['step']}: {first['loss']:.4f}  →  step {last['step']}: {last['loss']:.4f}")
bad = 0
if st.get("global_step") != st.get("max_steps"):
    print(f"  ✗ did not run to completion: {st.get('global_step')} / {st.get('max_steps')}"); bad += 1
nan = [h['step'] for h in hist if not math.isfinite(h['loss'])]
if nan:
    print(f"  ✗ {len(nan)} non-finite loss values, first at step {nan[0]}"); bad += 1
if last['loss'] >= first['loss']:
    print(f"  ⚠ final loss is not lower than initial — not necessarily wrong, but worth a look at the curve")
if ev:
    print(f"  eval_loss  first {ev[0]['eval_loss']:.4f} → last {ev[-1]['eval_loss']:.4f}  ({len(ev)} evals)")
    if ev[-1]['eval_loss'] > ev[0]['eval_loss']:
        print(f"  ⚠ eval_loss went up — possibly overfitting (val_size is only 20, very noisy, don't over-read it)")
else:
    print("  ⚠ no eval records")
if not bad:
    print("  ✓ training ran to completion and loss is finite")
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] || fail=$((fail+1))

# =============================================================================
# 3. Checkpoint file integrity
#
#    run_sft.sh's Phase 3 already: deletes text_config from config.json,
#    sets tie_word_embeddings=true, copies preprocessor_config.json from the base model,
#    and copies in a toolshed_config.yaml. This is a **re-check that Phase 3 actually ran** —
#    if training was restarted by hand, Phase 3 may have been skipped, and that raises no error at load time,
#    it just hands the RL side a model with the wrong tie_word_embeddings.
# =============================================================================
echo
echo "=== 3. checkpoint integrity ==="
[ -d "$CKPT" ] || { bad "checkpoint directory does not exist"; }
# chat_template.json is the processor-level template; sglang reads it when loading a VLM.
# run_sft.sh's Phase 3 only copied preprocessor_config.json from the base model, not this —
# if it's missing the RL side crashes, while the tokenizer-side chat_template.jinja existing would make the template check below go green anyway.
for f in config.json generation_config.json tokenizer_config.json \
         preprocessor_config.json chat_template.json toolshed_config.yaml; do
    [ -f "$CKPT/$f" ] && ok "$f" || bad "missing $f"
done
# tokenizer itself: either tokenizer.json or vocab+merges
if [ -f "$CKPT/tokenizer.json" ] || [ -f "$CKPT/vocab.json" ]; then
    ok "tokenizer itself"
else
    bad "missing tokenizer.json / vocab.json"
fi
# weights
NSAFE=$(ls -1 "$CKPT"/*.safetensors 2>/dev/null | wc -l)
if [ "$NSAFE" -gt 0 ]; then
    WSZ=$(du -ch "$CKPT"/*.safetensors 2>/dev/null | tail -1 | cut -f1)
    ok "$NSAFE weight shards, total $WSZ"
    [ "$NSAFE" -gt 1 ] && { [ -f "$CKPT/model.safetensors.index.json" ] && ok "index.json" || bad "multiple shards but index.json missing"; }
else
    bad "no .safetensors weights"
fi

python3 - "$CKPT" <<'PY'
import json, os, sys
ckpt = sys.argv[1]
p = os.path.join(ckpt, "config.json")
if not os.path.exists(p): sys.exit(1)
c = json.load(open(p))
bad = 0
# The two Phase 3 changes
if "text_config" in c:
    print("  ✗ config.json still contains text_config — Phase 3 did not run, RL-side loading will have problems"); bad += 1
else:
    print("  ✓ text_config removed")
if c.get("tie_word_embeddings") is not True:
    print(f"  ✗ tie_word_embeddings = {c.get('tie_word_embeddings')}, should be true — Phase 3 did not run"); bad += 1
else:
    print("  ✓ tie_word_embeddings = true")
print(f"  ✓ dtype {c.get('torch_dtype')} · {c.get('model_type')}")
# chat template: for qwen2_vl usually embedded in tokenizer_config.json
tc = os.path.join(ckpt, "tokenizer_config.json")
has_tpl = os.path.exists(os.path.join(ckpt, "chat_template.jinja"))
if not has_tpl and os.path.exists(tc):
    has_tpl = "chat_template" in json.load(open(tc))
print(("  ✓ chat template present" if has_tpl else
       "  ✗ chat template not found (neither chat_template.jinja nor in tokenizer_config.json)"))
sys.exit(1 if (bad or not has_tpl) else 0)
PY
[ $? -eq 0 ] || fail=$((fail+1))

# =============================================================================
# 4. Data definition — exactly which 7020 samples were trained on this time, and what the system prompt looks like
#
#    Without this section, in half a year nobody can prove this ckpt used v1's 11 schemas.
# =============================================================================
echo
echo "=== 4. Data definition ==="
python3 - "$DATA" "$EVID" <<'PY'
import json, os, sys, hashlib, re
data, evid = sys.argv[1], sys.argv[2]
j = os.path.join(data, "data", "train.json")
if not os.path.exists(j):
    print(f"  ⚠ {j} not found — the data may already have been cleaned up, skipping (not counted as a failure)"); sys.exit(0)
d = json.load(open(j))
sysprompts = {it.get("system", "") for it in d}
print(f"  {len(d)} samples  (v1 expects 7020 = 7907 − 887 robot-tool samples)")
if len(d) != 7020:
    print(f"  ⚠ not 7020 — check whether VERSION is v1")
if len(sysprompts) == 1:
    sp = next(iter(sysprompts))
    h = hashlib.sha256(sp.encode()).hexdigest()
    n = len(re.findall(r'"name"\s*:', sp))
    print(f"  ✓ system prompt consistent across the whole table  {len(sp)} chars  sha256 {h[:16]}")
    print(f"    number of tool schemas ≈ {n}  (v1 expects 11)")
    os.makedirs(evid, exist_ok=True)
    open(os.path.join(evid, "system_prompt.txt"), "w").write(sp)
    json.dump({"n_samples": len(d), "system_sha256": h, "system_chars": len(sp),
               "n_tool_schemas": n},
              open(os.path.join(evid, "data_provenance.json"), "w"), indent=2)
else:
    print(f"  ✗ system prompt has {len(sysprompts)} different versions — data preparation went wrong"); sys.exit(1)
PY
[ $? -eq 0 ] || fail=$((fail+1))

# =============================================================================
# 5. Collect evidence (a few hundred KB, but this is the only copy)
# =============================================================================
echo
echo "=== 5. Collect evidence → $EVID ==="
mkdir -p "$EVID"
for f in trainer_state.json all_results.json train_results.json eval_results.json \
         trainer_log.jsonl training_loss.png training_eval_loss.png; do
    [ -f "$CKPT/$f" ] && { cp "$CKPT/$f" "$EVID/"; echo "  + $f"; }
done
for f in sft_config.yaml tool_config.yaml; do
    [ -f "$EXP/$f" ] && { cp "$EXP/$f" "$EVID/"; echo "  + $f"; }
done
# Startup log from training (contains the line "GPUs N · per_device=X · ga=Y · global batch 8 ✓")
# ⚠️ The training log is not under EXP but in the handoff directory — that line
# "GPUs N · per_device=X · ga=Y · global batch 8 ✓" is exactly the definition evidence this script keeps stressing.
for f in "$EXP"/*.log "$HANDOFF"/probe*.log "$HANDOFF"/launcher*.log; do
    [ -f "$f" ] && { cp "$f" "$EVID/"; echo "  + $(basename "$f")"; }
done
# Snapshot of the live environment
{
  echo "# FINALIZE snapshot  $(date -u +%FT%TZ)"
  echo; echo "## GPU"
  nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total --format=csv
  echo; echo "## Key package versions"
  python3 -c "import torch,transformers,deepspeed,flash_attn,llamafactory as L; \
print('torch',torch.__version__); print('transformers',transformers.__version__); \
print('deepspeed',deepspeed.__version__); print('flash_attn',flash_attn.__version__); \
print('llamafactory',L.__version__)"
  echo; echo "## git"
  git -C "$REPO_DIR" log -1 --format='SpaceTools-SFT %H %ci %s' 2>/dev/null
  echo; echo "## checkpoint listing"
  ls -la "$CKPT"
} > "$EVID/SNAPSHOT.txt" 2>&1
echo "  + SNAPSHOT.txt"
( cd "$CKPT" && sha256sum ./*.safetensors ./*.json 2>/dev/null ) > "$EVID/CKPT_SHA256SUMS" 2>/dev/null
echo "  + CKPT_SHA256SUMS"
echo "  evidence directory $(du -sh "$EVID" | cut -f1)"

# =============================================================================
# 6. Clean up intermediate checkpoints (needs PRUNE=1)
#
#    sft_config.yaml has save_steps=500 and no save_total_limit
#    → 3000 steps leave 6 intermediate ckpts, about 7.6 GB each (bf16).
#    save_only_model=true so there's no optimizer state in them; deleting them doesn't affect any ability to resume training
#    — it couldn't resume in the first place.
# =============================================================================
echo
echo "=== 6. Intermediate checkpoints ==="
MIDS=$(ls -1d "$CKPT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n)
if [ -z "$MIDS" ]; then
    echo "  (no intermediate checkpoints)"
else
    echo "$MIDS" | while read -r m; do echo "  $(du -sh "$m" | cut -f1)  $(basename "$m")"; done
    TOT=$(du -sh -c $MIDS 2>/dev/null | tail -1 | cut -f1)
    echo "  total $TOT"
    if [ "$PRUNE" = "1" ] && [ "$fail" -ne 0 ]; then
        echo "  Refusing to delete — $fail earlier items failed. Whether the final weights are usable is not yet confirmed;"
        echo "  deleting intermediate ckpts now might delete the only usable copy. Sort out the problem first."
    elif [ "$PRUNE" = "1" ]; then
        echo "  PRUNE=1 → keep checkpoint-$KEEP_MID, delete the rest"
        echo "$MIDS" | while read -r m; do
            if [ "$(basename "$m")" = "checkpoint-$KEEP_MID" ]; then
                echo "    keep $(basename "$m")"
            else
                rm -rf "$m" && echo "    deleted $(basename "$m")"
            fi
        done
        df -h "$CKPT" | tail -1 | sed 's/^/    /'
    else
        echo "  Not deleted. To delete, rerun with PRUNE=1 (the final weights are in the root of $CKPT, unaffected)"
    fi
fi

# =============================================================================
# 7. Format smoke test + base comparison (skip with SMOKE=0; about 3–5 minutes, uses 1 GPU)
#
#    What it catches is this kind of silent failure: the loss curve looks completely normal, but the model never learned to emit tool calls.
#    This can still be the case even if the first six sections all pass — those sections only check engineering, not what the model learned.
#    ⚠️ This is not a capability measurement; samples are taken from the training set. Only run_eval.sh gives real scores.
# =============================================================================
echo
echo "=== 7. Format smoke test ==="
SMOKE="${SMOKE:-1}"
SMOKE_PY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/smoke_check.py"
TRAIN_JSON="$DATA/data/train.json"
if [ "$SMOKE" != "1" ]; then
    echo "  SMOKE=0, skipping"
elif [ ! -f "$SMOKE_PY" ]; then
    note "smoke_check.py not found (should be in the same directory as this script), skipping"
elif [ ! -f "$TRAIN_JSON" ]; then
    note "$TRAIN_JSON not found (data already cleaned up?), skipping"
else
    mkdir -p "$EVID"
    python3 "$SMOKE_PY" "$CKPT" "$TRAIN_JSON" --n "${SMOKE_N:-12}" \
            --base "${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}" \
            --out "$EVID/smoke_check.json" 2>&1 | sed 's/^/  /'
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then ok "format smoke test passed"
    else note "format smoke test did not pass — see the raw output above. Does not block upload, but figure it out before the full eval"; fi
fi

# =============================================================================
# 8. Upload (needs HF_REPO + HF_TOKEN)
# =============================================================================
echo
echo "=== 8. Upload ==="
if [ "$fail" -ne 0 ]; then
    echo "  Skipping — $fail earlier items failed; sort out the problem before uploading"
elif [ -z "${HF_REPO:-}" ] || [ -z "${HF_TOKEN:-}" ]; then
    echo "  Skipping — no HF_REPO / HF_TOKEN given. To upload:"
    echo "    HF_REPO=<username>/spacetools-sft-v1-4xa6000 HF_TOKEN=hf_xxx bash FINALIZE.sh"
    echo "  The repo will be created as private. You can also do it manually first:"
    echo "    huggingface-cli upload --private --repo-type model <repo> $CKPT ."
else
    echo "  → $HF_REPO (private)"
    export HF_TOKEN
    python3 - "$CKPT" "$EVID" "$HF_REPO" <<'PY'
import sys, os
from huggingface_hub import HfApi
ckpt, evid, repo = sys.argv[1], sys.argv[2], sys.argv[3]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
api.upload_folder(folder_path=ckpt, repo_id=repo, repo_type="model",
                  ignore_patterns=["checkpoint-*/**"])   # intermediate ckpts are not uploaded
api.upload_folder(folder_path=evid, repo_id=repo, repo_type="model",
                  path_in_repo="_evidence")
print(f"  ✓ https://huggingface.co/{repo}")
PY
    [ $? -eq 0 ] || { bad "upload failed"; }
fi

# =============================================================================
echo
echo "============================================"
if [ "$fail" -eq 0 ]; then
    echo "✓ all passed ($warn warnings)"
    echo "  ckpt:  $CKPT"
    echo "  evidence:  $EVID"
    echo
    echo "Before destroying the pod, confirm: both the ckpt and the evidence directory have a copy outside the pod."
else
    echo "✗ $fail items failed — do not destroy the pod, investigate first"
fi
echo "============================================"
exit "$fail"
