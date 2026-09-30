#!/bin/bash
# =============================================================================
# FINALIZE.sh —— SFT 训练结束后的收尾(4× A6000)。
#
# 为什么需要它:pod 一销毁,能**证明这次训练跑对了**的东西就全没了,
# 而这个 checkpoint 是后续 RL 两臂对比的共同起点 π_ref —— 口径存疑,整条 C 路作废。
# ckpt 本身只是其中一件。
#
# 用法:
#     bash FINALIZE.sh                              # 只验收 + 收集证据(不删不传)
#     PRUNE=1 bash FINALIZE.sh                      # 额外删掉中间 checkpoint
#     HF_REPO=用户名/spacetools-sft-v1-4xa6000 \
#     HF_TOKEN=hf_xxx PRUNE=1 bash FINALIZE.sh      # 全套:验收 → 收集 → 清理 → 上传
#
#     bash FINALIZE.sh /path/to/experiments/sft_v1_20260908_0130   # 手动指定实验目录
#
# 默认不做任何破坏性操作。删除和上传都要显式开关。
# =============================================================================

set -uo pipefail

# ---- 环境 -------------------------------------------------------------------
# ⚠️ 不要 `export PATH=/opt/conda-st/bin:$PATH` —— 若调用方已激活 spacetools-sft,
# 这会把 base 的 bin 前置,而随后的 `conda activate` 因「已激活」变成空操作,
# python3 于是解析到 base。后果不对称:第 1–4 节只用 stdlib,在 base 下照样全绿;
# 只有第 5 节的 import torch 和第 8 节的 huggingface_hub 会失败 ——
# 得到「验收全过 + 证据不全」这种最容易误判的组合。
eval "$(/opt/conda-st/bin/conda shell.bash hook)" 2>/dev/null
conda activate spacetools-sft 2>/dev/null
# 断言:环境变量会骗人,以能不能 import 为准
PY3="$(command -v python3)"
case "$PY3" in
  /opt/conda-st/envs/spacetools-sft/bin/python3) ;;
  *) echo "✗ python3 解析到 $PY3,不是 spacetools-sft 的。PATH 被污染,拒绝启动。"; exit 1 ;;
esac
python3 -c "import torch, huggingface_hub" 2>/dev/null || {
    echo "✗ $PY3 导不进 torch/huggingface_hub,环境没激活好,拒绝启动。"; exit 1; }

REPO_DIR="${REPO_DIR:-/workspace/SpaceTools-SFT}"   # LLaMA-Factory fork(取 git 信息用)
# ⚠️ run_sft.sh 里的 REPO_DIR 是 LLAMA_DIR/.. = /workspace,不是 fork 目录本身。
# 实验目录默认落在 /workspace/experiments/,而我们这次显式传了 OUTPUT_DIR=.../full。
EXP_ROOT="${EXP_ROOT:-/workspace}"
HANDOFF="${HANDOFF:-/workspace/runpod-handoff}"     # 训练日志实际所在
PRUNE="${PRUNE:-0}"
KEEP_MID="${KEEP_MID:-1500}"     # 除最终 ckpt 外额外保留的那一步(做 sanity 对比);设 0 表示不留

fail=0
warn=0
ok()   { echo "  ✓ $*"; }
bad()  { echo "  ✗ $*"; fail=$((fail+1)); }
note() { echo "  ⚠ $*"; warn=$((warn+1)); }

# ---- 定位实验目录 -----------------------------------------------------------
if [ $# -ge 1 ]; then
    EXP="$1"
else
    EXP=$(ls -1dt "$EXP_ROOT"/experiments/sft_v1_* "$EXP_ROOT"/experiments/full 2>/dev/null | head -1)
fi
[ -n "${EXP:-}" ] && [ -d "$EXP" ] || { echo "✗ 找不到实验目录。用 'bash FINALIZE.sh <目录>' 指定"; exit 1; }

CKPT="$EXP/sft_checkpoint"
DATA="$EXP/sft_data"
STAMP=$(date +%Y%m%d_%H%M%S)
EVID="$EXP/EVIDENCE_$STAMP"

echo "实验目录: $EXP"
echo "checkpoint: $CKPT"
echo

# =============================================================================
# 1. batch 口径验收 —— 这是最关键的一项
#
#    train_samples_per_second × train_runtime = 这次训练总共喂进去的样本次数
#                                             = max_steps × 全局 batch 8
#
#    ⚠️ 不要看 epoch。run_sft.sh 的 DATASET_SPEC 把数据集注册了 3 遍
#    (DATASET_SPEC="${NAME},${NAME},${NAME}"),HF 的 epoch 是按 7020×3=21060 算的,
#    看起来永远对不上,是假警报。
# =============================================================================
echo "=== 1. 全局 batch 口径 ==="
python3 - "$EXP" "$CKPT" <<'PY'
import json, os, sys, re
exp, ckpt = sys.argv[1], sys.argv[2]

# 从实际生成的配置里读,不要用假设值
cfg = os.path.join(exp, "sft_config.yaml")
vals = {}
if os.path.exists(cfg):
    for line in open(cfg):
        m = re.match(r'\s*(max_steps|per_device_train_batch_size|gradient_accumulation_steps|learning_rate|deepspeed|cutoff_len):\s*(\S+)', line)
        if m:
            vals[m.group(1)] = m.group(2)
else:
    print("  ✗ 找不到 sft_config.yaml"); sys.exit(1)

steps = int(vals.get("max_steps", 0))
pd    = int(vals.get("per_device_train_batch_size", 0))
ga    = int(vals.get("gradient_accumulation_steps", 0))
print(f"  配置: max_steps={steps} per_device={pd} ga={ga} lr={vals.get('learning_rate')} "
      f"deepspeed={os.path.basename(vals.get('deepspeed',''))} cutoff_len={vals.get('cutoff_len')}")

res = os.path.join(ckpt, "all_results.json")
if not os.path.exists(res):
    print("  ✗ 没有 all_results.json —— 训练没正常结束"); sys.exit(1)
r = json.load(open(res))
rt  = r.get("train_runtime")
sps = r.get("train_samples_per_second")
if rt is None or sps is None:
    print(f"  ✗ all_results.json 缺 train_runtime / train_samples_per_second: {list(r)}"); sys.exit(1)

seen = rt * sps
want = steps * 8
err  = abs(seen - want) / want
print(f"  train_runtime           {rt:.1f} s  ({rt/3600:.2f} h)")
print(f"  train_samples_per_second {sps:.4f}")
print(f"  样本次数 = {seen:.0f}   期望 = {steps} × 8 = {want}   偏差 {err*100:.2f}%")
if err < 0.01:
    print(f"  ✓ 全局 batch 确为 8,与论文口径一致")
else:
    print(f"  ✗ 对不上。每步实际喂了 {seen/steps:.2f} 条,不是 8 —— 这个 ckpt 不能用作 π_ref")
    sys.exit(1)
print(f"  train_loss              {r.get('train_loss')}")
print(f"  (epoch 字段 = {r.get('epoch')},按 7020×3=21060 计,不用管)")
PY
[ $? -eq 0 ] || fail=$((fail+1))

# =============================================================================
# 2. 训练过程健康度
# =============================================================================
echo
echo "=== 2. loss 轨迹 ==="
python3 - "$CKPT" <<'PY'
import json, os, sys, math
ckpt = sys.argv[1]
p = os.path.join(ckpt, "trainer_state.json")
if not os.path.exists(p):
    print("  ✗ 没有 trainer_state.json"); sys.exit(1)
st = json.load(open(p))
hist = [h for h in st.get("log_history", []) if "loss" in h]
ev   = [h for h in st.get("log_history", []) if "eval_loss" in h]
if not hist:
    print("  ✗ log_history 里没有 loss"); sys.exit(1)
first, last = hist[0], hist[-1]
print(f"  记录点 {len(hist)} 个,最后一步 global_step={st.get('global_step')}/{st.get('max_steps')}")
print(f"  loss  step {first['step']}: {first['loss']:.4f}  →  step {last['step']}: {last['loss']:.4f}")
bad = 0
if st.get("global_step") != st.get("max_steps"):
    print(f"  ✗ 没跑满:{st.get('global_step')} / {st.get('max_steps')}"); bad += 1
nan = [h['step'] for h in hist if not math.isfinite(h['loss'])]
if nan:
    print(f"  ✗ 有 {len(nan)} 个非有限 loss,首次在 step {nan[0]}"); bad += 1
if last['loss'] >= first['loss']:
    print(f"  ⚠ 末尾 loss 没低于开头 —— 不一定是错,但值得看一眼曲线")
if ev:
    print(f"  eval_loss  首 {ev[0]['eval_loss']:.4f} → 末 {ev[-1]['eval_loss']:.4f}  ({len(ev)} 次)")
    if ev[-1]['eval_loss'] > ev[0]['eval_loss']:
        print(f"  ⚠ eval_loss 上升 —— 可能过拟合(val_size 只有 20,噪声很大,别过度解读)")
else:
    print("  ⚠ 没有 eval 记录")
if not bad:
    print("  ✓ 训练跑满且 loss 有限")
sys.exit(1 if bad else 0)
PY
[ $? -eq 0 ] || fail=$((fail+1))

# =============================================================================
# 3. checkpoint 文件完整性
#
#    run_sft.sh 的 Phase 3 已经会:删 config.json 里的 text_config、
#    置 tie_word_embeddings=true、从 base model 拷 preprocessor_config.json、
#    拷一份 toolshed_config.yaml 进来。这里是**复核 Phase 3 真的跑过了** ——
#    如果训练是手工重启的,Phase 3 可能被跳过,而这在加载时不会报错,
#    只会让 RL 侧拿到一个 tie_word_embeddings 不对的模型。
# =============================================================================
echo
echo "=== 3. checkpoint 完整性 ==="
[ -d "$CKPT" ] || { bad "checkpoint 目录不存在"; }
# chat_template.json 是 processor 级模板,sglang 加载 VLM 时读它。
# run_sft.sh 的 Phase 3 只从 base model 拷了 preprocessor_config.json,没拷它 ——
# 缺了 RL 侧会崩,而 tokenizer 侧的 chat_template.jinja 存在会让下面的模板检查照样变绿。
for f in config.json generation_config.json tokenizer_config.json \
         preprocessor_config.json chat_template.json toolshed_config.yaml; do
    [ -f "$CKPT/$f" ] && ok "$f" || bad "缺 $f"
done
# tokenizer 本体:tokenizer.json 或 vocab+merges 二选一
if [ -f "$CKPT/tokenizer.json" ] || [ -f "$CKPT/vocab.json" ]; then
    ok "tokenizer 本体"
else
    bad "缺 tokenizer.json / vocab.json"
fi
# 权重
NSAFE=$(ls -1 "$CKPT"/*.safetensors 2>/dev/null | wc -l)
if [ "$NSAFE" -gt 0 ]; then
    WSZ=$(du -ch "$CKPT"/*.safetensors 2>/dev/null | tail -1 | cut -f1)
    ok "权重分片 $NSAFE 个,合计 $WSZ"
    [ "$NSAFE" -gt 1 ] && { [ -f "$CKPT/model.safetensors.index.json" ] && ok "index.json" || bad "多分片但缺 index.json"; }
else
    bad "没有 .safetensors 权重"
fi

python3 - "$CKPT" <<'PY'
import json, os, sys
ckpt = sys.argv[1]
p = os.path.join(ckpt, "config.json")
if not os.path.exists(p): sys.exit(1)
c = json.load(open(p))
bad = 0
# Phase 3 的两处改动
if "text_config" in c:
    print("  ✗ config.json 仍含 text_config —— Phase 3 没跑,RL 侧加载会出问题"); bad += 1
else:
    print("  ✓ text_config 已移除")
if c.get("tie_word_embeddings") is not True:
    print(f"  ✗ tie_word_embeddings = {c.get('tie_word_embeddings')},应为 true —— Phase 3 没跑"); bad += 1
else:
    print("  ✓ tie_word_embeddings = true")
print(f"  ✓ dtype {c.get('torch_dtype')} · {c.get('model_type')}")
# chat template:qwen2_vl 通常内嵌在 tokenizer_config.json
tc = os.path.join(ckpt, "tokenizer_config.json")
has_tpl = os.path.exists(os.path.join(ckpt, "chat_template.jinja"))
if not has_tpl and os.path.exists(tc):
    has_tpl = "chat_template" in json.load(open(tc))
print(("  ✓ chat template 存在" if has_tpl else
       "  ✗ 找不到 chat template(既没有 chat_template.jinja,tokenizer_config.json 里也没有)"))
sys.exit(1 if (bad or not has_tpl) else 0)
PY
[ $? -eq 0 ] || fail=$((fail+1))

# =============================================================================
# 4. 数据口径 —— 这次到底训的是哪 7020 条、system prompt 长什么样
#
#    没有这一节,半年后没人能证明这个 ckpt 用的是 v1 的 11 个 schema。
# =============================================================================
echo
echo "=== 4. 数据口径 ==="
python3 - "$DATA" "$EVID" <<'PY'
import json, os, sys, hashlib, re
data, evid = sys.argv[1], sys.argv[2]
j = os.path.join(data, "data", "train.json")
if not os.path.exists(j):
    print(f"  ⚠ 找不到 {j} —— 数据可能已被清掉,跳过(不算失败)"); sys.exit(0)
d = json.load(open(j))
sysprompts = {it.get("system", "") for it in d}
print(f"  样本数 {len(d)}  (v1 期望 7020 = 7907 − 887 条 robot 工具样本)")
if len(d) != 7020:
    print(f"  ⚠ 不是 7020 —— 确认 VERSION 是不是 v1")
if len(sysprompts) == 1:
    sp = next(iter(sysprompts))
    h = hashlib.sha256(sp.encode()).hexdigest()
    n = len(re.findall(r'"name"\s*:', sp))
    print(f"  ✓ system prompt 全表一致  {len(sp)} 字符  sha256 {h[:16]}")
    print(f"    工具 schema 数 ≈ {n}  (v1 期望 11)")
    os.makedirs(evid, exist_ok=True)
    open(os.path.join(evid, "system_prompt.txt"), "w").write(sp)
    json.dump({"n_samples": len(d), "system_sha256": h, "system_chars": len(sp),
               "n_tool_schemas": n},
              open(os.path.join(evid, "data_provenance.json"), "w"), indent=2)
else:
    print(f"  ✗ system prompt 有 {len(sysprompts)} 个不同版本 —— 数据准备出了问题"); sys.exit(1)
PY
[ $? -eq 0 ] || fail=$((fail+1))

# =============================================================================
# 5. 收集证据(几百 KB,但只此一份)
# =============================================================================
echo
echo "=== 5. 收集证据 → $EVID ==="
mkdir -p "$EVID"
for f in trainer_state.json all_results.json train_results.json eval_results.json \
         trainer_log.jsonl training_loss.png training_eval_loss.png; do
    [ -f "$CKPT/$f" ] && { cp "$CKPT/$f" "$EVID/"; echo "  + $f"; }
done
for f in sft_config.yaml tool_config.yaml; do
    [ -f "$EXP/$f" ] && { cp "$EXP/$f" "$EVID/"; echo "  + $f"; }
done
# 训练时的启动日志(含那行 "GPU 数 N · per_device=X · ga=Y · 全局 batch 8 ✓")
# ⚠️ 训练日志不在 EXP 下,而在 handoff 目录 —— 那行
# 「GPU 数 N · per_device=X · ga=Y · 全局 batch 8 ✓」正是本脚本反复强调的口径证据。
for f in "$EXP"/*.log "$HANDOFF"/probe*.log "$HANDOFF"/launcher*.log; do
    [ -f "$f" ] && { cp "$f" "$EVID/"; echo "  + $(basename "$f")"; }
done
# 现场快照
{
  echo "# FINALIZE 快照  $(date -u +%FT%TZ)"
  echo; echo "## GPU"
  nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total --format=csv
  echo; echo "## 关键包版本"
  python3 -c "import torch,transformers,deepspeed,flash_attn,llamafactory as L; \
print('torch',torch.__version__); print('transformers',transformers.__version__); \
print('deepspeed',deepspeed.__version__); print('flash_attn',flash_attn.__version__); \
print('llamafactory',L.__version__)"
  echo; echo "## git"
  git -C "$REPO_DIR" log -1 --format='SpaceTools-SFT %H %ci %s' 2>/dev/null
  echo; echo "## checkpoint 清单"
  ls -la "$CKPT"
} > "$EVID/SNAPSHOT.txt" 2>&1
echo "  + SNAPSHOT.txt"
( cd "$CKPT" && sha256sum ./*.safetensors ./*.json 2>/dev/null ) > "$EVID/CKPT_SHA256SUMS" 2>/dev/null
echo "  + CKPT_SHA256SUMS"
echo "  证据目录 $(du -sh "$EVID" | cut -f1)"

# =============================================================================
# 6. 清理中间 checkpoint(需要 PRUNE=1)
#
#    sft_config.yaml 里 save_steps=500 且没有 save_total_limit
#    → 3000 步会留下 6 个中间 ckpt,每个约 7.6 GB(bf16)。
#    save_only_model=true 所以里面没有优化器态,删掉不影响任何续训能力
#    —— 本来就不能续训。
# =============================================================================
echo
echo "=== 6. 中间 checkpoint ==="
MIDS=$(ls -1d "$CKPT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n)
if [ -z "$MIDS" ]; then
    echo "  (没有中间 checkpoint)"
else
    echo "$MIDS" | while read -r m; do echo "  $(du -sh "$m" | cut -f1)  $(basename "$m")"; done
    TOT=$(du -sh -c $MIDS 2>/dev/null | tail -1 | cut -f1)
    echo "  合计 $TOT"
    if [ "$PRUNE" = "1" ] && [ "$fail" -ne 0 ]; then
        echo "  拒绝删除 —— 前面有 $fail 项没通过。最终权重是否可用还没确认,"
        echo "  这时候删中间 ckpt 可能把唯一能用的那份也删了。先把问题查清楚。"
    elif [ "$PRUNE" = "1" ]; then
        echo "  PRUNE=1 → 保留 checkpoint-$KEEP_MID,其余删除"
        echo "$MIDS" | while read -r m; do
            if [ "$(basename "$m")" = "checkpoint-$KEEP_MID" ]; then
                echo "    保留 $(basename "$m")"
            else
                rm -rf "$m" && echo "    删除 $(basename "$m")"
            fi
        done
        df -h "$CKPT" | tail -1 | sed 's/^/    /'
    else
        echo "  未删除。要删就 PRUNE=1 重跑(最终权重在 $CKPT 根目录,不受影响)"
    fi
fi

# =============================================================================
# 7. 格式冒烟 + base 对照(SMOKE=0 可跳过,约 3–5 分钟,占 1 张卡)
#
#    抓的是这一类静默失败:loss 曲线完全正常,但模型根本没学会发工具调用。
#    前六节全过也照样可能是这种情况 —— 那几节只查工程,不查模型学到了什么。
#    ⚠️ 这不是能力度量,样本取自训练集。真实分数只有 run_eval.sh 能给。
# =============================================================================
echo
echo "=== 7. 格式冒烟 ==="
SMOKE="${SMOKE:-1}"
SMOKE_PY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/smoke_check.py"
TRAIN_JSON="$DATA/data/train.json"
if [ "$SMOKE" != "1" ]; then
    echo "  SMOKE=0,跳过"
elif [ ! -f "$SMOKE_PY" ]; then
    note "找不到 smoke_check.py(应与本脚本同目录),跳过"
elif [ ! -f "$TRAIN_JSON" ]; then
    note "找不到 $TRAIN_JSON(数据已清掉?),跳过"
else
    mkdir -p "$EVID"
    python3 "$SMOKE_PY" "$CKPT" "$TRAIN_JSON" --n "${SMOKE_N:-12}" \
            --base "${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}" \
            --out "$EVID/smoke_check.json" 2>&1 | sed 's/^/  /'
    rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then ok "格式冒烟通过"
    else note "格式冒烟未通过 —— 看上面的原始输出。不阻断上传,但全量评测之前要弄清楚"; fi
fi

# =============================================================================
# 8. 上传(需要 HF_REPO + HF_TOKEN)
# =============================================================================
echo
echo "=== 8. 上传 ==="
if [ "$fail" -ne 0 ]; then
    echo "  跳过 —— 前面有 $fail 项没通过,先把问题查清楚再传"
elif [ -z "${HF_REPO:-}" ] || [ -z "${HF_TOKEN:-}" ]; then
    echo "  跳过 —— 没给 HF_REPO / HF_TOKEN。要传就:"
    echo "    HF_REPO=<用户名>/spacetools-sft-v1-4xa6000 HF_TOKEN=hf_xxx bash FINALIZE.sh"
    echo "  仓库会建成 private。也可以先手动:"
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
                  ignore_patterns=["checkpoint-*/**"])   # 中间 ckpt 不传
api.upload_folder(folder_path=evid, repo_id=repo, repo_type="model",
                  path_in_repo="_evidence")
print(f"  ✓ https://huggingface.co/{repo}")
PY
    [ $? -eq 0 ] || { bad "上传失败"; }
fi

# =============================================================================
echo
echo "============================================"
if [ "$fail" -eq 0 ]; then
    echo "✓ 全部通过($warn 条提醒)"
    echo "  ckpt:  $CKPT"
    echo "  证据:  $EVID"
    echo
    echo "销毁 pod 之前确认:ckpt 和证据目录都已经在 pod 之外有一份。"
else
    echo "✗ $fail 项未通过 —— 不要销毁 pod,先查"
fi
echo "============================================"
exit "$fail"
