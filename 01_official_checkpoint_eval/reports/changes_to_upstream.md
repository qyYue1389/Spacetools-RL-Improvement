# Every change we made, and why

Five locations were touched to get the SpaceTools eval running on 4x A6000,
and §6 records what the move to 4x A100-SXM4-40GB (sm_80) then required.
This file is the complete list. `PROVENANCE.txt` carries the same information in
condensed form alongside weight commits and the environment matrix; this file is
the long-form version with the reasoning.

Base points (the upstream state each repo was on before we touched it):

| location            | branch          | upstream base | net diff                       |
|---------------------|-----------------|---------------|--------------------------------|
| SpaceTools-RL       | `repro-4xa6000` | `54270e82`    | 1 file, +55 -21, 6 commits     |
| SpaceTools-Toolshed | `repro-4xa6000` | `4f0512d`     | 2 files, +26 -8, 1 commit      |
| GraspGen            | `repro-4xa6000` | `2dd8852`     | 2 files, +4 -6, 1 commit       |
| RoboRefer           | `repro-4xa6000` | `d97a995`     | 2 files, +2 -2, 1 commit       |
| Official checkpoint | (not a repo)    | `f953b1a1`    | 2 config files, `.orig` kept   |

Read it with:

    git -C SpaceTools-RL              diff 54270e82..HEAD
    git -C SpaceTools-Toolshed        diff 4f0512d..HEAD
    git -C SpaceTools-Toolshed/GraspGen  diff main..HEAD
    git -C SpaceTools-Toolshed/RoboRefer diff main..HEAD
    cd /workspace/models/spacetools-ckpt && diff <(python3 -m json.tool config.json.orig) <(python3 -m json.tool config.json)

---

## 1. SpaceTools-RL

One file, `examples/toolshed/run_eval.sh`, across six commits.

### 1.1 BENCHMARKS parquet paths  (`f7560170`)

    - [robospatial]="robospatial_home_multiturn/test.parquet"
    + [robospatial]="data/robospatial.parquet"
      ... all nine keys

**Why.** The hardcoded paths do not match the layout of
`siyich/spacetools-eval-benchmarks` on HuggingFace. The dataset was evidently
reorganised after the script was written, and the benchmark keys already map
one-to-one onto the new filenames.

**Without it** line 113 exits 1 with `Missing: .../robospatial_home_multiturn/test.parquet`.
The eval does not start at all.

**Verified** against real data: all nine parquets resolve.
**Affects numbers:** no. Same files, different path.

### 1.2 NUM_GPUS 8 -> 4  (`f7560170`)

Upstream defaults assume 2x8 A100. We have 4x A6000. Telling Ray there are 8
GPUs on a 4-GPU box over-subscribes the scheduler.

**Affects numbers:** no.

### 1.3 DATA_DIR override  (`10037a43`)

    + if [ -z "${DATA_DIR:-}" ]; then
          <snapshot_download block>
          DATA_DIR="$EVAL_DIR/benchmarks"
    + else
    +     echo "Using preset DATA_DIR=$DATA_DIR (skipping benchmark download)"
    + fi

**Why, two reasons.** `EVAL_DIR` is timestamped, and `snapshot_download` is given
both `cache_dir` and `local_dir` inside it, so every run re-downloads the full
292 MB. And `run_eval.sh` has no sample-count option at all -- `data.val_files`
points straight at a parquet -- so the only way to run a subset is to hand it a
truncated parquet, which requires being able to point `DATA_DIR` somewhere.

Every P2 smoke run depended on this.

**Revertible:** yes, at the cost of losing subset runs and re-downloading each time.
**Affects numbers:** no.

### 1.4 Load conda's shell hook  (`a71f20f1`)

    + if ! declare -F conda >/dev/null 2>&1; then
    +     ... . "$_conda_base/etc/profile.d/conda.sh"
    + fi

**Why.** Line 105 calls `conda activate`, which is a shell *function*. A
non-interactive `bash run_eval.sh` inherits exported variables but not shell
functions, so it fails with `CondaError: Run 'conda init' before 'conda activate'`.
Upstream presumably always ran it from a conda-init'd shell.

**Affects numbers:** no. Pure portability.

### 1.5 dtype 'float16' -> 'auto'  (`f68fd4c6`)

This one looks like a behaviour change and is the opposite.

Upstream `vlm.py` hardcodes `torch_dtype="auto"` and ignores its own `dtype`
argument, so the `'float16'` in the config is dead code. `"auto"` loads Molmo at
its stored precision, which is fp32 on the Hub -- measured at 33 GB, not the
~15 GB fp16 would imply. **The paper's numbers were therefore produced in fp32.**

Once we fixed `vlm.py` (see 2.2) the config would have started taking effect,
which would have moved us *away* from the paper. Writing `'auto'` explicitly
keeps runtime behaviour byte-identical to upstream.

**Affects numbers:** no -- this is what keeps them unchanged.

### 1.6 Tool GPU fractions and EVAL_GPUS  (`f68fd4c6` then `910f6f5d`)

| tool               | upstream | interim | final |
|--------------------|---------:|--------:|------:|
| `vlm`              |      0.5 |    0.55 |   1.0 |
| `roborefer`        |      0.5 |    0.55 |   0.6 |
| `sam2` x2          |     0.15 |    0.15 |   0.2 |
| `depth_estimator` x2 |   0.15 |    0.15 |   0.4 |
| `bounding_box` x2  |      0.1 |     0.1 |  0.05 |
| total              |      1.9 |     2.0 |   3.0 |

plus `EVAL_GPUS` 4 -> 1.

**Why, in two rounds.** Ray PACKS fractional GPUs rather than spreading them, so
upstream's `roborefer 0.5 + vlm 0.5 = 1.0` put both large models on GPU 0:
18.6 GB + 33 GB against a 48 GB card. The interim 0.55/0.55 made them mutually
exclusive (1.1 > 1.0).

That was not enough. Measured footprints turned out to be much larger than the
plan assumed -- DepthPro is 12.5 GB PER ACTOR, not the ~4 GB estimated, and with
`num_actors: 2` that is 25 GB. Tools total ~80 GB against 94.8 GB on two cards:
84% occupancy with no headroom for activation spikes. With
`max_parallel_calls=8`, a 124-sample blinkdepth run OOMed 68 times. The affected
13 samples scored 38.5% against 88.3% for the clean ones, dragging the benchmark
from 88.3% to 83.1% -- **a silently wrong number, not a crash**.

The final split gives the tools three cards: Molmo alone, RoboRefer + SAM2,
DepthPro + bbox + grasp. Every upstream replica count is preserved; only
`EVAL_GPUS`, which was ours to choose, changed. The cost is one data-parallel
policy replica instead of two.

**Affects numbers:** no. Fractions are Ray bookkeeping, not memory limits, and
they only decide placement.

### 1.7 expandable_segments deliberately NOT set  (`77141944` then `ef2c7580`)

Added on the strength of the OOM traces' own recommendation, then reverted:
sglang's HYBRID rollout uses TorchMemorySaver to release and rebuild the KV
cache, and that refuses to run under it:

    RuntimeError: TorchMemorySaver is disabled for the current process because
    expandable_segments is not supported yet.

Only a comment survives, explaining why it must not be re-added.

---

## 2. SpaceTools-Toolshed

Two files, one commit (`712e557`).

### 2.1 `graspgen_franka_panda.yml` -- three checkpoint paths made absolute

    - checkpoint: ../../checkpoints/GraspGenModels/checkpoints/graspgen_franka_panda_dis.pth
    + checkpoint: /workspace/models/GraspGenModels/checkpoints/graspgen_franka_panda_dis.pth

Lines 170, 171 (`discriminator:` section) and 217 (`eval:` section).

**Why.** The paths are relative, and `grasp_gen` has no logic that resolves them
against the config file's location -- `from_config` passes the string straight to
`torch.load()`, so they resolve against the process CWD. Inside a Ray actor that
is not under our control.

Upstream is also self-contradictory about the location: this yml says
`../../checkpoints/`, `requirements/tool-graspgen.txt` says `toolshed/data/`, and
`install_graspgen.sh` uses `$MODELS_DIR`. We clone where the install script puts
it and point the yml there.

Semantics were checked before editing: 170/171 are the discriminator's own weights
plus the generator's object encoder used as pretrained init; 217 is the generator
used for eval sampling. The diff touches exactly those three lines.

**Verified** in the load logs, which show the new paths being read.
**Affects numbers:** no. Same weights, findable.

### 2.2 `vlm.py:112` -- honour the `dtype` argument

    -     model_kwargs["torch_dtype"] = "auto"
    +     model_kwargs["torch_dtype"] = (
    +         "auto" if dtype == "auto" else getattr(torch, dtype)
    +     )

**Why.** The constructor advertises `dtype: str = "float16"` but the
non-quantised path ignores it entirely. That made the config lie, and hid the
fact that Molmo was running in fp32 at 33 GB -- which is what made GPU 0 OOM once
RoboRefer shared it.

**Affects numbers:** no, because `run_eval.sh` passes `'auto'`, which takes the
same branch as upstream. Fixing it turns a future precision ablation into a
one-string change instead of a rediscovery.

### 2.3 / 2.4 `vlm.py:326, 346, 364` -- cast float inputs to the model dtype

    + model_dtype = next(self._model.parameters()).dtype
    ...
    + if v.is_floating_point() and v.dtype != model_dtype:
    +     v = v.to(model_dtype)

**Why.** The tensor-moving code only did `.to(model_device)` -- device, never
dtype. Harmless while the model was always fp32; the moment the model was fp16 it
failed with `expected mat1 and mat2 to have the same dtype, but got: float != c10::Half`.

**This is a second upstream bug that 2.2 was masking.** Neither fires as long as
everything stays fp32, at the cost of doubling the VRAM. Only floating tensors
are cast, so `input_ids` and `image_input_idx` stay integer. Both the normal and
the OOM-recovery path needed it.

**Affects numbers:** no. Under fp32 the condition never holds; it is a no-op.

---

## 3. GraspGen

One commit (`cb846e1`). **Only one of the five hunks in the diff is ours.**

### 3.1 Drop `pickle5`  (ours)

    # pyproject.toml
    -     "pickle5",
    # requirements.txt
    - pickle5

**Why.** `pickle5` backports pickle protocol 5 to Python 3.6/3.7. Protocol 5 has
been in the standard library since 3.8, and the private C API the backport uses
(`_PyObject_CallNoArg`, `_PySys_GetObjectId`, `Py_SIZE` as an lvalue) was removed
in 3.11, so it cannot build:

    pickle5/_pickle.c:6178:9: error: lvalue required as left operand of assignment
            Py_SIZE(self->stack) = len;

There is no py3.11 wheel either -- the package was last released in 2020.

Removing it is behaviour-preserving: the only import site already guards it.

    try:
        import pickle5 as pickle
    except:
        import pickle

**Revertible:** no. It literally cannot compile.
**Affects numbers:** no.

### 3.2 torch / torchvision / numpy / spconv pins  (NOT ours)

    - "torch==2.1.0"          + "torch>=2.3.0,<2.4"
    - "torchvision==0.16.0"   + "torchvision>=0.18.0,<0.19"
    - "numpy==1.26.4"         + "numpy>=2.0"
    - "spconv-cu120"          + "spconv-cu121"

`install_graspgen.sh` lines 94-98 apply these with `sed` on every run. Upstream's
own comment explains it: torch 2.1 cannot interoperate with numpy 2.x, and 2.3
restores compatibility. They are committed only so the working tree is clean.

**Knock-on effect worth remembering.** These pins put GraspGen on
`torch>=2.3,<2.4`, which collides with the `torch==2.9.1` in our
`tool-constraints.txt`:

    ERROR: Cannot install grasp-gen==1.0.0 ...
        grasp-gen 1.0.0 depends on torch<2.4 and >=2.3.0
        The user requested (constraint) torch==2.9.1

So `PIP_CONSTRAINT` must be unset when building that environment. PyG's
`torch-cluster` / `torch-scatter` wheels only exist for `2.3.0+cu121`, which locks
it further. This is why five environments carry four different torch versions.

---

## 4. RoboRefer

One commit (`7c50d8a`). One of the two hunks is ours.

### 4.1 Skip the hardcoded py3.10 flash-attn wheel  (ours)

    - pip install https://.../flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    + # patched: flash-attn is installed separately, matched to the torch this env ends up with

**Why.** The URL pins Python 3.10, torch 2.3 and cu122 all at once. This
environment is Python 3.11, so pip refuses outright ("not a supported wheel on
this platform"), and `set -e` then aborts the script before line 24 -- meaning
`pip install -e ".[train,eval]"` never runs and **llava is never installed**.

flash-attn is installed separately instead. That took two attempts: the first
used a wheel built for torch 2.9.1, but line 24 then pulled VILA's pins and
downgraded torch to 2.5.1, breaking the C++ ABI
(`undefined symbol: _ZN3c105ErrorC2E...`). The second used the official
`+cu12torch2.5cxx11abiFALSE-cp311` build, after checking
`torch._C._GLIBCXX_USE_CXX11_ABI` is `False`.

**Revertible:** no. Hard blocker.
**Affects numbers:** no. Same flash-attn version, ABI-matched.

### 4.2 Disable `lighteval`  (NOT ours)

`install_roborefer.sh` line 108 comments it out with `sed`; upstream's own note
says the dependency hangs during installation. Committed only so the tree is clean.

### 4.3 A patch we made and then reverted

`pip install triton==3.1.0` on line 27 was briefly commented out, on the theory
that it would break torch 2.9.1 (which requires triton 3.5.1).

**That reasoning was wrong.** Line 24 runs first and already downgrades torch to
2.5.1, whose own pin IS triton 3.1.0 -- so the line is a no-op in the real
execution order. Restored to upstream.

The mistake is worth recording because it recurred: inferring whole-system
behaviour from one component's current state, instead of walking the actual
execution order.

### Knock-on effect

`pip install -e ".[train,eval]"` downgrades this environment wholesale:

| package      | before        | after         |
|--------------|---------------|---------------|
| torch        | 2.9.1+cu128   | 2.5.1+cu124   |
| transformers | 4.57.1        | 4.49.0        |
| numpy        | 2.4.6         | 1.26.4        |

Not a fault -- VILA/llava is pinned to that stack, and Toolshed's per-tool conda
environments exist precisely to absorb this. But it makes roborefer the only
environment on transformers 4.49, which is worth remembering during attribution.

It also explains why the driver must be on numpy 2.x: roborefer produces arrays
under numpy 1.26.4, and numpy 2 can unpickle numpy-1 arrays while the reverse
fails. The consumer has to be at least as new as every producer.

---

## 5. Official checkpoint config

**The only place where we modified released weights rather than code.** Applied by
`/workspace/fix_checkpoint.py`; originals kept as `config.json.orig` and
`preprocessor_config.json.orig`. Those backups are not in any git -- keep them.

### 5.1 `config.json` -- remove `text_config`

    text_config = <dict, 26 keys, model_type='qwen2_5_vl_text'>   ->   absent

**Why.** With `text_config` present, transformers resolves `model_type` to the
inner `qwen2_5_vl_text` (the text submodel) instead of the top-level
`qwen2_5_vl`, and sglang refuses to load:

    RuntimeError: Unimplemented model type: qwen2_5_vl_text

`docs/SETUP.md:112-116` documents this exact error and fix, and says `run_sft.sh`
applies it automatically -- but that is for checkpoints you train yourself. The
released `siyich/spacetools-ckpt` ships unfixed.

**Revertible:** no. Hard blocker.

### 5.2 `config.json` -- set `tie_word_embeddings = True`

    absent   ->   True

**Why.** `run_sft.sh` lines 273-274 do this, and the base model
`Qwen/Qwen2.5-VL-3B-Instruct` also has `True`.

**This one was not copied blindly.** If the output head had been trained
separately, setting `True` would make transformers discard the trained `lm_head`
and reuse the embedding -- the model would still run and every output would be
wrong, silently. So the two tensors were compared directly:

    model.embed_tokens.weight   (151936, 2048)
    lm_head.weight              (151936, 2048)
    identical: True | max abs diff: 0.000e+00

Bit-identical, so the head really is tied and the checkpoint merely materialised a
copy. `fix_checkpoint.py` performs this check itself rather than assuming; on an
untied head it skips the step and says so.

### 5.3 `preprocessor_config.json` -- replaced with the base model's

    image_processor_type:  Qwen2VLImageProcessorFast  ->  Qwen2VLImageProcessor
    keys:                  26                         ->  9

**Why.** `docs/SETUP.md:117-119`: the Fast processor produces a different image
token count than the vision encoder expects, giving an off-by-one:

    ValueError: Image features and image tokens do not match

This only surfaces after 5.1 is fixed and the model actually starts processing
images, so both were done together.

**Which version, and does it matter?** Nothing in the repo pins a revision --
`run_sft.sh:289` calls `hf_hub_download('$BASE_MODEL', 'preprocessor_config.json')`
with no `revision=`, so it always takes current main. That looked like an open
question, and it is not:

- our installed file is byte-identical to the Hub's (sha256 `f2058c716eef96cc...`, 350 bytes)
- Qwen has not touched `preprocessor_config.json` since commit `1b989f2c` on
  2025-02-15; the later commits only changed tokenizer_config, generation_config
  and the README
- the paper is arXiv 2512.04069, December 2025

So the authors necessarily fetched the same file, by the same unpinned call.
**This is not a source of numeric deviation.**

---

## 6. Phase M: migrating from A6000 (sm_86) to A100-SXM4-40GB (sm_80)

Done 2026-08-27. Two things actually changed; a third that looked like it would
need changing did not.

### 6.1 `pointnet2_ops` rebuilt for `8.0;8.6;8.9`

Deviation [3] built this GraspGen dependency with `TORCH_CUDA_ARCH_LIST=8.6`,
because torch defaults the arch list to the local device and `setup.py` pins
nothing. That produces sm_86 cubins with **no PTX fallback**.

CUDA runs a `sm_X.y` cubin on `sm_X.z` only when `z >= y`. A100 is 8.0, so this
was a certainty rather than a risk. Confirmed before touching anything:

    cuobjdump --list-elf _ext...so   ->  4x sm_86, no PTX
    furthest_point_sample(...)       ->  CUDA kernel failed :
                                         no kernel image is available for
                                         execution on the device

After rebuilding: 4 cubins each for sm_80, sm_86 and sm_89, and the op returns.
Recipe kept at `/workspace/rebuild_pointnet2_sm80.sh`.

**Three conda-injected build traps had to be neutralised again** — they are
properties of the environment, not of the previous machine, so they were all
still live:

| injected | effect | what we did |
|----------|--------|-------------|
| `NVCC_PREPEND_FLAGS=-ccbin=x86_64-conda-linux-gnu-c++` | pins nvcc's host compiler to conda's **gcc 15.3**; nvcc 12.8 refuses > 14 | point it at `/usr/bin/g++-13` |
| `CFLAGS`/`CXXFLAGS` | carry conda's CUDA **13.3** headers | `unset` |
| `LDFLAGS` | `-L$CONDA_PREFIX/lib` — see below | `unset` |

We did **not** use `-allow-unsupported-compiler`. NVIDIA warns it can produce
silently wrong results, which is unacceptable for operators that emit geometric
poses.

A fourth trap is ours, not conda's: **`setup.py`'s `build/` directory survives**,
and setuptools will happily relink the stale sm_86 `.o` files into a "rebuilt"
extension. The script deletes `build/` first. Without that the rebuild is a
silent no-op — the same failure shape as the OOM problem, and just as invisible.

### 6.2 Correction: deviation [3]'s cudart diagnosis was wrong

[3] recorded that the old build linked the env's `libcudart.so.13` because
`LIBRARY_PATH` pointed at `.../lib64/stubs`, which has no `libcudart`. That is
not the mechanism. **`LIBRARY_PATH` is only searched after explicit `-L`
flags**, and conda's `LDFLAGS` puts `-L$CONDA_PREFIX/lib` — which does contain
`libcudart.so.13` — *ahead* of torch's `-L/usr/local/cuda/lib64`. Setting
`LIBRARY_PATH` correctly changes nothing; unsetting `LDFLAGS` is what fixes it.

The rebuilt extension links `/usr/local/cuda/lib64/libcudart.so.12` (12.8.90).
Nothing is lost at runtime: `/usr/local/cuda/lib64` is on the system `ldconfig`
path, so dropping conda's rpath still resolves.

This was worth chasing even though [3] called it harmless. It was harmless in
the narrow sense — cubins are fixed at compile time — but it meant every process
loading this extension had **two CUDA runtimes resident**, 12.8 via torch and
13.3 via pointnet2_ops. That is now gone.

The wider lesson is the one P2 already recorded under a different name: the
original note reasoned from one component's configuration (`LIBRARY_PATH` says
stubs) instead of tracing what the linker actually does with the whole command
line.

### 6.3 `p2_tool_chain.py`'s GPU fractions were stale — our bug, hidden by 48 GB

The tool-chain smoke test carries its own copy of `TOOL_CONFIGS`, with a
docstring promising it is "copied verbatim from `run_eval.sh`". It was not. It
still held the **interim** split — `vlm`/`roborefer` 0.55, `sam2`/`depth` 0.15 —
that §1.6 describes as superseded. It was never updated when `910f6f5d` moved
`run_eval.sh` to the final split.

On 48 GB cards that was survivable: Molmo 33.0 + DepthPro 12.5 + SAM2 1.2 =
46.7 GB fits in 48, so P2 ran green and nobody looked. On 40 GB it OOMs on the
very first depth call, with three processes on one card:

    torch.OutOfMemoryError: ... GPU 0 has a total capacity of 39.49 GiB of which
    98.31 MiB is free. Process ... 30.38 GiB ... 7.70 GiB ... 1.29 GiB

Now synced to `0.6 / 1.0 / 2x0.2 / 2x0.4 / 2x0.05 / 0 / 0.1 = 3.0`, identical to
`run_eval.sh`. Afterwards the chain scores **12/12** — better than the A6000's
11/12, where `grasp_generator` had failed on the kitchen photo as a scene
mismatch.

**Affects numbers:** no. `p2_tool_chain.py` is a diagnostic; it never produced a
reported result. But it would have wasted a day: it is the M3 gate, and a gate
that fails for a reason unrelated to the thing it guards is worse than no gate.

It also demonstrated the project's central failure mode live, which is worth
having seen rather than only read about. The OOM did not raise; the depth tool
returned **normally**, with the exception text in its `text` field and an empty
`variables` list:

    1/7  depth_estimator.estimate_depth_with_pointcloud
        elapsed: 1.2s
        text   : torch.OutOfMemoryError: CUDA out of memory. Tried to allocate...
        variables: []

The chain then carried on to tools 2-7 and printed a summary. Had this been the
eval rather than the smoke test, the model would have read that string as a tool
response, reasoned around it, and produced a plausible wrong answer. Exactly the
68-OOM cascade P2 describes, reproduced in miniature.

### 6.3b The checkout was stale, and the credentials were why

On arrival, `SpaceTools-RL`'s local branch ended at `ef2c7580` (patch 0007) and
`verl/trainer/ppo/ray_trainer.py` was byte-identical to upstream `54270e82` — so
the P3 uid/num_turns dump was not in the working tree, even though
`P4_HANDOFF.md`'s repo table lists it as present.

**It was present — on the remote.** It had been committed and pushed from the
A6000 host as `8d193ec3`, the same SHA that heads `patches/rl/0008`. The
volume's copy of the repo simply lagged the remote by that one commit.

The handoff's very first repository instruction is *"`git pull` all of them
first"*. That could not run. The remotes were `https://`, GitHub credentials had
lived on the previous host's **container** disk, and a recycle wipes container
disk — so every `git pull` died with `could not read Username for
'https://github.com'`, and the stale checkout went unnoticed. Fixing the
credentials (§6.3c) and pulling was the actual fix; my first attempt re-applied
the patch file by hand, which produced identical content under a different SHA
and had to be discarded in favour of the remote's commit.

**One trap worth recording, because I nearly fell in it.** `sample_uids` and
`sample_turns` appear all over `ray_trainer.py` *without* the patch — they are
verl's own variables, feeding the `val-aux/num_turns/*` metrics. What
`patches/rl/0008` adds is passing them into `_dump_generations`, which upstream
never does. Grepping for the names says "applied". The checks that tell the
truth are `git diff 54270e82..HEAD -- verl/trainer/ppo/ray_trainer.py` (empty
means absent) and `git apply --check` on the patch (applies cleanly means
missing).

That is the same failure mode P1 and P2 kept hitting, in a new costume: reading
one surface signal instead of the thing it stands for. Here the cost would have
been V2's `turn-count mismatch` check silently measuring nothing.

### 6.3c GitHub credentials, and why they vanished

Every `git fetch`/`pull`/`push` on this host failed with `could not read
Username for 'https://github.com'`. The cause is structural, not a
misconfiguration: **credentials lived on the previous host's container disk**
(`/root`, `/opt`, `/`), and recycling an instance rebuilds that from the image.
Only the mounted volume survives — which is why the conda envs, weights and
dumps all came across and the credentials did not.

No token was needed. `/root/.ssh/id_ed25519` (key comment
`vast-instance-c17760dabb09`) is **already authorised on the `qyYue1389`
account** — `ssh -T git@github.com` answers `Hi qyYue1389!`. Only the remote
scheme was wrong. Switching all five repos from `https://github.com/` to
`git@github.com:` fixed fetch, pull and push at once:

    git remote set-url mine git@github.com:qyYue1389/<repo>.git

**This is not durable.** That key is on `/root`, i.e. container disk, so a
recycle brings the same failure back — along with, once again, a checkout that
quietly lags the remote. Anyone who recycles this instance must redo the key (or
move it onto the volume, weighing that `/workspace` may be shared with other
instances on the same physical machine).

### 6.4 What did NOT need changing

**`run_eval.sh` needs no edit at all.** The fractional budget the P4 handoff
derives for 40 GB cards is byte-identical to what `910f6f5d` already committed
for the A6000 — `vlm 1.0` alone; `roborefer 0.6 + sam2 2x0.2`; `depth 2x0.4 +
bbox 2x0.05 + grasp 0.1`; `vision_ops 0`; tools 3.0 + policy 1.0 = 4.0. The
A6000 layout was already forced onto physical boundaries, and physical
boundaries are what changed size, not count. `EVAL_GPUS` stays 1.

**flash-attn in `spacetools-rl` needed no rebuild.** The handoff expected the
A6000 source build to have picked up only the local arch. It had not: 72 cubins,
**all sm_80**. It was running sm_80 code on the A6000 all along (legal, 8.6 >=
8.0) and is now an exact match for the hardware.

To be sure nothing else was lurking, every compiled extension in all five
environments was scanned with `cuobjdump` — **66 `.so` files, every one carrying
an sm_80 cubin**. `pointnet2_ops` was the only sm_80-incapable extension on the
machine.

**No `num_actors` touched**; all replica counts remain at upstream defaults.
**`dtype` stays `'auto'`** (Molmo fp32). Buying headroom with fp16 would be a
fidelity deviation, not a fix — see §1.5 and §2.2.

### 6.5 Placement, measured

With Toolshed up and every GPU tool warmed, otherwise idle:

| GPU | used | contents |
|-----|-----:|----------|
| 0 | 21020 MiB (20.5 GB) | RoboRefer 18.4 + SAM2 1.3 + SAM2 0.8 |
| 1 | 35337 MiB (34.5 GB) | **Molmo, alone** — 86.3% of 40 GB |
| 2 | 17931 MiB (17.5 GB) | DepthPro 12.2 + 4.0 + 1.3 |
| 3 | 4 MiB | free, reserved for the policy |

`ray status` reports **3.0/4.0 GPU**. Every per-process footprint matches P2's
A6000 table one-to-one (RoboRefer 18.6, Molmo 33.0, DepthPro 12.5, SAM2 1.2).

**A caveat on how this was established.** `nvidia-smi` reports *host* pids, and
this container cannot see its own host pids — `/proc/<pid>/status` has a single
`NSpid` entry — so per-process GPU rows **cannot be joined to Ray actors from
inside the container**. `CUDA_VISIBLE_DEVICES` is no help either: Ray sets it
after `exec`, so it is absent from `/proc/<pid>/environ`, which is a snapshot.
Placement is therefore established from three agreeing lines of evidence — `ray
status` at 3.0/4.0, exactly three occupied cards with process counts 3/1/3
matching the three groups, and footprints that identify each model uniquely —
and not from a direct pid join. That is weaker than a join and stronger than
reading the fractions, which is what the handoff warned against.

### 6.6 The open question this hands to V3

Molmo sits at **34.5 GB after a single `detect_one` on a 1.36 MP image**, above
the 33.0 GB in P2's table, leaving 5.6 GB. `robospatial` images are 2.76 MP.

Worth stating precisely, because it changes how V3 should be read: PyTorch's
caching allocator does not return freed blocks to the driver, so `nvidia-smi`
reports a **high-water mark of allocator reservation**, not live usage. That
makes `nvidia-smi` the right instrument for V3's peak hunt — but it also means
the 33.0 GB in P2's table was never a pure resting figure either. The honest
statement of the open question is not "resting or peak" but "how far does the
high-water mark climb at 2.76 MP", and V3 measures exactly that.

---

## 7. Phase V1: three bugs the parser regression test caught

`P2_RESULTS.md`'s hand-computed statistics for the blinkdepth **run4** dump are a
free regression test, and they earned their keep immediately. Run against the
parser as shipped, the numbers that come straight out of JSON fields matched
(108/124 = 87.10%, 16 wrong, 1 tool failure, zero OOM / truncation / max-turns)
and **every number that required actually parsing the transcript was wrong**.

That split is the tell. Score and wrong-count are read, not derived; chains,
tool histogram, turn counts and `no <answer>` are derived. Only the derived ones
broke.

### 7.1 The splitter silently dropped assistant turn 1

`parse_transcript` splits `output` on `<|im_start|>` and reads each chunk's first
line as a role. But verl dumps `output` as the **response only** — the opening
`<|im_start|>assistant` of the first turn lives at the tail of `input`. So the
first chunk has no role header, its first line is the model's own `<think>`, the
role matches neither `assistant` nor `user`, and the whole turn is discarded.

Turn 1 is where the standard blinkdepth chain issues `depth_estimator` +
`roborefer` x2. The consequences:

| statistic | shipped parser | truth |
|-----------|---------------|-------|
| `depth_estimator` calls | **absent entirely** | 122 |
| dominant chain | `vision_opsx2@2t` x107 | `depth_estimatorx1+roboreferx2+vision_opsx2@3t` x106 |
| wrong-sample chain | not the expected 12 | **12**, as P2 hand-counted |
| `no <answer>` | 2 | 0 |
| mean turns | 2.13 | 3.13 |

`depth_estimator` vanishing from the tool histogram of a **depth benchmark** is
the loudest possible symptom, and it was sitting in plain sight. The two
`no <answer>` samples were single-turn outputs swallowed whole — they show up in
the shipped parser's own chain list as `@0t` x2.

It would also have suppressed `hit_max_turns`: with every count one short, a
sample that actually hit the 8-turn cap reads as 7 and never trips the flag.

Fixed by treating a leading chunk with no `<|im_start|>` prefix as assistant turn
1. After the fix, all eight of the handoff's expected values reproduce, including
the 12-chain that the handoff singles out as the one that matters.

### 7.2 The turn-count cross-check compared two different things

Patch 0008 exists so the parser can check its own decomposition against verl's
count, and `--strict` treats a mismatch as contamination. But the two sides count
differently — `verl/experimental/agent_loop/tool_agent_loop.py:238`:

    num_turns = agent_data.user_turns + agent_data.assistant_turns + 1

while `num_turns_derived` counts **assistant turns only**. Comparing them
directly makes **every sample** a mismatch. With patch 0008 now in the tree,
V2 would have exited 1 on every benchmark, and the handoff says a non-zero value
"is a parser bug, not a run problem" — it would have been right, in the check
rather than the splitter.

The parser now also derives `num_turns_verl_convention`
(`assistant + user + 1`) and compares that. Validated against P2's recorded turn
counts on four runs — 6.26, 4.00, 4.00, 9.88 — all four reproduce exactly.

This is also what explains the "factor of two" between P2's turn counts and the
parser's: not a bug in either, two conventions. P2's mean turns of 6.26 for
blinkdepth is 3.13 assistant turns plus 2.13 tool-result turns plus the prompt.

### 7.3 P2's grasp success/failure split is wrong, and the parser is right

`P3_NOTES.md` offers P2's bopgrasp split — 27 succeeded / 33 errored — as "the
cheapest check that failure detection works". The parser says **19 / 41**, and a
raw text search of the dump, independent of the parser entirely, agrees with the
parser:

    "No collision-free grasps"        33 events in 33 samples
    "Top-down filtering removed all"   8 events in  8 samples
    at least one failure               41 samples  ->  19 clean

Exactly one `grasp_generator` call per sample, 60 calls total. P2 counted only
`No collision-free grasps` as a failure — hence exactly 33 — and scored the 8
`Top-down filtering removed all` samples as successes.

**P2's 96 vs 16 failure-cause counts came from the eval log, not the dump.**
`grep -c` on `/workspace/logs/p2-bopgrasp60.log` returns 96 and 16; the same
greps on the dump return 33 and 8. The tool logs several lines per call, so the
log counts internal attempts while the dump holds what the model actually
received. Both are real; they answer different questions, and P2 conflated them.

**Two P6 leads move as a result**, and P5/P6 should use these:

- "the grasp tool fails on 55% of samples" is really **68%** (41/60).
- "collision filtering dominates by 6x" is **4.1x** in what the model saw
  (33 vs 8), not 6x. Same direction, weaker margin.
- the succeeded/errored score split must be recomputed on 19/41. Note this needs
  `analysis/analyze_grasp_result.py`: `score` on bopgrasp is the RL NCE metric
  (lower is better, mean ~2.0), **not** MACE/SR.

**Nothing here changes a reported accuracy.** These are diagnostic statistics
only, and P2's headline numbers all reproduce: blinkdepth 108 / 107 / 111,
cvb2drelation 93.75%, robospatial Vacant 56.25%.

---

## 8. Phase V2: the policy does not fit a 40 GB card as configured

blinkdepth would not run at all. Four failed attempts, all dying at the same
place — `sglang_rollout.py:213`, `get_named_tensor_buckets` -> `tensor.clone()`,
inside `rollout_mode()`'s push of FSDP weights into the sglang engine:

    torch.OutOfMemoryError: Tried to allocate 20.00 MiB.
    GPU 0 has a total capacity of 39.49 GiB of which 19.12 MiB is free.
    Process A has 31.42 GiB in use.   Process B has 8.00 GiB in use.

The policy had a card to itself — `SGLangHttpServer ... cuda_visible_devices='3'`,
and the GPU trace shows the tools steady on 0/1/2 throughout. This is not tool
co-location. It is the policy alone overflowing 40 GB.

### 8.1 Three knobs that were not the answer

Recorded because each looked obvious and each cost a run:

| tried | result |
|-------|--------|
| `gpu_memory_utilization` 0.5 -> 0.35 | failure **byte-identical**, down to "19.12 MiB is free" |
| `actor.fsdp_config.param_offload=True` | 31.46 -> 31.42 GiB, i.e. nothing |
| `update_weights_bucket_megabytes` 2048 -> 512 | freed its ~1.5 GB transient; card still filled (GPU3 peaked 39.5 GB) |

`gpu_memory_utilization` is the intuitive suspect and is irrelevant here:
**TorchMemorySaver has already released the KV cache by the time the weight push
runs**, so sglang is down to 8.00 GB of bf16 weights and its static fraction no
longer applies. `param_offload` fails for a similar reason — `update_weights`
must gather the parameters back onto the GPU whatever their resting place.

The pattern is the one this project keeps re-learning: a parameter's *name*
suggested it governed the allocation; the *execution order* decided otherwise.

### 8.2 What it actually was, and the fix

verl's FSDP engine defaults `model_dtype` to **fp32**
(`verl/workers/config/engine.py:238`). The policy is 4.066 B parameters, so the
master copy is **16.3 GB**; in bf16 it is 8.1 GB. That is the only allocation on
the card big enough to matter.

    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16

GPU3's peak fell from **39.5 GB to 25.0 GB** and the run completed.

**Why this is numerically inert for evaluation** — checked, not assumed:

- Every tensor in `siyich/spacetools-ckpt` is stored **BF16**: 825 of 825
  tensors, 8.132 GB over 4.066 B parameters = **2.00 bytes/parameter**, read
  from the safetensors headers.
- verl upcasts that to the fp32 master and then downcasts it again to hand to
  sglang, which runs bf16 (`rollout dtype: bfloat16`). `bf16 -> fp32 -> bf16` is
  the identity, so the fp32 master round-trips exactly the bits it was given.
- `trainer.val_only=true`: no optimizer step, no gradient accumulation. Those
  are the two things an fp32 master exists for.

**The round trip was then measured, not just argued.** Both paths were applied
to the actual checkpoint on CPU -- `t.to(fp32).to(bf16)` against `t` -- for every
tensor:

    tensors compared    825
    elements compared   4.066 B
    non-bf16 tensors    0
    tensors differing   0
    elements differing  0
    max abs difference  0.000e+00      VERDICT: bit-identical

**And the FSDP model never runs a forward pass during eval.** The run's own
config says `calculate_log_probs: False`, and the generation batch carries
`recompute_log_prob: False`; generation is entirely sglang's. The FSDP engine's
only job in `val_only` is to hold the weights and hand them over. Since what it
hands over is bit-identical either way, `model_dtype` cannot reach the results.

That closes the chain by measurement rather than inference. The run landing at
**109/124 = 87.90%**, inside the 106-112 band, is consistent with it but is not
what the argument rests on -- a single draw from a 6-sample-wide band could not
distinguish "inert" from "small effect" anyway.

> **This reasoning does not extend to training.** If P7 ever runs `run_rl.sh`,
> the fp32 master matters and this flag must not be carried across. The comment
> in `run_eval.sh` says so at the point of use.

### 8.3 Why P0-P3 never saw it

48 GB absorbed the fp32 master with ~8 GB to spare. Every A6000 run had it.
This is the second time in this migration that the A6000's extra 8 GB was
silently load-bearing — the first was `p2_tool_chain.py`'s stale fractions
(§6.3). Neither showed up as a warning on the old hardware; both are hard
failures on 40 GB.

### 8.4 The alternative, and why it was rejected

fp32 could probably have been kept by stacking the three knobs in §8.1 together
to claw back the last ~1 GB. That would mean three deviations of marginal and
poorly-understood effect, leaving a card with no headroom, in exchange for
preserving a master copy that `val_only` never reads. One numerically-inert
change with 14.5 GB of headroom is the better trade, and it is far easier for P5
to explain.

---

## 9. Phase V3: Molmo's peak, and the answer to the handoff's open question

`P4_HANDOFF.md` leaves one question for this machine: is the 33.0 GB in P2's
table Molmo's resting footprint or its observed peak? If resting, the spike on
`robospatial`'s 2.76 MP images is unmeasured and 7 GB of headroom may not be
enough.

### 9.1 The prescribed test cannot answer it

The 32-sample `robospatial` stress run completed clean — zero OOM, `--strict`
exit 0, 18/32 = 56.25%, which is **identical to the A6000's number**. Peaks:

| GPU | contents | peak | of 40 GB |
|-----|----------|-----:|---------:|
| 0 | RoboRefer + SAM2 x2 | 20.8 GB | 52% |
| 1 | Molmo | 30.4 GB | 76% |
| 2 | DepthPro + bbox + grasp | 9.3 GB | 23% |
| 3 | policy | 27.4 GB | 69% |

But the tool histogram reads `{'roborefer': 32}` — **`vlm` never appears**. All
32 samples used `roboreferx1@2t`. The handoff anticipated exactly this: if
`vlm` is never invoked, Molmo cannot spike there, and GPU1's 30.4 GB is just its
resting size. The stress test proves the *run* is safe; it says nothing about
Molmo.

### 9.2 Measuring it directly

So Molmo was called directly, outside the eval, on the largest image in each
benchmark, sampling its card every 50 ms:

| case | image | Molmo peak | of 40 GB |
|------|------:|-----------:|---------:|
| blinkdepth | 0.19 MP | 35321 MiB | 86.2% |
| bopgrasp | 0.92 MP | 35321 MiB | 86.2% |
| boppose | 2.07 MP | 35321 MiB | 86.2% |
| robospatial | 2.76 MP | 35321 MiB | 86.2% |
| cvb2drelation, largest | 3.63 MP | 35321 MiB | 86.2% |

**34.49 GB, and not one MiB of variation across a 19x range of image size.**

### 9.3 Why — the crop grid saturates

Molmo's preprocessor emits a bounded multi-crop tensor, not a resized image:

    blinkdepth  0.19 MP  ->  images torch.Size([ 5, 576, 588])
    robospatial 2.76 MP  ->  images torch.Size([13, 576, 588])

The crop count rises with resolution and then **caps at 13** (12 tiles plus one
global view). Past that, a larger image adds nothing. And even at saturation the
activation cost of 13 x 576 x 588 is below `nvidia-smi`'s resolution against
33 GB of resident fp32 weights — which is why the peaks are byte-identical
rather than merely close.

### 9.4 The answer

The question "resting or peak" has a better answer than either: **for Molmo the
two are the same number**, because its input is normalised to a fixed-size crop
grid before the model sees it. 34.49 GB is a hard ceiling across every image in
all nine benchmarks, leaving **5.51 GB permanently free** on that card.

So the 86.2% occupancy that looks comparable to the 84% which produced P2's OOM
cascade is not comparable at all. That 84% was a *variable* load — several tools
sharing a card, spiking concurrently. This is a constant one with a proven
ceiling and no other tenant.

V3's gate — "peak below ~37 GB: fine" — passes at 34.49 GB.

**Still check every benchmark.** This bounds Molmo, not the other tools, and
`parse_dump.py --strict` after each benchmark remains the rule.

### 9.5 Image sizes across the nine benchmarks, for the record

Measured from the parquets; the totals confirm the handoff's 2121-sample scope.

| benchmark | n | mean MP | max MP |
|-----------|--:|--------:|-------:|
| robospatial | 350 | 2.76 | 2.76 |
| boppose | 60 | 2.07 | 2.07 |
| bopgrasp | 60 | 0.92 | 0.92 |
| cvb3ddepth | 600 | 0.79 | 0.79 |
| cvb2drelation | 650 | 0.44 | **3.15** |
| refplacement | 100 | 0.31 | 0.31 |
| reflocation | 100 | 0.30 | 0.31 |
| refunseen | 77 | 0.28 | 0.31 |
| blinkdepth | 124 | 0.18 | 0.25 |
| **total** | **2121** | | |

`cvb2drelation` is worth noting: a low mean but the largest single image in the
whole suite. It is also the second benchmark in the P4 order, so it exercises
the large-image path early, which is what the cheapest-first ordering is for.

---

## 10. Operating gotchas

None of these are changes to anything. They are the traps that cost real time in
Phase M and V, written down because every one of them will still be there for
whoever works on this next.

### 10.1 `set -u` breaks conda activation

    $ set -euo pipefail; conda activate spacetools-tool-graspgen
    .../etc/conda/activate.d/~cuda-nvcc_activate.sh: line 48:
        NVCC_PREPEND_FLAGS: unbound variable

The env's own activation hooks read variables that may not be set, so they are
not `-u` safe. Use `set -eo pipefail` in any script that activates a conda env
here, or `set +u` around the activate. The failure looks like a bug in your
script and is not.

### 10.2 `start_toolkit(detached=False)` — hold the handle

    RuntimeError: Could not find ToolRouterActor 'toolshed_router'
                  in namespace 'toolshed'

The router actor's lifetime is bound to the object `start_toolkit` returns. Drop
it on the floor -- `start_toolkit(...)` instead of `router = start_toolkit(...)`
-- and Ray garbage-collects the router out from under you, usually a few tool
calls in, so it reads as an intermittent failure rather than a mistake at line 1.
`p2_tool_chain.py` carries a comment about this; I still walked into it.

### 10.3 `make_subset.py --out` is a directory root, not a file

    python tools/make_subset.py robospatial 32 \
        --src /workspace/eval-benchmarks --out /workspace/tmp/rs32
    # writes /workspace/tmp/rs32/data/robospatial.parquet
    # then:  DATA_DIR=/workspace/tmp/rs32 bash run_eval.sh <ckpt> robospatial

It takes a benchmark NAME and a count, not input and output paths, and `--out`
becomes the `DATA_DIR` root that `run_eval.sh` expects -- the parquet lands in a
`data/` subdirectory. Passing a `.parquet` path as `--out` "works" and produces
`.../robospatial.parquet/data/robospatial.parquet`.

### 10.4 Sample GPU memory at 1 s, not 5 s

The policy-card OOM in §8 happens inside `rollout_mode()`'s weight push, which
fills the card and dies in **under five seconds**. A 5 s sampling loop showed the
card sitting calmly at 23 GB right up to the failure and never recorded the
spike, which sent the first round of diagnosis at the wrong component entirely.
1 s catches it; the direct Molmo measurement in §9 used 50 ms.

More generally: a peak you did not sample fast enough to see is indistinguishable
from a peak that did not happen.

### 10.5 The project's own tools need the conda env

`tools/make_subset.py`, `tools/parse_dump.py` and anything touching parquet need
`pandas` / `pyarrow`, which the system `python3` does not have. Run them under
`conda run -n spacetools-rl` **with a script path, never a heredoc** (§10.6).
`parse_dump.py` is stdlib-only by design and runs in any of the five envs; the
parquet tools are not.

### 10.6 `conda run` does not forward stdin -- silently

This is the single biggest source of wasted cycles in Phase M and V, because it
does not look like an error:

    $ echo "print('hi')" | conda run -n spacetools-rl python -
    $                                    # no output. exit code 1. no message.

    $ echo "print('hi')" | /workspace/envs/spacetools-rl/bin/python -
    hi

Every `conda run -n <env> python - <<'EOF' ... EOF` and every heredoc-fed
one-liner in these five environments returns **nothing at all**. It exits 1, so
`set -e` catches it, but run interactively you just see an empty result and
assume your script printed nothing.

Two things work: write the snippet to a file and pass the path
(`conda run -n <env> python /tmp/x.py`), or call the environment's interpreter
directly (`/workspace/envs/<env>/bin/python`). The interpreter path skips
`conda run`'s wrapper entirely, which also means it skips the env's activation
hooks -- fine for plain Python, not for anything needing `LD_LIBRARY_PATH` (see
`st_activate` in `env.sh`).

### 10.7 `pointnet2_ops` kernel failures kill the process, they do not raise

`_ext-src/include/cuda_utils.h:30`:

    #define CUDA_CHECK_ERRORS()                    \
      ... fprintf(stderr, "CUDA kernel failed : %s\n" ...);  \
          exit(-1);

`exit(-1)`, not a thrown exception. A `try/except` around
`furthest_point_sample()` catches nothing; the interpreter is gone. Inside a Ray
actor it takes the actor down.

**This is worth knowing precisely because it is the opposite of the OOM failure
mode this project keeps warning about.** An out-of-memory condition inside a
tool comes back as an error *string* the model reads as a tool response, and the
benchmark finishes with a quietly wrong number. An architecture mismatch in
`pointnet2_ops` cannot do that -- it kills the actor. So the sm_86 problem, had
it survived into P4, would have failed loudly. The silent-failure worry belongs
to OOM, not to this.

### 10.8 `pkill -f <pattern>` can match itself

`pkill -f "nvidia-smi --query-gpu"` matched the very shell that issued it,
because `-f` matches the whole command line and the pattern was in it. The shell
died mid-command and the work it was about to launch never started, which looks
exactly like a silent failure of the thing you were trying to run. Write the
watcher to a script file and kill it by recorded PID.

### 10.9 A crashing sglang worker writes a core the size of its RSS, onto the CONTAINER disk

2026-09-02. A `robospatial` run at `n=5` OOMed in sglang's vision encoder. The
worker died, and the kernel wrote a **50 GB** core:

    /var/lib/vastai_kaalia/data/core-ray::SGLangHttp.82949.60948e76dd01.1788331572

That is the vast.ai core directory, on the **50 GB container disk** — not the
volume. Everything else on this instance is deliberately pointed at the volume
(`envs`, `models`, `TMPDIR`, `HF_HOME`, ray's `--temp-dir`), but **`core_pattern`
obeys none of those**. The volume still had 58 GB free while `/` was at 100%.

**The symptom is not "the eval failed".** The eval had already failed, on GPU
memory; the core was written *during the crash*. What you actually meet, minutes
later, is **every command failing with `ENOSPC`** — including `df -h`, because a
shell needs to write its output somewhere. Diagnosis therefore has to run
backwards: probe which filesystem is full by *writing* to candidate paths
(`/workspace` succeeded, `/tmp` did not), and only then look for the file.

    find / -xdev -type f -size +500M -exec ls -lhS {} + 2>/dev/null

`-xdev` matters: without it the scan descends into the 199 GB volume and buries
the answer.

**Fix, applied to every runner** (`tools/p4_run.sh`, `tools/p7/p7_passk2_run.sh`,
the three `tools/p6/variants/run_*.sh`):

    ulimit -c 0        # inherited by all children; set once in the launching shell

This is worth more than it looks. The core filled the disk **mid-sequence**, so
the two benchmarks queued behind `robospatial` would have failed on ENOSPC even
if they never came near an OOM — one crash contaminating a whole run, which is
exactly the failure class `--strict` exists to prevent and cannot see.

---

## What could still move a number

After review, only two of the fifteen deviations plausibly could, and neither can
be avoided:

| deviation | why it might matter | why it cannot be avoided |
|-----------|--------------------|--------------------------|
| cudnn 9.16.0.29 (breaks torch's `==9.10.2.21` pin) | different conv results are exactly what pytorch#168167 is about | sglang reads `torch.backends.cudnn.version()` at startup and refuses below 9.15. `docs/SETUP.md:88` documents the same requirement and the same version number |
| numpy 2.x in `spacetools-rl` (verl declares `<2.0.0`) | none known; the pin is stale metadata | Ray cannot deserialise numpy-2 arrays from the tool environments into a numpy-1 driver (`ModuleNotFoundError: numpy._core.numeric`), and toolshed itself requires `>=2.0.0` |

Everything else is either a hard blocker, a verified no-op, or scheduling.

Ranked above both of these is something that is not a deviation at all:
**run-to-run nondeterminism**. blinkdepth run three times under an identical
configuration gave 108 / 107 / 111 correct, a range of 3.23 pp -- larger than any
single deviation is likely to be worth. See `P2_RESULTS.md`.

---

## Operational constraints we introduced

**Correction, 2026-08-27:** all four repos now DO track a remote (`mine/`,
pointing at personal forks), so the paragraph below is stale and
`install_graspgen.sh`'s `git pull` would no longer abort for that reason. What
is true on the A100 host is different and worse: **no GitHub credentials exist
on this machine at all** -- they lived on the previous host's container disk,
which a recycle wipes. `git fetch`/`git pull`/`git push` all fail with
`could not read Username for 'https://github.com'`. Nothing was lost (the
working trees came across on the volume), but **Phase D cannot push** until a
token is supplied.

All four repos now sit on a local branch `repro-4xa6000` with **no upstream
tracking**. That matters in exactly one place: `install_graspgen.sh:79` runs
`git pull` when the GraspGen directory already exists, and with `set -e` at line
17 the script aborts, because `git pull` on an untracked branch fails with "no
upstream configured". Check out `main` first if that script ever needs re-running:

    git -C SpaceTools-Toolshed/GraspGen checkout main

The others are unaffected: `install_roborefer.sh` skips the clone when the
directory exists rather than pulling, and nothing pulls Toolshed or SpaceTools-RL.
