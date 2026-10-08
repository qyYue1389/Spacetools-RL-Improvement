# Environment package revision 2026-09-12: POSTRESTORE.sh + VERIFY pre-check

Written 2026-09-12 · based on the findings of the 2026-09-11 eval in `03_sft_eval/sft_eval_results.md`
· already pushed to `qyYue1389/spacetools-eval-env`

---

## 0. In one sentence

The three "things missing from the package" and two "false green lights" exposed by that eval have been fixed into the HF repo as **small files**;
**the 22 GB envs tar was not touched**. The restore order on a new machine gains one step:

```
4. bash FETCH_WEIGHTS.sh
5. bash RESTORE.sh
5b. bash POSTRESTORE.sh     ← new
6. bash VERIFY.sh           ← now four items
```

## 1. Why upload files instead of repackaging

The real delta is only three lines: 1 line changed in each of two `node.py` files, and a 1-line `activate.d`.
Repackaging 22 GB would only buy "self-contained", at three costs:

| | Repackage | Upload small files |
|---|---|---|
| Upload/download volume | about 26–28 GB compressed (`/opt/conda-st` is now 52G, the original package unpacks to about 40G; the extra is `__pycache__` produced by eval runs and the conda cache) | a few KB |
| Download increment per machine switch | **+5 GB** | 0 |
| Can it be verified | **No**. Step 0 of `RESTORE.sh` is `[ -e /opt/conda-st ] && die`, which explicitly refuses to overwrite, so after packaging we could only confirm "compressed and uploaded successfully", not "the restore comes out right" | Yes, the script has built-in read-back assertions |
| Root-cause visibility | welding the workaround into the tar makes the root cause (Python version mismatch) completely invisible | the patch is explicit, reviewable, and can be deleted later |

A100 (sm_80) and A6000 (sm_86) use **the same package**; the architecture scan confirmed that self-built extensions missing sm_8x is 0,
so switching frequently between these two GPU types does not require per-architecture packages — the only things that change are `NUM_GPUS` / `EVAL_GPUS` and the GPU budget.

## 2. The three things `POSTRESTORE.sh` patches

Idempotent, safe to re-run, and every patch is immediately read back and asserted.

**① Ray's Python version match level.** In this package `spacetools-tool-vlm` / `spacetools-tool-bbox`
are Python 3.11.0, and the other three are 3.11.16. Ray's `check_version_info` compares the full version string by default,
so **none of the actors** from these two environments can join the cluster.

The script switches to Ray's built-in `minor` level (adding
`python_version_match_level="minor"` at the call site in `node.py:454`, with the original file backed up as `node.py.orig`).

Safety rationale: both sides have bytecode magic 3495 and default pickle protocol 4, so code objects and
pickles are interchangeable — exactly the scenario the minor level is designed for; the change only relaxes one version check and does not change any numerical value.

**The script is version-aware**: it first reads the Python of all five environments and only patches environments that differ from the head node (`spacetools-rl`).
If the package is rebuilt in the future with everything unified on 3.11.16, this item is skipped automatically.

**② `CUDA_HOME` for the roborefer environment.** The llava inference path hard-depends on deepspeed at module level (environment
README hole #17), and deepspeed reads `CUDA_HOME` at import. The packaging machine had the system
`/usr/local/cuda-12.8`, which hit torch's third fallback; a machine with only the driver installed does not have it.

**③ `/workspace/logs` and `/workspace/smoke`.** `28_chain.sh` / `04_smoke.sh` write to these
two directories, while MANIFEST only mentioned `checkpoints` and `hf`.

## 3. Two revisions to `VERIFY.sh` — this matters more than the three things above

**The original three items cannot catch ①.** Before the patch `28_chain.sh` **passed** (measured: all four chains STEP_OK);
the Python version mismatch only surfaces at eval / RL scale. In other words: forget to run POSTRESTORE,
all three items are green, and then RL silently runs with 23 fewer actors.

So a **pre-check item 0** was added; if it fails, it does `exit 1` and does not print the three items after it:

```
Python versions of the five envs + whether the patch is in place when they differ
CUDA_HOME for roborefer (automatically pointed at that env when unset)
whether /workspace/{logs,smoke,checkpoints,hf} exist
whether there is a live Ray cluster / leftover cluster address file
```

**Item 3 no longer trusts the exit code of `28_chain.sh`**; it now looks for a success marker in the output. Measured: it
prints "✗ chain not connected" while doing `exit 0`, and the summary printed "✓ all three passed" based on that.

### The leftover Ray cluster pitfall deserves its own note

`/root/tmp/ray/ray_current_cluster` (a 19-byte address file left by the previous eval) is enough to make
`28_chain.sh`'s `ray.init(num_cpus=8, num_gpus=1, ...)` think the cluster is still there:

```
Connecting to existing Ray cluster at address: ...:6379
ValueError: When connecting to an existing cluster, num_cpus and num_gpus must not be provided.
```

The python block dies within 1 second, and the script still does `exit 0`. **If the previous eval is not cleaned up, item 3 of the next VERIFY
fails and gives a false green light.** Item 0 now blocks this.

Also, `ray stop --force` reporting "57/58 stopped" is nothing to worry about: the remaining one is a zombie, already defunct.

### Detect a live cluster with `pgrep -x`, not `pgrep -f`

When item 0 decides "is there a live Ray cluster", it **must match the process name exactly**:

```bash
pgrep -x gcs_server || pgrep -x raylet        # correct
pgrep -f "gcs_server|raylet"                  # wrong, matches the caller's own command line
```

`pgrep -f` matches the whole command line, so a call like `ray stop --force && bash VERIFY.sh` would
count itself as "Ray is running" and give a false alarm. Ray's core binaries are named exactly `gcs_server` and `raylet`;
`-x` matches comm exactly and is not affected by the caller's command line.

## 4. Upstream issue fixed along the way

Step 1 of `RESTORE.sh` runs `sha256sum -c SHA256SUMS`, but the repo originally only had
`SHA256SUMS.remote`, so following the docs was bound to die with "package is corrupted". Now both files are in the repo
(identical content), and no manual `cp` is needed.

## 5. Verification

```
POSTRESTORE.sh  idempotency: with the patch already in place, correctly reports "already applied, skipping" and reads back to confirm, exit 0
VERIFY.sh item 0:
    with a leftover Ray cluster  → correctly BAD and exit 1 (tested under real leftover conditions)
    after cleanup                → all 8 checks OK, correctly proceeds to item 1
SHA256SUMS:     13/13 self-check passed (6 small files recomputed, 7 large files keep their original hashes)
both scripts:   bash -n syntax check passed
```

## 6. What changed in the repo

```
POSTRESTORE.sh      new         5535 bytes
VERIFY.sh           rewritten   5117 bytes (three items → four items)
README.md           appended    restore order + explanation of the three things + the SHA256SUMS note
MANIFEST.txt        appended    Python version table of the five envs + measured GPU budget
SHA256SUMS.remote   recomputed
SHA256SUMS          new         same content as .remote
spacetools-envs-*.tar.zst        untouched
spacetools-scripts-*.tar.zst     untouched
```

## 7. Not done yet

**Fix the Python version mismatch at the root** — when rebuilding the environment package, unify all five environments on 3.11.16; after that
item ① of `POSTRESTORE.sh` is skipped automatically and the whole script can be deleted.

It cannot be upgraded in place on this machine:

```
CondaToSNonInteractiveError: Terms of Service have not been accepted for:
    https://repo.anaconda.com/pkgs/main   https://repo.anaconda.com/pkgs/r
```

Accepting Anaconda's commercial terms is a legal decision for the organization; and re-solving risks moving pinned versions such as numpy 1.26.4 /
transformers 4.53.2, which is the silent definition change this project fears most.

**GPU budget** (see `03_sft_eval/sft_eval_results.md` §4, §6): with all tools alive the tools take 2.3 GPUs and
the policy needs a whole GPU, which is tight on a 4-GPU machine; RL's actor count needs roughly 7+ GPUs and must be recomputed.
This is not something the package can solve; it has to be computed when choosing machines.
