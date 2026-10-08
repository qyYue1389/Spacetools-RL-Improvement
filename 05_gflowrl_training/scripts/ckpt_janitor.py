#!/usr/bin/env python3
"""Checkpoint janitor + milestone weight snapshots (Route B).

Volume is 250 GB; after deleting the redundant 21 GB concatenation artifact in envpkg:
    fixed non-ckpt usage     111 GB
    available for ckpt       139 GB
      resume peak               88 GB   (old 44 + new 44 being written)
      eval output reserve       10 GB
      milestone snapshots       30 GB   = 2 × 15 GB, 11 GB left over

Does two things:
  1) keeps only the full ckpt with the largest step — for run_rl.sh's automatic resume.
     Never deletes the newest; only deletes old ones once the newest has been written to >= MIN_MB, so we never end up with neither.
  2) when a step in MILESTONES is reached, copies that step's **weight shards** out separately for archiving,
     without optim_*.pt (7 GB per rank, only needed for resuming, not for eval).
     Basis: verl/model_merger/fsdp_model_merger.py only reads model_world_size_*_rank_*.pt.
     So a 15 GB snapshot is enough to merge into HF format and feed to run_eval.sh.

Snapshots are written to .tmp first and then renamed, so dying midway never leaves a broken snapshot that looks complete.
"""
import os, re, shutil, subprocess, time

OUT = "/workspace/exp/p7_cprime/rl_output"
SNAP = "/workspace/exp/p7_cprime/eval_ckpts"
LOG = "/root/logs/ckpt_janitor.log"
MILESTONES = (30, 60)
MIN_MB = 25000
PERIOD = 120
# What a snapshot carries: weight shards + config/tokenizer + fsdp metadata. No optim_*.pt
KEEP_RE = re.compile(r"^(model_world_size_\d+_rank_\d+\.pt|extra_state_world_size_\d+_rank_\d+\.pt|fsdp_config\.json)$")


def log(msg):
    with open(LOG, "a") as f:
        f.write("[%s] %s\n" % (time.strftime("%F %H:%M:%S"), msg))


def size_mb(path):
    r = subprocess.run(["du", "-sm", path], capture_output=True, text=True)
    try:
        return int(r.stdout.split()[0])
    except (IndexError, ValueError):
        return 0


def steps():
    out = []
    try:
        for name in os.listdir(OUT):
            m = re.fullmatch(r"global_step_(\d+)", name)
            if m:
                out.append(int(m.group(1)))
    except FileNotFoundError:
        pass
    return sorted(out)


def snapshot(step):
    """Copy the weight part of global_step_<step> to SNAP. Skip if it already exists."""
    dst = os.path.join(SNAP, "global_step_%d" % step)
    if os.path.isdir(dst):
        return
    src_actor = os.path.join(OUT, "global_step_%d" % step, "actor")
    if not os.path.isdir(src_actor):
        return
    tmp = dst + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    dst_actor = os.path.join(tmp, "actor")
    os.makedirs(dst_actor, exist_ok=True)
    n = 0
    for name in sorted(os.listdir(src_actor)):
        s = os.path.join(src_actor, name)
        if os.path.isdir(s):
            if name == "huggingface":
                shutil.copytree(s, os.path.join(dst_actor, name))
                n += 1
            continue
        if KEEP_RE.match(name):
            shutil.copyfile(s, os.path.join(dst_actor, name))
            n += 1
    os.rename(tmp, dst)
    log("milestone snapshot global_step_%d done: %d items / %d MB (excluding optim_*.pt)" % (step, n, size_mb(dst)))


log("started (Route B): keep only the newest full ckpt; milestones %s get weight snapshots saved to %s" % (list(MILESTONES), SNAP))
os.makedirs(SNAP, exist_ok=True)
while True:
    s = steps()
    if s:
        newest = s[-1]
        mb = size_mb(os.path.join(OUT, "global_step_%d" % newest))
        if mb >= MIN_MB:
            # archive first, then delete — in the reverse order a milestone would be lost permanently
            for ms in MILESTONES:
                if ms in s:
                    try:
                        snapshot(ms)
                    except Exception as e:
                        log("snapshot global_step_%d failed (retry next round): %r" % (ms, e))
            for old in s[:-1]:
                p = os.path.join(OUT, "global_step_%d" % old)
                log("deleting full ckpt global_step_%d (newest global_step_%d is already %d MB)" % (old, newest, mb))
                shutil.rmtree(p, ignore_errors=True)
            if len(s) > 1:
                log("existing full ckpts: %s · weight snapshots: %s" % (steps(), sorted(os.listdir(SNAP))))
        else:
            log("newest global_step_%d is only %d MB, still being written, leaving it alone" % (newest, mb))
    time.sleep(PERIOD)
