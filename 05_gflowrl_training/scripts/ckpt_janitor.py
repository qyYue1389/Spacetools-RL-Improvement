#!/usr/bin/env python3
"""checkpoint 清道夫 + 里程碑权重快照(路 B)。

卷 250 GB,删掉 envpkg 里那个冗余的 21 GB 拼接产物后:
    非 ckpt 固定占用   111 GB
    可给 ckpt 的       139 GB
      续训峰值          88 GB   (旧 44 + 正在写的新 44)
      eval 输出预留     10 GB
      里程碑快照        30 GB   = 2 个 × 15 GB,余 11 GB

做两件事:
  1) 完整 ckpt 只保留 step 最大的那一个 —— 给 run_rl.sh 的自动 resume 用。
     永远不删最新的;只有最新的已写到 >= MIN_MB 才删旧的,避免两头空。
  2) 到 MILESTONES 的步数时,把那一步的**权重分片**单独复制出来存档,
     不带 optim_*.pt(每个 rank 7 GB,只有续训要,eval 用不到)。
     依据:verl/model_merger/fsdp_model_merger.py 只读 model_world_size_*_rank_*.pt。
     所以 15 GB 的快照足够 merge 成 HF 格式再喂给 run_eval.sh。

快照先写到 .tmp 再改名,半途死掉不会留下一个看起来完整的坏快照。
"""
import os, re, shutil, subprocess, time

OUT = "/workspace/exp/p7_cprime/rl_output"
SNAP = "/workspace/exp/p7_cprime/eval_ckpts"
LOG = "/root/logs/ckpt_janitor.log"
MILESTONES = (30, 60)
MIN_MB = 25000
PERIOD = 120
# 快照要带的:权重分片 + config/tokenizer + fsdp 元信息。不带 optim_*.pt
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
    """把 global_step_<step> 的权重部分复制到 SNAP。已存在则跳过。"""
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
    log("里程碑快照 global_step_%d 完成:%d 项 / %d MB(不含 optim_*.pt)" % (step, n, size_mb(dst)))


log("启动(路 B):完整 ckpt 只留最新;里程碑 %s 另存权重快照到 %s" % (list(MILESTONES), SNAP))
os.makedirs(SNAP, exist_ok=True)
while True:
    s = steps()
    if s:
        newest = s[-1]
        mb = size_mb(os.path.join(OUT, "global_step_%d" % newest))
        if mb >= MIN_MB:
            # 先存档,再删除 —— 顺序反了就永久丢掉里程碑
            for ms in MILESTONES:
                if ms in s:
                    try:
                        snapshot(ms)
                    except Exception as e:
                        log("快照 global_step_%d 失败(下一轮重试):%r" % (ms, e))
            for old in s[:-1]:
                p = os.path.join(OUT, "global_step_%d" % old)
                log("删完整 ckpt global_step_%d(最新 global_step_%d 已 %d MB)" % (old, newest, mb))
                shutil.rmtree(p, ignore_errors=True)
            if len(s) > 1:
                log("现存完整 ckpt: %s · 权重快照: %s" % (steps(), sorted(os.listdir(SNAP))))
        else:
            log("最新 global_step_%d 只有 %d MB,还在写,不动" % (newest, mb))
    time.sleep(PERIOD)
