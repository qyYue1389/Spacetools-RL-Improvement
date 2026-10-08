"""
P7 -- under our batch config, is the IS weight w in Eq.8 really identically 1?

dp_actor.py:550
    on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
dp_actor.py:598
    if on_policy:  old_log_prob = log_prob.detach()

When on_policy is true, the **value** of log_ratio is identically 0 (the gradient is not 0), so
    w = clamp(exp(0), max=1+eps_is) = 1        <- eps_is has no effect at all
    delta = g~ + 0                              <- loss value = mean(g~^2)
The update degenerates to REINFORCE form: grad = mean(2*g~_i * grad masked_sum(log pi_theta)).

This is not a defect, it is the inevitable result of "one rollout batch takes only one optimizer step" — there is no pi_theta/pi_old
mismatch to correct. But it is **config-dependent**: changing ppo_mini_batch_size or ppo_epochs flips it,
and then w becomes the probability ratio of the whole sequence exp(sum over ~1e3 tokens), while Eq.8 only clips the upper bound.
So this script is both a record and a guardrail.

This script also guards the divisibility assertion at fsdp_workers.py:263 — the number of training GPUs is not arbitrary:
  normalized ppo_mini_batch_size = (train_batch_size × rollout.n) // number of training GPUs
  it must be divisible by ppo_micro_batch_size_per_gpu, otherwise worker initialization throws.
  Training on 6 and 7 GPUs will FAIL (neither 53 nor 45 is divisible by 2), and this is not recorded anywhere else.
  In addition, fsdp_size=2 (HSDP, recommended on) requires an even number of training GPUs,
  because create_device_mesh uses mesh_shape=(world_size // fsdp_size, fsdp_size).

Same as p7_guard_check.py: the numbers are extracted from the real script, not copied by hand.
"""
import math, re, sys, yaml

RL = __import__("os").environ.get("SPACETOOLS_RL", "../../../SpaceTools-RL") + ""
sh = open(f"{RL}/examples/toolshed/run_rl.sh").read()
actor_yaml = yaml.safe_load(open(f"{RL}/verl/trainer/config/actor/actor.yaml"))


def grab(key, default=None):
    m = re.search(rf"{re.escape(key)}=(\d+)", sh)
    if m:
        return int(m.group(1))
    if default is None:
        sys.exit(f"FAIL: cannot find {key} in run_rl.sh")
    return default


train_bsz = grab("data.train_batch_size")
mini_cfg = grab("actor_rollout_ref.actor.ppo_mini_batch_size")
micro = grab("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu")
n = grab("actor_rollout_ref.rollout.n")
epochs = int(actor_yaml["ppo_epochs"])

print(f"run_rl.sh:  train_batch_size={train_bsz}  ppo_mini_batch_size={mini_cfg}  "
      f"micro/gpu={micro}  rollout.n={n}")
print(f"actor.yaml: ppo_epochs={epochs}")
print()

ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {label}")
    if detail:
        print(f"         {detail}")


for ranks in (2, 4, 6, 8):
    # fsdp_workers.py:249-250, unconditional
    mini = (mini_cfg * n) // ranks
    per_rank = (train_bsz * n) // ranks          # number of sequences this rank gets
    n_mini = math.ceil(per_rank / mini) if mini else 0
    on_policy = (n_mini == 1 and epochs == 1)
    good = on_policy
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] training GPUs {ranks}: normalized mini={mini} · "
          f"{per_rank} per GPU · {n_mini} mini-batches -> on_policy={on_policy}"
          f"  =>  w {'identically 1' if on_policy else 'is the real sequence ratio'}")

print()
# ---- validity of the number of training GPUs (divisibility assertion at fsdp_workers.py:263 + HSDP even-number requirement) ----
print("training GPU count validity: normalized mini must be divisible by micro; fsdp_size=2 also requires an even number")
legal_plain, legal_hsdp = [], []
for ranks in range(1, 9):
    mini = (train_bsz * n) // ranks
    ok_div = mini % micro == 0 and mini // micro > 0
    ok_hsdp = ranks % 2 == 0
    if ok_div:
        legal_plain.append(ranks)
        if ok_hsdp:
            legal_hsdp.append(ranks)
    mark = ("✓" if (ok_div and ok_hsdp)
            else "△ only fsdp_size=-1" if ok_div
            else "✗ fsdp_workers.py:263 will throw")
    print(f"    training GPUs {ranks}: mini={mini:4d}  mini%micro={mini % micro}  "
          f"{'even' if ok_hsdp else 'odd'}   {mark}")

# the three training GPU counts we plan to use must be valid, and 6/7 must be correctly identified as invalid
#   4-GPU machine -> tools 2 + training 2 | 6 GPUs -> tools 2 + training 4 | 8 GPUs -> tools 4 + training 4
check("planned training GPU counts 2 / 4 are both valid (including fsdp_size=2)",
      {2, 4} <= set(legal_hsdp), f"GPU counts valid under fsdp_size=2: {legal_hsdp}")
check("training on 6 and 7 GPUs is correctly identified as invalid",
      6 not in legal_plain and 7 not in legal_plain,
      "the intuitive split \"8 GPUs = tools 2 + training 6\" crashes on startup; this check is here to block it")

print()
# what would break it
print("changes that would flip on_policy (only then does eps_is start to matter, and w may collapse to 0):")
for label, mb, ep in (("ppo_mini_batch_size 64 -> 32", 32, epochs),
                      ("ppo_mini_batch_size 64 -> 16", 16, epochs),
                      ("ppo_epochs 1 -> 2", mini_cfg, 2)):
    mini = (mb * n) // 4
    per_rank = (train_bsz * n) // 4
    n_mini = math.ceil(per_rank / mini) if mini else 0
    print(f"    {label:32s} (4 GPUs) -> {n_mini} mini-batches, epochs {ep} "
          f"-> on_policy={n_mini == 1 and ep == 1}")

print("\n" + "=" * 62)
print("RESULT:", "PASS -- w is identically 1, eps_is has no effect under the current config" if ok
      else "FAIL -- w is no longer identically 1, eps_is and the IS-collapse risk both become real")
sys.exit(0 if ok else 1)
