"""
P7 -- can hydra actually set policy_loss.gflowrl.* ?

PolicyLossConfig is a structured dataclass instantiated through
omega_conf_to_dataclass -> hydra.utils.instantiate (fsdp_workers.py:935).
A key it has never heard of does not get ignored, it raises:

    TypeError: PolicyLossConfig.__init__() got an unexpected keyword argument 'gflowrl'

and it raises inside a Ray worker, minutes after Toolshed is already up.  So the
field must exist in BOTH the dataclass and the yaml, and the wrapper script must
not pass a key nobody reads.

Same discipline as p7_guard_check.py: read the repo's own text.
"""
import ast, re, sys, yaml

RL = __import__("os").environ.get("SPACETOOLS_RL", "../../../SpaceTools-RL") + ""
ACTOR_PY = f"{RL}/verl/workers/config/actor.py"
ACTOR_YAML = f"{RL}/verl/trainer/config/actor/actor.yaml"
WRAPPER = f"{RL}/examples/toolshed/run_rl_gflowrl.sh"
BASE_SH = f"{RL}/examples/toolshed/run_rl.sh"
ROLLOUT_YAML = f"{RL}/verl/trainer/config/rollout/rollout.yaml"
RCORR_YAML = f"{RL}/verl/trainer/config/algorithm/rollout_correction.yaml"
TRAINER = f"{RL}/verl/trainer/ppo/ray_trainer.py"
ALGOS = f"{RL}/verl/trainer/ppo/core_algos.py"

ok = True


def check(label, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {label}")
    if detail:
        print(f"         {detail}")


# 1. the dataclass has a gflowrl field
tree = ast.parse(open(ACTOR_PY).read())
fields = []
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "PolicyLossConfig":
        fields = [n.target.id for n in node.body if isinstance(n, ast.AnnAssign)]
check("PolicyLossConfig declares a gflowrl field", "gflowrl" in fields,
      f"fields: {fields}")

# 2. the yaml declares this key
y = yaml.safe_load(open(ACTOR_YAML))
check("actor.yaml has gflowrl under policy_loss",
      "gflowrl" in (y.get("policy_loss") or {}),
      f"policy_loss keys: {sorted((y.get('policy_loss') or {}).keys())}")

# 3. the set of keys the wrapper passes
sh = open(WRAPPER).read()
passed = set(re.findall(r"policy_loss\.gflowrl\.([a-z_]+)=", sh))
check("wrapper passes at least variant and beta", {"variant", "beta"} <= passed,
      f"passed: {sorted(passed)}")

# 4. the set of keys the code actually reads — half in the driver + half in the loss, must be covered by the wrapper
read_driver = set(re.findall(r'_gf\.get\("([a-z_]+)"', open(TRAINER).read()))
read_loss = set(re.findall(r'_cfg\.get\("([a-z_]+)"', open(ALGOS).read()))
read = read_driver | read_loss
check("every key the code reads is passed by the wrapper", read <= passed,
      f"driver reads {sorted(read_driver)} · loss reads {sorted(read_loss)}"
      + (f"  ← missing {sorted(read - passed)}" if read - passed else ""))
check("wrapper passes no extra keys that nobody reads", passed <= read,
      f"extra: {sorted(passed - read)}" if passed - read else "")

# 5. the only difference between both arms must be the objective
#    run_rl_gflowrl.sh is a thin wrapper around run_rl.sh; the whole delta is in GFLOWRL_OVERRIDES.
#    That array may only touch policy_loss.* and kl_loss_coef — touching anything else adds a second difference
#    besides the objective, and the paired McNemar immediately becomes meaningless.
ov = re.search(r"GFLOWRL_OVERRIDES=\((.*?)\n\)", sh, re.S)
keys = set(re.findall(r"\+?([A-Za-z_][\w.]*)=", ov.group(1))) if ov else set()
allowed = {k for k in keys
           if k.startswith("actor_rollout_ref.actor.policy_loss.")
           or k == "actor_rollout_ref.actor.kl_loss_coef"}
check("the delta between both arms only touches policy_loss.* and kl_loss_coef", keys and keys == allowed,
      f"override keys: {sorted(keys)}"
      + (f"  ← out of bounds: {sorted(keys - allowed)}" if keys - allowed else ""))

# 6. rollout correction must be off in both arms, and nobody may quietly turn it on
#    The SpaceTools release config has it off (see around P7 handoff §4.5). Turning it on in only one arm introduces a second difference;
#    turning it on in both arms deviates from the original config of the GRPO control arm. So the answer is: touch neither side.
ro = yaml.safe_load(open(ROLLOUT_YAML)) or {}
rc = yaml.safe_load(open(RCORR_YAML)) or {}
check("rollout.yaml calculate_log_probs is still False",
      ro.get("calculate_log_probs") is False,
      f"calculate_log_probs = {ro.get('calculate_log_probs')!r}"
      "   (if True, rollout_log_probs enters the batch and the first gate of rollout correction opens)")
check("rollout_correction rollout_is / rollout_rs are still null",
      rc.get("rollout_is") is None and rc.get("rollout_rs") is None,
      f"rollout_is = {rc.get('rollout_is')!r} · rollout_rs = {rc.get('rollout_rs')!r}"
      "   (note the dataclass defaults are 'sequence'/2.0; it is this yaml that overrides them to null)")

base = open(BASE_SH).read()
touched = {k for k in ("calculate_log_probs", "rollout_is", "rollout_rs", "rollout_correction")
           if k in base or k in sh}
check("neither launch script overrides these switches", not touched,
      f"touched: {sorted(touched)}" if touched else "")

# 7. the reward degeneracy rate must be logged in both arms — i.e. the call site must not sit inside the gflowrl branch
#    If only one arm has this number it proves nothing (P7 handoff 4.4).
trainer_src = open(TRAINER).read()


def indent_of(pattern):
    m = re.search(rf"\n([ \t]*){re.escape(pattern)}", trainer_src)
    return (len(m.group(1)), m.start()) if m else (None, None)


i_adv, p_adv = indent_of("batch = compute_advantage(")
i_deg, p_deg = indent_of("metrics.update(compute_group_degeneracy_metrics(batch))")
_, p_branch = indent_of('if (_pl.get("loss_mode"')
check("degeneracy metric has the same indentation as compute_advantage (= not inside the gflowrl branch)",
      i_deg is not None and i_deg == i_adv,
      f"compute_advantage indent {i_adv} · degeneracy call indent {i_deg}")
check("degeneracy metric comes before the gflowrl branch",
      None not in (p_deg, p_branch) and p_adv < p_deg < p_branch, "")

print("\n" + "=" * 62)
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
