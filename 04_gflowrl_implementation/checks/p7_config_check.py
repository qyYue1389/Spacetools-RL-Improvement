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


# 1. dataclass 里有 gflowrl 字段
tree = ast.parse(open(ACTOR_PY).read())
fields = []
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "PolicyLossConfig":
        fields = [n.target.id for n in node.body if isinstance(n, ast.AnnAssign)]
check("PolicyLossConfig 声明了 gflowrl 字段", "gflowrl" in fields,
      f"字段: {fields}")

# 2. yaml 里声明了这个 key
y = yaml.safe_load(open(ACTOR_YAML))
check("actor.yaml 的 policy_loss 下有 gflowrl",
      "gflowrl" in (y.get("policy_loss") or {}),
      f"policy_loss keys: {sorted((y.get('policy_loss') or {}).keys())}")

# 3. wrapper 传的 key 集合
sh = open(WRAPPER).read()
passed = set(re.findall(r"policy_loss\.gflowrl\.([a-z_]+)=", sh))
check("wrapper 至少传了 variant 和 beta", {"variant", "beta"} <= passed,
      f"传了: {sorted(passed)}")

# 4. 代码里真正读的 key 集合 —— driver 一半 + loss 一半,必须被 wrapper 覆盖
read_driver = set(re.findall(r'_gf\.get\("([a-z_]+)"', open(TRAINER).read()))
read_loss = set(re.findall(r'_cfg\.get\("([a-z_]+)"', open(ALGOS).read()))
read = read_driver | read_loss
check("代码读的每个 key 都被 wrapper 传了", read <= passed,
      f"driver 读 {sorted(read_driver)} · loss 读 {sorted(read_loss)}"
      + (f"  ← 缺 {sorted(read - passed)}" if read - passed else ""))
check("wrapper 没传多余的、没人读的 key", passed <= read,
      f"多余: {sorted(passed - read)}" if passed - read else "")

# 5. 两臂唯一的差异必须是目标函数
#    run_rl_gflowrl.sh 是 run_rl.sh 的薄封装,delta 全在 GFLOWRL_OVERRIDES 里。
#    那个数组只许碰 policy_loss.* 和 kl_loss_coef —— 碰别的就等于在目标函数之外
#    又加了一处差异,配对 McNemar 立刻失去意义。
ov = re.search(r"GFLOWRL_OVERRIDES=\((.*?)\n\)", sh, re.S)
keys = set(re.findall(r"\+?([A-Za-z_][\w.]*)=", ov.group(1))) if ov else set()
allowed = {k for k in keys
           if k.startswith("actor_rollout_ref.actor.policy_loss.")
           or k == "actor_rollout_ref.actor.kl_loss_coef"}
check("两臂的 delta 只碰 policy_loss.* 与 kl_loss_coef", keys and keys == allowed,
      f"override 的 key: {sorted(keys)}"
      + (f"  ← 越界: {sorted(keys - allowed)}" if keys - allowed else ""))

# 6. rollout correction 两臂都必须是关的,且没人偷偷打开
#    SpaceTools 发布配置是关的(见 P7 交接 §4.5 附近)。只给一臂开会引入第二处差异;
#    两臂一起开则偏离 GRPO 对照臂的原配置。所以答案是两边都别动。
ro = yaml.safe_load(open(ROLLOUT_YAML)) or {}
rc = yaml.safe_load(open(RCORR_YAML)) or {}
check("rollout.yaml 的 calculate_log_probs 仍为 False",
      ro.get("calculate_log_probs") is False,
      f"calculate_log_probs = {ro.get('calculate_log_probs')!r}"
      "   (为 True 则 rollout_log_probs 进 batch,rollout correction 的第一道门打开)")
check("rollout_correction 的 rollout_is / rollout_rs 仍为 null",
      rc.get("rollout_is") is None and rc.get("rollout_rs") is None,
      f"rollout_is = {rc.get('rollout_is')!r} · rollout_rs = {rc.get('rollout_rs')!r}"
      "   (注意 dataclass 的默认值是 'sequence'/2.0,是这份 yaml 把它覆盖成 null 的)")

base = open(BASE_SH).read()
touched = {k for k in ("calculate_log_probs", "rollout_is", "rollout_rs", "rollout_correction")
           if k in base or k in sh}
check("两个启动脚本都没有覆盖这些开关", not touched,
      f"被碰到的: {sorted(touched)}" if touched else "")

# 7. 奖励简并率必须两臂都记 —— 也就是调用点不能落在 gflowrl 分支里面
#    只有一臂有这个数的话它什么都证明不了(P7 交接 4.4)。
trainer_src = open(TRAINER).read()


def indent_of(pattern):
    m = re.search(rf"\n([ \t]*){re.escape(pattern)}", trainer_src)
    return (len(m.group(1)), m.start()) if m else (None, None)


i_adv, p_adv = indent_of("batch = compute_advantage(")
i_deg, p_deg = indent_of("metrics.update(compute_group_degeneracy_metrics(batch))")
_, p_branch = indent_of('if (_pl.get("loss_mode"')
check("简并率指标与 compute_advantage 同缩进(= 不在 gflowrl 分支内)",
      i_deg is not None and i_deg == i_adv,
      f"compute_advantage 缩进 {i_adv} · 简并率调用缩进 {i_deg}")
check("简并率指标在 gflowrl 分支之前",
      None not in (p_deg, p_branch) and p_adv < p_deg < p_branch, "")

print("\n" + "=" * 62)
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
