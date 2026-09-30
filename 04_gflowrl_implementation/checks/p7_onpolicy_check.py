"""
P7 -- 在我们的 batch 配置下,Eq.8 的 IS 权重 w 到底是不是恒等于 1?

dp_actor.py:550
    on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
dp_actor.py:598
    if on_policy:  old_log_prob = log_prob.detach()

on_policy 为真时 log_ratio 的**数值**恒为 0(梯度不为 0),于是
    w = clamp(exp(0), max=1+eps_is) = 1        <- eps_is 完全不起作用
    delta = g~ + 0                              <- loss 数值 = mean(g~^2)
更新退化成 REINFORCE 形式:grad = mean(2*g~_i * grad masked_sum(log pi_theta))。

这不是缺陷,是「一个 rollout batch 只走一步优化」的必然结果 —— 没有 pi_theta/pi_old
失配可纠。但它是**配置依赖**的:调 ppo_mini_batch_size 或 ppo_epochs 就会翻转,
届时 w 变成整条序列的概率比 exp(sum over ~1e3 tokens),而 Eq.8 只 clip 上界。
所以这个脚本既是记录也是护栏。

本脚本同时守 fsdp_workers.py:263 那条整除断言 —— 训练卡数不是随便选的:
  归一化后的 ppo_mini_batch_size = (train_batch_size × rollout.n) // 训练卡数
  它必须能被 ppo_micro_batch_size_per_gpu 整除,否则 worker 初始化就抛。
  6 卡和 7 卡训练会 FAIL(53 / 45 都不能被 2 整除),而这一点在别处没有任何记录。
  另外 fsdp_size=2(HSDP,建议开)要求训练卡数是偶数,
  因为 create_device_mesh 的 mesh_shape=(world_size // fsdp_size, fsdp_size)。

同 p7_guard_check.py:数字从真实脚本里抠,不手抄。
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
        sys.exit(f"FAIL: 在 run_rl.sh 里找不到 {key}")
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
    # fsdp_workers.py:249-250,无条件
    mini = (mini_cfg * n) // ranks
    per_rank = (train_bsz * n) // ranks          # 该 rank 拿到的序列数
    n_mini = math.ceil(per_rank / mini) if mini else 0
    on_policy = (n_mini == 1 and epochs == 1)
    good = on_policy
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] 训练卡 {ranks}: 归一化后 mini={mini} · "
          f"每卡 {per_rank} 条 · mini-batch {n_mini} 个 -> on_policy={on_policy}"
          f"  =>  w {'恒为 1' if on_policy else '是真实序列比值'}")

print()
# ---- 训练卡数的合法性(fsdp_workers.py:263 的整除断言 + HSDP 的偶数要求) ----
print("训练卡数合法性:归一化 mini 必须能被 micro 整除;fsdp_size=2 还要求偶数")
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
            else "△ 只能 fsdp_size=-1" if ok_div
            else "✗ fsdp_workers.py:263 会抛")
    print(f"    训练卡 {ranks}: mini={mini:4d}  mini%micro={mini % micro}  "
          f"{'偶数' if ok_hsdp else '奇数'}   {mark}")

# 我们计划用到的三种训练卡数必须合法,而 6/7 必须被正确识别为非法
#   4 张机器 -> 工具 2 + 训练 2 | 6 张 -> 工具 2 + 训练 4 | 8 张 -> 工具 4 + 训练 4
check("计划用的训练卡数 2 / 4 都合法(含 fsdp_size=2)",
      {2, 4} <= set(legal_hsdp), f"fsdp_size=2 下合法的卡数: {legal_hsdp}")
check("6 卡与 7 卡训练被正确识别为非法",
      6 not in legal_plain and 7 not in legal_plain,
      "「8 张 = 工具 2 + 训练 6」这个直觉配法会启动即崩,这一条就是拦它的")

print()
# 什么会打破它
print("会翻转 on_policy 的改动(届时 eps_is 才开始起作用,且 w 可能塌到 0):")
for label, mb, ep in (("ppo_mini_batch_size 64 -> 32", 32, epochs),
                      ("ppo_mini_batch_size 64 -> 16", 16, epochs),
                      ("ppo_epochs 1 -> 2", mini_cfg, 2)):
    mini = (mb * n) // 4
    per_rank = (train_bsz * n) // 4
    n_mini = math.ceil(per_rank / mini) if mini else 0
    print(f"    {label:32s} (4 卡) -> mini-batch {n_mini} 个, epochs {ep} "
          f"-> on_policy={n_mini == 1 and ep == 1}")

print("\n" + "=" * 62)
print("RESULT:", "PASS -- w 恒为 1,eps_is 在当前配置下不起作用" if ok
      else "FAIL -- w 不再恒为 1,eps_is 与 IS 塌陷风险都变成真的")
sys.exit(0 if ok else 1)
