"""
Infra 自检:在 ref / old_log_prob / update_actor 之前丢掉退化组,更新与 P1-a(只置零 g~)相同。

与 p1a_check.py 同样的纪律:select_nondegenerate_groups / compute_gflowrl_flow_gap 从
打了补丁的 ray_trainer.py 抽源码 exec,compute_policy_loss_gflowrl 从 core_algos.py 抽。
dp_actor 的梯度累加照 verl/workers/actor/dp_actor.py L562-585 复刻:
    gradient_accumulation = ppo_mini_batch_size(配置值,每卡) // micro
    loss_scale_factor     = 1 / gradient_accumulation     (与本 micro-batch 实际行数无关)
FSDP 在各 rank 间对梯度取平均。
用法: python infra_drop_check.py PATCHED_RAY_TRAINER CORE_ALGOS
"""
import ast, sys, types
import numpy as np
import torch

PATCHED, CORE = sys.argv[1:3]



def extract(path, fname):
    src = open(path).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == fname:
            return ast.get_source_segment(src, node)
    raise SystemExit(f"{fname} not found in {path}")


def masked_sum(x, mask, axis=None):
    return (x * mask).sum(axis=axis)


def masked_mean(x, mask, axis=None):
    return (x * mask).sum(axis=axis) / mask.sum(axis=axis).clamp(min=1)


class DataProto:
    def __init__(self, batch, non_tensor_batch):
        self.batch, self.non_tensor_batch = batch, non_tensor_batch

    def select_idxs(self, idx):
        t = torch.as_tensor(idx)
        return DataProto({k: v[t] for k, v in self.batch.items()},
                         {k: v[np.asarray(idx)] for k, v in self.non_tensor_batch.items()})


def load(path, fname):
    ns = {"torch": torch, "np": np, "masked_sum": masked_sum,
          "masked_mean": masked_mean,
          "verl_F": types.SimpleNamespace(masked_sum=masked_sum, masked_mean=masked_mean),
          "DataProto": DataProto, "Optional": __import__("typing").Optional,
          "Any": __import__("typing").Any, "ActorConfig": type("ActorConfig", (), {}),
          "__builtins__": __builtins__}
    exec(compile(extract(path, fname), f"<{path}:{fname}>", "exec"), ns)
    return ns[fname]


select = load(PATCHED, "select_nondegenerate_groups")
flow = load(PATCHED, "compute_gflowrl_flow_gap")
loss_fn = load(CORE, "compute_policy_loss_gflowrl")

DP, MICRO, G, NP, T = 4, 2, 5, 16, 40          # 80 行,每卡配置 mini = 20 行,GA = 10
B = G * NP
MINI = B // DP
GA = MINI // MICRO

fails = []
def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)


def make(seed, n_degen_groups):
    g = torch.Generator().manual_seed(seed)
    lengths = torch.randint(8, T + 1, (B,), generator=g)
    mask = (torch.arange(T)[None, :] < lengths[:, None]).float()
    uid = np.repeat([f"p{j:02d}" for j in range(NP)], G)
    r = torch.rand(B, generator=g)
    for j in range(n_degen_groups):                     # 前 n 组退化:全对 / 全错 / 连续全同
        r[j * G:(j + 1) * G] = [1.0, 0.0, 0.37][j % 3]
    rm = torch.zeros(B, T); rm[torch.arange(B), lengths - 1] = r
    old_lp = -torch.rand(B, T, generator=g) * mask
    ref_lp = -torch.rand(B, T, generator=g) * mask
    batch = {"response_mask": mask, "old_log_probs": old_lp, "ref_log_prob": ref_lp,
             "rm_scores": rm, "token_level_scores": rm.clone(),
             "attention_mask": torch.ones(B, 4 + T)}
    return DataProto(batch, {"uid": uid}), lengths


def grad_of(data, perm_seed=0):
    """dp_actor on-policy 更新的梯度,对 log_prob 求导;行按随机分区摊到 DP 个 rank。"""
    d, _ = flow(data, beta=8.0, filter_degenerate=True)
    n = d.batch["response_mask"].shape[0]
    assert n % DP == 0
    theta = d.batch["old_log_probs"].clone().requires_grad_(True)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(perm_seed))  # balance_batch 重排
    per_rank = n // DP
    total = 0.0
    for k in range(DP):
        rows = order[k * per_rank:(k + 1) * per_rank]
        rank_loss = 0.0
        for m in range(0, per_rank, MICRO):
            idx = rows[m:m + MICRO]
            lp = theta[idx]
            l, _ = loss_fn(old_log_prob=lp.detach(), log_prob=lp,
                           advantages=d.batch["advantages"][idx], response_mask=d.batch["response_mask"][idx])
            rank_loss = rank_loss + l * (1.0 / GA)
        total = total + rank_loss / DP                     # FSDP: 各 rank 梯度取平均
    total.backward()
    # 把梯度按 (uid, 行内序号) 对齐回原始行,方便比较
    return theta.grad, d


for n_degen in (0, 5, 11, 16):
    print(f"\n== 退化组 {n_degen}/{NP} ==")
    data, lengths = make(seed=n_degen, n_degen_groups=n_degen)
    kept, m = select(DataProto(dict(data.batch), dict(data.non_tensor_batch)), pad_multiple=DP * MICRO)
    n_keep = (NP - n_degen) * G
    n_after = int(m["gflowrl_drop/rows_after"])
    check("行数是 dp*micro 的整数倍", n_after % (DP * MICRO) == 0, f"({n_after})")
    check("非退化行全部保留", n_after - int(m["gflowrl_drop/pad_rows"]) == n_keep)
    npad = int(m["gflowrl_drop/pad_rows"])
    if npad:
        L_all = data.batch["response_mask"].sum(-1)
        deg_rows = [i for i in range(n_degen * G)]
        chosen = {float(kept.batch["old_log_probs"][i].sum()) for i in range(n_after)}
        pad_L = sorted(int(L_all[i]) for i in deg_rows if float(data.batch["old_log_probs"][i].sum()) in chosen)
        rest_L = sorted(int(L_all[i]) for i in deg_rows if float(data.batch["old_log_probs"][i].sum()) not in chosen)
        check("补齐用的是最短的退化行", not rest_L or max(pad_L) <= min(rest_L), f"(补 {npad} 行)")
    check("kept_rollout_frac", abs(m["gflowrl_drop/kept_rollout_frac"] - n_keep / B) < 1e-12)

    g_full, d_full = grad_of(DataProto(dict(data.batch), dict(data.non_tensor_batch)), perm_seed=1)
    g_drop, d_drop = grad_of(kept, perm_seed=2)
    # 按 (uid, old_log_probs 指纹) 对齐:kept 是原始行的子集
    key = lambda d: [(u, float(d.batch["old_log_probs"][i].sum())) for i, u in enumerate(d.non_tensor_batch["uid"])]
    full_idx = {k: i for i, k in enumerate(key(d_full))}
    sel = [full_idx[k] for k in key(d_drop)]
    others = sorted(set(range(B)) - set(sel))
    diff = (g_full[sel] - g_drop).abs().max().item()
    check("保留行上的梯度与全量(P1-a)一致", diff < 1e-6 * max(1.0, g_full.abs().max().item()), f"(max|diff|={diff:.1e})")
    check("被丢掉的行在全量里梯度恰为 0", bool((g_full[others] == 0).all()) if others else True)
    check("存在非零梯度", n_degen == NP or bool(g_drop.abs().sum() > 0))

print("\n== 反例:只补齐到 dp 的倍数、不管 micro(会出现 1 行的 micro-batch)==")
data, _ = make(seed=99, n_degen_groups=12)                  # 保留 4 组 = 20 行 -> 每卡 5 行
bad, _ = select(DataProto(dict(data.batch), dict(data.non_tensor_batch)), pad_multiple=DP)
per_rank = bad.batch["response_mask"].shape[0] // DP
check("反例确实产生了奇数行的 rank", per_rank % MICRO == 1, f"(每卡 {per_rank} 行)")
g_full, d_full = grad_of(DataProto(dict(data.batch), dict(data.non_tensor_batch)), perm_seed=1)
g_bad, d_bad = grad_of(bad, perm_seed=2)
key = lambda d: [(u, float(d.batch["old_log_probs"][i].sum())) for i, u in enumerate(d.non_tensor_batch["uid"])]
full_idx = {k: i for i, k in enumerate(key(d_full))}
sel = [full_idx[k] for k in key(d_bad)]
diff = (g_full[sel] - g_bad).abs().max().item()
check("反例下梯度确实不一致(所以补齐到 dp*micro 是必要的)", diff > 1e-6, f"(max|diff|={diff:.1e})")

print("\n== 全部退化 ==")
data, _ = make(seed=7, n_degen_groups=NP)
kept, m = select(DataProto(dict(data.batch), dict(data.non_tensor_batch)), pad_multiple=DP * MICRO)
check("全部退化时保留 dp*micro 行", int(m["gflowrl_drop/rows_after"]) == DP * MICRO)

print("\nRESULT:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
