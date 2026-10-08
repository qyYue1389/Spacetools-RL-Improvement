"""
Infra self-check: drop degenerate groups before ref / old_log_prob / update_actor; the update is identical to P1-a (only zeroing g~).

Same discipline as p1a_check.py: select_nondegenerate_groups / compute_gflowrl_flow_gap are extracted as source from
the patched ray_trainer.py and exec'd; compute_policy_loss_gflowrl is extracted from core_algos.py.
dp_actor gradient accumulation is replicated from verl/workers/actor/dp_actor.py L562-585:
    gradient_accumulation = ppo_mini_batch_size (config value, per GPU) // micro
    loss_scale_factor     = 1 / gradient_accumulation     (independent of the actual row count of this micro-batch)
FSDP averages gradients across ranks.
Usage: python infra_drop_check.py PATCHED_RAY_TRAINER CORE_ALGOS
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

DP, MICRO, G, NP, T = 4, 2, 5, 16, 40          # 80 rows, configured mini per GPU = 20 rows, GA = 10
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
    for j in range(n_degen_groups):                     # first n groups degenerate: all correct / all wrong / continuous all identical
        r[j * G:(j + 1) * G] = [1.0, 0.0, 0.37][j % 3]
    rm = torch.zeros(B, T); rm[torch.arange(B), lengths - 1] = r
    old_lp = -torch.rand(B, T, generator=g) * mask
    ref_lp = -torch.rand(B, T, generator=g) * mask
    batch = {"response_mask": mask, "old_log_probs": old_lp, "ref_log_prob": ref_lp,
             "rm_scores": rm, "token_level_scores": rm.clone(),
             "attention_mask": torch.ones(B, 4 + T)}
    return DataProto(batch, {"uid": uid}), lengths


def grad_of(data, perm_seed=0):
    """Gradient of the dp_actor on-policy update w.r.t. log_prob; rows are spread across DP ranks by a random partition."""
    d, _ = flow(data, beta=8.0, filter_degenerate=True)
    n = d.batch["response_mask"].shape[0]
    assert n % DP == 0
    theta = d.batch["old_log_probs"].clone().requires_grad_(True)
    order = torch.randperm(n, generator=torch.Generator().manual_seed(perm_seed))  # balance_batch reordering
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
        total = total + rank_loss / DP                     # FSDP: gradients averaged across ranks
    total.backward()
    # align gradients back to the original rows by (uid, in-row index) for comparison
    return theta.grad, d


for n_degen in (0, 5, 11, 16):
    print(f"\n== degenerate groups {n_degen}/{NP} ==")
    data, lengths = make(seed=n_degen, n_degen_groups=n_degen)
    kept, m = select(DataProto(dict(data.batch), dict(data.non_tensor_batch)), pad_multiple=DP * MICRO)
    n_keep = (NP - n_degen) * G
    n_after = int(m["gflowrl_drop/rows_after"])
    check("row count is a multiple of dp*micro", n_after % (DP * MICRO) == 0, f"({n_after})")
    check("all non-degenerate rows kept", n_after - int(m["gflowrl_drop/pad_rows"]) == n_keep)
    npad = int(m["gflowrl_drop/pad_rows"])
    if npad:
        L_all = data.batch["response_mask"].sum(-1)
        deg_rows = [i for i in range(n_degen * G)]
        chosen = {float(kept.batch["old_log_probs"][i].sum()) for i in range(n_after)}
        pad_L = sorted(int(L_all[i]) for i in deg_rows if float(data.batch["old_log_probs"][i].sum()) in chosen)
        rest_L = sorted(int(L_all[i]) for i in deg_rows if float(data.batch["old_log_probs"][i].sum()) not in chosen)
        check("padding uses the shortest degenerate rows", not rest_L or max(pad_L) <= min(rest_L), f"(padded {npad} rows)")
    check("kept_rollout_frac", abs(m["gflowrl_drop/kept_rollout_frac"] - n_keep / B) < 1e-12)

    g_full, d_full = grad_of(DataProto(dict(data.batch), dict(data.non_tensor_batch)), perm_seed=1)
    g_drop, d_drop = grad_of(kept, perm_seed=2)
    # align by (uid, old_log_probs fingerprint): kept is a subset of the original rows
    key = lambda d: [(u, float(d.batch["old_log_probs"][i].sum())) for i, u in enumerate(d.non_tensor_batch["uid"])]
    full_idx = {k: i for i, k in enumerate(key(d_full))}
    sel = [full_idx[k] for k in key(d_drop)]
    others = sorted(set(range(B)) - set(sel))
    diff = (g_full[sel] - g_drop).abs().max().item()
    check("gradients on kept rows match the full batch (P1-a)", diff < 1e-6 * max(1.0, g_full.abs().max().item()), f"(max|diff|={diff:.1e})")
    check("dropped rows have exactly 0 gradient in the full batch", bool((g_full[others] == 0).all()) if others else True)
    check("non-zero gradient exists", n_degen == NP or bool(g_drop.abs().sum() > 0))

print("\n== counterexample: pad only to a multiple of dp, ignoring micro (a 1-row micro-batch appears) ==")
data, _ = make(seed=99, n_degen_groups=12)                  # keep 4 groups = 20 rows -> 5 rows per GPU
bad, _ = select(DataProto(dict(data.batch), dict(data.non_tensor_batch)), pad_multiple=DP)
per_rank = bad.batch["response_mask"].shape[0] // DP
check("counterexample really produces a rank with an odd row count", per_rank % MICRO == 1, f"({per_rank} rows per GPU)")
g_full, d_full = grad_of(DataProto(dict(data.batch), dict(data.non_tensor_batch)), perm_seed=1)
g_bad, d_bad = grad_of(bad, perm_seed=2)
key = lambda d: [(u, float(d.batch["old_log_probs"][i].sum())) for i, u in enumerate(d.non_tensor_batch["uid"])]
full_idx = {k: i for i, k in enumerate(key(d_full))}
sel = [full_idx[k] for k in key(d_bad)]
diff = (g_full[sel] - g_bad).abs().max().item()
check("gradients really differ under the counterexample (so padding to dp*micro is necessary)", diff > 1e-6, f"(max|diff|={diff:.1e})")

print("\n== all degenerate ==")
data, _ = make(seed=7, n_degen_groups=NP)
kept, m = select(DataProto(dict(data.batch), dict(data.non_tensor_batch)), pad_multiple=DP * MICRO)
check("all degenerate: dp*micro rows kept", int(m["gflowrl_drop/rows_after"]) == DP * MICRO)

print("\nRESULT:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
