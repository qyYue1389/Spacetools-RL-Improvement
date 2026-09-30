"""
P7 step 2b -- does C-prime produce gradient on reward-degenerate groups?

P6 measured 56.3-79.0% of groups have all-identical rewards.  GRPO's advantage is
(r - mean)/std, so those groups contribute EXACTLY ZERO gradient -- more than half
the rollout budget is discarded.  If C-prime's g~ is non-zero there, that is a
concrete mechanism by which GFlowRL could beat GRPO, and it is testable now,
before any GPU is rented.

Same extraction discipline as p7_fixedpoint_check.py: the real source, not a copy.
"""
import ast, types
import numpy as np
import torch

RAY = __import__("os").environ.get("SPACETOOLS_RL", "../../../SpaceTools-RL") + "/verl/trainer/ppo/ray_trainer.py"


def masked_sum(x, mask, axis=None):
    return (x * mask).sum(axis=axis)


def masked_mean(x, mask, axis=None):
    return (x * mask).sum(axis=axis) / mask.sum(axis=axis).clamp(min=1)


class DataProto:
    def __init__(self, batch, non_tensor_batch):
        self.batch, self.non_tensor_batch = batch, non_tensor_batch


src = open(RAY).read()
seg = next(ast.get_source_segment(src, n) for n in ast.parse(src).body
           if isinstance(n, ast.FunctionDef) and n.name == "compute_gflowrl_flow_gap")
ns = {"torch": torch, "np": np, "masked_sum": masked_sum, "masked_mean": masked_mean,
      "DataProto": DataProto, "__builtins__": __builtins__}
exec(compile(seg, "<real>", "exec"), ns)
flow = ns["compute_gflowrl_flow_gap"]

torch.manual_seed(1)
G, N_PROMPT, T, BETA = 8, 6, 60, 8.0
B = G * N_PROMPT
lengths = torch.randint(20, T + 1, (B,))
response_mask = (torch.arange(T)[None, :] < lengths[:, None]).float()
uid = np.repeat([f"p{j}" for j in range(N_PROMPT)], G)

# groups 0,1,2 fully degenerate (all-wrong / all-right); 3,4,5 mixed
r = torch.zeros(B)
r[0 * G:1 * G] = 0.0      # all wrong
r[1 * G:2 * G] = 1.0      # all right
r[2 * G:3 * G] = 0.0      # all wrong
for j in range(3, N_PROMPT):
    r[j * G:(j + 1) * G] = (torch.rand(G) > 0.5).float()

tls = torch.zeros(B, T)
tls[torch.arange(B), lengths - 1] = r

# a policy that has drifted from ref (this is what step >0 looks like)
ref_lp = (-torch.rand(B, T) * 2) * response_mask
old_lp = (ref_lp / response_mask.clamp(min=1) - 0.05 * torch.randn(B, T)) * response_mask

data = DataProto({"response_mask": response_mask, "old_log_probs": old_lp,
                  "ref_log_prob": ref_lp, "token_level_scores": tls,
                  "attention_mask": torch.cat([torch.ones(B, 8), response_mask], -1)},
                 {"uid": uid})
data, m = flow(data, beta=BETA, variant="cprime")
g = (data.batch["advantages"] * response_mask).sum(-1) / response_mask.sum(-1)

print(f"\nclip_saturation overall = {m['gflowrl/clip_saturation']:.4f}\n")
print(f"{'group':>6} {'rewards':>18} {'degenerate':>11} {'mean|g~|':>10} {'std g~':>10}  GRPO adv")
for j in range(N_PROMPT):
    sl = slice(j * G, (j + 1) * G)
    rj, gj = r[sl], g[sl]
    deg = bool((rj == rj[0]).all())
    grpo = "0 (dropped)" if deg else f"{((rj - rj.mean()) / rj.std().clamp(min=1e-6)).abs().mean():.3f}"
    print(f"{j:>6} {str(rj.tolist()[:4])[:18]:>18} {str(deg):>11} "
          f"{gj.abs().mean():>10.4f} {gj.std():>10.4f}  {grpo}")

deg_mask = torch.tensor([bool((r[j * G:(j + 1) * G] == r[j * G]).all()) for j in range(N_PROMPT)])
deg_idx = torch.cat([torch.arange(j * G, (j + 1) * G) for j in range(N_PROMPT) if deg_mask[j]])
nod_idx = torch.cat([torch.arange(j * G, (j + 1) * G) for j in range(N_PROMPT) if not deg_mask[j]])
print(f"\ndegenerate groups     : {int(deg_mask.sum())}/{N_PROMPT}  "
      f"({100*len(deg_idx)/B:.1f}% of rollouts)")
print(f"  mean |g~| there     = {g[deg_idx].abs().mean():.4f}   "
      f"(GRPO would give exactly 0)")
print(f"  mean |g~| elsewhere = {g[nod_idx].abs().mean():.4f}")
print(f"  ratio               = {g[deg_idx].abs().mean()/g[nod_idx].abs().mean():.3f}")
print(f"\nfraction of ALL rollouts with |g~| < 1e-6 = "
      f"{(g.abs() < 1e-6).float().mean():.4f}  (GRPO's would be "
      f"{len(deg_idx)/B:.4f})")
