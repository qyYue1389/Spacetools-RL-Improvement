"""
P7 step 2 -- fixed-point self-check for the GFlowRL implementation.

We do NOT retype the loss.  We parse the two real source files, extract the exact
source segment of the two functions we wrote, and exec that text against stubs for
the verl symbols they touch.  Anything that passes here passes because the code in
the repo is right, not because a copy of it is right.

Prop. B.1: the fixed point of Eq. 8 is  pi_theta(y|x) = pi_ref(y|x) * exp(beta*r(x,y)) / Z(x).
At that point, for every sampled y_i,

    log pi_theta(y_i) - log pi_ref(y_i) = beta*r_i - log Z(x)

so with Z_t estimated by Eq. 4 in-batch, g_i = Z_t - d_i - beta*r_i must be 0 for
every i in the group (d_i = log pi_ref - log pi_old, and at the fixed point
pi_old == pi_theta), the clip is inactive, and the loss is exactly 0.
"""
import ast, sys, types
import numpy as np
import torch

CORE = __import__("os").environ.get("SPACETOOLS_RL", "../../../SpaceTools-RL") + "/verl/trainer/ppo/core_algos.py"
RAY = __import__("os").environ.get("SPACETOOLS_RL", "../../../SpaceTools-RL") + "/verl/trainer/ppo/ray_trainer.py"


def extract(path, fname):
    src = open(path).read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fname:
            seg = ast.get_source_segment(src, node)
            # drop decorators -- they are registry plumbing, not the maths
            return seg
    raise SystemExit(f"{fname} not found in {path}")


# ---- stubs for the verl symbols the two functions touch -------------------
def masked_sum(x, mask, axis=None):
    return (x * mask).sum(axis=axis)


def masked_mean(x, mask, axis=None):
    return (x * mask).sum(axis=axis) / mask.sum(axis=axis).clamp(min=1)


verl_F = types.SimpleNamespace(masked_sum=masked_sum, masked_mean=masked_mean)


class DataProto:
    def __init__(self, batch, non_tensor_batch):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch


ns = {
    "torch": torch, "np": np, "numpy": np,
    "masked_sum": masked_sum, "masked_mean": masked_mean, "verl_F": verl_F,
    "DataProto": DataProto, "Optional": __import__("typing").Optional,
    "Any": __import__("typing").Any,
    "ActorConfig": type("ActorConfig", (), {}), "AdvantageEstimator": object,
    "__builtins__": __builtins__,
}

flow_src = extract(RAY, "compute_gflowrl_flow_gap")
loss_src = extract(CORE, "compute_policy_loss_gflowrl")
exec(compile(flow_src, "<ray_trainer:compute_gflowrl_flow_gap>", "exec"), ns)
exec(compile(loss_src, "<core_algos:compute_policy_loss_gflowrl>", "exec"), ns)
compute_gflowrl_flow_gap = ns["compute_gflowrl_flow_gap"]
compute_policy_loss_gflowrl = ns["compute_policy_loss_gflowrl"]
print("extracted both functions from the real source files")

# ---- build an exact fixed point ------------------------------------------
torch.manual_seed(0)
G, N_PROMPT, T = 8, 4, 40          # group size, prompts, max response tokens
BETA = 8.0
B = G * N_PROMPT

# ragged lengths: this is exactly what C-prime is being tested against
lengths = torch.randint(12, T + 1, (B,))
response_mask = (torch.arange(T)[None, :] < lengths[:, None]).float()

uid = np.repeat([f"p{j}" for j in range(N_PROMPT)], G)

# arbitrary per-sequence rewards, incl. one fully degenerate group (all equal)
r = torch.rand(B)
r[:G] = 0.5                                        # degenerate group
token_level_scores = torch.zeros(B, T)
token_level_scores[torch.arange(B), lengths - 1] = r   # reward on last real token

# arbitrary reference log-probs
ref_lp = -torch.rand(B, T) * response_mask
d_ref = masked_sum(ref_lp, response_mask, axis=-1)

# construct pi_theta = pi_ref * exp(beta*r) / Z  ==>  log pi_theta = log pi_ref + beta*r - log Z
# log Z is per-prompt; any constant works, use the true in-batch Eq.4 constant.
gidx = torch.as_tensor(np.unique(uid, return_inverse=True)[1], dtype=torch.long)
target = BETA * r + d_ref
sums = torch.zeros(N_PROMPT).index_add_(0, gidx, target)
cnts = torch.zeros(N_PROMPT).index_add_(0, gidx, torch.ones(B))
logZ = (sums / cnts)[gidx]

# the sequence log-prob the fixed-point policy assigns; spread the shift over tokens
shift = (BETA * r - logZ) / lengths.float()
old_lp = (ref_lp + shift[:, None]) * response_mask
d_old = masked_sum(old_lp, response_mask, axis=-1)
print(f"  check  d_old - d_ref  vs  beta*r - logZ : "
      f"max|err| = {(d_old - d_ref - (BETA*r - logZ)).abs().max():.3e}")

data = DataProto(
    batch={
        "response_mask": response_mask,
        "old_log_probs": old_lp,
        "ref_log_prob": ref_lp,
        "token_level_scores": token_level_scores,
        "attention_mask": torch.cat([torch.ones(B, 8), response_mask], dim=-1),
    },
    non_tensor_batch={"uid": uid},
)

fails = []
for variant in ("cprime", "paper", "normalized"):
    d2 = DataProto({k: v.clone() for k, v in data.batch.items()}, dict(data.non_tensor_batch))
    d2, m = compute_gflowrl_flow_gap(d2, beta=BETA, variant=variant)
    g_tilde_seq = (d2.batch["advantages"] * response_mask).sum(-1) / response_mask.sum(-1)
    # at the fixed point pi_theta == pi_old, so log_prob = old_log_probs exactly
    loss, lm = compute_policy_loss_gflowrl(
        old_log_prob=old_lp, log_prob=old_lp.clone(),
        advantages=d2.batch["advantages"], response_mask=response_mask,
    )
    print(f"\n[{variant}]")
    print(f"  max |g~_i|        = {g_tilde_seq.abs().max().item():.3e}")
    print(f"  clip_saturation   = {m['gflowrl/clip_saturation']:.4f}")
    print(f"  loss              = {loss.item():.6e}")
    ok = loss.item() < 1e-10 and g_tilde_seq.abs().max().item() < 1e-5
    print(f"  fixed point       = {'PASS' if ok else 'FAIL'}")
    if variant == "cprime" and not ok:
        fails.append(variant)
    if variant == "cprime":
        # --- the beta-decomposition diagnostics added alongside Eq. 6 -----------
        # 1. the sweep at the configured beta must reproduce clip_saturation
        d_sat = abs(m["gflowrl/sat_at_beta_8"] - m["gflowrl/clip_saturation"])
        # 2. at the fixed point g == 0, so drift must be exactly minus reward and
        #    the two |.|-means must coincide -- this is what proves the split is the
        #    real decomposition and not two unrelated numbers
        d_split = abs(m["gflowrl/reward_term_abs_mean"] - m["gflowrl/drift_term_abs_mean"])
        # 3. this batch is built with exactly one fully degenerate group out of four
        d_nondeg = abs(m["gflowrl/nondegenerate_frac"] - 0.75)
        print(f"  sat_at_beta_8 vs clip_saturation : |d| = {d_sat:.3e}  (must be ~0)")
        print(f"  |reward_term| vs |drift_term|    : |d| = {d_split:.3e}  (must be ~0 at the fixed point)")
        print(f"  nondegenerate_frac               : {m['gflowrl/nondegenerate_frac']:.4f}  (must be 0.75)")
        print("  beta sweep  " + "  ".join(
            f"b={b}:{m['gflowrl/sat_at_beta_' + b]:.4f}" for b in ("2", "4", "8", "16")))
        if d_sat > 1e-12 or d_split > 1e-4 or d_nondeg > 1e-9:
            fails.append("cprime-diagnostics")
    if variant != "cprime":
        print(f"  (informational -- 'paper'/'normalized' are NOT expected to be 0 here;")
        print(f"   the Eq.4/Eq.5-6 normalization mismatch is exactly what P7 documented)")

# ---- gradient sanity: perturb theta away from the fixed point -------------
print("\n--- gradient check (cprime) ---")
d3 = DataProto({k: v.clone() for k, v in data.batch.items()}, dict(data.non_tensor_batch))
d3, _ = compute_gflowrl_flow_gap(d3, beta=BETA, variant="cprime")
theta = (old_lp.clone() + 0.01 * torch.randn(B, T) * response_mask).requires_grad_(True)
loss, lm = compute_policy_loss_gflowrl(
    old_log_prob=old_lp, log_prob=theta,
    advantages=d3.batch["advantages"], response_mask=response_mask,
)
loss.backward()
gnorm = theta.grad.norm().item()
print(f"  loss at perturbed theta = {loss.item():.6e}  (must be > 0)")
print(f"  ||dL/dlogpi||           = {gnorm:.6e}  (must be > 0)")
print(f"  grad on padding         = {(theta.grad * (1 - response_mask)).abs().max().item():.3e}  (must be 0)")
print(f"  is_weight_clipped_frac  = {lm['gflowrl/is_weight_clipped_frac']:.4f}")

grad_ok = loss.item() > 0 and gnorm > 0 and (theta.grad * (1 - response_mask)).abs().max().item() == 0
print(f"\n{'=' * 60}")
print("RESULT:", "PASS" if (not fails and grad_ok) else f"FAIL {fails}")
sys.exit(0 if (not fails and grad_ok) else 1)
