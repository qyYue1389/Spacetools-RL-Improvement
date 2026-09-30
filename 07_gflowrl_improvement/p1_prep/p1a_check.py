"""
P1-a 自检:过滤退化组(g~ 置零)在 on-policy 下与 loss mask 逐位等价。

与 p7_fixedpoint_check.py 同样的纪律:从真实源文件里抽出函数源码 exec,不重抄。
  PATCHED = 打了 P1-a 补丁的 ray_trainer.py
  ORIG    = 未打补丁的 ray_trainer.py(用来确认 filter 关闭时逐位无回归)
  CORE    = core_algos.py(compute_policy_loss_gflowrl,未改)
用法: python p1a_check.py PATCHED ORIG CORE
"""
import ast, sys, types
import numpy as np
import torch

PATCHED, ORIG, CORE = sys.argv[1:4]


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


def load(path, fname):
    ns = {"torch": torch, "np": np, "masked_sum": masked_sum, "masked_mean": masked_mean,
          "verl_F": types.SimpleNamespace(masked_sum=masked_sum, masked_mean=masked_mean),
          "DataProto": DataProto, "Optional": __import__("typing").Optional,
          "Any": __import__("typing").Any, "ActorConfig": type("ActorConfig", (), {}),
          "__builtins__": __builtins__}
    exec(compile(extract(path, fname), f"<{path}:{fname}>", "exec"), ns)
    return ns[fname]


flow_new = load(PATCHED, "compute_gflowrl_flow_gap")
flow_old = load(ORIG, "compute_gflowrl_flow_gap")
loss_fn = load(CORE, "compute_policy_loss_gflowrl")

torch.manual_seed(0)
G, NP, T = 5, 6, 60
B = G * NP
lengths = torch.randint(20, T + 1, (B,))
mask = (torch.arange(T)[None, :] < lengths[:, None]).float()
uid = np.repeat([f"p{j}" for j in range(NP)], G)
r = torch.rand(B)
r[0:G] = 1.0                 # 退化:全对
r[G:2 * G] = 0.0             # 退化:全错
r[2 * G:3 * G] = 0.37        # 退化:连续奖励但全同
r[3 * G] = r[3 * G + 1]      # 部分相同但不退化
tls = torch.zeros(B, T); tls[torch.arange(B), lengths - 1] = r
ref_lp = -torch.rand(B, T) * mask
old_lp = -torch.rand(B, T) * mask
batch = {"response_mask": mask, "old_log_probs": old_lp, "ref_log_prob": ref_lp,
         "token_level_scores": tls, "attention_mask": torch.cat([torch.ones(B, 8), mask], -1)}


def run(fn, **kw):
    d = DataProto({k: v.clone() for k, v in batch.items()}, {"uid": uid.copy()})
    d, m = fn(d, beta=8.0, **kw)
    return d.batch["advantages"], m


fails = []
def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails.append(name)


print("1) filter 关闭时与原函数逐位一致(无回归)")
a_old, _ = run(flow_old)
a_off, m_off = run(flow_new, filter_degenerate=False)
check("advantages bit-identical", torch.equal(a_old, a_off))

print("2) filter 打开:退化组 g~ 恰为 0,其余组不变")
a_on, m_on = run(flow_new, filter_degenerate=True)
seq = lambda a: (a * mask).sum(-1) / mask.sum(-1)
degen = torch.zeros(B, dtype=torch.bool); degen[:3 * G] = True
check("degenerate rows == 0", bool((seq(a_on)[degen] == 0).all()))
check("kept rows unchanged", torch.equal(a_on[~degen], a_off[~degen]))
check("kept_groups == 3", m_on["gflowrl/kept_groups"] == 3.0, f"(got {m_on['gflowrl/kept_groups']})")
check("kept_rollout_frac == 0.5", abs(m_on["gflowrl/kept_rollout_frac"] - 0.5) < 1e-9)
check("before filtering, degenerate rows had nonzero g~ (pure drift)", bool((seq(a_off)[degen].abs() > 0).any()))

print("3) on-policy 梯度:置零 g~ == 真正的 loss mask(分母保持全部序列),逐位相等")
# dp_actor on-policy 分支: old_log_prob = log_prob.detach()
theta1 = old_lp.clone().requires_grad_(True)
l1, _ = loss_fn(old_log_prob=theta1.detach(), log_prob=theta1, advantages=a_on, response_mask=mask)
l1.backward()
theta2 = old_lp.clone().requires_grad_(True)
g_seq = seq(a_off)
log_ratio = masked_sum(theta2 - theta2.detach(), mask, axis=-1)
w = torch.clamp(torch.exp(log_ratio.detach()), max=1.2)
l2 = (w * (g_seq + log_ratio).pow(2) * (~degen).float()).sum() / B
l2.backward()
check("grad bit-identical", torch.equal(theta1.grad, theta2.grad),
      f"(max|diff|={(theta1.grad - theta2.grad).abs().max().item():.1e})")
check("grad on degenerate rows == 0", bool((theta1.grad[degen] == 0).all()))
check("grad on kept rows != 0", bool((theta1.grad[~degen].abs().sum(-1) > 0).all()))

print("4) 反例:off-policy(log_prob != old_log_prob)时等价性不成立 —— 守卫存在的理由")
theta3 = (old_lp + 0.01 * torch.randn(B, T) * mask).requires_grad_(True)
l3, _ = loss_fn(old_log_prob=old_lp, log_prob=theta3, advantages=a_on, response_mask=mask)
l3.backward()
check("off-policy degenerate rows DO get gradient (proximal term)", bool((theta3.grad[degen].abs().sum() > 0)))

print("5) 新日志字段")
for k in ("gflowrl/clip_saturation_kept", "gflowrl/g_abs_p50", "gflowrl/g_abs_p90", "gflowrl/g_abs_p99"):
    check(k, k in m_on and m_on[k] == m_on[k], f"= {m_on.get(k):.4f}")

print("\nRESULT:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(1 if fails else 0)
