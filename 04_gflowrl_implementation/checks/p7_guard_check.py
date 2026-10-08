"""
P7 -- does the ray_trainer.py guard actually fire?

The two misconfigurations it exists for are SILENT: neither crashes on its own,
both just train a different objective.  So the guard is only worth having if it
really raises.  Extract the guard's source from the real file and run it.

Same discipline as p7_fixedpoint_check.py: test the repo's text, not a copy.
"""
import ast, re, sys

RAY = __import__("os").environ.get("SPACETOOLS_RL", "../../../SpaceTools-RL") + "/verl/trainer/ppo/ray_trainer.py"
src = open(RAY).read()

# pull the guard block verbatim out of fit()
m = re.search(
    r"\n(\s+)# Two misconfigurations here are otherwise SILENT.*?"
    r'\n\s+_gf = _pl\.get\("gflowrl", \{\}\) or \{\}',
    src, re.S,
)
if not m:
    sys.exit("FAIL: guard block not found in ray_trainer.py")
block = m.group(0)
indent = m.group(1)
# strip the common indentation so it can be exec'd at module level
body = "\n".join(l[len(indent):] if l.startswith(indent) else l
                 for l in block.strip("\n").split("\n"))
print("extracted guard from the real source:\n")
print("\n".join("    " + l for l in body.split("\n")[:4]) + "\n    ...\n")


class Cfg(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)


def run(use_kl_loss, kl_loss_coef):
    ns = {"_actor_cfg": Cfg(use_kl_loss=use_kl_loss, kl_loss_coef=kl_loss_coef),
          "_pl": Cfg(), "__builtins__": __builtins__}
    try:
        exec(compile(body, "<guard>", "exec"), ns)
        return None
    except ValueError as e:
        return str(e)


cases = [
    ("correct config          use_kl_loss=True,  coef=0",    True,  0.0,   False),
    ("forgot to zero coef     use_kl_loss=True,  coef=0.01", True,  0.01,  True),
    ("forgot use_kl_loss      use_kl_loss=False, coef=0",    False, 0.0,   True),
    ("both wrong              use_kl_loss=False, coef=0.01", False, 0.01,  True),
]

ok = True
for label, kl_on, kl_c, should_raise in cases:
    err = run(kl_on, kl_c)
    raised = err is not None
    good = raised == should_raise
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] {label}  ->  "
          f"{'raised' if raised else 'passed through'}")
    if raised:
        print(f"         {err.split('.')[0]}.")

print("\n" + "=" * 62)
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
