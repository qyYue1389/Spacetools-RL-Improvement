#!/usr/bin/env python3
"""P7 · fixed-point cost of the candidate fix (the to-do from section 6 of P7_CRITERION_IB.md).

Section 6 proposed: also normalize the logprob term of Eq.4 by |y|,
    C_norm = mean_i[ beta*r_i + (log pi_ref - log pi_old)_i / |y_i| ]
Synthetically this pushed "whole group of g flattened to the same value" from 36%-93% back to 0.0%, and on real drift also back to 0.0% (see P7_GPU_RESULTS section 4.5).
**But that section only measured the symptom disappearing, not the cost.** This script measures the cost.

Reuses the world construction and conventions of tools/p7/p7_fixedpoint.py (small world |Y|=64, exactly normalizable).

First write out the algebra, then verify numerically:

  Zero loss (Eq.5 with length normalization) requires, for every y:
      Z + (1/L_y)(log pi_theta,y - log pi_ref,y) - beta*r_y = 0
  =>  log pi_theta,y = log pi_ref,y + L_y*(beta*r_y - Z) - c(Z)
      where c(Z) = logsumexp(...) is the normalizing constant.
  Substituting back into the residual:
      residual_i = Z_t - Z - c(Z)/L_i
  When L differs, for it to be zero for all i, **c(Z)=0 and Z_t=Z** must both hold at the same time.

  c(Z)=0 is a monotone equation, **one unknown, one root, generically solvable** -> call it Z*.
  What remains, Z_t = Z*, is the "estimator self-consistency" step:

    Eq.4 as is (unnormalized):
      Z_t = E[beta*r] - E[L*beta*r] + Z*E[L]     -> in general != Z*   (T3b's "fixed point does not exist")
    Eq.4 after normalization (the section 6 fix):
      Z_t = E[ beta*r + (log pi_ref - log pi_theta)/L ] = E[ beta*r + Z - beta*r ] = Z   ✓ identically

  **So the fix algebraically restores the existence of the fixed point. The cost is not in existence, it is in the location of the fixed point.**
"""
import numpy as np

RNG = np.random.default_rng(20260902)
BETA = 8.0
EPS_LOW, EPS_HIGH, EPS_IS = 0.2, 0.28, 0.2


def lse(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


def lse_norm(v):
    return v - lse(v)


def make_world(n=64, lo=200, hi=3200, seed=None):
    rng = RNG if seed is None else np.random.default_rng(seed)
    logpi_ref = lse_norm(rng.normal(0, 1.0, n))
    r = rng.uniform(0.0, 1.0, n)
    lens = rng.integers(lo, hi, n).astype(float)
    return logpi_ref, r, lens


def c_of_Z(Z, logpi_ref, r, lens, beta=BETA):
    """log normalizing constant c(Z) of the unnormalized construction; zero loss needs c(Z)=0."""
    return lse(logpi_ref + lens * (beta * r - Z))


def solve_Zstar(logpi_ref, r, lens, beta=BETA):
    """Root of c(Z)=0. c is monotonically decreasing in Z; bisection."""
    lo, hi = -50.0, 50.0
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if c_of_Z(mid, logpi_ref, r, lens, beta) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_fixed_point(Z, logpi_ref, r, lens, beta=BETA):
    return lse_norm(logpi_ref + lens * (beta * r - Z))


def Zt_paper(logpi, logpi_ref, r, beta=BETA):
    """Eq.4 as is: no length normalization."""
    return float(np.sum(np.exp(logpi) * (beta * r + logpi_ref - logpi)))


def Zt_fixed(logpi, logpi_ref, r, lens, beta=BETA):
    """Section 6 fix: the logprob term of Eq.4 is also normalized by |y|."""
    return float(np.sum(np.exp(logpi) * (beta * r + (logpi_ref - logpi) / lens)))


def residual(logpi, logpi_ref, r, Zt, lens, beta=BETA):
    return Zt + (logpi - logpi_ref) / lens - beta * r


def loss_at(logpi, logpi_ref, r, Zt, lens, beta=BETA, clip=True):
    g = residual(logpi, logpi_ref, r, Zt, lens, beta)
    gt = np.clip(g, -EPS_LOW, EPS_HIGH) if clip else g
    return float(np.sum(np.exp(logpi) * gt ** 2)), g


def shape(logpi):
    p = np.exp(logpi)
    H = float(-(p * np.log(np.maximum(p, 1e-300))).sum())
    return H, float(np.exp(H)), float(p.max())


def report_world(tag, logpi_ref, r, lens):
    print("=" * 78)
    print(f"{tag}   |Y|={len(r)}  beta={BETA:g}  |y| range [{lens.min():.0f}, {lens.max():.0f}] "
          f"median {np.median(lens):.0f}")
    print("=" * 78)

    Zs = solve_Zstar(logpi_ref, r, lens)
    lp = build_fixed_point(Zs, logpi_ref, r, lens)
    print(f"  solving c(Z)=0 gives Z* = {Zs:.6f}   residual c(Z*) = {c_of_Z(Zs, logpi_ref, r, lens):.2e}")

    zt_fix = Zt_fixed(lp, logpi_ref, r, lens)
    zt_pap = Zt_paper(lp, logpi_ref, r)
    print(f"\n  [existence] the Z_t each version of Eq.4 gives at this point, compared with Z*:")
    print(f"    section 6 fix (normalized)   Z_t = {zt_fix:14.6f}   |Z_t - Z*| = {abs(zt_fix - Zs):.3e}")
    print(f"    paper as is (unnormalized)   Z_t = {zt_pap:14.6f}   |Z_t - Z*| = {abs(zt_pap - Zs):.3e}")

    l_fix, _ = loss_at(lp, logpi_ref, r, zt_fix, lens, clip=False)
    l_pap, _ = loss_at(lp, logpi_ref, r, zt_pap, lens, clip=False)
    print(f"\n  [zero loss] substitute each self-consistent Z_t back into Eq.8 (no clip):")
    print(f"    section 6 fix   loss = {l_fix:.3e}   {'<- exact zero' if l_fix < 1e-16 else ''}")
    print(f"    paper as is     loss = {l_pap:.3e}")

    print(f"\n  [cost] what the fixed point looks like (compared with pi_ref and with the beta-tilt point of Prop.B.1):")
    lp_beta = lse_norm(logpi_ref + BETA * r)
    for nm, v in (("pi_ref", logpi_ref), ("Prop.B.1  pi_ref*exp(beta*r)", lp_beta),
                  ("fixed point of the fix", lp)):
        H, eff, mx = shape(v)
        print(f"    {nm:30s} H={H:7.4f}  effective support {eff:8.3f}/{len(r)}  max p = {mx:.6f}")


def main():
    print(__doc__)

    logpi_ref, r, lens = make_world()
    report_world("A. synthetic world (original setting of p7_fixedpoint.py, |y| 200-3200)", logpi_ref, r, lens)

    # redo with the measured magnitude of |y|: robospatial median 334 tokens, within-group relative range ~0.41
    lp_ref2, r2, _ = make_world(seed=7)
    lens2 = RNG.integers(250, 450, len(r2)).astype(float)
    report_world("B. measured magnitude (|y| ~ 250-450 tokens, robospatial median 334)", lp_ref2, r2, lens2)

    # homogeneous length sweep: see how the fixed point's collapse grows with L
    print("=" * 78)
    print("C. homogeneous length sweep: degree of fixed-point collapse vs L (with equal L everything is analytic)")
    print("=" * 78)
    lp_ref3, r3, _ = make_world(seed=11)
    print(f"    {'L':>8} {'Z*':>12} {'loss(fix)':>14} {'H':>9} {'eff. support':>10} {'max p':>10}")
    for Lc in (1.0, 2.0, 5.0, 10.0, 50.0, 334.0):
        lens3 = np.full(len(r3), Lc)
        Zs = solve_Zstar(lp_ref3, r3, lens3)
        lp3 = build_fixed_point(Zs, lp_ref3, r3, lens3)
        zt = Zt_fixed(lp3, lp_ref3, r3, lens3)
        l3, _ = loss_at(lp3, lp_ref3, r3, zt, lens3, clip=False)
        H, eff, mx = shape(lp3)
        print(f"    {Lc:8.1f} {Zs:12.5f} {l3:14.3e} {H:9.4f} {eff:10.3f} {mx:10.6f}")
    H0, e0, m0 = shape(lp_ref3)
    Hb, eb, mb = shape(lse_norm(lp_ref3 + BETA * r3))
    print(f"    {'pi_ref':>8} {'':12} {'':14} {H0:9.4f} {e0:10.3f} {m0:10.6f}")
    print(f"    {'beta-tilt':>8} {'':12} {'':14} {Hb:9.4f} {eb:10.3f} {mb:10.6f}   <- target of Prop.B.1")

    print("""
=== Reading ===
[existence] the fix **restores a self-consistent zero-loss fixed point**: Z_t (normalized Eq.4) and Z* are identical (numerically to machine precision),
         while the paper's original Eq.4 gives, at the same point, a Z_t that differs from Z* by a sizable amount — this is exactly the other side of T3(b)
         "when L != 1 the self-consistent zero does not exist". **The fix repairs it.**

[cost]   The cost is not in existence, it is in **the location of the fixed point**. The zero-loss condition itself (the length normalization of Eq.5) forces
         log pi* = log pi_ref + L*(beta*r - Z), i.e. the line in Remark B.4
         "inverse temperature L*beta". At L ~ 300 tokens exp(L*beta*r) is a hard argmax:
         the fixed point collapses to near a point mass, **distribution matching degenerates into reward maximization** — exactly what GFlowRL is meant to avoid.

         Note this **does not contradict** the claim retracted in section 2.3 of P7_STEP23_RESULTS: that one said
         that **under the paper as is** the minimizer lands at c ~ 1, not c = L — because there is no zero there at all,
         and the minimum is a compromise between two incompatible conditions, "self-consistency" and "zero residual". The fix makes the zero actually exist,
         so the L*beta reading of Remark B.4 **comes back under the fix**.

[so]     The section 6 fix **is not free**. It pushes "whole group of g flattened to the same value" to 0,
         at the cost of pushing the fixed point toward argmax. The two are two sides of the same length normalization.
         The truly consistent alternative is **normalize neither side** (back to the setting of Prop. B.1, tilt = beta,
         fixed-point entropy on the same order as pi_ref) — the cost is that long sequences get more weight in the loss,
         which is exactly the problem the paper introduced length normalization to solve.
""")


if __name__ == "__main__":
    main()
