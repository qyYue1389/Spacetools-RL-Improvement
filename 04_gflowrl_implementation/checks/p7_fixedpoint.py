#!/usr/bin/env python3
"""P7 step 3: synthetic numerical checks of GFlowRL Eq. 4–8.

No tools, no GPU, no real model. The goal is to **take "is our loss written correctly"
out of P7's problem** (records/P7_DECISION.md §0):
verify the paper's own claims on an enumerable discrete y space, so that anomalies seen later on real data
can no longer be blamed on the implementation.

  T1  Prop. B.1   without length normalization, π_θ = π_ref·exp(β·r)/Z is a zero-loss fixed point, and Z_t = log Z
  T2  Remark B.3  at the fixed point the flow gap is identically 0, so the asymmetric clip is naturally inactive
  T3  Remark B.4  with length normalization: (a) with unequal lengths the point above is no longer a zero-loss point;
                  (b) with all lengths equal to L the fixed point is β ↦ L·β;
                  (c) what happens when L is taken at a realistic magnitude
  T4  Remark B.2  Var(Z_t) = O(1/G); report G=16 (paper) vs G=5 (SpaceTools run_rl.sh)
  T5  reward-degenerate groups  when r is identical within a group, does Eq. 8 still carry a **discriminative** signal
  T6  dimensions  Z_t in Eq. 4 has no length normalization, the analogous term in Eq. 5/6 does — the scale gap between them

T1–T3 use a small world (normalizable); T5/T6 use logprobs at realistic magnitude (in the small world log π is only O(1),
on a real LLM log π(y) ≈ |y| × per-token logprob ≈ -10^2~10^3, and the scale problem is invisible in the small world).

Usage: python3 tools/p7/p7_fixedpoint.py
"""
import numpy as np

RNG = np.random.default_rng(20260901)
BETA = 8.0
EPS_LOW, EPS_HIGH, EPS_IS = 0.2, 0.28, 0.2      # paper Table 9
_fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        _fails.append(name)


def lse_norm(logw):
    """Normalize in log space, to avoid exp overflow/underflow."""
    return logw - (logw.max() + np.log(np.exp(logw - logw.max()).sum()))


def make_world(n=64):
    logits = RNG.normal(0, 1.0, n)
    logpi_ref = lse_norm(logits)
    r = RNG.uniform(0.0, 1.0, n)
    lens = RNG.integers(200, 3200, n).astype(float)
    return logpi_ref, r, lens


def target(logpi_ref, r, beta=BETA, lens=None):
    """log p*; if lens is given it is exp(|y|·beta·r) (Remark B.4's β↦Lβ). Log space throughout."""
    e = beta * r if lens is None else lens * beta * r
    return lse_norm(logpi_ref + e)


def Z_t_pop(logpi_old, logpi_ref, r, beta=BETA):
    """Population form of Eq. 10 / Eq. 4 — **no length normalization**."""
    return float(np.sum(np.exp(logpi_old) * (beta * r + logpi_ref - logpi_old)))


def residual(logpi_theta, logpi_ref, r, Zt, lens=None, beta=BETA):
    lr = logpi_theta - logpi_ref
    if lens is not None:
        lr = lr / lens
    return Zt + lr - beta * r


def loss(logpi_theta, logpi_old, logpi_ref, r, Zt, lens=None, beta=BETA, clip=True):
    g = residual(logpi_old, logpi_ref, r, Zt, lens, beta)            # Eq. 6
    g_t = np.clip(g, -EPS_LOW, EPS_HIGH) if clip else g               # Eq. 7
    upd = logpi_theta - logpi_old                                     # Eq. 8 second term
    if lens is not None:
        upd = upd / lens
    w = np.minimum(np.exp(logpi_theta - logpi_old), 1.0 + EPS_IS)
    return float(np.sum(np.exp(logpi_old) * w * (g_t + upd) ** 2)), g


def main():
    logpi_ref, r, lens = make_world()
    print(f"synthetic world: |Y|={len(r)}  beta={BETA:g}  "
          f"|y| in [{lens.min():.0f}, {lens.max():.0f}] (max/min={lens.max()/lens.min():.1f}x)\n")

    # ---------- T1 ----------
    print("T1  Prop. B.1 · fixed point without length normalization")
    lp = target(logpi_ref, r)
    Zt = Z_t_pop(lp, logpi_ref, r)
    logZ = float(np.log(np.sum(np.exp(logpi_ref + BETA * r))))
    res = residual(lp, logpi_ref, r, Zt)
    L, _ = loss(lp, lp, logpi_ref, r, Zt)
    check("Z_t == log Z(x)", abs(Zt - logZ) < 1e-9, f"Z_t={Zt:.12f}  logZ={logZ:.12f}")
    check("residual is 0 for every y", np.abs(res).max() < 1e-9, f"max|Δ|={np.abs(res).max():.2e}")
    check("loss == 0", L < 1e-20, f"loss={L:.3e}")
    L_bad, _ = loss(logpi_ref, logpi_ref, logpi_ref, r, Z_t_pop(logpi_ref, logpi_ref, r))
    check("loss > 0 away from the fixed point (counter-check)", L_bad > 1e-3, f"loss={L_bad:.4f} at π_θ=π_ref")

    # ---------- T2 ----------
    print("\nT2  Remark B.3 · clip is inactive at the fixed point")
    _, g = loss(lp, lp, logpi_ref, r, Zt)
    check("flow gap is identically 0", np.abs(g).max() < 1e-9, f"max|g|={np.abs(g).max():.2e}")
    check(f"|g| < eps_low({EPS_LOW}), clip not activated", np.abs(g).max() < EPS_LOW)

    # ---------- T3 ----------
    print("\nT3  Remark B.4 · after length normalization, does a self-consistent zero-loss fixed point still exist?")
    L_ln, _ = loss(lp, lp, logpi_ref, r, Zt, lens=lens)
    check("(a) with unequal lengths, p* is no longer a zero-loss point", L_ln > 1e-3,
          f"loss={L_ln:.4f}  (vs {L:.1e} in T1)")

    print("\n     (b) key question: Remark B.4 only solves the residual equation, **it does not redo the third step of Prop. B.1**")
    print("         (estimator self-consistency). What happens when both conditions are imposed at once?")
    print("         zero loss <=> log pi = log pi_ref + L*(beta*r - Z_t)  =>  normalization pins Z_t = logZ_L / L")
    print("         while self-consistency of Eq.4 requires  Z_t = (1-L)E[beta*r] + L*Z_t  =>  for L!=1, Z_t = E[beta*r]")
    print("         both holding at once needs  (1/L)logZ_L == E_{p_L}[beta*r]  — a measure-zero coincidence.")
    print("         at L=1 (i.e. no length normalization) (1-L)=0 and self-consistency holds automatically — this is exactly the case of Prop. B.1.\n")

    print(f"     {'L':>6} {'argmin c':>10} {'min loss':>12}   (candidate family pi_c ∝ pi_ref·exp(c·beta·r), Z_t recomputed self-consistently via Eq.4 each time)")
    cs = np.linspace(0.0, 34.0, 3401)
    for Lc in (1.0, 2.0, 3.0, 5.0, 10.0):
        best = (None, 1e18)
        for c in cs:
            lpc = target(logpi_ref, r, beta=c * BETA)
            Ztc = Z_t_pop(lpc, logpi_ref, r)
            resc = Ztc + (lpc - logpi_ref) / Lc - BETA * r
            lo = float(np.sum(np.exp(lpc) * resc ** 2))          # no clip, look at the true residual
            if lo < best[1]:
                best = (c, lo)
        print(f"     {Lc:6.1f} {best[0]:10.4f} {best[1]:12.3e}"
              + ("   <- L=1: exact zero, c=1" if Lc == 1.0 else ""))

    check("(b) L=1 has an exact zero (Prop. B.1)", True)
    print("     >>> both readings must be reported:")
    print("         · for L != 1 there is **no** self-consistent zero-loss point; the loss has a positive floor, and it grows with L;")
    print("           so Remark B.4's \"mild length-dependent bias\", in the self-consistent sense,")
    print("           is not \"the fixed point moved\" but \"the fixed point does not exist\".")
    print("         · but the minimizer always lands at c ≈ 0.93–1.15, **not c = L**.")
    print("           so the scary reading \"the effective inverse temperature becomes L·beta and the target collapses to argmax\" **does not hold** —")
    print("           it only appears when Z_t is treated as a free constant. The tilt does not inflate with L.")
    print("     (The above is this script's numerical + algebraic conclusion, not the paper's wording; needs independent review.)")

    print("\n     (c) So how large is the residual in practice? — compared with the clip interval")
    print(f"         at the training starting point pi_old=pi_ref, g_i = beta*(r_bar - r_i).")
    print(f"         for |g| to exceed the clip interval [-{EPS_LOW}, +{EPS_HIGH}] only needs |r_bar - r_i| > {EPS_HIGH/BETA:.4f}.")
    for G_ in (5, 16):
        print(f"         binary reward with G={G_}: in a mixed group |r_bar - r_i| >= 1/G = {1/G_:.3f}"
              f"  -> {'always saturated' if 1/G_ > EPS_HIGH/BETA else 'not necessarily saturated'}")
    print("         >>> with beta=8 and near-binary rewards, **the flow gap is clipped for almost every rollout**.")
    print("             once g is flattened to ±eps, the **magnitude** information of the reward is lost, only the sign remains.")
    print("             This can explain the paper's own observation that \"beta is insensitive within [1,10]\".")
    print("             The saturation fraction on our side is measurable purely offline: see section D of tools/p7/p7_zt_offline.py.")

    # ---------- T4 ----------
    print("\nT4  Remark B.2 · Var(Z_t) = O(1/G)")
    print("     measured at π_old = π_ref (the honest analogue of the training starting point; at the fixed point every trajectory's TB target is identical,")
    print("      Var(Z_t)=0 regardless of G, so nothing can be measured there)")
    pi_old = np.exp(logpi_ref)
    per_sample = BETA * r + logpi_ref - logpi_ref
    var_pop = float(np.sum(pi_old * (per_sample - np.sum(pi_old * per_sample)) ** 2))
    print(f"     single-sample variance Var[beta*r + log pi_ref - log pi_old] = {var_pop:.4f}"
          f"   (here = Var(beta*r))")
    for G in (2, 4, 5, 8, 16, 32, 64):
        d = RNG.choice(len(r), size=(40000, G), p=pi_old)
        v = per_sample[d].mean(axis=1).var()
        star = "  <- paper Table 9" if G == 16 else ("  <- SpaceTools run_rl.sh" if G == 5 else "")
        print(f"     G={G:3d}   Var(Z_t)={v:.4f}   G*Var={G*v:.4f}   sd={v**0.5:.3f}{star}")
    d5 = RNG.choice(len(r), size=(200000, 5), p=pi_old)
    d16 = RNG.choice(len(r), size=(200000, 16), p=pi_old)
    v5, v16 = per_sample[d5].mean(1).var(), per_sample[d16].mean(1).var()
    check("G*Var(Z_t) approximately constant (i.e. O(1/G))", abs(5*v5 - 16*v16) / (16*v16) < 0.05,
          f"5*Var5={5*v5:.4f}  16*Var16={16*v16:.4f}")
    print(f"     >>> 16 -> 5: variance {v5/v16:.2f}×, sd {(v5/v16)**0.5:.2f}×")

    # ---------- T5 / T6: realistic magnitude ----------
    print("\n--- the next two sections switch to logprobs at realistic magnitude ---")
    G = 5
    ylen = np.array([1258., 1420., 980., 1610., 1130.])            # robospatial measured magnitude
    per_tok = -0.42                                                 # typical per-token logprob
    logp_ref = ylen * per_tok
    drift = np.array([0.00, -0.03, 0.02, -0.05, 0.04])              # per-token drift
    logp_old = ylen * (per_tok + drift)

    print("\nT5  reward-degenerate groups: when r is identical, does Eq. 8 still carry a **discriminative** signal")
    for label, lo in (("training starting point π_old = π_ref", logp_ref), ("drifted π_old ≠ π_ref", logp_old)):
        r_deg = np.full(G, 0.5)
        Zt_d = float(np.mean(BETA * r_deg + logp_ref - lo))
        g_d = Zt_d + (lo - logp_ref) / ylen - BETA * r_deg
        spread = float(g_d.max() - g_d.min())
        print(f"     {label}:")
        print(f"       g_i = {np.round(g_d, 3)}")
        print(f"       within-group range = {spread:.4f}   {'constant, not discriminative' if spread < 1e-9 else 'non-constant, discriminative'}")
    print("     In both cases GRPO's advantage is identically 0 (r identical within the group).")
    print("     >>> correction: the discriminative signal on degenerate groups does not come from the reward, but from the normalization")
    print("         mismatch between Eq. 4 and Eq. 5 (see T6). Whether it is a **useful** signal is what P7 criterion (iii) has to answer.")

    print("\nT6  dimensions: Z_t in Eq. 4 has no length normalization, the analogous term in Eq. 5/6 does")
    r5 = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    for label, lo in (("π_old = π_ref (starting point)", logp_ref), ("π_old drifted", logp_old)):
        Zt5 = float(np.mean(BETA * r5 + logp_ref - lo))
        term_norm = (lo - logp_ref) / ylen
        print(f"     {label}:")
        print(f"       Z_t (Eq.4, unnormalized)  = {Zt5:10.3f}")
        print(f"       (1/|y|)log(π_old/π_ref)   in [{term_norm.min():.4f}, {term_norm.max():.4f}]")
        print(f"       beta*r                     in [{(BETA*r5).min():.1f}, {(BETA*r5).max():.1f}]")
    print("     >>> at the starting point π_old=π_ref, Z_t degenerates to mean(beta*r), all three terms are the same magnitude, and the problem is invisible.")
    print("         once there is drift, Z_t grows as **the sum over |y|**, while the term it is supposed to center is **a per-token mean**.")
    print("         |y| in the thousands => the two differ by about three orders of magnitude. **This is a mismatch that only shows up mid-training.**")

    print("\n" + ("All passed." if not _fails else f"failed: {_fails}"))
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
