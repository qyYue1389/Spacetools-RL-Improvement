#!/usr/bin/env python3
"""P7 判据 i-b(合成一半):同一个批内常数的三种取法,到底差在哪、什么时候才开始有差别。

`P7_DECISION.md` 判据 i-b。**零 GPU。**

--- 先把关系说清楚(这一节推翻了「三种估计法」这个笼统说法)---

TB 残差:  Δ_i = C + (1/|y_i|)·log(π/π_ref) − β·r_i

  GFlowRL (Eq. 4)  C = mean_i[ β·r_i + log π_ref − log π_old ],用 **π_old**,且 **sg[]**
  VarGrad          C* = 使 Σ Δ_i² 最小的常数,用 **π_θ**,且 **带梯度**
  DevGrad (f-TB)   C* = 使 (1/B) Σ L_f(Δ_i + C) 最小的常数,**f 不是平方时不再是均值**

**所以:平方损失 + π_θ = π_old 时,三者数值上完全相同。** 差别只有两处:
  (a) 常数里走不走梯度(GFlowRL 的 sg[] vs VarGrad)—— 只在 minibatch 内 π_θ 漂开后显形;
  (b) f 的选择(平方 -> 均值;绝对值 -> 中位;Huber -> 稳健常数)。

本脚本量的就是 (a) 与 (b),并回答一个更前置的问题:
**在 flow gap 几乎全被 clip 削平的区间里(步骤 2 实测 β=8 时非简并 rollout 100% 饱和),
常数的选择还有没有影响?**

奖励用 `p6/passk` 的**真实组**;`log π_ref − log π_old` 这一项**是合成的**
(前向 pass 之前拿不到),按每 token 漂移 × |y| 生成,量级可扫。
前向 pass 之后应当用真值重跑本脚本。

用法: python3 tools/p7/p7_estimators.py
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))
from parse_dump import TURN_SPLIT                      # noqa: E402

EPS_LOW, EPS_HIGH = 0.2, 0.28                          # 论文 Table 9
RNG = np.random.default_rng(20260901)


# ---------- 三种常数 ----------
def c_mean(T):
    """GFlowRL Eq. 4:算术平均(= 平方损失下的最优常数)。"""
    return T.mean(axis=-1)


def c_median(T):
    """DevGrad 的一个实例:f 使损失为绝对值时,最优常数是中位数。"""
    return np.median(T, axis=-1)


def c_huber(T, delta=EPS_HIGH):
    """DevGrad 的另一个实例:Huber 损失的最优常数(用 IRLS 解,delta 取 clip 的宽度)。"""
    c = np.median(T, axis=-1)
    for _ in range(60):
        rres = T - c[..., None]
        w = np.where(np.abs(rres) <= delta, 1.0, delta / np.maximum(np.abs(rres), 1e-12))
        c = (w * T).sum(-1) / np.maximum(w.sum(-1), 1e-12)
    return c


EST = {"mean(GFlowRL)": c_mean, "median(DevGrad|·|)": c_median, "huber(DevGrad)": c_huber}


# ---------- 真实数据 ----------
def load_groups(key):
    """返回 (奖励矩阵 [n,G], assistant 字符长度矩阵 [n,G])。长度是 token 的代理。"""
    path = os.path.join(REPO, "01_official_checkpoint_eval", "p6", "passk", key, "0.jsonl")
    g = {}
    for line in open(path):
        d = json.loads(line)
        g.setdefault(d["index"], []).append(d)
    R, L = [], []
    for v in g.values():
        R.append([x["score"] for x in v])
        L.append([_alen(x["output"]) for x in v])
    return np.array(R, float), np.array(L, float)


def _alen(output):
    """assistant 生成的字符数,切分沿用 parse_dump 的约定(见步骤 2)。"""
    chunks = TURN_SPLIT.split(output)
    lead = not output.lstrip().startswith("<|im_start|>")
    n = 0
    for i, ch in enumerate(chunks):
        if not ch.strip():
            continue
        if i == 0 and lead:
            role, body = "assistant", ch
        else:
            head, _, body = ch.partition("\n")
            role = head.strip().replace("<|im_end|>", "").strip()
        if role == "assistant":
            n += len(body.replace("<|im_end|>", ""))
    return max(n, 1)


def main():
    print("# P7 判据 i-b · 三种批内常数(合成漂移,真实奖励)\n")
    print("奖励:p6/passk 的真实 G=5 组   漂移项 log π_ref − log π_old:合成,量级可扫")
    print("长度 |y|:assistant 字符数(token 代理)\n")

    for key in ("robospatial", "boppose"):
        R, L = load_groups(key)
        n, G = R.shape
        print(f"{'='*78}\n## {key}   prompts={n}  G={G}\n")

        # ---------- (0) 起点:三者是否真的相同 ----------
        T0 = 8.0 * R                                   # drift=0 => T_i = beta*r_i
        cs = {k: f(T0) for k, f in EST.items()}
        d_med = np.abs(cs["mean(GFlowRL)"] - cs["median(DevGrad|·|)"]).mean()
        d_hub = np.abs(cs["mean(GFlowRL)"] - cs["huber(DevGrad)"]).mean()
        print(f"   (0) 起点(π_old=π_ref,drift=0,beta=8):三个常数的平均绝对差")
        print(f"       mean vs median {d_med:.4f}   mean vs huber {d_hub:.4f}")
        print(f"       -> 平方损失下 mean 与 median/huber **本来就不同**(f 不同),")
        print(f"          但 π_θ=π_old 使 GFlowRL 与 VarGrad 在数值上重合。\n")

        # ---------- (1) beta × 漂移 二维扫描 ----------
        print(f"   (1) 饱和率 与 符号不一致率:beta × 每 token 漂移 sd")
        print(f"       饱和 = |g| 越出 clip 区间的比例;符号不一致 = clip 之后 mean 与 huber")
        print(f"       给出的 g̃ 正负号不同的比例(**饱和之后只有符号还活着**)")
        hdr = "       {:>10}".format("drift sd") + "".join(f"{('b=%g'%b_):>18}" for b_ in (0.5, 1.0, 2.0, 8.0))
        print(hdr)
        for sd in (0.0, 0.0003, 0.001, 0.003, 0.01, 0.03):
            drift = RNG.normal(0, sd, size=R.shape) * L if sd > 0 else np.zeros_like(R)
            cells = []
            for beta in (0.5, 1.0, 2.0, 8.0):
                T = beta * R + drift
                g_m = c_mean(T)[:, None] - drift / L - beta * R
                g_h = c_huber(T)[:, None] - drift / L - beta * R
                sat = np.mean((g_m < -EPS_LOW) | (g_m > EPS_HIGH))
                gt_m = np.clip(g_m, -EPS_LOW, EPS_HIGH)
                gt_h = np.clip(g_h, -EPS_LOW, EPS_HIGH)
                dis = np.mean(np.sign(gt_m) != np.sign(gt_h))
                cells.append(f"{100*sat:7.1f}% /{100*dis:6.1f}%")
            print("       {:>10}".format(f"{sd:g}") + "".join(f"{c:>18}" for c in cells))
        print(f"       格式:饱和率 / 符号不一致率")
        print(f"       >>> 两条,都与写脚本前的猜测相反:")
        print(f"           · **漂移一旦非零,饱和率就与 beta 无关地冲到 ~100%** ——")
        print(f"             因为 C 里的漂移项未归一化、被减去的那一项归一化(§T6 的量纲失配),")
        print(f"             |y| 上千把 g 推得远超 clip 区间。beta 只在 drift=0 那一行说了算。")
        print(f"           · **饱和不等于常数的选择无关紧要。** clip 削掉的是幅度,活下来的是符号,")
        print(f"             而符号仍然取决于 C:mean 与 huber 在 ~1/5 的 rollout 上给出相反方向。\n")

        # ---------- (1b) 组内 g̃ 是否被削成同一个值 ----------
        print(f"   (1b) **更尖锐的一问**:组内 G 个 g̃ 是不是被削成了同一个值?")
        print(f"        g_i = C − d_i/|y_i| − beta*r_i,而 C 里的漂移项**未归一化**、")
        print(f"        被减的 d_i/|y_i| **已归一化** => 漂移在组内近似是一个**公共平移**。")
        print(f"        公共平移一旦超过 clip 半宽,整组被推到同一侧,**奖励的贡献被完全抹掉**。")
        print(f"        {'drift sd':>10} {'|C 的漂移贡献| 中位':>20} {'整组 g̃ 相同的比例':>20}")
        for sd in (0.0, 0.0003, 0.0005, 0.001, 0.003, 0.01):
            drift = RNG.normal(0, sd, size=R.shape) * L if sd > 0 else np.zeros_like(R)
            T = 8.0 * R + drift
            C = c_mean(T)[:, None]
            g = C - drift / L - 8.0 * R
            gt = np.clip(g, -EPS_LOW, EPS_HIGH)
            uniform = np.mean(np.ptp(gt, axis=1) < 1e-12)
            contrib = np.median(np.abs(drift.mean(axis=1)))
            print(f"        {sd:10g} {contrib:20.4f} {100*uniform:19.1f}%")
        thr = EPS_HIGH * np.sqrt(G) / float(np.median(L))
        print(f"        >>> 阈值可以直接算:C 的漂移贡献 sd ≈ 每token漂移 × |y| / sqrt(G),")
        print(f"            令它等于 eps_high={EPS_HIGH} 得每 token 漂移 ≈ {thr:.1e} nat。")
        print(f"            **clip 半宽 {EPS_HIGH} 对上 |y|≈{np.median(L):.0f} 的未归一化求和,容差小到这个地步。**\n")

        # ---------- (2) 估计量自身的抽样方差 ----------
        print(f"   (2) 估计量的抽样方差(固定真值,重复抽 G 个样本 20000 次)")
        print(f"       {'G':>4} {'噪声分布':>10} {'Var(mean)':>11} {'Var(median)':>12} {'Var(huber)':>11}  {'相对 mean':>18}")
        for G_ in (5, 16):
            for label, gen in (("高斯", lambda s_: RNG.normal(0, 1.0, s_)),
                               ("重尾 t3", lambda s_: RNG.standard_t(3, s_)),
                               ("二值+噪声", lambda s_: 8.0 * RNG.integers(0, 2, s_) + RNG.normal(0, 1.0, s_))):
                T = gen((20000, G_))
                v = {k: float(np.var(f(T))) for k, f in EST.items()}
                rel = (f"median {v['median(DevGrad|·|)']/v['mean(GFlowRL)']:.2f}x  "
                       f"huber {v['huber(DevGrad)']/v['mean(GFlowRL)']:.2f}x")
                print(f"       {G_:4d} {label:>10} {v['mean(GFlowRL)']:11.4f}"
                      f" {v['median(DevGrad|·|)']:12.4f} {v['huber(DevGrad)']:11.4f}  {rel:>18}")
        print(f"       >>> 高斯下 mean 最优(median 约 1.2-1.5x);**重尾 t3 下 median/huber 明显更稳**;")
        print(f"           而**二值奖励 + 噪声**这种双峰分布上 median 反而更差 —— 它去追某一个峰。")
        print(f"           我们的漂移项到底是哪一种,**只有前向 pass 能回答**。这就是判据 i-b 的落点。\n")

        # ---------- (3) sg 与否:minibatch 内漂开之后常数会动多少 ----------
        print(f"   (3) sg[] 值不值(GFlowRL 冻结常数,VarGrad 不冻结)")
        print(f"       {'minibatch 内每 token 漂移':>26} {'常数移动(均值)':>16} {'相对 beta*r 量程':>16}")
        T_ref = 8.0 * R + drift
        C_ref = c_mean(T_ref)
        for extra in (0.001, 0.003, 0.01, 0.03):
            d2 = drift + RNG.normal(0, extra, size=R.shape) * L
            C_new = c_mean(8.0 * R + d2)
            move = np.abs(C_new - C_ref).mean()
            print(f"       {extra:26.3f} {move:16.4f} {move/8.0:16.4f}")
        print(f"       >>> 常数的移动量与 |y| 成正比(§T6 的量纲失配),")
        print(f"           所以 sg[] 冻结的东西在 minibatch 内可以漂得比 beta*r 的量程还大。\n")


        # ---------- (4) 候选修法:把 Eq. 4 也做长度归一化 ----------
        print(f"   (4) 候选修法:如果 Z_t(Eq. 4)里的 logprob 项**也**按 |y| 归一化会怎样?")
        print(f"       C_norm = mean_i[ beta*r_i + (log pi_ref − log pi_old)_i / |y_i| ]")
        print(f"       这样 C 与被它中心化的那一项同量纲,公共平移应当消失。")
        print(f"       {'drift sd':>10} | {'原样 Eq.4':>22} | {'长度归一化后':>22}")
        print(f"       {'':>10} | {'饱和率':>10}{'整组同值':>12} | {'饱和率':>10}{'整组同值':>12}")
        for sd in (0.0003, 0.001, 0.003, 0.01, 0.03):
            drift = RNG.normal(0, sd, size=R.shape) * L
            out = []
            for norm in (False, True):
                T = 8.0 * R + (drift / L if norm else drift)
                C = c_mean(T)[:, None]
                g = C - drift / L - 8.0 * R
                gt = np.clip(g, -EPS_LOW, EPS_HIGH)
                sat = np.mean((g < -EPS_LOW) | (g > EPS_HIGH))
                uni = np.mean(np.ptp(gt, axis=1) < 1e-12)
                out.append((sat, uni))
            print(f"       {sd:10g} | {100*out[0][0]:9.1f}%{100*out[0][1]:11.1f}%"
                  f" | {100*out[1][0]:9.1f}%{100*out[1][1]:11.1f}%")
        print(f"       >>> 归一化之后,饱和率与整组同值率**回到 drift=0 的水平并不再随漂移增长**。")
        print(f"           也就是说 §1b 那个「奖励贡献被抹掉」的机制,**来源是 Eq.4 与 Eq.5 的")
        print(f"           归一化不一致本身,而不是 GFlowRL 的批内估计思路**。")
        print(f"           ⚠ 但这个改法会**改变不动点**(Prop. B.1 是在无归一化下证的),")
        print(f"              代价未知 —— 这是一条待检验的候选,不是结论。\n")


if __name__ == "__main__":
    main()
