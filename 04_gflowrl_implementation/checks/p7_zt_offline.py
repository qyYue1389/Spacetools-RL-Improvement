#!/usr/bin/env python3
"""P7 步骤 2(离线部分):在 p6/passk 的 G=5 组上拆 Z_t 能离线拆的两项。

Z_t(x) = (1/G) Σ_i ( β·r(x,y_i) + log π_ref(y_i|x) − log π_old(y_i|x) )      [Eq. 4]

三项里 β·r 完全离线;两个 logprob 项需要一次前向 pass(不接工具、不起 sglang 池),
本脚本不碰。但**长度**是离线可得的,而按 Remark B.4,长度决定了不动点的有效逆温度:

    π_θ*(y|x) = π_ref(y|x) · exp( |y| · ( β·r − Z_t(x) ) )        [Remark B.4]

所以本脚本量三件事:
  A. β·r 的组内散布(含奖励简并组的比例)
  B. |y| 的分解:assistant 生成 vs <tool_response>。
     记录决定 §5.1 第 2 条:|y| 必须是策略生成的 token 数,不能是原始 response 长度。
  D. flow gap 的 clip 饱和比例(训练起点 π_old=π_ref 时 g_i = β·(r̄ − r_i),纯离线可算)
  C. 由 B 推出的有效逆温度 |y|·β 的散布 —— 即 Remark B.4 「lengths vary only
     modestly within rollout groups」这个前提在我们数据上被违反了多少。

⚠ 单位是**字符**,不是 token(本机没有 tokenizer,models/ 在 GPU 机器上)。
   字符是 token 的代理:比值与散布的量级可信,绝对值不可信。
   精确 token 数在前向 pass 那一步免费得到,届时应重跑本脚本的 B/C 两节。

切分沿用 tools/parse_dump.py 的 TURN_SPLIT 与 leading_is_assistant 约定 ——
**不自己重写切分**(P3 踩过:自建切分器静默丢掉 assistant 第一轮,
depth_estimator 从深度 benchmark 的工具直方图里整个消失)。
脚本自带一道回归校验:轮数必须与 verl 的 num_turns 对得上。

用法:  python3 tools/p7/p7_zt_offline.py [--beta 8]
"""
import argparse, json, os, statistics as st, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "tools"))
from parse_dump import TURN_SPLIT, parse_transcript   # noqa: E402

BENCHES = ["robospatial", "blinkdepth", "boppose"]


def split_lengths(output: str):
    """按 parse_dump 的约定切分,只累计字符数。

    返回 (assistant_chars, tool_chars, n_assistant_turns, n_user_turns)。
    """
    chunks = TURN_SPLIT.split(output)
    leading_is_assistant = not output.lstrip().startswith("<|im_start|>")
    a_chars = t_chars = 0
    n_a = n_u = 0
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        if idx == 0 and leading_is_assistant:
            role, body = "assistant", chunk
        else:
            head, _, body = chunk.partition("\n")
            role = head.strip().replace("<|im_end|>", "").strip()
        body = body.replace("<|im_end|>", "")
        if role == "assistant":
            a_chars += len(body); n_a += 1
        elif role in ("user", "tool"):
            t_chars += len(body); n_u += 1
    return a_chars, t_chars, n_a, n_u


def rel_range(xs):
    m = st.mean(xs)
    return (max(xs) - min(xs)) / m if m else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--root", default=os.path.join(REPO, "01_official_checkpoint_eval", "p6", "passk"))
    a = ap.parse_args()
    beta = a.beta

    print(f"# P7 步骤 2 · Z_t 的离线两项   beta={beta:g}   长度单位=字符(token 的代理)\n")

    for key in BENCHES:
        path = os.path.join(a.root, key, "0.jsonl")
        groups = defaultdict(list)
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                groups[d["index"]].append(d)

        sizes = {len(v) for v in groups.values()}
        # ---- 回归校验:切分出来的轮数必须与 verl 的 num_turns 自洽 ----
        bad = 0
        for v in groups.values():
            for d in v:
                _, _, n_a, n_u = split_lengths(d["output"])
                if n_a + n_u + 1 != d["num_turns"]:
                    bad += 1
        tot = sum(len(v) for v in groups.values())

        # ---- A. beta*r ----
        degen = 0
        br_sd_in_group, br_range = [], []
        # ---- B/C. 长度 ----
        a_all, raw_all, ratio_all = [], [], []
        a_relrange, raw_relrange = [], []
        for v in groups.values():
            r = [x["score"] for x in v]
            if max(r) - min(r) < 1e-9:
                degen += 1
            br = [beta * x for x in r]
            br_sd_in_group.append(st.pstdev(br))
            br_range.append(max(br) - min(br))

            ac, rawc = [], []
            for d in v:
                ach, tch, _, _ = split_lengths(d["output"])
                ac.append(max(ach, 1))
                rawc.append(max(ach + tch, 1))
            a_all += ac; raw_all += rawc
            ratio_all += [rr / aa for aa, rr in zip(ac, rawc)]
            a_relrange.append(rel_range(ac))
            raw_relrange.append(rel_range(rawc))

        n = len(groups)
        print(f"## {key}   prompts={n}  group sizes={sorted(sizes)}  rollouts={tot}")
        print(f"   回归校验(切分轮数 vs verl num_turns): 不符 {bad}/{tot}"
              f"   {'OK' if bad == 0 else '<<< 不通过,以下数字不可用'}")
        print(f"   A. beta*r 组内标准差   均值 {st.mean(br_sd_in_group):.3f}"
              f"   中位 {st.median(br_sd_in_group):.3f}"
              f"   极差均值 {st.mean(br_range):.3f}")
        print(f"      奖励简并组(G 内 r 全同)  {degen}/{n} = {100*degen/n:.1f}%")
        print(f"   B. |y| assistant 生成      均值 {st.mean(a_all):8.0f}"
              f"   中位 {st.median(a_all):8.0f}   min {min(a_all)}  max {max(a_all)}")
        print(f"      |y| 原始 response       均值 {st.mean(raw_all):8.0f}"
              f"   中位 {st.median(raw_all):8.0f}   min {min(raw_all)}  max {max(raw_all)}")
        print(f"      原始/assistant 倍数     中位 {st.median(ratio_all):.2f}×"
              f"   均值 {st.mean(ratio_all):.2f}×   max {max(ratio_all):.2f}×")
        print(f"   C. 组内 |y| 相对极差 (max-min)/mean")
        print(f"      用 assistant 长度        均值 {st.mean(a_relrange):.2f}"
              f"   中位 {st.median(a_relrange):.2f}   max {max(a_relrange):.2f}")
        print(f"      用原始 response 长度      均值 {st.mean(raw_relrange):.2f}"
              f"   中位 {st.median(raw_relrange):.2f}   max {max(raw_relrange):.2f}")
        print(f"      跨 prompt 的 |y| 极差比  max/min = {max(a_all)/min(a_all):.1f}×")

        # ---- D. clip 饱和 ----
        EPS_LOW, EPS_HIGH = 0.2, 0.28          # 论文 Table 9
        sat = tot_r = sat_nd = tot_nd = 0
        for v in groups.values():
            rr = [x["score"] for x in v]
            rbar = st.mean(rr)
            deg = (max(rr) - min(rr)) < 1e-9
            for x in rr:
                g = beta * (rbar - x)
                hit = (g < -EPS_LOW) or (g > EPS_HIGH)
                sat += hit; tot_r += 1
                if not deg:
                    sat_nd += hit; tot_nd += 1
        print(f"   D. flow gap 在起点被 clip 削掉的比例(g_i = beta*(r_bar - r_i))")
        print(f"      全部 rollout      {sat}/{tot_r} = {100*sat/tot_r:.1f}%")
        if tot_nd:
            print(f"      仅非简并组        {sat_nd}/{tot_nd} = {100*sat_nd/tot_nd:.1f}%"
                  f"   <- 简并组的 g 恒为 0,本来就不该算进来")
        print()


if __name__ == "__main__":
    main()
