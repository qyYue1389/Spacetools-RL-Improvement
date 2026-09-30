#!/usr/bin/env python3
# =============================================================================
# smoke_check.py —— SFT ckpt 的两个便宜检查。不需要起工具栈。
#
#   ① 格式冒烟:ckpt 生成的东西是否长成训练数据里 assistant 的样子
#      (合法的 tool call / <answer> 标签 / 工具名在这 11 个 schema 里)
#   ② 对照 base:同样的 prompt 喂 Qwen2.5-VL-3B-Instruct
#
# ⚠️ 这**不是**能力度量。样本取自训练集,只能判断"SFT 有没有生效",
#    不能判断"效果好不好"。真实分数只有 SpaceTools-RL 的 run_eval.sh 能给。
#
# 它抓的是这一类静默失败:loss 曲线完全正常,但模型根本没学会发工具调用。
#
#   python smoke_check.py <ckpt目录> <train.json> [--n 12] [--base <模型>] [--no-base]
# =============================================================================
import argparse, json, os, random, re, sys

MARKERS = {
    "<think> 标签":   re.compile(r"<think>"),
    "</think> 闭合":  re.compile(r"</think>"),
    "tool_call 标签": re.compile(r"<tool_call>"),
    '"name" 字段':   re.compile(r'"name"\s*:'),
    "<answer> 标签":  re.compile(r"<answer>"),
}

# 与 base 对照时用哪个标记最有判别力。<think> 在第一个 assistant 轮里真值 100%,
# 而 base(未见过这个格式)几乎不会自发产出 —— 差距是「近 0% vs 近 100%」。
# 用「JSON 合法」做对照则很脆:base 是 instruct 模型且 system prompt 里给全了
# 11 个 schema,它本来就可能产出结构合法的 JSON,差值容易落在阈值附近而误判。
DISCRIMINATIVE = "<think> 标签"


def census(texts, label):
    """先看真值长什么样 —— 不要硬编码格式,从数据里读出来。

    ⚠️ 传进来的必须是**第一个 assistant 轮**,因为 smoke_check 只生成这一轮。
    统计所有轮会得到完全不同的分布(实测:所有轮 tool_call 63.3% / answer 36.7%,
    第一轮 tool_call 99.8% / answer 0.2%),据此定阈值会把 75% 的坏 ckpt 判成通过。
    """
    n = len(texts)
    if n == 0:
        print(f"\n  ✗ {label}: 一条都没取到 —— conversations 的 schema 可能变了")
        return [], {}
    print(f"\n  {label}({n} 条):")
    present, rates = [], {}
    for name, rx in MARKERS.items():
        c = sum(1 for t in texts if rx.search(t))
        rates[name] = c / n
        print(f"    {name:16s} {c:4d} / {n}  ({c/n*100:5.1f}%)")
        if c / n > 0.20:
            present.append(name)
    return present, rates


def _objects(text):
    """按花括号配对(且不被字符串里的括号骗到)扫出所有顶层 JSON 对象候选。

    不能用正则 —— `{"name": "t", "arguments": {}}` 里非贪婪的 .*?\\} 会停在内层
    的 } 上,切出不平衡的片段,把合法调用误判成非法。
    """
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        depth, in_str, esc = 0, False, False
        for j in range(i, len(text)):
            c = text[j]
            if in_str:
                if esc:            esc = False
                elif c == "\\":    esc = True
                elif c == '"':     in_str = False
                continue
            if c == '"':   in_str = True
            elif c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    yield text[i:j + 1]
                    break


def check(text, allowed_tools):
    """对单条生成做和真值同样的标记检查。"""
    r = {name: bool(rx.search(text)) for name, rx in MARKERS.items()}
    r["JSON 合法"] = False
    r["工具名合法"] = False
    m = re.search(r"<tool_call>\s*(\{.*)</tool_call>", text, re.S)
    cands = list(_objects(m.group(1))) if m else list(_objects(text))
    for c in cands:
        try:
            obj = json.loads(c)
        except Exception:
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        r["JSON 合法"] = True
        nm = obj.get("name")
        r["工具名合法"] = nm in allowed_tools if allowed_tools else bool(nm)
        r["_tool"] = nm
        break
    return r


def resolve_model(path):
    """把 Hub id 解析成本地 snapshot —— 本机匿名 IP 已被 HF 限速,走 Hub 会失败。

    调用方(FINALIZE.sh 第 7 节)会显式传 Hub id,所以解析必须在这里做,
    不能只改 argparse 的默认值。
    """
    import glob
    if os.path.isdir(path):
        return path
    # ⚠️ 不能用 os.environ.setdefault("HF_HOME", ...) —— Vast 实例已经把 HF_HOME
    # 设成 /workspace/.hf_home,setdefault 不会覆盖它,于是 glob 到空目录、
    # 回落到 Hub 重下 7.1 GB。必须把训练实际用的 /workspace/hf 也纳入候选。
    roots = [r for r in (os.environ.get("HF_HOME"), "/workspace/hf",
                         os.path.expanduser("~/.cache/huggingface")) if r]
    sub = "models--" + path.replace("/", "--")
    for root in roots:
        cands = sorted(glob.glob(os.path.join(root, "hub", sub, "snapshots", "*")))
        cands = [c for c in cands if glob.glob(os.path.join(c, "*.safetensors"))]
        if cands:
            print(f"  (用本地 snapshot 替代 Hub: {path} → {cands[-1]})")
            return cands[-1]
    return path


def build_msgs(item, sysprompt):
    """取第一个 user 轮,连同它的图。"""
    user = None
    for c in item.get("conversations", []):
        if c.get("role") == "user":
            user = c.get("content", "")
            break
    if user is None:
        return None, None
    txt = re.sub(r"<image>", "", user).strip()
    imgs = item.get("images") or []
    content = ([{"type": "image"}] if imgs else []) + [{"type": "text", "text": txt}]
    msgs = [{"role": "system", "content": [{"type": "text", "text": sysprompt}]},
            {"role": "user", "content": content}]
    return msgs, (imgs[0] if imgs else None)


def load_image(path, max_pixels=262144):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if w * h > max_pixels:                       # 与训练的 image_max_pixels 对齐
        s = (max_pixels / (w * h)) ** 0.5
        im = im.resize((max(28, int(w * s)), max(28, int(h * s))))
    return im


def run_model(model_path, samples, sysprompt, max_new_tokens=256):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print(f"\n  加载 {model_path} ...", flush=True)
    model_path = resolve_model(model_path)
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True)
    model.eval()
    outs = []
    for i, item in enumerate(samples):
        msgs, imgpath = build_msgs(item, sysprompt)
        if msgs is None:
            outs.append(""); continue
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        images = [load_image(imgpath)] if imgpath and os.path.exists(imgpath) else None
        inputs = proc(text=[text], images=images, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None, top_k=None)
        gen = gen[0][inputs["input_ids"].shape[1]:]
        outs.append(proc.decode(gen, skip_special_tokens=True))
        print(f"    [{i+1}/{len(samples)}]", end="\r", flush=True)
    print(" " * 30, end="\r")
    del model
    torch.cuda.empty_cache()
    return outs


def summarize(outs, allowed, label):
    res = [check(o, allowed) for o in outs]
    n = len(res)
    print(f"\n  === {label} ===")
    keys = list(MARKERS) + ["JSON 合法", "工具名合法"]
    rates = {}
    for k in keys:
        c = sum(1 for r in res if r.get(k))
        rates[k] = c / n if n else 0
        print(f"    {k:16s} {c:3d} / {n}  ({rates[k]*100:5.1f}%)")
    tools = [r.get("_tool") for r in res if r.get("_tool")]
    if tools:
        from collections import Counter
        print(f"    调用到的工具: {dict(Counter(tools))}")
    empty = sum(1 for o in outs if not o.strip())
    if empty:
        print(f"    ⚠ {empty} 条生成为空")
    return rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("train_json")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # ⚠️ 训练期间不能跑:本脚本要在 cuda:0 上加载 ckpt(3B bf16 ≈ 6.2 GiB)再加载 base,
    # 而训练每卡已占 35–36 / 48 GiB。撞 OOM 会杀掉训练,而 save_only_model=true
    # 意味着不能续训 —— 只能从头再来。
    pidfile = os.environ.get("TRAIN_PIDFILE", "/workspace/experiments/train.pid")
    if os.path.exists(pidfile) and not os.environ.get("ALLOW_DURING_TRAINING"):
        try:
            pid = int(open(pidfile).read().strip())
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            pass
        else:
            print(f"✗ 训练仍在运行(pid={pid},{pidfile})。本脚本会占显存,可能把训练撞 OOM。")
            print("  等训练结束再跑。确实要现在跑就设 ALLOW_DURING_TRAINING=1。")
            sys.exit(2)

    data = json.load(open(a.train_json))
    sysprompt = data[0].get("system", "")
    allowed = set(re.findall(r'"name"\s*:\s*"([\w.]+)"', sysprompt))
    print(f"数据 {len(data)} 条 · system prompt {len(sysprompt)} 字符 · "
          f"工具 schema {len(allowed)} 个")
    if allowed:
        print(f"  {sorted(allowed)}")

    # 先看真值:期望的格式从数据里读,不硬编码。
    # 只取**第一个** assistant 轮 —— build_msgs 只喂第一个 user 轮,
    # 模型生成的就是这一轮,拿多轮统计定阈值会系统性偏低。
    gts = []
    for it in data:
        for c in it.get("conversations", []):
            if c.get("role") == "assistant":
                gts.append(c.get("content", ""))
                break
    expected, gt = census(gts, "训练数据里【第一个 assistant 轮】的标记分布")
    if not gt:
        sys.exit(1)
    print(f"  → 期望 ckpt 的生成里出现: {expected}")

    samples = random.Random(0).sample(data, min(a.n, len(data)))
    print(f"\n抽 {len(samples)} 条(seed=0,greedy 解码)")

    ck = run_model(a.ckpt, samples, sysprompt)
    rc = summarize(ck, allowed, "SFT ckpt")

    rb = None
    if not a.no_base:
        try:
            bs = run_model(a.base, samples, sysprompt)
            rb = summarize(bs, allowed, f"base ({a.base})")
        except Exception as e:
            print(f"\n  ⚠ base 对照跑不了({type(e).__name__}: {str(e)[:80]})—— 跳过")

    print("\n" + "=" * 60)
    # 阈值不写死,从真值率推导:样本抽自训练集(模型见过),所以合格线应贴近真值。
    key = "JSON 合法" if "tool_call 标签" in expected or '"name" 字段' in expected else "<answer> 标签"
    ref = gt.get("tool_call 标签", 1.0) if key == "JSON 合法" else gt.get("<answer> 标签", 1.0)
    pass_at, warn_at = 0.90 * ref, 0.50 * ref
    verdict = 0
    print(f"真值率 {ref*100:.1f}% → 合格线 {pass_at*100:.0f}% · 警戒线 {warn_at*100:.0f}%")
    if rc[key] >= pass_at:
        print(f"✓ ckpt 在 {rc[key]*100:.0f}% 的样本上产出了结构合法的工具调用")
    elif rc[key] >= warn_at:
        print(f"⚠ 只有 {rc[key]*100:.0f}%(真值 {ref*100:.0f}%)—— 偏低。"
              f"看下面的原始输出判断是模型没学会还是解析太严")
        verdict = 1
    else:
        print(f"✗ 只有 {rc[key]*100:.0f}%(真值 {ref*100:.0f}%)—— SFT 很可能没生效")
        verdict = 1
    if rb is not None:
        # 用判别力最强的格式标记做对照,而不是「JSON 合法」(base 本来就可能会)
        dk = DISCRIMINATIVE if DISCRIMINATIVE in rc and DISCRIMINATIVE in rb else key
        d = rc[dk] - rb[dk]
        print(f"  对照项「{dk}」: ckpt {rc[dk]*100:.0f}% vs base {rb[dk]*100:.0f}%,差 {d*100:+.0f} 个百分点")
        print(f"  (结构项「{key}」: ckpt {rc[key]*100:.0f}% vs base {rb[key]*100:.0f}%)")
        if d < 0.2:
            print("  ✗ 与 base 拉不开差距 —— 这正是「loss 正常但没学到东西」的样子")
            verdict = 1
        else:
            print("  ✓ 相对 base 有明显差距,SFT 确实改变了行为")

    print("\n--- 前 2 条原始输出(人工看一眼)---")
    for i, o in enumerate(ck[:2]):
        print(f"\n[{i+1}] {o[:600]}")

    if a.out:
        json.dump({"ckpt_rates": rc, "base_rates": rb, "expected_markers": expected,
                   "n": len(samples), "generations": ck},
                  open(a.out, "w"), ensure_ascii=False, indent=2)
        print(f"\n明细写入 {a.out}")

    print("\n提醒:这只判断「SFT 有没有生效」。真实分数要跑 SpaceTools-RL 的 run_eval.sh。")
    sys.exit(verdict)


if __name__ == "__main__":
    main()
