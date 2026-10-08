#!/usr/bin/env python3
# =============================================================================
# smoke_check.py — two cheap checks on an SFT ckpt. No need to bring up the tool stack.
#
#   ① format smoke test: does what the ckpt generates look like the assistant turns in the training data
#      (valid tool call / <answer> tag / tool name among these 11 schemas)
#   ② compare against base: feed the same prompts to Qwen2.5-VL-3B-Instruct
#
# ⚠️ This is **not** a capability measure. Samples come from the training set, so it can only tell "whether SFT took effect",
#    not "whether the result is good". Only SpaceTools-RL's run_eval.sh can give the real score.
#
# What it catches is this class of silent failure: the loss curve is perfectly normal, but the model never learned to emit tool calls.
#
#   python smoke_check.py <ckpt dir> <train.json> [--n 12] [--base <model>] [--no-base]
# =============================================================================
import argparse, json, os, random, re, sys

MARKERS = {
    "<think> tag":   re.compile(r"<think>"),
    "</think> closed":  re.compile(r"</think>"),
    "tool_call tag": re.compile(r"<tool_call>"),
    '"name" field':   re.compile(r'"name"\s*:'),
    "<answer> tag":  re.compile(r"<answer>"),
}

# Which marker discriminates best when comparing against base. <think> is 100% in the ground truth of the first assistant turn,
# while base (never saw this format) almost never produces it spontaneously — the gap is "near 0% vs near 100%".
# Using "JSON valid" for the comparison is fragile: base is an instruct model and the system prompt gives all
# 11 schemas, so it may well produce structurally valid JSON anyway; the difference easily lands near the threshold and gets misjudged.
DISCRIMINATIVE = "<think> tag"


def census(texts, label):
    """First look at what the ground truth looks like — do not hard-code the format, read it from the data.

    ⚠️ What is passed in must be the **first assistant turn**, because smoke_check only generates that turn.
    Counting all turns gives a completely different distribution (measured: all turns tool_call 63.3% / answer 36.7%,
    first turn tool_call 99.8% / answer 0.2%); setting the threshold from that would pass 75% of bad ckpts.
    """
    n = len(texts)
    if n == 0:
        print(f"\n  ✗ {label}: got nothing at all — the conversations schema may have changed")
        return [], {}
    print(f"\n  {label} ({n} entries):")
    present, rates = [], {}
    for name, rx in MARKERS.items():
        c = sum(1 for t in texts if rx.search(t))
        rates[name] = c / n
        print(f"    {name:16s} {c:4d} / {n}  ({c/n*100:5.1f}%)")
        if c / n > 0.20:
            present.append(name)
    return present, rates


def _objects(text):
    """Scan out all top-level JSON object candidates by matching braces (without being fooled by braces inside strings).

    Cannot use a regex — on `{"name": "t", "arguments": {}}` a non-greedy .*?\\} stops at the inner
    }, cuts out an unbalanced fragment, and misjudges a valid call as invalid.
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
    """Run the same marker checks on a single generation as on the ground truth."""
    r = {name: bool(rx.search(text)) for name, rx in MARKERS.items()}
    r["JSON valid"] = False
    r["tool name valid"] = False
    m = re.search(r"<tool_call>\s*(\{.*)</tool_call>", text, re.S)
    cands = list(_objects(m.group(1))) if m else list(_objects(text))
    for c in cands:
        try:
            obj = json.loads(c)
        except Exception:
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        r["JSON valid"] = True
        nm = obj.get("name")
        r["tool name valid"] = nm in allowed_tools if allowed_tools else bool(nm)
        r["_tool"] = nm
        break
    return r


def resolve_model(path):
    """Resolve a Hub id to a local snapshot — this machine's anonymous IP is rate-limited by HF, going through the Hub will fail.

    The caller (FINALIZE.sh section 7) explicitly passes a Hub id, so the resolution must happen here,
    not just by changing the argparse default.
    """
    import glob
    if os.path.isdir(path):
        return path
    # ⚠️ Cannot use os.environ.setdefault("HF_HOME", ...) — the Vast instance already sets HF_HOME
    # to /workspace/.hf_home, setdefault does not override it, so the glob hits an empty directory and
    # falls back to re-downloading 7.1 GB from the Hub. The /workspace/hf actually used by training must be among the candidates too.
    roots = [r for r in (os.environ.get("HF_HOME"), "/workspace/hf",
                         os.path.expanduser("~/.cache/huggingface")) if r]
    sub = "models--" + path.replace("/", "--")
    for root in roots:
        cands = sorted(glob.glob(os.path.join(root, "hub", sub, "snapshots", "*")))
        cands = [c for c in cands if glob.glob(os.path.join(c, "*.safetensors"))]
        if cands:
            print(f"  (using local snapshot instead of Hub: {path} → {cands[-1]})")
            return cands[-1]
    return path


def build_msgs(item, sysprompt):
    """Take the first user turn, together with its image."""
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
    if w * h > max_pixels:                       # match the training image_max_pixels
        s = (max_pixels / (w * h)) ** 0.5
        im = im.resize((max(28, int(w * s)), max(28, int(h * s))))
    return im


def run_model(model_path, samples, sysprompt, max_new_tokens=256):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print(f"\n  loading {model_path} ...", flush=True)
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
    keys = list(MARKERS) + ["JSON valid", "tool name valid"]
    rates = {}
    for k in keys:
        c = sum(1 for r in res if r.get(k))
        rates[k] = c / n if n else 0
        print(f"    {k:16s} {c:3d} / {n}  ({rates[k]*100:5.1f}%)")
    tools = [r.get("_tool") for r in res if r.get("_tool")]
    if tools:
        from collections import Counter
        print(f"    tools called: {dict(Counter(tools))}")
    empty = sum(1 for o in outs if not o.strip())
    if empty:
        print(f"    ⚠ {empty} generations are empty")
    return rates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("train_json")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--no-base", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # ⚠️ Cannot run during training: this script loads the ckpt on cuda:0 (3B bf16 ≈ 6.2 GiB) and then base,
    # while training already uses 35–36 / 48 GiB per GPU. Hitting OOM kills training, and save_only_model=true
    # means training cannot be resumed — it can only start over.
    pidfile = os.environ.get("TRAIN_PIDFILE", "/workspace/experiments/train.pid")
    if os.path.exists(pidfile) and not os.environ.get("ALLOW_DURING_TRAINING"):
        try:
            pid = int(open(pidfile).read().strip())
            os.kill(pid, 0)
        except (ValueError, ProcessLookupError, PermissionError, OSError):
            pass
        else:
            print(f"✗ training is still running (pid={pid}, {pidfile}). This script uses GPU memory and may push training into OOM.")
            print("  Wait until training finishes. If you really must run now, set ALLOW_DURING_TRAINING=1.")
            sys.exit(2)

    data = json.load(open(a.train_json))
    sysprompt = data[0].get("system", "")
    allowed = set(re.findall(r'"name"\s*:\s*"([\w.]+)"', sysprompt))
    print(f"data {len(data)} entries · system prompt {len(sysprompt)} chars · "
          f"tool schemas {len(allowed)}")
    if allowed:
        print(f"  {sorted(allowed)}")

    # Look at the ground truth first: the expected format is read from the data, not hard-coded.
    # Take only the **first** assistant turn — build_msgs only feeds the first user turn,
    # so that turn is what the model generates; setting thresholds from multi-turn stats is systematically too low.
    gts = []
    for it in data:
        for c in it.get("conversations", []):
            if c.get("role") == "assistant":
                gts.append(c.get("content", ""))
                break
    expected, gt = census(gts, "marker distribution of the [first assistant turn] in the training data")
    if not gt:
        sys.exit(1)
    print(f"  → expected to appear in ckpt generations: {expected}")

    samples = random.Random(0).sample(data, min(a.n, len(data)))
    print(f"\nsampled {len(samples)} entries (seed=0, greedy decoding)")

    ck = run_model(a.ckpt, samples, sysprompt)
    rc = summarize(ck, allowed, "SFT ckpt")

    rb = None
    if not a.no_base:
        try:
            bs = run_model(a.base, samples, sysprompt)
            rb = summarize(bs, allowed, f"base ({a.base})")
        except Exception as e:
            print(f"\n  ⚠ base comparison cannot run ({type(e).__name__}: {str(e)[:80]}) — skipped")

    print("\n" + "=" * 60)
    # Thresholds are not hard-coded but derived from the ground-truth rate: samples come from the training set (the model has seen them), so the pass line should be close to the ground truth.
    key = "JSON valid" if "tool_call tag" in expected or '"name" field' in expected else "<answer> tag"
    ref = gt.get("tool_call tag", 1.0) if key == "JSON valid" else gt.get("<answer> tag", 1.0)
    pass_at, warn_at = 0.90 * ref, 0.50 * ref
    verdict = 0
    print(f"ground-truth rate {ref*100:.1f}% → pass line {pass_at*100:.0f}% · warning line {warn_at*100:.0f}%")
    if rc[key] >= pass_at:
        print(f"✓ ckpt produced structurally valid tool calls on {rc[key]*100:.0f}% of samples")
    elif rc[key] >= warn_at:
        print(f"⚠ only {rc[key]*100:.0f}% (ground truth {ref*100:.0f}%) — on the low side. "
              f"Look at the raw output below to judge whether the model did not learn or the parsing is too strict")
        verdict = 1
    else:
        print(f"✗ only {rc[key]*100:.0f}% (ground truth {ref*100:.0f}%) — SFT very likely did not take effect")
        verdict = 1
    if rb is not None:
        # compare using the most discriminative format marker, not "JSON valid" (base may manage that anyway)
        dk = DISCRIMINATIVE if DISCRIMINATIVE in rc and DISCRIMINATIVE in rb else key
        d = rc[dk] - rb[dk]
        print(f"  comparison item \"{dk}\": ckpt {rc[dk]*100:.0f}% vs base {rb[dk]*100:.0f}%, difference {d*100:+.0f} percentage points")
        print(f"  (structural item \"{key}\": ckpt {rc[key]*100:.0f}% vs base {rb[key]*100:.0f}%)")
        if d < 0.2:
            print("  ✗ no clear gap from base — this is exactly what \"loss normal but nothing learned\" looks like")
            verdict = 1
        else:
            print("  ✓ clear gap relative to base, SFT really changed the behavior")

    print("\n--- first 2 raw outputs (for a human to glance at) ---")
    for i, o in enumerate(ck[:2]):
        print(f"\n[{i+1}] {o[:600]}")

    if a.out:
        json.dump({"ckpt_rates": rc, "base_rates": rb, "expected_markers": expected,
                   "n": len(samples), "generations": ck},
                  open(a.out, "w"), ensure_ascii=False, indent=2)
        print(f"\ndetails written to {a.out}")

    print("\nReminder: this only tells \"whether SFT took effect\". For real scores run SpaceTools-RL's run_eval.sh.")
    sys.exit(verdict)


if __name__ == "__main__":
    main()
