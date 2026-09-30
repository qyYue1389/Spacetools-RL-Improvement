#!/usr/bin/env python3
"""Apply the SFT checkpoint config fixes to a released SpaceTools checkpoint.

docs/SETUP.md documents two forms of config pollution produced by
SpaceTools-SFT under transformers 4.57.1, and run_sft.sh (phase 3) fixes them
automatically for checkpoints you train yourself. The released checkpoint
siyich/spacetools-ckpt ships WITHOUT those fixes applied, so eval crashes:

  1. config.json carries a `text_config` section, which makes transformers
     resolve model_type to "qwen2_5_vl_text" (the text submodel) instead of
     "qwen2_5_vl". sglang then dies with:
         RuntimeError: Unimplemented model type: qwen2_5_vl_text

  2. preprocessor_config.json uses Qwen2VLImageProcessorFast, whose image token
     count disagrees with what the vision encoder expects, giving an off-by-one:
         ValueError: Image features and image tokens do not match

run_sft.sh additionally sets tie_word_embeddings=True. That is only safe when
lm_head really is tied to the input embedding, so this script verifies it by
comparing the two tensors instead of assuming.

Originals are copied to *.orig before anything is written.

Usage:
    python fix_checkpoint.py [--ckpt /workspace/models/spacetools-ckpt]
                             [--base Qwen/Qwen2.5-VL-3B-Instruct]
                             [--dry-run]
"""

import argparse
import json
import os
import shutil


def backup(path):
    """Copy path to path.orig once, so the original stays auditable."""
    orig = path + ".orig"
    if os.path.exists(path) and not os.path.exists(orig):
        shutil.copy2(path, orig)
        print(f"  backed up -> {os.path.basename(orig)}")


def embeddings_are_tied(ckpt):
    """True when lm_head.weight is bit-identical to the input embedding.

    A checkpoint can ship a materialised copy of a tied head, which is fine.
    An actually untied head must keep tie_word_embeddings unset, otherwise
    transformers discards the trained output head and silently reuses the
    embedding.
    """
    import torch
    from safetensors import safe_open

    index = os.path.join(ckpt, "model.safetensors.index.json")
    if not os.path.isfile(index):
        print("  no safetensors index; cannot verify tying")
        return None
    weight_map = json.load(open(index))["weight_map"]
    emb_key = next((k for k in weight_map if k.endswith("embed_tokens.weight")), None)
    lm_key = next((k for k in weight_map if "lm_head" in k), None)
    if emb_key is None:
        print("  no embedding tensor found; cannot verify tying")
        return None
    if lm_key is None:
        print("  no lm_head tensor: head is tied by construction")
        return True

    def load(key):
        with safe_open(os.path.join(ckpt, weight_map[key]), framework="pt") as f:
            return f.get_tensor(key)

    emb, lm = load(emb_key), load(lm_key)
    if emb.shape != lm.shape:
        print(f"  lm_head {tuple(lm.shape)} != embedding {tuple(emb.shape)}: untied")
        return False
    tied = torch.equal(emb, lm)
    print(f"  lm_head vs embedding: {'identical (tied)' if tied else 'differ (untied)'}")
    return tied


def fix_config(ckpt, dry_run):
    path = os.path.join(ckpt, "config.json")
    cfg = json.load(open(path))
    changes = []

    if "text_config" in cfg:
        changes.append("remove text_config")
    else:
        print("  text_config already absent")

    tied = embeddings_are_tied(ckpt)
    if tied and cfg.get("tie_word_embeddings") is not True:
        changes.append("set tie_word_embeddings=True")
    elif tied is False:
        print("  leaving tie_word_embeddings alone: the head is genuinely untied")

    if not changes:
        print("  config.json: nothing to do")
        return
    print(f"  config.json: {', '.join(changes)}")
    if dry_run:
        return

    backup(path)
    cfg.pop("text_config", None)
    if tied:
        cfg["tie_word_embeddings"] = True
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print("  config.json written")


def fix_preprocessor(ckpt, base, dry_run):
    path = os.path.join(ckpt, "preprocessor_config.json")
    current = json.load(open(path)).get("image_processor_type")
    print(f"  current image_processor_type: {current}")
    if current != "Qwen2VLImageProcessorFast":
        print("  preprocessor_config.json: nothing to do")
        return
    if dry_run:
        print(f"  preprocessor_config.json: would replace with {base}'s copy")
        return

    if os.path.isdir(base):
        src = os.path.join(base, "preprocessor_config.json")
    else:
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(base, "preprocessor_config.json")
    backup(path)
    shutil.copy(src, path)
    replaced = json.load(open(path)).get("image_processor_type")
    print(f"  preprocessor_config.json replaced from {base} (now {replaced})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/models/spacetools-ckpt")
    ap.add_argument("--base", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"Checkpoint: {args.ckpt}")
    print(f"Base model: {args.base}")
    if args.dry_run:
        print("(dry run: nothing will be written)")
    print("\n[1/2] config.json")
    fix_config(args.ckpt, args.dry_run)
    print("\n[2/2] preprocessor_config.json")
    fix_preprocessor(args.ckpt, args.base, args.dry_run)

    print("\nCurrent state (unchanged):" if args.dry_run else "\nResult:")
    cfg = json.load(open(os.path.join(args.ckpt, "config.json")))
    pre = json.load(open(os.path.join(args.ckpt, "preprocessor_config.json")))
    print(f"  model_type           : {cfg.get('model_type')}")
    print(f"  text_config present  : {'text_config' in cfg}")
    print(f"  tie_word_embeddings  : {cfg.get('tie_word_embeddings', '<unset>')}")
    print(f"  image_processor_type : {pre.get('image_processor_type')}")


if __name__ == "__main__":
    main()
