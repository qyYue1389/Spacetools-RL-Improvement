#!/usr/bin/env python3
"""
Parse verl validation dumps into per-sample trajectory records.

verl already writes everything we need. `_dump_generations` emits one JSONL row
per sample under `trainer.validation_data_dir`, and the `output` field is decoded
with `skip_special_tokens=False`, so the entire multi-turn transcript -- every
<think>, <tool_call>, <tool_response> and <answer> -- is present as text. This
script turns that text into structured records. It does not touch the eval.

Two jobs:

  1. P4 guard. Run it right after every benchmark, before believing any score.
     An out-of-memory condition does NOT crash the eval: the tool returns an
     error string, the model reads it as a tool response and carries on, and the
     benchmark reports a plausible number that is quietly wrong. Use --strict.

  2. P6 input. --emit writes enriched records with the trajectory decomposed and
     every failure mode flagged, so error classification never has to re-read a
     log or re-run a GPU.

Usage
-----
    # guard: exit non-zero if the run is contaminated
    python parse_dump.py results/blinkdepth/0.jsonl --strict

    # full report
    python parse_dump.py results/blinkdepth/0.jsonl -v

    # structured records for P6
    python parse_dump.py results/*/0.jsonl --emit parsed/

    # several runs of the same benchmark, as a range
    python parse_dump.py runs/run{4,5,6}/blinkdepth/0.jsonl --compare

Stdlib only, so it runs in any of the five conda environments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

# run_eval.sh sets max_assistant_turns=8
DEFAULT_MAX_TURNS = 8

# tool_agent_loop.py inserts this literal when a tool response exceeds
# max_tool_response_length (2048) with truncate_side=middle
TRUNCATION_MARKER = "...(truncated)"

# Contamination, not a tool result. A run with any of these is not a measurement.
OOM_PATTERNS = [
    re.compile(r"OutOfMemoryError"),
    re.compile(r"CUDA out of memory"),
    re.compile(r"torch\.cuda\.OutOfMemoryError"),
]

# A tool that ran and reported it could not do the job. These are legitimate
# tool responses the model is expected to recover from -- they are signal for
# P6, not run contamination.
TOOL_FAILURE_PATTERNS = [
    re.compile(r"No collision-free grasps"),
    re.compile(r"Top-down filtering removed all"),
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"\bRuntimeError\b"),
    re.compile(r"\bValueError\b"),
    re.compile(r"\bERROR\b"),
    re.compile(r"^Error[: ]", re.MULTILINE),
]

# ---------------------------------------------------------------------------
# transcript parsing
# ---------------------------------------------------------------------------

TURN_SPLIT = re.compile(r"<\|im_start\|>")
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESP_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
# Toolshed hands large results back by reference, never inline: a response says
# "Use $depth_map (numpy array, 364x504) ... and $focal_length_px (float) ...".
# Which variables existed, and which the model actually referenced, is what
# separates P6 taxonomy 2d ("recomputed something it already had") from a
# model that reused its results properly. Both sides are cheap to extract.
VAR_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")


def _match_any(patterns, text):
    for p in patterns:
        if p.search(text):
            return p.pattern
    return None


def parse_transcript(output: str, max_turns: int = DEFAULT_MAX_TURNS) -> dict:
    """Decompose one raw `output` string into a structured trajectory.

    The Qwen template emits assistant generations as `<|im_start|>assistant ...`
    and feeds tool results back as user turns wrapping one or more
    `<tool_response>` blocks. Consecutive tool results share a single user turn,
    so responses are collected per user turn and paired positionally with the
    calls from the preceding assistant turn.

    Nothing here assumes well-formed output: a rollout that hit the response
    length cap can end mid-tag, and that must be reported rather than crash.
    """
    trajectory = []
    tool_calls_flat = []
    tool_responses_flat = []
    malformed_tool_calls = 0
    n_user_turns = 0  # tool-result turns, needed for verl's turn convention

    chunks = TURN_SPLIT.split(output)
    pending = None  # assistant turn awaiting its tool responses

    # verl dumps `output` as the RESPONSE ONLY: the opening
    # `<|im_start|>assistant` of the first turn lives at the tail of `input`,
    # so `output` begins part-way through that turn with no role header. Every
    # later turn is introduced by its own `<|im_start|>`. Splitting on
    # `<|im_start|>` therefore yields a first chunk that is the body of assistant
    # turn 1, and treating it like the others -- reading its first line as a role
    # -- silently discards it.
    #
    # That is not a cosmetic loss. Turn 1 is where the standard blinkdepth chain
    # issues depth_estimator + roborefer x2, so dropping it removed
    # depth_estimator from the tool histogram of a DEPTH benchmark entirely,
    # undercounted every turn count by one, and swallowed whole any sample that
    # answered in a single turn (which then reported as "no <answer>").
    leading_is_assistant = not output.lstrip().startswith("<|im_start|>")

    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        if idx == 0 and leading_is_assistant:
            # no role header to strip; the whole chunk is the turn body
            role, body = "assistant", chunk
        else:
            # first line is the role
            head, _, body = chunk.partition("\n")
            role = head.strip().replace("<|im_end|>", "").strip()
        body = body.replace("<|im_end|>", "")

        if role == "assistant":
            think = [m.strip() for m in THINK_RE.findall(body)]
            answer_m = ANSWER_RE.search(body)
            calls = []
            for raw in TOOL_CALL_RE.findall(body):
                try:
                    obj = json.loads(raw)
                    args = obj.get("arguments")
                    calls.append(
                        {
                            "name": obj.get("name"),
                            "arguments": args,
                            # $variables this call reads back from earlier tools
                            "arg_vars": sorted({
                                m for v in (args or {}).values()
                                if isinstance(v, str)
                                for m in VAR_RE.findall(v)
                            }) if isinstance(args, dict) else [],
                            "raw": None,
                        }
                    )
                except Exception:
                    malformed_tool_calls += 1
                    calls.append({"name": None, "arguments": None,
                                  "arg_vars": [], "raw": raw[:500]})
            turn = {
                "turn": len(trajectory) + 1,
                "role": "assistant",
                "think": think,
                "tool_calls": calls,
                "tool_responses": [],
                "answer": answer_m.group(1).strip() if answer_m else None,
                # a turn with neither a tool call nor an answer means the
                # generation was cut off, which is worth seeing
                "empty": not calls and not answer_m,
            }
            trajectory.append(turn)
            tool_calls_flat.extend(calls)
            pending = turn

        elif role in ("user", "tool"):
            n_user_turns += 1
            responses = TOOL_RESP_RE.findall(body)
            if not responses:
                continue
            recs = []
            for i, text in enumerate(responses):
                name = None
                if pending is not None and i < len(pending["tool_calls"]):
                    name = pending["tool_calls"][i]["name"]
                oom = _match_any(OOM_PATTERNS, text)
                rec = {
                    "name": name,
                    "text": text,
                    # $variables this response makes available to later turns
                    "vars": sorted(set(VAR_RE.findall(text))),
                    "chars": len(text),
                    "truncated": TRUNCATION_MARKER in text,
                    "oom": oom is not None,
                    "oom_pattern": oom,
                    # OOM is contamination; only classify a non-OOM response as
                    # a tool failure, or every OOM would also look like one
                    "tool_failure": (
                        _match_any(TOOL_FAILURE_PATTERNS, text) if oom is None else None
                    ),
                }
                recs.append(rec)
            if pending is not None:
                pending["tool_responses"] = recs
            tool_responses_flat.extend(recs)

    # ---- aggregates -------------------------------------------------------
    by_method = Counter(c["name"] for c in tool_calls_flat if c["name"])
    by_tool = Counter(
        c["name"].split(".")[0] for c in tool_calls_flat if c["name"] and "." in c["name"]
    )

    answers = [t["answer"] for t in trajectory if t["answer"] is not None]
    n_turns = len(trajectory)

    return {
        "trajectory": trajectory,
        # Variable-reuse signal for P6 taxonomy 2d. `vars_unused` is the
        # interesting one: a variable a tool published and the model never read
        # back is a candidate for "recomputed what it already had". It is a lead,
        # not a verdict -- some variables are simply not needed.
        "vars_exposed": sorted({v for t in trajectory
                                for r in t["tool_responses"] for v in r["vars"]}),
        "vars_used": sorted({v for t in trajectory
                             for c in t["tool_calls"] for v in c.get("arg_vars", [])}),
        "num_turns_derived": n_turns,
        "num_user_turns_derived": n_user_turns,
        # verl's own convention, for the cross-check against its dumped count:
        #   tool_agent_loop.py:238
        #   num_turns = agent_data.user_turns + agent_data.assistant_turns + 1
        # the +1 is the initial prompt turn, which lives in `input`, not `output`.
        "num_turns_verl_convention": n_turns + n_user_turns + 1,
        "hit_max_turns": n_turns >= max_turns,
        "tool_stats": dict(by_tool),
        "tool_stats_by_method": dict(by_method),
        "n_tool_calls": len(tool_calls_flat),
        "malformed_tool_calls": malformed_tool_calls,
        "answer": answers[-1] if answers else None,
        "has_answer": bool(answers),
        "n_answers": len(answers),
        "oom": any(r["oom"] for r in tool_responses_flat),
        "n_oom": sum(1 for r in tool_responses_flat if r["oom"]),
        "truncated_tool_response": any(r["truncated"] for r in tool_responses_flat),
        "n_truncated": sum(1 for r in tool_responses_flat if r["truncated"]),
        "tool_failures": [
            {"turn": t["turn"], "name": r["name"], "pattern": r["tool_failure"],
             "text": r["text"][:300]}
            for t in trajectory
            for r in t["tool_responses"]
            if r["tool_failure"]
        ],
        # A compact signature for grouping identical tool-use patterns.
        # `@Nt` counts ASSISTANT turns, which is the same thing
        # max_assistant_turns=8 and hit_max_turns count. Note P2_RESULTS.md's
        # hand analysis used a different convention -- turns that CALLED a tool
        # -- so its "12 x depth x1, roborefer x2, vision_ops x2 in 2 turns"
        # appears here as the same 12 samples at @3t: the third assistant turn
        # is the one that emits <answer> and calls nothing.
        "chain_signature": "+".join(
            f"{k}x{v}" for k, v in sorted(by_tool.items())
        ) + f"@{n_turns}t",
        "truncated_generation": bool(trajectory and trajectory[-1]["empty"]),
    }


# ---------------------------------------------------------------------------
# record building
# ---------------------------------------------------------------------------


_CHOICE = re.compile(r"\(([A-Z])\)")


def _answer_off_options(question, answer):
    """True when the question offers lettered choices and the answer is not one.

    Returns None when there are no choices to check against, so that free-form
    benchmarks (pointing, pose, grasp) are never flagged. Deliberately narrow:
    it only looks at the FIRST character of the answer, because that is how the
    reward functions read a multiple-choice answer.
    """
    if not question or not answer:
        return None
    opts = set(_CHOICE.findall(question))
    if len(opts) < 2:
        return None
    a = str(answer).strip().upper()
    return bool(a) and a[0] not in opts


def _extract_question(rendered_input) -> str | None:
    """Pull just the question out of verl's rendered prompt.

    `input` is the whole chat template -- system prompt, tool schemas, then the
    user turn -- about 10 KB per sample. Carrying all of it would add ~21 MB
    across the nine benchmarks for text that is identical in every row bar the
    last few lines. The question sits between the final `user` marker and the
    trailing `assistant` one.
    """
    if not rendered_input:
        return None
    text = str(rendered_input)
    tail = text.rsplit("\nuser\n", 1)[-1]
    tail = tail.rsplit("\nassistant\n", 1)[0] if "\nassistant\n" in tail else tail
    tail = tail.strip()
    return tail or None


def build_record(row: dict, benchmark: str, max_turns: int, threshold: float) -> dict:
    parsed = parse_transcript(row.get("output") or "", max_turns=max_turns)

    score = row.get("score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = None

    rec = {
        "benchmark": benchmark,
        # verl dumps `index` from extra_info; `uid` only if the dump patch is applied
        "sample_id": row.get("index", row.get("uid")),
        "uid": row.get("uid"),
        # verl's own image field is None in practice: multi_modal_data never
        # reaches _dump_generations on this eval path, so its PNG-saving branch
        # (ray_trainer.py:432) never fires. It is not needed. `sample_id` is the
        # parquet row index -- verified 124/124 against blinkdepth's ground truth
        # -- so P6 recovers the original image as
        #     parquet[sample_id]["images"][0]["image"]   (a base64 data URI)
        # which is exact and avoids duplicating 292 MB of images onto disk.
        # Subsets from make_subset.py use head(N), so row i of the subset IS row
        # i of the full file and the index means the same thing either way.
        # That equivalence would break under random sampling.
        "image": row.get("image"),
        "gt": row.get("gts", row.get("answer")),
        "question": _extract_question(row.get("input")),
        "score": score,
        "reward": row.get("reward"),
        "correct": (score is not None and score >= threshold),
        "raw_answer": parsed["answer"],
        "parse_ok": parsed["has_answer"],
        # parse_ok only says an <answer> block existed and parsed. It does NOT
        # say the answer is one of the offered choices. P6 found blinkdepth #69
        # answering "C" to an (A)/(B) question: tags well formed, parse_ok True,
        # --strict silent, and P6's "format errors = 0" was wrong because of it.
        # None when the question offers no lettered choices.
        "answer_off_options": _answer_off_options(
            _extract_question(row.get("input")), parsed["answer"]),
    }
    rec.update(
        {k: parsed[k] for k in (
            "num_turns_derived", "num_user_turns_derived",
            "num_turns_verl_convention", "hit_max_turns", "tool_stats",
            "vars_exposed", "vars_used",
            "tool_stats_by_method", "n_tool_calls", "malformed_tool_calls",
            "oom", "n_oom", "truncated_tool_response", "n_truncated",
            "tool_failures", "chain_signature", "truncated_generation",
            "trajectory",
        )}
    )

    # exposed but never read back -- the lead for P6 taxonomy 2d
    rec["vars_unused"] = sorted(set(rec["vars_exposed"]) - set(rec["vars_used"]))
    # referenced but NEVER published by any tool -- the model invented the name.
    # A far cleaner signal than vars_unused: taxonomy 2c, wrong arguments.
    # Observed live in P4 blinkdepth run1 sample 89, which asked bounding_box for
    # $point_cloud and $segmentation_mask having called neither
    # estimate_depth_with_pointcloud nor sam2. The tool rejected it
    # ("must be Nx3 numpy array, got str"), the model recovered and answered
    # correctly. Signal for P6, not contamination -- --strict passes it.
    rec["vars_phantom"] = sorted(set(rec["vars_used"]) - set(rec["vars_exposed"]))

    # cross-check against verl's own count when the dump patch is in place.
    # a mismatch means the transcript parser is wrong, and that must be loud.
    if "num_turns" in row:
        try:
            rec["num_turns_reported"] = int(row["num_turns"])
            # Compare like with like. verl counts user + assistant + 1; the
            # parser's num_turns_derived counts ASSISTANT turns only, so
            # comparing the two directly makes every sample look like a
            # mismatch and turns --strict into a benchmark-wide false alarm.
            rec["num_turns_agree"] = (
                rec["num_turns_reported"] == parsed["num_turns_verl_convention"]
            )
        except (TypeError, ValueError):
            pass

    return rec


def benchmark_name(path: str) -> str:
    """run_eval.sh writes $EVAL_DIR/<benchmark>/<step>.jsonl"""
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return parent or os.path.basename(path)


def load(path: str, max_turns: int, threshold: float):
    bench = benchmark_name(path)
    out = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                print(f"  !! {path}:{lineno} unparseable JSON: {e}", file=sys.stderr)
                continue
            out.append(build_record(row, bench, max_turns, threshold))
    return out


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def summarise(recs, threshold, verbose=False):
    n = len(recs)
    if n == 0:
        return {"n": 0}

    scored = [r["score"] for r in recs if r["score"] is not None]
    n_correct = sum(1 for r in recs if r["correct"])
    turns = [r["num_turns_derived"] for r in recs]

    s = {
        "n": n,
        "mean_score": statistics.fmean(scored) if scored else None,
        "n_correct": n_correct,
        "accuracy": n_correct / n,
        "mean_turns": statistics.fmean(turns) if turns else 0,
        "oom_samples": sum(1 for r in recs if r["oom"]),
        "oom_events": sum(r["n_oom"] for r in recs),
        "truncated_samples": sum(1 for r in recs if r["truncated_tool_response"]),
        "max_turn_samples": sum(1 for r in recs if r["hit_max_turns"]),
        "no_answer": sum(1 for r in recs if not r["parse_ok"]),
        "answer_off_options": sum(1 for r in recs if r.get("answer_off_options")),
        "multiple_choice": sum(1 for r in recs
                               if r.get("answer_off_options") is not None),
        "cut_off_generation": sum(1 for r in recs if r["truncated_generation"]),
        "malformed_tool_calls": sum(r["malformed_tool_calls"] for r in recs),
        "tool_failure_samples": sum(1 for r in recs if r["tool_failures"]),
        "tool_failure_events": sum(len(r["tool_failures"]) for r in recs),
    }

    checked = [r for r in recs if "num_turns_agree" in r]
    disagree = [r for r in recs if r.get("num_turns_agree") is False]
    s["turn_count_disagreements"] = len(disagree)
    s["turn_count_checked"] = len(checked) > 0

    tool_calls = Counter()
    for r in recs:
        tool_calls.update(r["tool_stats"])
    s["tool_calls"] = dict(tool_calls.most_common())

    fail_reasons = Counter()
    for r in recs:
        for tf in r["tool_failures"]:
            fail_reasons[(tf["name"] or "?", tf["pattern"] or "?")] += 1
    s["tool_failure_reasons"] = {f"{k[0]} :: {k[1]}": v for k, v in fail_reasons.most_common()}

    s["chains"] = dict(Counter(r["chain_signature"] for r in recs).most_common(10))

    # what distinguishes the wrong samples -- the P6 starting point
    wrong = [r for r in recs if not r["correct"]]
    s["wrong"] = {
        "n": len(wrong),
        "with_tool_failure": sum(1 for r in wrong if r["tool_failures"]),
        "with_truncation": sum(1 for r in wrong if r["truncated_tool_response"]),
        "hit_max_turns": sum(1 for r in wrong if r["hit_max_turns"]),
        "no_answer": sum(1 for r in wrong if not r["parse_ok"]),
        "oom": sum(1 for r in wrong if r["oom"]),
        "chains": dict(Counter(r["chain_signature"] for r in wrong).most_common(5)),
    }
    # a wrong sample with none of the above is a genuine tool-accuracy or
    # reasoning failure, which is where the interesting P6 work is
    s["wrong"]["clean"] = len(
        [
            r
            for r in wrong
            if not r["tool_failures"]
            and not r["truncated_tool_response"]
            and not r["hit_max_turns"]
            and r["parse_ok"]
            and not r["oom"]
        ]
    )

    if verbose:
        s["_grasp_split"] = grasp_split(recs)
    return s


def grasp_split(recs):
    """MACE/SR split by whether the grasp tool actually produced a pose.

    P2 measured 27 succeeded / 33 errored on bopgrasp, with SR 77.78% vs 36.36%.
    Reproducing that split is the cheapest check that the parser reads tool
    failures correctly.
    """
    groups = defaultdict(list)
    for r in recs:
        if "grasp_generator" not in r["tool_stats"]:
            continue
        errored = any(
            tf["name"] and tf["name"].startswith("grasp_generator")
            for tf in r["tool_failures"]
        )
        groups["errored" if errored else "succeeded"].append(r)
    out = {}
    for k, v in groups.items():
        scored = [r["score"] for r in v if r["score"] is not None]
        out[k] = {
            "n": len(v),
            "mean_score": statistics.fmean(scored) if scored else None,
            "frac_above_threshold": sum(1 for r in v if r["correct"]) / len(v) if v else None,
        }
    return out


def print_summary(path, s, threshold):
    print(f"\n=== {path} ===")
    if s["n"] == 0:
        print("  empty")
        return
    print(f"  samples             {s['n']}")
    if s["mean_score"] is not None:
        print(f"  mean score          {s['mean_score']:.4f}")
    print(f"  score >= {threshold:<10.2f} {s['n_correct']}/{s['n']} = {100*s['accuracy']:.2f}%")
    print(f"  mean turns          {s['mean_turns']:.2f}")

    print("  --- run health (all should be 0) ---")
    flag = lambda v: "  <-- CONTAMINATED" if v else ""
    print(f"  OOM samples         {s['oom_samples']} ({s['oom_events']} events){flag(s['oom_samples'])}")
    print(f"  cut-off generation  {s['cut_off_generation']}")
    print(f"  malformed tool_call {s['malformed_tool_calls']}")
    # Always print when the dump carries verl's own count (patches/rl/0008), not
    # only on failure: a check that is silent when it passes is a check nobody
    # can tell ran. "n/a" means the dump predates the patch and the parser's
    # decomposition is unverified.
    if s.get("turn_count_checked"):
        print(f"  turn-count mismatch {s['turn_count_disagreements']}"
              f"{'  <-- PARSER BUG' if s['turn_count_disagreements'] else ''}")
    else:
        print("  turn-count mismatch n/a  (dump has no verl num_turns)")

    print("  --- infrastructure limits hit ---")
    print(f"  truncated responses {s['truncated_samples']}")
    print(f"  hit max turns       {s['max_turn_samples']}")
    print(f"  no <answer>         {s['no_answer']}")
    if s.get("multiple_choice"):
        flag = "   <- answered outside the offered choices" if s["answer_off_options"] else ""
        print(f"  off-options answer  {s['answer_off_options']}"
              f" / {s['multiple_choice']} MC{flag}")

    print("  --- tools ---")
    print(f"  calls               {s['tool_calls']}")
    print(f"  tool failures       {s['tool_failure_samples']} samples, {s['tool_failure_events']} events")
    for k, v in s["tool_failure_reasons"].items():
        print(f"      {v:>4}  {k}")

    print("  --- chains (top) ---")
    for k, v in s["chains"].items():
        print(f"      {v:>4}  {k}")

    w = s["wrong"]
    print(f"  --- wrong samples: {w['n']} ---")
    print(f"      tool failure    {w['with_tool_failure']}")
    print(f"      truncation      {w['with_truncation']}")
    print(f"      max turns       {w['hit_max_turns']}")
    print(f"      no answer       {w['no_answer']}")
    print(f"      OOM             {w['oom']}")
    print(f"      clean           {w['clean']}   <- real tool-accuracy / reasoning failures")
    if w.get("chains"):
        print("      chains")
        for k, v in w["chains"].items():
            print(f"        {v:>4}  {k}")

    if "_grasp_split" in s and s["_grasp_split"]:
        print("  --- grasp split ---")
        for k, v in s["_grasp_split"].items():
            ms = f"{v['mean_score']:.4f}" if v["mean_score"] is not None else "n/a"
            fa = f"{100*v['frac_above_threshold']:.2f}%" if v["frac_above_threshold"] is not None else "n/a"
            print(f"      {k:<10} n={v['n']:<4} mean={ms}  >=thr={fa}")


def print_compare(per_file, threshold):
    """Several runs of one benchmark: report the band, not a point value."""
    print("\n=== run-to-run comparison ===")
    accs, corrects = [], []
    for path, s in per_file:
        if s["n"] == 0:
            continue
        accs.append(s["accuracy"])
        corrects.append(s["n_correct"])
        print(f"  {os.path.basename(os.path.dirname(path)) or path:<20} "
              f"{s['n_correct']:>4}/{s['n']:<5} {100*s['accuracy']:>7.2f}%")
    if len(accs) < 2:
        return
    lo, hi = min(accs), max(accs)
    print(f"  {'mean':<20} {statistics.fmean(corrects):>8.1f}      {100*statistics.fmean(accs):>7.2f}%")
    print(f"  {'range':<20} {min(corrects)}-{max(corrects)}        {100*(hi-lo):>7.2f} pp")
    print("\n  Report this as a band, not a point value. A difference smaller than")
    print("  the band is not a difference.")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Parse verl validation dumps into trajectory records.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("dumps", nargs="+", help="verl dump JSONL file(s)")
    ap.add_argument("--emit", metavar="DIR",
                    help="write enriched records to DIR/<benchmark>.jsonl")
    ap.add_argument("--emit-slim", action="store_true",
                    help="omit the full trajectory from --emit output")
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS,
                    help=f"max_assistant_turns used for the run (default {DEFAULT_MAX_TURNS})")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="score >= this counts as correct (default 0.5); for "
                         "multiple-choice benchmarks the score is already 0/1")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any run looks contaminated (OOM, malformed "
                         "tool calls, or turn-count mismatch). Use in P4.")
    ap.add_argument("--compare", action="store_true",
                    help="treat the inputs as repeat runs and report the band")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="include the grasp success/failure split")
    ap.add_argument("--json", action="store_true", help="emit summaries as JSON")
    args = ap.parse_args()

    per_file = []
    contaminated = []

    for path in args.dumps:
        if not os.path.isfile(path):
            print(f"!! not a file: {path}", file=sys.stderr)
            contaminated.append(path)
            continue
        recs = load(path, args.max_turns, args.threshold)
        s = summarise(recs, args.threshold, verbose=args.verbose)
        per_file.append((path, s))

        # A pass@k dump repeats each index n times. Detect that, because the
        # malformed-tool_call rule below was written for greedy runs, where a
        # malformed call means the pipeline is broken. Under do_sample=True the
        # model genuinely emits a low-probability garbage token now and then --
        # P6 experiment C hit exactly one in 1750 samplings on robospatial (a
        # stray "槛" before an otherwise well-formed call, on a sample whose
        # other four draws were correct, so it moved no number). Failing the
        # gate on that would make every sampling run un-gateable, so it is
        # reported as a warning and the other contamination rules still apply.
        idx_counts = Counter(r.get("index") for r in recs)
        n_draws = max(idx_counts.values()) if idx_counts else 1
        sampling = n_draws > 1

        hard = s["oom_samples"] or s["turn_count_disagreements"]
        if s["n"] and (hard or (s["malformed_tool_calls"] and not sampling)):
            contaminated.append(path)
        elif s["n"] and s["malformed_tool_calls"] and sampling:
            print(f"\n!! WARNING: {s['malformed_tool_calls']} malformed tool_call(s) "
                  f"in a sampling run (n={n_draws} draws/sample).", file=sys.stderr)
            print("   Not treated as contamination -- see the note in the source. "
                  "Inspect them before reporting.", file=sys.stderr)

        # WARNING, not contamination. An off-options answer is a real model
        # failure (format error), not a broken pipeline, so it must not gate a
        # run the way OOM or a turn-count mismatch does -- and making it gate
        # would retroactively change the status of runs that already passed.
        # P4's full sweep has exactly one (blinkdepth #69, "C" to an A/B
        # question) out of 774 multiple-choice samples.
        if s.get("answer_off_options"):
            print(f"\n!! WARNING: {s['answer_off_options']} sample(s) answered "
                  f"outside the offered choices ({path}).", file=sys.stderr)
            print("   parse_ok is True for these -- the <answer> block is well "
                  "formed, the content just is not one of the options.",
                  file=sys.stderr)
            print("   Counts as a format error in P6, not as contamination.",
                  file=sys.stderr)

        if args.emit:
            os.makedirs(args.emit, exist_ok=True)
            bench = benchmark_name(path)
            out = os.path.join(args.emit, f"{bench}.jsonl")
            # several runs of one benchmark would otherwise overwrite
            i = 2
            while os.path.exists(out):
                out = os.path.join(args.emit, f"{bench}.{i}.jsonl")
                i += 1
            with open(out, "w", encoding="utf-8") as f:
                for r in recs:
                    if args.emit_slim:
                        r = {k: v for k, v in r.items() if k != "trajectory"}
                    f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            print(f"wrote {len(recs)} records to {out}")

    if args.json:
        print(json.dumps({p: s for p, s in per_file}, indent=2, default=str))
    else:
        for path, s in per_file:
            print_summary(path, s, args.threshold)
        if args.compare:
            print_compare(per_file, args.threshold)

    if contaminated:
        print("\n!! CONTAMINATED OR UNREADABLE:", file=sys.stderr)
        for p in contaminated:
            print(f"     {p}", file=sys.stderr)
        print("   Do not report scores from these runs. Fix the cause and re-run.",
              file=sys.stderr)
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
