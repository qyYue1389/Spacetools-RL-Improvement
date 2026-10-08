#!/usr/bin/env python3
"""
Compare whether the text that two toolshed YAMLs ultimately splice into the SFT system prompt is verbatim identical.

Why not a plain `diff`:
    The two YAMLs almost certainly differ in key order, indentation and quoting style (one is checked into the repo,
    the other is exported by generate_toolshed_config.py from a live router), so a raw diff is all noise.
    What really needs comparing is **the string constructed** by lines 88–110 of run_sft.sh —
    because that is the only thing that goes into the system prompt.

This script replicates run_sft.sh's construction logic verbatim, then compares the two results.

Usage:
    python3 diff_tool_text.py A.yaml B.yaml

    # typical: static repo version  vs  the one from your P4 run
    python3 diff_tool_text.py \
        toolshed_v1_config.repo.yaml \
        /path/to/eval_v1_.../toolshed_config.yaml

Exit code:
    0 = verbatim identical, SFT can use either one
    1 = different, **use the P4 one** (that is what the model actually saw during eval)
"""

import sys
import json
import difflib
import hashlib

try:
    import yaml
except ImportError:
    sys.exit("needs pyyaml:  pip install pyyaml")


def build_tool_text(yaml_path):
    """Verbatim replica of the PREP_PY construction in scripts/spacetools/run_sft.sh."""
    with open(yaml_path) as f:
        tool_data = yaml.safe_load(f)

    schemas = [item["tool_schema"] for item in tool_data["tools"] if "tool_schema" in item]

    lines = [
        "\n\n# Tools\n",
        "You may call one or more functions to assist with the user query.\n",
        "You are provided with function signatures within <tools></tools> XML tags:",
        "<tools>",
    ]
    for schema in schemas:
        lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("</tools>\n")
    lines.append(
        "For each function call, return a json object with function name and "
        "arguments within <tool_call></tool_call> XML tags:\n"
        '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    )
    return "\n".join(lines), schemas


def names(schemas):
    return [s.get("function", {}).get("name") for s in schemas]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)

    a_path, b_path = sys.argv[1], sys.argv[2]
    a_text, a_schemas = build_tool_text(a_path)
    b_text, b_schemas = build_tool_text(b_path)

    print(f"A = {a_path}")
    print(f"    {len(a_schemas)} schemas · {len(a_text)} chars · sha256 {hashlib.sha256(a_text.encode()).hexdigest()[:16]}")
    print(f"B = {b_path}")
    print(f"    {len(b_schemas)} schemas · {len(b_text)} chars · sha256 {hashlib.sha256(b_text.encode()).hexdigest()[:16]}")
    print()

    # 1. count and method names (the loudest failure mode: some tool did not come up, a schema is missing)
    if len(a_schemas) != 11 or len(b_schemas) != 11:
        print(f"⚠️  v1 should have 11 schemas. A={len(a_schemas)} B={len(b_schemas)}")
        print("    Fewer usually means some tool actor was not up at the moment of export.")
        print()

    na, nb = names(a_schemas), names(b_schemas)
    if na != nb:
        print("❌ method name lists differ (order counts too):")
        only_a = [n for n in na if n not in nb]
        only_b = [n for n in nb if n not in na]
        if only_a:
            print(f"   only in A: {only_a}")
        if only_b:
            print(f"   only in B: {only_b}")
        if not only_a and not only_b:
            print(f"   same set but different order:\n     A: {na}\n     B: {nb}")
            print("   ⚠️ order also changes the system prompt bytes, still counts as different.")
        print()

    # 2. verbatim comparison of the final text
    if a_text == b_text:
        print("✅ identical — the text spliced into the system prompt is verbatim the same. SFT can use either.")
        return 0

    print("❌ different. Differences below (-=A  +=B):\n")
    # compare line by line per schema, so the whole block is not one unreadable line
    a_lines = a_text.split("\n")
    b_lines = b_text.split("\n")
    shown = 0
    for line in difflib.unified_diff(a_lines, b_lines, lineterm="", n=0):
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("@@"):
            print(line)
            continue
        # a single schema can be very long; truncate the display and point out the first divergence
        tag, body = line[0], line[1:]
        if len(body) > 400:
            body = body[:400] + f" … ({len(body)} chars total)"
        print(tag + body)
        shown += 1
        if shown >= 40:
            print("… too many differences, truncated")
            break

    print()
    print("Action: **use the one from the P4 run** as the SFT TOOL_CONFIG —")
    print("      it is the schema the model actually saw during eval; the training/inference definition must follow it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
