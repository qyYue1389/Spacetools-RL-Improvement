#!/usr/bin/env python3
"""
比较两份 toolshed YAML 最终会拼进 SFT system prompt 的那段文本是否逐字一致。

为什么不用普通的 `diff`:
    两份 YAML 的键顺序、缩进、引号风格几乎必然不同(一份是签在仓库里的,
    一份是 generate_toolshed_config.py 从活 router 导出的),裸 diff 全是噪声。
    真正要比的是 run_sft.sh 第 88–110 行**构造出来的那段字符串** ——
    因为进 system prompt 的只有它。

本脚本逐字复刻 run_sft.sh 的构造逻辑,然后比两段结果。

用法:
    python3 diff_tool_text.py A.yaml B.yaml

    # 典型:仓库静态版  vs  你 P4 那次跑出来的
    python3 diff_tool_text.py \
        toolshed_v1_config.repo.yaml \
        /path/to/eval_v1_.../toolshed_config.yaml

退出码:
    0 = 逐字一致,SFT 可以用任意一份
    1 = 不一致,**用 P4 那份**(它才是 eval 时模型真正看到的)
"""

import sys
import json
import difflib
import hashlib

try:
    import yaml
except ImportError:
    sys.exit("需要 pyyaml:  pip install pyyaml")


def build_tool_text(yaml_path):
    """逐字复刻 scripts/spacetools/run_sft.sh 里 PREP_PY 的构造。"""
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
    print(f"    {len(a_schemas)} 个 schema · {len(a_text)} 字符 · sha256 {hashlib.sha256(a_text.encode()).hexdigest()[:16]}")
    print(f"B = {b_path}")
    print(f"    {len(b_schemas)} 个 schema · {len(b_text)} 字符 · sha256 {hashlib.sha256(b_text.encode()).hexdigest()[:16]}")
    print()

    # 1. 数量与方法名(最响的失败模式:某个工具没起来,schema 少了)
    if len(a_schemas) != 11 or len(b_schemas) != 11:
        print(f"⚠️  v1 应当是 11 个 schema。A={len(a_schemas)} B={len(b_schemas)}")
        print("    少了通常意味着导出那一刻某个工具 actor 没起来。")
        print()

    na, nb = names(a_schemas), names(b_schemas)
    if na != nb:
        print("❌ 方法名列表不同(顺序也算):")
        only_a = [n for n in na if n not in nb]
        only_b = [n for n in nb if n not in na]
        if only_a:
            print(f"   只在 A: {only_a}")
        if only_b:
            print(f"   只在 B: {only_b}")
        if not only_a and not only_b:
            print(f"   集合相同但顺序不同:\n     A: {na}\n     B: {nb}")
            print("   ⚠️ 顺序也会改变 system prompt 的字节,仍算不一致。")
        print()

    # 2. 逐字比较最终文本
    if a_text == b_text:
        print("✅ 一致 —— 拼进 system prompt 的那段文本逐字相同。SFT 用哪一份都行。")
        return 0

    print("❌ 不一致。差异如下(-=A  +=B):\n")
    # 按 schema 拆行比,避免整段单行不可读
    a_lines = a_text.split("\n")
    b_lines = b_text.split("\n")
    shown = 0
    for line in difflib.unified_diff(a_lines, b_lines, lineterm="", n=0):
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("@@"):
            print(line)
            continue
        # 单个 schema 可能很长,截断显示并指出第一个分歧位置
        tag, body = line[0], line[1:]
        if len(body) > 400:
            body = body[:400] + f" …(共 {len(body)} 字符)"
        print(tag + body)
        shown += 1
        if shown >= 40:
            print("… 差异过多,已截断")
            break

    print()
    print("处置:**用 P4 那次跑出来的那份**做 SFT 的 TOOL_CONFIG ——")
    print("      它才是 eval 时模型真正看到的 schema,训练/推理口径必须以它为准。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
