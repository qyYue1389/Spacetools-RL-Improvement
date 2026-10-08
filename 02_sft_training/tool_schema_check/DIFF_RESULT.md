# SFT schema definition comparison result

> What is compared: the tool schema text that `run_sft.sh` splices into the SFT system prompt,
> versus the text **the model actually saw during the P4 eval** — are they verbatim identical.
> Data source: the `input` field of row 0 of `spacetools-repro/p4/dumps/run1/blinkdepth/0.jsonl`.
> Date: 2026-09-06

---

## Conclusion: **Not identical, two differences. Use `toolshed_v1_config.p4.yaml`, not the static version in the repo.**

| | Repo static version<br>`SpaceTools-RL/examples/toolshed/toolshed_v1_config.yaml` | **P4 actual** |
|---|---|---|
| Number of schemas | 11 | 11 ✓ |
| Set of method names | Same | Same ✓ |
| **Order** | See below | **Different** ❌ |
| **Description of `vlm.detect_one` / `detect_all`** | Missing one line | **169 more characters × 2** ❌ |
| `<tools>` block bytes | 8257 | **8595** |
| sha256 (first 16) | `436221881e80cacd` | `72c71c806f64e162` |

---

## Difference 1: the two `vlm` schemas lack the variable description (**this one has real impact**)

The descriptions of P4's `vlm.detect_one` / `vlm.detect_all` have this one extra line compared with the repo version:

```
Stored variables: Coordinates in ${obj_name}_detections variable of list of (x, y)
point coordinates in normalized pixel space [0, 1] for use in subsequent operations.
```

**Cause**: the tool docstring uses a `[[if:vars]] … [[/if:vars]]` conditional block; the rendered result depends on
`no_output_vars`. `run_eval.sh` configures `vlm` with `no_output_vars: False`, so this line is present.
The static YAML in the repo was exported under a different config (or an older Toolshed) and does not have it.

> For comparison: the two `roborefer` schemas are **completely identical on both sides** — because
> `roborefer.py` hardcodes `print("WARNING: RoboRefer forcing no_output_vars to True")`,
> so its vars line never appears, and there is no divergence.

**Why this one cannot be ignored**: in P4 trajectories the model really does use variables like
`$red_point_under_label_A_detections` — it knows these variables exist, **and it learned that from this line**.
If SFT uses the repo static version, the prompt during training never mentions these variables while the prompt during eval does.
**Training/inference definitions are inconsistent, and as usual nothing reports an error.**

## Difference 2: different order

```
Repo:  vision_ops · bounding_box · sam2×2 · depth×2 · grasp · roborefer×2 · vlm×2
P4  :  vision_ops · bounding_box · sam2×2 · grasp · vlm×2 · depth×2 · roborefer×2
```

Same set, same content (except vlm), only the arrangement differs — most likely reflecting the readiness order
of the actors at the moment of export. **But it still changes the bytes of the system prompt, so it still counts as inconsistent.**

---

## One piece of good news: **the wrapper text matches exactly**

The head and tail of the `# Tools` section built by `run_sft.sh` are **verbatim identical**
to what the verl chat template renders (checked against 400 characters before and after in the dump's `input`):

```
\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n
\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>
… schemas …
</tools>\n\nFor each function call, return a json object with function name and arguments
within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": …, "arguments": …}\n</tool_call>
```

**So the assembly logic of `run_sft.sh` is fine; the only stale thing is the schema list in that YAML.**
Replacing the YAML is enough; not one line of the script needs to change.

---

## Resolution

Use **`toolshed_v1_config.p4.yaml`** in this directory:

```bash
VERSION=v1 \
TOOL_CONFIG=/path/to/toolshed_v1_config.p4.yaml \
    bash scripts/spacetools/run_sft.sh
```

It was **reverse-engineered from the `input` of the P4 dump** and verified: after rebuilding it the way `run_sft.sh` does,
the `<tools>` block is byte-for-byte identical to P4 (8595 characters, sha `72c71c806f64e162`).
The full `# Tools` section is **8952 characters** — matching the "about 8.9 KB" recorded in the methods notes.

The format is the minimal form `run_sft.sh` needs (it only reads `item["tool_schema"]`):

```yaml
tools:
  - tool_schema: {...}
  - tool_schema: {...}
```

---

## How to re-check

```bash
python3 diff_tool_text.py toolshed_v1_config.repo.yaml toolshed_v1_config.p4.yaml
# exit code 1 + list of differences = reproduces the conclusion of this document
```

`diff_tool_text.py` replicates the construction logic of `run_sft.sh` verbatim and compares **the final string that goes into the prompt**,
not the YAML lines — the two YAMLs necessarily differ in indentation and key order, so a bare `diff` is all noise.

---

## Additional finding: **within P4's own 15 runs, there are three schema orders**

Extract the `<tools>` block from all 15 dumps and compare sha:

```
run1/blinkdepth      8595  72c71c806f64      run1/refplacement    8595  72c71c806f64
run1/bopgrasp        8595  72c71c806f64      run1/refunseen       8595  72c71c806f64
run1/boppose         8595  0c6f645a3ebe  ←   run1/robospatial     8595  72c71c806f64
run1/cvb2drelation   8595  0c6f645a3ebe  ←   run2/*  run3/*       8595  72c71c806f64
run1/cvb3ddepth      8595  72c71c806f64      run4/blinkdepth      8595  4f58f1391956  ←
run1/reflocation     8595  0c6f645a3ebe  ←
```

**Three shas, but the byte count is always 8595 and there are always 11 schemas — verified to be different permutations of the same set of schemas**;
not a single character of content changed. The only thing that changes is the relative position of `grasp_generator` / `vlm` / `depth_estimator`:

```
72c7… : … sam2×2 · grasp · vlm×2 · depth×2 · roborefer×2
0c6f… : … sam2×2 · vlm×2 · grasp · depth×2 · roborefer×2
4f58… : … sam2×2 · grasp · depth×2 · vlm×2 · roborefer×2
```

`vision_ops` / `bounding_box` / `sam2` are always first (fastest to load),
`roborefer` is always last (8B, 4 shards, slowest) —
**this is the actor readiness order, reshuffled every time Toolshed restarts.**

### Two consequences

**① Which order should SFT align with? — There is no "which one".**
The eval itself never fixed the order. So **aligning content is mandatory (the vlm lines), aligning order is impossible**.
Approach: use the content of `toolshed_v1_config.p4.yaml`, pick one order and **record it in PROVENANCE**,
stating it was taken from the majority (11 of 15).

**② This is a second source of run-to-run nondeterminism, not recorded before.**
P2/P4 attributed the three runs of the same config at 108/107/111 to "sglang batch composition changing the floating-point reduction order".
**But the tokens of the prompt itself also change** — the three permutations are three different token sequences,
with different attention patterns, enough to push low-confidence tokens to flip.

> Note that `run1/blinkdepth` (72c7…) and `run4/blinkdepth` (4f58…) have **different permutations** —
> and these two are exactly part of that 108 / 107 / 111 / … repeat experiment.
> **Previously attributing all of the spread to floating-point reduction may have overestimated that term.**
> This does not change any published conclusion (the spread width was measured), but it adds one more candidate explanation for "why there is spread",
> and one that is **controllable**: fix `TOOL_CONFIG` to a static YAML fed to eval, and this source disappears.

## Leftovers

- `toolshed_v2_config.yaml` (17 tools) was not checked; if v2 is run in the future this has to be redone.
- ② above is only a **possible** source; no controlled experiment was done to prove it actually affects scores.
  To prove it: fix `TOOL_CONFIG` and rerun blinkdepth three times, see whether the spread narrows. Cost about 3 × 5 minutes.
