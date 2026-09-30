# P3: trajectory capture

## What P3 turned out to be

The plan assumed P3 meant instrumenting `ToolAgentLoop` with hooks to record
every turn. Reading verl first made most of that unnecessary.

`_dump_generations` (`verl/trainer/ppo/ray_trainer.py:390`) already writes one
JSONL row per sample under `trainer.validation_data_dir`, carrying `input`,
`output`, `gts`, `score`, `step`, `messages`, and the `image` / `answer` /
`index` fields from `extra_info`. It also has a branch that saves images from
`multi_modal_data` as PNGs -- which **never fires on this eval path**, because
`multi_modal_data` does not reach `_dump_generations` here. That is fine and no
patch is wanted; see the conformance section below.

The decisive detail is line 601:

    # Use skip_special_tokens=False to preserve tool call tokens in the output
    output_texts = [self.tokenizer.decode(ids, skip_special_tokens=False) ...]

Because the response is decoded with special tokens preserved, and because verl
multi-turn keeps tool results inside the response token sequence, **the whole
transcript is already in the `output` string** -- every `<think>`,
`<tool_call>`, `<tool_response>` and `<answer>`, in order. Nothing needs to be
hooked.

So P3 is one small patch plus one offline script.

## The patch: `patches/rl/0008`

Two fields the validation loop already collects were never passed to the dump:

| field          | where collected                   | why we want it |
|----------------|-----------------------------------|----------------|
| `sample_uids`  | `ray_trainer.py:614` (`extend`)    | stable identity |
| `sample_turns` | `ray_trainer.py:655` (`append`)    | cross-check     |

The turn count is the one that matters. It lets the parser compare its own
transcript decomposition against verl's count, so a parser bug shows up as a
loud mismatch instead of quietly wrong statistics. `parse_dump.py` reports
`turn-count mismatch` and `--strict` treats it as contamination.

`sample_turns` is appended per batch rather than extended, so it arrives as a
list of arrays and is flattened before use. Both additions sit inside
`try/except`: this is diagnostics and must never be able to break an eval.

The parser does not require the patch -- it derives the turn count from the
transcript either way. Applying it only buys the cross-check.

## The script: `tools/parse_dump.py`

Stdlib only, so it runs in any of the five environments. It has two jobs.

### 1. P4 guard -- run it after every benchmark

    python tools/parse_dump.py "$EVAL_DIR/blinkdepth/0.jsonl" --strict

An out-of-memory condition **does not crash the eval**. The tool returns an
error string, the model reads it as a tool response and carries on, and the
benchmark reports a plausible number that is quietly wrong. One P2 run OOMed 68
times and lost 5 points that way. `--strict` exits 1 on OOM, malformed tool
calls, or a turn-count mismatch.

Treat a non-zero exit as "this benchmark has no result yet", not as a warning.

### 2. P6 input

    python tools/parse_dump.py "$EVAL_DIR"/*/0.jsonl --emit parsed/

Writes enriched records with the trajectory decomposed and every failure mode
flagged, so error classification never has to re-read a log or touch a GPU
again. Add `--emit-slim` to drop the full trajectory when only the flags are
wanted.

### Derived fields

| field                     | how |
|---------------------------|-----|
| `trajectory`              | split on `<\|im_start\|>`, then per-turn tag extraction |
| `num_turns_derived`       | count of assistant turns |
| `tool_stats`              | tool call names, grouped by top-level tool |
| `tool_failures`           | tool responses matching a failure pattern |
| `oom`                     | `OutOfMemoryError` / `CUDA out of memory` |
| `truncated_tool_response` | literal `...(truncated)` -- what `tool_agent_loop.py:515` inserts in middle mode |
| `hit_max_turns`           | `num_turns >= 8` (run_eval.sh's `max_assistant_turns`) |
| `parse_ok`                | an `<answer>` block exists |
| `truncated_generation`    | last assistant turn has neither a tool call nor an answer |
| `chain_signature`         | e.g. `depth_estimatorx1+roboreferx2+vision_opsx2@3t`, for grouping identical tool-use patterns |
| `num_turns_verl_convention` | `assistant + user + 1`, the only form comparable to verl's dumped `num_turns` |
| `vars` / `arg_vars`       | `$variables` a tool response exposes / a tool call reads back |
| `vars_exposed` / `vars_used` / `vars_unused` | per-sample roll-up; `vars_unused` is the lead for taxonomy 2d |
| `vars_phantom` | referenced but never published by any tool -- the model invented the name. Taxonomy 2c, and a cleaner signal than `vars_unused` |
| `question`                | the question alone, extracted from the ~10 KB rendered prompt |

OOM is classified before tool failure, so a response that OOMed is not also
counted as the tool declining the job. They are different things: one is run
contamination, the other is signal for P6.

### Reporting choices

`--compare` takes repeat runs of one benchmark and prints the band rather than
a point value, because P2 measured blinkdepth at 108 / 107 / 111 correct under
an identical configuration. A difference smaller than the band is not a
difference.

The wrong-sample breakdown separates failures with an infrastructure cause
(tool error, truncation, turn exhaustion, missing answer, OOM) from `clean`
ones -- wrong despite everything working. The clean ones are where the real P6
work is: genuine tool inaccuracy or genuine reasoning error.

`-v` adds the grasp success/failure split, which is the cheapest check that
failure detection works.

**Do not use P2's 27 succeeded / 33 errored as the expected value -- it is
wrong.** The real split is **19 / 41**, confirmed by a raw text search of the
dump independently of the parser: 33 samples hit `No collision-free grasps` and
a further 8 hit `Top-down filtering removed all`, which P2 scored as successes.
See `CHANGES.md` §7.3 and the correction block at the top of `P2_RESULTS.md`.

## Validating the parser

`P2_RESULTS.md` already contains hand-computed statistics for the blinkdepth
run4 dump: 16 wrong, zero truncated tool responses, zero max-turn exhaustions,
one tool error, every sample produced an `<answer>`, and 12 of the wrong ones
used exactly `depth x1, roborefer x2, vision_ops x2` in 2 turns.

Run the parser on that dump and those numbers should come back. They were
derived independently, so they are a free regression test -- use them before
trusting the parser on anything new.

    python tools/parse_dump.py <run4>/blinkdepth/0.jsonl -v

Expected, from `P2_RESULTS.md`:

    score >= 0.50        108/124 = 87.10%
    OOM samples          0
    truncated responses  0
    hit max turns        0
    no <answer>          0
    tool failures        1 sample
    wrong samples        16
      chains             12 x depth_estimatorx1+roboreferx2+vision_opsx2@3t

**The count is what matters, not the `@Nt` suffix.** P2's hand analysis wrote
`@2t` because it counted turns that *called a tool*; the parser counts assistant
turns, and the third is the one that emits `<answer>` and calls nothing. Same
twelve samples either way.

If the chain signature count differs, the transcript splitter is wrong before
anything else is worth reading -- which is exactly what happened when this was
first run in Phase V1, and it took three bugs with it. See `CHANGES.md` §7.

## Conformance against the plan's P3 section

Re-checked after Phase V, field by field against the plan artifact.

### The plan's field table

| plan's field | status |
|--------------|--------|
| `trajectory` (think / tool_calls / tool_results) | done |
| `tool_stats` / `tool_errors` | done (`tool_stats`, `tool_failures`) |
| `num_turns` / `hit_max_turns` / `parse_ok` | done, three turn conventions kept apart |
| `truncated_tool_response` | done, on the literal marker rather than a length heuristic |
| `latency_ms` | **not built** -- see below |
| `image_path` | **closed, and not needed** -- see below |

### `image_path`: closed by measurement, not by a patch

The plan flags `image` as null and calls for "a small patch". It is null on
**seven of the eight benchmarks measured so far**, and the reason is not a bug:
`multi_modal_data` never reaches `_dump_generations` on this eval path, so the
PNG-saving branch at `ray_trainer.py:432` never fires. `multi_modal_data`
appears zero times in a full run's log.

`boppose` is the exception, and it is a worse kind of non-null: it carries a
string like `images/scene_000001_frame_000000.png`, copied from the parquet's
`extra_info`. **No such file exists** -- it is a dangling relative path into the
original BOP dataset layout. Trusting it would be worse than the null.

Patching it would be the wrong fix anyway. **`sample_id` is the parquet row
index**, verified 124/124 against blinkdepth's ground truth, unique and
contiguous. So the original image is recovered exactly as

    parquet[sample_id]["images"][0]["image"]     # base64 data URI

which is free, lossless, and avoids writing a second copy of 292 MB of images
next to every dump. P6's prediction-vs-GT overlays have everything they need.

One caveat worth carrying: subsets from `make_subset.py` use `head(N)`, so row
*i* of a subset is row *i* of the full file and `sample_id` means the same thing
against either. **That equivalence would break the moment anyone switches the
subset to random sampling.**

### `pred`: deferred to P5/P6, and nothing is lost by deferring

The plan's schema separates `raw_answer` (the text inside `<answer>`) from
`pred` (a structured prediction). Only `raw_answer` is emitted, deliberately.

`gt` and `raw_answer` come back in the **same format** in every benchmark
family, so one parser handles both sides symmetrically:

    blinkdepth    gt "B"                              raw_answer "B"
    robospatial   gt "[(0.383, 0.873), ...]"          raw_answer "[(0.338, 0.934)]"
    bopgrasp      gt "Grasp center: [...], Left ..."   raw_answer "Grasp center: [...], ..."

So `pred` is pure offline text parsing over fields the record already carries --
it needs nothing captured at eval time. Writing it now would mean guessing the
answer formats of the six benchmarks whose dumps do not exist yet. It belongs
with the P5/P6 analysis, off this machine.

### The plan's seven P3 tasks

| # | task | status |
|---|------|--------|
| 1 | hook `ToolAgentLoop` for per-turn think / tool_call / tool_response | **superseded** -- verl already dumps the whole transcript (`skip_special_tokens=False`); no hook needed |
| 2 | store only summaries of large variables (shape, range, mean) | **satisfied upstream** -- Toolshed never inlines arrays; a response reads "Use `$depth_map` (numpy array, 364x504) ... Depth range: 3.09m to 1094.68m (mean: 20.95m)". The summary *is* the response. The parser now also records the variable names on both sides (below) |
| 3 | record tool errors explicitly | done (`tool_failures`, with OOM classified separately) |
| 4 | flag responses truncated at `max_tool_response_length=2048` | done |
| 5 | pose / grasp prediction-vs-GT overlays | **deferred to P6**, and now demonstrably feasible: `sample_id` -> parquet -> image |
| 6 | sample-level resume by `sample_id` | **not built** -- verl writes one `{step}.jsonl` per benchmark rather than appending per sample, so resume belongs in the shell loop (skip a benchmark whose dump exists), not in the eval |
| 7 | validate the dump is fully parseable on P2 samples before P4 | **done, and it earned its keep** -- it caught three real bugs; see `CHANGES.md` §7 |

### Added while checking task 2: variable-reuse tracking

Task 2 turned out to be satisfied by Toolshed's design, but checking it surfaced
something the plan asks for in its schema (`"vars": [...]`) and that P6 needs
directly. Toolshed hands large results back **by reference**:

    Use $depth_map (numpy array, 364x504) to reference the depth data
    and $focal_length_px (float) to reference the focal length.

and later calls read them back:

    vision_ops.index_at  arguments {"data": "$depth_map", "u": 0.443, ...}

Both sides are now recorded -- `vars` per tool response, `arg_vars` per tool
call, and rolled up per sample as `vars_exposed` / `vars_used` / `vars_unused`.

`vars_unused` is the point: a variable a tool published and the model never read
back is the signature of **taxonomy 2d, "recomputed what it already had"**.
It is a lead and not a verdict -- on blinkdepth 106 of 124 samples leave exactly
one variable unused, and it is always `focal_length_px`, which that benchmark
genuinely does not need. The samples worth looking at are the tail: the run
measured 11 samples with three unused and one with four, which is where a
roborefer detection was published and then abandoned in favour of a vlm retry.

### Also added

`question` (extracted from the rendered prompt rather than carrying all ~10 KB
of it -- that would add ~21 MB across the nine benchmarks for text that is
identical in every row bar the last few lines) and `reward` passthrough.

## Not built, deliberately

**Per-tool `latency_ms`.** It would require instrumenting Toolshed, and P2
already showed the bottleneck is policy generation, not the tools: per-sample
cost stayed within 3.3-3.7 s while turn count varied 2.5x and pixel count 16x.

**Sample-level resume.** verl writes one `{global_steps}.jsonl` per benchmark
rather than appending per sample, so resume belongs in the shell loop -- skip a
benchmark whose dump already exists -- not in the eval.

**Pose / grasp prediction-vs-GT overlays.** Worth having for P6 (the paper's
Fig. 10-12 style), but they are rendered from the parquet ground truth plus the
parsed prediction, entirely decoupled from the eval. Deferring them does not
block P4.
