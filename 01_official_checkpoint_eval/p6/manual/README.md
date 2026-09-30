# P6 人工归类

| 内容 | 说明 |
|---|---|
| `pending31.json` | 判据 A/B/C 之后仍需看图的 31 条(生成:`tools/p6/p6_relations.py` 的 `--dump-cases` 加上「检测对不上」与零头) |
| `verdicts.jsonl` | 32 条逐条结论与**证据**(生成:`tools/p6/verdicts.py`) |
| `cards/` | 32 张审阅卡:原图 + 工具实际返回的点 + 题面/GT/模型答案/判据/间距 |

卡片可重新生成,需要 `eval-benchmarks` 的 parquet 与 `pandas`/`pyarrow`/`PIL`:

    python3 tools/p6/make_cards.py --data <路径>/eval-benchmarks/data

## 两件必须说明的

**只有一轮标注,κ 算不出来。** 这 31 条是单标注(无第二标注者),
所以**没有标注一致性度量**。补救是把逐条证据公开在 `verdicts.jsonl` 里
——每条都写了「凭什么」——让别人可以复核结论而不必信任标注者。
若要报 κ,需请第二人独立标同一批。

**卡片上的点是工具**实际**返回的点**,不是 GT,也不是模型答案。
判「工具错」看的就是这个点落在哪个物体上。
