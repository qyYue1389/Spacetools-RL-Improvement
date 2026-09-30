# P7 开机的迁移验收(2026-09-02)

`run12` = 4×A100-SXM4-40GB 上的 `blinkdepth` 全量,**未打诊断开关**
(`dump_token_diagnostics` 默认关,字段列表与 P4 逐字相同 —— 见 `records/P7_GPU_RESULTS.md` §1.2)。

判读按新口径:**不看**旧文档那个来自 n=3 的 107–109 窄带,看**稳定上界与硬核**。

    单次    107/124 = 86.29%     bf16 五次实际范围 106–109  -> 落带内
    硬核    12 个 [20,22,37,43,55,61,76,84,93,94,114,119]  与文档记载逐个对上
    回归    每个 baseline 都对、run12 却错的样本  0 个

⚠ 四个 baseline 合并的上界是 111/124、硬核 13;**加进 run12 后变成 112/12**。
上界又升一格 —— 未见饱和,**不能当不变量用**(`P7_HANDOFF.md` §3.3 第 1 条)。
验收依据是「单次落带 + 硬核对上 + 零回归」,不是上界命中论文值。

复算:`python3 tools/p7/p7_accept_blinkdepth.py p7/accept/run12/blinkdepth/0.jsonl \
    p4/dumps/run1/blinkdepth/0.jsonl p4/dumps/run4/blinkdepth/0.jsonl \
    p6/gmu025/run5/blinkdepth/0.jsonl p6/gmu025/run6/blinkdepth/0.jsonl`
