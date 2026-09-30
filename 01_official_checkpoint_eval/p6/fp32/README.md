# 实验 F:fp32 对照(去掉 model_dtype=bf16)

    blinkdepth/           run7  107/124   <- 最先跑的一次
    run9/blinkdepth/      run9  105/124
    run10/blinkdepth/     run10 105/124

三次同配置(fp32 + gpu_memory_utilization=0.25)。对照臂 bf16 同配置在
`p6/gmu025/run{5,6}/blinkdepth/`,得 108 / 109。

**只看 run7 会得到错误结论。** 曾据它单次的 107「落在 107–109 带内」
判定偏离 [20] 已端到端排除,补跑 run9/run10 后收回。
两个范围不重叠(fp32 105–107,bf16 108–109),但样本量不足以下反向结论:
p≈0.1,结构上只有 1 个样本稳定不同。详见 `records/P6_GPU_RESULTS.md` §4。
