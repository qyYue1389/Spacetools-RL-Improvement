# Experiment F: fp32 control (drop model_dtype=bf16)

    blinkdepth/           run7  107/124   <- the first one run
    run9/blinkdepth/      run9  105/124
    run10/blinkdepth/     run10 105/124

Three runs with the same config (fp32 + gpu_memory_utilization=0.25). The control arm, bf16 with the same config, is in
`p6/gmu025/run{5,6}/blinkdepth/`, scoring 108 / 109.

**Looking at run7 alone leads to the wrong conclusion.** Based on its single 107 ("falls inside the 107–109 band")
deviation [20] was judged ruled out end to end; after the extra run9/run10 runs, that was retracted.
The two ranges do not overlap (fp32 105–107, bf16 108–109), but the sample size is too small to draw the opposite conclusion:
p≈0.1, and structurally only 1 sample is stably different. See `records/P6_GPU_RESULTS.md` §4 for details.
