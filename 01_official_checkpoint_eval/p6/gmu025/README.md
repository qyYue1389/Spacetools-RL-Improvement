# Runs under `gpu_memory_utilization=0.25`

    run5/blinkdepth/    108/124   evidence for deviation [23]
    run6/blinkdepth/    109/124   evidence for deviation [23]
    run3/robospatial/   222/350   third draw of experiment G
    run8/robospatial/   231/350   fourth draw of experiment G

The directory is named after the **value** of `gpu_memory_utilization`, but **comparability is judged by the size of the KV pool,
not by this value** — it is a static fraction of the whole GPU (see deviation `[23]` in `records/PROVENANCE.txt`).

## The two robospatial runs have the same config as the P4 runs

    run1 / run2   40 GB GPU × 0.5  = 20 GB pool   (original P4 machine)
    run3 / run8   80 GB GPU × 0.25 = 20 GB pool   (this directory)

**All four runs have a 20 GB KV pool**, so they are four draws under the same config, and
`records/P6_GPU_RESULTS.md` §5b is justified in merging the four into one interval
(VQA 70.61–74.12%, mean 72.92%).

> In other words: the number `0.25` looks like a deviation, but it is actually the value required **to reproduce
> P4's pool size on an 80 GB GPU**. Using 0.5 is what would really change the config.

## The two blinkdepth runs, on the other hand, really are a control group

    run4 (in p4/dumps/)  80 GB × 0.5  = 40 GB pool   <- differs from P4
    run5 / run6 (this directory) 80 GB × 0.25 = 20 GB pool   <- same as P4

This group is used to show how pool size affects the score (106 vs 108/109),
and is the direct evidence for deviation `[23]`.
