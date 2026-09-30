# `gpu_memory_utilization=0.25` 下的运行

    run5/blinkdepth/    108/124   偏离 [23] 的证据
    run6/blinkdepth/    109/124   偏离 [23] 的证据
    run3/robospatial/   222/350   实验 G 的第三次抽样
    run8/robospatial/   231/350   实验 G 的第四次抽样

目录按 `gpu_memory_utilization` 的**取值**分,但**判断可比性要看 KV 池的大小,
不是看这个取值**——它是占整卡的静态比例(见 `records/PROVENANCE.txt` 偏离 `[23]`)。

## 两个 robospatial 与 P4 的运行是同一配置

    run1 / run2   40 GB 卡 × 0.5  = 20 GB 池   (P4 原机)
    run3 / run8   80 GB 卡 × 0.25 = 20 GB 池   (本目录)

**四次的 KV 池都是 20 GB**,所以它们是同一配置下的四次抽样,
`records/P6_GPU_RESULTS.md` §5b 把四次合并成一个区间
(VQA 70.61–74.12%,均值 72.92%)是有依据的。

> 换句话说:`0.25` 这个数字看起来像偏离,实际是**为了在 80 GB 卡上复现
> P4 的池子大小**而必须取的值。用 0.5 才是真正改变了配置。

## 两个 blinkdepth 则确实是对照组

    run4(在 p4/dumps/)  80 GB × 0.5  = 40 GB 池   <- 与 P4 不同
    run5 / run6(本目录) 80 GB × 0.25 = 20 GB 池   <- 与 P4 相同

这一组是用来显示池子大小如何影响分数的(106 对 108/109),
是偏离 `[23]` 的直接证据。
