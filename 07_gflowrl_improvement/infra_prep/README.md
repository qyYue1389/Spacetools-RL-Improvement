# infra_prep:训练侧 infra 优化(开机前,无 GPU)

2026-09-30。不改目标函数;默认值下与训练用的 SpaceTools-RL `c6fef78a` 行为逐位相同。

## 文件

| 文件 | 作用 |
|---|---|
| `patched/` | 打好补丁的 5 个文件:`ray_trainer.py`(= c6fef78a + P1-a + 丢退化组 + dump 不存图开关)、`fsdp_workers.py`、`tool_agent_loop.py`、`run_rl.sh`、`run_rl_gflowrl.sh` |
| `patched_toolshed/verl.py` | Toolshed 侧计时补丁(SpaceTools-Toolshed `712e557`) |
| `infra_all_vs_c6fef78a.diff` | `patched/` 5 个文件相对 c6fef78a 的完整 diff,`git apply` 验证可用 |
| `toolshed_tool_timing_vs_712e557.diff` | Toolshed 侧计时补丁的 diff,`git apply` 验证可用 |
| `infra_drop_only.diff` | 只有「丢退化组」这一项(相对 P1-a) |
| `infra_drop_check.py` | 丢退化组的 CPU 自检(需 torch) |
| `stage_timing.py` | 从训练日志抽全部 `timing_s/*`,找出那 129 s |
| `nccl_bw_probe.py` + `preflight_nccl.sh` | 开机验收:P2P 能不能用、all-reduce 带宽多少 |
| `dump_images_check.py` | dump 不存图开关的 CPU 自检 |
| `check_tool_packing.py` | 开机验收:工具 actor 按卡排不排得下、份额够不够显存(读 run 脚本里的 `TOOL_CONFIGS`,不用 GPU) |
| `tool_timing_check.py` / `tool_timing_summary.py` | 按工具计时的自检 / 汇总 |
| `stage_timing_p7_85steps.csv` | P7 训练 85 步的逐阶段耗时 |

## 1. 丢退化组(`GF_DROP_DEGEN=true`,需同时 `GF_FILTER_DEGEN=true`)

拿到奖励后、old_log_prob 之前,把奖励全同的组整组删掉,old_log_prob / ref / update_actor 只算剩下的行。
退化组中位 70%,这三段(约 369 + 122 + 未记账的一部分 s)按行数缩小。

为什么更新不变:dp_actor 每个 micro-batch 的 loss 乘 `1/gradient_accumulation`,而它 = 配置的
`ppo_mini_batch_size / micro`(每卡 80/2),与实际行数无关 → 每卡梯度 = Σ本卡行的 loss / 80,
FSDP 跨卡取平均 → Σ全部 / 320,与 P1-a(只置零 g~、分母保留全部行)相同。
两个约束:行数补齐到 dp×micro 的倍数(否则会出现 1 行的 micro-batch,那一行权重翻倍,自检第 5 节有反例);
补齐用最短的退化行(filter 下 g~=0,不贡献梯度)。

日志:`reward/degenerate_*` 仍是全批的值;新增 `gflowrl_drop/{rows_before,rows_after,pad_rows,kept_rollout_frac,score_mean_full}`。
**注意** `critic/score/mean`、`perf/total_num_tokens`、`perf/throughput` 等变成只算保留行,跨运行比较用 `gflowrl_drop/score_mean_full`。
rollout dump(`trainer.rollout_data_dir`)仍写全部 320 条,`p1_monitor.py` 的读数不受影响。

## 2. ref 模型留在 GPU(`REF_PARAM_OFFLOAD=False`)

上游 `fsdp_workers.py` 对 ref **强制** `CPUOffload(offload_params=True)`,与 `ref.fsdp_config.param_offload` 无关 ——
原来把这个开关设成 False 也不会让 ref 上 GPU。补丁让 False 真正生效;默认 True 行为不变。
代价约 2 GB/卡(4.07 B 参数 bf16 分到 4 卡)。实测(§3)只省约 5 s/步(<1%),**不建议打开**;补丁保留,默认关。

## 3. 那 129 s —— 已查清(2026-09-30,读 HF `provenance/full_train.log.gz`)

`python3 stage_timing.py <full_train.log>`,85 步中位(秒):

```
gen            321.2   34.8%
old_log_prob   111.9   12.1%   ← 报告漏记
ref            116.8   12.7%
update_actor   351.8   38.1%
save_ckpt       39.6    4.3%   (只在 save_freq 命中的步)
update_weights   9.7    1.0%   ← 报告漏记
其余             0.3
step           922
```

step 85 那 981 s 里的 129 s = old_log_prob 118.8 + update_weights 10.0 + 0.3。逐步结果在 `stage_timing_p7_85steps.csv`。

由此得到的三个结论:

1. **随行数缩放的部分是 old_log_prob + ref + update_actor ≈ 580 s(63%)**,比之前估的大 —— 丢退化组(§1)能砍的就是这一块。
2. **ref 放 CPU 只多花约 5 s/步**(ref 117 vs 同样是一次前向、参数在 GPU 的 old_log_prob 112)。§2 的开关收益 < 1%,还要多占 2 GB/卡,**不建议打开**;补丁保留,默认关。
3. **gen 的 321 s 主要在等工具,不在生成。** 最慢那条轨迹工具耗时 281 s、生成只有 26 s;平均每条工具 135 s、生成 20 s。
   rollout 是被工具服务卡住的(工具 actor 为塞进 4 张卡缩到了 ×0.5,roborefer 只有 3 个副本)。
   丢退化组帮不了这一段(奖励要等生成完才知道)。下一步要查的是哪个工具在排队。

## 4. 按工具计时(`TOOL_TIMING_DIR=<目录>`)

§3 发现 gen 的 321 s 主要在等工具,但 verl 只记每轮工具的总时间。设了 `TOOL_TIMING_DIR` 后,
每次工具调用往 `$TOOL_TIMING_DIR/tool_calls_<pid>.jsonl` 写一行:工具名、提交时刻、总耗时,以及拆开的三段 ——

- `exec_wait`:在 agent loop 进程的默认线程池里排队(Toolshed 用 `run_in_executor(None, ...)` 同步调用,线程数有上限)
- `remote`:Toolshed 调用本身(router + 工具 actor 排队 + 计算)
- 剩下的是回到事件循环后的处理(`ray.put` 图片/变量等)

外加提交时该工具和全部工具的在途调用数、线程池大小、是否出错。不设这个变量时行为与原来完全一样。
`remote` 和 `exec_wait` 需要 Toolshed 侧补丁(`patched_toolshed/`);没打时只有总耗时。

```bash
export TOOL_TIMING_DIR=$OUTPUT_DIR/tool_timing        # 在 run_rl_gflowrl.sh 之前,随 ray start 传给 worker
bash examples/toolshed/run_rl_gflowrl.sh ...
python3 tool_timing_summary.py $OUTPUT_DIR/tool_timing
```

怎么读:`remote` p50 远大于 min → 工具 actor 在排队,加副本 / 调 GPU 分配;p50 ≈ min → 算得慢,
要批处理或换卡;`exec_wait` 大且在途数经常超过线程数 → 瓶颈是 agent loop 的线程池,不是工具。

## 5. 开机验收

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash preflight_nccl.sh     # 训练用的那几张卡
```
P2P_OK 才不设 `NCCL_P2P_DISABLE`;P2P_BROKEN 时考虑换机器。

```bash
python3 check_tool_packing.py examples/toolshed/run_rl.sh 4 --card-gib 46   # TOOL_GPUS=4,单卡 46 GiB
```
改工具 actor 数或 `num_gpus` 之后、开训之前跑。`run_rl.sh` 自己只断言份额合计 ≤ TOOL_GPUS,而且要等训练启动才触发;
这里多查两件事:每张卡上的份额加起来 ≤ 1.0(排不进任何一张卡的 actor 会一直等),以及份额 × 单卡显存 ≥ 实测峰值
(目前只有 vlm 有实测值 32.11 GiB,其余用 `--peak-gib 工具=GiB` 补)。工具表不在脚本里另抄一份,而是把 run 脚本的
`TOOL_CONFIGS` 段(含自动缩放)切出来执行。
PACK_FAIL 不要开训。PACK_ORDER_DEPENDENT 不拦:论文的 2 节点配置(7.8 / 8)和 P7 的配置(3.7 / 4)都是这个结果,
意思是没有余量,换一种创建顺序会有 actor 排不进去;开训后某个工具一直不响应时先查这里。

## 6. rollout dump 不存图片(`+trainer.dump_images=false`)

P1 的监控要开 `trainer.rollout_data_dir`,而上游的 `_dump_generations` 除了写 JSONL,还会把每条样本的图另存成 PNG
(每步一个 `images_<step>/`,320 条样本有几张图就存几张)。`p1_monitor.py` 只读 JSONL,这些图用不上。
加了一个配置项:`trainer.dump_images`,默认 True(行为与原来完全一样);设成 false 时跳过存图,JSONL 照写。
它不在上游的配置文件里,所以命令行要带 `+`:

```bash
GF_FILTER_DEGEN=true bash examples/toolshed/run_rl_gflowrl.sh \
    trainer.rollout_data_dir=$OUT/rollouts +trainer.dump_images=false
```

训练中验证的 dump(`validation_data_dir`)走同一个函数,同一个开关一起生效。eval 脚本不传这个开关,行为不变。

## 自检结果

- `infra_drop_check.py`:退化组 0/5/11/16 个四种情形,保留行梯度与 P1-a 全量完全一致;不按 micro 补齐的反例确实不一致。PASS
- `../p1_prep/p1a_check.py` 指向 `patched/ray_trainer.py`:PASS
- `04_gflowrl_implementation/checks/run_checks.sh` 指向打了补丁的完整树:config / guard / gpusplit / onpolicy / fixedpoint / degenerate 全部 PASS
- `infra_all_vs_c6fef78a.diff` 在干净的 c6fef78a 上 `git apply` 后与 `patched/` 逐字节相同

- `check_tool_packing.py --selftest`:合计超限、合计不超但单卡排不下、P7 配置(顺序相关)、vlm 份额 0.6 / 0.7 对 46 GiB 卡,判定都符合预期;对 `c6fef78a` 的 `run_rl.sh` 取 TOOL_GPUS=4 复现 3.70 与 P7 的 actor 数。PASS
- `dump_images_check.py`:不设 / 设 true 时存图与原来相同,设 false 时不建 `images_<step>/`、JSONL 逐行相同;同一脚本指向没有开关的 `../p1_prep/patched/ray_trainer.py` 会失败。PASS
- `tool_timing_check.py`:不设变量时返回值与原函数完全一致;设了之后每次调用一条记录,`remote` / `exec_wait` 拆分正确,线程数不够时 `exec_wait` 变大,Toolshed 报错与未知工具都记为失败。PASS

没做:checkpoint 瘦身。每 5 步存一次、每次 39 s,摊到每步约 8 s(<1%),而且去掉 optimizer 状态会让自动续训失效。
