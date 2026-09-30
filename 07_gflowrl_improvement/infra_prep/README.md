# infra_prep:训练侧 infra 优化(开机前,无 GPU)

2026-09-30。不改目标函数;默认值下与训练用的 SpaceTools-RL `c6fef78a` 行为逐位相同。

## 文件

| 文件 | 作用 |
|---|---|
| `patched/` | 打好补丁的 4 个文件:`ray_trainer.py`(= c6fef78a + P1-a + 丢退化组)、`fsdp_workers.py`、`run_rl.sh`、`run_rl_gflowrl.sh` |
| `infra_all_vs_c6fef78a.diff` | 上面 4 个文件相对 c6fef78a 的完整 diff,`git apply` 验证可用 |
| `infra_drop_only.diff` | 只有「丢退化组」这一项(相对 P1-a) |
| `infra_drop_check.py` | 丢退化组的 CPU 自检(需 torch) |
| `stage_timing.py` | 从训练日志抽全部 `timing_s/*`,找出那 129 s |
| `nccl_bw_probe.py` + `preflight_nccl.sh` | 开机验收:P2P 能不能用、all-reduce 带宽多少 |

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
代价约 2 GB/卡(4.07 B 参数 bf16 分到 4 卡)。收益要等 `stage_timing.py` 读出 old_log_prob(同样的前向、参数在 GPU)
和 ref 的差才能估,上机先跑 1 步确认不 OOM。

## 3. 那 129 s

verl 本来就给每个阶段计时(old_log_prob / reward / update_weights / dump_rollout_generations …),
报告只抄了四项。完整日志在 HF `provenance/`:

```bash
hf download qzpm55555/spacetools-p7-gflowrl-cprime-8xa40 --include "provenance/*" --local-dir p7_hf
python3 stage_timing.py p7_hf/provenance/<日志文件>
```

## 4. 开机验收

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash preflight_nccl.sh     # 训练用的那几张卡
```
P2P_OK 才不设 `NCCL_P2P_DISABLE`;P2P_BROKEN 时考虑换机器。

## 自检结果

- `infra_drop_check.py`:退化组 0/5/11/16 个四种情形,保留行梯度与 P1-a 全量完全一致;不按 micro 补齐的反例确实不一致。PASS
- `../p1_prep/p1a_check.py` 指向 `patched/ray_trainer.py`:PASS
- `04_gflowrl_implementation/checks/run_checks.sh` 指向打了补丁的完整树:config / guard / gpusplit / onpolicy / fixedpoint / degenerate 全部 PASS
- `infra_all_vs_c6fef78a.diff` 在干净的 c6fef78a 上 `git apply` 后与 `patched/` 逐字节相同

没做:checkpoint 瘦身。每 5 步存一次、每次 39 s,摊到每步约 8 s(<1%),而且去掉 optimizer 状态会让自动续训失效。
