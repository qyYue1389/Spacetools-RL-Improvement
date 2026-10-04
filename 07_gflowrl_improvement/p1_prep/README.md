# P1_prep:P1 / P2 / P3 开机前(无 GPU)的准备与结论

2026-09-25。对应 Design Doc 各节里标 ✅ 的条目。

## 文件

| 文件 | 作用 |
|---|---|
| `p1a_ray_trainer.diff` | P1-a 补丁(基于训练用的 SpaceTools-RL `c6fef78a`;2026-09-30 从 9-14 的旧底重做,旧版缺 β 分解日志):`compute_gflowrl_flow_gap(filter_degenerate=...)` 把组内奖励极差为 0 的组的 g̃ 置 0;driver 守卫(filter 打开时要求 `ppo_mini_batch_size == train_batch_size` 且 `ppo_epochs == 1`);新日志 `gflowrl/kept_groups`、`kept_rollout_frac`、`clip_saturation_kept`、`g_abs_p50/p90/p99` |
| `p1a_run_rl_gflowrl.diff` | 新开关 `GF_FILTER_DEGEN`(默认 false) |
| `patched/` | 打好补丁的两个文件(未动 SpaceTools-RL 仓库本身) |
| `p1a_check.py` | 自检,CPU 上跑,需 torch:`python p1a_check.py patched/ray_trainer.py <原 ray_trainer.py> <core_algos.py>` |
| `p2_query_type.py` | P2 开机前调查:三臂在 Vacant 上给 roborefer 的查询写法(只问锚物体 / 带位置描述)与各自的命中、改点、正确率 |
| `robospatial_vacant.parquet` | 训练中验证集:robospatial eval 里的 Vacant 122 题(与原 parquet 同 schema)。`data.val_files=<它> trainer.test_freq=10 trainer.val_before_train=True` |
| `p1_monitor.py` | 训练中监控:只问锚物体的查询占比(主判据)、改点率(± 按 prompt bootstrap SE)、0.05 网格、透传 / 改点正确率、工具点命中率;`--upper` 强制透传 |

## 自检结果(p1a_check.py 全 PASS)

1. filter 关闭时 advantages 与原函数逐位一致。
2. filter 打开时,退化组 g̃ = 0,其余行不变;kept_groups / kept_rollout_frac 正确。
3. on-policy(dp_actor 令 `old_log_prob = log_prob.detach()`)下,梯度与「真正的 loss mask、分母保留全部序列」逐位相等(max|diff| = 0)。
4. 反例:off-policy 时退化组会收到近端项梯度 → 守卫必要。
5. 另外 `04_gflowrl_implementation/checks/run_checks.sh` 指向打了补丁的完整树(2026-09-30 重做后),六个自检全部 PASS。

## 训练时怎么开

```bash
GF_FILTER_DEGEN=true bash examples/toolshed/run_rl_gflowrl.sh \
    trainer.rollout_data_dir=$OUT/rollouts            # 不开就读不到改点率
# P1-c B 臂:再加  actor_rollout_ref.actor.grad_clip=<L̄>   (AdamW 下 ≡ loss×1/L̄)
# P1-c C 臂:GF_EPS_LOW=0.28 GF_EPS_HIGH=0.20
python p1_monitor.py --window 10 $OUT/rollouts
python p1_monitor.py --window 1 $OUT/val_outputs   # 训练中验证:每次 122 题的只问物体题数
```

注意:`_dump_generations` 每步会把 320 条样本的图存成 PNG。开关已加在 `../infra_prep/patched/ray_trainer.py`(本目录的 P1-a 版本没有):训练命令再带 `+trainer.dump_images=false` 就只写 JSONL,见 `../infra_prep/README.md` §6。

## 监控脚本的校准

在 P0 的 9 份 robospatial dump 上逐位复现 P0 表:透传 72.7 / 100.3 / 77.0,改点 49.3 / 21.7 / 44.0,网格 18.7 / 7.0 / 18.3,P4 改点正确率 27.7%。

## 关键发现

- **Vacant 差距的根在查询写法。** 只问锚物体("cup")时工具点命中 0–3%、只能改点;带位置描述时命中 56%。只问物体的题每次 SFT 42.0 · C′ 34.3 · GRPO 13.3 / 122,三臂在每一类内部几乎一样;P4 对、C′ 错的 13 题全是这个模式。SFT 数据里 RoboSpatial 这个模板 410 / 410 条示范只问物体。主判据阈值:20 步窗口下降 ≥ 12 pp。

- **P2 的根在工具调用质量。** 工具点(最后一次 roborefer 的第一个点)落在 GT 内:SFT 46.3、GRPO 60.3、C′ 49.3 / 122(三次均值;GRPO vs C′ 稳定样本 13 : 4,p = 0.049)。改点样本上工具点命中只有 10–19%,改后答案 19–28%。
- **强制透传不是上界:** Vacant SFT 50.7 → 46.3、GRPO 62.3 → 60.3、C′ 53.7 → 49.3。
- **P1-c:** 按 P7 grad_norm,L̄ = 334 时 9 / 85 步(10.6%)仍会被裁;250 → 21%,200 → 40%。
- **P3:** SFT 数据里 RoboSpatial yes/no 977 条,调 depth_estimator 的 0 条;depth_estimator 只在 RefSpatial depth(A/B 模板)和 bopask 出现。front/behind 29 题三次均值 SFT 17.3 / GRPO 19.0 / C′ 19.3,旧的 72.4% → 62.1% 是单次噪声。
- **阈值:** 按 P0 数据模拟,只读 RoboSpatial vacant 两窗差 SD ≈ 9 pp;与 RefSpatial pointing 合并后 ≈ 5 pp。
