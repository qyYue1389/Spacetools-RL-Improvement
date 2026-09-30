# P6 开机用到的 run_eval.sh 变体与常驻脚本

没有这些,偏离 `[23]` 与实验 F 都无法复现。两个变体都是 `run_eval.sh` 的
**逐字副本 + 指定的最小改动**,放在 `examples/toolshed/` 同目录下运行,
因为 `run_eval.sh:64` 用 `BASH_SOURCE` 定位 `SCRIPT_DIR`,换目录会失效。

| 文件 | 与 `run_eval.sh` 的差异 | 用途 |
|---|---|---|
| `run_eval_gmu025.sh` | **一行**:`rollout.gpu_memory_utilization` `0.5` → `0.25` | run5 / run6。在 80 GB 上复现 40 GB 的 20 GB KV 池,见偏离 `[23]` |
| `run_eval_fp32_gmu025.sh` | **两行**:删掉 `actor.fsdp_config.model_dtype=bf16`;`gpu_memory_utilization` `0.5` → `0.25` | run7,实验 F。删掉 flag 后 `model_dtype` 落到 `verl/trainer/config/engine/fsdp.yaml:33` 的默认值 `fp32` |

两处 `0.25` 不是可选项:实验 F 的判据是 fp32 与 bf16 落在同一带,
两臂就只能差 dtype 一个变量,而 bf16 的参照臂(run5/run6)在 0.25 下。
用 0.5 跑 fp32 会同时动两个变量,结论作废。

    # 复核差异(应当分别是 1 行和 2 行)
    diff examples/toolshed/run_eval.sh examples/toolshed/run_eval_gmu025.sh
    diff examples/toolshed/run_eval.sh examples/toolshed/run_eval_fp32_gmu025.sh

## 运行脚本

`run_gmu025.sh` / `run_fp32.sh` 照搬 `tools/p4_run.sh` 的逻辑与两条守卫
(拒绝覆盖已有 dump、拒绝非 `run<N>` 标签),只把调用的 eval 脚本换掉。

> **文档 §1.5 给的标签 `run_fp32` 跑不起来**——`p4_run.sh` 的守卫是
> `^run[0-9]+$`,会直接拒绝。本次用的是 `run7`。

## Toolshed 常驻

`toolshed_v1_holder.py` 是 `run_eval.sh` 第 193–211 行(v1 分支)的**逐字副本**,
供实验 A / B 使用——那两个实验不需要 policy,只需要工具起来。

`start_toolkit(detached=False)` 的返回值必须由一个长期存活的进程持有
(原文件靠 `while True: time.sleep(60)`),否则 Ray 会回收 router,
几次调用之后报 `Could not find ToolRouterActor`,看起来像间歇性故障。

    source /workspace/env.sh && conda activate spacetools-rl
    ray stop --force; ray start --head --num-gpus=4 --port=6379
    export RAY_ADDRESS=127.0.0.1:6379
    nohup python toolshed_v1_holder.py "$ROBOREFER_MODEL" "$DEPTH_CHECKPOINT" &
    # 验活:ray status 应显示 3.0/4.0 GPU —— 但这还不够,见下

> **`ray status` 显示 3.0/4.0 GPU 不代表可以调用了。** 它只说明 actor 已经
> **占住**了卡,而模型还在 `Loading checkpoint shards`。据此过早发起调用,
> 397 条探测全部拿到 `ToolRouterActor.call_tool()` 异常——**而且失败长得和
> 真实结果一模一样**:一张完整的汇总表、所有数字 0.00%。
> `run_eval.sh` 写的是 `sleep 120`,照做。
>
> 识破它的三个信号:`工具失败` 计数等于样本数、每个 benchmark 只用 1–4 秒、
> 以及与基线差 210 个样本(远超「差十个就说明通路不对」的阈值)。
> **正式跑之前先用 `--limit 5` 冒烟一次**,五秒钟就能确认工具真的活了。

## 实验 C(pass@k)

| 文件 | 与 `run_eval.sh` 的差异 |
|---|---|
| `run_eval_passk.sh` | **五行**:`gpu_memory_utilization` 0.5 → 0.25,外加四条 `val_kwargs`(`do_sample=True` / `n=5` / `temperature=1.0` / `top_p=1.0`) |

verl 的 validation 默认是 greedy(`val_kwargs`:`temperature 0 / n 1 / do_sample False`),
不显式覆盖就跑不出采样。生效的验证方式:解析后配置里应出现 `do_sample: True`,
dump 行数应为 样本数 × 5,且逐样本的 5 次输出文本不应全同。
