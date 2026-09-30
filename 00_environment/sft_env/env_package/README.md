# SpaceTools SFT 环境包(2× A6000 / sm_80·sm_86)

**已通过完整的 30 步 SFT 训练验证**(2026-09-07):`run_sft.sh` 三个 Phase 全部走通,
checkpoint 修复生效,退出码 0。

## 新机器上从这里开始

恢复完环境后,**先读 `/workspace/runpod-handoff/NEXT_MACHINE.md`** ——
那是给新机器准备的任务书(4 卡跑全量),含已验证的数值、必踩的坑、
以及 `TASK.md` 里已被实测推翻的几条判断。

背景细节在 `REPORT_A.md`(阶段 A–C 的完整记录)。

## 用法

```bash
sudo bash RESTORE.sh          # 前置检查 → 校验和 → 解包 → 验收,一条龙
```

`RESTORE.sh` 任何一项前置检查不过都会停,不会半途解一半。解完自动跑 `VERIFY.sh`。
单独复验:`bash VERIFY.sh`。

## 目标机要求

| | |
|---|---|
| 架构 | x86_64 Linux |
| GPU | compute_cap **8.0 或 8.6**(A100 / A6000 / A4000 / A5000 / 3090) |
| 驱动 | ≥ 525(本包在 580.159.03 上构建) |
| glibc | ≥ 2.32 |
| CUDA toolkit | **不需要** —— 运行时库在包内 `site-packages/nvidia/` |
| 磁盘 | `/opt` 所在盘 ≥ 12 GB |

**H100(9.0)和 Blackwell 跑不了**:包内 flash-attn 只有 `sm_80` 的 cubin,
没有 PTX 回退(`FLASH_ATTN_CUDA_ARCHS=80`,gencode 只给 `code=sm_80`)。
换那些卡要重编 flash-attn。`RESTORE.sh` 会在解包前拦下。

跑 SFT 还有显存要求:实测 2 卡 `per_device=2` 峰值 **48.5 GiB**,所以需要 48 GB 级别的卡。
A4000(16 GB)能加载这个环境,但跑不了这个训练。

## 卡数

`run_sft.sh` 按**可见 GPU 数自动推导** `per_device` / `ga`,保证
`per_device x ga x 卡数 = 8`(论文口径)。启动时打印一行确认:

```
GPU 数 4 · per_device=2 · ga=1 · 全局 batch 8 ✓
```

| 卡数 | `per_device` | `ga` | 余量(48 GB 卡) | 预估全量 |
|---|---|---|---|---|
| 2 | 2 | 2 | 0.6 GiB(实测,偏紧) | 9.3 h |
| **4** | 2 | 1 | **~10 GiB** | **~5–6 h** |
| 8 | 1 | 1 | ~15 GiB | ~3–4 h |

**3 / 5 / 6 / 7 卡会被拒绝启动** —— 8 不能被它们整除,凑不出论文的全局 batch,
而这个 checkpoint 是后续 RL 两臂对比的共同起点,口径不能动。

卡数由 `CUDA_VISIBLE_DEVICES` 决定(上游的 `NUM_GPUS` 是空操作)。
数据集只有 4 个分片,超过 4 卡会有 rank 分不到数据。

## ⚠️ 路径不能改

`llamafactory` 是 editable 安装,`site-packages/_editable_impl_llamafactory.pth`
里写死了 `/workspace/SpaceTools-SFT/src`。conda 环境里的 shebang 也都是
`/opt/conda-st/...` 的绝对路径。所以两个位置都必须原样恢复:

```
/opt/conda-st/                    conda + spacetools-sft 环境(9.3 GB)
/workspace/SpaceTools-SFT/        LLaMA-Factory fork,含 run_sft.sh(145 MB)
/workspace/SpaceTools/            上游仓库(6 MB)
/workspace/wheels/                自编的 flash-attn wheel(57 MB)
/workspace/runpod-handoff/        TASK.md · setup.sh · 报告 · 各种验收脚本
```

`/workspace` 只是普通目录,目标机没有的话 `RESTORE.sh` 会建。

## 包里没有的

SFT 数据(`siyich/spacetools-sft`,7463 个文件 / 5.89 GiB)和 base model
(`Qwen/Qwen2.5-VL-3B-Instruct`,~8 GB)不在包内 —— `run_sft.sh` 会自动下。

⚠️ 上游的 `snapshot_download` 在这个数据集上只有 ~0.8 MB/s(并发没生效,进程里只有 2 个线程),
要跑 1–2 小时。包内的 `runpod-handoff/prefetch.py` 是并行替代品,21 分钟下完并逐文件核对大小。

## 环境内容

```
python 3.11 · torch 2.9.1+cu128 · cuda 12.8
torchvision 0.24.1+cu128 · torchaudio 2.9.1+cu128     ← 必须与 torch 同版本
transformers 4.57.1 · flash-attn 2.8.3.post1(自编,仅 sm_80)
deepspeed 0.19.6 · llamafactory 0.9.5.dev0(editable)
accelerate 1.11.0 · trl 0.24.0 · peft 0.18.1 · datasets 4.0.0
```

## 相对上游 setup_envs.sh 的偏离

| 偏离 | 影响 |
|---|---|
| `FLASH_ATTN_CUDA_ARCHS=80`(上游默认 `80;90;100;120`) | 仅 sm_80 cubin、无 PTX;编译量和内存降到 1/4 |
| 增补 `ninja setuptools wheel` | 构建期工具;缺 `ninja` 时 flash-attn 退回串行编译 |
| **钉死 `torchvision` / `torchaudio`** | **修正**,见下 |
| 依赖上界补全 + `transformers` 二次钉死 | **修正**,见下 |

两处"修正"都是在补上游的漏,不是我们引入的偏离 —— 上游 `setup_envs.sh` 对传递依赖不设约束:

- **`transformers`** 会被顶成 5.x。**静默生效**,产出与论文口径不同的 π_ref
- **`torchaudio`** 会被解析到 2.11.0(给 torch 2.11 编的)。`llamafactory/data/mm_plugin.py`
  会 `import torchaudio` → `undefined symbol: torch_library_impl`,训练直接起不来

`VERIFY.sh` 现在两类都查:钉版本 + 走训练真实的导入链。
后者是必要的 —— 光 `import llamafactory` 抓不到 torchaudio 的问题,它是惰性的。

四项都不改变训练的数学结果。
