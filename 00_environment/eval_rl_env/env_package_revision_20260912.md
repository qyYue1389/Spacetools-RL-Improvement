# 环境包修订 2026-09-12:POSTRESTORE.sh + VERIFY 前置检查

写于 2026-09-12 · 依据 `03_sft_eval/sft_eval_results.md` 里 2026-09-11 那次 eval 的发现
· 已推到 `qzpm55555/spacetools-eval-env`

---

## 0. 一句话

那次 eval 暴露的三个「包里缺东西」和两个「假绿灯」,已经以**小文件**的形式修进 HF repo,
**22 GB 的 envs tar 没动**。新机器的还原顺序多一步:

```
4. bash FETCH_WEIGHTS.sh
5. bash RESTORE.sh
5b. bash POSTRESTORE.sh     ← 新增
6. bash VERIFY.sh           ← 现在是四项
```

## 1. 为什么是传文件而不是重新打包

真实 delta 只有三行:两个 `node.py` 各改 1 行、一个 1 行的 `activate.d`。
重打 22 GB 换来的只是「自包含」,而代价有三项:

| | 重新打包 | 传小文件 |
|---|---|---|
| 上传/下载量 | 压缩后约 26–28 GB(`/opt/conda-st` 现在 52G,原包解开约 40G,多出来的是 eval 跑出的 `__pycache__` 和 conda 缓存) | 几 KB |
| 每次换机器的下载增量 | **+5 GB** | 0 |
| 能否验证 | **不能**。`RESTORE.sh` 第 0 步 `[ -e /opt/conda-st ] && die`,明确拒绝覆盖,所以打完包只能确认「压缩上传成功」,不能确认「还原出来是对的」 | 能,脚本自带回读断言 |
| 根因可见性 | 把 workaround 焊进 tar,Python 版本不一致这个根因彻底隐形 | 补丁显式、可审、将来可删 |

A100(sm_80)和 A6000(sm_86)用**同一个包**,架构实扫确认自编扩展缺 sm_8x 为 0,
所以频繁换这两种卡不需要按架构分包 —— 变的只有 `NUM_GPUS` / `EVAL_GPUS` 和 GPU 预算。

## 2. `POSTRESTORE.sh` 补的三件事

幂等、可重复跑、每一处补完立刻回读断言。

**① Ray 的 Python 版本档位。** 本包 `spacetools-tool-vlm` / `spacetools-tool-bbox`
是 Python 3.11.0,其余三个是 3.11.16。Ray 的 `check_version_info` 默认比完整版本串,
这两个环境的 actor **全部入不了集群**。

脚本改用 Ray 自带的 `minor` 档(`node.py:454` 的调用点补
`python_version_match_level="minor"`,原文件备份为 `node.py.orig`)。

安全性依据:两侧 bytecode magic 都是 3495、pickle 默认协议都是 4,code object 与
pickle 互通 —— 这正是 minor 档的设计场景;该改动只放宽一个版本检查,不改变任何数值。

**脚本是版本感知的**:它先读五个环境的 Python,只对与头节点(`spacetools-rl`)
不一致的环境打补丁。将来重建包统一到 3.11.16,这一项自动跳过。

**② roborefer 环境的 `CUDA_HOME`。** llava 推理路径模块级硬依赖 deepspeed(环境
README 洞 #17),deepspeed import 时读 `CUDA_HOME`。打包机有系统
`/usr/local/cuda-12.8` 命中 torch 的第三条 fallback,只装驱动的机器没有。

**③ `/workspace/logs` 和 `/workspace/smoke`。** `28_chain.sh` / `04_smoke.sh` 往这
两个目录写,MANIFEST 只提了 `checkpoints` 和 `hf`。

## 3. `VERIFY.sh` 的两处修订 —— 这一条比上面三件更重要

**原来的三项抓不到 ①。** 补丁之前 `28_chain.sh` 是**通过**的(实测四环全 STEP_OK),
Python 版本不一致只在 eval / RL 规模下暴露。也就是说:忘了跑 POSTRESTORE,
三项全绿,然后 RL 静默少 23 个 actor。

所以加了**第 0 项前置检查**,不过就 `exit 1` 且不打印后面三项:

```
五环境 Python 版本 + 不一致时补丁是否在位
roborefer 的 CUDA_HOME(未设时自动指向该环境)
/workspace/{logs,smoke,checkpoints,hf} 是否存在
有没有活着的 Ray 集群 / 残留的集群地址文件
```

**第 3 项不再相信 `28_chain.sh` 的退出码**,改为在输出里找成功标记。实测它
打印「✗ 链没通」的同时 `exit 0`,而汇总据此打了「✓ 三项全过」。

### 残留 Ray 集群这个坑值得单独记

`/root/tmp/ray/ray_current_cluster`(上一次 eval 留下的 19 字节地址文件)足以让
`28_chain.sh` 的 `ray.init(num_cpus=8, num_gpus=1, ...)` 以为集群还在:

```
Connecting to existing Ray cluster at address: ...:6379
ValueError: When connecting to an existing cluster, num_cpus and num_gpus must not be provided.
```

python 块第 1 秒就死,脚本仍然 `exit 0`。**上一次 eval 不清理,下一次 VERIFY 的
第 3 项就会失败并给假绿灯。** 第 0 项现在会拦住。

另外 `ray stop --force` 报「57/58 停掉」不必担心:剩下那个是 zombie,已 defunct。

### 检测活集群要用 `pgrep -x` 而不是 `pgrep -f`

第 0 项判断「有没有活着的 Ray 集群」时,**必须按进程名精确匹配**:

```bash
pgrep -x gcs_server || pgrep -x raylet        # 对
pgrep -f "gcs_server|raylet"                  # 错,会匹配调用方自己的命令行
```

`pgrep -f` 匹配整条命令行,所以 `ray stop --force && bash VERIFY.sh` 这类调用会
把自己算成「有 Ray 在跑」而误报。Ray 的 core 二进制名就叫 `gcs_server` 和 `raylet`,
`-x` 按 comm 精确匹配,不受调用方命令行影响。

## 4. 顺手修掉的源头问题

`RESTORE.sh` 第 1 步执行 `sha256sum -c SHA256SUMS`,而 repo 原来只有
`SHA256SUMS.remote`,照文档跑必然 die 在「包损坏了」。现在两个同名文件都在 repo 里
(内容相同),不用再手动 `cp`。

## 5. 验证过程

```
POSTRESTORE.sh  幂等性:补丁已在位时正确报「已打过,跳过」并回读确认,exit 0
VERIFY.sh 第 0 项:
    有残留 Ray 集群时  → 正确 BAD 并 exit 1(在真实残留条件下测到)
    清理之后          → 8 项全 OK,正确进入第 1 项
SHA256SUMS:     13/13 自校验通过(6 个小文件重算,7 个大文件沿用原哈希)
两个脚本:        bash -n 语法检查通过
```

## 6. repo 里改动了什么

```
POSTRESTORE.sh      新增   5535 字节
VERIFY.sh           改写   5117 字节(三项 → 四项)
README.md           追加   还原顺序 + 三件事的说明 + SHA256SUMS 那条
MANIFEST.txt        追加   五环境 Python 版本表 + GPU 预算实测
SHA256SUMS.remote   重算
SHA256SUMS          新增   与 .remote 内容相同
spacetools-envs-*.tar.zst        未动
spacetools-scripts-*.tar.zst     未动
```

## 7. 还没做的

**根治 Python 版本不一致** —— 重建环境包时把五个环境统一到 3.11.16,之后
`POSTRESTORE.sh` 的第 ① 项自动跳过,整个脚本可以删。

本机上没法就地升级:

```
CondaToSNonInteractiveError: Terms of Service have not been accepted for:
    https://repo.anaconda.com/pkgs/main   https://repo.anaconda.com/pkgs/r
```

接受 Anaconda 商业条款是组织的法务决定;而且重新求解有动到 numpy 1.26.4 /
transformers 4.53.2 等钉死版本的风险,那是本项目最怕的静默改口径。

**GPU 预算**(见 `03_sft_eval/sft_eval_results.md` §4、§6):全工具存活时工具占 2.3 张、
策略要一整张,4 卡机器上偏紧;RL 的 actor 数量约需 7 张以上,必须重新算。
这一条不是包能解决的,是选机器时要算的。
