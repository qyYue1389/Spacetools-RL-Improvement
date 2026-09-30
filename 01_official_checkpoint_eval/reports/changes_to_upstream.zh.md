# 我们做的每一处改动,以及为什么

> 本文是 `CHANGES.md` 的中文版。代码标识符、文件路径、配置键、命令、
> 错误原文、包名与 commit hash 一律保留英文。

## 译法对照

凡是做过取舍的译法都列在这里,附英文原文。对着原版核对时以此为准。

### 通用

| 中文 | 英文原文 | 说明 |
|---|---|---|
| 上游 | upstream | verl / Toolshed / GraspGen / RoboRefer 的原始仓库状态 |
| 偏离 | deviation | `PROVENANCE.txt` 里编号的那些改动 |
| 硬阻塞 | hard blocker | 不改就跑不起来 |
| 保真度 | fidelity | 「与论文一致」,而不是「更好」 |
| 落盘 | dump / written | verl 把样本写成 JSONL |
| 污染 | contamination | 一次运行的数字不可信,而不是崩溃 |
| 静默 / 静默错掉 | silent / silently wrong | 没有报错,但结果是错的 |
| 写死 | hardcoded | |
| 陈旧 | stale | 副本没跟上源头 |
| 逐位相同 | bit-identical | 张量逐元素比对 |
| 逐字节相同 | byte-identical | 文件或配置逐字节比对 |
| 逐字 | verbatim | |
| 恒等变换 | the identity | `bf16 → fp32 → bf16` |
| 数值惰性 | numerically inert | 改了,但够不到数值结果 |
| 承重 | load-bearing | 某个条件在暗中支撑着结果 |
| 线索 | lead | 指向原因但不是结论 |
| 物有所值 | earned its keep | |
| 悬念 | open question | |
| 微缩版 | in miniature | 「同一个失败模式的微缩版」 |
| 守门员 / 闸门 | guard / gate | 指 `--strict`、M3 gate |

### 三组容易混的词

原文用了不同的词,中文如果都译成同一个字就会丢信息。这三组在正文里也逐处标了原文:

| 中文 | 英文原文 | 区别 |
|---|---|---|
| **分数 GPU / GPU 分数** | **fractional GPU / GPU fraction** | Ray 的调度单位(0.6、1.0 这些) |
| **分数** | **score** | 评测得分 |
| | | |
| **回退** | **revert** | 撤销一次已经做出的改动(如 §1.7) |
| **退回 / 兜底** | **fall back / fallback** | 主路径失败后走的替代路径 |
| **可回退** | **Revertible** | 该条改动能否撤销,是每条改动的一个属性 |
| | | |
| **覆盖(配置)** | **override** | 如 `DATA_DIR` 覆盖 |
| **覆盖(文件)** | **overwrite** | 如 `p4_run.sh` 拒绝覆盖已有 dump |

### 本文特有

| 中文 | 英文原文 | 说明 |
|---|---|---|
| 基点 | base point | 每个仓库被我们动手之前的上游 commit |
| 净 diff | net diff | |
| 连锁影响 | knock-on effect | |
| 操作陷阱 | operating gotchas | §10 |
| 操作性约束 | operational constraints | 我们引入的、后来者会撞上的限制 |
| 构建陷阱 | build traps | conda 注入的那三个 |
| 中和 | neutralise | 把注入的环境变量抵消掉 |
| 超额分配 | over-subscribe | 告诉 Ray 的 GPU 数多于实际 |
| 装箱 | pack | Ray 把分数 GPU 装箱而不是分散 |
| 放置 | placement | actor 实际落在哪张卡上 |
| 预热 | warmed | 工具已加载完权重 |
| 尖峰 | spike | activation 尖峰 |
| 高水位线 | high-water mark | 分配器预留的峰值,不是实时占用 |
| 静息 | resting | 静息占用,与峰值相对 |
| 余量 | headroom | **显存语境**,指还剩多少 GB |
| 存档用 | for the record | |

为了让 SpaceTools 的 eval 在 4×A6000 上跑起来,一共动了五个位置,
而 §6 记录了之后迁移到 4×A100-SXM4-40GB(sm_80)所需要的改动。
本文是完整清单。`PROVENANCE.txt` 以浓缩形式携带同样的信息,
并附上权重 commit 与环境矩阵;本文是带着推理过程的长版本。

基点(我们动手之前每个仓库所处的上游状态):

| 位置 | 分支 | 上游基点 | 净 diff |
|---------------------|-----------------|---------------|--------------------------------|
| SpaceTools-RL       | `repro-4xa6000` | `54270e82`    | 1 个文件,+55 -21,6 个 commit |
| SpaceTools-Toolshed | `repro-4xa6000` | `4f0512d`     | 2 个文件,+26 -8,1 个 commit  |
| GraspGen            | `repro-4xa6000` | `2dd8852`     | 2 个文件,+4 -6,1 个 commit   |
| RoboRefer           | `repro-4xa6000` | `d97a995`     | 2 个文件,+2 -2,1 个 commit   |
| 官方 checkpoint      | (不是仓库)     | `f953b1a1`    | 2 个配置文件,保留 `.orig`     |

这样读它:

    git -C SpaceTools-RL              diff 54270e82..HEAD
    git -C SpaceTools-Toolshed        diff 4f0512d..HEAD
    git -C SpaceTools-Toolshed/GraspGen  diff main..HEAD
    git -C SpaceTools-Toolshed/RoboRefer diff main..HEAD
    cd /workspace/models/spacetools-ckpt && diff <(python3 -m json.tool config.json.orig) <(python3 -m json.tool config.json)

---

## 1. SpaceTools-RL

一个文件,`examples/toolshed/run_eval.sh`,跨六个 commit。

### 1.1 BENCHMARKS parquet 路径 (`f7560170`)

    - [robospatial]="robospatial_home_multiturn/test.parquet"
    + [robospatial]="data/robospatial.parquet"
      ... 全部九个 key

**为什么。** 写死的路径与 HuggingFace 上 `siyich/spacetools-eval-benchmarks`
的实际布局不符。数据集显然在脚本写完之后重组过,
而 benchmark key 已经与新文件名一一对应。

**不改的话**,第 113 行会以
`Missing: .../robospatial_home_multiturn/test.parquet` 退出 1。eval 根本启动不了。

**已用真实数据验证**:九个 parquet 全部命中。
**影响数字吗:** 不。同样的文件,不同的路径。

### 1.2 NUM_GPUS 8 -> 4 (`f7560170`)

上游默认值假设 2×8 A100。我们有 4×A6000。
在一台 4 GPU 的机器上告诉 Ray 有 8 个 GPU 会让调度器超额分配。

**影响数字吗:** 不。

### 1.3 DATA_DIR 覆盖(override) (`10037a43`)

    + if [ -z "${DATA_DIR:-}" ]; then
          <snapshot_download block>
          DATA_DIR="$EVAL_DIR/benchmarks"
    + else
    +     echo "Using preset DATA_DIR=$DATA_DIR (skipping benchmark download)"
    + fi

**为什么,两个理由。** `EVAL_DIR` 带时间戳,而 `snapshot_download` 的
`cache_dir` 和 `local_dir` 都被指到它里面,所以每次运行都会重下完整的 292 MB。
另外 `run_eval.sh` 完全没有样本数选项——`data.val_files` 直接指向一个 parquet——
所以跑子集的唯一办法是给它一个截断过的 parquet,而这要求能把 `DATA_DIR` 指到别处。

P2 的每一次冒烟运行都依赖这一条。

**可回退(Revertible):** 可以,代价是失去子集运行、且每次都重新下载。
**影响数字吗:** 不。

### 1.4 加载 conda 的 shell hook (`a71f20f1`)

    + if ! declare -F conda >/dev/null 2>&1; then
    +     ... . "$_conda_base/etc/profile.d/conda.sh"
    + fi

**为什么。** 第 105 行调用 `conda activate`,那是一个 shell *函数*。
非交互式的 `bash run_eval.sh` 继承导出的变量但不继承 shell 函数,
于是它以 `CondaError: Run 'conda init' before 'conda activate'` 失败。
上游大概总是从一个已经 conda init 过的 shell 里运行它。

**影响数字吗:** 不。纯粹是可移植性。

### 1.5 dtype 'float16' -> 'auto' (`f68fd4c6`)

这一条看起来像行为改动,实际上正相反。

上游的 `vlm.py` 把 `torch_dtype="auto"` 写死,忽略它自己的 `dtype` 参数,
所以配置里那个 `'float16'` 是死代码。`"auto"` 会按 Molmo 存储时的精度加载,
而 Hub 上存的是 fp32——实测 33 GB,而不是 fp16 所暗示的约 15 GB。
**因此论文的数字是在 fp32 下产生的。**

一旦我们修好 `vlm.py`(见 2.2),配置就会真的开始生效,
那会把我们*推离*论文。显式写 `'auto'` 才能让运行时行为与上游逐字节相同。

**影响数字吗:** 不——正是这一条让它们保持不变。

### 1.6 工具 GPU 分数(GPU fractions)与 EVAL_GPUS (`f68fd4c6` 之后 `910f6f5d`)

| 工具 | 上游 | 中间版 | 最终 |
|--------------------|---------:|--------:|------:|
| `vlm`              |      0.5 |    0.55 |   1.0 |
| `roborefer`        |      0.5 |    0.55 |   0.6 |
| `sam2` x2          |     0.15 |    0.15 |   0.2 |
| `depth_estimator` x2 |   0.15 |    0.15 |   0.4 |
| `bounding_box` x2  |      0.1 |     0.1 |  0.05 |
| 合计               |      1.9 |     2.0 |   3.0 |

外加 `EVAL_GPUS` 4 -> 1。

**为什么,分两轮。** Ray 对分数 GPU 是**装箱**而不是分散,
所以上游的 `roborefer 0.5 + vlm 0.5 = 1.0` 把两个大模型都放进了 GPU 0:
18.6 GB + 33 GB,而卡只有 48 GB。中间版的 0.55/0.55 让它们互斥(1.1 > 1.0)。

那还不够。实测占用比计划假设的大得多——
**DepthPro 是每个 actor 12.5 GB**,而不是估计的约 4 GB,
配上 `num_actors: 2` 就是 25 GB。工具合计约 80 GB,而两张卡是 94.8 GB:
84% 占用率,没有留给 activation 尖峰的余量。
在 `max_parallel_calls=8` 下,一次 124 样本的 blinkdepth 运行 OOM 了 68 次。
受影响的 13 个样本得分 38.5%,而干净的那些是 88.3%,
把这个 benchmark 从 88.3% 拖到 83.1%——**一个静默错误的数字,而不是一次崩溃。**

最终的拆分给了工具三张卡:Molmo 独占、RoboRefer + SAM2、DepthPro + bbox + grasp。
所有上游副本数都被保留;只有 `EVAL_GPUS` 改了,而那本来就是我们可选的。
代价是策略侧从两个数据并行副本降到一个。

**影响数字吗:** 不。分数(fractions)是 Ray 的记账方式,不是显存上限,它们只决定放置(placement)。

### 1.7 有意不设置 expandable_segments (`77141944` 之后 `ef2c7580`)

根据 OOM 报错自己的推荐加上过,然后回退了(reverted):
sglang 的 HYBRID rollout 用 TorchMemorySaver 来释放和重建 KV cache,而它拒绝在那下面运行:

    RuntimeError: TorchMemorySaver is disabled for the current process because
    expandable_segments is not supported yet.

只有一条注释留了下来,解释它为什么绝不能被重新加上。

---

## 2. SpaceTools-Toolshed

两个文件,一个 commit(`712e557`)。

### 2.1 `graspgen_franka_panda.yml` —— 三个 checkpoint 路径改为绝对路径

    - checkpoint: ../../checkpoints/GraspGenModels/checkpoints/graspgen_franka_panda_dis.pth
    + checkpoint: /workspace/models/GraspGenModels/checkpoints/graspgen_franka_panda_dis.pth

第 170、171 行(`discriminator:` 段)和 217 行(`eval:` 段)。

**为什么。** 这些路径是相对的,而 `grasp_gen` 没有任何逻辑把它们
相对于配置文件所在位置去解析——`from_config` 把字符串直接传给 `torch.load()`,
所以它们相对进程 CWD 解析。在一个 Ray actor 内部,那不由我们控制。

上游对位置的说法也自相矛盾:这个 yml 说 `../../checkpoints/`,
`requirements/tool-graspgen.txt` 说 `toolshed/data/`,
而 `install_graspgen.sh` 用 `$MODELS_DIR`。
我们按安装脚本放置的位置 clone,并把 yml 指到那里。

编辑之前核对过语义:170/171 是 discriminator 自己的权重
加上被用作预训练初始化的 generator 物体编码器;217 是 eval 采样用的 generator。
diff 正好只碰了那三行。

**已在加载日志中验证**,可以看到读取的是新路径。
**影响数字吗:** 不。同样的权重,只是找得到了。

### 2.2 `vlm.py:112` —— 尊重 `dtype` 参数

    -     model_kwargs["torch_dtype"] = "auto"
    +     model_kwargs["torch_dtype"] = (
    +         "auto" if dtype == "auto" else getattr(torch, dtype)
    +     )

**为什么。** 构造函数声明 `dtype: str = "float16"`,但非量化路径完全忽略它。
这让配置在撒谎,并且掩盖了 Molmo 正在以 fp32 运行、占 33 GB 这个事实——
而那正是 RoboRefer 与它共卡之后 GPU 0 会 OOM 的原因。

**影响数字吗:** 不,因为 `run_eval.sh` 传的是 `'auto'`,走的分支与上游相同。
修好它是把一次未来的精度消融从「重新发现」变成「改一个字符串」。

### 2.3 / 2.4 `vlm.py:326, 346, 364` —— 把浮点输入转成模型的 dtype

    + model_dtype = next(self._model.parameters()).dtype
    ...
    + if v.is_floating_point() and v.dtype != model_dtype:
    +     v = v.to(model_dtype)

**为什么。** 搬运张量的代码只做了 `.to(model_device)`——只管设备,从不管 dtype。
在模型一直是 fp32 时无害;模型一旦是 fp16,它就会以
`expected mat1 and mat2 to have the same dtype, but got: float != c10::Half` 失败。

**这是第二个被 2.2 掩盖的上游 bug。** 只要一切保持 fp32,两个都不会触发,
代价是显存翻倍。只有浮点张量会被转换,所以 `input_ids` 和 `image_input_idx` 保持整型。
正常路径和 OOM 恢复路径都需要它。

**影响数字吗:** 不。在 fp32 下条件永远不成立;它是个 no-op。

---

## 3. GraspGen

一个 commit(`cb846e1`)。**diff 里的五个 hunk 中只有一个是我们的。**

### 3.1 删掉 `pickle5`(我们的)

    # pyproject.toml
    -     "pickle5",
    # requirements.txt
    - pickle5

**为什么。** `pickle5` 把 pickle protocol 5 backport 到 Python 3.6/3.7。
protocol 5 自 3.8 起就在标准库里,而这个 backport 使用的私有 C API
(`_PyObject_CallNoArg`、`_PySys_GetObjectId`、把 `Py_SIZE` 当左值)在 3.11 已被移除,
所以它编不过:

    pickle5/_pickle.c:6178:9: error: lvalue required as left operand of assignment
            Py_SIZE(self->stack) = len;

也没有 py3.11 的 wheel——这个包最后一次发布是 2020 年。

删掉它是保行为的:唯一的 import 处本来就有兜底。

    try:
        import pickle5 as pickle
    except:
        import pickle

**可回退(Revertible):** 不可以。它根本编不过。
**影响数字吗:** 不。

### 3.2 torch / torchvision / numpy / spconv 的 pin(**不是**我们的)

    - "torch==2.1.0"          + "torch>=2.3.0,<2.4"
    - "torchvision==0.16.0"   + "torchvision>=0.18.0,<0.19"
    - "numpy==1.26.4"         + "numpy>=2.0"
    - "spconv-cu120"          + "spconv-cu121"

`install_graspgen.sh` 第 94-98 行每次运行都用 `sed` 施加这些改动。
上游自己的注释解释了原因:torch 2.1 无法与 numpy 2.x 互操作,而 2.3 恢复了兼容性。
提交它们只是为了让工作树干净。

**值得记住的连锁影响。** 这些 pin 把 GraspGen 放在 `torch>=2.3,<2.4`,
与我们 `tool-constraints.txt` 里的 `torch==2.9.1` 冲突:

    ERROR: Cannot install grasp-gen==1.0.0 ...
        grasp-gen 1.0.0 depends on torch<2.4 and >=2.3.0
        The user requested (constraint) torch==2.9.1

所以构建那个环境时必须 unset `PIP_CONSTRAINT`。
PyG 的 `torch-cluster` / `torch-scatter` wheel 只存在于 `2.3.0+cu121`,
进一步锁死了它。**这就是为什么五个环境里有四种不同的 torch 版本。**

---

## 4. RoboRefer

一个 commit(`7c50d8a`)。两个 hunk 里有一个是我们的。

### 4.1 跳过写死的 py3.10 flash-attn wheel(我们的)

    - pip install https://.../flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    + # patched: flash-attn is installed separately, matched to the torch this env ends up with

**为什么。** 那个 URL 同时钉死了 Python 3.10、torch 2.3 和 cu122。
这个环境是 Python 3.11,所以 pip 直接拒绝(`"not a supported wheel on this platform"`),
接着 `set -e` 在第 24 行之前就中止了脚本——也就是说
`pip install -e ".[train,eval]"` 从未运行,**llava 也就从未被安装**。

flash-attn 改为单独安装。那花了两次尝试:
第一次用的是为 torch 2.9.1 构建的 wheel,但第 24 行随后拉进 VILA 的 pin、
把 torch 降到 2.5.1,破坏了 C++ ABI(`undefined symbol: _ZN3c105ErrorC2E...`)。
第二次用的是官方的 `+cu12torch2.5cxx11abiFALSE-cp311` 构建,
在此之前核对了 `torch._C._GLIBCXX_USE_CXX11_ABI` 为 `False`。

**可回退(Revertible):** 不可以。硬阻塞。
**影响数字吗:** 不。同样的 flash-attn 版本,ABI 匹配。

### 4.2 禁用 `lighteval`(**不是**我们的)

`install_roborefer.sh` 第 108 行用 `sed` 把它注释掉;
上游自己的说明写着这个依赖在安装时会挂起。提交它只是为了让工作树干净。

### 4.3 一个我们打了又撤销的补丁

第 27 行的 `pip install triton==3.1.0` 曾被短暂注释掉,
理由是它会破坏 torch 2.9.1(后者要求 triton 3.5.1)。

**那个推理是错的。** 第 24 行先运行,已经把 torch 降到 2.5.1,
而 2.5.1 自己的 pin **就是** triton 3.1.0——所以在真实的执行顺序里那一行是 no-op。
已还原为上游原样。

这个错误值得记录,因为它反复出现:**从某个组件的当前状态去推断整个系统的行为,
而不是走一遍真实的执行顺序。**

### 连锁影响

`pip install -e ".[train,eval]"` 会整体降级这个环境:

| 包 | 之前 | 之后 |
|--------------|---------------|---------------|
| torch        | 2.9.1+cu128   | 2.5.1+cu124   |
| transformers | 4.57.1        | 4.49.0        |
| numpy        | 2.4.6         | 1.26.4        |

这不是缺陷——VILA/llava 就钉在那套栈上,
而 Toolshed 的逐工具 conda 环境存在的意义正是吸收这种情况。
但它让 roborefer 成为唯一一个用 transformers 4.49 的环境,做归因时值得记住。

它也解释了为什么 driver 侧必须用 numpy 2.x:
roborefer 在 numpy 1.26.4 下产生数组,而 numpy 2 能反序列化 numpy-1 的数组,
反过来则失败。**消费者必须至少和每一个生产者一样新。**

---

## 5. 官方 checkpoint 的配置

**这是唯一一处我们修改了发布的权重而不是代码的地方。**
由 `/workspace/fix_checkpoint.py` 施加;原件保留为 `config.json.orig`
和 `preprocessor_config.json.orig`。**这两个备份不在任何 git 里——留好它们。**

### 5.1 `config.json` —— 移除 `text_config`

    text_config = <dict, 26 keys, model_type='qwen2_5_vl_text'>   ->   不存在

**为什么。** 有 `text_config` 在时,transformers 会把 `model_type` 解析成
内层的 `qwen2_5_vl_text`(文本子模型)而不是顶层的 `qwen2_5_vl`,于是 sglang 拒绝加载:

    RuntimeError: Unimplemented model type: qwen2_5_vl_text

`docs/SETUP.md:112-116` 记录了这个确切的错误和修法,并说 `run_sft.sh` 会自动施加——
但那是针对你自己训练出来的 checkpoint。发布的 `siyich/spacetools-ckpt` 是未修复的。

**可回退(Revertible):** 不可以。硬阻塞。

### 5.2 `config.json` —— 设置 `tie_word_embeddings = True`

    不存在   ->   True

**为什么。** `run_sft.sh` 第 273-274 行这么做,
而基座模型 `Qwen/Qwen2.5-VL-3B-Instruct` 也是 `True`。

**这一条不是照抄的。** 如果输出头是单独训练过的,
设成 `True` 会让 transformers 丢弃训练好的 `lm_head` 而复用 embedding——
模型照样能跑,而每一个输出都会是错的,并且是静默的。所以两个张量被直接比对了:

    model.embed_tokens.weight   (151936, 2048)
    lm_head.weight              (151936, 2048)
    identical: True | max abs diff: 0.000e+00

逐位相同,所以这个头确实是 tied 的,checkpoint 只是把副本物化了出来。
`fix_checkpoint.py` 自己会做这个检查而不是假定;遇到未 tied 的头它会跳过这一步并说明。

### 5.3 `preprocessor_config.json` —— 换成基座模型的那份

    image_processor_type:  Qwen2VLImageProcessorFast  ->  Qwen2VLImageProcessor
    keys:                  26                         ->  9

**为什么。** `docs/SETUP.md:117-119`:Fast processor 产生的 image token 数
与视觉编码器期望的不同,导致 off-by-one:

    ValueError: Image features and image tokens do not match

这只有在 5.1 修好、模型真的开始处理图像之后才会浮现,所以两件事是一起做的。

**是哪个版本,重要吗?** 仓库里没有任何地方 pin revision——
`run_sft.sh:289` 调用 `hf_hub_download('$BASE_MODEL', 'preprocessor_config.json')`,
没有 `revision=`,所以它总是取当前 main。这看起来像个悬案,其实不是:

- 我们安装的文件与 Hub 上的逐字节相同(sha256 `f2058c716eef96cc...`,350 字节)
- 自 2025-02-15 的 commit `1b989f2c` 之后,Qwen 就没有再动过 `preprocessor_config.json`;
  之后的 commit 只改了 tokenizer_config、generation_config 和 README
- 论文是 arXiv 2512.04069,2025 年 12 月

所以作者必然通过同样那次未 pin 的调用取到了同一个文件。
**这不是数值偏差的来源。**

---

## 6. Phase M:从 A6000(sm_86)迁移到 A100-SXM4-40GB(sm_80)

于 2026-08-27 完成。真正改动了两件事;第三件看起来需要改的其实不需要。

### 6.1 `pointnet2_ops` 为 `8.0;8.6;8.9` 重编

偏离 [3] 用 `TORCH_CUDA_ARCH_LIST=8.6` 构建了这个 GraspGen 依赖,
因为 torch 把 arch 列表默认成本机设备,而 `setup.py` 什么都没 pin。
那产出的是 **没有 PTX 回退**的 sm_86 cubin。

CUDA 只在 `z >= y` 时才用 `sm_X.y` 的 cubin 跑 `sm_X.z`。A100 是 8.0,
所以这是必然失败而不是风险。动手之前确认过:

    cuobjdump --list-elf _ext...so   ->  4x sm_86, no PTX
    furthest_point_sample(...)       ->  CUDA kernel failed :
                                         no kernel image is available for
                                         execution on the device

重编之后:sm_80、sm_86 和 sm_89 各 4 个 cubin,算子恢复正常。
recipe 保存在 `/workspace/rebuild_pointnet2_sm80.sh`。

**三个由 conda 注入的构建陷阱(build traps)必须再次中和(neutralise)**——
它们是环境的性质而不是上一台机器的性质,所以全都还活着:

| 被注入的 | 效果 | 我们做了什么 |
|----------|--------|-------------|
| `NVCC_PREPEND_FLAGS=-ccbin=x86_64-conda-linux-gnu-c++` | 把 nvcc 的 host 编译器钉死在 conda 的 **gcc 15.3**;nvcc 12.8 拒绝 > 14 | 把它指向 `/usr/bin/g++-13` |
| `CFLAGS`/`CXXFLAGS` | 夹带 conda 的 CUDA **13.3** 头文件 | `unset` |
| `LDFLAGS` | `-L$CONDA_PREFIX/lib` —— 见下 | `unset` |

我们**没有**使用 `-allow-unsupported-compiler`。
NVIDIA 警告它可能产生静默错误的结果,对于要输出几何位姿的算子这不可接受。

第四个陷阱是我们自己的,不是 conda 的:**`setup.py` 的 `build/` 目录会存活**,
而 setuptools 会欣然把陈旧的 sm_86 `.o` 重新链接进「重建后」的扩展。
脚本会先删掉 `build/`。没有这一步,重建就是一次**静默的 no-op**——
和 OOM 问题同样的失败形状,也同样看不见。

### 6.2 更正:偏离 [3] 对 cudart 的诊断是错的

[3] 记录说旧的构建链接了环境里的 `libcudart.so.13`,
因为 `LIBRARY_PATH` 指向 `.../lib64/stubs`,而那里没有 `libcudart`。机制不是这样。
**`LIBRARY_PATH` 只在显式的 `-L` 之后才被搜索**,
而 conda 的 `LDFLAGS` 把 `-L$CONDA_PREFIX/lib`——那里确实有 `libcudart.so.13`——
放在 torch 的 `-L/usr/local/cuda/lib64` *之前*。
把 `LIBRARY_PATH` 设对什么都不改变;**unset `LDFLAGS` 才是修法。**

重建后的扩展链接的是 `/usr/local/cuda/lib64/libcudart.so.12`(12.8.90)。
运行时没有任何损失:`/usr/local/cuda/lib64` 在系统 `ldconfig` 路径上,
所以丢掉 conda 的 rpath 之后仍然能解析。

尽管 [3] 说它无害,这件事仍然值得追查。它在狭义上确实无害——cubin 在编译期就定死了——
但它意味着每一个加载这个扩展的进程都**同时驻留两套 CUDA runtime**:
torch 带来的 12.8 和 pointnet2_ops 带来的 13.3。现在没有了。

更宽的教训是 P2 已经用另一个名字记过的那一条:
原来那条笔记是从某个组件的配置(`LIBRARY_PATH` 说是 stubs)去推理,
而不是追踪链接器对整条命令行实际做了什么。

### 6.3 `p2_tool_chain.py` 的 GPU 分数(fractions)是陈旧的 —— 我们自己的 bug,被 48 GB 掩盖

这个工具链冒烟测试带着自己的一份 `TOOL_CONFIGS` 副本,
docstring 承诺它是「copied verbatim from `run_eval.sh`」。它不是。
它仍然停在 §1.6 描述为已被取代的那个**中间版**拆分——
`vlm`/`roborefer` 0.55、`sam2`/`depth` 0.15。
`910f6f5d` 把 `run_eval.sh` 改成最终拆分时,它从未被同步更新。

在 48 GB 的卡上这还能活:Molmo 33.0 + DepthPro 12.5 + SAM2 1.2 = 46.7 GB,
装得进 48,所以 P2 跑绿了,没有人去看。
在 40 GB 上,它在第一次深度调用就 OOM,一张卡上三个进程:

    torch.OutOfMemoryError: ... GPU 0 has a total capacity of 39.49 GiB of which
    98.31 MiB is free. Process ... 30.38 GiB ... 7.70 GiB ... 1.29 GiB

现在已同步为 `0.6 / 1.0 / 2x0.2 / 2x0.4 / 2x0.05 / 0 / 0.1 = 3.0`,与 `run_eval.sh` 一致。
之后这条链得分 **12/12**——比 A6000 的 11/12 还好,
在那边 `grasp_generator` 曾因场景不匹配在厨房照片上失败。

**影响数字吗:** 不。`p2_tool_chain.py` 是诊断工具,它从未产出过被报告的结果。
但它本来会浪费一天:它是 M3 的 gate,
而**一个因为与它所守护之物无关的原因而失败的 gate,比没有 gate 更糟。**

它还现场演示了这个项目的核心失败模式,这值得亲眼见到而不只是读到。
OOM 没有抛出;深度工具**正常返回**了,异常文本在它的 `text` 字段里,`variables` 是空列表:

    1/7  depth_estimator.estimate_depth_with_pointcloud
        elapsed: 1.2s
        text   : torch.OutOfMemoryError: CUDA out of memory. Tried to allocate...
        variables: []

这条链接着走完了工具 2-7 并打印了摘要。
如果这是 eval 而不是冒烟测试,模型会把那个字符串当作一次工具响应读下去、
绕着它推理,然后产出一个看似合理的错误答案。正是 P2 描述的那次 68 次 OOM 级联的微缩版。

### 6.3b 检出是陈旧的,而原因是凭证

到达时,`SpaceTools-RL` 的本地分支停在 `ef2c7580`(patch 0007),
而 `verl/trainer/ppo/ray_trainer.py` 与上游 `54270e82` 逐字节相同——
也就是说 P3 的 uid/num_turns dump 不在工作树里,
尽管 `P4_HANDOFF.md` 的仓库表把它列为已存在。

**它确实存在——在远端。** 它是从 A6000 主机以 `8d193ec3` 提交并推送的,
和 `patches/rl/0008` 的头部是同一个 SHA。卷上的仓库副本只是比远端落后了那一个 commit。

handoff 的第一条仓库指令就是*「先对所有仓库 `git pull`」*。那条跑不了。
remote 是 `https://`,GitHub 凭证曾经存在上一台主机的 **container** 磁盘上,
而一次 recycle 会擦掉 container 磁盘——
所以每次 `git pull` 都死在 `could not read Username for 'https://github.com'`,
陈旧的检出就这样没被注意到。修好凭证(§6.3c)再 pull 才是真正的修法;
我的第一次尝试是手工重新应用 patch 文件,那产出了内容相同但 SHA 不同的提交,
只好丢弃、改用远端那个 commit。

**有一个陷阱值得记录,因为我差点掉进去。** `sample_uids` 和 `sample_turns`
这两个名字在 `ray_trainer.py` 里到处都是——*即使没有打补丁*——
它们是 verl 自己的变量,喂给 `val-aux/num_turns/*` 指标。
`patches/rl/0008` 加的是**把它们传进 `_dump_generations`**,而上游从不这么做。
grep 名字会说「已应用」。说真话的检查是
`git diff 54270e82..HEAD -- verl/trainer/ppo/ray_trainer.py`(为空 = 没打)
和对 patch 执行 `git apply --check`(能干净应用 = 缺失)。

这与 P1 和 P2 反复撞上的是同一个失败模式,只是换了一身衣服:
**读一个表面信号,而不是它所代表的那个东西。**
在这里,代价会是 V2 的 `turn-count mismatch` 检查什么都没在测。

### 6.3c GitHub 凭证,以及它们为什么消失了

这台主机上每一次 `git fetch`/`pull`/`push` 都以
`could not read Username for 'https://github.com'` 失败。
原因是结构性的,不是配置错误:**凭证存在上一台主机的 container 磁盘上**
(`/root`、`/opt`、`/`),而 recycle 一个实例会从镜像重建那部分。
只有挂载的卷会存活——这也是为什么 conda 环境、权重和 dump 都过来了,而凭证没有。

不需要 token。`/root/.ssh/id_ed25519`(key comment `vast-instance-c17760dabb09`)
**已经授权在 `qyYue1389` 账号上**——`ssh -T git@github.com` 回答 `Hi qyYue1389!`。
只是 remote 的 scheme 写错了。把五个仓库从 `https://github.com/` 换成
`git@github.com:` 一次性修好了 fetch、pull 和 push:

    git remote set-url mine git@github.com:qyYue1389/<repo>.git

**这不是持久的。** 那个 key 在 `/root`,即 container 磁盘上,
所以一次 recycle 会把同样的失败带回来——并且再一次带回一个悄悄落后远端的检出。
任何 recycle 这个实例的人都必须重做这个 key(或者把它移到卷上,
但要权衡 `/workspace` 可能与同一台物理机上的其它实例共享)。

### 6.4 哪些**不**需要改

**`run_eval.sh` 完全不需要编辑。** P4 handoff 为 40 GB 卡推导出的分数预算
与 `910f6f5d` 已经为 A6000 提交的逐字节相同——
`vlm 1.0` 独占;`roborefer 0.6 + sam2 2x0.2`;`depth 2x0.4 + bbox 2x0.05 + grasp 0.1`;
`vision_ops 0`;工具 3.0 + policy 1.0 = 4.0。
A6000 的排布本来就已经被压到物理边界上,而变化的是物理边界的**大小**而不是**数量**。
`EVAL_GPUS` 保持 1。

**`spacetools-rl` 里的 flash-attn 不需要重建。** handoff 预期 A6000 上的源码构建
只会挑本机 arch。它没有:72 个 cubin,**全部是 sm_80**。
它一直在 A6000 上跑 sm_80 的代码(合法,8.6 >= 8.0),现在与硬件精确匹配。

为了确认没有别的东西潜伏,五个环境里的每一个编译扩展都用 `cuobjdump` 扫过——
**66 个 `.so` 文件,每一个都带 sm_80 cubin**。
`pointnet2_ops` 是这台机器上唯一一个不具备 sm_80 能力的扩展。

**没有动任何 `num_actors`**;所有副本数保持上游默认。
**`dtype` 保持 `'auto'`**(Molmo fp32)。
用 fp16 换余量会是一次保真度上的偏离,而不是修复——见 §1.5 和 §2.2。

### 6.5 实测的放置

Toolshed 起来、每个 GPU 工具都预热过,其余空闲时:

| GPU | 已用 | 内容 |
|-----|-----:|----------|
| 0 | 21020 MiB(20.5 GB) | RoboRefer 18.4 + SAM2 1.3 + SAM2 0.8 |
| 1 | 35337 MiB(34.5 GB) | **Molmo 独占** —— 40 GB 的 86.3% |
| 2 | 17931 MiB(17.5 GB) | DepthPro 12.2 + 4.0 + 1.3 |
| 3 | 4 MiB | 空闲,留给 policy |

`ray status` 报告 **3.0/4.0 GPU**。每个进程的占用都与 P2 的 A6000 表一一对应
(RoboRefer 18.6、Molmo 33.0、DepthPro 12.5、SAM2 1.2)。

**关于这是怎么确立的,有一条告诫。** `nvidia-smi` 报告的是*宿主机*的 pid,
而这个容器看不到自己的宿主 pid——`/proc/<pid>/status` 只有一个 `NSpid` 条目——
所以**在容器内部无法把逐进程的 GPU 行与 Ray actor 对应起来**。
`CUDA_VISIBLE_DEVICES` 也帮不上忙:Ray 是在 `exec` 之后才设它的,
所以它不在 `/proc/<pid>/environ` 里(那是一份快照)。
放置因此是由三条互相印证的证据确立的——
`ray status` 显示 3.0/4.0、恰好三张被占用的卡且进程数 3/1/3 与三个分组吻合、
以及能唯一识别每个模型的占用量——**而不是靠直接的 pid 关联**。
这比一次关联弱,但比读那些分数强,而后者正是 handoff 警告过的。

### 6.6 这一阶段交给 V3 的悬念

Molmo 在一次 1.36 MP 图像上的 `detect_one` 之后停在 **34.5 GB**,
高于 P2 表里的 33.0 GB,只剩 5.6 GB。而 `robospatial` 的图像是 2.76 MP。

值得精确表述,因为它改变了 V3 该怎么读:PyTorch 的缓存分配器不会把
释放的块还给 driver,所以 `nvidia-smi` 报告的是**分配器预留的高水位线**,而不是实时占用。
这让 `nvidia-smi` 成为 V3 追峰值的正确工具——
但也意味着 P2 表里那个 33.0 GB 本来也从来不是一个纯粹的静息值。
这个悬念诚实的表述不是「静息还是峰值」(resting or peak),
而是「在 2.76 MP 下高水位线(high-water mark)能爬多高」,而 V3 测的正是这个。

---

## 7. Phase V1:解析器回归测试抓到的三个 bug

`P2_RESULTS.md` 针对 blinkdepth **run4** dump 手工算出的统计量是一份免费的回归测试,
而它立刻就物有所值。对着交付时的解析器跑,
**直接从 JSON 字段出来的数字都对**(108/124 = 87.10%、16 个错、1 个工具失败、
零 OOM / 截断 / 顶轮数),而**每一个需要真正解析 transcript 的数字都错了**。

**那个分裂就是线索。** score 和错题数是**读**出来的,不是派生的;
chains、工具直方图、轮数和 `no <answer>` 是**派生**的。只有派生的那一半坏了。

### 7.1 切分器静默丢掉了 assistant 第一轮

`parse_transcript` 按 `<|im_start|>` 切分 `output`,并把每块的第一行当作 role。
但 verl 落盘的 `output` 只有**响应部分**——
第一轮开头的 `<|im_start|>assistant` 在 `input` 的尾巴上。
所以第一块没有 role 头,它的第一行是模型自己的 `<think>`,
role 既不匹配 `assistant` 也不匹配 `user`,于是整个这一轮被丢弃。

第一轮正是标准 blinkdepth 链路发出 `depth_estimator` + `roborefer` x2 的地方。后果:

| 统计量 | 交付时的解析器 | 真值 |
|-----------|---------------|-------|
| `depth_estimator` 调用数 | **完全不出现** | 122 |
| 主导链路 | `vision_opsx2@2t` x107 | `depth_estimatorx1+roboreferx2+vision_opsx2@3t` x106 |
| 错题链路 | 不是预期的 12 | **12**,与 P2 手工计数一致 |
| `no <answer>` | 2 | 0 |
| 平均轮数 | 2.13 | 3.13 |

**`depth_estimator` 从一个深度 benchmark 的工具直方图里消失,是最响亮的症状,
而它就摆在明面上。** 那两个 `no <answer>` 样本是被整个吞掉的单轮输出——
它们在交付版解析器自己的链路列表里显示为 `@0t` x2。

它还会压掉 `hit_max_turns`:每个计数都少一,
一个真正撞到 8 轮上限的样本会被读成 7,永远不会触发标记。

修法是把没有 `<|im_start|>` 前缀的首块当作 assistant 第 1 轮。
修好之后,handoff 的八个期望值全部重现,包括 handoff 特别点名为最重要的那个 12-chain。

### 7.2 轮数交叉校验比的是两个不同的东西

patch 0008 存在的意义,是让解析器可以拿自己的分解去对照 verl 的计数,
而 `--strict` 把不匹配当作污染。但两边的数法不同——
`verl/experimental/agent_loop/tool_agent_loop.py:238`:

    num_turns = agent_data.user_turns + agent_data.assistant_turns + 1

而 `num_turns_derived` 数的**只有 assistant 轮**。
直接比较会让**每一个样本**都成为一次不匹配。
在 patch 0008 已经进树的情况下,V2 会在每个 benchmark 上都以 1 退出,
而 handoff 说非零值「是解析器的 bug,不是运行的问题」——它本来会是对的,
只不过 bug 在检查里而不在切分器里。

解析器现在还派生 `num_turns_verl_convention`(`assistant + user + 1`)并拿它去比。
用 P2 记录的四次运行轮数验证过——6.26、4.00、4.00、9.88——四个全部精确重现。

这也解释了 P2 的轮数与解析器之间那个「差两倍」:
两边都没有 bug,是两种约定。P2 给 blinkdepth 的平均轮数 6.26,
等于 3.13 个 assistant 轮加 2.13 个工具结果轮加上 prompt。

### 7.3 P2 的 grasp 成功/失败拆分是错的,解析器是对的

`P3_NOTES.md` 曾把 P2 的 bopgrasp 拆分——27 succeeded / 33 errored——
当作「失败检测是否有效最便宜的检查」。解析器说 **19 / 41**,
而一次完全独立于解析器的 dump 原始文本搜索同意解析器:

    "No collision-free grasps"        33 events in 33 samples
    "Top-down filtering removed all"   8 events in  8 samples
    at least one failure               41 samples  ->  19 clean

每个样本正好一次 `grasp_generator` 调用,共 60 次。
P2 只把 `No collision-free grasps` 算作失败——所以正好 33——
并把那 8 个 `Top-down filtering removed all` 的样本算成了成功。

**P2 的 96 vs 16 失败原因计数来自 eval 日志,不是来自 dump。**
对 `/workspace/logs/p2-bopgrasp60.log` 做 `grep -c` 得到 96 和 16;
同样的 grep 打在 dump 上得到 33 和 8。
工具每次调用记好几行日志,所以**日志数的是内部尝试次数,
而 dump 里是模型实际收到的东西。** 两者都真实,它们回答的是不同的问题,而 P2 把它们混了。

**两条 P6 线索因此移动**,P5/P6 应当用这些:

- 「grasp 工具在 55% 的样本上失败」实际是 **68%**(41/60)。
- 「碰撞过滤主导 6 倍」在模型看到的东西里是 **4.1 倍**(33 对 8),不是 6 倍。
  方向相同,但幅度更弱。
- succeeded/errored 的分数拆分必须按 19/41 重算。注意这需要
  `analysis/analyze_grasp_result.py`:bopgrasp 上的 `score` 是 RL 的 NCE 指标
  (越低越好,均值约 2.0),**不是** MACE/SR。

**这里没有任何东西改变一个被报告的准确率。** 这些只是诊断统计量,
而 P2 的头条数字全部重现:blinkdepth 108 / 107 / 111、cvb2drelation 93.75%、
robospatial Vacant 56.25%。

---

## 8. Phase V2:按当前配置,策略装不进一张 40 GB 的卡

blinkdepth 根本跑不起来。四次失败的尝试,全都死在同一个地方——
`sglang_rollout.py:213`,`get_named_tensor_buckets` -> `tensor.clone()`,
发生在 `rollout_mode()` 把 FSDP 权重推进 sglang 引擎的过程中:

    torch.OutOfMemoryError: Tried to allocate 20.00 MiB.
    GPU 0 has a total capacity of 39.49 GiB of which 19.12 MiB is free.
    Process A has 31.42 GiB in use.   Process B has 8.00 GiB in use.

策略是独占一张卡的——`SGLangHttpServer ... cuda_visible_devices='3'`,
而 GPU 轨迹显示工具全程稳定在 0/1/2 上。这不是工具共卡。**是策略自己撑爆了 40 GB。**

### 8.1 三个不是答案的旋钮

记录下来,因为每一个都看起来显而易见,而每一个都花掉了一次运行:

| 尝试 | 结果 |
|-------|--------|
| `gpu_memory_utilization` 0.5 -> 0.35 | 失败**逐字节相同**,连 "19.12 MiB is free" 都一样 |
| `actor.fsdp_config.param_offload=True` | 31.46 -> 31.42 GiB,即什么也没省 |
| `update_weights_bucket_megabytes` 2048 -> 512 | 释放了它自己约 1.5 GB 的瞬时占用;卡照样填满(GPU3 峰值 39.5 GB) |

`gpu_memory_utilization` 是直觉上的嫌疑人,而它在这里无关:
**等到权重推送运行时,TorchMemorySaver 已经释放了 KV cache**,
所以 sglang 只剩 8.00 GB 的 bf16 权重,它的静态占比不再适用。
`param_offload` 失败的原因类似——不管参数平时放在哪,
`update_weights` 都必须把它们收集回 GPU。

这个模式是这个项目反复在学的那一条:
**一个参数的*名字*暗示它管着那块分配;而*执行顺序*作了相反的决定。**

### 8.2 真正的原因,以及修法

verl 的 FSDP engine 把 `model_dtype` 默认成 **fp32**
(`verl/workers/config/engine.py:238`)。策略有 4.066 B 参数,
所以 master 副本是 **16.3 GB**;bf16 下是 8.1 GB。
那是卡上唯一一块大到值得管的分配。

    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16

GPU3 的峰值从 **39.5 GB 降到 25.0 GB**,运行完成。

**为什么这对评测在数值上是惰性的**——是检查过的,不是假定的:

- `siyich/spacetools-ckpt` 里的每一个张量都以 **BF16** 存储:825 个张量全是,
  8.132 GB 对 4.066 B 参数 = **2.00 字节/参数**,从 safetensors 头部读出。
- verl 把它升到 fp32 master,再降回来交给 sglang,而 sglang 跑 bf16
  (`rollout dtype: bfloat16`)。`bf16 -> fp32 -> bf16` 是恒等变换,
  所以 fp32 master 原样往返了它拿到的那些比特。
- `trainer.val_only=true`:没有优化器步骤,没有梯度累积。
  而那正是 fp32 master 存在的两个理由。

**这次往返随后是被测量的,而不只是被论证的。** 两条路径都在 CPU 上
施加到真实 checkpoint 上——`t.to(fp32).to(bf16)` 对比 `t`——对每一个张量:

    tensors compared    825
    elements compared   4.066 B
    non-bf16 tensors    0
    tensors differing   0
    elements differing  0
    max abs difference  0.000e+00      VERDICT: bit-identical

**而且 FSDP 模型在 eval 期间从不做前向。** 这次运行自己的配置写着
`calculate_log_probs: False`,而生成 batch 携带 `recompute_log_prob: False`;
生成完全是 sglang 的事。FSDP engine 在 `val_only` 下唯一的工作是持有权重并交出去。
既然它交出去的东西两种情况下逐位相同,`model_dtype` 就够不到结果。

这条链是靠测量而不是推断闭合的。这次运行落在 **109/124 = 87.90%**、
在 106-112 的带内,与它一致但不是论证的依据——
从一个 6 样本宽的带里抽一次,本来也分不出「惰性」和「小效应」。

> **这套推理不延伸到训练。** 如果 P7 真的运行 `run_rl.sh`,
> fp32 master 就有意义,这个 flag 绝不能带过去。
> `run_eval.sh` 在使用处的注释写明了这一点。

### 8.3 为什么 P0-P3 从没看见它

48 GB 吸收了 fp32 master,还剩约 8 GB。每一次 A6000 运行都带着它。
**这是这次迁移中第二次,A6000 多出来的 8 GB 在静默地承重**——
第一次是 `p2_tool_chain.py` 的陈旧分数(§6.3)。
两者在旧硬件上都没有以警告的形式出现;在 40 GB 上两者都是硬失败。

### 8.4 另一个选项,以及它为什么被否决

把 §8.1 里那三个旋钮叠在一起,大概能抠回最后约 1 GB,从而保住 fp32。
那意味着三处效果边际且理解不足的偏离、留下一张毫无余量的卡,
换来的是保住一份 `val_only` 从不读取的 master 副本。
**一处数值惰性的改动加 14.5 GB 余量是更划算的交易,而且对 P5 来说容易解释得多。**

---

## 9. Phase V3:Molmo 的峰值,以及 handoff 那个悬念的答案

`P4_HANDOFF.md` 给这台机器留了一个问题:P2 表里的 33.0 GB
是 Molmo 的静息占用还是它的观测峰值?如果是静息,
那么 `robospatial` 2.76 MP 图像上的尖峰就没被测过,7 GB 的余量可能不够。

### 9.1 规定的测试回答不了它

32 样本的 `robospatial` 压力运行干净完成——零 OOM、`--strict` 退出 0、
18/32 = 56.25%,**与 A6000 的数字完全相同**。峰值:

| GPU | 内容 | 峰值 | 占 40 GB |
|-----|----------|-----:|---------:|
| 0 | RoboRefer + SAM2 x2 | 20.8 GB | 52% |
| 1 | Molmo | 30.4 GB | 76% |
| 2 | DepthPro + bbox + grasp | 9.3 GB | 23% |
| 3 | policy | 27.4 GB | 69% |

但工具直方图读出来是 `{'roborefer': 32}`——**`vlm` 从未出现**。
全部 32 个样本都用 `roboreferx1@2t`。handoff 正好预料到了这一点:
如果 `vlm` 从未被调用,Molmo 就不可能在那里出现尖峰,
GPU1 的 30.4 GB 只是它的静息大小。压力测试证明的是这次*运行*是安全的;
它对 Molmo 什么也没说。

### 9.2 直接测量它

于是绕开 eval 直接调用 Molmo,用每个 benchmark 里最大的那张图,每 50 ms 采样它那张卡:

| 场景 | 图像 | Molmo 峰值 | 占 40 GB |
|------|------:|-----------:|---------:|
| blinkdepth | 0.19 MP | 35321 MiB | 86.2% |
| bopgrasp | 0.92 MP | 35321 MiB | 86.2% |
| boppose | 2.07 MP | 35321 MiB | 86.2% |
| robospatial | 2.76 MP | 35321 MiB | 86.2% |
| cvb2drelation,最大的 | 3.63 MP | 35321 MiB | 86.2% |

**34.49 GB,在 19 倍的图像尺寸跨度上,一个 MiB 的变化都没有。**

### 9.3 为什么 —— crop 网格会饱和

Molmo 的预处理器发出的是一个有界的多裁剪张量,而不是一张被缩放的图:

    blinkdepth  0.19 MP  ->  images torch.Size([ 5, 576, 588])
    robospatial 2.76 MP  ->  images torch.Size([13, 576, 588])

crop 数随分辨率上升,然后**在 13 封顶**(12 块 tile 加一张全局视图)。
超过之后,更大的图什么也不增加。而且即使在饱和状态,
13 x 576 x 588 的 activation 成本相对于 33 GB 常驻 fp32 权重,
低于 `nvidia-smi` 的分辨率——这就是为什么这些峰值是逐字节相同而不只是接近。

### 9.4 答案

「静息还是峰值」这个问题有一个比两者都好的答案:**对 Molmo 而言两者是同一个数字**,
因为它的输入在模型看到之前就已经被归一化成固定大小的 crop 网格。
34.49 GB 是横跨全部九个 benchmark 里每一张图的硬上限,
在那张卡上**永久留出 5.51 GB**。

所以那个看起来与 P2 那次 OOM 级联的 84% 可比的 86.2% 占用率,其实完全不可比。
那 84% 是一个*可变*负载——多个工具共用一张卡、同时出现尖峰。
这一个是常量负载,有已证明的上限,而且没有第二个租户。

V3 的 gate——「峰值低于约 37 GB:可以」——在 34.49 GB 上通过。

**仍然要检查每一个 benchmark。** 这条只界定了 Molmo,不包括其它工具,
每个 benchmark 之后跑 `parse_dump.py --strict` 仍然是规矩。

### 9.5 九个 benchmark 的图像尺寸,存档用

从 parquet 测得;总数确认了 handoff 的 2121 样本规模。

| benchmark | n | 平均 MP | 最大 MP |
|-----------|--:|--------:|-------:|
| robospatial | 350 | 2.76 | 2.76 |
| boppose | 60 | 2.07 | 2.07 |
| bopgrasp | 60 | 0.92 | 0.92 |
| cvb3ddepth | 600 | 0.79 | 0.79 |
| cvb2drelation | 650 | 0.44 | **3.15** |
| refplacement | 100 | 0.31 | 0.31 |
| reflocation | 100 | 0.30 | 0.31 |
| refunseen | 77 | 0.28 | 0.31 |
| blinkdepth | 124 | 0.18 | 0.25 |
| **合计** | **2121** | | |

`cvb2drelation` 值得注意:平均值低,却拥有整套里最大的单张图像。
它也是 P4 顺序里的第二个 benchmark,所以它很早就走通了大图路径——
而这正是「从便宜到贵」这个排序的意义。

---

## 10. 操作陷阱

这些都不是对任何东西的改动。它们是在 Phase M 和 V 中真实花掉时间的陷阱,
写下来是因为对下一个接手的人来说,每一个都还会在那里。

### 10.1 `set -u` 会破坏 conda 激活

    $ set -euo pipefail; conda activate spacetools-tool-graspgen
    .../etc/conda/activate.d/~cuda-nvcc_activate.sh: line 48:
        NVCC_PREPEND_FLAGS: unbound variable

环境自己的激活钩子会读可能未设置的变量,所以它们不是 `-u` 安全的。
在任何激活这里的 conda 环境的脚本中用 `set -eo pipefail`,
或者在 activate 前后 `set +u`。**这个失败看起来像是你脚本的 bug,而它不是。**

### 10.2 `start_toolkit(detached=False)` —— 握住那个 handle

    RuntimeError: Could not find ToolRouterActor 'toolshed_router'
                  in namespace 'toolshed'

router actor 的生命周期绑定在 `start_toolkit` 返回的那个对象上。
把它丢在地上——写 `start_toolkit(...)` 而不是 `router = start_toolkit(...)`——
Ray 就会在你脚下把 router 回收掉,通常发生在几次工具调用之后,
所以它读起来像是间歇性故障,而不是第 1 行的一个错误。
`p2_tool_chain.py` 里有一条关于此事的注释;我还是走进去了。

### 10.3 `make_subset.py --out` 是目录根,不是文件

    python tools/make_subset.py robospatial 32 \
        --src /workspace/eval-benchmarks --out /workspace/tmp/rs32
    # 写出 /workspace/tmp/rs32/data/robospatial.parquet
    # 然后:  DATA_DIR=/workspace/tmp/rs32 bash run_eval.sh <ckpt> robospatial

它接受一个 benchmark 名字和一个数量,而不是输入和输出路径;
`--out` 成为 `run_eval.sh` 期望的那个 `DATA_DIR` 根——parquet 落在一个 `data/` 子目录里。
把一个 `.parquet` 路径当作 `--out` 传进去也「能用」,
产出的是 `.../robospatial.parquet/data/robospatial.parquet`。

### 10.4 GPU 显存要按 1 s 采样,不是 5 s

§8 里那次策略卡 OOM 发生在 `rollout_mode()` 的权重推送内部,
它填满卡并在**不到五秒内**死掉。
一个 5 s 的采样循环显示这张卡一直平静地停在 23 GB 直到失败,
从未记录到那次尖峰,这把第一轮诊断完全引向了错误的组件。
1 s 能抓到;§9 里对 Molmo 的直接测量用的是 50 ms。

更一般地说:**一个你采样得不够快而没看到的峰值,
与一个没有发生过的峰值是无法区分的。**

### 10.5 本项目自己的工具需要 conda 环境

`tools/make_subset.py`、`tools/parse_dump.py` 以及任何碰 parquet 的东西
都需要 `pandas` / `pyarrow`,而系统的 `python3` 没有。
在 `conda run -n spacetools-rl` 下运行它们,**要传脚本路径,绝不要用 heredoc**(§10.6)。
`parse_dump.py` 按设计只用标准库,能在五个环境的任何一个里跑;parquet 工具不行。

### 10.6 `conda run` 不转发 stdin —— 而且是静默的

这是 Phase M 和 V 中最大的一处时间浪费来源,因为它看起来不像错误:

    $ echo "print('hi')" | conda run -n spacetools-rl python -
    $                                    # 没有输出。退出码 1。没有任何消息。

    $ echo "print('hi')" | /workspace/envs/spacetools-rl/bin/python -
    hi

在这五个环境里,每一个 `conda run -n <env> python - <<'EOF' ... EOF`
和每一个 heredoc 喂进去的一行命令都返回**什么都没有**。
它以 1 退出,所以 `set -e` 会抓住;但交互式运行时,
你只会看到一个空结果,然后以为自己的脚本什么都没打印。

有两个办法可行:把片段写进文件并传路径(`conda run -n <env> python /tmp/x.py`),
或者直接调用环境的解释器(`/workspace/envs/<env>/bin/python`)。
解释器路径完全跳过 `conda run` 的包装,这也意味着它跳过环境的激活钩子——
对纯 Python 没问题,对任何需要 `LD_LIBRARY_PATH` 的东西就不行了
(见 `env.sh` 里的 `st_activate`)。

### 10.7 `pointnet2_ops` 的 kernel 失败会杀进程,而不是抛异常

`_ext-src/include/cuda_utils.h:30`:

    #define CUDA_CHECK_ERRORS()                    \
      ... fprintf(stderr, "CUDA kernel failed : %s\n" ...);  \
          exit(-1);

是 `exit(-1)`,不是抛出的异常。围绕 `furthest_point_sample()` 的
`try/except` 什么都抓不到;解释器已经没了。在一个 Ray actor 里面,它会把 actor 带走。

**这一点值得精确知道,恰恰因为它与这个项目反复警告的那个 OOM 失败模式相反。**
工具内部的 out-of-memory 会以一个错误*字符串*回来,
被模型当作一次工具响应读走,而 benchmark 以一个悄悄错掉的数字结束。
`pointnet2_ops` 里的架构不匹配做不到那样——**它会杀掉 actor。**
所以 sm_86 这个问题即使活到了 P4,也会以响亮的方式失败。
「静默失败」这份担忧属于 OOM,不属于这一类。

### 10.8 `pkill -f <pattern>` 可能匹配到自己

`pkill -f "nvidia-smi --query-gpu"` 匹配到了发出这条命令的那个 shell 自身,
因为 `-f` 匹配整条命令行,而 pattern 就在里面。
shell 在命令中途死掉,它本来要启动的工作从未开始,
这看起来正像是你试图运行的那个东西静默失败了。
把 watcher 写进脚本文件,并按记录下来的 PID 杀它。

---

## 还有什么可能移动一个数字

复核之后,十五处偏离里只有两处有可能,而且两处都无法避免:

| 偏离 | 为什么它可能有影响 | 为什么它无法避免 |
|-----------|--------------------|--------------------------|
| cudnn 9.16.0.29(违反 torch 的 `==9.10.2.21` pin) | 不同的卷积结果正是 pytorch#168167 所讨论的东西 | sglang 在启动时读 `torch.backends.cudnn.version()`,低于 9.15 就拒绝运行。`docs/SETUP.md:88` 记载了同样的要求和同样的版本号 |
| `spacetools-rl` 里的 numpy 2.x(verl 声明 `<2.0.0`) | 没有已知影响;那个 pin 是陈旧的元数据 | Ray 无法把工具环境里的 numpy-2 数组反序列化进一个 numpy-1 的 driver(`ModuleNotFoundError: numpy._core.numeric`),而且 toolshed 自己要求 `>=2.0.0` |

其余的要么是硬阻塞,要么是已验证的 no-op,要么是调度。

**排在这两者之上的,是一个根本不算偏离的东西:运行间的非确定性。**
在完全相同的配置下把 blinkdepth 跑三次得到 108 / 107 / 111 个正确,
极差 3.23 pp——比任何单一偏离可能值得的都要大。见 `P2_RESULTS.md`。

---

## 我们引入的操作性约束

**更正,2026-08-27:** 四个仓库现在**确实**都跟踪一个 remote(`mine/`,指向个人 fork),
所以下面那段是陈旧的,`install_graspgen.sh` 的 `git pull` 不会再因那个理由中止。
而 A100 主机上的真实情况不同,而且更糟:**这台机器上根本不存在 GitHub 凭证**——
它们在上一台主机的 container 磁盘上,而 recycle 会擦掉。
`git fetch`/`git pull`/`git push` 全部以
`could not read Username for 'https://github.com'` 失败。
什么都没丢(工作树随卷过来了),但**在提供 token 之前,Phase D 无法 push**。

四个仓库现在都在一个本地分支 `repro-4xa6000` 上,**没有上游跟踪**。
这只在一个地方要紧:`install_graspgen.sh:79` 在 GraspGen 目录已存在时运行 `git pull`,
配上第 17 行的 `set -e`,脚本会在那里中止,
因为在未跟踪的分支上 `git pull` 会以 "no upstream configured" 失败。
如果那个脚本需要重跑,先 checkout `main`:

    git -C SpaceTools-Toolshed/GraspGen checkout main

其余的不受影响:`install_roborefer.sh` 在目录已存在时跳过 clone 而不是 pull,
而 Toolshed 和 SpaceTools-RL 没有任何东西会 pull。
