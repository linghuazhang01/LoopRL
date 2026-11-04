# LoopRL

在 looped language model（Ouro）上做 RL 的实验代码。目前包含两个 baseline 和一个诊断工具：

- **GRPO baseline**：标准 verl GRPO，即 RLTT 论文中的 GRPO 对照（`trainer: grpo`）。
- **RLTT baseline**：Williams & Tureci, *Prioritize the Process, Not Just the Outcome: Rewarding
  Latent Thought Trajectories Improves Reasoning in Looped Language Models*, arXiv 2602.10520
  （`trainer: rltt`）。
- **逐轮梯度探测**（`rltt_baseline/diagnostics/`）：把 Ouro 四次循环共用参数的梯度拆成每一轮的贡献，
  统计范数、轮间余弦、抵消比和噪声基线。

原作代码（[jonwill8/RLTT](https://github.com/jonwill8/RLTT)）以 git submodule 的形式固定在
commit `06189850`，不做任何修改。本仓库只补上原作缺少的部分和我们需要的功能。

> 状态（2026-10-23）：CPU 单元测试和 tiny 随机模型测试通过；在 3×H200 上，RLTT 与 GRPO 各跑通
> 2 步 smoke run（GRPO 含 checkpoint 与 `hf_model` 保存，保存的模型可直接加载）；环境、数据、
> 模型三个准备脚本在服务器上从零跑通。逐轮梯度探测尚未在真实模型上运行。
> 2026-10-24 训练数据改为 SAPD 的 DeepMath 数据、prompt 2,048 / 生成 16,384 之后，只做了 CPU 测试和数据脚本实跑，
> GPU smoke run 尚未在新设置下运行。所有数字都不是实验结论。

## 本仓库补充了什么

| 模块 | 作用 |
|---|---|
| `rltt_baseline/ouro_rltt/` | 原作训练时加载 `Ouro-*-Thinking-RLTT` 模型，要求 forward 返回逐轮 log-prob、熵和 `exit_pdf`，但原仓库没有提供这份模型代码。这里按原作 actor 的调用接口重新实现 |
| `rltt_baseline/prepare_model.py`、`fetch_model.py` | 下载固定 revision 的 Ouro-2.6B-Thinking（或 `--model 1.4B` 时的 Ouro-1.4B-Thinking）并核对 SHA256，再生成训练用模型目录：权重软链接，官方 `modeling_ouro.py` 改名为 `modeling_ouro_base.py` 并加 transformers 4.57 兼容修改 |
| `rltt_baseline/prepare_data.py`、`prepare_math_train.py`、`verl_data.py` | 下载 SAPD 数学训练集（DeepMath-103K 难度 ≥ 6，57,046 题，核对 SHA256）、MATH 训练集（7,500 题）和 MATH-500；`train_file` 为 verl 格式 parquet 时原样使用其中的 prompt |
| `rltt_baseline/config.py`、`launch.py` | YAML 配置（`trainer: rltt / grpo`），支持命令行 dot-list 覆盖；`verl_overrides` 可以改任意 verl 配置 |
| `rltt_baseline/decode_last.py`、`vllm_decode_last.py` | 可选的 rollout KV 布局 `decode_last`（`rollout.kv_layout`，默认关闭）：prompt 保留每一轮的 KV，生成的 token 只保留最后一轮的 KV。只改变 vLLM rollout，训练 forward 不变 |
| `rltt_baseline/metrics.py` | driver 端的推理 RL 指标（训练正确率、组内全对/全错/混合比例、截断率、正确与错误回答长度等） |
| `rltt_baseline/workers.py` | RLTT 训练时额外记录逐轮熵、期望退出轮次和总 loss，原作的 loss 计算不变 |
| `rltt_baseline/plc.py`、`plc_workers.py` | PLC（Posterior Loop Credit，`plc.enabled`，默认关闭）：把 RLTT 的逐轮权重换成按 token 的后验权重 `softmax_t(log q_t + γ·log P_old^(t)(y_j))`，逐轮 PPO ratio，可选 gate 损失；γ=0 退回 RLTT，λ=0 退回 GRPO |
| `rltt_baseline/loop_audit.py`、`grpo_workers.py` | 逐轮分布审计（`audit.loop_distribution`，所有配置都打开）：每个生成 token 上各轮分布的 KL、熵和 top-k，按步写入 `audit/` |
| `rltt_baseline/diagnostics/` | 逐轮梯度探测：rollout 生成（`rollouts.py`）与探测（`probe.py`），GRPO loss 与 verl 0.6.1 逐项一致 |
| `configs/rltt_env.lock.txt`、`scripts/setup_env.sh` | 训练环境（verl 0.6.1、vLLM 0.12.0、torch 2.9.0、transformers 4.57.3）的锁定文件和安装脚本 |

对官方 Ouro 代码只做了一处修改：transformers 4.57 把 `check_model_inputs` 改成了 decorator
factory，官方 `modeling_ouro.py` 直接加载会报 `wrapped_fn() got an unexpected keyword argument`。
`prepare_model` 在生成的 `modeling_ouro_base.py` 中加了一个兼容 wrapper，同时支持新旧两种接口。

## 目录

```
rltt_baseline/              适配代码（Python 包）
configs/grpo_baseline_8gpu.yaml  GRPO 8 卡训练配置（Ouro-2.6B-Thinking）
configs/grpo_baseline_1p4b_8gpu.yaml  GRPO 8 卡训练配置（Ouro-1.4B-Thinking，生成 16K，每步更新一次）
configs/rltt_baseline.yaml  RLTT 论文 Table 6 口径的配置（Ouro-2.6B-Thinking，与 grpo_baseline_8gpu 对齐）
configs/rltt_baseline_1p4b_8gpu.yaml  RLTT 8 卡训练配置（Ouro-1.4B-Thinking，与 GRPO 1.4B / PLC 配置对齐）
configs/plc_posterior_exit_gate_1p4b_8gpu.yaml  PLC（γ=1，λ=1，exit 先验，gate 损失）on Ouro-1.4B-Thinking，8 卡
configs/plc_posterior_exit_gate_2p6b_8gpu.yaml  同上，Ouro-2.6B-Thinking（与 1.4B 版只差模型）
configs/rltt_env.lock.txt   训练环境锁定文件
scripts/                    setup_env / prepare_data / prepare_model / run_train / smoke_grpo
scripts/deepmath_filter_portable/  训练集数据过滤的便携采样脚本（见下文“数据过滤采样”）
scripts/deepmath_filter_results/   过滤采样的判分与训练子集导出（见该目录的 README）
data/deepmath_level6_filter12k_passes1to3_no_yesno_01_truefalse/  过滤后的训练集，2,865 题（见下文“过滤后的训练集”）
scripts/deepmath_filter_a100/      12K 过滤采样 A100 分担时的脚本，只作记录（见该目录的 README）
tests/                      单元测试与 tiny 模型集成测试
third_party/RLTT            原作仓库（submodule，固定 commit）
third_party/ouro_source     官方 Ouro modeling 文件（Apache-2.0，见该目录 README）
```

## 使用

在本仓库工作的 agent 须遵守 `AGENTS.md`：只用 14,684 条训练集、训练 seed 固定 42、每次启动训练前报告全部参数。

```bash
git clone --recurse-submodules https://github.com/linghuazhang01/LoopRL.git
cd LoopRL
W=/path/to/workdir        # venv、数据、模型、输出都放这里
```

1. **环境**（需要 [uv](https://docs.astral.sh/uv/)；用 uv 管理的 CPython 3.12，自带 Triton 需要的 `Python.h`）：
   ```bash
   bash scripts/setup_env.sh $W/.venv
   export LOOPRL_VENV=$W/.venv
   ```
2. **数据**（当前实验用 14,684 条合并过滤训练集，按下文“过滤后的训练集”下载并校验；
   以下脚本准备上游源数据和 MATH-500，不会生成这份合并过滤集）：
   ```bash
   bash scripts/prepare_data.sh $W/data      # 默认：SAPD 数据 + MATH-500，结束时打印 export 行
   # 另加 RLTT 论文的 MATH 训练集：--datasets deepmath_level6,math500,math_train
   ```
3. **模型**（下载 `ByteDance/Ouro-2.6B-Thinking` @ `f1edd81e` 并核对 SHA256，生成训练用目录；
   已有官方目录时加 `--source <dir>` 跳过下载）：
   ```bash
   bash scripts/prepare_model.sh $W/models
   bash scripts/prepare_model.sh $W/models --model 1.4B   # Ouro-1.4B-Thinking @ 3aaa2224
   ```
4. **路径变量**：
   ```bash
   export LOOPRL_MODEL_DIR=$W/models/Ouro-2.6B-Thinking-RLTT
   export LOOPRL_OURO_DIR=$W/models/Ouro-2.6B-Thinking        # 只有 RLTT 配置用到
   export LOOPRL_TRAIN_FILE=$W/data/deepmath_14684/data/train.parquet  # 按下文从 HF 下载
   export LOOPRL_MATH500_JSONL=$W/data/math500.jsonl
   export LOOPRL_OUTPUT_DIR=$W/outputs/grpo_run1
   ```
5. **smoke run**（任意 ≥ 2 张卡，2 步、512 tokens，只验证流程）：
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 bash scripts/smoke_grpo.sh
   ```
6. **训练**（8 卡 GRPO；RLTT 用 `configs/rltt_baseline.yaml`（2.6B）或 `configs/rltt_baseline_1p4b_8gpu.yaml`（1.4B））：
   ```bash
   bash scripts/run_train.sh configs/grpo_baseline_8gpu.yaml
   bash scripts/run_train.sh configs/grpo_baseline_8gpu.yaml args.max_steps=100 wandb.run_name=try1
   # 1.4B：LOOPRL_MODEL_DIR=$W/models/Ouro-1.4B-Thinking-RLTT，另设一个 LOOPRL_OUTPUT_DIR
   bash scripts/run_train.sh configs/grpo_baseline_1p4b_8gpu.yaml
   ```
   在 Slurm 集群上必须在 allocation 内运行；脚本检测到 `sbatch` 而没有 `SLURM_JOB_ID` 时会拒绝。

### 配置对照组

配置按对照组组织，组内文件除方法本身（`trainer`、loss、`plc` 段、wandb 命名）外每个参数都必须相同，
这样三种方法可以直接对比：

| 组 | GRPO | RLTT | PLC |
|---|---|---|---|
| 1.4B，8 卡，每步一次更新 | `grpo_baseline_1p4b_8gpu.yaml` | `rltt_baseline_1p4b_8gpu.yaml` | `plc_posterior_exit_gate_1p4b_8gpu.yaml` |
| 2.6B，8 卡，每步一次更新 | `grpo_baseline_8gpu.yaml` | `rltt_baseline.yaml` | `plc_posterior_exit_gate_2p6b_8gpu.yaml` |

测试 `test_matched_configs_share_all_non_method_settings` 检查这一点（允许不同的键和改配置的规则见
`CLAUDE.md`）。改一个共享超参要改整组文件；临时实验用命令行覆盖，不要另存配置。

随机种子全部锁定 42：`args.seed: 42`，且 `verl_overrides.data.seed: ${args.seed}`（verl 0.6.1 的 dataloader
打乱种子默认为空，不从 `args.seed` 派生）；vLLM 引擎 seed 在 verl 0.6.1 不可配置（固定 0，各方法相同）。
诊断脚本 `rltt_baseline.diagnostics.rollouts` / `probe` 和原作评估脚本的默认 seed 也都是 42。

### GRPO 8 卡配置要点（`configs/grpo_baseline_8gpu.yaml`）

数据与长度按 SAPD，优化超参数按 RLTT 论文 Table 6（论文中 GRPO 与 RLTT 共用同一套设置），
只有梯度累积改为 1；与 `configs/rltt_baseline.yaml` 一致：

- 模型 Ouro-2.6B-Thinking 全参数，4 次循环。
- 上游数据：SAPD 的数学训练集，即 G-OPD 发布的 DeepMath-103K 难度 ≥ 6 子集（57,046 题，
  `icemoon28/MOPD-Training-Data` 中的 `DeepMath-103K/train_filtered_level6.parquet`）。
  当前实验使用下文的 **14,684 条合并过滤训练集**；GRPO、RLTT、PLC 都用同一个
  `LOOPRL_TRAIN_FILE` 指向它。
  prompt 用文件自带的：题目后接 “Please reason step by step, and put your final answer within
  \boxed{}.”，单条 user 消息，再套 Ouro 的 chat template。奖励沿用 RLTT：取最后一个 `\boxed{}`，
  用 `math_verify` 判断与标准答案是否等价，0/1。
- prompt 上限 2,048 tokens（按 Ouro chat template，最长的题 743 tokens，没有被过滤的），
  生成上限 16,384 tokens；每卡 token 上限 32,768（须 ≥ 2,048 + 16,384）。
  vLLM 每步 token 预算 `max_num_batched_tokens` 也是 32,768：verl 默认 8,192 小于
  `max_model_len` 18,432，开着 chunked prefill 时 vLLM 会拒绝启动，launcher 在 dry-run 阶段就会报错。
- 每步 32 题 × 8 条 = 256 条 rollout，temperature 1.0；梯度累积 1（论文 2），
  即每步做一次优化器更新（mini-batch = 32 题），共 140 步 = 140 次更新。
- AdamW8bit，lr 1e-6（constant，前 10% 步 warmup），betas (0.9, 0.99)，weight decay 0.1，
  梯度裁剪 0.1；KL 关闭（`beta: 0`、`use_kl_loss: false`、`kl_loss_coef: 0`，所有配置相同）；
  clip 0.2，token-mean。
- 每 10 步存 checkpoint（含 `hf_model`，供逐轮梯度探测）；训练中不做验证（论文如此），
  需要时加 `args.no_validation=false`（MATH-500，用 RLTT prompt）。
- 显存：Ouro 每次循环有独立的 KV cache（48 层 × 4 次 = 192 层），每 token 约 1.57 MB。
  按此估算，vLLM 占 50% 显存时每张 H200 约能放 4 万 tokens 的 KV，即同时约 2 条满长度序列，
  长回答多时 rollout 会很慢（实际数值以 vLLM 日志为准）。

原作 `grpo_train.py` 与论文有四处不一致，配置里用 `verl_overrides` 改正：

| 项 | 原作脚本 | 本配置 |
|---|---|---|
| 每步题数 | `data.train_batch_size = 题数 × 8`，verl 当作题数（256 题） | 32 题 |
| 梯度裁剪 | `max_grad_norm` 只传给未启用的 critic，actor 用 verl 默认 1.0 | 0.1 |
| Adam betas | (0.9, 0.999)（RLTT 脚本相同） | (0.9, 0.99) |
| KL | 加在 reward 上；组内奖励全同时，KL 噪声会给这组产生非零 advantage | 不进 reward；KL loss 也关闭（见 `CLAUDE.md`） |

KL 一律不放进 reward：最终 verl 配置的 `algorithm.use_kl_in_reward` 为 true 时，launcher 拒绝启动
（GRPO 原作默认就是 true，所以配置里必须显式写 `false`）。

原作 GRPO 脚本总会注册参考模型，verl 0.6.1 只要注册就会加载它并每步多做一次前向。KL loss 也关闭时，
launcher 在构造 trainer 前去掉这个角色，GRPO 就和 RLTT / PLC 一样不加载参考模型；dry-run 输出和
`rltt_baseline_resolved.json` 的 `reference_policy` 字段写明是否去掉。重新打开 KL loss 时参考模型照常加载。

`configs/rltt_baseline.yaml` 同样改正了每步题数和 betas（RLTT 脚本本身已有梯度裁剪 0.1）。
warmup 长度按步数算（140 × 10% = 14），verl 的学习率调度器每次更新前进一次；梯度累积 1 时步数等于
更新次数，warmup 就是前 14 次更新（10%）。若以后改回累积 2，warmup 会占到更新数的 20%。
- 原作 `grpo_train.py` 会**原地修改**模型目录里的 `modeling_ouro.py`。launcher 只接受
  `prepare_model` 生成的目录，拒绝官方目录，避免改动共享的模型文件。

### GRPO 1.4B 8 卡配置（`configs/grpo_baseline_1p4b_8gpu.yaml`）

与 2.6B 配置只差模型（Ouro-1.4B-Thinking；RLTT 论文两种规模用同一套设置），其余（数据、prompt 2,048 /
生成 16,384、每步 32 题 × 8 条、梯度累积 1、mini-batch 32 题、140 步 = 140 次更新、优化器、KL、
checkpoint、显存参数）都相同，测试 `test_grpo_1p4b_config_matches_2p6b_except_model` 检查这一点。
每次更新用 256 条 rollout，GRPO 只跑 1 个 epoch；warmup 10% 即前 14 次更新。
测试 `test_mini_batch_is_one_accumulated_update` 检查每份配置的 mini-batch 都等于“每步题数 × 累积步数”。

- 模型 `ByteDance/Ouro-1.4B-Thinking` @ `3aaa2224`，`prepare_model.sh --model 1.4B` 下载并核对
  `model.safetensors`、`modeling_ouro.py`、`configuration_ouro.py`、`config.json` 的 SHA256。
  它的 `modeling_ouro.py` 与 2.6B 只差空白；`configuration_ouro.py` 少一行 `self.pad_token_id = 2`，
  这行在 2.6B 中也会被 `PretrainedConfig.__init__` 覆盖，所以两者代码行为相同。chat template 相同。
- 24 层 × 4 次循环，每 token 的 KV 约 0.79 MB，是 2.6B 的一半；同样 50% 显存时 vLLM 能放的序列约多一倍。
- 两份配置都从 `LOOPRL_MODEL_DIR` 读模型。1.4B 配置写了 `model.num_hidden_layers: 24`，
  launcher 训练前核对模型目录的 `config.json`，指向 2.6B 目录（48 层）时拒绝启动。
- wandb 实验名由 verl 按模型目录名生成（`GRPO-Ouro-1.4B-Thinking-RLTT`），tag 为 `ouro-1.4b-thinking`。

### 数据过滤采样（`scripts/deepmath_filter_portable/`）

GRPO 数据过滤：用 Ouro-1.4B-Thinking 在 D4 上对训练集每题采 4 次，过滤依据是 4 次中答对几次。这里是其中 `rest45k` 批次的便携采样脚本，覆盖随机 12,000 题以外的全部 45,046 题，共 180,184 条样本。脚本只依赖 vLLM，不依赖 `rltt_baseline`，可以单独拷到别的机器上。

仓库不含 `rest45k` 的题目文件。先用 CPU 从训练 parquet 生成，需要 numpy、pandas、pyarrow：

```bash
python scripts/prepare_deepmath_level6_sample.py --source "$LOOPRL_TRAIN_FILE" \
    --n 12000 --seed 20261028 --out /tmp/deepmath_filter12k
python scripts/prepare_deepmath_level6_sample.py --source "$LOOPRL_TRAIN_FILE" --rest \
    --out scripts/deepmath_filter_portable/data \
    --exclude /tmp/deepmath_filter12k/sample12000.jsonl
```

`sample.py` 启动时核对题目文件的 SHA256（`d942f80b…a052`），不一致会拒绝运行。之后在 `scripts/deepmath_filter_portable/` 目录下按该目录的 README 操作，开始前先跑 `--smoke`。

### 过滤后的训练集

当前实验使用 **14,684 条**的合并过滤版本，HF 私有数据仓库为
[`icemoon28/LoopRL-DeepMath-14684`](https://huggingface.co/datasets/icemoon28/LoopRL-DeepMath-14684)。
它由 12K 批次筛选出的 2,865 条与 rest45K 批次筛选出的 11,819 条合并，保持原有行序和内容。
两批次的过滤、判分口径及已知例外记录在仓库的 `source_manifest.json`；完整文件清单和校验值见
`manifest.json`、`SHA256SUMS`。

- split：`train`；HF 文件：`data/train.parquet`；条数：`14,684`；大小：`2,023,592` 字节。
- schema：`data_source`、`prompt`、`ability`、`reward_model`、`extra_info`，与现有 verl 训练入口兼容。
- Parquet SHA256：`353b572a7c3a492b1e50d87e4d5195a461950018331c12258ff704d60c480b41`。
- 固定数据 revision：`6e0be984159c007d0abd3d717a55617774390907`。

历史运行使用的 JuiceFS 文件如下，HF 保存的是它的逐字节副本：

```text
/mnt/tidalfs-awsatl01-ren/dataset/pai/haishan1/looprl-work/data/deepmath_level6_train_merged_12k_rest45k/train_filtered_level6_passes1to3_merged_12k_rest45k.parquet
```

数据仓库为 **Private**，下载账号须有访问权限。先在已有 `huggingface_hub` 环境中执行
`hf auth login`，或使用已安全配置的 `HF_TOKEN`。以下命令沿用上文的工作目录 `W`；在训练 Pod 中，
`W` 必须位于已验收的 JuiceFS `/mnt/tidalfs-awsatl01-ren/dataset/pai/sz/zlh` 下；在本机使用本机目录。

```bash
export LOOPRL_DATA_DIR=$W/data/deepmath_14684
hf download icemoon28/LoopRL-DeepMath-14684 \
    --repo-type dataset \
    --revision 6e0be984159c007d0abd3d717a55617774390907 \
    --local-dir "$LOOPRL_DATA_DIR"

(cd "$LOOPRL_DATA_DIR" && sha256sum -c SHA256SUMS)
export LOOPRL_TRAIN_FILE="$LOOPRL_DATA_DIR/data/train.parquet"

python - "$LOOPRL_TRAIN_FILE" <<'PY'
import sys
import pyarrow.parquet as pq

rows = pq.ParquetFile(sys.argv[1]).metadata.num_rows
assert rows == 14684, rows
print(f"Verified {rows} training rows")
PY
```

macOS 可将校验命令替换为 `shasum -a 256 -c SHA256SUMS`。下载并校验后，训练入口通过
`LOOPRL_TRAIN_FILE` 读取这份文件，三种方法使用相同路径。

#### 仓库内的 12K 过滤分量（2,865 条）

`data/deepmath_level6_filter12k_passes1to3_no_yesno_01_truefalse/` 是 12,000 题过滤后留下的 2,865 题，parquet 与原训练集同 schema，可以直接用于训练：

```bash
export LOOPRL_TRAIN_FILE=$PWD/data/deepmath_level6_filter12k_passes1to3_no_yesno_01_truefalse/train_filtered_level6_passes1to3_no_yesno_01_truefalse.parquet
```

这份 2,865 条数据已纳入上述合并训练集。单独使用时，140 步 × 32 题需要 4,480 题，
它只够约 89 步不重复。规则和构成见该目录的 README。

### PLC：Posterior Loop Credit（`configs/plc_posterior_exit_gate_1p4b_8gpu.yaml`）

`plc.enabled: true` 时 launcher 用 `PLCActorRolloutRefWorker` 代替原作的 RLTT worker（仍是 `trainer: rltt`，
仍需 `use_rltt_model: true` 的逐轮模型目录）。记号：`ℓ[j,t] = log P^(t)(y_j)` 为采样 token 在第 t 轮的
log-prob，`q_t` 为退出深度的先验。

- **权重**：`w[j,t] = softmax_t(log q_t + γ·ℓ_old[j,t])`，用旧策略（`compute_log_prob` 那次 forward）的
  逐轮 log-prob 计算，detach，整个更新内固定。γ=1 是退出混合策略 `Σ_t q_t P^(t)` 的精确后验；γ=0 时
  `w = q`，就是 RLTT（先验 uniform / progressive / exit 分别对应 RLTT 的三种权重）。
- **损失**：`-agg_j[(1-λ)·PPO^(T) + λ·Σ_t w[j,t]·PPO^(t)]`，每轮一个 PPO ratio `exp(ℓ_new - ℓ_old)`，
  clip 用 verl 的 `clip_ratio` / 双重 clip；λ=0 与 verl GRPO 的 loss 逐位相等（`test_lam0_is_verl_grpo_on_the_final_loop`）。
  熵项和 KL（`low_var_kl`）只加在最后一轮，与原作相同；对照组配置里 KL 关闭，熵系数为 0。
- **gate 损失**（`use_gate_loss`，需要 `prior: exit`）：`gate_coef·agg[Â·CE(sg[w], q_new)] − gate_entropy_coef·agg[H(q_new)]`，
  即混合策略梯度中 gate 的那一项加 Ouro Stage I 的熵正则（Ouro 预训练 β=0.1，末阶段 0.05）。
  它要求模型目录没有用 `--detach-exit-pdf` 构建，launcher 训练前检查 `config.json`。
- **可选**：`eps_floor`（权重下限）、`use_is_correction`（在 loop 项上乘
  `min(Σ_t q_t·exp(ℓ_old[t] − ℓ_old[T]), is_clip)`，把自归一化权重换成普通的逐轮重要性权重）。
- **指标**：`plc/w_mean_per_loop`、`plc/eff_loops_mean`、`plc/argmax_is_final_frac`、`plc/delta_*`、
  `plc/clipfrac_per_loop`、`plc/ratio_mean_per_loop`，gate 的 `gate/q_mean_per_loop`、`gate/expected_exit_step`、
  `gate/kl_w_q_mean`、`gate/entropy_mean`；原有的 `rltt/loop_<t>_entropy`、`actor/*` 照旧。
- **批次里新增的张量**：`old_log_probs_loops` `[B,L,T]`，`prior: exit` 时还有 `old_exit_pdf` `[B,L,T]`。
  `old_log_probs` 就是前者的最后一轮。

消融用命令行覆盖：`plc.gamma=0`（RLTT-exit）、`plc.lam=0`（GRPO）、`plc.use_gate_loss=false`、
`plc.prior=uniform plc.use_gate_loss=false`（PLC-uniform）、`plc.use_is_correction=true`。
状态：只有 CPU 单元测试和 tiny 随机模型测试（`tests/test_plc.py`、`test_rltt_baseline_modeling.py`），
没有在真实权重或 GPU 上运行过。

### 配置与覆盖

配置文件的顶层键：`trainer`（`rltt` / `grpo`）、`args`（与所选原作脚本的 argparse 参数同名，
未列出的用原作默认值；拼错的键或类型不符会直接报错）、`wandb`、`verl_overrides`、`env`、
`rollout`（`kv_layout: full | decode_last`，见下节）、`model`、`audit`、`plc`（见上节）。
命令行覆盖示例：`args.learning_rate=2e-6 verl_overrides.trainer.save_freq=20 wandb.run_name=try1`。
解析后的完整配置会保存到 `<output_dir>/rltt_baseline_config.yaml`。

### decode_last rollout（可选，默认关闭）

`rollout.kv_layout=decode_last` 让 rollout 采用 Ouro 论文 §5.4.2 的 “last-step reuse”：
prefill 时 prompt 的每一轮各有自己的 KV；decode 时生成的 token 跑满 4 轮，第 u 轮读 prompt 的第 u 轮 KV、
之前生成 token 的最后一轮 KV 和自己的第 u 轮 KV，算完最后一轮后只保留最后一轮的 KV。
默认 `full` 与原版 vLLM 完全相同；关闭只需不设或设回 `full`。

```bash
python -m rltt_baseline.launch --config configs/grpo_baseline_8gpu.yaml rollout.kv_layout=decode_last
```

打开后 launcher 自动加入以下 verl 设置（不要再手动设置冲突项，launcher 会拒绝）：
- `actor_rollout_ref.model.external_lib` 加入 `rltt_baseline.vllm_decode_last`：worker 建 vLLM 引擎前
  import 它，注册 `OuroDecodeLastKVForCausalLM` 并给 `GPUModelRunner._prepare_inputs` 加一个钩子；
- `rollout.engine_kwargs.vllm`：`hf_overrides.architectures` 选这个模型类；
  `compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}`（不用 torch.compile，纯 decode
  批次用 CUDA graph）。因此不能同时设置 `rollout.cudagraph_capture_sizes`；
- `rollout.calculate_log_probs=true`：verl 记录 rollout 与训练 log-prob 的差异
  （`training/rollout_probs_diff_*`、`rollout_corr/*`），loss 不变（`rollout_correction.rollout_is` 默认为空）。
  可用 `rollout.log_probs=false` 关掉。

注意：
- **只改 rollout。** 训练时 actor 重算 log-prob 仍用完整的逐轮 KV（teacher forcing 无法并行实现
  decode_last），所以 rollout 分布和训练策略略有不一致，属于 off-policy。差异见上面的指标；
  需要校正时可设 `verl_overrides.algorithm.rollout_correction.rollout_is=token`（GRPO 路径；RLTT 原作 loss
  未接入 IS 权重）。
- **不省显存。** 仍保留每一轮的 KV cache，只改变计算语义。
- 哪些 token 是“生成的”按位置判断（位置 ≥ prompt 长度），在 prefix caching（同一题的 n 个样本共享 prompt
  块）、chunked prefill 下都精确。被抢占的请求重算时，已生成的 token 在同一次 prefill 里互相读的是第 u 轮
  KV，算完后才换成最后一轮 KV，和逐 token 生成略有差别。
- 支持 TRITON_ATTN 与 FLASH_ATTN 后端、bf16/fp16 KV cache；其他后端或 fp8 KV 会直接报错。
- 实验依据：RecurrentTransformer 项目 `plan/20261024_ouro_last_loop_kv/`（Ouro-2.6B-Thinking 上
  decode_last 与完整 KV 的配对对比）。

### wandb

日志通过 verl 0.6.1 的 `Tracking` 写入 wandb。`wandb.mode` 默认 `offline`（计算节点通常不能联网），
训练后在能联网的机器上运行 `wandb sync <output_dir>/wandb/offline-run-*`。

### 训练指标

verl 自带：`critic/score/mean`（平均奖励）、`response_length/*`（含 `clip_ratio`）、
`actor/pg_loss`、`actor/kl_loss`、`actor/grad_norm`、`actor/lr`、`actor/entropy`（最后一轮）、
时间与吞吐。本仓库新增（两种 trainer 都有）：

- `reasoning/accuracy`（训练正确率）、`reasoning/pass_at_group`、
  `reasoning/group_{all_correct,all_wrong,mixed}_ratio`、`reasoning/group_acc_std_mean`；
- `reasoning/truncation_rate`、`reasoning/truncated_accuracy`、
  `reasoning/response_length/{correct,wrong}_mean`、`reasoning/answer_rate`（有 `\boxed{}`）、
  `reasoning/nonzero_adv_token_ratio`、`reasoning/accuracy/level_<k>`。

RLTT 另有原作的 `rltt/loop_<t>_log_prob`、各轮退出概率，以及本仓库新增的
`rltt/loop_<t>_entropy`、`rltt/expected_exit_step`、`actor/total_loss`。

### 逐轮分布审计

采样只用最后一轮，但每一轮 t 都有自己的下一 token 分布
`P^(t) = softmax(lm_head(h_t) / temperature)`。打开审计后，每个审计步对每个生成 token
记录各轮分布的差异，GRPO、RLTT 和 PLC 都支持。所有配置都打开了审计，它是共享键，组内必须一致：

```yaml
audit:
  loop_distribution:
    enabled: true
    every_n_steps: 1    # 审计第 1、1+n、1+2n... 步
    top_k: 5
    save_tokens: true   # false 时不留逐 token 数组，只留 summary.json 和 responses.jsonl
    chunk_tokens: 512   # 每次 lm_head 处理的 token 数；显存不够时调小
```

审计不加前向：它挂在训练本来就要做的 `compute_log_prob` 那次前向上，读出各轮隐状态再算
`lm_head`，所以不改变 loss 和梯度，只多花 `lm_head` 的计算和磁盘。逐 token 数组压缩前约每 token
220 字节：每步 256 条回答，平均每条 6K token 时约 340 MB/步，140 步约 47 GB/次运行（满长 16K 时约
2.5 倍）。磁盘不够时可以整组一起改 `save_tokens: false`（只留汇总）或调大 `every_n_steps`。

命令行写 `audit.loop_distribution.enabled=false` 可以临时关掉；关闭时不装 hook，GRPO 仍用原作的
worker。之前的 `*_loop_audit.yaml` 诊断变体在审计全部打开后与基线配置相同，已删除。

输出在 `<output_dir>/audit/step_<step>/`：

| 文件 | 内容 |
|---|---|
| `summary.json` | 全步所有 token 汇总的 mean、p5、p95，以及 token 数、回答数、错误记录 |
| `responses.jsonl` | 每条回答一行：`uid`、`rank`、`index`、长度、reward、是否答对、是否截断，以及该回答的 mean、p5、p95 |
| `tokens_rank<r>.npz` | 逐 token 数组：`kl` (n, T, T)、`entropy`、`sampled_log_prob` (n, T)、`topk_ids`、`topk_probs` (n, T, k)、`sampled_token` |

| 指标 | 含义 |
|---|---|
| `kl/<t>_<s>` | KL(P^(t) \|\| P^(s))，单位 nat；所有有序轮次对，共 12 项 |
| `entropy/loop_<t>` | 第 t 轮分布的熵 |
| `sampled_log_prob/loop_<t>` | 实际采样的 token 在第 t 轮的 log-prob |
| `top1_prob/loop_<t>`、`topk_mass/loop_<t>` | 第 t 轮最大概率、前 k 个 token 的概率和 |
| `top1_agree/<t>_<s>`、`topk_overlap/<t>_<s>` | 两轮 top-1 是否相同、top-k 集合的重合比例 |

同样的统计也写入 wandb：`loop_audit/<指标>/<mean|p5|p95>`。读取某条回答的逐 token 结果
（`rank`、`index` 来自 `responses.jsonl`）：

```python
from transformers import AutoTokenizer
from rltt_baseline.loop_audit import read_response

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
token = read_response("<output_dir>/audit/step_000001", rank=0, index=3, tokenizer=tokenizer)
token["kl"][17, 0, 3]              # 第 18 个 token 上的 KL(P^(1) || P^(4))
token["topk_tokens"][17][0]        # 第 18 个 token 上第 1 轮的 top-5 token
token["topk_probs"][17, 0]         # 对应的概率
```

- **不增加前向。** 审计读取 `compute_log_prob` 那次前向的逐轮 hidden state，只多算 `lm_head`。
  多加一次前向时，每层 FSDP 都要做 collective，各 rank 的前向次数必须相同；`lm_head` 属于 root
  FSDP 单元，各 rank 调用次数不同也不会卡住。
- **不存完整分布。** 一个词表的 float32 约 200 KB，每个 token 四轮约 800 KB，所以只存 top-k。
- **磁盘。** 压缩前每个 token 约 218 字节（四轮、top-5）。256 条回答、平均 7,738 token
  时约 430 MB/步，140 步约 60 GB；压缩比尚未在真实数据上测过。用 `every_n_steps` 或
  `save_tokens: false` 减少占用。
- **前几轮是假设分布。** 实际采样只来自最后一轮；其余各轮表示“在这个上下文下若停在第 t 轮”的分布，
  上下文都是实际生成的 token。
- **分布来自训练 forward。** 这次前向使用完整的逐轮 KV，`decode_last` rollout 时也一样，所以
  审计的最后一轮不等于 `decode_last` 实际采样时用的分布。
- **自检。** `last_loop_log_prob_gap` 比较审计的最后一轮与 verl 的 `old_log_probs`，应接近 0；
  `unmatched_responses` 是对不上 driver batch 的回答数，应为 0。
- **失败不影响训练。** 审计出错时写 `error_rank<r>.txt`，并记录在 `summary.json`，训练继续。
- 审计按调用次数从第 1 步计数，因此 `resume_mode` 不是 `disable` 或启用 LoRA 时会拒绝启动。

### 逐轮梯度探测

```bash
python -m rltt_baseline.diagnostics.rollouts --model $LOOPRL_MODEL_DIR --data <parquet> --out r.jsonl
python -m rltt_baseline.diagnostics.probe --model $LOOPRL_MODEL_DIR --rollouts r.jsonl --out probe_out
```

`probe.py` 对每个 batch 把题目组分成 A、B 两半，按 GRPO、第 4 轮 NLL、Ouro SFT 式（exit 概率混合 logits）
三种 loss 分别求每一轮的梯度，保存每个参数张量的 Gram 矩阵（`grams.pt`）和汇总（`summary.json`）。

## 测试

```bash
python -m pytest tests          # 需要 transformers >= 4.56 的测试会自动跳过；verl 一致性测试需要 verl
```

## 已知差异与注意事项

- RLTT：原作论文写 140 步、Ouro-2.6B-Thinking；原作 `run_rltt.slurm` 默认 1.4B、5 个 epoch。配置按论文。
- RLTT：原作 actor 没有对 `exit_pdf` 权重做 detach，梯度会流入 exit gate；`prepare_model --detach-exit-pdf` 可关闭。
- RLTT：不支持熵正则（`entropy_coeff` 必须为 0），逐轮熵只用于记录。
- 少于 2 张 GPU 时原作会换用自己的单卡训练器，本仓库的配置覆盖和新增指标不会生效，因此 launcher 直接拒绝。
- `rollout.kv_layout=decode_last` 只作用于 vLLM rollout；训练 forward 仍是完整 KV，见上文。
- RLTT 原作的 requirements 是自身有冲突的 pip freeze（numpy 1.26.4 与 opencv-python-headless 4.12），
  `setup_env.sh` 用 `--no-deps` 按锁定版本逐个安装。
