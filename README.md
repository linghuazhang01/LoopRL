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
| `rltt_baseline/plc.py`、`plc_workers.py` | PLC（`plc.enabled`，默认关闭）：最后一轮做 GRPO，前几轮用截断重要性比率的 off-policy PPO 代理目标加指向最后一轮的正向 KL 信赖域；λ=0 退回 GRPO |
| `rltt_baseline/loop_audit.py` | 逐轮分布审计（`audit.loop_distribution`，所有配置都打开）：每个生成 token 上各轮分布的 KL、熵和 top-k，按步写入 `audit/` |
| `rltt_baseline/loss_agg.py`、`grpo_workers.py` | loss 聚合方式（`loss_agg` 段）：Dr. GRPO 常数归一化、按题组内 token 平均；结果与 micro-batch 怎么切无关。GRPO 由 `grpo_workers.py` 的 `RowAggPPOActor` 实现，PLC 由 `plc_workers.py` 实现；`grpo_workers.py` 同时挂逐轮分布审计 |
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
configs/plc_grpo_loopaux_1p4b_8gpu.yaml  PLC：GRPO + 中间轮 off-policy 辅助项（截断重要性权重 + 正向 KL 信赖域、无 gate），1.4B
configs/plc_grpo_loopaux_nokl_1p4b_8gpu.yaml  同上但去掉正向 KL 项（`plc.kl_coef=0`），1.4B
configs/plc_grpo_loopkd_1p4b_8gpu.yaml  PLC：GRPO + 前几轮向 stop-gradient 第 4 轮做全词表蒸馏（`plc.loop_loss: kd`），1.4B
configs/plc_grpo_loopkd_rev_1p4b_8gpu.yaml  同上但用 reverse KL（`plc.kd_divergence: reverse`，OPD / LoopOPD 的方向），1.4B
configs/plc_grpo_loopaux_nocap_refkl_1p4b_8gpu.yaml  PLC：精确比率（无上界）+ 初始模型锚定项（k3，0.1），去掉 k1，1.4B
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

在本仓库工作的 agent 须遵守 `AGENTS.md`：只用 14,684 条训练集、训练 seed 固定 42、wandb online、每次启动训练前报告全部参数。
其中数据、seed 和 wandb 三条由 launcher 强制检查：训练文件的 SHA256 或行数不对、`args.seed` /
`data.seed` 不是 42、wandb 不是 online 或项目不是 `looprl-ouro`，都会在启动 Ray 之前拒绝训练
（dry-run 的 `policy_violations` 列出这些问题）。

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

配置按对照组组织，组内文件除方法本身（`trainer`、loss、`plc` 段、`loss_agg` 段、wandb 命名）外每个参数都必须相同，
这样三种方法可以直接对比：

| 组 | GRPO | RLTT | PLC |
|---|---|---|---|
| 1.4B，8 卡，每步一次更新 | `grpo_baseline_1p4b_8gpu.yaml` | `rltt_baseline_1p4b_8gpu.yaml` | `plc_grpo_loopaux_1p4b_8gpu.yaml` |
| 2.6B，8 卡，每步一次更新 | `grpo_baseline_8gpu.yaml` | `rltt_baseline.yaml` | —（暂无） |
| 1.4B 消融：PLC 去掉 KL 项 | — | — | `plc_grpo_loopaux_nokl_1p4b_8gpu.yaml` |
| 1.4B：PLC 前几轮改为跨轮蒸馏 | — | — | `plc_grpo_loopkd_1p4b_8gpu.yaml` |
| 1.4B：同上，reverse KL | — | — | `plc_grpo_loopkd_rev_1p4b_8gpu.yaml` |
| 1.4B：PLC 精确比率 + 初始模型锚定项 | — | — | `plc_grpo_loopaux_nocap_refkl_1p4b_8gpu.yaml` |

测试 `test_matched_configs_share_all_non_method_settings` 检查这一点（允许不同的键和改配置的规则见
`CLAUDE.md`）。改一个共享超参要改整组文件；临时实验用命令行覆盖，不要另存配置。

随机种子全部锁定 42：`args.seed: 42`，且 `verl_overrides.data.seed: ${args.seed}`（verl 0.6.1 的 dataloader
打乱种子默认为空，不从 `args.seed` 派生）；vLLM 引擎 seed 在 verl 0.6.1 不可配置（固定 0，各方法相同）。
诊断脚本 `rltt_baseline.diagnostics.rollouts` / `probe` 和原作评估脚本的默认 seed 也都是 42。

### loss 聚合（`loss_agg` 段，2026-11-14 起按方法设置）

verl 0.6.1 的 `token-mean` 在每个 micro-batch 内部按 token 平均，再乘 `该块条数 / mini-batch 条数`；
开着动态 batch 时 micro-batch 按 token 数切分，同一个 token 的权重因此取决于和谁分在一块。
`loss_agg` 段换成与切分无关的聚合：每条回答 i 有一个只取决于它自己的除数 `d_i`（每个 micro-batch
从自己的 `response_mask` 算出），一个 B 条的 micro-batch 贡献 `(1/B)·Σ_i S_i / d_i`（`S_i` 为该条回答所有 token 的 loss 之和），
乘上 verl 的 `B/N` 并经 FSDP 跨卡平均后，整步的 loss 就是 `Σ_i S_i / d_i / (总条数)`：

| `mode` | `d_i` | 整步 loss | 使用者 |
|---|---|---|---|
| `token-mean`（默认） | — | verl 原样，micro-batch 内 token 平均 | RLTT（原作 loss 写死在 `third_party`，不能改） |
| `drgrpo` | `norm_len`（M） | `Σ_i S_i / (总条数 · M)`，Dr. GRPO | 所有 PLC 配置，`norm_len: 16384` |
| `seq-mean-token-mean` | 该回答自己的长度 `L_i` | `Σ_i (S_i / L_i) / 总条数`，每条回答先按自己的 token 平均，再对回答平均（原始 GRPO） | 两份 GRPO 配置 |

`seq-mean-token-mean` 就是 DeepSeekMath 里 GRPO 的 `1/G Σ_i 1/|o_i| Σ_t`；Dr. GRPO 指出它有长度偏差
（长的错误回答每个 token 受的惩罚更小）。GRPO 不直接用 verl 的同名 `loss_agg_mode`，是因为 driver 的
`actor/entropy` 指标也读它，GRPO 的熵指标会和其他方法口径不同。verl 自己的
`loss_agg_mode` 保持 `token-mean`（driver 的 `actor/entropy` 指标用它），launcher 拒绝改成别的值。
记录的 `actor/entropy_loss`、`actor/kl_loss` 仍是 token 平均；`actor/pg_loss`、`actor/total_loss`
按新的聚合计算，数值量级与以前不同（`drgrpo` 约小 `平均长度/16384` 倍）。新指标
`loss_agg/row_div_mean` 是每个 micro-batch 的平均除数。`tests/test_loss_agg.py` 模拟 verl 的切分和
跨卡平均，检查两种模式的每 token 梯度与闭式解一致、与切分无关。

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
  同一处还显式写出 `top_p` 1.0、`top_k` -1、`do_sample` true、`ignore_eos` false、`max_num_seqs` 1024、
  `dtype` bfloat16，取值与 verl 默认值相同，所有配置一致。
- 每步 64 题 × 8 条 = 512 条 rollout，temperature 1.0；梯度累积 1，
  即每步做一次优化器更新（mini-batch = 64 题），共 100 步 = 100 次更新
  （论文：每步 32 题、累积 2、140 步）。
- AdamW8bit，lr 1e-6（constant，前 10% 步 warmup），betas (0.9, 0.99)，weight decay 0.1，
  梯度裁剪 0.1；KL 关闭（`beta: 0`、`use_kl_loss: false`、`kl_loss_coef: 0`，所有配置相同）；
  clip 0.2；loss 每条回答先按自己的 token 平均、再对回答平均，即原始 GRPO（`loss_agg.mode:
  seq-mean-token-mean`，2026-11-14 起；之前是 verl 的 micro-batch 内 token-mean）。
- 每步存 checkpoint（`save_steps: 1`，用户 2026-11-07 要求；含 `hf_model`，供逐轮梯度探测）。
  内容是 model + optimizer + extra + hf_model，verl 默认全部保留，100 步就是 100 份；粗估每份 1.4B 约 20 GB、
  2.6B 约 40 GB（未实测），要留足磁盘，或整组一起加 `verl_overrides.trainer.max_actor_ckpt_to_keep`。
  训练中不做验证（论文如此），
  需要时加 `args.no_validation=false`（MATH-500，用 RLTT prompt）。
- 显存：Ouro 每次循环有独立的 KV cache（48 层 × 4 次 = 192 层），每 token 约 1.57 MB。
  按此估算，vLLM 占 50% 显存时每张 H200 约能放 4 万 tokens 的 KV，即同时约 2 条满长度序列，
  长回答多时 rollout 会很慢（实际数值以 vLLM 日志为准）。

原作 `grpo_train.py` 与论文有四处不一致，配置里用 `verl_overrides` 改正：

| 项 | 原作脚本 | 本配置 |
|---|---|---|
| 每步题数 | `data.train_batch_size = 题数 × 8`，verl 当作题数（512 题） | 64 题 |
| 梯度裁剪 | `max_grad_norm` 只传给未启用的 critic，actor 用 verl 默认 1.0 | 0.1 |
| Adam betas | (0.9, 0.999)（RLTT 脚本相同） | (0.9, 0.99) |
| KL | 加在 reward 上；组内奖励全同时，KL 噪声会给这组产生非零 advantage | 不进 reward；KL loss 也关闭（见 `CLAUDE.md`） |

KL 一律不放进 reward：最终 verl 配置的 `algorithm.use_kl_in_reward` 为 true 时，launcher 拒绝启动
（GRPO 原作默认就是 true，所以配置里必须显式写 `false`）。

原作 GRPO 脚本总会注册参考模型，verl 0.6.1 只要注册就会加载它并每步多做一次前向。KL loss 也关闭时，
launcher 在构造 trainer 前去掉这个角色，GRPO 就和 RLTT / PLC 一样不加载参考模型；dry-run 输出和
`rltt_baseline_resolved.json` 的 `reference_policy` 字段写明是否去掉。重新打开 KL loss 时参考模型照常加载。

`configs/rltt_baseline.yaml` 同样改正了每步题数和 betas（RLTT 脚本本身已有梯度裁剪 0.1）。
warmup 长度按步数算（100 × 10% = 10），verl 的学习率调度器每次更新前进一次；梯度累积 1 时步数等于
更新次数，warmup 就是前 14 次更新（10%）。若以后改回累积 2，warmup 会占到更新数的 20%。
- 原作 `grpo_train.py` 会**原地修改**模型目录里的 `modeling_ouro.py`。launcher 只接受
  `prepare_model` 生成的目录，拒绝官方目录，避免改动共享的模型文件。

### GRPO 1.4B 8 卡配置（`configs/grpo_baseline_1p4b_8gpu.yaml`）

与 2.6B 配置只差模型（Ouro-1.4B-Thinking；RLTT 论文两种规模用同一套设置），其余（数据、prompt 2,048 /
生成 16,384、每步 64 题 × 8 条、梯度累积 1、mini-batch 64 题、100 步 = 100 次更新、优化器、KL、
checkpoint、显存参数）都相同，测试 `test_grpo_1p4b_config_matches_2p6b_except_model` 检查这一点。
每次更新用 512 条 rollout，GRPO 只跑 1 个 epoch；warmup 10% 即前 10 次更新。
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

早期运行读的是集群共享存储上的同一份文件，HF 上的是它的逐字节副本，SHA256 相同。

数据仓库为 **Private**，下载账号须有访问权限。先在已有 `huggingface_hub` 环境中执行
`hf auth login`，或使用已安全配置的 `HF_TOKEN`。以下命令沿用上文的工作目录 `W`；在训练集群上，
`W` 放在集群的共享存储里，在本机就用本机目录。

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

这份 2,865 条数据已纳入上述合并训练集。单独使用时，100 步 × 64 题需要 6,400 题，
它只够约 44 步不重复。规则和构成见该目录的 README。

### PLC（`configs/plc_grpo_loopaux_1p4b_8gpu.yaml`）

`plc.enabled: true` 时 launcher 用 `PLCActorRolloutRefWorker` 代替原作的 RLTT worker（仍是 `trainer: rltt`，
仍需 `use_rltt_model: true` 的逐轮模型目录）。2026-11-13 起这是唯一的 PLC 逻辑（旧的 posterior 权重、
gate 损失、softmax(logp) / RLTT-progressive 权重、组基线和 IS 修正连同配置一起删除）。
记号：`ℓ[j,t] = log P^(t)(y_j)` 为采样 token 在第 t 轮的 log-prob。

vLLM 从最后一轮 T 采样，所以最后一轮是前几轮的行为策略。最后一轮照常做 GRPO；前几轮用截断比率加权的
PPO 代理目标学习，再加指向最后一轮的正向 KL 信赖域：

- **损失**：`-agg_j[PPO^(T)(Â)] + λ·Σ_{t<T} α_t·( -agg_j[ρ̄[j,t]·PPO^(t)(Â)] + kl_coef·agg_j[k1[j,t]] )`，
  `ρ̄[j,t] = min(trunc_cap, P_old^(t)(y_j)/P_old^(T)(y_j))`（只截上限，不回传梯度），
  `k1[j,t] = log P_old^(T)(y_j) − log P^(t)(y_j)`（最后一轮自己样本上 `KL(P^(T)‖P^(t))` 的无偏估计，
  梯度就是第 t 轮在 rollout token 上的交叉熵），`α_t = 1/(T−1)`。`agg` 由 `loss_agg` 段决定：
  2026-11-14 起所有 PLC 配置用 Dr. GRPO（`mode: drgrpo`，`norm_len: 16384`），每一项都是
  `Σ token / (条数 · 16384)`；不写 `loss_agg` 时是 verl 的 micro-batch 内 token-mean。每轮一个 PPO ratio `exp(ℓ_new − ℓ_old)`，clip 用 verl 的 `clip_ratio` / 双重 clip。
  最后一轮那一项就是 GRPO，λ=0 且 `loss_agg` 为 token-mean 时与 verl GRPO 的 loss 逐位相等
  （`test_lam0_is_verl_grpo_on_the_final_loop`）；用 Dr. GRPO 聚合时，λ=0 是"Dr. GRPO 聚合的 GRPO"，
  与 GRPO 配置（按题组内 token 平均）不同。
  每步一次更新时，第 t 轮每个 token 的梯度系数是 `−(λ/N)·α_t·(ρ̄·Â + kl_coef)`
  （`test_trust_region_gradient_is_rho_a_plus_kl_coef_per_token`）。熵项和 KL（`low_var_kl`）只加在最后一轮，
  与原作相同；对照组配置里 KL 关闭，熵系数为 0。gate 不进入 loss。
- **`plc` 段**：`weight_mode`（`is_trunc`，或 `early_prior` 即 ρ̄=1）、`lam`（0.1）、`trunc_cap`（1.0）、
  `kl_coef`（0.2）、`log_weight_stats`；可选的初始模型锚定项 `init_kl_coef`（0 = 关）、`init_kl_form`
  （`k3_reverse`）、`init_kl_log_ratio_cap`（3），见下一条。
- **初始模型锚定项（可选，默认关闭）**：`init_kl_coef > 0` 时 loss 再加
  `λ·init_kl_coef·agg_j[Σ_{t<T} α_t·pen(d[j,t])]`，`d = log P^(t)(y_j) − log P_init^(t)(y_j)`，
  `P_init` 是冻结的初始模型（与 actor 同一个 `model.path`，verl 参考模型的构建方式：fp32 主权重、
  同样的 bf16 混合精度、参数 offload 到 CPU；`compute_log_prob` 时在同一批 micro-batch 上无梯度前向一次，
  结果放进批次的 `init_log_probs_loops` `[B,L,T]`）。`pen` 由 `init_kl_form` 选：
  `k3` = e^{−d}+d−1（r = P_init/P_new；|d| 较小时与 verl `low_var_kl` 相同，verl 在值 ≥ 10 处截断、梯度为 0，这里改为在 cap 处线性延伸；对被抬高的 token 回复力 ≤ 1）、
  `k3_reverse` = e^{d}−d−1（r = P_new/P_init；对被抬高的 token 回复力无上限，即最后一轮样本推动的方向）、
  `k2` = d²/2（两个方向对称）。三者在 d = 0 处值和梯度都为 0，锚点是第 t 轮自己的初始分布；
  |d| 超过 `init_kl_log_ratio_cap` 后惩罚按该处斜率线性延伸，梯度有界且不会被截成 0。
  采样 token 上的 k1 不能当锚：`log P_init` 没有梯度，它的梯度和指向最后一轮的 k1 完全相同。
  指标：`plc/init_term_mean`、`plc/init_pen/loop_<t>`、`plc/init_log_ratio/loop_<t>`、
  `plc/init_beyond_cap_frac/loop_<t>`（`loop_4` 只是监控，权重为 0）。锚定项乘 λ，实际强度是
  `λ·init_kl_coef`；`plc.lam=0` 时设 `init_kl_coef > 0` 会被拒绝（否则加载冻结模型却没有梯度）。
- **指标**（逐轮的量每轮一个键，`/loop_<t>` 即第 t 轮；verl 的 `reduce_metrics` 对每个键的列表取均值，
  所以不能把一个逐轮列表放在同一个键里，见 `plc.flatten_loop_metrics`）：
  `plc/clipfrac/loop_<t>`、`plc/ratio_mean/loop_<t>`、`plc/loop_term_mean`、`plc/final_term_mean`、
  `plc/kl_k1/loop_<t>`（loss 用的单样本估计）、`plc/kl_k3/loop_<t>`（≥0，低方差，只用于监控）、
  `plc/kl_term_mean`、`plc/w_mean/loop_<t>`，前几轮（t = 1..3）的 `plc/xloop_rho_bar_mean/loop_<t>`、
  `plc/xloop_capped_frac/loop_<t>`、`plc/xloop_rho_lt_0.1_frac/loop_<t>`、`plc/xloop_pg_weight_abs_mean/loop_<t>`
  （与 `kl_coef` 比大小），相邻轮之间的 `plc/delta_mean/loop_<t>_to_<t+1>`、`plc/delta_neg_frac/loop_<t>_to_<t+1>`；
  原有的 `rltt/loop_<t>_entropy`、`actor/*` 照旧。2026-11-13 之前的 run 里这些逐轮指标名是 `plc/*_per_loop`，
  wandb 上只有把各轮平均后的一个数；要看各轮，用审计 npz 的 `sampled_log_prob` 离线重算
  （k1⁽ᵗ⁾ = ℓ⁽⁴⁾ − ℓ⁽ᵗ⁾）。完整词表上的精确 KL 没有实现，监控用审计里已有的
  `loop_audit/kl/4_t`（精确的 `KL(P^(4)‖P^(t))`）。
- **批次里新增的张量**：`old_log_probs_loops` `[B,L,T]`；`old_log_probs` 就是它的最后一轮。
  锚定项打开时还有 `init_log_probs_loops` `[B,L,T]`。

消融用命令行覆盖：`plc.kl_coef=0.05 / 1.0`（信赖域大小）、`plc.kl_coef=0`（只有代理目标）、
`plc.trunc_cap=2`、`plc.weight_mode=early_prior`（不乘比率）、`plc.lam=0`（PLC 代码路径上的 GRPO）、
`plc.kl_coef=0 plc.init_kl_coef=0.05 [plc.init_kl_form=k3|k3_reverse|k2]`（用初始模型锚定项替换
指向最后一轮的 KL；每步多一次冻结模型前向，耗时不能直接和 GRPO / RLTT 比）。去掉 KL 项的消融另存为 `configs/plc_grpo_loopaux_nokl_1p4b_8gpu.yaml`（用户 2026-11-12 要求）。
状态：只有 CPU 单元测试和 tiny 随机模型测试（`tests/test_plc.py`、`test_rltt_baseline_modeling.py`），
没有在真实权重或 GPU 上运行过。

#### 前几轮改为跨轮蒸馏（`plc.loop_loss: kd`，`configs/plc_grpo_loopkd_1p4b_8gpu.yaml`）

- **损失**：`-agg_j[PPO^(T)(Â)] + λ·Σ_{t<T} α_t·agg_j[D_j^(t)]`，α_t = 1/3 固定，不用 advantage、不乘比率、
  没有 k1。`D_j^(t)` 是第 j 个回答位置 `s_j = (x, y_<j)` 上的全词表 KL：`kd_divergence: forward` 为
  `KL(sg P^(T)(·|s_j) ‖ P^(t)(·|s_j))`，`reverse` 为 `KL(P^(t) ‖ sg P^(T))`。rollout 就是第 4 轮采样的那一批，
  不额外生成。教师是当前第 4 轮（含 `lm_head`）的 stop-gradient，每步一次更新时就是旧策略。
- **梯度**：第 t 轮 logits 上 forward 为 `(λ/N)·α_t·(P^(t) − P^(T))`，reverse 为
  `(λ/N)·α_t·P^(t)⊙(log P^(t) − log P^(T) − D)`；不经过 gate。forward 是 k1 信赖域梯度 `P^(t) − e_y`
  在 `y ~ P^(T)` 下的精确期望。不动点是 `P^(t) = P^(T)`，所以前几轮的熵按设计会靠近第 4 轮；判断漂移看
  `entropy_gap < 0`（比教师还尖）的比例、第 4 轮自己的熵，以及答对 / 答错回答之间的差别。
- **实现**：模型 forward 的 `return_loop_kd`（`rltt_per_loop.per_loop_kd`）只在回答窗口上分块计算，每块都做
  checkpoint（反向时重算 logits，否则每个位置要留 T 份词表大小的张量）。**`LOOPRL_MODEL_DIR` 要用
  `prepare_model` 重建**；旧目录会在 actor 初始化时报错。
- **指标**（每步，`plc/kd_<name>/loop_t`，t = 1..3）：`kl_forward`、`kl_reverse`、`entropy_gap`
  （H(P^(t)) − H(P^(T))）、`below_teacher_frac`、`top1_agree`、`teacher_top1_prob`（P^(t) 在教师 top-1 上的概率）、
  `entropy_push`（只看蒸馏梯度时第 t 轮熵的一阶变化，负 = 降熵），以及 `plc/kd_loss/loop_t`、`plc/kd_term_mean`。
  梯度裁剪 0.1 是对总梯度，蒸馏项太大会压小 GRPO 那部分，要看 `actor/grad_norm`。
- **消融**：`plc.kd_divergence=reverse`、`plc.lam=1.0`、`plc.lam=0`（GRPO）。可以叠加 `plc.init_kl_coef`。

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

日志通过 verl 0.6.1 的 `Tracking` 写入 wandb。所有配置都是 `wandb.mode: online`，项目 `looprl-ouro`
（`AGENTS.md` 规则 4）。`WANDB_API_KEY` 放在训练机器本地的 `env.local`（仓库根目录，已 gitignore；
或用 `LOOPRL_ENV_FILE` 指定路径），格式见 `env.local.example`：

```bash
cp env.local.example env.local   # 然后填入 WANDB_API_KEY=...；不要提交这个文件
```

launcher 启动时读取它（不覆盖已设置的环境变量），dry-run 输出只写 `wandb_api_key: found / missing`
和读到的变量名，不写值。online 而缺少密钥时，launcher 在启动 Ray 之前报错。训练节点需要能访问 wandb。

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
    loop_weights: true  # 另记每个 token 的逐轮权重：exit_pdf，PLC 运行还有 plc_weight
```

`loop_weights` 打开时，每个 token 另记两组逐轮权重（各为 (n, T)，每 token 16 字节）：

- `exit_pdf`：exit gate 给出的各轮退出概率 `q_t = λ_t ∏_{s<t}(1 − λ_s)`，最后一轮取剩余部分。
  三种方法都记；它也是 RLTT（`loop_weighting: exit_pdf`）loss 用的逐轮权重。
- `plc_weight`：PLC 前几轮的权重 `α_t·min(trunc_cap, ρ_t)`（最后一轮为 0），用 PLC actor 在同一次
  `compute_log_prob` 里算出的逐轮 log-prob 求得，等于这一步 `update_policy` 实际使用的权重。
  只在 PLC 运行（`plc.enabled`）里有。

每一轮上采样 token 的概率就是已有的 `exp(sampled_log_prob)`，不另存。这个开关只控制上面两组新增数组，
关掉后审计与之前完全相同。

跨轮蒸馏的诊断（所有方法都记录，可直接比较；最后一轮是教师）：

- `teacher_topk_probs` (n, T, k)：第 t 轮在第 4 轮 top-k 词上的概率，求和就是第 t 轮放在教师候选上的质量；
- `kd_entropy_push_forward` / `kd_entropy_push_reverse` (n, T)：向第 4 轮走一步蒸馏时第 t 轮熵的一阶变化
  （负 = 降熵，最后一轮为 0）；forward 那个也是 PLC k1 信赖域的期望推力。
- summary 另有 `entropy_gap/loop_t`（H(P^(t)) − H(P^(4))）、`below_teacher/loop_t`（比第 4 轮还尖的 token
  比例）、`teacher_top1_prob/loop_t`、`teacher_topk_mass/loop_t`，并在 `metrics_by_correct` 里把熵、KL、
  `entropy_gap`、推力、`xloop_log_rho` 按答对 / 答错分开统计（wandb：`loop_audit/{correct,wrong}/<name>/mean`）。
  和 advantage 有关的漂移会表现为两组之间的系统差异。压缩前每个 token 多约 112 字节（top_k 5 时 80 + 32；不受 `loop_weights` 控制），下面的 220 字节估计因此变为约 330 字节。

审计不加前向：它挂在训练本来就要做的 `compute_log_prob` 那次前向上，读出各轮隐状态再算
`lm_head`，所以不改变 loss 和梯度，只多花 `lm_head` 的计算和磁盘。逐 token 数组压缩前约每 token
220 字节：每步 512 条回答，平均每条 6K token 时约 680 MB/步，100 步约 68 GB/次运行（满长 16K 时约
2.5 倍）。磁盘不够时可以整组一起改 `save_tokens: false`（只留汇总）或调大 `every_n_steps`。

命令行写 `audit.loop_distribution.enabled=false` 可以临时关掉；关闭时不装 hook，GRPO 仍用原作的
worker。之前的 `*_loop_audit.yaml` 诊断变体在审计全部打开后与基线配置相同，已删除。

输出在 `<output_dir>/audit/step_<step>/`：

| 文件 | 内容 |
|---|---|
| `summary.json` | 全步所有 token 汇总的 mean、p5、p95，以及 token 数、回答数、错误记录 |
| `responses.jsonl` | 每条回答一行：`uid`、`rank`、`index`、长度、reward、是否答对、是否截断，以及该回答的 mean、p5、p95 |
| `tokens_rank<r>.npz` | 逐 token 数组：`kl` (n, T, T)、`entropy`、`sampled_log_prob` (n, T)、`topk_ids`、`topk_probs` (n, T, k)、`sampled_token`；`loop_weights` 打开时还有 `exit_pdf`、`plc_weight` (n, T) |

| 指标 | 含义 |
|---|---|
| `kl/<t>_<s>` | KL(P^(t) \|\| P^(s))，单位 nat；所有有序轮次对，共 12 项 |
| `entropy/loop_<t>` | 第 t 轮分布的熵 |
| `sampled_log_prob/loop_<t>` | 实际采样的 token 在第 t 轮的 log-prob |
| `top1_prob/loop_<t>`、`topk_mass/loop_<t>` | 第 t 轮最大概率、前 k 个 token 的概率和 |
| `top1_agree/<t>_<s>`、`topk_overlap/<t>_<s>` | 两轮 top-1 是否相同、top-k 集合的重合比例 |
| `exit_pdf/loop_<t>`、`exit_pdf/expected_loop` | `loop_weights`：第 t 轮的退出概率、期望退出轮次 `Σ_t t·q_t` |
| `plc_weight/loop_<t>`、`plc_weight/expected_loop` | `loop_weights`，PLC 运行：第 t 轮的 PLC 权重、`Σ_t t·w_t` |

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
- **磁盘。** 压缩前每个 token 约 218 字节（四轮、top-5）。512 条回答、平均 7,738 token
  时约 860 MB/步，100 步约 86 GB；压缩比尚未在真实数据上测过。用 `every_n_steps` 或
  `save_tokens: false` 减少占用。
- **前几轮是假设分布。** 实际采样只来自最后一轮；其余各轮表示“在这个上下文下若停在第 t 轮”的分布，
  上下文都是实际生成的 token。
- **分布来自训练 forward。** 这次前向使用完整的逐轮 KV，`decode_last` rollout 时也一样，所以
  审计的最后一轮不等于 `decode_last` 实际采样时用的分布。
- **自检。** `last_loop_log_prob_gap` 比较审计的最后一轮与 verl 的 `old_log_probs`，应接近 0；
  `unmatched_responses` 是对不上 driver batch 的回答数，应为 0。
- **失败不影响训练。** 审计出错时写 `error_rank<r>.txt`，并记录在 `summary.json`，训练继续。
- 审计按 `compute_log_prob` 调用次数计步，启用 LoRA 时拒绝启动。续训（`verl_overrides.trainer.resume_mode=auto`
  或 `resume_path`）时 launcher 读出要加载的 checkpoint 步数 n，审计从第 n+1 步接着编号，写进同一个
  `audit/` 目录（解析后的 `loop_audit.first_step` 记录这个值）；已有 `summary.json` 的步不会被覆盖。

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
- loss 聚合三种方法不同（2026-11-14 起）：GRPO 用原始 GRPO 的逐回答平均，PLC 用 Dr. GRPO（M = 16384），RLTT 保留
  原作的 micro-batch 内 token-mean。PLC 的 `plc.lam=0` 因此不再等于 GRPO 配置（只差聚合方式），三种方法的
  曲线差异里也包含聚合方式的差异；之前跑的 GRPO / PLC 结果是旧的聚合方式。
- 少于 2 张 GPU 时原作会换用自己的单卡训练器，本仓库的配置覆盖和新增指标不会生效，因此 launcher 直接拒绝。
- `rollout.kv_layout=decode_last` 只作用于 vLLM rollout；训练 forward 仍是完整 KV，见上文。
- RLTT 原作的 requirements 是自身有冲突的 pip freeze（numpy 1.26.4 与 opencv-python-headless 4.12），
  `setup_env.sh` 用 `--no-deps` 按锁定版本逐个安装。
