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
| `rltt_baseline/prepare_model.py`、`fetch_model.py` | 下载固定 revision 的 Ouro-2.6B-Thinking 并核对 SHA256，再生成训练用模型目录：权重软链接，官方 `modeling_ouro.py` 改名为 `modeling_ouro_base.py` 并加 transformers 4.57 兼容修改 |
| `rltt_baseline/prepare_data.py`、`prepare_math_train.py`、`verl_data.py` | 下载 SAPD 数学训练集（DeepMath-103K 难度 ≥ 6，57,046 题，核对 SHA256）、MATH 训练集（7,500 题）和 MATH-500；`train_file` 为 verl 格式 parquet 时原样使用其中的 prompt |
| `rltt_baseline/config.py`、`launch.py` | YAML 配置（`trainer: rltt / grpo`），支持命令行 dot-list 覆盖；`verl_overrides` 可以改任意 verl 配置 |
| `rltt_baseline/decode_last.py`、`vllm_decode_last.py` | 可选的 rollout KV 布局 `decode_last`（`rollout.kv_layout`，默认关闭）：prompt 保留每一轮的 KV，生成的 token 只保留最后一轮的 KV。只改变 vLLM rollout，训练 forward 不变 |
| `rltt_baseline/metrics.py` | driver 端的推理 RL 指标（训练正确率、组内全对/全错/混合比例、截断率、正确与错误回答长度等） |
| `rltt_baseline/workers.py` | RLTT 训练时额外记录逐轮熵、期望退出轮次和总 loss，原作的 loss 计算不变 |
| `rltt_baseline/diagnostics/` | 逐轮梯度探测：rollout 生成（`rollouts.py`）与探测（`probe.py`），GRPO loss 与 verl 0.6.1 逐项一致 |
| `configs/rltt_env.lock.txt`、`scripts/setup_env.sh` | 训练环境（verl 0.6.1、vLLM 0.12.0、torch 2.9.0、transformers 4.57.3）的锁定文件和安装脚本 |

对官方 Ouro 代码只做了一处修改：transformers 4.57 把 `check_model_inputs` 改成了 decorator
factory，官方 `modeling_ouro.py` 直接加载会报 `wrapped_fn() got an unexpected keyword argument`。
`prepare_model` 在生成的 `modeling_ouro_base.py` 中加了一个兼容 wrapper，同时支持新旧两种接口。

## 目录

```
rltt_baseline/              适配代码（Python 包）
configs/grpo_baseline_8gpu.yaml  GRPO 8 卡训练配置
configs/rltt_baseline.yaml  RLTT 论文 Table 6 口径的配置
configs/rltt_env.lock.txt   训练环境锁定文件
scripts/                    setup_env / prepare_data / prepare_model / run_train / smoke_grpo
tests/                      单元测试与 tiny 模型集成测试
third_party/RLTT            原作仓库（submodule，固定 commit）
third_party/ouro_source     官方 Ouro modeling 文件（Apache-2.0，见该目录 README）
```

## 使用

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
2. **数据**（SAPD 数学训练集 `deepmath_level6.parquet`、MATH 训练集 7,500 题、MATH-500）：
   ```bash
   bash scripts/prepare_data.sh $W/data      # 默认：SAPD 数据 + MATH-500，结束时打印 export 行
   # 另加 RLTT 论文的 MATH 训练集：--datasets deepmath_level6,math500,math_train
   ```
3. **模型**（下载 `ByteDance/Ouro-2.6B-Thinking` @ `f1edd81e` 并核对 SHA256，生成训练用目录；
   已有官方目录时加 `--source <dir>` 跳过下载）：
   ```bash
   bash scripts/prepare_model.sh $W/models
   ```
4. **路径变量**：
   ```bash
   export LOOPRL_MODEL_DIR=$W/models/Ouro-2.6B-Thinking-RLTT
   export LOOPRL_OURO_DIR=$W/models/Ouro-2.6B-Thinking        # 只有 RLTT 配置用到
   export LOOPRL_TRAIN_FILE=$W/data/deepmath_level6.parquet   # 或 $W/data/math_train.jsonl
   export LOOPRL_MATH500_JSONL=$W/data/math500.jsonl
   export LOOPRL_OUTPUT_DIR=$W/outputs/grpo_run1
   ```
5. **smoke run**（任意 ≥ 2 张卡，2 步、512 tokens，只验证流程）：
   ```bash
   CUDA_VISIBLE_DEVICES=0,1 bash scripts/smoke_grpo.sh
   ```
6. **训练**（8 卡 GRPO；RLTT 用 `configs/rltt_baseline.yaml`）：
   ```bash
   bash scripts/run_train.sh configs/grpo_baseline_8gpu.yaml
   bash scripts/run_train.sh configs/grpo_baseline_8gpu.yaml args.max_steps=100 wandb.run_name=try1
   ```
   在 Slurm 集群上必须在 allocation 内运行；脚本检测到 `sbatch` 而没有 `SLURM_JOB_ID` 时会拒绝。

### GRPO 8 卡配置要点（`configs/grpo_baseline_8gpu.yaml`）

数据与长度按 SAPD，优化超参数按 RLTT 论文 Table 6（论文中 GRPO 与 RLTT 共用同一套设置），
与 `configs/rltt_baseline.yaml` 一致：

- 模型 Ouro-2.6B-Thinking 全参数，4 次循环。
- 数据：SAPD 的数学训练集，即 G-OPD 发布的 DeepMath-103K 难度 ≥ 6 子集（57,046 题，
  `icemoon28/MOPD-Training-Data` 中的 `DeepMath-103K/train_filtered_level6.parquet`）。
  prompt 用文件自带的：题目后接 “Please reason step by step, and put your final answer within
  \boxed{}.”，单条 user 消息，再套 Ouro 的 chat template。奖励沿用 RLTT：取最后一个 `\boxed{}`，
  用 `math_verify` 判断与标准答案是否等价，0/1。
- prompt 上限 2,048 tokens（按 Ouro chat template，最长的题 743 tokens，没有被过滤的），
  生成上限 16,384 tokens；每卡 token 上限 32,768（须 ≥ 2,048 + 16,384）。
- 每步 32 题 × 8 条 = 256 条 rollout，temperature 1.0；梯度累积 2，
  即每 64 题做一次优化器更新（mini-batch = 64 题），共 140 步（70 次更新）。
- AdamW8bit，lr 1e-6（constant，前 10% 步 warmup），betas (0.9, 0.99)，weight decay 0.1，
  梯度裁剪 0.1；KL β = 0.001，按 RLTT 的方式作为 `low_var_kl` loss；clip 0.2，token-mean。
- 每 10 步存 checkpoint（含 `hf_model`，供逐轮梯度探测）；训练中不做验证（论文如此），
  需要时加 `args.no_validation=false`（MATH-500，用 RLTT prompt）。
- 显存：Ouro 每次循环有独立的 KV cache（48 层 × 4 次 = 192 层），每 token 约 1.57 MB。
  按此估算，vLLM 占 45% 显存时每张 H200 约能放 3.5 万 tokens 的 KV，即同时约 2 条满长度序列，
  长回答多时 rollout 会很慢（实际数值以 vLLM 日志为准）。

原作 `grpo_train.py` 与论文有四处不一致，配置里用 `verl_overrides` 改正：

| 项 | 原作脚本 | 本配置 |
|---|---|---|
| 每步题数 | `data.train_batch_size = 题数 × 8`，verl 当作题数（256 题） | 32 题 |
| 梯度裁剪 | `max_grad_norm` 只传给未启用的 critic，actor 用 verl 默认 1.0 | 0.1 |
| Adam betas | (0.9, 0.999)（RLTT 脚本相同） | (0.9, 0.99) |
| KL | 加在 reward 上；组内奖励全同时，KL 噪声会给这组产生非零 advantage | `low_var_kl` loss，与 RLTT 相同 |

KL 一律不放进 reward：最终 verl 配置的 `algorithm.use_kl_in_reward` 为 true 时，launcher 拒绝启动
（GRPO 原作默认就是 true，所以配置里必须显式写 `false`）。

`configs/rltt_baseline.yaml` 同样改正了每步题数和 betas（RLTT 脚本本身已有梯度裁剪 0.1 和 KL loss）。
注意 warmup 长度按步数算（140 × 10% = 14），但 verl 的学习率调度器每次更新才前进一次，
梯度累积 2 时共 70 次更新，所以实际 warmup 占前 14 次更新（20%）。原作也是这样，这里不改。
- 原作 `grpo_train.py` 会**原地修改**模型目录里的 `modeling_ouro.py`。launcher 只接受
  `prepare_model` 生成的目录，拒绝官方目录，避免改动共享的模型文件。

### 配置与覆盖

配置文件有六个顶层键：`trainer`（`rltt` / `grpo`）、`args`（与所选原作脚本的 argparse 参数同名，
未列出的用原作默认值；拼错的键或类型不符会直接报错）、`wandb`、`verl_overrides`、`env`、
`rollout`（`kv_layout: full | decode_last`，见下节）。
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
