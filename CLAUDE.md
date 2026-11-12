# LoopRL：给 Claude 的项目约定

先读并遵守 `AGENTS.md`（训练数据、seed 42、启动训练前报告全部参数），它对所有 agent 生效，
与本文件冲突时以它为准：

@AGENTS.md

在 looped language model（Ouro）上做 RL：GRPO 对照、RLTT 对照（arXiv 2602.10520）和我们的方法
PLC（Posterior Loop Credit）。原作代码在 `third_party/RLTT`（submodule，固定 commit，**不修改**）；
本仓库的适配代码在 `rltt_baseline/`，配置在 `configs/`，细节见 README.md。

launcher 会强制检查 `AGENTS.md` 的规则 1、2、4（训练文件 SHA256 与行数、seed 42、wandb online 且项目
`looprl-ouro`），违反就拒绝启动，没有跳过开关；对应常量在 `rltt_baseline/config.py`。

## 分支：代码统一在 main

- **所有代码、配置和文档的修改都直接在 `main` 上进行。** 不再为新功能或新方法开长期分支；
  `plc` 等旧分支已合并进 `main`（2026-11-04），之后不再在上面提交。
- 开始工作前先 `git checkout main && git pull --ff-only`，确认工作区干净，避免在旧分支上改。
- 需要隔离的短期实验可以临时开分支，但完成后尽快合回 `main` 并删除，不要让 `main` 与分支长期分叉；
  合并时 `configs/` 必须重新满足下面的对照组规则。
- 正式实验只用 `main` 上的 commit 跑，实验记录里写明该 commit。

## 配置对照组：方法之外的参数必须完全一致

实验结论靠的是"同一套设置下只换方法"。`configs/` 里的文件按**对照组**组织，组内文件除方法本身外
每个参数都必须相同，这样三种方法的曲线才能直接对比。

| 组 | GRPO | RLTT | PLC |
|---|---|---|---|
| 1.4B，8 卡，每步一次更新 | `grpo_baseline_1p4b_8gpu.yaml` | `rltt_baseline_1p4b_8gpu.yaml` | `plc_posterior_exit_gate_1p4b_8gpu.yaml` |
| 2.6B，8 卡，每步一次更新 | `grpo_baseline_8gpu.yaml` | `rltt_baseline.yaml` | `plc_posterior_exit_gate_2p6b_8gpu.yaml` |
| 1.4B 消融：PLC 关闭 gate loss | — | — | `plc_posterior_exit_nogate_1p4b_8gpu.yaml` |
| 1.4B 权重：softmax(logp)（IDEA Two） | — | — | `plc_logp_softmax_1p4b_8gpu.yaml` |
| 1.4B 权重：RLTT-progressive（PLC 代码路径） | — | — | `plc_rltt_progressive_1p4b_8gpu.yaml` |
| 1.4B GRPO + 中间轮辅助项（PLC 代码路径） | — | — | `plc_grpo_loopaux_1p4b_8gpu.yaml` |
| 1.4B GRPO + 中间轮辅助项、无 KL 项（PLC 代码路径） | — | — | `plc_grpo_loopaux_nokl_1p4b_8gpu.yaml` |

组内**允许**不同的只有：

- `trainer`（`grpo` / `rltt`）、`plc` 段、`wandb` 的 `run_name / tags / notes`；
- 两份原作脚本参数名不同但取值等价的键：GRPO 的 `use_ppo`；RLTT 的 `use_rltt_model`、
  `rltt_model_path`（= GRPO 的 `model_path`，即 `LOOPRL_MODEL_DIR`）、`rltt_logprob_chunk_size`、
  `loop_weighting`、`progressive_alpha`、`ref_offload: true`、`no_gradient_checkpointing: false`、
  `lr_scheduler_type: constant`；
- GRPO 配置里为补原作脚本缺失项而写的 `verl_overrides`：`algorithm.use_kl_in_reward: false`、
  `actor.grad_clip: 0.1`、`ref.fsdp_config.param_offload: true`。
  RLTT 脚本通过自己的参数设出同样的值。

其余一切（数据、prompt / 生成长度、每步题数 × 采样数、步数、梯度累积、mini-batch、学习率与调度、
betas、weight decay、梯度裁剪、KL 系数、temperature、seed、vLLM 参数、checkpoint 内容、`model.num_hidden_layers`、
`rollout`、`env`、`audit`、`wandb.mode / dir / project / entity`（mode 必须是 `online`，见 `AGENTS.md`
规则 4；密钥从本地 gitignored 的 `env.local` 读取）、vLLM 的
`verl_overrides.actor_rollout_ref.rollout.max_num_batched_tokens`）组内必须一致。所有 run 都记到
同一个 wandb 项目 `looprl-ouro`，靠 run 名和 tag 区分方法与规模；消融和多 seed 的 run 用
`wandb.run_name=...` 起不同的名字。逐轮分布审计（`audit.loop_distribution`）
在所有配置里都打开；它只读 `compute_log_prob` 那次前向的隐状态，不改变训练，但每次运行写几十 GB，
要关或调稀疏时整组一起改。

这由 `tests/test_rltt_baseline.py::test_matched_configs_share_all_non_method_settings` 强制检查，
对照组列在该文件的 `MATCHED_GROUPS` 里。因此：

1. **改一个共享超参，就要改整组的所有文件**，然后跑 `pytest tests/test_rltt_baseline.py tests/test_plc.py`。
2. **新增一种方法或一份配置**，先从同组已有文件复制，只改方法相关的键，再把文件名加进 `MATCHED_GROUPS`
   以及 `test_mini_batch_is_one_accumulated_update`、`test_other_configs_have_plc_off` 的参数列表。
3. 临时实验用命令行 dot-list 覆盖（`args.max_steps=10 wandb.run_name=try`），不要为此另存一份配置。
4. 消融走同一份配置加覆盖：`plc.gamma=0`（RLTT-exit）、`plc.lam=0`（GRPO）、`plc.use_gate_loss=false`。
   PLC 配置加 `plc.gamma=0 plc.use_gate_loss=false` 是"与 PLC 完全同一条代码路径"的 RLTT 对照；
   `rltt_baseline_1p4b_8gpu.yaml` 是原作 REINFORCE 形式的 RLTT 对照。两者回答的问题不同，论文里都要有。
   例外：`plc_posterior_exit_nogate_1p4b_8gpu.yaml` 是用户 2026-11-05 要求单独存的 gate loss 消融，
   等于 1.4B PLC 配置加 `plc.use_gate_loss=false`。它登记在 `MATCHED_GROUPS` 里；
   `tests/test_plc.py` 还检查它与 PLC 配置只差这一项和 wandb 命名，所以改 PLC 配置时要同步改它。
   同样是用户要求单独存的（2026-11-07）：`plc_logp_softmax_1p4b_8gpu.yaml`（`plc.weight_mode=logp_softmax
   plc.prior=uniform`）和 `plc_rltt_progressive_1p4b_8gpu.yaml`（`plc.weight_mode=rltt plc.prior=progressive`），
   都等于 nogate 配置改这两个 `plc` 键和 `args.loop_weighting`（只影响原作脚本的 run 名）。`tests/test_plc.py`
   检查这一点，改 nogate 配置时要同步改它们。
   同样是用户要求单独存的（2026-11-08，2026-11-11 按用户要求改为信赖域形式）：
   `plc_grpo_loopaux_1p4b_8gpu.yaml`（GRPO + 中间轮 off-policy 辅助项：`plc.weight_mode=is_trunc
   plc.prior=uniform plc.lam=0.1 plc.objective=additive plc.loop_baseline=none plc.loop_agg_mode=same
   plc.trunc_cap=1.0 plc.kl_coef=0.2`，`args.loop_weighting=uniform`），等于 nogate 配置改这些键，
   `tests/test_plc.py` 同样检查。它的消融走覆盖：`plc.kl_coef=0.05 / 1.0`（信赖域大小）、
   `plc.kl_coef=0`（只有代理目标）、`plc.trunc_cap=2`、`plc.weight_mode=early_prior`（不乘比率）、
   `plc.lam=0`（GRPO）；`plc.kl_coef=0 plc.loop_baseline=group plc.loop_agg_mode=seq-mean-token-mean`
   是 2026-11-08 版本。
   同样是用户要求单独存的（2026-11-12）：`plc_grpo_loopaux_nokl_1p4b_8gpu.yaml`，等于 loopaux 配置改
   `plc.kl_coef=0`（去掉正向 KL 项，只留代理目标），用来区分中间轮熵坍塌来自 KL 项还是代理目标。
   `tests/test_plc.py` 检查它与 loopaux 配置只差这一项和 wandb 命名，改 loopaux 配置时要同步改它。

两个组之间只差模型（1.4B / 2.6B，`model.num_hidden_layers` 24 / 48；
`test_each_method_differs_between_sizes_only_in_the_model` 检查每种方法的两份配置）：都是每步 64 题 × 8 条
（512 条 rollout）、梯度累积 1、mini-batch 64 题、100 步 = 100 次优化器更新，warmup 10 步
（用户 2026-11-05 指定；论文 Table 6 是每步 32 题、累积 2、140 步）。
`test_matched_groups_are_one_update_per_step` 锁定这些值，改动要整组一起改。

## vLLM token 预算：32768

- 每份配置写 `verl_overrides.actor_rollout_ref.rollout.max_num_batched_tokens: 32768`。
  verl 0.6.1 默认 8192，而原作脚本开着 chunked prefill、`max_model_len` = 2048 + 16384 = 18432；
  这时 verl 要求 `max_num_batched_tokens >= max_model_len`，否则 vLLM 初始化直接报错。
- launcher 的 `check_rollout_token_budget` 在 dry-run 阶段就拒绝不满足这一条的配置。
- 同一处还显式写出其余与训练有关的 rollout 参数：`top_p` 1.0、`top_k` -1、`do_sample` true、
  `ignore_eos` false、`max_num_seqs` 1024、`dtype` bfloat16。取值等于 verl 0.6.1 默认值，写出来是为了
  不依赖默认值；它们是共享键，测试锁定，改任何一个都要整组一起改。

## KL：全部关闭

- 每份配置都写 `args.beta: 0.0`，并在 `verl_overrides.actor_rollout_ref.actor` 写 `use_kl_loss: false`、
  `kl_loss_coef: 0.0`。这是共享键，组内必须一致。
- 原作 RLTT 脚本没有注册参考模型 worker（只有 ActorRollout），它的 actor 在没有 `ref_log_prob` 时用
  `old_log_probs` 代替，所以 RLTT / PLC 即使开着 KL 也只是对旧策略算，接近 0。
- GRPO 脚本注册了参考模型；verl 0.6.1 只要注册了就会加载并每步前向一次，与 `use_kl_loss` 无关。
  launcher 在最终 verl 配置里 `use_kl_loss` 和 `use_kl_in_reward` 都为 false 时，从
  `role_worker_mapping` 里去掉 `Role.RefPolicy`，所以 KL 关闭时三种方法都不加载参考模型，
  每步耗时可以直接比较。结果记录在 `rltt_baseline_resolved.json` 的 `reference_policy` 字段。
- 要重新打开 KL，必须先解决 RLTT / PLC 没有参考模型的问题，否则三种方法的 KL 含义不同。

## 随机种子：全部锁定 42

- 训练配置 `args.seed: 42`（driver 的 `torch.manual_seed` / `set_seed`），并且
  `verl_overrides.data.seed: ${args.seed}`：verl 0.6.1 的 dataloader 打乱种子默认为空，不从 `args.seed`
  派生，必须显式写，否则数据顺序不受 seed 控制。改 seed 只改 `args.seed`，`data.seed` 跟着插值。
- vLLM 采样引擎的 seed 在 verl 0.6.1 里不可配置（`RolloutConfig` 没有 `seed` 字段，引擎固定用 0）；
  三种方法相同，但 temperature 1.0 下多卡批处理本身不保证逐位复现，不要指望两次训练 rollout 完全一致。
- 评估 / 诊断脚本默认 seed 也是 42：本仓库的 `rltt_baseline.diagnostics.rollouts` 与 `probe`
  （`--seed`，默认 42）；原作 `third_party/RLTT` 的评估脚本 `--seed` / `--base_seed` 默认本来就是 42，
  跑评估时不要传别的 seed，否则结果不能和同组比较。
- 新加任何带随机性的脚本，`--seed` 默认写 42。

## 配置文件看不出来、但会随 `trainer` 一起改变的东西

- `trainer: rltt`（RLTT 与 PLC）的 actor 用自定义前向：padded 输入、模型内分块算逐轮 log-prob；
  `trainer: grpo` 走 verl 标准 actor（remove-padding 路径）。数学等价，数值路径不同。KL 关闭时
  三者都没有参考模型（见上面"KL：全部关闭"）。
- exit gate 是否被训练由**模型目录**决定：`prepare_model` 默认写 `rltt_detach_exit_pdf: false`
  （PLC 的 gate loss 需要）。原作 RLTT 用 `exit_pdf` 权重时梯度会流进 gate，GRPO 则完全不碰 gate。
  三种方法共用 `LOOPRL_MODEL_DIR`，换模型目录相当于换了实验设置，要在实验记录里写明。
- 原作脚本会原地改模型目录：`rltt_train.py` 在 `rltt_logprob_chunk_size` 不一致时重写
  `config.json`，`grpo_train.py` 会改 `modeling_ouro.py`（launcher 拒绝官方目录以避免后者）。
  不要把 `LOOPRL_MODEL_DIR` 指向共享的官方模型目录。

## 其他

- 训练用 `bash scripts/run_train.sh <config> [overrides]`；它先跑 `--dry-run` 校验配置再启动。
  解析后的完整配置写到 `<output_dir>/rltt_baseline_config.yaml`，对比两次实验先 diff 这个文件。
- 单元测试只需 CPU：`pytest`。GPU 相关代码（verl / vLLM worker）只能在训练环境里验证。
- 代码风格：`ruff`，行宽 100；`third_party/` 不检查也不修改。
