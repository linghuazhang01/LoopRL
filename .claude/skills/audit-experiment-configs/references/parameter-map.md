# 配置键速查：每个键在做什么、属于哪一类、原作默认值是什么

读配置时逐键对照这张表。三类：

- **方法键**：方法本身，组内允许不同。
- **等价键**：GRPO 脚本和 RLTT 脚本参数名不同、但设的是同一个 verl 值；要核对**取值等价**。
- **共享键**：其余一切，组内必须逐字相同。

原作脚本 = `third_party/RLTT/grpo_experiments/grpo_train.py`（`trainer: grpo`）和
`third_party/RLTT/rltt_experiments/rltt_train.py`（`trainer: rltt`，PLC 也走它）。`args` 里没写的键
取原作 argparse 默认值；两份脚本默认值不同的键见文末，审计时必须看。

## 顶层段

| 段 | 做什么 | 类别 |
|---|---|---|
| `trainer` | 选原作脚本。`grpo`：verl 标准 actor（remove-padding）；`rltt`：自定义 padded 前向、模型内分块算逐轮 log-prob。数学等价，数值路径不同 | 方法 |
| `args` | 原作 argparse 的 dest 名；未列出的用原作默认值 | 见下 |
| `wandb` | `enabled / mode / dir / project / entity` 共享（`project` 统一为 `looprl-ouro`）；`run_name / tags / notes` 只是命名 | 命名键为方法，其余共享 |
| `verl_overrides` | 原作填完 verl 配置后再深合并进去；用来改正原作与论文不一致处 | 见下 |
| `env` | driver 和 Ray worker 的环境变量 | 共享 |
| `rollout` | `kv_layout: full`（标准 vLLM）或 `decode_last`（实验性 KV 布局） | 共享 |
| `model` | `num_hidden_layers`：launcher 开训前核对模型目录 `config.json`，防止 1.4B 配置跑到 2.6B 权重（24 = 1.4B，48 = 2.6B） | 共享 |
| `audit` | `loop_distribution`：逐轮分布审计，写 `<output_dir>/audit/`；所有配置都打开，只读不改训练，每次运行几十 GB | 共享 |
| `plc` | Posterior Loop Credit 的全部开关（见下） | 方法 |

## `args`：模型与数据

| 键 | 含义 | 类别 / 备注 |
|---|---|---|
| `model_path` | GRPO：训练的模型目录（= `LOOPRL_MODEL_DIR`，prepare_model 生成）。RLTT：只用于 run 命名，指官方目录 `LOOPRL_OURO_DIR` | 等价：GRPO `model_path` ↔ RLTT `rltt_model_path`，都应是 `LOOPRL_MODEL_DIR` |
| `use_rltt_model` / `rltt_model_path` | RLTT 专有：用逐轮模型类，训练 `rltt_model_path` | 等价（见上）；`use_rltt_model` 必须 true |
| `rltt_logprob_chunk_size` | RLTT 专有：逐轮 log-prob 的分块大小，只影响显存；原作会把它**写回模型目录的 config.json** | 等价键（GRPO 无此项）；默认 256，配置用 2048 |
| `total_ut_steps` | 循环次数，Ouro 固定 4 | 共享 |
| `no_lora` | true = 全参数训练；参考模型是独立的冻结副本 | 共享 |
| `no_sft_checkpoint` | 不从 SFT checkpoint 起训 | 共享 |
| `no_gradient_checkpointing` | RLTT 专有；GRPO 脚本永远开 checkpointing，所以 RLTT 这里必须是 false | 等价 |
| `ref_offload` | RLTT 专有：参考模型 offload 到 CPU；GRPO 脚本不设（False），配置用 `verl_overrides.actor_rollout_ref.ref.fsdp_config.param_offload: true` 补 | 等价 |
| `train_file` | 训练集。`.parquet` 走 verl 数据集、用文件自带 prompt（SAPD DeepMath level≥6）；`.jsonl` 走原作 MATH 转换加 RLTT prompt | 共享，且决定 prompt 格式 |
| `math500_test_file` | 验证集（默认关着） | 共享 |
| `max_prompt_length` / `max_completion_length` | prompt 上限（超长的题被过滤）/ 生成上限 | 共享 |

## `args`：目标与采样

| 键 | 含义 | 类别 / 备注 |
|---|---|---|
| `use_ppo` | GRPO 专有，必须 false（true 会跑带 critic 的 PPO） | 等价键（RLTT 无此项） |
| `loop_weighting` | RLTT 专有：逐轮权重 `uniform / progressive / exit_pdf / learned`。PLC 开着时权重由 `plc` 段决定，这里只影响 run 命名 | 方法（RLTT 的目标就是它） |
| `progressive_alpha` | RLTT 专有，`progressive` 权重的指数 | 方法 |
| `beta` | KL 系数（`kl_loss_coef`，`low_var_kl`，加在 loss 上）。对照组统一为 0，且 `verl_overrides` 里 `use_kl_loss: false`、`kl_loss_coef: 0.0`；RLTT 脚本没有参考模型，KL 只能对旧策略算 | 共享 |
| `num_generations` | 每题采样数（组大小，verl `rollout.n`） | 共享 |
| `num_prompts_per_batch` | 每步题数。原作把 `data.train_batch_size` 设成 题数×采样数 并被 verl 当作题数，所以配置在 `verl_overrides.data.train_batch_size` 里重设 | 共享；要和 `verl_overrides.data.train_batch_size` 相等 |
| `temperature` | 采样温度 | 共享 |

## `args`：优化

| 键 | 含义 | 类别 / 备注 |
|---|---|---|
| `learning_rate` | AdamW8bit 学习率（两份脚本都用 bitsandbytes AdamW8bit） | 共享 |
| `weight_decay` | 权重衰减 | 共享 |
| `max_grad_norm` | 梯度裁剪。**RLTT 脚本把它传给 actor `grad_clip`；GRPO 脚本只传给没启用的 critic**，actor 用 verl 默认 1.0，所以 GRPO 配置要在 `verl_overrides...actor.grad_clip` 再写一次，且必须等于 `max_grad_norm` | 共享 + GRPO 等价键 |
| `warmup_ratio` | warmup 占总**步数**的比例；verl 调度器每次优化器更新前进一次，梯度累积 >1 时实际 warmup 比例会放大（累积 2 → 20%）。两个对照组现在都是累积 1，warmup = 前 14 次更新 | 共享 |
| `lr_scheduler_type` | RLTT 专有，`constant` 或 `cosine`（默认 cosine！）；GRPO 脚本硬编码 constant，所以 RLTT 必须显式写 `constant` | 等价 |
| `gradient_accumulation_steps` | 累积多少步做一次更新；verl 侧要配 `ppo_mini_batch_size = train_batch_size × 累积步数` | 共享 |
| `num_train_epochs` | `max_steps = -1` 时才生效 | 共享 |
| `max_steps` | 总步数（一步 = 一个 rollout batch）；论文 140 | 共享 |
| `seed` | 随机种子 | 共享 |

## `args`：显存、rollout、资源、验证

| 键 | 含义 | 备注 |
|---|---|---|
| `ppo_max_token_len` / `log_prob_max_token_len` | 动态 batch 时每卡 token 上限，须 ≥ prompt + 生成。**两份脚本默认值不同**（`ppo_max_token_len` GRPO 16384、RLTT 8192），所以必须显式写 | 共享 |
| `dataloader_num_workers` | 数据加载进程数（默认 8） | 共享 |
| `use_vllm` / `no_vllm` | 用 vLLM 做 rollout | 共享 |
| `vllm_gpu_memory_utilization` | vLLM 占显存比例（默认 0.7；`free_cache_engine` 原作写死 false，训练时不释放） | 共享 |
| `vllm_tensor_parallel_size` / `vllm_enforce_eager` | vLLM 并行度 / 是否关 CUDA graph | 共享 |
| `n_gpus` / `nnodes` | 卡数（<2 会落到原作单卡简易 trainer，launcher 拒绝） | 共享 |
| `output_dir` | 输出目录；launcher 在这里写 `rltt_baseline_config.yaml` 和 `rltt_baseline_resolved.json` | 共享（取环境变量） |
| `save_steps` / `logging_steps` | checkpoint 间隔 / 日志间隔 | 共享 |
| `no_validation` / `skip_initial_validation` / `eval_steps` / `val_batch_size` / `val_max_new_tokens` | 验证开关与参数；论文训练中不验证 | 共享 |

## `verl_overrides`：原作脚本与论文不一致之处

两份原作脚本都写死：`loss_agg_mode token-mean`、`ppo_epochs 1`、`entropy_coeff 0`、`clip_ratio 0.2`（GRPO 写死，RLTT 不设、取 verl 默认 0.2）、
`kl_loss_type low_var_kl`、`betas (0.9, 0.999)`、`ppo_mini_batch_size = min(128, 题数×采样数)`、
`data.train_batch_size = 题数×采样数`、`free_cache_engine false`、`use_dynamic_bsz true`、
`save_contents [model, optimizer, extra]`。配置用下面这些覆盖改正：

| 覆盖 | 为什么 | 类别 |
|---|---|---|
| `data.seed: ${args.seed}` | verl dataloader 打乱种子默认 null，不跟 `args.seed` 走 | 共享 |
| `data.train_batch_size` | 原作把题数×采样数当题数；改回每步题数 | 共享（= `args.num_prompts_per_batch`） |
| `actor.ppo_mini_batch_size` | 原作 128；改成 题数 × 累积步数，使"累积 N 步 = 一次更新" | 共享 |
| `actor.optim.betas` | 论文 Table 6 是 (0.9, 0.99) | 共享 |
| `actor.checkpoint.save_contents` | 加 `hf_model`，供逐轮梯度探测直接加载 | 共享 |
| `trainer.logger` | wandb + console | 共享 |
| `rollout.max_num_batched_tokens: 32768` | verl 默认 8192 < `max_model_len` 18432，开着 chunked prefill 时 verl 直接报错；launcher dry-run 会拒绝 | 共享 |
| `rollout.top_p 1.0 / top_k -1 / do_sample true / ignore_eos false / max_num_seqs 1024 / dtype bfloat16` | 与 verl 0.6.1 默认值相同，显式写出以免依赖默认值 | 共享 |
| `algorithm.use_kl_in_reward: false` | **GRPO 专有**：原作 GRPO 把 KL 加进 reward（组内奖励全同时会制造非零 advantage）；RLTT 脚本本来就是 false。launcher 看到 true 直接拒绝 | 等价 |
| `actor.grad_clip` | **GRPO 专有**：见 `max_grad_norm` | 等价，须 = `args.max_grad_norm` |
| `actor.use_kl_loss: false` + `actor.kl_loss_coef: 0.0` | KL 全部关闭；每份配置都写（RLTT 脚本写死 true，必须覆盖） | 共享 |
| `ref.fsdp_config.param_offload: true` | **GRPO 专有**：= RLTT 的 `ref_offload` | 等价 |

## `plc` 段（方法键，只在 PLC 配置里开）

`enabled`、`weight_mode`（posterior / rltt）、`prior`（uniform / progressive / exit）、`progressive_alpha`、
`gamma`（0 = RLTT 同先验）、`lam`（0 = GRPO，只算最后一轮）、`eps_floor`、`use_gate_loss`（需 `prior: exit`
且模型目录 `rltt_detach_exit_pdf: false`）、`gate_coef`、`gate_entropy_coef`、`use_is_correction`、`is_clip`、
`log_weight_stats`。其他配置里这段必须不存在或 `enabled: false`。消融用命令行覆盖这些键，不另存文件。

## 两份原作脚本的默认值差异（`args` 没写就会在组内不一致）

| 键 | GRPO 默认 | RLTT 默认 |
|---|---|---|
| `ppo_max_token_len` | 16384 | 8192 |
| `lr_scheduler_type` | （硬编码 constant） | cosine |
| `ref.param_offload` | （硬编码 False） | `ref_offload` 默认 False |
| `enable_gradient_checkpointing` | （硬编码 True） | `not no_gradient_checkpointing`，默认 True |
| actor `grad_clip` | （不设，verl 默认 1.0） | `max_grad_norm`，默认 0.1 |
| `use_kl_in_reward` | True | False |
| `wandb_project` / `output_dir` | 不同字符串 | 不同字符串（配置里都显式给了；`wandb.project` 统一为 `looprl-ouro`） |

其余共有键默认值相同（`beta 0.001`、`num_generations 8`、`num_prompts_per_batch 128`、`temperature 0.9`、
`max_prompt_length 512`、`max_completion_length 4096`、`learning_rate 1e-6`、`weight_decay 0.1`、
`max_grad_norm 0.1`、`warmup_ratio 0.1`、`save_steps 5`、`eval_steps 7`、`dataloader_num_workers 8`、
`vllm_gpu_memory_utilization 0.7`、`seed 42`、`total_ut_steps 4`）。注意这些默认值很多与论文不同
（题数 128、温度 0.9、生成 4096），所以正式配置应当把它们全部显式写出，不依赖默认。

## 配置文件里看不见、但随方法或环境变化的东西

- 参考模型：只有 GRPO 脚本注册参考模型。KL loss 和 KL in reward 都关时 launcher 把它去掉，
  `rltt_baseline_resolved.json` 的 `reference_policy` 应为 `dropped`（GRPO）或 `none`（RLTT / PLC）；
  若 GRPO 的 run 里是 `built`，它每步多跑一次参考模型前向，耗时不能和另外两种方法比。

- 模型目录 `config.json` 的 `rltt_detach_exit_pdf`：决定 exit gate 是否被训练。prepare_model 默认 false
  （PLC gate loss 需要）；RLTT 用 `exit_pdf` 权重时梯度因此也会流进 gate；GRPO 完全不碰 gate。
- `rltt_train.py` 会把 `rltt_logprob_chunk_size` 写回模型目录；`grpo_train.py` 会原地改 `modeling_ouro.py`。
  所以同组各 run 必须共用同一个、由 prepare_model 生成的 `LOOPRL_MODEL_DIR`。
- `LOOPRL_TRAIN_FILE`、`LOOPRL_MATH500_JSONL`、`LOOPRL_OUTPUT_DIR`：配置里只是 `${oc.env:...}`，
  真实取值只能在 `<output_dir>/rltt_baseline_config.yaml` 里看到。
