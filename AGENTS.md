# LoopRL：给所有代码 agent 的硬性规则

本文件适用于在本仓库工作的所有 agent（Claude、Codex 等）。`CLAUDE.md` 写配置对照组、KL、
分支等细节；两者冲突时以本文件为准。下面三条规则没有例外，除非用户在对话里明确要求改变。

## 1. 训练数据：只用 14,684 条合并过滤训练集

- 所有训练（GRPO、RLTT、PLC，1.4B 和 2.6B，正式实验和消融）都用同一份数据：
  **14,684 条合并过滤训练集**。它由 12K 批次过滤出的 2,865 条和 rest45K 批次过滤出的
  11,819 条合并而成。

  | 项 | 值 |
  |---|---|
  | HF 数据仓库（私有） | `icemoon28/LoopRL-DeepMath-14684` |
  | 固定 revision | `6e0be984159c007d0abd3d717a55617774390907` |
  | 文件 | `data/train.parquet`，split `train` |
  | 条数 | 14,684 |
  | Parquet SHA256 | `353b572a7c3a492b1e50d87e4d5195a461950018331c12258ff704d60c480b41` |

- `LOOPRL_TRAIN_FILE` 必须指向这份文件。下载与校验命令见 README“过滤后的训练集”一节。
- **启动训练前必须核对**：文件的 SHA256 等于上表的值，行数为 14,684。任一项不符就不启动，
  先告诉用户。
- 不要用下面这些数据训练：仓库里的 2,865 条分量
  （`data/deepmath_level6_filter12k_passes1to3_no_yesno_01_truefalse/`）、未过滤的 57,046 题
  `deepmath_level6.parquet`、5,234 题或 3,006 题的中间版本、MATH 训练集。

## 2. 训练 seed 必须是 42

- 所有配置都写 `args.seed: 42`，`verl_overrides.data.seed: ${args.seed}` 让数据打乱也用 42。
- 不要在命令行覆盖 `args.seed` 或 `verl_overrides.data.seed`，也不要为了"多 seed"另开运行。
- 评估和诊断脚本的 seed 也用默认的 42，不要传别的值。
- 新写任何带随机性的脚本，`--seed` 默认值写 42。

## 3. 每次启动训练前，必须把全部配置参数告诉用户

适用于每一次训练启动，包括 smoke run、消融和重跑。顺序如下：

1. 用和正式启动**完全相同**的环境变量和命令行覆盖，先跑 dry-run：

   ```bash
   python -m rltt_baseline.launch --config <配置文件> --dry-run [与启动时相同的覆盖]
   ```

   dry-run 不启动 Ray，只打印解析后的完整配置（JSON）。`scripts/run_train.sh` 内部也会跑一次，
   但会丢弃输出，所以要单独跑。

2. 把下面这些内容完整告诉用户，再启动训练：
   - **代码**：`git rev-parse HEAD`、分支名（必须是 `main`）、工作区是否干净。
   - **配置文件**和全部**命令行覆盖**（`cli_overrides`）。
   - **环境变量**的实际值：`LOOPRL_TRAIN_FILE`、`LOOPRL_MODEL_DIR`、`LOOPRL_OURO_DIR`、
     `LOOPRL_MATH500_JSONL`、`LOOPRL_OUTPUT_DIR`。
   - **数据核对结果**：训练文件的 SHA256 和行数（规则 1）。
   - **全部参数**：dry-run 输出中的 `vendor_args`（原作脚本的全部参数，包括配置没写、取默认值的）、
     `verl_overrides`（含 `max_num_batched_tokens`、KL、mini-batch、betas、审计设置）、
     `config.plc`、`config.rollout`、`config.audit`、`config.wandb`、`env`、`reference_policy`。
     按类别整理成表格，不能只给摘要或只列改动过的项。
   - **与对照组的差异**：这次运行和同组其他方法的配置相比，除方法本身外是否还有差异；
     有差异必须逐项指出。

3. 发现违反规则 1 或规则 2，或者有用户没要求的共享参数差异时，不启动，先告诉用户。

4. 启动后告诉用户输出目录，以及 `<output_dir>/rltt_baseline_resolved.json` 的位置；这份文件
   记录了本次运行实际使用的全部参数。
