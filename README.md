# LoopRL

在 looped language model（Ouro）上做 RL 的实验代码。目前包含 **RLTT baseline**：
Williams & Tureci, *Prioritize the Process, Not Just the Outcome: Rewarding Latent Thought
Trajectories Improves Reasoning in Looped Language Models*, arXiv 2602.10520。

原作代码（[jonwill8/RLTT](https://github.com/jonwill8/RLTT)）以 git submodule 的形式固定在
commit `06189850`，不做任何修改。本仓库只补上原作缺少的部分和我们需要的功能。

> 状态：CPU 单元测试和 tiny 随机模型的集成测试已通过；**尚未在真实权重或 GPU 上运行训练**。

## 本仓库补充了什么

| 模块 | 作用 |
|---|---|
| `rltt_baseline/ouro_rltt/` | 原作训练时加载 `Ouro-*-Thinking-RLTT` 模型，要求 forward 返回逐轮 log-prob、熵和 `exit_pdf`，但原仓库没有提供这份模型代码。这里按原作 actor 的调用接口重新实现 |
| `rltt_baseline/prepare_model.py` | 用官方 `Ouro-2.6B-Thinking` 目录生成上述模型目录：权重用软链接，官方 `modeling_ouro.py` 先核对 SHA256，再改名为 `modeling_ouro_base.py` |
| `rltt_baseline/config.py`、`launch.py` | YAML 配置，支持命令行 dot-list 覆盖；`verl_overrides` 可以改任意 verl 配置 |
| `rltt_baseline/metrics.py` | driver 端的推理 RL 指标（训练正确率、组内全对/全错/混合比例、截断率、正确与错误回答长度等） |
| `rltt_baseline/workers.py` | 额外记录逐轮熵、期望退出轮次和总 loss，原作的 loss 计算不变 |
| `rltt_baseline/prepare_math_train.py` | 把已下载的 MATH 训练集（7,500 题）转成原作读取的 JSONL |

对官方 Ouro 代码只做了一处修改：transformers 4.57（原作 requirements 锁定的版本）把
`check_model_inputs` 改成了 decorator factory，官方 `modeling_ouro.py` 直接加载会报
`wrapped_fn() got an unexpected keyword argument`。`prepare_model` 在生成的
`modeling_ouro_base.py` 中加了一个兼容 wrapper，同时支持新旧两种接口。

## 目录

```
rltt_baseline/            适配代码（Python 包）
configs/rltt_baseline.yaml  论文 Table 6 口径的默认配置
scripts/run_rltt_baseline.sh  在 Slurm 计算节点上启动训练
tests/                    单元测试与 tiny 模型集成测试
third_party/RLTT          原作仓库（submodule，固定 commit）
third_party/ouro_source   官方 Ouro modeling 文件（Apache-2.0，见该目录 README）
```

## 使用

```bash
git clone --recurse-submodules https://github.com/linghuazhang01/LoopRL.git
```

1. **环境**：按 `third_party/RLTT/requirements.txt` 建一个独立 venv（verl 0.6.1、vLLM 0.12.0、
   torch 2.9.0、transformers 4.57.3），再安装 `omegaconf`、`pyyaml`。
2. **数据**：把 MATH 训练集转成 JSONL；MATH-500 可直接用 `HuggingFaceH4/MATH-500` 的 test.jsonl。
   ```bash
   python -m rltt_baseline.prepare_math_train --source <MATH 训练集目录> --out <math_train.jsonl>
   ```
3. **模型目录**：
   ```bash
   python -m rltt_baseline.prepare_model --source <Ouro-2.6B-Thinking 目录> --out <Ouro-2.6B-Thinking-RLTT 目录>
   ```
4. **配置**：设置 `OURO_THINKING_DIR`、`RLTT_MODEL_DIR`、`RLTT_MATH_TRAIN_JSONL`、
   `RLTT_MATH500_JSONL`、`RLTT_OUTPUT_DIR`，先检查解析结果：
   ```bash
   python -m rltt_baseline.launch --config configs/rltt_baseline.yaml --dry-run
   ```
5. **训练**（需要 ≥ 2 张 GPU；论文用 4×H200）：
   ```bash
   RLTT_VENV=<venv> bash scripts/run_rltt_baseline.sh configs/rltt_baseline.yaml args.max_steps=140 wandb.mode=offline
   ```

### 配置与覆盖

`configs/rltt_baseline.yaml` 有四节：`args`（与原作 `rltt_train.py` 的 argparse 参数同名，
未列出的用原作默认值；拼错的键或类型不符会直接报错）、`wandb`、`verl_overrides`、`env`。
命令行覆盖示例：`args.learning_rate=2e-6 verl_overrides.trainer.save_freq=20 wandb.run_name=try1`。
解析后的完整配置会保存到 `<output_dir>/rltt_baseline_config.yaml`。

### wandb

日志通过 verl 0.6.1 的 `Tracking` 写入 wandb。`wandb.mode` 默认 `offline`（计算节点通常不能联网），
训练后在能联网的机器上运行 `wandb sync <output_dir>/wandb/offline-run-*`。

### 训练指标

verl 自带：`critic/score/mean`（平均奖励）、`response_length/*`（含 `clip_ratio`）、
`actor/pg_loss`、`actor/kl_loss`、`actor/grad_norm`、`actor/lr`、`actor/entropy`（最后一轮）、
时间与吞吐。原作另记 `rltt/loop_<t>_log_prob` 和各轮退出概率。本仓库新增：

- `reasoning/accuracy`（训练正确率）、`reasoning/pass_at_group`、
  `reasoning/group_{all_correct,all_wrong,mixed}_ratio`、`reasoning/group_acc_std_mean`；
- `reasoning/truncation_rate`、`reasoning/truncated_accuracy`、
  `reasoning/response_length/{correct,wrong}_mean`、`reasoning/answer_rate`（有 `\boxed{}`）、
  `reasoning/nonzero_adv_token_ratio`、`reasoning/accuracy/level_<k>`；
- `rltt/loop_<t>_entropy`、`rltt/expected_exit_step`、`actor/total_loss`。

## 测试

```bash
pip install pytest torch omegaconf pyyaml numpy
python -m pytest tests/test_rltt_baseline.py                 # 不依赖 verl / ray
pip install "transformers==4.57.3"
python -m pytest tests/test_rltt_baseline_modeling.py        # tiny 随机 Ouro，需要 transformers >= 4.56
```

## 已知差异与注意事项

- 原作论文写 140 步、Ouro-2.6B-Thinking；原作 `run_rltt.slurm` 默认 1.4B、5 个 epoch。默认配置按论文。
- 原作 actor 没有对 `exit_pdf` 权重做 detach，梯度会流入 exit gate；`prepare_model --detach-exit-pdf` 可关闭。
- 不支持熵正则（`entropy_coeff` 必须为 0），逐轮熵只用于记录。
- 少于 2 张 GPU 时原作会换用自己的单卡训练器，本仓库的配置覆盖和新增指标不会生效，因此 launcher 直接拒绝。
