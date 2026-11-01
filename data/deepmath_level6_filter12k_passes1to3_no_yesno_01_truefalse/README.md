# 过滤后的 DeepMath 训练集（2,865 题）

来源是 SAPD 数学训练集（DeepMath-103K 难度 ≥ 6，57,046 题）。按 seed 20261028 随机抽 12,000 题，用 Ouro-1.4B-Thinking（4 次循环，temperature 1.0，top_p 1.0，生成上限 16K）每题采样 4 次，再按下面的规则过滤：

1. 保留 4 次中答对 1–3 次的题，去掉 0/4 和 4/4；
2. 去掉答案是 Yes/No、0/1 的题；
3. 去掉答案是 True/False 的题。

判分口径是 `last-boxed-plain-or-math-verify-natural-eos-v1`：取最后一个 boxed，先比纯文本，再用 math-verify，写满 16K 的算错。判分代码见 `scripts/deepmath_filter_results/`。

## 文件

| 文件 | 内容 |
|---|---|
| `train_filtered_level6_passes1to3_no_yesno_01_truefalse.parquet` | verl 格式，按行号从原 parquet 取子集，保持原顺序、schema 和元数据，可直接作为 `LOOPRL_TRAIN_FILE` |
| `sample2865.jsonl` | 同样的 2,865 题，列为 `id`、`row`（原 parquet 行号）、`question`、`gold_answer`、`passes`（4 次中答对几次）、`answer_type` |
| `manifest.json` | 规则、来源和 SHA256。其中 `sample_file`、`passes_file` 是原项目里的路径，不在本仓库 |

## 构成

| 答对次数 | 题数 |
|---|---|
| 1/4 | 722 |
| 2/4 | 854 |
| 3/4 | 1,289 |

答案类型：integer 843 题，other 2,022 题。

## 使用前注意

- **数量**：140 步 × 32 题需要 4,480 题，这份只够约 89 步不重复。
- **未处理的题**：金标是单个字母 A/B/C/D 或 a/d 的 64 题还没有逐题看，可能是选项，也可能是数学变量。
- **只用 1.4B 过滤**：`passes` 是 Ouro-1.4B-Thinking 的结果，换模型后难度分布可能不同。

## 来源与许可

题目来自 [`icemoon28/MOPD-Training-Data`](https://huggingface.co/datasets/icemoon28/MOPD-Training-Data)（Apache-2.0）的 `DeepMath-103K/train_filtered_level6.parquet`，原始数据集是 [`zwhe99/DeepMath-103K`](https://huggingface.co/datasets/zwhe99/DeepMath-103K)（MIT）。
