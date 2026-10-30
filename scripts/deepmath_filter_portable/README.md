# DeepMath 过滤采样便携包（deepmath-filter-1p4b-16k-v1，批次 rest45k）

用途：GRPO 数据过滤。RL 训练集是 DeepMath-103K level ≥ 6，共 57,046 题。用 Ouro-1.4B-Thinking 在 D4 上每题采 4 次，过滤依据是 4 次中答对几次。

57,046 题分成两批，互不重合，合起来正好覆盖整个训练集：

| 批次 | 题数 | chunk 数 | 样本数 | 选法 | 在哪里跑 |
|---|---:|---:|---:|---|---|
| 12K | 12,000 | 240 | 48,000 | seed 20261028 随机抽 | H200（不在本包） |
| `rest45k` | 45,046 | 901（最后一个 46 题） | 180,184 | 12K 以外的全部题目 | 本包 |

`rest45k` 由两部分合并而成：之前单独打包的 `ext20k`（排除 12K 后按 seed 20261030 抽的 20,000 题）和剩下的 25,046 题。合并后题目按 seeded shuffle 混在一起，跑到一半时已完成的部分也是随机子集。`sample.py` 仍认识 `ext20k`，用于核对旧 ext20k 包的输出；新的采样只用 rest45k。

包里只带一批题目，放在 `data/` 下。`sample.py` 会自动识别是哪一批，`--status` 的输出里也写着 `batch`。

采样配方与 H200 上 12,000 题的那次完全相同，key 和 seed 的规则也相同，两批之后可以直接合并：

- 模型：`ByteDance/Ouro-1.4B-Thinking`，revision `3aaa2224`，默认 4 轮循环（D4）。
- prompt：parquet 里的 user message 原样，套 Thinking 模板（`enable_thinking=True`）。
- 采样：temperature 1.0、top_p 1.0，生成上限 16,384 tokens。
- 每题 4 个样本，seed 由（题号, 第几个样本）哈希得到。

## 文件

| 文件 | 作用 |
|---|---|
| `sample.py` | 采样主程序；只依赖 vLLM |
| `run_worker.sh` | 在一张卡上跑一个 worker |
| `run_all.sh` | 一次给多张卡各起一个 worker，后台运行，退出登录也不会停 |
| `stop_all.sh` | 停止 worker |
| `fetch_model.py` | 下载模型的 9 个固定文件并核对 SHA256 |
| `requirements.txt` | 环境要求：`vllm==0.15.0`（H200 上实测用的版本） |
| `data/sample45046.jsonl`、`data/manifest.json` | 45,046 题及其来源与 SHA |
| `SHA256SUMS` | 本包所有文件的 SHA256 |

## 步骤（在包目录下执行）

1. **环境**：Python ≥ 3.10。
   ```bash
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   ```
   用 uv 也可以：`uv venv && uv pip install -r requirements.txt`。脚本会自动使用包目录下的 `.venv`。Triton 编译时需要 `Python.h`，系统要装好 `python3-dev`。

2. **模型**：
   ```bash
   .venv/bin/python fetch_model.py
   ```
   国内机器可先 `export HF_ENDPOINT=https://hf-mirror.com`。如果是从别处拷来的模型目录，用 `.venv/bin/python sample.py --check-model <目录>` 核对，再用 `MODEL_DIR=<目录>` 指定。目录里只能有这 9 个文件：多了 `generation_config.json` 会改变 vLLM 的采样默认值，脚本会拒绝。

3. **预检**：两步都不写 `outputs/`。
   ```bash
   .venv/bin/python sample.py --plan-only
   CUDA_VISIBLE_DEVICES=0 .venv/bin/python sample.py --smoke
   ```
   - `--plan-only` 不用 GPU，把本批全部 prompt 过一遍 tokenizer，应得到最长约 740 tokens。
   - `--smoke` 用一张卡约 1–2 分钟，检查引擎带 4 轮循环正常加载、能读到 KV 容量、流式取回的 token 与 `llm.generate` 一致。

4. **启动**：每张卡一个独立进程，例如四张卡：
   ```bash
   bash run_all.sh 0 1 2 3
   ```
   - 也可以单独起一张卡：`bash run_worker.sh 0`。
   - 采样参数可以接在 `--` 后面，例如 `bash run_all.sh 0 1 -- --attention-backend FLASH_ATTN --gpu-memory-utilization 0.85`。H200 上用的是 FLASH_ATTN、显存 0.9；不指定 attention backend 时由 vLLM 自己选。
   - 卡上剩余显存不够 `--gpu-memory-utilization`（默认 0.9）时，worker 会直接退出。
   - 同时在跑的请求数上限默认按 KV 容量算，每约 3,500 个 KV token 一个，限制在 8–64 之间（H200 约 48，80G A100 约 25）。可用 `--concurrency` 改。

5. **看进度**：`.venv/bin/python sample.py --status`。日志在 `logs/gpu<N>.log`，各卡进度在 `outputs/workers/`。

6. **停止与续跑**：`bash stop_all.sh`，或只停某几张卡 `bash stop_all.sh 2 3`。已提交的样本保留，正在生成的请求丢弃。再执行 `run_all.sh` 或 `run_worker.sh` 会从 journal 接着跑，不会重复。中途加卡、减卡都可以：各卡用文件锁认领 chunk，互不重复。

7. **完成后**：
   ```bash
   .venv/bin/python sample.py --verify --export-passes passes.csv
   ```
   核对所有 chunk 和样本各出现且只出现一次，并输出每题 provisional 答对次数的 CSV。然后把整个 `outputs/` 目录带回来，离线用 math-verify 正式判分。`outputs/` 含生成文本和 token ids，按平均 7.8K tokens 估计约 14–16 GB。

## 耗时参考（未在别的机器实测）

- 1.4B-Thinking 在这些题上平均约 7,760 tokens/条，180,184 条约 14 亿 tokens。
- H200 上实测稳定后每卡约 470 tok/s，约需 830 卡·小时：8 张 H200 约 4.3 天。
- A100 的显存带宽约为 H200 的 40%，估计慢 2–2.5 倍。以 `--status` 里的 `session_tokens_per_second` 为准。
