# 12K 过滤采样的 A100 分担（记录）

2026-10-30 起用 A100 分担 H200 上 12,000 题的过滤采样时用过的脚本，已跑完。这里保留的是当时的原样文件，**路径按 RecurrentTransformer 的目录结构写**（`code/…`、`budget_inference`），在本仓库里不能直接运行。新的采样用 `scripts/deepmath_filter_portable/`，判分和导出用 `scripts/deepmath_filter_results/`。

| 文件 | 作用 |
|---|---|
| `ouro_deepmath_filter_a100.py` | 加载便携采样器的冻结副本，换成 12K 题目，只认领名单里本节点的 chunk |
| `portable_sample_75fe831f.py` | 便携采样器的冻结副本，SHA256 开头 `75fe831f` |
| `run_ouro_deepmath_filter_a100_queue.sh` | Slurm 作业里的队列：卡连续空闲 30 分钟后先做 smoke，再起 worker，不停止别人的进程 |
| `hold_deepmath_filter_chunks.py` | 在另一台机器上持有名单中 chunk 的 claim 锁，让那边的 worker 跳过这些 chunk |
