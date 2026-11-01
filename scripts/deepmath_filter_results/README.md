# 从过滤采样结果到训练子集

把 `scripts/deepmath_filter_portable/` 跑出来的 journal 变成判分结果和过滤后的训练 parquet。

| 脚本 | 作用 | 依赖 |
|---|---|---|
| `project_journals.py` | 去掉生成文本和 token id，得到可以拷走的精简文件 | 只用标准库 |
| `summarize_passes.py` | 判分：要求自然结束，最后一个 `\boxed{}` 先比纯文本，再用 math-verify；写满 16K 算错。输出 `summary.json` 和逐题的 `passes.csv` | `math-verify` |
| `export_subset.py` | 按答对次数和答案类型取子集，输出 parquet、JSONL 和 manifest | `pyarrow`、`math-verify` |

```bash
# 在 journal 所在的机器上运行
python project_journals.py <采样输出目录> projection.jsonl.gz <来源标签>
# 可以同时传多个投影文件；同一个 key 出现两次会报错
python summarize_passes.py --out results projection.jsonl.gz
# 保留答对 1–3 次的题，并去掉靠猜也能答对的答案类型
python export_subset.py --source train_filtered_level6.parquet --passes results/passes.csv \
    --sample sample12000.jsonl --out filtered \
    --drop-answer-types yes_no zero_one --drop-true-false
```

`export_subset.py` 按行号从原 parquet 取子集，保持原来的顺序、schema 和元数据，并逐题核对题面和答案，写出后再读回校验。把 `LOOPRL_TRAIN_FILE` 指向输出的 parquet 就可以训练。
