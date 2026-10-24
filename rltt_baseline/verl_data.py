"""Use a verl-format parquet as the training set, with its own prompts.

The vendor trainers read MATH-style JSONL and build every prompt themselves (RLTT instruction +
``Problem: ...``). When ``train_file`` ends in ``.parquet`` the launcher instead copies the file
through ``copy_verl_parquet``: the ``prompt`` messages are kept unchanged, so the dataset's own
instruction is what the model sees.

Required columns: ``prompt`` (list of ``{"role", "content"}`` messages), ``data_source`` and
``reward_model.ground_truth`` (non-empty string). ``extra_info`` is kept; ``problem`` is added
when missing (the last user message), because the vendor reward logs it.

An example is the SAPD math training set, ``DeepMath-103K/train_filtered_level6.parquet`` from
the Hugging Face dataset ``icemoon28/MOPD-Training-Data`` (``rltt_baseline.prepare_data``).
"""

from pathlib import Path


def _check_row(i: int, row: dict) -> None:
    messages = row.get("prompt")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"row {i}: prompt must be a non-empty list of messages")
    for msg in messages:
        if not isinstance(msg, dict) or not msg.get("role") or msg.get("content") is None:
            raise ValueError(f"row {i}: each message needs role and content")
    gold = (row.get("reward_model") or {}).get("ground_truth")
    if not isinstance(gold, str) or not gold.strip():
        raise ValueError(f"row {i}: reward_model.ground_truth must be a non-empty string")
    if not row.get("data_source"):
        raise ValueError(f"row {i}: data_source is missing")


def load_rows(path) -> list:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    if not rows:
        raise ValueError(f"{path}: no rows")
    for i, row in enumerate(rows):
        _check_row(i, row)
        info = dict(row.get("extra_info") or {})
        if "problem" not in info:
            users = [m["content"] for m in row["prompt"] if m["role"] == "user"]
            info["problem"] = users[-1] if users else ""
        row["extra_info"] = info
    return rows


def copy_verl_parquet(src, dst) -> str:
    """Validate ``src`` and write it to ``dst`` with prompts unchanged; return ``dst``."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = load_rows(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), dst)
    return str(dst)
