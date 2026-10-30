"""Sample problems from the RL training set (DeepMath-103K level >= 6) for ouro-kv-share-v1.

The source is the LoopRL training file `train_filtered_level6.parquet` (G-OPD / SAPD, 57,046
problems, verl format, SHA256 de3350fd...0597). A seeded uniform sample without replacement is
written as JSONL with the user message unchanged (question + "Please reason step by step, and put
your final answer within \\boxed{}."), the ground truth and the source row index, plus a
manifest. CPU only. ``--exclude`` draws only from rows absent from earlier samples (their
JSONL files; the manifest records each one's SHA256), e.g. the 20,000 filter problems
disjoint from the first 12,000 (2026-10-30). ``--rest`` takes every row absent from the
excluded files instead of a draw, e.g. all 45,046 problems but the first 12,000:

    .venv/bin/python code/scripts/prepare_deepmath_level6_sample.py --source <parquet>
    .venv/bin/python code/scripts/prepare_deepmath_level6_sample.py --source <parquet> \\
        --n 20000 --seed 20261030 --out code/data/processed/deepmath_level6_filter20k \\
        --exclude code/data/processed/deepmath_level6_filter12k/sample12000.jsonl
    .venv/bin/python code/scripts/prepare_deepmath_level6_sample.py --source <parquet> --rest \\
        --out code/data/processed/deepmath_level6_filter_rest45k \\
        --exclude code/data/processed/deepmath_level6_filter12k/sample12000.jsonl
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_SHA256 = "de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597"
SOURCE_ROWS = 57046
OUT = Path("code/data/processed/deepmath_level6_sample500")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[],
                        help="earlier sample JSONL files whose rows are not drawn again")
    parser.add_argument("--rest", action="store_true",
                        help="every row not in the excluded files (ignores --n and --seed)")
    args = parser.parse_args()
    if args.rest and not args.exclude:
        parser.error("--rest needs --exclude")
    if sha256(args.source) != SOURCE_SHA256:
        raise ValueError("Training parquet differs from the pinned SHA256")
    frame = pd.read_parquet(args.source)
    if len(frame) != SOURCE_ROWS:
        raise ValueError("Training parquet does not hold 57,046 rows")
    excluded = sorted({json.loads(line)["row"] for path in args.exclude
                       for line in path.read_text(encoding="utf-8").splitlines() if line})
    pool = np.setdiff1d(np.arange(len(frame)), np.array(excluded, dtype=int))
    if args.rest:
        picks = pool
        args.n, args.seed = len(pool), None
    elif args.exclude:
        picks = np.sort(pool[np.random.RandomState(args.seed).choice(len(pool), args.n,
                                                                     replace=False)])
    else:  # the original draw over all rows (keeps earlier samples reproducible)
        picks = np.sort(np.random.RandomState(args.seed).choice(len(frame), args.n,
                                                                replace=False))
    rows = []
    for i in picks:
        row = frame.iloc[int(i)]
        messages = list(row["prompt"])
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError(f"Row {i} is not a single user message")
        rows.append({"id": f"dm6-{int(row['extra_info']['index']):05d}", "row": int(i),
                     "question": messages[0]["content"],
                     "gold_answer": str(row["reward_model"]["ground_truth"])})
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Sampled IDs are not unique")
    args.out.mkdir(parents=True, exist_ok=True)
    data = args.out / f"sample{args.n}.jsonl"
    data.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    manifest = {"source": "G-OPD DeepMath-103K/train_filtered_level6.parquet",
                "source_sha256": SOURCE_SHA256, "source_rows": SOURCE_ROWS,
                "sampler": "numpy RandomState(seed).choice(rows, n, replace=False), sorted",
                "seed": args.seed, "n": args.n, "file": data.name, "file_sha256": sha256(data)}
    if args.exclude:
        manifest["sampler"] = ("every row not in the excluded files, sorted" if args.rest else
                               "numpy RandomState(seed).choice(rows not in the excluded files, "
                               "n, replace=False), sorted")
        manifest["excluded"] = [{"file": str(path), "sha256": sha256(path)}
                                for path in args.exclude]
        manifest["excluded_rows"] = len(excluded)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
