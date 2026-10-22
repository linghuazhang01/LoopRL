"""Write the MATH training set as the JSONL the RLTT trainer reads.

RLTT trains on the Hendrycks MATH train split (7,500 problems; ``run_rltt.slurm`` hard-codes
``DATASET_SIZE=7500``). The vendor code reads ``problem`` and ``solution`` (the gold answer is
the last ``\\boxed{}`` of the solution) and logs ``level`` and ``type``.

Accepted sources (already downloaded; this script does no network access):
- the original release tree ``<root>/train/<subject>/<id>.json``;
- parquet or JSONL files with columns problem, solution, level, type (for example a local
  copy of the Hugging Face ``EleutherAI/hendrycks_math`` train splits).

    python -m rltt_baseline.prepare_math_train --source <dir> --out <file.jsonl>
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

FIELDS = ("problem", "solution", "level", "type")
EXPECTED_SIZE = 7500


def _records_from_tree(root: Path):
    for path in sorted((root / "train").glob("*/*.json")):
        item = json.loads(path.read_text())
        yield {k: item.get(k, "") for k in FIELDS}


def _records_from_tables(root: Path):
    files = sorted(root.rglob("*.parquet")) + sorted(root.rglob("*.jsonl"))
    files = [f for f in files if "test" not in f.name.lower()]
    for path in files:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            for item in pq.read_table(path).to_pylist():
                yield {k: item.get(k, "") for k in FIELDS}
        else:
            with open(path) as fh:
                for line in fh:
                    if line.strip():
                        item = json.loads(line)
                        yield {k: item.get(k, "") for k in FIELDS}


def collect(source: Path):
    records = list(_records_from_tree(source)) if (source / "train").is_dir() else []
    if not records:
        records = list(_records_from_tables(source))
    seen, unique = set(), []
    for rec in records:
        key = rec["problem"].strip()
        if key and key not in seen and "boxed" in rec["solution"]:
            seen.add(key)
            unique.append(rec)
    return records, unique


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-size-mismatch", action="store_true",
                        help=f"write even if the count differs from {EXPECTED_SIZE}")
    args = parser.parse_args(argv)
    raw, unique = collect(args.source)
    if len(unique) != EXPECTED_SIZE and not args.allow_size_mismatch:
        raise SystemExit(f"found {len(unique)} usable problems (raw {len(raw)}), "
                         f"expected {EXPECTED_SIZE}; pass --allow-size-mismatch to keep them")
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; refusing to overwrite")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for rec in unique:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(json.dumps({"out": str(args.out), "records": len(unique), "raw": len(raw),
                      "sha256": digest}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
