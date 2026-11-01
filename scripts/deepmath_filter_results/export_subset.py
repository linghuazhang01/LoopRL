"""Write a GRPO training subset chosen from the deepmath-filter passes (no model, no GPU).

Rows are taken from the source parquet by their row index, in source order, so the file keeps the
verl schema and metadata and can be used as LOOPRL_TRAIN_FILE. Every kept row is checked against
the sampled problem file and the written parquet is read back. A JSONL in the sample format and a
manifest go next to it. The default keeps 1 to 3 correct out of 4; the first filtered set also
drops the answers a guess can hit:

    python export_subset.py --source train_filtered_level6.parquet --passes passes.csv \\
        --sample sample12000.jsonl --out <dir> \\
        --drop-answer-types yes_no zero_one --drop-true-false
"""

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from grading import plain

SOURCE_SHA256 = "de3350fdd00bc0410550098ea65179e2be873da99e4075f80de575fc17670597"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--passes", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-passes", type=int, default=1)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--drop-answer-types", nargs="*", default=[],
                        choices=["yes_no", "zero_one", "integer", "other"],
                        help="also drop these answer types (passes.csv labels)")
    parser.add_argument("--drop-true-false", action="store_true",
                        help="also drop gold answers True / False (plain-text form)")
    args = parser.parse_args()
    if sha256(args.source) != SOURCE_SHA256:
        raise ValueError("Training parquet differs from the pinned SHA256")
    with args.passes.open(newline="") as stream:
        graded = list(csv.DictReader(stream))
    sample = {row["row"]: row for row in map(
        json.loads, args.sample.read_text(encoding="utf-8").splitlines())}
    if len(graded) != len(sample) or {int(g["row"]) for g in graded} != set(sample):
        raise ValueError("Passes and sample cover different problems")
    kept = sorted((g for g in graded
                   if args.min_passes <= int(g["passes"]) <= args.max_passes
                   and g["answer_type"] not in args.drop_answer_types
                   and not (args.drop_true_false
                            and plain(g["gold_answer"]) in ("true", "false"))),
                  key=lambda g: int(g["row"]))
    table = pq.read_table(args.source)
    subset = table.take([int(g["row"]) for g in kept])
    rows = []
    for g, source in zip(kept, subset.to_pylist()):
        problem = sample[int(g["row"])]
        if (source["prompt"][0]["content"] != problem["question"]
                or str(source["reward_model"]["ground_truth"]) != problem["gold_answer"]
                or f"dm6-{source['extra_info']['index']:05d}" != g["question_id"]):
            raise ValueError(f"Row {g['row']} differs from the sampled problem")
        rows.append({**problem, "passes": int(g["passes"]), "answer_type": g["answer_type"]})
    args.out.mkdir(parents=True, exist_ok=True)
    short = {"yes_no": "yesno", "zero_one": "01", "integer": "integer", "other": "other"}
    dropped = [short[name] for name in sorted(args.drop_answer_types)]
    dropped += ["truefalse"] if args.drop_true_false else []
    suffix = "_no_" + "_".join(dropped) if dropped else ""
    parquet = args.out / (f"train_filtered_level6_passes{args.min_passes}to{args.max_passes}"
                          f"{suffix}.parquet")
    pq.write_table(subset, parquet)
    back = pq.read_table(parquet)
    if not back.schema.equals(table.schema, check_metadata=True) or back.num_rows != len(kept):
        raise ValueError("Written parquet differs from the source schema or row count")
    if back.to_pylist() != subset.to_pylist():
        raise ValueError("Written parquet differs from the selected rows")
    jsonl = args.out / f"sample{len(rows)}.jsonl"
    jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                     encoding="utf-8")
    manifest = {
        "rule": f"{args.min_passes} <= correct out of 4 <= {args.max_passes}",
        "dropped_answer_types": sorted(args.drop_answer_types),
        "dropped_true_false": args.drop_true_false,
        "source": "G-OPD DeepMath-103K/train_filtered_level6.parquet",
        "source_sha256": SOURCE_SHA256, "source_rows": table.num_rows,
        "sample_file": str(args.sample), "sample_sha256": sha256(args.sample),
        "passes_file": str(args.passes), "passes_sha256": sha256(args.passes),
        "problems_graded": len(graded), "problems_kept": len(rows),
        "kept_by_passes": dict(sorted(collections.Counter(
            str(r["passes"]) for r in rows).items())),
        "kept_by_answer_type": dict(sorted(collections.Counter(
            r["answer_type"] for r in rows).items())),
        "parquet": parquet.name, "parquet_sha256": sha256(parquet),
        "jsonl": jsonl.name, "jsonl_sha256": sha256(jsonl)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
