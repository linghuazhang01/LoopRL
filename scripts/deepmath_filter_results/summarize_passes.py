"""Final scores of deepmath-filter-1p4b-16k-v1 from compact projections of the journals.

A trial is correct only if it ended naturally and its last \\boxed{} matches the gold answer under
grading.py (plain-text form, then math-verify); a 16K stop is wrong. Prints and
writes per-source totals and, for problems with all 4 trials, the number correct out of 4 (the
filter signal), overall and by answer type; passes.csv holds one row per problem.

    python summarize_passes.py --out <dir> proj1.jsonl.gz proj2.jsonl.gz
"""

import argparse
import collections
import csv
import gzip
import json
import multiprocessing
import re
from pathlib import Path

from grading import GRADER_VERSION, answers_match, plain


def match(pair: tuple[str, str]) -> bool:
    return answers_match(*pair)


def answer_type(gold: str) -> str:
    text = plain(gold)
    if text in ("yes", "no"):
        return "yes_no"
    if text in ("0", "1"):
        return "zero_one"
    return "integer" if re.fullmatch(r"-?\d+", text) else "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("projections", type=Path, nargs="+")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for path in args.projections:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["key"] in seen:
                    raise ValueError(f"Duplicate trial key: {row['key']}")
                seen.add(row["key"])
                rows.append(row)
    gradable = lambda r: r["finish_reason"] == "stop" and r["boxed_status"] == "found"  # noqa: E731
    pairs = sorted({(r["gold_answer"], r["boxed_answer"]) for r in rows if gradable(r)})
    with multiprocessing.Pool(args.jobs) as pool:
        verdicts = dict(zip(pairs, pool.map(match, pairs, chunksize=64)))
    for row in rows:
        row["correct"] = gradable(row) and verdicts[(row["gold_answer"], row["boxed_answer"])]

    def totals(subset: list[dict]) -> dict:
        n = len(subset)
        return {"trials": n, "correct": sum(r["correct"] for r in subset),
                "provisional_correct": sum(bool(r["provisional_correct"]) for r in subset),
                "length_stops": sum(r["finish_reason"] == "length" for r in subset),
                "missing_boxed_natural": sum(r["finish_reason"] == "stop"
                                             and r["boxed_status"] != "found" for r in subset),
                "mean_tokens": sum(r["generated_tokens"] for r in subset) / n if n else None}

    problems: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        problems[row["question_id"]].append(row)
    full = {q: trials for q, trials in problems.items() if len(trials) == 4}
    passes = {q: sum(t["correct"] for t in trials) for q, trials in full.items()}
    by_type: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for q, trials in full.items():
        by_type[answer_type(trials[0]["gold_answer"])][passes[q]] += 1
    summary = {"grader": GRADER_VERSION, "unique_answer_pairs": len(pairs),
               "all": totals(rows),
               "by_source": {s: totals([r for r in rows if r["source"] == s])
                             for s in sorted({r["source"] for r in rows})},
               "problems_with_4_trials": len(full),
               "pass_histogram": {str(k): sum(v == k for v in passes.values()) for k in range(5)},
               "pass_histogram_by_answer_type": {
                   t: {str(k): c[k] for k in range(5)} for t, c in sorted(by_type.items())}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out / "passes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["question_id", "row", "chunk", "source", "gold_answer", "answer_type",
                         "passes", "length_stops", "mean_tokens"])
        for q, trials in sorted(full.items(), key=lambda item: item[1][0]["order"]):
            first = trials[0]
            writer.writerow([q, first["row"], first["chunk"], first["source"],
                             first["gold_answer"], answer_type(first["gold_answer"]), passes[q],
                             sum(t["finish_reason"] == "length" for t in trials),
                             round(sum(t["generated_tokens"] for t in trials) / 4, 1)])
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
