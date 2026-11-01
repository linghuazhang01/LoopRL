"""Compact projection of deepmath-filter journals (standard library only).

Keeps what scoring needs and drops the generated text and token ids, so the result of a run can be
copied and scored elsewhere. A last line without its newline is an unfinished write and is skipped.

    python project_journals.py <output dir of the run> projection.jsonl.gz <source label>
"""
import gzip
import json
import sys
from pathlib import Path

root, out, source = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
KEEP = ("key", "question_id", "row", "order", "chunk", "replicate", "seed", "finish_reason",
        "boxed_answer", "boxed_status", "gold_answer", "generated_tokens", "thinking_tokens",
        "thinking_closed", "provisional_correct", "attention_backend", "host", "record_sha256")
n = 0
with gzip.open(out, "wt", encoding="utf-8") as sink:
    for journal in sorted(root.glob("chunks/c*/journal.jsonl")):
        payload = journal.read_bytes()
        if not payload.endswith(b"\n"):
            payload = payload[:payload.rfind(b"\n") + 1]
        complete = (journal.parent / "complete.json").exists()
        for line in filter(None, payload.decode("utf-8").split("\n")):
            record = json.loads(line)
            row = {k: record.get(k) for k in KEEP}
            row.update(source=source, chunk_complete=complete)
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
print(json.dumps({"records": n, "out": out}))
