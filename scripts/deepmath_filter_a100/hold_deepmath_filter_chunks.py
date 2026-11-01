#!/usr/bin/env python3
"""Keep the H200 workers of deepmath-filter-1p4b-16k-v1 off the chunks the A100 workers run.

Run on the H200 server from the project root (user request 2026-10-30: the A100 hold jobs take
part of the 12,000 problems). Every 30 s it reads <root>/reserved/a100_chunks.json
({"chunks": [...]}) and keeps a non-blocking flock on chunks/cNNN/claim.lock for every listed
chunk that is not complete. An H200 worker's claim of a held lock fails, so it skips the chunk
(ouro_deepmath_filter_sampling.next_claim). A listed chunk that an H200 worker holds, or that
already has an H200 journal, is reported as a conflict and left alone. Chunks dropped from the
list or completed (complete.json, e.g. copied over from the A100 output) are released.

Status: <root>/reserved/holder_status.json. The holder writes no journal and starts, stops or
signals no other process; a second holder exits with status 3. Stop: touch
<root>/reserved/holder.stop (releases every lock and exits 0), or SIGTERM / SIGINT.

    nohup python3 code/scripts/hold_deepmath_filter_chunks.py \
        --root code/outputs/ouro_deepmath_filter_1p4b_16k_v1 >> logs/... 2>&1 &
"""

import argparse
import fcntl
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def chunk_dir(root: Path, chunk: int) -> Path:
    return root / "chunks" / f"c{chunk:03d}"


def is_complete(root: Path, chunk: int) -> bool:
    marker = chunk_dir(root, chunk) / "complete.json"
    try:
        return marker.exists() and json.loads(marker.read_text())["chunk"] == chunk
    except (OSError, ValueError, KeyError):
        return False


def has_journal(root: Path, chunk: int) -> bool:
    journal = chunk_dir(root, chunk) / "journal.jsonl"
    return journal.exists() and journal.stat().st_size > 0


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True))
    os.replace(temporary, path)


class Holder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.held: dict[int, Any] = {}
        self.conflicts: dict[int, str] = {}

    def wanted(self, reserve_file: Path) -> list[int]:
        if not reserve_file.exists():
            return []
        return sorted({int(c) for c in json.loads(reserve_file.read_text())["chunks"]})

    def sync(self, wanted: list[int]) -> None:
        keep = set(wanted)
        for chunk in list(self.held):
            if chunk not in keep or is_complete(self.root, chunk):
                self.held.pop(chunk).close()  # closing the descriptor drops the flock
        conflicts: dict[int, str] = {}
        for chunk in wanted:
            if chunk in self.held or is_complete(self.root, chunk):
                continue
            if has_journal(self.root, chunk):  # started on the H200 server: leave it there
                conflicts[chunk] = "h200_journal"
                continue
            path = chunk_dir(self.root, chunk) / "claim.lock"
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                conflicts[chunk] = "held_by_h200_worker"
                continue
            if has_journal(self.root, chunk):  # a worker wrote between the check and the lock
                handle.close()
                conflicts[chunk] = "h200_journal"
                continue
            self.held[chunk] = handle
        self.conflicts = conflicts

    def release_all(self) -> None:
        for chunk in list(self.held):
            self.held.pop(chunk).close()

    def status(self, wanted: list[int], state: str) -> dict[str, Any]:
        return {"state": state, "pid": os.getpid(), "held": sorted(self.held),
                "conflicts": {f"c{c:03d}": why for c, why in sorted(self.conflicts.items())},
                "listed": wanted,
                "listed_complete": [c for c in wanted if is_complete(self.root, c)],
                "updated_at_utc": _utc()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true", help="one sync, then exit (tests)")
    args = parser.parse_args()
    reserved = args.root / "reserved"
    reserved.mkdir(parents=True, exist_ok=True)
    instance = (reserved / "holder.lock").open("a")
    try:
        fcntl.flock(instance, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("HOLDER_ALREADY_RUNNING", flush=True)
        sys.exit(3)

    def interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    holder, wanted = Holder(args.root), []
    status_file, stop_file = reserved / "holder_status.json", reserved / "holder.stop"
    print(f"{_utc()} HOLDER_START pid={os.getpid()} root={args.root}", flush=True)
    try:
        while not stop_file.exists():
            wanted = holder.wanted(reserved / "a100_chunks.json")
            before = (sorted(holder.held), dict(holder.conflicts))
            holder.sync(wanted)
            if (sorted(holder.held), holder.conflicts) != before:
                print(f"{_utc()} HOLDER_SYNC held={len(holder.held)} "
                      f"conflicts={sorted(holder.conflicts)}", flush=True)
            write_json(status_file, holder.status(wanted, "holding"))
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        holder.release_all()
        write_json(status_file, holder.status(wanted, "stopped"))
        print(f"{_utc()} HOLDER_STOP", flush=True)


if __name__ == "__main__":
    main()
