#!/usr/bin/env python3
"""Portable sampler for deepmath-filter-1p4b-16k-v1, batch rest45k (2026-10-30).

GRPO data filtering on the RL training file (DeepMath-103K level >= 6, 57,046 problems) beyond
the 12,000 sampled on the H200 server: rest45k holds all 45,046 others, so the two cover the file
once. It merges ext20k (20,000 of them drawn at random, packaged first) with the 25,046 left
after that; ext20k stays here so a run of its earlier package can still be checked. A package
carries one batch's problem file under data/ and runs that batch (--batch picks one if several
are present). The recipe is the H200 run's:
Ouro-1.4B-Thinking (revision 3aaa2224), its default 4 loops, the file's user message through the
Thinking chat template (enable_thinking=True), temperature 1.0, top_p 1.0, at most 16,384
generated tokens, 4 samples per problem, seed = f(protocol, problem id, replicate). The number
correct out of 4 is the filter signal; a length stop counts as wrong, and the final score is
computed offline with math-verify (records keep the text and a provisional string match).

Work split: the problems run in a seeded shuffled order, in chunks of 50 (rest45k 901 with 46
in the last, ext20k 400). One worker process
per GPU claims chunks in order with a non-blocking flock on outputs/chunks/cNNN/claim.lock,
holds a chunk until its 200 trials are committed to its journal, writes complete.json and claims
the next one as soon as its queue runs dry. Stop a worker at any time (SIGTERM or Ctrl-C): its
locks drop with the process, and whoever claims those chunks next skips the committed trials.
GPUs can be added or removed at any time. A front-end admission gate keeps the estimated KV in
use (prompt + generated tokens of the running requests, plus a 2,048-token growth reserve each)
below the engine's KV capacity, so long generations are not preempted over and over.

Needs only vLLM (tested 0.15.0 / torch 2.9.1+cu128 / transformers 4.57.6) and the 9 pinned
model files (fetch_model.py). Commands (from this directory):
    python sample.py --check-model MODEL_DIR       # file set, sizes and SHA256; writes provenance
    python sample.py --plan-only --model MODEL_DIR  # every prompt through the tokenizer, no GPU
    CUDA_VISIBLE_DEVICES=0 python sample.py --smoke --model MODEL_DIR   # 1-minute GPU check
    CUDA_VISIBLE_DEVICES=0 python sample.py --worker --model MODEL_DIR
    python sample.py --status
    python sample.py --verify [--export-passes passes.csv]   # after the run
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
PROTOCOL = "deepmath-filter-1p4b-16k-v1"  # same recipe and seed scheme as the 12,000 on H200
BATCHES = {  # name -> (problem file under data/, its SHA256, problems)
    "ext20k": ("sample20000.jsonl",
               "6317c0effdd818d932b83a2a242a1c8810d00811e574b3875d48d92ef7afe939", 20000),
    "rest45k": ("sample45046.jsonl",
                "d942f80b6c5011c6eaa128958889b3e98503fb9c28ea217668086f65fb71a052", 45046),
}
REPLICATES, CHUNK = 4, 50
OUTPUT = HERE / "outputs"
MODEL = {"dir": "model_1p4b_thinking_3aaa2224", "model_id": "ByteDance/Ouro-1.4B-Thinking",
         "revision": "3aaa2224253a92ca45cf2e3d427c360e1ef9c93d", "num_hidden_layers": 24}
# the 9 files of the H200 / HPC copies (no generation_config.json: vLLM defaults, as on H200)
MODEL_FILES = {
    "config.json": (1471, "44e42ec2ed97f49f20f87985688b6e1be9dd61bbfe9f8a349daac192126e2c55"),
    "configuration_ouro.py": (
        11550, "950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb"),
    "merges.txt": (466391, "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510"),
    "model.safetensors": (
        2869336434, "6730581fd366f6d1689728c61637520228385cdfc17c0b24f19aa840b00565fb"),
    "modeling_ouro.py": (35500, "cb8c980b016ae8ae35b6c1193633b9b4496e89993e3a0e8c7853324f01904b30"),
    "special_tokens_map.json": (
        965, "688ac94df221b18a4d2d4579dfa3c4f0fd5ca3e394da46cd0a317fb5d57d4685"),
    "tokenizer.json": (3522644, "fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d"),
    "tokenizer_config.json": (
        4392, "e41eafc6bf7269d6f90f2531124949073a57965f5f0765238a744c1c528e2448"),
    "vocab.json": (800656, "7b9de3f47796abf8d00ab96be299fea0dc9afdf1827f34e7e0b9fb44593efe5c"),
}
DEPTH = 4
TEMPERATURE, TOP_P = 1.0, 1.0  # the GRPO rollouts: no nucleus truncation
GENERATION_CAP, MAX_PROMPT_TOKENS = 16384, 2048
CONTEXT_LIMIT = MAX_PROMPT_TOKENS + GENERATION_CAP  # the GRPO vLLM max_model_len (18,432)
THINK_END = 4  # </think>
RESERVE_TOKENS = 2048  # growth reserve per running request (the H200 gate's value)
ENGINE_MAX_SEQS = 64  # engine ceiling; the front-end gate decides how many actually run
BATCHED_TOKENS = 4096


def packaged_batches() -> list[str]:
    return [name for name, (file, _, _) in BATCHES.items() if (HERE / "data" / file).exists()]


def select_batch(name: str) -> None:
    """Point BATCH, DATA, DATA_SHA256 and PROBLEMS at one batch."""
    global BATCH, DATA, DATA_SHA256, PROBLEMS
    file, DATA_SHA256, PROBLEMS = BATCHES[name]
    BATCH, DATA = name, HERE / "data" / file


# the batch this package carries (ext20k when there is none or several; main() then checks)
select_batch(packaged_batches()[0] if len(packaged_batches()) == 1 else "ext20k")


# ----------------------------------------------------------------------------- hashing
def canonical_hash(value: Any) -> str:
    """Identical to budget_inference.ouro_sampling_protocol.canonical_hash."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record_sha(record: Mapping[str, Any]) -> str:
    return canonical_hash({k: v for k, v in record.items() if k != "record_sha256"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True))
    os.replace(temporary, path)


def append_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


# ----------------------------------------------------------------------------- data and plan
def load_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """The frozen problems in their seeded shuffled order (``order`` 0..PROBLEMS-1)."""
    payload = (path or DATA).read_bytes()
    if hashlib.sha256(payload).hexdigest() != DATA_SHA256:
        raise ValueError("Problem file differs from its frozen SHA256")
    rows = [json.loads(line) for line in payload.decode("utf-8").split("\n") if line]
    if len(rows) != PROBLEMS or len({r["id"] for r in rows}) != PROBLEMS:
        raise ValueError(f"Problem file must hold {PROBLEMS} unique problems")
    rows.sort(key=lambda r: canonical_hash([PROTOCOL, "order", r["id"]]))
    return [{**r, "order": i} for i, r in enumerate(rows)]


def chunk_rows(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    return [list(rows[i:i + CHUNK]) for i in range(0, len(rows), CHUNK)]


def trial_key(question_id: str, replicate: int) -> str:
    return f"{PROTOCOL}|{question_id}|{replicate}"


def trial_seed(question_id: str, replicate: int) -> int:
    return int(canonical_hash([PROTOCOL, question_id, replicate])[:16], 16) & ((1 << 63) - 1)


def chunk_plan(rows: Sequence[Mapping[str, Any]], chunk: int) -> dict[str, dict[str, Any]]:
    """The chunk's trials in queue order: the four replicates of a problem are adjacent."""
    return {trial_key(r["id"], k): {
        "key": trial_key(r["id"], k), "question_id": r["id"], "row": r["row"],
        "order": r["order"], "chunk": chunk, "replicate": k, "seed": trial_seed(r["id"], k),
        "gold_answer": r["gold_answer"]} for r in rows for k in range(REPLICATES)}


def chunk_dir(chunk: int, root: Path = OUTPUT) -> Path:
    return root / "chunks" / f"c{chunk:03d}"


def read_journal(path: Path, planned: Mapping[str, Mapping[str, Any]],
                 partial_tail_ok: bool = False) -> dict[str, dict[str, Any]]:
    """Committed records of one chunk journal; every line must be a valid planned record.

    ``partial_tail_ok`` is for reading a journal another worker may be appending to: a last line
    without its newline is an unfinished write and is ignored.
    """
    if path.is_symlink():
        raise ValueError("Journal cannot be a symlink")
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    payload = path.read_bytes()
    if partial_tail_ok and not payload.endswith(b"\n"):
        payload = payload[:payload.rfind(b"\n") + 1]
    # split on "\n" only: generated text may hold U+2028, U+0085 and the like, which
    # json.dumps(ensure_ascii=False) leaves raw and str.splitlines() would treat as line breaks
    for line in filter(None, payload.decode("utf-8").split("\n")):
        record = json.loads(line)
        key = record.get("key")
        if (key not in planned or key in records
                or record.get("record_sha256") != record_sha(record)
                or record.get("seed") != planned[key]["seed"]
                or record.get("data_sha256") != DATA_SHA256):
            raise ValueError(f"Journal line is corrupt, foreign or duplicated: {key}")
        records[key] = record
    return records


def is_complete(chunk: int, root: Path = OUTPUT) -> bool:
    """A marker of another batch (a wrong --output) does not count; its journal then fails."""
    marker = chunk_dir(chunk, root) / "complete.json"
    if not marker.exists():
        return False
    entry = json.loads(marker.read_text())
    return entry["chunk"] == chunk and entry.get("batch") == BATCH


def mark_complete(chunk: int, planned: Mapping[str, Mapping[str, Any]],
                  root: Path = OUTPUT) -> dict[str, Any]:
    """Check the journal holds every planned trial and write complete.json with the counts."""
    journal = chunk_dir(chunk, root) / "journal.jsonl"
    records = read_journal(journal, planned)
    if set(records) != set(planned):
        raise RuntimeError(f"Chunk {chunk} is not fully committed")
    passes: dict[str, int] = {}
    for record in records.values():
        passes[record["question_id"]] = (passes.get(record["question_id"], 0)
                                         + bool(record["provisional_correct"]))
    marker = {"protocol": PROTOCOL, "batch": BATCH, "chunk": chunk, "trials": len(records),
              "journal_sha256": sha256_file(journal),
              "provisional_correct": sum(passes.values()),
              "length_stops": sum(r["finish_reason"] == "length" for r in records.values()),
              "generated_tokens": sum(r["generated_tokens"] for r in records.values()),
              "provisional_passes": dict(sorted(passes.items())), "completed_at_utc": _utc()}
    write_json(chunk_dir(chunk, root) / "complete.json", marker)
    return marker


# ----------------------------------------------------------------------------- answers
def boxed_content(text: str) -> tuple[str | None, str]:
    """The last \\boxed{...} with balanced braces; None with a status otherwise."""
    commands = list(re.finditer(r"\\boxed\b", text))
    if not commands:
        return None, "missing_boxed"
    position = commands[-1].end()
    while position < len(text) and text[position].isspace():
        position += 1
    if position == len(text) or text[position] != "{":
        return None, "unclosed_boxed"
    start = position = position + 1
    depth = 1
    while position < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[position], 0)
        position += 1
    if depth:
        return None, "unclosed_boxed"
    return text[start:position - 1], "found"


def normalize_answer(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\\text\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\(d|t)frac", r"\\frac", value)
    for token in ("\\left", "\\right", "\\,", "\\!", "\\;", "\\ ", "$", " ", "\n", "^\\circ",
                  "^{\\circ}", "\\%", "{,}"):
        value = value.replace(token, "")
    return value.rstrip(".")


def as_number(value: str) -> Decimal | None:
    value = normalize_answer(value).replace(",", "")
    if not re.fullmatch(r"[+-]?[0-9]+(?:\.[0-9]+)?", value):
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def provisional_match(prediction: str | None, gold: str) -> bool:
    """A conservative string/number match; math-verify gives the final verdict offline."""
    if prediction is None:
        return False
    predicted, expected = as_number(prediction), as_number(gold)
    if predicted is not None and expected is not None:
        return predicted == expected
    return normalize_answer(prediction) == normalize_answer(gold)


# ----------------------------------------------------------------------------- model
def check_model(directory: Path, full: bool = True) -> dict[str, Any]:
    """Exactly the 9 pinned files with their sizes (and SHA256 when ``full``).

    A full check writes provenance.json; a quick check (each worker start) needs it. Extra files
    such as generation_config.json would change vLLM's sampling defaults and are refused.
    """
    names = {p.name for p in directory.iterdir()
             if p.is_file() and p.name != "provenance.json" and not p.name.startswith(".")}
    if names != set(MODEL_FILES):
        raise ValueError(f"Model directory must hold exactly {sorted(MODEL_FILES)}; "
                         f"extra {sorted(names - set(MODEL_FILES))}, "
                         f"missing {sorted(set(MODEL_FILES) - names)}")
    for name, (size, _) in MODEL_FILES.items():
        if (directory / name).stat().st_size != size:
            raise ValueError(f"{name} has the wrong size")
    provenance_path = directory / "provenance.json"
    expected = {"model_id": MODEL["model_id"], "revision": MODEL["revision"],
                "files_sha256": {n: s for n, (_, s) in MODEL_FILES.items()}}
    if not full:
        if (not provenance_path.exists()
                or json.loads(provenance_path.read_text()) != expected):
            raise ValueError("Run `python sample.py --check-model MODEL_DIR` first")
        return expected
    for name, (_, digest) in MODEL_FILES.items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"{name} SHA256 differs from the pinned file")
    write_json(provenance_path, expected)
    return expected


def build_prompts(tokenizer: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    """The file's user message through the Thinking chat template (as the GRPO rollouts)."""
    prompts = {}
    for row in rows:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["question"]}], tokenize=True,
            add_generation_prompt=True, enable_thinking=True)
        if isinstance(encoded, dict) or hasattr(encoded, "keys"):
            encoded = encoded["input_ids"]
        ids = list(encoded.tolist() if hasattr(encoded, "tolist") else encoded)
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if not ids or len(ids) > MAX_PROMPT_TOKENS:
            raise ValueError(f"Prompt is empty or exceeds {MAX_PROMPT_TOKENS}: {row['id']}")
        prompts[row["id"]] = ids
    return prompts


# ----------------------------------------------------------------------------- chunk claims
class ChunkClaims:
    """Non-blocking flocks on chunks/cNNN/claim.lock, held until the chunk is complete."""

    def __init__(self, root: Path = OUTPUT) -> None:
        self.root = root
        self.held: dict[int, Any] = {}

    def try_claim(self, chunk: int) -> bool:
        import fcntl

        path = chunk_dir(chunk, self.root) / "claim.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        self.held[chunk] = handle
        return True

    def release(self, chunk: int) -> None:
        self.held.pop(chunk).close()  # closing the descriptor drops the flock


def next_claim(claims: ChunkClaims, total: int, root: Path = OUTPUT) -> int | None:
    """The first chunk that is neither complete nor held (by this or another worker)."""
    for chunk in range(total):
        if chunk in claims.held or is_complete(chunk, root):
            continue
        if not claims.try_claim(chunk):
            continue
        if is_complete(chunk, root):  # finished between the check and the claim
            claims.release(chunk)
            continue
        return chunk
    return None


# ----------------------------------------------------------------------------- admission
class KVGate:
    """Front-end admission: a new request enters only if the estimated free KV holds its prompt
    plus a growth reserve for every running request and itself (the H200 run's whole-context
    gate, estimated from the tokens each running request has so far)."""

    def __init__(self, capacity: int | None, ceiling: int, reserve: int = RESERVE_TOKENS) -> None:
        self.capacity, self.ceiling, self.reserve = capacity, ceiling, reserve
        self.closed = 0

    def admits(self, used_tokens: int, active: int, prompt_tokens: int) -> bool:
        if active == 0:
            return True
        if active >= self.ceiling:
            return False
        if self.capacity is not None and (
                self.capacity - used_tokens < prompt_tokens + self.reserve * (active + 1)):
            self.closed += 1
            return False
        return True


def default_ceiling(capacity: int | None) -> int:
    """About one running request per 3,500 KV tokens (H200 at 90%: 48), within 8..64."""
    return 32 if capacity is None else max(8, min(ENGINE_MAX_SEQS, capacity // 3500))


# ----------------------------------------------------------------------------- engine
def engine_kwargs(fields: set[str], directory: Path, memory: float,
                  backend: str | None) -> dict[str, Any]:
    """vLLM engine settings of the H200 run, passing only what this vLLM version accepts."""
    kwargs: dict[str, Any] = {
        "model": str(directory), "trust_remote_code": True,
        "hf_overrides": {"total_ut_steps": DEPTH}, "dtype": "bfloat16",
        "kv_cache_dtype": "auto", "tensor_parallel_size": 1, "max_model_len": CONTEXT_LIMIT,
        "max_num_seqs": ENGINE_MAX_SEQS, "gpu_memory_utilization": memory,
        "enable_prefix_caching": False, "enable_chunked_prefill": True,
        "max_num_batched_tokens": BATCHED_TOKENS, "seed": 0}
    optional = {"model_impl": "vllm", "async_scheduling": False}
    if backend:
        optional["attention_backend"] = backend
    for name, value in optional.items():
        if name in fields:
            kwargs[name] = value
        elif name == "attention_backend":  # older vLLM reads it from the environment
            os.environ["VLLM_ATTENTION_BACKEND"] = value
    return kwargs


def create_engine(directory: Path, memory: float, backend: str | None) -> Any:
    from vllm import LLM
    from vllm.engine.arg_utils import EngineArgs

    fields = {f.name for f in dataclasses.fields(EngineArgs)}
    llm = LLM(**engine_kwargs(fields, directory, memory, backend))
    hf_config = llm.llm_engine.vllm_config.model_config.hf_config
    if getattr(hf_config, "total_ut_steps", None) != DEPTH:
        raise RuntimeError("Engine did not apply the 4 Ouro loops")
    if getattr(hf_config, "num_hidden_layers", None) != MODEL["num_hidden_layers"]:
        raise RuntimeError("Engine did not load the 24-layer Ouro-1.4B model")
    return llm


def kv_capacity(llm: Any) -> int | None:
    cache = llm.llm_engine.vllm_config.cache_config
    blocks, size = getattr(cache, "num_gpu_blocks", None), getattr(cache, "block_size", None)
    return blocks * size if blocks and size else None


def runtime_identity() -> dict[str, Any]:
    versions = {}
    for module in ("vllm", "torch", "transformers"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001 - only recorded
            versions[module] = None
    return {"host": socket.gethostname().split(".", 1)[0], "versions": versions}


def gpu_identity(memory: float) -> dict[str, Any]:
    """The single visible GPU; refuses one without room for the requested memory fraction."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"[0-9]+", visible):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one GPU index")
    query = subprocess.run(
        ["nvidia-smi", "-i", visible, "--query-gpu=uuid,name,memory.total,memory.used",
         "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout
    gpu_uuid, name, total, used = [part.strip() for part in query.strip().split(",")]
    total_mib, used_mib = int(total), int(used)
    if total_mib - used_mib < memory * total_mib + 1024:
        raise RuntimeError(f"GPU {visible} has {total_mib - used_mib} MiB free, not enough for "
                           f"--gpu-memory-utilization {memory}; wait or lower it")
    return {"gpu_index": int(visible), "gpu_uuid": gpu_uuid, "gpu_name": name,
            "gpu_memory_mib": total_mib}


# ----------------------------------------------------------------------------- worker
def run_worker(model_dir: Path, root: Path = OUTPUT, memory: float = 0.9,
               concurrency: int | None = None, backend: str | None = None,
               engine_factory: Callable[[Path, float, str | None], Any] = create_engine,
               gpu: Mapping[str, Any] | None = None) -> dict[str, Any]:
    import fcntl

    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    rows = load_rows()
    parts = chunk_rows(rows)
    gpu = dict(gpu or gpu_identity(memory))
    lock_path = root / "locks" / f"gpu_{gpu['gpu_uuid']}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_lock = lock_path.open("a")
    try:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)  # one worker per card
    except BlockingIOError:
        raise RuntimeError(f"Another worker already runs on GPU {gpu['gpu_uuid']}") from None
    claims = ChunkClaims(root)
    probe = next_claim(claims, len(parts), root)
    if probe is None:  # every chunk is complete or held by another worker
        gpu_lock.close()
        print("NOTHING_TO_CLAIM", flush=True)
        return {"session_id": None, "trials": 0, "chunks": []}
    claims.release(probe)
    check_model(model_dir, full=False)
    llm = engine_factory(model_dir, memory, backend)
    tokenizer = llm.get_tokenizer()
    prompts = build_prompts(tokenizer, rows)
    engine = llm.llm_engine
    capacity = kv_capacity(llm)
    gate = KVGate(capacity, concurrency or default_ceiling(capacity))
    session, started = uuid.uuid4().hex, time.perf_counter()
    identity = {**gpu, **runtime_identity(), "gpu_memory_utilization": memory,
                "attention_backend": backend or "vllm-default", "session_id": session}
    held: dict[int, dict[str, Any]] = {}
    queue: list[dict[str, Any]] = []
    active: dict[str, dict[str, Any]] = {}
    done = {"trials": 0, "chunks": [], "provisional_correct": 0, "length_stops": 0,
            "generated_tokens": 0}
    progress_file = root / "workers" / f"{identity['host']}_gpu{gpu['gpu_index']}.json"
    print(f"WORKER_START session={session} kv_capacity={capacity} ceiling={gate.ceiling}",
          flush=True)

    def progress(state: str) -> None:
        write_json(progress_file, {
            "protocol": PROTOCOL, "batch": BATCH, **identity, "state": state, "pid": os.getpid(),
            "held_chunks": sorted(held), "queued": len(queue), "active": len(active),
            "kv_capacity_tokens": capacity, "ceiling": gate.ceiling,
            "estimated_kv_tokens": sum(a["prompt"] + len(a["ids"]) for a in active.values()),
            "gate_closed": gate.closed, "session": done,
            "elapsed_seconds": time.perf_counter() - started, "updated_at_utc": _utc()})

    def finish(chunk: int) -> None:
        marker = mark_complete(chunk, held.pop(chunk)["planned"], root)
        claims.release(chunk)
        done["chunks"].append(chunk)
        print(f"CHUNK_COMPLETE c{chunk:03d} trials={marker['trials']} "
              f"provisional_correct={marker['provisional_correct']}", flush=True)

    def claim() -> bool:
        while (chunk := next_claim(claims, len(parts), root)) is not None:
            planned = chunk_plan(parts[chunk], chunk)
            committed = read_journal(chunk_dir(chunk, root) / "journal.jsonl", planned)
            held[chunk] = {"planned": planned, "committed": committed}
            todo = [p for p in planned.values() if p["key"] not in committed]
            if not todo:
                finish(chunk)
                continue
            queue.extend(todo)
            print(f"CLAIM c{chunk:03d} resume={len(committed)} todo={len(todo)}", flush=True)
            return True
        return False

    def admit() -> None:
        while len(active) < gate.ceiling:
            if not queue and not claim():
                return
            plan = queue[0]
            prompt = prompts[plan["question_id"]]
            used = sum(a["prompt"] + len(a["ids"]) for a in active.values())
            if not gate.admits(used, len(active), len(prompt)):
                return
            queue.pop(0)
            engine.add_request(plan["key"], {"prompt_token_ids": list(prompt)},
                               SamplingParams(temperature=TEMPERATURE, top_p=TOP_P,
                                              max_tokens=GENERATION_CAP, seed=plan["seed"], n=1,
                                              ignore_eos=False, detokenize=False,
                                              output_kind=RequestOutputKind.DELTA))
            active[plan["key"]] = {"plan": plan, "prompt": len(prompt), "ids": [],
                                   "admitted": time.perf_counter()}

    def commit(entry: dict[str, Any], reason: Any) -> None:
        plan, ids = entry["plan"], entry["ids"]
        final = "length" if str(reason) == "length" else "stop"
        text = tokenizer.decode(ids, skip_special_tokens=True)
        boxed, status = boxed_content(text)
        think_end = ids.index(THINK_END) if THINK_END in ids else None
        prompt_ids = prompts[plan["question_id"]]
        record = {
            "protocol": PROTOCOL, "batch": BATCH, **plan, "data_sha256": DATA_SHA256,
            "model_id": MODEL["model_id"], "model_revision": MODEL["revision"], "depth": DEPTH,
            "prompt_tokens": len(prompt_ids), "prompt_token_ids_sha256": canonical_hash(prompt_ids),
            "generated_tokens": len(ids), "thinking_tokens": think_end,
            "thinking_closed": think_end is not None, "finish_reason": final,
            "boxed_answer": boxed, "boxed_status": status,
            "provisional_correct": bool(
                final == "stop" and provisional_match(boxed, plan["gold_answer"])),
            "temperature": TEMPERATURE, "top_p": TOP_P, "generation_cap": GENERATION_CAP,
            "admission_policy": "frontend_kv_estimate_gate", "ceiling": gate.ceiling,
            **identity, "latency_seconds": time.perf_counter() - entry["admitted"],
            "committed_at_utc": _utc(), "generated_text": text, "token_ids": ids}
        record["record_sha256"] = record_sha(record)
        chunk = plan["chunk"]
        append_record(chunk_dir(chunk, root) / "journal.jsonl", record)
        held[chunk]["committed"][plan["key"]] = record
        done["trials"] += 1
        done["provisional_correct"] += record["provisional_correct"]
        done["length_stops"] += final == "length"
        done["generated_tokens"] += len(ids)
        if len(held[chunk]["committed"]) == len(held[chunk]["planned"]):
            finish(chunk)

    last = 0.0
    try:
        admit()
        progress("running")
        while active:
            if not engine.has_unfinished_requests():
                raise RuntimeError("Engine drained while requests were active")
            for output in engine.step():
                entry = active.get(output.request_id)
                if entry is None:
                    continue
                entry["ids"].extend(int(t) for t in output.outputs[0].token_ids)
                if output.finished:
                    del active[output.request_id]
                    commit(entry, output.outputs[0].finish_reason)
            admit()
            now = time.perf_counter()
            if now - last > 30:
                progress("running")
                last = now
    except KeyboardInterrupt:
        progress("stopped")
        raise
    except BaseException:
        progress("failed")
        raise
    finally:  # a process exit drops these anyway; an in-process stop must too
        for chunk in list(claims.held):
            claims.release(chunk)
        gpu_lock.close()
    progress("completed")
    return {"session_id": session, **done, "elapsed_seconds": time.perf_counter() - started}


# ----------------------------------------------------------------------------- smoke
def smoke(model_dir: Path, memory: float, backend: str | None,
          engine_factory: Callable[[Path, float, str | None], Any] = create_engine,
          prompts_n: int = 4, tokens: int = 64) -> dict[str, Any]:
    """GPU self-check before a run (writes nothing under outputs/): the engine loads with 4
    loops, the KV capacity is readable for the gate, and greedy tokens accumulated from the
    streamed (DELTA) step loop equal those of llm.generate on the same prompts."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    check_model(model_dir, full=False)
    rows = load_rows()[:prompts_n]
    llm = engine_factory(model_dir, memory, backend)
    prompts = build_prompts(llm.get_tokenizer(), rows)

    def params(kind: Any) -> Any:
        return SamplingParams(temperature=0.0, max_tokens=tokens, ignore_eos=True,
                              detokenize=False, output_kind=kind)

    engine = llm.llm_engine
    streamed: dict[str, list[int]] = {r["id"]: [] for r in rows}
    for row in rows:
        engine.add_request(f"smoke-{row['id']}", {"prompt_token_ids": prompts[row["id"]]},
                           params(RequestOutputKind.DELTA))
    while engine.has_unfinished_requests():
        for output in engine.step():
            streamed[output.request_id.removeprefix("smoke-")].extend(
                int(t) for t in output.outputs[0].token_ids)
    final = llm.generate([{"prompt_token_ids": prompts[r["id"]]} for r in rows],
                         params(RequestOutputKind.FINAL_ONLY))
    reference = {r["id"]: [int(t) for t in o.outputs[0].token_ids] for r, o in zip(rows, final)}
    capacity = kv_capacity(llm)
    result = {"kv_capacity_tokens": capacity, "default_ceiling": default_ceiling(capacity),
              "streamed_tokens": {k: len(v) for k, v in streamed.items()},
              "delta_matches_generate": streamed == reference, **runtime_identity()}
    if not result["delta_matches_generate"] or any(len(v) != tokens for v in streamed.values()):
        raise RuntimeError(f"Streamed tokens differ from llm.generate: {result}")
    if capacity is None:
        print("WARNING: KV capacity unreadable; only --concurrency limits the running requests",
              flush=True)
    return result


# ----------------------------------------------------------------------------- plan, status
def plan_only(model_dir: Path, root: Path = OUTPUT) -> dict[str, Any]:
    """CPU preflight: data, model files and every prompt through the real tokenizer."""
    from transformers import AutoTokenizer

    rows = load_rows()
    check_model(model_dir, full=False)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True,
                                              trust_remote_code=True)
    sizes = [len(ids) for ids in build_prompts(tokenizer, rows).values()]
    return {"protocol": PROTOCOL, "batch": BATCH, "problems": len(rows),
            "chunks": len(chunk_rows(rows)), "trials": len(rows) * REPLICATES,
            "prompt_tokens_min": min(sizes), "prompt_tokens_max": max(sizes),
            "prompt_tokens_mean": sum(sizes) / len(sizes),
            "committed": status(root)["committed"], "cuda_engine_allocated": False}


def status(root: Path = OUTPUT) -> dict[str, Any]:
    """Read-only progress: complete chunks from their markers, others from their journals."""
    parts = chunk_rows(load_rows())
    committed = correct = stops = tokens = complete = 0
    partial: dict[str, int] = {}
    for chunk, rows in enumerate(parts):
        marker = chunk_dir(chunk, root) / "complete.json"
        if marker.exists():
            entry = json.loads(marker.read_text())
            complete += 1
            committed += entry["trials"]
            correct += entry["provisional_correct"]
            stops += entry["length_stops"]
            tokens += entry["generated_tokens"]
            continue
        records = read_journal(chunk_dir(chunk, root) / "journal.jsonl", chunk_plan(rows, chunk),
                               partial_tail_ok=True)
        if records:
            partial[f"c{chunk:03d}"] = len(records)
        committed += len(records)
        correct += sum(r["provisional_correct"] for r in records.values())
        stops += sum(r["finish_reason"] == "length" for r in records.values())
        tokens += sum(r["generated_tokens"] for r in records.values())
    workers = {}
    for path in sorted((root / "workers").glob("*.json")):
        w = json.loads(path.read_text())
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(w["updated_at_utc"])).total_seconds()
        session = w.get("session", {})
        workers[path.stem] = {
            "state": w.get("state"), "pid": w.get("pid"), "held_chunks": w.get("held_chunks"),
            "active": w.get("active"), "ceiling": w.get("ceiling"),
            "kv_capacity_tokens": w.get("kv_capacity_tokens"),
            "estimated_kv_tokens": w.get("estimated_kv_tokens"),
            "session_trials": session.get("trials"),
            "session_tokens_per_second": round(
                session.get("generated_tokens", 0) / max(w.get("elapsed_seconds", 0), 1), 1),
            "age_seconds": round(age)}
    return {"protocol": PROTOCOL, "batch": BATCH, "chunks": len(parts), "chunks_complete": complete,
            "partial_chunks": partial, "trials": PROBLEMS * REPLICATES, "committed": committed,
            "provisional_correct": correct, "length_stops": stops, "generated_tokens": tokens,
            "mean_generated_tokens": tokens / committed if committed else None,
            "workers": workers, "checked_at_utc": _utc()}


def verify(root: Path = OUTPUT, passes_csv: Path | None = None) -> dict[str, Any]:
    """After the run: every chunk complete, every trial exactly once, markers match journals."""
    parts = chunk_rows(load_rows())
    per_problem: dict[str, dict[str, Any]] = {}
    total = 0
    for chunk, rows in enumerate(parts):
        planned = chunk_plan(rows, chunk)
        journal = chunk_dir(chunk, root) / "journal.jsonl"
        records = read_journal(journal, planned)
        if set(records) != set(planned) or not is_complete(chunk, root):
            raise RuntimeError(f"Chunk {chunk} is incomplete: {len(records)}/{len(planned)}")
        marker = json.loads((chunk_dir(chunk, root) / "complete.json").read_text())
        if marker["journal_sha256"] != sha256_file(journal):
            raise RuntimeError(f"Chunk {chunk} journal changed after completion")
        total += len(records)
        for record in records.values():
            entry = per_problem.setdefault(record["question_id"], {
                "question_id": record["question_id"], "row": record["row"],
                "provisional_passes": 0, "length_stops": 0, "tokens": 0})
            entry["provisional_passes"] += bool(record["provisional_correct"])
            entry["length_stops"] += record["finish_reason"] == "length"
            entry["tokens"] += record["generated_tokens"]
    if total != PROBLEMS * REPLICATES or len(per_problem) != PROBLEMS:
        raise RuntimeError("Trial or problem count differs from the plan")
    histogram = {k: sum(e["provisional_passes"] == k for e in per_problem.values())
                 for k in range(REPLICATES + 1)}
    if passes_csv is not None:
        with passes_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "question_id", "row", "provisional_passes", "length_stops", "mean_tokens"])
            writer.writeheader()
            for e in sorted(per_problem.values(), key=lambda e: e["row"]):
                writer.writerow({"question_id": e["question_id"], "row": e["row"],
                                 "provisional_passes": e["provisional_passes"],
                                 "length_stops": e["length_stops"],
                                 "mean_tokens": round(e["tokens"] / REPLICATES, 1)})
    return {"protocol": PROTOCOL, "batch": BATCH, "trials": total, "problems": len(per_problem),
            "provisional_passes_histogram": histogram, "verified_at_utc": _utc()}


# ----------------------------------------------------------------------------- CLI
def _stop_gracefully() -> None:
    """SIGTERM/SIGINT -> KeyboardInterrupt, so a stop marks progress and exits with 130."""
    def interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)


def main() -> None:
    _stop_gracefully()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--worker", action="store_true", help="run on the visible GPU")
    action.add_argument("--smoke", action="store_true", help="1-minute GPU self-check")
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--check-model", type=Path, metavar="MODEL_DIR")
    action.add_argument("--status", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--batch", choices=sorted(BATCHES),
                        help="default: the one batch whose problem file is under data/")
    parser.add_argument("--model", type=Path, default=HERE / MODEL["dir"])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--concurrency", type=int, help="running-request ceiling (default: "
                        "about one per 3,500 KV tokens, 8..64)")
    parser.add_argument("--attention-backend", help="e.g. FLASH_ATTN (H200 run) or TRITON_ATTN; "
                        "default: vLLM's choice")
    parser.add_argument("--export-passes", type=Path, help="with --verify: per-problem CSV")
    args = parser.parse_args()
    if args.batch:
        select_batch(args.batch)
    elif not args.check_model and len(packaged_batches()) != 1:
        parser.error(f"data/ holds problem files of {packaged_batches() or 'no batch'}; "
                     "pass --batch")
    if args.check_model:
        result = check_model(args.check_model, full=True)
    elif args.plan_only:
        result = plan_only(args.model, args.output)
    elif args.status:
        result = status(args.output)
    elif args.verify:
        result = verify(args.output, args.export_passes)
    elif args.smoke:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # KV capacity in-process
        result = smoke(args.model, args.gpu_memory_utilization, args.attention_backend)
    else:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        try:
            result = run_worker(args.model, args.output, args.gpu_memory_utilization,
                                args.concurrency, args.attention_backend)
        except KeyboardInterrupt:
            print("WORKER_STOPPED", flush=True)
            sys.exit(130)
    print(json.dumps(result, indent=1, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
