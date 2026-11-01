"""A100 share of deepmath-filter-1p4b-16k-v1, the 12,000 problems of the H200 run.

User request 2026-10-30: take part of the H200 workload on the CityU HPC A100 hold jobs. The
recipe, keys, seeds, answer check and journal format are the H200 run's. The worker is the
portable sampler of the ext20k package (frozen copy
code/scripts/deepmath_filter_a100/portable_sample_75fe831f.py, pinned by SHA256, public vLLM
interfaces only), loaded unchanged and then pointed at the 12,000-problem file, with three changes:

- records and complete.json carry batch "12k-a100" (and this host's identity, as the sampler
  already records it); the H200 read_journal accepts them, so finished chunks can be copied into
  the H200 output as they are;
- a worker claims only the chunks listed for its node in <output>/allowed_chunks.json
  ({"by_node": {"gpu-a100-12": [...], ...}}), in that order, re-read before and after every
  claim. On the H200 server a holder process (code/scripts/hold_deepmath_filter_chunks.py) keeps
  the claim flocks of exactly these chunks, so the H200 workers skip them. A chunk is listed here
  only after the holder holds it, and unlisted here before the holder lets it go;
- ``gpu_index`` is the card's physical index on the node (SLURM_STEP_GPUS / SLURM_JOB_GPUS): the
  two hold jobs on one node both see CUDA device 0, and the sampler names a worker's progress file
  after host and gpu_index. ``cuda_visible_devices`` and ``slurm_job_id`` are recorded as well.

Nodes get disjoint lists, so flocks are never relied on across nodes; the jobs on one node share
their node's list through the chunk flocks on the shared file system.

    PYTHONPATH=code/src python -m budget_inference.ouro_deepmath_filter_a100 --worker \
        --model <dir> --attention-backend TRITON_ATTN    # or --smoke / --plan-only / --status
"""

import hashlib
import importlib.util
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# a frozen copy of the ext20k package's sampler (local/tmp/deepmath_filter_ext20k_package,
# 2026-10-30 01:20), so later edits of code/scripts/deepmath_filter_portable/ do not move it
PORTABLE = Path("code/scripts/deepmath_filter_a100/portable_sample_75fe831f.py")
PORTABLE_SHA256 = "75fe831f2dbd773557f1a98b3b024b22214a0d5889f35f35a4e8820eed8268d7"
DATA = Path("code/data/processed/deepmath_level6_filter12k/sample12000.jsonl")
DATA_SHA256 = "1d0ccfe21d3ef67b5e70bd9a68cae6a0b3827936d017f5aded0fdb9436f543db"
PROBLEMS = 12000
BATCH = "12k-a100"
OUTPUT = Path("code/outputs/ouro_deepmath_filter_1p4b_16k_v1_a100")
ALLOWED = "allowed_chunks.json"


def node_name() -> str:
    return socket.gethostname().split(".", 1)[0]


def allowed_chunks(root: Path, node: str | None = None) -> list[int]:
    """This node's chunks in claim order; an absent file or node means none."""
    path = root / ALLOWED
    if not path.exists():
        return []
    return [int(c) for c in json.loads(path.read_text())["by_node"].get(node or node_name(), [])]


def physical_gpu(environ: Mapping[str, str] = os.environ) -> int:
    for name in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
        value = environ.get(name, "").strip()
        if value.isdecimal():
            return int(value)
    raise RuntimeError("Need one Slurm GPU (SLURM_STEP_GPUS or SLURM_JOB_GPUS)")


def load_sampler(path: Path = PORTABLE, sha256: str = PORTABLE_SHA256) -> Any:
    """The portable sampler module, pointed at the 12,000 problems and the node's chunk list."""
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError("Portable sampler differs from its pinned SHA256")
    spec = importlib.util.spec_from_file_location("deepmath_filter_portable_sample_a100", path)
    sampler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sampler)
    sampler.DATA, sampler.DATA_SHA256, sampler.PROBLEMS = DATA, DATA_SHA256, PROBLEMS
    sampler.BATCH, sampler.OUTPUT = BATCH, OUTPUT
    original_gpu, original_runtime = sampler.gpu_identity, sampler.runtime_identity

    def next_claim(claims: Any, total: int, root: Path = OUTPUT) -> int | None:
        for chunk in allowed_chunks(root):
            if not 0 <= chunk < total or chunk in claims.held or sampler.is_complete(chunk, root):
                continue
            if not claims.try_claim(chunk):
                continue
            # unlisted (or finished) between the read and the claim: let it go
            if sampler.is_complete(chunk, root) or chunk not in allowed_chunks(root):
                claims.release(chunk)
                continue
            return chunk
        return None

    def gpu_identity(memory: float) -> dict[str, Any]:
        gpu = original_gpu(memory)
        return {**gpu, "gpu_index": physical_gpu(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    def runtime_identity() -> dict[str, Any]:
        return {**original_runtime(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}

    sampler.next_claim = next_claim
    sampler.gpu_identity = gpu_identity
    sampler.runtime_identity = runtime_identity
    return sampler


def main(argv: Sequence[str] | None = None) -> None:
    sampler = load_sampler()
    sys.argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    sampler.main()


if __name__ == "__main__":
    main()
