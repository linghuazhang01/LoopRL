#!/usr/bin/env bash
# Create the RLTT / GRPO training venv: uv-managed CPython 3.12 (ships Python.h, which Triton
# needs) + the pinned lock installed with --no-deps, then CPU-only import checks.
#   bash setup_env.sh <venv dir>        (UV=/path/to/uv to pick a uv binary)
set -euo pipefail
VENV="${1:?Usage: $0 <venv dir>}"
UV="${UV:-$(command -v uv || true)}"
[[ -x "$UV" ]] || { echo "uv not found; install it (https://docs.astral.sh/uv/) or set UV=" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
LOCK="$HERE/../configs/rltt_env.lock.txt"
[[ -f "$LOCK" ]] || { echo "lock file not found: $LOCK" >&2; exit 2; }
[[ ! -e "$VENV" ]] || { echo "$VENV exists; refusing to overwrite" >&2; exit 2; }

"$UV" python install 3.12
PY="$("$UV" python find --managed-python 3.12)"
INC="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
[[ -f "$INC/Python.h" ]] || { echo "Python.h missing under $INC" >&2; exit 2; }
"$UV" venv --python "$PY" "$VENV"
"$UV" pip install --no-deps --python "$VENV/bin/python" -r "$LOCK"

CUDA_VISIBLE_DEVICES="" "$VENV/bin/python" - <<'PY'
import importlib
for m in ["torch", "vllm", "verl", "transformers", "flash_attn", "ray", "omegaconf", "wandb"]:
    print(f"{m:13s}", getattr(importlib.import_module(m), "__version__", "?"))
import flash_attn.bert_padding  # noqa: F401  (imported by the RLTT actor)
import setuptools  # noqa: F401  (torch.utils.cpp_extension)
from vllm.model_executor.models.registry import ModelRegistry
assert "OuroForCausalLM" in ModelRegistry.get_supported_archs(), "vLLM lacks Ouro support"
print("ENV_OK")
PY
