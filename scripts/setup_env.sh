#!/usr/bin/env bash
# Create the B200 migration target training venv with CPython 3.12 and CUDA 13.0 wheels.
# Resolve dependencies normally, then run CPU-only import checks.
#   bash setup_env.sh <venv dir>        (UV=/path/to/uv to pick a uv binary)
set -euo pipefail
VENV="${1:?Usage: $0 <venv dir>}"
UV="${UV:-$(command -v uv || true)}"
[[ -x "$UV" ]] || { echo "uv not found; install it (https://docs.astral.sh/uv/) or set UV=" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REQUIREMENTS="$HERE/../configs/b200.requirements.txt"
[[ -f "$REQUIREMENTS" ]] || { echo "requirements file not found: $REQUIREMENTS" >&2; exit 2; }
[[ ! -e "$VENV" ]] || { echo "$VENV exists; refusing to overwrite" >&2; exit 2; }

"$UV" python install 3.12
PY="$("$UV" python find --managed-python 3.12)"
INC="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
[[ -f "$INC/Python.h" ]] || { echo "Python.h missing under $INC" >&2; exit 2; }
"$UV" venv --python "$PY" "$VENV"
"$UV" pip install --index-strategy unsafe-best-match --python "$VENV/bin/python" -r "$REQUIREMENTS"

CUDA_VISIBLE_DEVICES="" "$VENV/bin/python" - <<'PY'
import importlib
for m in ["torch", "vllm", "verl", "transformers", "flash_attn", "ray", "omegaconf", "wandb"]:
    print(f"{m:13s}", getattr(importlib.import_module(m), "__version__", "?"))
import setuptools  # noqa: F401  (torch.utils.cpp_extension)
from vllm.model_executor.models.registry import ModelRegistry
assert "OuroForCausalLM" in ModelRegistry.get_supported_archs(), "vLLM lacks Ouro support"
print("ENV_OK")
PY
