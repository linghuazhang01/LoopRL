#!/usr/bin/env bash
# Train with a YAML config (RLTT or GRPO). On a Slurm cluster run it inside an allocation.
#   LOOPRL_VENV=<venv> LOOPRL_MODEL_DIR=... LOOPRL_TRAIN_FILE=... LOOPRL_MATH500_JSONL=... \
#   LOOPRL_OUTPUT_DIR=... [LOOPRL_OURO_DIR=...] bash run_train.sh <config.yaml> [dot-list overrides]
set -euo pipefail
CONFIG="${1:?Usage: $0 <config.yaml> [overrides...]}"
shift
: "${LOOPRL_VENV:?set LOOPRL_VENV to the training venv}"
# A reachable Slurm controller means this is a cluster node: require an allocation.
# (Client binaries alone, without a controller, do not count.)
if command -v sinfo >/dev/null 2>&1 && timeout 10 sinfo -h >/dev/null 2>&1 \
    && [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Slurm cluster detected: run inside an allocation (srun/sbatch), not on a login node" >&2
  exit 2
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$HERE/.." && pwd -P)"
SRC="$ROOT/src"; [[ -d "$SRC/rltt_baseline" ]] || SRC="$ROOT"
source "$LOOPRL_VENV/bin/activate"
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONDONTWRITEBYTECODE=1
python3 -u -m rltt_baseline.launch --dry-run --config "$CONFIG" "$@" > /dev/null
exec python3 -u -m rltt_baseline.launch --config "$CONFIG" "$@"
