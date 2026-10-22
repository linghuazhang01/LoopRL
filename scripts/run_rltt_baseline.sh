#!/usr/bin/env bash
# RLTT baseline training. Run on a compute node inside a Slurm allocation with >= 2 GPUs,
# e.g. srun --jobid=<job> --overlap --ntasks=1 bash <scripts>/run_rltt_baseline.sh \
#        <configs>/rltt_baseline.yaml [dot-list overrides...]
# Required environment: RLTT_VENV (venv with the RLTT requirements.txt),
# OURO_THINKING_DIR, RLTT_MODEL_DIR, RLTT_MATH_TRAIN_JSONL, RLTT_MATH500_JSONL, RLTT_OUTPUT_DIR.
set -euo pipefail
USAGE="Usage: $0 <config.yaml> [overrides...]"
CONFIG="${1:?$USAGE}"
shift
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "Run inside a Slurm allocation (compute node)" >&2; exit 2; }
: "${RLTT_VENV:?set RLTT_VENV to the RLTT training venv}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$PROJECT_ROOT"
source "$RLTT_VENV/bin/activate"
SRC_DIR="$PROJECT_ROOT/code/src"
[[ -d "$SRC_DIR/rltt_baseline" ]] || SRC_DIR="$PROJECT_ROOT"  # LoopRL layout
export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python3 -u -m rltt_baseline.launch --dry-run --config "$CONFIG" "$@" > /dev/null
exec python3 -u -m rltt_baseline.launch --config "$CONFIG" "$@"
