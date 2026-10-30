#!/usr/bin/env bash
# One sampling worker on one GPU; it claims chunks until none is left, then exits 0.
# Stop it with `kill <pid>` (pid in logs/gpu<N>.pid) or `bash stop_all.sh`; running this again
# resumes from the journals. Extra arguments go to sample.py, e.g. --attention-backend FLASH_ATTN.
# Usage: bash run_worker.sh <gpu index> [sample.py options]
set -euo pipefail
GPU="${1:?Usage: $0 <gpu index> [sample.py options]}"
shift
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "GPU index must be a number" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$HERE"
if [[ -z "${PYTHON:-}" && -z "${VIRTUAL_ENV:-}" && -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
fi
PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES="$GPU" VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONHASHSEED=42
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p logs
exec 9>"logs/gpu${GPU}.lock"
flock -n 9 || { echo "A worker for GPU $GPU is already running" >&2; exit 3; }
state() { echo "$(date -Iseconds) $*" | tee -a "logs/gpu${GPU}.state"; }
state "START gpu=$GPU python=$PYTHON"
"$PYTHON" -u sample.py --worker --model "${MODEL_DIR:-$HERE/model_1p4b_thinking_3aaa2224}" \
  "$@" >> "logs/gpu${GPU}.log" 2>&1 &
child=$!
echo "$child" > "logs/gpu${GPU}.pid"
trap 'kill -TERM "$child" 2>/dev/null || true' TERM INT
set +e
wait "$child"
status=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; status=$?; done  # again after a trap
set -e
state "EXIT gpu=$GPU status=$status"
exit "$status"
