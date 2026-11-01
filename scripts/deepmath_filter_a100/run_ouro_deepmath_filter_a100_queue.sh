#!/usr/bin/env bash
# deepmath-filter-1p4b-16k-v1, A100 share of the 12,000 problems (user request 2026-10-30): run
# inside one hold job's allocation (srun --jobid=<job> --overlap --ntasks=1 --gres=gpu:a100:1).
# The card is shared with the user's other projects, so this queue never stops anything: it waits
# until its GPU has had no compute process and < 1 GiB in use for 30 minutes in a row (the H200
# run's rule), runs a one-off GPU smoke check per job, then a worker
# (budget_inference.ouro_deepmath_filter_a100) that claims only this node's listed chunks. When
# the worker ends normally it waits for idle again and runs another worker if listed chunks are
# still open (the list can grow). A stop (exit 130/143) or a failure ends the queue.
# Stop everything: touch logs/deepmath_filter_a100.stop, then kill the worker (pid file below).
set -euo pipefail
USAGE="Usage: $0 <slurm-job-id>"
JOB="${1:?$USAGE}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"
export PYTHONPATH="$PROJECT_ROOT/code/src"
export HF_HOME="$PROJECT_ROOT/cache/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=42 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENABLE_V1_MULTIPROCESSING=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
OUT="code/outputs/ouro_deepmath_filter_1p4b_16k_v1_a100"
MODEL="$OUT/model_1p4b_thinking_3aaa2224"
BACKEND="TRITON_ATTN"
IDLE_SECONDS=1800
STOP="logs/deepmath_filter_a100.stop"
LOG="logs/deepmath_filter_a100_job${JOB}"
[[ "${SLURM_JOB_ID:-}" == "$JOB" ]] || { echo "Not inside Slurm job $JOB" >&2; exit 2; }
[[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+$ ]] || { echo "Need one allocated CUDA device" >&2; exit 2; }
mkdir -p logs
exec 9>"${LOG}_queue.lock"
flock -n 9 || { echo "A filter queue for job $JOB already runs" >&2; exit 3; }
state() { echo "$(date -Iseconds) $*" | tee -a "${LOG}_state.log"; }
trap 'state "QUEUE_FAILED status=$? line=$LINENO"' ERR
NODE="$(hostname -s)"

open_chunks() {  # listed for this node and not complete
  python3 - "$OUT" "$NODE" <<'PY'
import json, sys
from pathlib import Path
root, node = Path(sys.argv[1]), sys.argv[2]
path = root / "allowed_chunks.json"
listed = json.loads(path.read_text())["by_node"].get(node, []) if path.exists() else []
print(sum(not (root / "chunks" / f"c{c:03d}" / "complete.json").exists() for c in listed))
PY
}

gpu_idle() {
  USED="$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  APPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)"
  [[ "$USED" =~ ^[0-9]+$ ]] && (( USED < 1024 && APPS == 0 ))
}

wait_idle() {  # 0 once idle for IDLE_SECONDS in a row; 1 if the stop file appears
  local since="" i=0
  while true; do
    [[ -e "$STOP" ]] && { state "STOP_FILE"; return 1; }
    if gpu_idle; then
      since="${since:-$(date +%s)}"
      (( $(date +%s) - since >= IDLE_SECONDS )) && { state "GPU_IDLE_30MIN used_mib=$USED"; return 0; }
    else
      [[ -n "$since" ]] && state "GPU_BUSY_AGAIN used_mib=$USED compute_apps=$APPS"
      since=""
      (( i % 30 == 0 )) && state "WAIT_GPU_IDLE used_mib=$USED compute_apps=$APPS"
    fi
    i=$((i + 1))
    sleep 60
  done
}

state "QUEUE_START job=$JOB node=$NODE cuda=$CUDA_VISIBLE_DEVICES step_gpus=${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-?}}"
while true; do
  open="$(open_chunks)"
  if (( open == 0 )); then state "NO_OPEN_CHUNKS node=$NODE"; sleep 600; [[ -e "$STOP" ]] && exit 0; continue; fi
  wait_idle || exit 0
  open="$(open_chunks)"
  (( open > 0 )) || continue
  if [[ ! -s "${LOG}_smoke.json" ]]; then
    state "SMOKE_START"
    if python3 -u -m budget_inference.ouro_deepmath_filter_a100 --smoke --model "$MODEL" \
        --attention-backend "$BACKEND" > "${LOG}_smoke.out" 2>> "${LOG}_srun.err"; then
      cp "${LOG}_smoke.out" "${LOG}_smoke.json"
      state "SMOKE_PASSED"
    else
      state "SMOKE_FAILED (see ${LOG}_smoke.out)"
      exit 5
    fi
    gpu_idle || { state "GPU_TAKEN_AFTER_SMOKE used_mib=$USED"; continue; }
  fi
  state "WORKER_START open_chunks=$open"
  set +e
  python3 -u -m budget_inference.ouro_deepmath_filter_a100 --worker --model "$MODEL" \
    --attention-backend "$BACKEND" --gpu-memory-utilization 0.9 &
  echo $! > "${LOG}_worker.pid"
  wait $!
  status=$?
  set -e
  state "WORKER_EXIT status=$status"
  case "$status" in
    0) ;;
    130|143) exit 0 ;;
    *) exit "$status" ;;
  esac
done
