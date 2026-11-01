#!/usr/bin/env bash
# Start one detached worker per listed GPU; they keep running after you log out.
# Usage: bash run_all.sh 0 1 2 3 [-- sample.py options]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$HERE"
GPUS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do GPUS+=("$1"); shift; done
[[ "${1:-}" == "--" ]] && shift
[[ ${#GPUS[@]} -gt 0 ]] || { echo "Usage: $0 <gpu index>... [-- sample.py options]" >&2; exit 2; }
mkdir -p logs
for gpu in "${GPUS[@]}"; do
  setsid nohup bash run_worker.sh "$gpu" "$@" > "logs/gpu${gpu}.out" 2>&1 < /dev/null &
  echo "started GPU $gpu (log: logs/gpu${gpu}.log)"
done
