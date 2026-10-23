#!/usr/bin/env bash
# Download MATH train + MATH-500 and write <data dir>/{math_train,math500}.jsonl + data_manifest.json.
#   LOOPRL_VENV=<venv> bash prepare_data.sh <data dir>
set -euo pipefail
OUT="${1:?Usage: $0 <data dir>}"
: "${LOOPRL_VENV:?set LOOPRL_VENV to the training venv}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$HERE/.." && pwd -P)"
SRC="$ROOT/src"; [[ -d "$SRC/rltt_baseline" ]] || SRC="$ROOT"   # project vs LoopRL layout
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$SRC" "$LOOPRL_VENV/bin/python" -m rltt_baseline.prepare_data --out "$OUT"
