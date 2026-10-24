#!/usr/bin/env bash
# Download the training / validation data into <data dir> and record revisions and SHA256 in
# <data dir>/data_manifest.json. Default: the SAPD math training set (deepmath_level6.parquet,
# 57,046 DeepMath-103K problems, the configs' training data) and MATH-500 (math500.jsonl).
# Add the RLTT paper's MATH train (math_train.jsonl) with --datasets deepmath_level6,math500,math_train.
#   LOOPRL_VENV=<venv> bash prepare_data.sh <data dir> [--datasets ...]
set -euo pipefail
OUT="${1:?Usage: $0 <data dir> [--datasets ...]}"
shift
: "${LOOPRL_VENV:?set LOOPRL_VENV to the training venv}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$HERE/.." && pwd -P)"
SRC="$ROOT/src"; [[ -d "$SRC/rltt_baseline" ]] || SRC="$ROOT"   # project vs LoopRL layout
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$SRC" "$LOOPRL_VENV/bin/python" -m rltt_baseline.prepare_data --out "$OUT" "$@"
OUT="$(cd "$OUT" && pwd -P)"
echo "# paths for run_train.sh / smoke_grpo.sh:"
[[ -f "$OUT/deepmath_level6.parquet" ]] && echo "export LOOPRL_TRAIN_FILE=$OUT/deepmath_level6.parquet"
[[ -f "$OUT/math500.jsonl" ]] && echo "export LOOPRL_MATH500_JSONL=$OUT/math500.jsonl"
exit 0
