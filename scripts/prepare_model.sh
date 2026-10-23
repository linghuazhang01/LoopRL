#!/usr/bin/env bash
# Fetch Ouro-2.6B-Thinking (pinned revision, SHA-checked) and build <models dir>/Ouro-2.6B-Thinking-RLTT.
#   LOOPRL_VENV=<venv> bash prepare_model.sh <models dir> [--source <existing official dir>]
set -euo pipefail
OUT="${1:?Usage: $0 <models dir> [--source <dir>]}"
shift
: "${LOOPRL_VENV:?set LOOPRL_VENV to the training venv}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$HERE/.." && pwd -P)"
SRC="$ROOT/src"; [[ -d "$SRC/rltt_baseline" ]] || SRC="$ROOT"
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$SRC" "$LOOPRL_VENV/bin/python" -m rltt_baseline.fetch_model --out "$OUT" "$@"
