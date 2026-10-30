#!/usr/bin/env bash
# Stop every worker started from this directory (SIGTERM; in-flight requests are dropped and
# redone on the next start, committed trials are kept). Usage: bash stop_all.sh [gpu index...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$HERE"
if [[ $# -gt 0 ]]; then
  files=(); for gpu in "$@"; do files+=("logs/gpu${gpu}.pid"); done
else
  files=(logs/gpu*.pid)
fi
for f in "${files[@]}"; do
  [[ -f "$f" ]] || continue
  pid="$(cat "$f")"
  if kill -0 "$pid" 2>/dev/null; then kill -TERM "$pid" && echo "stopped $f (pid $pid)"; fi
done
