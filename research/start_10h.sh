#!/bin/zsh
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
LOG_DIR="$ROOT/research"
LOG_FILE="$LOG_DIR/runner.log"
PID_FILE="$LOG_DIR/runner.pid"
HOURS=${1:-10}

mkdir -p "$LOG_DIR"

if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  if kill -0 "$PID" >/dev/null 2>&1; then
    echo "runner already active with pid $PID"
    exit 1
  fi
fi

cd "$ROOT"
nohup uv run python research/diff_research_runner.py --hours "$HOURS" >>"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "started runner pid $(cat "$PID_FILE")"
echo "log: $LOG_FILE"
