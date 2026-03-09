#!/bin/zsh
set -u

ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
LOG_DIR="$ROOT/research"
RUNNER_LOG="$LOG_DIR/runner.log"
SUPERVISOR_LOG="$LOG_DIR/supervisor.log"
STATE_FILE="$LOG_DIR/state.json"
HOURS=${1:-10}

mkdir -p "$LOG_DIR"

deadline_reached() {
  if [ ! -f "$STATE_FILE" ]; then
    return 1
  fi
  cd "$ROOT"
  uv run python - <<'PY'
import json
from datetime import datetime
from pathlib import Path

state = json.loads(Path("research/state.json").read_text())
deadline = state.get("deadline_at")
if not deadline:
    raise SystemExit(1)
raise SystemExit(0 if datetime.now() >= datetime.fromisoformat(deadline) else 1)
PY
}

{
  echo "[supervisor] started at $(date '+%F %T')"
  while true; do
    if deadline_reached; then
      echo "[supervisor] deadline reached at $(date '+%F %T'), exiting"
      break
    fi
    echo "[supervisor] launching runner at $(date '+%F %T')"
    cd "$ROOT"
    set +e
    uv run python research/diff_research_runner.py --hours "$HOURS" >>"$RUNNER_LOG" 2>&1
    status=$?
    set -e
    echo "[supervisor] runner exited with status $status at $(date '+%F %T')"
    if deadline_reached; then
      echo "[supervisor] deadline reached after runner exit"
      break
    fi
    sleep 5
  done
} >>"$SUPERVISOR_LOG" 2>&1
