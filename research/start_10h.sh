#!/bin/zsh
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
LOG_DIR="$ROOT/research"
LOG_FILE="$LOG_DIR/supervisor.log"
SESSION_NAME="diff_attn_10h"
HOURS=${1:-10}

mkdir -p "$LOG_DIR"

if screen -ls | grep -q "[.]$SESSION_NAME[[:space:]]"; then
  echo "session $SESSION_NAME is already active"
  exit 1
fi

cd "$ROOT"
screen -dmS "$SESSION_NAME" /bin/zsh -lc "cd \"$ROOT\" && ./research/supervise_10h.sh \"$HOURS\""
echo "started screen session $SESSION_NAME"
echo "log: $LOG_FILE"
