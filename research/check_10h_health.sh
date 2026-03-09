#!/bin/zsh
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
SESSION_NAME="diff_attn_10h"
RUNNER_LOG="$ROOT/research/runner.log"
RESULTS_FILE="$ROOT/research/diff_results.tsv"
REMOTE_BRANCH="codex%2Fdiff-attn-10h-20260309"

issues=0

echo "[health] root: $ROOT"

if screen -ls | /usr/bin/grep -qF ".$SESSION_NAME"; then
  echo "[health] screen: ok ($SESSION_NAME)"
else
  echo "[health] screen: missing ($SESSION_NAME)"
  issues=1
fi

supervisor_count=$(pgrep -fc '/Users/hy3/Desktop/diff-attn-wo/research/supervise_10h.sh' || true)
runner_count=$(pgrep -fc '/Users/hy3/Desktop/diff-attn-wo/.venv/bin/python3 research/diff_research_runner.py' || true)
echo "[health] supervisor_count: $supervisor_count"
echo "[health] runner_count: $runner_count"
if [ "$supervisor_count" -lt 1 ] || [ "$runner_count" -lt 1 ]; then
  issues=1
fi

if [ -f "$RUNNER_LOG" ]; then
  now=$(date +%s)
  mtime=$(stat -f %m "$RUNNER_LOG")
  age=$((now - mtime))
  echo "[health] runner_log_age_s: $age"
  if [ "$age" -gt 60 ]; then
    issues=1
  fi
else
  echo "[health] runner log missing"
  issues=1
fi

if [ -f "$RESULTS_FILE" ]; then
  rows=$(tail -n +2 "$RESULTS_FILE" | wc -l | tr -d ' ')
  echo "[health] completed_trials: $rows"
else
  echo "[health] results file missing"
  issues=1
fi

if command -v gh >/dev/null 2>&1; then
  remote_sha=$(gh api "repos/yanghu819/diff-attn-wo/branches/$REMOTE_BRANCH" --jq '.commit.sha' 2>/dev/null || true)
  if [ -n "$remote_sha" ]; then
    echo "[health] remote_head: $remote_sha"
  else
    echo "[health] remote_head: unavailable"
    issues=1
  fi
fi

exit "$issues"
