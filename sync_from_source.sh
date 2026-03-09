#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
SOURCE_DIR=${SOURCE_DIR:-"$SCRIPT_DIR/../autoresearch"}
MESSAGE=${1:-"backup: $(date '+%Y-%m-%d %H:%M:%S')"}

if [ ! -d "$SOURCE_DIR" ]; then
  echo "source repo not found: $SOURCE_DIR" >&2
  exit 1
fi

rsync -a \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$SOURCE_DIR/" "$SCRIPT_DIR/"

cd "$SCRIPT_DIR"
if [ -n "$(git status --porcelain)" ]; then
  git add -A
  git commit -m "$MESSAGE"
  if git remote get-url origin >/dev/null 2>&1; then
    git push origin HEAD
  fi
else
  echo "no changes to back up"
fi
