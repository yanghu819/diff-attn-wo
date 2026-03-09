#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
REPO_NAME=${1:-diff-attn-wo}
VISIBILITY=${2:-private}

cd "$SCRIPT_DIR"

if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI is not authenticated. Run: gh auth login" >&2
  exit 1
fi

if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin HEAD
  exit 0
fi

case "$VISIBILITY" in
  private)
    GH_VIS=(--private)
    ;;
  public)
    GH_VIS=(--public)
    ;;
  *)
    echo "visibility must be 'private' or 'public'" >&2
    exit 1
    ;;
esac

gh repo create "$REPO_NAME" "${GH_VIS[@]}" \
  --source=. \
  --remote=origin \
  --description "Backup of diff-attn-wo experiments from autoresearch-mlx" \
  --push
