# diff-attn-wo backup

This repository is a snapshot-first backup of the working diff-attention experiments from `../autoresearch`.

## Current best config

- `USE_DIFF_ATTN = True`
- `DIFF_ATTN_LAST_LAYERS = 1`
- `DIFF_ATTN_Q2_SOURCE = "wo"`
- `DIFF_ATTN_LAMBDA_INIT = -2.0`
- `DIFF_ATTN_WO_INIT = "small_random"`
- `DIFF_ATTN_WO_INIT_SCALE = 0.1`
- `TOTAL_BATCH_SIZE = 2**15`

## Best result on this machine

- `val_bpb = 2.078972`
- baseline without diff attention: `2.230553`
- improvement: `0.151581`

## Included artifacts

- `train.py`: current best diff-attention implementation
- `diff_attention_curves.png`
- `diff_attention_curves.svg`

## Local backup workflow

From this backup repo:

```bash
./sync_from_source.sh "backup: <short note>"
```

If no commit message is given, the script uses a timestamped default.

## GitHub push workflow

After `gh auth login`:

```bash
./push_to_github.sh diff-attn-wo private
```

The first run creates the repo and pushes. Later runs just push new commits.
