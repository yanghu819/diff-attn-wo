# Current Best Diff-Attn Run

- `val_bpb = 2.078972`
- `memory_gb = 11.3`
- `num_steps = 144`
- `description = layers=1, lam=-2.0, wo_scale=0.1, blend=1.0, y2norm=False, batch=2**15`

## Config

- `USE_DIFF_ATTN = True`
- `DIFF_ATTN_LAST_LAYERS = 1`
- `DIFF_ATTN_Q2_SOURCE = wo`
- `DIFF_ATTN_LAMBDA_INIT = -2.0`
- `DIFF_ATTN_WO_INIT = small_random`
- `DIFF_ATTN_WO_INIT_SCALE = 0.1`
- `DIFF_ATTN_Q2_BLEND = 1.0`
- `DIFF_ATTN_Y2_NORM = False`
- `TOTAL_BATCH_SIZE = 32768`
