# Current Best Diff-Attn Run

- `val_bpb = 2.036343`
- `memory_gb = 11.3`
- `num_steps = 169`
- `description = layers=1, lam=-2.25, wo_scale=0.12, lam_max=1.0, blend=1.0, q2_scale=1.0, y2norm=False, y2center=False, batch=2**15`

## Config

- `USE_DIFF_ATTN = True`
- `DIFF_ATTN_LAST_LAYERS = 1`
- `DIFF_ATTN_Q2_SOURCE = wo`
- `DIFF_ATTN_LAMBDA_INIT = -2.25`
- `DIFF_ATTN_WO_INIT = small_random`
- `DIFF_ATTN_WO_INIT_SCALE = 0.12`
- `DIFF_ATTN_LAMBDA_MAX = 1.0`
- `DIFF_ATTN_Q2_BLEND = 1.0`
- `DIFF_ATTN_Q2_SCALE = 1.0`
- `DIFF_ATTN_Y2_NORM = False`
- `DIFF_ATTN_Y2_CENTER_HEADS = False`
- `TOTAL_BATCH_SIZE = 32768`
