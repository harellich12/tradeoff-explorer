# Model Notes (v0) — Workload Primitives

## 1) Dtype sizes
Defaults:
- BF16/FP16: 2 bytes
- INT8: 1 byte
- INT4: 0.5 bytes (modeled as half-byte)

These sizes apply separately to:
- weights dtype
- KV dtype

## 2) FLOPs per token (high-level)
We approximate FLOPs/token for transformer inference using parameterized components:
- linear layers (QKV, output projection, MLP)
- attention compute (depends on seq_len and heads)

The tool should:
- keep formulas **explicit and documented**
- prefer “reasonable approximations” over complex derivations
- avoid claiming accuracy without calibration anchors

## 3) Bytes per token (high-level)
Bytes/token includes:
- weight reads (scaled by (1 - weight_cache_hit_rate))
- KV cache reads/writes:
  - grows with seq_len and n_layers
  - uses kv dtype bytes and kv_compression

DECODE should exhibit monotonic behavior:
- as seq_len increases, KV bytes/token increases
- reducing KV dtype from BF16 to INT8 reduces bytes/token

## 4) Roofline coupling
We translate FLOPs/token and bytes/token into a throughput upper bound using:
- peak compute * compute_util
- peak HBM BW * bw_util

Final bound uses an overlap factor.

## 5) Invariants (tests enforce these)
- tokens/s should not decrease when increasing HBM BW (when BW-limited)
- tokens/s should not decrease when increasing compute (when compute-limited)
- DECODE KV bytes/token increases with seq_len
- INT8 KV reduces bytes/token vs BF16
- all outputs finite and non-negative

## 6) Calibration (future)
If calibration anchors are introduced:
- fit utilization factors to anchors
- warn when extrapolating far outside anchor regimes
