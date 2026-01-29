# Assumptions & Interpretation Guide

This tool produces a **directional upper bound** on tokens/sec using a roofline-style model.

## 1) Core idea
For each hardware config:
- compute-limited upper bound:
  - tokens/s_compute = (peak_flops * compute_util) / flops_per_token
- bandwidth-limited upper bound:
  - tokens/s_bw = (hbm_bytes/s * bw_util) / bytes_per_token
- final:
  - tokens/s = min(tokens/s_compute, tokens/s_bw) * overlap_factor

The bottleneck is whichever bound is smaller.

## 2) Utilization factors (why they exist)
Peak specs are rarely achieved.
- compute_util (default ~0.3): accounts for kernel inefficiency, control overhead, under-occupancy.
- bw_util (default ~0.7): accounts for non-ideal memory patterns and contention.
- overlap_factor (default ~0.8): accounts for imperfect overlap of compute and memory.

These are user-controlled because they vary by implementation stack.

## 3) Caching assumptions
- weight_cache_hit_rate:
  - 0.0 = weights streamed from HBM (worst case)
  - >0.0 reduces weight-read bytes per token
In reality, caching depends on model size, batch, and memory hierarchy.

## 4) Workload modeling (v0)
Workloads are approximations of transformer inference.
Two modes:
- PREFILL: processes prompt tokens; attention is heavier; different traffic profile.
- DECODE: generates tokens sequentially; **KV cache traffic dominates** as seq_len grows.

KV cache bytes/token depends on:
- n_layers
- seq_len
- hidden size / heads / head_dim
- KV dtype (BF16 vs INT8)
- KV compression factor

## 5) What this tool is good for
- Correctly identifying **what dominates** (compute vs BW).
- Understanding **sensitivity**: which knobs matter most.
- Comparing **relative** outcomes between options under the same assumptions.

## 6) What it is NOT good for
- Predicting exact achieved tokens/sec.
- Comparing across radically different software stacks without recalibrating utilization factors.
- Modeling networked scaling (TP/EP) unless explicit comm modeling is added.

## 7) “What would change the decision?”
The decision can flip if any of these change materially:
- KV cache compression/quantization effectiveness
- batch size regime
- achieved memory BW vs peak
- kernel fusion / attention implementation quality
- model architecture deviations (GQA/MQA, SSMs, MoE)

The exported memo includes these as flags.
