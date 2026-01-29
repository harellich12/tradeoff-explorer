---
description: Project context for LLM Hardware-Model Tradeoff Explorer
---

# Tradeoff Explorer - Project Context

## High level context and planning of project

You are a senior product-minded engineer. Build a Streamlit-only MVP called “Model ↔ Hardware Tradeoff Explorer” in Python.

Goal:
A decision support tool (not a cycle-accurate simulator) that compares 2–4 hardware configurations for a given AI workload and outputs:
- bottleneck classification (COMPUTE / MEMORY_BW / INTERCONNECT / OTHER)
- required vs available compute + bandwidth
- a directional upper-bound throughput estimate (tokens/sec)
- top sensitivity levers (which knobs matter most)
- a clean 1-page exportable “Decision Memo” in HTML

Hard constraints:
- Streamlit-only UI (no FastAPI, no Next.js)
- All assumptions must be explicit in the UI and in the memo
- Must include tests enforcing monotonic invariants
- Must log each run to a local JSONL file (inputs, outputs, timestamp, git hash if available)

Scope (v0):
- Workloads: LLM Inference only, with two modes: PREFILL and DECODE
- Hardware model: peak compute, HBM bandwidth, optional SRAM size (informational in v0), optional interconnect BW/lat (v0 can be “not modeled” but keep the field)
- No training, no MoE in v0
- Do not claim exact perf; label outputs as “upper bound under assumptions”

Repo structure:
tradeoff_explorer/
  app.py                      # Streamlit UI entry
  model/
    schemas.py                # dataclasses / pydantic-like validation (no heavy deps needed)
    workloads.py              # LLM prefill/decode formulas for FLOPs + bytes per token
    hardware.py               # hardware schema + preset archetypes
    roofline.py               # computes upper-bound throughput and bottleneck classification
    sensitivity.py            # one-at-a-time sensitivity (tornado data)
    memo.py                   # HTML memo generator
  data/
    workload_presets.json     # a few model presets (7B/13B/70B style)
    hardware_presets.json     # 8–12 hardware archetypes (generic)
  logs/
    runs.jsonl                # appended logs
  tests/
    test_invariants.py        # monotonicity & sanity checks
    test_units.py             # unit tests for formula outputs
  README.md                   # setup & usage
  pyproject.toml or requirements.txt

UI requirements:
- Left panel: select workload mode (PREFILL/DECODE), choose model preset or custom fields:
  - params (billions), n_layers, hidden_size, n_heads, seq_len, batch
  - dtype for weights (BF16/INT8/INT4), dtype for KV (BF16/INT8), KV compression factor (default 1.0)
  - utilization factors: compute_util (0.3 default), bw_util (0.7 default), overlap factor (0.8 default)
- Main panel: compare 2–4 hardware configs (preset + editable):
  - peak_tflops (for selected dtype), hbm_gbps, optional interconnect_gbps, optional cost_per_hour
- Outputs per config:
  - bottleneck label and explanation
  - required_compute_tflops, required_bw_gbps (or per-token bytes)
  - compute_util_est, bw_util_est
  - throughput_upper_bound_tokens_per_s
  - top 3 levers (from sensitivity)
- Buttons:
  - “Run analysis”
  - “Export decision memo (HTML)” (download)

Modeling approach:
- Implement simple roofline-style upper bound:
  - tokens/s_compute = (peak_flops * compute_util) / flops_per_token
  - tokens/s_bw = (hbm_bytes_per_s * bw_util) / bytes_per_token
  - tokens/s = min(tokens/s_compute, tokens/s_bw) * overlap_factor
- Bottleneck classification:
  - if tokens/s_compute < tokens/s_bw → COMPUTE
  - else MEMORY_BW
  - INTERCONNECT only if modeled and dominates (optional)
- Bytes per token for PREFILL and DECODE must include:
  - weight reads (scaled by dtype size and cache hit assumptions; v0 can assume streaming weights from HBM with a “weight_cache_hit_rate” knob default 0)
  - KV reads/writes (dominant for DECODE; depends on seq_len, n_layers, n_heads, head_dim; apply KV dtype size and compression factor)
- FLOPs per token:
  - Provide a reasonable approximation using standard transformer components. Keep it parameterized and documented.

Sensitivity:
- One-at-a-time +/- 20% perturbation of key knobs:
  - hbm_gbps, peak_tflops, batch, seq_len, kv_dtype_bytes, kv_compression, compute_util, bw_util
- Output top levers sorted by absolute impact on tokens/s.

Logging:
- Append JSON object per run to logs/runs.jsonl with inputs, outputs, timestamp, app_version.

Decision memo (HTML):
- Scenario summary
- Table comparing configs
- Bottleneck + “top levers”
- Explicit assumptions & limitations
- “What would change this decision?” section listing dominant assumptions

Testing:
- Add invariant tests:
  - increasing hbm_gbps must not decrease tokens/s (when BW-limited)
  - increasing peak_tflops must not decrease tokens/s (when compute-limited)
  - increasing seq_len should not reduce KV bytes per token in DECODE
  - switching KV dtype BF16→INT8 should reduce bytes_per_token
- Add sanity tests for non-negative FLOPs/bytes and reasonable ranges

Deliverables:
- Fully runnable Streamlit app with presets
- README with install/run
- Tests passing (pytest)
- Export memo works

## Overview

**Tradeoff Explorer** is a Streamlit-based Python tool for analyzing LLM inference performance tradeoffs across different hardware configurations and model architectures using **roofline modeling**.

The goal is to help users understand:
- Whether a given workload/hardware combo is **compute-bound** or **memory-bound**
- What **throughput** (tokens/second) is achievable at the roofline
- How **parameter changes** (batch size, sequence length, precision) affect performance
- How to optimize inference by choosing the right hardware or tuning workload settings

---

## Architecture

```
tradeoff_explorer/
├── app.py                      # Streamlit UI entry point
├── model/
│   ├── schemas.py              # Dataclasses: WorkloadConfig, HardwareConfig, Assumptions
│   ├── workloads.py            # flops_per_token(), bytes_per_token() formulas
│   ├── hardware.py             # Hardware preset loading utilities
│   ├── roofline.py             # Roofline throughput & bottleneck classification
│   ├── sensitivity.py          # One-at-a-time sensitivity (tornado data)
│   └── memo.py                 # HTML memo/report generator
├── data/
│   ├── workload_presets.json   # Model presets (7B/13B/70B style)
│   └── hardware_presets.json   # 12 hardware archetypes (H100, A100, RTX 4090, etc.)
├── logs/
│   └── runs.jsonl              # Appended run logs
├── tests/
│   ├── test_invariants.py      # Monotonicity & sanity checks
│   └── test_units.py           # Unit tests for formulas
├── requirements.txt
└── README.md
```

---

## Core Schemas (model/schemas.py)

### WorkloadConfig
- `name`, `params_b` (billions), `n_layers`, `hidden_size`, `n_heads`
- `seq_len`, `batch`, `weight_dtype`, `kv_dtype`, `kv_compression`

### HardwareConfig
- `name`, `peak_tflops`, `hbm_gbps`
- Optional: `sram_gb`, `interconnect_gbps`, `cost_per_hour`

### Assumptions
- `compute_util`, `bw_util`, `overlap_factor`, `weight_cache_hit_rate`

---

## Key Formulas (model/workloads.py)

### FLOPs per Token
```
Linear FLOPs  = 2 × params × batch
Attention     = 4 × batch × n_layers × seq_len × hidden_size
```

### Bytes per Token (Decode)
```
Weights   = params × dtype_bytes × (1 - cache_hit_rate)
KV Read   = 2 × batch × n_layers × seq_len × hidden_size × kv_bytes × compression
KV Write  = 2 × batch × n_layers × hidden_size × kv_bytes × compression
```

### Roofline
```
Compute-limited throughput = peak_tflops / flops_per_token
Memory-limited throughput  = hbm_gbps / bytes_per_token
Achieved throughput        = min(compute, memory)
```

---

## Implementation Status

### Completed
- [x] Project scaffold (folders, requirements, README)
- [x] Core schemas with validation
- [x] Workload formulas (prefill/decode, weight cache, KV compression)
- [x] App.py with interactive configuration UI

### In Progress / TODO
- [ ] Hardware module with preset loading
- [ ] Roofline analysis integration in UI
- [ ] Sensitivity analysis (tornado charts)
- [ ] HTML memo export
- [ ] Tests update for new API

---

## Design Principles

1. **Dataclasses over Pydantic** - Keep dependencies minimal
2. **Explicit formulas** - Document every computation with markdown helpers
3. **Prefill vs Decode separation** - Different bottlenecks for each phase
4. **Utilization assumptions** - Real hardware doesn't hit peak; model this
5. **UI shows formulas** - Help users understand, not just compute

---

## Propmt Guardrails

1. Do NOT add new features not listed.
2. Keep changes minimal; prefer editing existing files.
3. If you need a new dependency, justify it and keep it lightweight.

---

## Usage

```bash
cd tradeoff_explorer
pip install -r requirements.txt
python -m streamlit run app.py
```

To run tests:
```bash
python -m pytest -q
```