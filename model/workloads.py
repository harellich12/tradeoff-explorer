"""
LLM workload computation formulas.

Implements FLOPs and memory transfer calculations for prefill and decode phases.
All formulas are documented with markdown explanations for UI display.
"""

from enum import Enum
from typing import Literal

from .schemas import WorkloadConfig, Assumptions, get_dtype_bytes


# =============================================================================
# Inference Mode
# =============================================================================

class InferenceMode(str, Enum):
    """Inference phase mode."""
    PREFILL = "prefill"
    DECODE = "decode"


# =============================================================================
# FLOPs Computation
# =============================================================================

def flops_per_token(cfg: WorkloadConfig, mode: Literal["prefill", "decode"] | InferenceMode) -> float:
    """
    Compute FLOPs per token for the given inference mode.
    
    **Prefill Phase:**
    - Process all input tokens in parallel
    - Attention is O(seq_len²) due to full sequence attention
    - Linear layers: 2 * params * batch (per token)
    
    **Decode Phase:**
    - Generate one token at a time
    - Attention reads from KV cache: O(seq_len) per token
    - Linear layers: 2 * params * batch (per token)
    
    Formula:
    ```
    Linear FLOPs = 2 × params × batch
    
    Prefill Attention FLOPs = 4 × batch × n_layers × seq_len × hidden_size
        (Q×K^T and Attention×V, each O(seq × d))
    
    Decode Attention FLOPs = 4 × batch × n_layers × seq_len × hidden_size  
        (attending to full KV cache)
    ```
    
    Args:
        cfg: Workload configuration
        mode: "prefill" or "decode"
        
    Returns:
        FLOPs per token (total FLOPs / seq_len for prefill, per-token for decode)
    """
    mode_str = mode.value if isinstance(mode, InferenceMode) else mode
    
    num_params = cfg.params_b * 1e9
    
    # Linear layers: 2 FLOPs per parameter per token per batch element
    # (1 multiply + 1 add for each weight)
    linear_flops_per_token = 2 * num_params * cfg.batch
    
    if mode_str == "prefill":
        # Prefill: full attention over sequence
        # Q×K^T: batch × n_layers × seq × seq × head_dim × 2 (mul + add)
        # Attn×V: batch × n_layers × seq × seq × head_dim × 2
        # Simplified: 4 × batch × n_layers × seq × hidden_size (per position)
        attention_flops_per_token = (
            4 * cfg.batch * cfg.n_layers * cfg.seq_len * cfg.hidden_size
        )
    else:  # decode
        # Decode: attend to all cached K,V (seq_len positions)
        # Each new token attends to all previous tokens
        attention_flops_per_token = (
            4 * cfg.batch * cfg.n_layers * cfg.seq_len * cfg.hidden_size
        )
    
    total_flops = linear_flops_per_token + attention_flops_per_token
    
    assert total_flops >= 0 and total_flops < float('inf'), \
        f"Invalid FLOPs: {total_flops}"
    
    return total_flops


def get_flops_formula_markdown(mode: Literal["prefill", "decode"]) -> str:
    """
    Get markdown explanation of FLOPs formula.
    
    Args:
        mode: "prefill" or "decode"
        
    Returns:
        Markdown string explaining the formula
    """
    if mode == "prefill":
        return """
**FLOPs per Token (Prefill)**

```
FLOPs = Linear + Attention

Linear  = 2 × params × batch
Attn    = 4 × batch × L × seq × d
```

Where:
- `params` = model parameters
- `L` = number of layers  
- `seq` = sequence length
- `d` = hidden dimension
"""
    else:
        return """
**FLOPs per Token (Decode)**

```
FLOPs = Linear + Attention

Linear  = 2 × params × batch
Attn    = 4 × batch × L × seq × d
```

Where:
- Attention reads full KV cache (all `seq` positions)
- `params` = model parameters
- `L` = number of layers
- `d` = hidden dimension
"""


# =============================================================================
# Memory Bytes Computation
# =============================================================================

def bytes_per_token(cfg: WorkloadConfig, assumptions: Assumptions) -> float:
    """
    Compute memory bytes transferred per token during decode.
    
    Memory bandwidth is dominated by:
    1. **Weight reads**: All model weights loaded per token (scaled by cache hit rate)
    2. **KV cache reads**: Read K,V for all previous positions per layer
    3. **KV cache writes**: Write new K,V for current position per layer
    
    Formula:
    ```
    Weight bytes = params × weight_dtype_bytes × (1 - cache_hit_rate)
    
    KV read bytes = 2 × batch × n_layers × seq_len × hidden_size × kv_dtype_bytes × kv_compression
    
    KV write bytes = 2 × batch × n_layers × hidden_size × kv_dtype_bytes × kv_compression
    
    Total = Weight bytes + KV read bytes + KV write bytes
    ```
    
    Args:
        cfg: Workload configuration
        assumptions: Efficiency assumptions (includes weight_cache_hit_rate)
        
    Returns:
        Bytes per token during decode
    """
    weight_bytes_per_param = get_dtype_bytes(cfg.weight_dtype)
    kv_bytes_per_element = cfg.kv_bytes_per_element()  # includes compression
    
    num_params = cfg.params_b * 1e9
    
    # Weight reads: full model weights, scaled by cache miss rate
    cache_miss_rate = 1.0 - assumptions.weight_cache_hit_rate
    weight_bytes = num_params * weight_bytes_per_param * cache_miss_rate
    
    # KV cache read: read all K and V for all previous tokens per layer
    # Shape: [batch, n_layers, seq_len, 2 (K,V), hidden_size]
    kv_read_bytes = (
        2  # K and V
        * cfg.batch
        * cfg.n_layers
        * cfg.seq_len  # all previous positions
        * cfg.hidden_size
        * kv_bytes_per_element
    )
    
    # KV cache write: write new K and V for current token per layer
    # Shape: [batch, n_layers, 1, 2 (K,V), hidden_size]
    kv_write_bytes = (
        2  # K and V
        * cfg.batch
        * cfg.n_layers
        * 1  # single new position
        * cfg.hidden_size
        * kv_bytes_per_element
    )
    
    total_bytes = weight_bytes + kv_read_bytes + kv_write_bytes
    
    assert total_bytes >= 0 and total_bytes < float('inf'), \
        f"Invalid bytes: {total_bytes}"
    
    return total_bytes


def bytes_per_token_prefill(cfg: WorkloadConfig, assumptions: Assumptions) -> float:
    """
    Compute memory bytes transferred per token during prefill.
    
    During prefill:
    1. **Weight reads**: Loaded once, amortized over seq_len tokens
    2. **KV cache writes**: Write all K,V for all positions
    
    Formula:
    ```
    Weight bytes (per token) = params × weight_dtype_bytes × (1 - cache_hit_rate) / seq_len
    
    KV write bytes (per token) = 2 × batch × n_layers × hidden_size × kv_dtype_bytes × kv_compression
    
    Total = Weight bytes + KV write bytes
    ```
    
    Args:
        cfg: Workload configuration
        assumptions: Efficiency assumptions
        
    Returns:
        Bytes per token during prefill
    """
    weight_bytes_per_param = get_dtype_bytes(cfg.weight_dtype)
    kv_bytes_per_element = cfg.kv_bytes_per_element()
    
    num_params = cfg.params_b * 1e9
    
    # Weight reads: amortized over all tokens in prefill
    cache_miss_rate = 1.0 - assumptions.weight_cache_hit_rate
    weight_bytes_per_token = (
        num_params * weight_bytes_per_param * cache_miss_rate / cfg.seq_len
    )
    
    # KV cache write: write K,V for each position
    kv_write_bytes = (
        2  # K and V
        * cfg.batch
        * cfg.n_layers
        * cfg.hidden_size
        * kv_bytes_per_element
    )
    
    total_bytes = weight_bytes_per_token + kv_write_bytes
    
    assert total_bytes >= 0 and total_bytes < float('inf'), \
        f"Invalid bytes: {total_bytes}"
    
    return total_bytes


def get_bytes_formula_markdown(mode: Literal["prefill", "decode"]) -> str:
    """
    Get markdown explanation of bytes per token formula.
    
    Args:
        mode: "prefill" or "decode"
        
    Returns:
        Markdown string explaining the formula
    """
    if mode == "prefill":
        return """
**Bytes per Token (Prefill)**

```
Bytes = Weights (amortized) + KV Write

Weights = params × dtype_bytes × (1 - cache_hit) / seq
KV Write = 2 × batch × L × d × kv_bytes × compression
```

Where:
- Weights amortized over `seq` tokens
- `L` = layers, `d` = hidden size
- `kv_bytes` = KV cache dtype size
"""
    else:
        return """
**Bytes per Token (Decode)**

```
Bytes = Weights + KV Read + KV Write

Weights  = params × dtype_bytes × (1 - cache_hit)
KV Read  = 2 × batch × L × seq × d × kv_bytes × comp
KV Write = 2 × batch × L × d × kv_bytes × comp
```

Where:
- Must read full KV cache (`seq` positions)
- `L` = layers, `d` = hidden size
- `comp` = KV compression ratio
"""


# =============================================================================
# Arithmetic Intensity
# =============================================================================

def arithmetic_intensity(
    cfg: WorkloadConfig, 
    assumptions: Assumptions,
    mode: Literal["prefill", "decode"] = "decode"
) -> float:
    """
    Compute arithmetic intensity (FLOP/byte) for the given mode.
    
    Higher intensity = more compute-bound
    Lower intensity = more memory-bound
    
    Args:
        cfg: Workload configuration
        assumptions: Efficiency assumptions
        mode: "prefill" or "decode"
        
    Returns:
        Arithmetic intensity in FLOP/byte
    """
    flops = flops_per_token(cfg, mode)
    
    if mode == "prefill":
        bytes_transferred = bytes_per_token_prefill(cfg, assumptions)
    else:
        bytes_transferred = bytes_per_token(cfg, assumptions)
    
    if bytes_transferred == 0:
        return float("inf")
    
    return flops / bytes_transferred


# =============================================================================
# Summary Helper
# =============================================================================

def compute_workload_summary(
    cfg: WorkloadConfig, 
    assumptions: Assumptions
) -> dict:
    """
    Compute a summary of workload metrics.
    
    Args:
        cfg: Workload configuration
        assumptions: Efficiency assumptions
        
    Returns:
        Dictionary with prefill and decode metrics
    """
    return {
        "prefill": {
            "flops_per_token": flops_per_token(cfg, "prefill"),
            "bytes_per_token": bytes_per_token_prefill(cfg, assumptions),
            "arithmetic_intensity": arithmetic_intensity(cfg, assumptions, "prefill"),
        },
        "decode": {
            "flops_per_token": flops_per_token(cfg, "decode"),
            "bytes_per_token": bytes_per_token(cfg, assumptions),
            "arithmetic_intensity": arithmetic_intensity(cfg, assumptions, "decode"),
        },
    }


# =============================================================================
# Example / Test
# =============================================================================

if __name__ == "__main__":
    from .schemas import example_instantiation
    
    workload, _, assumptions = example_instantiation()
    
    print("=== Workload Metrics ===")
    print(f"Model: {workload.name}")
    print(f"Params: {workload.params_b}B, Seq: {workload.seq_len}, Batch: {workload.batch}")
    
    print("\n--- Prefill ---")
    pf_flops = flops_per_token(workload, "prefill")
    pf_bytes = bytes_per_token_prefill(workload, assumptions)
    print(f"  FLOPs/token: {pf_flops:.2e}")
    print(f"  Bytes/token: {pf_bytes:.2e}")
    print(f"  Arithmetic intensity: {pf_flops/pf_bytes:.1f} FLOP/byte")
    
    print("\n--- Decode ---")
    dc_flops = flops_per_token(workload, "decode")
    dc_bytes = bytes_per_token(workload, assumptions)
    print(f"  FLOPs/token: {dc_flops:.2e}")
    print(f"  Bytes/token: {dc_bytes:.2e}")
    print(f"  Arithmetic intensity: {dc_flops/dc_bytes:.1f} FLOP/byte")
    
    # Verify: longer sequence = more bytes (for decode)
    from dataclasses import replace
    short_seq = replace(workload, seq_len=512)
    long_seq = replace(workload, seq_len=4096)
    
    short_bytes = bytes_per_token(short_seq, assumptions)
    long_bytes = bytes_per_token(long_seq, assumptions)
    print(f"\n--- Sequence Length Impact (Decode) ---")
    print(f"  seq=512:  {short_bytes:.2e} bytes/token")
    print(f"  seq=4096: {long_bytes:.2e} bytes/token")
    print(f"  Ratio: {long_bytes/short_bytes:.2f}x")
    
    # Verify: INT8 KV cache reduces bytes
    bf16_kv = replace(workload, kv_dtype="bf16")
    int8_kv = replace(workload, kv_dtype="int8")
    
    bf16_bytes = bytes_per_token(bf16_kv, assumptions)
    int8_bytes = bytes_per_token(int8_kv, assumptions)
    print(f"\n--- KV Dtype Impact (Decode) ---")
    print(f"  BF16: {bf16_bytes:.2e} bytes/token")
    print(f"  INT8: {int8_bytes:.2e} bytes/token")
    print(f"  Reduction: {(1 - int8_bytes/bf16_bytes)*100:.1f}%")
