"""
Memory fit estimation module.

Estimates memory requirements for LLM inference and checks if workload fits
in hardware memory with appropriate margins.
"""

from typing import Dict, Literal, Optional
from .schemas import WorkloadConfig, HardwareConfig, get_dtype_bytes


# Memory fit status literals
MemoryFitStatus = Literal["OK", "WARNING", "ERROR", "UNKNOWN"]


def estimate_weights_bytes(workload: WorkloadConfig) -> float:
    """
    Estimate memory required for model weights.
    
    Args:
        workload: Workload configuration
        
    Returns:
        Estimated weight memory in bytes
    """
    params = workload.params_b * 1e9  # Convert billions to actual count
    dtype_bytes = get_dtype_bytes(workload.weight_dtype)
    return params * dtype_bytes


def estimate_kv_bytes(workload: WorkloadConfig) -> float:
    """
    Estimate memory required for KV cache.
    
    KV cache stores keys and values for all layers:
    - 2 (K and V) * batch * seq_len * n_layers * hidden_size * dtype_bytes
    - Divided by kv_compression (capped at small epsilon to avoid division by zero)
    
    Args:
        workload: Workload configuration
        
    Returns:
        Estimated KV cache memory in bytes
    """
    kv_dtype_bytes = get_dtype_bytes(workload.kv_dtype)
    kv_compression = max(workload.kv_compression, 1e-6)  # Avoid division by zero
    
    # KV cache: 2 * batch * seq_len * n_layers * hidden_size * dtype_bytes / compression
    kv_bytes = (
        2  # K and V
        * workload.batch
        * workload.seq_len
        * workload.n_layers
        * workload.hidden_size
        * kv_dtype_bytes
        / kv_compression
    )
    
    return kv_bytes


def estimate_total_bytes(
    workload: WorkloadConfig,
    overhead_fraction: float = 0.15
) -> Dict[str, float]:
    """
    Estimate total memory required for inference.
    
    Args:
        workload: Workload configuration
        overhead_fraction: Additional overhead (activations, framework, etc.)
        
    Returns:
        Dictionary with:
        - weights_bytes: Memory for model weights
        - kv_bytes: Memory for KV cache
        - total_bytes: Sum of weights + KV
        - total_with_margin_bytes: Total with overhead margin
    """
    weights_bytes = estimate_weights_bytes(workload)
    kv_bytes = estimate_kv_bytes(workload)
    total_bytes = weights_bytes + kv_bytes
    total_with_margin_bytes = total_bytes * (1 + overhead_fraction)
    
    return {
        "weights_bytes": weights_bytes,
        "kv_bytes": kv_bytes,
        "total_bytes": total_bytes,
        "total_with_margin_bytes": total_with_margin_bytes,
    }


def memory_fit_status(
    workload: WorkloadConfig,
    hardware: HardwareConfig,
    overhead_fraction: float = 0.15,
    warn_threshold: float = 0.85,
    error_threshold: float = 0.95,
) -> Dict:
    """
    Check if workload fits in hardware memory.
    
    Args:
        workload: Workload configuration
        hardware: Hardware configuration
        overhead_fraction: Additional overhead (activations, framework, etc.)
        warn_threshold: Memory usage fraction that triggers WARNING (default 0.85)
        error_threshold: Memory usage fraction that triggers ERROR (default 0.95)
        
    Returns:
        Dictionary with:
        - status: "OK", "WARNING", "ERROR", or "UNKNOWN"
        - required_gb: Estimated memory required (with margin)
        - memory_gb: Hardware memory (or None)
        - headroom_gb: Available headroom (or None)
        - weights_gb: Memory for weights
        - kv_gb: Memory for KV cache
        - usage_fraction: Fraction of memory used (or None)
        - rationale: Human-readable explanation
    """
    mem_estimate = estimate_total_bytes(workload, overhead_fraction)
    
    weights_gb = mem_estimate["weights_bytes"] / 1e9
    kv_gb = mem_estimate["kv_bytes"] / 1e9
    required_gb = mem_estimate["total_with_margin_bytes"] / 1e9
    
    # Handle missing memory_gb
    if hardware.memory_gb is None:
        return {
            "status": "UNKNOWN",
            "required_gb": required_gb,
            "memory_gb": None,
            "headroom_gb": None,
            "weights_gb": weights_gb,
            "kv_gb": kv_gb,
            "usage_fraction": None,
            "rationale": f"Memory not specified for {hardware.name}. Please enter memory_gb to check fit.",
        }
    
    memory_gb = hardware.memory_gb
    headroom_gb = memory_gb - required_gb
    usage_fraction = required_gb / memory_gb
    
    # Determine status
    if required_gb > memory_gb * error_threshold:
        status: MemoryFitStatus = "ERROR"
        rationale = (
            f"⚠️ Model likely won't fit! Requires ~{required_gb:.1f}GB but {hardware.name} has {memory_gb:.0f}GB. "
            f"Reduce seq_len, batch, or quantize KV cache."
        )
    elif required_gb > memory_gb * warn_threshold:
        status = "WARNING"
        rationale = (
            f"⚡ Tight fit: ~{required_gb:.1f}GB required vs {memory_gb:.0f}GB available "
            f"({headroom_gb:.1f}GB headroom). Monitor for OOM."
        )
    else:
        status = "OK"
        rationale = (
            f"✅ Should fit: ~{required_gb:.1f}GB required vs {memory_gb:.0f}GB available "
            f"({headroom_gb:.1f}GB headroom)."
        )
    
    return {
        "status": status,
        "required_gb": required_gb,
        "memory_gb": memory_gb,
        "headroom_gb": headroom_gb,
        "weights_gb": weights_gb,
        "kv_gb": kv_gb,
        "usage_fraction": usage_fraction,
        "rationale": rationale,
    }


def format_memory_fit_for_memo(fit_result: Dict, overhead_fraction: float) -> str:
    """
    Format memory fit result for decision memo.
    
    Args:
        fit_result: Result from memory_fit_status()
        overhead_fraction: Overhead fraction used
        
    Returns:
        HTML-formatted memory fit section
    """
    status = fit_result["status"]
    
    # Status badge styling
    badge_colors = {
        "OK": "#28a745",
        "WARNING": "#ffc107",
        "ERROR": "#dc3545",
        "UNKNOWN": "#6c757d",
    }
    badge_color = badge_colors.get(status, "#6c757d")
    
    html = f"""
    <div style="margin: 10px 0; padding: 10px; border-left: 4px solid {badge_color}; background: #f8f9fa;">
        <strong>Status:</strong> 
        <span style="background: {badge_color}; color: white; padding: 2px 8px; border-radius: 4px;">{status}</span>
        <br><br>
        <strong>Breakdown:</strong>
        <ul style="margin: 5px 0;">
            <li>Weights: {fit_result['weights_gb']:.2f} GB</li>
            <li>KV Cache: {fit_result['kv_gb']:.2f} GB</li>
            <li>Overhead: {overhead_fraction*100:.0f}%</li>
            <li><strong>Total Required: {fit_result['required_gb']:.2f} GB</strong></li>
    """
    
    if fit_result["memory_gb"] is not None:
        html += f"""
            <li>Hardware Memory: {fit_result['memory_gb']:.0f} GB</li>
            <li>Headroom: {fit_result['headroom_gb']:.2f} GB</li>
        """
    
    html += f"""
        </ul>
        <p><em>{fit_result['rationale']}</em></p>
    </div>
    """
    
    return html


def format_memory_fit_for_log(fit_result: Dict) -> Dict:
    """
    Format memory fit result for JSONL logging.
    
    Args:
        fit_result: Result from memory_fit_status()
        
    Returns:
        Dictionary suitable for JSON serialization
    """
    return {
        "status": fit_result["status"],
        "required_gb": round(fit_result["required_gb"], 3),
        "memory_gb": fit_result["memory_gb"],
        "headroom_gb": round(fit_result["headroom_gb"], 3) if fit_result["headroom_gb"] is not None else None,
        "weights_gb": round(fit_result["weights_gb"], 3),
        "kv_gb": round(fit_result["kv_gb"], 3),
        "usage_fraction": round(fit_result["usage_fraction"], 4) if fit_result["usage_fraction"] is not None else None,
    }
