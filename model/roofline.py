"""
Roofline model computation.

Computes upper-bound throughput and bottleneck classification.
"""

from typing import Literal

from .schemas import WorkloadConfig, HardwareConfig, Assumptions, AnalysisResult
from .workloads import (
    flops_per_token,
    bytes_per_token,
    bytes_per_token_prefill,
    arithmetic_intensity,
)


def classify_bottleneck(
    ai: float, hardware: HardwareConfig
) -> Literal["compute", "memory"]:
    """
    Classify whether workload is compute-bound or memory-bound.

    Args:
        ai: Arithmetic intensity (FLOP/byte ratio) of the workload
        hardware: Hardware configuration

    Returns:
        "compute" if compute-bound, "memory" if memory-bound
    """
    ridge_point = hardware.arithmetic_intensity_ridge()

    if ai >= ridge_point:
        return "compute"
    else:
        return "memory"


def compute_roofline_throughput(
    workload: WorkloadConfig, 
    hardware: HardwareConfig,
    assumptions: Assumptions,
    mode: Literal["prefill", "decode"] = "decode"
) -> float:
    """
    Compute the roofline-limited throughput in tokens per second.

    The roofline model gives the upper bound on performance:
    - Compute bound: limited by peak FLOP/s
    - Memory bound: limited by memory bandwidth

    Args:
        workload: Workload configuration
        hardware: Hardware configuration
        assumptions: Efficiency assumptions
        mode: "prefill" or "decode"

    Returns:
        Maximum achievable tokens per second
    """
    flops = flops_per_token(workload, mode)
    
    if mode == "prefill":
        mem_bytes = bytes_per_token_prefill(workload, assumptions)
    else:
        mem_bytes = bytes_per_token(workload, assumptions)

    if flops == 0 or mem_bytes == 0:
        return 0.0

    # Compute-limited throughput (apply utilization)
    peak_flops = hardware.peak_tflops * 1e12 * assumptions.compute_util  # FLOP/s
    compute_limited = peak_flops / flops

    # Memory-limited throughput (apply utilization)
    peak_bandwidth = hardware.hbm_gbps * 1e9 * assumptions.bw_util  # bytes/s
    memory_limited = peak_bandwidth / mem_bytes

    # Roofline: take the minimum (the binding constraint)
    return min(compute_limited, memory_limited)


def compute_utilization(
    actual_throughput: float,
    workload: WorkloadConfig,
    hardware: HardwareConfig,
    assumptions: Assumptions,
) -> float:
    """
    Compute hardware utilization as fraction of roofline throughput.

    Args:
        actual_throughput: Measured or estimated throughput
        workload: Workload configuration
        hardware: Hardware configuration
        assumptions: Efficiency assumptions

    Returns:
        Utilization as a fraction (0.0 to 1.0)
    """
    roofline_throughput = compute_roofline_throughput(workload, hardware, assumptions)

    if roofline_throughput == 0:
        return 0.0

    return min(actual_throughput / roofline_throughput, 1.0)


def analyze(
    workload: WorkloadConfig, 
    hardware: HardwareConfig,
    assumptions: Assumptions | None = None
) -> AnalysisResult:
    """
    Perform full roofline analysis for a workload/hardware pair.

    Args:
        workload: Workload configuration
        hardware: Hardware configuration
        assumptions: Efficiency assumptions (defaults to Assumptions())

    Returns:
        Complete analysis result
    """
    if assumptions is None:
        assumptions = Assumptions()
    
    prefill_flops = flops_per_token(workload, "prefill")
    decode_flops = flops_per_token(workload, "decode")
    memory_bytes = bytes_per_token(workload, assumptions)
    ai = arithmetic_intensity(workload, assumptions, "decode")

    throughput = compute_roofline_throughput(workload, hardware, assumptions, "decode")
    bottleneck = classify_bottleneck(ai, hardware)

    # Utilization assumes we achieve roofline (ideal case)
    utilization = 1.0 if throughput > 0 else 0.0

    return AnalysisResult(
        workload=workload.name,
        hardware=hardware.name,
        throughput_tokens_per_sec=throughput,
        bottleneck=bottleneck,
        arithmetic_intensity=ai,
        utilization=utilization,
        prefill_flops=prefill_flops,
        decode_flops_per_token=decode_flops,
        memory_bytes_per_token=memory_bytes,
    )


def analyze_detailed(
    workload: WorkloadConfig,
    hardware: HardwareConfig,
    assumptions: Assumptions | None = None,
    mode: Literal["prefill", "decode"] = "decode"
) -> dict:
    """
    Perform detailed roofline analysis with comprehensive metrics.
    
    Returns a dict with:
    - tokens_s_compute: Compute-limited throughput (tokens/s)
    - tokens_s_bw: Bandwidth-limited throughput (tokens/s)
    - tokens_s_final: Final throughput after overlap factor
    - bottleneck_class: "COMPUTE" or "MEMORY_BW"
    - required_tflops: Required compute for this workload
    - required_gbps: Required bandwidth for this workload
    - compute_util_est: Estimated compute utilization
    - bw_util_est: Estimated bandwidth utilization
    - explanation: Human-readable explanation of the result
    
    Formula:
    - tokens_s_compute = (peak_tflops * 1e12 * compute_util) / flops_per_token
    - tokens_s_bw = (hbm_gbps * 1e9 * bw_util) / bytes_per_token
    - tokens_s_final = min(tokens_s_compute, tokens_s_bw) * overlap_factor
    - bottleneck = COMPUTE if tokens_s_compute < tokens_s_bw else MEMORY_BW
    
    Args:
        workload: Workload configuration
        hardware: Hardware configuration  
        assumptions: Efficiency assumptions (defaults to Assumptions())
        mode: "prefill" or "decode"
        
    Returns:
        Dict with detailed analysis results
    """
    if assumptions is None:
        assumptions = Assumptions()
    
    # Get per-token requirements
    flops = flops_per_token(workload, mode)
    if mode == "prefill":
        mem_bytes = bytes_per_token_prefill(workload, assumptions)
    else:
        mem_bytes = bytes_per_token(workload, assumptions)
    
    # Calculate throughput limits
    # tokens_s_compute = (peak_tflops * 1e12 * compute_util) / flops_per_token
    peak_flops_s = hardware.peak_tflops * 1e12 * assumptions.compute_util
    tokens_s_compute = peak_flops_s / flops if flops > 0 else float('inf')
    
    # tokens_s_bw = (hbm_gbps * 1e9 * bw_util) / bytes_per_token
    peak_bw_s = hardware.hbm_gbps * 1e9 * assumptions.bw_util
    tokens_s_bw = peak_bw_s / mem_bytes if mem_bytes > 0 else float('inf')
    
    # tokens_s_final = min(tokens_s_compute, tokens_s_bw) * overlap_factor
    # Note: overlap_factor > 0 means some compute/memory can overlap
    # Final throughput accounts for this overlap benefit
    overlap_multiplier = 1.0 + assumptions.overlap_factor  # overlap helps
    tokens_s_final = min(tokens_s_compute, tokens_s_bw) * overlap_multiplier
    
    # Bottleneck classification
    # COMPUTE if tokens_s_compute < tokens_s_bw else MEMORY_BW
    if tokens_s_compute < tokens_s_bw:
        bottleneck_class = "COMPUTE"
    else:
        bottleneck_class = "MEMORY_BW"
    
    # Required resources at final throughput
    required_tflops = (flops * tokens_s_final) / 1e12  # TFLOP/s needed
    required_gbps = (mem_bytes * tokens_s_final) / 1e9  # GB/s needed
    
    # Utilization estimates (actual / available)
    compute_util_est = required_tflops / (hardware.peak_tflops * assumptions.compute_util) if hardware.peak_tflops > 0 else 0
    bw_util_est = required_gbps / (hardware.hbm_gbps * assumptions.bw_util) if hardware.hbm_gbps > 0 else 0
    
    # Clamp to [0, 1]
    compute_util_est = min(max(compute_util_est, 0.0), 1.0)
    bw_util_est = min(max(bw_util_est, 0.0), 1.0)
    
    # Human-readable explanation
    explanation = _generate_explanation(
        workload=workload,
        hardware=hardware,
        mode=mode,
        tokens_s_compute=tokens_s_compute,
        tokens_s_bw=tokens_s_bw,
        tokens_s_final=tokens_s_final,
        bottleneck_class=bottleneck_class,
        required_tflops=required_tflops,
        required_gbps=required_gbps,
    )
    
    return {
        "tokens_s_compute": tokens_s_compute,
        "tokens_s_bw": tokens_s_bw,
        "tokens_s_final": tokens_s_final,
        "bottleneck_class": bottleneck_class,
        "required_tflops": required_tflops,
        "required_gbps": required_gbps,
        "compute_util_est": compute_util_est,
        "bw_util_est": bw_util_est,
        "flops_per_token": flops,
        "bytes_per_token": mem_bytes,
        "explanation": explanation,
    }


def _generate_explanation(
    workload: WorkloadConfig,
    hardware: HardwareConfig,
    mode: str,
    tokens_s_compute: float,
    tokens_s_bw: float,
    tokens_s_final: float,
    bottleneck_class: str,
    required_tflops: float,
    required_gbps: float,
) -> str:
    """Generate a human-readable explanation of the analysis."""
    
    # Format numbers nicely
    def fmt(val, suffix=""):
        if val >= 1e6:
            return f"{val/1e6:.1f}M{suffix}"
        elif val >= 1e3:
            return f"{val/1e3:.1f}K{suffix}"
        else:
            return f"{val:.1f}{suffix}"
    
    bottleneck_label = "**compute-bound**" if bottleneck_class == "COMPUTE" else "**memory bandwidth-bound**"
    
    lines = [
        f"### Analysis: {workload.name} on {hardware.name}",
        f"",
        f"**Mode:** {mode.capitalize()}",
        f"",
        f"**Bottleneck:** This workload is {bottleneck_label}.",
        f"",
        f"**Throughput Limits:**",
        f"- Compute-limited: {fmt(tokens_s_compute)} tokens/s",
        f"- Bandwidth-limited: {fmt(tokens_s_bw)} tokens/s",
        f"- **Achievable: {fmt(tokens_s_final)} tokens/s**",
        f"",
        f"**Resource Requirements:**",
        f"- Compute: {required_tflops:.1f} TFLOP/s (of {hardware.peak_tflops:.0f} TFLOP/s available)",
        f"- Bandwidth: {required_gbps:.1f} GB/s (of {hardware.hbm_gbps:.0f} GB/s available)",
    ]
    
    # Add recommendation based on bottleneck
    lines.append("")
    if bottleneck_class == "COMPUTE":
        lines.append("**Recommendation:** To improve throughput, consider:")
        lines.append("- Using a GPU with higher TFLOP/s")
        lines.append("- Reducing model size (fewer parameters)")
        lines.append("- Using lower-precision weights (INT8/INT4)")
    else:
        lines.append("**Recommendation:** To improve throughput, consider:")
        lines.append("- Using a GPU with higher memory bandwidth")
        lines.append("- Reducing batch size (less KV cache)")
        lines.append("- Using INT8 KV cache to reduce memory traffic")
        lines.append("- Enabling KV compression")
    
    return "\n".join(lines)

