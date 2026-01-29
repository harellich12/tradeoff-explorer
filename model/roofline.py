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
