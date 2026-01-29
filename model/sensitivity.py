"""
Sensitivity analysis for one-at-a-time parameter variation.

Generates data for tornado charts showing parameter impact on throughput.
"""

from dataclasses import replace
from typing import List, Dict, Any, Tuple

from .schemas import WorkloadConfig, HardwareConfig, Assumptions, SensitivityPoint
from .roofline import compute_roofline_throughput


# Default variation ranges (as fractions of base value)
DEFAULT_VARIATION = 0.25  # ±25%


def vary_parameter(
    workload: WorkloadConfig,
    hardware: HardwareConfig,
    assumptions: Assumptions,
    param_name: str,
    low_value: Any,
    high_value: Any,
) -> SensitivityPoint:
    """
    Compute sensitivity for a single parameter.

    Args:
        workload: Base workload configuration
        hardware: Hardware configuration
        assumptions: Efficiency assumptions
        param_name: Name of the parameter to vary
        low_value: Low value for the parameter
        high_value: High value for the parameter

    Returns:
        SensitivityPoint with throughput at low, base, and high values
    """
    base_value = getattr(workload, param_name)
    base_throughput = compute_roofline_throughput(workload, hardware, assumptions)

    # Create modified workloads
    low_workload = replace(workload, **{param_name: low_value})
    high_workload = replace(workload, **{param_name: high_value})

    low_throughput = compute_roofline_throughput(low_workload, hardware, assumptions)
    high_throughput = compute_roofline_throughput(high_workload, hardware, assumptions)

    return SensitivityPoint(
        parameter_name=param_name,
        low_value=float(low_value) if isinstance(low_value, (int, float)) else 0,
        base_value=float(base_value) if isinstance(base_value, (int, float)) else 0,
        high_value=float(high_value) if isinstance(high_value, (int, float)) else 0,
        low_throughput=low_throughput,
        base_throughput=base_throughput,
        high_throughput=high_throughput,
    )


def compute_sensitivity_analysis(
    workload: WorkloadConfig,
    hardware: HardwareConfig,
    assumptions: Assumptions | None = None,
    parameters: List[Tuple[str, Any, Any]] | None = None,
) -> List[SensitivityPoint]:
    """
    Perform one-at-a-time sensitivity analysis.

    Args:
        workload: Base workload configuration
        hardware: Hardware configuration
        assumptions: Efficiency assumptions (defaults to Assumptions())
        parameters: List of (param_name, low_value, high_value) tuples.
                   If None, uses default parameters with ±25% variation.

    Returns:
        List of SensitivityPoint for each parameter
    """
    if assumptions is None:
        assumptions = Assumptions()
    
    if parameters is None:
        # Default parameters to vary
        parameters = [
            (
                "batch",
                max(1, int(workload.batch * (1 - DEFAULT_VARIATION))),
                int(workload.batch * (1 + DEFAULT_VARIATION)) + 1,
            ),
            (
                "seq_len",
                max(128, int(workload.seq_len * (1 - DEFAULT_VARIATION))),
                int(workload.seq_len * (1 + DEFAULT_VARIATION)),
            ),
            (
                "params_b",
                workload.params_b * (1 - DEFAULT_VARIATION),
                workload.params_b * (1 + DEFAULT_VARIATION),
            ),
        ]

    results = []
    for param_name, low_val, high_val in parameters:
        point = vary_parameter(workload, hardware, assumptions, param_name, low_val, high_val)
        results.append(point)

    # Sort by impact (descending)
    results.sort(
        key=lambda p: abs(p.high_throughput - p.low_throughput), reverse=True
    )

    return results


def format_tornado_data(
    sensitivity_points: List[SensitivityPoint],
) -> Dict[str, List]:
    """
    Format sensitivity analysis data for tornado chart visualization.

    Args:
        sensitivity_points: List of SensitivityPoint from analysis

    Returns:
        Dictionary with 'parameters', 'low_deltas', 'high_deltas' lists
    """
    parameters = []
    low_deltas = []
    high_deltas = []

    for point in sensitivity_points:
        parameters.append(point.parameter_name)
        low_deltas.append(point.low_throughput - point.base_throughput)
        high_deltas.append(point.high_throughput - point.base_throughput)

    return {
        "parameters": parameters,
        "low_deltas": low_deltas,
        "high_deltas": high_deltas,
        "base_throughput": sensitivity_points[0].base_throughput if sensitivity_points else 0,
    }


# Dataclass for sensitivity lever result
from dataclasses import dataclass


@dataclass
class LeverResult:
    """Result of sensitivity analysis for a single lever."""
    param_name: str
    param_display_name: str
    base_value: float
    low_value: float
    high_value: float
    base_throughput: float
    low_throughput: float
    high_throughput: float
    percent_impact: float  # max percent change from base
    explanation: str


# Parameter display names and explanations
PARAM_INFO = {
    "hbm_gbps": {
        "display": "HBM Bandwidth (GB/s)",
        "explanation_up": "Increasing memory bandwidth helps when memory-bound",
        "explanation_down": "Reducing memory bandwidth hurts throughput when memory-bound",
    },
    "peak_tflops": {
        "display": "Peak Compute (TFLOP/s)",
        "explanation_up": "More compute capacity helps when compute-bound",
        "explanation_down": "Less compute capacity hurts throughput when compute-bound",
    },
    "batch": {
        "display": "Batch Size",
        "explanation_up": "Larger batches amortize weight reads, improving efficiency",
        "explanation_down": "Smaller batches reduce memory efficiency",
    },
    "seq_len": {
        "display": "Sequence Length",
        "explanation_up": "Longer sequences increase KV cache reads per token",
        "explanation_down": "Shorter sequences reduce KV cache overhead",
    },
    "kv_compression": {
        "display": "KV Compression",
        "explanation_up": "Less compression means more KV bytes per token",
        "explanation_down": "More compression reduces KV bytes per token",
    },
    "compute_util": {
        "display": "Compute Utilization",
        "explanation_up": "Higher utilization extracts more from available compute",
        "explanation_down": "Lower utilization wastes compute capacity",
    },
    "bw_util": {
        "display": "Bandwidth Utilization",
        "explanation_up": "Higher utilization extracts more from available bandwidth",
        "explanation_down": "Lower utilization wastes memory bandwidth",
    },
}


def compute_sensitivity(
    workload: WorkloadConfig,
    hardware: HardwareConfig,
    assumptions: Assumptions,
    mode: str = "decode",
    variation: float = 0.20,
    top_n: int = 3,
) -> List[LeverResult]:
    """
    Compute sensitivity analysis for key knobs.
    
    Performs one-at-a-time +/- variation on:
    - Hardware: hbm_gbps, peak_tflops
    - Workload: batch, seq_len, kv_compression
    - Assumptions: compute_util, bw_util
    
    Args:
        workload: Workload configuration
        hardware: Hardware configuration
        assumptions: Efficiency assumptions
        mode: "prefill" or "decode"
        variation: Fraction to vary (0.20 = ±20%)
        top_n: Number of top levers to return
    
    Returns:
        List of top N LeverResult sorted by absolute impact
    """
    from .roofline import analyze_detailed
    
    # Get baseline throughput
    base_analysis = analyze_detailed(workload, hardware, assumptions, mode)
    base_throughput = base_analysis["tokens_s_final"]
    
    results = []
    
    # Define parameters to vary: (param_name, object_type, getter, low_val, high_val)
    params_to_vary = []
    
    # Hardware params
    params_to_vary.append((
        "hbm_gbps", "hardware",
        hardware.hbm_gbps,
        hardware.hbm_gbps * (1 - variation),
        hardware.hbm_gbps * (1 + variation),
    ))
    params_to_vary.append((
        "peak_tflops", "hardware",
        hardware.peak_tflops,
        hardware.peak_tflops * (1 - variation),
        hardware.peak_tflops * (1 + variation),
    ))
    
    # Workload params
    params_to_vary.append((
        "batch", "workload",
        workload.batch,
        max(1, int(workload.batch * (1 - variation))),
        max(1, int(workload.batch * (1 + variation))),
    ))
    params_to_vary.append((
        "seq_len", "workload",
        workload.seq_len,
        max(128, int(workload.seq_len * (1 - variation))),
        int(workload.seq_len * (1 + variation)),
    ))
    params_to_vary.append((
        "kv_compression", "workload",
        workload.kv_compression,
        max(0.1, workload.kv_compression * (1 - variation)),
        min(1.0, workload.kv_compression * (1 + variation)),
    ))
    
    # Assumption params
    params_to_vary.append((
        "compute_util", "assumptions",
        assumptions.compute_util,
        max(0.1, assumptions.compute_util * (1 - variation)),
        min(1.0, assumptions.compute_util * (1 + variation)),
    ))
    params_to_vary.append((
        "bw_util", "assumptions",
        assumptions.bw_util,
        max(0.1, assumptions.bw_util * (1 - variation)),
        min(1.0, assumptions.bw_util * (1 + variation)),
    ))
    
    for param_name, obj_type, base_val, low_val, high_val in params_to_vary:
        # Create modified configs
        if obj_type == "hardware":
            hw_low = replace(hardware, **{param_name: low_val})
            hw_high = replace(hardware, **{param_name: high_val})
            wl_low, wl_high = workload, workload
            ass_low, ass_high = assumptions, assumptions
        elif obj_type == "workload":
            wl_low = replace(workload, **{param_name: low_val})
            wl_high = replace(workload, **{param_name: high_val})
            hw_low, hw_high = hardware, hardware
            ass_low, ass_high = assumptions, assumptions
        else:  # assumptions
            ass_low = replace(assumptions, **{param_name: low_val})
            ass_high = replace(assumptions, **{param_name: high_val})
            hw_low, hw_high = hardware, hardware
            wl_low, wl_high = workload, workload
        
        # Compute throughput at low and high
        low_analysis = analyze_detailed(wl_low, hw_low, ass_low, mode)
        high_analysis = analyze_detailed(wl_high, hw_high, ass_high, mode)
        
        low_throughput = low_analysis["tokens_s_final"]
        high_throughput = high_analysis["tokens_s_final"]
        
        # Calculate percent impact (max of low/high change from base)
        low_pct = abs(low_throughput - base_throughput) / base_throughput * 100 if base_throughput > 0 else 0
        high_pct = abs(high_throughput - base_throughput) / base_throughput * 100 if base_throughput > 0 else 0
        percent_impact = max(low_pct, high_pct)
        
        # Generate explanation
        param_info = PARAM_INFO.get(param_name, {"display": param_name, "explanation_up": "", "explanation_down": ""})
        if high_throughput > low_throughput:
            explanation = param_info["explanation_up"]
        else:
            explanation = param_info["explanation_down"]
        
        results.append(LeverResult(
            param_name=param_name,
            param_display_name=param_info["display"],
            base_value=float(base_val),
            low_value=float(low_val),
            high_value=float(high_val),
            base_throughput=base_throughput,
            low_throughput=low_throughput,
            high_throughput=high_throughput,
            percent_impact=percent_impact,
            explanation=explanation,
        ))
    
    # Sort by percent impact descending
    results.sort(key=lambda r: r.percent_impact, reverse=True)
    
    return results[:top_n]
