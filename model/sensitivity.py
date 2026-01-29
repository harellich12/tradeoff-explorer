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
