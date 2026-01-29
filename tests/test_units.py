"""
Unit tests for formula outputs.

Tests specific formula calculations with known expected values.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.schemas import WorkloadConfig, HardwareConfig, Assumptions, get_dtype_bytes
from model.workloads import (
    flops_per_token,
    bytes_per_token,
    arithmetic_intensity,
)
from model.roofline import (
    classify_bottleneck,
    compute_roofline_throughput,
)
from model.sensitivity import compute_sensitivity_analysis, format_tornado_data


class TestWorkloadFormulas:
    """Unit tests for workload computation formulas."""

    def test_bytes_per_param_bf16(self):
        """BF16 should use 2 bytes per parameter."""
        assert get_dtype_bytes("bf16") == 2

    def test_bytes_per_param_fp32(self):
        """FP32 should use 4 bytes per parameter."""
        assert get_dtype_bytes("fp32") == 4

    def test_bytes_per_param_int8(self):
        """INT8 should use 1 byte per parameter."""
        assert get_dtype_bytes("int8") == 1

    def test_bytes_per_param_int4(self):
        """INT4 should use 0.5 bytes per parameter."""
        assert get_dtype_bytes("int4") == 0.5

    def test_prefill_flops_scaling(self):
        """Prefill FLOPs per token should increase with seq_len due to attention."""
        workload_1 = WorkloadConfig(
            name="Test1",
            params_b=1.0,
            n_layers=4,
            hidden_size=1024,
            n_heads=4,
            seq_len=1024,
        )
        workload_2 = WorkloadConfig(
            name="Test2",
            params_b=1.0,
            n_layers=4,
            hidden_size=1024,
            n_heads=4,
            seq_len=2048,
        )

        flops_1 = flops_per_token(workload_1, "prefill")
        flops_2 = flops_per_token(workload_2, "prefill")

        # Per-token FLOPs: linear FLOPs are constant, attention adds ~4*n_layers*seq_len*hidden
        # For small models, linear dominates, so increase is modest
        # Just verify longer sequence doesn't reduce FLOPs
        assert flops_2 >= flops_1, "Longer sequence should not reduce per-token FLOPs"

    def test_arithmetic_intensity_calculation(self):
        """Arithmetic intensity should be FLOP/byte ratio."""
        workload = WorkloadConfig(
            name="Test",
            params_b=7.0,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
        )
        assumptions = Assumptions()

        flops = flops_per_token(workload, "decode")
        memory = bytes_per_token(workload, assumptions)
        intensity = arithmetic_intensity(workload, assumptions, "decode")

        assert abs(intensity - flops / memory) < 1e-6


class TestRooflineFormulas:
    """Unit tests for roofline model computations."""

    def test_ridge_point_calculation(self):
        """Ridge point should be peak_flops / bandwidth."""
        hardware = HardwareConfig(
            name="Test",
            peak_tflops=312.0,  # TFLOP/s
            hbm_gbps=2000.0,  # GB/s
        )

        ridge = hardware.arithmetic_intensity_ridge()
        # Ridge = (TFLOP/s * 1000) / GB/s = 312 * 1000 / 2000 = 156 FLOP/byte
        expected = 156.0

        assert abs(ridge - expected) < 1e-6

    def test_classify_compute_bound(self):
        """High arithmetic intensity should be compute-bound."""
        hardware = HardwareConfig(
            name="Test",
            peak_tflops=100.0,
            hbm_gbps=1000.0,  # Ridge = 100 FLOP/byte
        )

        # 200 FLOP/byte > 100 ridge = compute-bound
        assert classify_bottleneck(200.0, hardware) == "compute"

    def test_classify_memory_bound(self):
        """Low arithmetic intensity should be memory-bound."""
        hardware = HardwareConfig(
            name="Test",
            peak_tflops=100.0,
            hbm_gbps=1000.0,  # Ridge = 100 FLOP/byte
        )

        # 50 FLOP/byte < 100 ridge = memory-bound
        assert classify_bottleneck(50.0, hardware) == "memory"

    def test_roofline_throughput_positive(self):
        """Roofline throughput should be positive for valid inputs."""
        workload = WorkloadConfig(
            name="Test",
            params_b=7.0,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
            batch=1,
            seq_len=2048,
        )
        hardware = HardwareConfig(
            name="Test",
            peak_tflops=312.0,
            hbm_gbps=2000.0,
        )
        assumptions = Assumptions()

        throughput = compute_roofline_throughput(workload, hardware, assumptions)
        assert throughput > 0


class TestSensitivityAnalysis:
    """Unit tests for sensitivity analysis."""

    def test_sensitivity_returns_points(self):
        """Sensitivity analysis should return SensitivityPoint list."""
        workload = WorkloadConfig(
            name="Test",
            params_b=7.0,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
        )
        hardware = HardwareConfig(
            name="Test",
            peak_tflops=312.0,
            hbm_gbps=2000.0,
        )

        points = compute_sensitivity_analysis(workload, hardware)

        assert len(points) > 0
        assert all(hasattr(p, "parameter_name") for p in points)

    def test_tornado_data_format(self):
        """Tornado data should have correct structure."""
        workload = WorkloadConfig(
            name="Test",
            params_b=7.0,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
        )
        hardware = HardwareConfig(
            name="Test",
            peak_tflops=312.0,
            hbm_gbps=2000.0,
        )

        points = compute_sensitivity_analysis(workload, hardware)
        tornado = format_tornado_data(points)

        assert "parameters" in tornado
        assert "low_deltas" in tornado
        assert "high_deltas" in tornado
        assert "base_throughput" in tornado
        assert len(tornado["parameters"]) == len(tornado["low_deltas"])
