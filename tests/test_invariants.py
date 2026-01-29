"""
Invariant tests for roofline computations.

Tests monotonicity and sanity checks for the model computations.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.schemas import WorkloadConfig, HardwareConfig, Assumptions
from model.roofline import compute_roofline_throughput, analyze
from model.workloads import (
    flops_per_token,
    bytes_per_token,
)


# Test fixtures
@pytest.fixture
def base_workload():
    """Standard 7B model workload for testing."""
    return WorkloadConfig(
        name="Test-7B",
        params_b=7.0,
        n_layers=32,
        hidden_size=4096,
        n_heads=32,
        seq_len=2048,
        batch=1,
        weight_dtype="bf16",
        kv_dtype="bf16",
        kv_compression=1.0,
    )


@pytest.fixture
def base_hardware():
    """Standard datacenter GPU for testing."""
    return HardwareConfig(
        name="Test-GPU",
        peak_tflops=312.0,
        hbm_gbps=2000.0,
    )


@pytest.fixture
def base_assumptions():
    """Default assumptions for testing."""
    return Assumptions()


class TestMonotonicity:
    """Tests for monotonic relationships in computations."""

    def test_larger_model_more_flops(self):
        """Larger models should require more FLOPs."""
        small_workload = WorkloadConfig(
            name="Small",
            params_b=7.0,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
        )
        large_workload = WorkloadConfig(
            name="Large",
            params_b=70.0,
            n_layers=80,
            hidden_size=8192,
            n_heads=64,
        )

        small_flops = flops_per_token(small_workload, "decode")
        large_flops = flops_per_token(large_workload, "decode")

        assert large_flops > small_flops, "Larger model should have more FLOPs"

    def test_larger_batch_more_flops(self, base_workload):
        """Larger batch sizes should require more FLOPs."""
        from dataclasses import replace

        batch_1 = replace(base_workload, batch=1)
        batch_8 = replace(base_workload, batch=8)

        flops_1 = flops_per_token(batch_1, "decode")
        flops_8 = flops_per_token(batch_8, "decode")

        assert flops_8 > flops_1, "Larger batch should have more total FLOPs"

    def test_longer_sequence_more_memory(self, base_workload, base_assumptions):
        """Longer sequences should require more memory bandwidth (for KV cache)."""
        from dataclasses import replace

        seq_512 = replace(base_workload, seq_len=512)
        seq_4096 = replace(base_workload, seq_len=4096)

        mem_512 = bytes_per_token(seq_512, base_assumptions)
        mem_4096 = bytes_per_token(seq_4096, base_assumptions)

        assert mem_4096 > mem_512, "Longer sequence should need more memory"

    def test_faster_hardware_more_throughput(self, base_workload, base_assumptions):
        """Faster hardware should achieve higher throughput."""
        slow_hw = HardwareConfig(
            name="Slow",
            peak_tflops=100.0,
            hbm_gbps=1000.0,
        )
        fast_hw = HardwareConfig(
            name="Fast",
            peak_tflops=500.0,
            hbm_gbps=3000.0,
        )

        slow_throughput = compute_roofline_throughput(base_workload, slow_hw, base_assumptions)
        fast_throughput = compute_roofline_throughput(base_workload, fast_hw, base_assumptions)

        assert fast_throughput > slow_throughput, "Faster hardware should have higher throughput"


class TestSanityChecks:
    """Basic sanity checks for computation outputs."""

    def test_throughput_positive(self, base_workload, base_hardware, base_assumptions):
        """Throughput should always be positive."""
        throughput = compute_roofline_throughput(base_workload, base_hardware, base_assumptions)
        assert throughput > 0, "Throughput should be positive"

    def test_flops_positive(self, base_workload):
        """FLOPs should always be positive."""
        prefill = flops_per_token(base_workload, "prefill")
        decode = flops_per_token(base_workload, "decode")

        assert prefill > 0, "Prefill FLOPs should be positive"
        assert decode > 0, "Decode FLOPs should be positive"

    def test_memory_positive(self, base_workload, base_assumptions):
        """Memory usage should always be positive."""
        memory = bytes_per_token(base_workload, base_assumptions)
        assert memory > 0, "Memory usage should be positive"

    def test_analysis_result_valid(self, base_workload, base_hardware):
        """Analysis should return valid result object."""
        result = analyze(base_workload, base_hardware)

        assert result.workload == base_workload.name
        assert result.hardware == base_hardware.name
        assert result.throughput_tokens_per_sec > 0
        assert result.bottleneck in ("compute", "memory")
        assert result.arithmetic_intensity > 0
        assert 0 <= result.utilization <= 1

    def test_bottleneck_classification(self, base_workload, base_hardware):
        """Bottleneck should be either compute or memory bound."""
        result = analyze(base_workload, base_hardware)
        assert result.bottleneck in ("compute", "memory")
