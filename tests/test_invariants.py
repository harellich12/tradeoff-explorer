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


class TestBottleneckInvariants:
    """Tests for throughput invariants based on bottleneck type."""

    def test_bw_limited_increasing_hbm_gbps(self):
        """When BW-limited, increasing hbm_gbps should not decrease tokens_s_final."""
        from dataclasses import replace
        from model.roofline import analyze_detailed
        
        # Create a memory-bound scenario (small batch, long sequence)
        workload = WorkloadConfig(
            name="BW-Test",
            params_b=7.0,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
            seq_len=4096,
            batch=1,
            weight_dtype="bf16",
            kv_dtype="bf16",
            kv_compression=1.0,
        )
        assumptions = Assumptions()
        
        # Low bandwidth hardware
        hw_low_bw = HardwareConfig(
            name="Low-BW",
            peak_tflops=500.0,  # High compute to ensure BW-bound
            hbm_gbps=1000.0,
        )
        hw_high_bw = replace(hw_low_bw, name="High-BW", hbm_gbps=2000.0)
        
        result_low = analyze_detailed(workload, hw_low_bw, assumptions, "decode")
        result_high = analyze_detailed(workload, hw_high_bw, assumptions, "decode")
        
        # Verify we're actually memory-bound
        assert result_low["bottleneck_class"] == "MEMORY_BW", "Should be memory-bound for this test"
        
        # Increasing bandwidth should not decrease throughput
        assert result_high["tokens_s_final"] >= result_low["tokens_s_final"], \
            "Increasing hbm_gbps should not decrease throughput when BW-limited"

    def test_compute_limited_increasing_peak_tflops(self):
        """When compute-limited, increasing peak_tflops should not decrease tokens_s_final."""
        from dataclasses import replace
        from model.roofline import analyze_detailed
        
        # Create a compute-bound scenario (large batch, high BW)
        workload = WorkloadConfig(
            name="Compute-Test",
            params_b=70.0,
            n_layers=80,
            hidden_size=8192,
            n_heads=64,
            seq_len=512,
            batch=32,
            weight_dtype="bf16",
            kv_dtype="bf16",
            kv_compression=1.0,
        )
        assumptions = Assumptions()
        
        # High bandwidth, low compute hardware
        hw_low_compute = HardwareConfig(
            name="Low-Compute",
            peak_tflops=100.0,
            hbm_gbps=5000.0,  # Very high BW to ensure compute-bound
        )
        hw_high_compute = replace(hw_low_compute, name="High-Compute", peak_tflops=500.0)
        
        result_low = analyze_detailed(workload, hw_low_compute, assumptions, "decode")
        result_high = analyze_detailed(workload, hw_high_compute, assumptions, "decode")
        
        # Verify we're actually compute-bound
        assert result_low["bottleneck_class"] == "COMPUTE", "Should be compute-bound for this test"
        
        # Increasing compute should not decrease throughput
        assert result_high["tokens_s_final"] >= result_low["tokens_s_final"], \
            "Increasing peak_tflops should not decrease throughput when compute-limited"


class TestKVDtypeInvariants:
    """Tests for KV dtype impact on memory bandwidth."""

    def test_kv_int8_reduces_bytes_per_token(self):
        """KV dtype INT8 should reduce bytes/token vs BF16 (all else equal)."""
        from dataclasses import replace
        
        workload_bf16 = WorkloadConfig(
            name="KV-BF16",
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
        workload_int8 = replace(workload_bf16, name="KV-INT8", kv_dtype="int8")
        
        assumptions = Assumptions()
        
        bytes_bf16 = bytes_per_token(workload_bf16, assumptions)
        bytes_int8 = bytes_per_token(workload_int8, assumptions)
        
        assert bytes_int8 < bytes_bf16, \
            "INT8 KV cache should use fewer bytes per token than BF16"

    def test_kv_int4_reduces_bytes_more_than_int8(self):
        """KV dtype INT4 should reduce bytes/token more than INT8."""
        from dataclasses import replace
        
        workload_int8 = WorkloadConfig(
            name="KV-INT8",
            params_b=7.0,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
            seq_len=2048,
            batch=1,
            weight_dtype="bf16",
            kv_dtype="int8",
            kv_compression=1.0,
        )
        workload_int4 = replace(workload_int8, name="KV-INT4", kv_dtype="int4")
        
        assumptions = Assumptions()
        
        bytes_int8 = bytes_per_token(workload_int8, assumptions)
        bytes_int4 = bytes_per_token(workload_int4, assumptions)
        
        assert bytes_int4 < bytes_int8, \
            "INT4 KV cache should use fewer bytes per token than INT8"


class TestOutputSanity:
    """Enhanced sanity checks for finite and non-negative outputs."""

    def test_all_outputs_finite(self, base_workload, base_hardware, base_assumptions):
        """All computed outputs should be finite (not inf or nan)."""
        import math
        from model.roofline import analyze_detailed
        
        analysis = analyze_detailed(base_workload, base_hardware, base_assumptions, "decode")
        
        assert math.isfinite(analysis["tokens_s_final"]), "tokens_s_final should be finite"
        assert math.isfinite(analysis["tokens_s_compute"]), "tokens_s_compute should be finite"
        assert math.isfinite(analysis["tokens_s_bw"]), "tokens_s_bw should be finite"
        assert math.isfinite(analysis["required_tflops"]), "required_tflops should be finite"
        assert math.isfinite(analysis["required_gbps"]), "required_gbps should be finite"

    def test_all_outputs_non_negative(self, base_workload, base_hardware, base_assumptions):
        """All computed outputs should be non-negative."""
        from model.roofline import analyze_detailed
        
        analysis = analyze_detailed(base_workload, base_hardware, base_assumptions, "decode")
        
        assert analysis["tokens_s_final"] >= 0, "tokens_s_final should be non-negative"
        assert analysis["tokens_s_compute"] >= 0, "tokens_s_compute should be non-negative"
        assert analysis["tokens_s_bw"] >= 0, "tokens_s_bw should be non-negative"
        assert analysis["required_tflops"] >= 0, "required_tflops should be non-negative"
        assert analysis["required_gbps"] >= 0, "required_gbps should be non-negative"

    def test_utilization_in_valid_range(self, base_workload, base_hardware, base_assumptions):
        """Utilization estimates should be between 0 and 1."""
        from model.roofline import analyze_detailed
        
        analysis = analyze_detailed(base_workload, base_hardware, base_assumptions, "decode")
        
        assert 0 <= analysis["compute_util_est"] <= 1, "compute_util_est should be in [0, 1]"
        assert 0 <= analysis["bw_util_est"] <= 1, "bw_util_est should be in [0, 1]"

    def test_flops_and_bytes_consistent(self, base_workload, base_assumptions):
        """FLOPs and bytes should be consistent across modes."""
        from model.workloads import flops_per_token, bytes_per_token, bytes_per_token_prefill
        
        decode_flops = flops_per_token(base_workload, "decode")
        prefill_flops = flops_per_token(base_workload, "prefill")
        decode_bytes = bytes_per_token(base_workload, base_assumptions)
        prefill_bytes = bytes_per_token_prefill(base_workload, base_assumptions)
        
        # All should be positive finite numbers
        import math
        assert math.isfinite(decode_flops) and decode_flops > 0
        assert math.isfinite(prefill_flops) and prefill_flops > 0
        assert math.isfinite(decode_bytes) and decode_bytes > 0
        assert math.isfinite(prefill_bytes) and prefill_bytes > 0

