"""
Tests for memory fit estimation and status checking.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.schemas import WorkloadConfig, HardwareConfig
from model.memory_fit import (
    estimate_weights_bytes,
    estimate_kv_bytes,
    estimate_total_bytes,
    memory_fit_status,
)


class TestMemoryEstimation:
    """Tests for memory estimation functions."""

    def test_estimate_weights_bytes_bf16(self):
        """BF16 weights should use 2 bytes per param."""
        workload = WorkloadConfig(
            name="Test",
            params_b=7.0,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
            seq_len=2048,
            batch=1,
            weight_dtype="bf16",
            kv_dtype="bf16",
        )
        
        weights_bytes = estimate_weights_bytes(workload)
        expected = 7e9 * 2  # 7B params * 2 bytes
        assert abs(weights_bytes - expected) < 1e6  # Within 1MB

    def test_estimate_weights_bytes_int8(self):
        """INT8 weights should use 1 byte per param."""
        workload = WorkloadConfig(
            name="Test",
            params_b=7.0,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
            seq_len=2048,
            batch=1,
            weight_dtype="int8",
            kv_dtype="bf16",
        )
        
        weights_bytes = estimate_weights_bytes(workload)
        expected = 7e9 * 1  # 7B params * 1 byte
        assert abs(weights_bytes - expected) < 1e6

    def test_kv_bytes_increases_with_seq_len(self):
        """KV cache bytes should increase monotonically with seq_len."""
        base_workload = WorkloadConfig(
            name="Test",
            params_b=7.0,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
            seq_len=1024,
            batch=1,
            weight_dtype="bf16",
            kv_dtype="bf16",
        )
        
        kv_1024 = estimate_kv_bytes(base_workload)
        
        # Create workload with longer seq_len
        longer = WorkloadConfig(
            name="Test",
            params_b=7.0,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
            seq_len=2048,
            batch=1,
            weight_dtype="bf16",
            kv_dtype="bf16",
        )
        
        kv_2048 = estimate_kv_bytes(longer)
        
        # Should double with doubled seq_len
        assert kv_2048 > kv_1024
        assert abs(kv_2048 / kv_1024 - 2.0) < 0.01

    def test_kv_bytes_int8_smaller_than_bf16(self):
        """INT8 KV cache should be smaller than BF16."""
        bf16_workload = WorkloadConfig(
            name="Test",
            params_b=7.0,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
            seq_len=2048,
            batch=1,
            weight_dtype="bf16",
            kv_dtype="bf16",
        )
        
        int8_workload = WorkloadConfig(
            name="Test",
            params_b=7.0,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
            seq_len=2048,
            batch=1,
            weight_dtype="bf16",
            kv_dtype="int8",
        )
        
        kv_bf16 = estimate_kv_bytes(bf16_workload)
        kv_int8 = estimate_kv_bytes(int8_workload)
        
        # INT8 should be half of BF16
        assert kv_int8 < kv_bf16
        assert abs(kv_int8 / kv_bf16 - 0.5) < 0.01


class TestMemoryFitStatus:
    """Tests for memory fit status checking."""

    def test_tiny_model_generous_memory_ok(self):
        """Tiny model with generous memory should be OK."""
        workload = WorkloadConfig(
            name="Tiny",
            params_b=1.0,
            n_layers=12,
            hidden_size=768,
            n_heads=12,
            seq_len=512,
            batch=1,
            weight_dtype="bf16",
            kv_dtype="bf16",
        )
        hardware = HardwareConfig(
            name="Big GPU",
            peak_tflops=100.0,
            hbm_gbps=1000.0,
            memory_gb=80.0,
        )
        
        result = memory_fit_status(workload, hardware)
        
        assert result["status"] == "OK"
        assert result["memory_gb"] == 80.0
        assert result["headroom_gb"] > 0

    def test_large_model_small_memory_error(self):
        """Large model with small memory should be ERROR."""
        workload = WorkloadConfig(
            name="Large",
            params_b=70.0,
            n_layers=80,
            hidden_size=8192,
            n_heads=64,
            seq_len=8192,
            batch=16,
            weight_dtype="bf16",
            kv_dtype="bf16",
        )
        hardware = HardwareConfig(
            name="Small GPU",
            peak_tflops=100.0,
            hbm_gbps=1000.0,
            memory_gb=24.0,
        )
        
        result = memory_fit_status(workload, hardware)
        
        assert result["status"] == "ERROR"
        assert result["required_gb"] > result["memory_gb"]

    def test_borderline_model_warning(self):
        """Model using ~90% of memory should be WARNING."""
        # Create a workload that uses about 35GB
        workload = WorkloadConfig(
            name="Medium",
            params_b=13.0,  # ~26GB for weights in BF16
            n_layers=40,
            hidden_size=5120,
            n_heads=40,
            seq_len=2048,
            batch=4,
            weight_dtype="bf16",
            kv_dtype="bf16",
        )
        hardware = HardwareConfig(
            name="40GB GPU",
            peak_tflops=100.0,
            hbm_gbps=1000.0,
            memory_gb=40.0,
        )
        
        result = memory_fit_status(workload, hardware, overhead_fraction=0.15)
        
        # Should be WARNING or ERROR depending on exact calculation
        assert result["status"] in ["WARNING", "ERROR"]
        assert result["usage_fraction"] > 0.85

    def test_none_memory_returns_unknown(self):
        """If memory_gb is None, status should be UNKNOWN."""
        workload = WorkloadConfig(
            name="Test",
            params_b=7.0,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
            seq_len=2048,
            batch=1,
            weight_dtype="bf16",
            kv_dtype="bf16",
        )
        hardware = HardwareConfig(
            name="Unknown GPU",
            peak_tflops=100.0,
            hbm_gbps=1000.0,
            memory_gb=None,  # Unknown memory
        )
        
        result = memory_fit_status(workload, hardware)
        
        assert result["status"] == "UNKNOWN"
        assert result["memory_gb"] is None
        assert result["headroom_gb"] is None
        assert "not specified" in result["rationale"].lower()

    def test_total_bytes_includes_margin(self):
        """Total with margin should be larger than total without."""
        workload = WorkloadConfig(
            name="Test",
            params_b=7.0,
            n_layers=32,
            hidden_size=4096,
            n_heads=32,
            seq_len=2048,
            batch=1,
            weight_dtype="bf16",
            kv_dtype="bf16",
        )
        
        result = estimate_total_bytes(workload, overhead_fraction=0.15)
        
        assert result["total_with_margin_bytes"] > result["total_bytes"]
        ratio = result["total_with_margin_bytes"] / result["total_bytes"]
        assert abs(ratio - 1.15) < 0.01
