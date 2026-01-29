"""
Data schemas and validation for tradeoff analysis.

Uses dataclasses for lightweight validation without heavy dependencies.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
from enum import Enum


# =============================================================================
# Data Type Definitions
# =============================================================================

class DType(str, Enum):
    """Supported data types with their byte sizes."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


# Byte sizes for each dtype
DTYPE_BYTES: dict[DType | str, float] = {
    DType.FP32: 4.0,
    DType.FP16: 2.0,
    DType.BF16: 2.0,
    DType.INT8: 1.0,
    DType.INT4: 0.5,
    # String keys for convenience
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


def get_dtype_bytes(dtype: DType | str) -> float:
    """
    Get the byte size for a given dtype.
    
    Args:
        dtype: Data type (DType enum or string)
        
    Returns:
        Bytes per element
        
    Raises:
        ValueError: If dtype is not recognized
    """
    if dtype in DTYPE_BYTES:
        return DTYPE_BYTES[dtype]
    
    # Try lowercase string conversion
    dtype_str = str(dtype).lower()
    if dtype_str in DTYPE_BYTES:
        return DTYPE_BYTES[dtype_str]
    
    raise ValueError(f"Unknown dtype: {dtype}. Valid options: {list(DType)}")


# =============================================================================
# Validation Helpers
# =============================================================================

class ValidationError(ValueError):
    """Raised when dataclass validation fails."""
    pass


def validate_positive(value: float, name: str) -> None:
    """Validate that a value is positive (> 0)."""
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


def validate_non_negative(value: float, name: str) -> None:
    """Validate that a value is non-negative (>= 0)."""
    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")


def validate_range(value: float, name: str, min_val: float, max_val: float) -> None:
    """Validate that a value is within a range."""
    if not (min_val <= value <= max_val):
        raise ValidationError(f"{name} must be in [{min_val}, {max_val}], got {value}")


# =============================================================================
# Core Configuration Schemas
# =============================================================================

@dataclass
class WorkloadConfig:
    """
    Configuration for an LLM workload.
    
    Attributes:
        name: Human-readable name for this workload configuration
        params_b: Model parameters in billions
        n_layers: Number of transformer layers
        hidden_size: Hidden dimension size
        n_heads: Number of attention heads
        seq_len: Sequence length (context window)
        batch: Batch size
        weight_dtype: Data type for model weights
        kv_dtype: Data type for KV cache
        kv_compression: KV cache compression ratio (1.0 = no compression)
    """
    name: str
    params_b: float  # Parameters in billions
    n_layers: int
    hidden_size: int
    n_heads: int
    seq_len: int = 2048
    batch: int = 1
    weight_dtype: str = "bf16"
    kv_dtype: str = "bf16"
    kv_compression: float = 1.0  # 1.0 = no compression
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        validate_positive(self.params_b, "params_b")
        validate_positive(self.n_layers, "n_layers")
        validate_positive(self.hidden_size, "hidden_size")
        validate_positive(self.n_heads, "n_heads")
        validate_positive(self.seq_len, "seq_len")
        validate_positive(self.batch, "batch")
        validate_range(self.kv_compression, "kv_compression", 0.0, 1.0)
        
        # Validate dtypes
        if self.weight_dtype.lower() not in DTYPE_BYTES:
            raise ValidationError(f"Invalid weight_dtype: {self.weight_dtype}")
        if self.kv_dtype.lower() not in DTYPE_BYTES:
            raise ValidationError(f"Invalid kv_dtype: {self.kv_dtype}")
        
        # Sensible minimum checks
        if self.hidden_size < 64:
            raise ValidationError(f"hidden_size should be at least 64, got {self.hidden_size}")
        if self.hidden_size % self.n_heads != 0:
            raise ValidationError(
                f"hidden_size ({self.hidden_size}) must be divisible by n_heads ({self.n_heads})"
            )

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_size // self.n_heads

    def weight_bytes_per_param(self) -> float:
        """Return bytes per weight parameter based on weight_dtype."""
        return get_dtype_bytes(self.weight_dtype)
    
    def kv_bytes_per_element(self) -> float:
        """Return bytes per KV cache element based on kv_dtype and compression."""
        return get_dtype_bytes(self.kv_dtype) * self.kv_compression


@dataclass
class HardwareConfig:
    """
    Configuration for a hardware platform.
    
    Attributes:
        name: Human-readable name for this hardware
        peak_tflops: Peak compute performance in TFLOP/s (FP16/BF16)
        hbm_gbps: HBM memory bandwidth in GB/s
        memory_gb: Total device memory in GB (optional, for memory fit check)
        sram_gb: On-chip SRAM/cache in GB (optional)
        interconnect_gbps: Interconnect bandwidth in GB/s (optional, for multi-GPU)
        cost_per_hour: Cost per hour in USD (optional)
    """
    name: str
    peak_tflops: float  # TFLOP/s
    hbm_gbps: float  # GB/s
    memory_gb: Optional[float] = None  # GB (total device memory)
    sram_gb: Optional[float] = None  # GB
    interconnect_gbps: Optional[float] = None  # GB/s
    cost_per_hour: Optional[float] = None  # USD
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        validate_positive(self.peak_tflops, "peak_tflops")
        validate_positive(self.hbm_gbps, "hbm_gbps")
        
        if self.memory_gb is not None:
            validate_positive(self.memory_gb, "memory_gb")
        if self.sram_gb is not None:
            validate_non_negative(self.sram_gb, "sram_gb")
        if self.interconnect_gbps is not None:
            validate_non_negative(self.interconnect_gbps, "interconnect_gbps")
        if self.cost_per_hour is not None:
            validate_non_negative(self.cost_per_hour, "cost_per_hour")
        
        # Sensible minimum checks
        if self.peak_tflops < 1:
            raise ValidationError(f"peak_tflops should be at least 1, got {self.peak_tflops}")
        if self.hbm_gbps < 100:
            raise ValidationError(f"hbm_gbps should be at least 100, got {self.hbm_gbps}")

    def arithmetic_intensity_ridge(self) -> float:
        """
        Calculate the ridge point (FLOP/byte) where compute and memory
        bounds intersect on the roofline.
        
        Returns:
            Ridge point in FLOP/byte
        """
        # peak_tflops is in TFLOP/s, hbm_gbps is in GB/s
        # Ridge = (TFLOP/s * 1e12) / (GB/s * 1e9) = TFLOP/s * 1000 / (GB/s)
        return (self.peak_tflops * 1000) / self.hbm_gbps


@dataclass
class Assumptions:
    """
    Efficiency and utilization assumptions for analysis.
    
    Attributes:
        compute_util: Fraction of peak compute utilized (0.0 to 1.0)
        bw_util: Fraction of peak memory bandwidth utilized (0.0 to 1.0)
        overlap_factor: Degree of compute/memory overlap (0.0 = no overlap, 1.0 = perfect)
        weight_cache_hit_rate: Fraction of weights cached on-chip (0.0 to 1.0)
    """
    compute_util: float = 0.7  # 70% compute utilization
    bw_util: float = 0.8  # 80% bandwidth utilization  
    overlap_factor: float = 0.0  # No overlap by default
    weight_cache_hit_rate: float = 0.0  # No weight caching by default
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        validate_range(self.compute_util, "compute_util", 0.0, 1.0)
        validate_range(self.bw_util, "bw_util", 0.0, 1.0)
        validate_range(self.overlap_factor, "overlap_factor", 0.0, 1.0)
        validate_range(self.weight_cache_hit_rate, "weight_cache_hit_rate", 0.0, 1.0)


# =============================================================================
# Analysis Result Schemas
# =============================================================================

@dataclass
class AnalysisResult:
    """Result of a roofline analysis."""
    workload: str
    hardware: str
    throughput_tokens_per_sec: float
    bottleneck: Literal["compute", "memory"]
    arithmetic_intensity: float
    utilization: float
    prefill_flops: float
    decode_flops_per_token: float
    memory_bytes_per_token: float


@dataclass
class SensitivityPoint:
    """A single point in sensitivity analysis."""
    parameter_name: str
    low_value: float
    base_value: float
    high_value: float
    low_throughput: float
    base_throughput: float
    high_throughput: float


# =============================================================================
# Example Usage
# =============================================================================

def example_instantiation() -> tuple[WorkloadConfig, HardwareConfig, Assumptions]:
    """
    Demonstrate example instantiation of all schemas.
    
    Returns:
        Tuple of (WorkloadConfig, HardwareConfig, Assumptions)
    """
    # Example: Llama-2 7B style configuration
    workload = WorkloadConfig(
        name="Llama-2-7B",
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
    
    # Example: NVIDIA H100 SXM
    hardware = HardwareConfig(
        name="H100-SXM",
        peak_tflops=989.0,
        hbm_gbps=3350.0,
        sram_gb=0.050,  # 50 MB L2 cache
        interconnect_gbps=900.0,  # NVLink
        cost_per_hour=3.50,
    )
    
    # Example: Conservative assumptions
    assumptions = Assumptions(
        compute_util=0.65,
        bw_util=0.75,
        overlap_factor=0.1,
        weight_cache_hit_rate=0.0,
    )
    
    return workload, hardware, assumptions


if __name__ == "__main__":
    # Demonstrate schema usage
    workload, hardware, assumptions = example_instantiation()
    
    print("=== WorkloadConfig ===")
    print(f"  Name: {workload.name}")
    print(f"  Parameters: {workload.params_b}B")
    print(f"  Layers: {workload.n_layers}")
    print(f"  Hidden size: {workload.hidden_size}")
    print(f"  Heads: {workload.n_heads} (dim={workload.head_dim})")
    print(f"  Sequence length: {workload.seq_len}")
    print(f"  Batch: {workload.batch}")
    print(f"  Weight dtype: {workload.weight_dtype} ({workload.weight_bytes_per_param()} bytes)")
    print(f"  KV dtype: {workload.kv_dtype} ({workload.kv_bytes_per_element()} bytes)")
    
    print("\n=== HardwareConfig ===")
    print(f"  Name: {hardware.name}")
    print(f"  Peak compute: {hardware.peak_tflops} TFLOP/s")
    print(f"  HBM bandwidth: {hardware.hbm_gbps} GB/s")
    print(f"  SRAM: {hardware.sram_gb} GB")
    print(f"  Ridge point: {hardware.arithmetic_intensity_ridge():.1f} FLOP/byte")
    print(f"  Cost: ${hardware.cost_per_hour}/hour")
    
    print("\n=== Assumptions ===")
    print(f"  Compute utilization: {assumptions.compute_util*100:.0f}%")
    print(f"  Bandwidth utilization: {assumptions.bw_util*100:.0f}%")
    print(f"  Overlap factor: {assumptions.overlap_factor}")
    print(f"  Weight cache hit rate: {assumptions.weight_cache_hit_rate}")
    
    print("\n=== DType Bytes ===")
    for dtype in DType:
        print(f"  {dtype.value}: {get_dtype_bytes(dtype)} bytes")
