"""
Preset loaders for workload and hardware configurations.

Loads presets from JSON files and returns validated dataclass instances.
Supports extended metadata fields for richer preset information.
"""

import json
from pathlib import Path
from typing import Optional, List, Any
from dataclasses import dataclass, field

from model.schemas import WorkloadConfig, HardwareConfig


# Path to data directory (relative to this module)
DATA_DIR = Path(__file__).parent.parent / "data"


# =============================================================================
# Extended Preset Dataclasses (with metadata)
# =============================================================================

@dataclass
class HardwarePresetMeta:
    """
    Hardware preset with extended metadata.
    
    Contains the core HardwareConfig plus optional metadata fields
    for display and sourcing purposes.
    """
    config: HardwareConfig
    vendor: Optional[str] = None
    category: Optional[str] = None  # GPU, TPU, NPU, Other
    memory_gb: Optional[float] = None
    peak_tflops_int8: Optional[float] = None
    peak_tflops_int4: Optional[float] = None
    sources: List[str] = field(default_factory=list)
    is_placeholder: bool = False
    notes: Optional[str] = None
    description: Optional[str] = None


@dataclass
class WorkloadPresetMeta:
    """
    Workload preset with extended metadata.
    
    Contains the core WorkloadConfig plus optional metadata fields
    for model family info and sourcing.
    """
    config: WorkloadConfig
    model_family: Optional[str] = None
    gqa: Optional[bool] = None  # Grouped Query Attention
    mqa: Optional[bool] = None  # Multi-Query Attention
    sources: List[str] = field(default_factory=list)
    is_placeholder: bool = False
    notes: Optional[str] = None
    description: Optional[str] = None


# =============================================================================
# Preset Loaders
# =============================================================================

def load_workload_presets() -> list[WorkloadConfig]:
    """
    Load workload presets from JSON file (backward compatible).
    
    Returns:
        List of WorkloadConfig instances
        
    Raises:
        FileNotFoundError: If preset file doesn't exist
        ValueError: If preset data is invalid
    """
    preset_file = DATA_DIR / "workload_presets.json"
    
    with open(preset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    presets = []
    for item in data.get("presets", []):
        # Skip placeholder entries that lack required fields
        if item.get("is_placeholder", False):
            # Check if we have the minimum required fields
            if item.get("n_layers") is None or item.get("hidden_size") is None:
                continue  # Skip incomplete placeholders
        
        # Extract fields, ignoring extra keys
        config = WorkloadConfig(
            name=item["name"],
            params_b=item["params_b"],
            n_layers=item["n_layers"],
            hidden_size=item["hidden_size"],
            n_heads=item["n_heads"],
            seq_len=item.get("seq_len", 2048),
            batch=item.get("batch", 1),
            weight_dtype=item.get("weight_dtype", "bf16"),
            kv_dtype=item.get("kv_dtype", "bf16"),
            kv_compression=item.get("kv_compression", 1.0),
        )
        presets.append(config)
    
    return presets


def load_workload_presets_with_meta() -> list[WorkloadPresetMeta]:
    """
    Load workload presets with full metadata.
    
    Returns:
        List of WorkloadPresetMeta instances (including placeholders)
    """
    preset_file = DATA_DIR / "workload_presets.json"
    
    with open(preset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    presets = []
    for item in data.get("presets", []):
        is_placeholder = item.get("is_placeholder", False)
        
        # For placeholders, create a minimal config or None
        if is_placeholder and (item.get("n_layers") is None or item.get("hidden_size") is None):
            # Create a dummy config for display purposes
            config = None
        else:
            config = WorkloadConfig(
                name=item["name"],
                params_b=item["params_b"],
                n_layers=item["n_layers"],
                hidden_size=item["hidden_size"],
                n_heads=item["n_heads"],
                seq_len=item.get("seq_len", 2048),
                batch=item.get("batch", 1),
                weight_dtype=item.get("weight_dtype", "bf16"),
                kv_dtype=item.get("kv_dtype", "bf16"),
                kv_compression=item.get("kv_compression", 1.0),
            )
        
        meta = WorkloadPresetMeta(
            config=config,
            model_family=item.get("model_family"),
            gqa=item.get("gqa"),
            mqa=item.get("mqa"),
            sources=item.get("sources", []),
            is_placeholder=is_placeholder,
            notes=item.get("notes"),
            description=item.get("description"),
        )
        presets.append(meta)
    
    return presets


def load_hardware_presets() -> list[HardwareConfig]:
    """
    Load hardware presets from JSON file (backward compatible).
    
    Returns:
        List of HardwareConfig instances
        
    Raises:
        FileNotFoundError: If preset file doesn't exist
        ValueError: If preset data is invalid
    """
    preset_file = DATA_DIR / "hardware_presets.json"
    
    with open(preset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    presets = []
    for item in data.get("presets", []):
        # Skip placeholder entries that lack required fields
        if item.get("is_placeholder", False):
            if item.get("peak_tflops") is None or item.get("hbm_gbps") is None:
                continue  # Skip incomplete placeholders
        
        # Extract fields, ignoring extra keys
        config = HardwareConfig(
            name=item["name"],
            peak_tflops=item["peak_tflops"],
            hbm_gbps=item["hbm_gbps"],
            memory_gb=item.get("memory_gb"),
            sram_gb=item.get("sram_gb"),
            interconnect_gbps=item.get("interconnect_gbps"),
            cost_per_hour=item.get("cost_per_hour"),
        )
        presets.append(config)
    
    return presets


def load_hardware_presets_with_meta() -> list[HardwarePresetMeta]:
    """
    Load hardware presets with full metadata.
    
    Returns:
        List of HardwarePresetMeta instances (including placeholders)
    """
    preset_file = DATA_DIR / "hardware_presets.json"
    
    with open(preset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    presets = []
    for item in data.get("presets", []):
        is_placeholder = item.get("is_placeholder", False)
        
        # For placeholders without required fields, skip config creation
        if is_placeholder and (item.get("peak_tflops") is None or item.get("hbm_gbps") is None):
            config = None
        else:
            config = HardwareConfig(
                name=item["name"],
                peak_tflops=item["peak_tflops"],
                hbm_gbps=item["hbm_gbps"],
                memory_gb=item.get("memory_gb"),
                sram_gb=item.get("sram_gb"),
                interconnect_gbps=item.get("interconnect_gbps"),
                cost_per_hour=item.get("cost_per_hour"),
            )
        
        meta = HardwarePresetMeta(
            config=config,
            vendor=item.get("vendor"),
            category=item.get("category"),
            memory_gb=item.get("memory_gb"),
            peak_tflops_int8=item.get("peak_tflops_int8"),
            peak_tflops_int4=item.get("peak_tflops_int4"),
            sources=item.get("sources", []),
            is_placeholder=is_placeholder,
            notes=item.get("notes"),
            description=item.get("description"),
        )
        presets.append(meta)
    
    return presets


# =============================================================================
# Utility Functions
# =============================================================================

def get_workload_preset_by_name(name: str) -> Optional[WorkloadConfig]:
    """
    Get a specific workload preset by name.
    
    Args:
        name: Preset name to find
        
    Returns:
        WorkloadConfig if found, None otherwise
    """
    presets = load_workload_presets()
    for preset in presets:
        if preset.name == name:
            return preset
    return None


def get_hardware_preset_by_name(name: str) -> Optional[HardwareConfig]:
    """
    Get a specific hardware preset by name.
    
    Args:
        name: Preset name to find
        
    Returns:
        HardwareConfig if found, None otherwise
    """
    presets = load_hardware_presets()
    for preset in presets:
        if preset.name == name:
            return preset
    return None


if __name__ == "__main__":
    # Test loading presets
    print("=== Workload Presets ===")
    for preset in load_workload_presets():
        print(f"  {preset.name}: {preset.params_b}B, {preset.n_layers}L, {preset.hidden_size}d")
    
    print("\n=== Hardware Presets ===")
    for preset in load_hardware_presets():
        print(f"  {preset.name}: {preset.peak_tflops} TFLOP/s, {preset.hbm_gbps} GB/s")
    
    print("\n=== Hardware Presets (with meta) ===")
    for meta in load_hardware_presets_with_meta():
        vendor = meta.vendor or "Unknown"
        sources = len(meta.sources)
        print(f"  {meta.config.name if meta.config else 'PLACEHOLDER'}: {vendor}, {sources} sources")
