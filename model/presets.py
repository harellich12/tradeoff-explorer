"""
Preset loaders for workload and hardware configurations.

Loads presets from JSON files and returns validated dataclass instances.
"""

import json
from pathlib import Path
from typing import Optional

from model.schemas import WorkloadConfig, HardwareConfig


# Path to data directory (relative to this module)
DATA_DIR = Path(__file__).parent.parent / "data"


def load_workload_presets() -> list[WorkloadConfig]:
    """
    Load workload presets from JSON file.
    
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
        # Extract fields, ignoring extra keys like 'description'
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


def load_hardware_presets() -> list[HardwareConfig]:
    """
    Load hardware presets from JSON file.
    
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
        # Extract fields, ignoring extra keys like 'description'
        config = HardwareConfig(
            name=item["name"],
            peak_tflops=item["peak_tflops"],
            hbm_gbps=item["hbm_gbps"],
            sram_gb=item.get("sram_gb"),
            interconnect_gbps=item.get("interconnect_gbps"),
            cost_per_hour=item.get("cost_per_hour"),
        )
        presets.append(config)
    
    return presets


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
