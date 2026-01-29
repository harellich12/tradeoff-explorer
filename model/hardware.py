"""
Hardware configuration and preset management.

Provides hardware archetypes and utility functions for loading presets.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import HardwareConfig


# Default hardware presets (used if JSON file not found)
DEFAULT_HARDWARE_PRESETS: List[Dict] = [
    {
        "name": "NVIDIA A100 80GB SXM",
        "peak_flops": 312.0,
        "memory_bandwidth": 2.0,
        "memory_capacity": 80.0,
        "description": "High-end datacenter GPU for AI training and inference",
    },
    {
        "name": "NVIDIA H100 SXM",
        "peak_flops": 989.0,
        "memory_bandwidth": 3.35,
        "memory_capacity": 80.0,
        "description": "Latest-gen datacenter GPU with Transformer Engine",
    },
    {
        "name": "NVIDIA RTX 4090",
        "peak_flops": 82.6,
        "memory_bandwidth": 1.0,
        "memory_capacity": 24.0,
        "description": "Consumer flagship GPU for local inference",
    },
]


def load_hardware_presets(filepath: Optional[Path] = None) -> List[HardwareConfig]:
    """
    Load hardware presets from JSON file or use defaults.

    Args:
        filepath: Path to hardware_presets.json. If None, uses default location.

    Returns:
        List of HardwareConfig instances
    """
    if filepath is None:
        # Default path relative to this module
        filepath = Path(__file__).parent.parent / "data" / "hardware_presets.json"

    presets = []

    try:
        if filepath.exists():
            with open(filepath, "r") as f:
                data = json.load(f)
                presets_data = data.get("presets", data)
        else:
            presets_data = DEFAULT_HARDWARE_PRESETS
    except (json.JSONDecodeError, IOError):
        presets_data = DEFAULT_HARDWARE_PRESETS

    for preset in presets_data:
        presets.append(
            HardwareConfig(
                name=preset["name"],
                peak_flops=preset["peak_flops"],
                memory_bandwidth=preset["memory_bandwidth"],
                memory_capacity=preset["memory_capacity"],
                description=preset.get("description"),
            )
        )

    return presets


def get_hardware_by_name(name: str, presets: Optional[List[HardwareConfig]] = None) -> Optional[HardwareConfig]:
    """
    Find a hardware preset by name.

    Args:
        name: Hardware preset name to search for
        presets: List of presets to search. If None, loads from file.

    Returns:
        HardwareConfig if found, None otherwise
    """
    if presets is None:
        presets = load_hardware_presets()

    for preset in presets:
        if preset.name.lower() == name.lower():
            return preset

    return None
