"""
Run logging module.

Appends JSON lines to logs/runs.jsonl with timestamp and app version.
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


APP_VERSION = "0.5.0"


def get_git_hash() -> Optional[str]:
    """
    Get the current git commit hash if available.
    
    Returns:
        Short git hash or None if not in a git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return None


def log_run(
    payload: Dict[str, Any],
    path: str = "logs/runs.jsonl",
) -> None:
    """
    Append a run log entry to the JSONL file.
    
    Args:
        payload: Dictionary with run inputs and outputs
        path: Path to the log file (relative to app root)
    """
    # Ensure logs directory exists
    log_path = Path(__file__).parent.parent / path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build the log entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "app_version": APP_VERSION,
        "git_hash": get_git_hash(),
        **payload,
    }
    
    # Append to file
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def format_workload_for_log(workload) -> Dict[str, Any]:
    """Format WorkloadConfig for logging."""
    return {
        "name": workload.name,
        "params_b": workload.params_b,
        "n_layers": workload.n_layers,
        "hidden_size": workload.hidden_size,
        "n_heads": workload.n_heads,
        "seq_len": workload.seq_len,
        "batch": workload.batch,
        "weight_dtype": workload.weight_dtype,
        "kv_dtype": workload.kv_dtype,
        "kv_compression": workload.kv_compression,
    }


def format_hardware_for_log(hardware) -> Dict[str, Any]:
    """Format HardwareConfig for logging."""
    return {
        "name": hardware.name,
        "peak_tflops": hardware.peak_tflops,
        "hbm_gbps": hardware.hbm_gbps,
        "sram_gb": hardware.sram_gb,
        "interconnect_gbps": hardware.interconnect_gbps,
        "cost_per_hour": hardware.cost_per_hour,
    }


def format_assumptions_for_log(assumptions) -> Dict[str, Any]:
    """Format Assumptions for logging."""
    return {
        "compute_util": assumptions.compute_util,
        "bw_util": assumptions.bw_util,
        "overlap_factor": assumptions.overlap_factor,
        "weight_cache_hit_rate": assumptions.weight_cache_hit_rate,
    }


def format_analysis_for_log(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Format analysis result for logging."""
    return {
        "tokens_s_final": analysis.get("tokens_s_final"),
        "tokens_s_compute": analysis.get("tokens_s_compute"),
        "tokens_s_bw": analysis.get("tokens_s_bw"),
        "bottleneck_class": analysis.get("bottleneck_class"),
        "required_tflops": analysis.get("required_tflops"),
        "required_gbps": analysis.get("required_gbps"),
    }
