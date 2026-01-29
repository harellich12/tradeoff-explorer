"""
Tradeoff Explorer - Streamlit Application Entry Point

A tool for exploring LLM inference performance tradeoffs across
different hardware configurations and model architectures.
"""

import streamlit as st
from dataclasses import replace

from model.schemas import WorkloadConfig, HardwareConfig, Assumptions, get_dtype_bytes
from model.workloads import (
    flops_per_token,
    bytes_per_token,
    bytes_per_token_prefill,
    arithmetic_intensity,
    get_flops_formula_markdown,
    get_bytes_formula_markdown,
)
from model.presets import load_workload_presets, load_hardware_presets, load_hardware_presets_with_meta, load_workload_presets_with_meta
from model.roofline import analyze_detailed
from model.sensitivity import compute_sensitivity
from model.logging import (
    log_run,
    format_workload_for_log,
    format_hardware_for_log,
    format_assumptions_for_log,
    format_analysis_for_log,
)
from model.memory_fit import memory_fit_status, format_memory_fit_for_log


# App version
APP_VERSION = "0.7.0"

# Default assumption values for reset
DEFAULT_COMPUTE_UTIL = 0.7
DEFAULT_BW_UTIL = 0.8
DEFAULT_OVERLAP_FACTOR = 0.0
DEFAULT_WEIGHT_CACHE_HIT_RATE = 0.0
DEFAULT_OVERHEAD_FRACTION = 0.15


def format_number(value: float, precision: int = 2) -> str:
    """Format large numbers with SI suffixes."""
    if value >= 1e15:
        return f"{value/1e15:.{precision}f}P"
    elif value >= 1e12:
        return f"{value/1e12:.{precision}f}T"
    elif value >= 1e9:
        return f"{value/1e9:.{precision}f}G"
    elif value >= 1e6:
        return f"{value/1e6:.{precision}f}M"
    elif value >= 1e3:
        return f"{value/1e3:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"


def create_default_workload() -> WorkloadConfig:
    """Create default workload configuration."""
    return WorkloadConfig(
        name="Llama-2-7B",
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


def create_default_assumptions() -> Assumptions:
    """Create default assumptions."""
    return Assumptions(
        compute_util=DEFAULT_COMPUTE_UTIL,
        bw_util=DEFAULT_BW_UTIL,
        overlap_factor=DEFAULT_OVERLAP_FACTOR,
        weight_cache_hit_rate=DEFAULT_WEIGHT_CACHE_HIT_RATE,
    )


def check_extrapolation_warnings(workload: WorkloadConfig, assumptions: Assumptions) -> list:
    """Check for extreme values that may cause extrapolation issues."""
    warnings = []
    
    # Check utilization factors
    if assumptions.compute_util < 0.3:
        warnings.append(f"⚠️ **Very low compute utilization ({assumptions.compute_util*100:.0f}%)** — Real systems rarely see <30% utilization")
    if assumptions.bw_util < 0.4:
        warnings.append(f"⚠️ **Very low bandwidth utilization ({assumptions.bw_util*100:.0f}%)** — May indicate suboptimal memory access patterns")
    
    # Check extreme sequence lengths
    if workload.seq_len >= 16384:
        warnings.append(f"⚠️ **Very long sequence ({workload.seq_len:,} tokens)** — KV cache may exceed GPU memory; consider chunked attention")
    if workload.seq_len < 256:
        warnings.append(f"⚠️ **Very short sequence ({workload.seq_len} tokens)** — May not reflect typical LLM usage patterns")
    
    # Check extreme batch sizes
    if workload.batch >= 32:
        warnings.append(f"⚠️ **Large batch size ({workload.batch})** — Memory constraints may limit actual batch size")
    
    # Check model size vs typical ranges
    if workload.params_b > 70:
        warnings.append(f"⚠️ **Very large model ({workload.params_b}B params)** — May require multi-GPU deployment")
    
    return warnings


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Tradeoff Explorer",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Title and description
    st.title("⚖️ LLM Hardware-Model Tradeoff Explorer")
    st.caption(
        "Analyze performance tradeoffs between LLM workloads and hardware using roofline modeling. "
        "All results are **upper-bound estimates under assumptions**."
    )

    # Load presets once
    workload_presets = load_workload_presets()
    hardware_presets = load_hardware_presets()
    hardware_presets_meta = load_hardware_presets_with_meta()
    
    # Build preset name lists
    workload_names = ["Custom"] + [p.name for p in workload_presets]
    hardware_names = [p.name for p in hardware_presets]
    
    # Build hardware metadata lookup
    hw_meta_lookup = {m.config.name: m for m in hardware_presets_meta if m.config}

    # =========================================================================
    # Sidebar Configuration
    # =========================================================================
    with st.sidebar:
        st.header("⚙️ Workload Configuration")
        
        # Workload preset selector
        selected_workload = st.selectbox(
            "📦 Model Preset",
            options=workload_names,
            index=1,
            help="Select a preset to auto-fill model architecture, or 'Custom' for manual entry"
        )
        
        # Get preset values
        if selected_workload == "Custom":
            preset = create_default_workload()
        else:
            preset = next((p for p in workload_presets if p.name == selected_workload), create_default_workload())
        
        # Reset to preset defaults button
        if selected_workload != "Custom":
            if st.button("🔄 Reset to Preset Defaults", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        
        st.divider()
        
        # Model Architecture section
        st.subheader("🏗️ Model Architecture")
        
        params_b = st.slider(
            "Parameters (billions)", 
            1.0, 100.0, 
            float(preset.params_b), 
            0.5,
            disabled=(selected_workload != "Custom"),
            help="Total model parameters in billions"
        )
        n_layers = st.slider(
            "Transformer Layers", 
            4, 128, 
            int(preset.n_layers), 
            2,
            disabled=(selected_workload != "Custom")
        )
        
        hidden_options = [1024, 2048, 4096, 5120, 6144, 8192, 12288, 16384]
        hidden_idx = hidden_options.index(preset.hidden_size) if preset.hidden_size in hidden_options else 2
        hidden_size = st.select_slider(
            "Hidden Dimension",
            options=hidden_options,
            value=hidden_options[hidden_idx],
            disabled=(selected_workload != "Custom")
        )
        
        head_options = [8, 16, 32, 40, 48, 64, 80, 96, 128]
        head_idx = head_options.index(preset.n_heads) if preset.n_heads in head_options else 2
        n_heads = st.select_slider(
            "Attention Heads",
            options=head_options,
            value=head_options[head_idx],
            disabled=(selected_workload != "Custom")
        )
        
        st.divider()
        
        # Inference Settings section
        st.subheader("📊 Inference Settings")
        
        seq_options = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        seq_idx = seq_options.index(preset.seq_len) if preset.seq_len in seq_options else 3
        seq_len = st.select_slider(
            "Context Length (tokens)",
            options=seq_options,
            value=seq_options[seq_idx],
            help="Sequence length for KV cache sizing"
        )
        batch = st.slider(
            "Batch Size", 
            1, 64, 
            int(preset.batch),
            help="Number of concurrent sequences"
        )
        
        st.divider()
        
        # Precision Settings section
        st.subheader("🎯 Precision Settings")
        
        dtype_options = ["bf16", "fp16", "int8", "int4"]
        weight_dtype = st.selectbox(
            "Weight Precision",
            options=dtype_options,
            index=dtype_options.index(preset.weight_dtype) if preset.weight_dtype in dtype_options else 0,
            help="Data type for model weights"
        )
        kv_dtype = st.selectbox(
            "KV Cache Precision",
            options=dtype_options,
            index=dtype_options.index(preset.kv_dtype) if preset.kv_dtype in dtype_options else 0,
            help="Data type for key-value cache"
        )
        kv_compression = st.slider(
            "KV Compression Factor", 
            0.1, 1.0, 
            float(preset.kv_compression), 
            0.05,
            help="1.0 = no compression, lower = more aggressive compression"
        )
        
        st.divider()
        
        # Assumptions section with expander
        with st.expander("🔧 Efficiency Assumptions", expanded=False):
            st.caption("Adjust these to model real-world inefficiencies")
            
            compute_util = st.slider(
                "Compute Utilization", 
                0.1, 1.0, 
                DEFAULT_COMPUTE_UTIL, 
                0.05,
                help="Fraction of peak FLOP/s actually achieved (typical: 60-80%)"
            )
            bw_util = st.slider(
                "Bandwidth Utilization", 
                0.1, 1.0, 
                DEFAULT_BW_UTIL, 
                0.05,
                help="Fraction of peak memory bandwidth achieved (typical: 70-90%)"
            )
            overlap_factor = st.slider(
                "Compute-Memory Overlap", 
                0.0, 0.5, 
                DEFAULT_OVERLAP_FACTOR, 
                0.05,
                help="Degree of overlap between compute and memory ops (0 = none)"
            )
            weight_cache_hit_rate = st.slider(
                "Weight Cache Hit Rate", 
                0.0, 1.0, 
                DEFAULT_WEIGHT_CACHE_HIT_RATE, 
                0.05,
                help="Fraction of weights cached in faster memory (e.g., SRAM)"
            )
            overhead_fraction = st.slider(
                "Memory Overhead %",
                0.05, 0.50,
                DEFAULT_OVERHEAD_FRACTION,
                0.05,
                format="%.0f%%",
                help="Additional memory overhead for activations, framework, etc."
            )
        
        st.divider()
        st.caption(f"v{APP_VERSION}")

    # =========================================================================
    # Build Configuration Objects
    # =========================================================================
    try:
        if selected_workload == "Custom":
            workload = WorkloadConfig(
                name="Custom",
                params_b=params_b,
                n_layers=n_layers,
                hidden_size=hidden_size,
                n_heads=n_heads,
                seq_len=seq_len,
                batch=batch,
                weight_dtype=weight_dtype,
                kv_dtype=kv_dtype,
                kv_compression=kv_compression,
            )
        else:
            workload = WorkloadConfig(
                name=preset.name,
                params_b=preset.params_b,
                n_layers=preset.n_layers,
                hidden_size=preset.hidden_size,
                n_heads=preset.n_heads,
                seq_len=seq_len,
                batch=batch,
                weight_dtype=weight_dtype,
                kv_dtype=kv_dtype,
                kv_compression=kv_compression,
            )
        assumptions = Assumptions(
            compute_util=compute_util,
            bw_util=bw_util,
            overlap_factor=overlap_factor,
            weight_cache_hit_rate=weight_cache_hit_rate
        )
        config_valid = True
    except Exception as e:
        st.error(f"Configuration Error: {e}")
        workload = create_default_workload()
        assumptions = create_default_assumptions()
        config_valid = False

    # =========================================================================
    # Check for Extrapolation Warnings
    # =========================================================================
    extrapolation_warnings = check_extrapolation_warnings(workload, assumptions)
    if extrapolation_warnings:
        with st.expander("⚠️ Extrapolation Warnings", expanded=True):
            for warning in extrapolation_warnings:
                st.markdown(warning)

    # =========================================================================
    # Workload Summary Panel
    # =========================================================================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Workload Summary")
        
        if config_valid:
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| **Model** | {workload.params_b}B parameters |
| **Architecture** | {workload.n_layers}L × {workload.hidden_size}d × {workload.n_heads}h |
| **Context** | {workload.seq_len:,} tokens |
| **Batch** | {workload.batch} sequences |
| **Weights** | {workload.weight_dtype.upper()} ({get_dtype_bytes(workload.weight_dtype)} bytes) |
| **KV Cache** | {workload.kv_dtype.upper()} ({workload.kv_bytes_per_element():.1f} bytes) |
            """)
        else:
            st.warning("Invalid configuration. Using defaults.")

    with col2:
        with st.expander("📐 Current Assumptions", expanded=True):
            st.markdown(f"""
| Assumption | Value | Notes |
|------------|-------|-------|
| **Compute Util** | {assumptions.compute_util*100:.0f}% | Fraction of peak FLOP/s |
| **Bandwidth Util** | {assumptions.bw_util*100:.0f}% | Fraction of peak GB/s |
| **KV Compression** | {workload.kv_compression*100:.0f}% | Compression factor |
| **Weight Cache** | {assumptions.weight_cache_hit_rate*100:.0f}% | SRAM cache hits |
            """)

    st.divider()

    # =========================================================================
    # Per-Token Metrics
    # =========================================================================
    prefill_flops = flops_per_token(workload, "prefill")
    prefill_bytes = bytes_per_token_prefill(workload, assumptions)
    decode_flops = flops_per_token(workload, "decode")
    decode_bytes = bytes_per_token(workload, assumptions)

    st.subheader("📈 Per-Token Compute & Memory")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            label="Prefill FLOPs/token",
            value=f"{format_number(prefill_flops)}",
        )
        st.caption("Compute per token")

    with metric_col2:
        st.metric(
            label="Prefill Bytes/token",
            value=f"{format_number(prefill_bytes)}B",
        )
        st.caption("Memory per token")

    with metric_col3:
        st.metric(
            label="Decode FLOPs/token",
            value=f"{format_number(decode_flops)}",
        )
        st.caption("Compute per token")

    with metric_col4:
        st.metric(
            label="Decode Bytes/token",
            value=f"{format_number(decode_bytes)}B",
        )
        st.caption("Memory per token")

    # Arithmetic intensity
    ai_col1, ai_col2 = st.columns(2)
    
    with ai_col1:
        prefill_ai = prefill_flops / prefill_bytes if prefill_bytes > 0 else float('inf')
        st.metric(
            label="Prefill Arithmetic Intensity",
            value=f"{prefill_ai:.1f} FLOP/byte",
        )
        st.caption("Higher = more compute-bound")
    
    with ai_col2:
        decode_ai = decode_flops / decode_bytes if decode_bytes > 0 else float('inf')
        st.metric(
            label="Decode Arithmetic Intensity",
            value=f"{decode_ai:.1f} FLOP/byte",
        )
        st.caption("Lower = more memory-bound")

    st.divider()

    # =========================================================================
    # Hardware Comparison Section
    # =========================================================================
    st.subheader("🖥️ Hardware Comparison")
    
    num_configs = st.slider(
        "Number of hardware configurations",
        min_value=2,
        max_value=4,
        value=2,
        key="num_hw_configs",
        help="Compare 2-4 hardware configurations side by side"
    )
    
    # Create columns for hardware config selection
    hw_columns = st.columns(num_configs)
    edited_hw_configs = []
    
    # Callback to update input fields when preset changes
    def make_hw_preset_callback(idx, presets, preset_names):
        def callback():
            selected = st.session_state[f"hw_preset_{idx}"]
            config = next((p for p in presets if p.name == selected), presets[0])
            st.session_state[f"hw_name_{idx}"] = config.name
            st.session_state[f"hw_tflops_{idx}"] = float(config.peak_tflops)
            st.session_state[f"hw_hbm_{idx}"] = float(config.hbm_gbps)
            st.session_state[f"hw_memory_{idx}"] = float(config.memory_gb) if config.memory_gb else 0.0
            st.session_state[f"hw_cost_{idx}"] = float(config.cost_per_hour) if config.cost_per_hour else 0.0
        return callback
    
    for i, col in enumerate(hw_columns):
        config_idx = i + 1
        with col:
            st.markdown(f"#### Hardware #{config_idx}")
            
            default_idx = min(i, len(hardware_names) - 1)
            
            # Initialize session state for this config if needed
            init_key = f"hw_init_{config_idx}"
            if init_key not in st.session_state:
                init_config = hardware_presets[default_idx] if default_idx < len(hardware_presets) else hardware_presets[0]
                st.session_state[f"hw_preset_{config_idx}"] = init_config.name
                st.session_state[f"hw_name_{config_idx}"] = init_config.name
                st.session_state[f"hw_tflops_{config_idx}"] = float(init_config.peak_tflops)
                st.session_state[f"hw_hbm_{config_idx}"] = float(init_config.hbm_gbps)
                st.session_state[f"hw_memory_{config_idx}"] = float(init_config.memory_gb) if init_config.memory_gb else 0.0
                st.session_state[f"hw_cost_{config_idx}"] = float(init_config.cost_per_hour) if init_config.cost_per_hour else 0.0
                st.session_state[init_key] = True
            
            selected_preset = st.selectbox(
                f"Preset",
                options=hardware_names,
                key=f"hw_preset_{config_idx}",
                label_visibility="collapsed",
                on_change=make_hw_preset_callback(config_idx, hardware_presets, hardware_names)
            )
            
            # Display vendor/memory info from metadata
            hw_meta = hw_meta_lookup.get(selected_preset)
            if hw_meta:
                info_parts = []
                if hw_meta.vendor:
                    info_parts.append(hw_meta.vendor)
                if hw_meta.category:
                    info_parts.append(hw_meta.category)
                if hw_meta.memory_gb:
                    info_parts.append(f"{hw_meta.memory_gb:.0f}GB")
                if info_parts:
                    st.caption(" · ".join(info_parts))
            
            edited_name = st.text_input(
                "Name",
                key=f"hw_name_{config_idx}",
                label_visibility="collapsed"
            )
            
            edited_tflops = st.number_input(
                "Peak TFLOP/s (BF16)",
                min_value=1.0,
                max_value=10000.0,
                step=10.0,
                key=f"hw_tflops_{config_idx}"
            )
            
            edited_hbm_gbps = st.number_input(
                "Memory Bandwidth (GB/s)",
                min_value=100.0,
                max_value=10000.0,
                step=50.0,
                key=f"hw_hbm_{config_idx}"
            )
            
            edited_memory_gb = st.number_input(
                "Memory (GB)",
                min_value=0.0,
                max_value=1000.0,
                step=1.0,
                key=f"hw_memory_{config_idx}",
                help="Total device memory for memory fit check (0 = unknown)"
            )
            
            edited_cost = st.number_input(
                "Cost ($/hour)",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                key=f"hw_cost_{config_idx}"
            )
            
            # Get base config for sram and interconnect
            base_config = next((p for p in hardware_presets if p.name == selected_preset), hardware_presets[0])
            
            edited_config = HardwareConfig(
                name=edited_name,
                peak_tflops=edited_tflops,
                hbm_gbps=edited_hbm_gbps,
                memory_gb=edited_memory_gb if edited_memory_gb > 0 else None,
                sram_gb=base_config.sram_gb,
                interconnect_gbps=base_config.interconnect_gbps,
                cost_per_hour=edited_cost if edited_cost > 0 else None
            )
            edited_hw_configs.append(edited_config)
    
    st.divider()
    
    # Run Analysis button
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        st.session_state.run_analysis = True
    
    # Show results if analysis has been run
    if st.session_state.get("run_analysis", False):
        analyses = []
        for hw_config in edited_hw_configs:
            analysis = analyze_detailed(workload, hw_config, assumptions, "decode")
            analyses.append(analysis)
        
        # Compute memory fit results for logging
        memory_fits = [memory_fit_status(workload, hw, overhead_fraction) for hw in edited_hw_configs]
        
        # Log the run
        log_run({
            "workload": format_workload_for_log(workload),
            "assumptions": format_assumptions_for_log(assumptions),
            "overhead_fraction": overhead_fraction,
            "hardware_configs": [format_hardware_for_log(hw) for hw in edited_hw_configs],
            "results": [format_analysis_for_log(a) for a in analyses],
            "memory_fits": [format_memory_fit_for_log(mf) for mf in memory_fits],
        })
        
        st.divider()
        st.markdown("### 📊 Comparison Results")
        
        # Build comparison table
        table_data = {
            "Configuration": [],
            "Peak TFLOP/s": [],
            "HBM GB/s": [],
            "Bottleneck": [],
            "Throughput (tok/s)": [],
            "Required TFLOP/s": [],
            "Available TFLOP/s": [],
            "Required GB/s": [],
            "Available GB/s": [],
        }
        
        for hw_config, analysis in zip(edited_hw_configs, analyses):
            table_data["Configuration"].append(hw_config.name)
            table_data["Peak TFLOP/s"].append(f"{hw_config.peak_tflops:.0f}")
            table_data["HBM GB/s"].append(f"{hw_config.hbm_gbps:.0f}")
            table_data["Bottleneck"].append(analysis["bottleneck_class"])
            table_data["Throughput (tok/s)"].append(f"{analysis['tokens_s_final']:,.0f}")
            table_data["Required TFLOP/s"].append(f"{analysis['required_tflops']:.1f}")
            table_data["Available TFLOP/s"].append(f"{hw_config.peak_tflops * assumptions.compute_util:.1f}")
            table_data["Required GB/s"].append(f"{analysis['required_gbps']:.1f}")
            table_data["Available GB/s"].append(f"{hw_config.hbm_gbps * assumptions.bw_util:.1f}")
        
        import pandas as pd
        df = pd.DataFrame(table_data)
        
        def highlight_bottleneck(val):
            if val == "COMPUTE":
                return "background-color: #ffcccb; color: black"
            elif val == "MEMORY_BW":
                return "background-color: #ffffcc; color: black"
            return ""
        
        styled_df = df.style.applymap(highlight_bottleneck, subset=["Bottleneck"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Find winner
        throughputs = [a["tokens_s_final"] for a in analyses]
        max_throughput = max(throughputs)
        winner_idx = throughputs.index(max_throughput)
        winner_config = edited_hw_configs[winner_idx]
        
        speedup_msgs = []
        for i, (hw, analysis) in enumerate(zip(edited_hw_configs, analyses)):
            if i != winner_idx:
                speedup = max_throughput / analysis["tokens_s_final"] if analysis["tokens_s_final"] > 0 else float('inf')
                speedup_msgs.append(f"{speedup:.2f}× vs {hw.name}")
        
        st.success(f"🏆 **{winner_config.name}** achieves highest throughput: **{format_number(max_throughput)} tok/s** ({', '.join(speedup_msgs)})")
        
        # Detailed cards
        st.markdown("### 📋 Detailed Analysis")
        
        detail_cols = st.columns(num_configs)
        for i, (col, hw_config, analysis) in enumerate(zip(detail_cols, edited_hw_configs, analyses)):
            with col:
                is_winner = (i == winner_idx)
                header = f"{'🏆 ' if is_winner else ''}{hw_config.name}"
                st.markdown(f"#### {header}")
                
                bn = analysis["bottleneck_class"]
                if bn == "COMPUTE":
                    st.error(f"💻 **{bn}**")
                else:
                    st.warning(f"💾 **{bn}**")
                
                st.metric(
                    label="Throughput",
                    value=f"{format_number(analysis['tokens_s_final'])} tok/s",
                )
                
                with st.expander("📊 Metrics"):
                    st.markdown(f"""
| Metric | Value |
|--------|-------|
| Compute-limited | {format_number(analysis['tokens_s_compute'])} tok/s |
| BW-limited | {format_number(analysis['tokens_s_bw'])} tok/s |
| Required TFLOP/s | {analysis['required_tflops']:.1f} |
| Required GB/s | {analysis['required_gbps']:.1f} |
| Compute Util | {analysis['compute_util_est']*100:.0f}% |
| BW Util | {analysis['bw_util_est']*100:.0f}% |
                    """)
                
                # Memory Fit section
                fit_result = memory_fit_status(workload, hw_config, overhead_fraction)
                with st.expander("💾 Memory Fit", expanded=fit_result["status"] in ["WARNING", "ERROR"]):
                    status = fit_result["status"]
                    if status == "OK":
                        st.success(f"✅ **{status}** — {fit_result['rationale']}")
                    elif status == "WARNING":
                        st.warning(f"⚡ **{status}** — {fit_result['rationale']}")
                    elif status == "ERROR":
                        st.error(f"⚠️ **{status}** — {fit_result['rationale']}")
                    else:
                        st.info(f"❓ **{status}** — {fit_result['rationale']}")
                    
                    st.markdown(f"""
| Component | Size |
|-----------|------|
| Weights | {fit_result['weights_gb']:.2f} GB |
| KV Cache | {fit_result['kv_gb']:.2f} GB |
| Overhead | {overhead_fraction*100:.0f}% |
| **Total Required** | **{fit_result['required_gb']:.2f} GB** |
                    """)
                    
                    if fit_result["memory_gb"]:
                        st.markdown(f"**Available:** {fit_result['memory_gb']:.0f} GB • **Headroom:** {fit_result['headroom_gb']:.2f} GB")
                    
                    if status in ["WARNING", "ERROR"]:
                        st.caption("💡 **OOM levers:** reduce `seq_len`, reduce `batch`, or quantize/compress KV cache.")
                
                with st.expander("🎯 Top Levers"):
                    levers = compute_sensitivity(workload, hw_config, assumptions, "decode")
                    for idx, lever in enumerate(levers, 1):
                        direction = "↑" if lever.high_throughput > lever.low_throughput else "↓"
                        st.markdown(f"**{idx}. {lever.param_display_name}** ({lever.percent_impact:.1f}% {direction})")
                        st.caption(lever.explanation)
                
                with st.expander("📝 Explanation"):
                    st.markdown(analysis["explanation"])

    st.divider()

    # =========================================================================
    # Formula and Documentation Tabs
    # =========================================================================
    tab1, tab2, tab3 = st.tabs(["📐 Formula Details", "📈 Sensitivity Analysis", "📄 Memo Export"])

    with tab1:
        formula_col1, formula_col2 = st.columns(2)
        
        with formula_col1:
            with st.expander("FLOPs Formula (Decode)", expanded=False):
                st.markdown(get_flops_formula_markdown("decode"))
        
        with formula_col2:
            with st.expander("Bytes Formula (Decode)", expanded=False):
                st.markdown(get_bytes_formula_markdown("decode"))
        
        with st.expander("🔍 Understanding the Roofline Model", expanded=False):
            st.markdown("""
**The Roofline Model** provides an upper bound on achievable performance:

1. **Compute-Bound**: When arithmetic intensity is high, performance is limited by peak FLOP/s
2. **Memory-Bound**: When arithmetic intensity is low, performance is limited by memory bandwidth

The model computes:
- `tokens/s_compute = (peak_TFLOP/s × compute_util) / FLOP_per_token`
- `tokens/s_bw = (HBM_GB/s × bw_util) / bytes_per_token`
- `tokens/s_final = min(compute, bw) × (1 + overlap_factor)`

**Limitations**: This is a simplified model. Real performance depends on many factors not captured here, including kernel efficiency, memory access patterns, and system-level effects.
            """)

    with tab2:
        st.info(
            "💡 **Sensitivity Analysis** shows which parameters have the most impact on throughput. "
            "See the **Top Levers** section in each hardware config's detailed analysis above."
        )

    with tab3:
        st.info(
            "📄 **Memo Export** generates an HTML summary of your analysis. "
            "This feature is coming soon."
        )


if __name__ == "__main__":
    main()
