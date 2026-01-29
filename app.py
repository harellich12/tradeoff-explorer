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
from model.presets import load_workload_presets, load_hardware_presets
from model.roofline import analyze_detailed
from model.sensitivity import compute_sensitivity
from model.logging import (
    log_run,
    format_workload_for_log,
    format_hardware_for_log,
    format_assumptions_for_log,
    format_analysis_for_log,
)


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
        compute_util=0.7,
        bw_util=0.8,
        overlap_factor=0.0,
        weight_cache_hit_rate=0.0,
    )


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
    st.markdown(
        """
        Analyze and visualize the performance tradeoffs between LLM workloads
        and hardware configurations using roofline modeling.
        """
    )

    st.divider()

    # Load presets once
    workload_presets = load_workload_presets()
    hardware_presets = load_hardware_presets()
    
    # Build preset name lists with "Custom" option
    workload_names = ["Custom"] + [p.name for p in workload_presets]
    hardware_names = [p.name for p in hardware_presets]

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Workload Configuration")
        
        # Workload preset selector
        selected_workload = st.selectbox(
            "📦 Workload Preset",
            options=workload_names,
            index=1,  # Default to first real preset
            help="Select a preset to auto-fill parameters, or 'Custom' to enter manually"
        )
        
        # Get preset values if not custom
        if selected_workload == "Custom":
            preset = create_default_workload()
        else:
            preset = next((p for p in workload_presets if p.name == selected_workload), create_default_workload())
        
        st.divider()
        
        # Model parameters - use preset values as defaults
        params_b = st.slider(
            "Model Size (B params)", 
            1.0, 100.0, 
            float(preset.params_b), 
            0.5,
            disabled=(selected_workload != "Custom")
        )
        n_layers = st.slider(
            "Layers", 
            4, 128, 
            int(preset.n_layers), 
            2,
            disabled=(selected_workload != "Custom")
        )
        
        # For select_slider, we need to handle preset values
        hidden_options = [1024, 2048, 4096, 5120, 6144, 8192, 12288, 16384]
        hidden_idx = hidden_options.index(preset.hidden_size) if preset.hidden_size in hidden_options else 2
        hidden_size = st.select_slider(
            "Hidden Size",
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
        
        seq_options = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        seq_idx = seq_options.index(preset.seq_len) if preset.seq_len in seq_options else 2
        seq_len = st.select_slider(
            "Sequence Length",
            options=seq_options,
            value=seq_options[seq_idx]
        )
        batch = st.slider("Batch Size", 1, 64, int(preset.batch))
        
        st.divider()
        st.subheader("Data Types")
        
        dtype_options = ["bf16", "fp16", "int8", "int4"]
        weight_dtype = st.selectbox(
            "Weight Dtype",
            options=dtype_options,
            index=dtype_options.index(preset.weight_dtype) if preset.weight_dtype in dtype_options else 0
        )
        kv_dtype = st.selectbox(
            "KV Cache Dtype",
            options=dtype_options,
            index=dtype_options.index(preset.kv_dtype) if preset.kv_dtype in dtype_options else 0
        )
        kv_compression = st.slider("KV Compression", 0.1, 1.0, float(preset.kv_compression), 0.05)
        
        st.divider()
        st.subheader("Assumptions")
        
        compute_util = st.slider("Compute Utilization", 0.1, 1.0, 0.7, 0.05)
        bw_util = st.slider("Bandwidth Utilization", 0.1, 1.0, 0.8, 0.05)
        overlap_factor = st.slider("Overlap Factor", 0.0, 0.5, 0.0, 0.05,
            help="How much compute and memory operations can overlap (0=none, higher=more overlap benefit)")
        weight_cache_hit_rate = st.slider("Weight Cache Hit Rate", 0.0, 1.0, 0.0, 0.05)
        
        st.divider()
        st.caption("v0.5.0")

    # Build workload config using preset values when not custom
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
            # Use preset but allow seq_len and batch overrides
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

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Workload Summary")
        
        if config_valid:
            st.markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | **Model** | {workload.params_b}B params |
            | **Architecture** | {workload.n_layers}L × {workload.hidden_size}d × {workload.n_heads}h |
            | **Sequence** | {workload.seq_len:,} tokens |
            | **Batch** | {workload.batch} |
            | **Weight dtype** | {workload.weight_dtype} ({get_dtype_bytes(workload.weight_dtype)} bytes) |
            | **KV dtype** | {workload.kv_dtype} ({workload.kv_bytes_per_element():.1f} bytes) |
            """)
        else:
            st.warning("Invalid configuration. Using defaults.")

    with col2:
        st.subheader("📐 Assumptions")
        st.markdown(f"""
        | Assumption | Value |
        |------------|-------|
        | **Weight Cache Hit** | {assumptions.weight_cache_hit_rate*100:.0f}% |
        | **KV Compression** | {workload.kv_compression*100:.0f}% |
        | **Compute Util** | {assumptions.compute_util*100:.0f}% |
        | **Bandwidth Util** | {assumptions.bw_util*100:.0f}% |
        """)

    st.divider()

    # Compute metrics
    prefill_flops = flops_per_token(workload, "prefill")
    prefill_bytes = bytes_per_token_prefill(workload, assumptions)
    decode_flops = flops_per_token(workload, "decode")
    decode_bytes = bytes_per_token(workload, assumptions)

    # Results section
    st.subheader("📈 Per-Token Metrics")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            label="Prefill FLOPs/token",
            value=f"{format_number(prefill_flops)}",
        )
        st.caption("Compute per token (prefill)")

    with metric_col2:
        st.metric(
            label="Prefill Bytes/token",
            value=f"{format_number(prefill_bytes)}B",
        )
        st.caption("Memory per token (prefill)")

    with metric_col3:
        st.metric(
            label="Decode FLOPs/token",
            value=f"{format_number(decode_flops)}",
        )
        st.caption("Compute per token (decode)")

    with metric_col4:
        st.metric(
            label="Decode Bytes/token",
            value=f"{format_number(decode_bytes)}B",
        )
        st.caption("Memory per token (decode)")

    # Arithmetic intensity
    st.divider()
    
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
    # Hardware Comparison Section (2-4 configs)
    # =========================================================================
    st.subheader("🖥️ Hardware Comparison")
    
    # Select number of configs to compare
    num_configs = st.slider(
        "Number of configurations to compare",
        min_value=2,
        max_value=4,
        value=2,
        key="num_hw_configs"
    )
    
    # Initialize session state for hardware configs if needed
    if "hw_configs" not in st.session_state:
        st.session_state.hw_configs = {}
    
    # Create columns for hardware config selection and editing
    hw_columns = st.columns(num_configs)
    
    # Store edited hardware configs
    edited_hw_configs = []
    
    for i, col in enumerate(hw_columns):
        config_idx = i + 1
        with col:
            st.markdown(f"#### Config #{config_idx}")
            
            # Preset selector
            default_idx = min(i, len(hardware_names) - 1)
            selected_preset = st.selectbox(
                f"Preset",
                options=hardware_names,
                index=default_idx,
                key=f"hw_preset_{config_idx}"
            )
            
            # Get the preset as baseline
            base_config = next((p for p in hardware_presets if p.name == selected_preset), hardware_presets[0])
            
            # Editable fields
            edited_name = st.text_input(
                "Name",
                value=base_config.name,
                key=f"hw_name_{config_idx}"
            )
            
            edited_tflops = st.number_input(
                "Peak TFLOP/s",
                min_value=1.0,
                max_value=10000.0,
                value=float(base_config.peak_tflops),
                step=10.0,
                key=f"hw_tflops_{config_idx}"
            )
            
            edited_hbm_gbps = st.number_input(
                "HBM GB/s",
                min_value=100.0,
                max_value=10000.0,
                value=float(base_config.hbm_gbps),
                step=50.0,
                key=f"hw_hbm_{config_idx}"
            )
            
            edited_cost = st.number_input(
                "Cost $/hr (optional)",
                min_value=0.0,
                max_value=100.0,
                value=float(base_config.cost_per_hour) if base_config.cost_per_hour else 0.0,
                step=0.1,
                key=f"hw_cost_{config_idx}"
            )
            
            # Create the edited config
            edited_config = HardwareConfig(
                name=edited_name,
                peak_tflops=edited_tflops,
                hbm_gbps=edited_hbm_gbps,
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
        # Run analysis for each config
        analyses = []
        for hw_config in edited_hw_configs:
            analysis = analyze_detailed(workload, hw_config, assumptions, "decode")
            analyses.append(analysis)
        
        # Log the run
        log_run({
            "workload": format_workload_for_log(workload),
            "assumptions": format_assumptions_for_log(assumptions),
            "hardware_configs": [format_hardware_for_log(hw) for hw in edited_hw_configs],
            "results": [format_analysis_for_log(a) for a in analyses],
        })
        
        st.divider()
        st.markdown("### 📊 Comparison Results")
        
        # Build comparison table data
        table_data = {
            "Config": [],
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
            table_data["Config"].append(hw_config.name)
            table_data["Peak TFLOP/s"].append(f"{hw_config.peak_tflops:.0f}")
            table_data["HBM GB/s"].append(f"{hw_config.hbm_gbps:.0f}")
            table_data["Bottleneck"].append(analysis["bottleneck_class"])
            table_data["Throughput (tok/s)"].append(f"{analysis['tokens_s_final']:,.0f}")
            table_data["Required TFLOP/s"].append(f"{analysis['required_tflops']:.1f}")
            table_data["Available TFLOP/s"].append(f"{hw_config.peak_tflops * assumptions.compute_util:.1f}")
            table_data["Required GB/s"].append(f"{analysis['required_gbps']:.1f}")
            table_data["Available GB/s"].append(f"{hw_config.hbm_gbps * assumptions.bw_util:.1f}")
        
        # Display as dataframe
        import pandas as pd
        df = pd.DataFrame(table_data)
        
        # Style the dataframe
        def highlight_bottleneck(val):
            if val == "COMPUTE":
                return "background-color: #ffcccb; color: black"
            elif val == "MEMORY_BW":
                return "background-color: #ffffcc; color: black"
            return ""
        
        styled_df = df.style.applymap(
            highlight_bottleneck, 
            subset=["Bottleneck"]
        )
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Find winner
        throughputs = [a["tokens_s_final"] for a in analyses]
        max_throughput = max(throughputs)
        winner_idx = throughputs.index(max_throughput)
        winner_config = edited_hw_configs[winner_idx]
        
        # Calculate speedups
        speedup_msgs = []
        for i, (hw, analysis) in enumerate(zip(edited_hw_configs, analyses)):
            if i != winner_idx:
                speedup = max_throughput / analysis["tokens_s_final"] if analysis["tokens_s_final"] > 0 else float('inf')
                speedup_msgs.append(f"{speedup:.2f}x vs {hw.name}")
        
        st.success(f"🏆 **{winner_config.name}** is fastest: {format_number(max_throughput)} tok/s ({', '.join(speedup_msgs)})")
        
        # Detailed cards for each config
        st.markdown("### 📋 Detailed Analysis")
        
        detail_cols = st.columns(num_configs)
        for i, (col, hw_config, analysis) in enumerate(zip(detail_cols, edited_hw_configs, analyses)):
            with col:
                is_winner = (i == winner_idx)
                header = f"{'🏆 ' if is_winner else ''}{hw_config.name}"
                st.markdown(f"#### {header}")
                
                # Bottleneck badge
                bn = analysis["bottleneck_class"]
                if bn == "COMPUTE":
                    st.error(f"💻 **{bn}**")
                else:
                    st.warning(f"💾 **{bn}**")
                
                st.metric(
                    label="Throughput",
                    value=f"{format_number(analysis['tokens_s_final'])} tok/s",
                )
                
                with st.expander("Details"):
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
                
                with st.expander("Top Levers"):
                    # Compute sensitivity for this config
                    levers = compute_sensitivity(workload, hw_config, assumptions, "decode")
                    for idx, lever in enumerate(levers, 1):
                        direction = "↑" if lever.high_throughput > lever.low_throughput else "↓"
                        st.markdown(f"**{idx}. {lever.param_display_name}** ({lever.percent_impact:.1f}% impact {direction})")
                        st.caption(lever.explanation)
                
                with st.expander("Explanation"):
                    st.markdown(analysis["explanation"])

    st.divider()

    # Formula explanations
    tab1, tab2, tab3 = st.tabs(["Formula Details", "Sensitivity Analysis", "Memo Export"])

    with tab1:
        formula_col1, formula_col2 = st.columns(2)
        
        with formula_col1:
            with st.expander("FLOPs Formula (Decode)", expanded=True):
                st.markdown(get_flops_formula_markdown("decode"))
        
        with formula_col2:
            with st.expander("Bytes Formula (Decode)", expanded=True):
                st.markdown(get_bytes_formula_markdown("decode"))

    with tab2:
        st.info(
            "**Sensitivity Analysis:** Tornado charts showing parameter "
            "impact on throughput. (Coming soon)"
        )

    with tab3:
        st.info(
            "**Memo Export:** Generate an HTML summary report of the analysis. (Coming soon)"
        )


if __name__ == "__main__":
    main()
