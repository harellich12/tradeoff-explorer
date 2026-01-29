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
    hardware_names = ["Custom"] + [p.name for p in hardware_presets]

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
        
        weight_cache_hit_rate = st.slider("Weight Cache Hit Rate", 0.0, 1.0, 0.0, 0.05)
        
        st.divider()
        
        # Hardware preset selector
        st.header("🖥️ Hardware Configuration")
        selected_hardware = st.selectbox(
            "📦 Hardware Preset",
            options=hardware_names,
            index=1,  # Default to first real preset
            help="Select a hardware preset for comparison"
        )
        
        # Display selected hardware info
        if selected_hardware != "Custom":
            hw_preset = next((p for p in hardware_presets if p.name == selected_hardware), None)
            if hw_preset:
                st.caption(f"**{hw_preset.peak_tflops}** TFLOP/s | **{hw_preset.hbm_gbps}** GB/s")
        
        st.divider()
        st.caption("v0.3.0")

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
        assumptions = Assumptions(weight_cache_hit_rate=weight_cache_hit_rate)
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
