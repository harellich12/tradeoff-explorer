# Tradeoff Explorer

A Streamlit application for exploring LLM inference performance tradeoffs across different hardware configurations and model architectures using roofline modeling.

## Screenshots

> **To capture screenshots for documentation:**
> 1. Run the app: `streamlit run app.py`
> 2. Use browser dev tools or a screen capture tool
> 3. Capture the following views:
>    - **Main Dashboard**: Workload summary and per-token metrics
>    - **Hardware Comparison**: Comparison table with multiple configs
>    - **Detailed Analysis**: Individual hardware cards with bottleneck classification
>    - **Sidebar**: Workload configuration panel
> 4. Save to `docs/screenshots/` directory

## Features

- **Workload Configuration**: Configure model parameters (size, batch, sequence length, precision)
- **Hardware Presets**: Select from 12 pre-defined hardware archetypes or customize
- **Roofline Analysis**: Compute upper-bound throughput and bottleneck classification
- **Hardware Comparison**: Compare 2-4 hardware configurations side-by-side
- **Sensitivity Analysis**: Top levers showing which parameters impact performance most
- **Extrapolation Warnings**: Alerts when using extreme values that may cause inaccurate estimates
- **Run Logging**: All analyses logged to `logs/runs.jsonl` with timestamps and git hash
- **Memo Export**: Generate HTML summary reports (coming soon)

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Setup

1. **Clone or navigate to the project directory:**

   ```bash
   cd tradeoff_explorer
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**

   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

Start the Streamlit server:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Quick Start Guide

1. **Select a Model Preset** in the sidebar (or choose "Custom" for manual entry)
2. **Adjust Inference Settings**: Sequence length and batch size
3. **Configure Hardware**: Select 2-4 hardware configurations to compare
4. **Click "Run Analysis"** to see comparison results
5. **Review Results**: Check bottlenecks, throughput, and top levers

### Understanding Results

- **COMPUTE bottleneck** (red): Limited by GPU compute capacity
- **MEMORY_BW bottleneck** (yellow): Limited by memory bandwidth
- **Top Levers**: Parameters that most impact throughput (±20% sensitivity)

### Running Tests

Execute the test suite:

```bash
pytest -q
```

For verbose output:

```bash
pytest -v
```

## Project Structure

```
tradeoff_explorer/
├── app.py                    # Streamlit UI entry point
├── model/
│   ├── schemas.py            # Data validation schemas
│   ├── workloads.py          # LLM prefill/decode FLOPs formulas
│   ├── hardware.py           # Hardware schema + presets
│   ├── roofline.py           # Throughput & bottleneck computation
│   ├── sensitivity.py        # One-at-a-time sensitivity analysis
│   ├── logging.py            # Run logging to JSONL
│   └── memo.py               # HTML memo generator
├── data/
│   ├── workload_presets.json # Model presets (7B/13B/70B)
│   └── hardware_presets.json # Hardware archetypes
├── logs/
│   └── runs.jsonl            # Appended run logs
├── tests/
│   ├── test_invariants.py    # Monotonicity & sanity checks
│   └── test_units.py         # Unit tests for formulas
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Development

### Adding New Features

1. Define new schemas in `model/schemas.py`
2. Implement computation logic in the appropriate model module
3. Add UI components in `app.py`
4. Write tests in the `tests/` directory

### Logging

Run logs are appended to `logs/runs.jsonl` in JSON Lines format. Each entry includes:
- Timestamp (ISO 8601)
- App version
- Git commit hash (if available)
- Full workload, hardware, and assumption configurations
- Analysis results

### Testing Invariants

The test suite includes invariant tests to ensure model correctness:
- Monotonicity: More resources → higher throughput
- KV dtype: INT8/INT4 uses fewer bytes than BF16
- Sanity: All outputs finite and non-negative

## Disclaimer

⚠️ **Results are upper-bound estimates under assumptions.** Real-world performance depends on many factors not modeled here, including kernel efficiency, memory access patterns, and system-level effects.

## License

MIT License
