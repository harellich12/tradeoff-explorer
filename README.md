# Tradeoff Explorer

A Streamlit application for exploring LLM inference performance tradeoffs across different hardware configurations and model architectures using roofline modeling.

## Features

- **Workload Configuration**: Configure model parameters (size, batch, sequence length, precision)
- **Hardware Presets**: Select from pre-defined hardware archetypes or define custom specs
- **Roofline Analysis**: Compute upper-bound throughput and bottleneck classification
- **Sensitivity Analysis**: Tornado charts showing parameter impact on performance
- **Memo Export**: Generate HTML summary reports

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

Run logs are appended to `logs/runs.jsonl` in JSON Lines format.

## License

MIT License
