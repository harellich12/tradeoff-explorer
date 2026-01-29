"""
HTML memo generator for analysis reports.

Generates downloadable HTML summaries of tradeoff analysis.
"""

from datetime import datetime
from typing import List, Optional

from .schemas import AnalysisResult, SensitivityPoint


def generate_memo_html(
    result: AnalysisResult,
    sensitivity: Optional[List[SensitivityPoint]] = None,
    title: str = "Tradeoff Analysis Memo",
) -> str:
    """
    Generate an HTML memo summarizing the analysis.

    Args:
        result: Analysis result to summarize
        sensitivity: Optional sensitivity analysis results
        title: Memo title

    Returns:
        HTML string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            background: #f5f5f5;
            color: #333;
        }}
        .memo {{
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1a1a2e;
            border-bottom: 2px solid #4a90d9;
            padding-bottom: 0.5rem;
        }}
        h2 {{
            color: #4a90d9;
            margin-top: 1.5rem;
        }}
        .metric {{
            display: inline-block;
            background: #e8f4fd;
            padding: 1rem 1.5rem;
            border-radius: 6px;
            margin: 0.5rem 0.5rem 0.5rem 0;
        }}
        .metric-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: #1a1a2e;
        }}
        .metric-label {{
            font-size: 0.875rem;
            color: #666;
        }}
        .bottleneck-compute {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 0.75rem;
            margin: 1rem 0;
        }}
        .bottleneck-memory {{
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 0.75rem;
            margin: 1rem 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
        }}
        .timestamp {{
            color: #888;
            font-size: 0.875rem;
            margin-top: 2rem;
        }}
    </style>
</head>
<body>
    <div class="memo">
        <h1>⚖️ {title}</h1>

        <h2>Configuration</h2>
        <p><strong>Workload:</strong> {result.workload}</p>
        <p><strong>Hardware:</strong> {result.hardware}</p>

        <h2>Key Metrics</h2>
        <div class="metric">
            <div class="metric-value">{result.throughput_tokens_per_sec:,.1f}</div>
            <div class="metric-label">Tokens/second</div>
        </div>
        <div class="metric">
            <div class="metric-value">{result.arithmetic_intensity:.2f}</div>
            <div class="metric-label">FLOP/byte</div>
        </div>
        <div class="metric">
            <div class="metric-value">{result.utilization*100:.1f}%</div>
            <div class="metric-label">Utilization</div>
        </div>

        <h2>Bottleneck Analysis</h2>
        <div class="bottleneck-{result.bottleneck}">
            <strong>{'⚡ Compute-Bound' if result.bottleneck == 'compute' else '💾 Memory-Bound'}</strong>
            <p>
                {'Performance is limited by the GPU compute capacity (FLOP/s). Consider using lower precision or smaller batch sizes.'
                 if result.bottleneck == 'compute'
                 else 'Performance is limited by memory bandwidth. Consider increasing batch size or using tensor parallelism.'}
            </p>
        </div>

        <h2>Computation Details</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Prefill FLOPs</td>
                <td>{result.prefill_flops:.2e}</td>
            </tr>
            <tr>
                <td>Decode FLOPs/token</td>
                <td>{result.decode_flops_per_token:.2e}</td>
            </tr>
            <tr>
                <td>Memory bytes/token</td>
                <td>{result.memory_bytes_per_token:.2e}</td>
            </tr>
        </table>
"""

    if sensitivity:
        html += """
        <h2>Sensitivity Analysis</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Low</th>
                <th>Base</th>
                <th>High</th>
                <th>Impact</th>
            </tr>
"""
        for point in sensitivity:
            impact = abs(point.high_throughput - point.low_throughput)
            html += f"""
            <tr>
                <td>{point.parameter_name}</td>
                <td>{point.low_value:.2f}</td>
                <td>{point.base_value:.2f}</td>
                <td>{point.high_value:.2f}</td>
                <td>±{impact:,.1f} tok/s</td>
            </tr>
"""
        html += "        </table>\n"

    html += f"""
        <p class="timestamp">Generated: {timestamp}</p>
    </div>
</body>
</html>"""

    return html


def save_memo(html: str, filepath: str) -> None:
    """
    Save memo HTML to file.

    Args:
        html: HTML content
        filepath: Output file path
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
