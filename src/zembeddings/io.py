"""
YAML and Markdown I/O for experiment results.
==============================================

- **YAML** — machine-readable metric time-series and summary stats.
- **Markdown** — human-readable experiment reports with tables and
  ASCII-art sparklines.

All output files are timestamped so successive runs never overwrite
each other.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  YAML output
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _numpy_to_python(obj):
    """Recursively convert numpy types for YAML serialisation."""
    if isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_metrics_yaml(
    metrics_dict: dict[str, Any],
    params: dict[str, Any],
    experiment_name: str,
    out_dir: str | Path | None = None,
) -> Path:
    """Write the full metric suite to a timestamped YAML file.

    Returns the path to the written file.
    """
    if out_dir is None:
        out_dir = Path(params["paths"]["results_metrics"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{stamp}.yaml"
    path = out_dir / filename

    payload = {
        "experiment": experiment_name,
        "timestamp": stamp,
        "params": _numpy_to_python(params),
        "summary": _numpy_to_python(metrics_dict.get("summary", {})),
        "timeseries": _numpy_to_python(metrics_dict.get("timeseries", {})),
    }

    with open(path, "w") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True, width=120)

    print(f"✓ Metrics written to {path}")
    return path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Markdown report
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float], width: int = 40) -> str:
    """Render a list of floats as a unicode sparkline."""
    clean = [v for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return ""
    lo, hi = min(clean), max(clean)
    rng = hi - lo if hi != lo else 1.0
    chars = []
    step = max(1, len(clean) // width)
    for i in range(0, len(clean), step):
        chunk = clean[i : i + step]
        avg = sum(chunk) / len(chunk)
        idx = int((avg - lo) / rng * (len(_SPARKLINE_CHARS) - 1))
        chars.append(_SPARKLINE_CHARS[idx])
    return "".join(chars)


def write_report_markdown(
    metrics_dict: dict[str, Any],
    params: dict[str, Any],
    experiment_name: str,
    out_dir: str | Path | None = None,
) -> Path:
    """Write a human-readable Markdown report.

    Returns the path to the written file.
    """
    if out_dir is None:
        out_dir = Path(params["paths"]["results_reports"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{stamp}.md"
    path = out_dir / filename

    summary = metrics_dict.get("summary", {})
    ts = metrics_dict.get("timeseries", {})

    lines: list[str] = [
        f"# Experiment Report: {experiment_name}",
        f"*Generated {stamp}*",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    for k, v in summary.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Parameters",
        "",
        "```yaml",
        yaml.dump(_numpy_to_python(params), default_flow_style=False, sort_keys=False),
        "```",
        "",
    ]

    # Sparkline visualisations for key series
    sparkline_keys = [
        ("cosine_distance", "Cosine Distance"),
        ("velocity", "Velocity"),
        ("acceleration", "Acceleration"),
        ("jerk", "Jerk"),
        ("ema_drift", "EMA Drift"),
        ("kalman_mahalanobis", "Kalman Mahalanobis"),
        ("kalman_accel_mahalanobis", "Kalman Accel Mahalanobis"),
    ]

    lines += ["## Trajectory Sparklines", ""]
    for key, label in sparkline_keys:
        if key in ts:
            spark = _sparkline(ts[key])
            lines.append(f"**{label}:**  `{spark}`")
            lines.append("")

    # Boundary events
    if "boundary_indices" in summary:
        lines += [
            "## Detected Boundaries",
            "",
            "| Window Index | Velocity | Kalman d_M |",
            "|-------------|----------|-----------|",
        ]
        for idx in summary["boundary_indices"]:
            vel = ts.get("velocity", [None] * (idx + 1))[idx]
            dm = ts.get("kalman_mahalanobis", [None] * (idx + 1))[idx]
            lines.append(f"| {idx} | {_fmt(vel)} | {_fmt(dm)} |")
        lines.append("")

    # Return events
    if "return_indices" in summary:
        lines += [
            "## Detected Returns",
            "",
            "| Window Index | Cluster ID |",
            "|-------------|-----------|",
        ]
        for idx in summary["return_indices"]:
            cid = ts.get("return_cluster_id", [None] * (idx + 1))[idx]
            lines.append(f"| {idx} | {cid} |")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"✓ Report written to {path}")
    return path


def _fmt(val, decimals: int = 6) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val:.{decimals}f}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Text file reader
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def read_transcript(path: str | Path) -> str:
    """Read a plain-text transcript file."""
    return Path(path).read_text(encoding="utf-8")
