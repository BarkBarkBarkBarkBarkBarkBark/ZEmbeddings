"""
Run an experiment from a YAML config file.
==========================================

Usage::

    python scripts/run_experiment.py config/experiments/exp_001_synthetic_topic_shift.yaml

Or from Python::

    from zembeddings.params import load_params
    from zembeddings.pipeline import run_pipeline

    p = load_params("config/experiments/exp_001_synthetic_topic_shift.yaml")
    result = run_pipeline(p["experiment"]["source"], params=p)
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Add src to path for direct execution ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zembeddings.params import load_params, PARAMS
from zembeddings.pipeline import run_pipeline


def main(config_path: str | None = None) -> None:
    """Run a single experiment."""
    if config_path is None:
        if len(sys.argv) < 2:
            print("Usage: python scripts/run_experiment.py <config.yaml>")
            print("       or pass a config path to main()")
            sys.exit(1)
        config_path = sys.argv[1]

    config = Path(config_path)
    if not config.exists():
        print(f"✗ Config not found: {config}")
        sys.exit(1)

    params = load_params(config)

    # Resolve source path relative to project root
    source = params.get("experiment", {}).get("source", "")
    if not source:
        print("✗ Config must specify experiment.source")
        sys.exit(1)

    source_path = Path(source)
    if not source_path.is_file():
        # Try relative to project root
        project_root = Path(__file__).resolve().parent.parent
        source_path = project_root / source
        if not source_path.is_file():
            print(f"✗ Source not found: {source}")
            print(f"  (also tried: {source_path})")
            sys.exit(1)

    experiment_name = params.get("experiment", {}).get("name", config.stem)
    params.setdefault("experiment", {})
    params["experiment"]["name"] = experiment_name
    params["experiment"]["source"] = str(source_path)

    print(f"╔══════════════════════════════════════════╗")
    print(f"║  ZEmbeddings — Experiment Runner         ║")
    print(f"╚══════════════════════════════════════════╝")
    print(f"  Config:     {config}")
    print(f"  Source:     {source_path}")
    print(f"  Experiment: {experiment_name}")
    print()

    result = run_pipeline(
        str(source_path),
        params=params,
        experiment_name=experiment_name,
    )

    print()
    print(f"═══ Result ═══")
    print(f"  Windows:          {result.metrics.n_windows}")
    print(f"  Boundaries:       {result.metrics.n_boundaries}")
    print(f"  Returns:          {result.metrics.n_returns}")
    print(f"  Kalman violations:{result.kalman.n_violations}")
    print(f"  Path length:      {result.metrics.total_path_length:.6f}")
    print(f"  Mean velocity:    {result.metrics.mean_velocity:.6f}")
    print(f"  Cloud mean sim:   {result.cloud_stats['mean_pairwise_sim']:.4f}")
    print()

    return result


if __name__ == "__main__":
    main()
