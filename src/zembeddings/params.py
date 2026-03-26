"""
Central parameter dictionary for ZEmbeddings experiments.
=========================================================

Every tuneable knob lives in the ``PARAMS`` dict.  Import it into a
Python REPL, inspect / mutate values, then pass the dict to the
pipeline.  Parameters can also be loaded from YAML config files and
deep-merged over these defaults.

Usage (interactive)::

    >>> from zembeddings.params import PARAMS, get_params
    >>> PARAMS["window"]
    {'size': 10, 'stride': 1, 'encoding': 'cl100k_base'}
    >>> p = get_params(**{"window.size": 20, "ema.alpha": 0.5})

Usage (from YAML)::

    >>> from zembeddings.params import load_params
    >>> p = load_params("config/experiments/exp_001_synthetic_topic_shift.yaml")
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DEFAULT PARAMETERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PARAMS: dict[str, Any] = {
    # ── Embedding model ────────────────────────────────────────────────
    # backend: "openai" (API, costs $) or "local" (sentence-transformers, free)
    #
    # OpenAI reference: https://platform.openai.com/docs/guides/embeddings
    # Local models that fit in 8 GB RAM:
    #   - all-MiniLM-L6-v2          384-d,   ~80 MB  (fast, good baseline)
    #   - all-mpnet-base-v2         768-d,  ~420 MB  (best quality / size)
    #   - nomic-ai/nomic-embed-text-v1.5  768-d, ~550 MB (long context)
    "model": {
        "backend": "openai",           # "openai" | "local"
        "name": "text-embedding-3-small",
        "dimensions_full": 1536,
        "dimensions_reduced": 256,
        "max_tokens_per_input": 8192,
        "batch_size": 2048,  # max inputs per API call (OpenAI)
        # local-only settings (ignored when backend="openai")
        "local_model": "all-MiniLM-L6-v2",
        "local_batch_size": 256,
        "device": "mps",  # "mps" (Apple Silicon) | "cpu"
    },

    # ── Sliding window ────────────────────────────────────────────────
    # Causal: window at step t = tokens[t-W+1 : t+1].
    # Stride of 1 gives maximum temporal resolution.
    "window": {
        "size": 10,           # tokens per window
        "stride": 1,          # step size in tokens
        "encoding": "cl100k_base",  # tiktoken encoding for the model
    },

    # ── Exponential moving average (EMA) ──────────────────────────────
    # Causal one-sided smoothing of embeddings.
    #   α close to 1  →  fast adaptation (short memory)
    #   α close to 0  →  slow adaptation (long memory)
    "ema": {
        "alpha": 0.3,
    },

    # ── Boundary detection ────────────────────────────────────────────
    "boundary": {
        "k_sigma": 2.0,       # velocity > μ + kσ  → boundary
        "min_samples": 5,     # warmup: skip first N windows
    },

    # ── Kalman filter ─────────────────────────────────────────────────
    # Constant-velocity model for trajectory prediction.
    #
    # Ref: Kalman, R. E. (1960). "A New Approach to Linear Filtering
    #      and Prediction Problems." J. Basic Eng., 82(1), 35–45.
    "kalman": {
        "process_noise_scale": 1e-4,       # Q diagonal scale
        "measurement_noise_scale": 1e-2,   # R diagonal scale
        "initial_covariance_scale": 1.0,   # P₀ diagonal scale
        "innovation_threshold": 3.0,       # Mahalanobis cutoff
        "mode": "scalar",                  # "scalar" | "vector"
    },

    # ── Semantic cloud ────────────────────────────────────────────────
    # Concentration of measure in high dimensions means most random
    # unit vectors are nearly orthogonal.  Similarities below the
    # floor are indistinguishable from noise.
    #
    # Ref: Vershynin, R. (2018). High-Dimensional Probability.
    #      Cambridge University Press.  Ch. 5.
    "semantic_cloud": {
        "similarity_floor": 0.30,    # below this, cosine sim ≈ noise
        "cluster_threshold": 0.85,   # above this → same cluster
        "return_threshold": 0.80,    # sim to past centroid → "return"
    },

    # ── Derivative computation ────────────────────────────────────────
    "derivatives": {
        "method": "gradient",   # "gradient" (numpy.gradient) | "diff"
        "edge_order": 2,        # boundary accuracy for numpy.gradient
    },

    # ── Database (pgvector) ───────────────────────────────────────────
    "database": {
        "enabled": False,       # flip True when postgres is ready
        "host": "localhost",
        "port": 5432,
        "dbname": "zembeddings",
        "user": "postgres",
        "password": "",         # override via .env  DB_PASSWORD
    },

    # ── File paths (relative to project root) ─────────────────────────
    "paths": {
        "data_raw": "data/raw",
        "data_synthetic": "data/synthetic",
        "data_processed": "data/processed",
        "results_metrics": "results/metrics",
        "results_reports": "results/reports",
    },
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge *overrides* into *base* (in-place)."""
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def get_params(**overrides: Any) -> dict[str, Any]:
    """Return a deep copy of ``PARAMS`` with dot-notation overrides.

    Examples::

        get_params(**{"window.size": 20, "ema.alpha": 0.5})

    """
    p = copy.deepcopy(PARAMS)
    for dotkey, value in overrides.items():
        keys = dotkey.split(".")
        d = p
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
    return p


def load_params(yaml_path: str | Path) -> dict[str, Any]:
    """Load a YAML config and deep-merge over the defaults."""
    p = copy.deepcopy(PARAMS)
    with open(yaml_path) as f:
        overrides = yaml.safe_load(f) or {}
    _deep_merge(p, overrides)
    return p


def save_params(params: dict[str, Any], yaml_path: str | Path) -> None:
    """Persist a params dict to a YAML file."""
    Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)


def describe_params(params: dict[str, Any] | None = None, indent: int = 0) -> str:
    """Pretty-print a params dict for terminal inspection."""
    if params is None:
        params = PARAMS
    lines: list[str] = []
    prefix = "  " * indent
    for k, v in params.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.append(describe_params(v, indent + 1))
        else:
            lines.append(f"{prefix}{k}: {v!r}")
    return "\n".join(lines)
