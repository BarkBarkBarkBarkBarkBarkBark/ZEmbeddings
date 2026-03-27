"""
End-to-end experiment pipeline.
================================

Orchestrates the full flow:

    text → tokenise → embed → compute metrics → Kalman filter
                              → store to pgvector (optional)
                              → write YAML + Markdown reports

All parameters come from the ``params`` dict, which can be edited
interactively before calling ``run_pipeline()``.

Usage::

    from zembeddings.params import get_params
    from zembeddings.pipeline import run_pipeline

    p = get_params(**{"window.size": 15, "kalman.mode": "vector"})
    results = run_pipeline("data/synthetic/topic_shift_001.txt", params=p)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zembeddings.tokenizer import tokenize, windows_to_texts, TokenizedTranscript
from zembeddings.embeddings import embed_texts
from zembeddings.metrics import compute_metrics, TrajectoryMetrics, semantic_cloud_stats
from zembeddings.kalman import run_kalman, run_acceleration_kalman, KalmanResult
from zembeddings.io import (
    read_transcript,
    write_metrics_yaml,
    write_report_markdown,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Pipeline result
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PipelineResult:
    """Container for all pipeline outputs — inspectable at the REPL."""

    def __init__(
        self,
        params: dict[str, Any],
        transcript: TokenizedTranscript,
        embeddings_full: np.ndarray,
        embeddings_reduced: np.ndarray,
        metrics: TrajectoryMetrics,
        kalman: KalmanResult,
        kalman_accel: KalmanResult,
        cloud_stats: dict[str, float],
        experiment_name: str,
    ):
        self.params = params
        self.transcript = transcript
        self.embeddings_full = embeddings_full
        self.embeddings_reduced = embeddings_reduced
        self.metrics = metrics
        self.kalman = kalman
        self.kalman_accel = kalman_accel
        self.cloud_stats = cloud_stats
        self.experiment_name = experiment_name

    def __repr__(self) -> str:
        return (
            f"<PipelineResult "
            f"windows={self.metrics.n_windows} "
            f"boundaries={self.metrics.n_boundaries} "
            f"returns={self.metrics.n_returns} "
            f"kalman_violations={self.kalman.n_violations} "
            f"accel_violations={self.kalman_accel.n_violations} "
            f"path_length={self.metrics.total_path_length:.4f}>"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_pipeline(
    source: str | Path,
    params: dict[str, Any],
    *,
    experiment_name: str | None = None,
    write_outputs: bool = True,
    store_db: bool | None = None,
) -> PipelineResult:
    """Run the full analysis pipeline on a transcript.

    Parameters
    ----------
    source : str | Path
        Path to a plain-text transcript file, **or** a raw text string.
        If the string contains a newline or doesn't look like a path,
        it's treated as raw text.
    params : dict
        Full PARAMS dict (use ``get_params()`` or ``load_params()``).
    experiment_name : str, optional
        Name for output files.  Defaults to the source filename stem.
    write_outputs : bool
        Whether to write YAML + Markdown result files.
    store_db : bool, optional
        Whether to store embeddings & metrics in pgvector.  Defaults to
        ``params["database"]["enabled"]``.

    Returns
    -------
    PipelineResult
        All intermediate and final data, inspectable interactively.
    """
    # ── 0. Resolve source ─────────────────────────────────────────────
    source_str = str(source)
    is_file = False
    if "\n" not in source_str and len(source_str) < 1024:
        try:
            source_path = Path(source_str)
            is_file = source_path.is_file()
        except (OSError, ValueError):
            is_file = False

    if is_file:
        text = read_transcript(source_path)
        if experiment_name is None:
            experiment_name = source_path.stem
    else:
        text = source_str
        if experiment_name is None:
            experiment_name = "inline"

    print(f"── Pipeline: {experiment_name} ──")

    # ── 1. Tokenise + window ──────────────────────────────────────────
    transcript = tokenize(text, params)
    texts = windows_to_texts(transcript)
    n = len(texts)
    print(f"   {transcript.n_tokens} tokens → {n} windows "
          f"(size={params['window']['size']}, stride={params['window']['stride']})")

    if n < 2:
        raise ValueError(
            f"Need at least 2 windows but got {n}.  "
            f"Text has {transcript.n_tokens} tokens, window size is "
            f"{params['window']['size']}."
        )

    # ── 2. Embed ──────────────────────────────────────────────────────
    emb = embed_texts(texts, params)
    embeddings_full: np.ndarray = emb["full"]
    embeddings_reduced: np.ndarray = emb["reduced"]
    print(f"   Embedded: full {embeddings_full.shape}, "
          f"reduced {embeddings_reduced.shape}")

    # ── 3. Metrics ────────────────────────────────────────────────────
    met = compute_metrics(embeddings_full, params)
    print(f"   Metrics: velocity μ={met.mean_velocity:.6f} σ={met.std_velocity:.6f}, "
          f"boundaries={met.n_boundaries}, returns={met.n_returns}")

    # ── 4. Kalman filter ──────────────────────────────────────────────
    kal = run_kalman(
        embeddings_full, embeddings_reduced,
        met.cosine_distance, params,
    )
    print(f"   Kalman ({kal.mode}): {kal.n_violations} violations "
          f"(threshold={kal.threshold})")

    # ── 4b. Kalman on acceleration (context-change detector) ──────────
    kal_accel = run_acceleration_kalman(met.acceleration, params)
    print(f"   Kalman (acceleration): {kal_accel.n_violations} violations "
          f"(threshold={kal_accel.threshold})")

    # ── 5. Semantic cloud stats ───────────────────────────────────────
    cloud = semantic_cloud_stats(embeddings_full)
    print(f"   Cloud: mean_sim={cloud['mean_pairwise_sim']:.4f}, "
          f"std_sim={cloud['std_pairwise_sim']:.4f}")

    # ── 6. Build output dict ──────────────────────────────────────────
    boundary_indices = list(np.where(met.boundary_flags)[0].tolist())
    return_indices = list(np.where(met.return_flags)[0].tolist())

    metrics_dict = {
        "summary": {
            "n_windows": met.n_windows,
            "n_tokens": transcript.n_tokens,
            "mean_velocity": met.mean_velocity,
            "std_velocity": met.std_velocity,
            "n_boundaries": met.n_boundaries,
            "n_returns": met.n_returns,
            "total_path_length": met.total_path_length,
            "kalman_mode": kal.mode,
            "kalman_violations": kal.n_violations,
            "kalman_accel_violations": kal_accel.n_violations,
            "cloud_mean_sim": cloud["mean_pairwise_sim"],
            "cloud_std_sim": cloud["std_pairwise_sim"],
            "boundary_indices": boundary_indices,
            "return_indices": return_indices,
        },
        "timeseries": {
            "cosine_distance": met.cosine_distance.tolist(),
            "euclidean_distance": met.euclidean_distance.tolist(),
            "velocity": met.velocity.tolist(),
            "acceleration": met.acceleration.tolist(),
            "jerk": met.jerk.tolist(),
            "cosine_similarity": met.cosine_similarity.tolist(),
            "cosine_sim_d1": met.cosine_sim_d1.tolist(),
            "cosine_sim_d2": met.cosine_sim_d2.tolist(),
            "ema_drift": met.ema_drift.tolist(),
            "cumulative_path": met.cumulative_path.tolist(),
            "kalman_innovation": kal.innovation_norms.tolist(),
            "kalman_mahalanobis": kal.mahalanobis_distances.tolist(),
            "kalman_accel_innovation": kal_accel.innovation_norms.tolist(),
            "kalman_accel_mahalanobis": kal_accel.mahalanobis_distances.tolist(),
            "boundary_flags": met.boundary_flags.tolist(),
            "return_flags": met.return_flags.tolist(),
            "return_cluster_id": met.return_cluster_ids.tolist(),
            "fixation_flags": met.fixation_flags.tolist(),
            "cloud_valid": met.cloud_valid.tolist(),
        },
    }

    # ── 7. Write outputs ──────────────────────────────────────────────
    if write_outputs:
        write_metrics_yaml(metrics_dict, params, experiment_name)
        write_report_markdown(metrics_dict, params, experiment_name)

    # ── 8. Store to pgvector ──────────────────────────────────────────
    use_db = store_db if store_db is not None else params["database"]["enabled"]
    if use_db:
        _store_to_db(params, experiment_name, transcript, met, kal, kal_accel,
                     embeddings_full, embeddings_reduced, metrics_dict)

    # ── Done ──────────────────────────────────────────────────────────
    result = PipelineResult(
        params=params,
        transcript=transcript,
        embeddings_full=embeddings_full,
        embeddings_reduced=embeddings_reduced,
        metrics=met,
        kalman=kal,
        kalman_accel=kal_accel,
        cloud_stats=cloud,
        experiment_name=experiment_name,
    )
    print(f"   ✓ {result}")
    return result


def _store_to_db(
    params, experiment_name, transcript, met, kal, kal_accel,
    embeddings_full, embeddings_reduced, metrics_dict,
):
    """Store results in pgvector (guarded import)."""
    try:
        from zembeddings.database import (
            insert_experiment,
            insert_embeddings,
            insert_metrics,
        )
    except ImportError as exc:
        print(f"   ⚠ Database storage skipped (missing dependency: {exc})")
        return

    exp_id = insert_experiment(
        params, experiment_name,
        description=f"Windows={met.n_windows}, tokens={transcript.n_tokens}",
    )

    windows_data = [
        {
            "index": w.index,
            "text": w.text,
            "start_token": w.start_token,
            "end_token": w.end_token,
        }
        for w in transcript.windows
    ]
    insert_embeddings(params, exp_id, windows_data,
                      embeddings_full, embeddings_reduced)

    # Flatten metrics for DB
    ts = metrics_dict["timeseries"]
    db_metrics = {
        "cosine_distance": np.array(ts["cosine_distance"]),
        "euclidean_distance": np.array(ts["euclidean_distance"]),
        "velocity": np.array(ts["velocity"]),
        "acceleration": np.array(ts["acceleration"]),
        "jerk": np.array(ts["jerk"]),
        "cosine_similarity": np.array(ts["cosine_similarity"]),
        "cosine_sim_d1": np.array(ts["cosine_sim_d1"]),
        "cosine_sim_d2": np.array(ts["cosine_sim_d2"]),
        "ema_drift": np.array(ts["ema_drift"]),
        "cumulative_path": np.array(ts["cumulative_path"]),
        "kalman_innovation": np.array(ts["kalman_innovation"]),
        "kalman_mahalanobis": np.array(ts["kalman_mahalanobis"]),
        "is_boundary": np.array(ts["boundary_flags"]),
        "is_return": np.array(ts["return_flags"]),
        "return_cluster_id": np.array(ts["return_cluster_id"]),
        "is_fixation": np.array(ts["fixation_flags"]),
        "cloud_valid": np.array(ts["cloud_valid"]),
    }
    insert_metrics(params, exp_id, db_metrics)
    print(f"   ✓ Stored in pgvector (experiment_id={exp_id})")
