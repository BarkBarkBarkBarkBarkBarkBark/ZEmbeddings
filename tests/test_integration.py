"""
Integration test — full pipeline on synthetic data.

Runs the complete pipeline on the synthetic topic_shift_001 transcript
with mocked embeddings, verifying that all stages chain together
correctly and the output contracts hold.
"""

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
import yaml
from zembeddings.params import get_params
from zembeddings.pipeline import run_pipeline


# Use clustered synthetic embeddings to simulate real topic shifts
def _make_clustered_mock(n_windows, dims_full=1536, dims_reduced=256):
    """Create mock embeddings with clear cluster structure."""
    rng = np.random.default_rng(42)

    # Create 5 clusters of roughly equal size (matching topic_shift_001)
    cluster_size = n_windows // 5
    remainder = n_windows - cluster_size * 5
    sizes = [cluster_size] * 5
    sizes[-1] += remainder  # give remainder to last cluster

    full = []
    for size in sizes:
        center = rng.standard_normal(dims_full)
        center /= np.linalg.norm(center)
        noise = rng.standard_normal((size, dims_full)) * 0.03
        cluster = center + noise
        norms = np.linalg.norm(cluster, axis=1, keepdims=True)
        full.append(cluster / norms)
    full = np.vstack(full).astype(np.float32)

    # Reduced is just truncation for the mock
    reduced = full[:, :dims_reduced].copy()
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    reduced /= np.maximum(norms, 1e-12)

    return full, reduced


def _mock_embed_clustered(texts, params, *, use_cache=True):
    """Mock that returns clustered embeddings."""
    n = len(texts)
    dims_full = params["model"]["dimensions_full"]
    dims_reduced = params["model"]["dimensions_reduced"]
    full, reduced = _make_clustered_mock(n, dims_full, dims_reduced)
    return {"full": full, "reduced": reduced, "texts": texts}


class TestIntegration:
    @patch("zembeddings.pipeline.embed_texts", side_effect=_mock_embed_clustered)
    def test_full_pipeline_on_synthetic(self, mock_embed, tmp_path):
        """End-to-end: load synthetic text → pipeline → verify outputs."""
        synthetic_path = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "topic_shift_001.txt"

        # Generate if missing
        if not synthetic_path.exists():
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
            from generate_conversations import generate_all
            generate_all()

        p = get_params(**{
            "boundary.k_sigma": 1.5,
            "boundary.min_samples": 3,
            "kalman.mode": "scalar",
            "kalman.innovation_threshold": 2.5,
            "paths.results_metrics": str(tmp_path / "metrics"),
            "paths.results_reports": str(tmp_path / "reports"),
            "paths.data_processed": str(tmp_path / "processed"),
        })

        result = run_pipeline(
            str(synthetic_path), p,
            experiment_name="integration_test",
            write_outputs=True,
        )

        # ── Structural checks ────────────────────────────────────────
        assert result.metrics.n_windows > 10
        assert result.embeddings_full.shape[0] == result.metrics.n_windows
        assert result.embeddings_reduced.shape[0] == result.metrics.n_windows

        # ── Metric array shapes ──────────────────────────────────────
        N = result.metrics.n_windows
        assert result.metrics.cosine_distance.shape == (N,)
        assert result.metrics.velocity.shape == (N,)
        assert result.metrics.acceleration.shape == (N,)
        assert result.metrics.jerk.shape == (N,)
        assert result.metrics.ema_centroids.shape[0] == N
        assert result.metrics.boundary_flags.shape == (N,)
        assert result.metrics.return_flags.shape == (N,)

        # ── Kalman checks ────────────────────────────────────────────
        assert result.kalman.mode == "scalar"
        assert result.kalman.mahalanobis_distances.shape == (N,)
        assert result.kalman_accel.mode == "acceleration"
        assert result.kalman_accel.mahalanobis_distances.shape == (N,)

        # ── Boundary detection ───────────────────────────────────────
        # With clustered embeddings, we should detect at least some boundaries
        assert result.metrics.n_boundaries >= 1
        # Kalman should also flag violations at cluster transitions
        assert result.kalman.n_violations >= 1

        # ── Cloud stats ──────────────────────────────────────────────
        assert result.cloud_stats["mean_pairwise_sim"] > -1
        assert result.cloud_stats["std_pairwise_sim"] >= 0

        # ── Output files ─────────────────────────────────────────────
        yaml_files = list((tmp_path / "metrics").glob("*.yaml"))
        md_files = list((tmp_path / "reports").glob("*.md"))
        assert len(yaml_files) == 1
        assert len(md_files) == 1

        # ── YAML output parseable ────────────────────────────────────
        with open(yaml_files[0]) as f:
            data = yaml.safe_load(f)
        assert data["experiment"] == "integration_test"
        assert "summary" in data
        assert "timeseries" in data
        assert "kalman_accel_mahalanobis" in data["timeseries"]

    @patch("zembeddings.pipeline.embed_texts", side_effect=_mock_embed_clustered)
    def test_vector_kalman_integration(self, mock_embed, tmp_path):
        """Pipeline with vector Kalman mode."""
        p = get_params(**{
            "kalman.mode": "vector",
            "paths.results_metrics": str(tmp_path / "metrics"),
            "paths.results_reports": str(tmp_path / "reports"),
            "paths.data_processed": str(tmp_path / "processed"),
        })

        text = (
            "Dogs are wonderful companions who bring joy and happiness. "
            "Quantum mechanics describes the behavior of subatomic particles. "
            "The ocean covers seventy percent of the Earth's surface. "
        ) * 5  # repeat to get enough tokens

        result = run_pipeline(text, p, write_outputs=False, experiment_name="vec_test")
        assert result.kalman.mode == "vector"
        assert result.kalman_accel.mode == "acceleration"
        assert result.metrics.n_windows > 5

    @patch("zembeddings.pipeline.embed_texts", side_effect=_mock_embed_clustered)
    def test_cumulative_path_increases(self, mock_embed, tmp_path):
        """Cumulative path length should only increase."""
        p = get_params(**{
            "paths.results_metrics": str(tmp_path / "metrics"),
            "paths.results_reports": str(tmp_path / "reports"),
            "paths.data_processed": str(tmp_path / "processed"),
        })
        text = "Random text about various topics. " * 30
        result = run_pipeline(text, p, write_outputs=False)
        diffs = np.diff(result.metrics.cumulative_path)
        assert np.all(diffs >= -1e-10)

    @patch("zembeddings.pipeline.embed_texts", side_effect=_mock_embed_clustered)
    def test_ema_drift_nonnegative(self, mock_embed, tmp_path):
        """EMA drift (cosine distance) should be non-negative."""
        p = get_params(**{
            "paths.results_metrics": str(tmp_path / "metrics"),
            "paths.results_reports": str(tmp_path / "reports"),
            "paths.data_processed": str(tmp_path / "processed"),
        })
        text = "Some text about nature and science. " * 30
        result = run_pipeline(text, p, write_outputs=False)
        assert np.all(result.metrics.ema_drift >= -1e-10)
