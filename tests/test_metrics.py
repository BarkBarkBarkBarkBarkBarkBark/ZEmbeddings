"""
Tests for the metrics module.

Uses synthetic embeddings (no API calls) to verify that the metric
computations are mathematically correct.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
from zembeddings.params import PARAMS, get_params
from zembeddings.metrics import (
    compute_metrics,
    pairwise_cosine_matrix,
    semantic_cloud_stats,
)


def _make_embeddings(n: int = 50, dim: int = 1536, seed: int = 42) -> np.ndarray:
    """Generate random L2-normalised embeddings."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / norms).astype(np.float64)


def _make_clustered_embeddings(
    cluster_sizes: list[int] = [15, 15, 20],
    dim: int = 1536,
    cluster_spread: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Generate embeddings in distinct clusters (to test boundary detection)."""
    rng = np.random.default_rng(seed)
    vecs = []
    for size in cluster_sizes:
        center = rng.standard_normal(dim)
        center = center / np.linalg.norm(center)
        noise = rng.standard_normal((size, dim)) * cluster_spread
        cluster = center + noise
        norms = np.linalg.norm(cluster, axis=1, keepdims=True)
        vecs.append(cluster / norms)
    return np.vstack(vecs).astype(np.float64)


class TestComputeMetrics:
    def test_output_shapes(self):
        emb = _make_embeddings(30)
        met = compute_metrics(emb, PARAMS)
        assert met.n_windows == 30
        assert met.cosine_distance.shape == (30,)
        assert met.velocity.shape == (30,)
        assert met.acceleration.shape == (30,)
        assert met.jerk.shape == (30,)
        assert met.cosine_similarity.shape == (30,)
        assert met.ema_centroids.shape == (30, 1536)
        assert met.boundary_flags.shape == (30,)

    def test_first_distance_is_nan(self):
        """The first window has no predecessor → distance = NaN."""
        emb = _make_embeddings(20)
        met = compute_metrics(emb, PARAMS)
        assert np.isnan(met.cosine_distance[0])
        assert np.isnan(met.euclidean_distance[0])

    def test_cosine_distance_range(self):
        """Cosine distance ∈ [0, 2] for normalised vectors."""
        emb = _make_embeddings(30)
        met = compute_metrics(emb, PARAMS)
        valid = met.cosine_distance[~np.isnan(met.cosine_distance)]
        assert np.all(valid >= 0)
        assert np.all(valid <= 2)

    def test_cumulative_path_monotonic(self):
        """Cumulative path length should be non-decreasing."""
        emb = _make_embeddings(30)
        met = compute_metrics(emb, PARAMS)
        diffs = np.diff(met.cumulative_path)
        assert np.all(diffs >= -1e-10)  # allow tiny float errors

    def test_boundary_detection_on_clusters(self):
        """Clustered embeddings should produce boundaries at transitions."""
        emb = _make_clustered_embeddings([20, 20, 20], cluster_spread=0.01)
        p = get_params(**{"boundary.k_sigma": 1.5, "boundary.min_samples": 3})
        met = compute_metrics(emb, p)
        # Should detect at least 1 boundary (ideally 2, at indices ~20 and ~40)
        assert met.n_boundaries >= 1

    def test_identical_embeddings_zero_velocity(self):
        """If all embeddings are identical, velocity should be ~0."""
        vec = np.random.default_rng(0).standard_normal(1536)
        vec = vec / np.linalg.norm(vec)
        emb = np.tile(vec, (20, 1))
        met = compute_metrics(emb, PARAMS)
        valid_v = met.velocity[~np.isnan(met.velocity)]
        assert np.allclose(valid_v, 0, atol=1e-10)


class TestPairwiseCosine:
    def test_square_output(self):
        emb = _make_embeddings(10)
        mat = pairwise_cosine_matrix(emb)
        assert mat.shape == (10, 10)

    def test_diagonal_ones(self):
        """Diagonal should be ~1 (self-similarity)."""
        emb = _make_embeddings(10)
        mat = pairwise_cosine_matrix(emb)
        assert np.allclose(np.diag(mat), 1.0, atol=1e-6)


class TestSemanticCloudStats:
    def test_output_keys(self):
        emb = _make_embeddings(20)
        stats = semantic_cloud_stats(emb)
        assert "mean_pairwise_sim" in stats
        assert "std_pairwise_sim" in stats
        assert "min_pairwise_sim" in stats
        assert "max_pairwise_sim" in stats

    def test_random_vectors_near_orthogonal(self):
        """Random high-dim unit vectors should have mean similarity near 0."""
        emb = _make_embeddings(100, dim=1536)
        stats = semantic_cloud_stats(emb)
        assert abs(stats["mean_pairwise_sim"]) < 0.1
