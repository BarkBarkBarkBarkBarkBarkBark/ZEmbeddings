"""
Tests for the embeddings module.

Uses mock/patch to avoid real API calls or loading sentence-transformers
models. Tests caching logic, backend dispatch, and shape contracts.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
from zembeddings.params import get_params
from zembeddings.embeddings import (
    embed_texts,
    _cache_key,
    _cache_path,
    load_cached_embeddings,
)


SAMPLE_TEXTS = ["hello world", "the quick brown fox", "semantic embedding space"]


class TestCacheKey:
    def test_deterministic(self):
        k1 = _cache_key(SAMPLE_TEXTS, "model-a", 1536)
        k2 = _cache_key(SAMPLE_TEXTS, "model-a", 1536)
        assert k1 == k2

    def test_different_model(self):
        k1 = _cache_key(SAMPLE_TEXTS, "model-a", 1536)
        k2 = _cache_key(SAMPLE_TEXTS, "model-b", 1536)
        assert k1 != k2

    def test_different_dims(self):
        k1 = _cache_key(SAMPLE_TEXTS, "model-a", 1536)
        k2 = _cache_key(SAMPLE_TEXTS, "model-a", 256)
        assert k1 != k2

    def test_different_texts(self):
        k1 = _cache_key(["hello"], "model-a", 1536)
        k2 = _cache_key(["world"], "model-a", 1536)
        assert k1 != k2

    def test_key_length(self):
        k = _cache_key(SAMPLE_TEXTS, "model-a", 1536)
        assert len(k) == 16  # SHA-256 truncated to 16 hex chars


class TestCachePath:
    def test_path_format(self):
        p = _cache_path("/tmp/cache", "abcdef1234567890")
        assert p == Path("/tmp/cache/emb_abcdef1234567890.npz")


class TestEmbedTextsLocal:
    """Test embed_texts with the local backend using a mocked model."""

    @patch("zembeddings.embeddings._get_local_model")
    def test_local_backend_shapes(self, mock_get_model, tmp_path):
        """Local backend should return correct shapes."""
        dim = 384  # MiniLM-L6-v2 native dim
        n = len(SAMPLE_TEXTS)
        fake_embeddings = np.random.randn(n, dim).astype(np.float32)
        # Normalise
        fake_embeddings /= np.linalg.norm(fake_embeddings, axis=1, keepdims=True)

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings
        mock_get_model.return_value = mock_model

        p = get_params(**{
            "model.backend": "local",
            "model.local_model": "all-MiniLM-L6-v2",
            "model.dimensions_full": 384,
            "model.dimensions_reduced": 128,
            "paths.data_processed": str(tmp_path),
        })

        result = embed_texts(SAMPLE_TEXTS, p, use_cache=False)

        assert "full" in result
        assert "reduced" in result
        assert "texts" in result
        assert result["full"].shape == (n, dim)
        assert result["reduced"].shape[0] == n
        assert result["texts"] == SAMPLE_TEXTS

    @patch("zembeddings.embeddings._get_local_model")
    def test_local_caching(self, mock_get_model, tmp_path):
        """Cache should be written and loaded on second call."""
        dim = 384
        n = len(SAMPLE_TEXTS)
        fake_embeddings = np.random.randn(n, dim).astype(np.float32)
        fake_embeddings /= np.linalg.norm(fake_embeddings, axis=1, keepdims=True)

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings
        mock_get_model.return_value = mock_model

        p = get_params(**{
            "model.backend": "local",
            "model.local_model": "test-model",
            "model.dimensions_full": 384,
            "model.dimensions_reduced": 128,
            "paths.data_processed": str(tmp_path),
        })

        # First call — should embed and cache
        r1 = embed_texts(SAMPLE_TEXTS, p, use_cache=True)
        assert mock_model.encode.call_count == 1

        # Second call — should load from cache (no new encode call)
        r2 = embed_texts(SAMPLE_TEXTS, p, use_cache=True)
        # Still only 1 call because cache was hit
        assert mock_model.encode.call_count == 1
        np.testing.assert_array_almost_equal(r1["full"], r2["full"])

    @patch("zembeddings.embeddings._get_local_model")
    def test_reduced_truncation(self, mock_get_model, tmp_path):
        """Reduced dims should be truncated + renormalised."""
        dim = 384
        n = len(SAMPLE_TEXTS)
        fake_embeddings = np.random.randn(n, dim).astype(np.float32)
        fake_embeddings /= np.linalg.norm(fake_embeddings, axis=1, keepdims=True)

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_embeddings
        mock_get_model.return_value = mock_model

        p = get_params(**{
            "model.backend": "local",
            "model.local_model": "test-model",
            "model.dimensions_full": 384,
            "model.dimensions_reduced": 128,
            "paths.data_processed": str(tmp_path),
        })

        result = embed_texts(SAMPLE_TEXTS, p, use_cache=False)
        reduced = result["reduced"]
        assert reduced.shape == (n, 128)
        # Check normalisation
        norms = np.linalg.norm(reduced, axis=1)
        np.testing.assert_array_almost_equal(norms, 1.0, decimal=5)


class TestLoadCachedEmbeddings:
    def test_returns_none_when_missing(self, tmp_path):
        result = load_cached_embeddings(tmp_path, ["test"], "model", 384)
        assert result is None

    def test_returns_array_when_cached(self, tmp_path):
        texts = ["hello", "world"]
        model = "test-model"
        dims = 384
        key = _cache_key(texts, model, dims)
        path = _cache_path(tmp_path, key)
        fake = np.random.randn(2, dims).astype(np.float32)
        np.savez_compressed(path, embeddings=fake)

        loaded = load_cached_embeddings(tmp_path, texts, model, dims)
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded, fake)
