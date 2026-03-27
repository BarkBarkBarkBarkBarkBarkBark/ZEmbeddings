"""
Tests for the pipeline module.

Uses mock/patch to avoid real API calls and isolate the orchestration
logic. Tests the full pipeline flow with synthetic data.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
from zembeddings.params import get_params
from zembeddings.pipeline import run_pipeline, PipelineResult


def _mock_embed_texts(texts, params, *, use_cache=True):
    """Return fake embeddings matching the expected shape."""
    n = len(texts)
    dims_full = params["model"]["dimensions_full"]
    dims_reduced = params["model"]["dimensions_reduced"]
    rng = np.random.default_rng(42)

    full = rng.standard_normal((n, dims_full)).astype(np.float32)
    full /= np.linalg.norm(full, axis=1, keepdims=True)

    reduced = rng.standard_normal((n, dims_reduced)).astype(np.float32)
    reduced /= np.linalg.norm(reduced, axis=1, keepdims=True)

    return {"full": full, "reduced": reduced, "texts": texts}


SAMPLE_TEXT = (
    "I've always loved dogs. Golden retrievers are my favorite breed "
    "because of their gentle temperament and loyalty. When I was growing "
    "up we had a golden named Biscuit who would follow me everywhere. "
    "Dogs really are remarkable companions. They can sense your emotions "
    "and respond with genuine affection. Training a puppy takes patience "
    "but the bond you build is incredibly rewarding. Walking a dog every "
    "morning also keeps you healthy and connected to your neighborhood. "
    "Speaking of travel, I recently took a flight to Tokyo. The Boeing "
    "787 Dreamliner is an incredible piece of engineering. The composite "
    "fuselage allows for higher cabin pressure and larger windows. "
    "Modern jet engines are remarkably efficient compared to earlier "
    "turbofans."
)


class TestRunPipeline:
    @patch("zembeddings.pipeline.embed_texts", side_effect=_mock_embed_texts)
    def test_returns_pipeline_result(self, mock_embed, tmp_path):
        p = get_params(**{
            "paths.results_metrics": str(tmp_path / "metrics"),
            "paths.results_reports": str(tmp_path / "reports"),
            "paths.data_processed": str(tmp_path / "processed"),
        })
        result = run_pipeline(
            SAMPLE_TEXT, p,
            experiment_name="test_pipeline",
            write_outputs=True,
        )
        assert isinstance(result, PipelineResult)
        assert result.experiment_name == "test_pipeline"

    @patch("zembeddings.pipeline.embed_texts", side_effect=_mock_embed_texts)
    def test_metrics_populated(self, mock_embed, tmp_path):
        p = get_params(**{
            "paths.results_metrics": str(tmp_path / "metrics"),
            "paths.results_reports": str(tmp_path / "reports"),
            "paths.data_processed": str(tmp_path / "processed"),
        })
        result = run_pipeline(SAMPLE_TEXT, p, write_outputs=False)
        assert result.metrics.n_windows > 0
        assert result.metrics.velocity.shape[0] == result.metrics.n_windows
        assert result.metrics.acceleration.shape[0] == result.metrics.n_windows

    @patch("zembeddings.pipeline.embed_texts", side_effect=_mock_embed_texts)
    def test_kalman_populated(self, mock_embed, tmp_path):
        p = get_params(**{
            "paths.results_metrics": str(tmp_path / "metrics"),
            "paths.results_reports": str(tmp_path / "reports"),
            "paths.data_processed": str(tmp_path / "processed"),
        })
        result = run_pipeline(SAMPLE_TEXT, p, write_outputs=False)
        assert result.kalman.mode in ("scalar", "vector")
        assert result.kalman.mahalanobis_distances.shape[0] == result.metrics.n_windows

    @patch("zembeddings.pipeline.embed_texts", side_effect=_mock_embed_texts)
    def test_kalman_accel_populated(self, mock_embed, tmp_path):
        p = get_params(**{
            "paths.results_metrics": str(tmp_path / "metrics"),
            "paths.results_reports": str(tmp_path / "reports"),
            "paths.data_processed": str(tmp_path / "processed"),
        })
        result = run_pipeline(SAMPLE_TEXT, p, write_outputs=False)
        assert result.kalman_accel.mode == "acceleration"
        assert result.kalman_accel.mahalanobis_distances.shape[0] == result.metrics.n_windows

    @patch("zembeddings.pipeline.embed_texts", side_effect=_mock_embed_texts)
    def test_cloud_stats_populated(self, mock_embed, tmp_path):
        p = get_params(**{
            "paths.results_metrics": str(tmp_path / "metrics"),
            "paths.results_reports": str(tmp_path / "reports"),
            "paths.data_processed": str(tmp_path / "processed"),
        })
        result = run_pipeline(SAMPLE_TEXT, p, write_outputs=False)
        assert "mean_pairwise_sim" in result.cloud_stats
        assert "std_pairwise_sim" in result.cloud_stats

    @patch("zembeddings.pipeline.embed_texts", side_effect=_mock_embed_texts)
    def test_writes_output_files(self, mock_embed, tmp_path):
        p = get_params(**{
            "paths.results_metrics": str(tmp_path / "metrics"),
            "paths.results_reports": str(tmp_path / "reports"),
            "paths.data_processed": str(tmp_path / "processed"),
        })
        run_pipeline(
            SAMPLE_TEXT, p,
            experiment_name="test_output",
            write_outputs=True,
        )
        metrics_files = list((tmp_path / "metrics").glob("*.yaml"))
        report_files = list((tmp_path / "reports").glob("*.md"))
        assert len(metrics_files) == 1
        assert len(report_files) == 1

    @patch("zembeddings.pipeline.embed_texts", side_effect=_mock_embed_texts)
    def test_from_file(self, mock_embed, tmp_path):
        txt_file = tmp_path / "transcript.txt"
        txt_file.write_text(SAMPLE_TEXT, encoding="utf-8")
        p = get_params(**{
            "paths.results_metrics": str(tmp_path / "metrics"),
            "paths.results_reports": str(tmp_path / "reports"),
            "paths.data_processed": str(tmp_path / "processed"),
        })
        result = run_pipeline(
            str(txt_file), p,
            write_outputs=False,
        )
        assert result.experiment_name == "transcript"  # stem of filename

    @patch("zembeddings.pipeline.embed_texts", side_effect=_mock_embed_texts)
    def test_repr(self, mock_embed, tmp_path):
        p = get_params(**{
            "paths.results_metrics": str(tmp_path / "metrics"),
            "paths.results_reports": str(tmp_path / "reports"),
            "paths.data_processed": str(tmp_path / "processed"),
        })
        result = run_pipeline(SAMPLE_TEXT, p, write_outputs=False)
        r = repr(result)
        assert "PipelineResult" in r
        assert "windows=" in r
        assert "accel_violations=" in r

    @patch("zembeddings.pipeline.embed_texts", side_effect=_mock_embed_texts)
    def test_short_text_raises(self, mock_embed, tmp_path):
        p = get_params(**{
            "window.size": 100,
            "paths.results_metrics": str(tmp_path / "metrics"),
            "paths.results_reports": str(tmp_path / "reports"),
            "paths.data_processed": str(tmp_path / "processed"),
        })
        with pytest.raises(ValueError, match="Need at least 2 windows"):
            run_pipeline("hello world", p, write_outputs=False)

    @patch("zembeddings.pipeline.embed_texts", side_effect=_mock_embed_texts)
    def test_vector_kalman_mode(self, mock_embed, tmp_path):
        p = get_params(**{
            "kalman.mode": "vector",
            "paths.results_metrics": str(tmp_path / "metrics"),
            "paths.results_reports": str(tmp_path / "reports"),
            "paths.data_processed": str(tmp_path / "processed"),
        })
        result = run_pipeline(SAMPLE_TEXT, p, write_outputs=False)
        assert result.kalman.mode == "vector"
        # Acceleration Kalman should still be scalar
        assert result.kalman_accel.mode == "acceleration"
