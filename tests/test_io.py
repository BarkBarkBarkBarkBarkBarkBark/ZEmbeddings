"""
Tests for the I/O module.

Tests YAML and Markdown output formatting, sparkline generation,
and file reading. All file I/O uses tmp_path — never writes to
project directories.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
import yaml
from zembeddings.params import get_params
from zembeddings.io import (
    write_metrics_yaml,
    write_report_markdown,
    read_transcript,
    _sparkline,
    _numpy_to_python,
    _fmt,
)


def _sample_metrics_dict():
    """Build a minimal metrics_dict matching pipeline output format."""
    N = 20
    return {
        "summary": {
            "n_windows": N,
            "n_tokens": 100,
            "mean_velocity": 0.05,
            "std_velocity": 0.02,
            "n_boundaries": 2,
            "n_returns": 1,
            "total_path_length": 0.45,
            "kalman_mode": "scalar",
            "kalman_violations": 3,
            "kalman_accel_violations": 1,
            "cloud_mean_sim": 0.42,
            "cloud_std_sim": 0.08,
            "boundary_indices": [5, 15],
            "return_indices": [18],
        },
        "timeseries": {
            "cosine_distance": [float(np.nan)] + [0.02 + 0.01 * i for i in range(N - 1)],
            "euclidean_distance": [float(np.nan)] + [0.1 * i for i in range(N - 1)],
            "velocity": [float(np.nan)] + [0.05] * (N - 1),
            "acceleration": [float(np.nan), float(np.nan)] + [0.01] * (N - 2),
            "jerk": [float(np.nan)] * 3 + [0.001] * (N - 3),
            "cosine_similarity": [float(np.nan)] + [0.95] * (N - 1),
            "cosine_sim_d1": [float(np.nan)] * 2 + [0.0] * (N - 2),
            "cosine_sim_d2": [float(np.nan)] * 3 + [0.0] * (N - 3),
            "ema_drift": [0.01] * N,
            "cumulative_path": list(np.cumsum([0.02] * N)),
            "kalman_innovation": [float(np.nan)] + [0.01] * (N - 1),
            "kalman_mahalanobis": [float(np.nan)] + [1.5] * (N - 1),
            "kalman_accel_innovation": [float(np.nan)] * 2 + [0.005] * (N - 2),
            "kalman_accel_mahalanobis": [float(np.nan)] * 2 + [1.2] * (N - 2),
            "boundary_flags": [False] * 5 + [True] + [False] * 9 + [True] + [False] * 4,
            "return_flags": [False] * 18 + [True] + [False],
            "return_cluster_id": [-1] * 18 + [0] + [-1],
            "fixation_flags": [False] * N,
            "cloud_valid": [False] + [True] * (N - 1),
        },
    }


class TestNumpyToPython:
    def test_ndarray(self):
        result = _numpy_to_python(np.array([1.0, 2.0]))
        assert isinstance(result, list)
        assert result == [1.0, 2.0]

    def test_np_int(self):
        result = _numpy_to_python(np.int64(42))
        assert isinstance(result, int)
        assert result == 42

    def test_np_float(self):
        result = _numpy_to_python(np.float32(3.14))
        assert isinstance(result, float)

    def test_np_nan_to_none(self):
        result = _numpy_to_python(np.float64(np.nan))
        assert result is None

    def test_np_bool(self):
        result = _numpy_to_python(np.bool_(True))
        assert isinstance(result, bool)
        assert result is True

    def test_nested_dict(self):
        data = {"a": np.array([1, 2]), "b": {"c": np.int64(3)}}
        result = _numpy_to_python(data)
        assert result == {"a": [1, 2], "b": {"c": 3}}

    def test_list_with_numpy(self):
        data = [np.float64(1.0), np.int32(2), "hello"]
        result = _numpy_to_python(data)
        assert result == [1.0, 2, "hello"]


class TestSparkline:
    def test_basic(self):
        spark = _sparkline([0.0, 0.5, 1.0])
        assert isinstance(spark, str)
        assert len(spark) > 0

    def test_empty(self):
        spark = _sparkline([])
        assert spark == ""

    def test_all_nan(self):
        spark = _sparkline([float("nan"), float("nan")])
        assert spark == ""

    def test_constant(self):
        spark = _sparkline([0.5] * 10)
        # All same value → all same character
        assert len(set(spark)) == 1

    def test_with_nan(self):
        spark = _sparkline([0.0, float("nan"), 1.0])
        assert isinstance(spark, str)
        assert len(spark) > 0


class TestFmt:
    def test_normal_value(self):
        assert _fmt(3.14159, decimals=2) == "3.14"

    def test_none(self):
        assert _fmt(None) == "—"

    def test_nan(self):
        assert _fmt(float("nan")) == "—"


class TestWriteMetricsYaml:
    def test_creates_file(self, tmp_path):
        metrics = _sample_metrics_dict()
        p = get_params(**{"paths.results_metrics": str(tmp_path)})
        path = write_metrics_yaml(metrics, p, "test_experiment", out_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".yaml"

    def test_valid_yaml(self, tmp_path):
        metrics = _sample_metrics_dict()
        p = get_params(**{"paths.results_metrics": str(tmp_path)})
        path = write_metrics_yaml(metrics, p, "test_experiment", out_dir=tmp_path)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["experiment"] == "test_experiment"
        assert "summary" in data
        assert "timeseries" in data

    def test_summary_values(self, tmp_path):
        metrics = _sample_metrics_dict()
        p = get_params(**{"paths.results_metrics": str(tmp_path)})
        path = write_metrics_yaml(metrics, p, "test_experiment", out_dir=tmp_path)
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["summary"]["n_boundaries"] == 2
        assert data["summary"]["n_returns"] == 1


class TestWriteReportMarkdown:
    def test_creates_file(self, tmp_path):
        metrics = _sample_metrics_dict()
        p = get_params(**{"paths.results_reports": str(tmp_path)})
        path = write_report_markdown(metrics, p, "test_experiment", out_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".md"

    def test_contains_headers(self, tmp_path):
        metrics = _sample_metrics_dict()
        p = get_params(**{"paths.results_reports": str(tmp_path)})
        path = write_report_markdown(metrics, p, "test_experiment", out_dir=tmp_path)
        content = path.read_text()
        assert "# Experiment Report: test_experiment" in content
        assert "## Summary" in content
        assert "## Trajectory Sparklines" in content

    def test_contains_sparklines(self, tmp_path):
        metrics = _sample_metrics_dict()
        p = get_params(**{"paths.results_reports": str(tmp_path)})
        path = write_report_markdown(metrics, p, "test_experiment", out_dir=tmp_path)
        content = path.read_text()
        assert "Velocity" in content
        assert "Acceleration" in content

    def test_contains_boundary_table(self, tmp_path):
        metrics = _sample_metrics_dict()
        p = get_params(**{"paths.results_reports": str(tmp_path)})
        path = write_report_markdown(metrics, p, "test_experiment", out_dir=tmp_path)
        content = path.read_text()
        assert "Detected Boundaries" in content


class TestReadTranscript:
    def test_reads_file(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("Hello, world!", encoding="utf-8")
        result = read_transcript(txt)
        assert result == "Hello, world!"

    def test_unicode(self, tmp_path):
        txt = tmp_path / "unicode.txt"
        txt.write_text("Ähello wörld 日本語", encoding="utf-8")
        result = read_transcript(txt)
        assert "日本語" in result
