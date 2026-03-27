"""
Tests for the Kalman filter module.

Uses synthetic distance / embedding series — no API calls.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
from zembeddings.params import PARAMS, get_params
from zembeddings.kalman import (
    run_scalar_kalman,
    run_vector_kalman,
    run_kalman,
    run_acceleration_kalman,
)


def _smooth_then_spike(n: int = 50, spike_at: int = 25) -> np.ndarray:
    """Cosine distance series: smooth baseline with a spike."""
    rng = np.random.default_rng(42)
    series = 0.02 + rng.normal(0, 0.005, n)
    series = np.abs(series)
    series[0] = np.nan  # no predecessor for first window
    # Inject a spike
    series[spike_at] = 0.5
    return series


class TestScalarKalman:
    def test_output_shape(self):
        series = _smooth_then_spike(50)
        result = run_scalar_kalman(series, PARAMS)
        assert result.mode == "scalar"
        assert result.innovations.shape == (50,)
        assert result.mahalanobis_distances.shape == (50,)
        assert result.violation_flags.shape == (50,)

    def test_spike_detected(self):
        """The spike at index 25 should trigger a Kalman violation."""
        series = _smooth_then_spike(50, spike_at=25)
        result = run_scalar_kalman(series, PARAMS)
        assert result.violation_flags[25] == True
        assert result.n_violations >= 1

    def test_smooth_no_violations(self):
        """A smooth series should produce no or very few violations."""
        series = np.full(50, 0.02)
        series[0] = np.nan
        p = get_params(**{"kalman.innovation_threshold": 5.0})
        result = run_scalar_kalman(series, p)
        assert result.n_violations <= 2  # allow edge effects

    def test_nan_handling(self):
        """NaN in the input should propagate gracefully."""
        series = _smooth_then_spike(30)
        series[5] = np.nan
        result = run_scalar_kalman(series, PARAMS)
        assert np.isnan(result.innovations[5])


class TestVectorKalman:
    def test_output_shape(self):
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((30, 256))
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        result = run_vector_kalman(emb, PARAMS)
        assert result.mode == "vector"
        assert result.innovations.shape == (30, 256)
        assert result.mahalanobis_distances.shape == (30,)

    def test_abrupt_direction_change(self):
        """An abrupt direction change should trigger a violation."""
        rng = np.random.default_rng(42)
        dim = 256
        # Build a trajectory: smooth then sudden jump
        emb = np.zeros((40, dim))
        direction1 = rng.standard_normal(dim)
        direction1 /= np.linalg.norm(direction1)
        direction2 = rng.standard_normal(dim)
        direction2 /= np.linalg.norm(direction2)

        for t in range(20):
            emb[t] = direction1 + rng.normal(0, 0.01, dim)
        for t in range(20, 40):
            emb[t] = direction2 + rng.normal(0, 0.01, dim)

        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

        p = get_params(**{"kalman.mode": "vector", "kalman.innovation_threshold": 2.0})
        result = run_vector_kalman(emb, p)
        # Should detect violation around index 20
        assert result.n_violations >= 1
        # At least one violation should be near the transition
        violation_indices = np.where(result.violation_flags)[0]
        assert any(18 <= idx <= 25 for idx in violation_indices)


class TestKalmanDispatcher:
    def test_scalar_mode(self):
        series = _smooth_then_spike(30)
        rng = np.random.default_rng(42)
        emb_full = rng.standard_normal((30, 1536)).astype(np.float32)
        emb_red = rng.standard_normal((30, 256)).astype(np.float32)
        result = run_kalman(emb_full, emb_red, series, PARAMS)
        assert result.mode == "scalar"

    def test_vector_mode(self):
        series = _smooth_then_spike(30)
        rng = np.random.default_rng(42)
        emb_full = rng.standard_normal((30, 1536)).astype(np.float32)
        emb_red = rng.standard_normal((30, 256)).astype(np.float32)
        p = get_params(**{"kalman.mode": "vector"})
        result = run_kalman(emb_full, emb_red, series, p)
        assert result.mode == "vector"


class TestAccelerationKalman:
    """Tests for the 2nd-derivative Kalman filter (context-change neuron)."""

    def test_output_shape(self):
        """Output arrays match input length."""
        series = _smooth_then_spike(50)
        result = run_acceleration_kalman(series, PARAMS)
        assert result.mode == "acceleration"
        assert result.innovations.shape == (50,)
        assert result.mahalanobis_distances.shape == (50,)
        assert result.violation_flags.shape == (50,)

    def test_first_value_nan(self):
        """First value should be NaN (distance[0] is NaN → accel[0] is NaN)."""
        series = _smooth_then_spike(50)
        result = run_acceleration_kalman(series, PARAMS)
        assert np.isnan(result.innovations[0])

    def test_spike_onset_detected(self):
        """A sudden jump should produce a Mahalanobis spike on acceleration."""
        series = _smooth_then_spike(50, spike_at=25)
        p = get_params(**{"kalman.innovation_threshold": 2.0})
        result = run_acceleration_kalman(series, p)
        assert result.n_violations >= 1
        # The acceleration spike should be near the velocity spike
        violation_indices = np.where(result.violation_flags)[0]
        assert any(24 <= idx <= 27 for idx in violation_indices)

    def test_smooth_series_no_violations(self):
        """Constant series → zero acceleration → no violations."""
        series = np.full(60, 0.02)
        series[0] = np.nan
        p = get_params(**{"kalman.innovation_threshold": 5.0})
        result = run_acceleration_kalman(series, p)
        assert result.n_violations == 0

    def test_linear_trend_no_violations(self):
        """Linear increase in distance → constant velocity → zero accel."""
        series = np.linspace(0.01, 0.20, 60)
        series[0] = np.nan
        p = get_params(**{"kalman.innovation_threshold": 5.0})
        result = run_acceleration_kalman(series, p)
        # Constant velocity ⇒ ~0 acceleration ⇒ no violations
        assert result.n_violations <= 1  # allow small edge effect

    def test_double_spike_two_events(self):
        """Two spikes should produce at least two violation clusters."""
        rng = np.random.default_rng(42)
        series = 0.02 + np.abs(rng.normal(0, 0.003, 80))
        series[0] = np.nan
        series[25] = 0.5
        series[55] = 0.5
        p = get_params(**{"kalman.innovation_threshold": 2.0})
        result = run_acceleration_kalman(series, p)
        viol = np.where(result.violation_flags)[0]
        # Should have violations in two distinct regions
        assert len(viol) >= 2
        assert any(v < 40 for v in viol) and any(v > 40 for v in viol)

    def test_nan_propagation(self):
        """NaN mid-series should not crash; output NaN at that position."""
        series = _smooth_then_spike(40)
        series[10] = np.nan
        result = run_acceleration_kalman(series, PARAMS)
        assert result.innovations.shape == (40,)
        # Positions depending on NaN should propagate NaN
        assert np.isnan(result.innovations[10])
