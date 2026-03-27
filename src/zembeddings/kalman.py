"""
Kalman filter for semantic trajectory prediction.
==================================================

This module implements a **linear** constant-velocity Kalman filter with
linear observations. For that model the filter is **exact**. A full **Extended
Kalman Filter (EKF)** reduces to the same update when ``f`` and ``h`` are
linear; EKF is needed for nonlinear ``h`` (e.g. sphere-constrained embeddings).
**update_gain_scale** (default 1.0) scales the correction ``K y`` for interactive
tuning; at 1.0 behaviour matches the standard filter.

Implements two modes:

Scalar mode
    Tracks the **cosine distance** time-series as a 1-D signal.
    State = [distance, velocity]ᵀ.  Lightweight and interpretable.
    A spike in the Mahalanobis distance of the innovation signals a
    semantic boundary.

Vector mode
    Tracks the **reduced embedding** trajectory as a d-dimensional
    signal (default d = 256).  State = [embedding, velocity]ᵀ with
    diagonal covariance.  Richer, captures directional information.

Both modes use a constant-velocity transition model, which is the
simplest assumption: "the speaker will keep going in the same
semantic direction."  Violations of that assumption are exactly what
we want to detect.

Mathematical formulation
------------------------

State:      x_t = [e_t, v_t]ᵀ       (position + velocity)
Transition: F   = [[I, I], [0, I]]   (constant velocity)
Process:    w_t ~ N(0, Q)
Observe:    z_t = H x_t + ν_t        H = [I, 0]
Measure:    ν_t ~ N(0, R)

Predict:    x̂_{t|t-1} = F x_{t-1|t-1}
            P_{t|t-1}  = F P_{t-1|t-1} Fᵀ + Q

Innovation: y_t = z_t − H x̂_{t|t-1}
            S_t = H P_{t|t-1} Hᵀ + R

Kalman gain: K_t = P_{t|t-1} Hᵀ S_t⁻¹

Update:     x̂_{t|t} = x̂_{t|t-1} + K_t y_t
            P_{t|t}  = (I − K_t H) P_{t|t-1}

Mahalanobis: d_M = √(y_tᵀ S_t⁻¹ y_t)

References
----------
- Kalman, R. E. (1960). "A New Approach to Linear Filtering and
  Prediction Problems." J. Basic Eng., 82(1), 35–45.
  doi:10.1115/1.3662552
- Adams, R. P. & MacKay, D. J. C. (2007). "Bayesian Online
  Changepoint Detection." arXiv:0710.3742
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Result containers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class KalmanResult:
    """Output of a Kalman filter run over a trajectory."""
    mode: str                          # "scalar" or "vector"
    innovations: np.ndarray            # raw innovation at each step
    innovation_norms: np.ndarray       # ||innovation|| or abs(innovation)
    mahalanobis_distances: np.ndarray  # d_M at each step
    predicted_states: np.ndarray       # x̂_{t|t-1}  (position component)
    filtered_states: np.ndarray        # x̂_{t|t}    (position component)
    violation_flags: np.ndarray        # bool: d_M > threshold
    threshold: float                   # the Mahalanobis threshold used
    n_violations: int = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Scalar Kalman filter  (on cosine-distance time series)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_scalar_kalman(
    cosine_distances: np.ndarray,
    params: dict[str, Any],
) -> KalmanResult:
    """Run a 2-D Kalman filter on the 1-D cosine distance series.

    State vector: x = [distance, velocity]ᵀ  (2×1)
    """
    kp = params["kalman"]
    q_scale = kp["process_noise_scale"]
    r_scale = kp["measurement_noise_scale"]
    p0_scale = kp["initial_covariance_scale"]
    threshold = kp["innovation_threshold"]
    gain_scale = float(kp.get("update_gain_scale", 1.0))
    gain_scale = max(0.0, min(gain_scale, 4.0))

    N = len(cosine_distances)

    # Transition matrix (constant velocity)
    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    # Observation matrix
    H = np.array([[1.0, 0.0]])
    # Process noise
    Q = q_scale * np.eye(2)
    # Measurement noise
    R = np.array([[r_scale]])
    # Initial state
    x = np.array([0.0, 0.0])  # [distance, velocity]
    P = p0_scale * np.eye(2)

    # Output arrays
    innovations = np.full(N, np.nan)
    mahal = np.full(N, np.nan)
    predicted = np.full(N, np.nan)
    filtered = np.full(N, np.nan)
    violations = np.zeros(N, dtype=bool)

    for t in range(N):
        z = cosine_distances[t]
        if np.isnan(z):
            predicted[t] = np.nan
            filtered[t] = np.nan
            continue

        # ── Predict ───────────────────────────────────────────────────
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # ── Innovation ────────────────────────────────────────────────
        y = z - (H @ x_pred)[0]
        S = (H @ P_pred @ H.T + R)[0, 0]

        innovations[t] = y
        predicted[t] = x_pred[0]

        # Mahalanobis distance (scalar: |y| / √S)
        d_m = abs(y) / np.sqrt(S) if S > 0 else 0.0
        mahal[t] = d_m

        if d_m > threshold:
            violations[t] = True

        # ── Update ────────────────────────────────────────────────────
        K = P_pred @ H.T / S          # 2×1
        Ks = gain_scale * K
        x = x_pred + Ks.flatten() * y
        P = (np.eye(2) - Ks @ H) @ P_pred

        filtered[t] = x[0]

    return KalmanResult(
        mode="scalar",
        innovations=innovations,
        innovation_norms=np.abs(innovations),
        mahalanobis_distances=mahal,
        predicted_states=predicted,
        filtered_states=filtered,
        violation_flags=violations,
        threshold=threshold,
        n_violations=int(violations.sum()),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Vector Kalman filter  (on reduced-embedding trajectory)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_vector_kalman(
    embeddings: np.ndarray,
    params: dict[str, Any],
) -> KalmanResult:
    """Run a Kalman filter on the d-dimensional embedding trajectory.

    Uses **diagonal covariance** for computational efficiency when
    d is large (e.g. 256).

    State vector: x = [e (d,), v (d,)]  →  2d-dimensional
    All covariances are stored as 1-D vectors (diagonal only).
    """
    kp = params["kalman"]
    q_scale = kp["process_noise_scale"]
    r_scale = kp["measurement_noise_scale"]
    p0_scale = kp["initial_covariance_scale"]
    threshold = kp["innovation_threshold"]
    gain_scale = float(kp.get("update_gain_scale", 1.0))
    gain_scale = max(0.0, min(gain_scale, 4.0))

    N, dim = embeddings.shape
    state_dim = 2 * dim  # [position, velocity]

    # Diagonal covariance vectors (length state_dim)
    Q_diag = q_scale * np.ones(state_dim)
    R_diag = r_scale * np.ones(dim)
    P_diag = p0_scale * np.ones(state_dim)

    # State: [position (dim), velocity (dim)]
    x = np.zeros(state_dim)
    x[:dim] = embeddings[0]  # initialise position to first embedding

    # Output arrays
    innovations = np.full((N, dim), np.nan)
    innov_norms = np.full(N, np.nan)
    mahal = np.full(N, np.nan)
    predicted = np.full((N, dim), np.nan)
    filtered = np.full((N, dim), np.nan)
    violations = np.zeros(N, dtype=bool)

    filtered[0] = embeddings[0]

    for t in range(1, N):
        z = embeddings[t]

        # ── Predict (constant velocity, diagonal cov) ─────────────────
        # x_pred = F @ x  but with diagonal structure:
        #   position_pred = position + velocity
        #   velocity_pred = velocity
        x_pred = x.copy()
        x_pred[:dim] = x[:dim] + x[dim:]   # pos += vel

        # P_pred = F P Fᵀ + Q   (diagonal approx)
        #   P_pos_pred = P_pos + 2*P_vel + P_vel  (simplified: P_pos + P_vel for diag)
        #   Actually for diagonal, F P Fᵀ mixes pos and vel:
        #     P_pos_pred ≈ P_pos + P_vel   (dominant terms)
        #     P_vel_pred = P_vel
        P_pred = P_diag.copy()
        P_pred[:dim] = P_diag[:dim] + P_diag[dim:] + Q_diag[:dim]
        P_pred[dim:] = P_diag[dim:] + Q_diag[dim:]

        # ── Innovation ────────────────────────────────────────────────
        y = z - x_pred[:dim]
        innovations[t] = y
        innov_norms[t] = float(np.linalg.norm(y))

        # Innovation covariance (diagonal): S = P_pos_pred + R
        S_diag = P_pred[:dim] + R_diag

        # Mahalanobis distance (diagonal covariance):
        #   d_M = sqrt( Σ  y_i² / S_i )
        d_m = float(np.sqrt(np.sum(y ** 2 / S_diag)))
        mahal[t] = d_m

        predicted[t] = x_pred[:dim]

        if d_m > threshold:
            violations[t] = True

        # ── Update (diagonal Kalman gain) ─────────────────────────────
        # K_pos = P_pos_pred / S   (element-wise)
        # K_vel = P_vel_pred / S   (element-wise, assuming cross-terms)
        K_pos = P_pred[:dim] / S_diag
        K_vel = P_pred[dim:] / S_diag

        x[:dim] = x_pred[:dim] + gain_scale * (K_pos * y)
        x[dim:] = x_pred[dim:] + gain_scale * (K_vel * y)

        P_diag[:dim] = (1.0 - gain_scale * K_pos) * P_pred[:dim]
        P_diag[dim:] = (1.0 - gain_scale * K_vel) * P_pred[dim:]

        filtered[t] = x[:dim]

    return KalmanResult(
        mode="vector",
        innovations=innovations,
        innovation_norms=innov_norms,
        mahalanobis_distances=mahal,
        predicted_states=predicted,
        filtered_states=filtered,
        violation_flags=violations,
        threshold=threshold,
        n_violations=int(violations.sum()),
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Dispatcher
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_kalman(
    embeddings_full: np.ndarray,
    embeddings_reduced: np.ndarray,
    cosine_distances: np.ndarray,
    params: dict[str, Any],
) -> KalmanResult:
    """Run the Kalman filter in the mode specified by params.

    Parameters
    ----------
    embeddings_full : ndarray (N, 1536)
    embeddings_reduced : ndarray (N, 256)
    cosine_distances : ndarray (N,)
        Pre-computed cosine distance series.
    params : dict
        Full PARAMS dict.

    Returns
    -------
    KalmanResult
    """
    mode = params["kalman"]["mode"]
    if mode == "scalar":
        return run_scalar_kalman(cosine_distances, params)
    elif mode == "vector":
        return run_vector_kalman(embeddings_reduced, params)
    else:
        raise ValueError(f"Unknown Kalman mode: {mode!r}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Kalman on acceleration (2nd derivative of cosine distance)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_acceleration_kalman(
    acceleration: np.ndarray,
    params: dict[str, Any],
) -> KalmanResult:
    """Run a scalar Kalman filter on the acceleration (2nd derivative).

    This models the *rate of change of velocity* as a constant-value
    process.  A spike in the Mahalanobis distance means the acceleration
    itself is surprising — analogous to a neuron that fires when the
    *context is changing*, not just when the position shifts.

    Think of it as: the regular Kalman on distance detects topic shifts;
    this one detects the *onset and offset* of topic shifts — the event
    boundary signal that could cue hippocampal resetting.

    State vector: x = [acceleration, jerk]ᵀ  (2×1)
    """
    kp = params["kalman"]
    q_scale = kp["process_noise_scale"]
    r_scale = kp["measurement_noise_scale"]
    p0_scale = kp["initial_covariance_scale"]
    threshold = kp["innovation_threshold"]
    gain_scale = float(kp.get("update_gain_scale", 1.0))
    gain_scale = max(0.0, min(gain_scale, 4.0))

    N = len(acceleration)

    # Transition matrix (constant-velocity model on acceleration)
    F = np.array([[1.0, 1.0],
                  [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = q_scale * np.eye(2)
    R = np.array([[r_scale]])
    x = np.array([0.0, 0.0])
    P = p0_scale * np.eye(2)

    innovations = np.full(N, np.nan)
    mahal = np.full(N, np.nan)
    predicted = np.full(N, np.nan)
    filtered = np.full(N, np.nan)
    violations = np.zeros(N, dtype=bool)

    for t in range(N):
        z = acceleration[t]
        if np.isnan(z):
            predicted[t] = np.nan
            filtered[t] = np.nan
            continue

        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        y = z - (H @ x_pred)[0]
        S = (H @ P_pred @ H.T + R)[0, 0]

        innovations[t] = y
        predicted[t] = x_pred[0]

        d_m = abs(y) / np.sqrt(S) if S > 0 else 0.0
        mahal[t] = d_m

        if d_m > threshold:
            violations[t] = True

        K = P_pred @ H.T / S
        Ks = gain_scale * K
        x = x_pred + Ks.flatten() * y
        P = (np.eye(2) - Ks @ H) @ P_pred

        filtered[t] = x[0]

    return KalmanResult(
        mode="acceleration",
        innovations=innovations,
        innovation_norms=np.abs(innovations),
        mahalanobis_distances=mahal,
        predicted_states=predicted,
        filtered_states=filtered,
        violation_flags=violations,
        threshold=threshold,
        n_violations=int(violations.sum()),
    )
