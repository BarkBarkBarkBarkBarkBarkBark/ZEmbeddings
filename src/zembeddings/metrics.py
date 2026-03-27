"""
Semantic trajectory metrics.
============================

Computes a comprehensive suite of metrics on a time-series of embedding
vectors, treating the sequence as a trajectory through semantic space.

All computations are **strictly causal** — at time *t* we use only
information from steps 0 … t.

Metric catalogue
----------------
1. **Cosine distance** series  d_t = 1 − e_t · e_{t−1}
2. **Euclidean distance** series  ||e_t − e_{t−1}||₂
3. **Velocity** (1st derivative of cosine distance)
4. **Acceleration** (2nd derivative)
5. **Jerk** (3rd derivative)
6. **Cosine similarity** series  s_t = e_t · e_{t−1}
7. **1st / 2nd derivative of cosine similarity**
8. **EMA centroid** + drift from centroid
9. **Cumulative path length**
10. **Boundary flags** (velocity k-sigma)
11. **Fixation / return flags** (similarity to historical centroids)
12. **Semantic cloud** validity mask

References
----------
- numpy.gradient:
  https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
- scipy.spatial.distance.cosine:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html
- Aggarwal et al. (2001). "On the Surprising Behavior of Distance
  Metrics in High Dimensional Space." ICDT 2001, LNCS 1973.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Result container
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class TrajectoryMetrics:
    """All metric arrays for a single embedding trajectory.

    Every array has length *N* (number of windows) unless noted.
    The first element of derivative arrays is NaN where a predecessor
    is required.
    """
    n_windows: int

    # ── Distance series (length N, first element = NaN) ──────────────
    cosine_distance: np.ndarray       # d_t = 1 − cos(e_t, e_{t−1})
    euclidean_distance: np.ndarray    # ||e_t − e_{t−1}||₂

    # ── Kinematic derivatives of cosine distance ─────────────────────
    velocity: np.ndarray              # 1st deriv of cosine distance
    acceleration: np.ndarray          # 2nd deriv
    jerk: np.ndarray                  # 3rd deriv

    # ── Cosine similarity + derivatives ──────────────────────────────
    cosine_similarity: np.ndarray     # s_t = e_t · e_{t−1}
    cosine_sim_d1: np.ndarray         # ṡ_t
    cosine_sim_d2: np.ndarray         # s̈_t

    # ── EMA ──────────────────────────────────────────────────────────
    ema_centroids: np.ndarray         # shape (N, dim)
    ema_drift: np.ndarray             # ||e_t − ēma_t|| cosine distance

    # ── Cumulative path length ───────────────────────────────────────
    cumulative_path: np.ndarray       # Σ d_i^cos  from i=0..t

    # ── Flags ────────────────────────────────────────────────────────
    boundary_flags: np.ndarray        # bool array
    return_flags: np.ndarray          # bool array
    return_cluster_ids: np.ndarray    # int array (-1 = no return)
    fixation_flags: np.ndarray        # bool array

    # ── Semantic cloud ───────────────────────────────────────────────
    cloud_valid: np.ndarray           # bool: similarity above floor?

    # ── Summary stats ────────────────────────────────────────────────
    mean_velocity: float = 0.0
    std_velocity: float = 0.0
    n_boundaries: int = 0
    n_returns: int = 0
    total_path_length: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Core computation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_metrics(
    embeddings: np.ndarray,
    params: dict[str, Any],
) -> TrajectoryMetrics:
    """Compute the full metric suite on an embedding trajectory.

    Parameters
    ----------
    embeddings : ndarray, shape (N, dim)
        One embedding per sliding window, in temporal order.
    params : dict
        Full PARAMS dict.

    Returns
    -------
    TrajectoryMetrics
    """
    N, dim = embeddings.shape
    method = params["derivatives"]["method"]
    edge_order = params["derivatives"]["edge_order"]
    alpha = params["ema"]["alpha"]
    k_sigma = params["boundary"]["k_sigma"]
    min_samples = params["boundary"]["min_samples"]
    sim_floor = params["semantic_cloud"]["similarity_floor"]
    return_thresh = params["semantic_cloud"]["return_threshold"]

    # ── 1. Consecutive distance series ────────────────────────────────
    cos_dist = np.full(N, np.nan, dtype=np.float64)
    euc_dist = np.full(N, np.nan, dtype=np.float64)
    cos_sim = np.full(N, np.nan, dtype=np.float64)

    for t in range(1, N):
        # OpenAI embeddings are L2-normalised → dot = cosine similarity
        dot = float(np.dot(embeddings[t], embeddings[t - 1]))
        cos_sim[t] = dot
        cos_dist[t] = 1.0 - dot
        euc_dist[t] = float(np.linalg.norm(embeddings[t] - embeddings[t - 1]))

    # ── 2. Derivatives of cosine distance ─────────────────────────────
    #       (velocity / acceleration / jerk)
    velocity = _derivative(cos_dist, method, edge_order)
    acceleration = _derivative(velocity, method, edge_order)
    jerk = _derivative(acceleration, method, edge_order)

    # ── 3. Derivatives of cosine similarity ───────────────────────────
    cos_sim_d1 = _derivative(cos_sim, method, edge_order)
    cos_sim_d2 = _derivative(cos_sim_d1, method, edge_order)

    # ── 4. EMA centroid + drift ───────────────────────────────────────
    ema = np.zeros((N, dim), dtype=np.float64)
    ema[0] = embeddings[0].copy()
    for t in range(1, N):
        ema[t] = alpha * embeddings[t] + (1 - alpha) * ema[t - 1]

    ema_drift = np.zeros(N, dtype=np.float64)
    for t in range(N):
        ema_drift[t] = cosine_distance(embeddings[t], ema[t])

    # ── 5. Cumulative path length ─────────────────────────────────────
    cum_path = np.zeros(N, dtype=np.float64)
    for t in range(1, N):
        cum_path[t] = cum_path[t - 1] + (cos_dist[t] if not np.isnan(cos_dist[t]) else 0.0)

    # ── 6. Boundary detection (k-sigma on velocity, CAUSAL) ─────────────
    #    At each step t, compute running mean & std from min_samples..t
    #    so that only past information is used (no future leakage).
    boundary_flags = np.zeros(N, dtype=bool)
    for t in range(min_samples, N):
        if np.isnan(velocity[t]):
            continue
        # Collect all valid velocities from min_samples up to and including t
        past_v = velocity[min_samples:t + 1]
        past_valid = past_v[~np.isnan(past_v)]
        if len(past_valid) < 2:
            continue
        mu_v = np.mean(past_valid)
        sigma_v = np.std(past_valid)
        threshold = mu_v + k_sigma * sigma_v
        if velocity[t] > threshold:
            boundary_flags[t] = True

    # ── 7. Fixation & return detection ────────────────────────────────
    return_flags = np.zeros(N, dtype=bool)
    return_cluster_ids = np.full(N, -1, dtype=int)
    fixation_flags = np.zeros(N, dtype=bool)

    # Build cluster centroids from segments between boundaries
    centroids: list[np.ndarray] = []
    segment_start = 0
    for t in range(N):
        if boundary_flags[t] or t == N - 1:
            if t > segment_start:
                centroid = embeddings[segment_start:t].mean(axis=0)
                centroids.append(centroid)
            segment_start = t

    # Check returns to past centroids
    for t in range(1, N):
        for ci, centroid in enumerate(centroids):
            sim = float(np.dot(embeddings[t], centroid))
            if sim > return_thresh:
                return_flags[t] = True
                return_cluster_ids[t] = ci
                break  # earliest matching centroid

    # Fixation: velocity near zero for extended run (5+ steps)
    fixation_run = 0
    for t in range(1, N):
        if not np.isnan(velocity[t]) and abs(velocity[t]) < 1e-4:
            fixation_run += 1
        else:
            fixation_run = 0
        if fixation_run >= 5:
            fixation_flags[t] = True

    # ── 8. Semantic cloud validity ────────────────────────────────────
    cloud_valid = np.zeros(N, dtype=bool)
    for t in range(1, N):
        if not np.isnan(cos_sim[t]) and cos_sim[t] >= sim_floor:
            cloud_valid[t] = True

    # ── Summary stats ─────────────────────────────────────────────────
    valid_vel = velocity[~np.isnan(velocity)]
    mean_v = float(np.mean(valid_vel)) if len(valid_vel) > 0 else 0.0
    std_v = float(np.std(valid_vel)) if len(valid_vel) > 0 else 0.0

    return TrajectoryMetrics(
        n_windows=N,
        cosine_distance=cos_dist,
        euclidean_distance=euc_dist,
        velocity=velocity,
        acceleration=acceleration,
        jerk=jerk,
        cosine_similarity=cos_sim,
        cosine_sim_d1=cos_sim_d1,
        cosine_sim_d2=cos_sim_d2,
        ema_centroids=ema,
        ema_drift=ema_drift,
        cumulative_path=cum_path,
        boundary_flags=boundary_flags,
        return_flags=return_flags,
        return_cluster_ids=return_cluster_ids,
        fixation_flags=fixation_flags,
        cloud_valid=cloud_valid,
        mean_velocity=mean_v,
        std_velocity=std_v,
        n_boundaries=int(boundary_flags.sum()),
        n_returns=int(return_flags.sum()),
        total_path_length=float(cum_path[-1]) if N > 0 else 0.0,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Internal helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _derivative(
    series: np.ndarray,
    method: str = "gradient",
    edge_order: int = 2,
) -> np.ndarray:
    """Compute the numerical derivative of a 1-D series.

    NaN values in the input propagate to the output.
    """
    if method == "gradient":
        # numpy.gradient handles NaN gracefully at edges
        return np.gradient(series, edge_order=min(edge_order, 2))
    elif method == "diff":
        d = np.diff(series, n=1)
        # Prepend NaN so the output length matches the input
        return np.concatenate([[np.nan], d])
    else:
        raise ValueError(f"Unknown derivative method: {method!r}")


def pairwise_cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    """NxN cosine similarity matrix (useful for heatmap visualisation)."""
    # Embeddings are L2-normalised → gram matrix = cosine similarity
    return embeddings @ embeddings.T


def semantic_cloud_stats(embeddings: np.ndarray) -> dict[str, float]:
    """Compute distribution statistics for the semantic cloud.

    Returns the mean and std of all pairwise cosine similarities,
    useful for calibrating the similarity_floor parameter.
    """
    from scipy.spatial.distance import pdist

    dists = pdist(embeddings, metric="cosine")  # 1 - cos_sim
    sims = 1.0 - dists
    return {
        "mean_pairwise_sim": float(np.mean(sims)),
        "std_pairwise_sim": float(np.std(sims)),
        "min_pairwise_sim": float(np.min(sims)),
        "max_pairwise_sim": float(np.max(sims)),
        "median_pairwise_sim": float(np.median(sims)),
        "n_pairs": len(sims),
    }
