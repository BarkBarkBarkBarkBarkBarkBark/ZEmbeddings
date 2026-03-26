# MANIFESTO — Semantic Trajectory Analysis

> *"The path through meaning-space is itself meaningful."*

---

## 1 · Purpose

This codebase exists to answer a deceptively simple question:

**When someone speaks, how does the *meaning* of their speech move through
semantic space — and can we characterise that motion mathematically?**

We treat a conversation transcript as a time-series of positions in a
high-dimensional semantic manifold, then apply the tools of classical
mechanics — displacement, velocity, acceleration, jerk — plus Kalman
filtering and change-point detection to classify the trajectory.

---

## 2 · Core Assumptions

### 2.1  Biological Plausibility — Causal Attention Only

In real life we have access to the present and the past, never the future.
Modern language models often use bidirectional attention or streaming
buffers that peek ahead, but we deliberately **reject** future information.

Every computation in this pipeline is **strictly causal**:

- Sliding windows include only tokens at or before time *t*.
- The exponential moving average (EMA) is one-sided.
- The Kalman filter predicts forward, then corrects — it never
  retrospectively smooths.

This mirrors the constraint a biological listener operates under.

> **Reference:** Friston, K. (2010). "The free-energy principle: a unified
> brain theory?" *Nature Reviews Neuroscience*, 11(2), 127–138.
> doi:[10.1038/nrn2787](https://doi.org/10.1038/nrn2787)

### 2.2  Semantic Space Is Not Euclidean (But Locally It's Close Enough)

OpenAI's text-embedding-3 models produce L2-normalised vectors on a
1536-dimensional unit hypersphere.  Cosine similarity equals the dot
product, and cosine *distance* $d = 1 - \cos(\theta)$ is our primary
distance metric.

Crucially, **the midpoint between two embeddings is not necessarily
meaningful**.  "Dog" and "Airplane" each occupy well-defined regions,
but their centroid may correspond to nothing interpretable.  What *is*
meaningful is the **local neighbourhood** — nearby embeddings belong to
the same semantic cluster, and smooth trajectories through clusters
indicate coherent discourse.

> **Reference:** Aggarwal, C. C., Hinneburg, A., & Keim, D. A. (2001).
> "On the Surprising Behavior of Distance Metrics in High Dimensional
> Space." *ICDT 2001*, LNCS 1973, 420–434.
> doi:[10.1007/3-540-44503-X_27](https://doi.org/10.1007/3-540-44503-X_27)

### 2.3  Clusters Are Real; Trajectories Between Them Are Transitions

We assume that coherent topics map to **clusters** in embedding space.
While a speaker remains on-topic, successive window embeddings stay
within a cluster (low velocity, low cosine distance).  When the speaker
changes topic, the trajectory crosses a **semantic boundary** — a region
of high velocity that the Kalman filter flags as an *innovation violation*.

> **Reference:** Adams, R. P. & MacKay, D. J. C. (2007). "Bayesian
> Online Changepoint Detection." arXiv:[0710.3742](https://arxiv.org/abs/0710.3742).

### 2.4  The Semantic Cloud — Where Proximity Loses Meaning

In high dimensions, random unit vectors concentrate near orthogonality.
For $d = 1536$, the expected cosine similarity between two random vectors
is approximately $0$, with standard deviation $\approx 1/\sqrt{d} \approx 0.026$.

Any measured cosine similarity below a **similarity floor** (~0.3 in
practice) is therefore indistinguishable from noise.  We call this the
*semantic cloud threshold* — below it, we cannot claim two embeddings
are meaningfully related.

> **Reference:** Vershynin, R. (2018). *High-Dimensional Probability:
> An Introduction with Applications in Data Science.* Cambridge
> University Press.  Chapter 5: Concentration of Measure.

---

## 3 · The Measurement Pipeline

```
  raw text
     │
     ▼
  ┌──────────────┐     tiktoken (cl100k_base)
  │  Tokeniser   │──── encode full transcript into token IDs
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐     causal: window = tokens[t-W+1 : t+1]
  │  Windowing   │──── stride-1 sliding window, W = 10 tokens
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐     OpenAI text-embedding-3-small
  │  Embedding   │──── batch API call → full (1536-d) + reduced (256-d)
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐     cache .npz + pgvector INSERT
  │   Storage    │──── embeddings + metadata persisted
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │   Metrics    │──── see §4 below
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │   Output     │──── YAML (machine) + Markdown (human) reports
  └──────────────┘
```

---

## 4 · Metric Catalogue

All metrics are computed **causally** — at time $t$ we use only
information from windows $0, 1, \ldots, t$.

### 4.1  Distance Series

| Symbol | Definition | Notes |
|--------|-----------|-------|
| $d_t^{\cos}$ | $1 - e_t \cdot e_{t-1}$ | Cosine distance (primary) |
| $d_t^{L2}$   | $\lVert e_t - e_{t-1} \rVert_2$ | Euclidean distance |

### 4.2  Derivative Series (Kinematic Analogy)

| Symbol | Order | Kinematic Name | Meaning |
|--------|-------|---------------|---------|
| $v_t = d_t^{\cos}$     | 1st | **Velocity**     | Rate of semantic displacement per step |
| $a_t = \dot{v}_t$       | 2nd | **Acceleration** | Change in velocity — topic is *shifting* |
| $j_t = \dot{a}_t$       | 3rd | **Jerk**         | Change in acceleration — onset/offset of a shift |

Computed via `numpy.gradient()` with `edge_order=2` for stable boundary
estimates.

### 4.3  Cosine Similarity Derivatives

The raw similarity $s_t = e_t \cdot e_{t-1}$ and its first two
derivatives $\dot{s}_t$, $\ddot{s}_t$ capture the *attraction* side of
the dynamics — how strongly successive windows cohere.

### 4.4  Exponential Moving Average (EMA) Centroid

$$\bar{e}_t = \alpha \, e_t + (1 - \alpha) \, \bar{e}_{t-1}$$

The EMA centroid is a causal estimate of "where the conversation has
been."  Drift from the centroid ($\lVert e_t - \bar{e}_t \rVert$)
measures how far the current utterance has strayed from the running
context.

### 4.5  Kalman Filter — Predicted Trajectory & Innovation

We model the embedding trajectory as a **constant-velocity** linear
dynamical system:

$$\hat{e}_{t|t-1} = e_{t-1} + v_{t-1}$$

The Kalman filter maintains an estimate of position and velocity along
with uncertainty (covariance).  When the **innovation** — the difference
between the predicted and actual embedding — exceeds a Mahalanobis
distance threshold, we flag a **trajectory violation**: the speaker
has departed from the anticipated semantic course.

Two modes are implemented:

- **Scalar mode:** Kalman filter on the 1-D cosine-distance time series.
  Lightweight, interpretable.
- **Vector mode:** Kalman filter on the reduced (256-d) embedding
  trajectory, with diagonal covariance approximation.

> **Reference:** Kalman, R. E. (1960). "A New Approach to Linear
> Filtering and Prediction Problems." *Journal of Basic Engineering*,
> 82(1), 35–45.
> doi:[10.1115/1.3662552](https://doi.org/10.1115/1.3662552)

### 4.6  Boundary Detection

A semantic boundary is flagged when **any** of these conditions hold:

1. Velocity exceeds $\mu_v + k\sigma_v$ (configurable $k$, default 2.0).
2. Kalman innovation exceeds the Mahalanobis threshold (default 3.0).
3. Both velocity *and* jerk spike simultaneously (conjunction rule for
   high-confidence boundaries).

### 4.7  Fixation & Return Detection

Fixation (semantic perseveration) and return (revisiting a prior topic)
are detected by comparing the current embedding to **historical cluster
centroids**:

- Maintain a list of centroids from detected clusters (segments between
  boundaries).
- If $\cos(e_t, c_k) >$ `return_threshold` for any past centroid $c_k$,
  flag a **return** to topic $k$.
- If velocity remains near zero for an extended run, flag **fixation**.

### 4.8  Cumulative Path Length

$$L_t = \sum_{i=1}^{t} d_i^{\cos}$$

Total distance traversed through semantic space.  Useful for comparing
speakers or sessions: a rambling speaker accumulates more path length
per unit time than a focused one.

---

## 5 · Storage: pgvector

Embeddings and metrics are persisted to a PostgreSQL database with the
`pgvector` extension, enabling:

- **Exact and approximate nearest-neighbour search** across all stored
  windows — "find every moment in the corpus that sounds like this one."
- **Cross-experiment similarity** — compare trajectories across different
  speakers or sessions.
- **Temporal queries** — retrieve the embedding trajectory for a time
  range and re-analyse with different parameters.

> **Reference:** Johnson, J., Douze, M., & Jégou, H. (2021).
> "Billion-scale similarity search with GPUs." *IEEE Transactions on
> Big Data*, 7(3), 535–547.
> doi:[10.1109/TBDATA.2019.2921572](https://doi.org/10.1109/TBDATA.2019.2921572)

---

## 6 · Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Parameters are data, not code** | Single `PARAMS` dict; YAML config; editable at the REPL before any run |
| **Causal only** | No future tokens, no bidirectional smoothing, no look-ahead |
| **Reproducibility** | Every experiment saves its full param snapshot alongside results |
| **Dual-format output** | YAML for machines, Markdown for humans |
| **Cache everything** | Embeddings are expensive — `.npz` on disk, pgvector in DB |

---

## 7 · References (Collected)

1. Kalman, R. E. (1960). "A New Approach to Linear Filtering and
   Prediction Problems." *J. Basic Eng.*, 82(1), 35–45.

2. Aggarwal, C. C., Hinneburg, A., & Keim, D. A. (2001). "On the
   Surprising Behavior of Distance Metrics in High Dimensional Space."
   *ICDT 2001*, LNCS 1973.

3. Mikolov, T. et al. (2013). "Efficient Estimation of Word
   Representations in Vector Space." arXiv:1301.3781.

4. Pennington, J., Socher, R., & Manning, C. D. (2014). "GloVe:
   Global Vectors for Word Representation." *EMNLP 2014*.

5. Adams, R. P. & MacKay, D. J. C. (2007). "Bayesian Online
   Changepoint Detection." arXiv:0710.3742.

6. Devlin, J. et al. (2019). "BERT: Pre-training of Deep
   Bidirectional Transformers." *NAACL-HLT 2019*.

7. Reimers, N. & Gurevych, I. (2019). "Sentence-BERT: Sentence
   Embeddings using Siamese BERT-Networks." *EMNLP 2019*.

8. Friston, K. (2010). "The free-energy principle: a unified brain
   theory?" *Nature Rev. Neuroscience*, 11(2), 127–138.

9. Vershynin, R. (2018). *High-Dimensional Probability.* Cambridge
   University Press.

10. Johnson, J., Douze, M., & Jégou, H. (2021). "Billion-scale
    similarity search with GPUs." *IEEE Trans. Big Data*, 7(3).

11. OpenAI (2024). "Embeddings — API Reference."
    https://platform.openai.com/docs/guides/embeddings

---

*This document is the ground truth for the project's scientific logic.
If the code disagrees with the manifesto, the code has a bug.*
