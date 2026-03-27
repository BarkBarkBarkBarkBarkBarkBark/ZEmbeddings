# Experiment Report: mvp_smoke_fixation
*Generated 20260326_190710*

## Summary

| Metric | Value |
|--------|-------|
| n_windows | 65 |
| n_tokens | 74 |
| mean_velocity | 0.0057038390447223 |
| std_velocity | 0.06727504050404406 |
| n_boundaries | 2 |
| n_returns | 0 |
| total_path_length | 7.690104454755783 |
| kalman_mode | scalar |
| kalman_violations | 1 |
| kalman_accel_violations | 0 |
| cloud_mean_sim | 0.5714543597864673 |
| cloud_std_sim | 0.1587020398215344 |
| boundary_indices | [39, 64] |
| return_indices | [] |

## Parameters

```yaml
model:
  backend: local
  name: text-embedding-3-small
  dimensions_full: 1536
  dimensions_reduced: 256
  max_tokens_per_input: 8192
  batch_size: 2048
  cache_embeddings: false
  local_model: all-MiniLM-L6-v2
  local_batch_size: 256
  device: cpu
window:
  size: 10
  stride: 1
  encoding: cl100k_base
ema:
  alpha: 0.3
boundary:
  k_sigma: 2.0
  min_samples: 5
kalman:
  process_noise_scale: 0.0001
  measurement_noise_scale: 0.01
  initial_covariance_scale: 1.0
  innovation_threshold: 3.0
  mode: scalar
semantic_cloud:
  similarity_floor: 0.3
  cluster_threshold: 0.85
  return_threshold: 0.8
derivatives:
  method: gradient
  edge_order: 2
database:
  enabled: false
  host: localhost
  port: 5432
  dbname: zembeddings
  user: postgres
  password: ''
paths:
  data_raw: data/raw
  data_synthetic: data/synthetic
  data_processed: data/processed
  results_metrics: results/metrics
  results_reports: results/reports
experiment:
  name: mvp_smoke_fixation
  description: Fast local run on fixation_test.txt
  source: data/synthetic/fixation_test.txt

```

## Trajectory Sparklines

**Cosine Distance:**  `▁▁▂▁▁▃▂▁▂▁▄▄▄▂▁▁▂▂▄▁▁▂▁▂▁▂▂▁▂▁▁▂▁▃▂▁▃▃▁█▂▅▁▁▁▄▁▁▂▂▂▃▄▃▁▂▁▁▃▂▁▂▁▃`

**Velocity:**  `▄▃▂▄▄▁▃▃▄▄▂▂▁▂▄▄▄▂▁▄▃▃▃▃▃▂▃▃▂▃▃▃▃▂▄▄▂▆▃▁▂▁▃▅▃▁▃▄▃▃▄▃▁▂▃▃▄▃▁▃▃▄█`

**Acceleration:**  `▂▃▃▁▂▃▃▃▂▁▂▃▄▃▃▂▁▃▃▂▃▃▃▂▂▃▂▃▃▂▃▂▃▄▂▃▃▁▂▂▃▄▃▁▃▄▃▂▃▂▁▂▄▃▃▃▁▂▃▃▅█`

**Jerk:**  `▄▂▂▄▄▃▂▂▃▄▄▃▂▂▂▄▄▂▂▃▃▂▃▄▃▂▃▃▃▂▃▄▂▃▄▁▂▄▃▄▃▁▃▅▃▂▃▃▂▃▄▃▃▃▂▃▄▃▄▆█`

**EMA Drift:**  `▁▁▁▃▂▁▅▃▂▂▂▅▆▂▂▂▁▃▂▅▃▂▃▂▂▁▂▃▁▂▂▂▄▂▄▂▂▄▃▂█▄▅▃▂▂▄▂▂▂▃▃▄▄▃▂▃▂▂▃▃▂▂▂▅`

**Kalman Mahalanobis:**  `▁▁▁▁▂▃▁▃▁▂▄▁▁▃▄▃▁▁▄▃▁▁▁▁▁▁▁▂▁▁▁▂▂▃▁▁▂▂▃█▄▂▅▃▃▄▂▁▁▂▂▂▂▁▄▁▂▁▂▁▂▁▁▃`

**Kalman Accel Mahalanobis:**  `▁▁▁▄▁▃▁▁▃▃▁▂▄▁▁▃▃▃▂▂▁▁▁▂▁▂▁▁▁▁▁▂▁▃▃▂▁▅▁▁▂▅▂▅▁▃▁▁▁▁▃▁▃▁▁▁▃▁▂▁▄█`

## Detected Boundaries

| Window Index | Velocity | Kalman d_M |
|-------------|----------|-----------|
| 39 | 0.158864 | 0.927935 |
| 64 | 0.273102 | 1.230310 |

## Detected Returns

| Window Index | Cluster ID |
|-------------|-----------|
