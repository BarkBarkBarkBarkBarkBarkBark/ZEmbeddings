# Experiment Report: exp_001_synthetic_topic_shift
*Generated 20260326_184537*

## Summary

| Metric | Value |
|--------|-------|
| n_windows | 429 |
| n_tokens | 438 |
| mean_velocity | 6.828012176084853e-07 |
| std_velocity | 0.04254700911019822 |
| n_boundaries | 21 |
| n_returns | 19 |
| total_path_length | 56.01857286691666 |
| kalman_mode | scalar |
| kalman_violations | 0 |
| kalman_accel_violations | 0 |
| cloud_mean_sim | 0.1740663468413102 |
| cloud_std_sim | 0.1383677829672637 |
| boundary_indices | [64, 75, 102, 103, 139, 142, 160, 167, 187, 192, 202, 203, 213, 214, 312, 345, 368, 389, 402, 423, 428] |
| return_indices | [99, 100, 101, 102, 139, 140, 141, 142, 188, 189, 197, 199, 200, 201, 202, 211, 212, 213, 426] |

## Parameters

```yaml
model:
  backend: openai
  name: text-embedding-3-small
  dimensions_full: 1536
  dimensions_reduced: 256
  max_tokens_per_input: 8192
  batch_size: 2048
  local_model: all-MiniLM-L6-v2
  local_batch_size: 256
  device: mps
window:
  size: 10
  stride: 1
  encoding: cl100k_base
ema:
  alpha: 0.3
boundary:
  k_sigma: 1.5
  min_samples: 5
kalman:
  process_noise_scale: 0.0001
  measurement_noise_scale: 0.01
  initial_covariance_scale: 1.0
  innovation_threshold: 2.5
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
  name: exp_001_synthetic_topic_shift
  description: 'Baseline experiment: embed a synthetic multi-topic conversation, compute
    full metric suite, and verify that Kalman-filter innovations spike at known topic
    boundaries.

    '
  source: data/synthetic/topic_shift_001.txt

```

## Trajectory Sparklines

**Cosine Distance:**  `▃▃▂▂▃▂▂▃▂▂▃▂▂▃▃▃▃▂▃▃▃▃▂▃▃▃▂▂▂▃▂▃▃▂▃▃▂▂▃▃▃▂▃`

**Velocity:**  `▄▄▄▅▄▅▄▄▄▅▄▄▄▄▅▅▄▅▅▄▅▅▄▄▄▅▄▄▅▄▅▄▄▅▄▄▄▄▅▄▄▅▅`

**Acceleration:**  `▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄`

**Jerk:**  `▄▄▄▄▄▄▄▄▄▃▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▅`

**EMA Drift:**  `▃▄▄▃▄▃▄▄▄▃▃▃▃▃▄▃▄▃▃▄▄▄▃▃▄▄▃▃▃▃▃▃▃▃▄▄▃▃▄▄▃▃▄`

**Kalman Mahalanobis:**  `▃▂▂▂▂▁▂▃▂▂▃▂▁▂▂▂▂▁▂▂▂▃▂▂▁▂▂▂▃▂▁▂▂▂▃▂▂▂▂▂▃▂▃`

**Kalman Accel Mahalanobis:**  `▃▃▁▁▂▁▂▂▁▂▃▂▁▂▂▂▂▂▁▂▂▃▂▁▂▁▂▂▂▁▂▂▂▁▂▂▂▂▂▂▃▁▄`

## Detected Boundaries

| Window Index | Velocity | Kalman d_M |
|-------------|----------|-----------|
| 64 | 0.080391 | 0.437895 |
| 75 | 0.081175 | 0.238097 |
| 102 | 0.091971 | 0.063890 |
| 103 | 0.099196 | 1.679304 |
| 139 | 0.067153 | 0.496170 |
| 142 | 0.069033 | 0.440576 |
| 160 | 0.065194 | 0.325765 |
| 167 | 0.079341 | 0.319900 |
| 187 | 0.076545 | 0.004144 |
| 192 | 0.080034 | 0.106840 |
| 202 | 0.070971 | 0.137978 |
| 203 | 0.069649 | 1.208424 |
| 213 | 0.119894 | 0.291985 |
| 214 | 0.085209 | 1.808324 |
| 312 | 0.064364 | 0.738163 |
| 345 | 0.085217 | 0.247819 |
| 368 | 0.090601 | 0.530654 |
| 389 | 0.065982 | 0.668814 |
| 402 | 0.070249 | 1.072109 |
| 423 | 0.090280 | 0.673215 |
| 428 | 0.080842 | 0.844913 |

## Detected Returns

| Window Index | Cluster ID |
|-------------|-----------|
| 99 | 3 |
| 100 | 3 |
| 101 | 3 |
| 102 | 3 |
| 139 | 5 |
| 140 | 5 |
| 141 | 5 |
| 142 | 5 |
| 188 | 9 |
| 189 | 9 |
| 197 | 10 |
| 199 | 11 |
| 200 | 11 |
| 201 | 11 |
| 202 | 11 |
| 211 | 13 |
| 212 | 13 |
| 213 | 13 |
| 426 | 20 |
