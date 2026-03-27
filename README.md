

# ZEmbeddings

### *When someone speaks, how does meaning move?*

**Quantify the trajectory of thought through semantic space.**

---

## The Problem

You have conversation transcripts — hours of natural speech — and you
want to answer questions that **no word-count or sentiment tool can
touch:**

- *Where exactly does the speaker change topic?*
- *How erratic is this conversation compared to that one?*
- *Did the speaker return to an earlier idea, or is this new?*
- *How much semantic ground did they cover in 10 minutes?*

These are geometric questions about **trajectories through meaning
space**, and they need geometric answers.

## What ZEmbeddings Gives You

Every utterance gets embedded into a 1536-dimensional semantic manifold.
Then we treat that sequence of points as a *trajectory* and apply the
tools of classical mechanics:

| Metric | What it measures | Neuroscience analogy |
|--------|-----------------|---------------------|
| **Semantic velocity** | Rate of topic change per step | Attention shift speed |
| **Acceleration** | Is the speaker *beginning* to shift? | Onset of context switch |
| **Jerk** | Onset/offset of a shift — the "snap" | Event boundary signal |
| **Kalman innovation** | Surprise vs. predicted trajectory | Prediction error |
| **Kalman on acceleration** | Surprise in the *rate of change* | Context-change neuron |
| **EMA drift** | How far from the running context | Working memory deviation |
| **Cumulative path** | Total distance through meaning-space | Cognitive effort proxy |
| **Boundary flags** | Binary: topic changed here | Hippocampal reset cue |
| **Fixation flags** | Speaker stuck on one idea | Perseveration detector |
| **Return flags** | Speaker revisited a past topic | Episodic memory retrieval |

Every computation is **strictly causal** — at time *t* we use only the
past and present, never the future. This mirrors the constraint a
biological listener operates under.

> *"The path through meaning-space is itself meaningful."*
> — [MANIFESTO.md](MANIFESTO.md)

---

## Quick Start

```bash
# 1. Clone & install
cd ZEmbeddings
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Configure (OpenAI key for API embeddings, or use local models for free)
cp .env.example .env && $EDITOR .env

# 3. Generate synthetic test data & run your first experiment
python scripts/generate_conversations.py
python scripts/run_experiment.py config/experiments/exp_001_synthetic_topic_shift.yaml

# 4. Or use the one-liner shell scripts
./scripts/run_tests.sh              # Run full test suite with coverage
./scripts/run_experiment.sh          # Run default experiment with live display
```

### Free local embeddings (no API key needed)

```python
from zembeddings.params import get_params
from zembeddings.pipeline import run_pipeline

p = get_params(**{"model.backend": "local"})  # uses all-MiniLM-L6-v2, ~80 MB
result = run_pipeline("data/synthetic/topic_shift_001.txt", params=p)
```

---

## Interactive Parameter Editing

Every tuneable knob lives in a single Python dictionary. Open a REPL
and explore:

```python
from zembeddings.params import PARAMS, get_params

PARAMS["window"]    # {'size': 10, 'stride': 1, 'encoding': 'cl100k_base'}
PARAMS["kalman"]    # {'process_noise_scale': 0.0001, ...}

# Override for a run
p = get_params(**{"window.size": 20, "kalman.mode": "vector", "ema.alpha": 0.5})

# Or load from YAML
from zembeddings.params import load_params
p = load_params("config/experiments/exp_001_synthetic_topic_shift.yaml")
```

---

## Repo Map

Every file, what it does, and where it fits in the pipeline.

### Core Library (`src/zembeddings/`)

| Module | Role | Key Functions / Classes |
|--------|------|------------------------|
| [`params.py`](src/zembeddings/params.py) | **Single source of truth** for all parameters | `PARAMS`, `get_params()`, `load_params()` |
| [`tokenizer.py`](src/zembeddings/tokenizer.py) | Causal sliding-window tokenisation | `tokenize()`, `Window`, `TokenizedTranscript` |
| [`embeddings.py`](src/zembeddings/embeddings.py) | OpenAI API + local sentence-transformers | `embed_texts()`, `.npz` caching |
| [`metrics.py`](src/zembeddings/metrics.py) | Full kinematic metric suite | `compute_metrics()` → `TrajectoryMetrics` |
| [`kalman.py`](src/zembeddings/kalman.py) | Kalman filter boundary detection | `run_scalar_kalman()`, `run_vector_kalman()` |
| [`pipeline.py`](src/zembeddings/pipeline.py) | End-to-end orchestrator | `run_pipeline()` → `PipelineResult` |
| [`io.py`](src/zembeddings/io.py) | YAML metrics + Markdown reports | `write_metrics_yaml()`, `write_report_markdown()` |
| [`ingest.py`](src/zembeddings/ingest.py) | Corpus parsing & text posturing | `ingest_conversation()`, `walk_conversations()` |
| [`database.py`](src/zembeddings/database.py) | pgvector storage & ANN search | `create_schema()`, `insert_*()`, `find_similar()` |

### Pipeline Data Flow

```
  raw text / corpus
        │
        ▼
  ┌─────────────┐   tokenizer.py     ┌─────────────┐
  │  Tokenise   │──────────────────▶│   Windows    │
  └─────────────┘                    └──────┬──────┘
                                            │
  ┌─────────────┐   embeddings.py    ┌──────▼──────┐
  │   Embed     │◀──────────────────│  Window texts│
  └──────┬──────┘                    └─────────────┘
         │
         ▼
  ┌─────────────┐   metrics.py       velocity, acceleration, jerk,
  │  Metrics    │──────────────────▶ EMA drift, boundaries, fixation,
  └──────┬──────┘                    returns, cumulative path
         │
         ▼
  ┌─────────────┐   kalman.py        Mahalanobis distance,
  │   Kalman    │──────────────────▶ innovation norms,
  └──────┬──────┘                    violation flags
         │
         ▼
  ┌─────────────┐   io.py            YAML (machine) +
  │   Output    │──────────────────▶ Markdown (human)
  └─────────────┘
```

### Configuration

| File | Purpose |
|------|---------|
| [`config/default.yaml`](config/default.yaml) | Default parameters (mirrors `PARAMS` dict) |
| [`config/experiments/*.yaml`](config/experiments/) | Per-experiment parameter overrides |
| [`.env.example`](.env.example) | Template for API keys and DB credentials |

### Scripts

| Script | Purpose |
|--------|---------|
| [`scripts/run_experiment.py`](scripts/run_experiment.py) | Run one experiment from a YAML config |
| [`scripts/generate_conversations.py`](scripts/generate_conversations.py) | Create synthetic test transcripts with ground-truth boundaries |
| [`scripts/ingest_conversations.py`](scripts/ingest_conversations.py) | Parse & ingest the no_media corpus |
| [`scripts/setup_database.py`](scripts/setup_database.py) | Create pgvector schema (optional) |
| [`scripts/run_tests.sh`](scripts/run_tests.sh) | One-command test suite with coverage display |
| [`scripts/run_experiment.sh`](scripts/run_experiment.sh) | One-command experiment runner with display |

### Context & Documentation

| File | Audience |
|------|----------|
| [`MANIFESTO.md`](MANIFESTO.md) | **Ground truth** — scientific rationale, metric definitions, references |
| [`context/architecture.yaml`](context/architecture.yaml) | Machine-readable module dependency graph & data flow |
| [`context/glossary.yaml`](context/glossary.yaml) | Domain terms with definitions and units |
| [`context/conventions.yaml`](context/conventions.yaml) | Coding rules, naming patterns, test expectations |
| [`context/ingestion_strategy.yaml`](context/ingestion_strategy.yaml) | Full corpus ingestion plan |
| [`.agent-hints.md`](.agent-hints.md) | Quick-reference for AI copilots |

### Data

| Directory | Contents |
|-----------|----------|
| `data/synthetic/` | 3 synthetic transcripts + ground-truth JSON manifests |
| `data/processed/` | Postured text files from corpus ingestion |
| `results/metrics/` | YAML metric time-series from experiment runs |
| `results/reports/` | Markdown experiment reports with sparklines |

---

## Embedding Backends

### OpenAI API (default)

Uses `text-embedding-3-small` with native Matryoshka dimensionality
reduction (1536-d full, 256-d reduced).

| Scope | Conversations | Est. Cost |
|-------|--------------|-----------|
| 1 speaker, 1 conversation | 1 | < $0.01 |
| All 3 modes, 10% sample | 165 | ~$1.03 |
| All 3 modes, full corpus | 1,655 | ~$10.29 |

### Local models (free, runs on 8 GB Mac)

| Model | Dimensions | Size | Notes |
|-------|-----------|------|-------|
| `all-MiniLM-L6-v2` | 384 | ~80 MB | Fast baseline |
| `all-mpnet-base-v2` | 768 | ~420 MB | Best quality/size |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | ~550 MB | Long context |

```bash
pip install sentence-transformers  # one-time install
```

---

## Testing

```bash
./scripts/run_tests.sh                      # Pretty terminal display
.venv/bin/python -m pytest tests/ -v        # Standard pytest
.venv/bin/python -m pytest tests/ --cov     # With coverage
```

---

## The Science

Read [MANIFESTO.md](MANIFESTO.md) for the full scientific rationale,
including:

- Why **causal-only** computation matters (biological plausibility)
- Why **cosine distance** is the right metric on the embedding hypersphere
- How the **Kalman filter** models expected semantic trajectory
- What the **semantic cloud threshold** means in high dimensions
- All 11 academic references

> *This document is the ground truth for the project's scientific logic.
> If the code disagrees with the manifesto, the code has a bug.*

---

## License

MIT — see [LICENSE](LICENSE).
]]>