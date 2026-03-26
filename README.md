# ZEmbeddings

**Trajectory analysis in semantic embedding space.**

Embed conversation transcripts with OpenAI's `text-embedding-3-small`,
then compute kinematic metrics — velocity, acceleration, jerk — plus
Kalman-filter boundary detection and fixation/return classification to
characterise how meaning *moves* during natural speech.

Read [MANIFESTO.md](MANIFESTO.md) for the full scientific rationale and
references.

---

## Quick Start

```bash
# 1. Clone & enter
cd ZEmbeddings

# 2. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install
pip install -e ".[dev]"

# 4. Add your OpenAI key
cp .env.example .env   # then edit .env

# 5. Generate synthetic data & run first experiment
python scripts/generate_conversations.py
python scripts/run_experiment.py config/experiments/exp_001_synthetic_topic_shift.yaml
```

## Interactive Parameter Editing

Every tuneable knob lives in a single dictionary.  Open a Python REPL
and explore before running:

```python
from zembeddings.params import PARAMS, get_params

# Inspect defaults
PARAMS["window"]          # {'size': 10, 'stride': 1, ...}
PARAMS["kalman"]          # {'process_noise_scale': 0.0001, ...}

# Override for a run
p = get_params(**{"window.size": 20, "kalman.mode": "vector"})

# Or load from YAML
from zembeddings.params import load_params
p = load_params("config/experiments/exp_001_synthetic_topic_shift.yaml")
```

Then pass `p` to the pipeline:

```python
from zembeddings.pipeline import run_pipeline
results = run_pipeline("data/synthetic/topic_shift_001.txt", params=p)
```

## Project Layout

```
├── MANIFESTO.md            ← Scientific rationale & references
├── config/
│   ├── default.yaml        ← Default parameters (mirrors PARAMS dict)
│   └── experiments/        ← Per-experiment YAML overrides
├── context/
│   └── ingestion_strategy.yaml  ← Full ingestion plan (machine-readable)
├── src/zembeddings/
│   ├── params.py           ← Central parameter dictionary
│   ├── tokenizer.py        ← Causal sliding-window tokenisation
│   ├── embeddings.py       ← OpenAI + local embedding backends
│   ├── metrics.py          ← Velocity, acceleration, jerk, boundaries
│   ├── kalman.py           ← Scalar & vector Kalman filters
│   ├── database.py         ← pgvector storage & retrieval
│   ├── pipeline.py         ← End-to-end orchestration
│   ├── ingest.py           ← Corpus ingestion & text posturing
│   └── io.py               ← YAML / Markdown readers & writers
├── scripts/
│   ├── run_experiment.py
│   ├── generate_conversations.py
│   ├── ingest_conversations.py  ← Ingest no_media corpus
│   └── setup_database.py
├── data/                   ← raw / synthetic / processed
├── results/                ← metrics (YAML) + reports (Markdown)
└── tests/
```

---

## Database Schema

All tables are created idempotently by `scripts/setup_database.py`.
The schema lives in [`src/zembeddings/database.py`](src/zembeddings/database.py).

```
┌─────────────────────┐
│   conversations      │
├─────────────────────┤       ┌──────────────────────┐
│ id            SERIAL │──┐   │   speakers            │
│ convo_uuid    TEXT   │  │   ├──────────────────────┤
│ session_id    TEXT   │  ├──▶│ id             SERIAL │──┐
│ created_at    TSTZ   │  │   │ conversation_id INT   │  │
│ duration_seconds DBL │  │   │ user_id        TEXT   │  │
│ n_turns       INT    │  │   │ channel        TEXT   │  │
│ n_words       INT    │  │   │ n_turns        INT    │  │
│ metadata      JSONB  │  │   │ n_words        INT    │  │
│ ingested_at   TSTZ   │  │   │ survey_data    JSONB  │  │
└─────────────────────┘  │   │ demographics   JSONB  │  │
                          │   └──────────────────────┘  │
┌─────────────────────┐  │                              │
│   turns              │  │   ┌──────────────────────┐  │
├─────────────────────┤  │   │   experiments          │  │
│ id            SERIAL │  │   ├──────────────────────┤  │
│ conversation_id INT  │◀─┤   │ id             SERIAL │  │
│ speaker_id    INT    │◀─┼──▶│ conversation_id INT   │◀─┘
│ turn_index    INT    │  │   │ speaker_id     INT    │
│ start_time    DBL    │  │   │ name           TEXT   │
│ stop_time     DBL    │  │   │ description    TEXT   │
│ utterance     TEXT   │  │   │ analysis_mode  TEXT   │
│ backchannel   TEXT   │  │   │ params         JSONB  │
│ backchannel_count INT│  │   │ created_at     TSTZ   │
│ n_words       INT    │  │   └──────────────────────┘
│ is_question   BOOL   │  │            │
│ has_overlap   BOOL   │  │            ▼
└─────────────────────┘  │   ┌──────────────────────┐
                          │   │   embeddings           │
                          │   ├──────────────────────┤
                          │   │ id             SERIAL │
                          │   │ experiment_id  INT    │
                          │   │ window_index   INT    │
                          │   │ window_text    TEXT   │
                          │   │ start_token    INT    │
                          │   │ end_token      INT    │
                          │   │ embedding_full vector │
                          │   │ embedding_reduced vec │
                          │   │ created_at     TSTZ   │
                          │   └──────────────────────┘
                          │
                          │   ┌──────────────────────┐
                          │   │   metrics              │
                          │   ├──────────────────────┤
                          │   │ id             SERIAL │
                          │   │ experiment_id  INT    │
                          │   │ window_index   INT    │
                          │   │ cosine_distance  DBL  │
                          │   │ velocity         DBL  │
                          │   │ acceleration     DBL  │
                          │   │ jerk             DBL  │
                          │   │ ema_drift        DBL  │
                          │   │ kalman_innovation DBL │
                          │   │ is_boundary     BOOL  │
                          │   │ is_return       BOOL  │
                          │   │ is_fixation     BOOL  │
                          │   │ …               …     │
                          │   └──────────────────────┘
                          │
    conversations ─1:N─▶ speakers ─1:N─▶ turns
    conversations ─1:N─▶ experiments ─1:N─▶ embeddings
                                       ─1:N─▶ metrics
```

Full DDL: [`src/zembeddings/database.py` § `_SCHEMA_SQL`](src/zembeddings/database.py)

---

## Embedding Backends

### OpenAI API (default)

Uses `text-embedding-3-small` with native Matryoshka dimensionality
reduction (1536-d full, 256-d reduced).

| Scope | Conversations | API Tokens | Cost |
|-------|--------------|------------|------|
| 1 speaker, 1 convo | 1 | ~65K | < $0.01 |
| 1 speaker, 1/10 sample | 165 | ~12M | ~$0.24 |
| All 3 modes, 1/10 sample | 165 | ~51M | ~$1.03 |
| All 3 modes, full corpus | 1,655 | ~514M | ~$10.29 |

### Local models (free, 8 GB Mac)

```python
p = get_params(**{"model.backend": "local"})
# optionally pick a model:
p = get_params(**{"model.backend": "local", "model.local_model": "all-mpnet-base-v2"})
```

| Model | Dimensions | Size | Quality | Speed |
|-------|-----------|------|---------|-------|
| `all-MiniLM-L6-v2` | 384 | ~80 MB | Good baseline | ★★★ |
| `all-mpnet-base-v2` | 768 | ~420 MB | Best quality/size | ★★☆ |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | ~550 MB | Long context | ★★☆ |

Requires `pip install sentence-transformers`.  Uses Apple Silicon MPS
acceleration by default (falls back to CPU).

---

## License

MIT
