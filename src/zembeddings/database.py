"""
pgvector database layer for embedding storage and retrieval.
=============================================================

Schema
------
- **experiments** — one row per experiment run, storing the full
  params snapshot as JSONB.
- **embeddings** — one row per sliding window, storing both the
  full-dimensional and reduced-dimensional vectors, plus the
  decoded window text.
- **metrics** — one row per window with all scalar metric values.

Indexes use ``ivfflat`` with cosine-distance ops for approximate
nearest-neighbour search.

Requires PostgreSQL with the ``pgvector`` extension installed::

    CREATE EXTENSION IF NOT EXISTS vector;

Reference
---------
- pgvector: https://github.com/pgvector/pgvector
- Johnson, J. et al. (2021). "Billion-scale similarity search with
  GPUs." IEEE Trans. Big Data, 7(3), 535–547.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Connection helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_connection(params: dict[str, Any]):
    """Open a psycopg2 connection using params + .env overrides."""
    import psycopg2

    db = params["database"]
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", db["host"]),
        port=int(os.getenv("DB_PORT", db["port"])),
        dbname=os.getenv("DB_NAME", db["dbname"]),
        user=os.getenv("DB_USER", db["user"]),
        password=os.getenv("DB_PASSWORD", db.get("password", "")),
    )
    conn.autocommit = False
    return conn


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Schema creation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SCHEMA_SQL = """
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Conversations ────────────────────────────────────────────────────
-- One row per dyadic conversation (UUID directory from no_media).
CREATE TABLE IF NOT EXISTS conversations (
    id                SERIAL PRIMARY KEY,
    convo_uuid        TEXT UNIQUE NOT NULL,
    session_id        TEXT,
    created_at        TIMESTAMPTZ,
    duration_seconds  DOUBLE PRECISION,
    n_turns           INTEGER,
    n_words           INTEGER,
    metadata          JSONB,
    ingested_at       TIMESTAMPTZ DEFAULT NOW()
);

-- ── Speakers ─────────────────────────────────────────────────────────
-- One row per speaker-in-conversation (2 per conversation).
CREATE TABLE IF NOT EXISTS speakers (
    id                SERIAL PRIMARY KEY,
    conversation_id   INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    user_id           TEXT NOT NULL,
    channel           TEXT,
    n_turns           INTEGER,
    n_words           INTEGER,
    survey_data       JSONB,
    demographics      JSONB
);

-- ── Turns ────────────────────────────────────────────────────────────
-- One row per backbiter transcript turn.
CREATE TABLE IF NOT EXISTS turns (
    id                  SERIAL PRIMARY KEY,
    conversation_id     INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
    speaker_id          INTEGER REFERENCES speakers(id) ON DELETE CASCADE,
    turn_index          INTEGER NOT NULL,
    start_time          DOUBLE PRECISION,
    stop_time           DOUBLE PRECISION,
    utterance           TEXT NOT NULL,
    backchannel         TEXT,
    backchannel_count   INTEGER DEFAULT 0,
    n_words             INTEGER,
    is_question         BOOLEAN DEFAULT FALSE,
    has_overlap         BOOLEAN DEFAULT FALSE
);

-- ── Experiments ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS experiments (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT,
    params          JSONB NOT NULL,
    conversation_id INTEGER REFERENCES conversations(id),
    speaker_id      INTEGER REFERENCES speakers(id),
    analysis_mode   TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Embeddings ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS embeddings (
    id              SERIAL PRIMARY KEY,
    experiment_id   INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
    window_index    INTEGER NOT NULL,
    window_text     TEXT NOT NULL,
    start_token     INTEGER,
    end_token       INTEGER,
    embedding_full  vector({dims_full}),
    embedding_reduced vector({dims_reduced}),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Metrics ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS metrics (
    id                   SERIAL PRIMARY KEY,
    experiment_id        INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
    window_index         INTEGER NOT NULL,
    cosine_distance      DOUBLE PRECISION,
    euclidean_distance   DOUBLE PRECISION,
    velocity             DOUBLE PRECISION,
    acceleration         DOUBLE PRECISION,
    jerk                 DOUBLE PRECISION,
    cosine_similarity    DOUBLE PRECISION,
    cosine_sim_d1        DOUBLE PRECISION,
    cosine_sim_d2        DOUBLE PRECISION,
    ema_drift            DOUBLE PRECISION,
    cumulative_path      DOUBLE PRECISION,
    kalman_innovation    DOUBLE PRECISION,
    kalman_mahalanobis   DOUBLE PRECISION,
    is_boundary          BOOLEAN DEFAULT FALSE,
    is_return            BOOLEAN DEFAULT FALSE,
    return_cluster_id    INTEGER DEFAULT -1,
    is_fixation          BOOLEAN DEFAULT FALSE,
    cloud_valid          BOOLEAN DEFAULT TRUE,
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

-- ── Indexes ──────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_speakers_conversation
    ON speakers(conversation_id);
CREATE INDEX IF NOT EXISTS idx_speakers_user_id
    ON speakers(user_id);
CREATE INDEX IF NOT EXISTS idx_turns_conversation
    ON turns(conversation_id);
CREATE INDEX IF NOT EXISTS idx_turns_speaker
    ON turns(speaker_id);
CREATE INDEX IF NOT EXISTS idx_experiments_conversation
    ON experiments(conversation_id);

-- ivfflat requires at least some rows to build;
-- CREATE INDEX IF NOT EXISTS so re-runs are safe.
"""

_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_emb_full_cosine
    ON embeddings USING ivfflat (embedding_full vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_emb_reduced_cosine
    ON embeddings USING ivfflat (embedding_reduced vector_cosine_ops)
    WITH (lists = 100);
"""


def create_schema(params: dict[str, Any]) -> None:
    """Create all tables (idempotent)."""
    dims_full = params["model"]["dimensions_full"]
    dims_reduced = params["model"]["dimensions_reduced"]

    sql = _SCHEMA_SQL.format(dims_full=dims_full, dims_reduced=dims_reduced)

    conn = _get_connection(params)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print(f"✓ Schema created (dims_full={dims_full}, dims_reduced={dims_reduced})")
    finally:
        conn.close()


def create_indexes(params: dict[str, Any]) -> None:
    """Create ivfflat indexes (requires ≥1 row in embeddings)."""
    conn = _get_connection(params)
    try:
        with conn.cursor() as cur:
            cur.execute(_INDEX_SQL)
        conn.commit()
        print("✓ IVFFlat indexes created")
    finally:
        conn.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Insert helpers — conversations / speakers / turns
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def insert_conversation(
    params: dict[str, Any],
    *,
    convo_uuid: str,
    session_id: str | None = None,
    created_at: str | None = None,
    duration_seconds: float | None = None,
    n_turns: int | None = None,
    n_words: int | None = None,
    metadata: dict | None = None,
) -> int:
    """Insert a conversation row and return its id."""
    conn = _get_connection(params)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO conversations
                   (convo_uuid, session_id, created_at,
                    duration_seconds, n_turns, n_words, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (convo_uuid) DO UPDATE
                   SET session_id = EXCLUDED.session_id,
                       duration_seconds = EXCLUDED.duration_seconds,
                       n_turns = EXCLUDED.n_turns,
                       n_words = EXCLUDED.n_words,
                       metadata = EXCLUDED.metadata
                   RETURNING id""",
                (
                    convo_uuid,
                    session_id,
                    created_at,
                    duration_seconds,
                    n_turns,
                    n_words,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            row = cur.fetchone()
            convo_id = row[0] if row else -1
        conn.commit()
        return convo_id
    finally:
        conn.close()


def insert_speaker(
    params: dict[str, Any],
    *,
    conversation_id: int,
    user_id: str,
    channel: str | None = None,
    n_turns: int | None = None,
    n_words: int | None = None,
    survey_data: dict | None = None,
    demographics: dict | None = None,
) -> int:
    """Insert a speaker row and return its id."""
    conn = _get_connection(params)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO speakers
                   (conversation_id, user_id, channel,
                    n_turns, n_words, survey_data, demographics)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (
                    conversation_id,
                    user_id,
                    channel,
                    n_turns,
                    n_words,
                    json.dumps(survey_data) if survey_data else None,
                    json.dumps(demographics) if demographics else None,
                ),
            )
            row = cur.fetchone()
            spk_id = row[0] if row else -1
        conn.commit()
        return spk_id
    finally:
        conn.close()


def insert_turns(
    params: dict[str, Any],
    conversation_id: int,
    speaker_ids: dict[str, int],
    turns: list[dict],
) -> None:
    """Batch-insert turn rows.

    Parameters
    ----------
    speaker_ids : dict
        Mapping from user_id → speakers table id.
    turns : list[dict]
        Each dict has keys matching the turns table columns.
    """
    conn = _get_connection(params)
    try:
        with conn.cursor() as cur:
            for t in turns:
                cur.execute(
                    """INSERT INTO turns
                       (conversation_id, speaker_id, turn_index,
                        start_time, stop_time, utterance,
                        backchannel, backchannel_count, n_words,
                        is_question, has_overlap)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        conversation_id,
                        speaker_ids.get(t["speaker"]),
                        t["turn_index"],
                        t.get("start_time"),
                        t.get("stop_time"),
                        t["utterance"],
                        t.get("backchannel"),
                        t.get("backchannel_count", 0),
                        t.get("n_words"),
                        t.get("is_question", False),
                        t.get("has_overlap", False),
                    ),
                )
        conn.commit()
    finally:
        conn.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Insert helpers — experiments / embeddings / metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def insert_experiment(
    params: dict[str, Any],
    name: str,
    description: str = "",
    *,
    conversation_id: int | None = None,
    speaker_id: int | None = None,
    analysis_mode: str | None = None,
) -> int:
    """Insert an experiment row and return its id."""
    conn = _get_connection(params)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO experiments
                   (name, description, params,
                    conversation_id, speaker_id, analysis_mode)
                   VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
                (
                    name,
                    description,
                    json.dumps(_serialisable(params)),
                    conversation_id,
                    speaker_id,
                    analysis_mode,
                ),
            )
            exp_id = cur.fetchone()[0]
        conn.commit()
        return exp_id
    finally:
        conn.close()


def insert_embeddings(
    params: dict[str, Any],
    experiment_id: int,
    windows: list[dict],
    embeddings_full: np.ndarray,
    embeddings_reduced: np.ndarray,
) -> None:
    """Batch-insert embedding rows.

    Parameters
    ----------
    windows : list[dict]
        Each dict has keys: index, text, start_token, end_token.
    embeddings_full : ndarray (N, dims_full)
    embeddings_reduced : ndarray (N, dims_reduced)
    """
    conn = _get_connection(params)
    try:
        with conn.cursor() as cur:
            for i, win in enumerate(windows):
                cur.execute(
                    """INSERT INTO embeddings
                       (experiment_id, window_index, window_text,
                        start_token, end_token,
                        embedding_full, embedding_reduced)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (
                        experiment_id,
                        win["index"],
                        win["text"],
                        win.get("start_token"),
                        win.get("end_token"),
                        embeddings_full[i].tolist(),
                        embeddings_reduced[i].tolist(),
                    ),
                )
        conn.commit()
    finally:
        conn.close()


def insert_metrics(
    params: dict[str, Any],
    experiment_id: int,
    metrics_dict: dict[str, np.ndarray],
) -> None:
    """Batch-insert metric rows.

    Parameters
    ----------
    metrics_dict : dict
        Keys match the metrics table columns; values are 1-D arrays of
        length N (one per window).
    """
    conn = _get_connection(params)
    n = len(metrics_dict["cosine_distance"])
    try:
        with conn.cursor() as cur:
            for i in range(n):
                row = {k: _safe_float(v[i]) for k, v in metrics_dict.items()}
                cur.execute(
                    """INSERT INTO metrics
                       (experiment_id, window_index,
                        cosine_distance, euclidean_distance,
                        velocity, acceleration, jerk,
                        cosine_similarity, cosine_sim_d1, cosine_sim_d2,
                        ema_drift, cumulative_path,
                        kalman_innovation, kalman_mahalanobis,
                        is_boundary, is_return, return_cluster_id,
                        is_fixation, cloud_valid)
                       VALUES (%s,%s, %s,%s, %s,%s,%s, %s,%s,%s,
                               %s,%s, %s,%s, %s,%s,%s, %s,%s)""",
                    (
                        experiment_id, i,
                        row.get("cosine_distance"),
                        row.get("euclidean_distance"),
                        row.get("velocity"),
                        row.get("acceleration"),
                        row.get("jerk"),
                        row.get("cosine_similarity"),
                        row.get("cosine_sim_d1"),
                        row.get("cosine_sim_d2"),
                        row.get("ema_drift"),
                        row.get("cumulative_path"),
                        row.get("kalman_innovation"),
                        row.get("kalman_mahalanobis"),
                        bool(row.get("is_boundary", False)),
                        bool(row.get("is_return", False)),
                        int(row.get("return_cluster_id", -1)),
                        bool(row.get("is_fixation", False)),
                        bool(row.get("cloud_valid", True)),
                    ),
                )
        conn.commit()
    finally:
        conn.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Query helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def find_similar(
    params: dict[str, Any],
    query_embedding: np.ndarray,
    *,
    use_reduced: bool = True,
    top_k: int = 10,
    experiment_id: int | None = None,
) -> list[dict]:
    """Find the *top_k* most similar stored embeddings (cosine).

    Returns a list of dicts with keys: window_index, window_text,
    distance, experiment_id.
    """
    col = "embedding_reduced" if use_reduced else "embedding_full"
    where = ""
    bind: list = [query_embedding.tolist(), top_k]
    if experiment_id is not None:
        where = "WHERE experiment_id = %s"
        bind = [query_embedding.tolist(), experiment_id, top_k]

    sql = f"""
        SELECT window_index, window_text, experiment_id,
               {col} <=> %s AS distance
        FROM embeddings
        {where}
        ORDER BY {col} <=> %s
        LIMIT %s
    """
    # Adjust bind order: <=> needs the vector twice in different spots
    conn = _get_connection(params)
    try:
        with conn.cursor() as cur:
            if experiment_id is not None:
                cur.execute(
                    f"""SELECT window_index, window_text, experiment_id,
                               {col} <=> %s::vector AS distance
                        FROM embeddings
                        WHERE experiment_id = %s
                        ORDER BY distance
                        LIMIT %s""",
                    (query_embedding.tolist(), experiment_id, top_k),
                )
            else:
                cur.execute(
                    f"""SELECT window_index, window_text, experiment_id,
                               {col} <=> %s::vector AS distance
                        FROM embeddings
                        ORDER BY distance
                        LIMIT %s""",
                    (query_embedding.tolist(), top_k),
                )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "window_index": r[0],
            "window_text": r[1],
            "experiment_id": r[2],
            "distance": r[3],
        }
        for r in rows
    ]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _safe_float(val) -> float | None:
    """Convert numpy scalar to Python float; NaN → None for SQL."""
    if val is None:
        return None
    f = float(val)
    if np.isnan(f):
        return None
    return f


def _serialisable(obj):
    """Make a params dict JSON-serialisable (numpy types → Python)."""
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialisable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
