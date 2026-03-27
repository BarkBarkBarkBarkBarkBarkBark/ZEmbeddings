"""
OpenAI embedding client with batching and optional disk cache.
==============================================================

Handles:
1. Loading the API key from ``.env``
2. Batching window texts into single API calls (up to 2048 per call)
3. Requesting both **full** (1536-d) and **reduced** (256-d) embeddings
4. Optional ``.npz`` cache under ``paths.data_processed`` when
   ``params["model"]["cache_embeddings"]`` is true (or ``use_cache=True``)

Reference
---------
- OpenAI Embeddings guide: https://platform.openai.com/docs/guides/embeddings
- Model ``text-embedding-3-small`` supports the ``dimensions`` param
  for native dimensionality reduction (Matryoshka Representation
  Learning).
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

# Load .env once at import time so OPENAI_API_KEY is available.
load_dotenv()

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  OpenAI helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_client():
    """Instantiate an OpenAI client (reads OPENAI_API_KEY from env)."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.startswith("sk-your"):
        raise EnvironmentError(
            "Set OPENAI_API_KEY in your .env file before embedding."
        )
    return OpenAI(api_key=api_key)


def _batch_embed_openai(
    texts: list[str],
    model: str,
    dimensions: int,
    batch_size: int,
) -> np.ndarray:
    """Call the OpenAI embeddings API in batches."""
    from openai import OpenAI  # noqa: F811 — guarded import

    client = _get_client()
    all_vecs: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(
            input=batch,
            model=model,
            dimensions=dimensions,
        )
        batch_vecs = [
            item.embedding
            for item in sorted(resp.data, key=lambda x: x.index)
        ]
        all_vecs.extend(batch_vecs)
    return np.array(all_vecs, dtype=np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Local (sentence-transformers) helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Module-level cache so the model is loaded once per process.
_LOCAL_MODEL_CACHE: dict[str, Any] = {}


def _get_local_model(model_name: str, device: str = "cpu"):
    """Load a sentence-transformers model (cached)."""
    key = f"{model_name}@{device}"
    if key not in _LOCAL_MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Install sentence-transformers for local embeddings:\n"
                "  pip install sentence-transformers\n"
                "Recommended models (8 GB Mac):\n"
                "  all-MiniLM-L6-v2          384-d   ~80 MB\n"
                "  all-mpnet-base-v2         768-d  ~420 MB\n"
                "  nomic-ai/nomic-embed-text-v1.5  768-d  ~550 MB"
            )
        logger.info("Loading local model %s on %s …", model_name, device)
        _LOCAL_MODEL_CACHE[key] = SentenceTransformer(
            model_name, device=device,
        )
    return _LOCAL_MODEL_CACHE[key]


def _batch_embed_local(
    texts: list[str],
    model_name: str,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    """Embed texts with a local sentence-transformers model."""
    model = _get_local_model(model_name, device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 500,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Cache helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _cache_key(texts: list[str], model: str, dims: int) -> str:
    """Deterministic hash for a batch of texts + model + dims."""
    blob = f"{model}|{dims}|{'||'.join(texts)}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _cache_path(cache_dir: str | Path, key: str) -> Path:
    return Path(cache_dir) / f"emb_{key}.npz"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def embed_texts(
    texts: list[str],
    params: dict[str, Any],
    *,
    use_cache: bool | None = None,
) -> dict[str, np.ndarray]:
    """Embed a list of texts at both full and reduced dimensionality.

    Supports two backends controlled by ``params["model"]["backend"]``:

    - ``"openai"`` — calls the OpenAI API (requires OPENAI_API_KEY).
      Supports native dimensionality reduction via the ``dimensions``
      parameter (Matryoshka Representation Learning).
    - ``"local"`` — uses ``sentence-transformers`` (free, runs on CPU
      or Apple-Silicon MPS).  The reduced embedding is produced by
      simple truncation + re-normalisation (Matryoshka style).

    Parameters
    ----------
    texts : list[str]
        One string per sliding window.
    params : dict
        Full PARAMS dict.
    use_cache : bool, optional
        If set, overrides ``params["model"]["cache_embeddings"]``.
        If *None*, uses the YAML/Python default (usually *False* for MVP).

    Returns
    -------
    dict with keys:
        ``"full"``    – ndarray of shape ``(N, dims_full)``
        ``"reduced"`` – ndarray of shape ``(N, dims_reduced)``
        ``"texts"``   – the input texts (echoed back for alignment)
    """
    if use_cache is None:
        use_cache = bool(params["model"].get("cache_embeddings", False))

    backend: str = params["model"].get("backend", "openai")
    model_name: str = params["model"]["name"]
    dims_full: int = params["model"]["dimensions_full"]
    dims_reduced: int = params["model"]["dimensions_reduced"]
    cache_dir = Path(params["paths"]["data_processed"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve effective model name for cache keys ───────────────────
    effective_model = (
        params["model"].get("local_model", model_name)
        if backend == "local"
        else model_name
    )

    # ── Try cache ─────────────────────────────────────────────────────
    ck_full = _cache_key(texts, effective_model, dims_full)
    ck_red = _cache_key(texts, effective_model, dims_reduced)
    cp_full = _cache_path(cache_dir, ck_full)
    cp_red = _cache_path(cache_dir, ck_red)

    if use_cache and cp_full.exists() and cp_red.exists():
        full = np.load(cp_full)["embeddings"]
        reduced = np.load(cp_red)["embeddings"]
        return {"full": full, "reduced": reduced, "texts": texts}

    # ── Embed ─────────────────────────────────────────────────────────
    if backend == "local":
        local_model = params["model"].get("local_model", "all-MiniLM-L6-v2")
        local_batch = params["model"].get("local_batch_size", 256)
        device = params["model"].get("device", "cpu")

        full = _batch_embed_local(texts, local_model, local_batch, device)
        native_dim = full.shape[1]

        # Override dims_full to match the model's actual output
        dims_full = native_dim

        # Produce reduced via Matryoshka-style truncation + renorm
        if dims_reduced >= native_dim:
            reduced = full.copy()
        else:
            reduced = full[:, :dims_reduced].copy()
            norms = np.linalg.norm(reduced, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            reduced /= norms
    else:
        # OpenAI backend — native dimensionality reduction
        batch_size: int = params["model"]["batch_size"]
        full = _batch_embed_openai(texts, model_name, dims_full, batch_size)
        reduced = _batch_embed_openai(texts, model_name, dims_reduced, batch_size)

    # ── Write cache ───────────────────────────────────────────────────
    if use_cache:
        np.savez_compressed(cp_full, embeddings=full)
        np.savez_compressed(cp_red, embeddings=reduced)

    return {"full": full, "reduced": reduced, "texts": texts}


def load_cached_embeddings(
    cache_dir: str | Path,
    texts: list[str],
    model: str,
    dims: int,
) -> np.ndarray | None:
    """Load cached embeddings if they exist, else return None."""
    ck = _cache_key(texts, model, dims)
    cp = _cache_path(Path(cache_dir), ck)
    if cp.exists():
        return np.load(cp)["embeddings"]
    return None
