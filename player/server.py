"""
ZEmbeddings Live Player — Local-only FastAPI server.

Captures audio via WebSocket, transcribes with Whisper (local),
embeds with sentence-transformers, runs the full metric + Kalman
pipeline, and streams results back to the browser for 3D visualisation.

Usage:
    python player/server.py [--port 8765] [--whisper-model base] [--embed-model all-MiniLM-L6-v2]
"""

import asyncio
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sklearn.decomposition import PCA

# ZEmbeddings imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zembeddings.params import get_params
from zembeddings.tokenizer import tokenize_and_window
from zembeddings.metrics import compute_metrics, semantic_cloud_stats
from zembeddings.kalman import run_scalar_kalman, run_acceleration_kalman

logger = logging.getLogger("zembeddings.player")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Models (lazy-loaded singletons)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_whisper_model = None
_embed_model = None


def get_whisper(model_name: str = "base"):
    global _whisper_model
    if _whisper_model is None:
        import whisper
        logger.info("Loading Whisper model '%s'...", model_name)
        _whisper_model = whisper.load_model(model_name)
        logger.info("Whisper ready.")
    return _whisper_model


def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model '%s'...", model_name)
        _embed_model = SentenceTransformer(model_name)
        logger.info("Embedder ready (%d dims).", _embed_model.get_sentence_embedding_dimension())
    return _embed_model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Session state
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Session:
    """Holds the rolling state for one connected client."""

    def __init__(self, params: dict[str, Any]):
        self.params = params
        self.transcript_chunks: list[str] = []
        self.full_text: str = ""
        self.embeddings: list[np.ndarray] = []
        self.pca = PCA(n_components=3)
        self.pca_fitted = False
        self.window_texts: list[str] = []
        self.step = 0

    def add_chunk(self, text: str) -> dict[str, Any] | None:
        """Process a new transcript chunk through the pipeline."""
        if not text.strip():
            return None

        self.transcript_chunks.append(text)
        self.full_text = " ".join(self.transcript_chunks)
        self.step += 1

        # Tokenise
        transcript = tokenize_and_window(self.full_text, self.params)
        if transcript.n_windows < 3:
            return {
                "type": "waiting",
                "step": self.step,
                "transcript": self.full_text,
                "n_windows": transcript.n_windows,
                "message": f"Accumulating text... ({transcript.n_windows} windows, need ≥3)",
            }

        # Embed
        model = get_embedder(self.params["model"].get("local_model", "all-MiniLM-L6-v2"))
        embeddings = model.encode(transcript.window_texts, show_progress_bar=False)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Metrics
        metrics = compute_metrics(embeddings, self.params)
        cloud = semantic_cloud_stats(embeddings)

        # Kalman (scalar on cosine distance)
        kalman = run_scalar_kalman(metrics.cosine_distance, self.params)
        kalman_accel = run_acceleration_kalman(metrics.acceleration, self.params)

        # PCA → 3D for visualisation
        if embeddings.shape[0] >= 3:
            coords_3d = self.pca.fit_transform(embeddings).tolist()
            self.pca_fitted = True
        else:
            coords_3d = embeddings[:, :3].tolist()

        # Build response
        N = metrics.n_windows
        return {
            "type": "update",
            "step": self.step,
            "transcript": self.full_text,
            "n_windows": N,
            "latest_chunk": text,

            # 3D trajectory (PCA-projected)
            "trajectory_3d": coords_3d,

            # Scalar timeseries (JSON-safe)
            "velocity": _safe_list(metrics.velocity),
            "acceleration": _safe_list(metrics.acceleration),
            "jerk": _safe_list(metrics.jerk),
            "cosine_distance": _safe_list(metrics.cosine_distance),
            "ema_drift": _safe_list(metrics.ema_drift),
            "cumulative_path": _safe_list(metrics.cumulative_path),

            # Kalman
            "kalman_mahalanobis": _safe_list(kalman.mahalanobis_distances),
            "kalman_violations": kalman.violation_flags.tolist(),
            "kalman_accel_mahalanobis": _safe_list(kalman_accel.mahalanobis_distances),
            "kalman_accel_violations": kalman_accel.violation_flags.tolist(),

            # Boundaries
            "boundary_flags": metrics.boundary_flags.tolist(),
            "n_boundaries": int(metrics.n_boundaries),
            "n_kalman_violations": int(kalman.n_violations),
            "n_accel_violations": int(kalman_accel.n_violations),

            # Cloud
            "cloud_mean_sim": float(cloud["mean_pairwise_sim"]),
            "cloud_std_sim": float(cloud["std_pairwise_sim"]),

            # Summary
            "total_path_length": float(metrics.total_path_length),
            "mean_velocity": float(np.nanmean(metrics.velocity)),
        }


def _safe_list(arr: np.ndarray) -> list[float | None]:
    """Convert numpy array to JSON-safe list (NaN → null)."""
    return [None if np.isnan(v) else round(float(v), 6) for v in arr]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FastAPI app
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = FastAPI(title="ZEmbeddings Live Player", version="0.1.0")

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "whisper_loaded": _whisper_model is not None,
            "embedder_loaded": _embed_model is not None}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")

    # Build params for this session
    params = get_params(**{
        "model.backend": "local",
        "model.local_model": app.state.embed_model,
        "model.dimensions_full": 384,
        "model.dimensions_reduced": 128,
        "window.size": 100,
        "window.stride": 1,
        "kalman.mode": "scalar",
        "kalman.innovation_threshold": 2.5,
    })

    session = Session(params)

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                # Audio data — transcribe with Whisper
                audio_bytes = data["bytes"]
                text = await _transcribe_audio(audio_bytes)
                if text:
                    result = session.add_chunk(text)
                    if result:
                        await websocket.send_json(result)

            elif "text" in data:
                # Text input (for testing without mic)
                msg = json.loads(data["text"])
                if msg.get("type") == "text":
                    result = session.add_chunk(msg["text"])
                    if result:
                        await websocket.send_json(result)
                elif msg.get("type") == "reset":
                    session = Session(params)
                    await websocket.send_json({"type": "reset"})

    except WebSocketDisconnect:
        logger.info("Client disconnected")


async def _transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio bytes with Whisper (runs in thread pool)."""
    import tempfile
    import wave

    whisper_model = get_whisper(app.state.whisper_model)

    # Write to temp WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        # Assume 16kHz mono 16-bit PCM
        with wave.open(f, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(audio_bytes)

    # Transcribe in thread pool to not block the event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: whisper_model.transcribe(tmp_path, language="en", fp16=False),
    )

    Path(tmp_path).unlink(missing_ok=True)
    return result.get("text", "").strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="ZEmbeddings Live Player")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--whisper-model", default="base",
                        choices=["tiny", "base", "small", "medium"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model name")
    args = parser.parse_args()

    app.state.whisper_model = args.whisper_model
    app.state.embed_model = args.embed_model

    print(f"\n  🎧 ZEmbeddings Live Player")
    print(f"  ─────────────────────────────")
    print(f"  Server:   http://{args.host}:{args.port}")
    print(f"  Whisper:  {args.whisper_model}")
    print(f"  Embedder: {args.embed_model}")
    print(f"  Mode:     local only (no API keys needed)")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
