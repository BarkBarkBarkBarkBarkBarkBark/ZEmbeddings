from __future__ import annotations

import argparse
import asyncio
import copy
import threading
import uuid
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.decomposition import PCA

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zembeddings.embeddings import load_cached_embeddings
from zembeddings.ingest import build_interleaved_text, parse_backbiter_transcript, parse_metadata
from zembeddings.kalman import run_acceleration_kalman, run_kalman
from zembeddings.metrics import compute_metrics, semantic_cloud_stats
from zembeddings.params import PARAMS
from zembeddings.pipeline import PipelineResult, run_pipeline
from zembeddings.tokenizer import tokenize


app = FastAPI(title="ZEmbeddings Inspector", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.state.preview_store = {}
app.state.focus_run_id = None


class PreviewRequest(BaseModel):
    run_id: str
    window_size: int | None = None
    stride: int | None = None
    k_sigma: float | None = None
    ema_alpha: float | None = None
    kalman_mode: str | None = None
    process_noise_scale: float | None = None
    measurement_noise_scale: float | None = None
    update_gain_scale: float | None = None
    innovation_threshold: float | None = None


@dataclass
class PreviewComputation:
    result: PipelineResult
    source_label: str
    cache_hit: bool


def _ensure_kalman_defaults(params: dict[str, Any]) -> None:
    params.setdefault("kalman", {})
    for key, val in PARAMS["kalman"].items():
        params["kalman"].setdefault(key, copy.deepcopy(val))



@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "inspector.html"))


@app.get("/api/bootstrap")
async def bootstrap() -> dict[str, Any]:
    runs = discover_runs()
    default_run_id = app.state.focus_run_id or (runs[0]["id"] if runs else None)
    schema = get_schema_payload()
    return {
        "default_run_id": default_run_id,
        "runs": runs,
        "schema": schema,
    }


@app.get("/api/runs")
async def list_runs() -> dict[str, Any]:
    return {"runs": discover_runs()}


@app.get("/api/schema")
async def schema() -> dict[str, Any]:
    return get_schema_payload()


@app.post("/api/recompute")
async def recompute(request: PreviewRequest) -> dict[str, Any]:
    run_meta = get_run_by_id(request.run_id)
    if run_meta is None:
        raise HTTPException(status_code=404, detail="Run not found")

    source_input, source_label = resolve_source_input(run_meta)
    if source_input is None:
        raise HTTPException(status_code=400, detail="Source transcript not found for selected run")

    params = copy.deepcopy(run_meta["params"])
    _ensure_kalman_defaults(params)
    params.setdefault("experiment", {})
    params["experiment"]["source"] = source_label
    params["experiment"]["name"] = run_meta["name"]

    if request.window_size is not None:
        params["window"]["size"] = max(2, int(request.window_size))
    if request.stride is not None:
        params["window"]["stride"] = max(1, int(request.stride))
    if request.k_sigma is not None:
        params["boundary"]["k_sigma"] = float(request.k_sigma)
    if request.ema_alpha is not None:
        params["ema"]["alpha"] = max(0.01, min(0.99, float(request.ema_alpha)))
    if request.kalman_mode is not None:
        params["kalman"]["mode"] = request.kalman_mode
    if request.process_noise_scale is not None:
        params["kalman"]["process_noise_scale"] = max(1e-12, float(request.process_noise_scale))
    if request.measurement_noise_scale is not None:
        params["kalman"]["measurement_noise_scale"] = max(1e-12, float(request.measurement_noise_scale))
    if request.update_gain_scale is not None:
        params["kalman"]["update_gain_scale"] = max(0.0, min(4.0, float(request.update_gain_scale)))
    if request.innovation_threshold is not None:
        params["kalman"]["innovation_threshold"] = max(0.1, float(request.innovation_threshold))

    if params == run_meta["params"]:
        try:
            preview = await asyncio.to_thread(build_saved_preview, run_meta, source_input, source_label)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        preview_id = uuid.uuid4().hex[:12]
        app.state.preview_store[preview_id] = preview
        trim_preview_store()
        return {
            "preview_id": preview_id,
            "cache_hit": True,
            **preview,
        }

    try:
        preview_run = await asyncio.to_thread(
            compute_preview_run,
            source_input,
            source_label,
            params,
            run_meta["name"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    preview = build_preview(run_meta, preview_run.source_label, preview_run.result)
    preview_id = uuid.uuid4().hex[:12]
    app.state.preview_store[preview_id] = preview
    trim_preview_store()
    return {
        "preview_id": preview_id,
        "cache_hit": preview_run.cache_hit,
        **preview,
    }


def _max_stride_for_two_windows(n_tokens: int, window_size: int) -> int:
    """Largest stride such that tokenize() still yields at least two windows."""
    if n_tokens < window_size + 1:
        return 1
    return max(1, n_tokens - window_size)


def _ensure_minimum_windows(transcript, params: dict[str, Any]) -> None:
    nw = len(transcript.windows)
    if nw >= 2:
        return
    w = int(params["window"]["size"])
    st = int(params["window"]["stride"])
    mx = _max_stride_for_two_windows(transcript.n_tokens, w)
    raise ValueError(
        f"Need at least 2 sliding windows (got {nw}). "
        f"Lower stride (try ≤ {mx}) or window size "
        f"(tokens={transcript.n_tokens}, size={w}, stride={st})."
    )


@app.websocket("/ws/stream/{preview_id}")
async def stream_preview(
    websocket: WebSocket,
    preview_id: str,
    pace_ms: int = Query(default=160, ge=20, le=5000),
) -> None:
    preview = app.state.preview_store.get(preview_id)
    if preview is None:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    try:
        await websocket.send_json({
            "type": "start",
            "frames": len(preview["frames"]),
            "preview_id": preview_id,
        })
        for frame in preview["frames"]:
            await websocket.send_json({"type": "frame", "frame": frame})
            await asyncio.sleep(pace_ms / 1000)
        await websocket.send_json({"type": "done", "preview_id": preview_id})
    except WebSocketDisconnect:
        return


def trim_preview_store(max_items: int = 8) -> None:
    keys = list(app.state.preview_store.keys())
    while len(keys) > max_items:
        oldest = keys.pop(0)
        app.state.preview_store.pop(oldest, None)


def discover_runs() -> list[dict[str, Any]]:
    metrics_dir = PROJECT_ROOT / "results" / "metrics"
    if not metrics_dir.exists():
        return []

    runs: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue

        params = payload.get("params") or copy.deepcopy(PARAMS)
        _ensure_kalman_defaults(params)
        summary = payload.get("summary") or {}
        name = (
            payload.get("experiment")
            or params.get("experiment", {}).get("name")
            or path.stem
        )
        run_id = str(path.relative_to(PROJECT_ROOT))
        source_input, source_label = resolve_source_input({
            "id": run_id,
            "name": name,
            "params": params,
            "path": path,
        })
        runs.append({
            "id": run_id,
            "name": name,
            "path": run_id,
            "timestamp": payload.get("timestamp"),
            "windows": summary.get("n_windows"),
            "boundaries": summary.get("n_boundaries"),
            "path_length": summary.get("total_path_length"),
            "source": source_label,
            "source_exists": source_input is not None,
            "params": params,
        })
    return runs


def get_run_by_id(run_id: str) -> dict[str, Any] | None:
    for run in discover_runs():
        if run["id"] == run_id:
            run["path"] = PROJECT_ROOT / run["path"]
            return run
    return None


def resolve_source_input(run_meta: dict[str, Any]) -> tuple[str | None, str | None]:
    params = run_meta.get("params") or {}
    exp = params.get("experiment", {}) if isinstance(params, dict) else {}
    source = exp.get("source")
    if source:
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        if source_path.exists():
            return str(source_path), display_path(source_path)

    name = run_meta.get("name", "")
    if len(name) == 36 and name.count("-") == 4:
        base = PROJECT_ROOT / "data" / "processed" / name
        for posture in ("interleaved.txt", "speaker_L.txt", "speaker_R.txt"):
            candidate = base / posture
            if candidate.exists():
                return str(candidate), display_path(candidate)

        no_media_root = Path("/Users/marco/Downloads/no_media") / name
        if no_media_root.exists():
            metadata = parse_metadata(no_media_root)
            turns = parse_backbiter_transcript(no_media_root)
            if metadata and turns:
                labels = {
                    speaker["user_id"]: f"SPEAKER_{speaker.get('channel', 'X')}"
                    for speaker in metadata.get("speakers", [])
                }
                return (
                    build_interleaved_text(turns, labels),
                    str(no_media_root.relative_to(no_media_root.parent.parent) / "transcription/transcript_backbiter.csv"),
                )
    return None, None


def get_schema_payload() -> dict[str, Any]:
    try:
        import os
        import psycopg2

        db = copy.deepcopy(PARAMS)["database"]
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", db["host"]),
            port=int(os.getenv("DB_PORT", db["port"])),
            dbname=os.getenv("DB_NAME", db["dbname"]),
            user=os.getenv("DB_USER", db["user"]),
            password=os.getenv("DB_PASSWORD", db.get("password", "")),
            connect_timeout=1,
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "error": "PostgreSQL is not running locally. Schema browsing is optional.",
            "tables": [],
        }

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name, column_name, data_type, udt_name, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position
                """
            )
            rows = cur.fetchall()
        tables: dict[str, list[dict[str, Any]]] = {}
        for table_name, column_name, data_type, udt_name, is_nullable in rows:
            tables.setdefault(table_name, []).append({
                "name": column_name,
                "data_type": data_type,
                "udt_name": udt_name,
                "nullable": is_nullable == "YES",
            })
        return {
            "status": "ok",
            "error": None,
            "tables": [
                {"name": name, "columns": columns}
                for name, columns in sorted(tables.items())
            ],
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "error": f"Schema query failed: {exc}",
            "tables": [],
        }
    finally:
        conn.close()


def build_preview(run_meta: dict[str, Any], source_label: str, result: PipelineResult) -> dict[str, Any]:
    coords = project_trajectory(result)
    frames = build_frames(result)
    timeseries = {
        "velocity": safe_array(result.metrics.velocity),
        "acceleration": safe_array(result.metrics.acceleration),
        "jerk": safe_array(result.metrics.jerk),
        "cosine_distance": safe_array(result.metrics.cosine_distance),
        "ema_drift": safe_array(result.metrics.ema_drift),
        "cumulative_path": safe_array(result.metrics.cumulative_path),
        "kalman_mahalanobis": safe_array(result.kalman.mahalanobis_distances),
        "kalman_accel_mahalanobis": safe_array(result.kalman_accel.mahalanobis_distances),
        "boundary_flags": [bool(v) for v in result.metrics.boundary_flags.tolist()],
        "return_flags": [bool(v) for v in result.metrics.return_flags.tolist()],
        "return_cluster_id": [int(v) for v in result.metrics.return_cluster_ids.tolist()],
        "kalman_violations": [bool(v) for v in result.kalman.violation_flags.tolist()],
        "kalman_accel_violations": [bool(v) for v in result.kalman_accel.violation_flags.tolist()],
    }
    summary = {
        "windows": result.metrics.n_windows,
        "tokens": result.transcript.n_tokens,
        "boundaries": result.metrics.n_boundaries,
        "returns": result.metrics.n_returns,
        "kalman_violations": result.kalman.n_violations,
        "kalman_accel_violations": result.kalman_accel.n_violations,
        "path_length": round(float(result.metrics.total_path_length), 6),
        "mean_velocity": round(float(result.metrics.mean_velocity), 6),
        "cloud_mean_sim": round(float(result.cloud_stats["mean_pairwise_sim"]), 6),
        "cloud_std_sim": round(float(result.cloud_stats["std_pairwise_sim"]), 6),
    }
    return {
        "run": {
            "id": run_meta["id"],
            "name": run_meta["name"],
            "source": source_label,
        },
        "params": {
            "window_size": result.params["window"]["size"],
            "stride": result.params["window"]["stride"],
            "k_sigma": result.params["boundary"]["k_sigma"],
            "ema_alpha": result.params["ema"]["alpha"],
            "kalman_mode": result.params["kalman"]["mode"],
            "kalman_innovation_threshold": float(
                result.params["kalman"].get("innovation_threshold", 3.0)
            ),
            "kalman_process_noise_scale": float(
                result.params["kalman"].get("process_noise_scale", 1e-4)
            ),
            "kalman_measurement_noise_scale": float(
                result.params["kalman"].get("measurement_noise_scale", 1e-2)
            ),
            "kalman_update_gain_scale": float(
                result.params["kalman"].get("update_gain_scale", 1.0)
            ),
        },
        "summary": summary,
        "trajectory_3d": coords,
        "timeseries": timeseries,
        "frames": frames,
        **_embedding_meta_from_reduced(result.embeddings_reduced, frames),
        "transcript_text": result.transcript.raw_text,
        "replay": {
            "kind": "offline",
            "note": (
                "This inspector replays a precomputed run from disk. "
                "Play steps through sliding windows at a fixed pace — not live audio or real-time ASR."
            ),
        },
    }


def compute_preview_run(
    source_input: str,
    source_label: str,
    params: dict[str, Any],
    run_name: str,
) -> PreviewComputation:
    transcript = tokenize(source_input, params)
    _ensure_minimum_windows(transcript, params)
    texts = [window.text for window in transcript.windows]
    full, reduced = load_embeddings_from_cache(texts, params)

    if full is not None and reduced is not None:
        metrics = compute_metrics(full, params)
        kalman = run_kalman(full, reduced, metrics.cosine_distance, params)
        kalman_accel = run_acceleration_kalman(metrics.acceleration, params)
        cloud_stats = semantic_cloud_stats(full)
        result = PipelineResult(
            params=params,
            transcript=transcript,
            embeddings_full=full,
            embeddings_reduced=reduced,
            metrics=metrics,
            kalman=kalman,
            kalman_accel=kalman_accel,
            cloud_stats=cloud_stats,
            experiment_name=f"{run_name}_inspector",
        )
        return PreviewComputation(result=result, source_label=source_label, cache_hit=True)

    result = run_pipeline(
        source_input,
        params,
        experiment_name=f"{run_name}_inspector",
        write_outputs=False,
        store_db=False,
    )
    return PreviewComputation(result=result, source_label=source_label, cache_hit=False)


def load_embeddings_from_cache(
    texts: list[str],
    params: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    model = params["model"]
    backend = model.get("backend", "openai")
    effective_model = model.get("local_model", model["name"]) if backend == "local" else model["name"]
    cache_dir = PROJECT_ROOT / params["paths"]["data_processed"]
    full = load_cached_embeddings(cache_dir, texts, effective_model, model["dimensions_full"])
    reduced = load_cached_embeddings(cache_dir, texts, effective_model, model["dimensions_reduced"])
    return full, reduced


def build_saved_preview(
    run_meta: dict[str, Any],
    source_input: str,
    source_label: str,
) -> dict[str, Any]:
    _ensure_kalman_defaults(run_meta["params"])
    payload = yaml.safe_load(Path(run_meta["path"]).read_text(encoding="utf-8")) or {}
    transcript = tokenize(source_input, run_meta["params"])
    _ensure_minimum_windows(transcript, run_meta["params"])
    texts = [window.text for window in transcript.windows]
    _full, reduced = load_embeddings_from_cache(texts, run_meta["params"])

    if reduced is None:
        preview_run = compute_preview_run(source_input, source_label, run_meta["params"], run_meta["name"])
        return build_preview(run_meta, source_label, preview_run.result)

    timeseries = payload.get("timeseries", {})
    summary = payload.get("summary", {})
    threshold = float(run_meta["params"]["kalman"].get("innovation_threshold", 3.0))
    coords = project_embeddings(reduced)
    frames = build_frames_from_timeseries(transcript, timeseries, threshold, reduced=reduced)
    return {
        "run": {
            "id": run_meta["id"],
            "name": run_meta["name"],
            "source": source_label,
        },
        "params": {
            "window_size": run_meta["params"]["window"]["size"],
            "stride": run_meta["params"]["window"]["stride"],
            "k_sigma": run_meta["params"]["boundary"]["k_sigma"],
            "ema_alpha": run_meta["params"]["ema"]["alpha"],
            "kalman_mode": run_meta["params"]["kalman"]["mode"],
            "kalman_innovation_threshold": float(
                run_meta["params"]["kalman"].get("innovation_threshold", 3.0)
            ),
            "kalman_process_noise_scale": float(
                run_meta["params"]["kalman"].get("process_noise_scale", 1e-4)
            ),
            "kalman_measurement_noise_scale": float(
                run_meta["params"]["kalman"].get("measurement_noise_scale", 1e-2)
            ),
            "kalman_update_gain_scale": float(
                run_meta["params"]["kalman"].get("update_gain_scale", 1.0)
            ),
        },
        "summary": {
            "windows": summary.get("n_windows", transcript.n_tokens),
            "tokens": summary.get("n_tokens", transcript.n_tokens),
            "boundaries": summary.get("n_boundaries", 0),
            "returns": summary.get("n_returns", 0),
            "kalman_violations": summary.get("kalman_violations", 0),
            "kalman_accel_violations": summary.get("kalman_accel_violations", 0),
            "path_length": round(float(summary.get("total_path_length", 0.0)), 6),
            "mean_velocity": round(float(summary.get("mean_velocity", 0.0)), 6),
            "cloud_mean_sim": round(float(summary.get("cloud_mean_sim", 0.0)), 6),
            "cloud_std_sim": round(float(summary.get("cloud_std_sim", 0.0)), 6),
        },
        "trajectory_3d": coords,
        "timeseries": sanitize_timeseries(timeseries, threshold),
        "frames": frames,
        **_embedding_meta_from_reduced(reduced, frames),
        "transcript_text": transcript.raw_text,
        "replay": {
            "kind": "offline",
            "note": (
                "This inspector replays a precomputed run from disk. "
                "Play steps through sliding windows at a fixed pace — not live audio or real-time ASR."
            ),
        },
    }


def sanitize_timeseries(timeseries: dict[str, Any], threshold: float) -> dict[str, Any]:
    kalman_mahalanobis = sanitize_series(timeseries.get("kalman_mahalanobis", []))
    kalman_accel_mahalanobis = sanitize_series(timeseries.get("kalman_accel_mahalanobis", []))
    return {
        "velocity": sanitize_series(timeseries.get("velocity", [])),
        "acceleration": sanitize_series(timeseries.get("acceleration", [])),
        "jerk": sanitize_series(timeseries.get("jerk", [])),
        "cosine_distance": sanitize_series(timeseries.get("cosine_distance", [])),
        "ema_drift": sanitize_series(timeseries.get("ema_drift", [])),
        "cumulative_path": sanitize_series(timeseries.get("cumulative_path", [])),
        "kalman_mahalanobis": kalman_mahalanobis,
        "kalman_accel_mahalanobis": kalman_accel_mahalanobis,
        "boundary_flags": [bool(v) for v in timeseries.get("boundary_flags", [])],
        "return_flags": [bool(v) for v in timeseries.get("return_flags", [])],
        "return_cluster_id": [int(v) for v in timeseries.get("return_cluster_id", [])],
        "kalman_violations": [value is not None and float(value) > threshold for value in kalman_mahalanobis],
        "kalman_accel_violations": [value is not None and float(value) > threshold for value in kalman_accel_mahalanobis],
    }


def sanitize_series(values: list[Any]) -> list[float | None]:
    return [safe_value(value) for value in values]


def project_embeddings(embeddings: np.ndarray) -> list[list[float]]:
    if embeddings.shape[0] == 0:
        return []
    if embeddings.shape[1] < 3:
        pad = np.zeros((embeddings.shape[0], 3 - embeddings.shape[1]))
        coords = np.concatenate([embeddings, pad], axis=1)
        return np.round(coords[:, :3], 6).tolist()
    if embeddings.shape[0] >= 3:
        coords = PCA(n_components=3).fit_transform(embeddings)
    else:
        coords = embeddings[:, :3]
    return np.round(coords, 6).tolist()


def build_frames_from_timeseries(
    transcript,
    timeseries: dict[str, Any],
    threshold: float,
    reduced: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    previous_end = -1
    velocity = sanitize_series(timeseries.get("velocity", []))
    acceleration = sanitize_series(timeseries.get("acceleration", []))
    jerk = sanitize_series(timeseries.get("jerk", []))
    cosine_distance = sanitize_series(timeseries.get("cosine_distance", []))
    ema_drift = sanitize_series(timeseries.get("ema_drift", []))
    cumulative_path = sanitize_series(timeseries.get("cumulative_path", []))
    kalman_mahalanobis = sanitize_series(timeseries.get("kalman_mahalanobis", []))
    kalman_accel_mahalanobis = sanitize_series(timeseries.get("kalman_accel_mahalanobis", []))
    boundary_flags = [bool(v) for v in timeseries.get("boundary_flags", [])]
    return_flags = [bool(v) for v in timeseries.get("return_flags", [])]
    return_cluster_id = [int(v) for v in timeseries.get("return_cluster_id", [])]

    for i, window in enumerate(transcript.windows):
        delta_ids = transcript.token_ids[previous_end + 1 : window.end_token + 1]
        delta_text = decode_tokens(transcript.encoding_name, delta_ids)
        previous_end = window.end_token
        frames.append({
            "index": i,
            "window_text": window.text,
            "delta_text": delta_text,
            "start_token": window.start_token,
            "end_token": window.end_token,
            "velocity": velocity[i] if i < len(velocity) else None,
            "acceleration": acceleration[i] if i < len(acceleration) else None,
            "jerk": jerk[i] if i < len(jerk) else None,
            "cosine_distance": cosine_distance[i] if i < len(cosine_distance) else None,
            "ema_drift": ema_drift[i] if i < len(ema_drift) else None,
            "cumulative_path": cumulative_path[i] if i < len(cumulative_path) else None,
            "kalman_mahalanobis": kalman_mahalanobis[i] if i < len(kalman_mahalanobis) else None,
            "kalman_accel_mahalanobis": kalman_accel_mahalanobis[i] if i < len(kalman_accel_mahalanobis) else None,
            "boundary_flag": boundary_flags[i] if i < len(boundary_flags) else False,
            "return_flag": return_flags[i] if i < len(return_flags) else False,
            "return_cluster_id": return_cluster_id[i] if i < len(return_cluster_id) else -1,
            "kalman_violation": (kalman_mahalanobis[i] if i < len(kalman_mahalanobis) else None) is not None and (kalman_mahalanobis[i] if i < len(kalman_mahalanobis) else 0.0) > threshold,
            "kalman_accel_violation": (kalman_accel_mahalanobis[i] if i < len(kalman_accel_mahalanobis) else None) is not None and (kalman_accel_mahalanobis[i] if i < len(kalman_accel_mahalanobis) else 0.0) > threshold,
        })
        if reduced is not None and i < reduced.shape[0]:
            frames[-1]["embedding_chip"] = embedding_chip(reduced[i])
        elif "embedding_chip" not in frames[-1]:
            frames[-1]["embedding_chip"] = []
    return frames


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


EMBEDDING_CHIP_MAX = 64


def embedding_chip(row: np.ndarray, max_dims: int = EMBEDDING_CHIP_MAX) -> list[float]:
    """Evenly sample up to *max_dims* components from a reduced embedding row for JSON payloads."""
    vec = np.asarray(row, dtype=np.float64).ravel()
    n = int(vec.shape[0])
    if n == 0:
        return []
    if n <= max_dims:
        return np.round(vec, 6).tolist()
    idx = np.linspace(0, n - 1, max_dims, dtype=int)
    return np.round(vec[idx], 6).tolist()


def _embedding_meta_from_reduced(reduced: np.ndarray, frames: list[dict]) -> dict[str, Any]:
    if reduced.size == 0 or not frames:
        return {
            "embedding_chip_len": 0,
            "embedding_source_dim": 0,
            "embedding_note": "",
        }
    source_dim = int(reduced.shape[1])
    chip_n = len(frames[0].get("embedding_chip") or [])
    return {
        "embedding_chip_len": chip_n,
        "embedding_source_dim": source_dim,
        "embedding_note": (
            f"{chip_n} of {source_dim} reduced dimensions (evenly sampled)."
        ),
    }


def build_frames(result: PipelineResult) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    previous_end = -1
    for i, window in enumerate(result.transcript.windows):
        delta_ids = result.transcript.token_ids[previous_end + 1 : window.end_token + 1]
        delta_text = decode_tokens(result.transcript.encoding_name, delta_ids)
        previous_end = window.end_token
        emb_row = result.embeddings_reduced[i] if result.embeddings_reduced.size else np.array([])
        frames.append({
            "index": i,
            "window_text": window.text,
            "delta_text": delta_text,
            "start_token": window.start_token,
            "end_token": window.end_token,
            "embedding_chip": embedding_chip(emb_row) if emb_row.size else [],
            "velocity": safe_value(result.metrics.velocity[i]),
            "acceleration": safe_value(result.metrics.acceleration[i]),
            "jerk": safe_value(result.metrics.jerk[i]),
            "cosine_distance": safe_value(result.metrics.cosine_distance[i]),
            "ema_drift": safe_value(result.metrics.ema_drift[i]),
            "cumulative_path": safe_value(result.metrics.cumulative_path[i]),
            "kalman_mahalanobis": safe_value(result.kalman.mahalanobis_distances[i]),
            "kalman_accel_mahalanobis": safe_value(result.kalman_accel.mahalanobis_distances[i]),
            "boundary_flag": bool(result.metrics.boundary_flags[i]),
            "return_flag": bool(result.metrics.return_flags[i]),
            "return_cluster_id": int(result.metrics.return_cluster_ids[i]),
            "kalman_violation": bool(result.kalman.violation_flags[i]),
            "kalman_accel_violation": bool(result.kalman_accel.violation_flags[i]),
        })
    return frames


def decode_tokens(encoding_name: str, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)
    return enc.decode(token_ids)


def project_trajectory(result: PipelineResult) -> list[list[float]]:
    emb = result.embeddings_reduced if result.embeddings_reduced.size else result.embeddings_full
    if emb.shape[0] == 0:
        return []
    if emb.shape[1] < 3:
        pad = np.zeros((emb.shape[0], 3 - emb.shape[1]))
        coords = np.concatenate([emb, pad], axis=1)
        return np.round(coords[:, :3], 6).tolist()
    if emb.shape[0] >= 3:
        coords = PCA(n_components=3).fit_transform(emb)
    else:
        coords = emb[:, :3]
    return np.round(coords, 6).tolist()


def safe_array(values: np.ndarray) -> list[float | None]:
    return [safe_value(value) for value in values.tolist()]


def safe_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(numeric):
        return None
    return round(numeric, 6)


def main() -> None:
    parser = argparse.ArgumentParser(description="ZEmbeddings inspector")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--focus-run", default=None, help="Workspace-relative metrics YAML path to preselect")
    args = parser.parse_args()

    if args.focus_run:
        focus_path = Path(args.focus_run)
        if not focus_path.is_absolute():
            focus_path = PROJECT_ROOT / focus_path
        try:
            app.state.focus_run_id = str(focus_path.relative_to(PROJECT_ROOT))
        except ValueError:
            app.state.focus_run_id = None

    if args.open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{args.host}:{args.port}/")).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
