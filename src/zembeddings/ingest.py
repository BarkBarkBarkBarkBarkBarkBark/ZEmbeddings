"""
Ingestion module for dyadic conversation corpus.
=================================================

Parses the ``no_media`` dataset structure:

    {uuid}/
        metadata.json
        survey.csv
        audio_video_features.csv
        transcription/
            transcript_backbiter.csv   ← primary (recommended)
            transcript_audiophile.csv
            transcript_cliffhanger.csv
            transcribe_output.json

The **backbiter** format is used by default because it consolidates
backchannels into the primary speaker's turn, producing semantically
coherent utterances suitable for embedding while preserving
backchannel metadata in separate columns.

Usage::

    from zembeddings.ingest import ingest_conversation, walk_conversations

    # Ingest a single conversation
    result = ingest_conversation("/path/to/no_media/uuid-dir", params)

    # Walk and ingest all conversations
    for uuid_dir in walk_conversations("/path/to/no_media"):
        ingest_conversation(uuid_dir, params)

Reference
---------
See ``context/ingestion_strategy.yaml`` for the full ingestion plan.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Parsing helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_metadata(convo_dir: Path) -> dict[str, Any] | None:
    """Parse metadata.json from a conversation directory.

    Returns
    -------
    dict with keys: convo_uuid, session_id, created_at, speakers
    (list of dicts with user_id and channel), or None on failure.
    """
    meta_path = convo_dir / "metadata.json"
    if not meta_path.exists():
        logger.warning("Missing metadata.json in %s", convo_dir.name)
        return None

    try:
        with open(meta_path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Corrupt metadata.json in %s: %s", convo_dir.name, e)
        return None

    speakers = []
    for s in raw.get("speakers", []):
        speakers.append({
            "user_id": s.get("user_id", ""),
            "channel": s.get("channel", ""),
        })

    # Convert epoch ms to ISO timestamp
    created_ms = raw.get("createdAt")
    created_at = None
    if created_ms:
        try:
            created_at = datetime.fromtimestamp(
                created_ms / 1000, tz=timezone.utc
            ).isoformat()
        except (OSError, ValueError, OverflowError):
            pass

    return {
        "convo_uuid": raw.get("id", convo_dir.name),
        "session_id": raw.get("sessionId"),
        "created_at": created_at,
        "speakers": speakers,
        "raw": raw,
    }


def parse_backbiter_transcript(convo_dir: Path) -> list[dict[str, Any]] | None:
    """Parse transcript_backbiter.csv into a list of turn dicts.

    Returns
    -------
    list[dict] with keys: turn_index, speaker, start_time, stop_time,
    utterance, backchannel, backchannel_count, n_words, is_question,
    has_overlap.  Returns None if the file is missing or empty.
    """
    csv_path = convo_dir / "transcription" / "transcript_backbiter.csv"
    if not csv_path.exists():
        logger.warning("Missing transcript_backbiter.csv in %s", convo_dir.name)
        return None

    turns: list[dict[str, Any]] = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    turn = {
                        "turn_index": int(row.get("turn_id", 0)),
                        "speaker": row.get("speaker", ""),
                        "start_time": _safe_parse_float(row.get("start")),
                        "stop_time": _safe_parse_float(row.get("stop")),
                        "utterance": row.get("utterance", "").strip(),
                        "backchannel": row.get("backchannel", "").strip() or None,
                        "backchannel_count": int(row.get("backchannel_count", 0)),
                        "n_words": int(row.get("n_words", 0)),
                        "is_question": (
                            row.get("end_question", "False").strip().lower() == "true"
                        ),
                        "has_overlap": (
                            row.get("overlap", "False").strip().lower() == "true"
                        ),
                    }
                    if turn["utterance"]:
                        turns.append(turn)
                except (ValueError, KeyError) as e:
                    logger.debug("Skipping malformed row in %s: %s", csv_path, e)
                    continue
    except OSError as e:
        logger.error("Error reading %s: %s", csv_path, e)
        return None

    return turns if turns else None


def parse_survey(convo_dir: Path) -> dict[str, dict[str, Any]]:
    """Parse survey.csv into a dict keyed by user_id.

    Returns
    -------
    dict mapping user_id → full survey row (as dict).
    Empty dict if file is missing.
    """
    csv_path = convo_dir / "survey.csv"
    if not csv_path.exists():
        logger.debug("No survey.csv in %s", convo_dir.name)
        return {}

    result: dict[str, dict[str, Any]] = {}
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("user_id", "").strip()
                if uid:
                    # Convert numeric strings to floats where possible
                    cleaned: dict[str, Any] = {}
                    for k, v in row.items():
                        cleaned[k] = _coerce_value(v)
                    result[uid] = cleaned
    except OSError as e:
        logger.debug("Error reading survey.csv in %s: %s", convo_dir.name, e)

    return result


def extract_demographics(survey_row: dict[str, Any]) -> dict[str, Any]:
    """Extract demographic fields from a survey row for quick access."""
    demo_keys = ["sex", "politics", "race", "edu", "employ", "age"]
    return {k: survey_row.get(k) for k in demo_keys if k in survey_row}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ValidationResult:
    """Result of validating a conversation directory."""

    def __init__(self) -> None:
        self.valid = True
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def fail(self, msg: str) -> None:
        self.valid = False
        self.errors.append(msg)

    def __repr__(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        return f"<ValidationResult {status} warnings={len(self.warnings)} errors={len(self.errors)}>"


def validate_conversation(
    convo_dir: Path,
    metadata: dict[str, Any] | None,
    turns: list[dict[str, Any]] | None,
    *,
    min_turns: int = 10,
    min_words: int = 50,
) -> ValidationResult:
    """Validate a conversation for ingestion suitability."""
    result = ValidationResult()

    # Metadata checks
    if metadata is None:
        result.fail("Missing or corrupt metadata.json")
        return result

    if len(metadata.get("speakers", [])) != 2:
        result.fail(
            f"Expected 2 speakers, found {len(metadata.get('speakers', []))}"
        )

    # Transcript checks
    if turns is None:
        result.fail("Missing or empty transcript_backbiter.csv")
        return result

    # Speaker consistency
    transcript_speakers = set(t["speaker"] for t in turns)
    metadata_speakers = set(s["user_id"] for s in metadata.get("speakers", []))

    if len(transcript_speakers) < 2:
        result.fail(f"Only {len(transcript_speakers)} speaker(s) in transcript")

    if metadata_speakers and not transcript_speakers.issubset(metadata_speakers):
        extra = transcript_speakers - metadata_speakers
        result.warn(
            f"Transcript speakers not in metadata: {extra}"
        )

    # Minimum thresholds
    if len(turns) < min_turns:
        result.fail(f"Only {len(turns)} turns (minimum: {min_turns})")

    total_words = sum(t.get("n_words", 0) for t in turns)
    if total_words < min_words:
        result.fail(f"Only {total_words} words (minimum: {min_words})")

    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Text construction for embedding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_speaker_texts(
    turns: list[dict[str, Any]],
    speakers: list[dict[str, Any]],
) -> dict[str, str]:
    """Build per-speaker text from turn list.

    Returns a dict mapping user_id → concatenated utterance text.
    Turns are ordered chronologically for each speaker.
    """
    speaker_turns: dict[str, list[str]] = {}
    for t in turns:
        uid = t["speaker"]
        if uid not in speaker_turns:
            speaker_turns[uid] = []
        speaker_turns[uid].append(t["utterance"])

    return {uid: "\n".join(texts) for uid, texts in speaker_turns.items()}


def build_interleaved_text(
    turns: list[dict[str, Any]],
    speaker_labels: dict[str, str] | None = None,
) -> str:
    """Build interleaved conversation text with speaker labels.

    Parameters
    ----------
    speaker_labels : dict, optional
        Mapping from user_id → display label (e.g. 'SPEAKER_L').
        If None, uses 'SPEAKER_0', 'SPEAKER_1' in order of appearance.
    """
    if speaker_labels is None:
        seen: list[str] = []
        for t in turns:
            if t["speaker"] not in seen:
                seen.append(t["speaker"])
        speaker_labels = {uid: f"SPEAKER_{i}" for i, uid in enumerate(seen)}

    lines: list[str] = []
    for t in turns:
        label = speaker_labels.get(t["speaker"], t["speaker"])
        lines.append(f"{label}: {t['utterance']}")

    return "\n".join(lines)


def write_postured_texts(
    convo_dir_name: str,
    turns: list[dict[str, Any]],
    speakers: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, Path]:
    """Write per-speaker and interleaved text files to disk.

    Returns dict mapping analysis_key → file path.
    """
    out_dir = output_root / convo_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build speaker label mapping from metadata
    speaker_labels = {}
    for s in speakers:
        ch = s.get("channel", "X")
        speaker_labels[s["user_id"]] = f"SPEAKER_{ch}"

    paths: dict[str, Path] = {}

    # Per-speaker texts
    speaker_texts = build_speaker_texts(turns, speakers)
    for uid, text in speaker_texts.items():
        ch = "X"
        for s in speakers:
            if s["user_id"] == uid:
                ch = s.get("channel", "X")
                break
        fpath = out_dir / f"speaker_{ch}.txt"
        fpath.write_text(text, encoding="utf-8")
        paths[f"speaker_{ch}"] = fpath

    # Interleaved text
    interleaved = build_interleaved_text(turns, speaker_labels)
    fpath = out_dir / "interleaved.txt"
    fpath.write_text(interleaved, encoding="utf-8")
    paths["interleaved"] = fpath

    return paths


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Main ingestion orchestrator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class IngestionResult:
    """Container for the result of ingesting one conversation."""

    def __init__(
        self,
        convo_uuid: str,
        valid: bool,
        *,
        conversation_id: int | None = None,
        speaker_ids: dict[str, int] | None = None,
        n_turns: int = 0,
        n_words: int = 0,
        text_paths: dict[str, Path] | None = None,
        validation: ValidationResult | None = None,
    ):
        self.convo_uuid = convo_uuid
        self.valid = valid
        self.conversation_id = conversation_id
        self.speaker_ids = speaker_ids or {}
        self.n_turns = n_turns
        self.n_words = n_words
        self.text_paths = text_paths or {}
        self.validation = validation

    def __repr__(self) -> str:
        status = "OK" if self.valid else "SKIP"
        return (
            f"<IngestionResult {self.convo_uuid[:12]}… "
            f"{status} turns={self.n_turns} words={self.n_words}>"
        )


def ingest_conversation(
    convo_dir: str | Path,
    params: dict[str, Any],
    *,
    store_db: bool = False,
    write_texts: bool = True,
    output_root: Path | None = None,
    min_turns: int = 10,
    min_words: int = 50,
) -> IngestionResult:
    """Parse and ingest a single conversation directory.

    Parameters
    ----------
    convo_dir : Path
        Path to a UUID conversation directory.
    params : dict
        Full PARAMS dict (used for DB connection if store_db=True).
    store_db : bool
        If True, write to PostgreSQL database.
    write_texts : bool
        If True, write postured text files to output_root.
    output_root : Path, optional
        Root directory for postured text output.
        Defaults to ``data/processed``.
    min_turns, min_words : int
        Validation thresholds.

    Returns
    -------
    IngestionResult
    """
    convo_dir = Path(convo_dir)
    convo_uuid = convo_dir.name

    if output_root is None:
        output_root = Path(params["paths"]["data_processed"])

    # ── 1. Parse ──────────────────────────────────────────────────────
    metadata = parse_metadata(convo_dir)
    turns = parse_backbiter_transcript(convo_dir)
    survey = parse_survey(convo_dir)

    # ── 2. Validate ───────────────────────────────────────────────────
    validation = validate_conversation(
        convo_dir, metadata, turns,
        min_turns=min_turns, min_words=min_words,
    )

    if not validation.valid:
        for err in validation.errors:
            logger.info("SKIP %s: %s", convo_uuid[:12], err)
        return IngestionResult(convo_uuid, valid=False, validation=validation)

    # At this point metadata and turns are guaranteed non-None
    assert metadata is not None
    assert turns is not None

    speakers = metadata["speakers"]
    n_turns = len(turns)
    n_words = sum(t.get("n_words", 0) for t in turns)
    duration = max(t.get("stop_time", 0) or 0 for t in turns)

    # ── 3. Build speaker stats ────────────────────────────────────────
    speaker_stats: dict[str, dict[str, int]] = {}
    for t in turns:
        uid = t["speaker"]
        if uid not in speaker_stats:
            speaker_stats[uid] = {"n_turns": 0, "n_words": 0}
        speaker_stats[uid]["n_turns"] += 1
        speaker_stats[uid]["n_words"] += t.get("n_words", 0)

    # ── 4. Store to database ──────────────────────────────────────────
    conversation_id = None
    speaker_ids: dict[str, int] = {}

    if store_db:
        from zembeddings.database import (
            insert_conversation,
            insert_speaker,
            insert_turns,
        )

        conversation_id = insert_conversation(
            params,
            convo_uuid=convo_uuid,
            session_id=metadata.get("session_id"),
            created_at=metadata.get("created_at"),
            duration_seconds=duration,
            n_turns=n_turns,
            n_words=n_words,
            metadata=metadata.get("raw"),
        )

        for s in speakers:
            uid = s["user_id"]
            stats = speaker_stats.get(uid, {"n_turns": 0, "n_words": 0})
            survey_row = survey.get(uid, {})

            spk_id = insert_speaker(
                params,
                conversation_id=conversation_id,
                user_id=uid,
                channel=s.get("channel"),
                n_turns=stats["n_turns"],
                n_words=stats["n_words"],
                survey_data=survey_row if survey_row else None,
                demographics=extract_demographics(survey_row) if survey_row else None,
            )
            speaker_ids[uid] = spk_id

        insert_turns(params, conversation_id, speaker_ids, turns)

    # ── 5. Write postured text files ──────────────────────────────────
    text_paths: dict[str, Path] = {}
    if write_texts:
        text_paths = write_postured_texts(
            convo_uuid, turns, speakers, output_root,
        )

    return IngestionResult(
        convo_uuid,
        valid=True,
        conversation_id=conversation_id,
        speaker_ids=speaker_ids,
        n_turns=n_turns,
        n_words=n_words,
        text_paths=text_paths,
        validation=validation,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Directory walker
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def walk_conversations(root: str | Path) -> list[Path]:
    """List all UUID conversation directories under *root*, sorted."""
    root = Path(root)
    dirs = sorted(
        d for d in root.iterdir()
        if d.is_dir() and len(d.name) == 36 and d.name.count("-") == 4
    )
    return dirs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _safe_parse_float(val: str | None) -> float | None:
    """Parse a string to float, returning None on failure."""
    if val is None or val.strip() == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _coerce_value(val: str | None) -> Any:
    """Coerce a CSV string value to the most appropriate Python type."""
    if val is None or val.strip() == "":
        return None
    v = val.strip()
    # Try int
    try:
        return int(v)
    except ValueError:
        pass
    # Try float
    try:
        return float(v)
    except ValueError:
        pass
    # Boolean-ish
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v
