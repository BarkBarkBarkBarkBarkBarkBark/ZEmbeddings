"""
Tests for the ingestion module.

Tests metadata/transcript parsing, validation, text construction,
and directory walking. Uses fixture data — no real corpus needed.
"""

import sys
import json
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from zembeddings.params import get_params
from zembeddings.ingest import (
    parse_metadata,
    parse_backbiter_transcript,
    parse_survey,
    extract_demographics,
    validate_conversation,
    build_speaker_texts,
    build_interleaved_text,
    write_postured_texts,
    ingest_conversation,
    walk_conversations,
    ValidationResult,
    _safe_parse_float,
    _coerce_value,
)


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def convo_dir(tmp_path):
    """Create a realistic conversation directory with all expected files."""
    d = tmp_path / "abc12345-1234-5678-9abc-def012345678"
    d.mkdir()

    # metadata.json
    metadata = {
        "id": "abc12345-1234-5678-9abc-def012345678",
        "sessionId": "sess-001",
        "createdAt": 1700000000000,  # epoch ms
        "speakers": [
            {"user_id": "alice", "channel": "L"},
            {"user_id": "bob", "channel": "R"},
        ],
    }
    (d / "metadata.json").write_text(json.dumps(metadata))

    # transcription/transcript_backbiter.csv
    trans_dir = d / "transcription"
    trans_dir.mkdir()
    rows = [
        {
            "turn_id": "0", "speaker": "alice", "start": "0.0", "stop": "5.0",
            "utterance": "Hello how are you doing today",
            "backchannel": "", "backchannel_count": "0", "n_words": "6",
            "end_question": "True", "overlap": "False",
        },
        {
            "turn_id": "1", "speaker": "bob", "start": "5.0", "stop": "10.0",
            "utterance": "I am doing well thank you for asking",
            "backchannel": "yeah", "backchannel_count": "1", "n_words": "8",
            "end_question": "False", "overlap": "False",
        },
    ]
    # Add more turns to pass validation (min_turns=10)
    for i in range(2, 12):
        spk = "alice" if i % 2 == 0 else "bob"
        rows.append({
            "turn_id": str(i), "speaker": spk,
            "start": str(i * 5.0), "stop": str((i + 1) * 5.0),
            "utterance": f"This is turn number {i} with enough words to count",
            "backchannel": "", "backchannel_count": "0", "n_words": "10",
            "end_question": "False", "overlap": "False",
        })

    csv_path = trans_dir / "transcript_backbiter.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # survey.csv
    survey_rows = [
        {"user_id": "alice", "sex": "F", "age": "28", "politics": "moderate", "race": "white", "edu": "bachelors", "employ": "employed"},
        {"user_id": "bob", "sex": "M", "age": "35", "politics": "liberal", "race": "asian", "edu": "masters", "employ": "employed"},
    ]
    survey_path = d / "survey.csv"
    with open(survey_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=survey_rows[0].keys())
        writer.writeheader()
        writer.writerows(survey_rows)

    return d


@pytest.fixture
def minimal_convo_dir(tmp_path):
    """Minimal directory with only metadata — no transcript."""
    d = tmp_path / "minimal-1234-5678-9abc-def012345678"
    d.mkdir()
    metadata = {
        "id": d.name,
        "speakers": [
            {"user_id": "alice", "channel": "L"},
            {"user_id": "bob", "channel": "R"},
        ],
    }
    (d / "metadata.json").write_text(json.dumps(metadata))
    return d


# ── Parse tests ──────────────────────────────────────────────────────

class TestParseMetadata:
    def test_valid_metadata(self, convo_dir):
        meta = parse_metadata(convo_dir)
        assert meta is not None
        assert meta["convo_uuid"] == "abc12345-1234-5678-9abc-def012345678"
        assert meta["session_id"] == "sess-001"
        assert len(meta["speakers"]) == 2
        assert meta["speakers"][0]["user_id"] == "alice"
        assert meta["speakers"][0]["channel"] == "L"

    def test_created_at_parsed(self, convo_dir):
        meta = parse_metadata(convo_dir)
        assert meta["created_at"] is not None
        assert "2023" in meta["created_at"]  # epoch 1700000000000 → Nov 2023

    def test_missing_metadata(self, tmp_path):
        d = tmp_path / "no-meta"
        d.mkdir()
        assert parse_metadata(d) is None

    def test_corrupt_json(self, tmp_path):
        d = tmp_path / "corrupt"
        d.mkdir()
        (d / "metadata.json").write_text("{invalid json")
        assert parse_metadata(d) is None


class TestParseBackbiterTranscript:
    def test_valid_transcript(self, convo_dir):
        turns = parse_backbiter_transcript(convo_dir)
        assert turns is not None
        assert len(turns) == 12
        assert turns[0]["speaker"] == "alice"
        assert turns[0]["utterance"] == "Hello how are you doing today"
        assert turns[0]["n_words"] == 6
        assert turns[0]["is_question"] is True

    def test_missing_transcript(self, minimal_convo_dir):
        assert parse_backbiter_transcript(minimal_convo_dir) is None


class TestParseSurvey:
    def test_valid_survey(self, convo_dir):
        survey = parse_survey(convo_dir)
        assert "alice" in survey
        assert "bob" in survey
        assert survey["alice"]["sex"] == "F"
        assert survey["bob"]["age"] == 35  # coerced to int

    def test_missing_survey(self, minimal_convo_dir):
        survey = parse_survey(minimal_convo_dir)
        assert survey == {}


class TestExtractDemographics:
    def test_extracts_keys(self):
        row = {"sex": "F", "age": 28, "politics": "moderate", "extra_key": "ignored"}
        demo = extract_demographics(row)
        assert demo["sex"] == "F"
        assert demo["age"] == 28
        assert "extra_key" not in demo


# ── Validation tests ─────────────────────────────────────────────────

class TestValidateConversation:
    def test_valid_conversation(self, convo_dir):
        meta = parse_metadata(convo_dir)
        turns = parse_backbiter_transcript(convo_dir)
        result = validate_conversation(convo_dir, meta, turns)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_no_metadata(self, convo_dir):
        result = validate_conversation(convo_dir, None, [])
        assert result.valid is False
        assert any("metadata" in e.lower() for e in result.errors)

    def test_no_turns(self, convo_dir):
        meta = parse_metadata(convo_dir)
        result = validate_conversation(convo_dir, meta, None)
        assert result.valid is False

    def test_too_few_turns(self, convo_dir):
        meta = parse_metadata(convo_dir)
        turns = parse_backbiter_transcript(convo_dir)[:3]  # only 3 turns
        result = validate_conversation(convo_dir, meta, turns, min_turns=10)
        assert result.valid is False

    def test_single_speaker(self, convo_dir):
        meta = parse_metadata(convo_dir)
        turns = parse_backbiter_transcript(convo_dir)
        # Make all turns from one speaker
        for t in turns:
            t["speaker"] = "alice"
        result = validate_conversation(convo_dir, meta, turns)
        assert result.valid is False


# ── Text construction tests ──────────────────────────────────────────

class TestBuildSpeakerTexts:
    def test_builds_per_speaker(self, convo_dir):
        turns = parse_backbiter_transcript(convo_dir)
        meta = parse_metadata(convo_dir)
        texts = build_speaker_texts(turns, meta["speakers"])
        assert "alice" in texts
        assert "bob" in texts
        assert "Hello" in texts["alice"]

    def test_turn_order(self, convo_dir):
        turns = parse_backbiter_transcript(convo_dir)
        meta = parse_metadata(convo_dir)
        texts = build_speaker_texts(turns, meta["speakers"])
        # Alice's text should have her turns in order
        assert "Hello" in texts["alice"]


class TestBuildInterleavedText:
    def test_interleaved_output(self, convo_dir):
        turns = parse_backbiter_transcript(convo_dir)
        text = build_interleaved_text(turns)
        lines = text.strip().split("\n")
        assert len(lines) == 12
        assert lines[0].startswith("SPEAKER_0:")
        assert lines[1].startswith("SPEAKER_1:")

    def test_custom_labels(self, convo_dir):
        turns = parse_backbiter_transcript(convo_dir)
        labels = {"alice": "SPEAKER_L", "bob": "SPEAKER_R"}
        text = build_interleaved_text(turns, speaker_labels=labels)
        assert "SPEAKER_L:" in text
        assert "SPEAKER_R:" in text


class TestWritePosturedTexts:
    def test_creates_files(self, convo_dir, tmp_path):
        turns = parse_backbiter_transcript(convo_dir)
        meta = parse_metadata(convo_dir)
        paths = write_postured_texts(
            convo_dir.name, turns, meta["speakers"], tmp_path,
        )
        assert "speaker_L" in paths
        assert "speaker_R" in paths
        assert "interleaved" in paths
        for p in paths.values():
            assert p.exists()
            assert p.stat().st_size > 0


# ── Integration: ingest_conversation ─────────────────────────────────

class TestIngestConversation:
    def test_full_ingest(self, convo_dir, tmp_path):
        p = get_params(**{"paths.data_processed": str(tmp_path)})
        result = ingest_conversation(
            convo_dir, p,
            store_db=False, write_texts=True, output_root=tmp_path,
        )
        assert result.valid is True
        assert result.n_turns == 12
        assert result.n_words > 0
        assert len(result.text_paths) == 3

    def test_skip_invalid(self, minimal_convo_dir, tmp_path):
        p = get_params(**{"paths.data_processed": str(tmp_path)})
        result = ingest_conversation(
            minimal_convo_dir, p,
            store_db=False, write_texts=True, output_root=tmp_path,
        )
        assert result.valid is False


# ── Directory walker ─────────────────────────────────────────────────

class TestWalkConversations:
    def test_finds_uuid_dirs(self, tmp_path):
        # Create some UUID-like directories
        (tmp_path / "abc12345-1234-5678-9abc-def012345678").mkdir()
        (tmp_path / "00000000-0000-0000-0000-000000000001").mkdir()
        (tmp_path / "not-a-uuid").mkdir()  # should be excluded
        (tmp_path / "README.md").touch()  # files excluded

        dirs = walk_conversations(tmp_path)
        assert len(dirs) == 2
        assert all(len(d.name) == 36 for d in dirs)

    def test_sorted_output(self, tmp_path):
        (tmp_path / "zzz00000-0000-0000-0000-000000000000").mkdir()
        (tmp_path / "aaa00000-0000-0000-0000-000000000000").mkdir()
        dirs = walk_conversations(tmp_path)
        assert dirs[0].name < dirs[1].name


# ── Utility functions ────────────────────────────────────────────────

class TestSafeParseFloat:
    def test_valid_float(self):
        assert _safe_parse_float("3.14") == pytest.approx(3.14)

    def test_integer_string(self):
        assert _safe_parse_float("42") == 42.0

    def test_none(self):
        assert _safe_parse_float(None) is None

    def test_empty_string(self):
        assert _safe_parse_float("") is None

    def test_invalid(self):
        assert _safe_parse_float("not-a-number") is None


class TestCoerceValue:
    def test_int(self):
        assert _coerce_value("42") == 42

    def test_float(self):
        assert _coerce_value("3.14") == pytest.approx(3.14)

    def test_bool_true(self):
        assert _coerce_value("true") is True

    def test_bool_false(self):
        assert _coerce_value("False") is False

    def test_string(self):
        assert _coerce_value("hello") == "hello"

    def test_none(self):
        assert _coerce_value(None) is None

    def test_empty(self):
        assert _coerce_value("  ") is None
