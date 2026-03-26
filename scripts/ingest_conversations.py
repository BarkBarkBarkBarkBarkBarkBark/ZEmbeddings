#!/usr/bin/env python3
"""
Ingest the no_media corpus into the ZEmbeddings database + text files.
=====================================================================

Walks all UUID conversation directories, validates, parses metadata
/ survey / backbiter transcripts, and:

  1. Writes per-speaker and interleaved text files to data/processed/
  2. Optionally inserts into PostgreSQL (if --db flag is set)

Usage
-----
::

    # Text-only (no database) — the default
    python scripts/ingest_conversations.py /path/to/no_media

    # With database storage
    python scripts/ingest_conversations.py /path/to/no_media --db

    # Limit to first N conversations (useful for testing)
    python scripts/ingest_conversations.py /path/to/no_media --limit 10

    # Resume from a specific conversation UUID
    python scripts/ingest_conversations.py /path/to/no_media --resume <uuid>

    # Dry-run: validate only, no writes
    python scripts/ingest_conversations.py /path/to/no_media --dry-run

"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from zembeddings.ingest import (
    IngestionResult,
    ingest_conversation,
    validate_conversation,
    parse_metadata,
    parse_backbiter_transcript,
    walk_conversations,
)
from zembeddings.params import PARAMS, load_params

logger = logging.getLogger("ingest_conversations")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest the no_media corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "corpus_dir",
        type=Path,
        help="Path to the no_media root directory.",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        default=False,
        help="Write to PostgreSQL database (requires running postgres).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N conversations.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from conversation UUID (skip all before it).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate only — do not write any files or DB rows.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file (merged over defaults).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for postured text files. Default: data/processed/",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=10,
        help="Minimum turns to accept a conversation (default: 10).",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=50,
        help="Minimum total words to accept a conversation (default: 50).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Verbose logging (DEBUG level).",
    )

    args = parser.parse_args()

    # ── Logging ───────────────────────────────────────────────────────
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Params ────────────────────────────────────────────────────────
    if args.config:
        params = load_params(args.config)
    else:
        params = PARAMS.copy()

    if args.db:
        params["database"]["enabled"] = True

    output_root = args.output_dir or Path(params["paths"]["data_processed"])
    output_root = Path(output_root)

    # ── Discover conversations ────────────────────────────────────────
    corpus_dir = args.corpus_dir.resolve()
    if not corpus_dir.is_dir():
        logger.error("Corpus directory not found: %s", corpus_dir)
        sys.exit(1)

    all_dirs = walk_conversations(corpus_dir)
    logger.info("Found %d conversation directories in %s", len(all_dirs), corpus_dir)

    # Resume support
    if args.resume:
        try:
            idx = next(
                i for i, d in enumerate(all_dirs) if d.name == args.resume
            )
            all_dirs = all_dirs[idx:]
            logger.info("Resuming from %s (%d remaining)", args.resume, len(all_dirs))
        except StopIteration:
            logger.error("Resume UUID not found: %s", args.resume)
            sys.exit(1)

    # Limit support
    if args.limit:
        all_dirs = all_dirs[: args.limit]
        logger.info("Limited to %d conversations", len(all_dirs))

    # ── Counters ──────────────────────────────────────────────────────
    total = len(all_dirs)
    ok = 0
    skipped = 0
    errors = 0
    total_turns = 0
    total_words = 0
    t0 = time.time()

    results_log: list[dict] = []

    # ── Process ───────────────────────────────────────────────────────
    for i, convo_dir in enumerate(all_dirs, 1):
        uuid_short = convo_dir.name[:12]
        try:
            if args.dry_run:
                # Validate only
                metadata = parse_metadata(convo_dir)
                turns = parse_backbiter_transcript(convo_dir)
                validation = validate_conversation(
                    convo_dir, metadata, turns,
                    min_turns=args.min_turns, min_words=args.min_words,
                )
                status = "VALID" if validation.valid else "SKIP"
                n_t = len(turns) if turns else 0
                n_w = sum(t.get("n_words", 0) for t in turns) if turns else 0
                logger.info(
                    "[%d/%d] %s… %s  turns=%d words=%d%s",
                    i, total, uuid_short, status, n_t, n_w,
                    f"  errors={validation.errors}" if validation.errors else "",
                )
                if validation.valid:
                    ok += 1
                    total_turns += n_t
                    total_words += n_w
                else:
                    skipped += 1
                results_log.append({
                    "uuid": convo_dir.name,
                    "status": status,
                    "turns": n_t,
                    "words": n_w,
                    "errors": validation.errors,
                    "warnings": validation.warnings,
                })
            else:
                result = ingest_conversation(
                    convo_dir,
                    params,
                    store_db=args.db,
                    write_texts=True,
                    output_root=output_root,
                    min_turns=args.min_turns,
                    min_words=args.min_words,
                )
                if result.valid:
                    ok += 1
                    total_turns += result.n_turns
                    total_words += result.n_words
                    logger.info(
                        "[%d/%d] %s… OK  turns=%d words=%d",
                        i, total, uuid_short, result.n_turns, result.n_words,
                    )
                else:
                    skipped += 1
                    logger.info(
                        "[%d/%d] %s… SKIP  %s",
                        i, total, uuid_short,
                        result.validation.errors if result.validation else "unknown",
                    )
                results_log.append({
                    "uuid": convo_dir.name,
                    "status": "OK" if result.valid else "SKIP",
                    "turns": result.n_turns,
                    "words": result.n_words,
                })

        except Exception as e:
            errors += 1
            logger.error("[%d/%d] %s… ERROR: %s", i, total, uuid_short, e)
            results_log.append({
                "uuid": convo_dir.name,
                "status": "ERROR",
                "error": str(e),
            })

    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Ingestion complete in %.1fs", elapsed)
    logger.info("  Total processed:  %d", total)
    logger.info("  Accepted:         %d", ok)
    logger.info("  Skipped:          %d", skipped)
    logger.info("  Errors:           %d", errors)
    logger.info("  Total turns:      %d", total_turns)
    logger.info("  Total words:      %d", total_words)
    if not args.dry_run:
        logger.info("  Output dir:       %s", output_root.resolve())
    logger.info("=" * 60)

    # Write summary report
    report_dir = Path(params["paths"]["results_reports"])
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "ingestion_report.json"
    report = {
        "corpus_dir": str(corpus_dir),
        "dry_run": args.dry_run,
        "db_enabled": args.db,
        "total": total,
        "accepted": ok,
        "skipped": skipped,
        "errors": errors,
        "total_turns": total_turns,
        "total_words": total_words,
        "elapsed_seconds": round(elapsed, 2),
        "conversations": results_log,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
