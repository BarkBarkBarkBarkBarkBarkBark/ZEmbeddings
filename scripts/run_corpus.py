"""
Run the pipeline on transcripts from the processed corpus.
===========================================================

Walks ``data/processed/`` (or a custom directory), finds interleaved
transcripts, and runs the full pipeline on each one.

Usage::

    python scripts/run_corpus.py                              # all conversations
    python scripts/run_corpus.py --limit 5                    # first 5 only
    python scripts/run_corpus.py --corpus data/processed      # custom dir
    python scripts/run_corpus.py --posture interleaved        # default
    python scripts/run_corpus.py --posture speaker_L          # left speaker only
    python scripts/run_corpus.py --backend local              # local embeddings
    python scripts/run_corpus.py --window 200 --stride 1      # custom tokeniser
    python scripts/run_corpus.py --k-sigma 2.0                # boundary threshold
    python scripts/run_corpus.py --kalman-mode vector         # vector Kalman
    python scripts/run_corpus.py --no-write                   # suppress file output
    python scripts/run_corpus.py --uuid 0020a0c5-1658-4747-99c1-2839e736b481
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zembeddings.params import get_params
from zembeddings.pipeline import run_pipeline


def find_conversations(corpus_dir: Path, posture: str = "interleaved") -> list[tuple[str, Path]]:
    """Find all conversation transcript files in corpus_dir.

    Returns list of (uuid, transcript_path) sorted by uuid.
    """
    results = []
    for d in sorted(corpus_dir.iterdir()):
        if not d.is_dir():
            continue
        # UUID dirs have 36-char names with hyphens
        if len(d.name) != 36 or d.name.count("-") != 4:
            continue
        txt = d / f"{posture}.txt"
        if txt.exists() and txt.stat().st_size > 0:
            results.append((d.name, txt))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run ZEmbeddings pipeline on corpus transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpus", default="data/processed",
                        help="Directory containing UUID conversation folders (default: data/processed)")
    parser.add_argument("--posture", default="interleaved",
                        choices=["interleaved", "speaker_L", "speaker_R"],
                        help="Which transcript posture to analyse (default: interleaved)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of conversations to process")
    parser.add_argument("--uuid", default=None,
                        help="Process only this specific conversation UUID")
    parser.add_argument("--backend", default=None,
                        choices=["openai", "local"],
                        help="Embedding backend (default: from config)")
    parser.add_argument("--window", type=int, default=None,
                        help="Window size in tokens")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride in windows")
    parser.add_argument("--k-sigma", type=float, default=None,
                        help="Boundary detection threshold (k × σ)")
    parser.add_argument("--kalman-mode", default=None,
                        choices=["scalar", "vector"],
                        help="Kalman filter mode")
    parser.add_argument("--no-write", action="store_true",
                        help="Don't write YAML/Markdown output files")
    args = parser.parse_args()

    # Build params overrides
    overrides = {}
    if args.backend:
        overrides["model.backend"] = args.backend
    if args.window:
        overrides["window.size"] = args.window
    if args.stride:
        overrides["window.stride"] = args.stride
    if args.k_sigma:
        overrides["boundary.k_sigma"] = args.k_sigma
    if args.kalman_mode:
        overrides["kalman.mode"] = args.kalman_mode

    params = get_params(**overrides)

    # Find conversations
    corpus_dir = Path(args.corpus)
    if not corpus_dir.is_dir():
        print(f"✗ Corpus directory not found: {corpus_dir}")
        sys.exit(1)

    conversations = find_conversations(corpus_dir, args.posture)

    # Filter by UUID if specified
    if args.uuid:
        conversations = [(u, p) for u, p in conversations if u == args.uuid]
        if not conversations:
            print(f"✗ UUID not found in corpus: {args.uuid}")
            sys.exit(1)

    if args.limit:
        conversations = conversations[:args.limit]

    if not conversations:
        print(f"✗ No {args.posture}.txt transcripts found in {corpus_dir}")
        sys.exit(1)

    print(f"Found {len(conversations)} conversations to process")
    print()

    # Process each
    results_summary = []
    failed = 0
    t0 = time.time()

    for i, (uuid, transcript_path) in enumerate(conversations, 1):
        label = f"[{i}/{len(conversations)}] {uuid[:12]}…"
        try:
            result = run_pipeline(
                str(transcript_path),
                params=params,
                experiment_name=uuid,
                write_outputs=not args.no_write,
            )
            results_summary.append({
                "uuid": uuid,
                "windows": result.metrics.n_windows,
                "boundaries": result.metrics.n_boundaries,
                "returns": result.metrics.n_returns,
                "kalman_v": result.kalman.n_violations,
                "accel_v": result.kalman_accel.n_violations,
                "path": result.metrics.total_path_length,
                "mean_vel": result.metrics.mean_velocity,
            })
            print(f"  ✓ {label}  win={result.metrics.n_windows}  "
                  f"bound={result.metrics.n_boundaries}  "
                  f"kalman={result.kalman.n_violations}  "
                  f"accel={result.kalman_accel.n_violations}  "
                  f"path={result.metrics.total_path_length:.4f}")
        except Exception as e:
            print(f"  ✗ {label}  {e}")
            failed += 1

    elapsed = time.time() - t0
    print()
    print(f"{'═' * 56}")
    print(f"  Processed: {len(results_summary)}")
    print(f"  Failed:    {failed}")
    print(f"  Time:      {elapsed:.1f}s")

    if results_summary:
        import numpy as np
        windows = [r["windows"] for r in results_summary]
        boundaries = [r["boundaries"] for r in results_summary]
        kalman_v = [r["kalman_v"] for r in results_summary]
        accel_v = [r["accel_v"] for r in results_summary]
        paths = [r["path"] for r in results_summary]

        print()
        print(f"  Aggregate statistics:")
        print(f"    Windows:     {np.mean(windows):.0f} ± {np.std(windows):.0f}  (total {np.sum(windows)})")
        print(f"    Boundaries:  {np.mean(boundaries):.1f} ± {np.std(boundaries):.1f}  (total {np.sum(boundaries)})")
        print(f"    Kalman viol: {np.mean(kalman_v):.1f} ± {np.std(kalman_v):.1f}  (total {np.sum(kalman_v)})")
        print(f"    Accel viol:  {np.mean(accel_v):.1f} ± {np.std(accel_v):.1f}  (total {np.sum(accel_v)})")
        print(f"    Path length: {np.mean(paths):.4f} ± {np.std(paths):.4f}")
    print()


if __name__ == "__main__":
    main()
