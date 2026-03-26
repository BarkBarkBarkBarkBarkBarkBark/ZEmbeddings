"""
Set up the pgvector database schema.
=====================================

Usage::

    python scripts/setup_database.py

Requires PostgreSQL with the ``pgvector`` extension.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zembeddings.params import PARAMS
from zembeddings.database import create_schema, create_indexes


def main() -> None:
    print("╔══════════════════════════════════════════╗")
    print("║  ZEmbeddings — Database Setup            ║")
    print("╚══════════════════════════════════════════╝")
    print()

    params = PARAMS.copy()
    params["database"]["enabled"] = True

    try:
        create_schema(params)
        print()
        print("Note: IVFFlat indexes require data in the table.")
        print("Run an experiment first, then call:")
        print("  python -c \"from zembeddings.database import create_indexes; "
              "from zembeddings.params import PARAMS; create_indexes(PARAMS)\"")
        print()
        print("✓ Database setup complete")
    except Exception as e:
        print(f"✗ Database setup failed: {e}")
        print()
        print("Make sure PostgreSQL is running and pgvector is installed:")
        print("  brew install postgresql pgvector   # macOS")
        print("  sudo apt install postgresql-16-pgvector  # Ubuntu")
        print()
        print("Then create the database:")
        print("  createdb zembeddings")
        sys.exit(1)


if __name__ == "__main__":
    main()
