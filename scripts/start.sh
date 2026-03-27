#!/usr/bin/env bash
# One-shot MVP: ensure .venv + deps, then run the local smoke experiment.
# Same flags as run_experiment.sh (e.g. --inspect, --inspect-port 8770).
# Pass a YAML path to override the default smoke config.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Keep Hugging Face / sentence-transformers weights inside the repo (ignored by git).
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
mkdir -p "$TRANSFORMERS_CACHE"

DEFAULT_CONFIG="config/experiments/mvp_smoke.yaml"
VENV_PY="$PROJECT_ROOT/.venv/bin/python"
VENV_PIP="$PROJECT_ROOT/.venv/bin/pip"

has_yaml=0
for a in "$@"; do
  case "$a" in
    *.yaml|*.yml) has_yaml=1; break ;;
  esac
done

if [[ $has_yaml -eq 0 ]]; then
  set -- "$DEFAULT_CONFIG" "$@"
fi

if [[ ! -x "$VENV_PY" ]]; then
  echo "→ Creating .venv …"
  python -m venv .venv
  echo "→ pip install -e '.[dev,player]' (first run may take a minute) …"
  "$VENV_PIP" install -U pip
  "$VENV_PIP" install -e ".[dev,player]"
fi

exec "$SCRIPT_DIR/run_experiment.sh" "$@"
