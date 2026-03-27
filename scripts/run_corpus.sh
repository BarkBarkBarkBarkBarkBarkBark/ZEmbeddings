#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ZEmbeddings — Corpus Runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  Run the pipeline on real corpus transcripts from data/processed.
#
#  Usage:
#    ./scripts/run_corpus.sh                              # all conversations
#    ./scripts/run_corpus.sh --limit 5                    # first 5 only
#    ./scripts/run_corpus.sh --uuid 0020a0c5-...          # single conversation
#    ./scripts/run_corpus.sh --posture speaker_L           # left speaker only
#    ./scripts/run_corpus.sh --backend local               # local embeddings
#    ./scripts/run_corpus.sh --window 200 --stride 1       # custom window
#    ./scripts/run_corpus.sh --k-sigma 2.0                 # boundary threshold
#    ./scripts/run_corpus.sh --no-write                    # suppress output files
#    ./scripts/run_corpus.sh --inspect                     # launch local inspector after success
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Colours ──────────────────────────────────────────────────────────
BOLD='\033[1m'
DIM='\033[2m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
MAGENTA='\033[0;35m'
RESET='\033[0m'

VENV=".venv/bin/python"
INSPECT=0
INSPECT_PORT=8766

free_port() {
    local port="$1"
    local pids
    pids=$(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null || true)

    if [[ -z "$pids" ]]; then
        return 0
    fi

    echo -e "  ${YELLOW}▸ Port ${port} already in use — stopping existing listener(s)${RESET}"
    while IFS= read -r pid; do
        [[ -z "$pid" ]] && continue
        echo -e "  ${DIM}  kill ${pid}${RESET}"
        kill "$pid" 2>/dev/null || true
    done <<< "$pids"

    sleep 1
    pids=$(lsof -ti tcp:"$port" -sTCP:LISTEN 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        echo -e "  ${YELLOW}  forcing shutdown on port ${port}${RESET}"
        while IFS= read -r pid; do
            [[ -z "$pid" ]] && continue
            echo -e "  ${DIM}  kill -9 ${pid}${RESET}"
            kill -9 "$pid" 2>/dev/null || true
        done <<< "$pids"
        sleep 1
    fi

    if lsof -ti tcp:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        echo -e "  ${RED}✗ Could not free port ${port}${RESET}"
        return 1
    fi
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --inspect)
            INSPECT=1
            shift
            ;;
        --inspect-port)
            INSPECT=1
            INSPECT_PORT="$2"
            shift 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
    set -- "${POSITIONAL[@]}"
else
    set --
fi

# ── Banner ───────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║                                                          ║"
echo "  ║   🧠  Z E M B E D D I N G S   C O R P U S   R U N     ║"
echo "  ║                                                          ║"
echo "  ║   Pipeline over real conversation transcripts            ║"
echo "  ║   tokenise → embed → metrics → Kalman → report          ║"
echo "  ║                                                          ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Validate ─────────────────────────────────────────────────────────
if [[ ! -f "$VENV" ]]; then
    echo -e "${RED}✗ Virtual environment not found at ${VENV}${RESET}"
    echo -e "${DIM}  Run: python -m venv .venv && .venv/bin/pip install -e '.[dev]'${RESET}"
    exit 1
fi

# Count available conversations
CORPUS="data/processed"
prev_arg=""
for arg in "$@"; do
    if [[ "$prev_arg" == "--corpus" ]]; then
        CORPUS="$arg"
    fi
    prev_arg="$arg"
done

N_DIRS=$(find "$CORPUS" -maxdepth 1 -type d -name '*-*-*-*-*' 2>/dev/null | wc -l | tr -d ' ')
echo -e "  ${DIM}Corpus:       ${RESET}${CORPUS}"
echo -e "  ${DIM}Conversations:${RESET} ${N_DIRS}"
echo -e "  ${DIM}Python:       ${RESET}$($VENV --version)"
echo ""

# ── Run ──────────────────────────────────────────────────────────────
echo -e "${YELLOW}▸ Running pipeline on corpus...${RESET}"
echo ""

START_TIME=$(date +%s)

"$VENV" scripts/run_corpus.py "$@" 2>&1 | while IFS= read -r line; do
    if [[ "$line" == *"✓"* ]]; then
        echo -e "  ${GREEN}${line}${RESET}"
    elif [[ "$line" == *"✗"* ]]; then
        echo -e "  ${RED}${line}${RESET}"
    elif [[ "$line" == *"═"* || "$line" == *"Aggregate"* || "$line" == *"Processed"* ]]; then
        echo -e "  ${BOLD}${line}${RESET}"
    elif [[ "$line" == *"Found"* ]]; then
        echo -e "  ${CYAN}${line}${RESET}"
    else
        echo -e "  ${DIM}${line}${RESET}"
    fi
done

EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "  ${DIM}────────────────────────────────────────────────────${RESET}"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "  ${GREEN}${BOLD}✓ Corpus run complete${RESET} ${DIM}(${ELAPSED}s)${RESET}"
    if [[ $INSPECT -eq 1 ]]; then
        LATEST_METRIC=$(ls -t results/metrics/*.yaml 2>/dev/null | head -1 || true)
        echo ""
        echo -e "  ${YELLOW}▸ Launching inspector on http://127.0.0.1:${INSPECT_PORT}${RESET}"
        free_port "$INSPECT_PORT"
        : >/tmp/zembeddings_inspector.log
        if [[ -n "$LATEST_METRIC" ]]; then
            "$VENV" player/inspector.py --host 127.0.0.1 --port "$INSPECT_PORT" --open-browser --focus-run "$LATEST_METRIC" >/tmp/zembeddings_inspector.log 2>&1 &
        else
            "$VENV" player/inspector.py --host 127.0.0.1 --port "$INSPECT_PORT" --open-browser >/tmp/zembeddings_inspector.log 2>&1 &
        fi
        echo -e "  ${DIM}  Inspector log: /tmp/zembeddings_inspector.log${RESET}"
    fi
else
    echo -e "  ${RED}${BOLD}✗ Corpus run failed${RESET} ${DIM}(${ELAPSED}s)${RESET}"
fi
echo ""

exit $EXIT_CODE
