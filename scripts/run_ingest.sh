#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ZEmbeddings — Quick Ingest
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  Parse and ingest conversations from the raw corpus.
#  No database required — writes postured text files only.
#
#  Usage:
#    ./scripts/run_ingest.sh                              # default: ../no_media
#    ./scripts/run_ingest.sh --corpus /path/to/corpus     # custom corpus
#    ./scripts/run_ingest.sh --limit 10                   # process only 10
#    ./scripts/run_ingest.sh --output data/processed      # custom output dir
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
RESET='\033[0m'

VENV=".venv/bin/python"
CORPUS="../no_media"
LIMIT=""
OUTPUT="data/processed"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --corpus)  CORPUS="$2";  shift 2 ;;
        --limit)   LIMIT="$2";   shift 2 ;;
        --output)  OUTPUT="$2";  shift 2 ;;
        --help|-h)
            head -16 "$0" | tail -12
            exit 0 ;;
        *)
            echo -e "${RED}Unknown argument: $1${RESET}"
            exit 1 ;;
    esac
done

echo ""
echo -e "${CYAN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║                                                      ║"
echo "  ║   📂  Z E M B E D D I N G S   I N G E S T I O N    ║"
echo "  ║                                                      ║"
echo "  ║   Parse corpus → validate → write postured texts     ║"
echo "  ║                                                      ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${RESET}"

if [[ ! -f "$VENV" ]]; then
    echo -e "${RED}✗ Virtual environment not found${RESET}"
    exit 1
fi

echo -e "  ${DIM}Corpus:  ${RESET}${CORPUS}"
echo -e "  ${DIM}Output:  ${RESET}${OUTPUT}"
if [[ -n "$LIMIT" ]]; then
    echo -e "  ${DIM}Limit:   ${RESET}${LIMIT} conversations"
fi
echo ""

# Count available conversations
N_DIRS=$(find "$CORPUS" -maxdepth 1 -type d -name '*-*-*-*-*' 2>/dev/null | wc -l | tr -d ' ')
echo -e "  ${DIM}Found ${N_DIRS} conversation directories${RESET}"
echo ""

PYCMD="$VENV scripts/ingest_conversations.py --corpus \"$CORPUS\" --output \"$OUTPUT\" --no-db"
if [[ -n "$LIMIT" ]]; then
    PYCMD+=" --limit $LIMIT"
fi

echo -e "${YELLOW}▸ Ingesting...${RESET}"
START_TIME=$(date +%s)

eval "$PYCMD" 2>&1 | while IFS= read -r line; do
    if [[ "$line" == *"OK"* ]]; then
        echo -e "  ${GREEN}${line}${RESET}"
    elif [[ "$line" == *"SKIP"* ]]; then
        echo -e "  ${YELLOW}${line}${RESET}"
    elif [[ "$line" == *"ERROR"* ]]; then
        echo -e "  ${RED}${line}${RESET}"
    elif [[ "$line" == *"Summary"* || "$line" == *"Total"* ]]; then
        echo -e "  ${BOLD}${line}${RESET}"
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
    N_PROCESSED=$(find "$OUTPUT" -maxdepth 1 -type d -name '*-*-*-*-*' 2>/dev/null | wc -l | tr -d ' ')
    echo -e "  ${GREEN}${BOLD}✓ Ingestion complete${RESET} ${DIM}— ${N_PROCESSED} conversations processed (${ELAPSED}s)${RESET}"
else
    echo -e "  ${RED}${BOLD}✗ Ingestion failed${RESET} ${DIM}(${ELAPSED}s)${RESET}"
fi
echo ""

exit $EXIT_CODE
