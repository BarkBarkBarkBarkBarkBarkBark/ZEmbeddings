#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ZEmbeddings — Experiment Runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  Run the pipeline using a YAML experiment config.
#
#  Usage:
#    ./scripts/run_experiment.sh                                          # default: exp_001
#    ./scripts/run_experiment.sh config/experiments/exp_001_synthetic_topic_shift.yaml
#    ./scripts/run_experiment.sh --list                                   # list available configs
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
set -euo pipefail

# ── Resolve project root ─────────────────────────────────────────────
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

# ── Parse arguments ──────────────────────────────────────────────────
CONFIG=""

if [[ $# -eq 0 ]]; then
    CONFIG="config/experiments/exp_001_synthetic_topic_shift.yaml"
elif [[ "$1" == "--list" ]]; then
    echo ""
    echo -e "${CYAN}${BOLD}Available experiment configs:${RESET}"
    echo ""
    for f in config/experiments/*.yaml; do
        if [[ -f "$f" ]]; then
            name=$(grep -m1 'name:' "$f" 2>/dev/null | sed 's/.*name: *"\{0,1\}\([^"]*\)"\{0,1\}/\1/' || echo "")
            echo -e "  ${GREEN}$f${RESET}"
            [[ -n "$name" ]] && echo -e "    ${DIM}name: ${name}${RESET}"
            echo ""
        fi
    done
    exit 0
elif [[ "$1" == "--help" || "$1" == "-h" ]]; then
    head -14 "$0" | tail -8
    exit 0
else
    CONFIG="$1"
fi

# ── Banner ───────────────────────────────────────────────────────────
echo ""
echo -e "${MAGENTA}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║                                                          ║"
echo "  ║   🔬  Z E M B E D D I N G S   E X P E R I M E N T      ║"
echo "  ║                                                          ║"
echo "  ║   Semantic trajectory analysis pipeline                  ║"
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

if [[ ! -f "$CONFIG" ]]; then
    echo -e "${RED}✗ Config not found: ${CONFIG}${RESET}"
    echo -e "${DIM}  Run: ./scripts/run_experiment.sh --list  to see available configs${RESET}"
    exit 1
fi

# ── Display config ───────────────────────────────────────────────────
EXP_NAME=$(grep -m1 'name:' "$CONFIG" 2>/dev/null | sed 's/.*name: *"\{0,1\}\([^"]*\)"\{0,1\}/\1/' || echo "unknown")
EXP_SOURCE=$(grep -m1 'source:' "$CONFIG" 2>/dev/null | sed 's/.*source: *"\{0,1\}\([^"]*\)"\{0,1\}/\1/' || echo "unknown")

echo -e "  ${CYAN}${BOLD}Configuration${RESET}"
echo -e "  ${DIM}────────────────────────────────────────────────────${RESET}"
echo -e "  ${DIM}Config:       ${RESET}${CONFIG}"
echo -e "  ${DIM}Experiment:   ${RESET}${EXP_NAME}"
echo -e "  ${DIM}Source:       ${RESET}${EXP_SOURCE}"
echo -e "  ${DIM}Python:       ${RESET}$($VENV --version)"
echo ""

# ── Auto-generate synthetic source if needed ─────────────────────────
if [[ ! -f "$EXP_SOURCE" && "$EXP_SOURCE" == *"synthetic"* ]]; then
    echo -e "${YELLOW}⚠ Source not found: ${EXP_SOURCE}${RESET}"
    echo -e "${DIM}  Generating synthetic conversations...${RESET}"
    "$VENV" scripts/generate_conversations.py
    if [[ ! -f "$EXP_SOURCE" ]]; then
        echo -e "${RED}✗ Generation failed — file still missing${RESET}"
        exit 1
    fi
    echo -e "${GREEN}✓ Generated: ${EXP_SOURCE}${RESET}"
    echo ""
fi

# ── Run ──────────────────────────────────────────────────────────────
echo -e "${YELLOW}▸ Running experiment...${RESET}"
echo -e "${DIM}  $VENV scripts/run_experiment.py $CONFIG${RESET}"
echo ""

START_TIME=$(date +%s)

"$VENV" scripts/run_experiment.py "$CONFIG" 2>&1 | while IFS= read -r line; do
    if [[ "$line" == *"── Pipeline"* ]]; then
        echo -e "  ${CYAN}${BOLD}${line}${RESET}"
    elif [[ "$line" == *"✓"* ]]; then
        echo -e "  ${GREEN}${line}${RESET}"
    elif [[ "$line" == *"✗"* || "$line" == *"ERROR"* || "$line" == *"Error"* ]]; then
        echo -e "  ${RED}${line}${RESET}"
    elif [[ "$line" == *"═"* || "$line" == *"Result"* ]]; then
        echo -e "  ${BOLD}${line}${RESET}"
    elif [[ "$line" == *"Boundaries"* || "$line" == *"violations"* || "$line" == *"boundaries"* ]]; then
        echo -e "  ${MAGENTA}${line}${RESET}"
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
    echo -e "  ${GREEN}${BOLD}✓ Experiment complete${RESET} ${DIM}(${ELAPSED}s)${RESET}"
    echo ""
    echo -e "  ${CYAN}${BOLD}Output files:${RESET}"
    for f in results/metrics/*.yaml; do
        [[ -f "$f" ]] && echo -e "    ${DIM}📊${RESET} $f"
    done
    for f in results/reports/*.md; do
        [[ -f "$f" ]] && echo -e "    ${DIM}📝${RESET} $f"
    done
else
    echo -e "  ${RED}${BOLD}✗ Experiment failed${RESET} ${DIM}(${ELAPSED}s)${RESET}"
fi
echo ""

exit $EXIT_CODE
