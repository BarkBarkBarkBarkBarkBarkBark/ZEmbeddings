#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ZEmbeddings — Test Runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
#  Usage:
#    ./scripts/run_tests.sh              # all tests, with coverage
#    ./scripts/run_tests.sh -k kalman    # only tests matching 'kalman'
#    ./scripts/run_tests.sh -m fast      # only @pytest.mark.fast tests
#    ./scripts/run_tests.sh --no-cov     # skip coverage report
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

# ── Config ───────────────────────────────────────────────────────────
VENV=".venv/bin/python"
COV_ENABLED=true
PYTEST_ARGS=()

# Parse args
for arg in "$@"; do
    if [[ "$arg" == "--no-cov" ]]; then
        COV_ENABLED=false
    else
        PYTEST_ARGS+=("$arg")
    fi
done

# ── Banner ───────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║                                                      ║"
echo "  ║   ⚡ Z E M B E D D I N G S   T E S T   S U I T E   ║"
echo "  ║                                                      ║"
echo "  ║   Semantic trajectory analysis · Kalman boundary     ║"
echo "  ║   detection · Causal sliding windows                 ║"
echo "  ║                                                      ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Check venv ───────────────────────────────────────────────────────
if [[ ! -f "$VENV" ]]; then
    echo -e "${RED}✗ Virtual environment not found at ${VENV}${RESET}"
    echo -e "${DIM}  Run: python -m venv .venv && .venv/bin/pip install -e '.[dev]'${RESET}"
    exit 1
fi

# ── Summarise what we're running ─────────────────────────────────────
echo -e "${DIM}  Project:   ${PROJECT_ROOT}${RESET}"
echo -e "${DIM}  Python:    $($VENV --version)${RESET}"
echo -e "${DIM}  Coverage:  $(if $COV_ENABLED; then echo "enabled"; else echo "disabled"; fi)${RESET}"
if [[ ${#PYTEST_ARGS[@]} -gt 0 ]]; then
    echo -e "${DIM}  Filters:   ${PYTEST_ARGS[*]}${RESET}"
fi
echo ""

# ── Build pytest command ─────────────────────────────────────────────
CMD=("$VENV" "-m" "pytest" "tests/" "-v" "--tb=short" "-q")

if $COV_ENABLED; then
    CMD+=("--cov=src/zembeddings" "--cov-report=term-missing:skip-covered" "--cov-report=html:results/coverage")
fi

if [[ ${#PYTEST_ARGS[@]} -gt 0 ]]; then
    CMD+=("${PYTEST_ARGS[@]}")
fi

# ── Run ──────────────────────────────────────────────────────────────
echo -e "${YELLOW}▸ Running: ${DIM}${CMD[*]}${RESET}"
echo ""

START_TIME=$(date +%s)

"${CMD[@]}" 2>&1 | while IFS= read -r line; do
    # Colourise PASSED / FAILED / ERROR lines
    if [[ "$line" == *"PASSED"* ]]; then
        echo -e "  ${GREEN}${line}${RESET}"
    elif [[ "$line" == *"FAILED"* ]]; then
        echo -e "  ${RED}${line}${RESET}"
    elif [[ "$line" == *"ERROR"* ]]; then
        echo -e "  ${RED}${line}${RESET}"
    elif [[ "$line" == *"passed"* || "$line" == *"warnings summary"* ]]; then
        echo -e "  ${BOLD}${line}${RESET}"
    else
        echo "  $line"
    fi
done

EXIT_CODE=${PIPESTATUS[0]}
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "  ${DIM}────────────────────────────────────────────────────${RESET}"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "  ${GREEN}${BOLD}✓ All tests passed${RESET} ${DIM}(${ELAPSED}s)${RESET}"
else
    echo -e "  ${RED}${BOLD}✗ Some tests failed${RESET} ${DIM}(${ELAPSED}s)${RESET}"
fi

if $COV_ENABLED && [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "  ${DIM}Coverage report: results/coverage/index.html${RESET}"
fi
echo ""

exit $EXIT_CODE
