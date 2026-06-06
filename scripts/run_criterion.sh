#!/usr/bin/env bash
# scripts/run_criterion.sh — run criterion benchmarks, archive the HTML report,
# and optionally save/compare baselines for regression tracking.
#
# Usage:
#   ./scripts/run_criterion.sh                        # run all groups
#   ./scripts/run_criterion.sh thread_scaling         # run one group
#   ./scripts/run_criterion.sh --save baseline_name   # run + save named baseline
#   ./scripts/run_criterion.sh --compare baseline_name # run + compare vs saved baseline

set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
GIT_HASH="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS="$REPO_ROOT/bench_results"
CRITERION_DIR="$REPO_ROOT/target/criterion"
BASELINES_DIR="$RESULTS/baselines"

mkdir -p "$RESULTS" "$BASELINES_DIR"

# ── argument parsing ──────────────────────────────────────────────────────────

FILTER=""
MODE="run"          # run | save | compare
BASELINE_NAME=""
CUDA=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --save)    MODE="save";    BASELINE_NAME="$2"; shift 2 ;;
        --compare) MODE="compare"; BASELINE_NAME="$2"; shift 2 ;;
        --cuda)    CUDA=1; shift ;;
        *)         FILTER="$1";   shift ;;
    esac
done

# ── build the cargo bench command ─────────────────────────────────────────────

BENCH_ARGS=()
[ -n "$FILTER" ] && BENCH_ARGS+=("$FILTER")

case "$MODE" in
    save)
        BENCH_ARGS+=("--save-baseline" "$BASELINE_NAME")
        echo "==> Saving baseline: $BASELINE_NAME"
        ;;
    compare)
        BENCH_ARGS+=("--baseline" "$BASELINE_NAME")
        echo "==> Comparing against baseline: $BASELINE_NAME"
        ;;
esac

# ── run ───────────────────────────────────────────────────────────────────────

FEATURES_FLAG=()
[ "$CUDA" = "1" ] && FEATURES_FLAG=(--features cuda)

echo "==> Running criterion benchmarks (release)  git=$GIT_HASH"
[ -n "$FILTER" ]    && echo "    filter : $FILTER"
[ "$CUDA" = "1" ]   && echo "    cuda   : enabled"

cd "$REPO_ROOT"

if [[ ${#BENCH_ARGS[@]} -gt 0 ]]; then
    cargo bench --bench fractal_bench "${FEATURES_FLAG[@]}" -- "${BENCH_ARGS[@]}"
else
    cargo bench --bench fractal_bench "${FEATURES_FLAG[@]}"
fi

# ── archive the HTML report ───────────────────────────────────────────────────

REPORT_SRC="$CRITERION_DIR/report/index.html"
ARCHIVE_DIR="$RESULTS/criterion_${TIMESTAMP}_${GIT_HASH}"

# Copy the ENTIRE criterion directory so relative links between the overview
# page and per-benchmark sub-pages all resolve correctly.
if [ -d "$CRITERION_DIR" ]; then
    cp -r "$CRITERION_DIR" "$ARCHIVE_DIR"
    echo ""
    echo "==> HTML report archived to:"
    echo "    $ARCHIVE_DIR/report/index.html"

    # Symlink 'latest' points at the versioned archive directory
    ln -sfn "$ARCHIVE_DIR" "$RESULTS/criterion_latest"
    echo "    $RESULTS/criterion_latest/report/index.html  (symlink → latest)"
else
    echo "==> No criterion output found at $CRITERION_DIR"
fi

# ── archive the named baseline if we saved one ────────────────────────────────

if [ "$MODE" = "save" ] && [ -n "$BASELINE_NAME" ]; then
    BASELINE_SRC="$CRITERION_DIR/$BASELINE_NAME"   # criterion stores baselines here
    # Actually criterion stores baselines inside each benchmark's directory;
    # the data lives at target/criterion/<group>/<bench>/base/ for --save-baseline.
    # We snapshot the whole criterion dir so comparisons remain self-contained.
    BASELINE_DEST="$BASELINES_DIR/${BASELINE_NAME}_${TIMESTAMP}_${GIT_HASH}"
    cp -r "$CRITERION_DIR" "$BASELINE_DEST"
    echo "==> Baseline '$BASELINE_NAME' archived to:"
    echo "    $BASELINE_DEST"
fi

# ── open in browser ───────────────────────────────────────────────────────────

OPEN_TARGET="$ARCHIVE_DIR/report/index.html"
if [ -f "$OPEN_TARGET" ]; then
    echo ""
    echo "==> Opening report..."
    xdg-open "$OPEN_TARGET" 2>/dev/null || open "$OPEN_TARGET" 2>/dev/null || \
        echo "    Open manually: $OPEN_TARGET"
fi
