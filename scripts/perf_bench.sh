#!/usr/bin/env bash
# scripts/perf_bench.sh — hardware-counter profiling for the fractal renderer.
#
# Captures cycles, IPC, cache behaviour, branch prediction, and FP throughput
# using Linux perf stat.  Results are timestamped and stored in bench_results/.
#
# Requires: perf (linux-tools), cargo (release build)
#
# Usage:
#   ./scripts/perf_bench.sh [fractal] [width] [height] [iters] [runs]
#
#   ./scripts/perf_bench.sh                          # default 1920×1080, 1000 iters
#   ./scripts/perf_bench.sh mandelbrot 3840 2160 2000 3
#   BENCH_BACKEND=wgpu ./scripts/perf_bench.sh       # GPU path
#   BENCH_SCALING=1 ./scripts/perf_bench.sh          # thread-scaling sweep

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────

FRACTAL="${1:-all}"
WIDTH="${2:-1920}"
HEIGHT="${3:-1080}"
ITERS="${4:-1000}"
RUNS="${5:-5}"
BACKEND="${BENCH_BACKEND:-cpu}"
SCALING="${BENCH_SCALING:-0}"

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
BINARY="$REPO_ROOT/target/release/bench_runner"
RESULTS="$REPO_ROOT/bench_results"

GIT_HASH="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LABEL="${TIMESTAMP}_${GIT_HASH}_${FRACTAL}_${WIDTH}x${HEIGHT}"

mkdir -p "$RESULTS"

# ── build ─────────────────────────────────────────────────────────────────────

echo "==> Building bench_runner (release)..."
cargo build --release --bin bench_runner --manifest-path "$REPO_ROOT/Cargo.toml"

# ── perf event selection ──────────────────────────────────────────────────────
#
# Probes each candidate event individually; only includes those that the
# current kernel+PMU driver actually supports.

EVENT_LIST=()

try_event() {
    # Suppress stderr; check for known error strings — always returns 0 so
    # set -e does not abort the script when an event is unsupported.
    local err
    err=$(perf stat -e "$1" -- true 2>&1) || true
    if echo "$err" | grep -qE "Bad event|syntax error|Unable to find|not supported"; then
        return 0   # event unavailable — skip silently
    fi
    EVENT_LIST+=("$1")
}

# Core counters (generic — work on any modern x86)
try_event cycles
try_event instructions
try_event cache-references
try_event cache-misses
try_event branches
try_event branch-misses
try_event L1-dcache-loads
try_event L1-dcache-load-misses
try_event LLC-loads
try_event LLC-load-misses

# FP counters — AMD Zen 3 retire-based FP operation counts
# (fp_ret_sse_avx_ops.* counts actual FLOP retirements, not instructions)
try_event fp_ret_sse_avx_ops.all
try_event fp_ret_sse_avx_ops.add_sub_flops
try_event fp_ret_sse_avx_ops.mult_flops
try_event fp_ret_sse_avx_ops.mac_flops     # each MAC = 2 FLOPs
try_event fp_ret_sse_avx_ops.div_flops

# Intel fallback (Skylake+)
try_event fp_arith_inst_retired.scalar_double
try_event fp_arith_inst_retired.128b_packed_double
try_event fp_arith_inst_retired.256b_packed_double

# Build comma-separated event string
ALL_EVENTS="$(IFS=','; echo "${EVENT_LIST[*]}")"

echo "==> Active perf events (${#EVENT_LIST[@]}):"
printf '     %s\n' "${EVENT_LIST[@]}"

# ── helper: run one perf stat measurement ─────────────────────────────────────

run_perf() {
    local label="$1"; shift
    local out_file="$RESULTS/${LABEL}_${label}.txt"

    echo ""
    echo "==> perf stat: $label  →  $out_file"

    # Disable ASLR for more reproducible results
    setarch "$(uname -m)" --addr-no-randomize \
    perf stat \
        --big-num \
        -e "$ALL_EVENTS" \
        --output "$out_file" \
        -- "$@" 2>&1 | tee -a "$out_file"

    echo ""
}

# ── helper: perf record (call-graph) for flamegraph ──────────────────────────

run_record() {
    local label="$1"; shift
    local perf_data="$RESULTS/${LABEL}_${label}.perf.data"
    local flame_svg="$RESULTS/${LABEL}_${label}.flamegraph.svg"

    echo "==> perf record: $label  →  $perf_data"
    perf record \
        --call-graph dwarf \
        --freq 999 \
        --output "$perf_data" \
        -- "$@" || true          # don't abort if perf record needs privileges

    # Generate flamegraph if flamegraph.pl is in PATH
    if command -v flamegraph.pl &>/dev/null && command -v stackcollapse-perf.pl &>/dev/null; then
        perf script -i "$perf_data" | stackcollapse-perf.pl | flamegraph.pl > "$flame_svg"
        echo "    flamegraph: $flame_svg"
    elif command -v cargo-flamegraph &>/dev/null; then
        echo "    (install flamegraph.pl from https://github.com/brendangregg/FlameGraph for SVG)"
    fi
}

# ── benchmark matrix ─────────────────────────────────────────────────────────

BASE_ARGS=(
    --fractal "$FRACTAL"
    --width   "$WIDTH"
    --height  "$HEIGHT"
    --iters   "$ITERS"
    --runs    "$RUNS"
    --backend "$BACKEND"
)

if [ "$SCALING" = "1" ]; then
    # Thread-scaling sweep — output JSON for later plotting
    echo "==> Thread-scaling sweep..."
    JSON_OUT="$RESULTS/${LABEL}_scaling.json"
    "$BINARY" "${BASE_ARGS[@]}" --scaling --json > "$JSON_OUT"
    echo "    Scaling data: $JSON_OUT"
    python3 - "$JSON_OUT" <<'EOF'
import json, sys
data = json.load(open(sys.argv[1]))
print(f"\n{'Fractal':<12} {'Threads':>8} {'Mpix/s':>10} {'GFLOPs':>10} {'Speedup':>9}")
print("-" * 55)
base = {}
for s in data:
    k = s['fractal']
    if s['threads'] == 1:
        base[k] = s['mpix_per_sec']
    sp = s['mpix_per_sec'] / base.get(k, s['mpix_per_sec'])
    print(f"{s['fractal']:<12} {s['threads']:>8} {s['mpix_per_sec']:>10.2f} {s['gflops']:>10.3f} {sp:>8.2f}x")
print()
EOF
else
    # Standard perf stat run
    run_perf "stat" "$BINARY" "${BASE_ARGS[@]}"

    # Also a perf record for flamegraph (single fractal, fewer runs)
    RECORD_ARGS=(
        --fractal "$([ "$FRACTAL" = "all" ] && echo "mandelbrot" || echo "$FRACTAL")"
        --width   "$WIDTH"
        --height  "$HEIGHT"
        --iters   "$ITERS"
        --runs    3
        --backend "$BACKEND"
    )
    run_record "flamegraph" "$BINARY" "${RECORD_ARGS[@]}"
fi

# ── plain timing table (no perf overhead) ────────────────────────────────────
echo "==> Plain timing (no perf overhead):"
"$BINARY" "${BASE_ARGS[@]}"

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
echo "Results saved to: $RESULTS/"
ls -lh "$RESULTS/${LABEL}"* 2>/dev/null | awk '{print "  " $NF " (" $5 ")"}'
