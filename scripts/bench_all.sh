#!/usr/bin/env bash
# scripts/bench_all.sh — master benchmark runner for the HPC thesis.
#
# Runs every relevant combination of backend × measurement tool, then saves
# all results to a single timestamped directory under bench_results/.
#
# Usage:
#   ./scripts/bench_all.sh                # CPU + wgpu + hybrid (CPU+wgpu)
#   ./scripts/bench_all.sh --cpu-only     # CPU rayon only
#   ./scripts/bench_all.sh --cuda         # + CUDA and hybrid (CPU+CUDA)
#
# Options (can be combined):
#   --cpu-only       skip all GPU benchmarks
#   --cuda           include CUDA benchmarks (requires --features cuda build)
#   --width  N       render width  (default 1920)
#   --height N       render height (default 1080)
#   --iters  N       max iterations (default 1000)
#   --runs   N       timing repetitions (default 5)
#   --no-perf        skip perf stat (useful if perf needs elevated privileges)
#   --no-criterion   skip criterion statistical benchmarks
#
# Output layout:
#   bench_results/run_<timestamp>_<git>_<mode>/
#     config.txt          system + build info
#     perf/               perf stat counter files (one per backend)
#     timing/             bench_runner JSON + plain text tables
#     criterion/          full criterion HTML tree
#     summary.txt         side-by-side comparison of all backends

set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

# ── argument parsing ──────────────────────────────────────────────────────────

CPU_ONLY=0
CUDA=0
WIDTH=1920
HEIGHT=1080
ITERS=1000
RUNS=5
RUN_PERF=1
RUN_CRITERION=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu-only)     CPU_ONLY=1;              shift ;;
        --cuda)         CUDA=1;                  shift ;;
        --width)        WIDTH="$2";              shift 2 ;;
        --height)       HEIGHT="$2";             shift 2 ;;
        --iters)        ITERS="$2";              shift 2 ;;
        --runs)         RUNS="$2";               shift 2 ;;
        --no-perf)      RUN_PERF=0;              shift ;;
        --no-criterion) RUN_CRITERION=0;         shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

MODE="$( [[ $CPU_ONLY -eq 1 ]] && echo "cpu" || ( [[ $CUDA -eq 1 ]] && echo "cuda" || echo "full" ) )"

# ── output directory ──────────────────────────────────────────────────────────

GIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="bench_results/run_${TIMESTAMP}_${GIT_HASH}_${MODE}"
BINARY="./target/release/bench_runner"

mkdir -p "$RUN_DIR/perf" "$RUN_DIR/timing" "$RUN_DIR/criterion"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  bench_all.sh — fractal renderer HPC benchmark suite        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  mode    : $MODE"
echo "  output  : $RUN_DIR"
echo "  git     : $GIT_HASH"
echo "  size    : ${WIDTH}×${HEIGHT}  iters: $ITERS  runs: $RUNS"
echo ""

# ── build ─────────────────────────────────────────────────────────────────────

echo "── [1/5] Build ──────────────────────────────────────────────────"
cargo build --release --bin bench_runner
[[ $CUDA -eq 1 ]] && cargo build --release --bin bench_runner --features cuda
echo ""

# ── system info ───────────────────────────────────────────────────────────────

{
    echo "=== Benchmark run: $TIMESTAMP ==="
    echo "git          : $GIT_HASH"
    echo "mode         : $MODE"
    echo "resolution   : ${WIDTH}×${HEIGHT}"
    echo "max_iter     : $ITERS"
    echo "runs         : $RUNS"
    echo ""
    echo "=== System ==="
    echo "cpu          : $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
    echo "physical_cores: $(grep 'cpu cores' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
    echo "logical_cores : $(nproc)"
    echo "ram_gb        : $(awk '/MemTotal/{printf "%.1f\n", $2/1024/1024}' /proc/meminfo)"
    echo "kernel        : $(uname -r)"
    echo "perf          : $(perf --version 2>/dev/null || echo 'not found')"
    echo ""
    echo "=== Rust toolchain ==="
    rustc --version
    cargo --version
} | tee "$RUN_DIR/config.txt"
echo ""

# ── perf event probing ────────────────────────────────────────────────────────

EVENT_LIST=()
if [[ $RUN_PERF -eq 1 ]]; then
    try_event() {
        local err
        err=$(perf stat -e "$1" -- true 2>&1) || true
        echo "$err" | grep -qE "Bad event|syntax error|Unable to find|not supported" && return 0
        EVENT_LIST+=("$1")
    }
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
    # AMD Zen 3 retire-based FP counters
    try_event fp_ret_sse_avx_ops.all
    try_event fp_ret_sse_avx_ops.add_sub_flops
    try_event fp_ret_sse_avx_ops.mult_flops
    try_event fp_ret_sse_avx_ops.mac_flops
    try_event fp_ret_sse_avx_ops.div_flops
    # Intel fallback
    try_event fp_arith_inst_retired.scalar_double
    try_event fp_arith_inst_retired.128b_packed_double
    try_event fp_arith_inst_retired.256b_packed_double

    ALL_EVENTS="$(IFS=','; echo "${EVENT_LIST[*]}")"
    echo "Active perf events (${#EVENT_LIST[@]}): $ALL_EVENTS" >> "$RUN_DIR/config.txt"
fi

# ── helpers ───────────────────────────────────────────────────────────────────

run_perf_stat() {
    local label="$1"; shift    # e.g. "cpu", "wgpu"
    local out="$RUN_DIR/perf/${label}.txt"
    echo "  [perf] $label → $out"
    setarch "$(uname -m)" --addr-no-randomize \
        perf stat -e "$ALL_EVENTS" --big-num --output "$out" \
        -- "$@" 2>&1 | tee -a "$out"
}

run_timing() {
    local label="$1"; shift    # e.g. "cpu", "wgpu", "hybrid"
    local txt="$RUN_DIR/timing/${label}.txt"
    local json="$RUN_DIR/timing/${label}.json"
    echo "  [timing] $label"
    "$@" 2>/dev/null | tee "$txt"
    "$@" --json 2>/dev/null > "$json"
}

COMMON_ARGS=(--width "$WIDTH" --height "$HEIGHT" --iters "$ITERS" --runs "$RUNS")

# ── [2/5] perf stat ───────────────────────────────────────────────────────────

if [[ $RUN_PERF -eq 1 ]]; then
    echo "── [2/5] perf stat ─────────────────────────────────────────────"

    # CPU — all fractals
    echo "  CPU: all fractals"
    run_perf_stat "cpu_all" \
        "$BINARY" "${COMMON_ARGS[@]}" --fractal all --backend cpu

    # CPU — Mandelbrot only (cleaner signal for per-counter analysis)
    echo "  CPU: Mandelbrot only"
    run_perf_stat "cpu_mandelbrot" \
        "$BINARY" "${COMMON_ARGS[@]}" --fractal mandelbrot --backend cpu

    if [[ $CPU_ONLY -eq 0 ]]; then
        # wgpu — all fractals (shows host-side overhead + PCIe sync)
        echo "  wgpu: all fractals"
        run_perf_stat "wgpu_all" \
            "$BINARY" "${COMMON_ARGS[@]}" --fractal all --backend wgpu

        # Hybrid CPU+wgpu
        echo "  hybrid: Mandelbrot"
        run_perf_stat "hybrid_mandelbrot" \
            "$BINARY" "${COMMON_ARGS[@]}" --fractal mandelbrot --backend hybrid
    fi

    if [[ $CUDA -eq 1 ]]; then
        CUDA_BINARY="./target/release/bench_runner"   # same binary, cuda feature baked in
        echo "  CUDA: all fractals"
        run_perf_stat "cuda_all" \
            "$CUDA_BINARY" "${COMMON_ARGS[@]}" --fractal all --backend wgpu
    fi

    echo ""
fi

# ── [3/5] timing tables (bench_runner, no perf overhead) ──────────────────────

echo "── [3/5] Timing tables ─────────────────────────────────────────"

run_timing "cpu" \
    "$BINARY" "${COMMON_ARGS[@]}" --fractal all --backend cpu

if [[ $CPU_ONLY -eq 0 ]]; then
    run_timing "wgpu" \
        "$BINARY" "${COMMON_ARGS[@]}" --fractal all --backend wgpu

    run_timing "hybrid" \
        "$BINARY" "${COMMON_ARGS[@]}" --fractal all --backend hybrid
fi

# Thread-scaling sweep (CPU only — always useful)
echo "  [timing] cpu_scaling (thread sweep)"
"$BINARY" "${COMMON_ARGS[@]}" --fractal mandelbrot --backend cpu --scaling \
    --json 2>/dev/null > "$RUN_DIR/timing/cpu_scaling.json"
"$BINARY" "${COMMON_ARGS[@]}" --fractal mandelbrot --backend cpu --scaling \
    2>/dev/null > "$RUN_DIR/timing/cpu_scaling.txt"

echo ""

# ── [4/5] criterion statistical benchmarks ───────────────────────────────────

if [[ $RUN_CRITERION -eq 1 ]]; then
    echo "── [4/5] Criterion benchmarks ──────────────────────────────────"

    FEATURES_FLAG=()
    [[ $CUDA -eq 1 ]] && FEATURES_FLAG=(--features cuda)

    if [[ $CPU_ONLY -eq 1 ]]; then
        # Only run CPU groups — faster, no GPU warm-up
        cargo bench --bench fractal_bench "${FEATURES_FLAG[@]}" \
            -- "cpu/" 2>&1
    else
        # Run everything: CPU + wgpu + hybrid + (cuda)
        cargo bench --bench fractal_bench "${FEATURES_FLAG[@]}" 2>&1
    fi

    # Archive the full criterion tree so HTML links work
    if [[ -d target/criterion ]]; then
        cp -r target/criterion/. "$RUN_DIR/criterion/"
        echo ""
        echo "  Criterion report: $RUN_DIR/criterion/report/index.html"
        # Keep a convenience symlink to the latest run's report
        ln -sfn "$(pwd)/$RUN_DIR/criterion" bench_results/criterion_latest
    fi
    echo ""
fi

# ── [5/5] summary ─────────────────────────────────────────────────────────────

echo "── [5/5] Summary ───────────────────────────────────────────────"
{
    echo "=== Backend comparison — ${WIDTH}×${HEIGHT}  iter=$ITERS  git=$GIT_HASH ==="
    echo ""

    for label in cpu wgpu hybrid; do
        f="$RUN_DIR/timing/${label}.txt"
        [[ -f "$f" ]] && { echo "--- $label ---"; cat "$f"; echo ""; }
    done

    echo "--- cpu thread scaling ---"
    [[ -f "$RUN_DIR/timing/cpu_scaling.txt" ]] && cat "$RUN_DIR/timing/cpu_scaling.txt"
    echo ""

    if command -v python3 &>/dev/null && [[ -f "$RUN_DIR/timing/cpu.json" ]]; then
        python3 - "$RUN_DIR/timing" <<'PYEOF'
import json, os, sys

timing_dir = sys.argv[1]
backends = ["cpu", "wgpu", "hybrid"]
all_data = {}

for b in backends:
    p = os.path.join(timing_dir, f"{b}.json")
    if os.path.exists(p):
        all_data[b] = {s["fractal"]: s for s in json.load(open(p))}

if not all_data:
    sys.exit(0)

fractals = list(next(iter(all_data.values())).keys())
print("=== Throughput comparison (Mpix/s) ===")
hdr = f"{'Fractal':<12}"
for b in backends:
    if b in all_data:
        hdr += f"  {b:>10}"
print(hdr)
print("-" * len(hdr))
for frac in fractals:
    row = f"{frac:<12}"
    cpu_mpix = all_data.get("cpu", {}).get(frac, {}).get("mpix_per_sec", None)
    for b in backends:
        if b not in all_data:
            continue
        v = all_data[b].get(frac, {}).get("mpix_per_sec", 0)
        row += f"  {v:>10.2f}"
    print(row)

print()
print("=== Speedup vs CPU ===")
hdr = f"{'Fractal':<12}"
for b in backends:
    if b in all_data and b != "cpu":
        hdr += f"  {b:>10}"
print(hdr)
print("-" * len(hdr))
for frac in fractals:
    cpu_v = all_data.get("cpu", {}).get(frac, {}).get("mpix_per_sec", 1)
    row = f"{frac:<12}"
    for b in backends:
        if b not in all_data or b == "cpu":
            continue
        v = all_data[b].get(frac, {}).get("mpix_per_sec", 0)
        row += f"  {v/cpu_v:>9.2f}x"
    print(row)
PYEOF
    fi

} | tee "$RUN_DIR/summary.txt"

# ── done ─────────────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  All benchmarks complete                                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "  Results : $RUN_DIR/"
echo "  Report  : $RUN_DIR/criterion/report/index.html"
echo "  Summary : $RUN_DIR/summary.txt"
echo ""
echo "  Open report:"
echo "    xdg-open $RUN_DIR/criterion/report/index.html"
echo ""
du -sh "$RUN_DIR"
