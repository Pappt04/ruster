#!/usr/bin/env bash
# scripts/energy_bench.sh — GFLOPS/watt for CPU vs GPU, following the same
# conventions as scripts/perf_bench.sh (timestamped output under bench_results/).
#
# CPU energy: Intel RAPL package counter (/sys/class/powercap/intel-rapl:0/energy_uj),
# read before/after a --runs N batch. Needs root to read on most kernels — this
# script uses `sudo cat`, which will prompt interactively the first time.
#
# GPU energy: background-polls `nvidia-smi --query-gpu=power.draw` for the
# duration of a --cuda-loop N run (needs a --features cuda build), no root
# required.
#
# Usage:
#   ./scripts/energy_bench.sh [fractal] [width] [height] [iters] [runs]
#
#   ./scripts/energy_bench.sh                       # default 1920x1080, 1000 iters, 200 runs
#   ./scripts/energy_bench.sh mandelbrot 3840 2160 2000 100
#   BENCH_SKIP_GPU=1 ./scripts/energy_bench.sh       # CPU-only (no cuda build needed)

set -euo pipefail

FRACTAL="${1:-mandelbrot}"
WIDTH="${2:-1920}"
HEIGHT="${3:-1080}"
ITERS="${4:-1000}"
RUNS="${5:-200}"
SKIP_GPU="${BENCH_SKIP_GPU:-0}"

REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
BINARY="$REPO_ROOT/target/release/bench_runner"
RESULTS="$REPO_ROOT/results"
mkdir -p "$RESULTS"

RAPL="/sys/class/powercap/intel-rapl:0/energy_uj"
RAPL_MAX="/sys/class/powercap/intel-rapl:0/max_energy_range_uj"

echo "==> Building bench_runner (release, cpu)..."
cargo build --release --bin bench_runner --manifest-path "$REPO_ROOT/Cargo.toml"

# ── CPU: RAPL package energy around a --runs N batch ──────────────────────────

echo
echo "==> CPU energy (RAPL package-0), fractal=$FRACTAL ${WIDTH}x${HEIGHT} iters=$ITERS runs=$RUNS"

if [ ! -r "$RAPL" ]; then
    echo "    (energy_uj not readable as current user — trying sudo, will prompt for password)"
fi

read_energy_uj() {
    if [ -r "$RAPL" ]; then cat "$RAPL"; else sudo cat "$RAPL"; fi
}

MAX_RANGE="$(cat "$RAPL_MAX")"
E0="$(read_energy_uj)"
T0="$(date +%s.%N)"

"$BINARY" --fractal "$FRACTAL" --width "$WIDTH" --height "$HEIGHT" --iters "$ITERS" --runs "$RUNS" \
    --json > "$RESULTS/energy_cpu_run.json"

T1="$(date +%s.%N)"
E1="$(read_energy_uj)"

# RAPL counter wraps around at max_energy_range_uj — correct for one wrap
# (good enough for runs under a few minutes; multi-wrap not handled).
DELTA_UJ=$((E1 - E0))
if [ "$DELTA_UJ" -lt 0 ]; then DELTA_UJ=$((DELTA_UJ + MAX_RANGE)); fi

WALL_S="$(echo "$T1 - $T0" | bc)"
ENERGY_J="$(echo "scale=4; $DELTA_UJ / 1000000" | bc)"
AVG_WATTS="$(echo "scale=2; $ENERGY_J / $WALL_S" | bc)"
GFLOPS="$(python3 -c "import json; d=json.load(open('$RESULTS/energy_cpu_run.json')); print(d[0]['gflops'])" 2>/dev/null || echo "0")"
GFLOPS_PER_WATT="$(echo "scale=3; $GFLOPS / $AVG_WATTS" | bc 2>/dev/null || echo "n/a")"

echo "    wall time    : ${WALL_S}s"
echo "    energy       : ${ENERGY_J} J"
echo "    avg power    : ${AVG_WATTS} W"
echo "    GFLOPS       : ${GFLOPS}"
echo "    GFLOPS/watt  : ${GFLOPS_PER_WATT}"

{
    echo "backend,fractal,width,height,iters,wall_s,energy_j,avg_watts,gflops,gflops_per_watt"
    echo "cpu,$FRACTAL,$WIDTH,$HEIGHT,$ITERS,$WALL_S,$ENERGY_J,$AVG_WATTS,$GFLOPS,$GFLOPS_PER_WATT"
} > "$RESULTS/energy_cpu.csv"

# ── GPU: nvidia-smi power polling around a --cuda-loop N run ──────────────────

if [ "$SKIP_GPU" = "1" ]; then
    echo
    echo "==> Skipping GPU energy (BENCH_SKIP_GPU=1)"
    exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "==> nvidia-smi not found — skipping GPU energy"
    exit 0
fi

echo
echo "==> Building bench_runner (release, cuda)..."
cargo build --release --features cuda --bin bench_runner --manifest-path "$REPO_ROOT/Cargo.toml"

echo "==> GPU energy (nvidia-smi power.draw), fractal=$FRACTAL ${WIDTH}x${HEIGHT} iters=$ITERS loop=$RUNS"

POWER_LOG="$RESULTS/energy_gpu_power.csv"
nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -lms 100 > "$POWER_LOG" &
SMI_PID=$!

T0="$(date +%s.%N)"
"$BINARY" --cuda-loop "$RUNS" --fractal "$FRACTAL" --width "$WIDTH" --height "$HEIGHT" --iters "$ITERS" \
    --json > "$RESULTS/energy_gpu_run.json"
T1="$(date +%s.%N)"

kill "$SMI_PID" 2>/dev/null || true
wait "$SMI_PID" 2>/dev/null || true

WALL_S="$(echo "$T1 - $T0" | bc)"
AVG_WATTS="$(awk '{sum+=$1; n++} END {if (n>0) print sum/n; else print 0}' "$POWER_LOG")"
TOTAL_S="$(python3 -c "import json; d=json.load(open('$RESULTS/energy_gpu_run.json')); print(d['total_s'])" 2>/dev/null || echo "$WALL_S")"
ENERGY_J="$(echo "scale=4; $AVG_WATTS * $TOTAL_S" | bc)"
# Real iteration count, not a pixels*max_iter worst-case bound: CPU and GPU
# compute bit-identical escape counts on this codebase (see the "math must be
# perfect" fixes), so the CPU JSON's total_iters from the run above is exactly
# how much work the GPU loop also did per launch — reusing it here gives true
# GFLOPS instead of an inflated estimate (most pixels escape long before
# max_iter, so pixels*max_iter would overstate total FLOPs substantially).
ITERS_PER_FRAME="$(python3 -c "import json; d=json.load(open('$RESULTS/energy_cpu_run.json')); print(d[0]['total_iters'])" 2>/dev/null || echo 0)"
MANDELBROT_FLOPS_PER_ITER=8
TOTAL_FLOPS=$(echo "$ITERS_PER_FRAME * $MANDELBROT_FLOPS_PER_ITER * $RUNS" | bc)
GFLOPS="$(echo "scale=4; $TOTAL_FLOPS / $TOTAL_S / 1000000000" | bc)"
GFLOPS_PER_WATT="$(echo "scale=3; $GFLOPS / $AVG_WATTS" | bc 2>/dev/null || echo "n/a")"

echo "    wall time    : ${WALL_S}s ($RUNS launches)"
echo "    avg power    : ${AVG_WATTS} W"
echo "    GFLOPS (real iter count, from the CPU run above)  : ${GFLOPS}"
echo "    GFLOPS/watt  : ${GFLOPS_PER_WATT}"

{
    echo "backend,fractal,width,height,iters,wall_s,avg_watts,gflops,gflops_per_watt"
    echo "gpu,$FRACTAL,$WIDTH,$HEIGHT,$ITERS,$WALL_S,$AVG_WATTS,$GFLOPS,$GFLOPS_PER_WATT"
} >> "$RESULTS/energy_cpu.csv"

echo
echo "==> Results written to $RESULTS/energy_cpu.csv (both rows) and $RESULTS/energy_gpu_power.csv"
