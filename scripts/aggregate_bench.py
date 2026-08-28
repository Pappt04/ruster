# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
#!/usr/bin/env python3
"""
aggregate_bench.py — cross-project fractal-renderer benchmark aggregator.

Normalizes results from every fractal renderer in ../.. (ruster, Fractals-rs,
FractalRendererCpp, cpp, zigger, nanobrot, fraqtive) into the shared schema
described by bench_schema.json, then emits, all scoped to a single fractal
(Mandelbrot by default — see --fractal):

  - bench_results/combined.json          full normalized record list
  - bench_results/comparison_report.md    two tables (see below)
  - bench_results/comparison.csv          flat row-per-record, for spreadsheets/thesis appendix
  - bench_results/comparison_charts.pdf   charts, generated with matplotlib (see PEP 723
                                           header above — run via `uv run scripts/aggregate_bench.py`
                                           so the dependency is sandboxed, no system pip install needed)

Criterion is the primary, most-trusted source (Rust's `criterion` gives a
bootstrapped mean/median/stddev + 95% CI over ~100 samples — the statistical
gold standard here); Google Benchmark run with --benchmark_format=json and
->Repetitions(N)->ReportAggregatesOnly(true) is the C++ analogue and is
ingested the same way, just flagged as lacking a bootstrap CI. Hand-rolled
`bench_runner --json` output (perf/energy counters, GFLOPs, thread scaling)
is ingested as 'manual' — still normalized, still reported, just without
Criterion's statistical apparatus.

Nothing is ever dropped for being "not comparable". A record that has no
peer in any other project (ruster's wgpu/CUDA/hybrid/perturbation backends,
today) is kept with comparability.class = "project-unique" and shown in the
full-capability table. A project with no benchmark data yet (nanobrot,
fraqtive, cpp, zigger) gets an explicit status: "no-data" row instead of
silently not appearing.

Usage:
    uv run scripts/aggregate_bench.py [--out-dir bench_results] [--fractal mandelbrot] [--all-fractals]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

SCHEMA_VERSION = "1"
RUSTER_ROOT = Path(__file__).resolve().parent.parent
FRACTAL_ROOT = RUSTER_ROOT.parent          # .../diplomski/fractal
OTHER = FRACTAL_ROOT / "other-projects"

MAX_ITER_DEFAULT = 1000  # matches MAX_ITER const in fractal_bench.rs / application_bench.rs


# ── normalized record ──────────────────────────────────────────────────────────

def new_record(**kwargs) -> dict:
    rec = {
        "schema_version": SCHEMA_VERSION,
        "project": None,
        "impl": None,
        "language": "unknown",
        "git_hash": None,
        "run_id": None,
        "generated_at": None,
        "source": {"kind": "manual", "tool_version": None, "path": None},
        "backend": {"family": "unknown", "detail": None, "threads": None},
        "workload": {
            "fractal": None, "measurement": "unclassified", "algorithm": None,
            "resolution": None, "max_iter": None, "precision": None, "param": None,
        },
        "comparability": {"class": "unclassified", "baseline_key": None, "note": None},
        "stats": None,
        "derived": None,
        "status": "ok",
    }
    for k, v in kwargs.items():
        if isinstance(v, dict) and isinstance(rec.get(k), dict):
            rec[k].update(v)
        else:
            rec[k] = v
    return rec


def baseline_key(measurement: str, fractal: str, max_iter: int, precision: str, algorithm: str,
                  resolution: Optional[list[int]] = None, threads: Optional[int] = None) -> str:
    """Join key for Table A. Two records only merge if fractal, measurement kind,
    iteration count, precision, and algorithm class all match — plus resolution
    and thread count where those axes apply. Deliberately strict: a shared key
    is meant to license "renderer X is Nx faster than renderer Y", so anything
    that could make that statement unfair belongs in the key, not left out."""
    key = f"{fractal.lower()}@{measurement}@iter{max_iter}@{precision}@{algorithm}"
    if resolution:
        key += f"@{resolution[0]}x{resolution[1]}"
    if threads is not None:
        key += f"@threads{threads}"
    return key


POINTS_PER_CALL_RE = re.compile(r"points_per_call=(\d+)")


def apply_pixel_kernel_throughput(rec: dict) -> None:
    """pixel_kernel groups have no Criterion/GoogleBenchmark 'throughput' field
    (they're not full-frame renders), but they still represent N point
    evaluations per call — a rate that's dimensionally identical to Mpix/s
    (millions of evaluations/sec) and lets e.g. ruster's 5-points-per-call
    micro-bench compare fairly against FractalRendererCpp's 1-point-per-call
    one, despite the raw mean_ms not being comparable at all."""
    if rec["workload"]["measurement"] != "pixel_kernel" or rec["derived"]:
        return
    m = POINTS_PER_CALL_RE.search(rec["workload"].get("param") or "")
    stats = rec["stats"] or {}
    if not m or not stats.get("mean_ms"):
        return
    points = int(m.group(1))
    mpix_s = points / (stats["mean_ms"] / 1000.0) / 1_000_000.0
    rec["derived"] = {"mpix_s": round(mpix_s, 4), "gflops": None,
                       "gflops_note": "Normalized M-evaluations/s (points_per_call / time) — not a full-frame pixel count."}


RES_RE = re.compile(r"(\d+)\s*[×xX]\s*(\d+)")  # matches "1920×1080" / "1920x1080", NOT "zoom_1e12"


def parse_resolution(s: Optional[str]) -> Optional[list[int]]:
    if not s:
        return None
    m = RES_RE.search(s)
    if not m:
        return None
    return [int(m.group(1)), int(m.group(2))]


def fractal_norm(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = name.lower().strip()
    if n.startswith("julia"):
        return "julia"
    if n.startswith("mandelbrot") or n.startswith("mandel"):
        return "mandelbrot"
    if n.startswith("newton"):
        return "newton"
    if n.startswith("nova"):
        return "nova"
    if n.startswith("burning"):
        return "burning_ship"
    if n.startswith("tricorn"):
        return "tricorn"
    return n


# ── Criterion adapter (ruster) ─────────────────────────────────────────────────
#
# ruster's own convention (see benches/fractal_bench.rs), verified against the
# actual bench_results/criterion_latest tree:
#   group_id  = "<backend>/<measurement>[/<Fractal>]"
#   function_id = compute variant (rayon/scalar/f64x4/f32x8/f32x8_ilp/hilbert/
#                 rows/cuda/wgpu/threads/...) or, for pixel_kernel/pipeline
#                 groups, the fractal name itself
#   value_str = resolution label ("1920×1080") when the group has throughput,
#               otherwise the raw parameter (e.g. a thread count)
#
# Only cpu/render/<Fractal>::rayon is treated as the common CPU-naive baseline
# (matches Stage 0 in results/summary.md). Confirmed scalar, not auto-vectorized:
# fractal_bench.rs's own doc comment on bench_simd_render says it "Compares
# scalar render() vs f64x4 render_simd() vs f32x8 render_simd_f32()" — `render()`
# is the same function used here in the cpu/render group.

# ── Which physical device each GPU backend actually runs on ───────────────────
#
# Both GPU backends now target the discrete RTX 3050, so wgpu-vs-CUDA rows are
# a genuine same-silicon comparison.
#
# This was NOT true before 2026-08-08. The NVIDIA stack had been installed
# COMPUTE-ONLY (`libnvidia-compute-580`, no `nvidia-driver-*`/`libnvidia-gl-*`),
# so there was no `nvidia_icd.json` and no `libGLX_nvidia` on the system;
# Vulkan could not see the RTX 3050 at all and `PowerPreference::HighPerformance`
# silently fell back to the integrated AMD Vega. Every wgpu number in runs up to
# and including `criterion_20260725_215202_8d2dbdc` came from the integrated GPU
# and must not be compared against a CUDA number. Installing `nvidia-driver-580`
# fixed it; `cargo run --release --example adapters` verifies which device wgpu
# selects, and the bench harness now prints it (`[wgpu] ...`) at startup.
GPU_DEVICE = {
    "gpu-wgpu": "NVIDIA RTX 3050 Laptop — DISCRETE (via Vulkan)",
    "gpu-cuda": "NVIDIA RTX 3050 Laptop — DISCRETE (via CUDA)",
    "hybrid":   "CPU + NVIDIA RTX 3050 Laptop (discrete)",
}


def ruster_precision(fam: str, function_id: Optional[str], fractal: Optional[str],
                     param: Optional[str]) -> str:
    """Actual arithmetic precision of a ruster measurement.

    Previously hardcoded to "f64" for every ruster record, which mislabelled
    every GPU and f32-SIMD row and would have let an f32 GPU number join a
    Table A group against f64 CPU numbers. The real rules, read off the source:

    * wgpu (`fractal.wgsl`): WGSL has no f64 type at all — always f32, every
      fractal, every zoom, no exceptions.
    * CUDA (`fractal.cu` + `CudaFractal::render`): dispatches `fractal_kernel_f32`
      for Mandelbrot/Julia below `F32_PRECISION_THRESHOLD` (1e6) and the f64
      `fractal_kernel` otherwise. Newton/Nova have no f32 kernel at all.
    * SIMD (`fractal.rs`): `f32x8`/`f32x8_ilp` are f32; `f64x4`/`scalar` are f64.
    * The perturbation groups' reference orbits are f64 or f128 (double-double
      `Dd`), carried in function_id.
    """
    fid = (function_id or "").lower()
    if fam == "wgpu":
        return "f32"
    if fam == "cuda":
        if fractal_norm(fractal) not in ("mandelbrot", "julia"):
            return "f64"
        # Perturbation-sweep arms carry an explicit zoom label; everything else
        # is benched at zoom=1 (see fractal_bench.rs's `vp()`).
        if param and param.startswith("zoom="):
            try:
                exp = int(param.split("1e")[1])
            except (IndexError, ValueError):
                return "f64"
            return "f32" if exp < 6 else "f64"
        return "f32"
    if fam == "simd":
        return "f32" if fid.startswith("f32") else "f64"
    if fam == "hybrid":
        # CPU tiles: f32 SIMD (SchedulerConfig::simd_cpu_tiles, default true);
        # GPU tiles: f32 (wgpu always, CUDA via gpu_tiles_f32, default true).
        return "f32"
    if fam == "perturbation":
        if "f128" in fid:
            return "f128"
        return "f64"
    return "f64"


def classify_ruster(group_id: str, function_id: str, value_str: Optional[str]) -> dict:
    parts = group_id.split("/")
    fam = parts[0] if parts else "unknown"
    measurement = parts[1] if len(parts) > 1 else "unclassified"
    fractal_from_group = parts[2] if len(parts) > 2 else None

    # The plain-CPU perturbation group is named "perturbation/Mandelbrot_1080p"
    # (fractal+resolution folded into position 1), unlike the wgpu/cuda variants
    # which are "wgpu/perturbation/Mandelbrot_1080p" (measurement at position 1,
    # like every other family). Without this, these records silently fall
    # through with measurement="Mandelbrot_1080p" and fractal=None instead of
    # joining the wgpu/cuda perturbation numbers under measurement="perturbation".
    if fam == "perturbation" and measurement not in ("reference_orbit", "series_approx"):
        fractal_from_group = measurement.split("_")[0]
        measurement = "perturbation"

    backend_family_map = {
        "cpu": "cpu-scalar", "simd": "cpu-simd", "wgpu": "gpu-wgpu",
        "cuda": "gpu-cuda", "hybrid": "hybrid", "perturbation": "cpu-scalar",
    }
    backend_family = backend_family_map.get(fam, "unknown")
    device = GPU_DEVICE.get(backend_family)

    # measurement-specific fractal / algorithm extraction. `variant_label` feeds
    # `impl` — for groups where function_id is actually the fractal name (not a
    # compute variant), it's set to the measurement instead so `impl` identifies
    # the backend variant consistently across fractals rather than conflating
    # the two (e.g. "ruster-hybrid-cpu+wgpu", not "ruster-hybrid-Mandelbrot").
    fractal = fractal_from_group
    algorithm = "naive"
    param = None
    threads = None
    variant_label = function_id

    resolution_override = None

    if measurement in ("pixel_kernel", "pipeline"):
        fractal = function_id           # group has no /<Fractal>, function_id carries it
        variant_label = measurement
        if measurement == "pixel_kernel":
            param = "points_per_call=5"  # SAMPLE_POINTS has 5 entries — see bench_pixel_kernels
    elif fam == "hybrid" and measurement == "heterogeneous_deep":
        # bench_heterogeneous_deep (fractal_bench.rs) is Mandelbrot-only and its
        # function_id is the compute arm ("cpu" | "cuda" | "hybrid"), not the
        # fractal — the opposite convention from the other hybrid groups below.
        # Without this branch, fractal_norm(function_id) returns "cpu"/"cuda"/
        # "hybrid" verbatim (none of those match a known fractal prefix), which
        # silently drops every one of this group's records from the
        # Mandelbrot-filtered dataset (filter_by_fractal only keeps fractal in
        # {None, target}). Keeping the three arms as separate impls, rather
        # than folding them into one "heterogeneous_deep" bucket, is what lets
        # a chart plot cpu/cuda/hybrid as separate series.
        fractal = "mandelbrot"
        variant_label = f"{measurement}_{function_id}"
        if value_str:
            param = f"zoom={value_str}"
    elif fam == "hybrid":
        # All hybrid groups (cpu+wgpu static split, heterogeneous [CUDA
        # adaptive scheduler], heterogeneous_wgpu [wgpu adaptive scheduler])
        # call bench_function(fractal.name()) / BenchmarkId::new(fractal.name(),
        # ...) — function_id is always the fractal, never a compute variant.
        fractal = function_id
        variant_label = measurement      # "cpu+wgpu" | "heterogeneous" | "heterogeneous_wgpu"
        if measurement in ("heterogeneous", "heterogeneous_wgpu") and value_str:
            param = f"zoom={value_str}"  # value_str is a zoom label here, not a resolution
    elif measurement == "thread_scaling":
        fractal = "mandelbrot"          # only Mandelbrot_1080p is benchmarked
        resolution_override = [1920, 1080]
        try:
            threads = int(function_id) if function_id.isdigit() else int(value_str)
        except (TypeError, ValueError):
            pass
        param = f"threads={threads}" if threads else None
        variant_label = measurement
    elif measurement.startswith("perturbation") or measurement in ("reference_orbit", "series_approx"):
        fid_lower = (function_id or "").lower()
        algorithm = ("naive" if fid_lower == "scalar" else            # perturbation group's own scalar-baseline arm
                     "perturbation_sa" if "sa" in fid_lower or measurement == "series_approx" else
                     "perturbation")
        fractal = fractal or "mandelbrot"
        if value_str and not RES_RE.search(value_str):
            param = f"zoom={value_str}"  # perturbation groups key by zoom label too, not resolution
    elif measurement == "render_tiled":
        algorithm = "tiled_hilbert" if function_id == "hilbert" else "naive"

    resolution = resolution_override or parse_resolution(value_str)

    # Baseline-eligible ruster measurements: all three are confirmed scalar f64,
    # naive, max_iter=1000 (see module comment above + bench_pixel_kernels /
    # bench_thread_scaling source). Each is a distinct measurement *kind*, so
    # baseline_key keeps them in separate groups — a render number never merges
    # with a pixel_kernel number just because both are "baseline-common".
    is_baseline = (
        (measurement == "render" and function_id == "rayon") or
        (measurement == "pixel_kernel") or
        (measurement == "thread_scaling")
    ) and fam == "cpu"

    if is_baseline:
        bkey = baseline_key(measurement, fractal_norm(fractal) or "unknown", MAX_ITER_DEFAULT,
                             "f64", "naive", resolution=resolution, threads=threads)
        comparability = {"class": "baseline-common", "baseline_key": bkey, "note": None}
    else:
        comparability = {
            "class": "project-unique",
            "baseline_key": None,
            "note": "No equivalent measurement in another project yet." if backend_family != "cpu-scalar"
                     else "Non-baseline CPU measurement (not the naive rayon/render/pixel_kernel/thread_scaling path).",
        }

    impl = f"ruster-{fam}-{variant_label}" if variant_label else f"ruster-{fam}"

    return {
        "impl": impl,
        "backend": {"family": backend_family, "detail": function_id, "threads": threads,
                    "device": device},
        "workload": {
            "fractal": fractal_norm(fractal), "measurement": measurement, "algorithm": algorithm,
            "resolution": resolution, "max_iter": MAX_ITER_DEFAULT,
            "precision": ruster_precision(fam, function_id, fractal, param), "param": param,
        },
        "comparability": comparability,
    }


# ── Criterion adapter (Fractals-rs, github.com/Maxime-Cllt/Fractals-rs) ────────
#
# Verified against benches/application_bench.rs:
#   "full_frame_800x600"        fixed 800x600, max_iter=1000, no throughput.
#                                function_id = "<fractal>_fast" (f32 SIMD) or
#                                "<fractal>_high" (f64 SIMD) — there is NO
#                                scalar/non-SIMD path in this project at all.
#   "full_frame/<Fractal>"      multi-res sweep, max_iter=1000, throughput set.
#                                function_id = "fast" | "high", value_str = resolution.
#   "thread_scaling/Mandelbrot_1080p_fast"  max_iter=1000, f32 SIMD only.
#   "scalar_kernels" / "simd_kernels"       single-pixel kernel calls.
#
# max_iter is baked in per-group from source, not read from data — Criterion's
# own output doesn't carry it. If Fractals-rs's bench consts change, update here.
# (Both full-frame groups used to run at max_iter=300 vs ruster's 1000 — fixed
# to 1000 in benches/application_bench.rs so iteration count now matches.)
#
# Nothing here reaches class == "baseline-common" against ruster today, but not
# because of iteration count anymore: every Fractals-rs full-frame path is SIMD
# (f32 "fast" or f64 "high"), while ruster's naive-cpu baseline is scalar f64 —
# different algorithm class, so a merge would still be a false equivalence on
# that axis alone. They're kept as project-unique with an explicit note instead.
# Re-running either project's bench to match algorithm/precision too would make
# an exact pair line up automatically — the join key is purely mechanical.

FRACTALS_RS_FULLFRAME_800_MAXITER = 1000
FRACTALS_RS_MULTIRES_MAXITER = 1000
FRACTALS_RS_THREAD_SCALING_MAXITER = 1000


def classify_fractals_rs(group_id: str, function_id: str, value_str: Optional[str]) -> dict:
    precision = "f32" if function_id.endswith("fast") or function_id == "fast" else \
                "f64" if function_id.endswith("high") or function_id == "high" else None
    backend_family = "cpu-simd"  # this project has no non-SIMD render path
    algorithm = "simd"
    threads = None
    param = None
    resolution = parse_resolution(value_str)
    note = None

    if group_id == "full_frame_800x600":
        fractal = re.sub(r"_(fast|high)$", "", function_id)
        max_iter = FRACTALS_RS_FULLFRAME_800_MAXITER
        resolution = [800, 600]
        measurement = "pipeline"
        note = (f"Fixed 800x600, max_iter={max_iter}, {precision} SIMD ('{function_id}') — "
                "no scalar path exists in this project to compare against ruster's naive-cpu baseline.")
    elif group_id.startswith("full_frame/"):
        fractal = group_id.split("/", 1)[1]
        max_iter = FRACTALS_RS_MULTIRES_MAXITER
        measurement = "render"
        note = (f"max_iter={max_iter}, matches ruster's baseline — but {precision} SIMD vs ruster's "
                "scalar naive-cpu baseline is still a different algorithm class, not comparable as-is; "
                "would need a matching-algorithm rerun on one side.")
    elif group_id.startswith("thread_scaling"):
        fractal = "mandelbrot"
        max_iter = FRACTALS_RS_THREAD_SCALING_MAXITER
        resolution = [1920, 1080]
        measurement = "thread_scaling"
        precision = "f32"
        try:
            threads = int(function_id) if function_id.isdigit() else int(value_str)
        except (TypeError, ValueError):
            pass
        param = f"threads={threads}" if threads else None
        note = "f32 SIMD thread-scaling sweep; ruster's equivalent is scalar f64 — precision differs."
    elif group_id in ("scalar_kernels", "simd_kernels"):
        fractal = re.sub(r"_(f32|f64|simd_f32|simd_f64)$", "", function_id)
        max_iter = None
        measurement = "pixel_kernel"
        algorithm = "naive" if group_id == "scalar_kernels" else "simd"
        precision = "f64" if function_id.endswith("f64") else "f32"
        if group_id == "scalar_kernels":
            # Empirically this group measures ~0.6ns/call for a max_iter=1000 escape-time
            # computation — physically impossible (even 1 FLOP takes longer). TEST_X/TEST_Y/
            # MAX_ITERATIONS are compile-time literals passed by value with no black_box on
            # the input side (only the output is boxed), so LLVM appears to constant-fold the
            # whole call away; simd_kernels takes array *references* instead, which resists
            # this. No points_per_call is set here, so apply_pixel_kernel_throughput leaves
            # derived empty rather than reporting a bogus multi-GHz-equivalent throughput.
            note = "Measurement appears constant-folded by the compiler (inputs aren't black_box'd) — mean_ms is not real per-call cost, excluded from throughput charts."
        else:
            param = "points_per_call=1"  # single-point call, like FractalRendererCpp's BM_Evaluate
            note = "Single-pixel kernel call, not a full-frame render — compare only against ruster's cpu/pixel_kernel group, and even then mind the precision axis."
    else:
        fractal = None
        max_iter = None
        measurement = "unclassified"
        note = f"Unrecognized Fractals-rs group_id '{group_id}' — add a case in classify_fractals_rs()."

    impl = f"fractals-rs-{function_id}" if measurement == "pixel_kernel" else f"fractals-rs-{group_id.split('/')[0]}-{function_id}"

    return {
        "impl": impl,
        "backend": {"family": backend_family, "detail": function_id, "threads": threads},
        "workload": {
            "fractal": fractal_norm(fractal), "measurement": measurement, "algorithm": algorithm,
            "resolution": resolution, "max_iter": max_iter, "precision": precision, "param": param,
        },
        "comparability": {"class": "project-unique", "baseline_key": None, "note": note},
    }


CLASSIFIERS = {
    "ruster": classify_ruster,
    "fractals-rs": classify_fractals_rs,
}


def ns_to_ms(x: Optional[float]) -> Optional[float]:
    return None if x is None else x / 1_000_000.0


def iter_criterion_tree(root: Path) -> Iterator[tuple[dict, dict]]:
    """Yield (benchmark.json, estimates.json) pairs from a Criterion output tree,
    using only the 'new' phase (latest measurement) to avoid double-counting
    against 'base'/'change' (criterion's own before/after comparison phases)."""
    if not root.exists():
        return
    for est_path in root.rglob("new/estimates.json"):
        bench_path = est_path.parent / "benchmark.json"
        if not bench_path.exists():
            continue
        try:
            bench = json.loads(bench_path.read_text())
            est = json.loads(est_path.read_text())
        except json.JSONDecodeError:
            continue
        yield bench, est, est_path


def parse_criterion_project(project: str, language: str, root: Path, run_id: Optional[str],
                             git_hash: Optional[str]) -> list[dict]:
    classify = CLASSIFIERS.get(project)
    records = []
    for bench, est, est_path in iter_criterion_tree(root):
        group_id = bench.get("group_id", "")
        function_id = bench.get("function_id", "")
        value_str = bench.get("value_str")

        mean = est.get("mean", {})
        median = est.get("median", {})
        std_dev = est.get("std_dev", {})
        mean_ci = mean.get("confidence_interval", {})

        stats = {
            "mean_ms": ns_to_ms(mean.get("point_estimate")),
            "mean_ci_low_ms": ns_to_ms(mean_ci.get("lower_bound")),
            "mean_ci_high_ms": ns_to_ms(mean_ci.get("upper_bound")),
            "median_ms": ns_to_ms(median.get("point_estimate")),
            "std_dev_ms": ns_to_ms(std_dev.get("point_estimate")),
            "confidence_level": mean_ci.get("confidence_level"),
            "n_samples": None,
        }

        rec = new_record(
            project=project, language=language, git_hash=git_hash, run_id=run_id,
            source={"kind": "criterion", "tool_version": "criterion-rs", "path": str(est_path.relative_to(root.parent) if root.parent in est_path.parents else est_path)},
            stats=stats,
        )

        if classify:
            rec.update(classify(group_id, function_id, value_str))
        else:
            # Unknown project naming convention: keep the raw labels, don't drop the record.
            rec["impl"] = f"{project}-{function_id or group_id}"
            rec["workload"]["fractal"] = fractal_norm(function_id) or fractal_norm(group_id)
            rec["workload"]["measurement"] = group_id or "unclassified"
            rec["workload"]["resolution"] = parse_resolution(value_str)
            rec["workload"]["max_iter"] = MAX_ITER_DEFAULT
            rec["comparability"] = {
                "class": "unclassified", "baseline_key": None,
                "note": f"No classifier registered for project '{project}' yet — add one to CLASSIFIERS in aggregate_bench.py.",
            }

        throughput = bench.get("throughput")
        pixels = None
        if throughput and isinstance(throughput, dict) and "Elements" in throughput:
            pixels = throughput["Elements"]
        elif rec["workload"].get("resolution"):
            # Criterion group set no explicit throughput (e.g. Fractals-rs's
            # full_frame_800x600) — fall back to the resolution the classifier
            # already determined from source, since pixel count is fixed either way.
            w, h = rec["workload"]["resolution"]
            pixels = w * h
        if pixels and stats["mean_ms"]:
            mpix_s = pixels / (stats["mean_ms"] / 1000.0) / 1_000_000.0
            rec["derived"] = {"mpix_s": round(mpix_s, 3), "gflops": None,
                               "gflops_note": "FLOPs/iter known analytically but iteration count isn't exposed by Criterion throughput — use bench_runner's GFLOPs for that."}
        apply_pixel_kernel_throughput(rec)
        records.append(rec)
    return records


# ── Manual adapter (ruster bench_runner.rs / energy scripts) ───────────────────
# Flat JSON arrays, already close to normalized. Not comparable across
# projects today (no other project emits this), but reported in full since it
# carries GFLOPs / thread-count / perturbation data Criterion's own output
# doesn't expose.

def parse_manual_json(project: str, language: str, path: Path, git_hash: Optional[str]) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    is_scaling_sweep = "scaling" in path.stem
    records = []
    for s in data:
        if "fractal" not in s:
            continue
        backend_raw = s.get("backend", "cpu")
        backend_family = ("hybrid" if backend_raw.startswith("hybrid") else
                          {"cpu": "cpu-scalar", "wgpu": "gpu-wgpu", "cuda": "gpu-cuda"}.get(backend_raw, "unknown"))
        resolution = [s["width"], s["height"]] if "width" in s and "height" in s else None
        threads = s.get("threads")
        measurement = "thread_scaling" if is_scaling_sweep else "render"
        impl = f"{project}-{backend_raw}-bench_runner" + (f"-t{threads}" if is_scaling_sweep and threads else "")
        rec = new_record(
            project=project, language=language, git_hash=git_hash,
            impl=impl,
            source={"kind": "manual", "tool_version": "bench_runner", "path": str(path)},
            backend={"family": backend_family, "detail": backend_raw, "threads": threads,
                     "device": GPU_DEVICE.get(backend_family)},
            workload={
                "fractal": fractal_norm(s.get("fractal")), "measurement": measurement, "algorithm": "naive",
                "resolution": resolution, "max_iter": s.get("max_iter"),
                # Same rules as the criterion path (see `ruster_precision`) — the
                # backend name in bench_runner's JSON maps onto the same kernels.
                "precision": ruster_precision(
                    {"gpu-wgpu": "wgpu", "gpu-cuda": "cuda", "hybrid": "hybrid"}.get(backend_family, "cpu"),
                    backend_raw, s.get("fractal"), None,
                ),
                "param": f"threads={threads}" if is_scaling_sweep and threads else None,
            },
            comparability={
                "class": "project-unique", "baseline_key": None,
                "note": "bench_runner is ruster-only tooling (perf/GFLOPs counters); no other project emits this format yet.",
            },
            stats={
                "mean_ms": s.get("median_ms"), "mean_ci_low_ms": s.get("min_ms"), "mean_ci_high_ms": s.get("max_ms"),
                "median_ms": s.get("median_ms"), "std_dev_ms": None, "confidence_level": None, "n_samples": s.get("runs"),
            },
            derived={"mpix_s": s.get("mpix_per_sec"), "gflops": s.get("gflops"), "gflops_note": None},
        )
        records.append(rec)
    return records


def dedup_manual_records(records: list[dict]) -> list[dict]:
    """bench_results/ accumulates one timing/ directory per historical
    `bench_all.sh` run, and results/ has its own one-off diagnostic captures —
    several of these cover the exact same (backend, fractal, resolution,
    threads) config. Without this, older/superseded runs show up alongside
    the current one as if they were distinct benchmark variations, which is
    exactly the kind of noise that made earlier charts hard to read. Keep
    only the most-recently-modified source file's record per config."""
    best: dict[tuple, tuple[float, dict]] = {}
    passthrough = []
    for r in records:
        if r["source"]["kind"] != "manual":
            passthrough.append(r)
            continue
        path = r["source"].get("path")
        try:
            mtime = Path(path).stat().st_mtime if path else 0.0
        except OSError:
            mtime = 0.0
        w = r["workload"]
        key = (r["project"], r["backend"]["detail"], w.get("fractal"),
               tuple(w.get("resolution") or []), r["backend"].get("threads"))
        if key not in best or mtime > best[key][0]:
            best[key] = (mtime, r)
    return passthrough + [v[1] for v in best.values()]


# ── Google Benchmark adapter (FractalRendererCpp criterion_bench) ─────────────
# Not wired to real data yet (no run has been made) — implemented so the
# first `./criterion_bench --benchmark_format=json > result.json` run slots
# in immediately. Google Benchmark's ->Repetitions(N)->ReportAggregatesOnly
# gives mean/median/stddev/cv but NOT a bootstrap confidence interval — that
# gap is recorded in comparability.note rather than silently presented as
# equivalent to Criterion's numbers.

# ── Google Benchmark adapter classifiers ───────────────────────────────────────
#
# FractalRendererCpp (Benchmarks/criterion_bench.cpp), verified against source:
# every fractal is `double` (f64) scalar, naive escape-time — no SIMD/GPU axis
# in this project at all (see RenderCore.h's own module doc). Four benchmark
# families, each with its own params baked in at the C++ call site (Google
# Benchmark's JSON carries none of max_iter/resolution/threads as data, only
# as name path segments for the ones passed via ->Args/->Arg):
#   BM_Evaluate/<fractal>                    max_iter=1000, single point, no threads
#   BM_FullFrame/<fractal>/<w>/<h>            max_iter=1000, 8 threads (hardcoded)
#   BM_ThreadScaling_Mandelbrot_1080p/<n>      max_iter=1000, 1920x1080, n threads
#   BM_FullPipeline_Mandelbrot_1080p           max_iter=1000, 1920x1080, 8 threads
#
# All four land in Table A now: same fractal, same max_iter, same f64-scalar-
# naive algorithm as ruster's cpu/pixel_kernel, cpu/thread_scaling, cpu/render,
# and cpu/pipeline respectively. BM_FullFrame/BM_FullPipeline used to run at
# max_iter=300 (chosen to mirror Fractals-rs's full-frame benches, which were
# ALSO max_iter=300 at the time) — both were bumped to 1000 in
# Benchmarks/criterion_bench.cpp / benches/application_bench.rs so every
# project's full-frame numbers are now a genuine apples-to-apples iteration
# count (algorithm/precision can still differ — see each project's own note).

def classify_fractalrenderercpp(base_name: str) -> dict:
    # Real name shape (verified against an actual criterion_bench --benchmark_format=json
    # run): "<kind>/<bare-positional-args>/<key:value-args from ->ArgNames>", e.g.
    #   BM_Evaluate/mandelbrot/repeats:20
    #   BM_FullFrame/mandelbrot/width:1920/height:1080/min_time:0.100/repeats:10
    #   BM_ThreadScaling_Mandelbrot_1080p/threads:8/min_time:0.100/repeats:10
    #   BM_FullPipeline_Mandelbrot_1080p/min_time:0.100/repeats:10
    # ->Repetitions()/->MinTime() add their own "repeats:"/"min_time:" segments —
    # ignored below, not a real workload axis.
    segments = base_name.split("/")
    kind = segments[0]
    kv: dict[str, str] = {}
    positional: Optional[str] = None
    for seg in segments[1:]:
        if ":" in seg:
            k, v = seg.split(":", 1)
            kv[k] = v
        elif positional is None:
            positional = seg

    fractal = None
    resolution = None
    threads = None
    param = None
    max_iter = None
    measurement = "unclassified"
    comparability = {"class": "unclassified", "baseline_key": None,
                      "note": f"Unrecognized benchmark name '{base_name}' — add a case in classify_fractalrenderercpp()."}

    if kind == "BM_Evaluate":
        fractal = positional
        measurement = "pixel_kernel"
        max_iter = 1000
        param = "points_per_call=1"
        bkey = baseline_key(measurement, fractal_norm(fractal) or "unknown", max_iter, "f64", "naive")
        comparability = {"class": "baseline-common", "baseline_key": bkey,
                          "note": "1 point/call here vs ruster's 5 points/call — compared via normalized M-evals/s (see apply_pixel_kernel_throughput), not raw mean_ms."}
    elif kind == "BM_FullFrame":
        fractal = positional
        resolution = [int(kv["width"]), int(kv["height"])]
        measurement = "render"
        max_iter = 1000
        threads = 8
        param = "threads=8"
        bkey = baseline_key(measurement, fractal_norm(fractal) or "unknown", max_iter, "f64", "naive", resolution=resolution)
        comparability = {"class": "baseline-common", "baseline_key": bkey,
                          "note": "Same fractal/resolution/max_iter/precision/algorithm as ruster's cpu/render — "
                                   "but 8 raw std::threads spawned per call here vs a persistent rayon pool in ruster, "
                                   "a real threading-strategy difference, not a flaw."}
    elif kind == "BM_ThreadScaling_Mandelbrot_1080p":
        fractal = "mandelbrot"
        resolution = [1920, 1080]
        measurement = "thread_scaling"
        max_iter = 1000
        threads = int(kv["threads"]) if "threads" in kv else None
        param = f"threads={threads}" if threads else None
        bkey = baseline_key(measurement, fractal, max_iter, "f64", "naive", resolution=resolution, threads=threads)
        comparability = {"class": "baseline-common", "baseline_key": bkey,
                          "note": "Same fractal/resolution/max_iter/precision/algorithm as ruster's cpu/thread_scaling — "
                                   "but a different threading strategy (raw std::thread spawn+join per call here vs a "
                                   "persistent rayon pool in ruster), which is a real, worth-reporting difference, not a flaw."}
    elif kind == "BM_FullPipeline_Mandelbrot_1080p":
        fractal = "mandelbrot"
        resolution = [1920, 1080]
        measurement = "pipeline"
        max_iter = 1000
        threads = 8
        param = "threads=8"
        bkey = baseline_key(measurement, fractal, max_iter, "f64", "naive", resolution=resolution)
        comparability = {"class": "baseline-common", "baseline_key": bkey,
                          "note": "Same fractal/resolution/max_iter/precision/algorithm as ruster's cpu/pipeline — "
                                   "but 8 raw std::threads spawned per call here vs a persistent rayon pool in ruster, "
                                   "a real threading-strategy difference, not a flaw."}

    impl_suffix = kind.replace("BM_", "").replace("_Mandelbrot_1080p", "").lower()
    impl = f"fractalrenderercpp-{impl_suffix}" + (f"-{fractal}" if kind in ("BM_Evaluate", "BM_FullFrame") else "")

    return {
        "impl": impl,
        "backend": {"family": "cpu-scalar", "detail": kind, "threads": threads},
        "workload": {
            "fractal": fractal_norm(fractal), "measurement": measurement, "algorithm": "naive",
            "resolution": resolution, "max_iter": max_iter, "precision": "f64", "param": param,
        },
        "comparability": comparability,
    }


# ── Google Benchmark adapter classifier: XaoS ──────────────────────────────
#
# XaoS (Benchmarks/criterion_bench.cpp), verified against source and an
# actual run: bench_runner/criterion_bench force `number_t` back to plain
# `double` (-UUSE_LONG_DOUBLE, see Benchmarks/CMakeLists.txt) even though
# the real GUI build defaults to 80-bit extended precision (the top-level
# CMakeLists.txt always adds -DUSE_LONG_DOUBLE unless -DDEEPZOOM=ON) --
# bench_runner prints sizeof(number_t) at startup to confirm this rather
# than just asserting it. Same benchmark-name shapes as
# classify_fractalrenderercpp() above (BM_Evaluate/<fractal>,
# BM_FullFrame/<fractal>/<w>/<h>, BM_ThreadScaling_Mandelbrot_1080p/<n>),
# minus a BM_FullPipeline group: XaoS's `calculate()` always fuses
# iteration and color/palette lookup (see Benchmarks/BenchCore.h's module
# doc for why -- there's no separate raw-iteration API to call instead),
# so there's no unfused-vs-fused split to benchmark here the way
# FractalRendererCpp's Evaluate/colorizeFractal split allows.
#
# One real difference from FractalRendererCpp's BM_FullFrame: XaoS's uses
# std::thread::hardware_concurrency() at bench time, not a hardcoded
# thread count -- so `threads` is left out of its baseline_key (like
# ruster's own "render"/rayon baseline, which also doesn't fix a thread
# count), rather than reporting a number that would vary by machine.
#
# XaoS's real defining feature -- boundary-tracing "guessing"
# (src/engine/btrace.cpp) -- is not exercised by any of these numbers; see
# BENCHMARKING.md for why that distinction matters more here than for the
# other three (already brute-force-every-pixel) projects.

def classify_xaos(base_name: str) -> dict:
    segments = base_name.split("/")
    kind = segments[0]
    kv: dict[str, str] = {}
    positional: Optional[str] = None
    for seg in segments[1:]:
        if ":" in seg:
            k, v = seg.split(":", 1)
            kv[k] = v
        elif positional is None:
            positional = seg

    fractal = None
    resolution = None
    threads = None
    param = None
    max_iter = None
    measurement = "unclassified"
    comparability = {"class": "unclassified", "baseline_key": None,
                      "note": f"Unrecognized benchmark name '{base_name}' — add a case in classify_xaos()."}

    if kind == "BM_Evaluate":
        fractal = positional
        measurement = "pixel_kernel"
        max_iter = 1000
        param = "points_per_call=1"
        bkey = baseline_key(measurement, fractal_norm(fractal) or "unknown", max_iter, "f64", "naive")
        comparability = {"class": "baseline-common", "baseline_key": bkey,
                          "note": "1 point/call, matching FractalRendererCpp's BM_Evaluate — compared via normalized "
                                   "M-evals/s (see apply_pixel_kernel_throughput), not raw mean_ms. Also includes a "
                                   "cpalette.pixels[] array-index lookup XaoS's calculate() always does (see "
                                   "Benchmarks/BenchCore.h) — not purely the iteration math, since XaoS has no way to "
                                   "call just that."}
    elif kind == "BM_FullFrame":
        fractal = positional
        resolution = [int(kv["width"]), int(kv["height"])]
        measurement = "render"
        max_iter = 1000
        param = "threads=hardware_concurrency()"
        bkey = baseline_key(measurement, fractal_norm(fractal) or "unknown", max_iter, "f64", "naive", resolution=resolution)
        comparability = {"class": "baseline-common", "baseline_key": bkey,
                          "note": "Same fractal/resolution/max_iter/precision/algorithm as ruster's cpu/render — "
                                   "row-striped std::threads spawned per call (count = hardware_concurrency() at "
                                   "bench time, not fixed), a real threading-strategy difference from ruster's "
                                   "persistent rayon pool, not a flaw."}
    elif kind == "BM_ThreadScaling_Mandelbrot_1080p":
        fractal = "mandelbrot"
        resolution = [1920, 1080]
        measurement = "thread_scaling"
        max_iter = 1000
        threads = int(kv["threads"]) if "threads" in kv else None
        param = f"threads={threads}" if threads else None
        bkey = baseline_key(measurement, fractal, max_iter, "f64", "naive", resolution=resolution, threads=threads)
        comparability = {"class": "baseline-common", "baseline_key": bkey,
                          "note": "Same fractal/resolution/max_iter/precision/algorithm as ruster's cpu/thread_scaling — "
                                   "but row-striped std::thread spawn+join per call here vs a persistent rayon pool "
                                   "in ruster, a real threading-strategy difference, not a flaw."}

    impl_suffix = kind.replace("BM_", "").replace("_Mandelbrot_1080p", "").lower()
    impl = f"xaos-{impl_suffix}" + (f"-{fractal}" if kind in ("BM_Evaluate", "BM_FullFrame") else "")

    return {
        "impl": impl,
        "backend": {"family": "cpu-scalar", "detail": kind, "threads": threads},
        "workload": {
            "fractal": fractal_norm(fractal), "measurement": measurement, "algorithm": "naive",
            "resolution": resolution, "max_iter": max_iter, "precision": "f64", "param": param,
        },
        "comparability": comparability,
    }


# ── Google Benchmark adapter classifier: FractalNow ─────────────────────────
#
# FractalNow (Benchmarks/criterion_bench.cpp), verified against source and an
# actual run: unlike XaoS, FractalNow's compute API genuinely separates the
# iteration loop from color — `RunFractalEngine()` returns a `CacheEntry`
# whose `.value` field IS the raw per-pixel iteration count (IC_DISCRETE/
# CM_ITERATIONCOUNT, -1 sentinel for "never escaped"), with no gradient/
# transfer-function math involved at all (see Benchmarks/BenchCore.h's module
# doc). So BM_Evaluate here is a genuinely unfused kernel call, unlike XaoS's
# BM_Evaluate (which always pays a cpalette.pixels[] lookup it has no way to
# skip). BM_FullFrame/BM_ThreadScaling both call RunFractalEngine per pixel
# too (Benchmarks/BenchCore.h's RunRawKernel), row-striped across
# hardware_concurrency() real std::threads (own layer, not FractalNow's own
# Threads/Task machinery, which has no per-pixel API — only DrawFractal/
# AntiAliaseFractal do) — so, like XaoS's BM_FullFrame, `threads` is left out
# of the baseline_key.
#
# BM_FullPipeline_Mandelbrot_1080p calls DrawFractal() with
# quadInterpolationSize=1 — brute force, not FractalNow's real default
# (quadInterpolationSize=5, its "solid guessing" — see BENCHMARKING.md for
# why quadInterpolationSize=1 is what's actually comparable here). This is a
# real second pipeline stage (transfer function + gradient lookup), not a
# reconstruction, so it lands in Table A as a genuine pipeline row, the way
# FractalRendererCpp's BM_FullPipeline does — unlike XaoS, which has no
# unfused-vs-fused split to benchmark at all.

def classify_fractalnow(base_name: str) -> dict:
    segments = base_name.split("/")
    kind = segments[0]
    kv: dict[str, str] = {}
    positional: Optional[str] = None
    for seg in segments[1:]:
        if ":" in seg:
            k, v = seg.split(":", 1)
            kv[k] = v
        elif positional is None:
            positional = seg

    fractal = None
    resolution = None
    threads = None
    param = None
    max_iter = None
    measurement = "unclassified"
    comparability = {"class": "unclassified", "baseline_key": None,
                      "note": f"Unrecognized benchmark name '{base_name}' — add a case in classify_fractalnow()."}

    if kind == "BM_Evaluate":
        fractal = positional
        measurement = "pixel_kernel"
        max_iter = 1000
        param = "points_per_call=1"
        bkey = baseline_key(measurement, fractal_norm(fractal) or "unknown", max_iter, "f64", "naive")
        comparability = {"class": "baseline-common", "baseline_key": bkey,
                          "note": "1 point/call, matching FractalRendererCpp's/XaoS's BM_Evaluate — compared via "
                                   "normalized M-evals/s (see apply_pixel_kernel_throughput), not raw mean_ms. "
                                   "Genuinely unfused (RunFractalEngine's raw CacheEntry.value, no color/gradient "
                                   "math) — no palette-lookup caveat needed here, unlike XaoS's BM_Evaluate."}
    elif kind == "BM_FullFrame":
        fractal = positional
        resolution = [int(kv["width"]), int(kv["height"])]
        measurement = "render"
        max_iter = 1000
        param = "threads=hardware_concurrency()"
        bkey = baseline_key(measurement, fractal_norm(fractal) or "unknown", max_iter, "f64", "naive", resolution=resolution)
        comparability = {"class": "baseline-common", "baseline_key": bkey,
                          "note": "Same fractal/resolution/max_iter/precision/algorithm as ruster's cpu/render — "
                                   "row-striped std::threads spawned per call (count = hardware_concurrency() at "
                                   "bench time, not fixed), a real threading-strategy difference from ruster's "
                                   "persistent rayon pool, not a flaw. Raw RunFractalEngine kernel only, no color."}
    elif kind == "BM_ThreadScaling_Mandelbrot_1080p":
        fractal = "mandelbrot"
        resolution = [1920, 1080]
        measurement = "thread_scaling"
        max_iter = 1000
        threads = int(kv["threads"]) if "threads" in kv else None
        param = f"threads={threads}" if threads else None
        bkey = baseline_key(measurement, fractal, max_iter, "f64", "naive", resolution=resolution, threads=threads)
        comparability = {"class": "baseline-common", "baseline_key": bkey,
                          "note": "Same fractal/resolution/max_iter/precision/algorithm as ruster's cpu/thread_scaling — "
                                   "but row-striped std::thread spawn+join per call here vs a persistent rayon pool "
                                   "in ruster, a real threading-strategy difference, not a flaw. Scaling plateaus "
                                   "sharply between 2 and 4 threads on the reference machine (see BENCHMARKING.md) — "
                                   "reported as observed, not smoothed over; root cause not fully diagnosed."}
    elif kind == "BM_FullPipeline_Mandelbrot_1080p":
        fractal = "mandelbrot"
        resolution = [1920, 1080]
        measurement = "pipeline"
        max_iter = 1000
        param = "threads=hardware_concurrency(),quad_interp=1"
        bkey = baseline_key(measurement, fractal, max_iter, "f64", "naive", resolution=resolution)
        comparability = {"class": "baseline-common", "baseline_key": bkey,
                          "note": "Same fractal/resolution/max_iter/precision/algorithm as ruster's cpu/pipeline — "
                                   "DrawFractal() with quadInterpolationSize=1 (brute force, not FractalNow's real "
                                   "quadInterpolationSize=5 'solid guessing' default — see BENCHMARKING.md), "
                                   "including its real transfer-function + gradient-lookup stage, not a "
                                   "reconstruction. Threading strategy differs from ruster's persistent rayon pool, "
                                   "not a flaw."}

    impl_suffix = kind.replace("BM_", "").replace("_Mandelbrot_1080p", "").lower()
    impl = f"fractalnow-{impl_suffix}" + (f"-{fractal}" if kind in ("BM_Evaluate", "BM_FullFrame") else "")

    return {
        "impl": impl,
        "backend": {"family": "cpu-scalar", "detail": kind, "threads": threads},
        "workload": {
            "fractal": fractal_norm(fractal), "measurement": measurement, "algorithm": "naive",
            "resolution": resolution, "max_iter": max_iter, "precision": "f64", "param": param,
        },
        "comparability": comparability,
    }


GB_CLASSIFIERS = {
    "fractalrenderercpp": classify_fractalrenderercpp,
    "xaos": classify_xaos,
    "fractalnow": classify_fractalnow,
}


def parse_google_benchmark_json(project: str, language: str, path: Path, git_hash: Optional[str]) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    benchmarks = data.get("benchmarks", [])
    by_name: dict[str, dict] = {}
    for b in benchmarks:
        if b.get("run_type") != "aggregate":
            continue
        base_name = b["name"].rsplit("_", 1)[0]
        by_name.setdefault(base_name, {})[b["aggregate_name"]] = b

    classify = GB_CLASSIFIERS.get(project)
    records = []
    for base_name, aggs in by_name.items():
        mean_b = aggs.get("mean", {})
        unit = mean_b.get("time_unit", "ns")
        scale = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1000.0}.get(unit, 1e-6)

        def to_ms(b):
            return b.get("real_time", 0) * scale if b else None

        stats = {
            "mean_ms": to_ms(mean_b), "mean_ci_low_ms": None, "mean_ci_high_ms": None,
            "median_ms": to_ms(aggs.get("median")), "std_dev_ms": to_ms(aggs.get("stddev")),
            "confidence_level": None, "n_samples": None,
        }
        rec = new_record(
            project=project, language=language, git_hash=git_hash,
            source={"kind": "google_benchmark", "tool_version": "google_benchmark", "path": str(path)},
            stats=stats,
        )

        if classify:
            rec.update(classify(base_name))
            rec["comparability"]["note"] = (rec["comparability"].get("note") or "") + \
                " No bootstrap CI (Google Benchmark repetitions, not Criterion resampling)."
        else:
            rec["impl"] = f"{project}-{base_name}"
            rec["workload"]["fractal"] = fractal_norm(base_name)
            rec["workload"]["max_iter"] = MAX_ITER_DEFAULT
            rec["comparability"] = {
                "class": "unclassified", "baseline_key": None,
                "note": f"No classifier registered for project '{project}' in GB_CLASSIFIERS yet.",
            }

        w = rec["workload"]
        if w.get("resolution"):
            pixels = w["resolution"][0] * w["resolution"][1]
            if stats["mean_ms"]:
                mpix_s = pixels / (stats["mean_ms"] / 1000.0) / 1_000_000.0
                rec["derived"] = {"mpix_s": round(mpix_s, 3), "gflops": None, "gflops_note": None}
        apply_pixel_kernel_throughput(rec)
        records.append(rec)
    return records


# ── XaoS boundary-trace adapter (Benchmarks/bench_btrace.cpp) ──────────────────
#
# bench_btrace drives XaoS's real boundary-trace "guessing" algorithm
# (src/engine/btrace.cpp) directly -- the one thing bench_runner/
# criterion_bench deliberately never exercise (see BenchCore.h's file header,
# BENCHMARKING.md caveat 12, and bench_btrace.cpp's own module doc). Each
# JSON entry already carries both a single-threaded raw-kernel number and the
# boundary-trace number measured back-to-back at the same resolution/view, so
# this parser emits one Table A record per mode rather than reusing
# parse_manual_json's single-measurement-per-entry shape.
#
# Two things keep this out of Table A's baseline-common class, unlike
# xaos-fullframe-mandelbrot above: (1) it's forced single-threaded --
# btrace.cpp's own multithreaded path (tracerectangle2) is permanently
# disabled upstream (unresolved deadlocks, see btrace.cpp's comment above
# tracerectangle2) -- so it isn't comparable to any of this document's
# hardware_concurrency()-threaded bars without saying so; and (2) it's a
# single static frame with no previous-frame buffer (bench_btrace.cpp:
# nimages=1, no oldlines), so it measures the boundary-trace-and-flood-fill
# algorithm alone, not the additional inter-frame "guess from last frame's
# edges" reuse XaoS's real zoom animator layers on top in zoom.cpp (not
# linked into any Benchmarks/ target, including this one). A real, honest
# subset of "boundary tracing enabled" -- a lower bound on the interactive
# engine's actual speedup, not the whole of it. Also not independently
# verified pixel-for-pixel against the raw kernel's output -- timed only.

def parse_xaos_btrace_json(project: str, language: str, path: Path, git_hash: Optional[str]) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    records = []
    for s in data:
        resolution = [s["width"], s["height"]]
        max_iter = s.get("max_iter")
        base_workload = {"fractal": "mandelbrot", "algorithm": "naive",
                          "resolution": resolution, "max_iter": max_iter, "precision": "f64"}
        base_kwargs = dict(
            project=project, language=language, git_hash=git_hash,
            source={"kind": "manual", "tool_version": "bench_btrace", "path": str(path)},
        )
        records.append(new_record(
            impl="xaos-btrace-raw1t-mandelbrot",
            backend={"family": "cpu-scalar", "detail": "BM_FullFrame_1thread", "threads": 1},
            workload={**base_workload, "measurement": "render", "param": "threads=1"},
            comparability={"class": "project-unique", "baseline_key": None,
                            "note": "XaoS's raw per-pixel kernel pinned to threads=1, so it's comparable to the "
                                     "boundary-trace number below (btrace.cpp's multithreaded path is disabled "
                                     "upstream) -- NOT the hardware_concurrency()-threaded xaos-fullframe-mandelbrot "
                                     "number elsewhere in this document."},
            stats={"mean_ms": s.get("raw_mean_ms"), "median_ms": s.get("raw_mean_ms")},
            derived={"mpix_s": s.get("raw_mpix_s")},
            **base_kwargs,
        ))
        records.append(new_record(
            impl="xaos-btrace-mandelbrot",
            backend={"family": "cpu-scalar", "detail": "boundary_trace_1thread", "threads": 1},
            workload={**base_workload, "measurement": "render", "param": "threads=1,guessing=on"},
            comparability={"class": "project-unique", "baseline_key": None,
                            "note": "XaoS's real boundary-trace 'guessing' algorithm (src/engine/btrace.cpp), single "
                                     "static frame, single-threaded (see module doc above) -- not comparable to any "
                                     "multithreaded bar in this document, and not independently verified pixel-for-"
                                     "pixel against the raw kernel's output."},
            stats={"mean_ms": s.get("trace_mean_ms"), "median_ms": s.get("trace_mean_ms")},
            derived={"mpix_s": s.get("trace_mpix_s")},
            **base_kwargs,
        ))
    return records


# ── no-data placeholders ───────────────────────────────────────────────────────

def no_data(project: str, language: str, reason: str) -> dict:
    return new_record(
        project=project, language=language, impl=f"{project}-(unbenchmarked)",
        source={"kind": "no-data", "path": None},
        status="no-data",
        comparability={"class": "unclassified", "baseline_key": None, "note": reason},
    )


# ── discovery ───────────────────────────────────────────────────────────────────

def git_hash_of(repo: Path) -> Optional[str]:
    import subprocess
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def collect_all() -> list[dict]:
    records: list[dict] = []

    # ruster — real data
    ruster_criterion = RUSTER_ROOT / "bench_results" / "criterion_latest"
    run_id = ruster_criterion.resolve().name if ruster_criterion.exists() else None
    records += parse_criterion_project("ruster", "rust", ruster_criterion, run_id, git_hash_of(RUSTER_ROOT))

    for p in (RUSTER_ROOT / "results").glob("*.json"):
        records += parse_manual_json("ruster", "rust", p, git_hash_of(RUSTER_ROOT))
    for p in (RUSTER_ROOT / "bench_results").glob("run_*/timing/*.json"):
        records += parse_manual_json("ruster", "rust", p, git_hash_of(RUSTER_ROOT))

    # Fractals-rs — uses criterion too (see benches/application_bench.rs), classified
    # by classify_fractals_rs() above. Only re-run cargo bench there to refresh.
    fractals_rs = OTHER / "Fractals-rs"
    fr_criterion = fractals_rs / "target" / "criterion"
    if fr_criterion.exists():
        records += parse_criterion_project("fractals-rs", "rust", fr_criterion, None, git_hash_of(fractals_rs))
    else:
        records.append(no_data("fractals-rs", "rust",
            "criterion benches exist (benches/application_bench.rs) but haven't been run: "
            "`cd other-projects/Fractals-rs && cargo bench`."))

    # FractalRendererCpp — Google Benchmark, mirrors criterion stats via Repetitions()
    cpp_bench = OTHER / "FractalRendererCpp" / "build" / "Benchmarks" / "criterion_bench_result.json"
    if cpp_bench.exists():
        records += parse_google_benchmark_json("fractalrenderercpp", "cpp", cpp_bench, git_hash_of(OTHER / "FractalRendererCpp"))
    else:
        records.append(no_data("fractalrenderercpp", "cpp",
            "criterion_bench.cpp exists (Google Benchmark w/ Repetitions()->ReportAggregatesOnly) "
            "but hasn't been run: `./criterion_bench --benchmark_format=json > "
            "build/Benchmarks/criterion_bench_result.json`."))

    # XaoS — Google Benchmark, same adapter shape as fractalrenderercpp above.
    xaos_bench = OTHER / "XaoS" / "build" / "Benchmarks" / "criterion_bench_result.json"
    if xaos_bench.exists():
        records += parse_google_benchmark_json("xaos", "cpp", xaos_bench, git_hash_of(OTHER / "XaoS"))
    else:
        records.append(no_data("xaos", "cpp",
            "criterion_bench.cpp exists (Google Benchmark w/ Repetitions()->ReportAggregatesOnly) "
            "but hasn't been run: `cmake -S other-projects/XaoS -B other-projects/XaoS/build "
            "-DBUILD_BENCHMARKS=ON && cmake --build other-projects/XaoS/build --target criterion_bench && "
            "./other-projects/XaoS/build/Benchmarks/criterion_bench --benchmark_format=json > "
            "other-projects/XaoS/build/Benchmarks/criterion_bench_result.json`."))

    # XaoS boundary trace — hand-rolled bench_btrace, see parse_xaos_btrace_json's
    # module doc for exactly what this does and doesn't measure.
    xaos_btrace = OTHER / "XaoS" / "build" / "Benchmarks" / "bench_btrace_result.json"
    if xaos_btrace.exists():
        records += parse_xaos_btrace_json("xaos", "cpp", xaos_btrace, git_hash_of(OTHER / "XaoS"))
    else:
        records.append(no_data("xaos-btrace", "cpp",
            "bench_btrace exists (Benchmarks/bench_btrace.cpp, links engine/btrace.cpp) but hasn't been run: "
            "`cmake --build other-projects/XaoS/build --target bench_btrace && "
            "./other-projects/XaoS/build/Benchmarks/bench_btrace --json > "
            "other-projects/XaoS/build/Benchmarks/bench_btrace_result.json`."))

    # FractalNow — Google Benchmark, same adapter shape as fractalrenderercpp/xaos above.
    fractalnow_bench = OTHER / "fractalnow-0.8.2" / "build" / "Benchmarks" / "criterion_bench_result.json"
    if fractalnow_bench.exists():
        records += parse_google_benchmark_json("fractalnow", "c", fractalnow_bench, git_hash_of(OTHER / "fractalnow-0.8.2"))
    else:
        records.append(no_data("fractalnow", "c",
            "criterion_bench.cpp exists (Google Benchmark w/ Repetitions()->ReportAggregatesOnly) "
            "but hasn't been run: `cmake -S other-projects/fractalnow-0.8.2 -B other-projects/fractalnow-0.8.2/build "
            "-DBUILD_BENCHMARKS=ON && cmake --build other-projects/fractalnow-0.8.2/build --target criterion_bench && "
            "./other-projects/fractalnow-0.8.2/build/Benchmarks/criterion_bench --benchmark_format=json > "
            "other-projects/fractalnow-0.8.2/build/Benchmarks/criterion_bench_result.json`."))

    for project, language, reason in [
        ("cpp", "cpp", "No benchmark harness yet (plain port, no benches/ or scripts/bench_all.sh)."),
        ("zigger", "zig", "No benchmark harness yet."),
        ("nanobrot", "unknown", "Third-party reference implementation — no benchmark harness added."),
        ("fraqtive", "unknown", "Third-party reference implementation — no benchmark harness added."),
    ]:
        records.append(no_data(project, language, reason))

    return dedup_manual_records(records)


# ── reporting ───────────────────────────────────────────────────────────────────

def fmt(v, spec=".2f"):
    return format(v, spec) if isinstance(v, (int, float)) else "—"


def build_fair_comparison_table(records: list[dict]) -> str:
    groups: dict[str, list[dict]] = {}
    for r in records:
        if r["comparability"]["class"] == "baseline-common" and r["comparability"]["baseline_key"]:
            groups.setdefault(r["comparability"]["baseline_key"], []).append(r)

    if not groups:
        return ("_No records currently share a `baseline_key` — right now only ruster's "
                "`cpu/render/*::rayon` path is classified as `baseline-common`. This table "
                "fills in as soon as another project's naive-CPU render numbers are ingested "
                "(add a classifier for that project in `CLASSIFIERS`)._\n")

    lines = ["| Baseline | Impl | Mean ms | Mpix/s | Speedup vs slowest in group |",
             "|---|---|---:|---:|---:|"]
    for key, rs in sorted(groups.items()):
        rs_sorted = sorted(rs, key=lambda r: (r["derived"] or {}).get("mpix_s") or 0, reverse=True)
        slowest = min((r["derived"] or {}).get("mpix_s") or 1e-9 for r in rs_sorted)
        for r in rs_sorted:
            mpix = (r["derived"] or {}).get("mpix_s")
            speedup = f"{mpix / slowest:.2f}x" if mpix else "—"
            lines.append(f"| {key} | {r['impl']} | {fmt((r['stats'] or {}).get('mean_ms'))} | {fmt(mpix)} | {speedup} |")
    return "\n".join(lines) + "\n"


def build_full_capability_table(records: list[dict]) -> str:
    lines = ["| Project | Impl | Backend | Fractal | Measurement | Resolution | Mean ms | Mpix/s | GFLOPs | Class | Status |",
             "|---|---|---|---|---|---|---:|---:|---:|---|---|"]
    for r in sorted(records, key=lambda r: (r["project"], r["impl"])):
        if r["status"] == "no-data":
            lines.append(f"| {r['project']} | {r['impl']} | — | — | — | — | — | — | — | — | "
                         f"**no-data** ({r['comparability']['note']}) |")
            continue
        w = r["workload"]
        res = f"{w['resolution'][0]}×{w['resolution'][1]}" if w.get("resolution") else "—"
        st = r["stats"] or {}
        d = r["derived"] or {}
        cls = r["comparability"]["class"]
        cls_display = {"baseline-common": "baseline", "project-unique": "★ unique", "unclassified": "? unclassified"}.get(cls, cls)
        lines.append(f"| {r['project']} | {r['impl']} | {r['backend']['family']} ({r['backend']['detail']}) | "
                     f"{w.get('fractal') or '—'} | {w.get('measurement')} | {res} | "
                     f"{fmt(st.get('mean_ms'))} | {fmt(d.get('mpix_s'))} | {fmt(d.get('gflops'))} | {cls_display} | ok |")
    return "\n".join(lines) + "\n"


def build_report(records: list[dict]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_ok = sum(1 for r in records if r["status"] == "ok")
    n_nodata = sum(1 for r in records if r["status"] == "no-data")
    return f"""# Cross-project fractal renderer benchmark comparison

Generated {now}. {n_ok} measured records, {n_nodata} project(s) with no data yet.

> ## Scope and provenance of this run
>
> **Mandelbrot only.** The latest criterion run was filtered to Mandelbrot (plus the
> fractal-agnostic `colorize`/`reference_orbit`/`series_approx` groups). Julia, Newton and
> Nova entries still present in `criterion_latest` are **carry-over from the previous run**
> and were not re-measured; the default Mandelbrot scoping of this report excludes them.
>
> **Both GPU backends now run on the discrete RTX 3050**, so wgpu-vs-CUDA is a genuine
> same-silicon comparison. This was not true before 2026-08-08: the NVIDIA stack was
> installed compute-only, there was no NVIDIA Vulkan ICD, and wgpu silently fell back to
> the integrated AMD Vega. **Every wgpu number in runs up to and including
> `criterion_20260725_215202_8d2dbdc` came from the integrated GPU** and must not be
> compared against a CUDA number from those runs.
>
> **Cross-run CPU comparisons are unreliable.** The previous archive was recorded
> back-to-back with another full sweep on a thermally-loaded laptop; identical CPU
> benchmarks moved 15–30% between that run and this one with no code change. Only compare
> CPU rows recorded in the *same* run. FractalRendererCpp and Fractals-rs numbers here
> predate this run and were not re-measured, so cross-project ratios carry that caveat —
> re-run all three in one session before quoting a headline speedup.

## Table A — Fair comparison (same fractal/resolution/max_iter/precision, naive CPU path)

{build_fair_comparison_table(records)}

## Table B — Full capability matrix (every measurement, including ruster-only backends and un-instrumented projects)

{build_full_capability_table(records)}

Notes:
- Table A is deliberately narrow: it's the only place where "renderer X is Nx faster than
  renderer Y" is a fair statement, because everything in it shares fractal, resolution,
  iteration count, precision, and algorithm class.
- Table B is deliberately everything else: ruster's SIMD/wgpu/CUDA/hybrid/perturbation
  backends have no peer in another project today, so they'd disappear from a
  comparison-only report. They're kept here, tagged `★ unique`, so ruster's full
  performance envelope stays visible even where there's nothing to compare it to.
- `no-data` rows are real projects (nanobrot, fraqtive, cpp, zigger) that simply haven't
  produced numbers yet, kept as rows instead of omitted so the comparison's *coverage*
  is visible, not just its results.
- FractalRendererCpp (Google Benchmark, run via `./criterion_bench --benchmark_format=json`)
  lands in Table A for ALL FOUR of its benchmark groups: `BM_Evaluate`, `BM_FullFrame`,
  `BM_ThreadScaling_Mandelbrot_1080p`, and `BM_FullPipeline_Mandelbrot_1080p` are all
  confirmed f64 scalar naive at max_iter=1000, matching ruster's baseline exactly (see
  classify_fractalrenderercpp()) — the one real difference being threading strategy (raw
  std::thread spawn/join per call vs a persistent rayon pool), which the comparison
  reports rather than hides. (`BM_FullFrame`/`BM_FullPipeline` used to run at max_iter=300,
  mirroring Fractals-rs's own full-frame benches at the time — both were bumped to 1000 in
  Benchmarks/criterion_bench.cpp so iteration count no longer blocks the join.)
- Fractals-rs has real Criterion data (Table B) but none of it lands in Table A: its
  full-frame benches run SIMD-only (f32 "fast" / f64 "high"), while ruster's baseline is
  scalar f64 — different algorithm class, so treating them as equal would be a false
  equivalence, not a fair comparison, even now that both run at max_iter=1000 (bumped
  from 300 in benches/application_bench.rs — iteration count is no longer the blocker,
  algorithm class still is). Re-run one side to match the other's precision/algorithm and
  the join is automatic — see the notes in classify_fractals_rs().
- **`precision` is now derived per-record, not assumed.** Every ruster row used to be
  stamped `f64` unconditionally, which was wrong for the majority of them: wgpu is f32
  always (WGSL has no f64 type), CUDA takes `fractal_kernel_f32` for Mandelbrot/Julia
  below zoom 1e6, `f32x8`/`f32x8_ilp` SIMD are f32, and the hybrid scheduler runs f32 on
  both halves by default. The headline "GPU is 5x the CPU baseline" numbers are therefore
  **f32 GPU vs f64 CPU** and are not a like-for-like precision comparison — the honest
  like-for-like pairing is wgpu/CUDA against ruster's own `f32x8_ilp` SIMD row, which is
  the same precision on the same frame. `ruster_precision()` carries the rules and the
  source references.
"""


# ── fractal scoping ─────────────────────────────────────────────────────────────
# Default view is Mandelbrot-only: it's the one fractal every project here
# renders, so it's the fair basis for the flagship charts/tables. `no-data`
# rows and fractal-agnostic records (reference_orbit, series_approx — cost is
# independent of which fractal is on screen) are kept regardless of filter,
# since dropping them would misrepresent coverage rather than just scope it.

def filter_by_fractal(records: list[dict], fractal: Optional[str]) -> list[dict]:
    if not fractal:
        return records
    target = fractal.lower()
    return [r for r in records
            if r["status"] == "no-data"
            or r["workload"].get("fractal") is None
            or r["workload"]["fractal"] == target]


# ── CSV export ──────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "project", "impl", "language", "backend_family", "backend_detail", "backend_device", "threads",
    "fractal", "measurement", "algorithm", "resolution", "max_iter", "precision", "param",
    "comparability_class", "baseline_key", "comparability_note",
    "mean_ms", "mean_ci_low_ms", "mean_ci_high_ms", "median_ms", "std_dev_ms",
    "mpix_s", "gflops", "status", "run_id", "git_hash",
]


def record_to_csv_row(r: dict) -> dict:
    w, b, c = r["workload"], r["backend"], r["comparability"]
    st, d = r["stats"] or {}, r["derived"] or {}
    res = f"{w['resolution'][0]}x{w['resolution'][1]}" if w.get("resolution") else ""
    return {
        "project": r["project"], "impl": r["impl"], "language": r["language"],
        "backend_family": b["family"], "backend_detail": b.get("detail"),
        "backend_device": b.get("device"), "threads": b.get("threads"),
        "fractal": w.get("fractal"), "measurement": w.get("measurement"), "algorithm": w.get("algorithm"),
        "resolution": res, "max_iter": w.get("max_iter"), "precision": w.get("precision"), "param": w.get("param"),
        "comparability_class": c["class"], "baseline_key": c.get("baseline_key"), "comparability_note": c.get("note"),
        "mean_ms": st.get("mean_ms"), "mean_ci_low_ms": st.get("mean_ci_low_ms"), "mean_ci_high_ms": st.get("mean_ci_high_ms"),
        "median_ms": st.get("median_ms"), "std_dev_ms": st.get("std_dev_ms"),
        "mpix_s": d.get("mpix_s"), "gflops": d.get("gflops"),
        "status": r["status"], "run_id": r.get("run_id"), "git_hash": r.get("git_hash"),
    }


def write_csv(records: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in sorted(records, key=lambda r: (r["project"], r["impl"])):
            writer.writerow(record_to_csv_row(r))


# ── PDF charts ──────────────────────────────────────────────────────────────────
# Requires matplotlib — declared in the PEP 723 header at the top of this file,
# so `uv run scripts/aggregate_bench.py` gets it in an ephemeral env without
# touching system Python (Debian/Ubuntu's externally-managed-environment
# guard blocks a plain `pip install` here).
#
# Chart pages are organized around the actual Criterion group taxonomy in
# benches/fractal_bench.rs (cpu/simd/wgpu/cuda/hybrid render, pipeline,
# thread_scaling, render_tiled, colorize, pixel_kernel, perturbation +
# reference_orbit + series_approx) — every distinct benchmark variation gets
# its own clearly labeled page instead of one flattened, hard-to-read chart.
#
# Two color dimensions, never mixed on the same chart:
#   PROJECT_COLOR — which app (ruster / Fractals-rs / FractalRendererCpp) —
#                   cross-project charts.
#   BACKEND_COLOR — which compute backend within ruster (cpu-scalar / cpu-simd
#                   / cpu-tiled / gpu-cuda / gpu-wgpu / hybrid) — ruster-internal
#                   charts, always plotted in BACKEND_ORDER (the expected
#                   performance hierarchy: scalar < SIMD < GPU/hybrid) so any
#                   bar breaking that order is visible, not hidden by sorting
#                   bars by value (which was the earlier chart's problem).
# Every chart also gets a "higher/lower is better" badge — direction isn't
# always "higher": reference_orbit/series_approx report raw per-call cost in
# ms, where lower is better.

PROJECT_COLOR = {"ruster": "#2b6cb0", "fractals-rs": "#b83280", "fractalrenderercpp": "#c05621",
                  "xaos": "#d69e2e", "fractalnow": "#319795"}
PROJECT_COLOR_DEFAULT = "#a0aec0"

BACKEND_COLOR = {
    "cpu-scalar": "#718096", "cpu-simd": "#3182ce", "cpu-tiled": "#63b3ed",
    "gpu-cuda": "#805ad5", "gpu-wgpu": "#38a169", "hybrid": "#c53030",
}
BACKEND_ORDER = ["cpu-scalar", "cpu-simd", "cpu-tiled", "gpu-cuda", "gpu-wgpu", "hybrid"]
BACKEND_LABEL = {
    "cpu-scalar": "CPU scalar", "cpu-simd": "CPU SIMD", "cpu-tiled": "CPU tiled (Hilbert)",
    "gpu-cuda": "GPU (CUDA)", "gpu-wgpu": "GPU (wgpu)", "hybrid": "Hybrid CPU+GPU",
}

ZOOM_ORDER = ["zoom_1e0", "zoom_1e3", "zoom_1e6", "zoom_1e9", "zoom_1e12"]
ZOOM_LABELS = ["1", "1e3", "1e6", "1e9", "1e12"]


def _direction_badge(ax, better: str, metric: str) -> None:
    arrow = "↑" if better == "higher" else "↓"
    ax.text(0.99, 0.02, f"{arrow} {better} {metric} = better", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, style="italic", color="#4a5568",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f7fafc", ec="#cbd5e0", alpha=0.9))


def _bar_value_labels(ax, bars, fmt: str = "{:.0f}") -> None:
    for b in bars:
        h = b.get_height()
        if not h:
            continue
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7, color="#2d3748")


def _hbar_value_labels(ax, bars, fmt: str = "{:.0f}") -> None:
    for b in bars:
        w = b.get_width()
        if not w:
            continue
        ax.annotate(fmt.format(w), (w, b.get_y() + b.get_height() / 2), xytext=(3, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=7, color="#2d3748")


def _backend_legend(ax, plt, families: list[str]) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=BACKEND_COLOR[f]) for f in families]
    ax.legend(handles, [BACKEND_LABEL[f] for f in families], loc="upper left", fontsize=7.5, framealpha=0.9)


def build_pdf(records: list[dict], path: Path, fractal_label: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("matplotlib not installed — skipping PDF. Run via `uv run scripts/aggregate_bench.py` "
              "so the PEP 723 dependency block installs it automatically.", file=sys.stderr)
        return False

    ok = [r for r in records if r["status"] == "ok"]

    with PdfPages(path) as pdf:
        _chart_render_hierarchy(ok, plt, pdf, fractal_label)
        _chart_other_projects_render(ok, plt, pdf, fractal_label)
        _chart_hybrid_vs_solo(ok, plt, pdf, fractal_label)
        _chart_pipeline_overhead(ok, plt, pdf, fractal_label)
        _chart_thread_scaling(ok, plt, pdf, fractal_label)
        _chart_cpu_microopt(ok, plt, pdf, fractal_label)
        _chart_pixel_kernel(ok, plt, pdf, fractal_label)
        _chart_perturbation_render(ok, plt, pdf, fractal_label)
        _chart_perturbation_internals(ok, plt, pdf, fractal_label)
        _chart_fair_comparison(ok, plt, pdf, fractal_label)
    return True


def _chart_render_hierarchy(records: list[dict], plt, pdf, fractal_label: str) -> None:
    """Page 1 — the headline chart. Grouped bars: one group per resolution,
    ruster's bars in the EXPECTED performance hierarchy order (CPU scalar <
    CPU SIMD < GPU/hybrid, colored by backend family) — a fixed x-order, not
    sorted by value, so any backend that breaks that order (see the CUDA note
    below) is visible rather than smoothed away — PLUS Fractals-rs,
    FractalRendererCpp, XaoS and FractalNow's own render numbers on the same
    axes, colored by project instead of backend family since they're a
    different axis (each has only one or two variants, not ruster's full
    backend spread). XaoS and FractalNow are their raw per-pixel kernel with
    guessing/quad-interpolation forced off (Table A, caveats 12 and 17) — not
    a measurement of either project's real interactive engine.

    All projects now run max_iter=1000 (Fractals-rs's and FractalRendererCpp's
    full-frame benches used to run at max_iter=300 — bumped to match ruster's
    baseline in their own bench sources, see classify_fractals_rs()/
    classify_fractalrenderercpp()'s module comments). FractalRendererCpp's bar
    is now a genuine same-algorithm (f64 scalar naive) apples-to-apples pair
    with ruster's cpu-scalar bar (Table A), differing only in threading
    strategy. Fractals-rs's bar is still a different axis (SIMD-only, no
    scalar path in that project) rather than a different iteration count, so
    it's kept in this "own render numbers, colored by project" bucket
    alongside ruster's backend-family bars rather than merged into them.
    Only the f64 ("high") variant is plotted here — the f32 ("fast") one used
    to sit alongside it, but at 1920x1080/3840x2160 it renders at or above
    wgpu's throughput, which misrepresents a CPU-only project as beating a
    real GPU backend. It isn't: Fractals-rs's "full frame" numbers fuse
    iteration with a cheap non-histogram palette lookup (BENCHMARKING.md
    caveat 1), not the histogram-equalized colorize() every other bar here
    pays for, so the f32 number under-counts relative to everything else on
    this axis by more than precision alone explains. The f64 bar carries the
    same color-pipeline asterisk but stays, since it doesn't visually cross
    the GPU bars and is the more conservative of the two.

    A fifth XaoS bar (hatched) adds its real boundary-trace "guessing"
    algorithm (src/engine/btrace.cpp, driven directly by Benchmarks/
    bench_btrace.cpp) alongside its raw-kernel bar — the one thing every
    other XaoS/FractalNow number here deliberately excludes (see caveat 12).
    Two things keep it out of the same class as the hierarchy bars: it's
    forced single-threaded (btrace.cpp's own multithreaded path is disabled
    upstream, unresolved deadlocks) so it isn't compared at
    hardware_concurrency() like the rest of this figure, and it's a single
    static frame with no previous-frame buffer, so it measures the
    boundary-trace-and-flood-fill algorithm alone, not the additional
    inter-frame reuse XaoS's real zoom animator layers on top of it — a real
    lower bound on the interactive engine's speedup, not the whole of it."""
    representative = {
        "cpu-scalar": "ruster-cpu-rayon", "cpu-simd": "ruster-simd-f32x8",
        "cpu-tiled": "ruster-cpu-hilbert", "gpu-cuda": "ruster-cuda-cuda",
        "gpu-wgpu": "ruster-wgpu-gpu",
    }
    other_series = [
        ("fractals-rs-full_frame-high", "Fractals-rs (f64 SIMD, iter=1000)", PROJECT_COLOR["fractals-rs"], ".."),
        ("fractalrenderercpp-fullframe-mandelbrot", "FractalRendererCpp (f64 scalar, iter=1000 — apples-to-apples)", PROJECT_COLOR["fractalrenderercpp"], None),
        ("xaos-fullframe-mandelbrot", "XaoS (f64 scalar, raw kernel, iter=1000 — guessing off)", PROJECT_COLOR["xaos"], None),
        ("xaos-btrace-mandelbrot", "XaoS (boundary trace \"guessing\" on, 1 thread, static frame)", PROJECT_COLOR["xaos"], "//"),
        ("fractalnow-fullframe-mandelbrot", "FractalNow (f64 scalar, raw kernel, iter=1000 — quad-interp off)", PROJECT_COLOR["fractalnow"], None),
    ]
    resolutions = [[800, 600], [1920, 1080], [3840, 2160]]
    lookup = {}
    for r in records:
        w = r["workload"]
        if w["measurement"] == "render" and w.get("resolution") and r.get("derived"):
            lookup[(r["impl"], tuple(w["resolution"]))] = r

    families = [f for f in BACKEND_ORDER if f != "hybrid"
                and any((representative[f], tuple(res)) in lookup for res in resolutions)]
    other_present = [s for s in other_series if any((s[0], tuple(res)) in lookup for res in resolutions)]

    # The two adaptive hybrid schedulers only ever run at a fixed 1920x1080
    # (bench_heterogeneous/bench_heterogeneous_wgpu don't sweep resolution —
    # see their zoom-sweep-at-fixed-1920x1080 doc comments), and their records
    # carry a zoom `param`, not a `resolution`, so they can't join the
    # `(impl, resolution)`-keyed `lookup` above. Look them up separately and
    # only place a bar in the 1920x1080 group (0 elsewhere — same convention
    # `_bar_value_labels` already treats as "no bar" for `other_present`).
    hybrid_by_impl = {}
    for r in records:
        w = r["workload"]
        if w["measurement"] in ("heterogeneous", "heterogeneous_wgpu") and w.get("param") == "zoom=zoom_1e0" and r.get("derived"):
            hybrid_by_impl[r["impl"]] = r
    hybrid_series = [
        ("ruster-hybrid-heterogeneous", "Hybrid CPU+CUDA (adaptive)", "xx"),
        ("ruster-hybrid-heterogeneous_wgpu", "Hybrid CPU+wgpu (adaptive)", "oo"),
    ]
    hybrid_present = [s for s in hybrid_series if s[0] in hybrid_by_impl]

    if not families and not other_present and not hybrid_present:
        return

    bars_spec = [(representative[f], BACKEND_COLOR[f], None) for f in families] + \
                [(impl, color, hatch) for impl, _, color, hatch in other_present] + \
                [(impl, BACKEND_COLOR["hybrid"], hatch) for impl, _, hatch in hybrid_present]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    n = len(bars_spec)
    width = 0.8 / n
    x = list(range(len(resolutions)))
    for i, (impl, color, hatch) in enumerate(bars_spec):
        if impl in hybrid_by_impl:
            ys = [hybrid_by_impl[impl]["derived"]["mpix_s"] if tuple(res) == (1920, 1080) else 0
                  for res in resolutions]
        else:
            ys = [lookup[(impl, tuple(res))]["derived"]["mpix_s"] if (impl, tuple(res)) in lookup else 0
                  for res in resolutions]
        xs = [xi + (i - (n - 1) / 2) * width for xi in x]
        bars = ax.bar(xs, ys, width=width * 0.9, color=color, hatch=hatch, edgecolor="white" if hatch else None)
        _bar_value_labels(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{w}×{h}" for w, h in resolutions])
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Mpix/s")
    ax.set_title(f"{fractal_label.title()} — render throughput, every project and backend\n"
                 "(ruster bars ordered CPU-scalar → CPU-SIMD → CPU-tiled → GPU-CUDA → GPU-wgpu → Hybrid, the expected hierarchy;\n"
                 "hybrid bars only exist at 1920×1080 — see caption)")
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.32)  # headroom so the legend and callout box below don't overlap the bars
    handles = [plt.Rectangle((0, 0), 1, 1, color=BACKEND_COLOR[f]) for f in families]
    labels = [BACKEND_LABEL[f] for f in families]
    for impl, label, color, hatch in other_present:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color, hatch=hatch, ec="white" if hatch else "none"))
        labels.append(label)
    for impl, label, hatch in hybrid_present:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=BACKEND_COLOR["hybrid"], hatch=hatch, ec="white"))
        labels.append(label)
    ax.legend(handles, labels, loc="upper left", fontsize=7.5, framealpha=0.9)
    _direction_badge(ax, "higher", "Mpix/s")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _chart_other_projects_render(records: list[dict], plt, pdf, fractal_label: str) -> None:
    """Page 2 — the same 'render throughput by resolution' idea as page 1,
    but for Fractals-rs and FractalRendererCpp. Neither project has ruster's
    full backend hierarchy: Fractals-rs is SIMD-only (no scalar path, two
    precision variants — 'fast' f32 vs 'high' f64), FractalRendererCpp is
    scalar-only (no SIMD/GPU at all — see RenderCore.h's module doc). Each
    gets its own panel colored by its own axis of variation rather than
    ruster's backend-family palette, since these aren't the same axis.
    All three projects now run max_iter=1000 (see Table A notes) — this page
    is still each project's own internal comparison rather than a
    cross-project speed claim, but iteration count is no longer a caveat for
    either panel (FractalRendererCpp is a genuine same-algorithm match with
    ruster's cpu-scalar bar; Fractals-rs's remaining difference is SIMD vs
    scalar, not iteration count)."""
    resolutions = [[800, 600], [1920, 1080], [3840, 2160]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Fractals-rs: fast (f32 SIMD) vs high (f64 SIMD), from the full_frame/<Fractal> sweep
    ax = axes[0]
    lookup = {}
    for r in records:
        if (r["project"] != "fractals-rs" or r["workload"]["measurement"] != "render"
                or not r["workload"].get("resolution") or not r.get("derived")):
            continue
        prec = r["workload"].get("precision")
        if prec in ("f32", "f64"):
            lookup[(prec, tuple(r["workload"]["resolution"]))] = r["derived"]["mpix_s"]
    variants = [("f32", "#63b3ed", "fast (f32 SIMD)"), ("f64", "#2b6cb0", "high (f64 SIMD)")]
    x = list(range(len(resolutions)))
    n = len(variants)
    width = 0.8 / n
    plotted_a = False
    for i, (prec, color, label) in enumerate(variants):
        ys = [lookup.get((prec, tuple(res)), 0) for res in resolutions]
        if any(ys):
            xs = [xi + (i - (n - 1) / 2) * width for xi in x]
            bars = ax.bar(xs, ys, width=width * 0.9, color=color, label=label)
            _bar_value_labels(ax, bars)
            plotted_a = True
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w}×{h}" for w, h in resolutions])
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Mpix/s")
    ax.set_title("Fractals-rs — Mandelbrot render throughput\n(max_iter=1000, matches ruster's baseline — SIMD only, no scalar path)", fontsize=9)
    if plotted_a:
        ax.legend(fontsize=8)
    _direction_badge(ax, "higher", "Mpix/s")
    if not plotted_a:
        ax.text(0.5, 0.5, "No full_frame/<Fractal> multi-resolution data yet —\nrun `cargo bench` in other-projects/Fractals-rs",
                transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#718096")

    # FractalRendererCpp: single backend (f64 scalar naive, its only compute path)
    ax = axes[1]
    rows = [r for r in records if r["impl"] == "fractalrenderercpp-fullframe-mandelbrot" and r.get("derived")]
    rows.sort(key=lambda r: r["workload"]["resolution"][0] * r["workload"]["resolution"][1])
    if rows:
        xs = [f"{r['workload']['resolution'][0]}×{r['workload']['resolution'][1]}" for r in rows]
        ys = [r["derived"]["mpix_s"] for r in rows]
        bars = ax.bar(xs, ys, color=PROJECT_COLOR["fractalrenderercpp"])
        _bar_value_labels(ax, bars)
    else:
        ax.text(0.5, 0.5, "No data — run criterion_bench --benchmark_format=json",
                transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#718096")
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Mpix/s")
    ax.set_title("FractalRendererCpp — Mandelbrot render throughput\n(f64 scalar naive, max_iter=1000 — apples-to-apples with ruster's cpu-scalar, see Table A)", fontsize=9)
    _direction_badge(ax, "higher", "Mpix/s")

    fig.suptitle(f"{fractal_label.title()} — render throughput, other projects (their own internal comparison, not vs ruster — see Table A)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _chart_hybrid_vs_solo(records: list[dict], plt, pdf, fractal_label: str) -> None:
    """Page 2 — solo CPU/GPU backends next to ruster's three hybrid designs,
    all at 1920x1080 / zoom=1 (the only resolution/zoom the solo and
    static-split benchmarks run at — both heterogeneous scheduler bars are
    pinned to their own zoom=1e0 case for the same reason).

    Three "hybrid" bars appear here, and they are NOT interchangeable:
    - "CPU+wgpu (static 50/50 split)": bench_hybrid_cpu_wgpu — CPU always
      renders the top half, GPU the bottom half, concurrently, no
      classification at all. Still genuinely naive today — kept so the
      adaptive wgpu scheduler bar has a "what it replaced" to compare against.
    - "CPU+CUDA (adaptive scheduler)" / "CPU+wgpu (adaptive scheduler)":
      scheduler::render_heterogeneous / render_heterogeneous_wgpu —
      corner-sampling classification + work-stealing queues, on top of CUDA
      or wgpu respectively. Both replace what used to be (or, for wgpu, still
      is above) an equally-naive static split. Do not resurrect a
      "Hybrid CPU+CUDA (static split)" bar here — bench_hybrid_cpu_cuda was
      deleted from the source on purpose.
    Expectation: the static-split bar trails solo GPU when GPU is much faster
    than CPU (the CPU half becomes the bottleneck); the adaptive scheduler
    bars are not expected to have that specific failure mode, since neither
    forces a fixed 50/50 split — see `results/summary.md`'s Stage 3 discussion
    and `src/scheduler/classifier.rs`'s module doc for why."""
    wanted = [
        ("cpu-scalar", "ruster-cpu-rayon", None, "CPU\nscalar"),
        ("cpu-simd", "ruster-simd-f32x8", None, "CPU\nSIMD f32x8"),
        ("gpu-cuda", "ruster-cuda-cuda", None, "GPU\nCUDA (solo)"),
        ("gpu-wgpu", "ruster-wgpu-gpu", None, "GPU\nwgpu (solo)"),
        ("hybrid", "ruster-hybrid-cpu+wgpu", None, "Hybrid CPU+wgpu\n(static 50/50 split)"),
        ("hybrid", "ruster-hybrid-heterogeneous", "zoom=zoom_1e0", "Hybrid CPU+CUDA\n(adaptive scheduler)"),
        ("hybrid", "ruster-hybrid-heterogeneous_wgpu", "zoom=zoom_1e0", "Hybrid CPU+wgpu\n(adaptive scheduler)"),
    ]
    by_key = {}
    for r in records:
        if not r.get("derived"):
            continue
        by_key.setdefault(r["impl"], []).append(r)

    def _lookup(impl: str, param: str | None):
        rows = by_key.get(impl) or []
        if param is None:
            return rows[0] if rows else None
        for r in rows:
            if r["workload"].get("param") == param:
                return r
        return None

    labels, values, colors, families_present = [], [], [], []
    for fam, impl, param, label in wanted:
        rec = _lookup(impl, param)
        if not rec:
            continue
        labels.append(label)
        values.append(rec["derived"]["mpix_s"])
        colors.append(BACKEND_COLOR[fam])
        if fam not in families_present:
            families_present.append(fam)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ymax = max(values)
    ax.set_ylim(0, ymax * 1.3)
    bars = ax.bar(labels, values, color=colors)
    _bar_value_labels(ax, bars)
    ax.set_ylabel("Mpix/s")
    ax.set_title(f"{fractal_label.title()} @ 1920×1080 — solo backends vs ruster's hybrid designs")
    ax.tick_params(axis="x", labelsize=8)
    _direction_badge(ax, "higher", "Mpix/s")
    _backend_legend(ax, plt, sorted(families_present, key=BACKEND_ORDER.index))
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _chart_pipeline_overhead(records: list[dict], plt, pdf, fractal_label: str) -> None:
    """Page 3 — render-only vs render+colorize ('pipeline') per backend at
    1920x1080. The gap between the paired bars is the colorize stage's cost."""
    pairs = [
        ("cpu-scalar", "ruster-cpu-rayon", "ruster-cpu-pipeline", "CPU"),
        ("gpu-wgpu", "ruster-wgpu-gpu", "ruster-wgpu-pipeline", "GPU (wgpu)"),
        ("gpu-cuda", "ruster-cuda-cuda", "ruster-cuda-pipeline", "GPU (CUDA)"),
    ]
    by_impl = {r["impl"]: r for r in records if r.get("derived")}
    render_vals, pipeline_vals, colors, labels = [], [], [], []
    for fam, render_impl, pipeline_impl, label in pairs:
        r1, r2 = by_impl.get(render_impl), by_impl.get(pipeline_impl)
        if not r1 or not r2:
            continue
        render_vals.append(r1["derived"]["mpix_s"])
        pipeline_vals.append(r2["derived"]["mpix_s"])
        colors.append(BACKEND_COLOR[fam])
        labels.append(label)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = list(range(len(labels)))
    width = 0.32
    b1 = ax.bar([i - width / 2 for i in xs], render_vals, width=width, color=colors, alpha=1.0)
    b2 = ax.bar([i + width / 2 for i in xs], pipeline_vals, width=width, color=colors, alpha=0.45)
    _bar_value_labels(ax, b1)
    _bar_value_labels(ax, b2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mpix/s")
    ax.set_title(f"{fractal_label.title()} @ 1920×1080 — colorize overhead\n(solid = render only, faded = render + colorize)")
    _direction_badge(ax, "higher", "Mpix/s")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#4a5568", alpha=1.0),
               plt.Rectangle((0, 0), 1, 1, color="#4a5568", alpha=0.45)]
    ax.legend(handles, ["render only", "render + colorize"], fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _chart_thread_scaling(records: list[dict], plt, pdf, fractal_label: str) -> None:
    """Page 4 — Mpix/s vs thread count, all three projects on one axis.
    ruster and FractalRendererCpp are confirmed f64 scalar naive (max_iter=
    1000, 1920x1080) — a fair pair, the one real difference being threading
    strategy: rayon pool vs raw std::thread spawn/join per call. Fractals-rs
    runs f32 SIMD, not scalar — NOT an apples-to-apples precision match with
    the other two — so its line is dashed and its legend label says "f32 SIMD"
    plainly rather than being hidden in a separate panel; a reader comparing
    absolute heights should still notice the precision difference from the
    label, not have it "explained away" by a physical split."""
    fig, ax = plt.subplots(figsize=(9, 6))

    series = [
        ("ruster-cpu-thread_scaling", "ruster (rayon pool, f64 scalar)", PROJECT_COLOR["ruster"], "-"),
        ("fractalrenderercpp-threadscaling", "FractalRendererCpp (std::thread/call, f64 scalar)", PROJECT_COLOR["fractalrenderercpp"], "-"),
        ("fractals-rs-thread_scaling-threads", "Fractals-rs (rayon pool, f32 SIMD — not a scalar pair)", PROJECT_COLOR["fractals-rs"], "--"),
    ]
    plotted = False
    missing = []
    for impl, label, color, linestyle in series:
        rows = [r for r in records if r["impl"] == impl and r["backend"].get("threads") and r.get("derived")]
        rows.sort(key=lambda r: r["backend"]["threads"])
        if not rows:
            missing.append(label)
            continue
        xs = [r["backend"]["threads"] for r in rows]
        ys = [r["derived"]["mpix_s"] for r in rows]
        ax.plot(xs, ys, marker="o", color=color, linestyle=linestyle, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Threads")
    ax.set_ylabel("Mpix/s (log scale)")
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.legend(fontsize=8)
    _direction_badge(ax, "higher", "Mpix/s")
    if missing:
        ax.text(0.02, 0.02, "No data yet: " + "; ".join(missing),
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7, color="#718096")
    ax.set_title(f"{fractal_label.title()} @ 1920×1080 — CPU thread scaling, max_iter=1000\n"
                 "(ruster/FractalRendererCpp: f64 scalar, a fair pair — Fractals-rs: f32 SIMD, dashed, not the same precision)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _chart_cpu_microopt(records: list[dict], plt, pdf, fractal_label: str) -> None:
    """Page 5 — two CPU-only micro-optimizations from CURSOR_OPTIMIZATIONS.md:
    cache-locality (Hilbert tiling vs row-major) and instruction-level
    parallelism (interleaved f32x8 vs plain f32x8)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    plotted_a = False
    for impl, label, color in [("ruster-cpu-rows", "Row-major", "#718096"),
                                ("ruster-cpu-hilbert", "Hilbert tiled", "#63b3ed")]:
        rows = [r for r in records if r["impl"] == impl and r["workload"].get("resolution") and r.get("derived")]
        rows.sort(key=lambda r: r["workload"]["resolution"][0] * r["workload"]["resolution"][1])
        if not rows:
            continue
        xs = [f"{r['workload']['resolution'][0]}×{r['workload']['resolution'][1]}" for r in rows]
        ys = [r["derived"]["mpix_s"] for r in rows]
        ax.plot(xs, ys, marker="o", color=color, label=label)
        plotted_a = True
    ax.set_ylabel("Mpix/s")
    ax.set_title("Cache locality: row-major vs Hilbert-tiled traversal")
    if plotted_a:
        ax.legend(fontsize=8)
    _direction_badge(ax, "higher", "Mpix/s")

    ax = axes[1]
    plotted_b = False
    # Both arms must come from the same criterion group ("simd_render_ilp_Mandelbrot_1080p",
    # despite its "_1080p" name it sweeps all three resolutions) — that group is the only
    # place "f32x8_ilp" is measured, and it measures a "f32x8" baseline arm alongside it
    # under identical conditions. The *separate* "simd_render_Mandelbrot" sweep also produces
    # "ruster-simd-f32x8" rows at the same resolutions, but from a different benchmark run;
    # mixing the two in one line double-plots every x with unrelated y values.
    for impl, label, color in [("ruster-simd-f32x8", "f32x8 SIMD", "#3182ce"),
                                ("ruster-simd-f32x8_ilp", "f32x8 + ILP (2× interleaved)", "#805ad5")]:
        rows = [r for r in records if r["impl"] == impl and r["workload"].get("resolution") and r.get("derived")
                and "simd_render_ilp_Mandelbrot_1080p" in r.get("source", {}).get("path", "")]
        rows.sort(key=lambda r: r["workload"]["resolution"][0] * r["workload"]["resolution"][1])
        if not rows:
            continue
        xs = [f"{r['workload']['resolution'][0]}×{r['workload']['resolution'][1]}" for r in rows]
        ys = [r["derived"]["mpix_s"] for r in rows]
        ax.plot(xs, ys, marker="o", color=color, label=label)
        plotted_b = True
    ax.set_ylabel("Mpix/s")
    ax.set_title("Instruction-level parallelism: f32x8 vs interleaved f32x8")
    if plotted_b:
        ax.legend(fontsize=8)
    _direction_badge(ax, "higher", "Mpix/s")

    if not (plotted_a or plotted_b):
        plt.close(fig)
        return
    fig.suptitle(f"{fractal_label.title()} — CPU micro-optimizations")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _chart_pixel_kernel(records: list[dict], plt, pdf, fractal_label: str) -> None:
    """Page 6 — single-pixel kernel cost, every project that has one, colored
    by project. Normalized to M-evaluations/s so ruster's 5-points-per-call
    micro-bench and FractalRendererCpp's 1-point-per-call one are on the same
    footing (see apply_pixel_kernel_throughput)."""
    rows = [r for r in records if r["workload"]["measurement"] == "pixel_kernel" and r.get("derived")]
    if not rows:
        return
    rows.sort(key=lambda r: r["derived"]["mpix_s"])
    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(rows) + 1.5)))
    labels = [f"{r['project']}: {r['impl']}" for r in rows]
    values = [r["derived"]["mpix_s"] for r in rows]
    colors = [PROJECT_COLOR.get(r["project"], PROJECT_COLOR_DEFAULT) for r in rows]
    bars = ax.barh(labels, values, color=colors)
    _hbar_value_labels(ax, bars, fmt="{:.2f}")
    ax.set_xlabel("Normalized M-evaluations/s")
    ax.set_title(f"{fractal_label.title()} — single-pixel kernel cost, by project")
    _direction_badge(ax, "higher", "M-evals/s")
    projects_present = sorted({r["project"] for r in rows})
    handles = [plt.Rectangle((0, 0), 1, 1, color=PROJECT_COLOR.get(p, PROJECT_COLOR_DEFAULT)) for p in projects_present]
    # Anchored outside the axes (not "upper right") so the legend box never
    # covers the longest bar's value label, regardless of how many projects
    # are present.
    ax.legend(handles, projects_present, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.text(0.01, 0.01,
             "Normalized for points-per-call (ruster: 5 sample points/call, FractalRendererCpp: 1) — raw mean_ms isn't comparable, this rate is.",
             fontsize=7, color="#4a5568")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    pdf.savefig(fig)
    plt.close(fig)


def _chart_perturbation_render(records: list[dict], plt, pdf, fractal_label: str) -> None:
    """Page 7 — perturbation-theory render cost vs zoom depth, one subplot per
    backend, one line per algorithm (naive escape-time vs perturbation vs
    perturbation+series-approximation). Naive should get relatively slower as
    zoom deepens (more pixels need full-precision fallback); perturbation (and
    especially +SA) should hold up much better."""
    backends = [("cpu-scalar", "CPU"), ("gpu-wgpu", "GPU (wgpu)"), ("gpu-cuda", "GPU (CUDA)")]
    algo_style = {
        "naive": ("naive (full escape-time)", "#718096", "o"),
        "perturbation": ("perturbation", "#3182ce", "s"),
        "perturbation_sa": ("perturbation + series-approx", "#c53030", "^"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    any_plotted = False
    for ax, (fam, label) in zip(axes, backends):
        for algo, (algo_label, color, marker) in algo_style.items():
            rows = [r for r in records if r["workload"]["measurement"] == "perturbation"
                    and r["workload"]["algorithm"] == algo and r["backend"]["family"] == fam and r.get("derived")]
            by_zoom = {r["workload"]["param"]: r for r in rows if r["workload"].get("param")}
            xs, ys = [], []
            for z, zl in zip(ZOOM_ORDER, ZOOM_LABELS):
                rec = by_zoom.get(f"zoom={z}")
                if rec:
                    xs.append(zl)
                    ys.append(rec["derived"]["mpix_s"])
            if xs:
                ax.plot(xs, ys, marker=marker, color=color, label=algo_label)
                any_plotted = True
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Zoom level")
    if not any_plotted:
        plt.close(fig)
        return
    axes[0].set_ylabel("Mpix/s")
    axes[0].legend(fontsize=7.5, loc="upper right")
    _direction_badge(axes[-1], "higher", "Mpix/s")
    fig.suptitle(f"{fractal_label.title()} — perturbation-theory render cost vs zoom depth, by backend")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _chart_perturbation_internals(records: list[dict], plt, pdf, fractal_label: str) -> None:
    """Page 8 — cost of the two perturbation-theory precompute stages:
    reference-orbit computation (f64 vs f128 — paid once per frame) and
    series-approximation coefficients (paid once per frame, cost should be
    roughly zoom-independent, unlike the render itself)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    any_plotted = False

    ax = axes[0]
    orbit = [r for r in records if r["workload"]["measurement"] == "reference_orbit" and (r["stats"] or {}).get("mean_ms") is not None]
    by_precision: dict[str, list[float]] = {}
    for r in orbit:
        prec = "f128" if "f128" in r["impl"] else "f64" if "f64" in r["impl"] else None
        if prec:
            by_precision.setdefault(prec, []).append(r["stats"]["mean_ms"])
    if by_precision:
        labels = [p for p in ("f64", "f128") if p in by_precision]
        values = [sum(by_precision[p]) / len(by_precision[p]) for p in labels]
        bars = ax.bar(labels, values, color=["#3182ce", "#c53030"][:len(labels)])
        _bar_value_labels(ax, bars, fmt="{:.4f}")
        any_plotted = True
    ax.set_ylabel("Mean ms/call (avg across zoom levels)")
    ax.set_title("Reference-orbit cost: f64 vs f128")
    _direction_badge(ax, "lower", "ms")

    ax = axes[1]
    sa = [r for r in records if r["workload"]["measurement"] == "series_approx" and (r["stats"] or {}).get("mean_ms") is not None]
    by_zoom = {r["impl"].rsplit("-", 1)[-1]: r for r in sa}
    xs, ys = [], []
    for z, zl in zip(ZOOM_ORDER, ZOOM_LABELS):
        rec = by_zoom.get(z)
        if rec:
            xs.append(zl)
            ys.append(rec["stats"]["mean_ms"])
    if xs:
        ax.plot(xs, ys, marker="o", color="#3182ce")
        any_plotted = True
    ax.set_xlabel("Zoom level")
    ax.set_ylabel("Mean ms/call")
    ax.set_title("Series-approximation coefficient cost vs zoom")
    _direction_badge(ax, "lower", "ms")

    if not any_plotted:
        plt.close(fig)
        return
    fig.suptitle(f"{fractal_label.title()} — perturbation-theory internals (precompute stage cost)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _chart_fair_comparison(records: list[dict], plt, pdf, fractal_label: str) -> None:
    """Page 9 — every baseline-common group (Table A) with more than one
    project in it: the strictly fair, same-params, cross-project comparison.
    Colored by project, small multiples since pixel_kernel/thread_scaling
    live on very different Mpix/s scales."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        if r["comparability"]["class"] == "baseline-common" and r["comparability"]["baseline_key"] and r.get("derived"):
            groups.setdefault(r["comparability"]["baseline_key"], []).append(r)
    groups = {k: v for k, v in groups.items() if len({r["project"] for r in v}) > 1}
    if not groups:
        return

    def sort_key(k: str):
        m = re.search(r"threads(\d+)$", k)
        return (0, int(m.group(1))) if m else (-1, k)
    keys = sorted(groups, key=sort_key)

    # A single row of panels becomes unreadably wide once enough baseline
    # groups pile up (pixel_kernel + render x 3 resolutions + thread_scaling
    # x 5 thread counts + pipeline, x however many projects share a key) —
    # wrap into a grid instead, capped at 4 panels per row.
    ncols = min(len(keys), 4)
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 5 * nrows), squeeze=False)
    flat_axes = axes.flatten()
    for ax, key in zip(flat_axes, keys):
        rows = sorted(groups[key], key=lambda r: r["derived"]["mpix_s"], reverse=True)
        labels = [r["project"] for r in rows]
        values = [r["derived"]["mpix_s"] for r in rows]
        colors = [PROJECT_COLOR.get(r["project"], PROJECT_COLOR_DEFAULT) for r in rows]
        bars = ax.bar(labels, values, color=colors)
        fmt = "{:.2f}" if max(values, default=0) < 10 else "{:.0f}"
        _bar_value_labels(ax, bars, fmt=fmt)
        segs = key.split("@")
        short_title = segs[1] + (f" ({segs[-1]})" if segs[-1].startswith("threads") else "")
        ax.set_title(short_title, fontsize=9)
        ax.set_ylabel("Mpix/s")
        ax.tick_params(axis="x", labelsize=7, rotation=20)
    for ax in flat_axes[len(keys):]:
        ax.axis("off")
    _direction_badge(flat_axes[len(keys) - 1], "higher", "Mpix/s")
    fig.suptitle(f"{fractal_label.title()} — fair cross-project comparison (Table A: same fractal/resolution/max_iter/precision/algorithm)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(RUSTER_ROOT / "bench_results"))
    ap.add_argument("--fractal", default="mandelbrot",
                    help="Scope all outputs to one fractal (default: mandelbrot — the one every project renders).")
    ap.add_argument("--all-fractals", action="store_true", help="Disable fractal scoping, include everything.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fractal = None if args.all_fractals else args.fractal
    records = filter_by_fractal(collect_all(), fractal)
    label = fractal or "all-fractals"

    combined_path = out_dir / "combined.json"
    combined_path.write_text(json.dumps(records, indent=2))

    report_path = out_dir / "comparison_report.md"
    report_path.write_text(build_report(records))

    csv_path = out_dir / "comparison.csv"
    write_csv(records, csv_path)

    pdf_path = out_dir / "comparison_charts.pdf"
    pdf_ok = build_pdf(records, pdf_path, label)

    print(f"{len(records)} records ({label}) -> {combined_path}")
    print(f"report -> {report_path}")
    print(f"csv -> {csv_path}")
    print(f"pdf -> {pdf_path}" if pdf_ok else "pdf -> skipped (matplotlib unavailable)")


if __name__ == "__main__":
    main()
