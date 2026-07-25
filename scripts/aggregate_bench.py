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
    elif fam == "hybrid":
        # All three hybrid groups (cpu+wgpu, cpu+cuda, heterogeneous) call
        # bench_function(fractal.name()) / BenchmarkId::new(fractal.name(), ...) —
        # function_id is always the fractal, never a compute variant.
        fractal = function_id
        variant_label = measurement      # "cpu+wgpu" | "cpu+cuda" | "heterogeneous"
        if measurement == "heterogeneous" and value_str:
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
        "backend": {"family": backend_family, "detail": function_id, "threads": threads},
        "workload": {
            "fractal": fractal_norm(fractal), "measurement": measurement, "algorithm": algorithm,
            "resolution": resolution, "max_iter": MAX_ITER_DEFAULT, "precision": "f64", "param": param,
        },
        "comparability": comparability,
    }


# ── Criterion adapter (Fractals-rs, github.com/Maxime-Cllt/Fractals-rs) ────────
#
# Verified against benches/application_bench.rs:
#   "full_frame_800x600"        fixed 800x600, max_iter=300, no throughput.
#                                function_id = "<fractal>_fast" (f32 SIMD) or
#                                "<fractal>_high" (f64 SIMD) — there is NO
#                                scalar/non-SIMD path in this project at all.
#   "full_frame/<Fractal>"      multi-res sweep, max_iter=300, throughput set.
#                                function_id = "fast" | "high", value_str = resolution.
#   "thread_scaling/Mandelbrot_1080p_fast"  max_iter=1000, f32 SIMD only.
#   "scalar_kernels" / "simd_kernels"       single-pixel kernel calls.
#
# max_iter is baked in per-group from source, not read from data — Criterion's
# own output doesn't carry it. If Fractals-rs's bench consts change, update here.
#
# Nothing here reaches class == "baseline-common" against ruster today: every
# Fractals-rs full-frame path is SIMD (f32 "fast" or f64 "high") at max_iter=300,
# while ruster's naive-cpu baseline is scalar f64 at max_iter=1000 — different
# algorithm class AND different iteration count, so a merge would be a false
# equivalence. They're kept as project-unique with an explicit note instead.
# Re-running either project's bench to match (iters, precision, algorithm) would
# make an exact pair line up automatically — the join key is purely mechanical.

FRACTALS_RS_FULLFRAME_800_MAXITER = 300
FRACTALS_RS_MULTIRES_MAXITER = 300
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
        note = (f"max_iter={max_iter} vs ruster's 1000, {precision} SIMD vs ruster's scalar baseline — "
                "not comparable as-is; would need a matching-params rerun on one side.")
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
            backend={"family": backend_family, "detail": backend_raw, "threads": threads},
            workload={
                "fractal": fractal_norm(s.get("fractal")), "measurement": measurement, "algorithm": "naive",
                "resolution": resolution, "max_iter": s.get("max_iter"), "precision": "f64",
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
#   BM_FullFrame/<fractal>/<w>/<h>            max_iter=300,  8 threads (hardcoded)
#   BM_ThreadScaling_Mandelbrot_1080p/<n>      max_iter=1000, 1920x1080, n threads
#   BM_FullPipeline_Mandelbrot_1080p           max_iter=300,  1920x1080, 8 threads
#
# BM_Evaluate and BM_ThreadScaling land in Table A: same fractal, same max_iter,
# same f64-scalar-naive algorithm as ruster's cpu/pixel_kernel and
# cpu/thread_scaling. BM_FullFrame/BM_FullPipeline stay project-unique — their
# max_iter=300 was chosen to mirror Fractals-rs's full-frame benches, not
# ruster's 1000, so merging them into ruster's render baseline would be a false
# equivalence for the same reason Fractals-rs's full-frame numbers are excluded.

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
        max_iter = 300
        threads = 8
        param = "threads=8"
        comparability = {"class": "project-unique", "baseline_key": None,
                          "note": "max_iter=300 here (mirrors Fractals-rs's full-frame benches) vs ruster's cpu/render baseline at max_iter=1000 — not directly comparable."}
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
        max_iter = 300
        threads = 8
        param = "threads=8"
        comparability = {"class": "project-unique", "baseline_key": None,
                          "note": "max_iter=300 vs ruster's cpu/pipeline baseline at max_iter=1000 — not directly comparable."}

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


GB_CLASSIFIERS = {
    "fractalrenderercpp": classify_fractalrenderercpp,
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
  lands in Table A twice: its `BM_Evaluate` pixel-kernel and `BM_ThreadScaling_Mandelbrot_1080p`
  benches are confirmed f64 scalar naive at max_iter=1000, matching ruster's baseline exactly
  (see classify_fractalrenderercpp()) — the one real difference being threading strategy
  (raw std::thread spawn/join per call vs a persistent rayon pool), which the comparison
  reports rather than hides. Its `BM_FullFrame`/`BM_FullPipeline` groups stay project-unique:
  they run at max_iter=300 (mirroring Fractals-rs's own full-frame benches), not ruster's 1000.
- Fractals-rs has real Criterion data (Table B) but none of it lands in Table A: its
  full-frame benches run SIMD-only (f32 "fast" / f64 "high") at max_iter=300, while
  ruster's baseline is scalar f64 at max_iter=1000 — different algorithm class and
  iteration count, so treating them as equal would be a false equivalence, not a fair
  comparison. Re-run one side to match the other's (max_iter, precision, algorithm) and
  the join is automatic — see the notes in classify_fractals_rs().
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
    "project", "impl", "language", "backend_family", "backend_detail", "threads",
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
        "backend_family": b["family"], "backend_detail": b.get("detail"), "threads": b.get("threads"),
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

PROJECT_COLOR = {"ruster": "#2b6cb0", "fractals-rs": "#b83280", "fractalrenderercpp": "#c05621"}
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
    below) is visible rather than smoothed away — PLUS Fractals-rs and
    FractalRendererCpp's own render numbers on the same axes, colored by
    project instead of backend family since they're a different axis (each
    has only one or two variants, not ruster's full backend spread).

    These cross-project bars are NOT apples-to-apples with ruster's (see
    Table A): Fractals-rs/FractalRendererCpp run at max_iter=300, ruster at
    1000 — fewer iterations per pixel means higher Mpix/s independent of any
    real speed difference. That's spelled out in both the bar labels and the
    caption so it reads as "everything measured", not "ruster is slower"."""
    representative = {
        "cpu-scalar": "ruster-cpu-rayon", "cpu-simd": "ruster-simd-f32x8",
        "cpu-tiled": "ruster-cpu-hilbert", "gpu-cuda": "ruster-cuda-cuda",
        "gpu-wgpu": "ruster-wgpu-gpu",
    }
    other_series = [
        ("fractals-rs-full_frame-fast", "Fractals-rs (f32, iter=300)", PROJECT_COLOR["fractals-rs"], "//"),
        ("fractals-rs-full_frame-high", "Fractals-rs (f64, iter=300)", PROJECT_COLOR["fractals-rs"], ".."),
        ("fractalrenderercpp-fullframe-mandelbrot", "FractalRendererCpp (f64, iter=300)", PROJECT_COLOR["fractalrenderercpp"], None),
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
    if not families and not other_present:
        return

    bars_spec = [(representative[f], BACKEND_COLOR[f], None) for f in families] + \
                [(impl, color, hatch) for impl, _, color, hatch in other_present]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    n = len(bars_spec)
    width = 0.8 / n
    x = list(range(len(resolutions)))
    for i, (impl, color, hatch) in enumerate(bars_spec):
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
                 "(ruster bars ordered CPU-scalar → CPU-SIMD → CPU-tiled → GPU-CUDA → GPU-wgpu, the expected hierarchy)")
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.32)  # headroom so the legend and callout box below don't overlap the bars
    handles = [plt.Rectangle((0, 0), 1, 1, color=BACKEND_COLOR[f]) for f in families]
    labels = [BACKEND_LABEL[f] for f in families]
    for impl, label, color, hatch in other_present:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color, hatch=hatch, ec="white" if hatch else "none"))
        labels.append(label)
    ax.legend(handles, labels, loc="upper left", fontsize=7.5, framealpha=0.9)
    _direction_badge(ax, "higher", "Mpix/s")
    note = ("Note: GPU (CUDA) trails even CPU-scalar here — kernel-launch/PCIe-copy overhead dominates at\n"
            "zoom=1 on this laptop-class RTX 3050 (see results/summary.md's Stage 3 discussion); GPU (wgpu)\n"
            "doesn't show the same gap because its compute-shader dispatch path has different overhead.\n"
            "Fractals-rs/FractalRendererCpp run at max_iter=300 vs ruster's 1000 — higher Mpix/s here isn't\n"
            "a real speed advantage, it's fewer iterations/pixel. See Table A for the apples-to-apples pairs.")
    ax.text(0.99, 0.98, note, transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            color="#742a2a", bbox=dict(boxstyle="round,pad=0.4", fc="#fff5f5", ec="#feb2b2"))
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
    Reminder in the caption: max_iter differs from ruster's baseline (300 vs
    1000 — see Table A notes), so this is these projects' own internal
    comparison, not a cross-project speed claim."""
    resolutions = [[800, 600], [1920, 1080], [3840, 2160]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

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
    ax.set_title("Fractals-rs — Mandelbrot render throughput\n(max_iter=300 — this project's own default, not ruster's 1000)")
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
    ax.set_title("FractalRendererCpp — Mandelbrot render throughput\n(f64 scalar naive, max_iter=300 — the only backend this project has)")
    _direction_badge(ax, "higher", "Mpix/s")

    fig.suptitle(f"{fractal_label.title()} — render throughput, other projects (their own internal comparison, not vs ruster — see Table A)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _chart_hybrid_vs_solo(records: list[dict], plt, pdf, fractal_label: str) -> None:
    """Page 2 — solo CPU/GPU backends next to the two hybrid (CPU+GPU
    concurrent) combos, all at 1920x1080 (the only resolution hybrid
    benchmarks run at). Expectation: hybrid should approach the sum of its
    two concurrent halves' throughput."""
    wanted = [
        ("cpu-scalar", "ruster-cpu-rayon", "CPU\nscalar"),
        ("cpu-simd", "ruster-simd-f32x8", "CPU\nSIMD f32x8"),
        ("gpu-cuda", "ruster-cuda-cuda", "GPU\nCUDA (solo)"),
        ("gpu-wgpu", "ruster-wgpu-gpu", "GPU\nwgpu (solo)"),
        ("hybrid", "ruster-hybrid-cpu+cuda", "Hybrid\nCPU+CUDA"),
        ("hybrid", "ruster-hybrid-cpu+wgpu", "Hybrid\nCPU+wgpu"),
    ]
    by_impl = {r["impl"]: r for r in records if r.get("derived")}
    labels, values, colors, families_present = [], [], [], []
    for fam, impl, label in wanted:
        rec = by_impl.get(impl)
        if not rec:
            continue
        labels.append(label)
        values.append(rec["derived"]["mpix_s"])
        colors.append(BACKEND_COLOR[fam])
        if fam not in families_present:
            families_present.append(fam)
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ymax = max(values)
    ax.set_ylim(0, ymax * 1.3)
    bars = ax.bar(labels, values, color=colors)
    _bar_value_labels(ax, bars)
    ax.set_ylabel("Mpix/s")
    ax.set_title(f"{fractal_label.title()} @ 1920×1080 — solo backends vs hybrid CPU+GPU")
    _direction_badge(ax, "higher", "Mpix/s")
    _backend_legend(ax, plt, sorted(families_present, key=BACKEND_ORDER.index))
    wgpu_solo = by_impl.get("ruster-wgpu-gpu")
    hybrid_wgpu = by_impl.get("ruster-hybrid-cpu+wgpu")
    if wgpu_solo and hybrid_wgpu and hybrid_wgpu["derived"]["mpix_s"] < wgpu_solo["derived"]["mpix_s"]:
        note = ("Note: hybrid CPU+GPU trails solo GPU here — this is a naive 50/50 frame split (CPU renders\n"
                "the top half, GPU the bottom half, concurrently); since GPU alone is ~5x faster than CPU,\n"
                "the CPU half becomes the bottleneck and the GPU half sits idle waiting for it. A load-aware\n"
                "split (giving CPU a smaller share) is exactly what the heterogeneous scheduler targets — see\n"
                "results/summary.md's Stage 3 discussion.")
        ax.text(0.5, 0.98, note, transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
                color="#742a2a", bbox=dict(boxstyle="round,pad=0.4", fc="#fff5f5", ec="#feb2b2"))
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
    """Page 4 — Mpix/s vs thread count. Left panel: ruster overlaid with
    FractalRendererCpp (both confirmed f64 scalar naive, max_iter=1000,
    1920x1080 — the one real difference is threading strategy: rayon pool vs
    raw std::thread spawn/join per call). Fractals-rs gets its own separate
    right panel rather than being overlaid on the left: it runs f32 SIMD, not
    scalar, so sharing an axis with the other two would visually imply a fair
    comparison the precision mismatch doesn't support."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    series = [("ruster-cpu-thread_scaling", "ruster (rayon pool, f64 scalar)", PROJECT_COLOR["ruster"]),
              ("fractalrenderercpp-threadscaling", "FractalRendererCpp (std::thread/call, f64 scalar)", PROJECT_COLOR["fractalrenderercpp"])]
    plotted = False
    for impl, label, color in series:
        rows = [r for r in records if r["impl"] == impl and r["backend"].get("threads") and r.get("derived")]
        rows.sort(key=lambda r: r["backend"]["threads"])
        if not rows:
            continue
        xs = [r["backend"]["threads"] for r in rows]
        ys = [r["derived"]["mpix_s"] for r in rows]
        ax.plot(xs, ys, marker="o", color=color, label=label)
        plotted = True
    ax.set_xlabel("Threads")
    ax.set_ylabel("Mpix/s (log scale)")
    ax.set_yscale("log")
    ax.set_title("ruster vs FractalRendererCpp\n(both f64 scalar naive — a fair pair)")
    ax.set_xticks([1, 2, 4, 8, 16])
    if plotted:
        ax.legend(fontsize=8)
    _direction_badge(ax, "higher", "Mpix/s")

    ax = axes[1]
    rows = [r for r in records if r["impl"] == "fractals-rs-thread_scaling-threads" and r["backend"].get("threads") and r.get("derived")]
    rows.sort(key=lambda r: r["backend"]["threads"])
    if rows:
        xs = [r["backend"]["threads"] for r in rows]
        ys = [r["derived"]["mpix_s"] for r in rows]
        ax.plot(xs, ys, marker="o", color=PROJECT_COLOR["fractals-rs"], label="Fractals-rs (rayon pool, f32 SIMD)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No thread_scaling data yet —\nrun `cargo bench` in other-projects/Fractals-rs",
                transform=ax.transAxes, ha="center", va="center", fontsize=9, color="#718096")
    ax.set_xlabel("Threads")
    ax.set_ylabel("Mpix/s (log scale)")
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_title("Fractals-rs, own panel\n(f32 SIMD — not scalar, kept separate on purpose)")
    _direction_badge(ax, "higher", "Mpix/s")

    if not plotted and not rows:
        plt.close(fig)
        return
    fig.suptitle(f"{fractal_label.title()} @ 1920×1080 — CPU thread scaling, max_iter=1000")
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
    for impl, label, color in [("ruster-simd-f32x8", "f32x8 SIMD", "#3182ce"),
                                ("ruster-simd-f32x8_ilp", "f32x8 + ILP (2× interleaved)", "#805ad5")]:
        rows = [r for r in records if r["impl"] == impl and r["workload"].get("resolution") and r.get("derived")]
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
    ax.legend(handles, projects_present, loc="upper right", fontsize=8)
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
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 5))
    if len(keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, keys):
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
    _direction_badge(axes[-1], "higher", "Mpix/s")
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
