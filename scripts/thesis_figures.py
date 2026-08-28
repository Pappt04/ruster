#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8", "numpy>=1.26"]
# ///
"""Publication-quality figures for the thesis, built from the criterion archives.

Reads `bench_results/criterion_*/**/new/estimates.json` — no re-measurement — and
emits one multi-page PDF plus an individual PDF/PNG per figure, sized for a thesis
page.

Every figure is meant to carry an argument, not just display numbers:

  1  scheduler across the three precision regimes   ← the headline result
  2  backend comparison at 1080p
  3  where a GPU frame actually goes
  4  end-to-end pipeline and the Amdahl ceiling
  5  resolution scaling
  6  thread scaling and parallel efficiency
  7  SIMD progression
  8  floating-point precision, f32 vs f64
  9  perturbation vs scalar across zoom
 10  cross-project comparison
 11  the fp64 bulb-check ablation

Colours come from a validated categorical palette (checked with the dataviz
validator: lightness band, chroma floor, CVD separation, normal-vision floor all
PASS). Series identity is fixed — a backend keeps its colour in every figure —
and every figure is direct-labelled, which is also what discharges the
validator's contrast warning for the lighter slots.

Usage:
    ./scripts/thesis_figures.py [--out results/figures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FULL = REPO / "bench_results" / "criterion_latest"
SCHED = REPO / "bench_results" / "criterion_20260808_210551_fc5ca0d_sched_improved"

# ── palette (validated; see scripts/validate note in the docstring) ───────────
C_CUDA = "#2a78d6"   # slot 1 blue    — CUDA / discrete GPU
C_CPU = "#eb6834"    # slot 2 orange  — CPU
C_HYB = "#1baf7a"    # slot 3 aqua    — hybrid / scheduler
C_WGPU = "#eda100"   # slot 4 yellow  — wgpu
C_ALT = "#e87ba4"    # slot 5 magenta — competitor projects
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a85"
GRID = "#d8d8d4"

# Regime shading — neutral greys, deliberately not categorical hues, so the
# bands never compete with the series for identity.
BAND_LOSE = "#f0efec"
BAND_WIN = "#e4f3ec"

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": ":",
    "xtick.color": INK2,
    "ytick.color": INK2,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
})

F32_THRESHOLD = 1e6
F128_THRESHOLD = 1e12


# ── data access ──────────────────────────────────────────────────────────────
def est(base: Path, path: str):
    """(mean_ms, ci_low_ms, ci_high_ms) or None."""
    p = base / path / "new" / "estimates.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())["mean"]
    ci = m["confidence_interval"]
    return (m["point_estimate"] / 1e6, ci["lower_bound"] / 1e6, ci["upper_bound"] / 1e6)


def ms(base: Path, path: str):
    e = est(base, path)
    return e[0] if e else None


def finish(fig, text, width=112, top=1.0):
    """Attach a wrapped caption and lay the figure out around it.

    Caption length varies from two lines to five, so a fixed `tight_layout`
    rect either wastes space or lets the text collide with the x-axis labels.
    Reserving room proportional to the wrapped line count fixes both. Also
    hard-wraps rather than using matplotlib's `wrap=True`, which wraps at the
    figure boundary and produces full-bleed text.
    """
    import textwrap
    para = " ".join(line.strip() for line in text.strip().splitlines())
    body = textwrap.fill(para, width)
    n = body.count("\n") + 1
    # Reserve from real text metrics rather than a guess: one line occupies
    # (pt * linespacing / 72) inches, converted to a figure fraction.
    fs, spacing, pad = 7.6, 1.5, 0.022
    line_frac = fs * spacing / 72.0 / fig.get_figheight()
    fig.tight_layout(rect=(0, pad + n * line_frac, 1, top))
    fig.text(0.5, 0.006, body, ha="center", va="bottom",
             fontsize=fs, color=INK2, style="italic", linespacing=spacing)


def label_bars(ax, bars, values, fmt="{:.2f}", extra=None, rotation=0):
    for i, (b, v) in enumerate(zip(bars, values)):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        t = fmt.format(v)
        if extra:
            t += "\n" + extra[i]
        ax.annotate(t, (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom",
                    fontsize=7.5, color=INK, rotation=rotation)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — the headline: scheduler across the three precision regimes
# ═════════════════════════════════════════════════════════════════════════════
def fig_scheduler_regimes():
    zooms = [("zoom_1e4", 1e4), ("zoom_1e5", 1e5), ("zoom_1e6", 1e6),
             ("zoom_1e7", 1e7), ("zoom_1e9", 1e9), ("zoom_1e12", 1e12),
             ("zoom_1e15", 1e15)]
    rows = []
    for key, z in zooms:
        g = est(SCHED, f"hybrid_heterogeneous_deep/cuda/{key}")
        c = est(SCHED, f"hybrid_heterogeneous_deep/cpu/{key}")
        h = est(SCHED, f"hybrid_heterogeneous_deep/hybrid/{key}")
        if not (g and c and h):
            continue
        rows.append((z, g, c, h))
    if not rows:
        return None

    x = [r[0] for r in rows]
    gpu, cpu, hyb = [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows]
    best = [min(a[0], b[0]) for a, b in zip(gpu, cpu)]
    speedup = [b / h[0] for b, h in zip(best, hyb)]

    fig, (axh, ax1, ax2) = plt.subplots(
        3, 1, figsize=(8.0, 8.4), sharex=True,
        gridspec_kw={"height_ratios": [0.42, 2.0, 1.15], "hspace": 0.10})

    # Extra left margin so the leftmost data label has room without
    # being nudged into its neighbour.
    xlo, xhi = x[0] * 0.30, x[-1] * 2.2

    # Regime bands on every panel, drawn under everything.
    for ax in (axh, ax1, ax2):
        ax.axvspan(xlo, F32_THRESHOLD, color=BAND_LOSE, zorder=0)
        ax.axvspan(F32_THRESHOLD, F128_THRESHOLD, color=BAND_WIN, zorder=0)
        ax.axvspan(F128_THRESHOLD, xhi, color=BAND_WIN, alpha=0.55, zorder=0)
        for b in (F32_THRESHOLD, F128_THRESHOLD):
            ax.axvline(b, color=INK2, lw=1.0, ls=(0, (5, 3)), zorder=1)
        ax.set_xscale("log")
        ax.set_xlim(xlo, xhi)

    # ── header strip: the three regimes, with nothing to collide with ────────
    axh.set_ylim(0, 1)
    axh.set_yticks([])
    axh.grid(False)
    for sp in axh.spines.values():
        sp.set_visible(False)
    # Text is sized to the band it sits in — the first band is only ~2 decades
    # wide on this axis, so it gets the shortest label.
    for lo, hi, title, why in [
        (xlo, F32_THRESHOLD, "BELOW 1e6", "scheduler loses\nGPU f32 ≈ 20× CPU"),
        (F32_THRESHOLD, F128_THRESHOLD, "1e6 – 1e12",
         "SCHEDULER WINS\nGPU falls back to f64 at 1/64 rate — CPU becomes competitive"),
        (F128_THRESHOLD, xhi, "BEYOND 1e12", "still wins\nf64 loses the grid"),
    ]:
        mid = np.sqrt(lo * hi)
        axh.text(mid, 0.80, title, ha="center", va="center", fontsize=9.0,
                 fontweight="bold", color=INK)
        axh.text(mid, 0.30, why, ha="center", va="center", fontsize=7.2, color=INK2,
                 linespacing=1.5)

    def series(ax, ys, color, label, marker):
        v = [y[0] for y in ys]
        lo = [y[0] - y[1] for y in ys]
        hi = [y[2] - y[0] for y in ys]
        ax.errorbar(x, v, yerr=[lo, hi], color=color, marker=marker, markersize=6,
                    linewidth=2.0, capsize=2.5, elinewidth=1.0, label=label,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=4)

    series(ax1, gpu, C_CUDA, "CUDA alone (RTX 3050)", "o")
    series(ax1, cpu, C_CPU, "CPU alone (16 threads)", "s")
    series(ax1, hyb, C_HYB, "Heterogeneous scheduler", "D")
    ax1.set_yscale("log")
    ax1.set_ylabel("frame time (ms, log)")
    # Legend sits in the empty low-time region at the right of the log axis.
    ax1.legend(loc="lower right", ncol=1, framealpha=0.95, facecolor="white",
               edgecolor=GRID, frameon=True)

    # One direct label: the decisive point. The rest live in panel 2.
    i6 = x.index(1e6) if 1e6 in x else None
    if i6 is not None:
        ax1.annotate(f"{best[i6] / hyb[i6][0]:.2f}× faster than\neither backend alone",
                     (x[i6], hyb[i6][0]), textcoords="offset points", xytext=(14, -34),
                     fontsize=8.4, fontweight="bold", color=INK,
                     arrowprops=dict(arrowstyle="->", color=INK2, lw=1.0))

    # ── panel 3: speedup vs whichever solo backend is faster ─────────────────
    ax2.axhline(1.0, color=INK2, lw=1.1, zorder=3)
    ax2.plot(x, speedup, color=C_HYB, marker="D", markersize=6, linewidth=2.0,
             markeredgecolor="white", markeredgewidth=0.8, zorder=4)
    ax2.fill_between(x, 1.0, speedup, where=[s >= 1 for s in speedup],
                     color=C_HYB, alpha=0.20, interpolate=True, zorder=2)
    ax2.fill_between(x, 1.0, speedup, where=[s < 1 for s in speedup],
                     color=C_CPU, alpha=0.18, interpolate=True, zorder=2)
    ax2.set_xlabel("zoom level")
    ax2.set_ylabel("speedup vs best\nsolo backend")
    ax2.set_ylim(0.62, 1.95)
    for xi, s in zip(x, speedup):
        # Nudge only the 1e12 label, which would otherwise sit on the divider.
        dx = 16 if abs(xi - F128_THRESHOLD) < 1 else 0
        ax2.annotate(f"{s:.2f}×", (xi, s), textcoords="offset points",
                     xytext=(dx, 11 if s >= 1 else -17), ha="center", fontsize=7.9,
                     color=INK, fontweight="bold" if s > 1.5 else "normal")

    fig.suptitle("The adaptive scheduler wins exactly where the GPU loses its f32 fast path",
                 fontsize=12.5, fontweight="bold", y=0.975)
    finish(fig, "1920×1080 Mandelbrot, max_iter=1000. All three arms measured back-to-back in one criterion process,\n"
            "so session-to-session thermal drift cancels. Error bars are criterion's 95% confidence intervals.\n"
            "Wins past 1e7 are real but marginal (1.02–1.16×): the load balancer still leaves ~45% unclaimed there.", top=0.96)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — backend comparison at 1080p
# ═════════════════════════════════════════════════════════════════════════════
def fig_backends():
    items = [
        ("CUDA", ms(FULL, "cuda_render_Mandelbrot/cuda/1920×1080"), C_CUDA, "f32"),
        ("Scheduler\n(zoom 1e4)", ms(SCHED, "hybrid_heterogeneous/Mandelbrot/zoom_1e4"), C_HYB, "f32"),
        ("wgpu", ms(FULL, "wgpu_render_Mandelbrot/gpu/1920×1080"), C_WGPU, "f32"),
        ("SIMD\nf32x8+ILP", ms(FULL, "simd_render_ilp_Mandelbrot_1080p/f32x8_ilp/1920×1080"), C_CPU, "f32"),
        ("SIMD\nf64x4", ms(FULL, "simd_render_Mandelbrot/f64x4/1920×1080"), C_CPU, "f64"),
        ("CPU scalar\n(baseline)", ms(FULL, "cpu_render_Mandelbrot/rayon/1920×1080"), C_CPU, "f64"),
    ]
    items = [i for i in items if i[1]]
    if not items:
        return None
    items.sort(key=lambda t: t[1])
    names = [i[0] for i in items]
    vals = [i[1] for i in items]
    cols = [i[2] for i in items]
    precs = [i[3] for i in items]
    base = max(vals)

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    bars = ax.bar(range(len(vals)), vals, color=cols, edgecolor="white", linewidth=1.6,
                  width=0.68)
    for b, p in zip(bars, precs):
        if p == "f32":
            b.set_hatch("///")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8.4)
    ax.set_ylabel("frame time (ms)")
    ax.set_title("Backend comparison — 1920×1080 Mandelbrot, max_iter=1000")
    label_bars(ax, bars, vals, "{:.2f} ms",
               extra=[f"{base / v:.2f}× vs baseline" for v in vals])
    ax.set_ylim(0, base * 1.22)
    ax.legend(handles=[
        Patch(facecolor=C_CUDA, label="CUDA (RTX 3050)"),
        Patch(facecolor=C_WGPU, label="wgpu (RTX 3050)"),
        Patch(facecolor=C_HYB, label="CPU + GPU scheduler"),
        Patch(facecolor=C_CPU, label="CPU"),
        Patch(facecolor="white", edgecolor=INK2, hatch="///", label="f32 (hatched)"),
    ], loc="upper left", ncol=2)
    finish(fig, "Hatched bars are f32, solid are f64 — not a like-for-like comparison across the two. The "
            "precision-controlled pairing is CUDA (1.61 ms) against the CPU's own best f32 path (3.79 ms) = 2.36×; "
            "against the f64 baseline it reads 4.37×.", top=1.0)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — where a GPU frame actually goes
# ═════════════════════════════════════════════════════════════════════════════
def fig_gpu_breakdown():
    # From examples/gpu_probe.rs — median of 5 process launches, post-fix.
    data = [("CUDA", 0.28, 1.33, C_CUDA), ("wgpu", 0.29, 2.14, C_WGPU)]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = np.arange(len(data))
    kern = [d[1] for d in data]
    read = [d[2] for d in data]
    ax.bar(xs, kern, 0.5, color=[d[3] for d in data], edgecolor="white", linewidth=1.6)
    ax.bar(xs, read, 0.5, bottom=kern, color=[d[3] for d in data], alpha=0.32,
           edgecolor="white", linewidth=1.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([d[0] for d in data])
    ax.set_ylabel("frame time (ms)")
    ax.set_title("A GPU frame is mostly data movement, not computation")
    for i, d in enumerate(data):
        total = d[1] + d[2]
        ax.annotate(f"{d[1]:.2f} ms kernel", (i, d[1] / 2), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        ax.annotate(f"{d[2]:.2f} ms readback\n({d[2] / total * 100:.0f}% of frame)",
                    (i, d[1] + d[2] / 2), ha="center", va="center", fontsize=8, color=INK)
        ax.annotate(f"total {total:.2f} ms", (i, total), ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color=INK)
    ax.set_ylim(0, max(k + r for k, r in zip(kern, read)) * 1.2)
    # Colour identifies the backend (also on the x-axis); saturation identifies
    # the segment. Neutral legend swatches keep those two encodings distinct.
    ax.legend(handles=[Patch(facecolor=MUTED, label="iteration kernel"),
                       Patch(facecolor=MUTED, alpha=0.32, label="host readback")],
              loc="upper left")
    # No finish()/baked caption here on purpose — this figure is embedded in
    # the thesis with its own \caption in diplomski.tex, and the surrounding
    # prose already covers the PCIe/readback-share explanation finish()
    # used to print into the image itself.
    fig.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — end-to-end pipeline: the Amdahl ceiling
# ═════════════════════════════════════════════════════════════════════════════
def fig_pipeline():
    col = ms(FULL, "cpu_colorize/inferno/1920×1080")
    rows = [
        ("CPU", ms(FULL, "cpu_render_Mandelbrot/rayon/1920×1080"), C_CPU),
        ("wgpu", ms(FULL, "wgpu_render_Mandelbrot/gpu/1920×1080"), C_WGPU),
        ("CUDA", ms(FULL, "cuda_render_Mandelbrot/cuda/1920×1080"), C_CUDA),
    ]
    if col is None or any(r[1] is None for r in rows):
        return None
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    xs = np.arange(len(rows))
    rend = [r[1] for r in rows]
    ax.bar(xs, rend, 0.5, color=[r[2] for r in rows], edgecolor="white", linewidth=1.6,
           label="render (iteration)")
    ax.bar(xs, [col] * len(rows), 0.5, bottom=rend, color=MUTED, edgecolor="white",
           linewidth=1.6, label="colorize (CPU, backend-independent)")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("frame time (ms)")
    ax.set_title("The CPU colour stage caps every backend's end-to-end gain")
    slow = rend[0] + col
    for i, r in enumerate(rows):
        tot = r[1] + col
        ax.annotate(f"{r[1]:.2f}", (i, r[1] / 2), ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")
        ax.annotate(f"{col:.2f}", (i, r[1] + col / 2), ha="center", va="center",
                    fontsize=8, color=INK)
        ax.annotate(f"{tot:.2f} ms\n{slow / tot:.2f}× end-to-end", (i, tot), ha="center",
                    va="bottom", fontsize=8.2, color=INK, fontweight="bold")
    ax.set_ylim(0, slow * 1.24)
    ax.legend(loc="upper right")
    finish(fig, "colorize() — histogram equalisation + palette LUT — is CPU-only and identical for every backend. "
            "It is 47% of the CUDA pipeline. Of a 4.97 ms CUDA pipeline roughly 0.3 ms is actual fractal iteration; "
            "moving this stage onto the GPU is worth 2.18×, more than any remaining kernel optimisation.", top=1.0)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — resolution scaling
# ═════════════════════════════════════════════════════════════════════════════
def fig_resolution():
    res = [("800×600", 800 * 600), ("1920×1080", 1920 * 1080), ("3840×2160", 3840 * 2160)]
    series = [
        ("CUDA", C_CUDA, "cuda_render_Mandelbrot/cuda/{}"),
        ("wgpu", C_WGPU, "wgpu_render_Mandelbrot/gpu/{}"),
        ("CPU scalar", C_CPU, "cpu_render_Mandelbrot/rayon/{}"),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for name, colr, tmpl in series:
        ys = []
        for label, px in res:
            v = ms(FULL, tmpl.format(label))
            ys.append(px / v / 1e3 if v else np.nan)
        ax.plot([r[0] for r in res], ys, marker="o", markersize=7, linewidth=2.0,
                color=colr, label=name, markeredgecolor="white", markeredgewidth=0.8)
        ax.annotate(f"{ys[-1]:.0f}", (2, ys[-1]), textcoords="offset points",
                    xytext=(8, 0), fontsize=8, color=INK, va="center")
    ax.set_ylabel("throughput (Mpix/s)")
    ax.set_xlabel("resolution")
    ax.set_title("Throughput is flat in resolution — the kernel is embarrassingly parallel")
    ax.legend()
    finish(fig, "Mandelbrot, max_iter=1000. A per-pixel kernel with no shared state should hold constant Mpix/s as the "
            "frame grows; it does. The GPU curves rise slightly because their fixed per-frame costs — dispatch and "
            "the readback — amortise over more pixels.", top=1.0)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — thread scaling and parallel efficiency
# ═════════════════════════════════════════════════════════════════════════════
def fig_threads():
    ns = [1, 2, 4, 8, 16]
    v = [ms(FULL, f"cpu_thread_scaling_Mandelbrot_1080p/threads/{n}") for n in ns]
    if any(x is None for x in v):
        return None
    sp = [v[0] / x for x in v]
    eff = [s / n * 100 for s, n in zip(sp, ns)]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.9))
    a1.plot(ns, sp, marker="o", markersize=7, linewidth=2.0, color=C_CPU,
            markeredgecolor="white", markeredgewidth=0.8, label="measured")
    a1.plot(ns, ns, ls=(0, (5, 3)), color=MUTED, linewidth=1.4, label="ideal (linear)")
    a1.set_xscale("log", base=2)
    a1.set_yscale("log", base=2)
    a1.set_xticks(ns); a1.set_xticklabels(ns)
    a1.set_yticks(ns); a1.set_yticklabels(ns)
    a1.set_xlabel("threads"); a1.set_ylabel("speedup")
    a1.set_title("Parallel speedup")
    a1.legend(loc="upper left")
    for n, s in zip(ns, sp):
        a1.annotate(f"{s:.2f}×", (n, s), textcoords="offset points", xytext=(6, -10),
                    fontsize=7.8, color=INK)

    bars = a2.bar([str(n) for n in ns], eff, color=C_CPU, edgecolor="white",
                  linewidth=1.6, width=0.62)
    a2.axvline(3.5, color=INK2, ls=(0, (5, 3)), lw=1.0)
    a2.annotate("8 physical cores →\nSMT beyond here", (3.55, 96), fontsize=7.6,
                color=INK2, va="top")
    a2.set_ylim(0, 116)
    a2.set_ylabel("parallel efficiency (%)"); a2.set_xlabel("threads")
    a2.set_title("Efficiency")
    label_bars(a2, bars, eff, "{:.0f}%")
    finish(fig, "Mandelbrot 1920×1080, max_iter=1000, rayon. Efficiency holds above 90% to 4 threads and 80% at 8 — "
            "the Ryzen 7 5800H's physical core count — then halves at 16. That is the SMT signature: sibling "
            "threads share execution units, so the last doubling buys 25% rather than another 2×.", top=1.0)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — SIMD progression
# ═════════════════════════════════════════════════════════════════════════════
def fig_simd():
    items = [
        ("scalar\nf64", ms(FULL, "simd_render_Mandelbrot/scalar/1920×1080"), "f64"),
        ("f64x4", ms(FULL, "simd_render_Mandelbrot/f64x4/1920×1080"), "f64"),
        ("f32x8", ms(FULL, "simd_render_ilp_Mandelbrot_1080p/f32x8/1920×1080"), "f32"),
        ("f32x8\n+ ILP", ms(FULL, "simd_render_ilp_Mandelbrot_1080p/f32x8_ilp/1920×1080"), "f32"),
    ]
    items = [i for i in items if i[1]]
    if not items:
        return None
    vals = [i[1] for i in items]
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    bars = ax.bar(range(len(vals)), vals, color=C_CPU, edgecolor="white", linewidth=1.6,
                  width=0.62)
    for b, i in zip(bars, items):
        if i[2] == "f32":
            b.set_hatch("///")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([i[0] for i in items], fontsize=8.4)
    ax.set_ylabel("frame time (ms)")
    ax.set_title("CPU vectorisation — AVX2, 1920×1080")
    label_bars(ax, bars, vals, "{:.2f} ms",
               extra=[f"{vals[0] / v:.2f}×" for v in vals])
    ax.set_ylim(0, vals[0] * 1.24)
    ax.legend(handles=[Patch(facecolor="white", edgecolor=INK2, hatch="///", label="f32"),
                       Patch(facecolor=C_CPU, label="f64")], loc="upper right")
    finish(fig, "Vectorisation buys 1.87× over scalar. ILP — two interleaved f32x8 dependency chains, bit-identical "
            "output — adds only 4%, at the low end of expectations, because Mandelbrot's cardioid and bulb "
            "rejection already prune many pixels before the hot loop, leaving less latency to hide.", top=1.0)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — precision
# ═════════════════════════════════════════════════════════════════════════════
def fig_precision():
    pairs = [
        ("CPU SIMD", ms(FULL, "simd_render_Mandelbrot/f64x4/1920×1080"),
         ms(FULL, "simd_render_ilp_Mandelbrot_1080p/f32x8_ilp/1920×1080"), C_CPU),
    ]
    # CUDA: f64 arm comes from the deep-zoom perturbation group's scalar arm at 1e6+
    cuda_f64 = ms(FULL, "cuda_perturbation_Mandelbrot_1080p/scalar/zoom_1e6")
    cuda_f32 = ms(FULL, "cuda_render_Mandelbrot/cuda/1920×1080")
    if cuda_f64 and cuda_f32:
        pairs.append(("CUDA", cuda_f64, cuda_f32, C_CUDA))
    pairs = [p for p in pairs if p[1] and p[2]]
    if not pairs:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    idx = np.arange(len(pairs))
    w = 0.34
    b64 = ax.bar(idx - w / 2, [p[1] for p in pairs], w, color=[p[3] for p in pairs],
                 edgecolor="white", linewidth=1.6)
    b32 = ax.bar(idx + w / 2, [p[2] for p in pairs], w, color=[p[3] for p in pairs],
                 edgecolor="white", linewidth=1.6, hatch="///")
    for b, p in zip(b64, pairs):
        ax.annotate(f"{p[1]:.2f}", (b.get_x() + w / 2, p[1]), ha="center", va="bottom",
                    fontsize=8, color=INK)
    for b, p in zip(b32, pairs):
        ax.annotate(f"{p[2]:.2f}\n{p[1] / p[2]:.1f}× faster", (b.get_x() + w / 2, p[2]),
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(idx)
    ax.set_xticklabels([p[0] for p in pairs])
    # Deliberately LINEAR. A bar encodes magnitude by length from zero; on a log
    # axis that correspondence breaks and a 17x gap stops looking like one.
    ax.set_ylabel("frame time (ms)")
    ax.set_ylim(0, max(p[1] for p in pairs) * 1.28)
    ax.set_title("Floating-point precision: what f32 buys")
    ax.legend(handles=[Patch(facecolor=MUTED, label="f64"),
                       Patch(facecolor=MUTED, hatch="///", label="f32")], loc="upper right")
    finish(fig, "f32 is only NUMERICALLY VALID below zoom 1e6 — above it the mantissa can no longer separate "
            "adjacent pixels and the result is fast but wrong. CUDA's f64 penalty is severe because Ampere "
            "GeForce runs fp64 at 1/64 the fp32 rate; on the CPU the gap is far smaller. This asymmetry is the "
            "whole reason the scheduler has a regime where it wins.", top=1.0)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — perturbation across zoom
# ═════════════════════════════════════════════════════════════════════════════
def fig_perturbation():
    zs = [("zoom_1e0", 1e0), ("zoom_1e3", 1e3), ("zoom_1e6", 1e6),
          ("zoom_1e9", 1e9), ("zoom_1e12", 1e12)]
    arms = [("scalar f64", "scalar", C_CPU, "o"),
            ("perturbation", "perturb", C_CUDA, "s"),
            ("perturbation + SA", "perturb_sa", C_HYB, "D")]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for label, key, colr, mk in arms:
        ys, xs = [], []
        for zk, z in zs:
            v = ms(FULL, f"perturbation_Mandelbrot_1080p/{key}/{zk}")
            if v:
                xs.append(z); ys.append(v)
        if xs:
            ax.plot(xs, ys, marker=mk, markersize=6.5, linewidth=2.0, color=colr,
                    label=label, markeredgecolor="white", markeredgewidth=0.8)
    ax.axvline(F128_THRESHOLD, color=INK2, ls=(0, (5, 3)), lw=1.0)
    ax.annotate("f128 reference\norbit past here", (F128_THRESHOLD, ax.get_ylim()[1] * 0.9),
                fontsize=7.6, color=INK2, ha="right", va="top")
    ax.set_xscale("log")
    ax.set_xlabel("zoom level"); ax.set_ylabel("frame time (ms)")
    ax.set_title("Perturbation theory is a correctness feature, not a throughput one")
    ax.legend()
    finish(fig, "1920×1080, max_iter=1000, seahorse valley. Plain scalar f64 is faster than perturbation at every "
            "zoom tested — series approximation skips only 2–36 iterations of 1000 for this reference point. "
            "Perturbation earns its place past f64's ~1e15 ceiling, where scalar iteration cannot resolve "
            "individual pixels at all; that regime is beyond what this sweep measures.", top=1.0)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — cross-project
# ═════════════════════════════════════════════════════════════════════════════
def fig_cross_project():
    # 1-thread f64 scalar, naive CPU path: the fully controlled cross-project
    # comparison. ruster/FractalRendererCpp/XaoS/FractalNow were all measured in
    # one session — bench_results/comparison_report.md (2026-08-18), Table A.
    ruster_1t = ms(FULL, "cpu_thread_scaling_Mandelbrot_1080p/threads/1")
    if ruster_1t is None:
        return None
    cpp_1t = 893.02     # FractalRendererCpp, Google Benchmark
    xaos_1t = 720.30    # XaoS, raw per-pixel kernel, guessing disabled (caveat 12)
    fnow_1t = 2013.25   # FractalNow, raw per-pixel kernel, quad-interp disabled (caveat 17)
    frs_1t, frs_16t = 18.07, 2.58   # Fractals-rs (f32 SIMD — different algorithm class, right panel only)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.2))

    names = ["ruster", "FractalRenderer\nCpp", "XaoS", "FractalNow"]
    vals = [ruster_1t, cpp_1t, xaos_1t, fnow_1t]
    bars = a1.bar(names, vals, color=[C_CUDA, C_ALT, C_WGPU, C_CPU], edgecolor="white",
                  linewidth=1.6, width=0.6)
    # Linear, for the same reason as Figure 8: the 36x gap should LOOK like 36x.
    a1.set_ylabel("frame time (ms)")
    a1.set_ylim(0, fnow_1t * 1.22)
    a1.set_title("Single-thread f64 scalar\n(fully controlled comparison)")
    label_bars(a1, bars, vals, "{:.0f} ms")
    for i, v in enumerate(vals[1:], start=1):
        a1.annotate(f"{v / ruster_1t:.0f}×", (i, v), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8.5, fontweight="bold", color=INK)

    ns = [1, 2, 4, 8, 16]
    r = [ms(FULL, f"cpu_thread_scaling_Mandelbrot_1080p/threads/{n}") for n in ns]
    cpp = [893.02, 458.05, 237.37, 135.22, 113.00]
    xaos = [720.30, 362.81, 185.16, 103.70, 87.96]
    fnow = [2013.25, 1019.89, 987.55, 707.70, 437.42]
    frs = [18.07, 9.06, 5.18, 3.15, 2.58]
    for ys, colr, lbl, mk in [(r, C_CUDA, "ruster (f64 scalar)", "o"),
                              (frs, C_HYB, "Fractals-rs (f32 SIMD)", "D"),
                              (cpp, C_ALT, "FractalRendererCpp (f64)", "s"),
                              (xaos, C_WGPU, "XaoS (f64, raw kernel)", "^"),
                              (fnow, C_CPU, "FractalNow (f64, raw kernel)", "v")]:
        a2.plot(ns, [ys[0] / y for y in ys], marker=mk, markersize=6.5, linewidth=2.0,
                color=colr, label=lbl, markeredgecolor="white", markeredgewidth=0.8)
    a2.plot(ns, ns, ls=(0, (5, 3)), color=MUTED, lw=1.3, label="ideal")
    a2.set_xscale("log", base=2); a2.set_yscale("log", base=2)
    a2.set_xticks(ns); a2.set_xticklabels(ns)
    a2.set_yticks(ns); a2.set_yticklabels(ns)
    a2.set_xlabel("threads"); a2.set_ylabel("speedup vs own 1 thread")
    a2.set_title("Parallel scaling (normalised)")
    a2.legend(loc="upper left", fontsize=7.0)

    finish(fig, "LEFT is the fully controlled comparison — same fractal, resolution, iteration count, precision, "
            "algorithm class and thread count, ruster/FractalRendererCpp/XaoS/FractalNow all measured in one "
            "session (bench_results/comparison_report.md, 2026-08-18) — so it isolates kernel quality: ruster "
            "rejects cardioid/period-2/period-3 interiors and detects cycles; none of the other three do. XaoS "
            "and FractalNow are measured on their raw per-pixel kernel with guessing disabled (caveats 12 and "
            "17); their real interactive engines spend most of a frame avoiding this work, not doing it. "
            "Fractals-rs's own full-frame bench is SIMD-only, a different algorithm class, so it is left out of "
            "LEFT and appears only at RIGHT, which normalises each project against its own single thread and so "
            "compares scaling shape, not speed.", top=1.0)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 11 — the fp64 bulb-check ablation
# ═════════════════════════════════════════════════════════════════════════════
def fig_ablation():
    # examples/ptx_variants.rs — kernel-only, within one process (immune to drift).
    variants = [("shipped now\nf32 check", 0.452, C_CUDA),
                ("--fmad=true\nf32 check", 0.448, C_CUDA),
                ("pre-fix\nfp64 check", 0.910, MUTED),
                ("pre-fix\n+ --fmad=true", 0.842, MUTED)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 4.0),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    vals = [v[1] for v in variants]
    bars = a1.bar(range(len(vals)), vals, color=[v[2] for v in variants],
                  edgecolor="white", linewidth=1.6, width=0.62)
    a1.set_xticks(range(len(variants)))
    a1.set_xticklabels([v[0] for v in variants], fontsize=7.8)
    a1.set_ylabel("kernel time (ms)")
    a1.set_title("fractal_kernel_f32 ablation")
    label_bars(a1, bars, vals, "{:.3f}")
    a1.annotate("2.01×", (0.5, 0.96), xycoords="axes fraction", ha="center",
                fontsize=11, fontweight="bold", color=INK)
    a1.set_ylim(0, max(vals) * 1.25)

    full = [("before", 1.892, MUTED), ("after", 1.605, C_CUDA)]
    b = a2.bar([f[0] for f in full], [f[1] for f in full],
               color=[f[2] for f in full], edgecolor="white", linewidth=1.6, width=0.5)
    label_bars(a2, b, [f[1] for f in full], "{:.3f} ms")
    a2.set_ylabel("full frame (ms)")
    a2.set_title("Effect on the whole frame")
    a2.annotate("−15%", (0.5, 0.9), xycoords="axes fraction", ha="center",
                fontsize=11, fontweight="bold", color=INK)
    a2.set_ylim(0, 2.3)

    finish(fig, "mandelbrot_f32 called the fp64 in_period3_bulb once per pixel. On Ampere GeForce, which runs fp64 at "
            "1/64 the fp32 rate, those 8 operations cost as much as the entire 1000-iteration f32 loop. Replacing "
            "them with an f32 predicate is bit-identical — zero differing pixels of 2,073,600 across seven "
            "viewports including three centred on the period-3 boundary arc. The full-frame gain is only 15% "
            "because the frame is readback-bound (Figure 3).", top=1.0)
    return fig


# ── driver ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=REPO / "results" / "figures", type=Path)
    args = ap.parse_args()

    if not FULL.exists():
        raise SystemExit(f"missing {FULL} — run the benchmarks first")
    args.out.mkdir(parents=True, exist_ok=True)

    figures = [
        ("01_scheduler_regimes", fig_scheduler_regimes),
        ("02_backends", fig_backends),
        ("03_gpu_frame_breakdown", fig_gpu_breakdown),
        ("04_pipeline_amdahl", fig_pipeline),
        ("05_resolution_scaling", fig_resolution),
        ("06_thread_scaling", fig_threads),
        ("07_simd", fig_simd),
        ("08_precision", fig_precision),
        ("09_perturbation", fig_perturbation),
        ("10_cross_project", fig_cross_project),
        ("11_fp64_ablation", fig_ablation),
    ]

    pdf_path = args.out / "thesis_figures.pdf"
    made = 0
    with PdfPages(pdf_path) as pdf:
        for name, fn in figures:
            try:
                fig = fn()
            except Exception as e:                      # keep going; report which
                print(f"  !! {name}: {type(e).__name__}: {e}")
                continue
            if fig is None:
                print(f"  -- {name}: skipped (data missing from the archives)")
                continue
            pdf.savefig(fig)
            fig.savefig(args.out / f"{name}.pdf")
            fig.savefig(args.out / f"{name}.png", dpi=200)
            plt.close(fig)
            made += 1
            print(f"  ok {name}")
    print(f"\n{made}/{len(figures)} figures -> {pdf_path}")
    print(f"individual PDF/PNG -> {args.out}/")


if __name__ == "__main__":
    main()
