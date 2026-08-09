# Thesis figures

Regenerate with `uv run scripts/thesis_figures.py`. Reads the criterion archives —
no re-measurement — and writes `thesis_figures.pdf` (all figures) plus an
individual `.pdf` and `.png` per figure.

Each figure is built to carry one argument. The claim column is what the figure
is *for*; if a figure ends up in the thesis without that sentence near it, it is
doing less work than it could.

| # | File | The claim it makes | Source |
|---|---|---|---|
| 1 | `01_scheduler_regimes` | **The scheduler wins above zoom 1e6 and loses below it, and the boundary is exactly `F32_PRECISION_THRESHOLD`.** 1.71× at 1e6; wins through 1e15. | `hybrid/heterogeneous_deep` |
| 2 | `02_backends` | CUDA is fastest at 1080p; the gap to the CPU is 4.37× against the f64 baseline but only 2.36× against the CPU's own f32 path. | `cuda/wgpu/cpu/simd render` |
| 3 | `03_gpu_frame_breakdown` | A GPU frame is 83–88% data movement. The kernel is the small end. | `examples/gpu_probe.rs` |
| 4 | `04_pipeline_amdahl` | The CPU colour stage is 47% of the CUDA pipeline and caps every backend. | `*_pipeline`, `cpu/colorize` |
| 5 | `05_resolution_scaling` | Throughput is flat in resolution — the kernel is embarrassingly parallel. | `*_render` at 3 resolutions |
| 6 | `06_thread_scaling` | Efficiency holds to 8 threads then halves — the SMT signature. | `cpu/thread_scaling` |
| 7 | `07_simd` | Vectorisation buys 1.87×; ILP adds only 4%, because bulb rejection already prunes the work. | `simd/render*` |
| 8 | `08_precision` | f32 is 1.4× on the CPU but 17× on CUDA — and only legal below zoom 1e6. That asymmetry is *why* figure 1 has a win regime. | `simd`, `cuda` f32/f64 arms |
| 9 | `09_perturbation` | Perturbation is a correctness feature at extreme zoom, not a throughput one. | `perturbation/*` |
| 10 | `10_cross_project` | ruster's kernel is ~20× the C++ one single-threaded — algorithmic, not language. | `cpu/thread_scaling` + archives |
| 11 | `11_fp64_ablation` | One fp64 line cost half the CUDA kernel; removing it is bit-identical and 2.01×. | `examples/ptx_variants.rs` |

## Conventions

- **Colour = backend**, fixed across every figure: CUDA blue, CPU orange,
  scheduler aqua, wgpu yellow, competitor projects magenta.
- **Hatching = f32**, solid = f64. Precision is never encoded as colour, because
  an f32/f64 mix-up is the easiest way to produce a misleading fractal benchmark.
- **Bars are always on a linear axis from zero.** A bar encodes magnitude by
  length; on a log axis that correspondence breaks and a 17× gap stops looking
  like one. Lines and scatter use log axes freely.
- **Error bars are criterion's 95% confidence intervals** where the underlying
  group provides them.
- Every figure carries its own caption, including caveats, because figures get
  extracted from documents.

The palette was checked with the dataviz validator (lightness band, chroma floor,
CVD separation, normal-vision floor — all PASS). Its contrast warning for the
lighter slots is discharged by direct-labelling every mark.

## Provenance

Figures 1 and the scheduler arms come from
`bench_results/criterion_20260808_210551_fc5ca0d_sched_improved`; everything else
from `bench_results/criterion_latest`. Figures 3 and 11 use constants measured by
the probe examples rather than criterion — those are annotated in the script with
their source, since they cannot be re-derived from the archives alone.

⚠ Figure 10's two projects were measured in **different sessions**, and identical
benchmarks on this machine move 15–30% between sessions from thermal state. Quote
it as "roughly 17–20×" until all three projects are re-run together — which is
what `other-projects/tri-compare` exists to do.
