# Benchmark Results Summary

**Mandelbrot only.** Resolution 1920×1080 and `max_iter=1000` unless noted. Criterion means
from `bench_results/criterion_latest` (run `criterion_20260808_200032_8d2dbdc_rtx3050_f32bulb`, git
`8d2dbdc`), filtered to Mandelbrot plus the fractal-agnostic groups; phase decompositions from
the probe examples in §1.4.

**Hardware:**

| Part | Detail |
|---|---|
| CPU | AMD Ryzen 7 5800H, 8 cores / 16 threads |
| GPU | NVIDIA RTX 3050 Laptop, `sm_86`, CUDA 12.0, driver 580.173.02 |
| (also present) | AMD Radeon Vega (Renoir) integrated — see §1.1 |

**Both GPU backends now run on the RTX 3050**, so wgpu-vs-CUDA is finally a same-silicon
comparison. This was not true for any earlier run in this project — §1.1.

**Two measurement caveats that govern how these numbers may be used:**

1. **Never compare a CPU row across runs.** Identical CPU benchmarks moved 15–30% between the
   previous archive and this one with no code change, because the previous archive was
   recorded back-to-back with another full sweep on a thermally-loaded laptop. Within-run
   comparisons are sound; cross-run ones are not.
2. **FractalRendererCpp and Fractals-rs were not re-measured** for this run, so §4's
   cross-project ratios pair a fresh ruster number against an older competitor number and
   inherit caveat 1. Re-run all three in one session before quoting a headline speedup.

---

## 1. Headline results

### 1.1 The two GPU backends used to run on different GPUs — fixed 2026-08-08

`nvidia-smi` reported the RTX 3050, so it was natural to assume every GPU number came from it.
That was false for the entire wgpu half of this project, for every run up to and including
`criterion_20260725_215202_8d2dbdc`.

The machine had the NVIDIA stack installed **compute-only**:

```
$ dpkg -l | grep -E 'nvidia-(driver|compute)'
libnvidia-compute-580     580.173.02      # CUDA / OpenCL / NVML
nvidia-utils-580          580.173.02
# no nvidia-driver-580, no libnvidia-gl-580

$ ls /usr/share/vulkan/icd.d/
asahi_icd.json  intel_icd.json  lvp_icd.json  nouveau_icd.json  radeon_icd.json  ...
# no nvidia_icd.json — and no libGLX_nvidia.so anywhere on the system
```

So there was **no NVIDIA Vulkan driver**, and wgpu could not see the RTX 3050 at all.
`enumerate_adapters` returned only the integrated AMD Vega, llvmpipe, and an AMD GL adapter,
and `PowerPreference::HighPerformance` picked the integrated Vega. Setting
`__NV_PRIME_RENDER_OFFLOAD=1` changed nothing — there was no NVIDIA ICD to offload to.

Installing `nvidia-driver-580` supplies `nvidia_icd.json` and `libGLX_nvidia`, after which:

```
$ cargo run --release --example adapters
vulkan | AMD Radeon Graphics (RADV RENOIR)  | IntegratedGpu
vulkan | NVIDIA GeForce RTX 3050 Laptop GPU | DiscreteGpu     <- now visible
vulkan | llvmpipe (LLVM 20.1.2, 256 bits)   | Cpu
=== HighPerformance selects ===
  NVIDIA GeForce RTX 3050 Laptop GPU (DiscreteGpu, Vulkan)
```

No code change was needed — the existing `PowerPreference::HighPerformance` request picks the
discrete GPU as soon as one is visible. The bench harness now prints the selected adapter at
startup (`[wgpu] NVIDIA GeForce RTX 3050 Laptop GPU (Vulkan)`), and `examples/adapters.rs`
verifies it on demand, so this cannot silently regress again.

**Effect on the results: the ranking flips.** On the integrated GPU, wgpu appeared to beat
CUDA (1.79 vs 1.92 ms) — that advantage was entirely the iGPU not paying a PCIe transfer.
Moving wgpu onto the same discrete GPU as CUDA reverses it:

| 1920×1080 Mandelbrot | wgpu on Vega iGPU | wgpu on RTX 3050 | CUDA on RTX 3050 |
|---|---:|---:|---:|
| Full frame | 1.79 ms | **2.35 ms** | **1.89 ms** |

*(All three columns are pre-§2.2-fix, so the driver change is the only variable. Post-fix the
same three benchmarks read 1.79 / 2.26 / **1.61** ms — CUDA's lead widens from 1.24× to 1.41×.)*

### 1.2 Fastest at 1920×1080, Mandelbrot, `max_iter=1000`, zoom 1

All ruster rows from one run on one machine; the two competitor rows are older (see caveats).

| Rank | Implementation | Device | Prec. | Mean ms | Mpix/s |
|---:|---|---|---|---:|---:|
| 🥇 | **ruster — CUDA** | RTX 3050 | f32 | **1.61** | 1292 |
| 🥈 | ruster — heterogeneous (CUDA), zoom 1e4 | CPU + RTX 3050 | f32 | 2.03 | 1020 |
| 🥉 | ruster — wgpu | RTX 3050 | f32 | 2.26 | 918 |
| 4 | ruster — heterogeneous (CUDA), zoom 1e0 | CPU + RTX 3050 | f32 | 2.72 | 762 |
| 5 | ruster — heterogeneous (wgpu), zoom 1e4 | CPU + RTX 3050 | f32 | 3.38 | 613 |
| 6 | **ruster — SIMD f32x8 + ILP** *(fastest CPU)* | CPU | f32 | **3.79** | 548 |
| 7 | ruster — hybrid CPU+wgpu (static split) | CPU + RTX 3050 | f32 | 4.95 | 419 |
| 8 | ruster — SIMD f64x4 | CPU | f64 | 5.18 | 400 |
| 9 | Fractals-rs — SIMD "fast" *(older run)* | CPU | f32 | 5.62 | 369 |
| 10 | **ruster — CPU scalar / rayon** *(baseline)* | CPU | f64 | **7.02** | 296 |
| 11 | Fractals-rs — "high" *(older run)* | CPU | f64 | 16.00 | 130 |
| 12 | **FractalRendererCpp** *(older run)* | CPU | f64 | **247.87** | 8.4 |

**Precision caveat, now enforced by the tooling.** Rows 1–7 and 9 are f32; the rest are f64.
`scripts/aggregate_bench.py` used to stamp every ruster record `precision: "f64"`
unconditionally, which mislabelled all of them — it now derives precision per record
(`ruster_precision()`), so `bench_results/comparison.csv` no longer implies a like-for-like
comparison that isn't there. The honest same-precision GPU-vs-CPU pairing is
**CUDA 1.61 ms vs SIMD f32x8+ILP 3.79 ms = 2.36×**, not the 4.4× you get by racing an f32 GPU
against the f64 CPU baseline.

### 1.3 The GPU advantage, stated three honest ways

| Comparison | Ratio |
|---|---:|
| CUDA vs the f64 CPU scalar baseline | 4.37× |
| CUDA vs the CPU's *own best f32 path* (`f32x8_ilp`) | **2.36×** |
| CUDA vs CPU **end-to-end**, including `colorize()` (§5.4) | 2.02× |

The middle row is the one to quote for a precision-controlled kernel comparison; the bottom row
is the one a user actually experiences. For an embarrassingly-parallel per-pixel kernel on a
mid-range laptop dGPU against 16 Zen 3 threads running AVX2, ~2.4× is a reasonable and
defensible result — and a far more honest headline than the "5.2×" this document once carried.

The margin stays this narrow because of §2: **the frame is dominated by the host transfer, not
the kernel.** Returning 8.29 MB of results costs 1.33 ms; computing them costs 0.28 ms. Every
GPU backend here spends ~4× longer handing the answer back than producing it — and the fp64
bulb fix (§2.2) made that ratio *worse*, not better, by halving the numerator.

### 1.4 Reproducing the diagnostics

Every measurement in this document is reproducible from a probe binary in `examples/`:

```bash
cargo run --release --example adapters                            # which GPU wgpu selects
cargo run --release --example sched_overhead                      # scheduler overhead, work removed
cargo run --release --features cuda --example gpu_probe           # kernel vs readback split
cargo run --release --features cuda --example ptx_variants        # fmad / fp64-bulb ablation + pixel diff
cargo run --release --features cuda --example readback_probe      # readback scaling, pinning, per-call cost
cargo run --release --features cuda --example sched_probe         # scheduler vs solo backends
cargo run --release --features cuda --example sched_deepzoom      # scheduler across the f32/f64 boundary
cargo run --release --features cuda --example tiled_dispatch_probe # tiled vs Morton launch geometry
```

`ptx_variants` regenerates its kernel variants from the shipped `fractal.cu` with `nvcc` at run
time, so it stays honest if the kernel changes.

---

## 2. Where a GPU frame actually goes — measured, not guessed

Both backends, same GPU, 1920×1080 Mandelbrot, zoom 1, f32, 30 reps after warm-up, **median of
5 independent process launches**:

| Phase | CUDA | wgpu |
|---|---:|---:|
| Kernel | **0.28 ms** *(0.27 – 0.44)* | 0.29 ms *(0.29 – 0.44)* |
| Host readback (8.29 MB) | **1.33 ms** *(1.33 – 1.34)* | 2.14 ms *(1.83 – 2.42)* |
| Full frame | **1.61 ms** | 2.26 ms |
| Readback as share of frame | ~82% | ~90% |

Three findings, in order of importance:

1. **Both frames are readback-dominated — overwhelmingly so.** Returning the results costs 5–7×
   what computing them does. This is the single most important structural fact about GPU
   rendering in this project; it caps §1.3's speedup and it drives §3 entirely.
2. **CUDA wins the frame (1.61 vs 2.26 ms) purely on the readback.** The two kernels are now
   within 3% of each other (0.28 vs 0.29 ms). CUDA issues one `dtoh_sync_copy`; wgpu's
   `dispatch_and_readback` does a device→device `copy_buffer_to_buffer` into a separate
   `MAP_READ` buffer *first*, then `map_async` + `poll(Wait)` + a memcpy into a fresh `Vec` +
   `unmap`. That extra staging copy is ~0.8 ms of pure overhead and is the entire gap between
   the two backends. CUDA's copy is also remarkably stable (1.33 ms ±1% across every run ever
   measured here) — a fixed 8.29 MB PCIe DMA behaving exactly like one.
3. **The two kernels converging is itself the confirmation of §2.2.** Before the fp64 bulb fix,
   CUDA's kernel was 0.59 ms against wgpu's 0.30 — and the WGSL shader is essentially the
   "already fixed" version of the same kernel (no period-3 check, FMA contraction enabled). The
   ablation predicted CUDA would land at 0.59 × 0.495 ≈ 0.29 ms once fixed. It measures 0.28 ms.
   A prediction made from an ablation, confirmed by an independent implementation, and then
   confirmed again by applying the change.

*Methodology note — one measurement I had to throw away.* The wgpu kernel time was originally
derived by timing a standalone `readback()` (GPU idle, so no kernel behind it) and subtracting
it from `render()`. That is invalid: `render()` encodes the compute pass and the copy into one
encoder and submits once, whereas a standalone `readback()` is a second encoder, a second
submit and its own map/unmap round trip, so it costs materially more than the copy does
*inside* `render()`. On the RTX 3050 it over-counted past `render()`'s own total and produced a
**negative** kernel time (−0.642 ms). The number now comes from `dispatch_tiled` over one
full-frame tile plus `poll(Wait)` — same per-pixel math, no copy at all. The flawed method
happened to give plausible-looking numbers on the integrated GPU, which is precisely why it
went unnoticed until the hardware changed.

*Variance note:* any GPU claim in this document smaller than ~20% is inside the run-to-run
noise of this machine unless it comes from a paired within-run comparison (§2.1) or a ratio
measured in a single process (§2.2). Both of those are clock-state-controlled by construction.

### 2.1 Morton dispatch is *not* the problem (hypothesis tested and rejected)

`CudaFractal::morton_cfg` pads the launch to the smallest enclosing power-of-two **square**, so
a 1920×1080 frame launches a 2048×2048 grid:

| Launch geometry | Threads | vs 2,073,600 pixels |
|---|---:|---:|
| `morton_cfg` padded square | 4,194,304 | **2.02×** |
| plain 2D grid (wgpu-shaped) | 2,088,960 | 1.01× |

That looks damning — CUDA launches twice the threads wgpu does. It costs almost nothing.
`dispatch_tiled_f32` over one full-frame tile runs the *same* `mandelbrot_f32` math on a plain
2D grid, so the two are directly comparable — and because both are timed in the same process
back to back, the comparison is immune to the clock-state noise in §2:

| Run | Morton (padded square) | Plain 2D grid | Penalty |
|---|---:|---:|---:|
| 1 | 0.837 | 0.821 | +1.9% |
| 2 | 0.916 | 0.799 | +14.6% |
| 3 | 0.895 | 0.904 | −1.0% |
| 4 | 0.569 | 0.553 | +2.9% |
| 5 | 0.561 | 0.545 | +2.9% |
| 6 | 0.579 | 0.590 | −1.9% |
| 7 | 0.585 | 0.575 | +1.7% |
| | | **median** | **+1.9%** |

**~2%, with two runs showing Morton slightly ahead.** The ~2.1M surplus threads land in a
contiguous L-shaped dead region, so entire blocks fail the `x >= width || y >= height` test and
retire immediately. Worth recording as a negative result: the obvious suspect was innocent, and
the doubled thread count is a red herring.

### 2.2 The CUDA kernel was carrying a 2× self-inflicted penalty — now fixed

`ptx_variants` ablates the two questionable choices in the f32 CUDA path, timing
`fractal_kernel_f32` alone with no host copy. Run against the current tree, where the third
column is what the project shipped before this fix:

| Variant | Kernel ms | vs shipped |
|---|---:|---:|
| **shipped now:** `--fmad=false` + f32 bulb check | **0.452** | — |
| `--fmad=true` + f32 bulb check | 0.448 | −0.9% |
| `--fmad=false` + fp64 bulb check *(pre-fix)* | 0.910 | **+101.2%** |
| `--fmad=true` + fp64 bulb check *(pre-fix)* | 0.842 | +86.2% |

**`mandelbrot_f32` was spending half its runtime on one fp64 line.** `fractal.cu` used to read:

```c
// Period-3 check stays f64, same as the CPU's `bulb_precheck_x8` — it
// runs once per pixel, not per iteration, so it isn't the bottleneck.
if (in_period3_bulb((double)cr, (double)ci)) return (float)max_iter;
```

That comment was measurably wrong *on this GPU*. The reasoning holds on a CPU, where f64 costs
the same as f32 — and the CPU's `bulb_precheck_x8` (`fractal.rs:1811`) really does call the f64
version per lane, so the kernel faithfully mirrored it. But Ampere GeForce runs fp64 at 1/64 the
fp32 rate. PTX inspection confirmed 8 fp64 arithmetic ops surviving in the f32 kernel and 0 in
the f32-check variant; at a 1/64 rate those 8 ops cost roughly as much issue bandwidth as the
entire 1000-iteration f32 loop they were supposed to be an optimization for.

**The fix is bit-identical, not merely close.** `in_period3_bulb_f32` now backs the check.
`ptx_variants` renders seven viewports with both kernels and compares every pixel:

| Viewport | differing pixels (of 2,073,600) |
|---|---:|
| whole set, zoom 1 | 0 |
| period-3 bulb, boundary in view (zoom 30) | 0 |
| bulb interior (zoom 3000) | 0 |
| **centred ON the boundary arc, zoom 1e2 / 1e4 / 1e6** | **0 / 0 / 0** |
| seahorse valley, zoom 100 | 0 |

The boundary-arc views are the load-bearing ones — centred at nucleus + radius, so the arc
crosses the frame, at zooms up to `F32_PRECISION_THRESHOLD` (1e6), the deepest this kernel is
ever used. Zero differences, because `cr`/`ci` arrive already rounded to f32: widening them for
the comparison adds no information the predicate can act on. (An earlier version of this check
used a zoom-3000 view labelled "bulb edge" that was actually deep *inside* the bulb — the radius
is 0.0737 but the half-width there is 0.0007 — and so proved nothing. The arc views replaced it.)

`--fmad=false` (set file-wide in `build.rs` so CUDA's rounding matches Rust's, for CPU/GPU
bit-exactness at deep zoom) cost a further 7.6% while the fp64 check dominated; with the check
gone the difference collapses to **0.9%**. The fmad setting is cheap and keeps earning its
correctness guarantee.

**Realised win: −15% on the CUDA full frame at every resolution**, and the change propagates to
`fractal_kernel_tiled_f32` (the scheduler's GPU tiles) for free:

| CUDA Mandelbrot | pre-fix | post-fix | change |
|---|---:|---:|---:|
| 800×600 | 0.574 ms | 0.479 ms | −16.5% |
| 1920×1080 | 1.892 ms | **1.605 ms** | **−15.2%** |
| 3840×2160 | 7.407 ms | 6.016 ms | −18.8% |
| heterogeneous scheduler, zoom 1e0 | 3.595 ms | 2.722 ms | −24.3% |
| heterogeneous scheduler, zoom 1e4 | 2.474 ms | 2.033 ms | −17.8% |

The full-frame gain is smaller than the kernel's 2.01× precisely because the frame is
readback-bound (§2): the kernel was only ~31% of the budget, and is now ~18%. Note the ablation
ratio is measured within a single process, so unlike the absolute times it is immune to the
clock-state variance in §2.

---

## 3. The heterogeneous scheduler: where it loses, where it wins

**Summary of this section.** The scheduler loses to GPU-only at every zoom below 1e6 and wins
at every zoom above it, and the boundary is exactly `F32_PRECISION_THRESHOLD`. §3.1–§3.3
explain the shallow-zoom loss (a fixed full-frame readback the scheduler cannot shrink);
§3.4–§3.6 establish the ceiling and the strategic picture; §3.7 covers two improvements now
applied; §3.8 is the new deep-zoom benchmark group where it wins up to 1.71x.

### 3.1 The arithmetic that makes it structural

From §2: of CUDA's 1.61 ms frame, **1.33 ms is a fixed full-frame `dtoh_sync_copy`** and only
~0.28 ms is kernel. `CudaFractal::readback` copies the entire `width×height` buffer regardless
of how many tiles the GPU was actually given — the TODO at `src/gpu/cuda.rs:247` says so.

So when the scheduler moves work to the CPU, it can only shrink the ~0.28 ms slice:

| Zoom | GPU's share of pixels | Best possible kernel saving | Measured scheduler overhead |
|---|---:|---:|---:|
| 1e0 | 96.9% | 0.01 ms | +1.24 … +1.54 ms |
| 1e2 | 86.1% | 0.04 ms | (CPU becomes critical path) |
| 1e4 | 100% | 0.00 ms | +0.00 … +0.32 ms |

The scheduler pays more in fixed machinery than the entire kernel saving it can theoretically
achieve — by **two** orders of magnitude. **No tuning of thresholds or tile sizes can fix
this**, because the term it needs to reduce is not on the table.

The fp64 bulb fix (§2.2) made this *worse*, which is worth stating plainly: halving the kernel
halved the only quantity the scheduler can attack, taking the readback from 69% to 82% of the
frame. Every future kernel optimisation moves the scheduler further underwater. The only repair
is partial readback (§3.4).

### 3.2 Phase breakdown (`gpu_probe`, center −0.75/0.1)

| Zoom | tiles (GPU / CPU) | CPU px | partition | gpu_ms | cpu_ms | wall | plain CUDA |
|---|---|---:|---:|---:|---:|---:|---:|
| 1e0 | 213 / 127 | 3.1% | 0.072 ms | 1.877 | 0.910 | 3.10 ms | 1.90 ms |
| 1e2 | 256 / 569 | 13.9% | 0.316 ms | 1.922 | 4.674 | 5.61 ms | 2.59 ms |
| 1e4 | 135 / 0 | 0.0% | 0.033 ms | 1.826 | 1.036 | 2.05 ms | 1.91 ms |

Three things to read off this:

- **`partition_frame` is cheap** (0.03–0.32 ms). The corner-sampling classifier was the second
  suspect and it is also innocent — parallelising it over rayon did its job.
- **`gpu_ms` ≈ plain full-frame render time at every zoom**, even at zoom 1e0 where the GPU
  holds 96.9% of pixels and at 1e2 where it holds 86.1%. That is §3.1 made visible: the
  readback floor dominates, so the GPU side costs the same whatever fraction it renders.
- **At zoom 1e2 the CPU is the critical path with 13.9% of the pixels** (`cpu_ms` 4.67 ms vs
  `gpu_ms` 1.92 ms). This is the one original Stage 3 finding that survives intact: boundary
  tiles are exactly the tiles that take the most iterations, so a pixel-count-balanced split is
  badly iteration-count-imbalanced.

### 3.3 Isolating the fixed overhead (`sched_probe`, 3 runs)

Forcing a degenerate all-GPU partition (`threshold = 1e9`, `steal_reserve_frac = 0`) leaves the
scheduler doing exactly what plain `cuda.render()` does, plus its own machinery:

| Zoom | plain CUDA | scheduler, all-GPU | fixed overhead |
|---|---:|---:|---:|
| 1e0 | 2.25–2.27 ms | 3.50–3.78 ms | **+1.24 / +1.45 / +1.54 ms** |
| 1e2 | 3.04–3.39 ms | 2.86–3.48 ms | −0.53 / −0.02 / +0.19 ms |
| 1e4 | 1.90–2.08 ms | 2.06–2.32 ms | −0.01 / +0.25 / +0.32 ms |

The zoom-1e0 penalty (~1.3 ms) reproduces across all three runs. It is the cost of
`rayon::scope` spawning 16 workers, an mpsc message per worker, `Mutex<VecDeque>` traffic over
340 tiles, a `Vec<f32>` allocation per CPU tile, and the serial row-by-row merge at the end.

**Adaptive scheduler vs the better solo backend**, same three runs:

| Zoom | best solo | adaptive | penalty |
|---|---:|---:|---:|
| 1e0 | 2.25 ms (GPU) | 2.62–2.82 ms | +17% … +24% |
| 1e2 | 3.04 ms (GPU) | 5.56–6.20 ms | +64% … +91% |
| 1e4 | 1.90 ms (GPU) | 2.27–2.75 ms | +10% … +45% |

**The adaptive scheduler does not beat plain CUDA at any tested zoom.** Note the run-to-run
spread (plain CUDA ranges 1.89–3.39 ms for the same viewport across runs) — this is a
power-managed laptop GPU, and any claim under ~15% here is inside the noise.

One measurement that looks like a win but isn't: the all-CPU degenerate partition (7.5–11.6 ms)
beats plain `render()` (6.6–33.9 ms) by 3–4× at zoom ≥ 1e2. That is not the scheduler being
clever — `SchedulerConfig::simd_cpu_tiles` routes CPU tiles through the f32 SIMD kernel while
plain `render()` is scalar f64. Different kernel, not different scheduling.

### 3.4 Can it be made to beat GPU-only? — measured answer

**Yes, but only outside the regime the benchmarks currently cover, and the win is
small where they do cover.** Four probes were built to answer this
(`readback_probe`, `sched_overhead`, `tiled_dispatch_probe`, `sched_deepzoom`).

**Finding 1 — the scheduler already wins at zoom 1e6, and nothing measures it.**
The shipped benches stop at zoom 1e4, which is entirely inside the regime where the GPU
has an f32 fast path and is ~4x the CPU. No scheduler can improve on that. Above
`F32_PRECISION_THRESHOLD` (1e6) the GPU drops to its f64 kernel at 1/64 rate and the CPU
becomes competitive.

*This table is the original `examples/sched_deepzoom.rs` probe, measured **before** the
improvements in §3.7 and in a different thermal session. §3.8 has the current, post-improvement
criterion numbers; the two tables' milliseconds are not comparable, only their verdicts are.*

| zoom | CUDA | CPU | best solo | hybrid | ideal split | verdict |
|---|---:|---:|---:|---:|---:|---|
| 1e4 | 6.05 | 60.61 | 6.05 | 7.10 | 5.5 | 0.85x lose |
| 1e5 | 5.73 | 82.42 | 5.73 | 6.38 | 5.4 | 0.90x lose |
| **1e6** | **215.75** | **81.89** | **81.89** | **56.34** | 59.4 | **1.45x WIN** |
| 1e7 | 91.29 | 58.25 | 58.25 | 78.34 | 35.6 | 0.74x lose |
| 1e9 | 91.11 | 56.42 | 56.42 | 78.45 | 34.8 | 0.72x lose |

At zoom 1e6 it beats not just the better solo backend but the *ideal proportional split*
(56.3 vs 59.4 ms) — corner-sampling routes expensive tiles to the CPU rather than dividing
work blindly, which is exactly the thesis claim, demonstrated. Reproduced across runs
(1.45x, 1.49x).

At 1e7/1e9 it loses badly while an ideal split would give 36 ms against 78 ms measured —
a 2.2x miss, i.e. a load-balancing failure, not a structural one. Both halves are on their
slow paths there (`use_gpu_f32` and `simd_cpu_tiles` both switch off above 1e6) and the
controller hands the GPU far too much work.

**Finding 2 — the overhead is not where §3.3 assumed.** With the CPU assigned nothing, the
scheduler's own machinery (partition + queues + `rayon::scope` + merge) costs **0.10 ms**,
not the ~1.3 ms the black-box measurement suggested. Isolating each piece:

| component | cost |
|---|---:|
| `rayon::scope` + 16 spawns + 16 channel msgs | 0.009 ms |
| `Mutex<VecDeque>` drain, 135 tiles | +0.010 ms |
| `Mutex<VecDeque>` drain, 8160 tiles (16px) | +0.328 ms |
| **`Vec`-per-tile alloc + serial row-by-row merge** | **1.0 – 3.9 ms** |
| (reference) one flat 8.29 MB host memcpy | 0.360 ms |

The merge dominates by an order of magnitude — allocating one `Vec<f32>` per tile and
copying it back row by row costs 3–11x a single flat memcpy of the whole frame. Two
hypotheses were tested and rejected: tiled GPU dispatch is *faster* than the Morton
whole-frame launch (−8 to −12%), and rayon workers start within 0.06 ms even while the
calling thread is blocked in CUDA.

**Finding 3 — partial readback is viable, but only row-contiguous.** Readback time scales
almost linearly with size (50% of the buffer = 51.6% of the time, 25% = 28.9%), so shrinking
the copy genuinely helps. But splitting the same bytes into many copies is catastrophic:

| copies | 1 | 8 | 64 | 135 | 340 | 825 |
|---|---:|---:|---:|---:|---:|---:|
| vs one copy | 0.97x | 1.40x | 1.86x | **2.60x** | 3.21x | 5.93x |

135/340/825 are the actual GPU tile counts at zoom 1e4/1e0/1e2. **A per-tile readback would
be 2.6–5.9x worse than the full-frame copy it replaces.** So partial readback requires the
GPU's share to be one contiguous row range — which means abandoning scattered tile
assignment for a horizontal split, i.e. the "dumb" static split the adaptive scheduler was
built to replace. That is a genuine architectural tension, not a tuning knob.

**Finding 4 — the backends are not independent.** Loading all 16 CPU threads slows plain
`cuda.render()` by +1.9% / +33.3% / +7.2% at zoom 1e0/1e2/1e4. The readback's host half is a
memcpy competing for memory bandwidth. Any hybrid model that assumes CPU work is free
because it overlaps GPU work is wrong by up to a third.

**Finding 5 — the transfer is at the hardware ceiling.** The RTX 3050 negotiates **PCIe 3.0
x8** (`pcie.link.width.current = 8` against a max of 16) = 7.9 GB/s theoretical. Measured
6.7 GB/s with a page-locked destination, 85% efficiency. Pinning the destination and reusing
it is worth **−7.5%** on the readback for no algorithmic change, and that is all that is
available: the copy cannot be made faster, only smaller.

### 3.5 The ceiling, from measured constants

GPU 0.28 ms kernel + 1.33 ms readback; best CPU f32 path 3.79 ms; combining as
`1/(1/T_gpu + 1/T_cpu)`:

| scenario | hybrid | vs GPU-only |
|---|---:|---:|
| today (full readback, tile merge) | 2.33 ms | 0.69x |
| + direct-write CPU tiles (drop the merge) | 1.28 ms | 1.26x |
| + partial readback, row-contiguous, pinned | 1.23 ms | 1.31x |
| ...with the measured CPU/GPU contention | 1.34 ms | **1.20x** |
| perfect: zero overhead, no contention | 1.08 ms | 1.49x |

**So a realistic ceiling is ~1.2–1.3x on the render stage at shallow zoom**, requiring the
merge removal and a row-contiguous rewrite. The theoretical maximum is 1.49x and is
unreachable.

### 3.6 Strategic verdict

End-to-end, including the 2.34 ms CPU `colorize()` stage:

| | pipeline | vs today |
|---|---:|---:|
| GPU-only today | 3.95 ms | — |
| best realistic hybrid scheduler | 3.68 ms | 1.07x |
| **move `colorize()` to the GPU** | 1.81 ms | **2.18x** |
| GPU colorize + render straight to texture (no readback) | 0.48 ms | **8.2x** |

**A perfect heterogeneous scheduler is worth 7% end-to-end. Moving colorize onto the GPU is
worth 118%, and eliminating the readback entirely is worth 720%.** The scheduler is
optimizing a stage that is already the small half of a pipeline dominated by data movement.

That is not an argument that the scheduler is worthless — §3.8 shows it winning at *every*
zoom from 1e6 to 1e15, by up to 1.71x, and it would win broadly on hardware where CPU and GPU
are closer (a stronger CPU, a weaker or busier GPU, or any f64 workload). It is an argument
about **where the remaining engineering effort belongs** for shallow-zoom interactive
rendering, and about benchmarking the scheduler in the regime where its premise actually holds
rather than the one where the GPU trivially wins.

### 3.7 Applied: pooled tile buffers + pinned readback

Two of the improvements identified above are now implemented.

**1. Per-worker scratch instead of a `Vec` per tile.** CPU workers used to
`vec![0.0; tw*th]` for every tile they claimed, hand the `Vec` back over a channel, and
have the calling thread merge each one row by row. §3.4 measured that at 1.0–3.9 ms per
frame — the single largest piece of scheduler overhead, and dominated by page faults on
freshly-mapped pages rather than by the copying.

Workers now append every tile they render into one flat, geometrically-grown scratch buffer
and report a `(tile, offset)` index alongside it: **one allocation per worker per frame
instead of one per tile.** New `_into` variants (`render_tile_exact_into`,
`render_tile_exact_simd_into`, `render_cpu_tile_into`) render into a caller-owned slice; the
allocating forms are kept as thin wrappers so nothing else had to change.

Full direct-write into the frame buffer — workers writing straight to their final pixels —
was considered and rejected. It cannot work while the readback is full-frame: the readback
overwrites every pixel including the CPU's, so CPU results must be applied *after* it. Making
that safe needs partial readback, which needs row-contiguous GPU regions (§3.4, finding 3).
The residual copy this leaves is only the CPU's share of the frame — 3–23% of pixels, tens of
microseconds — so essentially all of the available win is captured without the unsafe aliasing
a true direct-write would require.

**2. `PinnedBuf` + `readback_into`.** `CudaFractal::readback_into(&mut [f32])` writes into a
caller-owned destination, and `PinnedBuf` is a page-locked (`cuMemHostRegister`) reusable
frame buffer that unregisters on drop and degrades gracefully to plain heap memory if
page-locking fails. `render_heterogeneous_into` takes that destination and DMAs directly into
it.

The API split matters: page-locking is only worth its cost when amortized over many frames, so
it cannot be done inside a per-call API that allocates and returns a fresh buffer. Reusing a
*pageable* buffer is worth nothing on its own (measured +1.4%, i.e. noise — the allocator hands
back the same mapped pages); the win is the pinning, and the pinning requires the reuse.
`render_heterogeneous` is kept as an allocating wrapper for callers that don't have a render
loop.

**Measured, 1920×1080 Mandelbrot, own controller per variant, 2 runs:**

| zoom | owned `Vec` | `_into` pageable | `_into` **pinned** | pinned vs owned |
|---|---:|---:|---:|---:|
| 1e0 | 2.16 / 2.32 | 1.88 / 1.86 | **1.78 / 1.87** | −17.8% / −19.6% |
| 1e2 | 5.69 / 5.66 | 5.14 / 5.33 | **5.10 / 5.36** | −10.3% / −5.4% |
| 1e4 | 5.68 / 5.93 | 5.47 / 6.04 | **5.48 / 5.54** | −3.5% / −6.7% |
| 1e6 | 57.2 / 59.4 | 57.5 / 60.1 | 58.2 / 60.5 | +1.8% / +1.9% |
| 1e7 | 79.0 / 79.4 | 78.8 / 79.1 | **78.7 / 79.0** | −0.4% / −0.5% |

**The gain is inversely proportional to how much real work the frame does**, which is exactly
right: both changes attack fixed per-frame costs, so they matter most at shallow zoom where the
frame is short, and vanish at deep zoom where a 60–80 ms compute dwarfs a 1.3 ms readback. That
also means neither change moves the deep-zoom regime where the scheduler actually wins — those
wins come from the split itself, not from overhead removal.

**Correctness.** `render_heterogeneous` and `render_heterogeneous_into` produce byte-identical
frames (0 differing pixels of 2,073,600 at zoom 1e0/1e2/1e6). With `simd_cpu_tiles` and
`gpu_tiles_f32` both off — the bit-exact mode — output still matches plain CPU `render()` to
within the 3/24/184 pixels of pre-existing GPU-vs-CPU fp64 non-determinism documented in
`classifier`'s module doc. No new divergence class was introduced.

### 3.8 New benchmark group: `hybrid/heterogeneous_deep`

`bench_heterogeneous` (zoom 1e0–1e4) is **deliberately left unchanged** — its losses are the
evidence for when the scheduler is *not* viable, and that is a result worth keeping verbatim.
The new group extends the sweep across `F32_PRECISION_THRESHOLD` and fixes two design flaws in
how the shallow one reports:

1. **Solo-backend reference arms in the same group.** Each zoom benchmarks `cuda`, `cpu` and
   `hybrid` side by side. The shallow group has no reference arm, which is precisely why its
   numbers looked acceptable in isolation while actually being 20–30% worse than just using the
   GPU. A hybrid number is meaningless without the two it has to beat, and measuring all three
   back to back also makes them immune to the cross-session thermal drift that makes this
   laptop's absolute numbers untrustworthy.
2. **Uses `render_heterogeneous_into` with a `PinnedBuf`** — the form a real render loop would
   use — instead of allocating a frame per call.

Run `criterion_20260808_210551_fc5ca0d_sched_improved` (scheduler groups only; the full-suite
archive `criterion_latest` predates these changes and is not mixed with them):

| zoom | CUDA | CPU | hybrid | best solo | ideal split | verdict |
|---|---:|---:|---:|---:|---:|---|
| 1e4 | **4.25** | 91.84 | 4.55 | 4.25 | 4.06 | 0.93x lose |
| 1e5 | **4.37** | 125.59 | 4.71 | 4.37 | 4.22 | 0.93x lose |
| **1e6** | 218.09 | 117.19 | **68.58** | 117.19 | 76.23 | **1.71x WIN** |
| 1e7 | 92.05 | 80.93 | **78.97** | 80.93 | 43.07 | 1.02x win |
| 1e9 | 91.86 | 82.27 | **78.49** | 82.27 | 43.40 | 1.05x win |
| 1e12 | 91.47 | 103.80 | **78.51** | 91.47 | 48.62 | 1.16x win |
| 1e15 | 91.46 | 81.30 | **78.47** | 81.30 | 43.04 | 1.04x win |

**The scheduler wins at every zoom from 1e6 up, and loses at every zoom below it.** The
boundary is exactly `F32_PRECISION_THRESHOLD`, which is the whole story in one line: below it
the GPU runs f32 and is 20x the CPU, so there is nothing to schedule; above it the GPU falls
back to f64 at 1/64 rate, the two processors come within ~1.1x of each other, and splitting the
frame pays. At 1e6 it beats the better solo backend by 1.71x and beats the ideal *proportional*
split too (68.6 vs 76.2 ms) — corner-sampling routes expensive tiles to the CPU instead of
dividing work blindly, which is the thesis claim demonstrated.

**The 1e7+ rows are the remaining work.** Winning by 1.02–1.16x against an available 43–49 ms
means the load balancer leaves ~45% on the table. Both halves are on their slow paths there
(`use_gpu_f32` and `simd_cpu_tiles` both switch off above 1e6) and the controller, which
balances on measured `gpu_ms` vs `cpu_ms`, converges to a split that keeps the GPU on the
critical path. This is a tuning problem in a regime that until now was never benchmarked, not a
structural one — the 1e6 row proves the machinery can reach past the proportional bound.

⚠ Absolute times here are not comparable with §3.4's earlier probe run (CPU at zoom 1e4 reads
91.8 ms here against 60–71 ms there) — that is this laptop's thermal drift, the same effect
documented at the top of this document. The within-group `cuda`/`cpu`/`hybrid` ratios are the
result; the milliseconds are not.

### 3.9 What would actually make it win

Two are done (§3.7). In priority order by measured headroom, what remains:

1. **Partial readback.** Copy only the rows the GPU actually touched, or compact tiles device-side
   before the copy. This is the only change that attacks the 1.32 ms fixed cost, and it is what
   turns the scheduler's premise from false into true. Everything else is second-order.
2. **Iteration-weighted balancing.** Weight the split by measured iteration count rather than
   pixel count, so "half the work" means half the FLOPs (§3.2, zoom 1e2).
3. ~~Cut the per-frame allocation and merge.~~ **Done (§3.7)** — one scratch buffer per worker
   instead of one `Vec` per tile, and a page-locked reusable frame buffer. −18…−20% at zoom 1e0.
4. ~~Fix the fp64 bulb check.~~ **Done (§2.2)** — 2.01x on the kernel, and it helps every CUDA
   path including the scheduler's GPU tiles.
5. **Benchmark and tune above zoom 1e6.** The `hybrid/heterogeneous_deep` group now covers
   1e4–1e15 with solo-backend reference arms, which is where the load-balancing failure at
   1e7+ (78 ms measured against 36 ms available) is visible and fixable.

The architecture itself is sound: classification adapts sensibly with zoom (CPU share rises 3.1%
→ 13.9% as the boundary fills the frame), work stealing functions in both directions, and the
output is correct. It is bottlenecked by one fixed cost, not by its design.

### 3.10 Confirmation from the criterion run on the RTX 3050

The criterion sweep reaches the same verdict independently of the probes, and now with both
scheduler variants on the same GPU:

| Mandelbrot 1080p | zoom 1e0 | zoom 1e2 | zoom 1e4 | best solo backend |
|---|---:|---:|---:|---:|
| heterogeneous (CUDA) | 2.72 | 5.19 | **2.03** | CUDA **1.61** |
| heterogeneous (wgpu) | 3.42 | 5.07 | 3.38 | wgpu 2.26 |
| static 50/50 CPU+wgpu | — | — | 4.95 | — |

**At these shallow zooms every scheduler configuration is slower than simply using the better
GPU alone**, on both backends. The closest case is heterogeneous-CUDA at zoom 1e4 (2.03 vs
1.61 ms, +26%), which is also the zoom where the classifier assigns the CPU nothing at all —
i.e. the scheduler is at its best precisely when it is doing least. That is the §3.1 arithmetic
showing through: its overhead is fixed and its upside is bounded by a kernel term now worth only
~18% of the frame. §3.8 shows this reversing above zoom 1e6, where the GPU loses its f32 path.

The scheduler did get faster in absolute terms (zoom 1e0: 3.60 → 2.72 ms) because its GPU tiles
run `fractal_kernel_tiled_f32`, which inherited the §2.2 fix. But plain CUDA got faster by the
same mechanism, so the *gap* did not close.

---

## 4. Cross-project comparison

Full tables in `bench_results/comparison_report.md`. FractalRendererCpp is C++/CMake with raw
`std::thread` and no SIMD; Fractals-rs is Rust/rayon with `wide` SIMD (SSE width) and no GPU.

### 4.1 The single fairest number

Same fractal, resolution, iteration count, precision (f64), algorithm class (naive), and thread
count (1) — so nothing but kernel quality varies:

| Single-threaded f64 scalar, 1920×1080 | ms | Speedup |
|---|---:|---:|
| ruster | 54.93 | **20.0×** |
| FractalRendererCpp | 1096.38 | 1.00× |

⚠ The two rows come from different sessions — see the cross-run caveat at the top. The same
ruster benchmark read 63.64 ms in an earlier archive, which would give 17.2×. **Quote this as
"roughly 17–20×" until both projects are re-run in one session**; the order of magnitude is
solid, the second digit is not.

This is not a language effect. ruster's kernel has cardioid + period-2 + period-3 bulb rejection
and Brent cycle detection; FractalRendererCpp's has none of them, and additionally re-squares `z`
for its bailout test instead of reusing the update step's intermediates.

### 4.2 Thread scaling (Mandelbrot 1080p, criterion)

| Threads | ruster ms | speedup | efficiency | FractalRendererCpp ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 54.93 | 1.00× | 100% | 1096.38 | 1.00× |
| 2 | 28.38 | 1.94× | 97% | 736.27 | 1.49× |
| 4 | 15.12 | 3.63× | 91% | 392.74 | 2.79× |
| 8 | 8.56 | 6.42× | 80% | 213.93 | 5.12× |
| 16 | 6.85 | 8.02× | 50% | 167.07 | 6.56× |

Efficiency holds through 8 threads (the 5800H's physical core count) then falls off sharply at
16 — the textbook SMT signature: sibling threads share execution units, so the last doubling
buys ~25% rather than another 2×. The *shape* of this curve is a within-run result and is
therefore reliable even though the absolute milliseconds are not comparable across sessions.
FractalRendererCpp scales worse at every point partly because it spawns and joins fresh
`std::thread`s every frame rather than reusing a pool.

### 4.3 Where the comparison legitimately stops

- **GPU, hybrid, perturbation, scheduler**: ruster-only. Neither other project has any of them.
- **SIMD**: ruster (f32x8, AVX2) vs Fractals-rs (f32x4/f64x2, SSE width); FractalRendererCpp has
  none. Comparable in kind, not in width.
- **Fractals-rs full-frame numbers include a fused palette lookup**; ruster's `render()` returns
  raw iteration counts and colorizes separately. Compare render-only against render-only.

---

## 5. Per-stage detail

### 5.1 Stage 0 — CPU scalar baseline (Mandelbrot)

| Resolution | ms | Mpix/s |
|---|---:|---:|
| 800×600 | 1.74 | 276 |
| 1920×1080 | 7.02 | 296 |
| 3840×2160 | 26.47 | 313 |

Mpix/s is nearly flat across resolutions (276 → 313), the expected signature of an
embarrassingly-parallel per-pixel kernel with no shared state.

### 5.2 Stage 1 — SIMD (Mandelbrot)

| Resolution | scalar | f64x4 | f32x8 | f32x8+ILP |
|---|---:|---:|---:|---:|
| 1920×1080 | 7.06 ms | 5.18 ms (1.36×) | 3.95 ms (1.79×) | **3.79 ms (1.87×)** |
| 3840×2160 | 27.44 ms | 19.79 ms (1.39×) | 14.99 ms (1.83×) | **14.17 ms (1.94×)** |

The ILP variant (two interleaved `f32x8` chains) buys **+4.3%** over plain `f32x8` at 1080p and
+5.4% at 4K — real, but at the low end of the 10–25% `CURSOR_OPTIMIZATIONS.md` predicted.
Plausible cause: Mandelbrot's cardioid/bulb rejection already prunes a large share of pixels
before the hot loop runs, leaving less latency for a second dependency chain to hide.

⚠ The `f32x8` figure above is taken from the `simd/render_ilp` group. The `simd/render` group
measured the *same kernel on the same frame* at 6.157 ms in this run — a 56% disagreement
within a single run. That is a bad sample, not a real effect, and it is a useful calibration of
how noisy this laptop is even within one criterion invocation: distrust any single SIMD number
that is not corroborated by a second group or a second run.

### 5.3 Stage 2 — GPU, both backends on the RTX 3050

| Mandelbrot | CPU scalar | CUDA | wgpu | CUDA vs wgpu |
|---|---:|---:|---:|---:|
| 800×600 | 1.74 | **0.479** (3.63×) | 0.570 (3.05×) | 1.19× |
| 1920×1080 | 7.02 | **1.605** (4.37×) | 2.258 (3.11×) | **1.41×** |
| 3840×2160 | 26.47 | **6.016** (4.40×) | 8.783 (3.01×) | 1.46× |

CUDA leads at every resolution now that its kernel no longer carries the fp64 penalty (§2.2) —
the 800×600 tie in the pre-fix run is gone. Peak throughput is **1379 Mpix/s** (CUDA at 4K).
Note the advantage over the CPU scalar baseline is 4.4×, not the 5.2× reported when wgpu was
silently running on the integrated GPU — and against the CPU's own best f32 path it is 2.36×
(§1.3).

This table also supersedes an older revision of this document that reported CUDA Mandelbrot at
16.69 ms and concluded "plain GPU rendering loses to the CPU at every fractal type." That
predated the `--fmad` correction and the f32 fast-path kernel (commits `8e49b4b`, `3929093`).

### 5.4 End-to-end pipeline — the Amdahl ceiling

`colorize()` (histogram equalization + palette LUT) is a CPU stage that no GPU backend touches:

| | render | colorize | pipeline | end-to-end advantage |
|---|---:|---:|---:|---:|
| CPU | 7.02 | 2.34 | 10.02 ms | — |
| CUDA | 1.61 | 2.34 | **4.97 ms** | **2.02×**, not 4.37× |
| wgpu | 2.26 | 2.34 | 4.63 ms | 2.16× |

A fixed ~2.3 ms serial stage caps the achievable end-to-end speedup regardless of how fast the
iteration kernel gets — it is now **47%** of the CUDA pipeline's total. Combined with §2's
finding that the readback is 82% of the render stage itself, the picture is stark: of a 4.97 ms
CUDA pipeline, roughly **0.3 ms is actual fractal iteration** and the other 4.7 ms is data
movement and colouring. The §2.2 fix halved that 0.3 ms and moved the pipeline by 4%.

(wgpu edges CUDA on the *pipeline* row while losing the *render* row — the two pipeline numbers
are 7% apart, inside this machine's noise band, so read them as tied.)

Moving histogram equalization onto the GPU is therefore the highest-leverage remaining
optimization in the entire project — larger than the fp64 bulb fix (§2.2), larger than anything
in the scheduler (§3) — and nothing in the current roadmap covers it. It would also remove the
readback from the critical path entirely if the iteration buffer never had to come back to the
CPU at all.

### 5.5 Stage 2b — Perturbation theory

Sweep centre: seahorse valley `[-0.75, 0.1]`, `max_iter=1000`, 1920×1080.

| Zoom | scalar | perturb | perturb+SA | speedup (perturb) | speedup (SA) |
|---|---:|---:|---:|---:|---:|
| 1e0 | 7.05 | 7.45 | 8.44 | 0.95× | 0.84× |
| 1e3 | 44.96 | 48.05 | 49.50 | 0.94× | 0.91× |
| 1e6 | 20.03 | 48.17 | 24.13 | 0.42× | 0.83× |
| 1e9 | 21.41 | 35.65 | 22.85 | 0.60× | 0.94× |
| 1e12 | 20.28 | 35.16 | 22.24 | 0.58× | 0.91× |

**Perturbation is not a throughput win at any zoom tested** — every ratio is below 1.0, at every
zoom, for both the plain and series-approximation variants. (The previous run showed 1.28× at
zoom 1e0; this run shows 0.95× for the identical benchmark, so that lone above-1.0 entry was
noise, not a real crossover.) Two reasons, both visible in the data:

1. Series approximation skips only 2–36 iterations out of 1000 for this reference point, so it
   buys a ~1–3% head start before falling back to the per-pixel perturbation loop.
2. Plain f64 `render()` is still numerically valid at zoom 1e12 (doubles carry ~15–17 significant
   digits), so this sweep never *needs* perturbation for correctness. Its real payoff — extending
   correct rendering past f64's ~1e15 ceiling via the f128 double-double reference orbit — lies
   at zooms beyond what this sweep reaches.

Perturbation here is a **correctness feature at extreme zoom**, not a throughput feature.
The reference-orbit and series-approximation setup costs are both sub-microsecond and never the
bottleneck; f128 orbits cost ~2× f64 orbits and are still negligible against frame time.

### 5.6 Box-counting dimension of the boundary

4096×4096, zoom 1.0, centre −0.5, `max_iter=1000`. A box counts only if it straddles the
boundary (contains both an in-set and an escaped pixel).

| box px | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| boxes | 6848 | 4370 | 2302 | 1154 | 568 | 246 | 108 | 42 | 16 |

**Dimension 1.10 (R² = 0.9907)** — an excellent power-law fit, well below the known Hausdorff
dimension of exactly 2.0 (Shishikura 1998). Expected: reaching the asymptotic value needs detail
at scales far finer than any achievable render. Viewport choice matters enormously — a zoom-1e5
crop gave only 2–15 straddling boxes across all scales, too few for a meaningful fit.

### 5.7 Area estimate vs the literature constant (~1.50659177)

| max_iter | 100 | 500 | 1000 | 5000 | 20000 |
|---|---:|---:|---:|---:|---:|
| area | 1.548642 | 1.513429 | 1.509575 | 1.506461 | 1.505981 |
| error | +2.791% | +0.454% | +0.198% | −0.009% | −0.041% |

Error shrinks monotonically with `max_iter`, exactly as theory predicts: pixels that would escape
past a too-small cap get misclassified as in-set, inflating the estimate. Resolution (480×360 →
3840×2160) shows no directional bias, staying within ±0.3% — consistent with the two error
sources being independent.

### 5.8 Perturbation glitch-rate statistics

512×512, `max_iter=2000`, zoom 1e9, 15 random reference points: **mean glitch rate 6.67%, stddev
24.94%, range 0–100%**. The spread is the finding — a single reference-point sample (as §5.5 uses)
is not representative. A 100% sample means the reference point itself escaped early, invalidating
perturbation for the whole frame. This is direct quantified motivation for
`render_perturbation_multiref` and `render_perturbation_rebase`.

### 5.9 CPU cache locality — row-major vs Hilbert

| Traversal | 1080p | 4K | LLC miss | L1 miss |
|---|---:|---:|---:|---:|
| row (`render()`) | 6.85 ms | 27.30 ms | 11.69% | 0.65% |
| Hilbert (`render_tiled()`) | 9.52 ms | 34.03 ms | 22.87% | 0.78% |

(Miss rates are from the earlier `perf stat` capture and were not re-measured; the timings are
from the current run. Hilbert is 1.39× slower at 1080p and 1.25× at 4K — the penalty has been
reproduced across every run in this project.)

**Hilbert tiling is worse on every axis** — 39% slower at 1080p and with *worse* cache-miss rates,
contradicting `CURSOR_OPTIMIZATIONS.md`'s predicted 5–15% gain. So this is not "better locality at
the cost of index arithmetic," it is worse at both. Likely cause: an 8 MB f32 buffer sits
comfortably in L3 with hardware prefetching already handling row-major access well, while
`hilbert::d2xy`'s per-pixel bit-interleaving is pure added cost.

### 5.10 GPU occupancy / warp divergence (Nsight Compute)

| Metric | zoom 1 (whole set) | zoom 1e6 (seahorse) |
|---|---:|---:|
| `sm__warps_active` (occupancy) | 48.98% | 60.70% |
| warp efficiency (/32) | 18.48 (57.8%) | 19.66 (61.4%) |
| L1/tex hit rate | 50.06% | 50.28% |

This **reverses** the hypothesis the scheduler's design rests on: the deep-zoom, boundary-heavy
view shows *higher* occupancy and *better* warp efficiency, not worse. The measurement cannot
settle the question either way, though — `ncu` profiles one whole-frame launch averaged over all
warps, while classification operates at 16–128 px tile granularity, so a few genuinely divergent
tiles are diluted by large uniform regions.

One unambiguous result: **occupancy is register-limited**, not warp- or shared-memory-limited —
`launch__occupancy_limit_registers` caps blocks/SM at 5 (46 registers/thread against Ampere's
65536-register file), well under the 16 the other limits would allow. This explains the 50–60%
ceiling on both viewports.

### 5.11 Energy efficiency

CPU (16 threads): 100.35 GFLOPS at 33.82 W = **2.97 GFLOPS/W**.
GPU: 54.37 GFLOPS at 13.49 W = **4.03 GFLOPS/W**.

The GPU is ~36% more energy-efficient per FLOP despite being slower in absolute terms on the
fractals measured there — its power draw is less than half the CPU's. "Faster" and "more
efficient" are different questions on this hardware. Full data in `results/energy.md`.

---

## 6. Stale results removed in this revision

For traceability, the following claims from the previous revision were **measured again and did
not hold**, and have been replaced above:

| Old claim | Status |
|---|---|
| **"wgpu is the fastest backend (1.79 ms)"** | **Artifact of wgpu running on the integrated GPU. On the RTX 3050 wgpu is 2.26 ms and CUDA wins at 1.61 ms** |
| **"The GPU is 5.2× the CPU baseline"** | **4.37× vs the f64 baseline; 2.36× vs the CPU's own best f32 path; 2.02× end-to-end** |
| "CUDA Mandelbrot 1080p = 16.69 ms" | Superseded — 1.61 ms after the f32 kernel, `--fmad` and fp64-bulb fixes |
| "Plain GPU loses to CPU at every fractal type" | False for Mandelbrot (4.37×) |
| "The CUDA kernel is 23% faster than wgpu's" | Wrong — from a single run. Pre-fix, wgpu's kernel was ~2× *faster*; post-fix the two are within 3% and CUDA wins the frame on readback |
| wgpu kernel time derived by subtracting a standalone `readback()` | Invalid method — produced a negative kernel time on NVIDIA. Now measured directly (§2) |
| "The fp64 bulb check is a ~15% available win, not applied" | **Applied.** Kernel 2.01× faster, frame −15%, output bit-identical (§2.2) |
| "The readback is 69% of the CUDA frame" | Now **82%** — halving the kernel raised the readback's share |
| Correctness check using a zoom-3000 "bulb edge" viewport | Worthless — that view is deep inside the bulb (radius 0.0737 vs half-width 0.0007). Replaced with three views centred on the boundary arc |
| "Static 50/50 hybrid beats the adaptive scheduler on 3 of 4 fractals" | Benchmark deleted; adaptive now compared against plain single-backend rendering |
| "CPU tiles are routed to the exact scalar path" | Fixed — `SchedulerConfig::simd_cpu_tiles` defaults on |
| "`fractal_kernel_tiled` has no f32 fast path" | Fixed — `fractal_kernel_tiled_f32` + `gpu_tiles_f32`, defaults on |
| Scheduler tunable sweep (`tile_size` 16–128 × `threshold` 10–500) | Stale — thresholds are now normalized to ~[0.001, 0.5]; the sweep predates the rewrite and has not been re-run |
| Time-to-first-paint table | Not re-measured this revision; the CUDA cold-start figures predate the f32 kernel |
