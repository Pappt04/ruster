# ruster — State of the Application

A complete description of every compute path in the renderer, the pipeline each one runs, and
how they compare against each other and against two external implementations.

**Scope of the measurements.** Mandelbrot, 1920×1080, `max_iter = 1000` unless stated. Criterion
means from `bench_results/criterion_20260808_200032_8d2dbdc_rtx3050_f32bulb` (full suite) and
`bench_results/criterion_20260808_210551_fc5ca0d_sched_improved` (scheduler groups). Every
number in this document was read out of those archives, not transcribed from notes.

**Hardware.** AMD Ryzen 7 5800H (8 cores / 16 threads, AVX2) · NVIDIA RTX 3050 Laptop
(`sm_86`, CUDA 12.0, driver 580.173.02) · **PCIe 3.0 x8** link (7.9 GB/s theoretical) · an AMD
Radeon Vega (Renoir) iGPU is also present but is no longer used by any backend.

**Two rules for reading any table here.**

1. **Never compare numbers across runs.** Identical CPU benchmarks moved 15–30% between
   archives with no code change, from thermal state alone. Within-run comparisons are sound.
2. **Mind the precision column.** The fastest paths are f32; the baseline is f64. An f32-vs-f64
   comparison measures two different computations.

---

## 1. Architecture at a glance

Four modules. Every compute path produces the same intermediate — an `IterBuf`
(`Vec<f32>`, one smooth escape-time value per pixel) — which is then coloured once.

```
  Viewport {center, zoom, width, height}
        │
        ▼
  pixel_grid()  ──►  PixelGrid {re_start, im_start, re_step, im_step}
        │            (precomputed once per frame; never call
        │             pixel_to_complex() inside a pixel loop)
        ▼
  ┌───────────────────── ONE OF THE COMPUTE PATHS (§3) ─────────────────────┐
  │  CPU scalar · CPU SIMD · CPU tiled · Mariani-Silver · perturbation      │
  │  wgpu · CUDA · static hybrid · heterogeneous scheduler                  │
  └────────────────────────────────────────────────────────────────────────┘
        │
        ▼
  IterBuf : Vec<f32>            ← the universal intermediate, width*height
        │
        ▼
  colorize()  (gui/color.rs, CPU, rayon)
        │  histogram of escaped pixels → CDF → equalised t ∈ [0,1]
        │  → 8192-entry palette LUT      (in-set pixels → BLACK)
        ▼
  Vec<Color32>  →  egui ColorImage  →  GPU texture  →  screen
```

`colorize` is CPU-only and backend-independent. It costs **2.34 ms** at 1080p and is the
single largest fixed cost in the whole pipeline — see §6.

### 1.1 Fractal kernels

| Fractal | id | Iteration | FLOPs/iter | Interior shortcut | Cycle detection |
|---|---:|---|---:|---|---|
| Mandelbrot | 0 | `z ← z² + c`, `z₀ = 0` | 8 | cardioid + period-2 + period-3 bulb | Brent |
| Julia | 1 | `z ← z² + c`, `z₀ = pixel` | 8 | — | Brent |
| Newton | 2 | `z ← z − f/f′`, `f = z³−1` | 25 | — | step-size < 1e-12 |
| Nova | 3 | Newton + `c` perturbation | 27 | — | step-size < 1e-12 |

Escape radius² = 4 everywhere. Smooth colouring is
`i + 1 − log₂(log|zₙ|)`, fed `zr²+zi²` directly rather than recomputed.

### 1.2 The three precision regimes

One constant decides which path everything takes:

| Zoom | Regime | CPU | CUDA | wgpu |
|---|---|---|---|---|
| `< 1e6` | f32 fast path | `render_simd_f32_ilp` | `fractal_kernel_f32` | f32 (always) |
| `1e6 … 1e12` | f64 | `render_simd` (f64x4) / scalar | `fractal_kernel` | f32 — **cannot** go higher |
| `> 1e12` | f128 reference orbit | `compute_reference_orbit_f128` + perturbation | perturbation kernel | perturbation |

`F32_PRECISION_THRESHOLD = 1e6`, `F128_ZOOM_THRESHOLD = 1e12`. **This boundary explains most
of the results in this document**: below 1e6 the GPU runs f32 and is 20× the CPU; above it the
GPU falls back to f64, which Ampere GeForce executes at 1/64 the fp32 rate, and the CPU becomes
competitive or better. WGSL has no f64 type at all, so wgpu is f32 unconditionally and is
simply wrong above ~1e6 unless perturbation carries it.

---

## 2. The interactive pipeline (what the app actually runs)

`gui/render.rs` runs a background worker; the UI thread never computes.

```
UI thread                          RenderWorker (background)
─────────                          ─────────────────────────
 mouse/key
   │ mutates Viewport
   │ needs_render = true
   ▼
 request_render() ──mpsc──►  drain channel, keep only the NEWEST request
                                   │      (stale frames are discarded, not queued)
                                   ▼
                             ┌── cache key match? ────► recolor only ──► Final
                             │
                             ├── axis-aligned integer pan? ──► shift_and_fill()
                             │      reuses the overlapping region, computes only
                             │      the newly exposed strip ──► Final (no preview)
                             │
                             └── otherwise: full recompute
                                    │
                                    ├─► ½-resolution PREVIEW ──► Quality::Preview
                                    │
                                    └─► full resolution ──────► Quality::Final
                                   │
   ◄────────────mpsc──────────────┘  ColorImage
 upload as GPU texture, draw
```

Two properties worth noting: the worker **drains** its channel so a fast-moving camera never
builds a backlog, and `busy` only clears on `Quality::Final`, so a preview keeps the UI polling
rather than looking finished.

**Dispatch order inside the worker** (first match wins):
`perturbation+SA` → `perturbation+multiref` → `perturbation` → `Mariani-Silver` →
`neighbour-capped` → `SIMD` (f32 ILP below 1e6, else f64x4) → `scalar rayon`.

The CUDA path is single-pass — no preview stage — and pays a one-time `CudaFractal::new()` cost
(PTX JIT + context init) on the worker's first request.

---

## 3. Compute paths — pipeline and cost

Every path below produces an identical-shaped `IterBuf`. They differ in *how*.

### 3.1 CPU scalar — the baseline

```
rayon par_chunks over rows
  └─ per pixel: cardioid/bulb reject → z²+c loop (f64) → Brent cycle check → smooth_iter
```

| Resolution | ms | Mpix/s |
|---|---:|---:|
| 800×600 | 1.74 | 276 |
| 1920×1080 | **7.02** | 296 |
| 3840×2160 | 26.47 | 313 |

Throughput is flat across resolutions — the expected signature of an embarrassingly-parallel
per-pixel kernel with no shared state.

**Thread scaling** (1080p):

| Threads | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| ms | 54.93 | 28.38 | 15.12 | 8.56 | 6.85 |
| speedup | 1.00× | 1.94× | 3.63× | 6.42× | **8.02×** |
| efficiency | 100% | 97% | 91% | 80% | **50%** |

Efficiency holds through 8 threads — the physical core count — then halves at 16. That is the
textbook SMT signature: sibling threads share execution units, so the last doubling buys 25%
rather than another 2×.

### 3.2 CPU SIMD

```
rayon rows → 8 (f32) or 4 (f64) pixels per vector lane
  └─ bulb_precheck_x8 → vectorised z²+c → per-lane escape mask → smooth_iter
     (no cycle detection in the f32 path — the tradeoff for width)
```

| Variant | 800×600 | 1920×1080 | 3840×2160 | vs scalar @1080p |
|---|---:|---:|---:|---:|
| scalar (f64) | 1.80 | 7.06 | 27.44 | 1.00× |
| f64x4 | 1.31 | 5.18 | 19.79 | 1.36× |
| f32x8 | 1.01 | 3.95 | 14.99 | 1.79× |
| **f32x8 + ILP** | **1.02** | **3.79** | **14.17** | **1.87×** |

ILP interleaves two independent `f32x8` dependency chains; it is bit-identical to plain `f32x8`
and worth ~4%. The gain is at the low end of what was predicted, because Mandelbrot's bulb
rejection already prunes many pixels before the hot loop, leaving less latency to hide.

⚠ In this run the `simd/render` group reported `f32x8` at 6.157 ms while `simd/render_ilp`
reported the *same kernel on the same frame* at 3.954 ms — a 56% disagreement inside one
criterion invocation. The 3.95 figure is used above. This is a useful calibration of how noisy
this machine is: distrust any single SIMD number not corroborated by a second group.

### 3.3 CPU traversal order

```
render()        row-major, rayon over row chunks
render_tiled()  64×64 tiles, either row-order or Hilbert-curve order
```

| Traversal | 800×600 | 1920×1080 | 3840×2160 | LLC miss | L1 miss |
|---|---:|---:|---:|---:|---:|
| rows | 1.83 | **6.85** | **27.30** | 11.69% | 0.65% |
| Hilbert | 3.12 | 9.52 | 34.03 | 22.87% | 0.78% |

**Hilbert tiling is worse on every axis** — 39% slower at 1080p *and* with worse cache-miss
rates, so it is not "better locality at the cost of index arithmetic", it is worse at both. An
8 MB f32 buffer sits comfortably in L3 with hardware prefetching already handling row-major
access well, while `hilbert::d2xy`'s per-pixel bit-interleaving is pure added cost. (Miss rates
are from an earlier `perf stat` capture; the timings are current.)

### 3.4 Perturbation theory

```
compute_reference_orbit(c₀)        f64  — or _f128 (double-double Dd) above zoom 1e12
        │  Zₙ for n = 0..max_iter
        ▼
compute_series_approx(orbit)       optional: Taylor coefficients to skip early iterations
        │
        ▼
per pixel:  δ = c − c₀
   εₙ₊₁ = 2·Zₙ·εₙ + εₙ² + δ            ← all f64, small numbers stay representable
   escape test on  zₙ = Zₙ + εₙ
   glitch test: |ε|² > |Z|²·1e-6  →  fall back to the exact scalar kernel for that pixel
```

| Zoom | scalar | perturb | perturb+SA | speedup | speedup (SA) |
|---|---:|---:|---:|---:|---:|
| 1e0 | 7.05 | 7.45 | 8.44 | 0.95× | 0.84× |
| 1e3 | 44.96 | 48.05 | 49.50 | 0.94× | 0.91× |
| 1e6 | 20.03 | 48.17 | 24.13 | 0.42× | 0.83× |
| 1e9 | 21.41 | 35.65 | 22.85 | 0.60× | 0.94× |
| 1e12 | 20.28 | 35.16 | 22.24 | 0.58× | 0.91× |

**Perturbation is not a throughput win at any tested zoom** — every ratio is below 1.0. Reasons,
both visible in the data: series approximation skips only 2–36 iterations of 1000 for this
reference point, and plain f64 `render()` is still numerically valid at zoom 1e12, so the sweep
never actually *needs* perturbation. Its real payoff — correct rendering past f64's ~1e15
ceiling via the f128 orbit — lies beyond what this sweep reaches.

**Perturbation here is a correctness feature at extreme zoom, not a throughput feature.**

Setup costs are negligible at any zoom: reference orbits are ~0.4 µs (f64) / ~1 µs (f128), and
series-approximation coefficients ~0.3 µs, against frame times of 7–50 ms.

Glitch behaviour is why the multi-reference and rebasing variants exist: across 15 random
reference points at zoom 1e9 the glitch rate averaged 6.67% with a **24.94% standard deviation
and a full 0–100% range**. A single reference point is not representative — a 100% sample means
the reference itself escaped early, invalidating the whole frame relative to it.

### 3.5 wgpu (Vulkan compute)

```
Uniforms {re_start, im_step, …}  ──write_buffer──►  uniform buffer
        │
        ▼
compute pass, workgroup 16×16, dispatch ⌈w/16⌉ × ⌈h/16⌉
        │  main()        whole frame
        │  main_tiled()  tile-descriptor list, workgroup_id.z selects the tile
        ▼
storage buffer ──copy_buffer_to_buffer──► MAP_READ buffer
        ──map_async + poll(Wait)──► memcpy into Vec<f32>
```

| Resolution | ms | Mpix/s |
|---|---:|---:|
| 800×600 | 0.570 | 843 |
| 1920×1080 | **2.258** | 918 |
| 3840×2160 | 8.783 | 944 |

**Always f32, every fractal, no exceptions** — WGSL has no f64 type. Frame breakdown at 1080p:
kernel **0.29 ms**, readback **2.14 ms** (~90%). The readback does a device→device staging copy
into a separate `MAP_READ` buffer *before* the DMA, then map + memcpy + unmap; that extra
staging step is ~0.8 ms of pure overhead and is the entire gap to CUDA.

**Known defect:** `fractal.wgsl`'s Newton/Nova spell out `.../denom` four times per iteration
where the CPU and CUDA versions factor the same algebra into two divides. Algebraically
identical, twice the divisions.

### 3.6 CUDA

```
build.rs: nvcc --ptx -arch=sm_86 -O3 --ftz=false --prec-div=true --fmad=false
        │  (--fmad=false keeps CUDA's rounding identical to Rust's for CPU/GPU bit-exactness)
        ▼
fractal_kernel      f64, Morton (Z-order) dispatch, 1-D grid of 16×16 blocks
fractal_kernel_f32  f32 fast path, Mandelbrot/Julia below zoom 1e6
fractal_kernel_tiled[_f32]   tile-descriptor list, blockIdx.z selects the tile
fractal_perturb_kernel       reference orbit uploaded per call
        ▼
dtoh_sync_copy  (one PCIe DMA; readback_into() for a caller-owned/pinned destination)
```

| Resolution | ms | Mpix/s | vs CPU scalar |
|---|---:|---:|---:|
| 800×600 | 0.479 | 1001 | 3.63× |
| 1920×1080 | **1.605** | **1292** | **4.37×** |
| 3840×2160 | 6.016 | 1379 | 4.40× |

Frame breakdown at 1080p: kernel **0.28 ms**, readback **1.33 ms** (~82%).

Two design details that were measured rather than assumed:

- **Morton padding is free.** `morton_cfg` pads to the enclosing power-of-two *square*, so
  1920×1080 launches 4,194,304 threads for 2,073,600 pixels — 2.02× oversubscription. Measured
  against the identical math on a plain 2-D grid: **+1.9% median over 7 paired runs**, with two
  runs favouring Morton. The dead threads land in one contiguous L-shaped region and whole
  blocks retire immediately.
- **The fp64 bulb check used to cost half the kernel.** `mandelbrot_f32` called the f64
  `in_period3_bulb`; at Ampere's 1/64 fp64 rate those 8 operations cost as much as the entire
  1000-iteration f32 loop. Replacing it with an f32 predicate made the kernel **2.01× faster**
  (0.910 → 0.452 ms) with **zero differing pixels** across seven viewports including three
  centred on the period-3 boundary arc at zoom 1e2/1e4/1e6.

### 3.7 Static hybrid (CPU + wgpu)

```
frame split into fixed top/bottom halves
  ├─ top half  → CPU  (rayon)          ┐ concurrent
  └─ bottom half → wgpu compute        ┘
concatenate
```

1080p Mandelbrot: **4.95 ms**. Simple, no classification, no work stealing — and on Newton
(18.44 ms) and Nova (61.61 ms) it is the fastest path in the entire project, because those
fractals are the ones where CPU and GPU are closest.

### 3.8 Heterogeneous scheduler (the thesis contribution)

```
partition_frame()   recursive corner-sampling, 128px cells down to 16px
   │  4 corner pixels per tile; spread = (max−min)/max_iter
   │  spread < threshold → GPU tile (coherent)      | no GPU prepass needed
   │  else subdivide; at min_tile → CPU tile (boundary)
   ▼
reserve steal_reserve_frac (20%) of GPU tiles into a steal queue
   ▼
rayon::scope ─┬─ N CPU workers: pop cpu_queue, then steal_queue
              │     render_cpu_tile_into() → per-worker flat scratch + (tile,offset) index
              │
              └─ calling thread: dispatch_tiled[_f32] → optional steal mop-up dispatch
                                 → readback_into(frame buffer)
   ▼
scatter CPU tiles over the readback  →  ThresholdController.update(gpu_ms, cpu_ms)
```

Defaults: `max_tile_size 128`, `min_tile_size 16`, `steal_reserve_frac 0.20`,
`min_steal_tiles 64`, `simd_cpu_tiles true`, `gpu_tiles_f32 true`.

**Shallow zoom (1e0–1e4), against the better solo backend:**

| zoom | heterogeneous (CUDA) | heterogeneous (wgpu) | best solo |
|---|---:|---:|---:|
| 1e0 | 2.60 | 4.20 | CUDA 1.61 |
| 1e2 | 5.07 | 4.80 | CUDA 1.61 |
| 1e4 | 2.21 | 2.96 | CUDA 1.61 |

**Deep zoom (`hybrid/heterogeneous_deep`, all three arms measured back to back):**

| zoom | CUDA | CPU | hybrid | ideal split | verdict |
|---|---:|---:|---:|---:|---|
| 1e4 | **4.25** | 91.84 | 4.55 | 4.06 | 0.93× lose |
| 1e5 | **4.37** | 125.59 | 4.71 | 4.22 | 0.93× lose |
| **1e6** | 218.09 | 117.19 | **68.58** | 76.23 | **1.71× WIN** |
| 1e7 | 92.05 | 80.93 | **78.97** | 43.07 | 1.02× win |
| 1e9 | 91.86 | 82.27 | **78.49** | 43.40 | 1.05× win |
| 1e12 | 91.47 | 103.80 | **78.51** | 48.62 | 1.16× win |
| 1e15 | 91.46 | 81.30 | **78.47** | 43.04 | 1.04× win |

**The scheduler loses at every zoom below 1e6 and wins at every zoom above it.** The boundary is
exactly `F32_PRECISION_THRESHOLD`. At 1e6 it beats not only the better solo backend but the
*ideal proportional split* (68.6 vs 76.2 ms) — corner-sampling routes expensive tiles to the CPU
rather than dividing work blindly, which is the design claim demonstrated.

---

## 4. Why shallow zoom cannot be won

Of a 1.61 ms CUDA frame, **1.33 ms is a fixed full-frame readback** and only 0.28 ms is kernel.
`readback` copies the whole `width×height` buffer regardless of how few tiles the GPU rendered.
So moving work to the CPU can only shrink the 0.28 ms part:

| zoom | GPU's pixel share | best possible kernel saving | scheduler overhead |
|---|---:|---:|---:|
| 1e0 | 96.9% | 0.01 ms | ~0.10 ms machinery + contention |
| 1e2 | 86.1% | 0.04 ms | CPU becomes critical path |

Four supporting measurements:

- **Readback scales with size** (50% of bytes = 51.6% of time), so partial readback would help —
  but splitting the same bytes per tile costs **2.60× (135 copies)** to **5.93× (825 copies)** a
  single copy. Partial readback therefore requires a contiguous row range, not scattered tiles.
- **The link is the ceiling.** PCIe 3.0 x8 = 7.9 GB/s theoretical; measured 6.7 GB/s page-locked
  (85%). The copy can only get smaller, never faster.
- **The backends are not independent.** Saturating 16 CPU threads slows plain `cuda.render()` by
  up to **33%** — the readback's host half competes for memory bandwidth.
- **Two suspects were tested and cleared:** tiled dispatch is 8–12% *faster* than the Morton
  whole-frame launch, and rayon workers start within 0.06 ms even with the caller blocked in CUDA.

Ceiling from these constants:

| scenario | hybrid | vs GPU-only |
|---|---:|---:|
| before the improvements below | 2.33 ms | 0.69× |
| + pooled tile buffers | 1.28 ms | 1.26× |
| + partial readback, row-contiguous, pinned | 1.23 ms | 1.31× |
| …with measured contention | 1.34 ms | **1.20×** |
| perfect: zero overhead, no contention | 1.08 ms | 1.49× (unreachable) |

---

## 5. Cross-project comparison

| | ruster | Fractals-rs | FractalRendererCpp |
|---|---|---|---|
| Language | Rust | Rust | C++17 |
| Threading | rayon pool | rayon pool | raw `std::thread`, spawn per frame |
| SIMD | f32x8 / f64x4 (AVX2) | f32x4 / f64x2 (SSE) | none |
| GPU | wgpu + CUDA | none | none (OpenGL is display-only) |
| Perturbation / f128 | yes | no | no |
| Scheduler | yes | no | no |

### 5.1 The single fairest number

Same fractal, resolution, iteration count, precision (f64), algorithm class, and thread count —
so nothing but kernel quality varies:

| 1 thread, f64 scalar, 1080p | ms | speedup |
|---|---:|---:|
| ruster | 54.93 | **20.0×** |
| FractalRendererCpp | 1096.38 | 1.00× |

⚠ The two rows come from different sessions; the same ruster benchmark read 63.64 ms in an
earlier archive, giving 17.2×. **Quote this as "roughly 17–20×"** until all three projects are
re-run in one session.

This is not a language effect. ruster's kernel has cardioid + period-2 + period-3 bulb rejection
and Brent cycle detection; FractalRendererCpp's has none of them and additionally re-squares `z`
for its bailout test instead of reusing the update step's intermediates.

### 5.2 Full-frame render, 1080p

| Implementation | ms | Prec. | note |
|---|---:|---|---|
| ruster CUDA | **1.61** | f32 | |
| ruster wgpu | 2.26 | f32 | |
| ruster SIMD f32x8+ILP | 3.79 | f32 | fastest CPU anywhere |
| ruster SIMD f64x4 | 5.18 | f64 | |
| Fractals-rs "fast" | 5.62 | f32 | includes a fused palette lookup |
| ruster CPU scalar | 7.02 | f64 | baseline |
| Fractals-rs "high" | 16.00 | f64 | includes a fused palette lookup |
| FractalRendererCpp | 247.87 | f64 | 8 threads, hardcoded |

Fractals-rs numbers include colour mapping that ruster's `render()` does separately — compare
render-only against render-only where possible.

### 5.3 Thread scaling

| Threads | ruster | speedup | Fractals-rs | speedup | FractalRendererCpp | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 54.93 | 1.00× | 18.07 | 1.00× | 1096.38 | 1.00× |
| 2 | 28.38 | 1.94× | 9.06 | 1.99× | 736.27 | 1.49× |
| 4 | 15.12 | 3.63× | 5.18 | 3.49× | 392.74 | 2.79× |
| 8 | 8.56 | 6.42× | 3.15 | 5.74× | 213.93 | 5.12× |
| 16 | 6.85 | 8.02× | 2.58 | 7.00× | 167.07 | 6.56× |

(Fractals-rs is f32 SIMD here against ruster's f64 scalar — the *shapes* are comparable, the
absolute times are not.) Both Rust projects scale nearly identically. FractalRendererCpp scales
worse at every point, partly because it spawns and joins fresh threads every frame instead of
reusing a pool.

### 5.4 What cannot be compared

GPU, hybrid, perturbation and the scheduler are ruster-only. SIMD is comparable in kind but not
in width. FractalRendererCpp is `double`-only project-wide, so it has no precision axis at all.

---

## 6. The end-to-end picture

`colorize()` is a CPU stage no backend touches:

| | render | colorize | pipeline | end-to-end |
|---|---:|---:|---:|---:|
| CPU | 7.02 | 2.34 | 10.02 ms | — |
| CUDA | 1.61 | 2.34 | **4.97 ms** | **2.02×**, not 4.37× |
| wgpu | 2.26 | 2.34 | 4.63 ms | 2.16× |

**Of a 4.97 ms CUDA pipeline, roughly 0.3 ms is actual fractal iteration.** The other 4.7 ms is
data movement and colouring. The GPU advantage stated three honest ways:

| comparison | ratio |
|---|---:|
| CUDA vs f64 CPU scalar baseline | 4.37× |
| CUDA vs the CPU's own best f32 path | **2.36×** |
| CUDA vs CPU end-to-end | 2.02× |

The middle row is the precision-controlled kernel comparison; the bottom row is what a user
experiences.

---

## 7. Where the remaining performance is

Ranked by measured headroom, not by appeal:

| # | Change | Effect |
|---|---|---|
| 1 | **Move `colorize()` onto the GPU** | pipeline 3.95 → 1.81 ms = **2.18×** |
| 2 | **Render straight to a texture, no readback** | pipeline → 0.48 ms = **8.2×** |
| 3 | Partial (row-contiguous) readback | render stage ~1.3× |
| 4 | Tune the load balancer above zoom 1e6 | 78 ms measured against 43 ms available |
| 5 | Iteration-weighted rather than pixel-weighted balancing | mitigates the 1e2 critical-path inversion |
| 6 | Fix WGSL Newton/Nova division count | Newton is wgpu's worst fractal |

**A perfect heterogeneous scheduler is worth 7% end-to-end. Moving colorize to the GPU is worth
118%.** The scheduler optimises the small half of a pipeline dominated by data movement — which
is an argument about where effort belongs, not that the scheduler is worthless: it wins at every
zoom from 1e6 to 1e15, and would win broadly on hardware where CPU and GPU sit closer together.

### Already applied

| Change | Effect |
|---|---|
| f32 period-3 bulb check in `mandelbrot_f32` | kernel **2.01×**, frame −15%, bit-identical |
| Pooled per-worker tile scratch | −18…−20% on the scheduler at zoom 1e0 |
| `PinnedBuf` + `readback_into` | −7.5% on the readback |
| f32 fast-path CUDA kernel + `--fmad` correction | CUDA 16.69 → 1.61 ms since the first revision |

---

## 8. Reproducing everything

```bash
cargo run --release                                   # the app
cargo bench --bench fractal_bench --features cuda      # full suite
uv run scripts/aggregate_bench.py                      # cross-project report

# diagnostics — every measurement in this document comes from one of these
cargo run --release --example adapters                            # which GPU wgpu selects
cargo run --release --example sched_overhead                      # scheduler overhead, work removed
cargo run --release --features cuda --example gpu_probe           # kernel vs readback split
cargo run --release --features cuda --example ptx_variants        # kernel ablation + pixel diff
cargo run --release --features cuda --example readback_probe      # readback scaling, pinning
cargo run --release --features cuda --example sched_probe         # scheduler vs solo backends
cargo run --release --features cuda --example sched_deepzoom      # across the f32/f64 boundary
cargo run --release --features cuda --example tiled_dispatch_probe # tiled vs Morton geometry
```

**Deeper detail:** `results/summary.md` (per-stage analysis, methodology, corrections log),
`bench_results/comparison_report.md` (cross-project tables),
`../other-projects/BENCHMARKING.md` (how to run all three projects and what is comparable).

---

## 9. Correctness and caveats

- **Bit-exactness.** With `simd_cpu_tiles` and `gpu_tiles_f32` off, the scheduler matches plain
  CPU `render()` to within 3–184 pixels of 2,073,600 — pre-existing GPU-vs-CPU fp64
  non-determinism on chaotic pixels near a cycle-detection threshold, documented in
  `classifier`'s module doc. With both defaults on (f32 paths) the divergence is larger and
  intentional.
- **`render_heterogeneous` and `render_heterogeneous_into` are byte-identical.**
- **wgpu ran on the wrong GPU until 2026-08-08.** The NVIDIA stack was installed compute-only,
  so there was no Vulkan ICD and wgpu silently used the integrated AMD Vega. Every wgpu number
  in archives up to `criterion_20260725_215202` came from that iGPU and must not be compared
  against a CUDA number. Verify with `cargo run --release --example adapters`.
- **This report covers Mandelbrot.** Julia, Newton and Nova entries in the latest archive are
  carried over from an earlier run and were not re-measured.
- **Thermal drift.** Identical benchmarks move 15–30% between sessions on this laptop. Treat
  sub-20% cross-run differences as noise; within-run and within-process ratios are sound.
