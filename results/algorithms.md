# Algorithm inventory — what runs where, and why

Every algorithm in the renderer, the pipeline stage it belongs to, the condition
that selects it, and the reason it exists. Where it has been measured, the result
is given — including the ones that lost.

Intended as the source for the thesis's implementation chapter, so each entry
answers *why this and not the obvious alternative*.

Constants referenced throughout:

| Constant | Value | Meaning |
|---|---|---|
| `F32_PRECISION_THRESHOLD` | `1e6` | above this zoom, f32 can no longer separate adjacent pixels |
| `F128_ZOOM_THRESHOLD` | `1e12` | above this, the f64 reference orbit itself is insufficient |
| `ESCAPE_RADIUS_SQ` | `4.0` | bailout radius² (radius 2) |
| `GLITCH_SQ` | `1e-6` | perturbation glitch trigger: `|ε|² > |Z|²·1e-6` |
| `MAX_REFS` | `8` | reference-orbit cap in multi-reference mode |
| `LUT_SIZE` | `8192` | palette lookup entries |

---

## Stage 0 — Viewport → pixel grid

**Algorithm:** precompute `(re_start, im_start, re_step, im_step)` once per frame;
map pixel `(x,y)` to `re_start + x·re_step`.

**Why:** the naïve alternative recomputes the mapping per pixel from
centre/zoom/aspect, which is several extra flops and a dependent chain in the
innermost loop. This is a standing rule — never call
`viewport.pixel_to_complex()` inside a pixel loop.

**Note for the thesis:** the grid samples pixel *centres*
(`re_start = min_edge + 0.5·re_step`). Both comparison projects sample pixel
*edges*. Half a pixel of difference, no effect on cost — but it matters when
claiming two renderers drew the same image, and it is the reason
`tri-compare`'s equivalence check normalises phase before comparing.

---

## Stage 1 — Choosing a rendering strategy

`gui/render.rs` picks in this order; **first match wins**:

| Order | Strategy | Selected when | Why it is first/last |
|---|---|---|---|
| 1 | perturbation + series approximation | user enables both | most specialised; only valid for Mandelbrot |
| 2 | perturbation + multi-reference | user enables both | glitch-robust variant |
| 3 | perturbation | user enables | needed for correctness past f64's reach |
| 4 | Mariani–Silver | user enables | work-skipping, sequential |
| 5 | neighbour-capped | user enables | work-bounding |
| 6 | SIMD | `simd` feature on | the default fast path |
| 7 | scalar rayon | fallback | the reference implementation |

The ordering is *specificity*, not speed: the narrow, correctness-critical paths
must not be shadowed by the general fast one.

---

## Stage 2 — The per-pixel kernel

### 2.1 Interior shortcuts (Mandelbrot only)

Three closed-form tests run before iteration:

1. **Main cardioid**: `q(q + (cr − ¼)) < ¼·ci²` where `q = (cr − ¼)² + ci²`
2. **Period-2 bulb**: `(cr + 1)² + ci² < 1/16`
3. **Period-3 bulbs**: distance² to each of two known nuclei vs. known radius²

**Why:** these regions are provably inside the set, so they would otherwise run
the full `max_iter` loop — the most expensive possible outcome. They are the
single largest algorithmic advantage this renderer has over the comparison
projects, neither of which implements any of them. §Cross-project shows a ~20×
single-thread gap that is almost entirely attributable to these plus cycle
detection.

**Measured cost of getting this wrong:** the CUDA f32 kernel originally
evaluated test 3 in **f64**. On Ampere GeForce (fp64 at 1/64 rate) that single
line cost **half the entire kernel** — 0.910 → 0.452 ms once made f32, with
**zero differing pixels** across seven viewports including three centred on the
period-3 boundary arc. A correctness-neutral 2.01×.

### 2.2 Cycle detection — Brent's algorithm

Track a saved reference point; if the orbit returns within `1e-20` of it, the
point is periodic and therefore interior. The comparison interval doubles
(8, 16, 32 … capped at 512).

**Why Brent and not Floyd:** Floyd's tortoise-and-hare advances two iterators,
costing an extra complex multiply per step. Brent's needs one iterator plus a
saved point and a counter, so the per-iteration cost is a subtract and compare.
In a loop this hot that difference is the whole reason it is affordable.

**Why it is absent from the f32 SIMD and f32 CUDA kernels:** the saved-point
comparison is a per-lane branch, which breaks SIMD lockstep and causes warp
divergence on the GPU. The trade is deliberate: cycle detection is *most*
valuable at deep zoom, and deep zoom is exactly where those f32 paths are not
used anyway.

### 2.3 Smooth (continuous) colouring

`i + 1 − log₂(log|z_n|)` computed from the escape magnitude.

**Why:** integer escape counts produce visible banding. The implementation is
fed `zr² + zi²` directly from the loop rather than recomputing the magnitude —
a rule enforced here because it was violated once.

---

## Stage 3 — Backend selection and precision

This is the decision the whole thesis turns on.

| Zoom | CPU | CUDA | wgpu |
|---|---|---|---|
| `< 1e6` | `render_simd_f32_ilp` (8×f32, AVX2) | `fractal_kernel_f32` | f32 |
| `1e6 – 1e12` | `render_simd` (4×f64) | `fractal_kernel` (f64) | **f32 — incorrect here** |
| `> 1e12` | perturbation + f128 orbit | perturbation kernel | perturbation |

**Why the threshold exists:** f32 carries ~7 decimal digits. At zoom 1e6 the
pixel step approaches that resolution, so adjacent pixels stop being
distinguishable and the image is wrong — fast, but wrong.

**Why this creates the scheduler's win regime:** consumer GPUs execute fp64 at a
small fraction of their fp32 rate (1/64 on this Ampere part), while a CPU pays a
much smaller penalty for f64. So crossing the threshold changes the *CPU/GPU
throughput ratio* by more than an order of magnitude. Below it the GPU is ~20×
the CPU and nothing is worth scheduling; above it they are within ~1.1× and a
split pays. **The scheduler's viability is a function of required precision, not
of the hardware alone.**

**wgpu's limitation is structural:** WGSL has no `f64` type. The wgpu backend is
therefore f32 unconditionally and cannot be correct above zoom ~1e6 without
going through perturbation.

### 3.1 SIMD variants

| Variant | Width | Selected for | Measured @1080p |
|---|---|---|---|
| `render_simd` | 4×f64 | Mandelbrot/Julia above 1e6 | 5.18 ms (1.36× vs scalar) |
| `render_simd_f32` | 8×f32 | Julia below 1e6 | — |
| `render_simd_f32_ilp` | 8×f32, two chains | Mandelbrot below 1e6 | **3.79 ms (1.87×)** |

**Why ILP only helps a little (+4%):** it interleaves two independent dependency
chains to hide FP latency. But the bulb rejection in 2.1 already removes many
interior pixels before the loop, so there is less latency left to hide. Julia,
which has no interior shortcut, gains far more from vectorisation (2.20×) than
Mandelbrot does (1.28× at plain f32x8) for exactly this reason.

### 3.2 GPU dispatch order — Morton (Z-order)

CUDA launches a 1-D grid and decodes each thread's index to `(x,y)` via a Morton
curve, so spatially-adjacent pixels land in adjacent threads.

**Why:** better L2 locality on writes, and neighbouring pixels have similar
iteration counts, which should reduce warp divergence.

**Measured:** the padding to a power-of-two square launches **2.02× more threads
than pixels** — which looks alarming and costs **+1.9%** (median of 7 paired
runs, two runs favouring Morton). The dead threads fall in one contiguous
L-shaped region and whole blocks retire immediately. Recorded as a
tested-and-rejected hypothesis so it does not get "optimised" again.

---

## Stage 4 — Deep zoom: perturbation

### 4.1 Why perturbation at all

Beyond f64's reach, per-pixel arbitrary precision is correct but far too slow.
Perturbation computes **one** high-precision reference orbit `Z_n` per frame and
expresses every other pixel as an offset:

```
ε_{n+1} = 2·Z_n·ε_n + ε_n² + δ        (δ = c − c_ref)
escape test on z_n = Z_n + ε_n
```

`ε` stays small, so it survives in ordinary f64 even when the absolute
coordinates do not.

**Measured, and this is the honest finding:** perturbation is **slower than
plain scalar f64 at every zoom tested** (0.42–0.95×). Series approximation
narrows but does not close the gap. It is a **correctness feature for extreme
zoom, not a throughput feature** — its payoff is past f64's ~1e15 ceiling, which
this sweep does not reach.

### 4.2 Reference orbit precision

`f64` below zoom 1e12; **double-double (`Dd`)** above it — a number stored as an
unevaluated sum of two f64s giving ~32 decimal digits, built from `two_sum` and
`two_prod` error-free transformations.

**Why double-double and not a bignum library:** it is a handful of f64 flops per
operation with no allocation and no dependency, and 32 digits is ample for the
reference orbit. Measured cost is ~1 µs against frame times of tens of
milliseconds — completely negligible, which is the point.

### 4.3 Series approximation

Fits a truncated power series `ε ≈ A·δ + B·δ² + C·δ³ + D·δ⁴` to the reference
orbit, letting early iterations be skipped in bulk.

**Measured:** skips only **2–36 iterations out of 1000** for the tested reference
point, i.e. a 1–3% head start. The coefficients diverge quickly at this field of
view. A deeper mini-brot reference or a smaller `delta_max_sq` would likely
improve it — untested.

### 4.4 Glitch handling — three strategies

A glitch is a pixel where the reference stops being a valid linearisation,
triggered by `|ε|² > |Z|²·1e-6`.

**Why this matters:** glitch rate across 15 random reference points at zoom 1e9
was **6.67% mean, 24.94% standard deviation, 0–100% range**. A single reference
is a lottery — a 100% sample means the reference itself escaped early and the
whole frame is invalid relative to it.

| Strategy | Mechanism | Termination guarantee |
|---|---|---|
| single-reference | glitched pixel falls back to exact scalar | trivially |
| multi-reference | seed new references at glitch sites, up to `MAX_REFS = 8`; then fall back to scalar | the cap |
| rebasing | restart perturbation from the current orbit point instead of falling back | bounded restarts |

The `MAX_REFS` cap exists specifically so termination does not depend on the
glitch pattern.

**Measured cost of robustness (1920×1080, `max_iter=1000`, seahorse valley):**

| zoom | scalar | single-ref | multi-ref | rebase |
|---|---:|---:|---:|---:|
| 1e0 | 8.8 ms | 8.8 ms | 45.3 ms | 119.4 ms |
| 1e3 | 59.9 ms | 60.4 ms | 124.3 ms | 180.3 ms |
| 1e6 | **25.1 ms** | 59.7 ms | 313.3 ms | 68.0 ms |
| 1e9 | **25.1 ms** | 42.8 ms | 45.4 ms | 69.4 ms |
| 1e12 | **25.0 ms** | 43.2 ms | 43.3 ms | 68.8 ms |

**Both robustness strategies are expensive — 1.6× to 5.2× over single-reference,
and plain scalar beats all three at every zoom from 1e6 up.** Multi-reference is
worst at 1e6 (5.2× single-reference) and converges toward it by 1e12, which fits
the mechanism: at moderate zoom the glitch sites are scattered and each new
reference orbit costs a full high-precision computation, whereas at extreme zoom
one or two references cover the frame.

**What this measurement does and does not show.** It measures *cost only*. The
benefit — fewer glitched pixels — is not measured here; §5.8 of `summary.md`
measured the underlying glitch rate separately (6.67% mean, 24.94% stddev,
0–100% range across 15 random references). So the honest statement is: these
strategies buy variance reduction at 1.6–5.2× the runtime, and whether that
trade is worth taking depends on a glitch rate this benchmark does not observe.
Quantifying correctness-per-millisecond would need a pixel-accuracy metric
against a high-precision ground truth — a clear piece of future work.

---

## Stage 5 — Work-skipping algorithms

All four were implemented and, until now, never measured.

### 5.1 Mariani–Silver

Trace a rectangle's border; if every border pixel shares a value the interior
must too (the escape-time field has no holes), so flood-fill it. Otherwise
subdivide.

**The first measurement of this was wrong, and the way it was wrong matters.**
Against the default 16-thread `render()` it looked ~10× slower. But
`render_mariani_silver` is **inherently sequential** — its recursive subdivision
carries a data dependency this implementation does not parallelise — so that
number reported the thread count, not the algorithm. The suite now runs two
baselines and the sequential arms are compared against the 1-thread one.

Full run, 1920×1080, `max_iter=1000`:

| viewport | baseline 16T | baseline 1T | Mariani–Silver | vs 1T | vs 16T |
|---|---:|---:|---:|---:|---:|
| whole set | 9.1 ms | 55.8 ms | 67.3 ms | 0.83× | 0.14× |
| boundary (1e2) | 50.5 ms | 368.2 ms | **304.2 ms** | **1.21×** | 0.17× |
| deep boundary (1e4) | 88.9 ms | 721.9 ms | 744.9 ms | 0.97× | 0.12× |

**Conclusion:** the work-skipping is real but narrow — it wins only where large
uniform high-iteration regions exist (the boundary view, 1.21×), and even there
a 16-thread parallel render is 6× faster. **On a multi-core CPU, parallelism
beats work-skipping.** To compete it would need parallelising, which its data
dependency makes non-trivial. A good illustration that an algorithmic win can be
entirely erased by an execution-model loss.

### 5.2 Mariani–Silver + distance estimation

Same border-tracing, but the subdivision floor is set by an exterior distance
estimate rather than a fixed minimum tile size: stop subdividing once the
estimate proves the rectangle cannot contain boundary.

**This is the best work-skipping result in the project, and it was the surprise
of the run:**

| viewport | baseline 1T | MS + DEM | vs 1T | vs plain MS |
|---|---:|---:|---:|---:|
| whole set | 55.8 ms | **33.2 ms** | **1.68×** | **2.03×** |
| boundary (1e2) | 368.2 ms | 329.6 ms | 1.12× | 0.92× |
| deep boundary (1e4) | 721.9 ms | 715.5 ms | 1.01× | 1.04× |

The distance estimate is worth **2.03× over plain Mariani–Silver** on the
whole-set view, and it inverts that view's verdict from a 0.83× loss to a 1.68×
win. The mechanism is intuitive: on the whole-set view most of the frame is far
from the boundary, so the estimator terminates recursion early over large areas.
At deep zoom nearly everything is near the boundary and the advantage vanishes.

Still ~1.7× slower than a 16-thread parallel render, so the same conclusion
holds — but this is the one work-skipping technique whose algorithmic value is
unambiguous.

### 5.3 Interior distance estimation

Bounds the distance from an interior point to the boundary by tracking the
derivative of the attracting cycle, proving interiority without iterating to
`max_iter`. Unlike Mariani–Silver this **is** rayon-parallel, so it compares
fairly against the 16-thread baseline.

| viewport | baseline 16T | IDE | ratio |
|---|---:|---:|---:|
| whole set | 9.1 ms | 8.6 ms | 1.05× |
| boundary | 50.5 ms | 52.3 ms | 0.96× |
| deep boundary | 88.9 ms | 90.5 ms | 0.98× |

**Verdict: neutral.** Within noise at every viewport. The derivative tracking
costs about what the saved iterations return. The closed-form cardioid/bulb
tests in §2.1 already catch the cheap interior cases, so IDE is left proving
interiority only for the awkward ones — where it is no cheaper than iterating.

### 5.4 Neighbour capping

A pixel's iteration count differs from its left neighbour's by a bounded amount
in smooth regions, so cap the loop at `neighbour + 16` and recompute exactly only
when the cap is hit.

| viewport | baseline 16T | capped | ratio |
|---|---:|---:|---:|
| whole set | 9.1 ms | 10.4 ms | 0.87× |
| boundary | 50.5 ms | 50.6 ms | 1.00× |
| deep boundary | 88.9 ms | 103.5 ms | 0.86× |

**Verdict: loses.** 13–14% slower where it differs at all. The bookkeeping plus
the recompute-on-miss costs more than the iterations it avoids, and it breaks
the row-level SIMD the baseline gets.

## Stage 6 — Frame-to-frame reuse

### 6.1 Incremental pan (`shift_and_fill`)

An axis-aligned pan by a whole number of pixels leaves most of the previous frame
valid: memmove the overlap, compute only the newly exposed strip.

**Measured — the largest unmeasured effect in the project:**

| pan distance | time | vs 44.6 ms full render |
|---|---:|---:|
| 1 px | 0.549 ms | 81× |
| 32 px | 0.120 ms | **370×** |
| 192 px | 0.215 ms | 208× |
| 960 px (half frame) | 0.742 ms | 60× |

The effect size is unambiguous — one to two orders of magnitude — but the
ordering across distances is **not monotonic**, and it should be. A 1 px pan
recomputes a 1-pixel strip and cannot legitimately cost 4.5× a 32 px pan. At
these magnitudes the measurement is dominated by the `iter_batched` clone of the
8.29 MB frame buffer (~0.36 ms as a flat memcpy) rather than by the strip work,
so the per-distance numbers should not be read as a scaling curve. The
comparison against a full render is sound; the shape is not. Isolating the strip
cost would need the clone moved out of the measured region.

**The precondition that makes it fragile:** `Viewport::delta_pixels` requires the
shift to round to within `1e-6` of a whole pixel. An arbitrary real-valued centre
shift silently falls through to a full recompute. The benchmark therefore
constructs its viewports from the pixel step and **asserts** the offset landed,
because otherwise it would measure `render()` under another name — a mistake an
earlier version of this benchmark actually made.

### 6.2 Colour cache

If only the palette changed, re-run `colorize()` over the cached iteration buffer
and skip rendering entirely.

**Why it is worth having:** the iteration buffer is backend-independent, so this
is free reuse. **Caveat:** both this and 6.1 live in the CPU worker, which is
`#[cfg(not(feature = "cuda"))]` — so a CUDA build compiles them out. That is an
application-architecture issue, not a benchmark one.

---

## Stage 7 — The heterogeneous scheduler

### 7.1 Classification — corner sampling

Recursively subdivide 128 px cells down to 16 px. At each level sample the tile's
**4 corner pixels** and compute `spread = (max − min)/max_iter`.

- `spread < threshold` → **GPU tile**, stop subdividing at whatever size reached
- at minimum size and still divergent → **CPU tile**
- otherwise bisect the longer axis and recurse

**Why corners and not a prepass:** an earlier design rendered a coarse
prepass on the GPU, then classified from it — an extra dispatch and an extra
transfer before any real work started. Corner sampling reuses the same `pixel()`
the render uses, costs 4 evaluations per tile, and needs no GPU round trip.
Measured at **0.03–0.32 ms** per frame, and it parallelises over rayon because
top-level cells are independent.

**Why this cannot corrupt the image:** classification chooses *where* a tile is
computed, never *whether*. Both backends compute every pixel of whatever tile
they receive. A missed filament costs GPU-side warp divergence — never a wrong
pixel. This is categorically different from a flood-fill shortcut that skips
computation on a boundary assumption.

**Why terminate early on coherence:** a uniform region resolves as one large GPU
tile instead of being forced down to many small ones — fewer, fatter dispatches.

### 7.2 The steal reserve

20% of GPU-classified tiles are held back in a separate queue *before* dispatch.

**Why reserve rather than let the CPU drain the GPU's queue:** a CUDA dispatch is
asynchronous and returns in microseconds, so the GPU queue would be empty long
before any CPU worker looked. The reserve is what creates a real window.

### 7.3 Bidirectional work stealing

CPU workers drain their own queue, then the reserve. The GPU issues a second
"mop-up" dispatch if ≥ 64 tiles remain unclaimed.

**Measured — both directions genuinely fire:** at zoom 1e0/1e2 the CPU steals
(1271 and 1754 tiles over 30 frames) and the GPU never mops up; at zoom 1e4 the
GPU mops up on **30/30 frames** and the CPU steals nothing. Whichever processor
is faster on that frame absorbs the slack, without anyone predicting which.

*(I initially calculated that the 64-tile threshold could never be met and was
about to report the mop-up as dead code. Measuring it disproved that. Worth
recording as a caution about arithmetic on remembered tile counts.)*

### 7.4 Adaptive threshold — PI controller

```
error = gpu_ms − cpu_ms
threshold −= K_P·error + K_I·mean(last 16 errors)
threshold = clamp(threshold, 0.001, 0.5)
```

Lowering the threshold makes "coherent enough for GPU" stricter, routing more
work to the CPU. Converges over a few frames onto whatever balances *this*
viewport on *this* machine.

**Honest status:** `K_P`, `K_I` and the clamp bounds are starting guesses, not
derived — the source says so. The 1e7+ results show it: the scheduler achieves
78 ms where an ideal split would give 43 ms, leaving ~45% unclaimed. This is the
clearest remaining piece of future work.

### 7.5 What bounds the whole thing

`CudaFractal::readback` copies the **entire** frame regardless of how few tiles
the GPU rendered — 1.33 ms against a 0.28 ms kernel, ~82% of the frame.

So the scheduler can only shrink the smaller 18%. At zoom 1e0 the best possible
kernel saving is ~0.01 ms against ~1.3 ms of machinery: **two orders of magnitude
underwater**. And the fp64 bulb fix made this *worse* by halving the numerator.

**Why partial readback is not a drop-in fix:** readback time does scale with size
(50% of bytes = 51.6% of time), but splitting the same bytes per-tile costs
**2.6× (135 copies) to 5.9× (825 copies)** a single copy. Partial readback
therefore requires the GPU's share to be one **contiguous row range** — which
means abandoning scattered tile assignment for a horizontal split, i.e. the
"dumb" static split the adaptive scheduler was built to replace. That tension is
architectural, not a tuning knob.

---

## Stage 8 — Colouring

**Algorithm:** histogram of escaped pixels → CDF → equalised `t ∈ [0,1]` →
8192-entry palette LUT. In-set pixels are black.

**Why histogram equalisation:** the escape-time distribution is extremely
non-uniform — most pixels escape in a few iterations while a few take thousands.
A linear map wastes almost all the palette. Equalisation allocates colour by
population, so detail is visible at every depth.

**Why a LUT:** the palette function is evaluated 8192 times per `colorize()` call
instead of once per pixel (2.07M times at 1080p).

**Measured, and this is the most important number in the pipeline:** 2.34 ms,
CPU-only, identical for every backend. That is **47% of the CUDA pipeline**. Of a
4.97 ms CUDA pipeline, roughly **0.3 ms is actual fractal iteration** — the rest
is data movement and colouring.

**Consequence:** moving this stage onto the GPU is worth **2.18× end-to-end**,
more than any remaining kernel optimisation and far more than a perfect
scheduler (7%). It is the highest-leverage work left in the project.

---

## Summary — what won, what lost

| Technique | Stage | Verdict |
|---|---|---|
| Cardioid / bulb / period-3 rejection | kernel | **Large win** — most of the ~20× cross-project gap |
| Brent cycle detection | kernel | **Win**, cheaper than Floyd |
| f32 fast path below zoom 1e6 | precision | **Win** — and creates the scheduler's win regime |
| SIMD f32x8 + ILP | CPU | **1.87×**; ILP itself only +4% |
| Incremental pan | reuse | **60–370×** — largest measured effect in the project |
| Mariani–Silver **+ distance estimation** | work-skipping | **1.68× vs 1 thread** on the whole-set view; 2.03× over plain MS |
| Heterogeneous scheduler | scheduling | **Wins only above zoom 1e6** (1.71×); loses below |
| Mariani–Silver (plain) | work-skipping | **Narrow** — 1.21× vs 1 thread at the boundary only |
| Morton dispatch | GPU | **Neutral** (+1.9%) — tested, rejected |
| Interior distance estimation | work-skipping | **Neutral** (0.96–1.05×) |
| Neighbour capping | work-bounding | **Loses** (13–14% slower) |
| Hilbert traversal | memory order | **Loses** (39% slower, worse cache misses) |
| Perturbation | deep zoom | **Correctness win, throughput loss** (0.42–0.95×) |
| Multi-reference / rebasing | glitch handling | **Costs 1.6–5.2×**; benefit unmeasured |
| Series approximation | deep zoom | **Marginal** — skips 2–36 of 1000 iterations |
| GPU-side colouring | colour | **Not implemented — the biggest remaining win (2.18×)** |

## Two results worth putting in the discussion chapter

**Parallelism beat work-skipping.** Every sequential work-skipping technique
here loses to a 16-thread parallel render even when it demonstrably does less
arithmetic — Mariani-Silver at the boundary saves 17% of the work and is still
6× slower than the parallel baseline. On a many-core CPU, the cost of *not*
parallelising exceeds the benefit of *not computing*. Whether that generalises
depends on core count: on a 2-core machine the verdict would likely invert.

**Almost none of the frame is fractal iteration.** Of a 4.76 ms CUDA pipeline,
~0.3 ms is the escape-time loop. The rest is a PCIe readback (1.33 ms) and CPU
colour mapping (2.29 ms). Every optimisation in Stage 2 and Stage 3 — the bulb
rejection, the SIMD widths, the f32 fast path, the fp64 bulb fix — competes for
that 6%. The two largest untaken wins (GPU colouring, eliminating the readback)
are both in stages that do no fractal mathematics at all.

> Numbers from `bench_results/criterion_20260809_204455_2174b94_full`
> (240 benchmarks, one session). Scheduler deep-zoom figures come from
> `criterion_20260808_210551_fc5ca0d_sched_improved`, which the current run did
> not re-execute. See `results/summary.md` for methodology and the standing
> caveat that cross-session comparisons on this machine are unreliable.
