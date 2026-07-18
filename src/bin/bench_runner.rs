/// bench_runner — deterministic workload for `perf stat` / `perf record`.
///
/// Usage:
///   bench_runner [options]
///
///   --fractal  mandelbrot|julia|newton|nova|all   (default: all)
///   --width    N                                  (default: 1920)
///   --height   N                                  (default: 1080)
///   --iters    N                                  (default: 1000)
///   --runs     N   warm-up + timed repetitions    (default: 5)
///   --threads  N   rayon threads, 0 = num_cpus    (default: 0)
///   --backend  cpu|wgpu|hybrid                    (default: cpu)
///   --zoom     F                                  (default: 1.0)
///   --center   RE,IM                              (default: -0.5,0.0)
///   --perturbation     use render_perturbation()  (CPU only, Mandelbrot)
///   --compare-perturb  scalar vs perturbation timing + pixel diff (Mandelbrot)
///   --perturb-sweep    zoom sweep [1,1e3,1e6,1e9,1e12] at seahorse valley
///   --compare-bulb-reject   validate period-3 bulb disk against raw escape ground truth
///   --compare-neighbor-cap  validate render_neighbor_capped vs render() (max_diff + recompute_pct)
///   --cap-slack N           slack for --compare-neighbor-cap             (default: 16)
///   --tile-check            validate render_tiled() vs render() (expect max_diff == 0)
///   --compare-pan-recycle DX,DY  validate shift_and_fill vs a from-scratch render at
///                                 the panned viewport (expect a small, near-zero diff —
///                                 see shift_and_fill's doc comment for why it isn't
///                                 always bit-exact)
///   --compare-multiref      count glitched pixels: primary-ref only vs after multi-ref
///                            correction, plus a pixel_diff_stats vs scalar ground truth
///   --compare-rebase        rebased perturbation vs single-ref: timing + mismatch counts
///                            vs scalar (rebase keeps chaotic boundary pixels in perturbation,
///                            so a small mismatch fraction vs f64 scalar is expected)
///   --compare-dem-cull [--dem-k F]  DEM-culled Mariani-Silver vs exact MS (approximate
///                                    bilinear fill — report pixel diff, tune k)
///   --compare-ide           render_ide_biased vs render() (derivative-bailout interior
///                            detection is approximate — report pixel diff)
///   --heterogeneous         validate scheduler::render_heterogeneous vs render() (expect
///                            max_diff == 0.0), report gpu_ms/cpu_ms/cpu_tile_frac
///                            (requires --features cuda)
///   --scaling       run thread-scaling sweep      (flag)
///   --json          emit JSON to stdout            (flag)
///
/// Examples:
///   perf stat -e cycles,instructions,cache-misses \
///       ./target/release/bench_runner --fractal mandelbrot --runs 3
///
///   ./target/release/bench_runner --backend wgpu --width 3840 --height 2160
///
///   ./target/release/bench_runner --scaling --json > bench_results/scaling.json

use std::time::{Duration, Instant};
use novafractal::fractal::{render, render_perturbation, render_perturbation_sa, compute_reference_orbit, compute_series_approx, flops_per_iter, FractalType, render_neighbor_capped, in_period3_bulb, pixel_grid, render_tiled, shift_and_fill, render_perturbation_multiref, perturb_mandelbrot_flagged, render_perturbation_rebase, render_mariani_silver_dem, render_ide_biased};
use novafractal::gui::viewport::Viewport;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Args {
    fractals:   Vec<FractalType>,
    width:      u32,
    height:     u32,
    max_iter:   u32,
    runs:       usize,
    threads:    usize,
    backend:            Backend,
    zoom:               f64,
    center:             [f64; 2],
    perturbation:       bool,
    use_sa:             bool,
    compare_perturb:    bool,
    perturb_sweep:      bool,
    scaling:            bool,
    json:               bool,
    compare_bulb_reject: bool,
    compare_neighbor_cap: bool,
    cap_slack:           u32,
    tile_check:           bool,
    compare_pan_recycle:  Option<(i32, i32)>,
    compare_multiref:     bool,
    compare_rebase:       bool,
    compare_dem_cull:     bool,
    dem_k:                f64,
    compare_ide:          bool,
    heterogeneous:        bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Backend { Cpu, Wgpu, Hybrid }

impl Default for Args {
    fn default() -> Self {
        Self {
            fractals:  FractalType::ALL.to_vec(),
            width:     1920,
            height:    1080,
            max_iter:  1000,
            runs:      5,
            threads:   0,
            backend:         Backend::Cpu,
            zoom:            1.0,
            center:          [-0.5, 0.0],
            perturbation:    false,
            use_sa:          false,
            compare_perturb: false,
            perturb_sweep:   false,
            scaling:         false,
            json:            false,
            compare_bulb_reject: false,
            compare_neighbor_cap: false,
            cap_slack:           16,
            tile_check:           false,
            compare_pan_recycle:  None,
            compare_multiref:     false,
            compare_rebase:       false,
            compare_dem_cull:     false,
            dem_k:                4.0,
            compare_ide:          false,
            heterogeneous:        false,
        }
    }
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--fractal" => {
                i += 1;
                a.fractals = match raw[i].as_str() {
                    "mandelbrot" => vec![FractalType::Mandelbrot],
                    "julia"      => vec![FractalType::Julia],
                    "newton"     => vec![FractalType::Newton],
                    "nova"       => vec![FractalType::Nova],
                    "all"        => FractalType::ALL.to_vec(),
                    x => panic!("unknown fractal: {x}"),
                };
            }
            "--width"   => { i += 1; a.width    = raw[i].parse().unwrap(); }
            "--height"  => { i += 1; a.height   = raw[i].parse().unwrap(); }
            "--iters"   => { i += 1; a.max_iter = raw[i].parse().unwrap(); }
            "--runs"    => { i += 1; a.runs     = raw[i].parse().unwrap(); }
            "--threads" => { i += 1; a.threads  = raw[i].parse().unwrap(); }
            "--backend" => {
                i += 1;
                a.backend = match raw[i].as_str() {
                    "cpu"    => Backend::Cpu,
                    "wgpu"   => Backend::Wgpu,
                    "hybrid" => Backend::Hybrid,
                    x => panic!("unknown backend: {x}"),
                };
            }
            "--zoom" => {
                i += 1;
                a.zoom = raw[i].parse().unwrap_or_else(|_| {
                    panic!("invalid --zoom value: {}", raw[i]);
                });
            }
            "--center" => {
                i += 1;
                let parts: Vec<&str> = raw[i].split(',').collect();
                assert!(parts.len() == 2, "--center expects RE,IM");
                a.center[0] = parts[0].trim().parse().unwrap();
                a.center[1] = parts[1].trim().parse().unwrap();
            }
            "--perturbation"    => a.perturbation    = true,
            "--sa"              => a.use_sa          = true,
            "--compare-perturb" => a.compare_perturb = true,
            "--perturb-sweep"   => a.perturb_sweep   = true,
            "--compare-bulb-reject" => a.compare_bulb_reject = true,
            "--compare-neighbor-cap" => a.compare_neighbor_cap = true,
            "--cap-slack" => { i += 1; a.cap_slack = raw[i].parse().unwrap(); }
            "--tile-check" => a.tile_check = true,
            "--compare-pan-recycle" => {
                i += 1;
                let parts: Vec<&str> = raw[i].split(',').collect();
                assert!(parts.len() == 2, "--compare-pan-recycle expects DX,DY");
                let dx: i32 = parts[0].trim().parse().unwrap();
                let dy: i32 = parts[1].trim().parse().unwrap();
                a.compare_pan_recycle = Some((dx, dy));
            }
            "--compare-multiref" => a.compare_multiref = true,
            "--compare-rebase" => a.compare_rebase = true,
            "--compare-dem-cull" => a.compare_dem_cull = true,
            "--dem-k" => { i += 1; a.dem_k = raw[i].parse().unwrap(); }
            "--compare-ide" => a.compare_ide = true,
            "--heterogeneous" => a.heterogeneous = true,
            "--scaling" => a.scaling = true,
            "--json"    => a.json    = true,
            x => panic!("unknown flag: {x}"),
        }
        i += 1;
    }
    a
}

// ── measurement types ─────────────────────────────────────────────────────────

#[derive(Debug)]
struct Sample {
    fractal:        &'static str,
    backend:        &'static str,
    threads:        usize,
    width:          u32,
    height:         u32,
    max_iter:       u32,
    runs:           usize,
    median_ms:      f64,
    min_ms:         f64,
    max_ms:         f64,
    mpix_per_sec:   f64,
    total_iters:    u64,
    gflops:         f64,
    /// bytes read (iteration buffer) + bytes written (color buffer)
    mem_traffic_mb: f64,
}

#[derive(Debug)]
struct PerturbCompare {
    zoom:                  f64,
    center:                [f64; 2],
    scalar_median_ms:      f64,
    perturb_median_ms:     f64,
    sa_median_ms:          f64,
    scalar_mpix_per_sec:   f64,
    perturb_mpix_per_sec:  f64,
    sa_mpix_per_sec:       f64,
    speedup_perturb:       f64,
    speedup_sa:            f64,
    sa_skip:               usize,
    max_pixel_diff:        f32,
    mean_pixel_diff:       f64,
    pixels_above_0_01:     u64,
    total_pixels:          u64,
}

fn cpu_render(
    vp: &Viewport,
    fractal: FractalType,
    julia_c: [f64; 2],
    max_iter: u32,
    perturbation: bool,
    use_sa: bool,
) -> Vec<f32> {
    if perturbation && use_sa {
        render_perturbation_sa(vp, fractal, julia_c, max_iter)
    } else if perturbation {
        render_perturbation(vp, fractal, julia_c, max_iter)
    } else {
        render(vp, fractal, julia_c, max_iter)
    }
}

fn pixel_diff_stats(a: &[f32], b: &[f32]) -> (f32, f64, u64) {
    assert_eq!(a.len(), b.len());
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    let mut above = 0u64;
    for (&x, &y) in a.iter().zip(b) {
        let d = (x - y).abs();
        max_diff = max_diff.max(d);
        sum_diff += d as f64;
        if d > 0.01 {
            above += 1;
        }
    }
    let mean = if a.is_empty() { 0.0 } else { sum_diff / a.len() as f64 };
    (max_diff, mean, above)
}

/// Validate `scheduler::render_heterogeneous` against plain CPU `render()` and
/// report the GPU/CPU tile split + timing, alongside a plain full-frame
/// `cuda.render()` baseline for context. Both columns are expected to show
/// `max_diff == 0.0` — CPU and GPU now agree bit-for-bit (see the fixes in
/// `src/gpu/cuda.rs::morton_cfg`, `src/fractal/fractal.cu::mandelbrot`,
/// `build.rs` (`--fmad=false`), and `src/fractal/fractal.rs::render_tile_exact`
/// for the three independent divergence sources this closed).
#[cfg(feature = "cuda")]
fn run_heterogeneous_check(vp: &Viewport, args: &Args) {
    use novafractal::gpu::cuda::CudaFractal;
    use novafractal::scheduler::{render_heterogeneous, controller::ThresholdController, SchedulerConfig};

    let julia_c = [-0.4f64, 0.6];
    let mut cuda = CudaFractal::new(vp.width, vp.height);
    let mut controller = ThresholdController::new(50.0);
    let cfg = SchedulerConfig::default();

    for &fractal in &args.fractals {
        let expected = render(vp, fractal, julia_c, args.max_iter);

        let pg = pixel_grid(vp);
        let plain_gpu = cuda.render(pg.re_start, pg.im_start, pg.re_step, pg.im_step, julia_c[0], julia_c[1], args.max_iter, fractal.as_u32());
        let (base_max, base_mean, base_above) = pixel_diff_stats(&expected, &plain_gpu);

        let result = render_heterogeneous(vp, fractal, julia_c, args.max_iter, &mut cuda, &mut controller, &cfg);
        let (max_diff, mean_diff, above) = pixel_diff_stats(&expected, &result.buf);

        if args.json {
            println!(
                "{{\"fractal\":\"{}\",\"max_diff\":{max_diff},\"mean_diff\":{mean_diff},\"above_0_01\":{above},\"baseline_max_diff\":{base_max},\"baseline_mean_diff\":{base_mean},\"baseline_above_0_01\":{base_above},\"gpu_ms\":{},\"cpu_ms\":{},\"cpu_tile_frac\":{}}}",
                fractal.name(), result.gpu_ms, result.cpu_ms, result.cpu_tile_frac,
            );
        } else {
            println!("\nheterogeneous scheduler validation: {} ({}×{}, max_iter={}, zoom={})", fractal.name(), args.width, args.height, args.max_iter, args.zoom);
            println!("  vs CPU render()          : max_diff={max_diff:<10} mean_diff={mean_diff:<12.6} above_0.01={above}");
            println!("  plain cuda.render() base : max_diff={base_max:<10} mean_diff={base_mean:<12.6} above_0.01={base_above}  (context, see fn doc comment)");
            println!("  gpu_ms        : {:.2}", result.gpu_ms);
            println!("  cpu_ms        : {:.2}", result.cpu_ms);
            println!("  cpu_tile_frac : {:.2}%", result.cpu_tile_frac * 100.0);
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn run_heterogeneous_check(_vp: &Viewport, _args: &Args) {
    eprintln!("[heterogeneous] cuda feature not enabled — rerun with: cargo run --release --bin bench_runner --features cuda -- --heterogeneous");
}

fn time_cpu_render(
    vp: &Viewport,
    fractal: FractalType,
    max_iter: u32,
    runs: usize,
    perturbation: bool,
    use_sa: bool,
) -> (Sample, Vec<f32>) {
    let pixels = (vp.width * vp.height) as u64;
    let julia_c = [-0.4f64, 0.6];

    let _ = cpu_render(vp, fractal, julia_c, max_iter, perturbation, use_sa);

    let mut times: Vec<Duration> = Vec::with_capacity(runs);
    let mut last_buf = Vec::new();

    for _ in 0..runs {
        let t = Instant::now();
        last_buf = cpu_render(vp, fractal, julia_c, max_iter, perturbation, use_sa);
        times.push(t.elapsed());
    }

    let total_iters: u64 = last_buf.iter()
        .map(|&v| v.min(max_iter as f32) as u64)
        .sum();
    let fpi = flops_per_iter(fractal);

    times.sort();
    let med_s = times[runs / 2].as_secs_f64();
    let backend = match (perturbation, use_sa) {
        (true,  true)  => "cpu+perturb+sa",
        (true,  false) => "cpu+perturb",
        (false, _)     => "cpu",
    };

    let sample = Sample {
        fractal:        fractal.name(),
        backend,
        threads:        rayon::current_num_threads(),
        width:          vp.width,
        height:         vp.height,
        max_iter,
        runs,
        median_ms:      med_s * 1e3,
        min_ms:         times[0].as_secs_f64() * 1e3,
        max_ms:         times[runs - 1].as_secs_f64() * 1e3,
        mpix_per_sec:   pixels as f64 / 1e6 / med_s,
        total_iters,
        gflops:         (total_iters * fpi) as f64 / med_s / 1e9,
        mem_traffic_mb: pixels as f64 * 8.0 / 1e6,
    };

    (sample, last_buf)
}

// ── CPU benchmarking ──────────────────────────────────────────────────────────

fn bench_cpu(fractal: FractalType, vp: &Viewport, max_iter: u32, runs: usize, perturbation: bool, use_sa: bool) -> Sample {
    time_cpu_render(vp, fractal, max_iter, runs, perturbation, use_sa).0
}

fn sa_skip_count(vp: &Viewport, max_iter: u32) -> usize {
    let aspect       = vp.width as f64 / vp.height as f64;
    let half         = 2.0 / vp.zoom;
    let delta_max_sq = (half * aspect) * (half * aspect) + half * half;
    let orbit = compute_reference_orbit(vp.center[0], vp.center[1], max_iter);
    compute_series_approx(&orbit, delta_max_sq).skip
}

fn bench_compare_perturb(vp: &Viewport, max_iter: u32, runs: usize) -> PerturbCompare {
    let fractal = FractalType::Mandelbrot;
    let (scalar_sample,  scalar_buf)  = time_cpu_render(vp, fractal, max_iter, runs, false, false);
    let (perturb_sample, perturb_buf) = time_cpu_render(vp, fractal, max_iter, runs, true,  false);
    let (sa_sample,      sa_buf)      = time_cpu_render(vp, fractal, max_iter, runs, true,  true);
    let (max_diff,  mean_diff,  above)  = pixel_diff_stats(&scalar_buf, &perturb_buf);
    let (max_diff2, mean_diff2, above2) = pixel_diff_stats(&scalar_buf, &sa_buf);
    let speedup_perturb = if perturb_sample.median_ms > 0.0 {
        scalar_sample.median_ms / perturb_sample.median_ms
    } else { 0.0 };
    let speedup_sa = if sa_sample.median_ms > 0.0 {
        scalar_sample.median_ms / sa_sample.median_ms
    } else { 0.0 };

    PerturbCompare {
        zoom:                  vp.zoom,
        center:                vp.center,
        scalar_median_ms:      scalar_sample.median_ms,
        perturb_median_ms:     perturb_sample.median_ms,
        sa_median_ms:          sa_sample.median_ms,
        scalar_mpix_per_sec:   scalar_sample.mpix_per_sec,
        perturb_mpix_per_sec:  perturb_sample.mpix_per_sec,
        sa_mpix_per_sec:       sa_sample.mpix_per_sec,
        speedup_perturb,
        speedup_sa,
        sa_skip:               sa_skip_count(vp, max_iter),
        max_pixel_diff:        max_diff.max(max_diff2),
        mean_pixel_diff:       mean_diff.max(mean_diff2),
        pixels_above_0_01:     above.max(above2),
        total_pixels:          scalar_buf.len() as u64,
    }
}

const PERTURB_SWEEP_ZOOMS: [f64; 5] = [1.0, 1e3, 1e6, 1e9, 1e12];
const PERTURB_SWEEP_CENTER: [f64; 2] = [-0.75, 0.1];

fn run_perturb_sweep(width: u32, height: u32, max_iter: u32, runs: usize) -> Vec<PerturbCompare> {
    PERTURB_SWEEP_ZOOMS.iter().map(|&zoom| {
        let vp = Viewport {
            center: PERTURB_SWEEP_CENTER,
            zoom,
            width,
            height,
        };
        bench_compare_perturb(&vp, max_iter, runs)
    }).collect()
}

// ── Phase B validation: bulb rejection + neighbor cap ──────────────────────────

/// Ground-truth Mandelbrot escape test with NO early-outs (no cardioid/period-2/
/// period-3 bulb checks, no period detection) — used only to validate that
/// `in_period3_bulb` never misclassifies an escaping pixel as in-set.
fn mandelbrot_raw(cr: f64, ci: f64, max_iter: u32) -> f32 {
    let mut zr = cr;
    let mut zi = ci;
    if max_iter <= 1 { return max_iter as f32; }
    let zr2 = zr * zr;
    let zi2 = zi * zi;
    zi = 2.0 * zr * zi + ci;
    zr = zr2 - zi2 + cr;
    if max_iter <= 2 { return max_iter as f32; }
    for i in 2..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        let zn_sq = zr2 + zi2;
        if zn_sq > 256.0 * 256.0 {
            return i as f32;
        }
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }
    max_iter as f32
}

/// Validates `in_period3_bulb` against `mandelbrot_raw`: every pixel the disk test
/// claims is in the bulb must NOT escape under the raw ground-truth loop. Scans a
/// wide overview viewport. `max_diff == 0` (i.e. `bad_pixels == 0`) means the disk
/// parameters are safe; any bad pixel means they must be shrunk before trusting.
fn bench_compare_bulb_reject(width: u32, height: u32, max_iter: u32) -> (u64, u64) {
    let vp = Viewport { center: [-0.5, 0.0], zoom: 1.0, width, height };
    let pg = pixel_grid(&vp);
    let mut checked = 0u64;
    let mut bad = 0u64;
    for y in 0..height as usize {
        let im = pg.im_start + y as f64 * pg.im_step;
        for x in 0..width as usize {
            let re = pg.re_start + x as f64 * pg.re_step;
            if in_period3_bulb(re, im) {
                checked += 1;
                if mandelbrot_raw(re, im, max_iter) < max_iter as f32 {
                    bad += 1;
                }
            }
        }
    }
    (checked, bad)
}

/// Validates `render_neighbor_capped` against `render()`: max_diff must be 0 (the
/// cap is self-correcting by construction — any nonzero diff is an implementation
/// bug, not tunable error). Also reports `recompute_pct`, the real tuning metric —
/// the fraction of pixels that hit the cap and paid for a full max_iter re-run.
fn bench_compare_neighbor_cap(vp: &Viewport, max_iter: u32, slack: u32) -> (f32, f64, f64) {
    let julia_c = [-0.4f64, 0.6];
    let baseline = render(vp, FractalType::Mandelbrot, julia_c, max_iter);
    let capped = render_neighbor_capped(vp, FractalType::Mandelbrot, julia_c, max_iter, slack);
    let (max_diff, mean_diff, _above) = pixel_diff_stats(&baseline, &capped);

    // recompute_pct: fraction of pixels whose row-scan cap was hit (i.e. the pixel's
    // value equals max_iter but a smaller cap would have been insufficient — we
    // approximate by recomputing the row-cap sequence and counting cap hits).
    let pg = pixel_grid(vp);
    let w = vp.width as usize;
    let h = vp.height as usize;
    let mut hit_cap = 0u64;
    let mut total = 0u64;
    for y in 0..h {
        let im = pg.im_start + y as f64 * pg.im_step;
        let mut cap = max_iter;
        for x in 0..w {
            let re = pg.re_start + x as f64 * pg.re_step;
            let guess = novafractal::fractal::pixel(FractalType::Mandelbrot, re, im, julia_c, cap);
            total += 1;
            if guess >= cap as f32 && cap < max_iter {
                hit_cap += 1;
            }
            cap = ((guess as u32).saturating_add(slack)).min(max_iter);
        }
    }
    let recompute_pct = if total > 0 { hit_cap as f64 / total as f64 * 100.0 } else { 0.0 };

    (max_diff, mean_diff, recompute_pct)
}

/// Validates `shift_and_fill` against a from-scratch render of the panned viewport.
/// See `shift_and_fill`'s doc comment: recycled pixels can differ from a fresh
/// render by a tiny amount (floating-point non-associativity between the old and
/// new viewport's independently-computed coordinate bases), so this reports the
/// diff rather than hard-asserting zero.
fn bench_compare_pan_recycle(vp: &Viewport, max_iter: u32, dx: i32, dy: i32) -> (f32, f64, u64) {
    let julia_c = [-0.4f64, 0.6];
    let w = vp.width as usize;
    let h = vp.height as usize;

    let mut buf = render(vp, FractalType::Mandelbrot, julia_c, max_iter);

    let half = 2.0 / vp.zoom;
    let aspect = vp.width as f64 / vp.height as f64;
    let re_per_px = half * aspect * 2.0 / vp.width as f64;
    let im_per_px = half * 2.0 / vp.height as f64;
    let new_vp = Viewport {
        center: [vp.center[0] - dx as f64 * re_per_px, vp.center[1] - dy as f64 * im_per_px],
        zoom: vp.zoom,
        width: vp.width,
        height: vp.height,
    };

    shift_and_fill(&mut buf, w, h, dx, dy, &new_vp, FractalType::Mandelbrot, julia_c, max_iter);
    let expected = render(&new_vp, FractalType::Mandelbrot, julia_c, max_iter);

    pixel_diff_stats(&buf, &expected)
}

/// Counts glitched pixels against the primary reference orbit only (what
/// `render_perturbation` falls back to scalar for), then validates
/// `render_perturbation_multiref`'s output against scalar ground truth. The
/// glitch-pixel-count before/after is the real success metric for 4a — it's
/// directly measurable, unlike pixel diffs (both single-ref-with-fallback and
/// multi-ref converge to the exact same correct values, by construction).
fn bench_compare_multiref(vp: &Viewport, max_iter: u32) -> (usize, u64, f32) {
    let julia_c = [0.0f64, 0.0];
    let w = vp.width as usize;
    let h = vp.height as usize;
    let pg = pixel_grid(vp);

    let orbit = compute_reference_orbit(vp.center[0], vp.center[1], max_iter);
    let mut primary_glitches = 0usize;
    for y in 0..h {
        let im = pg.im_start + y as f64 * pg.im_step;
        for x in 0..w {
            let re = pg.re_start + x as f64 * pg.re_step;
            if perturb_mandelbrot_flagged(&orbit, re - vp.center[0], im - vp.center[1], max_iter).is_none() {
                primary_glitches += 1;
            }
        }
    }

    let multi = render_perturbation_multiref(vp, FractalType::Mandelbrot, julia_c, max_iter);
    let scalar = render(vp, FractalType::Mandelbrot, julia_c, max_iter);
    let (max_diff, _mean, _above) = pixel_diff_stats(&multi, &scalar);

    (primary_glitches, (w * h) as u64, max_diff)
}

/// Rebased perturbation (4b) vs single-ref-with-scalar-fallback: median timing of
/// each, plus mismatch counts vs f64 scalar ground truth. Rebase keeps chaotic
/// boundary pixels inside perturbation instead of recomputing them exactly, so a
/// small mismatch fraction vs scalar is expected — report, don't assert.
fn bench_compare_rebase(vp: &Viewport, max_iter: u32, runs: usize) -> (f64, f64, usize, usize, u64) {
    let julia_c = [0.0f64, 0.0];
    let time_it = |f: &dyn Fn() -> Vec<f32>| -> (f64, Vec<f32>) {
        let _ = f();
        let mut times: Vec<f64> = Vec::with_capacity(runs);
        let mut buf = Vec::new();
        for _ in 0..runs {
            let t = Instant::now();
            buf = f();
            times.push(t.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (times[runs / 2], buf)
    };
    let (single_ms, single) = time_it(&|| render_perturbation(vp, FractalType::Mandelbrot, julia_c, max_iter));
    let (rebase_ms, rebase) = time_it(&|| render_perturbation_rebase(vp, FractalType::Mandelbrot, julia_c, max_iter));
    let scalar = render(vp, FractalType::Mandelbrot, julia_c, max_iter);
    let d = |a: &[f32], b: &[f32]| a.iter().zip(b).filter(|(x, y)| (**x - **y).abs() > 0.5).count();
    (single_ms, rebase_ms, d(&single, &scalar), d(&rebase, &scalar), scalar.len() as u64)
}

fn print_perturb_table(results: &[PerturbCompare]) {
    println!("\n{:-<140}", "");
    println!("{:<8} {:>10} {:>10} {:>10} {:>8} {:>8} {:>6} {:>10} {:>10}",
             "Zoom", "Scalar ms", "Perturb ms", "SA ms",
             "Sp×perturb", "Sp×SA", "SAskip", "Max diff", ">0.01 px");
    println!("{:-<140}", "");
    for r in results {
        println!("{:<8.0e} {:>10.2} {:>10.2} {:>10.2} {:>8.2}x {:>7.2}x {:>6} {:>10.4} {:>10}",
                 r.zoom,
                 r.scalar_median_ms, r.perturb_median_ms, r.sa_median_ms,
                 r.speedup_perturb, r.speedup_sa, r.sa_skip,
                 r.max_pixel_diff, r.pixels_above_0_01);
    }
    println!("{:-<140}", "");
    println!("center = [{}, {}]", PERTURB_SWEEP_CENTER[0], PERTURB_SWEEP_CENTER[1]);
}

fn print_perturb_json(results: &[PerturbCompare]) {
    println!("[");
    for (i, r) in results.iter().enumerate() {
        let comma = if i + 1 < results.len() { "," } else { "" };
        println!("  {{\"zoom\":{:.6e},\"center\":[{},{}],\
                  \"scalar_median_ms\":{:.4},\"perturb_median_ms\":{:.4},\"sa_median_ms\":{:.4},\
                  \"scalar_mpix_per_sec\":{:.4},\"perturb_mpix_per_sec\":{:.4},\"sa_mpix_per_sec\":{:.4},\
                  \"speedup_perturb\":{:.4},\"speedup_sa\":{:.4},\"sa_skip\":{},\
                  \"max_pixel_diff\":{:.6},\"mean_pixel_diff\":{:.6},\
                  \"pixels_above_0_01\":{},\"total_pixels\":{}}}{}",
                 r.zoom, r.center[0], r.center[1],
                 r.scalar_median_ms, r.perturb_median_ms, r.sa_median_ms,
                 r.scalar_mpix_per_sec, r.perturb_mpix_per_sec, r.sa_mpix_per_sec,
                 r.speedup_perturb, r.speedup_sa, r.sa_skip,
                 r.max_pixel_diff, r.mean_pixel_diff,
                 r.pixels_above_0_01, r.total_pixels, comma);
    }
    println!("]");
}

// ── wgpu benchmarking ─────────────────────────────────────────────────────────

fn init_wgpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    use wgpu::*;
    pollster::block_on(async {
        let instance = Instance::default();
        let adapter = instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await?;
        let (device, queue) = adapter.request_device(
            &DeviceDescriptor {
                label: Some("bench"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
                memory_hints: MemoryHints::Performance,
            },
            None,
        ).await.ok()?;
        eprintln!("[wgpu] adapter: {}", adapter.get_info().name);
        Some((device, queue))
    })
}

fn bench_wgpu(fractal: FractalType, vp: &Viewport, max_iter: u32, runs: usize) -> Option<Sample> {
    use novafractal::gpu::fractal_compute::FractalCompute;
    use novafractal::gpu::unifroms::Uniforms;

    let (device, queue) = init_wgpu()?;
    let compute = FractalCompute::new(&device, vp.width, vp.height);

    let aspect = vp.width as f64 / vp.height as f64;
    let half = 2.0 / vp.zoom;
    let uniforms = Uniforms {
        re_start: (vp.center[0] + (0.5 / vp.width  as f64 - 0.5) * half * aspect * 2.0) as f32,
        im_start: (vp.center[1] + (0.5 / vp.height as f64 - 0.5) * half * 2.0) as f32,
        re_step:  (half * aspect * 2.0 / vp.width  as f64) as f32,
        im_step:  (half * 2.0          / vp.height as f64) as f32,
        julia_cr: -0.4,
        julia_ci:  0.6,
        max_iter,
        fractal:  fractal_to_u32(fractal),
        width:    vp.width,
        height:   vp.height,
        _pad:     [0; 2],
    };

    // warm-up
    let _ = compute.render(&device, &queue, uniforms);

    let pixels = (vp.width * vp.height) as u64;
    let mut times: Vec<Duration> = Vec::with_capacity(runs);
    let mut last_buf = Vec::new();

    for _ in 0..runs {
        let t = Instant::now();
        last_buf = compute.render(&device, &queue, uniforms);
        times.push(t.elapsed());
    }

    let total_iters: u64 = last_buf.iter()
        .map(|&v| v.min(max_iter as f32) as u64)
        .sum();
    let fpi = flops_per_iter(fractal);

    times.sort();
    let med_s = times[runs / 2].as_secs_f64();

    Some(Sample {
        fractal:        fractal.name(),
        backend:        "wgpu",
        threads:        1,
        width:          vp.width,
        height:         vp.height,
        max_iter,
        runs,
        median_ms:      med_s * 1e3,
        min_ms:         times[0].as_secs_f64() * 1e3,
        max_ms:         times[runs - 1].as_secs_f64() * 1e3,
        mpix_per_sec:   pixels as f64 / 1e6 / med_s,
        total_iters,
        gflops:         (total_iters * fpi) as f64 / med_s / 1e9,
        mem_traffic_mb: pixels as f64 * 8.0 / 1e6,
    })
}

fn fractal_to_u32(f: FractalType) -> u32 {
    match f {
        FractalType::Mandelbrot => 0,
        FractalType::Julia      => 1,
        FractalType::Newton     => 2,
        FractalType::Nova       => 3,
    }
}

// ── hybrid benchmarking ───────────────────────────────────────────────────────
//
// Splits the image into top (CPU) and bottom (GPU) halves, renders both
// concurrently, and combines the buffers.  The wall time is max(t_cpu, t_gpu).

fn bench_hybrid(fractal: FractalType, vp: &Viewport, max_iter: u32, runs: usize) -> Option<Sample> {
    use novafractal::gpu::fractal_compute::FractalCompute;
    use novafractal::gpu::unifroms::Uniforms;

    let (device, queue) = init_wgpu()?;

    let h_top    = vp.height / 2;
    let h_bottom = vp.height - h_top;

    let vp_top = Viewport {
        center: [vp.center[0], vp.center[1]],
        zoom:   vp.zoom,
        width:  vp.width,
        height: h_top,
    };

    // Recompute im_start for the bottom half: it starts where the top half ends.
    let aspect    = vp.width as f64 / vp.height as f64;
    let half      = 2.0 / vp.zoom;
    let im_step   = half * 2.0 / vp.height as f64;
    let im_start  = vp.center[1] + (0.5 / vp.height as f64 - 0.5) * half * 2.0;
    let im_start_bottom = im_start + h_top as f64 * im_step;

    let compute_bottom = FractalCompute::new(&device, vp.width, h_bottom);
    let uniforms_bottom = Uniforms {
        re_start: (vp.center[0] + (0.5 / vp.width as f64 - 0.5) * half * aspect * 2.0) as f32,
        im_start: im_start_bottom as f32,
        re_step:  (half * aspect * 2.0 / vp.width as f64) as f32,
        im_step:  im_step as f32,
        julia_cr: -0.4,
        julia_ci:  0.6,
        max_iter,
        fractal:  fractal_to_u32(fractal),
        width:    vp.width,
        height:   h_bottom,
        _pad:     [0; 2],
    };

    let julia_c = [-0.4f64, 0.6f64];
    // warm-up
    let _ = render(&vp_top, fractal, julia_c, max_iter);
    let _ = compute_bottom.render(&device, &queue, uniforms_bottom);

    let pixels = (vp.width * vp.height) as u64;
    let mut wall_times: Vec<Duration> = Vec::with_capacity(runs);

    for _ in 0..runs {
        let t_wall = Instant::now();

        // CPU top half in a separate thread; GPU bottom on this thread.
        // std::thread::scope ensures both finish before we record wall time.
        std::thread::scope(|s| {
            let vp_top_ref  = &vp_top;
            s.spawn(move || render(vp_top_ref, fractal, julia_c, max_iter));
            let _ = compute_bottom.render(&device, &queue, uniforms_bottom);
        });

        wall_times.push(t_wall.elapsed());
    }

    wall_times.sort();
    let med_s = wall_times[runs / 2].as_secs_f64();

    Some(Sample {
        fractal:        fractal.name(),
        backend:        "hybrid(cpu+wgpu)",
        threads:        rayon::current_num_threads(),
        width:          vp.width,
        height:         vp.height,
        max_iter,
        runs,
        median_ms:      med_s * 1e3,
        min_ms:         wall_times[0].as_secs_f64() * 1e3,
        max_ms:         wall_times[runs - 1].as_secs_f64() * 1e3,
        mpix_per_sec:   pixels as f64 / 1e6 / med_s,
        total_iters:    0,   // not tracked in hybrid mode
        gflops:         0.0, // not tracked in hybrid mode
        mem_traffic_mb: pixels as f64 * 8.0 / 1e6,
    })
}

// ── output ────────────────────────────────────────────────────────────────────

fn print_table(samples: &[Sample]) {
    println!("\n{:-<100}", "");
    println!("{:<12} {:<8} {:>8} {:>10} {:>10} {:>10} {:>12} {:>12} {:>12}",
             "Fractal", "Backend", "Threads", "Mpix/s", "GFLOPs", "Med ms",
             "Min ms", "Max ms", "MemMB");
    println!("{:-<100}", "");
    for s in samples {
        println!("{:<12} {:<8} {:>8} {:>10.2} {:>10.3} {:>10.2} {:>12.2} {:>12.2} {:>12.1}",
                 s.fractal, s.backend, s.threads,
                 s.mpix_per_sec, s.gflops,
                 s.median_ms, s.min_ms, s.max_ms,
                 s.mem_traffic_mb);
    }
    println!("{:-<100}", "");
}

fn print_json(samples: &[Sample]) {
    println!("[");
    for (i, s) in samples.iter().enumerate() {
        let comma = if i + 1 < samples.len() { "," } else { "" };
        println!("  {{\"fractal\":\"{}\",\"backend\":\"{}\",\"threads\":{},\
                  \"width\":{},\"height\":{},\"max_iter\":{},\"runs\":{},\
                  \"median_ms\":{:.4},\"min_ms\":{:.4},\"max_ms\":{:.4},\
                  \"mpix_per_sec\":{:.4},\"total_iters\":{},\"gflops\":{:.4},\
                  \"mem_traffic_mb\":{:.2}}}{}",
                 s.fractal, s.backend, s.threads,
                 s.width, s.height, s.max_iter, s.runs,
                 s.median_ms, s.min_ms, s.max_ms,
                 s.mpix_per_sec, s.total_iters, s.gflops,
                 s.mem_traffic_mb, comma);
    }
    println!("]");
}

fn print_flop_budget(fractals: &[FractalType], max_iter: u32) {
    eprintln!("\n── Analytical FLOP budget (lower bound, no FMA fusion) ──────────");
    eprintln!("{:<12} {:>10} {:>16}", "Kernel", "FLOPs/iter", "FLOPs @ max_iter");
    for &f in fractals {
        let fpi = flops_per_iter(f);
        eprintln!("{:<12} {:>10} {:>16}", f.name(), fpi, fpi * max_iter as u64);
    }
    eprintln!("smooth_iter() (ln-based) adds ~6 FLOPs per escaped pixel.");
    eprintln!("────────────────────────────────────────────────────────────────\n");
}

fn git_hash() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
        .trim()
        .to_string()
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .expect("rayon pool init");
    }

    let vp = Viewport {
        center: args.center,
        zoom:   args.zoom,
        width:  args.width,
        height: args.height,
    };

    if !args.json {
        eprintln!("bench_runner  git={}", git_hash());
        eprintln!("  resolution : {}×{}", args.width, args.height);
        eprintln!("  center     : [{}, {}]", args.center[0], args.center[1]);
        eprintln!("  zoom       : {}", args.zoom);
        eprintln!("  max_iter   : {}", args.max_iter);
        eprintln!("  runs       : {}", args.runs);
        eprintln!("  backend    : {:?}", args.backend);
        if args.perturbation {
            eprintln!("  mode       : {}", if args.use_sa { "perturbation+SA" } else { "perturbation" });
        }
        eprintln!("  rayon cpus : {}", rayon::current_num_threads());
        if !args.perturb_sweep && !args.compare_perturb {
            print_flop_budget(&args.fractals, args.max_iter);
        }
    }

    // ── perturbation zoom sweep ──────────────────────────────────────────────
    if args.perturb_sweep {
        let results = run_perturb_sweep(args.width, args.height, args.max_iter, args.runs);
        if args.json {
            print_perturb_json(&results);
        } else {
            print_perturb_table(&results);
        }
        return;
    }

    // ── period-3 bulb rejection validation ───────────────────────────────────
    if args.compare_bulb_reject {
        let (checked, bad) = bench_compare_bulb_reject(args.width, args.height, args.max_iter);
        if args.json {
            println!("{{\"checked_pixels\":{checked},\"bad_pixels\":{bad}}}");
        } else {
            println!("\nperiod-3 bulb rejection validation ({}×{}, max_iter={})", args.width, args.height, args.max_iter);
            println!("  pixels claimed in-bulb : {checked}");
            println!("  pixels that escape (BAD): {bad}");
            if bad == 0 {
                println!("  RESULT: SAFE (disk parameters are conservative)");
            } else {
                println!("  RESULT: UNSAFE — shrink PERIOD3_RADIUS_SQ in fractal.rs");
            }
        }
        return;
    }

    // ── neighbor iteration cap validation ────────────────────────────────────
    if args.compare_neighbor_cap {
        let (max_diff, mean_diff, recompute_pct) = bench_compare_neighbor_cap(&vp, args.max_iter, args.cap_slack);
        if args.json {
            println!("{{\"max_diff\":{max_diff},\"mean_diff\":{mean_diff},\"recompute_pct\":{recompute_pct}}}");
        } else {
            println!("\nneighbor iteration cap validation (slack={})", args.cap_slack);
            println!("  max_diff       : {max_diff} (expect 0.0 — correctness by construction)");
            println!("  mean_diff      : {mean_diff}");
            println!("  recompute_pct  : {recompute_pct:.2}% (pixels that hit the cap and paid for a full re-run)");
        }
        return;
    }

    // ── incremental pan (shift_and_fill) validation ──────────────────────────
    if let Some((dx, dy)) = args.compare_pan_recycle {
        let (max_diff, mean_diff, above) = bench_compare_pan_recycle(&vp, args.max_iter, dx, dy);
        if args.json {
            println!("{{\"dx\":{dx},\"dy\":{dy},\"max_diff\":{max_diff},\"mean_diff\":{mean_diff},\"above_0_01\":{above}}}");
        } else {
            println!("\nincremental pan validation (dx={dx}, dy={dy})");
            println!("  max_diff  : {max_diff} (expect a small non-zero value — see shift_and_fill doc comment)");
            println!("  mean_diff : {mean_diff}");
            println!("  pixels with diff > 0.01 : {above}");
        }
        return;
    }

    // ── multi-reference glitch correction validation ─────────────────────────
    if args.compare_multiref {
        let (primary_glitches, total, max_diff) = bench_compare_multiref(&vp, args.max_iter);
        if args.json {
            println!("{{\"primary_glitches\":{primary_glitches},\"total_pixels\":{total},\"max_diff_vs_scalar\":{max_diff}}}");
        } else {
            println!("\nmulti-reference glitch correction validation ({}×{}, max_iter={}, zoom={})", args.width, args.height, args.max_iter, args.zoom);
            println!("  primary-ref-only glitches : {primary_glitches}/{total} ({:.2}%)", primary_glitches as f64 / total as f64 * 100.0);
            println!("  max_diff vs scalar ground truth (after multiref+fallback): {max_diff} (expect 0.0 — correctness guaranteed by scalar fallback cap)");
        }
        return;
    }

    // ── DEM-culled Mariani-Silver validation ─────────────────────────────────
    if args.compare_dem_cull {
        let julia_c = [-0.4f64, 0.6];
        let exact = novafractal::fractal::render_mariani_silver(&vp, FractalType::Mandelbrot, julia_c, args.max_iter);
        let dem = render_mariani_silver_dem(&vp, FractalType::Mandelbrot, julia_c, args.max_iter, args.dem_k);
        let (max_diff, mean_diff, above) = pixel_diff_stats(&exact, &dem);
        if args.json {
            println!("{{\"k\":{},\"max_diff\":{max_diff},\"mean_diff\":{mean_diff},\"above_0_01\":{above}}}", args.dem_k);
        } else {
            println!("\nDEM-culled Mariani-Silver validation (k={})", args.dem_k);
            println!("  max_diff  : {max_diff} (interpolated fill is approximate — tune k against this)");
            println!("  mean_diff : {mean_diff}");
            println!("  pixels with diff > 0.01 : {above}");
        }
        return;
    }

    // ── IDE biased interior checking validation ──────────────────────────────
    if args.compare_ide {
        let julia_c = [-0.4f64, 0.6];
        let exact = render(&vp, FractalType::Mandelbrot, julia_c, args.max_iter);
        let ide = render_ide_biased(&vp, FractalType::Mandelbrot, julia_c, args.max_iter);
        let (max_diff, mean_diff, above) = pixel_diff_stats(&exact, &ide);
        if args.json {
            println!("{{\"max_diff\":{max_diff},\"mean_diff\":{mean_diff},\"above_0_01\":{above}}}");
        } else {
            println!("\nIDE biased interior checking validation");
            println!("  max_diff  : {max_diff} (derivative-bailout interior detection is approximate)");
            println!("  mean_diff : {mean_diff}");
            println!("  pixels with diff > 0.01 : {above}");
        }
        return;
    }

    // ── rebased perturbation validation ──────────────────────────────────────
    if args.compare_rebase {
        let (single_ms, rebase_ms, single_mm, rebase_mm, total) = bench_compare_rebase(&vp, args.max_iter, args.runs);
        if args.json {
            println!("{{\"single_ms\":{single_ms},\"rebase_ms\":{rebase_ms},\"single_mismatches\":{single_mm},\"rebase_mismatches\":{rebase_mm},\"total_pixels\":{total}}}");
        } else {
            println!("\nrebased perturbation validation ({}×{}, max_iter={}, zoom={})", args.width, args.height, args.max_iter, args.zoom);
            println!("  single-ref median : {single_ms:.2} ms ({single_mm} mismatches vs scalar)");
            println!("  rebase median     : {rebase_ms:.2} ms ({rebase_mm} mismatches vs scalar, small fraction expected)");
            println!("  speedup           : {:.2}x", single_ms / rebase_ms);
        }
        return;
    }

    // ── Hilbert tile traversal validation ────────────────────────────────────
    if args.tile_check {
        for &fractal in &args.fractals {
            let julia_c = [-0.4f64, 0.6];
            let a = render(&vp, fractal, julia_c, args.max_iter);
            let b = render_tiled(&vp, fractal, julia_c, args.max_iter);
            let (max_diff, mean_diff, above) = pixel_diff_stats(&a, &b);
            if args.json {
                println!("{{\"fractal\":\"{}\",\"max_diff\":{max_diff},\"mean_diff\":{mean_diff},\"above_0_01\":{above}}}", fractal.name());
            } else {
                println!("\ntile traversal validation: {} ({}×{}, max_iter={})", fractal.name(), args.width, args.height, args.max_iter);
                println!("  max_diff  : {max_diff} (expect 0.0 — same math, different write order)");
                println!("  mean_diff : {mean_diff}");
            }
        }
        return;
    }

    // ── heterogeneous CPU+GPU scheduler validation ───────────────────────────
    if args.heterogeneous {
        run_heterogeneous_check(&vp, &args);
        return;
    }

    // ── single zoom scalar vs perturbation compare ───────────────────────────
    if args.compare_perturb {
        let result = bench_compare_perturb(&vp, args.max_iter, args.runs);
        if args.json {
            print_perturb_json(&[result]);
        } else {
            print_perturb_table(&[result]);
        }
        return;
    }

    let mut samples: Vec<Sample> = Vec::new();

    if args.scaling {
        let ncpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let thread_counts: Vec<usize> = (0..=ncpus.trailing_zeros())
            .map(|e| 1usize << e)
            .collect();

        for &t in &thread_counts {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .expect("rayon pool");

            for &fractal in &args.fractals {
                let mut s = pool.install(|| {
                    bench_cpu(fractal, &vp, args.max_iter, args.runs, args.perturbation, args.use_sa)
                });
                s.threads = t;
                samples.push(s);
            }
        }
    } else {
        for &fractal in &args.fractals {
            match args.backend {
                Backend::Cpu => {
                    samples.push(bench_cpu(
                        fractal, &vp, args.max_iter, args.runs, args.perturbation, args.use_sa,
                    ));
                }
                Backend::Wgpu => {
                    if args.perturbation {
                        eprintln!("[wgpu] --perturbation is CPU-only — skipping");
                        continue;
                    }
                    if let Some(s) = bench_wgpu(fractal, &vp, args.max_iter, args.runs) {
                        samples.push(s);
                    } else {
                        eprintln!("[wgpu] no GPU available — skipping");
                    }
                }
                Backend::Hybrid => {
                    if args.perturbation {
                        eprintln!("[hybrid] --perturbation is CPU-only — skipping");
                        continue;
                    }
                    samples.push(bench_cpu(
                        fractal, &vp, args.max_iter, args.runs, false, false,
                    ));
                    if let Some(s) = bench_hybrid(fractal, &vp, args.max_iter, args.runs) {
                        samples.push(s);
                    } else {
                        eprintln!("[hybrid] no GPU available — only CPU baseline shown");
                    }
                }
            }
        }
    }

    if args.json {
        print_json(&samples);
    } else {
        print_table(&samples);
    }
}
