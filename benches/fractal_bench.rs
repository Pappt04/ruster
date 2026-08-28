use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use novafractal::fractal::{pixel, render, render_perturbation, render_perturbation_sa, compute_reference_orbit, compute_reference_orbit_f128, compute_series_approx, flops_per_iter, FractalType};
use novafractal::gpu::wgpu::fractal_compute::FractalCompute;
use novafractal::gpu::wgpu::uniforms::{PerturbUniforms, Uniforms};
use novafractal::gui::color::{colorize, ColorScheme};
use novafractal::gui::viewport::Viewport;
use rayon::ThreadPoolBuilder;
#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::sync::OnceLock;

// ── shared constants ──────────────────────────────────────────────────────────

const MAX_ITER: u32 = 1000;
const JULIA_C: [f64; 2] = [-0.4, 0.6];

const SAMPLE_POINTS: &[(f64, f64)] = &[
    (-0.75,  0.1),
    (-0.12,  0.75),
    (-0.5,   0.0),
    ( 0.28,  0.53),
    (-1.401, 0.0),
];

fn vp(w: u32, h: u32) -> Viewport {
    Viewport { center: [-0.5, 0.0], zoom: 1.0, width: w, height: h }
}

// ── wgpu shared state (initialized once for the whole benchmark run) ──────────

struct GpuState {
    device: wgpu::Device,
    queue:  wgpu::Queue,
    info:   String,
}

static GPU: OnceLock<Option<GpuState>> = OnceLock::new();

// ── CUDA shared device (initialized once for the whole benchmark run) ─────────
//
// Each `bench_cuda_*` function used to construct its own `CudaFractal::new()`
// — a fresh `CudaDevice::new(0)` (fresh primary-context retain + PTX module
// load) per benchmark group/resolution, ~25+ over a full `cuda/` sweep.
// Churning through that many contexts back-to-back in one process was
// observed to destabilize the driver on this laptop's Optimus setup (a sweep
// hung indefinitely, 0% GPU utilization, deep into the run — the same
// operations completed in well under a second when run in isolation). One
// shared device, mirroring `GPU`/`gpu()` above, avoids the repeated
// retain/release/module-load churn entirely.
#[cfg(feature = "cuda")]
static CUDA_DEV: OnceLock<Arc<cudarc::driver::CudaDevice>> = OnceLock::new();

#[cfg(feature = "cuda")]
fn cuda_dev() -> Arc<cudarc::driver::CudaDevice> {
    CUDA_DEV.get_or_init(|| cudarc::driver::CudaDevice::new(0).expect("no CUDA device")).clone()
}

// `WGPU_ADAPTER=integrated cargo bench --bench fractal_bench` reruns the whole
// wgpu suite (bench_wgpu_*) against the AMD Vega instead of the RTX 3050 — an
// env var rather than a second copy of every bench_wgpu_* function, since
// criterion binaries don't get to define their own CLI flags. Selecting by
// explicit `DeviceType` rather than `PowerPreference` removes any ambiguity
// about which adapter actually got measured (results/summary.md §1.1).
fn gpu() -> Option<&'static GpuState> {
    GPU.get_or_init(|| {
        pollster::block_on(async {
            let integrated = std::env::var("WGPU_ADAPTER").as_deref() == Ok("integrated");
            let wanted = if integrated {
                wgpu::DeviceType::IntegratedGpu
            } else {
                wgpu::DeviceType::DiscreteGpu
            };
            let instance = wgpu::Instance::default();
            let adapter = instance
                .enumerate_adapters(wgpu::Backends::all())
                .into_iter()
                .find(|a| a.get_info().device_type == wanted)?;
            let info = format!("{} ({:?}, {:?})", adapter.get_info().name, adapter.get_info().device_type, adapter.get_info().backend);
            let (device, queue) = adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("bench"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            ).await.ok()?;
            Some(GpuState { device, queue, info })
        })
    }).as_ref()
}

// ── viewport → wgpu Uniforms ──────────────────────────────────────────────────

fn uniforms(vp: &Viewport, fractal: FractalType) -> Uniforms {
    let aspect = vp.width as f64 / vp.height as f64;
    let half   = 2.0 / vp.zoom;
    Uniforms {
        re_start: (vp.center[0] + (0.5 / vp.width  as f64 - 0.5) * half * aspect * 2.0) as f32,
        im_start: (vp.center[1] + (0.5 / vp.height as f64 - 0.5) * half * 2.0) as f32,
        re_step:  (half * aspect * 2.0 / vp.width  as f64) as f32,
        im_step:  (half * 2.0          / vp.height as f64) as f32,
        julia_cr: JULIA_C[0] as f32,
        julia_ci: JULIA_C[1] as f32,
        max_iter: MAX_ITER,
        fractal:  fractal_u32(fractal),
        width:    vp.width,
        height:   vp.height,
        _pad:     [0; 2],
    }
}

fn fractal_u32(f: FractalType) -> u32 {
    match f {
        FractalType::Mandelbrot => 0,
        FractalType::Julia      => 1,
        FractalType::Newton     => 2,
        FractalType::Nova       => 3,
    }
}

// ── build viewport uniforms for the bottom half of a split image ──────────────

fn uniforms_bottom_half(full_vp: &Viewport, fractal: FractalType) -> (Viewport, Uniforms) {
    let h_top    = full_vp.height / 2;
    let h_bottom = full_vp.height - h_top;
    let aspect   = full_vp.width as f64 / full_vp.height as f64;
    let half     = 2.0 / full_vp.zoom;
    let im_step  = half * 2.0 / full_vp.height as f64;
    let im_start = full_vp.center[1] + (0.5 / full_vp.height as f64 - 0.5) * half * 2.0;
    let im_start_bottom = im_start + h_top as f64 * im_step;

    let vp_top = Viewport { width: full_vp.width, height: h_top, ..full_vp.clone() };

    let u = Uniforms {
        re_start: (full_vp.center[0] + (0.5 / full_vp.width as f64 - 0.5) * half * aspect * 2.0) as f32,
        im_start: im_start_bottom as f32,
        re_step:  (half * aspect * 2.0 / full_vp.width as f64) as f32,
        im_step:  im_step as f32,
        julia_cr: JULIA_C[0] as f32,
        julia_ci: JULIA_C[1] as f32,
        max_iter: MAX_ITER,
        fractal:  fractal_u32(fractal),
        width:    full_vp.width,
        height:   h_bottom,
        _pad:     [0; 2],
    };
    (vp_top, u)
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPU benchmarks
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_pixel_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu/pixel_kernel");

    for &fractal in FractalType::ALL {
        group.bench_function(fractal.name(), |b| {
            b.iter(|| {
                let mut acc = 0.0f32;
                for &(re, im) in SAMPLE_POINTS {
                    acc += pixel(black_box(fractal), black_box(re), black_box(im),
                                 JULIA_C, MAX_ITER);
                }
                acc
            })
        });
    }
    group.finish();
}

fn bench_cpu_render(c: &mut Criterion) {
    let resolutions: &[(u32, u32, &str)] = &[
        (800,  600,  "800×600"),
        (1920, 1080, "1920×1080"),
        (3840, 2160, "3840×2160"),
    ];

    for &fractal in FractalType::ALL {
        let mut group = c.benchmark_group(format!("cpu/render/{}", fractal.name()));
        group.sample_size(10);

        for &(w, h, label) in resolutions {
            let vp = vp(w, h);
            group.throughput(Throughput::Elements(w as u64 * h as u64));
            group.bench_with_input(
                BenchmarkId::new("rayon", label),
                &vp,
                |b, vp| b.iter(|| render(black_box(vp), fractal, JULIA_C, MAX_ITER)),
            );
        }
        group.finish();
    }
}

/// Hilbert-tile traversal vs plain row-parallel render (2b in
/// CURSOR_OPTIMIZATIONS.md). Bit-identical output — measures cache-locality gain.
fn bench_tiled_render(c: &mut Criterion) {
    use novafractal::fractal::render_tiled;

    let resolutions: &[(u32, u32, &str)] = &[
        (800,  600,  "800×600"),
        (1920, 1080, "1920×1080"),
        (3840, 2160, "3840×2160"),
    ];

    for &fractal in FractalType::ALL {
        let mut group = c.benchmark_group(format!("cpu/render_tiled/{}", fractal.name()));
        group.sample_size(10);

        for &(w, h, label) in resolutions {
            let vp = vp(w, h);
            group.throughput(Throughput::Elements(w as u64 * h as u64));
            group.bench_with_input(BenchmarkId::new("rows",    label), &vp,
                |b, vp| b.iter(|| render(       black_box(vp), fractal, JULIA_C, MAX_ITER)));
            group.bench_with_input(BenchmarkId::new("hilbert", label), &vp,
                |b, vp| b.iter(|| render_tiled( black_box(vp), fractal, JULIA_C, MAX_ITER)));
        }
        group.finish();
    }
}

fn bench_colorize(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu/colorize");

    for &(w, h, label) in &[(800u32, 600u32, "800×600"), (1920, 1080, "1920×1080")] {
        let buf = render(&vp(w, h), FractalType::Mandelbrot, JULIA_C, MAX_ITER);
        group.throughput(Throughput::Elements(w as u64 * h as u64));
        group.bench_with_input(
            BenchmarkId::new("inferno", label),
            &buf,
            |b, buf| b.iter(|| colorize(black_box(buf), MAX_ITER, ColorScheme::Inferno)),
        );
    }
    group.finish();
}

fn bench_cpu_pipeline(c: &mut Criterion) {
    let vp = vp(1920, 1080);
    let mut group = c.benchmark_group("cpu/pipeline");
    group.throughput(Throughput::Elements(1920 * 1080));
    group.sample_size(10);

    for &fractal in FractalType::ALL {
        group.bench_function(fractal.name(), |b| {
            b.iter(|| {
                let buf = render(black_box(&vp), fractal, JULIA_C, MAX_ITER);
                colorize(black_box(&buf), MAX_ITER, ColorScheme::Inferno)
            })
        });
    }
    group.finish();
}

fn bench_thread_scaling(c: &mut Criterion) {
    let thread_counts = [1usize, 2, 4, 8, 16];
    let vp = vp(1920, 1080);
    let mut group = c.benchmark_group("cpu/thread_scaling/Mandelbrot_1080p");
    group.throughput(Throughput::Elements(1920 * 1080));
    group.sample_size(10);

    for &n in &thread_counts {
        let pool = ThreadPoolBuilder::new().num_threads(n).build().unwrap();
        group.bench_with_input(
            BenchmarkId::new("threads", n),
            &n,
            |b, _| pool.install(|| {
                b.iter(|| render(black_box(&vp), FractalType::Mandelbrot, JULIA_C, MAX_ITER))
            }),
        );
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIMD benchmarks  (only compiled and run with --features simd)
// Compares scalar render() vs f64x4 render_simd() vs f32x8 render_simd_f32()
// for Mandelbrot and Julia at three resolutions.
// Note: render_simd_f32 is only accurate below zoom ~1e6; all runs use zoom=1.
// ═══════════════════════════════════════════════════════════════════════════════

const SIMD_FRACTALS: &[FractalType] = &[FractalType::Mandelbrot, FractalType::Julia];

fn bench_simd_render(c: &mut Criterion) {
    use novafractal::fractal::{render_simd, render_simd_f32};

    let resolutions: &[(u32, u32, &str)] = &[
        (800,  600,  "800×600"),
        (1920, 1080, "1920×1080"),
        (3840, 2160, "3840×2160"),
    ];

    for &fractal in SIMD_FRACTALS {
        let mut group = c.benchmark_group(format!("simd/render/{}", fractal.name()));
        group.sample_size(10);

        for &(w, h, label) in resolutions {
            let vp = vp(w, h);
            group.throughput(Throughput::Elements(w as u64 * h as u64));

            group.bench_with_input(BenchmarkId::new("scalar", label), &vp,
                |b, vp| b.iter(|| render(          black_box(vp), fractal, JULIA_C, MAX_ITER)));
            group.bench_with_input(BenchmarkId::new("f64x4",  label), &vp,
                |b, vp| b.iter(|| render_simd(     black_box(vp), fractal, JULIA_C, MAX_ITER)));
            group.bench_with_input(BenchmarkId::new("f32x8",  label), &vp,
                |b, vp| b.iter(|| render_simd_f32( black_box(vp), fractal, JULIA_C, MAX_ITER)));
        }
        group.finish();
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIMD ILP: f32x8 vs interleaved 2×f32x8 (CURSOR_OPTIMIZATIONS.md 2a). Mandelbrot
// only — bit-identical output to render_simd_f32, only faster scheduling.
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_simd_ilp_render(c: &mut Criterion) {
    use novafractal::fractal::{render_simd_f32, render_simd_f32_ilp};

    let resolutions: &[(u32, u32, &str)] = &[
        (800,  600,  "800×600"),
        (1920, 1080, "1920×1080"),
        (3840, 2160, "3840×2160"),
    ];

    let mut group = c.benchmark_group("simd/render_ilp/Mandelbrot_1080p");
    group.sample_size(10);

    for &(w, h, label) in resolutions {
        let vp = vp(w, h);
        group.throughput(Throughput::Elements(w as u64 * h as u64));

        group.bench_with_input(BenchmarkId::new("f32x8",     label), &vp,
            |b, vp| b.iter(|| render_simd_f32(    black_box(vp), FractalType::Mandelbrot, JULIA_C, MAX_ITER)));
        group.bench_with_input(BenchmarkId::new("f32x8_ilp", label), &vp,
            |b, vp| b.iter(|| render_simd_f32_ilp(black_box(vp), FractalType::Mandelbrot, JULIA_C, MAX_ITER)));
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// wgpu (GPU) benchmarks
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_wgpu_render(c: &mut Criterion) {
    let Some(g) = gpu() else {
        eprintln!("[wgpu] no GPU available — skipping wgpu/render benchmarks");
        return;
    };
    eprintln!("[wgpu] {}", g.info);

    let resolutions: &[(u32, u32, &str)] = &[
        (800,  600,  "800×600"),
        (1920, 1080, "1920×1080"),
        (3840, 2160, "3840×2160"),
    ];

    for &fractal in FractalType::ALL {
        let mut group = c.benchmark_group(format!("wgpu/render/{}", fractal.name()));
        group.sample_size(10);

        for &(w, h, label) in resolutions {
            let vp = vp(w, h);
            let compute = FractalCompute::new(&g.device, w, h);
            let uni = uniforms(&vp, fractal);

            group.throughput(Throughput::Elements(w as u64 * h as u64));
            group.bench_with_input(
                BenchmarkId::new("gpu", label),
                &uni,
                |b, &uni| b.iter(|| compute.render(&g.device, &g.queue, black_box(uni))),
            );
        }
        group.finish();
    }
}

fn bench_wgpu_pipeline(c: &mut Criterion) {
    let Some(g) = gpu() else {
        eprintln!("[wgpu] no GPU — skipping wgpu/pipeline");
        return;
    };

    let vp = vp(1920, 1080);
    let mut group = c.benchmark_group("wgpu/pipeline");
    group.throughput(Throughput::Elements(1920 * 1080));
    group.sample_size(10);

    for &fractal in FractalType::ALL {
        let compute = FractalCompute::new(&g.device, 1920, 1080);
        let uni = uniforms(&vp, fractal);

        group.bench_function(fractal.name(), |b| {
            b.iter(|| {
                let buf = compute.render(&g.device, &g.queue, black_box(uni));
                colorize(black_box(&buf), MAX_ITER, ColorScheme::Inferno)
            })
        });
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPU + wgpu hybrid benchmarks
// Splits the image: CPU renders the top half, wgpu renders the bottom half,
// both in concurrent threads.  Wall time = max(t_cpu, t_gpu).
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_hybrid_cpu_wgpu(c: &mut Criterion) {
    let Some(g) = gpu() else {
        eprintln!("[hybrid] no GPU — skipping hybrid/cpu+wgpu benchmarks");
        return;
    };

    let full_vp = vp(1920, 1080);
    let pixels  = 1920u64 * 1080;

    let mut group = c.benchmark_group("hybrid/cpu+wgpu");
    group.throughput(Throughput::Elements(pixels));
    group.sample_size(10);

    for &fractal in FractalType::ALL {
        let (vp_top, uni_bottom) = uniforms_bottom_half(&full_vp, fractal);
        let h_bottom = full_vp.height - full_vp.height / 2;
        let compute  = FractalCompute::new(&g.device, full_vp.width, h_bottom);

        group.bench_function(fractal.name(), |b| {
            b.iter(|| {
                std::thread::scope(|s| {
                    // CPU: top half (rayon inside)
                    let top = s.spawn(|| render(black_box(&vp_top), fractal, JULIA_C, MAX_ITER));
                    // GPU: bottom half
                    let bottom = compute.render(&g.device, &g.queue, black_box(uni_bottom));
                    let _ = top.join().unwrap();
                    bottom
                })
            })
        });
    }
    group.finish();
}

// Corner-sampling, work-stealing heterogeneous scheduler (see src/scheduler),
// driving wgpu instead of CUDA — vs plain GPU-only render, across zoom levels.
// `bench_hybrid_cpu_wgpu` above is the OLD static top/bottom-half split, kept
// for comparison; this is the adaptive replacement for the wgpu backend,
// mirroring `bench_heterogeneous`'s CUDA version below.
// `scheduler` itself is only compiled with the `cuda` feature (see src/lib.rs),
// even though this particular benchmark drives the wgpu backend.
#[cfg(feature = "cuda")]
fn bench_heterogeneous_wgpu(c: &mut Criterion) {
    use novafractal::scheduler::{render_heterogeneous_wgpu, controller::ThresholdController, SchedulerConfig};

    let Some(g) = gpu() else {
        eprintln!("[hybrid] no GPU — skipping hybrid/heterogeneous_wgpu benchmarks");
        return;
    };

    const ZOOMS: &[(f64, &str)] = &[(1.0, "zoom_1e0"), (1e2, "zoom_1e2"), (1e4, "zoom_1e4")];
    const CENTER: [f64; 2] = [-0.75, 0.1];

    let mut group = c.benchmark_group("hybrid/heterogeneous_wgpu");
    group.throughput(Throughput::Elements(1920 * 1080));
    group.sample_size(10);

    for &fractal in FractalType::ALL {
        for &(zoom, label) in ZOOMS {
            let frame_vp = Viewport { center: CENTER, zoom, width: 1920, height: 1080 };
            let compute = FractalCompute::new(&g.device, 1920, 1080);
            // Starting guess for the normalized corner-spread threshold
            // (~[0,1]) — see scheduler::controller's doc comment.
            let mut controller = ThresholdController::new(0.02);
            let cfg = SchedulerConfig::default();

            group.bench_with_input(
                BenchmarkId::new(fractal.name(), label),
                &frame_vp,
                |b, frame_vp| {
                    b.iter(|| {
                        render_heterogeneous_wgpu(
                            black_box(frame_vp), fractal, JULIA_C, MAX_ITER,
                            &compute, &g.device, &g.queue, &mut controller, &cfg,
                        ).buf
                    })
                },
            );
        }
    }
    group.finish();
}

// Stub when cuda feature is off — criterion_group! needs a consistent list.
#[cfg(not(feature = "cuda"))]
fn bench_heterogeneous_wgpu(c: &mut Criterion) { let _ = c; }

// ═══════════════════════════════════════════════════════════════════════════════
// CUDA benchmarks  (only compiled and run with --features cuda)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
fn bench_cuda_render(c: &mut Criterion) {
    use novafractal::gpu::cuda::CudaFractal;

    let resolutions: &[(u32, u32, &str)] = &[
        (800,  600,  "800×600"),
        (1920, 1080, "1920×1080"),
        (3840, 2160, "3840×2160"),
    ];

    for &fractal in FractalType::ALL {
        let mut group = c.benchmark_group(format!("cuda/render/{}", fractal.name()));
        group.sample_size(10);

        for &(w, h, label) in resolutions {
            let vp = vp(w, h);
            let aspect = w as f64 / h as f64;
            let half   = 2.0 / vp.zoom;
            let re_start = vp.center[0] + (0.5 / w as f64 - 0.5) * half * aspect * 2.0;
            let im_start = vp.center[1] + (0.5 / h as f64 - 0.5) * half * 2.0;
            let re_step  = half * aspect * 2.0 / w as f64;
            let im_step  = half * 2.0 / h as f64;

            let mut cuda = CudaFractal::from_device(cuda_dev(), w, h);
            group.throughput(Throughput::Elements(w as u64 * h as u64));
            group.bench_with_input(
                BenchmarkId::new("cuda", label),
                &(re_start, im_start, re_step, im_step),
                |b, &(re_s, im_s, re_step, im_step)| {
                    b.iter(|| cuda.render(
                        black_box(re_s), black_box(im_s),
                        black_box(re_step), black_box(im_step),
                        JULIA_C[0], JULIA_C[1],
                        MAX_ITER, fractal_u32(fractal),
                    ))
                },
            );
        }
        group.finish();
    }
}

#[cfg(feature = "cuda")]
fn bench_cuda_pipeline(c: &mut Criterion) {
    use novafractal::gpu::cuda::CudaFractal;
    let vp = vp(1920, 1080);
    let aspect = 1920.0f64 / 1080.0;
    let half   = 2.0 / vp.zoom;
    let re_start = vp.center[0] + (0.5 / 1920.0 - 0.5) * half * aspect * 2.0;
    let im_start = vp.center[1] + (0.5 / 1080.0 - 0.5) * half * 2.0;
    let re_step  = half * aspect * 2.0 / 1920.0;
    let im_step  = half * 2.0 / 1080.0;

    let mut group = c.benchmark_group("cuda/pipeline");
    group.throughput(Throughput::Elements(1920 * 1080));
    group.sample_size(10);

    for &fractal in FractalType::ALL {
        let mut cuda = CudaFractal::from_device(cuda_dev(), 1920, 1080);
        group.bench_function(fractal.name(), |b| {
            b.iter(|| {
                let buf = cuda.render(
                    re_start, im_start, re_step, im_step,
                    JULIA_C[0], JULIA_C[1], MAX_ITER, fractal_u32(fractal),
                );
                colorize(black_box(&buf), MAX_ITER, ColorScheme::Inferno)
            })
        });
    }
    group.finish();
}

// Corner-sampling, work-stealing heterogeneous scheduler (see src/scheduler)
// vs plain GPU-only render — across zoom levels to show the CPU/GPU tile
// split adapting as the boundary-to-interior ratio grows. (The old static
// top/bottom-half "50/50" comparison benchmark, bench_hybrid_cpu_cuda, was
// deleted — it wasn't part of the real scheduler design, and the meaningful
// comparison is against plain single-backend rendering, not a dumb split.)
#[cfg(feature = "cuda")]
fn bench_heterogeneous(c: &mut Criterion) {
    use novafractal::gpu::cuda::CudaFractal;
    use novafractal::scheduler::{render_heterogeneous, controller::ThresholdController, SchedulerConfig};

    const ZOOMS: &[(f64, &str)] = &[(1.0, "zoom_1e0"), (1e2, "zoom_1e2"), (1e4, "zoom_1e4")];
    const CENTER: [f64; 2] = [-0.75, 0.1];

    let mut group = c.benchmark_group("hybrid/heterogeneous");
    group.throughput(Throughput::Elements(1920 * 1080));
    group.sample_size(10);

    for &fractal in FractalType::ALL {
        for &(zoom, label) in ZOOMS {
            let frame_vp = Viewport { center: CENTER, zoom, width: 1920, height: 1080 };
            let mut cuda = CudaFractal::from_device(cuda_dev(), 1920, 1080);
            // Starting guess for the normalized corner-spread threshold
            // (~[0,1]) — see scheduler::controller's doc comment.
            let mut controller = ThresholdController::new(0.02);
            let cfg = SchedulerConfig::default();

            group.bench_with_input(
                BenchmarkId::new(fractal.name(), label),
                &frame_vp,
                |b, frame_vp| {
                    b.iter(|| {
                        render_heterogeneous(
                            black_box(frame_vp), fractal, JULIA_C, MAX_ITER,
                            &mut cuda, &mut controller, &cfg,
                        ).buf
                    })
                },
            );
        }
    }
    group.finish();
}

#[cfg(not(feature = "cuda"))]
fn bench_heterogeneous(c: &mut Criterion) { let _ = c; }



// ═══════════════════════════════════════════════════════════════════════════════
// Algorithmic optimisations that skip or reuse work
// ═══════════════════════════════════════════════════════════════════════════════
//
// Everything above measures how fast a full per-pixel evaluation runs. These
// groups measure techniques that try to avoid doing it: proving a region is
// uniform and flood-filling it, bounding a pixel's iteration count from its
// neighbour, or reusing the previous frame outright. They were all implemented
// but never benchmarked, so their value in this renderer was unknown.
//
// All are compared against plain `render()` on the SAME viewport, since the
// question is only ever "is this cheaper than just computing every pixel?".
// Every one of them returns a full IterBuf, so a slower result means the
// bookkeeping cost more than the arithmetic it avoided.
//
// TWO baselines, and the distinction is load-bearing. `render()`,
// `render_ide_biased` and `render_neighbor_capped` are rayon-parallel over rows;
// `render_mariani_silver[_dem]` is inherently SEQUENTIAL — its recursive
// rectangle subdivision carries a data dependency that this implementation does
// not parallelise. Measuring it against a 16-thread baseline therefore reports
// the thread count, not the algorithm, and would make a work-skipping technique
// look ~10x worse than it is. So:
//
//   baseline_16t  — "should I use this in this renderer?"  (engineering answer)
//   baseline_1t   — "does this technique save work at all?" (algorithmic answer)
//
// Compare the sequential techniques against `baseline_1t` and the parallel ones
// against `baseline_16t`.

fn bench_work_skipping(c: &mut Criterion) {
    use novafractal::fractal::{
        render_mariani_silver, render_mariani_silver_dem, render_ide_biased,
        render_neighbor_capped,
    };
    let pool1 = ThreadPoolBuilder::new().num_threads(1).build().unwrap();

    // Three viewports with deliberately different interior/boundary ratios —
    // these techniques all exploit large uniform regions, so a whole-set view
    // (much interior) and a boundary-dominated crop should behave differently.
    const VIEWS: &[(f64, [f64; 2], &str)] = &[
        (1.0, [-0.5, 0.0], "whole_set"),          // large interior + large exterior
        (1e2, [-0.745, 0.113], "boundary"),       // seahorse valley, boundary-heavy
        (1e4, [-0.745, 0.113], "deep_boundary"),
    ];

    let mut group = c.benchmark_group("skipping/mandelbrot_1080p");
    group.throughput(Throughput::Elements(1920 * 1080));
    group.sample_size(10);

    for &(zoom, center, label) in VIEWS {
        let v = Viewport { center, zoom, width: 1920, height: 1080 };
        group.bench_with_input(BenchmarkId::new("baseline_16t", label), &v, |b, v| {
            b.iter(|| black_box(render(black_box(v), FractalType::Mandelbrot, JULIA_C, MAX_ITER)))
        });
        // The comparison point for the sequential techniques below.
        group.bench_with_input(BenchmarkId::new("baseline_1t", label), &v, |b, v| {
            b.iter(|| pool1.install(|| black_box(
                render(black_box(v), FractalType::Mandelbrot, JULIA_C, MAX_ITER))))
        });
        // Mariani-Silver: trace a rectangle's border; if every border pixel has
        // the same value the interior must too (the escape-time field has no
        // holes), so flood-fill it. Otherwise subdivide.
        group.bench_with_input(BenchmarkId::new("mariani_silver_SEQ", label), &v, |b, v| {
            b.iter(|| black_box(render_mariani_silver(black_box(v), FractalType::Mandelbrot, JULIA_C, MAX_ITER)))
        });
        // Same, but the subdivision floor is driven by a distance estimate
        // rather than a fixed minimum size.
        group.bench_with_input(BenchmarkId::new("mariani_silver_dem_SEQ", label), &v, |b, v| {
            b.iter(|| black_box(render_mariani_silver_dem(black_box(v), FractalType::Mandelbrot, JULIA_C, MAX_ITER, 1.0)))
        });
        // Interior distance estimation: prove a point is interior without
        // iterating to max_iter, by tracking the derivative of the cycle.
        group.bench_with_input(BenchmarkId::new("interior_dist_est", label), &v, |b, v| {
            b.iter(|| black_box(render_ide_biased(black_box(v), FractalType::Mandelbrot, JULIA_C, MAX_ITER)))
        });
        // Neighbour capping: a pixel's iteration count differs from its
        // left neighbour's by a bounded amount in smooth regions, so cap the
        // loop at neighbour+slack and only recompute exactly when the cap is hit.
        group.bench_with_input(BenchmarkId::new("neighbour_capped", label), &v, |b, v| {
            b.iter(|| black_box(render_neighbor_capped(black_box(v), FractalType::Mandelbrot, JULIA_C, MAX_ITER, 16)))
        });
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Perturbation glitch-handling strategies
// ═══════════════════════════════════════════════════════════════════════════════
//
// `bench_perturbation_render` above measures single-reference perturbation, whose
// weakness is glitches: pixels where the chosen reference orbit stops being a
// valid linearisation and which must fall back to full-precision iteration.
// Measured glitch rate across 15 random references at zoom 1e9 was 6.67% mean
// with a 24.94% standard deviation and a full 0-100% range (results/summary.md
// §5.8), i.e. a single reference is a lottery. These two strategies exist to
// remove that variance; this group measures what they cost.
fn bench_perturbation_strategies(c: &mut Criterion) {
    use novafractal::fractal::{render_perturbation_multiref, render_perturbation_rebase};

    let mut group = c.benchmark_group("perturbation/glitch_strategy");
    group.throughput(Throughput::Elements(1920 * 1080));
    group.sample_size(10);

    for &(zoom, label) in PERTURB_ZOOMS {
        let vp = Viewport { center: PERTURB_CENTER, zoom, width: 1920, height: 1080 };
        group.bench_with_input(BenchmarkId::new("scalar", label), &vp, |b, vp| {
            b.iter(|| black_box(render(black_box(vp), FractalType::Mandelbrot, JULIA_C, MAX_ITER)))
        });
        group.bench_with_input(BenchmarkId::new("single_ref", label), &vp, |b, vp| {
            b.iter(|| black_box(render_perturbation(black_box(vp), FractalType::Mandelbrot, JULIA_C, MAX_ITER)))
        });
        // Several reference orbits; each pixel uses the nearest valid one, so a
        // single bad reference no longer poisons the frame.
        group.bench_with_input(BenchmarkId::new("multi_ref", label), &vp, |b, vp| {
            b.iter(|| black_box(render_perturbation_multiref(black_box(vp), FractalType::Mandelbrot, JULIA_C, MAX_ITER)))
        });
        // Rebasing: when a pixel's epsilon grows too large relative to the
        // reference, restart the perturbation from the current orbit point
        // instead of falling back to full-precision iteration.
        group.bench_with_input(BenchmarkId::new("rebase", label), &vp, |b, vp| {
            b.iter(|| black_box(render_perturbation_rebase(black_box(vp), FractalType::Mandelbrot, JULIA_C, MAX_ITER)))
        });
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Frame-to-frame reuse: incremental pan
// ═══════════════════════════════════════════════════════════════════════════════
//
// `shift_and_fill` exploits the fact that panning by a whole number of pixels
// leaves most of the previous frame valid: memmove the overlap and compute only
// the newly exposed strip. Cost should scale with the strip, not the frame, so
// this sweeps pan distance to check that it does.
//
// The pan MUST be an exact integer pixel count. `Viewport::delta_pixels`
// requires the shift to round to within 1e-6 of a whole pixel; an arbitrary
// real-valued centre shift silently falls through to a full recompute, which
// would make this benchmark measure `render()` under another name. The
// viewports here are constructed from the pixel step so that cannot happen,
// and the assertion below fails loudly if it ever does.
fn bench_incremental_pan(c: &mut Criterion) {
    use novafractal::fractal::{pixel_grid, shift_and_fill};

    let base = Viewport { center: [-0.745, 0.113], zoom: 1e2, width: 1920, height: 1080 };
    let pg = pixel_grid(&base);

    let mut group = c.benchmark_group("incremental_pan/1080p");
    group.throughput(Throughput::Elements(1920 * 1080));
    group.sample_size(20);

    group.bench_function("full_render", |b| {
        b.iter(|| black_box(render(black_box(&base), FractalType::Mandelbrot, JULIA_C, MAX_ITER)))
    });

    // Pan distances as a fraction of the frame: a 1px nudge, a slow drag, a
    // fast fling. The last is close to "most of the frame is new".
    for &dx in &[1i32, 32, 192, 960] {
        let panned = Viewport {
            center: [base.center[0] + dx as f64 * pg.re_step, base.center[1]],
            ..base
        };
        let delta = base.delta_pixels(&panned)
            .expect("pan must round to a whole pixel or shift_and_fill degrades to a full render");
        assert_eq!(delta.0.abs(), dx, "constructed pan did not land on the intended pixel offset");

        let seed = render(&base, FractalType::Mandelbrot, JULIA_C, MAX_ITER);
        group.bench_with_input(BenchmarkId::new("shift_and_fill", dx), &dx, |b, _| {
            b.iter_batched(
                || seed.clone(),
                |mut buf| {
                    shift_and_fill(&mut buf, 1920, 1080, delta.0, delta.1,
                                   &panned, FractalType::Mandelbrot, JULIA_C, MAX_ITER);
                    black_box(buf)
                },
                criterion::BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// Deep-zoom heterogeneous scheduler  (--features cuda)
// ═══════════════════════════════════════════════════════════════════════════════
//
// `bench_heterogeneous` above is deliberately LEFT ALONE: its 1e0..1e4 sweep is
// the evidence for when the scheduler is *not* viable, and that is a result
// worth keeping verbatim.
//
// This group extends the sweep across `F32_PRECISION_THRESHOLD` (1e6), which is
// the boundary that decides everything. Below it, Mandelbrot takes
// `fractal_kernel_f32` and the GPU is ~4x the CPU, so no scheduler can win — the
// best it can do is lose by less. Above it the GPU falls back to the f64 kernel,
// which Ampere GeForce runs at 1/64 the fp32 rate, the CPU becomes competitive,
// and a split has something real to exploit.
//
// Two design changes over the shallow group, both aimed at making the result
// interpretable rather than just fast:
//
//   1. **Solo-backend reference arms in the same group.** Each zoom benchmarks
//      `cuda`, `cpu` and `hybrid` side by side. The shallow group has no
//      reference arm at all, which is exactly why its numbers looked acceptable
//      in isolation while actually being 20-30% worse than just using the GPU.
//      A hybrid number is meaningless without the two it must beat.
//   2. **`render_heterogeneous_into` with a `PinnedBuf`**, i.e. the form a real
//      render loop would use, rather than allocating a frame per call.
#[cfg(feature = "cuda")]
fn bench_heterogeneous_deep(c: &mut Criterion) {
    use novafractal::gpu::cuda::{CudaFractal, PinnedBuf};
    use novafractal::scheduler::{render_heterogeneous_into, controller::ThresholdController, SchedulerConfig};

    // Straddles F32_PRECISION_THRESHOLD (1e6). 1e15 is past f64's ability to
    // resolve individual pixels, where perturbation becomes the only correct
    // path — included to show the scheduler's behaviour there rather than
    // stopping short of it.
    const ZOOMS: &[(f64, &str)] = &[
        (1e4, "zoom_1e4"), (1e5, "zoom_1e5"), (1e6, "zoom_1e6"),
        (1e7, "zoom_1e7"), (1e9, "zoom_1e9"), (1e12, "zoom_1e12"), (1e15, "zoom_1e15"),
    ];
    const CENTER: [f64; 2] = [-0.745, 0.113];

    let mut group = c.benchmark_group("hybrid/heterogeneous_deep");
    group.throughput(Throughput::Elements(1920 * 1080));
    // Deep-zoom frames run 60-200 ms, so the default 100 samples would push
    // this group past 20 s per arm.
    group.sample_size(10);

    let mut cuda = CudaFractal::from_device(cuda_dev(), 1920, 1080);
    let mut pinned = PinnedBuf::new(1920 * 1080);
    assert!(pinned.is_pinned(), "page-locking failed — deep-zoom readback numbers would not be representative");
    let cfg = SchedulerConfig::default();

    for &(zoom, label) in ZOOMS {
        let vp = Viewport { center: CENTER, zoom, width: 1920, height: 1080 };
        let pg = novafractal::fractal::pixel_grid(&vp);

        // Solo GPU.
        group.bench_function(BenchmarkId::new("cuda", label), |b| {
            b.iter(|| black_box(cuda.render(
                pg.re_start, pg.im_start, pg.re_step, pg.im_step,
                JULIA_C[0], JULIA_C[1], MAX_ITER, 0,
            )))
        });

        // Solo CPU.
        group.bench_function(BenchmarkId::new("cpu", label), |b| {
            b.iter(|| black_box(render(black_box(&vp), FractalType::Mandelbrot, JULIA_C, MAX_ITER)))
        });

        // Hybrid. The adaptive threshold is converged before timing — an
        // unconverged controller measures the transient, not the scheduler.
        let mut controller = ThresholdController::new(0.02);
        for _ in 0..25 {
            render_heterogeneous_into(&vp, FractalType::Mandelbrot, JULIA_C, MAX_ITER,
                                      &mut cuda, &mut controller, &cfg, &mut pinned);
        }
        group.bench_function(BenchmarkId::new("hybrid", label), |b| {
            b.iter(|| black_box(render_heterogeneous_into(
                black_box(&vp), FractalType::Mandelbrot, JULIA_C, MAX_ITER,
                &mut cuda, &mut controller, &cfg, &mut pinned,
            )))
        });
    }
    group.finish();
}

#[cfg(not(feature = "cuda"))]
fn bench_heterogeneous_deep(c: &mut Criterion) { let _ = c; }

// Stubs when cuda feature is off — criterion_group! needs a consistent list.
#[cfg(not(feature = "cuda"))]
fn bench_cuda_render(c: &mut Criterion) {
    let _ = c;
    eprintln!("[cuda] feature not enabled — rerun with: cargo bench --features cuda");
}
#[cfg(not(feature = "cuda"))]
fn bench_cuda_pipeline(c: &mut Criterion) { let _ = c; }

// ═══════════════════════════════════════════════════════════════════════════════
// Perturbation theory benchmarks
// Compares render() (scalar) vs render_perturbation() at five zoom levels.
// At low zoom almost all pixels fall back to scalar (ε grows large immediately),
// so speedup ≈ 1×.  At high zoom (1e9+) most pixels stay within the linear
// approximation — speedup grows proportionally with max_iter.
// ═══════════════════════════════════════════════════════════════════════════════

// Zoom levels and center matching the bench_runner --perturb-sweep defaults.
const PERTURB_ZOOMS: &[(f64, &str)] = &[
    (1.0,  "zoom_1e0"),
    (1e3,  "zoom_1e3"),
    (1e6,  "zoom_1e6"),
    (1e9,  "zoom_1e9"),
    (1e12, "zoom_1e12"),
];
const PERTURB_CENTER: [f64; 2] = [-0.75, 0.1];

fn bench_perturbation_render(c: &mut Criterion) {
    let mut group = c.benchmark_group("perturbation/Mandelbrot_1080p");
    group.throughput(Throughput::Elements(1920 * 1080));
    group.sample_size(10);

    for &(zoom, label) in PERTURB_ZOOMS {
        let vp = Viewport { center: PERTURB_CENTER, zoom, width: 1920, height: 1080 };

        group.bench_with_input(
            BenchmarkId::new("scalar", label),
            &vp,
            |b, vp| b.iter(|| render(black_box(vp), FractalType::Mandelbrot, JULIA_C, MAX_ITER)),
        );
        group.bench_with_input(
            BenchmarkId::new("perturb", label),
            &vp,
            |b, vp| b.iter(|| render_perturbation(black_box(vp), FractalType::Mandelbrot, JULIA_C, MAX_ITER)),
        );
        group.bench_with_input(
            BenchmarkId::new("perturb_sa", label),
            &vp,
            |b, vp| b.iter(|| render_perturbation_sa(black_box(vp), FractalType::Mandelbrot, JULIA_C, MAX_ITER)),
        );
    }
    group.finish();
}

// Cost of computing the reference orbit alone — O(max_iter), paid once per frame.
// Compares f64 vs f128 orbit cost so the thesis can quote the overhead ratio.
fn bench_perturbation_reference_orbit(c: &mut Criterion) {
    let mut group = c.benchmark_group("perturbation/reference_orbit");

    for &(_, label) in PERTURB_ZOOMS {
        group.bench_function(
            BenchmarkId::new("f64", label),
            |b| b.iter(|| compute_reference_orbit(
                black_box(PERTURB_CENTER[0]),
                black_box(PERTURB_CENTER[1]),
                black_box(MAX_ITER),
            )),
        );
        group.bench_function(
            BenchmarkId::new("f128", label),
            |b| b.iter(|| compute_reference_orbit_f128(
                black_box(PERTURB_CENTER[0]),
                black_box(PERTURB_CENTER[1]),
                black_box(MAX_ITER),
            )),
        );
    }
    group.finish();
}

// Cost of computing SA coefficients on top of the reference orbit, and the
// resulting skip count at each zoom level (printed to stderr for inspection).
fn bench_series_approx(c: &mut Criterion) {
    let mut group = c.benchmark_group("perturbation/series_approx");

    for &(zoom, label) in PERTURB_ZOOMS {
        let aspect       = 1920.0f64 / 1080.0;
        let half         = 2.0 / zoom;
        let delta_max_sq = (half * aspect) * (half * aspect) + half * half;
        let orbit = compute_reference_orbit(PERTURB_CENTER[0], PERTURB_CENTER[1], MAX_ITER);

        // Print skip count so the thesis can quote concrete numbers.
        let sa = compute_series_approx(&orbit, delta_max_sq);
        eprintln!("[SA skip] zoom={zoom:.0e}  skip={}/{}  ({:.1}%)",
            sa.skip, MAX_ITER,
            100.0 * sa.skip as f64 / MAX_ITER as f64);

        group.bench_function(label, |b| {
            b.iter(|| compute_series_approx(black_box(&orbit), black_box(delta_max_sq)))
        });
    }
    group.finish();
}

// ── viewport → PerturbUniforms ────────────────────────────────────────────────

fn perturb_uniforms(vp: &Viewport, orbit_len: usize) -> PerturbUniforms {
    let aspect = vp.width as f64 / vp.height as f64;
    let half   = 2.0 / vp.zoom;
    PerturbUniforms {
        re_start:  (vp.center[0] + (0.5 / vp.width  as f64 - 0.5) * half * aspect * 2.0) as f32,
        im_start:  (vp.center[1] + (0.5 / vp.height as f64 - 0.5) * half * 2.0) as f32,
        re_step:   (half * aspect * 2.0 / vp.width  as f64) as f32,
        im_step:   (half * 2.0          / vp.height as f64) as f32,
        ref_re:    vp.center[0] as f32,
        ref_im:    vp.center[1] as f32,
        orbit_len: orbit_len as u32,
        max_iter:  MAX_ITER,
        width:     vp.width,
        height:    vp.height,
        _pad:      [0; 2],
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// wgpu perturbation benchmarks
// Compares gpu scalar vs gpu perturbation at the same zoom sweep used for CPU.
// The orbit is computed on CPU, cast f64→f32, then uploaded each iteration.
// ═══════════════════════════════════════════════════════════════════════════════

fn bench_wgpu_perturbation(c: &mut Criterion) {
    let Some(g) = gpu() else {
        eprintln!("[wgpu] no GPU — skipping wgpu/perturbation");
        return;
    };

    let mut group = c.benchmark_group("wgpu/perturbation/Mandelbrot_1080p");
    group.throughput(Throughput::Elements(1920 * 1080));
    group.sample_size(10);

    for &(zoom, label) in PERTURB_ZOOMS {
        let vp      = Viewport { center: PERTURB_CENTER, zoom, width: 1920, height: 1080 };
        let compute = FractalCompute::new(&g.device, 1920, 1080);
        let uni     = uniforms(&vp, FractalType::Mandelbrot);

        // Pre-compute orbit outside the timed loop.
        let orbit      = compute_reference_orbit(PERTURB_CENTER[0], PERTURB_CENTER[1], MAX_ITER);
        let orbit_re_f: Vec<f32> = orbit.zr[..=orbit.len].iter().map(|&v| v as f32).collect();
        let orbit_im_f: Vec<f32> = orbit.zi[..=orbit.len].iter().map(|&v| v as f32).collect();
        let perturb_uni = perturb_uniforms(&vp, orbit.len);

        group.bench_with_input(
            BenchmarkId::new("scalar", label), &uni,
            |b, &uni| b.iter(|| compute.render(&g.device, &g.queue, black_box(uni))),
        );
        group.bench_function(
            BenchmarkId::new("perturb", label),
            |b| b.iter(|| compute.render_perturbation(
                &g.device, &g.queue,
                black_box(perturb_uni),
                black_box(&orbit_re_f),
                black_box(&orbit_im_f),
            )),
        );
    }
    group.finish();
}

// ═══════════════════════════════════════════════════════════════════════════════
// CUDA perturbation benchmarks
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "cuda")]
fn bench_cuda_perturbation(c: &mut Criterion) {
    use novafractal::gpu::cuda::CudaFractal;

    let mut group = c.benchmark_group("cuda/perturbation/Mandelbrot_1080p");
    group.throughput(Throughput::Elements(1920 * 1080));
    group.sample_size(10);

    for &(zoom, label) in PERTURB_ZOOMS {
        let vp      = Viewport { center: PERTURB_CENTER, zoom, width: 1920, height: 1080 };
        let aspect  = 1920.0f64 / 1080.0;
        let half    = 2.0 / zoom;
        let re_start = vp.center[0] + (0.5 / 1920.0 - 0.5) * half * aspect * 2.0;
        let im_start = vp.center[1] + (0.5 / 1080.0 - 0.5) * half * 2.0;
        let re_step  = half * aspect * 2.0 / 1920.0;
        let im_step  = half * 2.0 / 1080.0;

        let orbit = compute_reference_orbit(PERTURB_CENTER[0], PERTURB_CENTER[1], MAX_ITER);

        let mut cuda = CudaFractal::from_device(cuda_dev(), 1920, 1080);

        group.bench_function(
            BenchmarkId::new("scalar", label),
            |b| b.iter(|| cuda.render(
                black_box(re_start), black_box(im_start),
                black_box(re_step),  black_box(im_step),
                JULIA_C[0], JULIA_C[1], MAX_ITER, 0, // 0 = Mandelbrot
            )),
        );
        group.bench_function(
            BenchmarkId::new("perturb", label),
            |b| b.iter(|| cuda.render_perturbation(
                black_box(&orbit),
                black_box(re_start), black_box(im_start),
                black_box(re_step),  black_box(im_step),
                MAX_ITER,
            )),
        );
    }
    group.finish();
}

#[cfg(not(feature = "cuda"))]
fn bench_cuda_perturbation(c: &mut Criterion) { let _ = c; }

// ── startup info ──────────────────────────────────────────────────────────────

fn print_header() {
    eprintln!("\n── Analytical FLOP budget ──────────────────────────────────────");
    eprintln!("{:<12}  {:>18}  {:>20}", "Kernel", "FLOPs/iter", "FLOPs @ 1000 iters");
    for &ft in FractalType::ALL {
        let fpi = flops_per_iter(ft);
        eprintln!("{:<12}  {:>18}  {:>20}", ft.name(), fpi, fpi * 1000);
    }
    eprintln!("(Smooth-coloring ln() calls add ~6 FLOPs per escaped pixel.)");
    eprintln!("────────────────────────────────────────────────────────────────\n");

    eprintln!("Backends compiled in:");
    eprintln!("  cpu   always");
    eprintln!("  wgpu  always (GPU required at runtime)");
    #[cfg(feature = "cuda")]
    eprintln!("  cuda  YES (--features cuda)");
    #[cfg(not(feature = "cuda"))]
    eprintln!("  cuda  NO  (rerun with --features cuda to enable)");
    eprintln!();
}

// ── registration ──────────────────────────────────────────────────────────────

criterion_group! {
    name = benches;
    config = {
        let c = Criterion::default()
            .warm_up_time(std::time::Duration::from_secs(2))
            .measurement_time(std::time::Duration::from_secs(5));
        print_header();
        c
    };
    targets =
        // CPU
        bench_pixel_kernels,
        bench_cpu_render,
        bench_tiled_render,
        bench_colorize,
        bench_cpu_pipeline,
        bench_thread_scaling,
        // SIMD  (no-op stub when feature is off)
        bench_simd_render,
        bench_simd_ilp_render,
        // Perturbation theory (scalar vs perturb vs perturb+SA × zoom sweep)
        bench_perturbation_render,
        bench_perturbation_reference_orbit,
        bench_series_approx,
        // wgpu GPU
        bench_wgpu_render,
        bench_wgpu_pipeline,
        bench_wgpu_perturbation,
        // Hybrid CPU + wgpu
        bench_hybrid_cpu_wgpu,
        bench_heterogeneous_wgpu,
        // CUDA GPU  (no-op stubs when feature is off)
        bench_cuda_render,
        bench_cuda_pipeline,
        bench_cuda_perturbation,
        // Heterogeneous CPU+CUDA scheduler
        bench_heterogeneous,
        bench_heterogeneous_deep,
        // Algorithmic work-skipping and reuse
        bench_work_skipping,
        bench_perturbation_strategies,
        bench_incremental_pan
}

criterion_main!(benches);
