use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use novafractal::fractal::{pixel, render, flops_per_iter, FractalType};
use novafractal::gpu::fractal_compute::FractalCompute;
use novafractal::gpu::unifroms::Uniforms;
use novafractal::gui::color::{colorize, ColorScheme};
use novafractal::gui::viewport::Viewport;
use rayon::ThreadPoolBuilder;
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

fn gpu() -> Option<&'static GpuState> {
    GPU.get_or_init(|| {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }).await?;
            let info = format!("{} ({:?})", adapter.get_info().name, adapter.get_info().backend);
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

            let mut cuda = CudaFractal::new(w, h);
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
        let mut cuda = CudaFractal::new(1920, 1080);
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

// CPU top half + CUDA bottom half, concurrent
#[cfg(feature = "cuda")]
fn bench_hybrid_cpu_cuda(c: &mut Criterion) {
    use novafractal::gpu::cuda::CudaFractal;

    let full_vp  = vp(1920, 1080);
    let h_top    = full_vp.height / 2;
    let h_bottom = full_vp.height - h_top;
    let aspect   = full_vp.width as f64 / full_vp.height as f64;
    let half     = 2.0 / full_vp.zoom;
    let im_step  = half * 2.0 / full_vp.height as f64;
    let im_start = full_vp.center[1] + (0.5 / full_vp.height as f64 - 0.5) * half * 2.0;
    let re_start = full_vp.center[0] + (0.5 / full_vp.width as f64 - 0.5) * half * aspect * 2.0;
    let re_step  = half * aspect * 2.0 / full_vp.width as f64;
    let im_start_bottom = im_start + h_top as f64 * im_step;

    let vp_top = Viewport { width: full_vp.width, height: h_top, ..full_vp.clone() };

    let mut group = c.benchmark_group("hybrid/cpu+cuda");
    group.throughput(Throughput::Elements(1920 * 1080));
    group.sample_size(10);

    for &fractal in FractalType::ALL {
        let mut cuda = CudaFractal::new(full_vp.width, h_bottom);

        group.bench_function(fractal.name(), |b| {
            b.iter(|| {
                std::thread::scope(|s| {
                    let top = s.spawn(|| render(black_box(&vp_top), fractal, JULIA_C, MAX_ITER));
                    let bottom = cuda.render(
                        black_box(re_start), black_box(im_start_bottom),
                        black_box(re_step), black_box(im_step),
                        JULIA_C[0], JULIA_C[1], MAX_ITER, fractal_u32(fractal),
                    );
                    let _ = top.join().unwrap();
                    bottom
                })
            })
        });
    }
    group.finish();
}

// Stubs when cuda feature is off — criterion_group! needs a consistent list.
#[cfg(not(feature = "cuda"))]
fn bench_cuda_render(c: &mut Criterion) {
    let _ = c;
    eprintln!("[cuda] feature not enabled — rerun with: cargo bench --features cuda");
}
#[cfg(not(feature = "cuda"))]
fn bench_cuda_pipeline(c: &mut Criterion) { let _ = c; }
#[cfg(not(feature = "cuda"))]
fn bench_hybrid_cpu_cuda(c: &mut Criterion) { let _ = c; }

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
        bench_colorize,
        bench_cpu_pipeline,
        bench_thread_scaling,
        // wgpu GPU
        bench_wgpu_render,
        bench_wgpu_pipeline,
        // Hybrid CPU + wgpu
        bench_hybrid_cpu_wgpu,
        // CUDA GPU  (no-op stubs when feature is off)
        bench_cuda_render,
        bench_cuda_pipeline,
        // Hybrid CPU + CUDA
        bench_hybrid_cpu_cuda
}

criterion_main!(benches);
