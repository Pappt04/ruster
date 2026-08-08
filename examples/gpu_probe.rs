//! Diagnostic probe: decomposes the CUDA and wgpu full-frame render paths into
//! kernel time vs host-transfer time, and the heterogeneous scheduler into its
//! serial phases. Answers "why is CUDA slower than wgpu" and "why is the
//! scheduler slower than GPU-only" with measurements instead of reasoning.
//!
//! Run: cargo run --release --features cuda --example gpu_probe

use novafractal::fractal::{pixel_grid, FractalType};
use novafractal::gpu::fractal_compute::FractalCompute;
use novafractal::gpu::unifroms::Uniforms;
use novafractal::gui::viewport::Viewport;
use std::time::Instant;

const MAX_ITER: u32 = 1000;
const JULIA_C: [f64; 2] = [-0.4, 0.6];
const W: u32 = 1920;
const H: u32 = 1080;
const REPS: usize = 30;

fn bench<F: FnMut()>(name: &str, mut f: F) -> f64 {
    for _ in 0..5 { f(); }
    let t0 = Instant::now();
    for _ in 0..REPS { f(); }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
    println!("  {name:<52} {ms:8.3} ms");
    ms
}

fn uniforms(vp: &Viewport, fractal: u32) -> Uniforms {
    let pg = pixel_grid(vp);
    Uniforms {
        re_start: pg.re_start as f32,
        im_start: pg.im_start as f32,
        re_step: pg.re_step as f32,
        im_step: pg.im_step as f32,
        julia_cr: JULIA_C[0] as f32,
        julia_ci: JULIA_C[1] as f32,
        max_iter: MAX_ITER,
        fractal,
        width: vp.width,
        height: vp.height,
        _pad: [0; 2],
    }
}

fn main() {
    let vp = Viewport { center: [-0.5, 0.0], zoom: 1.0, width: W, height: H };
    let pg = pixel_grid(&vp);

    println!("\n=== Frame: {W}x{H}, Mandelbrot, max_iter={MAX_ITER}, zoom=1 ===");
    println!("    pixels = {}", W * H);

    // ── Launch geometry arithmetic (no measurement needed, just the facts) ────
    let dim = W.max(H).next_power_of_two();
    let morton_threads = dim as u64 * dim as u64;
    let grid2d_threads = (((W + 15) / 16) * ((H + 15) / 16) * 256) as u64;
    println!("\n--- launch geometry ---");
    println!("  morton_cfg pads to {dim}x{dim} square -> {morton_threads} threads");
    println!("  plain 2D grid ({}x{} blocks)          -> {grid2d_threads} threads",
             (W + 15) / 16, (H + 15) / 16);
    println!("  morton oversubscription vs pixels: {:.2}x",
             morton_threads as f64 / (W as f64 * H as f64));
    println!("  plain2d oversubscription vs pixels: {:.2}x",
             grid2d_threads as f64 / (W as f64 * H as f64));

    // ── wgpu ─────────────────────────────────────────────────────────────────
    println!("\n--- wgpu ---");
    let (device, queue) = pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("no adapter");
        println!("  adapter: {}", adapter.get_info().name);
        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .unwrap()
    });

    let compute = FractalCompute::new(&device, W, H);
    let uni = uniforms(&vp, 0);

    let wgpu_total = bench("render() [dispatch + copy + map + to_vec]", || {
        std::hint::black_box(compute.render(&device, &queue, uni));
    });
    // Kernel measured DIRECTLY, not by subtracting a standalone readback.
    // `render()` encodes the compute pass and the buffer copy into one encoder
    // and submits once; a standalone `readback()` is a second encoder + a second
    // submit + its own map/unmap round trip, so it costs materially more than
    // the copy does *inside* render(). Subtracting it therefore over-counts the
    // transfer — on the RTX 3050 it over-counts it past render()'s own total and
    // yields a negative kernel time. `dispatch_tiled` over one full-frame tile
    // runs the same per-pixel math with no copy at all, so dispatch + poll(Wait)
    // is the honest kernel measurement.
    let full_tile_wgpu = [[0u32, 0, W, H]];
    let wgpu_kernel = bench("dispatch_tiled(full frame) + poll [kernel only]", || {
        compute.dispatch_tiled(&device, &queue, &full_tile_wgpu, uni);
        device.poll(wgpu::MaintainBase::Wait);
    });
    println!("  => wgpu kernel = {wgpu_kernel:.3} ms, transfer+map = {:.3} ms",
             wgpu_total - wgpu_kernel);

    // ── CUDA ─────────────────────────────────────────────────────────────────
    #[cfg(feature = "cuda")]
    {
        use novafractal::gpu::cuda::CudaFractal;
        println!("\n--- cuda ---");
        let mut cuda = CudaFractal::new(W, H);
        let full_tile = [[0u32, 0, W, H]];

        let cuda_total = bench("render() [morton f32 kernel + dtoh_sync_copy]", || {
            std::hint::black_box(cuda.render(
                pg.re_start, pg.im_start, pg.re_step, pg.im_step,
                JULIA_C[0], JULIA_C[1], MAX_ITER, 0,
            ));
        });
        // Same mandelbrot_f32 math, but launched as a plain 2D grid over one
        // full-frame tile instead of the padded-square Morton grid.
        let cuda_tiled = bench("dispatch_tiled_f32(full frame) + readback()", || {
            cuda.dispatch_tiled_f32(
                &full_tile, pg.re_start, pg.im_start, pg.re_step, pg.im_step,
                JULIA_C[0], JULIA_C[1], MAX_ITER, 0,
            );
            std::hint::black_box(cuda.readback());
        });
        // GPU idle -> pure dtoh transfer.
        let cuda_readback = bench("readback() only [dtoh_sync_copy]", || {
            std::hint::black_box(cuda.readback());
        });
        println!("  => cuda morton kernel  ~= {:.3} ms", cuda_total - cuda_readback);
        println!("  => cuda plain2d kernel ~= {:.3} ms", cuda_tiled - cuda_readback);
        println!("  => cuda transfer       ~= {:.3} ms", cuda_readback);
        println!("  => morton vs plain2d launch penalty: {:.3} ms ({:+.1}%)",
                 cuda_total - cuda_tiled,
                 (cuda_total - cuda_tiled) / cuda_tiled * 100.0);

        // ── heterogeneous scheduler phase breakdown ──────────────────────────
        use novafractal::scheduler::{
            classifier::partition_frame, controller::ThresholdController,
            render_heterogeneous, SchedulerConfig,
        };
        println!("\n--- heterogeneous scheduler (center -0.75,0.1) ---");
        let cfg = SchedulerConfig::default();
        for &(zoom, label) in &[(1.0f64, "zoom_1e0"), (1e2, "zoom_1e2"), (1e4, "zoom_1e4")] {
            let fvp = Viewport { center: [-0.75, 0.1], zoom, width: W, height: H };
            let fpg = pixel_grid(&fvp);
            println!("  [{label}]");

            let mut ctl = ThresholdController::new(0.02);
            // Warm the controller to the same steady state the real bench sees.
            for _ in 0..5 {
                std::hint::black_box(render_heterogeneous(
                    &fvp, FractalType::Mandelbrot, JULIA_C, MAX_ITER, &mut cuda, &mut ctl, &cfg,
                ));
            }
            let thr = ctl.threshold;

            let part = bench("    partition_frame() alone", || {
                std::hint::black_box(partition_frame(
                    &fpg, FractalType::Mandelbrot, JULIA_C, MAX_ITER, W, H,
                    cfg.max_tile_size, cfg.min_tile_size, thr,
                ));
            });
            let (g, c) = partition_frame(
                &fpg, FractalType::Mandelbrot, JULIA_C, MAX_ITER, W, H,
                cfg.max_tile_size, cfg.min_tile_size, thr,
            );
            let gpu_px: u64 = g.iter().map(|t| t[2] as u64 * t[3] as u64).sum();
            let cpu_px: u64 = c.iter().map(|t| t[2] as u64 * t[3] as u64).sum();
            println!("    tiles: {} gpu ({:.1}% px) + {} cpu ({:.1}% px), threshold={thr:.4}",
                     g.len(), gpu_px as f64 / (W * H) as f64 * 100.0,
                     c.len(), cpu_px as f64 / (W * H) as f64 * 100.0);

            let mut ctl2 = ThresholdController::new(thr);
            let mut last = None;
            let total = bench("    render_heterogeneous() full", || {
                last = Some({
                    let r = render_heterogeneous(
                        &fvp, FractalType::Mandelbrot, JULIA_C, MAX_ITER,
                        &mut cuda, &mut ctl2, &cfg,
                    );
                    (r.gpu_ms, r.cpu_ms, r.cpu_tile_frac)
                });
            });
            let plain = bench("    plain cuda render() same viewport", || {
                std::hint::black_box(cuda.render(
                    fpg.re_start, fpg.im_start, fpg.re_step, fpg.im_step,
                    JULIA_C[0], JULIA_C[1], MAX_ITER, 0,
                ));
            });
            if let Some((gms, cms, frac)) = last {
                println!("    reported gpu_ms={gms:.3} cpu_ms={cms:.3} cpu_tile_frac={:.1}%", frac * 100.0);
                println!("    unaccounted (wall - partition - max(gpu,cpu)) = {:.3} ms",
                         total - part - gms.max(cms) as f64);
            }
            println!("    scheduler vs plain cuda: {:+.1}%", (total - plain) / plain * 100.0);
        }
    }

    #[cfg(not(feature = "cuda"))]
    println!("\n(cuda feature off — rerun with --features cuda for the CUDA half)");
}
