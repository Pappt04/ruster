//! Isolates the heterogeneous scheduler's *fixed* per-frame overhead from its
//! actual GPU/CPU work, by forcing degenerate partitions:
//!   * all-GPU (threshold huge, no steal reserve) -> everything the scheduler
//!     costs that plain `cuda.render()` does not
//!   * all-CPU (threshold ~0)                     -> CPU-side cost with no GPU
//! and compares against the plain single-backend paths.
//!
//! Run: cargo run --release --features cuda --example sched_probe

#[cfg(not(feature = "cuda"))]
fn main() { println!("needs --features cuda"); }

#[cfg(feature = "cuda")]
fn main() {
    use novafractal::fractal::{pixel_grid, render, FractalType};
    use novafractal::gpu::cuda::CudaFractal;
    use novafractal::gui::viewport::Viewport;
    use novafractal::scheduler::{
        classifier::partition_frame, controller::ThresholdController,
        render_heterogeneous, SchedulerConfig,
    };
    use std::time::Instant;

    const W: u32 = 1920;
    const H: u32 = 1080;
    const MAX_ITER: u32 = 1000;
    const JULIA_C: [f64; 2] = [-0.4, 0.6];
    const REPS: usize = 25;

    fn bench<F: FnMut()>(name: &str, mut f: F) -> f64 {
        for _ in 0..5 { f(); }
        let t0 = Instant::now();
        for _ in 0..REPS { f(); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
        println!("  {name:<50} {ms:8.3} ms");
        ms
    }

    let mut cuda = CudaFractal::new(W, H);

    for &(zoom, label) in &[(1.0f64, "zoom_1e0"), (1e2, "zoom_1e2"), (1e4, "zoom_1e4")] {
        let vp = Viewport { center: [-0.75, 0.1], zoom, width: W, height: H };
        let pg = pixel_grid(&vp);
        println!("\n=== {label} (center -0.75,0.1, {W}x{H}, max_iter={MAX_ITER}) ===");

        let plain_gpu = bench("plain cuda.render()", || {
            std::hint::black_box(cuda.render(
                pg.re_start, pg.im_start, pg.re_step, pg.im_step,
                JULIA_C[0], JULIA_C[1], MAX_ITER, 0,
            ));
        });
        let plain_cpu = bench("plain cpu render()", || {
            std::hint::black_box(render(&vp, FractalType::Mandelbrot, JULIA_C, MAX_ITER));
        });

        // --- all-GPU degenerate partition: measures scheduler fixed overhead ---
        let mut cfg_gpu = SchedulerConfig::default();
        cfg_gpu.steal_reserve_frac = 0.0;
        let mut ctl = ThresholdController::new(1e9);
        let (g, c) = partition_frame(
            &pg, FractalType::Mandelbrot, JULIA_C, MAX_ITER, W, H,
            cfg_gpu.max_tile_size, cfg_gpu.min_tile_size, 1e9,
        );
        println!("  [all-gpu partition: {} gpu tiles, {} cpu tiles]", g.len(), c.len());
        let all_gpu = bench("render_heterogeneous(), all tiles -> GPU", || {
            let mut c = ThresholdController::new(1e9);
            std::hint::black_box(render_heterogeneous(
                &vp, FractalType::Mandelbrot, JULIA_C, MAX_ITER, &mut cuda, &mut c, &cfg_gpu,
            ));
        });
        let _ = &mut ctl;
        println!("  => scheduler fixed overhead vs plain GPU: {:+.3} ms ({:+.1}%)",
                 all_gpu - plain_gpu, (all_gpu - plain_gpu) / plain_gpu * 100.0);

        // --- all-CPU degenerate partition ---
        let mut cfg_cpu = SchedulerConfig::default();
        cfg_cpu.steal_reserve_frac = 0.0;
        let (g2, c2) = partition_frame(
            &pg, FractalType::Mandelbrot, JULIA_C, MAX_ITER, W, H,
            cfg_cpu.max_tile_size, cfg_cpu.min_tile_size, 0.0,
        );
        println!("  [all-cpu partition: {} gpu tiles, {} cpu tiles]", g2.len(), c2.len());
        let all_cpu = bench("render_heterogeneous(), all tiles -> CPU", || {
            let mut c = ThresholdController::new(0.0);
            std::hint::black_box(render_heterogeneous(
                &vp, FractalType::Mandelbrot, JULIA_C, MAX_ITER, &mut cuda, &mut c, &cfg_cpu,
            ));
        });
        println!("  => tile-split + merge cost vs plain CPU render: {:+.3} ms ({:+.1}%)",
                 all_cpu - plain_cpu, (all_cpu - plain_cpu) / plain_cpu * 100.0);

        // --- real adaptive scheduler for reference ---
        let mut ctl3 = ThresholdController::new(0.02);
        for _ in 0..5 {
            std::hint::black_box(render_heterogeneous(
                &vp, FractalType::Mandelbrot, JULIA_C, MAX_ITER, &mut cuda, &mut ctl3,
                &SchedulerConfig::default(),
            ));
        }
        let adaptive = bench("render_heterogeneous(), adaptive (shipped cfg)", || {
            std::hint::black_box(render_heterogeneous(
                &vp, FractalType::Mandelbrot, JULIA_C, MAX_ITER, &mut cuda, &mut ctl3,
                &SchedulerConfig::default(),
            ));
        });
        println!("  => best solo backend = {:.3} ms; adaptive = {:.3} ms ({:+.1}%)",
                 plain_gpu.min(plain_cpu), adaptive,
                 (adaptive - plain_gpu.min(plain_cpu)) / plain_gpu.min(plain_cpu) * 100.0);
    }
}
