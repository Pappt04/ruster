//! The heterogeneous scheduler's benches stop at zoom 1e4, which is entirely
//! inside the regime where the GPU has an f32 fast path and is 4x the CPU — a
//! regime no scheduler can improve on. Above `F32_PRECISION_THRESHOLD` (1e6)
//! the GPU drops to its f64 kernel, which Ampere GeForce runs at 1/64 rate, the
//! CPU becomes competitive or better, and a split genuinely pays.
//!
//! This probe sweeps across that boundary and compares the scheduler against
//! both solo backends and against the ideal harmonic split
//! `1/(1/T_gpu + 1/T_cpu)` — the best any perfectly-balanced, zero-overhead
//! scheduler could achieve.
//!
//! Run: cargo run --release --features cuda --example sched_deepzoom

#[cfg(not(feature = "cuda"))]
fn main() { println!("needs --features cuda"); }

#[cfg(feature = "cuda")]
fn main() {
    use novafractal::fractal::{pixel_grid, render, FractalType};
    use novafractal::gpu::cuda::CudaFractal;
    use novafractal::gui::viewport::Viewport;
    use novafractal::scheduler::{controller::ThresholdController, render_heterogeneous, SchedulerConfig};
    use std::time::Instant;

    const W: u32 = 1920;
    const H: u32 = 1080;
    const MI: u32 = 1000;
    const R: usize = 8;

    let mut cuda = CudaFractal::new(W, H);
    let cfg = SchedulerConfig::default();

    println!("\n=== scheduler across the f32/f64 precision boundary (1920x1080) ===");
    println!("  {:<7} {:>10} {:>10} {:>11} {:>10} {:>11} {:>12}",
             "zoom", "CUDA", "CPU", "best solo", "hybrid", "ideal split", "verdict");
    for (label, zoom) in [("1e4", 1e4f64), ("1e5", 1e5), ("1e6", 1e6), ("1e7", 1e7), ("1e9", 1e9)] {
        let vp = Viewport { center: [-0.745, 0.113], zoom, width: W, height: H };
        let pg = pixel_grid(&vp);
        let mut t = |f: &mut dyn FnMut()| {
            f();
            let s = Instant::now();
            for _ in 0..R { f(); }
            s.elapsed().as_secs_f64() * 1000.0 / R as f64
        };
        let g = t(&mut || { std::hint::black_box(cuda.render(
            pg.re_start, pg.im_start, pg.re_step, pg.im_step, -0.4, 0.6, MI, 0)); });
        let c = t(&mut || { std::hint::black_box(
            render(&vp, FractalType::Mandelbrot, [-0.4, 0.6], MI)); });

        // Let the adaptive threshold converge before timing.
        let mut ctl = ThresholdController::new(0.02);
        for _ in 0..25 {
            render_heterogeneous(&vp, FractalType::Mandelbrot, [-0.4, 0.6], MI, &mut cuda, &mut ctl, &cfg);
        }
        let h = t(&mut || {
            let r = render_heterogeneous(&vp, FractalType::Mandelbrot, [-0.4, 0.6], MI, &mut cuda, &mut ctl, &cfg);
            std::hint::black_box(r.buf);
        });

        let best = g.min(c);
        let ideal = 1.0 / (1.0 / g + 1.0 / c);
        println!("  {label:<7} {g:>9.2}ms {c:>9.2}ms {best:>10.2}ms {h:>9.2}ms {ideal:>10.2}ms {:>12}",
                 if h < best { format!("{:.2}x WIN", best / h) } else { format!("{:.2}x lose", best / h) });
    }
    println!("\n  'ideal split' is the ceiling for a perfectly balanced, zero-overhead");
    println!("  scheduler. Where hybrid is far above it, the load balancer is at fault;");
    println!("  where ideal is close to 'best solo', no scheduler can help at all.");
}
