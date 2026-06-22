use novafractal::gui::app::FractalApp;

fn main() -> eframe::Result<()> {
    if std::env::args().any(|a| a == "--headless" || a == "--bench") {
        headless_bench();
        return Ok(());
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([900.0, 700.0]),
        ..Default::default()
    };
    eframe::run_native(
        "NovaFractal",
        options,
        Box::new(|_cc| Ok(Box::new(FractalApp::default()))),
    )
}

// Single-binary headless mode for `perf stat` / `perf record`.
// Renders 1920×1080 Mandelbrot 6 times (1 warm-up, 5 timed) and prints timing.
fn headless_bench() {
    use novafractal::fractal::{render, FractalType};
    use novafractal::gui::viewport::Viewport;

    const W: u32 = 1920;
    const H: u32 = 1080;
    const MAX_ITER: u32 = 512;
    const RUNS: usize = 5;

    let vp = Viewport { center: [-0.5, 0.0], zoom: 1.0, width: W, height: H };
    let _ = render(&vp, FractalType::Mandelbrot, [0.0, 0.0], MAX_ITER); // warm-up

    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = std::time::Instant::now();
        let _ = render(&vp, FractalType::Mandelbrot, [0.0, 0.0], MAX_ITER);
        times.push(t.elapsed());
    }
    times.sort();

    let med_s  = times[RUNS / 2].as_secs_f64();
    let min_s  = times[0].as_secs_f64();
    let max_s  = times[RUNS - 1].as_secs_f64();
    let mpix_s = (W as f64 * H as f64) / 1e6 / med_s;
    eprintln!("headless_bench  {W}×{H}  mandelbrot  max_iter={MAX_ITER}  runs={RUNS}");
    eprintln!("  median={:.1}ms  min={:.1}ms  max={:.1}ms  {:.2} Mpix/s",
              med_s * 1e3, min_s * 1e3, max_s * 1e3, mpix_s);
}
