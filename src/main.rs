use novafractal::gui::app::FractalApp;

fn main() -> eframe::Result<()> {
    // A minimal headless path lets timing be captured without a display
    // (e.g. in CI or over SSH) rather than requiring the interactive
    // window; the real benchmark suite (bench_runner, criterion) is more
    // thorough, this is a quick sanity check only.
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

/// Renders a fixed 1920x1080 Mandelbrot frame 5 times on the CPU scalar
/// backend and reports median/min/max timing and throughput. The first
/// (untimed) call warms up the rayon thread pool and page-faults the
/// output buffer so it doesn't skew the first timed run.
fn headless_bench() {
    use novafractal::fractal::{render, FractalType};
    use novafractal::gui::viewport::Viewport;

    const W: u32 = 1920;
    const H: u32 = 1080;
    const MAX_ITER: u32 = 512;
    const RUNS: usize = 5;

    let vp = Viewport { center: [-0.5, 0.0], zoom: 1.0, width: W, height: H };
    let _ = render(&vp, FractalType::Mandelbrot, [0.0, 0.0], MAX_ITER); 

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
