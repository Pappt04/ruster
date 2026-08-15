//! Validates `CudaFractal::colorize_into` (GPU histogram-equalization +
//! palette LUT) against `gui::color::colorize` (the CPU implementation it
//! mirrors) on the same escape-time buffer, then times the two end-to-end
//! paths a caller actually has available: `cuda.render()` + CPU `colorize()`
//! (the old path every CUDA render used to take) vs `cuda.render_and_colorize()`
//! (skips the escape-value D2H copy and does histogram/CDF/LUT on-device).
//!
//! Run: cargo run --release --features cuda --example colorize_probe

#[cfg(not(feature = "cuda"))]
fn main() { println!("needs --features cuda"); }

#[cfg(feature = "cuda")]
fn main() {
    use novafractal::fractal::pixel_grid;
    use novafractal::gpu::cuda::CudaFractal;
    use novafractal::gui::color::{colorize, lut_bytes, ColorScheme};
    use novafractal::gui::viewport::Viewport;
    use std::time::Instant;

    const W: u32 = 1920;
    const H: u32 = 1080;
    const MAX_ITER: u32 = 512;
    const REPS: usize = 60;

    fn bench<F: FnMut()>(name: &str, mut f: F) -> f64 {
        for _ in 0..8 { f(); }
        let t0 = Instant::now();
        for _ in 0..REPS { f(); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
        println!("  {name:<52} {ms:8.3} ms");
        ms
    }

    let scheme = ColorScheme::Inferno;
    let scheme_id = scheme.palette_index() as u8;
    let lut = lut_bytes(scheme);

    let views = [
        (1.0f64,  "zoom_1e0", [-0.5, 0.0]),
        (1e4,     "zoom_1e4", [-0.75, 0.1]),
        (1e8,     "zoom_1e8", [-1.401155, 0.0]),
    ];

    for &(zoom, label, center) in &views {
        let vp = Viewport { center, zoom, width: W, height: H };
        let pg = pixel_grid(&vp);
        let mut cuda = CudaFractal::new(W, H);

        // --- correctness: GPU colorize_into vs CPU colorize on the same buffer ---
        let buf = cuda.render(pg.re_start, pg.im_start, pg.re_step, pg.im_step, -0.4, 0.6, MAX_ITER, 0);
        let cpu_pixels = colorize(&buf, MAX_ITER, scheme);

        let mut gpu_rgba = vec![0u8; (W * H * 4) as usize];
        cuda.colorize_into(MAX_ITER, scheme_id, lut, &mut gpu_rgba);

        let mut max_channel_diff = 0i32;
        let mut n_mismatched = 0usize;
        for (i, px) in cpu_pixels.iter().enumerate() {
            let [cr, cg, cb, ca] = px.to_array();
            let o = i * 4;
            let [gr, gg, gb, ga] = [gpu_rgba[o], gpu_rgba[o + 1], gpu_rgba[o + 2], gpu_rgba[o + 3]];
            let d = [(cr as i32 - gr as i32).abs(), (cg as i32 - gg as i32).abs(),
                     (cb as i32 - gb as i32).abs(), (ca as i32 - ga as i32).abs()];
            let dm = *d.iter().max().unwrap();
            if dm > 0 { n_mismatched += 1; }
            max_channel_diff = max_channel_diff.max(dm);
        }

        println!("\n=== {label} (center {center:?}) ===");
        println!("  [correctness] max_channel_diff={max_channel_diff} (0-255 scale), {n_mismatched}/{} pixels differ at all",
                 cpu_pixels.len());

        // --- timing: old path (render + CPU colorize) vs new (render_and_colorize) ---
        let old_ms = bench("render() + CPU colorize() (old)", || {
            let buf = cuda.render(pg.re_start, pg.im_start, pg.re_step, pg.im_step, -0.4, 0.6, MAX_ITER, 0);
            let pixels = colorize(&buf, MAX_ITER, scheme);
            std::hint::black_box(&pixels);
        });

        let mut rgba = vec![0u8; (W * H * 4) as usize];
        let new_ms = bench("render_and_colorize() (new, GPU colorize)", || {
            cuda.render_and_colorize(
                pg.re_start, pg.im_start, pg.re_step, pg.im_step, -0.4, 0.6,
                MAX_ITER, 0, scheme_id, lut, &mut rgba,
            );
            std::hint::black_box(&rgba);
        });

        println!("  => {:+.3} ms ({:+.1}%) end-to-end vs render()+CPU colorize()",
                 new_ms - old_ms, (new_ms - old_ms) / old_ms * 100.0);
    }
}
