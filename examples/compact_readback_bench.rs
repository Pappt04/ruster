//! Benchmarks the D2H copy alone: `CudaFractal::dispatch_tiled_compact` +
//! `readback_compact` (compact buffer, sized to dispatched pixels only)
//! against `dispatch_tiled` + `readback_into` (full-frame).
//!
//! This is the follow-up to `windowed_readback_bench.rs`'s finding (see git
//! history): a row-banded readback window couldn't shrink the transfer
//! because `classifier::partition_frame`'s GPU-tagged cells at the frame's
//! own top/bottom edges pin the band to full frame height regardless of how
//! little of the frame is actually GPU work. The compact buffer sidesteps
//! that — tiles are written back-to-back in dispatch order instead of at
//! their frame position, so the readback *itself* is sized to the dispatched
//! pixel count. And indeed this file shows that D2H copy getting smaller.
//!
//! **But this is not the whole picture, and reading only this file will give
//! the wrong conclusion.** A compact readback still has to be placed back
//! into a full-resolution frame buffer — unlike a direct `readback_into`,
//! which DMAs straight into the caller's buffer. That placement is a
//! mandatory host-side scatter this file does not measure. Once it's
//! included (`examples/readback_scatter_ab.rs`), the smaller copy shown here
//! is *not* a net win: the scatter costs more than the copy saves at every
//! GPU-tile fraction tested, so `crate::scheduler` does not use the compact
//! path. Read that file for the real end-to-end number; this one exists only
//! to show which piece of the cost the compact buffer does and doesn't
//! address.
//!
//! Run: cargo run --release --features cuda --example compact_readback_bench

#[cfg(not(feature = "cuda"))]
fn main() { println!("needs --features cuda"); }

#[cfg(feature = "cuda")]
fn main() {
    use novafractal::fractal::{pixel_grid, FractalType};
    use novafractal::gpu::cuda::CudaFractal;
    use novafractal::gui::viewport::Viewport;
    use novafractal::scheduler::{classifier::partition_frame, SchedulerConfig};
    use std::time::Instant;

    const W: u32 = 1920;
    const H: u32 = 1080;
    const MAX_ITER: u32 = 512;
    const JULIA_C: [f64; 2] = [-0.4, 0.6];
    const REPS: usize = 40;

    fn bench<F: FnMut()>(name: &str, mut f: F) -> f64 {
        for _ in 0..8 { f(); }
        let t0 = Instant::now();
        for _ in 0..REPS { f(); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
        println!("  {name:<40} {ms:8.3} ms");
        ms
    }

    let mut cuda = CudaFractal::new(W, H);
    let cfg = SchedulerConfig::default();

    let centers = [
        (1.0f64,  "zoom_1e0", [-0.5, 0.0]),
        (1e2,     "zoom_1e2", [-0.75, 0.1]),
        (1e4,     "zoom_1e4", [-0.75, 0.1]),
        (1e6,     "zoom_1e6 (f64 kernel boundary)", [-0.75, 0.1]),
        (1e8,     "zoom_1e8", [-1.401155, 0.0]),
    ];

    println!("=== full-frame vs compact CUDA readback, same dispatched tile set ({W}x{H}, max_iter={MAX_ITER}) ===");

    for &(zoom, label, center) in &centers {
        let vp = Viewport { center, zoom, width: W, height: H };
        let pg = pixel_grid(&vp);

        let (gpu_tiles, cpu_tiles) = partition_frame(
            &pg, FractalType::Mandelbrot, JULIA_C, MAX_ITER, W, H,
            cfg.max_tile_size, cfg.min_tile_size, 0.02,
        );
        let gpu_pixels: u64 = gpu_tiles.iter().map(|&[_, _, tw, th]| (tw as u64) * (th as u64)).sum();
        let total_pixels = (W as u64) * (H as u64);
        let gpu_frac = gpu_pixels as f64 / total_pixels as f64 * 100.0;

        println!("\n=== {label} (center {center:?}) ===");
        println!("  [{} gpu tiles / {} cpu tiles, gpu_pixels={:.1}% of frame]",
                 gpu_tiles.len(), cpu_tiles.len(), gpu_frac);

        let mut tagged: Vec<[u32; 5]> = Vec::with_capacity(gpu_tiles.len());
        let mut offset = 0u32;
        for &[x0, y0, tw, th] in &gpu_tiles {
            tagged.push([x0, y0, tw, th, offset]);
            offset += tw * th;
        }

        // f32 kernels, matching `SchedulerConfig::gpu_tiles_f32`'s default
        // (`true`) — the config real usage (including the live app) runs
        // with. The f64 tiled kernel is ~6-7x slower (see that field's doc
        // comment), which would swamp the readback difference this probe
        // exists to isolate.
        let full_ms = bench("dispatch_tiled_f32 + readback_into (old)", || {
            cuda.dispatch_tiled_f32(
                &gpu_tiles, pg.re_start, pg.im_start, pg.re_step, pg.im_step,
                JULIA_C[0], JULIA_C[1], MAX_ITER, FractalType::Mandelbrot.as_u32(),
            );
            let mut dst = vec![0.0f32; (W * H) as usize];
            cuda.readback_into(&mut dst);
            std::hint::black_box(&dst);
        });

        let compact_ms = bench("dispatch_tiled_f32_compact + readback_compact (new)", || {
            cuda.dispatch_tiled_f32_compact(
                &tagged, pg.re_start, pg.im_start, pg.re_step, pg.im_step,
                JULIA_C[0], JULIA_C[1], MAX_ITER, FractalType::Mandelbrot.as_u32(),
            );
            std::hint::black_box(cuda.readback_compact(offset));
        });

        println!("  => {:+.3} ms ({:+.1}%) vs full-frame readback",
                 compact_ms - full_ms, (compact_ms - full_ms) / full_ms * 100.0);
    }
}
