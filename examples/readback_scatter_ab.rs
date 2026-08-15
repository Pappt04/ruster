//! Isolated A/B of exactly the piece that changed in an earlier version of
//! the heterogeneous scheduler: how GPU-dispatched tiles land in the frame
//! buffer. Kept as the record of *why* `crate::scheduler` uses plain
//! `dispatch_tiled`/`readback_into` rather than a compact buffer — see
//! `CudaFractal::dispatch_tiled`'s doc comment.
//!
//! OLD (what the scheduler actually does): `dispatch_tiled_f32` (writes each
//! tile straight into its row-major frame position in the full-size device
//! buffer) + `readback_into` (one direct DMA into `out`, no host-side scatter
//! needed for GPU pixels).
//!
//! NEW (tried, rejected): `dispatch_tiled_f32_compact` (tiles land
//! contiguously in dispatch order in a compact device buffer) +
//! `readback_compact` (one DMA sized to only the dispatched pixels) + a
//! parallel per-row scatter into `out` (`examples/compact_readback_bench.rs`
//! shows that DMA alone getting smaller — the scatter is what this file adds
//! back in).
//!
//! Kernel dispatch and CPU tile compute are identical either way and excluded
//! from both timings — this isolates the readback+placement cost alone.
//! Correctness is checked only over pixels the current view's GPU tiles
//! cover: `out_old`/`out_new` are reused across zoom levels without
//! resetting (matching how the persistent `output`/`compact_output` device
//! buffers behave in real use), so pixels *outside* the current tile set
//! legitimately hold stale data from whichever zoom ran before them in
//! either buffer — comparing those would just be comparing two different
//! flavors of stale, not a real check.
//!
//! Run: cargo run --release --features cuda --example readback_scatter_ab

#[cfg(not(feature = "cuda"))]
fn main() { println!("needs --features cuda"); }

#[cfg(feature = "cuda")]
fn main() {
    use novafractal::fractal::{pixel_grid, FractalType};
    use novafractal::gpu::cuda::CudaFractal;
    use novafractal::gui::viewport::Viewport;
    use novafractal::scheduler::{classifier::partition_frame, SchedulerConfig};
    use rayon::prelude::*;
    use std::time::Instant;

    const W: u32 = 1920;
    const H: u32 = 1080;
    const MAX_ITER: u32 = 1000;
    const JULIA_C: [f64; 2] = [-0.4, 0.6];
    const REPS: usize = 80;

    fn bench<F: FnMut()>(name: &str, mut f: F) -> f64 {
        for _ in 0..15 { f(); }
        let t0 = Instant::now();
        for _ in 0..REPS { f(); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
        println!("  {name:<28} {ms:8.4} ms");
        ms
    }

    let mut cuda = CudaFractal::new(W, H);
    let cfg = SchedulerConfig::default();

    let centers = [
        (1.0f64, "zoom_1e0 (~97% gpu)", [-0.5, 0.0]),
        (1e2,    "zoom_1e2 (~85% gpu)", [-0.75, 0.1]),
        (1e4,    "zoom_1e4 (100% gpu)", [-0.75, 0.1]),
    ];

    for &(zoom, label, center) in &centers {
        let vp = Viewport { center, zoom, width: W, height: H };
        let pg = pixel_grid(&vp);
        let (gpu_tiles, _cpu_tiles) = partition_frame(
            &pg, FractalType::Mandelbrot, JULIA_C, MAX_ITER, W, H,
            cfg.max_tile_size, cfg.min_tile_size, 0.02,
        );
        let gpu_pixels: u64 = gpu_tiles.iter().map(|&[_, _, tw, th]| (tw as u64) * (th as u64)).sum();
        println!("\n=== {label}: {} gpu tiles, {:.1}% of frame ===",
                 gpu_tiles.len(), gpu_pixels as f64 / (W as u64 * H as u64) as f64 * 100.0);

        let mut tagged: Vec<[u32; 5]> = Vec::with_capacity(gpu_tiles.len());
        let mut offset = 0u32;
        for &[x0, y0, tw, th] in &gpu_tiles {
            tagged.push([x0, y0, tw, th, offset]);
            offset += tw * th;
        }

        let mut out_old = vec![0.0f32; (W * H) as usize];
        let old_ms = bench("OLD: dispatch + direct DMA", || {
            cuda.dispatch_tiled_f32(
                &gpu_tiles, pg.re_start, pg.im_start, pg.re_step, pg.im_step,
                JULIA_C[0], JULIA_C[1], MAX_ITER, FractalType::Mandelbrot.as_u32(),
            );
            cuda.readback_into(&mut out_old);
        });

        let mut out_new = vec![0.0f32; (W * H) as usize];
        let new_ms = bench("NEW: compact + par scatter", || {
            cuda.dispatch_tiled_f32_compact(
                &tagged, pg.re_start, pg.im_start, pg.re_step, pg.im_step,
                JULIA_C[0], JULIA_C[1], MAX_ITER, FractalType::Mandelbrot.as_u32(),
            );
            let compact = cuda.readback_compact(offset);
            let mut row_segments: Vec<Vec<(u32, u32, u32)>> = vec![Vec::new(); H as usize];
            for &[x0, y0, tw, th, off] in &tagged {
                for row in 0..th {
                    row_segments[(y0 + row) as usize].push((x0, tw, off + row * tw));
                }
            }
            out_new.par_chunks_mut(W as usize).zip(row_segments.par_iter()).for_each(|(row_slice, segs)| {
                for &(x0, tw, src) in segs {
                    let (x0, tw, src) = (x0 as usize, tw as usize, src as usize);
                    row_slice[x0..x0 + tw].copy_from_slice(&compact[src..src + tw]);
                }
            });
        });

        // Compare only the pixels this view's gpu_tiles actually cover (see
        // module doc comment for why the rest would be a false positive).
        let mut mismatches = 0u64;
        for &[x0, y0, tw, th] in &gpu_tiles {
            for row in 0..th {
                let base = ((y0 + row) * W + x0) as usize;
                for i in 0..tw as usize {
                    if out_old[base + i] != out_new[base + i] { mismatches += 1; }
                }
            }
        }
        println!("  [correctness] {mismatches} mismatches over {gpu_pixels} gpu-tile pixels");
        println!("  => {:+.4} ms ({:+.1}%) new vs old", new_ms - old_ms, (new_ms - old_ms) / old_ms * 100.0);
    }
}
