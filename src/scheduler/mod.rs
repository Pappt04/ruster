//! Adaptive prepass-guided heterogeneous CPU+GPU scheduler (see `TUTORIAL.md`
//! Stage 3). Classifies the frame into tiles from a cheap coarse GPU prepass,
//! routes divergent "boundary" tiles to the CPU (Mariani-Silver fill) and
//! coherent tiles to the GPU (`fractal_kernel_tiled`), running both
//! concurrently, then composites into one full-resolution buffer.
//!
//! Output is bit-identical to a plain full-frame render — no tile is skipped
//! or approximated, only *where* it's computed changes.

pub mod classifier;
pub mod controller;

use rayon::prelude::*;
use crate::fractal::{pixel_grid, render_tile_ms, IterBuf};
use crate::fractal::fractal_type::FractalType;
use crate::gpu::cuda::CudaFractal;
use crate::gui::viewport::Viewport;
use classifier::TileKind;
use controller::ThresholdController;

pub struct SchedulerConfig {
    pub tile_size: u32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        // Multiple of the 16x16 GPU block dim used by fractal_kernel_tiled.
        Self { tile_size: 32 }
    }
}

pub struct HeterogeneousResult {
    pub buf: IterBuf,
    pub gpu_ms: f32,
    pub cpu_ms: f32,
    pub cpu_tile_frac: f32,
}

/// Render one frame by adaptively splitting it into GPU and CPU tiles and
/// running both concurrently. `cuda` must already be sized for `vp`'s
/// width/height (the caller is responsible for recreating it on resize, same
/// as the existing plain-CUDA render path).
pub fn render_heterogeneous(
    vp: &Viewport,
    fractal: FractalType,
    julia_c: [f64; 2],
    max_iter: u32,
    cuda: &mut CudaFractal,
    controller: &mut ThresholdController,
    cfg: &SchedulerConfig,
) -> HeterogeneousResult {
    let w = vp.width;
    let h = vp.height;
    debug_assert_eq!(w, cuda.width(), "CudaFractal not sized for this viewport");
    debug_assert_eq!(h, cuda.height(), "CudaFractal not sized for this viewport");

    let pg = pixel_grid(vp);

    // --- Phase 1: coarse GPU prepass at 1/8 resolution (ceiling-divided so
    // edge tiles still get real prepass coverage instead of falling back to
    // the zero-sample default in `tile_stats`). ---
    let pw = (w + 7) / 8;
    let ph = (h + 7) / 8;
    let prepass = cuda.render_prepass(
        pg.re_start, pg.im_start, pg.re_step, pg.im_step,
        julia_c[0], julia_c[1], max_iter, fractal.as_u32(),
        pw, ph,
    );

    // --- Phase 2: build the tile grid and classify each tile from the prepass. ---
    let tile_size = cfg.tile_size.max(1);
    let tiles_x = (w + tile_size - 1) / tile_size;
    let tiles_y = (h + tile_size - 1) / tile_size;

    let mut gpu_tiles: Vec<[u32; 4]> = Vec::new();
    let mut cpu_tiles: Vec<[u32; 4]> = Vec::new();

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x0 = tx * tile_size;
            let y0 = ty * tile_size;
            let tw = tile_size.min(w - x0);
            let th = tile_size.min(h - y0);

            // Corresponding rect in prepass space (same tile grid, 1/8 scale).
            let px0 = x0 / 8;
            let py0 = y0 / 8;
            let ptw = ((tw + 7) / 8).max(1);
            let pth = ((th + 7) / 8).max(1);

            let stats = classifier::tile_stats(&prepass, pw, px0, py0, ptw, pth, max_iter);
            match classifier::classify(&stats, controller.threshold, max_iter) {
                TileKind::CpuBoundary => cpu_tiles.push([x0, y0, tw, th]),
                TileKind::GpuInterior | TileKind::GpuExterior | TileKind::GpuSmooth => gpu_tiles.push([x0, y0, tw, th]),
            }
        }
    }
    let total_tiles = gpu_tiles.len() + cpu_tiles.len();

    // --- Phase 3: concurrent dispatch. CPU tiles run on a scoped thread (only
    // needs Send, not 'static, so it can borrow pg/cpu_tiles/julia_c freely);
    // the GPU call runs synchronously on this thread since it never yields
    // anyway (cudarc's copies are blocking) and keeps `&mut cuda` local. ---
    let (buf, gpu_ms, cpu_ms) = std::thread::scope(|s| {
        let cpu_handle = s.spawn(|| {
            let t0 = std::time::Instant::now();
            let results: Vec<([u32; 4], Vec<f32>)> = cpu_tiles
                .par_iter()
                .map(|&tile| (tile, render_tile_ms(&pg, fractal, julia_c, max_iter, tile)))
                .collect();
            (results, t0.elapsed())
        });

        let t1 = std::time::Instant::now();
        let gpu_buf = cuda.render_tiled(
            &gpu_tiles,
            pg.re_start, pg.im_start, pg.re_step, pg.im_step,
            julia_c[0], julia_c[1], max_iter, fractal.as_u32(),
        );
        let gpu_ms = t1.elapsed().as_secs_f32() * 1000.0;

        let (cpu_results, cpu_elapsed) = cpu_handle.join().unwrap();
        let cpu_ms = cpu_elapsed.as_secs_f32() * 1000.0;

        // gpu_buf is already correct at every gpu_tile position (full w*h,
        // dtoh-copied from the persistent device output); everything else is
        // stale from a previous frame. Patch in each CPU tile's own buffer —
        // together gpu_tiles/cpu_tiles partition every pixel exactly once.
        let mut buf = gpu_buf;
        for ([x0, y0, tw, th], local) in cpu_results {
            for row in 0..th {
                let dst = ((y0 + row) * w + x0) as usize;
                let src = (row * tw) as usize;
                buf[dst..dst + tw as usize].copy_from_slice(&local[src..src + tw as usize]);
            }
        }
        (buf, gpu_ms, cpu_ms)
    });

    controller.update(gpu_ms, cpu_ms);

    HeterogeneousResult {
        buf,
        gpu_ms,
        cpu_ms,
        cpu_tile_frac: if total_tiles > 0 { cpu_tiles.len() as f32 / total_tiles as f32 } else { 0.0 },
    }
}
