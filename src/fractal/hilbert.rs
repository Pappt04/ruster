//! Hilbert-curve pixel ordering for cache-friendly tile traversal (2b in
//! CURSOR_OPTIMIZATIONS.md). Standard bit-rotation algorithm.

use crate::fractal::fractal::{compute, pixel_grid, IterBuf};
use crate::fractal::fractal_type::FractalType;
use crate::gui::viewport::Viewport;
use rayon::prelude::*;

/// Side length of one traversal tile: 2^6 = 64.
pub const TILE: usize = 64;
const ORDER: u32 = 6;

/// Converts a Hilbert-curve distance `d` (0..4^order) to (x,y) within a
/// 2^order × 2^order square.
pub fn d2xy(d: usize) -> (u32, u32) {
    let mut rx: u32;
    let mut ry: u32;
    let mut t = d as u32;
    let mut x = 0u32;
    let mut y = 0u32;
    let mut s = 1u32;
    while s < (1u32 << ORDER) {
        rx = 1 & (t / 2);
        ry = 1 & (t ^ rx);
        // rotate
        if ry == 0 {
            if rx == 1 {
                x = s.wrapping_sub(1).wrapping_sub(x);
                y = s.wrapping_sub(1).wrapping_sub(y);
            }
            std::mem::swap(&mut x, &mut y);
        }
        x += s * rx;
        y += s * ry;
        t /= 4;
        s *= 2;
    }
    (x, y)
}

/// Precomputed Hilbert traversal order for one TILE×TILE tile (4096 (x,y) pairs).
pub fn tile_order() -> Vec<(u32, u32)> {
    (0..TILE * TILE).map(d2xy).collect()
}

/// Row-parallel render that traverses pixels within each 64×64 tile in Hilbert-curve
/// order for better L1/L2 cache locality (2b in CURSOR_OPTIMIZATIONS.md). Bit-
/// identical output to `render()` — only the write order differs. Parallelism grain
/// is a disjoint band of up to `TILE` rows (via `par_chunks_mut`, the same idiom
/// `render()` already uses), avoiding `unsafe` for finer per-tile dispatch.
pub fn render_tiled(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    let w = vp.width as usize;
    let h = vp.height as usize;

    let pg = pixel_grid(vp);
    let mut buf = vec![0.0f32; w * h];
    let order = tile_order();

    buf.par_chunks_mut(TILE * w).enumerate().for_each(|(band_idx, band)| {
        let y0 = band_idx * TILE;
        let band_h = (h - y0).min(TILE);

        let mut tx0 = 0usize;
        while tx0 < w {
            let tile_w = (w - tx0).min(TILE);
            for &(lx, ly) in &order {
                let (lx, ly) = (lx as usize, ly as usize);
                if lx >= tile_w || ly >= band_h {
                    continue;
                }
                let x = tx0 + lx;
                let y_local = ly;
                let re = pg.re_start + x as f64 * pg.re_step;
                let im = pg.im_start + (y0 + y_local) as f64 * pg.im_step;
                band[y_local * w + x] = compute(fractal, re, im, julia_c, max_iter);
            }
            tx0 += TILE;
        }
    });

    buf
}
