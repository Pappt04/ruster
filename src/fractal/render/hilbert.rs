//! Tiled rendering with pixels visited in Hilbert-curve order within each
//! tile, so that consecutively-computed pixels are also spatially close.
//! Escape-time cost (iteration count) is spatially correlated with
//! distance to the set boundary, so a locality-preserving visiting order
//! keeps a rayon thread's recently-computed neighbors — and the branch
//! predictor state built up iterating them — relevant for longer than a
//! naive row-major scan would.

use crate::fractal::fractal::{compute, pixel_grid, IterBuf};
use crate::fractal::fractal_type::FractalType;
use crate::gui::viewport::Viewport;
use rayon::prelude::*;

/// Side length of one Hilbert-ordered tile, in pixels. Matches `2^ORDER`.
pub const TILE: usize = 64;
/// Hilbert curve order: `2^ORDER == TILE`, so the curve exactly covers one
/// tile with no unused indices.
const ORDER: u32 = 6;

/// Maps a linear index `d` along a `2^ORDER x 2^ORDER` Hilbert curve to its
/// `(x, y)` position within the tile. Standard bit-interleaving
/// construction: at each of the `ORDER` recursion levels (`s` doubling
/// from 1 to `2^ORDER`), the two bits of `d` at that level select a
/// quadrant and, when `ry == 0`, trigger the curve's characteristic
/// reflection (swap x/y, and for `rx == 1` also mirror both axes) that
/// keeps the path continuous across quadrant boundaries.
pub fn hilbert_d_to_xy(d: usize) -> (u32, u32) {
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

/// Precomputes the full within-tile visiting order once, so
/// [`render_tiled`] can reuse the same `(x, y)` offset sequence for every
/// tile in the frame instead of recomputing [`hilbert_d_to_xy`] per pixel
/// per tile.
pub fn tile_order() -> Vec<(u32, u32)> {
    (0..TILE * TILE).map(hilbert_d_to_xy).collect()
}

/// Renders a full frame as a grid of [`TILE`]-sized tiles, with pixels
/// inside each tile visited in Hilbert order via the precomputed
/// [`tile_order`].
///
/// Horizontal bands of height `TILE` are distributed across the rayon
/// thread pool; each band is then swept tile by tile left to right, and
/// within a tile, offsets outside the frame or the (possibly clipped)
/// final band/column are skipped so partial tiles at the frame edges are
/// handled without a separate code path.
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
