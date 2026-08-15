use crate::fractal::fractal::{compute, pixel_grid, IterBuf};
use crate::fractal::fractal_type::FractalType;
use crate::gui::viewport::Viewport;
use rayon::prelude::*;

/// Reuses a previous frame's iteration buffer across an axis-aligned pan by
/// shifting the still-valid pixels in place and only recomputing the
/// newly-exposed strip.
///
/// A pan by `dx`/`dy` pixels moves every existing pixel's screen position
/// but not its complex-plane value, so `buf.copy_within` slides the
/// reusable region into its new position (row-by-row for a horizontal
/// pan, one bulk copy for vertical, since vertical shifts move whole
/// contiguous rows) and only the vacated `|dx|`- or `|dy|`-pixel-wide
/// strip along the leading edge is recomputed from scratch. This turns an
/// O(w*h) full re-render into an O(w*h) copy plus an O(w*|dy|) (or
/// O(h*|dx|)) recompute, which is cheaper whenever the pan is small
/// relative to the frame — the common case for interactive dragging.
/// Diagonal pans are not supported (`dx` and `dy` cannot both be nonzero)
/// since the shifted and newly-exposed regions would overlap in a way a
/// single `copy_within` cannot express.
pub fn shift_and_fill(
    buf: &mut IterBuf,
    w: usize,
    h: usize,
    dx: i32,
    dy: i32,
    vp: &Viewport,
    fractal: FractalType,
    julia_c: [f64; 2],
    max_iter: u32,
) {
    debug_assert!(dx == 0 || dy == 0, "shift_and_fill is axis-aligned only");
    debug_assert!((dx.unsigned_abs() as usize) < w && (dy.unsigned_abs() as usize) < h);

    let pg = pixel_grid(vp);

    if dy != 0 {
        if dy > 0 {
            let dy = dy as usize;
            buf.copy_within(0..(h - dy) * w, dy * w);
        } else {
            let dy = (-dy) as usize;
            buf.copy_within(dy * w..h * w, 0);
        }
        let (fill_y0, fill_y1) = if dy > 0 { (0, dy as usize) } else { (h - (-dy) as usize, h) };
        buf[fill_y0 * w..fill_y1 * w]
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let y = fill_y0 + row_idx;
                let im = pg.im_start + y as f64 * pg.im_step;
                for x in 0..w {
                    let re = pg.re_start + x as f64 * pg.re_step;
                    row[x] = compute(fractal, re, im, julia_c, max_iter);
                }
            });
    } else if dx != 0 {
        let (shift_dst, shift_src_len, fill_x0, fill_x1) = if dx > 0 {
            (dx as usize, w - dx as usize, 0usize, dx as usize)
        } else {
            (0usize, w - (-dx) as usize, w - (-dx) as usize, w)
        };
        buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            if dx > 0 {
                row.copy_within(0..shift_src_len, shift_dst);
            } else {
                row.copy_within((-dx) as usize..w, shift_dst);
            }
            let im = pg.im_start + y as f64 * pg.im_step;
            for x in fill_x0..fill_x1 {
                let re = pg.re_start + x as f64 * pg.re_step;
                row[x] = compute(fractal, re, im, julia_c, max_iter);
            }
        });
    }
}
