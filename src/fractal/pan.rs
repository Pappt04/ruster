use crate::fractal::fractal::{compute, pixel_grid, IterBuf};
use crate::fractal::fractal_type::FractalType;
use crate::gui::viewport::Viewport;
use rayon::prelude::*;

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
        // Vertical pan: shift whole rows (each row is a contiguous w-element chunk).
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
        // Horizontal pan: shift within each row.
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
