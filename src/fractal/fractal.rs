use crate::fractal::fractal_type::FractalType;
use crate::fractal::mandelbrot::{mandelbrot, mandelbrot_dem, mandelbrot_x4, mandelbrot_x8, mandelbrot_x8x2};
use crate::fractal::julia::{julia, julia_x4, julia_x8};
use crate::fractal::newton::newton;
use crate::fractal::nova::nova;
use crate::fractal::bulb_precheck::{in_cardioid_or_period2, in_period3_bulb};
use crate::gui::viewport::Viewport;
use rayon::prelude::*;

pub const ESCAPE_RADIUS_SQ: f64 = 4.0;
pub const ESCAPE_RADIUS_SQ_F32: f32 = 4.0;

pub const F32_PRECISION_THRESHOLD: f64 = 1e6;

pub type IterBuf = Vec<f32>;

#[derive(Clone, Copy, Debug)]
pub struct PixelGrid {
    pub re_start: f64,
    pub re_step: f64,
    pub im_start: f64,
    pub im_step: f64,
}

pub fn pixel_grid(vp: &Viewport) -> PixelGrid {
    let aspect = vp.width as f64 / vp.height as f64;
    let half = 2.0 / vp.zoom;
    let re_step = half * aspect * 2.0 / vp.width as f64;
    let im_step = half * 2.0 / vp.height as f64;
    let re_start = vp.center[0] + (0.5 / vp.width as f64 - 0.5) * half * aspect * 2.0;
    let im_start = vp.center[1] + (0.5 / vp.height as f64 - 0.5) * half * 2.0;
    PixelGrid { re_start, re_step, im_start, im_step }
}

pub fn render(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    let w = vp.width as usize;
    let h = vp.height as usize;

    let pg = pixel_grid(vp);
    let mut buf = vec![0.0f32; w * h];

    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im = pg.im_start + y as f64 * pg.im_step;
        for x in 0..w {
            let re = pg.re_start + x as f64 * pg.re_step;
            row[x] = compute(fractal, re, im, julia_c, max_iter);
        }
    });

    buf
}

/// Computes at a capped iteration count first; if the pixel doesn't resolve within
/// `cap` (i.e. it hit the cap without escaping/converging), re-runs at the true
/// `max_iter`. Self-correcting: a too-tight cap only costs extra work for that one
/// pixel, it can never produce a wrong result.
#[inline]
fn compute_capped(fractal: FractalType, re: f64, im: f64, julia_c: [f64; 2], cap: u32, true_max: u32) -> f32 {
    let guess = compute(fractal, re, im, julia_c, cap);
    if guess >= cap as f32 && cap < true_max {
        compute(fractal, re, im, julia_c, true_max)
    } else {
        guess
    }
}

/// Row-parallel scalar render using a neighbor-coherence iteration cap: within a row,
/// each pixel's cap is `min(prev_pixel_iter + slack, max_iter)`, reset to `max_iter`
/// at the start of every row (1-D coherence only). Pixels that don't resolve within
/// the cap are re-verified at the true `max_iter` via `compute_capped`, so output is
/// always correct — the cap only affects how much work is spent, never correctness.
/// Restricted to plain scalar rendering; not used by MS/perturbation/SIMD paths.
pub fn render_neighbor_capped(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32, slack: u32) -> IterBuf {
    let w = vp.width as usize;
    let h = vp.height as usize;

    let pg = pixel_grid(vp);
    let mut buf = vec![0.0f32; w * h];

    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im = pg.im_start + y as f64 * pg.im_step;
        let mut cap = max_iter;
        for x in 0..w {
            let re = pg.re_start + x as f64 * pg.re_step;
            let v = compute_capped(fractal, re, im, julia_c, cap, max_iter);
            row[x] = v;
            cap = ((v as u32).saturating_add(slack)).min(max_iter);
        }
    });

    buf
}

pub fn render_simd(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    use wide::f64x4;

    let w = vp.width as usize;
    let h = vp.height as usize;

    let pg = pixel_grid(vp);
    let mut buf = vec![0.0f32; w * h];

    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im = pg.im_start + y as f64 * pg.im_step;
        let im4 = f64x4::splat(im);
        let mut x = 0usize;

        while x + 4 <= w {
            let re = f64x4::from([
                pg.re_start + x as f64       * pg.re_step,
                pg.re_start + (x + 1) as f64 * pg.re_step,
                pg.re_start + (x + 2) as f64 * pg.re_step,
                pg.re_start + (x + 3) as f64 * pg.re_step,
            ]);
            let lanes = match fractal {
                FractalType::Mandelbrot => mandelbrot_x4(re, im4, max_iter),
                FractalType::Julia => {
                    let cr4 = f64x4::splat(julia_c[0]);
                    let ci4 = f64x4::splat(julia_c[1]);
                    julia_x4(re, im4, cr4, ci4, max_iter)
                }
                _ => unreachable!(),
            };
            row[x..x+4].copy_from_slice(&lanes);
            x += 4;
        }

        // scalar tail (0–3 remaining pixels)
        for x in x..w {
            let re = pg.re_start + x as f64 * pg.re_step;
            row[x] = match fractal {
                FractalType::Mandelbrot => mandelbrot(re, im, max_iter),
                FractalType::Julia => julia(re, im, julia_c[0], julia_c[1], max_iter),
                _ => unreachable!(),
            };
        }
    });

    buf
}

/// 8-wide f32 SIMD path — 2× more pixels per instruction than f64x4.
/// Coordinate values are computed in f64 then narrowed to f32 per lane so
/// the initial pixel mapping stays accurate before the cast.
pub fn render_simd_f32(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    use wide::f32x8;

    let w = vp.width as usize;
    let h = vp.height as usize;

    let pg = pixel_grid(vp);
    let mut buf = vec![0.0f32; w * h];

    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im = (pg.im_start + y as f64 * pg.im_step) as f32;
        let im8 = f32x8::splat(im);
        let mut x = 0usize;

        while x + 8 <= w {
            let re = f32x8::from([
                (pg.re_start + (x    ) as f64 * pg.re_step) as f32,
                (pg.re_start + (x + 1) as f64 * pg.re_step) as f32,
                (pg.re_start + (x + 2) as f64 * pg.re_step) as f32,
                (pg.re_start + (x + 3) as f64 * pg.re_step) as f32,
                (pg.re_start + (x + 4) as f64 * pg.re_step) as f32,
                (pg.re_start + (x + 5) as f64 * pg.re_step) as f32,
                (pg.re_start + (x + 6) as f64 * pg.re_step) as f32,
                (pg.re_start + (x + 7) as f64 * pg.re_step) as f32,
            ]);
            let lanes = match fractal {
                FractalType::Mandelbrot => mandelbrot_x8(re, im8, max_iter),
                FractalType::Julia => {
                    let cr8 = f32x8::splat(julia_c[0] as f32);
                    let ci8 = f32x8::splat(julia_c[1] as f32);
                    julia_x8(re, im8, cr8, ci8, max_iter)
                }
                _ => unreachable!(),
            };
            row[x..x + 8].copy_from_slice(&lanes);
            x += 8;
        }

        // scalar tail (0–7 remaining pixels)
        let im_f64 = pg.im_start + y as f64 * pg.im_step;
        for x in x..w {
            let re = pg.re_start + x as f64 * pg.re_step;
            row[x] = match fractal {
                FractalType::Mandelbrot => mandelbrot(re, im_f64, max_iter),
                FractalType::Julia => julia(re, im_f64, julia_c[0], julia_c[1], max_iter),
                _ => unreachable!(),
            };
        }
    });

    buf
}

/// Like `render_simd_f32` but processes 16 pixels/iteration via two interleaved
/// `f32x8` chains (`mandelbrot_x8x2`) to increase instruction-level parallelism.
/// Mandelbrot only (falls back to the 8-wide kernel for Julia and for the 8-15
/// pixel remainder, then scalar for the final <8 tail). Bit-identical output to
/// `render_simd_f32` — see CURSOR_OPTIMIZATIONS.md 2a.
pub fn render_simd_f32_ilp(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    use wide::f32x8;

    let w = vp.width as usize;
    let h = vp.height as usize;

    let pg = pixel_grid(vp);
    let mut buf = vec![0.0f32; w * h];

    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im = (pg.im_start + y as f64 * pg.im_step) as f32;
        let im8 = f32x8::splat(im);
        let mut x = 0usize;

        if fractal == FractalType::Mandelbrot {
            while x + 16 <= w {
                let re0 = f32x8::from(std::array::from_fn::<f32, 8, _>(|k| {
                    (pg.re_start + (x + k) as f64 * pg.re_step) as f32
                }));
                let re1 = f32x8::from(std::array::from_fn::<f32, 8, _>(|k| {
                    (pg.re_start + (x + 8 + k) as f64 * pg.re_step) as f32
                }));
                let (lanes0, lanes1) = mandelbrot_x8x2(re0, im8, re1, im8, max_iter);
                row[x..x + 8].copy_from_slice(&lanes0);
                row[x + 8..x + 16].copy_from_slice(&lanes1);
                x += 16;
            }
        }

        // 8-15 remaining pixels: fall back to the single-batch 8-wide kernel
        while x + 8 <= w {
            let re = f32x8::from(std::array::from_fn::<f32, 8, _>(|k| {
                (pg.re_start + (x + k) as f64 * pg.re_step) as f32
            }));
            let lanes = match fractal {
                FractalType::Mandelbrot => mandelbrot_x8(re, im8, max_iter),
                FractalType::Julia => {
                    let cr8 = f32x8::splat(julia_c[0] as f32);
                    let ci8 = f32x8::splat(julia_c[1] as f32);
                    julia_x8(re, im8, cr8, ci8, max_iter)
                }
                _ => unreachable!(),
            };
            row[x..x + 8].copy_from_slice(&lanes);
            x += 8;
        }

        // scalar tail (0-7 remaining pixels)
        let im_f64 = pg.im_start + y as f64 * pg.im_step;
        for x in x..w {
            let re = pg.re_start + x as f64 * pg.re_step;
            row[x] = match fractal {
                FractalType::Mandelbrot => mandelbrot(re, im_f64, max_iter),
                FractalType::Julia => julia(re, im_f64, julia_c[0], julia_c[1], max_iter),
                _ => unreachable!(),
            };
        }
    });

    buf
}

/// Row-parallel render that traverses pixels within each 64×64 tile in Hilbert-curve
/// order for better L1/L2 cache locality (2b in CURSOR_OPTIMIZATIONS.md). Bit-
/// identical output to `render()` — only the write order differs. Parallelism grain
/// is a disjoint band of up to `TILE` rows (via `par_chunks_mut`, the same idiom
/// `render()` already uses), avoiding `unsafe` for finer per-tile dispatch.
pub fn render_tiled(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    use crate::fractal::hilbert::{tile_order, TILE};

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


pub fn render_mariani_silver_dem(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32, k: f64) -> IterBuf {
    if fractal != FractalType::Mandelbrot {
        return render_mariani_silver(vp, fractal, julia_c, max_iter);
    }
    let w = vp.width as usize;
    let h = vp.height as usize;
    let pg = pixel_grid(vp);
    let mut buf = vec![f32::NAN; w * h];
    ms_fill_dem(&mut buf, w, julia_c, max_iter, &pg, k, 0, 0, w, h);
    buf
}

fn ms_fill_dem(
    buf: &mut [f32],
    stride: usize,
    julia_c: [f64; 2],
    max_iter: u32,
    pg: &PixelGrid,
    k: f64,
    x0: usize, y0: usize, x1: usize, y1: usize,
) {
    let fractal = FractalType::Mandelbrot;
    let w = x1 - x0;
    let h = y1 - y0;

    if w <= MS_MIN || h <= MS_MIN {
        ms_fill(buf, stride, fractal, julia_c, max_iter, pg.re_start, pg.im_start, pg.re_step, pg.im_step, x0, y0, x1, y1);
        return;
    }

    // DEM corner cull, tried before computing the full border. Corner coords:
    let cs = [(x0, y0), (x1 - 1, y0), (x0, y1 - 1), (x1 - 1, y1 - 1)];
    let diag = ((w as f64 * pg.re_step).powi(2) + (h as f64 * pg.im_step).powi(2)).sqrt();
    let mut vals = [0.0f32; 4];
    let mut culled = true;
    for (i, &(cx, cy)) in cs.iter().enumerate() {
        let (v, d) = mandelbrot_dem(pg.re_start + cx as f64 * pg.re_step, pg.im_start + cy as f64 * pg.im_step, max_iter);
        vals[i] = v;
        if v >= max_iter as f32 || d < k * diag {
            culled = false;
            break;
        }
    }
    // Secondary guard: corner smooth values must agree closely, or interpolation bands.
    if culled {
        let vmin = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let vmax = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if vmax - vmin > 2.0 { culled = false; }
    }
    if culled {
        let fw = (w - 1).max(1) as f32;
        let fh = (h - 1).max(1) as f32;
        for y in y0..y1 {
            let ty = (y - y0) as f32 / fh;
            for x in x0..x1 {
                let tx = (x - x0) as f32 / fw;
                let top = vals[0] + (vals[1] - vals[0]) * tx;
                let bot = vals[2] + (vals[3] - vals[2]) * tx;
                buf[y * stride + x] = top + (bot - top) * ty;
            }
        }
        return;
    }

    // Fall back to the exact MS logic for this rect, recursing through the DEM
    // variant: compute border, uniform-fill or subdivide.
    // (Reuses ms_fill for border+uniform by delegating one level when small.)
    // Compute border pixels.
    for x in x0..x1 {
        for &y in &[y0, y1 - 1] {
            let idx = y * stride + x;
            if buf[idx].is_nan() {
                buf[idx] = compute(fractal, pg.re_start + x as f64 * pg.re_step, pg.im_start + y as f64 * pg.im_step, julia_c, max_iter);
            }
        }
    }
    for y in (y0 + 1)..(y1 - 1) {
        for &x in &[x0, x1 - 1] {
            let idx = y * stride + x;
            if buf[idx].is_nan() {
                buf[idx] = compute(fractal, pg.re_start + x as f64 * pg.re_step, pg.im_start + y as f64 * pg.im_step, julia_c, max_iter);
            }
        }
    }

    let border_val = buf[y0 * stride + x0];
    let uniform = 'check: {
        for x in x0..x1 {
            if buf[y0 * stride + x] != border_val { break 'check false; }
            if buf[(y1 - 1) * stride + x] != border_val { break 'check false; }
        }
        for y in (y0 + 1)..(y1 - 1) {
            if buf[y * stride + x0] != border_val { break 'check false; }
            if buf[y * stride + (x1 - 1)] != border_val { break 'check false; }
        }
        true
    };

    if uniform {
        for y in (y0 + 1)..(y1 - 1) {
            for x in (x0 + 1)..(x1 - 1) {
                buf[y * stride + x] = border_val;
            }
        }
    } else if w >= h {
        let mid = x0 + w / 2;
        ms_fill_dem(buf, stride, julia_c, max_iter, pg, k, x0, y0, mid, y1);
        ms_fill_dem(buf, stride, julia_c, max_iter, pg, k, mid, y0, x1, y1);
    } else {
        let mid = y0 + h / 2;
        ms_fill_dem(buf, stride, julia_c, max_iter, pg, k, x0, y0, x1, mid);
        ms_fill_dem(buf, stride, julia_c, max_iter, pg, k, x0, mid, x1, y1);
    }
}

/// Derivative threshold for interior detection: once |dz/dz₀| (the running product
/// of 2·z_n) falls below this, the orbit is inside an attracting cycle basin and
/// the pixel is classified interior. This is an approximation (a true boundary
/// pixel can have a small-but-nonzero derivative) — validate via `--compare-ide`.
const IDE_DER_SQ: f64 = 1e-24;

/// Mandelbrot kernel with interior distance-style early exit: tracks the running
/// derivative product `der ← 2·z·der` and returns `max_iter` (interior) as soon as
/// |der| collapses below `IDE_DER_SQ` — attracting-cycle basins are detected many
/// iterations before Brent's cycle check would fire. Keeps the exact
/// cardioid/period-2/period-3 pre-checks and escape logic of `mandelbrot()`.
pub fn mandelbrot_ide(cr: f64, ci: f64, max_iter: u32) -> f32 {
    if in_cardioid_or_period2(cr, ci) || in_period3_bulb(cr, ci) {
        return max_iter as f32;
    }

    let mut zr = cr;
    let mut zi = ci;
    let mut der_r = 1.0f64;
    let mut der_i = 0.0f64;

    for i in 1..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        let zn_sq = zr2 + zi2;
        if zn_sq > ESCAPE_RADIUS_SQ {
            return smooth_iter(i, zn_sq, max_iter);
        }
        // der ← 2·z·der (pre-update z)
        let new_der_r = 2.0 * (zr * der_r - zi * der_i);
        let new_der_i = 2.0 * (zr * der_i + zi * der_r);
        der_r = new_der_r;
        der_i = new_der_i;
        if der_r * der_r + der_i * der_i < IDE_DER_SQ {
            return max_iter as f32; // attracting cycle → interior
        }
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }
    max_iter as f32
}

/// Row-parallel Mandelbrot render with biased interior checking (1b): the
/// derivative-tracking `mandelbrot_ide` kernel is only used when the previous
/// pixel in the row was interior (interior regions cluster, so the fast interior
/// exit is likely to pay off); after an exterior pixel, the plain `mandelbrot()`
/// kernel runs without the ~4 FLOPs/iter derivative overhead.
pub fn render_ide_biased(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    if fractal != FractalType::Mandelbrot {
        return render(vp, fractal, julia_c, max_iter);
    }
    let w = vp.width as usize;
    let h = vp.height as usize;
    let pg = pixel_grid(vp);
    let mut buf = vec![0.0f32; w * h];

    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im = pg.im_start + y as f64 * pg.im_step;
        let mut prev_interior = false;
        for x in 0..w {
            let re = pg.re_start + x as f64 * pg.re_step;
            let v = if prev_interior {
                mandelbrot_ide(re, im, max_iter)
            } else {
                mandelbrot(re, im, max_iter)
            };
            prev_interior = v >= max_iter as f32;
            row[x] = v;
        }
    });

    buf
}

/// Shifts `buf` in place by `(dx, dy)` pixels and fills only the newly exposed
/// strip using the same `pixel_grid(vp)` formula a full render would use — `vp` is
/// the NEW viewport being panned to (`buf` on entry holds the OLD frame's data).
/// Axis-aligned only (`dx == 0 || dy == 0`) — the caller (the render worker) must
/// verify this, along with `|dx| < w` and `|dy| < h` (no overlap ⇒ do a full render
/// instead; this function does not handle that case and will panic/behave
/// incorrectly if called with an out-of-range or diagonal delta).
///
/// Note: recycled (copied) pixels are NOT always bit-identical to a from-scratch
/// render of the new viewport — `new_vp.re_start` is recomputed from
/// `new_vp.center` via an independent floating-point chain, which can differ from
/// `old_vp.re_start + dx * re_step` in the last few ULPs. This occasionally tips a
/// pixel's `smooth_iter` log evaluation across a rounding boundary (observed:
/// ~0.01% of pixels, magnitude ~1e-4 relative to max_iter). This is expected
/// floating-point noise, not a logic bug — validate with a small tolerance, not
/// exact equality, in `--compare-pan-recycle`.
/// See 3b in CURSOR_OPTIMIZATIONS.md.
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

/// Mariani-Silver rectangle subdivision: compute only border pixels and fill
/// uniform interiors without evaluating every point individually.
pub fn render_mariani_silver(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    let w = vp.width as usize;
    let h = vp.height as usize;

    let pg = pixel_grid(vp);
    let mut buf = vec![f32::NAN; w * h];
    ms_fill(&mut buf, w, fractal, julia_c, max_iter, pg.re_start, pg.im_start, pg.re_step, pg.im_step, 0, 0, w, h);
    buf
}

/// Render one `[x0, y0, tw, th]` tile (in full-frame pixel coordinates)
/// exactly, one `compute()` call per pixel, returning a tile-local (not
/// full-frame) buffer.
///
/// Used by the heterogeneous scheduler (`crate::scheduler`) to fill CPU-routed
/// boundary tiles. This deliberately does NOT use `ms_fill`'s border-uniformity
/// flood-fill shortcut: that check only samples a rectangle's border, so a
/// thin escaping filament entirely contained in the interior (never touching
/// a border at any recursion depth) can be silently flood-filled with the
/// wrong value. The scheduler's whole point is to route exactly the tiles
/// most likely to contain that kind of feature to the CPU, which makes this
/// approximation's failure mode land precisely where the scheduler can least
/// afford it — so CPU tiles pay full per-pixel cost for a bit-exact match to
/// `render()` instead.
pub fn render_tile_exact(pg: &PixelGrid, fractal: FractalType, julia_c: [f64; 2], max_iter: u32, tile: [u32; 4]) -> Vec<f32> {
    let [_, _, tw, th] = tile;
    let mut local = vec![0.0f32; (tw * th) as usize];
    render_tile_exact_into(pg, fractal, julia_c, max_iter, tile, &mut local);
    local
}

/// `render_tile_exact` writing into a caller-owned `out` (row-major, `tw*th`
/// long) instead of allocating. Exists so the heterogeneous scheduler can hand
/// workers a reused buffer: allocating one `Vec` per tile was measured at
/// 1.0-3.9 ms per frame — dominated by page faults on freshly-mapped pages,
/// and an order of magnitude above every other piece of scheduler machinery
/// (see results/summary.md §3.4 and `examples/sched_overhead.rs`).
///
/// # Panics
/// If `out.len() != tw * th`.
pub fn render_tile_exact_into(
    pg: &PixelGrid, fractal: FractalType, julia_c: [f64; 2], max_iter: u32,
    tile: [u32; 4], out: &mut [f32],
) {
    let [x0, y0, tw, th] = tile;
    assert_eq!(out.len(), (tw * th) as usize, "tile buffer size mismatch");
    for ly in 0..th {
        let im = pg.im_start + (y0 + ly) as f64 * pg.im_step;
        for lx in 0..tw {
            let re = pg.re_start + (x0 + lx) as f64 * pg.re_step;
            out[(ly * tw + lx) as usize] = compute(fractal, re, im, julia_c, max_iter);
        }
    }
}

/// Like `render_tile_exact` but uses the f32 SIMD kernels (`mandelbrot_x8`/
/// `julia_x8`) for 8-wide chunks within each row, falling back to scalar
/// `pixel()`-equivalent math for the remainder. Mandelbrot/Julia only,
/// mirroring `render_simd_f32`'s fractal support.
///
/// NOT bit-identical to `render_tile_exact`/`render()` — f32 SIMD vs f64
/// scalar arithmetic diverge in the last bit or two on chaotic escape-time
/// iteration, the same documented tradeoff `CudaFractal::render()`'s f32 fast
/// path already makes for GPU. Callers that need the heterogeneous
/// scheduler's bit-exact-vs-CPU-`render()` guarantee must not use this
/// unconditionally — see `SchedulerConfig::simd_cpu_tiles` and
/// `render_cpu_tile`, which gates it.
pub fn render_tile_exact_simd(pg: &PixelGrid, fractal: FractalType, julia_c: [f64; 2], max_iter: u32, tile: [u32; 4]) -> Vec<f32> {
    let [_, _, tw, th] = tile;
    let mut local = vec![0.0f32; (tw * th) as usize];
    render_tile_exact_simd_into(pg, fractal, julia_c, max_iter, tile, &mut local);
    local
}

/// `render_tile_exact_simd` writing into a caller-owned `out` instead of
/// allocating — see `render_tile_exact_into` for why.
///
/// # Panics
/// If `out.len() != tw * th`.
pub fn render_tile_exact_simd_into(
    pg: &PixelGrid, fractal: FractalType, julia_c: [f64; 2], max_iter: u32,
    tile: [u32; 4], out: &mut [f32],
) {
    use wide::f32x8;

    let [x0, y0, tw, th] = tile;
    assert_eq!(out.len(), (tw * th) as usize, "tile buffer size mismatch");
    let local = out;

    for ly in 0..th {
        let y = y0 + ly;
        let im = (pg.im_start + y as f64 * pg.im_step) as f32;
        let im8 = f32x8::splat(im);
        let row = &mut local[(ly * tw) as usize..(ly * tw + tw) as usize];
        let mut lx = 0u32;

        while lx + 8 <= tw {
            let x = x0 + lx;
            let re = f32x8::from(std::array::from_fn::<f32, 8, _>(|k| {
                (pg.re_start + (x + k as u32) as f64 * pg.re_step) as f32
            }));
            let lanes = match fractal {
                FractalType::Mandelbrot => mandelbrot_x8(re, im8, max_iter),
                FractalType::Julia => {
                    let cr8 = f32x8::splat(julia_c[0] as f32);
                    let ci8 = f32x8::splat(julia_c[1] as f32);
                    julia_x8(re, im8, cr8, ci8, max_iter)
                }
                _ => unreachable!("render_tile_exact_simd only supports Mandelbrot/Julia"),
            };
            row[lx as usize..lx as usize + 8].copy_from_slice(&lanes);
            lx += 8;
        }

        // scalar tail (0-7 remaining columns)
        let im_f64 = pg.im_start + y as f64 * pg.im_step;
        while lx < tw {
            let x = x0 + lx;
            let re = pg.re_start + x as f64 * pg.re_step;
            row[lx as usize] = match fractal {
                FractalType::Mandelbrot => mandelbrot(re, im_f64, max_iter),
                FractalType::Julia => julia(re, im_f64, julia_c[0], julia_c[1], max_iter),
                _ => unreachable!("render_tile_exact_simd only supports Mandelbrot/Julia"),
            };
            lx += 1;
        }
    }
}

/// Dispatches a CPU tile render to the SIMD fast path
/// (`render_tile_exact_simd`, Mandelbrot/Julia below `F32_PRECISION_THRESHOLD`,
/// when `use_simd` is set) or the exact scalar path (`render_tile_exact`)
/// otherwise. `use_simd` is `SchedulerConfig::simd_cpu_tiles` gated at the
/// call site — see `render_tile_exact_simd`'s doc comment for why this isn't
/// unconditional.
pub fn render_cpu_tile(pg: &PixelGrid, fractal: FractalType, julia_c: [f64; 2], max_iter: u32, tile: [u32; 4], use_simd: bool, zoom: f64) -> Vec<f32> {
    let [_, _, tw, th] = tile;
    let mut out = vec![0.0f32; (tw * th) as usize];
    render_cpu_tile_into(pg, fractal, julia_c, max_iter, tile, use_simd, zoom, &mut out);
    out
}

/// `render_cpu_tile` writing into a caller-owned `out` instead of allocating.
/// This is what the heterogeneous scheduler calls; see `render_tile_exact_into`.
///
/// # Panics
/// If `out.len() != tw * th`.
#[allow(clippy::too_many_arguments)]
pub fn render_cpu_tile_into(
    pg: &PixelGrid, fractal: FractalType, julia_c: [f64; 2], max_iter: u32,
    tile: [u32; 4], use_simd: bool, zoom: f64, out: &mut [f32],
) {
    if use_simd
        && matches!(fractal, FractalType::Mandelbrot | FractalType::Julia)
        && zoom < F32_PRECISION_THRESHOLD
    {
        render_tile_exact_simd_into(pg, fractal, julia_c, max_iter, tile, out)
    } else {
        render_tile_exact_into(pg, fractal, julia_c, max_iter, tile, out)
    }
}

/// Minimum side length at which we stop subdividing and compute all pixels directly.
const MS_MIN: usize = 2;

fn ms_fill(
    buf: &mut [f32],
    stride: usize,
    fractal: FractalType,
    julia_c: [f64; 2],
    max_iter: u32,
    re_start: f64,
    im_start: f64,
    re_step: f64,
    im_step: f64,
    x0: usize, y0: usize, x1: usize, y1: usize,
) {
    let w = x1 - x0;
    let h = y1 - y0;

    // Base case: too small to subdivide further — compute every pixel.
    if w <= MS_MIN || h <= MS_MIN {
        for y in y0..y1 {
            let im = im_start + y as f64 * im_step;
            for x in x0..x1 {
                let idx = y * stride + x;
                if buf[idx].is_nan() {
                    buf[idx] = compute(fractal, re_start + x as f64 * re_step, im, julia_c, max_iter);
                }
            }
        }
        return;
    }

    // Compute uncomputed border pixels.
    // Top row
    {
        let im = im_start + y0 as f64 * im_step;
        for x in x0..x1 {
            let idx = y0 * stride + x;
            if buf[idx].is_nan() {
                buf[idx] = compute(fractal, re_start + x as f64 * re_step, im, julia_c, max_iter);
            }
        }
    }
    // Bottom row
    {
        let im = im_start + (y1 - 1) as f64 * im_step;
        for x in x0..x1 {
            let idx = (y1 - 1) * stride + x;
            if buf[idx].is_nan() {
                buf[idx] = compute(fractal, re_start + x as f64 * re_step, im, julia_c, max_iter);
            }
        }
    }
    // Left column (interior rows only)
    {
        let re = re_start + x0 as f64 * re_step;
        for y in (y0 + 1)..(y1 - 1) {
            let idx = y * stride + x0;
            if buf[idx].is_nan() {
                buf[idx] = compute(fractal, re, im_start + y as f64 * im_step, julia_c, max_iter);
            }
        }
    }
    // Right column (interior rows only)
    {
        let re = re_start + (x1 - 1) as f64 * re_step;
        for y in (y0 + 1)..(y1 - 1) {
            let idx = y * stride + (x1 - 1);
            if buf[idx].is_nan() {
                buf[idx] = compute(fractal, re, im_start + y as f64 * im_step, julia_c, max_iter);
            }
        }
    }

    // Check whether all border pixels share the same value.
    let border_val = buf[y0 * stride + x0];
    let uniform = 'check: {
        for x in x0..x1 {
            if buf[y0 * stride + x] != border_val { break 'check false; }
            if buf[(y1 - 1) * stride + x] != border_val { break 'check false; }
        }
        for y in (y0 + 1)..(y1 - 1) {
            if buf[y * stride + x0] != border_val { break 'check false; }
            if buf[y * stride + (x1 - 1)] != border_val { break 'check false; }
        }
        true
    };

    if uniform {
        // Flood-fill the interior.
        for y in (y0 + 1)..(y1 - 1) {
            for x in (x0 + 1)..(x1 - 1) {
                buf[y * stride + x] = border_val;
            }
        }
    } else {
        // Subdivide along the longer axis (non-overlapping halves).
        if w >= h {
            let mid = x0 + w / 2;
            ms_fill(buf, stride, fractal, julia_c, max_iter, re_start, im_start, re_step, im_step, x0, y0, mid, y1);
            ms_fill(buf, stride, fractal, julia_c, max_iter, re_start, im_start, re_step, im_step, mid, y0, x1, y1);
        } else {
            let mid = y0 + h / 2;
            ms_fill(buf, stride, fractal, julia_c, max_iter, re_start, im_start, re_step, im_step, x0, y0, x1, mid);
            ms_fill(buf, stride, fractal, julia_c, max_iter, re_start, im_start, re_step, im_step, x0, mid, x1, y1);
        }
    }
}

/// Single-pixel entry point exposed for benchmarking.
pub fn pixel(fractal: FractalType, re: f64, im: f64, julia_c: [f64; 2], max_iter: u32) -> f32 {
    compute(fractal, re, im, julia_c, max_iter)
}

/// Analytical floating-point operation count per kernel call (lower bound, no FMA fusion).
pub const fn flops_per_iter(fractal: FractalType) -> u64 {
    match fractal {
        // 2 mul (zr²,zi²) + 1 add (zn_sq) + 2 mul + 1 add (zi) + 1 sub + 1 add (zr) = 8
        FractalType::Mandelbrot | FractalType::Julia => 8,
        // z³ (6) + f (2) + f' (4) + complex div+sub (10) + step check (3) = 25
        FractalType::Newton => 25,
        // Newton + 2 adds for nova perturbation = 27
        FractalType::Nova => 27,
    }
}

fn compute(fractal: FractalType, re: f64, im: f64, julia_c: [f64; 2], max_iter: u32) -> f32 {
    match fractal {
        FractalType::Nova => nova(re, im, max_iter),
        FractalType::Newton => newton(re, im, max_iter),
        FractalType::Mandelbrot => mandelbrot(re, im, max_iter),
        FractalType::Julia => julia(re, im, julia_c[0], julia_c[1], max_iter),
    }
}

pub(crate) fn smooth_iter(iter: u32, zn_sq: f64, max_iter: u32) -> f32 {
    if iter >= max_iter {
        return max_iter as f32;
    }
    const INV_LN2: f64 = std::f64::consts::LOG2_E;
    let log_zn = zn_sq.ln() / 2.0;
    let nu = (log_zn * INV_LN2).ln() * INV_LN2;
    (iter as f64 + 1.0 - nu) as f32
}

pub(crate) fn smooth_iter_f32(iter: u32, zn_sq: f32, max_iter: u32) -> f32 {
    if iter >= max_iter { return max_iter as f32; }
    const INV_LN2: f32 = std::f32::consts::LOG2_E;
    let log_zn = zn_sq.ln() / 2.0;
    let nu = (log_zn * INV_LN2).ln() * INV_LN2;
    iter as f32 + 1.0 - nu
}