use crate::fractal::fractal_type::FractalType;
use crate::gui::viewport::Viewport;
use rayon::prelude::*;

// Bailout radius 2 (radius² = 4) — the conventional escape-time threshold.
// Previously 256² (65536), which cost every escaping pixel ~3-4 extra
// iterations for negligible benefit: smooth_iter()'s correction term is
// log(log|z|)-based (doubly logarithmic in the bailout radius), so a real
// full-frame render diff against the old 65536 radius showed only a small
// shift in the fractional coloring value (mean ~0.055, max ~2.75, on a
// max_iter=1000 scale) that is imperceptible after histogram-equalized
// palette mapping — confirmed by direct visual comparison of both radii at
// the same view, not just the idealized per-pixel math.
pub const ESCAPE_RADIUS_SQ: f64 = 4.0;
pub const ESCAPE_RADIUS_SQ_F32: f32 = 4.0;

/// Below this zoom level f32 has enough precision (~7 sig-figs); use f32x8.
/// Above it fall back to f64x4 to avoid glitching artefacts.
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

// ── Exterior/interior distance estimation (1b/1c in CURSOR_OPTIMIZATIONS.md) ──

/// Mandelbrot kernel tracking dz/dc alongside z (recurrence dz_{n+1} = 2·z_n·dz_n + 1,
/// ~4 extra FLOPs/iter). Returns (smooth iteration value, exterior distance estimate
/// d ≈ |z|·ln|z| / |dz| on escape; 0.0 for interior pixels, where the estimate is
/// meaningless). No bulb/period early-outs — callers wanting them should gate first.
pub fn mandelbrot_dem(cr: f64, ci: f64, max_iter: u32) -> (f32, f64) {
    let mut zr = 0.0f64;
    let mut zi = 0.0f64;
    let mut dzr = 0.0f64;
    let mut dzi = 0.0f64;

    for i in 0..max_iter {
        let zn_sq = zr * zr + zi * zi;
        if zn_sq > ESCAPE_RADIUS_SQ {
            let z_mag = zn_sq.sqrt();
            let dz_mag = (dzr * dzr + dzi * dzi).sqrt();
            let d = if dz_mag > 0.0 { z_mag * z_mag.ln() / dz_mag } else { 0.0 };
            return (smooth_iter(i, zn_sq, max_iter), d);
        }
        // dz ← 2·z·dz + 1 (must use pre-update z)
        let new_dzr = 2.0 * (zr * dzr - zi * dzi) + 1.0;
        let new_dzi = 2.0 * (zr * dzi + zi * dzr);
        dzr = new_dzr;
        dzi = new_dzi;
        let new_zr = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = new_zr;
    }
    (max_iter as f32, 0.0)
}

/// DEM-culled Mariani-Silver (Mandelbrot only; other fractals fall back to the
/// exact `render_mariani_silver`). Adds one extra fast path to the subdivision:
/// when a rectangle is not border-uniform but all four corners' exterior distance
/// estimates exceed `k ×` the rectangle diagonal (the distance function is
/// 1-Lipschitz, so with k ≥ 1 the set boundary cannot enter the rectangle) AND the
/// corner smooth values agree within a small tolerance (guards against banding),
/// the interior is filled by bilinear interpolation of the corner values instead
/// of per-pixel iteration. This interpolated fill is an APPROXIMATION — validate
/// via `--compare-dem-cull`, tune `k` down only as the diff gate allows.
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
    let [x0, y0, tw, th] = tile;
    let mut local = vec![0.0f32; (tw * th) as usize];
    for ly in 0..th {
        let im = pg.im_start + (y0 + ly) as f64 * pg.im_step;
        for lx in 0..tw {
            let re = pg.re_start + (x0 + lx) as f64 * pg.re_step;
            local[(ly * tw + lx) as usize] = compute(fractal, re, im, julia_c, max_iter);
        }
    }
    local
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

/// Reference orbit for perturbation theory.
///
/// Stores the Mandelbrot orbit of the reference point C (the viewport center):
/// Z_0=0, Z_1, ..., Z_len, where `len` iterations were computed before escape
/// (or `len == max_iter` if the reference never escaped).
/// The arrays hold `len + 1` valid entries indexed 0..=len.
pub struct RefOrbit {
    pub zr: Vec<f64>,
    pub zi: Vec<f64>,
    pub len: usize,
}

/// Compute the Mandelbrot reference orbit for C=(cr,ci) up to max_iter steps.
pub fn compute_reference_orbit(cr: f64, ci: f64, max_iter: u32) -> RefOrbit {
    let n = max_iter as usize;
    let mut zr = vec![0.0f64; n + 1];
    let mut zi = vec![0.0f64; n + 1];
    for i in 0..n {
        let r2 = zr[i] * zr[i];
        let i2 = zi[i] * zi[i];
        if r2 + i2 > ESCAPE_RADIUS_SQ {
            return RefOrbit { zr, zi, len: i };
        }
        zr[i + 1] = r2 - i2 + cr;
        zi[i + 1] = 2.0 * zr[i] * zi[i] + ci;
    }
    RefOrbit { zr, zi, len: n }
}

/// Zoom threshold above which the reference orbit is computed with double-double precision.
///
/// At zoom ~10^13 f64 runs out of mantissa bits and the orbit drifts, producing
/// block artifacts. Double-double (~31 decimal digits) extends clean rendering to
/// ~10^26 — equivalent to software f128 for this use case, on stable Rust.
/// The orbit is computed once per frame; even at 5–10× f64 cost it adds < 5 ms.
pub const F128_ZOOM_THRESHOLD: f64 = 1e12;

// ── Double-double arithmetic ──────────────────────────────────────────────────
//
// Each value is (hi + lo) where |lo| ≤ ½ ulp(hi). Gives ~31 significant decimal
// digits — sufficient to push the clean-rendering zoom limit from ~10^13 to ~10^26.
// Uses FMA (f64::mul_add) for the error-free product, available on all x86-64.

#[derive(Clone, Copy)]
struct Dd(f64, f64); // (hi, lo)

impl Dd {
    #[inline] fn from_f64(x: f64) -> Self { Dd(x, 0.0) }
    #[inline] fn hi(self) -> f64 { self.0 }
}

#[inline]
fn two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    let v = s - a;
    Dd(s, (a - (s - v)) + (b - v))
}

#[inline]
fn two_prod(a: f64, b: f64) -> Dd {
    let p = a * b;
    Dd(p, a.mul_add(b, -p)) // FMA computes the rounding error exactly
}

impl std::ops::Add for Dd {
    type Output = Dd;
    fn add(self, b: Dd) -> Dd {
        let s = two_sum(self.0, b.0);
        let e = s.1 + (self.1 + b.1);
        two_sum(s.0, e)
    }
}

impl std::ops::Sub for Dd {
    type Output = Dd;
    fn sub(self, b: Dd) -> Dd { self + Dd(-b.0, -b.1) }
}

impl std::ops::Mul for Dd {
    type Output = Dd;
    fn mul(self, b: Dd) -> Dd {
        let p = two_prod(self.0, b.0);
        let e = p.1 + self.0 * b.1 + self.1 * b.0;
        two_sum(p.0, e)
    }
}

impl std::ops::Mul<Dd> for f64 {
    type Output = Dd;
    fn mul(self, b: Dd) -> Dd { Dd::from_f64(self) * b }
}

impl PartialEq for Dd { fn eq(&self, o: &Dd) -> bool { self.0 == o.0 && self.1 == o.1 } }
impl PartialOrd for Dd {
    fn partial_cmp(&self, o: &Dd) -> Option<std::cmp::Ordering> {
        // hi parts determine order; lo breaks ties within the same ulp.
        match self.0.partial_cmp(&o.0) {
            Some(std::cmp::Ordering::Equal) => self.1.partial_cmp(&o.1),
            other => other,
        }
    }
}

/// Compute the reference orbit using double-double precision, stored as f64.
///
/// Iterates Z_{n+1} = Z_n² + C in double-double (~31 decimal digits), then
/// downcasts each orbit term to f64 for storage. The per-pixel delta loop stays
/// in f64 — deltas remain representable because they are small fractions of the
/// full coordinate. Equivalent to f128 for this application on stable Rust.
pub fn compute_reference_orbit_f128(cr: f64, ci: f64, max_iter: u32) -> RefOrbit {
    let n = max_iter as usize;
    let mut zr_out = vec![0.0f64; n + 1];
    let mut zi_out = vec![0.0f64; n + 1];

    let cr_dd     = Dd::from_f64(cr);
    let ci_dd     = Dd::from_f64(ci);
    let escape_sq = Dd::from_f64(ESCAPE_RADIUS_SQ);

    // zr_out[0] = zi_out[0] = 0 already (Z_0 = 0).
    let mut zr = Dd::from_f64(0.0);
    let mut zi = Dd::from_f64(0.0);

    for i in 0..n {
        let r2 = zr * zr;
        let i2 = zi * zi;
        if r2 + i2 > escape_sq {
            return RefOrbit { zr: zr_out, zi: zi_out, len: i };
        }
        let new_zr = r2 - i2 + cr_dd;
        zi = 2.0 * zr * zi + ci_dd;
        zr = new_zr;
        zr_out[i + 1] = zr.hi();
        zi_out[i + 1] = zi.hi();
    }
    RefOrbit { zr: zr_out, zi: zi_out, len: n }
}

/// |ε|²/|Z|² ratio above which the linear approximation is no longer trusted.
/// Equivalent to |ε| > 1e-3 × |Z| (0.1 % of the reference magnitude).
const GLITCH_SQ: f64 = 1e-6;

/// Core perturbation recurrence, factored out of `perturb_mandelbrot` so it can be
/// reused by both the single-reference renderer and the multi-reference glitch
/// corrector (4a in CURSOR_OPTIMIZATIONS.md). Returns `None` on glitch (ε grew too
/// large relative to Z) or when the reference orbit escaped before this pixel did —
/// callers decide how to handle that (single-ref: scalar fallback immediately;
/// multi-ref: try another reference first).
///
/// Recurrence: ε_{n+1} = 2·Z_n·ε_n + ε_n² + δ  (δ = c − C, ε_0 = 0)
/// Escape check on z_{n+1} = Z_{n+1} + ε_{n+1}.
/// `pub` so `bench_runner` can directly count glitches for validation purposes.
#[inline]
pub fn perturb_mandelbrot_flagged(
    orbit: &RefOrbit,
    dc_re: f64, dc_im: f64, // δ = pixel c − reference C
    max_iter: u32,
) -> Option<f32> {
    let mut er = 0.0f64;
    let mut ei = 0.0f64;

    for n in 0..orbit.len {
        let zr = orbit.zr[n];
        let zi = orbit.zi[n];

        // ε_{n+1} = 2·Z_n·ε_n + ε_n² + δ
        let two_zr = 2.0 * zr;
        let two_zi = 2.0 * zi;
        let new_er = two_zr * er - two_zi * ei + (er * er - ei * ei) + dc_re;
        let new_ei = two_zr * ei + two_zi * er + (2.0 * er * ei)     + dc_im;
        er = new_er;
        ei = new_ei;

        // z_{n+1} ≈ Z_{n+1} + ε_{n+1}
        let az = orbit.zr[n + 1] + er;
        let bz = orbit.zi[n + 1] + ei;
        let zn_sq = az * az + bz * bz;

        if zn_sq > ESCAPE_RADIUS_SQ {
            return Some(smooth_iter(n as u32 + 1, zn_sq, max_iter));
        }

        // Glitch: ε has grown too large relative to Z — approximation unreliable.
        let ref_sq = orbit.zr[n + 1] * orbit.zr[n + 1] + orbit.zi[n + 1] * orbit.zi[n + 1];
        if er * er + ei * ei > ref_sq * GLITCH_SQ {
            return None;
        }
    }

    if orbit.len >= max_iter as usize {
        Some(max_iter as f32)
    } else {
        // Reference escaped before this pixel did.
        None
    }
}

/// Approximate one Mandelbrot pixel via perturbation theory.
/// Falls back to the exact scalar kernel on glitch or when the reference orbit ends early.
#[inline]
fn perturb_mandelbrot(
    orbit: &RefOrbit,
    dc_re: f64, dc_im: f64,   // δ = pixel c − reference C
    full_re: f64, full_im: f64, // actual pixel coordinate for fallback
    max_iter: u32,
) -> f32 {
    perturb_mandelbrot_flagged(orbit, dc_re, dc_im, max_iter)
        .unwrap_or_else(|| mandelbrot(full_re, full_im, max_iter))
}

/// Perturbation with mathr/Zhuoran-style rebasing: the pixel's iteration count `n`
/// is decoupled from the reference-orbit index `m`. Whenever the full value
/// `z = Z_m + ε` comes closer to 0 than ε itself (the "flip" — the point where
/// catastrophic cancellation would start corrupting ε), fold z into ε and restart
/// the orbit index at 0 (Z_0 = 0, so ε := z exactly). The same rebase handles the
/// reference orbit escaping early (`m == orbit.len`). ε therefore always stays
/// small relative to the orbit, no glitch criterion or scalar fallback is needed,
/// and the whole pixel stays inside perturbation against the one shared orbit.
/// See 4b in CURSOR_OPTIMIZATIONS.md.
#[inline]
fn perturb_mandelbrot_rebase(
    orbit: &RefOrbit,
    dc_re: f64, dc_im: f64,
    max_iter: u32,
) -> f32 {
    let mut er = 0.0f64;
    let mut ei = 0.0f64;
    let mut m = 0usize; // reference-orbit index, resets to 0 on rebase

    for n in 0..max_iter {
        let zr = orbit.zr[m];
        let zi = orbit.zi[m];

        // ε ← 2·Z_m·ε + ε² + δ
        let two_zr = 2.0 * zr;
        let two_zi = 2.0 * zi;
        let new_er = two_zr * er - two_zi * ei + (er * er - ei * ei) + dc_re;
        let new_ei = two_zr * ei + two_zi * er + (2.0 * er * ei)     + dc_im;
        er = new_er;
        ei = new_ei;
        m += 1;

        let az = orbit.zr[m] + er;
        let bz = orbit.zi[m] + ei;
        let zn_sq = az * az + bz * bz;

        if zn_sq > ESCAPE_RADIUS_SQ {
            return smooth_iter(n + 1, zn_sq, max_iter);
        }

        // Flip detection / rebase: z closer to 0 than ε, or orbit data exhausted.
        if zn_sq < er * er + ei * ei || m == orbit.len {
            er = az;
            ei = bz;
            m = 0;
        }
    }
    max_iter as f32
}

/// Render Mandelbrot using perturbation theory with early rebase-on-drift instead of
/// full-restart glitch fallback (4b in CURSOR_OPTIMIZATIONS.md). See
/// `perturb_mandelbrot_rebase` for the scoped-down interpretation of "rebase" used
/// here — a cheaper glitch recovery, not true per-pixel local-reference rebasing.
pub fn render_perturbation_rebase(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    if fractal != FractalType::Mandelbrot {
        return render(vp, fractal, julia_c, max_iter);
    }

    let w = vp.width as usize;
    let h = vp.height as usize;

    let pg = pixel_grid(vp);
    let center_re = vp.center[0];
    let center_im = vp.center[1];

    let orbit = if vp.zoom > F128_ZOOM_THRESHOLD {
        compute_reference_orbit_f128(center_re, center_im, max_iter)
    } else {
        compute_reference_orbit(center_re, center_im, max_iter)
    };

    let mut buf = vec![0.0f32; w * h];
    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im    = pg.im_start + y as f64 * pg.im_step;
        let dc_im = im - center_im;
        for x in 0..w {
            let re    = pg.re_start + x as f64 * pg.re_step;
            let dc_re = re - center_re;
            row[x] = perturb_mandelbrot_rebase(&orbit, dc_re, dc_im, max_iter);
        }
    });

    buf
}

/// Render Mandelbrot using perturbation theory (all other fractals fall back to scalar).
///
/// Computes one reference orbit at the viewport center, then derives every pixel as a
/// perturbation ε around that orbit.  Glitched pixels fall back to the full scalar kernel.
pub fn render_perturbation(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    if fractal != FractalType::Mandelbrot {
        return render(vp, fractal, julia_c, max_iter);
    }

    let w = vp.width as usize;
    let h = vp.height as usize;

    let pg = pixel_grid(vp);
    let center_re = vp.center[0];
    let center_im = vp.center[1];

    let orbit = if vp.zoom > F128_ZOOM_THRESHOLD {
        compute_reference_orbit_f128(center_re, center_im, max_iter)
    } else {
        compute_reference_orbit(center_re, center_im, max_iter)
    };

    let mut buf = vec![0.0f32; w * h];
    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im    = pg.im_start + y as f64 * pg.im_step;
        let dc_im = im - center_im;
        for x in 0..w {
            let re    = pg.re_start + x as f64 * pg.re_step;
            let dc_re = re - center_re;
            row[x] = perturb_mandelbrot(&orbit, dc_re, dc_im, re, im, max_iter);
        }
    });

    buf
}

/// Maximum number of secondary reference orbits `render_perturbation_multiref`
/// will compute before giving up and falling back to the exact scalar kernel for
/// any pixels still glitched.
pub const MAX_REFS: usize = 8;

/// A center point and its precomputed reference orbit — used by
/// `render_perturbation_multiref` to track secondary references added to recover
/// glitched pixels. See 4a in CURSOR_OPTIMIZATIONS.md.
pub struct RefOrbitSet {
    pub refs: Vec<RefOrbit>,
    pub centers: Vec<(f64, f64)>,
}

/// Render Mandelbrot using perturbation theory with automatic multi-reference
/// glitch correction: pixels that glitch against the primary reference orbit
/// (centered at the viewport) are retried against additional reference orbits
/// seeded at glitch locations, up to `MAX_REFS`. Remaining glitches after that cap
/// fall back to the exact scalar kernel, guaranteeing termination.
///
/// v1 scope (deliberately simpler than a full nearest-reference implementation):
/// - References are grown sequentially, always adding a new one for whatever is
///   still glitched — no per-pixel nearest-reference search across existing refs.
/// - Does not use series approximation (SA) for corrected pixels; combine with SA
///   is deferred (see CURSOR_OPTIMIZATIONS.md 4a note on interaction with 4c).
pub fn render_perturbation_multiref(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    if fractal != FractalType::Mandelbrot {
        return render(vp, fractal, julia_c, max_iter);
    }

    let w = vp.width as usize;
    let h = vp.height as usize;
    let pg = pixel_grid(vp);

    let make_orbit = |cr: f64, ci: f64| -> RefOrbit {
        if vp.zoom > F128_ZOOM_THRESHOLD {
            compute_reference_orbit_f128(cr, ci, max_iter)
        } else {
            compute_reference_orbit(cr, ci, max_iter)
        }
    };

    let mut ref_set = RefOrbitSet { refs: vec![], centers: vec![] };
    ref_set.refs.push(make_orbit(vp.center[0], vp.center[1]));
    ref_set.centers.push((vp.center[0], vp.center[1]));

    let mut buf = vec![0.0f32; w * h];

    // Pass 1: primary reference. Glitched pixels are sentinel-marked with NaN
    // (reusing the same sentinel convention render_mariani_silver already uses).
    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im = pg.im_start + y as f64 * pg.im_step;
        let (cx, cy) = ref_set.centers[0];
        let dc_im = im - cy;
        for x in 0..w {
            let re = pg.re_start + x as f64 * pg.re_step;
            let dc_re = re - cx;
            row[x] = perturb_mandelbrot_flagged(&ref_set.refs[0], dc_re, dc_im, max_iter)
                .unwrap_or(f32::NAN);
        }
    });

    // Sequential rescan + secondary-reference retry loop.
    loop {
        let mut first_glitch: Option<(usize, usize)> = None;
        let mut any_glitch = false;
        for y in 0..h {
            for x in 0..w {
                if buf[y * w + x].is_nan() {
                    any_glitch = true;
                    if first_glitch.is_none() {
                        first_glitch = Some((x, y));
                    }
                }
            }
        }
        if !any_glitch {
            break;
        }
        if ref_set.refs.len() >= MAX_REFS {
            break;
        }

        let (gx, gy) = first_glitch.unwrap();
        let g_re = pg.re_start + gx as f64 * pg.re_step;
        let g_im = pg.im_start + gy as f64 * pg.im_step;
        let new_orbit = make_orbit(g_re, g_im);
        ref_set.refs.push(new_orbit);
        ref_set.centers.push((g_re, g_im));
        let ref_idx = ref_set.refs.len() - 1;
        let (cx, cy) = ref_set.centers[ref_idx];
        let orbit = &ref_set.refs[ref_idx];

        buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let im = pg.im_start + y as f64 * pg.im_step;
            let dc_im = im - cy;
            for x in 0..w {
                if !row[x].is_nan() {
                    continue;
                }
                let re = pg.re_start + x as f64 * pg.re_step;
                let dc_re = re - cx;
                if let Some(v) = perturb_mandelbrot_flagged(orbit, dc_re, dc_im, max_iter) {
                    row[x] = v;
                }
            }
        });
    }

    // Any pixels still glitched after MAX_REFS: exact scalar fallback (terminates).
    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im = pg.im_start + y as f64 * pg.im_step;
        for x in 0..w {
            if row[x].is_nan() {
                let re = pg.re_start + x as f64 * pg.re_step;
                row[x] = mandelbrot(re, im, max_iter);
            }
        }
    });

    buf
}

// ── Series Approximation (SA) ─────────────────────────────────────────────────
//
// Precomputes four complex power-series coefficients (A, B, C, D) along the
// reference orbit so that, for any pixel offset δ = c − C:
//
//   ε_n ≈ A_n·δ + B_n·δ² + C_n·δ³ + D_n·δ⁴
//
// Recurrences (derived by expanding the perturbation recurrence order by order —
// each right-hand term is the matching-order coefficient of the Cauchy product ε²):
//
//   A_{n+1} = 2·Z_n·A_n + 1
//   B_{n+1} = 2·Z_n·B_n + A_n²
//   C_{n+1} = 2·Z_n·C_n + 2·A_n·B_n
//   D_{n+1} = 2·Z_n·D_n + 2·A_n·C_n + B_n²
//
// We advance until the highest-order term exceeds SA_THRESHOLD × the linear term for the
// worst-case (corner) pixel.  That iteration becomes the "skip" — all pixels
// jump directly to ε_skip via a cheap polynomial evaluation and then continue
// with the ordinary perturbation loop.

/// SA coefficients at the skip point, plus the skip count itself.
pub struct SeriesApprox {
    /// Iterations that can be skipped for the entire frame.
    pub skip: usize,
    /// A coefficient (complex): linear term.
    pub ar: f64, pub ai: f64,
    /// B coefficient (complex): quadratic term.
    pub br: f64, pub bi: f64,
    /// C coefficient (complex): cubic term.
    pub cr: f64, pub ci: f64,
    /// D coefficient (complex): quartic term (4c in CURSOR_OPTIMIZATIONS.md).
    pub dr: f64, pub di: f64,
}

/// max(|C·δ³|, |D·δ⁴|) / |A·δ| ratio above which the SA polynomial is no longer trusted.
const SA_THRESHOLD: f64 = 1e-6;

/// Walk the reference orbit accumulating SA coefficients; return the largest
/// safe skip and the coefficient values at that point.
///
/// `delta_max_sq` is |δ_corner|² — the squared distance from center to the
/// corner pixel.  Use it as a conservative bound on all pixel offsets.
pub fn compute_series_approx(orbit: &RefOrbit, delta_max_sq: f64) -> SeriesApprox {
    let (mut ar, mut ai) = (0.0f64, 0.0f64);
    let (mut br, mut bi) = (0.0f64, 0.0f64);
    let (mut cr, mut ci) = (0.0f64, 0.0f64);
    let (mut dr, mut di) = (0.0f64, 0.0f64);
    let mut skip = 0usize;

    for n in 0..orbit.len {
        let zr = orbit.zr[n];
        let zi = orbit.zi[n];
        let two_zr = 2.0 * zr;
        let two_zi = 2.0 * zi;

        // A_n² and 2·A_n·B_n in complex arithmetic (needed for B and C updates).
        let a_sq_r   = ar * ar - ai * ai;
        let a_sq_i   = 2.0 * ar * ai;
        let two_ab_r = 2.0 * (ar * br - ai * bi);
        let two_ab_i = 2.0 * (ar * bi + ai * br);
        // 2·A_n·C_n + B_n² (the δ⁴ coefficient of ε²)
        let d_src_r = 2.0 * (ar * cr - ai * ci) + (br * br - bi * bi);
        let d_src_i = 2.0 * (ar * ci + ai * cr) + 2.0 * br * bi;

        // A_{n+1} = 2·Z_n·A_n + 1
        let new_ar = two_zr * ar - two_zi * ai + 1.0;
        let new_ai = two_zr * ai + two_zi * ar;
        // B_{n+1} = 2·Z_n·B_n + A_n²
        let new_br = two_zr * br - two_zi * bi + a_sq_r;
        let new_bi = two_zr * bi + two_zi * br + a_sq_i;
        // C_{n+1} = 2·Z_n·C_n + 2·A_n·B_n
        let new_cr = two_zr * cr - two_zi * ci + two_ab_r;
        let new_ci = two_zr * ci + two_zi * cr + two_ab_i;
        // D_{n+1} = 2·Z_n·D_n + 2·A_n·C_n + B_n²
        let new_dr = two_zr * dr - two_zi * di + d_src_r;
        let new_di = two_zr * di + two_zi * dr + d_src_i;

        ar = new_ar; ai = new_ai;
        br = new_br; bi = new_bi;
        cr = new_cr; ci = new_ci;
        dr = new_dr; di = new_di;

        // Accuracy guard, on the worst of the two highest-order terms:
        // stop when |C·δ³| or |D·δ⁴| ≥ SA_THRESHOLD × |A·δ| for the corner pixel.
        // Squared, ÷δ²: |C|²·δ⁴ or |D|²·δ⁶ ≥ SA_THRESHOLD²·|A|²
        let a_mag_sq = ar * ar + ai * ai;
        let c_mag_sq = cr * cr + ci * ci;
        let d_mag_sq = dr * dr + di * di;
        let d2 = delta_max_sq * delta_max_sq;
        let bound = SA_THRESHOLD * SA_THRESHOLD * a_mag_sq;
        if c_mag_sq * d2 > bound || d_mag_sq * d2 * delta_max_sq > bound {
            break;
        }
        skip = n + 1;
    }

    SeriesApprox { skip, ar, ai, br, bi, cr, ci, dr, di }
}

/// Evaluate the SA polynomial and run perturbation from the skip point.
#[inline]
fn perturb_mandelbrot_sa(
    orbit: &RefOrbit,
    sa:    &SeriesApprox,
    dc_re: f64, dc_im: f64,
    full_re: f64, full_im: f64,
    max_iter: u32,
) -> f32 {
    // δ², δ³ (complex powers of the pixel offset).
    let d2r = dc_re * dc_re - dc_im * dc_im;
    let d2i = 2.0 * dc_re * dc_im;
    let d3r = dc_re * d2r - dc_im * d2i;
    let d3i = dc_re * d2i + dc_im * d2r;
    let d4r = d2r * d2r - d2i * d2i;
    let d4i = 2.0 * d2r * d2i;

    // ε_skip = A·δ + B·δ² + C·δ³ + D·δ⁴  (complex multiplications).
    let mut er = sa.ar * dc_re - sa.ai * dc_im
               + sa.br * d2r   - sa.bi * d2i
               + sa.cr * d3r   - sa.ci * d3i
               + sa.dr * d4r   - sa.di * d4i;
    let mut ei = sa.ar * dc_im + sa.ai * dc_re
               + sa.br * d2i   + sa.bi * d2r
               + sa.cr * d3i   + sa.ci * d3r
               + sa.dr * d4i   + sa.di * d4r;

    // Continue with the standard perturbation loop from iteration `skip`.
    for n in sa.skip..orbit.len {
        let zr = orbit.zr[n];
        let zi = orbit.zi[n];
        let two_zr = 2.0 * zr;
        let two_zi = 2.0 * zi;
        let new_er = two_zr * er - two_zi * ei + (er * er - ei * ei) + dc_re;
        let new_ei = two_zr * ei + two_zi * er + (2.0 * er * ei)     + dc_im;
        er = new_er;
        ei = new_ei;

        let az = orbit.zr[n + 1] + er;
        let bz = orbit.zi[n + 1] + ei;
        let zn_sq = az * az + bz * bz;
        if zn_sq > ESCAPE_RADIUS_SQ {
            return smooth_iter(n as u32 + 1, zn_sq, max_iter);
        }

        let ref_sq = orbit.zr[n + 1] * orbit.zr[n + 1] + orbit.zi[n + 1] * orbit.zi[n + 1];
        if er * er + ei * ei > ref_sq * GLITCH_SQ {
            return mandelbrot(full_re, full_im, max_iter);
        }
    }

    if orbit.len >= max_iter as usize { max_iter as f32 } else { mandelbrot(full_re, full_im, max_iter) }
}

/// Render Mandelbrot with perturbation theory + series approximation.
///
/// Builds the reference orbit and SA coefficients once per frame, then for each
/// pixel evaluates a 3-term polynomial to skip the first `sa.skip` iterations
/// and continues with the perturbation recurrence from there.
pub fn render_perturbation_sa(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    if fractal != FractalType::Mandelbrot {
        return render(vp, fractal, julia_c, max_iter);
    }

    let w = vp.width  as usize;
    let h = vp.height as usize;

    let pg = pixel_grid(vp);
    let center_re = vp.center[0];
    let center_im = vp.center[1];

    // Conservative corner-pixel |δ|² used for the SA validity bound.
    let half = 2.0 / vp.zoom;
    let aspect = vp.width as f64 / vp.height as f64;
    let delta_max_sq = (half * aspect) * (half * aspect) + half * half;

    let orbit = if vp.zoom > F128_ZOOM_THRESHOLD {
        compute_reference_orbit_f128(center_re, center_im, max_iter)
    } else {
        compute_reference_orbit(center_re, center_im, max_iter)
    };
    let sa = compute_series_approx(&orbit, delta_max_sq);

    let mut buf = vec![0.0f32; w * h];
    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im    = pg.im_start + y as f64 * pg.im_step;
        let dc_im = im - center_im;
        for x in 0..w {
            let re    = pg.re_start + x as f64 * pg.re_step;
            let dc_re = re - center_re;
            row[x] = perturb_mandelbrot_sa(&orbit, &sa, dc_re, dc_im, re, im, max_iter);
        }
    });

    buf
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

fn smooth_iter(iter: u32, zn_sq: f64, max_iter: u32) -> f32 {
    if iter >= max_iter {
        return max_iter as f32;
    }
    const INV_LN2: f64 = std::f64::consts::LOG2_E;
    let log_zn = zn_sq.ln() / 2.0;
    let nu = (log_zn * INV_LN2).ln() * INV_LN2;
    (iter as f64 + 1.0 - nu) as f32
}

/// Exact cardioid (period-1) and period-2 bulb membership test.
#[inline]
fn in_cardioid_or_period2(cr: f64, ci: f64) -> bool {
    let q = (cr - 0.25) * (cr - 0.25) + ci * ci;
    if q * (q + cr - 0.25) < 0.25 * ci * ci {
        return true;
    }
    (cr + 1.0) * (cr + 1.0) + ci * ci < 0.0625
}

/// Center (nucleus) and squared radius of the two largest period-3 bulbs attached
/// directly to the main cardioid (rotation numbers 1/3 and 2/3, symmetric about the
/// real axis). NOT an exact boundary — period-3 bulbs are not circles. These
/// constants were derived numerically (Newton's method on f_c^3(0)=0 for the exact
/// nucleus, then an empirical binary search for the largest circle around the
/// nucleus with zero escapes at 50,000 iterations over 768 sampled points, shrunk by
/// a further 20% safety margin) — NOT taken from a literature formula. A disk that is
/// too large would misclassify escaping boundary pixels; this one is deliberately
/// conservative. Validate with `--compare-bulb-reject` before trusting further.
const PERIOD3_CENTER_RE: f64 = -0.122561_1668766536;
const PERIOD3_CENTER_IM: f64 = 0.744861_7666197442;
const PERIOD3_RADIUS_SQ: f64 = 0.073714_84375 * 0.073714_84375;

/// Approximate test for the two largest period-3 bulbs (symmetric above/below the
/// real axis, attached left of the main cardioid). See constants doc comment above
/// for derivation and safety-margin details. `pub` so `bench_runner` can validate it
/// against a raw (no-early-out) ground truth via `--compare-bulb-reject`.
#[inline]
pub fn in_period3_bulb(cr: f64, ci: f64) -> bool {
    let dr = cr - PERIOD3_CENTER_RE;
    let di_pos = ci - PERIOD3_CENTER_IM;
    let di_neg = ci + PERIOD3_CENTER_IM;
    (dr * dr + di_pos * di_pos < PERIOD3_RADIUS_SQ) || (dr * dr + di_neg * di_neg < PERIOD3_RADIUS_SQ)
}

#[inline]
fn mandelbrot(cr: f64, ci: f64, max_iter: u32) -> f32 {
    if in_cardioid_or_period2(cr, ci) || in_period3_bulb(cr, ci) {
        return max_iter as f32;
    }

    let mut zr = cr;
    let mut zi = ci;
    if max_iter <= 1 { return max_iter as f32; }
    // iter 1: z = c² + c
    let zr2 = zr * zr;
    let zi2 = zi * zi;
    let new_zi = (2.0 * zr).mul_add(zi, ci);
    zr = zr2 - zi2 + cr;
    zi = new_zi;
    if max_iter <= 2 { return max_iter as f32; }

    let (mut zr_b, mut zi_b) = (0.0f64, 0.0f64);
    let mut period = 0u32;
    let mut check = 8u32;

    for i in 2..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        let zn_sq = zr2 + zi2;
        if zn_sq > ESCAPE_RADIUS_SQ {
            return smooth_iter(i, zn_sq, max_iter);
        }
        let new_zi = (2.0 * zr).mul_add(zi, ci);
        zr = zr2 - zi2 + cr;
        zi = new_zi;

        let dr = zr - zr_b;
        let di = zi - zi_b;
        if dr * dr + di * di < 1e-20 {
            return max_iter as f32;
        }
        period += 1;
        if period == check {
            period = 0;
            check = check.saturating_mul(2).min(512);
            zr_b = zr;
            zi_b = zi;
        }
    }
    max_iter as f32
}

#[inline]
fn julia(zr0: f64, zi0: f64, cr: f64, ci: f64, max_iter: u32) -> f32 {
    let (mut zr, mut zi) = (zr0, zi0);
    let (mut zr_b, mut zi_b) = (zr0, zi0);
    let mut period = 0u32;
    let mut check = 8u32;

    for i in 0..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        let zn_sq = zr2 + zi2;
        if zn_sq > ESCAPE_RADIUS_SQ {
            return smooth_iter(i, zn_sq, max_iter);
        }
        let new_zi = (2.0 * zr).mul_add(zi, ci);
        zr = zr2 - zi2 + cr;
        zi = new_zi;

        let dr = zr - zr_b;
        let di = zi - zi_b;
        if dr * dr + di * di < 1e-20 {
            return max_iter as f32;
        }
        period += 1;
        if period == check {
            period = 0;
            check = check.saturating_mul(2).min(512);
            zr_b = zr;
            zi_b = zi;
        }
    }
    max_iter as f32
}

#[inline]
fn newton(cr: f64, ci: f64, max_iter: u32) -> f32 {
    let (mut zr, mut zi) = (cr, ci);
    for i in 0..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        let z3r = zr * (zr2 - 3.0 * zi2);
        let z3i = zi * (3.0 * zr2 - zi2);
        let fr = z3r - 1.0;
        let fi = z3i;
        let d_re = 3.0 * (zr2 - zi2);
        let d_im = 6.0 * zr * zi;
        let denom = d_re * d_re + d_im * d_im;
        if denom < 1e-20 {
            break;
        }
        let new_zr = zr - (fr * d_re + fi * d_im) / denom;
        let new_zi = zi - (fi * d_re - fr * d_im) / denom;
        let dr = new_zr - zr;
        let di = new_zi - zi;
        zr = new_zr;
        zi = new_zi;
        if dr * dr + di * di < 1e-12 {
            return i as f32;
        }
    }
    max_iter as f32
}

#[inline]
fn nova(cr: f64, ci: f64, max_iter: u32) -> f32 {
    let (mut zr, mut zi) = (1.0, 0.0);
    for i in 0..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        let z3r = zr * (zr2 - 3.0 * zi2);
        let z3i = zi * (3.0 * zr2 - zi2);
        let fr = z3r - 1.0;
        let fi = z3i;
        let d_re = 3.0 * (zr2 - zi2);
        let d_im = 6.0 * zr * zi;
        let denom = d_re * d_re + d_im * d_im;
        if denom < 1e-20 {
            break;
        }
        let new_zr = zr - (fr * d_re + fi * d_im) / denom + cr;
        let new_zi = zi - (fi * d_re - fr * d_im) / denom + ci;
        let dr = new_zr - zr;
        let di = new_zi - zi;
        zr = new_zr;
        zi = new_zi;
        if dr * dr + di * di < 1e-12 {
            return i as f32;
        }
    }
    max_iter as f32
}

fn smooth_iter_f32(iter: u32, zn_sq: f32, max_iter: u32) -> f32 {
    if iter >= max_iter { return max_iter as f32; }
    const INV_LN2: f32 = std::f32::consts::LOG2_E;
    let log_zn = zn_sq.ln() / 2.0;
    let nu = (log_zn * INV_LN2).ln() * INV_LN2;
    iter as f32 + 1.0 - nu
}

fn mandelbrot_x8(cr: wide::f32x8, ci: wide::f32x8, max_iter: u32) -> [f32; 8] {
    use wide::{f32x8, CmpGt};

    let escape  = f32x8::splat(ESCAPE_RADIUS_SQ_F32);
    let two     = f32x8::splat(2.0f32);
    let all_one = f32x8::splat(f32::from_bits(u32::MAX));

    let in_set = bulb_precheck_x8(cr, ci);
    if in_set.iter().all(|&b| b) {
        return [max_iter as f32; 8];
    }

    let mut zr = cr;
    let mut zi = ci;
    if max_iter <= 1 { return [max_iter as f32; 8]; }
    {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        zi = (two * zr).mul_add(zi, ci);
        zr = zr2 - zi2 + cr;
    }
    if max_iter <= 2 { return [max_iter as f32; 8]; }

    let mut active = f32x8::from([
        f32::from_bits(if in_set[0] { 0 } else { u32::MAX }),
        f32::from_bits(if in_set[1] { 0 } else { u32::MAX }),
        f32::from_bits(if in_set[2] { 0 } else { u32::MAX }),
        f32::from_bits(if in_set[3] { 0 } else { u32::MAX }),
        f32::from_bits(if in_set[4] { 0 } else { u32::MAX }),
        f32::from_bits(if in_set[5] { 0 } else { u32::MAX }),
        f32::from_bits(if in_set[6] { 0 } else { u32::MAX }),
        f32::from_bits(if in_set[7] { 0 } else { u32::MAX }),
    ]);
    let mut escape_iter = [max_iter; 8];
    let mut escape_zn   = [0.0f32; 8];

    for i in 2..max_iter {
        let zr2   = zr * zr;
        let zi2   = zi * zi;
        let zn_sq = zr2 + zi2;

        let just_escaped = zn_sq.cmp_gt(escape) & active;
        if just_escaped.any() {
            let mask: [f32; 8] = just_escaped.into();
            let zn:   [f32; 8] = zn_sq.into();
            for lane in 0..8 {
                if mask[lane].to_bits() != 0 {
                    escape_iter[lane] = i;
                    escape_zn[lane]   = zn[lane];
                }
            }
            active = active & (all_one ^ just_escaped);
            if !active.any() { break; }
        }

        zi = (two * zr).mul_add(zi, ci);
        zr = zr2 - zi2 + cr;
    }

    let mut out = [0.0f32; 8];
    for lane in 0..8 {
        out[lane] = if escape_iter[lane] >= max_iter {
            max_iter as f32
        } else {
            smooth_iter_f32(escape_iter[lane], escape_zn[lane], max_iter)
        };
    }
    out
}

/// Per-lane bulb pre-check shared by `mandelbrot_x8` and `mandelbrot_x8x2` — factored
/// out so the two-batch ILP variant doesn't duplicate the scalar-per-lane logic.
#[inline]
/// Cardioid + period-2 bulb test done as true SIMD compares across all 8
/// lanes at once (mirroring how Fractals-rs's `mandelbrot_simd_f32` does the
/// same check — see BENCHMARKING.md), instead of the scalar per-lane loop
/// this used to be. Only the period-3 bulb check remains a scalar per-lane
/// call (it needs `in_period3_bulb`'s f64 arithmetic, and per investigation
/// isn't the bottleneck — this runs once per 8-pixel batch, not per
/// iteration, either way). Identical results to the previous all-scalar
/// version — verify via the scalar-vs-SIMD pixel-diff probe.
fn bulb_precheck_x8(cr: wide::f32x8, ci: wide::f32x8) -> [bool; 8] {
    use wide::{f32x8, CmpLt};

    let quarter = f32x8::splat(0.25);
    let ci_sq = ci * ci;
    let x_offset = cr - quarter;
    let q = x_offset.mul_add(x_offset, ci_sq);
    let cardioid_in = (q * (q + x_offset)).cmp_lt(quarter * ci_sq);
    let x_plus = cr + f32x8::splat(1.0);
    let bulb_in = x_plus.mul_add(x_plus, ci_sq).cmp_lt(f32x8::splat(0.0625));
    let vec_in_set: [f32; 8] = (cardioid_in | bulb_in).into();

    let cr_arr: [f32; 8] = cr.into();
    let ci_arr: [f32; 8] = ci.into();
    let mut in_set = [false; 8];
    for lane in 0..8 {
        if vec_in_set[lane].to_bits() != 0 {
            in_set[lane] = true;
            continue;
        }
        if in_period3_bulb(cr_arr[lane] as f64, ci_arr[lane] as f64) {
            in_set[lane] = true;
        }
    }
    in_set
}

/// Two independent `f32x8` Mandelbrot iteration chains interleaved within one loop
/// body, processing 16 pixels/iteration. Bit-identical output to running
/// `mandelbrot_x8` twice — only the instruction scheduling differs, which is the
/// entire point: interleaving two independent dependency chains lets the CPU
/// pipeline both, increasing instruction-level parallelism (2a in
/// CURSOR_OPTIMIZATIONS.md).
fn mandelbrot_x8x2(
    cr0: wide::f32x8, ci0: wide::f32x8,
    cr1: wide::f32x8, ci1: wide::f32x8,
    max_iter: u32,
) -> ([f32; 8], [f32; 8]) {
    use wide::{f32x8, CmpGt};

    let escape  = f32x8::splat(ESCAPE_RADIUS_SQ_F32);
    let two     = f32x8::splat(2.0f32);
    let all_one = f32x8::splat(f32::from_bits(u32::MAX));

    let in_set0 = bulb_precheck_x8(cr0, ci0);
    let in_set1 = bulb_precheck_x8(cr1, ci1);
    if in_set0.iter().all(|&b| b) && in_set1.iter().all(|&b| b) {
        return ([max_iter as f32; 8], [max_iter as f32; 8]);
    }

    let mut zr0 = cr0;
    let mut zi0 = ci0;
    let mut zr1 = cr1;
    let mut zi1 = ci1;
    if max_iter <= 1 {
        return ([max_iter as f32; 8], [max_iter as f32; 8]);
    }
    {
        let zr2_0 = zr0 * zr0; let zi2_0 = zi0 * zi0;
        zi0 = (two * zr0).mul_add(zi0, ci0);
        zr0 = zr2_0 - zi2_0 + cr0;
        let zr2_1 = zr1 * zr1; let zi2_1 = zi1 * zi1;
        zi1 = (two * zr1).mul_add(zi1, ci1);
        zr1 = zr2_1 - zi2_1 + cr1;
    }
    if max_iter <= 2 {
        return ([max_iter as f32; 8], [max_iter as f32; 8]);
    }

    let mk_active = |in_set: &[bool; 8]| -> f32x8 {
        f32x8::from(std::array::from_fn::<f32, 8, _>(|lane| {
            f32::from_bits(if in_set[lane] { 0 } else { u32::MAX })
        }))
    };
    let mut active0 = mk_active(&in_set0);
    let mut active1 = mk_active(&in_set1);
    let mut escape_iter0 = [max_iter; 8];
    let mut escape_iter1 = [max_iter; 8];
    let mut escape_zn0   = [0.0f32; 8];
    let mut escape_zn1   = [0.0f32; 8];

    for i in 2..max_iter {
        // batch 0
        let zr2_0   = zr0 * zr0;
        let zi2_0   = zi0 * zi0;
        let zn_sq_0 = zr2_0 + zi2_0;
        // batch 1 (interleaved so the CPU can pipeline both chains)
        let zr2_1   = zr1 * zr1;
        let zi2_1   = zi1 * zi1;
        let zn_sq_1 = zr2_1 + zi2_1;

        let just_escaped0 = zn_sq_0.cmp_gt(escape) & active0;
        if just_escaped0.any() {
            let mask: [f32; 8] = just_escaped0.into();
            let zn:   [f32; 8] = zn_sq_0.into();
            for lane in 0..8 {
                if mask[lane].to_bits() != 0 {
                    escape_iter0[lane] = i;
                    escape_zn0[lane]   = zn[lane];
                }
            }
            active0 = active0 & (all_one ^ just_escaped0);
        }
        let just_escaped1 = zn_sq_1.cmp_gt(escape) & active1;
        if just_escaped1.any() {
            let mask: [f32; 8] = just_escaped1.into();
            let zn:   [f32; 8] = zn_sq_1.into();
            for lane in 0..8 {
                if mask[lane].to_bits() != 0 {
                    escape_iter1[lane] = i;
                    escape_zn1[lane]   = zn[lane];
                }
            }
            active1 = active1 & (all_one ^ just_escaped1);
        }
        if !active0.any() && !active1.any() { break; }

        zi0 = (two * zr0).mul_add(zi0, ci0);
        zr0 = zr2_0 - zi2_0 + cr0;
        zi1 = (two * zr1).mul_add(zi1, ci1);
        zr1 = zr2_1 - zi2_1 + cr1;
    }

    let finish = |escape_iter: &[u32; 8], escape_zn: &[f32; 8]| -> [f32; 8] {
        std::array::from_fn(|lane| {
            if escape_iter[lane] >= max_iter {
                max_iter as f32
            } else {
                smooth_iter_f32(escape_iter[lane], escape_zn[lane], max_iter)
            }
        })
    };
    (finish(&escape_iter0, &escape_zn0), finish(&escape_iter1, &escape_zn1))
}

fn julia_x8(zr0: wide::f32x8, zi0: wide::f32x8, cr: wide::f32x8, ci: wide::f32x8, max_iter: u32) -> [f32; 8] {
    use wide::{f32x8, CmpGt};

    let escape  = f32x8::splat(ESCAPE_RADIUS_SQ_F32);
    let two     = f32x8::splat(2.0f32);
    let all_one = f32x8::splat(f32::from_bits(u32::MAX));
    let mut zr  = zr0;
    let mut zi  = zi0;

    let mut active = all_one;
    let mut escape_iter = [max_iter; 8];
    let mut escape_zn   = [0.0f32; 8];

    for i in 0..max_iter {
        let zr2   = zr * zr;
        let zi2   = zi * zi;
        let zn_sq = zr2 + zi2;

        let just_escaped = zn_sq.cmp_gt(escape) & active;
        if just_escaped.any() {
            let mask: [f32; 8] = just_escaped.into();
            let zn:   [f32; 8] = zn_sq.into();
            for lane in 0..8 {
                if mask[lane].to_bits() != 0 {
                    escape_iter[lane] = i;
                    escape_zn[lane]   = zn[lane];
                }
            }
            active = active & (all_one ^ just_escaped);
            if !active.any() { break; }
        }

        zi = (two * zr).mul_add(zi, ci);
        zr = zr2 - zi2 + cr;
    }

    let mut out = [0.0f32; 8];
    for lane in 0..8 {
        out[lane] = if escape_iter[lane] >= max_iter {
            max_iter as f32
        } else {
            smooth_iter_f32(escape_iter[lane], escape_zn[lane], max_iter)
        };
    }
    out
}

/// Cardioid + period-2 bulb test as true SIMD compares across all 4 lanes at
/// once — f64x4 counterpart of `bulb_precheck_x8`. Only period-3 stays a
/// scalar per-lane call. See `bulb_precheck_x8`'s doc comment.
fn bulb_precheck_x4(cr: wide::f64x4, ci: wide::f64x4) -> [bool; 4] {
    use wide::{f64x4, CmpLt};

    let quarter = f64x4::splat(0.25);
    let ci_sq = ci * ci;
    let x_offset = cr - quarter;
    let q = x_offset.mul_add(x_offset, ci_sq);
    let cardioid_in = (q * (q + x_offset)).cmp_lt(quarter * ci_sq);
    let x_plus = cr + f64x4::splat(1.0);
    let bulb_in = x_plus.mul_add(x_plus, ci_sq).cmp_lt(f64x4::splat(0.0625));
    let vec_in_set: [f64; 4] = (cardioid_in | bulb_in).into();

    let cr_arr: [f64; 4] = cr.into();
    let ci_arr: [f64; 4] = ci.into();
    let mut in_set = [false; 4];
    for lane in 0..4 {
        if vec_in_set[lane].to_bits() != 0 {
            in_set[lane] = true;
            continue;
        }
        if in_period3_bulb(cr_arr[lane], ci_arr[lane]) {
            in_set[lane] = true;
        }
    }
    in_set
}

fn mandelbrot_x4(cr: wide::f64x4, ci: wide::f64x4, max_iter: u32) -> [f32; 4] {
    use wide::{f64x4, CmpGt};

    let escape  = f64x4::splat(ESCAPE_RADIUS_SQ);
    let two     = f64x4::splat(2.0);
    let all_one = f64x4::splat(f64::from_bits(u64::MAX));

    let in_set = bulb_precheck_x4(cr, ci);
    if in_set.iter().all(|&b| b) {
        return [max_iter as f32; 4];
    }

    let mut zr = cr;
    let mut zi = ci;
    if max_iter <= 1 {
        return [max_iter as f32; 4];
    }
    {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        zi = (two * zr).mul_add(zi, ci);
        zr = zr2 - zi2 + cr;
    }
    if max_iter <= 2 {
        return [max_iter as f32; 4];
    }

    // Exclude already-classified in-set lanes from the SIMD loop.
    let mut active = f64x4::from([
        f64::from_bits(if in_set[0] { 0 } else { u64::MAX }),
        f64::from_bits(if in_set[1] { 0 } else { u64::MAX }),
        f64::from_bits(if in_set[2] { 0 } else { u64::MAX }),
        f64::from_bits(if in_set[3] { 0 } else { u64::MAX }),
    ]);
    let mut escape_iter = [max_iter; 4];
    let mut escape_zn   = [0.0f64; 4];

    for i in 2..max_iter {
        let zr2   = zr * zr;
        let zi2   = zi * zi;
        let zn_sq = zr2 + zi2;

        let just_escaped = zn_sq.cmp_gt(escape) & active;
        if just_escaped.any() {
            let mask: [f64; 4] = just_escaped.into();
            let zn:   [f64; 4] = zn_sq.into();
            for lane in 0..4 {
                if mask[lane].to_bits() != 0 {
                    escape_iter[lane] = i;
                    escape_zn[lane]   = zn[lane];
                }
            }
            active = active & (all_one ^ just_escaped);
            if !active.any() { break; }
        }

        zi = (two * zr).mul_add(zi, ci);
        zr = zr2 - zi2 + cr;
    }

    let mut out = [0.0f32; 4];
    for lane in 0..4 {
        out[lane] = if escape_iter[lane] >= max_iter {
            max_iter as f32
        } else {
            smooth_iter(escape_iter[lane], escape_zn[lane], max_iter)
        };
    }
    out
}

fn julia_x4(zr0: wide::f64x4, zi0: wide::f64x4, cr: wide::f64x4, ci: wide::f64x4, max_iter: u32) -> [f32; 4] {
    use wide::{f64x4, CmpGt};

    let escape  = f64x4::splat(ESCAPE_RADIUS_SQ);
    let two     = f64x4::splat(2.0);
    let all_one = f64x4::splat(f64::from_bits(u64::MAX));
    let mut zr  = zr0;
    let mut zi  = zi0;

    let mut active = all_one;
    let mut escape_iter = [max_iter; 4];
    let mut escape_zn   = [0.0f64; 4];

    for i in 0..max_iter {
        let zr2   = zr * zr;
        let zi2   = zi * zi;
        let zn_sq = zr2 + zi2;

        let just_escaped = zn_sq.cmp_gt(escape) & active;
        if just_escaped.any() {
            let mask: [f64; 4] = just_escaped.into();
            let zn:   [f64; 4] = zn_sq.into();
            for lane in 0..4 {
                if mask[lane].to_bits() != 0 {
                    escape_iter[lane] = i;
                    escape_zn[lane]   = zn[lane];
                }
            }
            active = active & (all_one ^ just_escaped);
            if !active.any() { break; }
        }

        zi = (two * zr).mul_add(zi, ci);
        zr = zr2 - zi2 + cr;
    }

    let mut out = [0.0f32; 4];
    for lane in 0..4 {
        out[lane] = if escape_iter[lane] >= max_iter {
            max_iter as f32
        } else {
            smooth_iter(escape_iter[lane], escape_zn[lane], max_iter)
        };
    }
    out
}