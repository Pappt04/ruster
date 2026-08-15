//! Mariani-Silver boundary tracing: recursively subdivides the frame into
//! rectangles, and whenever a rectangle's border pixels all share one
//! iteration count, fills its interior with that value instead of
//! computing it. This exploits the fact that connected level sets of the
//! escape-time function are large relative to a pixel — a rectangle whose
//! entire perimeter belongs to the same level set is, in the overwhelming
//! majority of cases, entirely inside it. It can occasionally miss a thin
//! filament that threads through a rectangle's interior without touching
//! its border, trading a rare, small rendering artifact for skipping the
//! computation of large uniform regions (deep set interior, or far
//! exterior where escape time barely changes).

use crate::fractal::fractal::{compute, pixel_grid, IterBuf, PixelGrid};
use crate::fractal::fractal_type::FractalType;
use crate::fractal::kernels::mandelbrot::mandelbrot_dem;
use crate::gui::viewport::Viewport;

/// Rectangles at or below this size are always computed pixel-by-pixel;
/// subdividing further would cost more in recursion overhead than it saves.
const MS_MIN: usize = 2;

pub fn render_mariani_silver(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    let w = vp.width as usize;
    let h = vp.height as usize;

    let pg = pixel_grid(vp);
    let mut buf = vec![f32::NAN; w * h];
    ms_fill(&mut buf, w, fractal, julia_c, max_iter, pg.re_start, pg.im_start, pg.re_step, pg.im_step, 0, 0, w, h);
    buf
}

/// Fills the rectangle `[x0, x1) x [y0, y1)` of `buf`, recursing per the
/// module-level algorithm description. `buf` is pre-seeded with `NAN` so a
/// pixel already computed by a sibling call sharing this border (adjacent
/// rectangles share an edge) is not recomputed — every write checks
/// `is_nan()` first.
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

    {
        let im = im_start + y0 as f64 * im_step;
        for x in x0..x1 {
            let idx = y0 * stride + x;
            if buf[idx].is_nan() {
                buf[idx] = compute(fractal, re_start + x as f64 * re_step, im, julia_c, max_iter);
            }
        }
    }
    {
        let im = im_start + (y1 - 1) as f64 * im_step;
        for x in x0..x1 {
            let idx = (y1 - 1) * stride + x;
            if buf[idx].is_nan() {
                buf[idx] = compute(fractal, re_start + x as f64 * re_step, im, julia_c, max_iter);
            }
        }
    }
    {
        let re = re_start + x0 as f64 * re_step;
        for y in (y0 + 1)..(y1 - 1) {
            let idx = y * stride + x0;
            if buf[idx].is_nan() {
                buf[idx] = compute(fractal, re, im_start + y as f64 * im_step, julia_c, max_iter);
            }
        }
    }
    {
        let re = re_start + (x1 - 1) as f64 * re_step;
        for y in (y0 + 1)..(y1 - 1) {
            let idx = y * stride + (x1 - 1);
            if buf[idx].is_nan() {
                buf[idx] = compute(fractal, re, im_start + y as f64 * im_step, julia_c, max_iter);
            }
        }
    }

    // Border fully computed above; if every border pixel agrees, assume the
    // interior does too and skip computing it.
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
    } else {
        // Split along the longer axis so recursion approaches a square
        // aspect ratio rather than degenerating into thin slivers.
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

/// Mariani-Silver variant for Mandelbrot that additionally culls rectangles
/// using the exterior distance estimate ([`mandelbrot_dem`]): a rectangle
/// whose four corners are all provably at least `k` diagonal-lengths away
/// from the set boundary cannot contain any boundary detail, so its
/// interior is filled by bilinear interpolation between the corner values
/// instead of either full computation or the exact-uniform-border test.
/// `k` trades accuracy for speed — smaller `k` culls more aggressively at
/// the risk of visibly flattening fine detail. Non-Mandelbrot fractals have
/// no distance estimator here and fall back to [`render_mariani_silver`].
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

    // A corner escaping within max_iter with distance estimate d means the
    // boundary is at least ~d away; requiring d >= k * diagonal for all
    // four corners bounds how much boundary detail this rectangle could
    // possibly contain. A corner that never escaped has no valid distance
    // estimate (v >= max_iter), so it always disqualifies the cull.
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
    if culled {
        // Extra sanity check: even with all corners individually "safe",
        // a large spread in their smooth iteration counts suggests the
        // interior is not as flat as the distance estimate implied, so
        // bilinear interpolation would misrepresent it.
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
