//! Fractal-geometry analysis over an already-rendered escape-time buffer
//! (`IterBuf`) — box-counting dimension, pixel-counting area estimate, and
//! the escape-iteration histogram they're both derived from. Kernel-agnostic:
//! takes a buffer produced by any `render*()` function, does no rendering
//! itself, so a single render can feed all three analyses without repeating
//! the (potentially expensive) fractal computation.

use crate::fractal::IterBuf;

pub struct BoxCountResult {
    /// (box_size_px, occupied_box_count) pairs, one per requested box size.
    pub counts: Vec<(usize, u64)>,
    /// Least-squares slope of log(count) vs log(1/box_size) — the box-counting
    /// dimension estimate. For the full Mandelbrot set boundary this should
    /// tend toward 2.0 (Shishikura's theorem) as box sizes shrink; small crops
    /// at finite resolution will underestimate this since box-counting needs
    /// scales spanning several orders of magnitude to converge.
    pub dimension: f64,
    /// Coefficient of determination of the log-log fit — how well the data
    /// follows a single power law over the tested range of box sizes.
    pub r_squared: f64,
}

/// Counts, for each `box_size` in `box_sizes`, how many non-overlapping
/// `box_size x box_size` boxes straddle the boundary — contain both an
/// "in set" pixel (`iter >= max_iter`) and an "escaped" pixel — then fits a
/// line to `log(count)` vs `log(1/box_size)`. Boxes that are uniformly
/// interior or uniformly exterior don't count: they cover the solid region
/// or empty space, not the boundary curve whose dimension this estimates.
pub fn box_count_dimension(buf: &IterBuf, w: usize, h: usize, max_iter: u32, box_sizes: &[usize]) -> BoxCountResult {
    let max_f = max_iter as f32;
    let mut counts = Vec::with_capacity(box_sizes.len());

    for &size in box_sizes {
        let mut occupied = 0u64;
        let mut by = 0;
        while by < h {
            let mut bx = 0;
            while bx < w {
                // A box counts toward the *boundary's* dimension only if it
                // straddles the boundary — i.e. contains both an "in set"
                // pixel (iter >= max_iter) and an "escaped" pixel. A box that
                // is uniformly interior or uniformly exterior is covering the
                // solid region/empty space, not the (dimension ~2, per
                // Shishikura) boundary curve itself.
                let mut saw_interior = false;
                let mut saw_exterior = false;
                'inner: for y in by..(by + size).min(h) {
                    for x in bx..(bx + size).min(w) {
                        if buf[y * w + x] >= max_f { saw_interior = true; } else { saw_exterior = true; }
                        if saw_interior && saw_exterior { break 'inner; }
                    }
                }
                if saw_interior && saw_exterior { occupied += 1; }
                bx += size;
            }
            by += size;
        }
        counts.push((size, occupied));
    }

    // Least-squares fit of ln(count) = dimension * ln(1/size) + intercept.
    let points: Vec<(f64, f64)> = counts.iter()
        .filter(|&&(_, c)| c > 0)
        .map(|&(s, c)| ((1.0 / s as f64).ln(), (c as f64).ln()))
        .collect();

    let (dimension, r_squared) = linear_fit(&points);

    BoxCountResult { counts, dimension, r_squared }
}

/// Ordinary least-squares slope + R² for a set of (x, y) points.
fn linear_fit(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    if points.len() < 2 { return (0.0, 0.0); }

    let sum_x: f64 = points.iter().map(|&(x, _)| x).sum();
    let sum_y: f64 = points.iter().map(|&(_, y)| y).sum();
    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    let mut ss_yy = 0.0;
    for &(x, y) in points {
        ss_xy += (x - mean_x) * (y - mean_y);
        ss_xx += (x - mean_x) * (x - mean_x);
        ss_yy += (y - mean_y) * (y - mean_y);
    }

    let slope = if ss_xx > 0.0 { ss_xy / ss_xx } else { 0.0 };
    let r_squared = if ss_xx > 0.0 && ss_yy > 0.0 { (ss_xy * ss_xy) / (ss_xx * ss_yy) } else { 0.0 };
    (slope, r_squared)
}

/// Pixel-counting area estimate: fraction of pixels classified "in set"
/// (`iter >= max_iter`) times the viewport's total complex-plane area.
/// Two independent error sources, both worth sweeping separately: resolution
/// (thin filaments near the boundary are under/over-counted depending on how
/// they land on the pixel grid) and `max_iter` (points that would eventually
/// escape past `max_iter` are counted as "in set", inflating the estimate —
/// this bias only shrinks as `max_iter` grows, never from finer resolution).
pub fn estimate_area(buf: &IterBuf, max_iter: u32, viewport_area: f64) -> f64 {
    let max_f = max_iter as f32;
    let in_set = buf.iter().filter(|&&v| v >= max_f).count();
    (in_set as f64 / buf.len() as f64) * viewport_area
}

/// Escape-iteration histogram over `bins` equal-width buckets spanning
/// `[0, max_iter]` (the last bin also catches in-set pixels, `iter == max_iter`).
pub fn iteration_histogram(buf: &IterBuf, max_iter: u32, bins: usize) -> Vec<u64> {
    let mut hist = vec![0u64; bins];
    let scale = bins as f32 / (max_iter as f32 + 1.0);
    for &v in buf.iter() {
        let idx = ((v * scale) as usize).min(bins - 1);
        hist[idx] += 1;
    }
    hist
}
