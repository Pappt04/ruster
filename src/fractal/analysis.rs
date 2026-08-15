//! Post-render analysis of a rendered [`IterBuf`]: fractal dimension
//! estimation, in-set area estimation, and iteration-count histograms.
//! These operate on the finished escape-time buffer and are independent of
//! which backend produced it.

use crate::fractal::IterBuf;

/// Result of a box-counting dimension estimate: the raw `(box size,
/// boundary-box count)` pairs, the fitted dimension, and the fit's
/// R^2 as a quality-of-fit indicator.
pub struct BoxCountResult {
    pub counts: Vec<(usize, u64)>,
    pub dimension: f64,
    pub r_squared: f64,
}

/// Estimates the fractal (box-counting/Minkowski) dimension of the set
/// boundary visible in `buf`, using the standard method: at each box size
/// in `box_sizes`, tile the image into boxes of that size and count how
/// many boxes contain both interior (`>= max_iter`) and exterior pixels —
/// i.e. how many boxes the boundary actually passes through. The
/// dimension is the slope of `ln(count)` against `ln(1/size)`; for a true
/// fractal boundary this relationship is linear, so a linear regression
/// over box sizes recovers the exponent in `count ~ size^-D`.
pub fn box_count_dimension(buf: &IterBuf, w: usize, h: usize, max_iter: u32, box_sizes: &[usize]) -> BoxCountResult {
    let max_f = max_iter as f32;
    let mut counts = Vec::with_capacity(box_sizes.len());

    for &size in box_sizes {
        let mut occupied = 0u64;
        let mut by = 0;
        while by < h {
            let mut bx = 0;
            while bx < w {
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

    let points: Vec<(f64, f64)> = counts.iter()
        .filter(|&&(_, c)| c > 0)
        .map(|&(s, c)| ((1.0 / s as f64).ln(), (c as f64).ln()))
        .collect();

    let (dimension, r_squared) = linear_fit(&points);

    BoxCountResult { counts, dimension, r_squared }
}

/// Ordinary least-squares fit of `y = slope * x + intercept` (intercept
/// unused by the caller), returning `(slope, r_squared)`. `r_squared` is
/// the standard coefficient of determination,
/// `(sum_xy)^2 / (sum_xx * sum_yy)` in mean-centered form.
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

/// Monte Carlo-style area estimate: the fraction of sampled pixels that
/// never escaped (`>= max_iter`) times the total area the viewport covers
/// in the complex plane. Accuracy is bounded by pixel resolution, not
/// sample count — this is a deterministic full-image sweep, not a random
/// sampling scheme, so its error comes from boundary pixels being
/// misclassified at the current zoom level rather than from statistical
/// variance.
pub fn estimate_area(buf: &IterBuf, max_iter: u32, viewport_area: f64) -> f64 {
    let max_f = max_iter as f32;
    let in_set = buf.iter().filter(|&&v| v >= max_f).count();
    (in_set as f64 / buf.len() as f64) * viewport_area
}

/// Bins every pixel's (possibly fractional) iteration count into `bins`
/// equal-width buckets spanning `[0, max_iter]`. This is the raw
/// per-value histogram [`crate::gui::color::colorize`] accumulates into a
/// CDF for histogram-equalized coloring; kept separate here since it is
/// also useful standalone for analysis/diagnostics.
pub fn iteration_histogram(buf: &IterBuf, max_iter: u32, bins: usize) -> Vec<u64> {
    let mut hist = vec![0u64; bins];
    let scale = bins as f32 / (max_iter as f32 + 1.0);
    for &v in buf.iter() {
        let idx = ((v * scale) as usize).min(bins - 1);
        hist[idx] += 1;
    }
    hist
}
