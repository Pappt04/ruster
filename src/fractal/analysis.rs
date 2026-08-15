
use crate::fractal::IterBuf;

pub struct BoxCountResult {
    pub counts: Vec<(usize, u64)>,
    pub dimension: f64,
    pub r_squared: f64,
}

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

pub fn estimate_area(buf: &IterBuf, max_iter: u32, viewport_area: f64) -> f64 {
    let max_f = max_iter as f32;
    let in_set = buf.iter().filter(|&&v| v >= max_f).count();
    (in_set as f64 / buf.len() as f64) * viewport_area
}

pub fn iteration_histogram(buf: &IterBuf, max_iter: u32, bins: usize) -> Vec<u64> {
    let mut hist = vec![0u64; bins];
    let scale = bins as f32 / (max_iter as f32 + 1.0);
    for &v in buf.iter() {
        let idx = ((v * scale) as usize).min(bins - 1);
        hist[idx] += 1;
    }
    hist
}
