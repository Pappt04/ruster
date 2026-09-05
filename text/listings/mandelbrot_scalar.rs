pub(crate) fn mandelbrot(cr: f64, ci: f64, max_iter: u32) -> f32 {
    if in_cardioid_or_period2(cr, ci) || in_period3_bulb(cr, ci) {
        return max_iter as f32;
    }

    let mut zr = cr;
    let mut zi = ci;
    let (mut zr_b, mut zi_b) = (0.0, 0.0);
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
