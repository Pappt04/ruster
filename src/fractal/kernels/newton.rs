/// Newton fractal for `f(z) = z^3 - 1`, seeded at the pixel's own complex
/// coordinate `z_0 = c`.
///
/// Applies the complex Newton-Raphson update
/// `z_{n+1} = z_n - f(z_n) / f'(z_n)` with `f'(z) = 3z^2`, computed by
/// expanding `f(z)/f'(z)` into real/imaginary components and dividing by
/// `|f'(z)|^2` (multiplying by the conjugate of the derivative). Every
/// starting point converges to one of the three cube roots of unity except
/// on a measure-zero set of boundaries between their basins of attraction —
/// it is this boundary that produces the fractal structure. The returned
/// value is the iteration count at convergence (not which root was
/// reached), so coloring shows convergence speed rather than basin
/// identity.
///
/// `denom < 1e-20` guards against `f'(z) == 0`, the critical point where
/// Newton's method is undefined; `dr*dr + di*di < 1e-12` is the
/// step-size convergence threshold — iteration stops once successive `z`
/// values stop moving in a way that matters at f32 output precision.
#[inline]
pub fn newton(cr: f64, ci: f64, max_iter: u32) -> f32 {
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