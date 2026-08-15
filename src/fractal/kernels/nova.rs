/// Nova fractal for `f(z) = z^3 - 1`: the same Newton-Raphson update as
/// [`crate::fractal::kernels::newton::newton`], but with the pixel
/// coordinate `c` added back in at every step and `z` always started at the
/// fixed point `1 + 0i` (a root of `f`, chosen so the first step is
/// well-defined).
///
/// `z_{n+1} = z_n - f(z_n)/f'(z_n) + c` turns the fractal from a
/// root-finding *basin* picture (Newton, boundaries in the `z`-plane) into
/// one parametrized by `c`, structurally analogous to how the Mandelbrot
/// set parametrizes the Julia family: here `c` perturbs an otherwise
/// convergent iteration just enough to produce chaotic orbits near the
/// basin boundaries. Convergence and critical-point thresholds are the
/// same as in [`crate::fractal::kernels::newton::newton`].
#[inline]
pub fn nova(cr: f64, ci: f64, max_iter: u32) -> f32 {
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