fn perturb_mandelbrot_rebase(
    orbit: &RefOrbit,
    dc_re: f64, dc_im: f64,
    max_iter: u32,
) -> f32 {
    let mut er = 0.0f64;
    let mut ei = 0.0f64;
    let mut m = 0usize;

    for n in 0..max_iter {
        let zr = orbit.zr[m];
        let zi = orbit.zi[m];

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

        if zn_sq < er * er + ei * ei || m == orbit.len {
            er = az;
            ei = bz;
            m = 0;
        }
    }
    max_iter as f32
}
