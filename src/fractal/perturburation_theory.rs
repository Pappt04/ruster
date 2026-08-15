use crate::fractal::fractal::{render, pixel_grid, IterBuf, ESCAPE_RADIUS_SQ, smooth_iter};
use crate::fractal::mandelbrot::mandelbrot;
use crate::fractal::double_double::DoubleDouble;
use crate::fractal::fractal_type::FractalType;
use crate::gui::viewport::Viewport;
use rayon::prelude::*;

pub struct RefOrbit {
    pub zr: Vec<f64>,
    pub zi: Vec<f64>,
    pub len: usize,
}

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

pub const F128_ZOOM_THRESHOLD: f64 = 1e12;

pub fn compute_reference_orbit_f128(cr: f64, ci: f64, max_iter: u32) -> RefOrbit {
    let n = max_iter as usize;
    let mut zr_out = vec![0.0f64; n + 1];
    let mut zi_out = vec![0.0f64; n + 1];

    let cr_dd     = DoubleDouble::from_f64(cr);
    let ci_dd     = DoubleDouble::from_f64(ci);
    let escape_sq = DoubleDouble::from_f64(ESCAPE_RADIUS_SQ);

    let mut zr = DoubleDouble::from_f64(0.0);
    let mut zi = DoubleDouble::from_f64(0.0);

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

const GLITCH_SQ: f64 = 1e-6;

#[inline]
pub fn perturb_mandelbrot_flagged(
    orbit: &RefOrbit,
    dc_re: f64, dc_im: f64, 
    max_iter: u32,
) -> Option<f32> {
    let mut er = 0.0f64;
    let mut ei = 0.0f64;

    for n in 0..orbit.len {
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
            return Some(smooth_iter(n as u32 + 1, zn_sq, max_iter));
        }

        let ref_sq = orbit.zr[n + 1] * orbit.zr[n + 1] + orbit.zi[n + 1] * orbit.zi[n + 1];
        if er * er + ei * ei > ref_sq * GLITCH_SQ {
            return None;
        }
    }

    if orbit.len >= max_iter as usize {
        Some(max_iter as f32)
    } else {
        None
    }
}

#[inline]
fn perturb_mandelbrot(
    orbit: &RefOrbit,
    dc_re: f64, dc_im: f64,   
    full_re: f64, full_im: f64, 
    max_iter: u32,
) -> f32 {
    perturb_mandelbrot_flagged(orbit, dc_re, dc_im, max_iter)
        .unwrap_or_else(|| mandelbrot(full_re, full_im, max_iter))
}

#[inline]
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

pub const MAX_REFS: usize = 8;

pub struct RefOrbitSet {
    pub refs: Vec<RefOrbit>,
    pub centers: Vec<(f64, f64)>,
}

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


pub struct SeriesApprox {
    pub skip: usize,
    pub ar: f64, pub ai: f64,
    pub br: f64, pub bi: f64,
    pub cr: f64, pub ci: f64,
    pub dr: f64, pub di: f64,
}

const SA_THRESHOLD: f64 = 1e-6;

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

        let a_sq_r   = ar * ar - ai * ai;
        let a_sq_i   = 2.0 * ar * ai;
        let two_ab_r = 2.0 * (ar * br - ai * bi);
        let two_ab_i = 2.0 * (ar * bi + ai * br);
        let d_src_r = 2.0 * (ar * cr - ai * ci) + (br * br - bi * bi);
        let d_src_i = 2.0 * (ar * ci + ai * cr) + 2.0 * br * bi;

        let new_ar = two_zr * ar - two_zi * ai + 1.0;
        let new_ai = two_zr * ai + two_zi * ar;
        let new_br = two_zr * br - two_zi * bi + a_sq_r;
        let new_bi = two_zr * bi + two_zi * br + a_sq_i;
        let new_cr = two_zr * cr - two_zi * ci + two_ab_r;
        let new_ci = two_zr * ci + two_zi * cr + two_ab_i;
        let new_dr = two_zr * dr - two_zi * di + d_src_r;
        let new_di = two_zr * di + two_zi * dr + d_src_i;

        ar = new_ar; ai = new_ai;
        br = new_br; bi = new_bi;
        cr = new_cr; ci = new_ci;
        dr = new_dr; di = new_di;

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

#[inline]
fn perturb_mandelbrot_sa(
    orbit: &RefOrbit,
    sa:    &SeriesApprox,
    dc_re: f64, dc_im: f64,
    full_re: f64, full_im: f64,
    max_iter: u32,
) -> f32 {
    let d2r = dc_re * dc_re - dc_im * dc_im;
    let d2i = 2.0 * dc_re * dc_im;
    let d3r = dc_re * d2r - dc_im * d2i;
    let d3i = dc_re * d2i + dc_im * d2r;
    let d4r = d2r * d2r - d2i * d2i;
    let d4i = 2.0 * d2r * d2i;

    let mut er = sa.ar * dc_re - sa.ai * dc_im
               + sa.br * d2r   - sa.bi * d2i
               + sa.cr * d3r   - sa.ci * d3i
               + sa.dr * d4r   - sa.di * d4i;
    let mut ei = sa.ar * dc_im + sa.ai * dc_re
               + sa.br * d2i   + sa.bi * d2r
               + sa.cr * d3i   + sa.ci * d3r
               + sa.dr * d4i   + sa.di * d4r;

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

pub fn render_perturbation_sa(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    if fractal != FractalType::Mandelbrot {
        return render(vp, fractal, julia_c, max_iter);
    }

    let w = vp.width  as usize;
    let h = vp.height as usize;

    let pg = pixel_grid(vp);
    let center_re = vp.center[0];
    let center_im = vp.center[1];

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