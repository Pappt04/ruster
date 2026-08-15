//! Perturbation-theory rendering of the Mandelbrot set for zoom levels
//! where f64 (or even double-double) precision on every pixel would be too
//! slow, or where the zoom exceeds what a per-pixel high-precision
//! iteration could resolve in reasonable time.
//!
//! Instead of iterating `z_{n+1} = z_n^2 + c` at full precision for every
//! pixel, a single *reference* orbit is iterated at high precision for one
//! point (the view center), and every other pixel is expressed as a small
//! offset `delta_c = c_pixel - c_ref` from it. Writing `Z_n` for the
//! reference orbit and `z_n = Z_n + delta_n` for a nearby pixel's orbit,
//! substituting into the iteration and canceling `Z_n^2 + c_ref = Z_{n+1}`
//! gives the perturbation recurrence
//!
//! ```text
//! delta_{n+1} = 2 Z_n delta_n + delta_n^2 + delta_c
//! ```
//!
//! `delta_n` starts at (and typically stays) many orders of magnitude
//! smaller than `Z_n`, so it can be tracked in plain f64 even where `Z_n`
//! itself would need extended precision — only the one reference orbit
//! pays for that precision. This is the algorithm behind essentially all
//! modern deep-zoom Mandelbrot renderers.
//!
//! Two complications this module handles: perturbation breaks down
//! ("glitches") when `delta_n` grows large relative to `Z_n`, handled by
//! [`perturb_mandelbrot_flagged`]/glitch fallback and by
//! [`perturb_mandelbrot_rebase`]'s rebasing trick; and the early
//! iterations of `delta_n` near the reference center can be predicted by a
//! power series in `delta_c` instead of iterated at all, handled by
//! [`compute_series_approx`]/[`perturb_mandelbrot_sa`].

use crate::fractal::fractal::{render, pixel_grid, IterBuf, ESCAPE_RADIUS_SQ, smooth_iter};
use crate::fractal::kernels::mandelbrot::mandelbrot;
use crate::fractal::perturbation::double_double::DoubleDouble;
use crate::fractal::fractal_type::FractalType;
use crate::gui::viewport::Viewport;
use rayon::prelude::*;

/// A precomputed high-precision orbit for one reference point, truncated
/// at `len` if the reference itself escaped before `max_iter`. `zr`/`zi`
/// are stored as f64 (the high part only, when computed in double-double)
/// since the perturbation recurrence only needs `Z_n` to f64 precision —
/// the extended precision is only required while *computing* the orbit,
/// to avoid catastrophic cancellation accumulating over many iterations.
pub struct RefOrbit {
    pub zr: Vec<f64>,
    pub zi: Vec<f64>,
    pub len: usize,
}

/// Computes a reference orbit at f64 precision. Sufficient below
/// [`F128_ZOOM_THRESHOLD`], where the reference point's own coordinates
/// still have enough bits of resolution relative to the view.
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

/// Zoom level above which the reference orbit itself must be computed in
/// [`DoubleDouble`] precision rather than plain f64 — beyond this, f64's
/// ~15-16 significant decimal digits are no longer enough to place the
/// reference point distinctly from its neighborhood at the current zoom,
/// which would corrupt the orbit `Z_n` that every pixel's perturbation is
/// measured against.
pub const F128_ZOOM_THRESHOLD: f64 = 1e12;

/// Double-double-precision counterpart of [`compute_reference_orbit`],
/// used past [`F128_ZOOM_THRESHOLD`]. Only the reference iteration itself
/// runs in extended precision; the result is immediately truncated back to
/// f64 (via [`DoubleDouble::hi`]) for storage, since perturbation deltas
/// computed against it only need f64.
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

/// Relative threshold (as a ratio of squared magnitudes) at which
/// `|delta_n|` is considered to have grown too large relative to `|Z_n|`
/// for perturbation to remain a valid approximation of the true orbit —
/// a "glitch". `1e-6` corresponds to `|delta_n| / |Z_n| > ~1e-3`.
const GLITCH_SQ: f64 = 1e-6;

/// Iterates the perturbation recurrence `delta_{n+1} = 2 Z_n delta_n +
/// delta_n^2 + delta_c` for one pixel against a shared [`RefOrbit`],
/// starting from `delta_0 = 0` (i.e. `z_0 = Z_0 = 0`, matching the
/// standard Mandelbrot iteration start).
///
/// Returns `Some(iteration)` on ordinary escape, or once the reference
/// orbit itself is exhausted without escaping (`orbit.len >= max_iter`,
/// meaning both the reference and this pixel are in the set). Returns
/// `None` if a glitch is detected (`|delta_n|^2` exceeds `GLITCH_SQ` times
/// `|Z_n|^2`) or if the reference orbit ran out before `max_iter` without
/// the pixel escaping — either way the caller must fall back to a direct,
/// full-precision computation for this pixel.
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

/// [`perturb_mandelbrot_flagged`] with the glitch/exhaustion case resolved
/// by falling back to a direct scalar [`mandelbrot`] computation at this
/// pixel's true coordinates — correct but pays full iteration cost for
/// every glitched pixel.
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

/// Perturbation iteration with automatic *rebasing* instead of falling
/// back to full precision on a glitch.
///
/// `m` indexes into the reference orbit independently of the true
/// iteration count `n`. Normally `m` tracks `n` step for step, but
/// whenever the true orbit position `Z_m + delta` becomes smaller in
/// magnitude than `delta` itself — i.e. the reference value has stopped
/// contributing anything useful, which happens when the reference orbit
/// happens to pass back near the origin, a common occurrence for
/// Mandelbrot reference points — the pixel's actual value `(az, bz)` is
/// adopted as a fresh `delta` and `m` resets to `0`. Since `orbit.zr[0] =
/// orbit.zi[0] = 0`, restarting at `m = 0` with `delta = (az, bz)` is
/// exactly equivalent to restarting perturbation with a zero reference
/// contribution, letting the same precomputed orbit be reused as if a new
/// reference had been recomputed at this pixel — without the cost of
/// actually doing so. The same reset also triggers when the orbit array is
/// exhausted (`m == orbit.len`), recycling it from the start rather than
/// stopping. This removes the need for [`render_perturbation_multiref`]'s
/// multi-reference search in most cases.
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

/// Perturbation render using [`perturb_mandelbrot_rebase`]: one reference
/// orbit at the view center, every pixel handled by the rebasing kernel so
/// glitches never fall back to per-pixel full-precision computation. Other
/// fractal types have no perturbation kernel and fall back to the general
/// [`render`].
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

/// Perturbation render using the plain [`perturb_mandelbrot`] kernel: one
/// reference orbit at the view center, glitched pixels individually
/// recomputed at full precision rather than rebased. Simpler and slightly
/// slower on heavily glitched views than [`render_perturbation_rebase`].
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

/// Upper bound on how many reference orbits [`render_perturbation_multiref`]
/// will compute for a single frame, capping worst-case cost when a view
/// contains many disjoint glitch-prone regions.
pub const MAX_REFS: usize = 8;

/// Accumulated reference orbits and their centers, used by
/// [`render_perturbation_multiref`] to track every reference computed so
/// far for one frame.
pub struct RefOrbitSet {
    pub refs: Vec<RefOrbit>,
    pub centers: Vec<(f64, f64)>,
}

/// Perturbation render with iterative multi-reference glitch correction:
/// an alternative to [`render_perturbation_rebase`]'s single-orbit
/// rebasing trick, for cases rebasing alone cannot resolve.
///
/// Renders the whole frame against one reference orbit at the view center
/// using [`perturb_mandelbrot_flagged`], marking every glitched pixel
/// `NaN`. If any remain, a new reference orbit is computed centered at the
/// first glitched pixel found (glitches cluster spatially, so one new
/// reference tends to resolve a whole neighboring region at once) and
/// every still-`NaN` pixel is retried against it. This repeats until no
/// glitches remain or [`MAX_REFS`] references have been used; any pixels
/// still unresolved after that are individually recomputed at full
/// precision as a last resort.
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

    // Any pixel still glitched after MAX_REFS references is resolved
    // directly, bypassing perturbation entirely.
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

/// Power-series coefficients (`A`, `B`, `C`, `D` for the `delta_c^1..4`
/// terms) approximating the perturbation orbit `delta_n` as a polynomial
/// in `delta_c`, valid for the first `skip` iterations. See
/// [`compute_series_approx`].
pub struct SeriesApprox {
    pub skip: usize,
    pub ar: f64, pub ai: f64,
    pub br: f64, pub bi: f64,
    pub cr: f64, pub ci: f64,
    pub dr: f64, pub di: f64,
}

/// Relative-error threshold (as a ratio of magnitudes, before squaring)
/// used to decide how many terms of the series remain trustworthy — see
/// [`compute_series_approx`].
const SA_THRESHOLD: f64 = 1e-6;

/// Derives, iteration by iteration alongside the reference orbit, a
/// 4th-order power series `delta_n ~= A_n*dc + B_n*dc^2 + C_n*dc^3 +
/// D_n*dc^4` that approximates the perturbation delta as a polynomial in
/// `delta_c` — allowing [`perturb_mandelbrot_sa`] to evaluate the first
/// `skip` iterations of a pixel's orbit in closed form instead of
/// iterating them.
///
/// The coefficients follow by substituting that series ansatz into the
/// perturbation recurrence `delta_{n+1} = 2 Z_n delta_n + delta_n^2 +
/// delta_c` and matching terms of equal power in `delta_c`:
/// `A_{n+1} = 2 Z_n A_n + 1`, `B_{n+1} = 2 Z_n B_n + A_n^2`,
/// `C_{n+1} = 2 Z_n C_n + 2 A_n B_n`,
/// `D_{n+1} = 2 Z_n D_n + (2 A_n C_n + B_n^2)` — each the coefficient of
/// `delta_n^2`'s corresponding power once `delta_n` itself is expanded and
/// squared. `a_sq`, `two_ab`, `d_src` below are exactly those squared/cross
/// terms.
///
/// The recursion is valid only while the dropped 5th-order term stays
/// negligible; `skip` is the last iteration where, for the worst-case
/// `delta_c` in the frame (`delta_max_sq`, the squared distance from
/// center to the farthest viewport corner), the `C`- and `D`-order terms
/// stay below [`SA_THRESHOLD`] relative to the leading `A` term. Iteration
/// stops there rather than continuing to accumulate an invalid series.
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

/// Evaluates [`SeriesApprox`] at this pixel's `delta_c` to get `delta_n`
/// directly at `n = sa.skip` (via Horner-free direct powers `dc^2`, `dc^3`,
/// `dc^4` computed by repeated complex multiplication), then continues
/// ordinary perturbation iteration ([`perturb_mandelbrot_flagged`]'s logic
/// inlined) from there to escape or `max_iter`. Falls back to full-precision
/// [`mandelbrot`] on a glitch, the same as [`perturb_mandelbrot`].
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

/// Perturbation render combining one reference orbit at the view center
/// with a [`SeriesApprox`] fitted to it, so every pixel's iteration starts
/// at `sa.skip` instead of `0` — the series replaces the early iterations,
/// which are identical in structure for every pixel and would otherwise be
/// repeated `width * height` times.
pub fn render_perturbation_sa(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    if fractal != FractalType::Mandelbrot {
        return render(vp, fractal, julia_c, max_iter);
    }

    let w = vp.width  as usize;
    let h = vp.height as usize;

    let pg = pixel_grid(vp);
    let center_re = vp.center[0];
    let center_im = vp.center[1];

    // Worst-case |delta_c|^2 over the frame: the squared distance from the
    // view center to a viewport corner, used to bound the series' error.
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