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

/// Compute the Mandelbrot reference orbit for C=(cr,ci) up to max_iter steps.
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

/// Zoom threshold above which the reference orbit is computed with double-double precision.
///
/// At zoom ~10^13 f64 runs out of mantissa bits and the orbit drifts, producing
/// block artifacts. Double-double (~31 decimal digits) extends clean rendering to
/// ~10^26 — equivalent to software f128 for this use case, on stable Rust.
/// The orbit is computed once per frame; even at 5–10× f64 cost it adds < 5 ms.
pub const F128_ZOOM_THRESHOLD: f64 = 1e12;

/// Compute the reference orbit using double-double precision, stored as f64.
///
/// Iterates Z_{n+1} = Z_n² + C in double-double (~31 decimal digits), then
/// downcasts each orbit term to f64 for storage. The per-pixel delta loop stays
/// in f64 — deltas remain representable because they are small fractions of the
/// full coordinate. Equivalent to f128 for this application on stable Rust.
pub fn compute_reference_orbit_f128(cr: f64, ci: f64, max_iter: u32) -> RefOrbit {
    let n = max_iter as usize;
    let mut zr_out = vec![0.0f64; n + 1];
    let mut zi_out = vec![0.0f64; n + 1];

    let cr_dd     = DoubleDouble::from_f64(cr);
    let ci_dd     = DoubleDouble::from_f64(ci);
    let escape_sq = DoubleDouble::from_f64(ESCAPE_RADIUS_SQ);

    // zr_out[0] = zi_out[0] = 0 already (Z_0 = 0).
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

/// |ε|²/|Z|² ratio above which the linear approximation is no longer trusted.
/// Equivalent to |ε| > 1e-3 × |Z| (0.1 % of the reference magnitude).
const GLITCH_SQ: f64 = 1e-6;

/// Core perturbation recurrence, factored out of `perturb_mandelbrot` so it can be
/// reused by both the single-reference renderer and the multi-reference glitch
/// corrector (4a in CURSOR_OPTIMIZATIONS.md). Returns `None` on glitch (ε grew too
/// large relative to Z) or when the reference orbit escaped before this pixel did —
/// callers decide how to handle that (single-ref: scalar fallback immediately;
/// multi-ref: try another reference first).
///
/// Recurrence: ε_{n+1} = 2·Z_n·ε_n + ε_n² + δ  (δ = c − C, ε_0 = 0)
/// Escape check on z_{n+1} = Z_{n+1} + ε_{n+1}.
/// `pub` so `bench_runner` can directly count glitches for validation purposes.
#[inline]
pub fn perturb_mandelbrot_flagged(
    orbit: &RefOrbit,
    dc_re: f64, dc_im: f64, // δ = pixel c − reference C
    max_iter: u32,
) -> Option<f32> {
    let mut er = 0.0f64;
    let mut ei = 0.0f64;

    for n in 0..orbit.len {
        let zr = orbit.zr[n];
        let zi = orbit.zi[n];

        // ε_{n+1} = 2·Z_n·ε_n + ε_n² + δ
        let two_zr = 2.0 * zr;
        let two_zi = 2.0 * zi;
        let new_er = two_zr * er - two_zi * ei + (er * er - ei * ei) + dc_re;
        let new_ei = two_zr * ei + two_zi * er + (2.0 * er * ei)     + dc_im;
        er = new_er;
        ei = new_ei;

        // z_{n+1} ≈ Z_{n+1} + ε_{n+1}
        let az = orbit.zr[n + 1] + er;
        let bz = orbit.zi[n + 1] + ei;
        let zn_sq = az * az + bz * bz;

        if zn_sq > ESCAPE_RADIUS_SQ {
            return Some(smooth_iter(n as u32 + 1, zn_sq, max_iter));
        }

        // Glitch: ε has grown too large relative to Z — approximation unreliable.
        let ref_sq = orbit.zr[n + 1] * orbit.zr[n + 1] + orbit.zi[n + 1] * orbit.zi[n + 1];
        if er * er + ei * ei > ref_sq * GLITCH_SQ {
            return None;
        }
    }

    if orbit.len >= max_iter as usize {
        Some(max_iter as f32)
    } else {
        // Reference escaped before this pixel did.
        None
    }
}

/// Approximate one Mandelbrot pixel via perturbation theory.
/// Falls back to the exact scalar kernel on glitch or when the reference orbit ends early.
#[inline]
fn perturb_mandelbrot(
    orbit: &RefOrbit,
    dc_re: f64, dc_im: f64,   // δ = pixel c − reference C
    full_re: f64, full_im: f64, // actual pixel coordinate for fallback
    max_iter: u32,
) -> f32 {
    perturb_mandelbrot_flagged(orbit, dc_re, dc_im, max_iter)
        .unwrap_or_else(|| mandelbrot(full_re, full_im, max_iter))
}

/// Perturbation with mathr/Zhuoran-style rebasing: the pixel's iteration count `n`
/// is decoupled from the reference-orbit index `m`. Whenever the full value
/// `z = Z_m + ε` comes closer to 0 than ε itself (the "flip" — the point where
/// catastrophic cancellation would start corrupting ε), fold z into ε and restart
/// the orbit index at 0 (Z_0 = 0, so ε := z exactly). The same rebase handles the
/// reference orbit escaping early (`m == orbit.len`). ε therefore always stays
/// small relative to the orbit, no glitch criterion or scalar fallback is needed,
/// and the whole pixel stays inside perturbation against the one shared orbit.
/// See 4b in CURSOR_OPTIMIZATIONS.md.
#[inline]
fn perturb_mandelbrot_rebase(
    orbit: &RefOrbit,
    dc_re: f64, dc_im: f64,
    max_iter: u32,
) -> f32 {
    let mut er = 0.0f64;
    let mut ei = 0.0f64;
    let mut m = 0usize; // reference-orbit index, resets to 0 on rebase

    for n in 0..max_iter {
        let zr = orbit.zr[m];
        let zi = orbit.zi[m];

        // ε ← 2·Z_m·ε + ε² + δ
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

        // Flip detection / rebase: z closer to 0 than ε, or orbit data exhausted.
        if zn_sq < er * er + ei * ei || m == orbit.len {
            er = az;
            ei = bz;
            m = 0;
        }
    }
    max_iter as f32
}

/// Render Mandelbrot using perturbation theory with early rebase-on-drift instead of
/// full-restart glitch fallback (4b in CURSOR_OPTIMIZATIONS.md). See
/// `perturb_mandelbrot_rebase` for the scoped-down interpretation of "rebase" used
/// here — a cheaper glitch recovery, not true per-pixel local-reference rebasing.
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

/// Render Mandelbrot using perturbation theory (all other fractals fall back to scalar).
///
/// Computes one reference orbit at the viewport center, then derives every pixel as a
/// perturbation ε around that orbit.  Glitched pixels fall back to the full scalar kernel.
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

/// Maximum number of secondary reference orbits `render_perturbation_multiref`
/// will compute before giving up and falling back to the exact scalar kernel for
/// any pixels still glitched.
pub const MAX_REFS: usize = 8;

/// A center point and its precomputed reference orbit — used by
/// `render_perturbation_multiref` to track secondary references added to recover
/// glitched pixels. See 4a in CURSOR_OPTIMIZATIONS.md.
pub struct RefOrbitSet {
    pub refs: Vec<RefOrbit>,
    pub centers: Vec<(f64, f64)>,
}

/// Render Mandelbrot using perturbation theory with automatic multi-reference
/// glitch correction: pixels that glitch against the primary reference orbit
/// (centered at the viewport) are retried against additional reference orbits
/// seeded at glitch locations, up to `MAX_REFS`. Remaining glitches after that cap
/// fall back to the exact scalar kernel, guaranteeing termination.
///
/// v1 scope (deliberately simpler than a full nearest-reference implementation):
/// - References are grown sequentially, always adding a new one for whatever is
///   still glitched — no per-pixel nearest-reference search across existing refs.
/// - Does not use series approximation (SA) for corrected pixels; combine with SA
///   is deferred (see CURSOR_OPTIMIZATIONS.md 4a note on interaction with 4c).
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

    // Pass 1: primary reference. Glitched pixels are sentinel-marked with NaN
    // (reusing the same sentinel convention render_mariani_silver already uses).
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

    // Sequential rescan + secondary-reference retry loop.
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

    // Any pixels still glitched after MAX_REFS: exact scalar fallback (terminates).
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

// ── Series Approximation (SA) ─────────────────────────────────────────────────
//
// Precomputes four complex power-series coefficients (A, B, C, D) along the
// reference orbit so that, for any pixel offset δ = c − C:
//
//   ε_n ≈ A_n·δ + B_n·δ² + C_n·δ³ + D_n·δ⁴
//
// Recurrences (derived by expanding the perturbation recurrence order by order —
// each right-hand term is the matching-order coefficient of the Cauchy product ε²):
//
//   A_{n+1} = 2·Z_n·A_n + 1
//   B_{n+1} = 2·Z_n·B_n + A_n²
//   C_{n+1} = 2·Z_n·C_n + 2·A_n·B_n
//   D_{n+1} = 2·Z_n·D_n + 2·A_n·C_n + B_n²
//
// We advance until the highest-order term exceeds SA_THRESHOLD × the linear term for the
// worst-case (corner) pixel.  That iteration becomes the "skip" — all pixels
// jump directly to ε_skip via a cheap polynomial evaluation and then continue
// with the ordinary perturbation loop.

/// SA coefficients at the skip point, plus the skip count itself.
pub struct SeriesApprox {
    /// Iterations that can be skipped for the entire frame.
    pub skip: usize,
    /// A coefficient (complex): linear term.
    pub ar: f64, pub ai: f64,
    /// B coefficient (complex): quadratic term.
    pub br: f64, pub bi: f64,
    /// C coefficient (complex): cubic term.
    pub cr: f64, pub ci: f64,
    /// D coefficient (complex): quartic term (4c in CURSOR_OPTIMIZATIONS.md).
    pub dr: f64, pub di: f64,
}

/// max(|C·δ³|, |D·δ⁴|) / |A·δ| ratio above which the SA polynomial is no longer trusted.
const SA_THRESHOLD: f64 = 1e-6;

/// Walk the reference orbit accumulating SA coefficients; return the largest
/// safe skip and the coefficient values at that point.
///
/// `delta_max_sq` is |δ_corner|² — the squared distance from center to the
/// corner pixel.  Use it as a conservative bound on all pixel offsets.
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

        // A_n² and 2·A_n·B_n in complex arithmetic (needed for B and C updates).
        let a_sq_r   = ar * ar - ai * ai;
        let a_sq_i   = 2.0 * ar * ai;
        let two_ab_r = 2.0 * (ar * br - ai * bi);
        let two_ab_i = 2.0 * (ar * bi + ai * br);
        // 2·A_n·C_n + B_n² (the δ⁴ coefficient of ε²)
        let d_src_r = 2.0 * (ar * cr - ai * ci) + (br * br - bi * bi);
        let d_src_i = 2.0 * (ar * ci + ai * cr) + 2.0 * br * bi;

        // A_{n+1} = 2·Z_n·A_n + 1
        let new_ar = two_zr * ar - two_zi * ai + 1.0;
        let new_ai = two_zr * ai + two_zi * ar;
        // B_{n+1} = 2·Z_n·B_n + A_n²
        let new_br = two_zr * br - two_zi * bi + a_sq_r;
        let new_bi = two_zr * bi + two_zi * br + a_sq_i;
        // C_{n+1} = 2·Z_n·C_n + 2·A_n·B_n
        let new_cr = two_zr * cr - two_zi * ci + two_ab_r;
        let new_ci = two_zr * ci + two_zi * cr + two_ab_i;
        // D_{n+1} = 2·Z_n·D_n + 2·A_n·C_n + B_n²
        let new_dr = two_zr * dr - two_zi * di + d_src_r;
        let new_di = two_zr * di + two_zi * dr + d_src_i;

        ar = new_ar; ai = new_ai;
        br = new_br; bi = new_bi;
        cr = new_cr; ci = new_ci;
        dr = new_dr; di = new_di;

        // Accuracy guard, on the worst of the two highest-order terms:
        // stop when |C·δ³| or |D·δ⁴| ≥ SA_THRESHOLD × |A·δ| for the corner pixel.
        // Squared, ÷δ²: |C|²·δ⁴ or |D|²·δ⁶ ≥ SA_THRESHOLD²·|A|²
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

/// Evaluate the SA polynomial and run perturbation from the skip point.
#[inline]
fn perturb_mandelbrot_sa(
    orbit: &RefOrbit,
    sa:    &SeriesApprox,
    dc_re: f64, dc_im: f64,
    full_re: f64, full_im: f64,
    max_iter: u32,
) -> f32 {
    // δ², δ³ (complex powers of the pixel offset).
    let d2r = dc_re * dc_re - dc_im * dc_im;
    let d2i = 2.0 * dc_re * dc_im;
    let d3r = dc_re * d2r - dc_im * d2i;
    let d3i = dc_re * d2i + dc_im * d2r;
    let d4r = d2r * d2r - d2i * d2i;
    let d4i = 2.0 * d2r * d2i;

    // ε_skip = A·δ + B·δ² + C·δ³ + D·δ⁴  (complex multiplications).
    let mut er = sa.ar * dc_re - sa.ai * dc_im
               + sa.br * d2r   - sa.bi * d2i
               + sa.cr * d3r   - sa.ci * d3i
               + sa.dr * d4r   - sa.di * d4i;
    let mut ei = sa.ar * dc_im + sa.ai * dc_re
               + sa.br * d2i   + sa.bi * d2r
               + sa.cr * d3i   + sa.ci * d3r
               + sa.dr * d4i   + sa.di * d4r;

    // Continue with the standard perturbation loop from iteration `skip`.
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

/// Render Mandelbrot with perturbation theory + series approximation.
///
/// Builds the reference orbit and SA coefficients once per frame, then for each
/// pixel evaluates a 3-term polynomial to skip the first `sa.skip` iterations
/// and continues with the perturbation recurrence from there.
pub fn render_perturbation_sa(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    if fractal != FractalType::Mandelbrot {
        return render(vp, fractal, julia_c, max_iter);
    }

    let w = vp.width  as usize;
    let h = vp.height as usize;

    let pg = pixel_grid(vp);
    let center_re = vp.center[0];
    let center_im = vp.center[1];

    // Conservative corner-pixel |δ|² used for the SA validity bound.
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