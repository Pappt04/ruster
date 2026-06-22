use crate::fractal::fractal_type::FractalType;
use crate::gui::viewport::Viewport;
use rayon::prelude::*;

pub const ESCAPE_RADIUS_SQ: f64 = 256.0 * 256.0;
pub const ESCAPE_RADIUS_SQ_F32: f32 = 256.0 * 256.0;

/// Below this zoom level f32 has enough precision (~7 sig-figs); use f32x8.
/// Above it fall back to f64x4 to avoid glitching artefacts.
pub const F32_PRECISION_THRESHOLD: f64 = 1e6;

pub type IterBuf = Vec<f32>;

pub fn render(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    let w = vp.width as usize;
    let h = vp.height as usize;

    let aspect = vp.width as f64 / vp.height as f64;
    let half = 2.0 / vp.zoom;
    let re_step = half * aspect * 2.0 / vp.width as f64;
    let im_step = half * 2.0 / vp.height as f64;
    let re_start = vp.center[0] + (0.5 / vp.width as f64 - 0.5) * half * aspect * 2.0;
    let im_start = vp.center[1] + (0.5 / vp.height as f64 - 0.5) * half * 2.0;

    let mut buf = vec![0.0f32; w * h];

    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im = im_start + y as f64 * im_step;
        for x in 0..w {
            let re = re_start + x as f64 * re_step;
            row[x] = compute(fractal, re, im, julia_c, max_iter);
        }
    });

    buf
}

#[cfg(feature = "simd")]
pub fn render_simd(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    use wide::f64x4;

    let w = vp.width as usize;
    let h = vp.height as usize;
    let aspect = vp.width as f64 / vp.height as f64;
    let half = 2.0 / vp.zoom;
    let re_step = half * aspect * 2.0 / vp.width as f64;
    let im_step = half * 2.0 / vp.height as f64;
    let re_start = vp.center[0] + (0.5 / vp.width as f64 - 0.5) * half * aspect * 2.0;
    let im_start = vp.center[1] + (0.5 / vp.height as f64 - 0.5) * half * 2.0;

    let mut buf = vec![0.0f32; w * h];

    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im = im_start + y as f64 * im_step;
        let im4 = f64x4::splat(im);
        let mut x = 0usize;

        while x + 4 <= w {
            let re = f64x4::from([
                re_start + x as f64       * re_step,
                re_start + (x + 1) as f64 * re_step,
                re_start + (x + 2) as f64 * re_step,
                re_start + (x + 3) as f64 * re_step,
            ]);
            let lanes = match fractal {
                FractalType::Mandelbrot => mandelbrot_x4(re, im4, max_iter),
                FractalType::Julia => {
                    let cr4 = f64x4::splat(julia_c[0]);
                    let ci4 = f64x4::splat(julia_c[1]);
                    julia_x4(re, im4, cr4, ci4, max_iter)
                }
                _ => unreachable!(),
            };
            row[x..x+4].copy_from_slice(&lanes);
            x += 4;
        }

        // scalar tail (0–3 remaining pixels)
        for x in x..w {
            let re = re_start + x as f64 * re_step;
            row[x] = match fractal {
                FractalType::Mandelbrot => mandelbrot(re, im, max_iter),
                FractalType::Julia => julia(re, im, julia_c[0], julia_c[1], max_iter),
                _ => unreachable!(),
            };
        }
    });

    buf
}

/// 8-wide f32 SIMD path — 2× more pixels per instruction than f64x4.
/// Coordinate values are computed in f64 then narrowed to f32 per lane so
/// the initial pixel mapping stays accurate before the cast.
#[cfg(feature = "simd")]
pub fn render_simd_f32(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    use wide::f32x8;

    let w = vp.width as usize;
    let h = vp.height as usize;
    let aspect = vp.width as f64 / vp.height as f64;
    let half = 2.0 / vp.zoom;
    let re_step = half * aspect * 2.0 / vp.width as f64;
    let im_step = half * 2.0 / vp.height as f64;
    let re_start = vp.center[0] + (0.5 / vp.width as f64 - 0.5) * half * aspect * 2.0;
    let im_start = vp.center[1] + (0.5 / vp.height as f64 - 0.5) * half * 2.0;

    let mut buf = vec![0.0f32; w * h];

    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im = (im_start + y as f64 * im_step) as f32;
        let im8 = f32x8::splat(im);
        let mut x = 0usize;

        while x + 8 <= w {
            let re = f32x8::from([
                (re_start + (x    ) as f64 * re_step) as f32,
                (re_start + (x + 1) as f64 * re_step) as f32,
                (re_start + (x + 2) as f64 * re_step) as f32,
                (re_start + (x + 3) as f64 * re_step) as f32,
                (re_start + (x + 4) as f64 * re_step) as f32,
                (re_start + (x + 5) as f64 * re_step) as f32,
                (re_start + (x + 6) as f64 * re_step) as f32,
                (re_start + (x + 7) as f64 * re_step) as f32,
            ]);
            let lanes = match fractal {
                FractalType::Mandelbrot => mandelbrot_x8(re, im8, max_iter),
                FractalType::Julia => {
                    let cr8 = f32x8::splat(julia_c[0] as f32);
                    let ci8 = f32x8::splat(julia_c[1] as f32);
                    julia_x8(re, im8, cr8, ci8, max_iter)
                }
                _ => unreachable!(),
            };
            row[x..x + 8].copy_from_slice(&lanes);
            x += 8;
        }

        // scalar tail (0–7 remaining pixels)
        let im_f64 = im_start + y as f64 * im_step;
        for x in x..w {
            let re = re_start + x as f64 * re_step;
            row[x] = match fractal {
                FractalType::Mandelbrot => mandelbrot(re, im_f64, max_iter),
                FractalType::Julia => julia(re, im_f64, julia_c[0], julia_c[1], max_iter),
                _ => unreachable!(),
            };
        }
    });

    buf
}

/// Mariani-Silver rectangle subdivision: compute only border pixels and fill
/// uniform interiors without evaluating every point individually.
pub fn render_mariani_silver(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    let w = vp.width as usize;
    let h = vp.height as usize;

    let aspect = vp.width as f64 / vp.height as f64;
    let half = 2.0 / vp.zoom;
    let re_step = half * aspect * 2.0 / vp.width as f64;
    let im_step = half * 2.0 / vp.height as f64;
    let re_start = vp.center[0] + (0.5 / vp.width as f64 - 0.5) * half * aspect * 2.0;
    let im_start = vp.center[1] + (0.5 / vp.height as f64 - 0.5) * half * 2.0;

    let mut buf = vec![f32::NAN; w * h];
    ms_fill(&mut buf, w, fractal, julia_c, max_iter, re_start, im_start, re_step, im_step, 0, 0, w, h);
    buf
}

/// Minimum side length at which we stop subdividing and compute all pixels directly.
const MS_MIN: usize = 2;

fn ms_fill(
    buf: &mut [f32],
    stride: usize,
    fractal: FractalType,
    julia_c: [f64; 2],
    max_iter: u32,
    re_start: f64,
    im_start: f64,
    re_step: f64,
    im_step: f64,
    x0: usize, y0: usize, x1: usize, y1: usize,
) {
    let w = x1 - x0;
    let h = y1 - y0;

    // Base case: too small to subdivide further — compute every pixel.
    if w <= MS_MIN || h <= MS_MIN {
        for y in y0..y1 {
            let im = im_start + y as f64 * im_step;
            for x in x0..x1 {
                let idx = y * stride + x;
                if buf[idx].is_nan() {
                    buf[idx] = compute(fractal, re_start + x as f64 * re_step, im, julia_c, max_iter);
                }
            }
        }
        return;
    }

    // Compute uncomputed border pixels.
    // Top row
    {
        let im = im_start + y0 as f64 * im_step;
        for x in x0..x1 {
            let idx = y0 * stride + x;
            if buf[idx].is_nan() {
                buf[idx] = compute(fractal, re_start + x as f64 * re_step, im, julia_c, max_iter);
            }
        }
    }
    // Bottom row
    {
        let im = im_start + (y1 - 1) as f64 * im_step;
        for x in x0..x1 {
            let idx = (y1 - 1) * stride + x;
            if buf[idx].is_nan() {
                buf[idx] = compute(fractal, re_start + x as f64 * re_step, im, julia_c, max_iter);
            }
        }
    }
    // Left column (interior rows only)
    {
        let re = re_start + x0 as f64 * re_step;
        for y in (y0 + 1)..(y1 - 1) {
            let idx = y * stride + x0;
            if buf[idx].is_nan() {
                buf[idx] = compute(fractal, re, im_start + y as f64 * im_step, julia_c, max_iter);
            }
        }
    }
    // Right column (interior rows only)
    {
        let re = re_start + (x1 - 1) as f64 * re_step;
        for y in (y0 + 1)..(y1 - 1) {
            let idx = y * stride + (x1 - 1);
            if buf[idx].is_nan() {
                buf[idx] = compute(fractal, re, im_start + y as f64 * im_step, julia_c, max_iter);
            }
        }
    }

    // Check whether all border pixels share the same value.
    let border_val = buf[y0 * stride + x0];
    let uniform = 'check: {
        for x in x0..x1 {
            if buf[y0 * stride + x] != border_val { break 'check false; }
            if buf[(y1 - 1) * stride + x] != border_val { break 'check false; }
        }
        for y in (y0 + 1)..(y1 - 1) {
            if buf[y * stride + x0] != border_val { break 'check false; }
            if buf[y * stride + (x1 - 1)] != border_val { break 'check false; }
        }
        true
    };

    if uniform {
        // Flood-fill the interior.
        for y in (y0 + 1)..(y1 - 1) {
            for x in (x0 + 1)..(x1 - 1) {
                buf[y * stride + x] = border_val;
            }
        }
    } else {
        // Subdivide along the longer axis (non-overlapping halves).
        if w >= h {
            let mid = x0 + w / 2;
            ms_fill(buf, stride, fractal, julia_c, max_iter, re_start, im_start, re_step, im_step, x0, y0, mid, y1);
            ms_fill(buf, stride, fractal, julia_c, max_iter, re_start, im_start, re_step, im_step, mid, y0, x1, y1);
        } else {
            let mid = y0 + h / 2;
            ms_fill(buf, stride, fractal, julia_c, max_iter, re_start, im_start, re_step, im_step, x0, y0, x1, mid);
            ms_fill(buf, stride, fractal, julia_c, max_iter, re_start, im_start, re_step, im_step, x0, mid, x1, y1);
        }
    }
}

/// Reference orbit for perturbation theory.
///
/// Stores the Mandelbrot orbit of the reference point C (the viewport center):
/// Z_0=0, Z_1, ..., Z_len, where `len` iterations were computed before escape
/// (or `len == max_iter` if the reference never escaped).
/// The arrays hold `len + 1` valid entries indexed 0..=len.
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

/// |ε|²/|Z|² ratio above which the linear approximation is no longer trusted.
/// Equivalent to |ε| > 1e-3 × |Z| (0.1 % of the reference magnitude).
const GLITCH_SQ: f64 = 1e-6;

/// Approximate one Mandelbrot pixel via perturbation theory.
///
/// Recurrence: ε_{n+1} = 2·Z_n·ε_n + ε_n² + δ  (δ = c − C, ε_0 = 0)
/// Escape check on z_{n+1} = Z_{n+1} + ε_{n+1}.
/// Falls back to the exact scalar kernel on glitch or when the reference orbit ends early.
#[inline]
fn perturb_mandelbrot(
    orbit: &RefOrbit,
    dc_re: f64, dc_im: f64,   // δ = pixel c − reference C
    full_re: f64, full_im: f64, // actual pixel coordinate for fallback
    max_iter: u32,
) -> f32 {
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
            return smooth_iter(n as u32 + 1, zn_sq, max_iter);
        }

        // Glitch: ε has grown too large relative to Z — approximation unreliable.
        let ref_sq = orbit.zr[n + 1] * orbit.zr[n + 1] + orbit.zi[n + 1] * orbit.zi[n + 1];
        if er * er + ei * ei > ref_sq * GLITCH_SQ {
            return mandelbrot(full_re, full_im, max_iter);
        }
    }

    if orbit.len >= max_iter as usize {
        max_iter as f32
    } else {
        // Reference escaped before this pixel did — fall back to exact kernel.
        mandelbrot(full_re, full_im, max_iter)
    }
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

    let aspect   = vp.width as f64 / vp.height as f64;
    let half     = 2.0 / vp.zoom;
    let re_step  = half * aspect * 2.0 / vp.width  as f64;
    let im_step  = half * 2.0          / vp.height as f64;
    let re_start = vp.center[0] + (0.5 / vp.width  as f64 - 0.5) * half * aspect * 2.0;
    let im_start = vp.center[1] + (0.5 / vp.height as f64 - 0.5) * half * 2.0;
    let center_re = vp.center[0];
    let center_im = vp.center[1];

    let orbit = compute_reference_orbit(center_re, center_im, max_iter);

    let mut buf = vec![0.0f32; w * h];
    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im    = im_start + y as f64 * im_step;
        let dc_im = im - center_im;
        for x in 0..w {
            let re    = re_start + x as f64 * re_step;
            let dc_re = re - center_re;
            row[x] = perturb_mandelbrot(&orbit, dc_re, dc_im, re, im, max_iter);
        }
    });

    buf
}

// ── Series Approximation (SA) ─────────────────────────────────────────────────
//
// Precomputes three complex power-series coefficients (A, B, C) along the
// reference orbit so that, for any pixel offset δ = c − C:
//
//   ε_n ≈ A_n·δ + B_n·δ² + C_n·δ³
//
// Recurrences (derived by expanding the perturbation recurrence order by order):
//
//   A_{n+1} = 2·Z_n·A_n + 1
//   B_{n+1} = 2·Z_n·B_n + A_n²
//   C_{n+1} = 2·Z_n·C_n + 2·A_n·B_n
//
// We advance until the cubic term exceeds SA_THRESHOLD × the linear term for the
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
}

/// |C·δ³| / |A·δ| ratio above which the 3-term SA is no longer trusted.
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

        // A_{n+1} = 2·Z_n·A_n + 1
        let new_ar = two_zr * ar - two_zi * ai + 1.0;
        let new_ai = two_zr * ai + two_zi * ar;
        // B_{n+1} = 2·Z_n·B_n + A_n²
        let new_br = two_zr * br - two_zi * bi + a_sq_r;
        let new_bi = two_zr * bi + two_zi * br + a_sq_i;
        // C_{n+1} = 2·Z_n·C_n + 2·A_n·B_n
        let new_cr = two_zr * cr - two_zi * ci + two_ab_r;
        let new_ci = two_zr * ci + two_zi * cr + two_ab_i;

        ar = new_ar; ai = new_ai;
        br = new_br; bi = new_bi;
        cr = new_cr; ci = new_ci;

        // Accuracy guard: stop when |C·δ³| ≥ SA_THRESHOLD × |A·δ| for the corner pixel.
        // Squared: |C|²·δ⁴ ≥ SA_THRESHOLD²·|A|²·δ²  →  |C|²·delta_max_sq ≥ SA_THRESHOLD²·|A|²
        let a_mag_sq = ar * ar + ai * ai;
        let c_mag_sq = cr * cr + ci * ci;
        if c_mag_sq * delta_max_sq * delta_max_sq > SA_THRESHOLD * SA_THRESHOLD * a_mag_sq {
            break;
        }
        skip = n + 1;
    }

    SeriesApprox { skip, ar, ai, br, bi, cr, ci }
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

    // ε_skip = A·δ + B·δ² + C·δ³  (complex multiplications).
    let mut er = sa.ar * dc_re - sa.ai * dc_im
               + sa.br * d2r   - sa.bi * d2i
               + sa.cr * d3r   - sa.ci * d3i;
    let mut ei = sa.ar * dc_im + sa.ai * dc_re
               + sa.br * d2i   + sa.bi * d2r
               + sa.cr * d3i   + sa.ci * d3r;

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

    let aspect    = vp.width  as f64 / vp.height as f64;
    let half      = 2.0 / vp.zoom;
    let re_step   = half * aspect * 2.0 / vp.width  as f64;
    let im_step   = half * 2.0          / vp.height as f64;
    let re_start  = vp.center[0] + (0.5 / vp.width  as f64 - 0.5) * half * aspect * 2.0;
    let im_start  = vp.center[1] + (0.5 / vp.height as f64 - 0.5) * half * 2.0;
    let center_re = vp.center[0];
    let center_im = vp.center[1];

    // Conservative corner-pixel |δ|² used for the SA validity bound.
    let delta_max_sq = (half * aspect) * (half * aspect) + half * half;

    let orbit = compute_reference_orbit(center_re, center_im, max_iter);
    let sa    = compute_series_approx(&orbit, delta_max_sq);

    let mut buf = vec![0.0f32; w * h];
    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im    = im_start + y as f64 * im_step;
        let dc_im = im - center_im;
        for x in 0..w {
            let re    = re_start + x as f64 * re_step;
            let dc_re = re - center_re;
            row[x] = perturb_mandelbrot_sa(&orbit, &sa, dc_re, dc_im, re, im, max_iter);
        }
    });

    buf
}

/// Single-pixel entry point exposed for benchmarking.
pub fn pixel(fractal: FractalType, re: f64, im: f64, julia_c: [f64; 2], max_iter: u32) -> f32 {
    compute(fractal, re, im, julia_c, max_iter)
}

/// Analytical floating-point operation count per kernel call (lower bound, no FMA fusion).
pub const fn flops_per_iter(fractal: FractalType) -> u64 {
    match fractal {
        // 2 mul (zr²,zi²) + 1 add (zn_sq) + 2 mul + 1 add (zi) + 1 sub + 1 add (zr) = 8
        FractalType::Mandelbrot | FractalType::Julia => 8,
        // z³ (6) + f (2) + f' (4) + complex div+sub (10) + step check (3) = 25
        FractalType::Newton => 25,
        // Newton + 2 adds for nova perturbation = 27
        FractalType::Nova => 27,
    }
}

fn compute(fractal: FractalType, re: f64, im: f64, julia_c: [f64; 2], max_iter: u32) -> f32 {
    match fractal {
        FractalType::Nova => nova(re, im, max_iter),
        FractalType::Newton => newton(re, im, max_iter),
        FractalType::Mandelbrot => mandelbrot(re, im, max_iter),
        FractalType::Julia => julia(re, im, julia_c[0], julia_c[1], max_iter),
    }
}

fn smooth_iter(iter: u32, zn_sq: f64, max_iter: u32) -> f32 {
    if iter >= max_iter {
        return max_iter as f32;
    }
    const INV_LN2: f64 = std::f64::consts::LOG2_E;
    let log_zn = zn_sq.ln() / 2.0;
    let nu = (log_zn * INV_LN2).ln() * INV_LN2;
    (iter as f64 + 1.0 - nu) as f32
}

#[inline]
fn mandelbrot(cr: f64, ci: f64, max_iter: u32) -> f32 {
    let q = (cr - 0.25) * (cr - 0.25) + ci * ci;
    if q * (q + cr - 0.25) < 0.25 * ci * ci {
        return max_iter as f32;
    }
    if (cr + 1.0) * (cr + 1.0) + ci * ci < 0.0625 {
        return max_iter as f32;
    }

    let mut zr = cr;
    let mut zi = ci;
    if max_iter <= 1 { return max_iter as f32; }
    // iter 1: z = c² + c
    let zr2 = zr * zr;
    let zi2 = zi * zi;
    zi = 2.0 * zr * zi + ci;
    zr = zr2 - zi2 + cr;
    if max_iter <= 2 { return max_iter as f32; }

    let (mut zr_b, mut zi_b) = (0.0f64, 0.0f64);
    let mut period = 0u32;
    let mut check = 8u32;

    for i in 2..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        let zn_sq = zr2 + zi2;
        if zn_sq > ESCAPE_RADIUS_SQ {
            return smooth_iter(i, zn_sq, max_iter);
        }
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;

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

#[inline]
fn julia(zr0: f64, zi0: f64, cr: f64, ci: f64, max_iter: u32) -> f32 {
    let (mut zr, mut zi) = (zr0, zi0);
    let (mut zr_b, mut zi_b) = (zr0, zi0);
    let mut period = 0u32;
    let mut check = 8u32;

    for i in 0..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        let zn_sq = zr2 + zi2;
        if zn_sq > ESCAPE_RADIUS_SQ {
            return smooth_iter(i, zn_sq, max_iter);
        }
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;

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

#[inline]
fn newton(cr: f64, ci: f64, max_iter: u32) -> f32 {
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

#[inline]
fn nova(cr: f64, ci: f64, max_iter: u32) -> f32 {
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

fn smooth_iter_f32(iter: u32, zn_sq: f32, max_iter: u32) -> f32 {
    if iter >= max_iter { return max_iter as f32; }
    const INV_LN2: f32 = std::f32::consts::LOG2_E;
    let log_zn = zn_sq.ln() / 2.0;
    let nu = (log_zn * INV_LN2).ln() * INV_LN2;
    iter as f32 + 1.0 - nu
}

#[cfg(feature = "simd")]
fn mandelbrot_x8(cr: wide::f32x8, ci: wide::f32x8, max_iter: u32) -> [f32; 8] {
    use wide::{f32x8, CmpGt};

    let escape  = f32x8::splat(ESCAPE_RADIUS_SQ_F32);
    let two     = f32x8::splat(2.0f32);
    let all_one = f32x8::splat(f32::from_bits(u32::MAX));

    // Cardioid / period-2 bulb test (per-lane scalar).
    let cr_arr: [f32; 8] = cr.into();
    let ci_arr: [f32; 8] = ci.into();
    let mut in_set = [false; 8];
    for lane in 0..8 {
        let (c_re, c_im) = (cr_arr[lane], ci_arr[lane]);
        let q = (c_re - 0.25) * (c_re - 0.25) + c_im * c_im;
        if q * (q + c_re - 0.25) < 0.25 * c_im * c_im {
            in_set[lane] = true;
            continue;
        }
        let d = c_re + 1.0;
        if d * d + c_im * c_im < 0.0625 {
            in_set[lane] = true;
        }
    }
    if in_set.iter().all(|&b| b) {
        return [max_iter as f32; 8];
    }

    let mut zr = cr;
    let mut zi = ci;
    if max_iter <= 1 { return [max_iter as f32; 8]; }
    {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        zi = two * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }
    if max_iter <= 2 { return [max_iter as f32; 8]; }

    let mut active = f32x8::from([
        f32::from_bits(if in_set[0] { 0 } else { u32::MAX }),
        f32::from_bits(if in_set[1] { 0 } else { u32::MAX }),
        f32::from_bits(if in_set[2] { 0 } else { u32::MAX }),
        f32::from_bits(if in_set[3] { 0 } else { u32::MAX }),
        f32::from_bits(if in_set[4] { 0 } else { u32::MAX }),
        f32::from_bits(if in_set[5] { 0 } else { u32::MAX }),
        f32::from_bits(if in_set[6] { 0 } else { u32::MAX }),
        f32::from_bits(if in_set[7] { 0 } else { u32::MAX }),
    ]);
    let mut escape_iter = [max_iter; 8];
    let mut escape_zn   = [0.0f32; 8];

    for i in 2..max_iter {
        let zr2   = zr * zr;
        let zi2   = zi * zi;
        let zn_sq = zr2 + zi2;

        let just_escaped = zn_sq.cmp_gt(escape) & active;
        if just_escaped.any() {
            let mask: [f32; 8] = just_escaped.into();
            let zn:   [f32; 8] = zn_sq.into();
            for lane in 0..8 {
                if mask[lane].to_bits() != 0 {
                    escape_iter[lane] = i;
                    escape_zn[lane]   = zn[lane];
                }
            }
            active = active & (all_one ^ just_escaped);
            if !active.any() { break; }
        }

        zi = two * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }

    let mut out = [0.0f32; 8];
    for lane in 0..8 {
        out[lane] = if escape_iter[lane] >= max_iter {
            max_iter as f32
        } else {
            smooth_iter_f32(escape_iter[lane], escape_zn[lane], max_iter)
        };
    }
    out
}

#[cfg(feature = "simd")]
fn julia_x8(zr0: wide::f32x8, zi0: wide::f32x8, cr: wide::f32x8, ci: wide::f32x8, max_iter: u32) -> [f32; 8] {
    use wide::{f32x8, CmpGt};

    let escape  = f32x8::splat(ESCAPE_RADIUS_SQ_F32);
    let two     = f32x8::splat(2.0f32);
    let all_one = f32x8::splat(f32::from_bits(u32::MAX));
    let mut zr  = zr0;
    let mut zi  = zi0;

    let mut active = all_one;
    let mut escape_iter = [max_iter; 8];
    let mut escape_zn   = [0.0f32; 8];

    for i in 0..max_iter {
        let zr2   = zr * zr;
        let zi2   = zi * zi;
        let zn_sq = zr2 + zi2;

        let just_escaped = zn_sq.cmp_gt(escape) & active;
        if just_escaped.any() {
            let mask: [f32; 8] = just_escaped.into();
            let zn:   [f32; 8] = zn_sq.into();
            for lane in 0..8 {
                if mask[lane].to_bits() != 0 {
                    escape_iter[lane] = i;
                    escape_zn[lane]   = zn[lane];
                }
            }
            active = active & (all_one ^ just_escaped);
            if !active.any() { break; }
        }

        zi = two * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }

    let mut out = [0.0f32; 8];
    for lane in 0..8 {
        out[lane] = if escape_iter[lane] >= max_iter {
            max_iter as f32
        } else {
            smooth_iter_f32(escape_iter[lane], escape_zn[lane], max_iter)
        };
    }
    out
}

#[cfg(feature = "simd")]
fn mandelbrot_x4(cr: wide::f64x4, ci: wide::f64x4, max_iter: u32) -> [f32; 4] {
    use wide::{f64x4, CmpGt};

    let escape  = f64x4::splat(ESCAPE_RADIUS_SQ);
    let two     = f64x4::splat(2.0);
    let all_one = f64x4::splat(f64::from_bits(u64::MAX));

    // Per-lane cardioid / period-2 bulb test — same math as the scalar path.
    let cr_arr: [f64; 4] = cr.into();
    let ci_arr: [f64; 4] = ci.into();
    let mut in_set = [false; 4];
    for lane in 0..4 {
        let (c_re, c_im) = (cr_arr[lane], ci_arr[lane]);
        let q = (c_re - 0.25) * (c_re - 0.25) + c_im * c_im;
        if q * (q + c_re - 0.25) < 0.25 * c_im * c_im {
            in_set[lane] = true;
            continue;
        }
        let d = c_re + 1.0;
        if d * d + c_im * c_im < 0.0625 {
            in_set[lane] = true;
        }
    }
    if in_set.iter().all(|&b| b) {
        return [max_iter as f32; 4];
    }

    let mut zr = cr;
    let mut zi = ci;
    if max_iter <= 1 {
        return [max_iter as f32; 4];
    }
    {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        zi = two * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }
    if max_iter <= 2 {
        return [max_iter as f32; 4];
    }

    // Exclude already-classified in-set lanes from the SIMD loop.
    let mut active = f64x4::from([
        f64::from_bits(if in_set[0] { 0 } else { u64::MAX }),
        f64::from_bits(if in_set[1] { 0 } else { u64::MAX }),
        f64::from_bits(if in_set[2] { 0 } else { u64::MAX }),
        f64::from_bits(if in_set[3] { 0 } else { u64::MAX }),
    ]);
    let mut escape_iter = [max_iter; 4];
    let mut escape_zn   = [0.0f64; 4];

    for i in 2..max_iter {
        let zr2   = zr * zr;
        let zi2   = zi * zi;
        let zn_sq = zr2 + zi2;

        let just_escaped = zn_sq.cmp_gt(escape) & active;
        if just_escaped.any() {
            let mask: [f64; 4] = just_escaped.into();
            let zn:   [f64; 4] = zn_sq.into();
            for lane in 0..4 {
                if mask[lane].to_bits() != 0 {
                    escape_iter[lane] = i;
                    escape_zn[lane]   = zn[lane];
                }
            }
            active = active & (all_one ^ just_escaped);
            if !active.any() { break; }
        }

        zi = two * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }

    let mut out = [0.0f32; 4];
    for lane in 0..4 {
        out[lane] = if escape_iter[lane] >= max_iter {
            max_iter as f32
        } else {
            smooth_iter(escape_iter[lane], escape_zn[lane], max_iter)
        };
    }
    out
}

#[cfg(feature = "simd")]
fn julia_x4(zr0: wide::f64x4, zi0: wide::f64x4, cr: wide::f64x4, ci: wide::f64x4, max_iter: u32) -> [f32; 4] {
    use wide::{f64x4, CmpGt};

    let escape  = f64x4::splat(ESCAPE_RADIUS_SQ);
    let two     = f64x4::splat(2.0);
    let all_one = f64x4::splat(f64::from_bits(u64::MAX));
    let mut zr  = zr0;
    let mut zi  = zi0;

    let mut active = all_one;
    let mut escape_iter = [max_iter; 4];
    let mut escape_zn   = [0.0f64; 4];

    for i in 0..max_iter {
        let zr2   = zr * zr;
        let zi2   = zi * zi;
        let zn_sq = zr2 + zi2;

        let just_escaped = zn_sq.cmp_gt(escape) & active;
        if just_escaped.any() {
            let mask: [f64; 4] = just_escaped.into();
            let zn:   [f64; 4] = zn_sq.into();
            for lane in 0..4 {
                if mask[lane].to_bits() != 0 {
                    escape_iter[lane] = i;
                    escape_zn[lane]   = zn[lane];
                }
            }
            active = active & (all_one ^ just_escaped);
            if !active.any() { break; }
        }

        zi = two * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }

    let mut out = [0.0f32; 4];
    for lane in 0..4 {
        out[lane] = if escape_iter[lane] >= max_iter {
            max_iter as f32
        } else {
            smooth_iter(escape_iter[lane], escape_zn[lane], max_iter)
        };
    }
    out
}