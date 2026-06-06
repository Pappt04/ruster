use crate::fractal::fractal_type::FractalType;
use crate::gui::viewport::Viewport;
use rayon::prelude::*;

pub const ESCAPE_RADIUS_SQ: f64 = 256.0 * 256.0;

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
    let log_zn = zn_sq.ln() / 2.0;
    let nu = (log_zn / std::f64::consts::LN_2).ln() / std::f64::consts::LN_2;
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

    let (mut zr, mut zi) = (0.0f64, 0.0f64);
    let (mut zr_b, mut zi_b) = (0.0f64, 0.0f64);
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

#[cfg(feature = "simd")]
fn mandelbrot_x4(cr: wide::f64x4, ci: wide::f64x4, max_iter: u32) -> [f32; 4] {
    use wide::{f64x4, CmpGt};

    let escape  = f64x4::splat(ESCAPE_RADIUS_SQ);
    let two     = f64x4::splat(2.0);
    let all_one = f64x4::splat(f64::from_bits(u64::MAX));
    let mut zr  = f64x4::ZERO;
    let mut zi  = f64x4::ZERO;

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