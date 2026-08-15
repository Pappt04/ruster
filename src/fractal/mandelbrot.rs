use crate::fractal::fractal::{ESCAPE_RADIUS_SQ, ESCAPE_RADIUS_SQ_F32, smooth_iter, smooth_iter_f32};
use crate::fractal::bulb_precheck::{bulb_precheck_x4, bulb_precheck_x8, in_cardioid_or_period2, in_period3_bulb};

pub fn mandelbrot_x8(cr: wide::f32x8, ci: wide::f32x8, max_iter: u32) -> [f32; 8] {
    use wide::{f32x8, CmpGt};

    let escape  = f32x8::splat(ESCAPE_RADIUS_SQ_F32);
    let two     = f32x8::splat(2.0f32);
    let all_one = f32x8::splat(f32::from_bits(u32::MAX));

    let in_set = bulb_precheck_x8(cr, ci);
    if in_set.iter().all(|&b| b) {
        return [max_iter as f32; 8];
    }

    let mut zr = cr;
    let mut zi = ci;
    if max_iter <= 1 { return [max_iter as f32; 8]; }
    {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        zi = (two * zr).mul_add(zi, ci);
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

        zi = (two * zr).mul_add(zi, ci);
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

pub(crate) fn mandelbrot_x8x2(
    cr0: wide::f32x8, ci0: wide::f32x8,
    cr1: wide::f32x8, ci1: wide::f32x8,
    max_iter: u32,
) -> ([f32; 8], [f32; 8]) {
    use wide::{f32x8, CmpGt};

    let escape  = f32x8::splat(ESCAPE_RADIUS_SQ_F32);
    let two     = f32x8::splat(2.0f32);
    let all_one = f32x8::splat(f32::from_bits(u32::MAX));

    let in_set0 = bulb_precheck_x8(cr0, ci0);
    let in_set1 = bulb_precheck_x8(cr1, ci1);
    if in_set0.iter().all(|&b| b) && in_set1.iter().all(|&b| b) {
        return ([max_iter as f32; 8], [max_iter as f32; 8]);
    }

    let mut zr0 = cr0;
    let mut zi0 = ci0;
    let mut zr1 = cr1;
    let mut zi1 = ci1;
    if max_iter <= 1 {
        return ([max_iter as f32; 8], [max_iter as f32; 8]);
    }
    {
        let zr2_0 = zr0 * zr0; let zi2_0 = zi0 * zi0;
        zi0 = (two * zr0).mul_add(zi0, ci0);
        zr0 = zr2_0 - zi2_0 + cr0;
        let zr2_1 = zr1 * zr1; let zi2_1 = zi1 * zi1;
        zi1 = (two * zr1).mul_add(zi1, ci1);
        zr1 = zr2_1 - zi2_1 + cr1;
    }
    if max_iter <= 2 {
        return ([max_iter as f32; 8], [max_iter as f32; 8]);
    }

    let mk_active = |in_set: &[bool; 8]| -> f32x8 {
        f32x8::from(std::array::from_fn::<f32, 8, _>(|lane| {
            f32::from_bits(if in_set[lane] { 0 } else { u32::MAX })
        }))
    };
    let mut active0 = mk_active(&in_set0);
    let mut active1 = mk_active(&in_set1);
    let mut escape_iter0 = [max_iter; 8];
    let mut escape_iter1 = [max_iter; 8];
    let mut escape_zn0   = [0.0f32; 8];
    let mut escape_zn1   = [0.0f32; 8];

    for i in 2..max_iter {
        // batch 0
        let zr2_0   = zr0 * zr0;
        let zi2_0   = zi0 * zi0;
        let zn_sq_0 = zr2_0 + zi2_0;
        // batch 1 (interleaved so the CPU can pipeline both chains)
        let zr2_1   = zr1 * zr1;
        let zi2_1   = zi1 * zi1;
        let zn_sq_1 = zr2_1 + zi2_1;

        let just_escaped0 = zn_sq_0.cmp_gt(escape) & active0;
        if just_escaped0.any() {
            let mask: [f32; 8] = just_escaped0.into();
            let zn:   [f32; 8] = zn_sq_0.into();
            for lane in 0..8 {
                if mask[lane].to_bits() != 0 {
                    escape_iter0[lane] = i;
                    escape_zn0[lane]   = zn[lane];
                }
            }
            active0 = active0 & (all_one ^ just_escaped0);
        }
        let just_escaped1 = zn_sq_1.cmp_gt(escape) & active1;
        if just_escaped1.any() {
            let mask: [f32; 8] = just_escaped1.into();
            let zn:   [f32; 8] = zn_sq_1.into();
            for lane in 0..8 {
                if mask[lane].to_bits() != 0 {
                    escape_iter1[lane] = i;
                    escape_zn1[lane]   = zn[lane];
                }
            }
            active1 = active1 & (all_one ^ just_escaped1);
        }
        if !active0.any() && !active1.any() { break; }

        zi0 = (two * zr0).mul_add(zi0, ci0);
        zr0 = zr2_0 - zi2_0 + cr0;
        zi1 = (two * zr1).mul_add(zi1, ci1);
        zr1 = zr2_1 - zi2_1 + cr1;
    }

    let finish = |escape_iter: &[u32; 8], escape_zn: &[f32; 8]| -> [f32; 8] {
        std::array::from_fn(|lane| {
            if escape_iter[lane] >= max_iter {
                max_iter as f32
            } else {
                smooth_iter_f32(escape_iter[lane], escape_zn[lane], max_iter)
            }
        })
    };
    (finish(&escape_iter0, &escape_zn0), finish(&escape_iter1, &escape_zn1))
}
pub(crate) fn mandelbrot_x4(cr: wide::f64x4, ci: wide::f64x4, max_iter: u32) -> [f32; 4] {
    use wide::{f64x4, CmpGt};

    let escape  = f64x4::splat(ESCAPE_RADIUS_SQ);
    let two     = f64x4::splat(2.0);
    let all_one = f64x4::splat(f64::from_bits(u64::MAX));

    let in_set = bulb_precheck_x4(cr, ci);
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
        zi = (two * zr).mul_add(zi, ci);
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

        zi = (two * zr).mul_add(zi, ci);
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
pub fn mandelbrot_dem(cr: f64, ci: f64, max_iter: u32) -> (f32, f64) {
    let mut zr = 0.0f64;
    let mut zi = 0.0f64;
    let mut dzr = 0.0f64;
    let mut dzi = 0.0f64;

    for i in 0..max_iter {
        let zn_sq = zr * zr + zi * zi;
        if zn_sq > ESCAPE_RADIUS_SQ {
            let z_mag = zn_sq.sqrt();
            let dz_mag = (dzr * dzr + dzi * dzi).sqrt();
            let d = if dz_mag > 0.0 { z_mag * z_mag.ln() / dz_mag } else { 0.0 };
            return (smooth_iter(i, zn_sq, max_iter), d);
        }
        // dz ← 2·z·dz + 1 (must use pre-update z)
        let new_dzr = 2.0 * (zr * dzr - zi * dzi) + 1.0;
        let new_dzi = 2.0 * (zr * dzi + zi * dzr);
        dzr = new_dzr;
        dzi = new_dzi;
        let new_zr = zr * zr - zi * zi + cr;
        zi = 2.0 * zr * zi + ci;
        zr = new_zr;
    }
    (max_iter as f32, 0.0)
}
#[inline]
pub(crate) fn mandelbrot(cr: f64, ci: f64, max_iter: u32) -> f32 {
    if in_cardioid_or_period2(cr, ci) || in_period3_bulb(cr, ci) {
        return max_iter as f32;
    }

    let mut zr = cr;
    let mut zi = ci;
    if max_iter <= 1 { return max_iter as f32; }
    // iter 1: z = c² + c
    let zr2 = zr * zr;
    let zi2 = zi * zi;
    let new_zi = (2.0 * zr).mul_add(zi, ci);
    zr = zr2 - zi2 + cr;
    zi = new_zi;
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