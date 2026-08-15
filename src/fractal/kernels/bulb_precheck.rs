//! Closed-form membership tests for the largest interior regions of the
//! Mandelbrot set, used to skip iterating points that are known in advance
//! to never escape.
//!
//! For `c` inside the main cardioid or the period-2 bulb, the orbit of `z`
//! is bounded and never leaves the escape radius, so escape-time iteration
//! would otherwise run all the way to `max_iter` for every such pixel — a
//! significant fraction of the total area near the classic view. These
//! tests replace that wasted iteration with a handful of arithmetic ops.

/// Center of the larger of the two period-3 bulbs attached to the main
/// cardioid (the other is its mirror image below the real axis).
const PERIOD3_CENTER_RE: f64 = -0.122561_1668766536;
const PERIOD3_CENTER_IM: f64 = 0.744861_7666197442;
const PERIOD3_RADIUS_SQ: f64 = 0.073714_84375 * 0.073714_84375;

/// Tests whether `c` lies in the main cardioid or the period-2 bulb, the
/// two largest components of the Mandelbrot set's interior.
///
/// The cardioid boundary satisfies `q(q + (x - 1/4)) = y^2/4` where
/// `q = (x - 1/4)^2 + y^2`, the standard algebraic parametrization of the
/// main cardioid; the period-2 bulb is the disk of radius 1/4 centered at
/// `-1`. Both are exact, not approximations — only the smaller bulbs
/// further out are left to plain iteration or [`in_period3_bulb`].
#[inline]
pub fn in_cardioid_or_period2(cr: f64, ci: f64) -> bool {
    let q = (cr - 0.25) * (cr - 0.25) + ci * ci;
    if q * (q + cr - 0.25) < 0.25 * ci * ci {
        return true;
    }
    (cr + 1.0) * (cr + 1.0) + ci * ci < 0.0625
}


/// Tests whether `c` lies in either of the two period-3 bulbs attached to
/// the main cardioid (symmetric about the real axis). These are the next
/// largest interior components after the cardioid and period-2 bulb, so
/// checking them catches a further set of pixels that would otherwise
/// iterate to `max_iter`.
#[inline]
pub fn in_period3_bulb(cr: f64, ci: f64) -> bool {
    let dr = cr - PERIOD3_CENTER_RE;
    let di_pos = ci - PERIOD3_CENTER_IM;
    let di_neg = ci + PERIOD3_CENTER_IM;
    (dr * dr + di_pos * di_pos < PERIOD3_RADIUS_SQ) || (dr * dr + di_neg * di_neg < PERIOD3_RADIUS_SQ)
}
/// SIMD (8-lane f32) form of [`in_cardioid_or_period2`] combined with
/// [`in_period3_bulb`]. The cardioid/bulb-2 test vectorizes directly; the
/// period-3 test does not (it is two disk checks with an OR that does not
/// map cleanly onto the same comparison mask), so it is evaluated in
/// scalar f64 per lane, but only for lanes the vectorized test missed.
#[inline]
pub fn bulb_precheck_x8(cr: wide::f32x8, ci: wide::f32x8) -> [bool; 8] {
    use wide::{f32x8, CmpLt};

    let quarter = f32x8::splat(0.25);
    let ci_sq = ci * ci;
    let x_offset = cr - quarter;
    let q = x_offset.mul_add(x_offset, ci_sq);
    let cardioid_in = (q * (q + x_offset)).cmp_lt(quarter * ci_sq);
    let x_plus = cr + f32x8::splat(1.0);
    let bulb_in = x_plus.mul_add(x_plus, ci_sq).cmp_lt(f32x8::splat(0.0625));
    let vec_in_set: [f32; 8] = (cardioid_in | bulb_in).into();

    let cr_arr: [f32; 8] = cr.into();
    let ci_arr: [f32; 8] = ci.into();
    let mut in_set = [false; 8];
    for lane in 0..8 {
        if vec_in_set[lane].to_bits() != 0 {
            in_set[lane] = true;
            continue;
        }
        if in_period3_bulb(cr_arr[lane] as f64, ci_arr[lane] as f64) {
            in_set[lane] = true;
        }
    }
    in_set
}
/// 4-lane f64 counterpart of [`bulb_precheck_x8`], used by the f64 SIMD
/// Mandelbrot path once the render has crossed
/// [`crate::fractal::fractal::F32_PRECISION_THRESHOLD`].
pub(crate) fn bulb_precheck_x4(cr: wide::f64x4, ci: wide::f64x4) -> [bool; 4] {
    use wide::{f64x4, CmpLt};

    let quarter = f64x4::splat(0.25);
    let ci_sq = ci * ci;
    let x_offset = cr - quarter;
    let q = x_offset.mul_add(x_offset, ci_sq);
    let cardioid_in = (q * (q + x_offset)).cmp_lt(quarter * ci_sq);
    let x_plus = cr + f64x4::splat(1.0);
    let bulb_in = x_plus.mul_add(x_plus, ci_sq).cmp_lt(f64x4::splat(0.0625));
    let vec_in_set: [f64; 4] = (cardioid_in | bulb_in).into();

    let cr_arr: [f64; 4] = cr.into();
    let ci_arr: [f64; 4] = ci.into();
    let mut in_set = [false; 4];
    for lane in 0..4 {
        if vec_in_set[lane].to_bits() != 0 {
            in_set[lane] = true;
            continue;
        }
        if in_period3_bulb(cr_arr[lane], ci_arr[lane]) {
            in_set[lane] = true;
        }
    }
    in_set
}