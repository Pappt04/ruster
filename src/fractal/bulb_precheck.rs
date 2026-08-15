const PERIOD3_CENTER_RE: f64 = -0.122561_1668766536;
const PERIOD3_CENTER_IM: f64 = 0.744861_7666197442;
const PERIOD3_RADIUS_SQ: f64 = 0.073714_84375 * 0.073714_84375;

#[inline]
pub fn in_cardioid_or_period2(cr: f64, ci: f64) -> bool {
    let q = (cr - 0.25) * (cr - 0.25) + ci * ci;
    if q * (q + cr - 0.25) < 0.25 * ci * ci {
        return true;
    }
    (cr + 1.0) * (cr + 1.0) + ci * ci < 0.0625
}


#[inline]
pub fn in_period3_bulb(cr: f64, ci: f64) -> bool {
    let dr = cr - PERIOD3_CENTER_RE;
    let di_pos = ci - PERIOD3_CENTER_IM;
    let di_neg = ci + PERIOD3_CENTER_IM;
    (dr * dr + di_pos * di_pos < PERIOD3_RADIUS_SQ) || (dr * dr + di_neg * di_neg < PERIOD3_RADIUS_SQ)
}
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