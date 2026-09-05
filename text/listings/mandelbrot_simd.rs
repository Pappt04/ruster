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
