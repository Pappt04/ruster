fn corner_spread(corners: [f32; 4], max_iter: u32) -> f32 {
    let mut max_val = f32::MIN;
    let mut min_val = f32::MAX;
    for &v in &corners {
        max_val = max_val.max(v);
        min_val = min_val.min(v);
    }
    (max_val - min_val) / (max_iter.max(1) as f32)
}

fn sample_corners(ctx: &PartitionCtx, x0: u32, y0: u32, tw: u32, th: u32) -> [f32; 4] {
    let re0 = ctx.pg.re_start + x0 as f64 * ctx.pg.re_step;
    let re1 = ctx.pg.re_start + (x0 + tw - 1) as f64 * ctx.pg.re_step;
    let im0 = ctx.pg.im_start + y0 as f64 * ctx.pg.im_step;
    let im1 = ctx.pg.im_start + (y0 + th - 1) as f64 * ctx.pg.im_step;
    [
        pixel(ctx.fractal, re0, im0, ctx.julia_c, ctx.max_iter),
        pixel(ctx.fractal, re1, im0, ctx.julia_c, ctx.max_iter),
        pixel(ctx.fractal, re0, im1, ctx.julia_c, ctx.max_iter),
        pixel(ctx.fractal, re1, im1, ctx.julia_c, ctx.max_iter),
    ]
}
