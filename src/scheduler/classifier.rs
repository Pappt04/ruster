
use crate::fractal::fractal_type::FractalType;
use crate::fractal::{pixel, PixelGrid};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileKind {
    Gpu,
    Cpu,
}

struct PartitionCtx<'a> {
    pg: &'a PixelGrid,
    fractal: FractalType,
    julia_c: [f64; 2],
    max_iter: u32,
    min_tile: u32,
    threshold: f32,
}

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

pub fn partition_frame(
    pg: &PixelGrid,
    fractal: FractalType,
    julia_c: [f64; 2],
    max_iter: u32,
    width: u32,
    height: u32,
    max_tile: u32,
    min_tile: u32,
    threshold: f32,
) -> (Vec<[u32; 4]>, Vec<[u32; 4]>) {
    use rayon::prelude::*;

    let min_tile = min_tile.max(1);
    let max_tile = max_tile.max(min_tile);
    let ctx = PartitionCtx { pg, fractal, julia_c, max_iter, min_tile, threshold };

    let mut cells: Vec<(u32, u32, u32, u32)> = Vec::new();
    let mut y0 = 0u32;
    while y0 < height {
        let th = max_tile.min(height - y0);
        let mut x0 = 0u32;
        while x0 < width {
            let tw = max_tile.min(width - x0);
            cells.push((x0, y0, tw, th));
            x0 += tw;
        }
        y0 += th;
    }

    let per_cell: Vec<(Vec<[u32; 4]>, Vec<[u32; 4]>)> = cells
        .par_iter()
        .map(|&(x0, y0, tw, th)| {
            let corners = sample_corners(&ctx, x0, y0, tw, th);
            let mut gpu_tiles = Vec::new();
            let mut cpu_tiles = Vec::new();
            partition_tile(&ctx, x0, y0, tw, th, corners, &mut gpu_tiles, &mut cpu_tiles);
            (gpu_tiles, cpu_tiles)
        })
        .collect();

    let mut gpu_tiles = Vec::new();
    let mut cpu_tiles = Vec::new();
    for (mut g, mut c) in per_cell {
        gpu_tiles.append(&mut g);
        cpu_tiles.append(&mut c);
    }

    (gpu_tiles, cpu_tiles)
}

fn partition_tile(
    ctx: &PartitionCtx,
    x0: u32, y0: u32, tw: u32, th: u32,
    corners: [f32; 4],
    gpu_tiles: &mut Vec<[u32; 4]>,
    cpu_tiles: &mut Vec<[u32; 4]>,
) {
    if corner_spread(corners, ctx.max_iter) < ctx.threshold {
        gpu_tiles.push([x0, y0, tw, th]);
        return;
    }
    if tw <= ctx.min_tile || th <= ctx.min_tile {
        cpu_tiles.push([x0, y0, tw, th]);
        return;
    }

    let [tl, tr, bl, br] = corners;

    if tw >= th {
        let left_w = tw / 2;
        let right_w = tw - left_w;
        let mid = x0 + left_w;
        let re_left_edge = ctx.pg.re_start + (mid - 1) as f64 * ctx.pg.re_step;
        let re_right_edge = ctx.pg.re_start + mid as f64 * ctx.pg.re_step;
        let im_top = ctx.pg.im_start + y0 as f64 * ctx.pg.im_step;
        let im_bot = ctx.pg.im_start + (y0 + th - 1) as f64 * ctx.pg.im_step;
        let left_tr = pixel(ctx.fractal, re_left_edge, im_top, ctx.julia_c, ctx.max_iter);
        let left_br = pixel(ctx.fractal, re_left_edge, im_bot, ctx.julia_c, ctx.max_iter);
        let right_tl = pixel(ctx.fractal, re_right_edge, im_top, ctx.julia_c, ctx.max_iter);
        let right_bl = pixel(ctx.fractal, re_right_edge, im_bot, ctx.julia_c, ctx.max_iter);

        partition_tile(ctx, x0, y0, left_w, th, [tl, left_tr, bl, left_br], gpu_tiles, cpu_tiles);
        partition_tile(ctx, mid, y0, right_w, th, [right_tl, tr, right_bl, br], gpu_tiles, cpu_tiles);
    } else {
        let top_h = th / 2;
        let bot_h = th - top_h;
        let mid = y0 + top_h;
        let im_top_edge = ctx.pg.im_start + (mid - 1) as f64 * ctx.pg.im_step;
        let im_bot_edge = ctx.pg.im_start + mid as f64 * ctx.pg.im_step;
        let re_left = ctx.pg.re_start + x0 as f64 * ctx.pg.re_step;
        let re_right = ctx.pg.re_start + (x0 + tw - 1) as f64 * ctx.pg.re_step;
        let top_bl = pixel(ctx.fractal, re_left, im_top_edge, ctx.julia_c, ctx.max_iter);
        let top_br = pixel(ctx.fractal, re_right, im_top_edge, ctx.julia_c, ctx.max_iter);
        let bot_tl = pixel(ctx.fractal, re_left, im_bot_edge, ctx.julia_c, ctx.max_iter);
        let bot_tr = pixel(ctx.fractal, re_right, im_bot_edge, ctx.julia_c, ctx.max_iter);

        partition_tile(ctx, x0, y0, tw, top_h, [tl, tr, top_bl, top_br], gpu_tiles, cpu_tiles);
        partition_tile(ctx, x0, mid, tw, bot_h, [bot_tl, bot_tr, bl, br], gpu_tiles, cpu_tiles);
    }
}
