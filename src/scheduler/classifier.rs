//! Corner-sampling classifier: recursively subdivides the frame into
//! tiles and, without computing a single interior pixel, estimates
//! whether each tile is "uniform" (its four corners have similar escape
//! counts, so the whole tile is likely deep in the set or far outside it)
//! or "divergent" (corners disagree, so the set boundary likely passes
//! through it). Uniform tiles are cheap and their per-pixel cost is
//! predictable, which suits the GPU's lockstep SIMT execution — divergent
//! escape counts within a GPU warp otherwise force every thread in the
//! warp to run for as long as its slowest member. Divergent tiles are
//! routed to the CPU instead, where each core runs its own pixels
//! independently and pays no such penalty. This upfront partition is what
//! [`crate::scheduler::render_heterogeneous`] dispatches from.

use crate::fractal::fractal_type::FractalType;
use crate::fractal::{pixel, PixelGrid};

/// Which backend a tile was routed to. Currently only used as a return
/// discriminant conceptually — [`partition_frame`] returns separate GPU/CPU
/// vectors directly rather than tagged tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileKind {
    Gpu,
    Cpu,
}

/// Shared read-only parameters threaded through the recursive partition
/// calls, to avoid an ever-growing argument list.
struct PartitionCtx<'a> {
    pg: &'a PixelGrid,
    fractal: FractalType,
    julia_c: [f64; 2],
    max_iter: u32,
    min_tile: u32,
    threshold: f32,
}

/// Normalized spread of the four corner samples: `(max - min) / max_iter`,
/// so the result is comparable across different `max_iter` settings. A
/// tile is treated as "uniform enough for the GPU" when this stays below
/// [`PartitionCtx::threshold`], which
/// [`crate::scheduler::controller::ThresholdController`] tunes at runtime.
fn corner_spread(corners: [f32; 4], max_iter: u32) -> f32 {
    let mut max_val = f32::MIN;
    let mut min_val = f32::MAX;
    for &v in &corners {
        max_val = max_val.max(v);
        min_val = min_val.min(v);
    }
    (max_val - min_val) / (max_iter.max(1) as f32)
}

/// Evaluates the escape-time kernel at just the four corners of a tile —
/// the only per-pixel computation this classifier performs, keeping
/// partitioning cost negligible relative to actually rendering the frame.
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

/// Splits the frame into `max_tile`-sized cells (the largest unit a tile
/// can be), classifies each independently in parallel via
/// [`partition_tile`], and merges the resulting GPU/CPU tile lists. Cells
/// are independent of each other, so this parallelizes trivially over
/// rayon with no shared mutable state during classification.
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

/// Recursively bisects one cell along its longer axis (mirroring
/// [`crate::fractal::render::mariani_silver`]'s split rule) until either
/// its corner spread drops below threshold — classified GPU, since the
/// tile is now small and uniform enough to be predictable — or it hits
/// [`PartitionCtx::min_tile`] and is handed to the CPU regardless, since
/// tiles that small are not worth subdividing further no matter how
/// divergent they are.
///
/// Splitting reuses two of the four corner samples from the parent tile
/// (the pair on the shared edge) and only evaluates the two *new* corners
/// created by the split, rather than resampling all four for each half.
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
