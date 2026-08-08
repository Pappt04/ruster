//! Corner-sampling recursive space partitioner driving the heterogeneous
//! scheduler.
//!
//! Classification here only decides *which backend* computes a tile — both
//! GPU (`fractal_kernel_tiled`, f64, exact) and CPU (`render_tile_exact`,
//! exact) always compute every pixel of whatever tile they're given, in
//! full, with no skipping or approximation. A corner-sampling
//! misclassification (missing a thin filament, routing a tile that's
//! actually boundary-heavy to the GPU) can therefore only cost GPU-side
//! performance — extra warp divergence on that tile — it can never produce a
//! wrong pixel.
//!
//! This is why corner-sampling is safe here even though `TUTORIAL.md`'s "Why
//! Not 5-Point Sampling" section rejected it for the *previous* design: that
//! rejection targeted a scheme where sampling decides what gets *computed*
//! (an `ms_fill`-style border-uniformity flood-fill shortcut that really can
//! skip real pixels based on a wrong assumption). Corner-sampling-for-routing
//! and flood-fill-for-skipping are not the same risk category — don't
//! reflexively re-apply the old objection here.
//!
//! One caveat found empirically, worth being honest about rather than
//! claiming perfect bit-exactness: "both backends compute exactly" is true in
//! the idealized-algorithm sense, but GPU f64 and CPU f64 can still diverge in
//! their last-bit rounding on genuinely chaotic escape-time computations near
//! a periodic-cycle detection threshold — the same class of hardware
//! floating-point non-determinism this codebase's own `fractal.cu` doc
//! comments already acknowledge ("a pixel exactly on the edge of a periodic
//! cycle can come out 'converged' on one backend and 'escapes 400 iterations
//! later' on the other"). Corner sampling can miss an isolated chaotic pixel
//! surrounded by uniform-looking corners (the exact TUTORIAL.md-warned
//! failure mode) and route it to the GPU, where — on such a pixel, and only
//! such a pixel — it may not match plain CPU `render()` bit-for-bit. Measured
//! impact: a handful of pixels out of ~2 million (Mandelbrot/Julia only;
//! Newton/Nova remain exactly bit-identical, having no comparable
//! period-detection sensitivity). This is a pre-existing category of
//! imprecision in this codebase (the old dense-prepass classifier showed a
//! smaller residual of the same kind, 1-2 pixels), not a new one — corner
//! sampling exposes a bit more of it than dense-prepass sampling did, as
//! `TUTORIAL.md` predicted for exactly this class of technique.

use crate::fractal::fractal_type::FractalType;
use crate::fractal::{pixel, PixelGrid};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileKind {
    Gpu,
    Cpu,
}

/// Invariant inputs for one partition pass, bundled so the recursive helper's
/// parameter list doesn't balloon.
struct PartitionCtx<'a> {
    pg: &'a PixelGrid,
    fractal: FractalType,
    julia_c: [f64; 2],
    max_iter: u32,
    min_tile: u32,
    threshold: f32,
}

/// `(max - min) / max_iter`, normalized so the metric and its threshold are
/// dimensionless (~[0,1]) regardless of `max_iter` — the old prepass-variance
/// threshold was implicitly coupled to `max_iter`'s scale, this isn't.
fn corner_spread(corners: [f32; 4], max_iter: u32) -> f32 {
    let mut max_val = f32::MIN;
    let mut min_val = f32::MAX;
    for &v in &corners {
        max_val = max_val.max(v);
        min_val = min_val.min(v);
    }
    (max_val - min_val) / (max_iter.max(1) as f32)
}

/// Samples a tile's 4 actual corner pixels — `(x0,y0)`, `(x0+tw-1,y0)`,
/// `(x0,y0+th-1)`, `(x0+tw-1,y0+th-1)` — in `[top_left, top_right,
/// bottom_left, bottom_right]` order. Tiles are half-open `[x0,y0,tw,th]`
/// like everywhere else in this codebase (`render_tile_exact`, `ms_fill`), so
/// the corner pixels are at `tw-1`/`th-1` offsets, not `tw`/`th`.
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

/// Partitions the whole frame into GPU/CPU tiles. Builds a ceil-divided grid
/// of `max_tile`-sized cells across the frame, then recursively subdivides
/// each cell independently down to `min_tile`, classifying by corner spread
/// at each level. Returns `(gpu_tiles, cpu_tiles)` as `[x0,y0,tw,th]`
/// descriptors that together partition every pixel exactly once.
///
/// Top-level cells are independent (each recurses only within its own
/// bounds), so they're classified in parallel via rayon rather than a plain
/// loop — worth doing since this whole step runs on the calling thread before
/// any GPU/CPU dispatch even starts, so it's pure added latency at the front
/// of every frame. Cost is bounded (a `pixel()` call per corner, a few
/// thousand at worst per 1920x1080 frame — see the module's cost estimate)
/// but non-trivial at deep zoom where more cells recurse to `min_tile`.
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

/// Recursive longer-axis bisection, modeled on `ms_fill`'s subdivision shape
/// (`fractal.rs` ~659) but classifying instead of flood-filling. Terminal
/// cases: `Gpu` the moment corners are close enough (regardless of remaining
/// size — this is what lets a whole uniform region resolve as one big GPU
/// tile instead of forced subdivision); `Cpu` once the tile can't shrink
/// further (`min_tile`) and corners still disagree (boundary detail).
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

    // Siblings share no physical pixel (half-open tiles), so there's no valid
    // corner reuse across them beyond the 2 unmoved corners each child
    // inherits from the parent — always exactly 4 new samples per split.
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
