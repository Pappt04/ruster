//! Tile statistics and classification driving the heterogeneous scheduler.
//!
//! `tile_stats` reads a sub-rectangle of a coarse (1/8-res) GPU prepass buffer
//! and summarizes it; `classify` uses that summary to decide whether a tile is
//! cheap/coherent enough for the GPU or divergent enough to route to the CPU.
//! Sampling every pixel at 1/8 resolution (rather than a handful of corner
//! points) is what makes this robust — a tile where the true fractal boundary
//! cuts through it will show up as high variance even if a small number of
//! discrete sample points happen to land on the same side.

pub struct TileStats {
    pub mean: f32,
    pub variance: f32,
    pub interior_frac: f32, // fraction of samples where value >= max_iter
    pub exterior_frac: f32, // fraction of samples where value < 5.0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileKind {
    GpuInterior,
    GpuExterior,
    GpuSmooth,
    CpuBoundary,
}

/// Summarize the prepass samples covering `[tile_x, tile_x+tile_pw) x [tile_y, tile_y+tile_ph)`
/// in prepass-buffer coordinates (`prepass_w` is the prepass buffer's row stride).
pub fn tile_stats(
    prepass: &[f32],
    prepass_w: u32,
    tile_x: u32, tile_y: u32, tile_pw: u32, tile_ph: u32,
    max_iter: u32,
) -> TileStats {
    let max_val = max_iter as f32;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut n_interior = 0u32;
    let mut n_exterior = 0u32;
    let mut count = 0u32;

    let prepass_h = (prepass.len() as u32) / prepass_w.max(1);

    for py in tile_y..(tile_y + tile_ph).min(prepass_h) {
        for px in tile_x..(tile_x + tile_pw).min(prepass_w) {
            let v = prepass[(py * prepass_w + px) as usize];
            sum += v as f64;
            sum_sq += (v * v) as f64;
            if v >= max_val { n_interior += 1; }
            if v < 5.0 { n_exterior += 1; }
            count += 1;
        }
    }

    if count == 0 {
        // No prepass coverage for this tile (shouldn't happen with ceiling-divided
        // prepass dimensions, but stay conservative rather than mis-flag as flat).
        return TileStats { mean: 0.0, variance: f32::MAX, interior_frac: 0.0, exterior_frac: 0.0 };
    }

    let mean = (sum / count as f64) as f32;
    let variance = ((sum_sq / count as f64) - (mean * mean) as f64) as f32;
    TileStats {
        mean,
        variance: variance.max(0.0),
        interior_frac: n_interior as f32 / count as f32,
        exterior_frac: n_exterior as f32 / count as f32,
    }
}

pub fn classify(stats: &TileStats, threshold_variance: f32, _max_iter: u32) -> TileKind {
    if stats.interior_frac > 0.95 && stats.variance < 4.0 {
        return TileKind::GpuInterior;
    }
    if stats.exterior_frac > 0.90 && stats.variance < 25.0 {
        return TileKind::GpuExterior;
    }
    if stats.variance < threshold_variance {
        return TileKind::GpuSmooth;
    }
    TileKind::CpuBoundary
}
