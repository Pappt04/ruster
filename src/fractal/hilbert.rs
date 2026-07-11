//! Hilbert-curve pixel ordering for cache-friendly tile traversal (2b in
//! CURSOR_OPTIMIZATIONS.md). Standard bit-rotation algorithm.

/// Side length of one traversal tile: 2^6 = 64.
pub const TILE: usize = 64;
const ORDER: u32 = 6;

/// Converts a Hilbert-curve distance `d` (0..4^order) to (x,y) within a
/// 2^order × 2^order square.
pub fn d2xy(d: usize) -> (u32, u32) {
    let mut rx: u32;
    let mut ry: u32;
    let mut t = d as u32;
    let mut x = 0u32;
    let mut y = 0u32;
    let mut s = 1u32;
    while s < (1u32 << ORDER) {
        rx = 1 & (t / 2);
        ry = 1 & (t ^ rx);
        // rotate
        if ry == 0 {
            if rx == 1 {
                x = s.wrapping_sub(1).wrapping_sub(x);
                y = s.wrapping_sub(1).wrapping_sub(y);
            }
            std::mem::swap(&mut x, &mut y);
        }
        x += s * rx;
        y += s * ry;
        t /= 4;
        s *= 2;
    }
    (x, y)
}

/// Precomputed Hilbert traversal order for one TILE×TILE tile (4096 (x,y) pairs).
pub fn tile_order() -> Vec<(u32, u32)> {
    (0..TILE * TILE).map(d2xy).collect()
}
