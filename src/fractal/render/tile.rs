//! Renders a single rectangular tile of an already-computed [`PixelGrid`]
//! rather than a full frame. This is the unit of work the heterogeneous
//! scheduler dispatches to CPU workers: tiles are small enough to balance
//! load across threads and GPU launches, while each tile call still
//! amortizes its own setup cost over many pixels.

use crate::fractal::fractal::{compute, PixelGrid, F32_PRECISION_THRESHOLD};
use crate::fractal::fractal_type::FractalType;
use crate::fractal::kernels::julia::{julia, julia_x8};
use crate::fractal::kernels::mandelbrot::{mandelbrot, mandelbrot_x8};

/// `tile = [x0, y0, width, height]` in pixel coordinates. Allocates and
/// returns a fresh buffer; see [`render_tile_exact_into`] for the
/// caller-owned-buffer form used by the scheduler's hot path.
pub fn render_tile_exact(pg: &PixelGrid, fractal: FractalType, julia_c: [f64; 2], max_iter: u32, tile: [u32; 4]) -> Vec<f32> {
    let [_, _, tw, th] = tile;
    let mut local = vec![0.0f32; (tw * th) as usize];
    render_tile_exact_into(pg, fractal, julia_c, max_iter, tile, &mut local);
    local
}

/// Scalar f64 tile render into a caller-owned buffer, one worker-owned
/// scratch allocation reused across many tiles instead of one `Vec` per
/// tile call.
pub fn render_tile_exact_into(
    pg: &PixelGrid, fractal: FractalType, julia_c: [f64; 2], max_iter: u32,
    tile: [u32; 4], out: &mut [f32],
) {
    let [x0, y0, tw, th] = tile;
    assert_eq!(out.len(), (tw * th) as usize, "tile buffer size mismatch");
    for ly in 0..th {
        let im = pg.im_start + (y0 + ly) as f64 * pg.im_step;
        for lx in 0..tw {
            let re = pg.re_start + (x0 + lx) as f64 * pg.re_step;
            out[(ly * tw + lx) as usize] = compute(fractal, re, im, julia_c, max_iter);
        }
    }
}

/// Allocating form of [`render_tile_exact_simd_into`].
pub fn render_tile_exact_simd(pg: &PixelGrid, fractal: FractalType, julia_c: [f64; 2], max_iter: u32, tile: [u32; 4]) -> Vec<f32> {
    let [_, _, tw, th] = tile;
    let mut local = vec![0.0f32; (tw * th) as usize];
    render_tile_exact_simd_into(pg, fractal, julia_c, max_iter, tile, &mut local);
    local
}

/// 8-lane f32 SIMD tile render (Mandelbrot/Julia only) into a caller-owned
/// buffer, with a scalar f64 tail for the last `tw % 8` pixels of each
/// row. Rows are written directly into `out`'s tile-local layout rather
/// than the full-frame stride used by [`render_tile_exact_into`].
pub fn render_tile_exact_simd_into(
    pg: &PixelGrid, fractal: FractalType, julia_c: [f64; 2], max_iter: u32,
    tile: [u32; 4], out: &mut [f32],
) {
    use wide::f32x8;

    let [x0, y0, tw, th] = tile;
    assert_eq!(out.len(), (tw * th) as usize, "tile buffer size mismatch");
    let local = out;

    for ly in 0..th {
        let y = y0 + ly;
        let im = (pg.im_start + y as f64 * pg.im_step) as f32;
        let im8 = f32x8::splat(im);
        let row = &mut local[(ly * tw) as usize..(ly * tw + tw) as usize];
        let mut lx = 0u32;

        while lx + 8 <= tw {
            let x = x0 + lx;
            let re = f32x8::from(std::array::from_fn::<f32, 8, _>(|k| {
                (pg.re_start + (x + k as u32) as f64 * pg.re_step) as f32
            }));
            let lanes = match fractal {
                FractalType::Mandelbrot => mandelbrot_x8(re, im8, max_iter),
                FractalType::Julia => {
                    let cr8 = f32x8::splat(julia_c[0] as f32);
                    let ci8 = f32x8::splat(julia_c[1] as f32);
                    julia_x8(re, im8, cr8, ci8, max_iter)
                }
                _ => unreachable!("render_tile_exact_simd only supports Mandelbrot/Julia"),
            };
            row[lx as usize..lx as usize + 8].copy_from_slice(&lanes);
            lx += 8;
        }

        let im_f64 = pg.im_start + y as f64 * pg.im_step;
        while lx < tw {
            let x = x0 + lx;
            let re = pg.re_start + x as f64 * pg.re_step;
            row[lx as usize] = match fractal {
                FractalType::Mandelbrot => mandelbrot(re, im_f64, max_iter),
                FractalType::Julia => julia(re, im_f64, julia_c[0], julia_c[1], max_iter),
                _ => unreachable!("render_tile_exact_simd only supports Mandelbrot/Julia"),
            };
            lx += 1;
        }
    }
}

/// Allocating form of [`render_cpu_tile_into`].
pub fn render_cpu_tile(pg: &PixelGrid, fractal: FractalType, julia_c: [f64; 2], max_iter: u32, tile: [u32; 4], use_simd: bool, zoom: f64) -> Vec<f32> {
    let [_, _, tw, th] = tile;
    let mut out = vec![0.0f32; (tw * th) as usize];
    render_cpu_tile_into(pg, fractal, julia_c, max_iter, tile, use_simd, zoom, &mut out);
    out
}


/// Dispatches a tile to the SIMD or scalar CPU renderer based on caller
/// preference and precision requirements: SIMD is only valid for
/// Mandelbrot/Julia, and only below [`F32_PRECISION_THRESHOLD`], since past
/// that zoom f32 coordinates round together and the tile would render
/// wrong. This is the entry point the scheduler's CPU worker calls per
/// tile.
#[allow(clippy::too_many_arguments)]
pub fn render_cpu_tile_into(
    pg: &PixelGrid, fractal: FractalType, julia_c: [f64; 2], max_iter: u32,
    tile: [u32; 4], use_simd: bool, zoom: f64, out: &mut [f32],
) {
    if use_simd
        && matches!(fractal, FractalType::Mandelbrot | FractalType::Julia)
        && zoom < F32_PRECISION_THRESHOLD
    {
        render_tile_exact_simd_into(pg, fractal, julia_c, max_iter, tile, out)
    } else {
        render_tile_exact_into(pg, fractal, julia_c, max_iter, tile, out)
    }
}
