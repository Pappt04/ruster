use crate::fractal::fractal_type::FractalType;
use crate::fractal::kernels::mandelbrot::mandelbrot;
use crate::fractal::kernels::julia::julia;
use crate::fractal::kernels::newton::newton;
use crate::fractal::kernels::nova::nova;
use crate::gui::viewport::Viewport;
use rayon::prelude::*;

/// Squared escape radius (bailout threshold) for smooth escape-time
/// iteration: a point is classified as escaped once `|z|^2` exceeds this.
/// 4.0 (i.e. `|z| > 2`) is the standard choice for Mandelbrot/Julia — it is
/// the smallest radius that provably encloses the whole set, and it keeps
/// the smooth-coloring formula in [`smooth_iter`] well-conditioned.
pub const ESCAPE_RADIUS_SQ: f64 = 4.0;
pub const ESCAPE_RADIUS_SQ_F32: f32 = 4.0;

/// Zoom level below which the f32 fast paths (SIMD CPU kernels, CUDA
/// `fractal_kernel_f32`) retain enough mantissa bits to place each pixel
/// distinctly in the complex plane. Above this zoom, f32 rounds neighboring
/// pixels to the same coordinate, so rendering falls back to f64 (or, past
/// [`crate::fractal::perturbation::perturbation_theory::F128_ZOOM_THRESHOLD`],
/// to perturbation theory). Newton and Nova always use f64 regardless of
/// zoom, since their basins of convergence are sensitive to rounding error
/// in a way plain escape-time iteration is not.
pub const F32_PRECISION_THRESHOLD: f64 = 1e6;

/// One (possibly fractional, see [`smooth_iter`]) iteration count per pixel,
/// row-major. This is the common intermediate format every render backend
/// (CPU scalar/SIMD, wgpu, CUDA) produces; [`crate::gui::color::colorize`]
/// turns it into displayable RGB.
pub type IterBuf = Vec<f32>;

/// Affine pixel-to-complex-plane mapping for one frame, expanded into a
/// start coordinate and a constant per-pixel step for each axis. Computing
/// this once per frame and reusing it in the pixel loop avoids repeating
/// the division/multiplication in `Viewport::pixel_to_complex` for every
/// pixel.
#[derive(Clone, Copy, Debug)]
pub struct PixelGrid {
    pub re_start: f64,
    pub re_step: f64,
    pub im_start: f64,
    pub im_step: f64,
}

/// Derives the per-frame [`PixelGrid`] from the current camera state.
///
/// The view spans `2/zoom` complex units in each direction from the center
/// (so `zoom == 1` shows the classic `[-2, 2]` real range), scaled by the
/// window aspect ratio on the real axis so pixels stay square. Start
/// coordinates are offset by half a pixel so each sample falls at a pixel
/// *center* rather than its top-left corner.
pub fn pixel_grid(vp: &Viewport) -> PixelGrid {
    let aspect = vp.width as f64 / vp.height as f64;
    let half = 2.0 / vp.zoom;
    let re_step = half * aspect * 2.0 / vp.width as f64;
    let im_step = half * 2.0 / vp.height as f64;
    let re_start = vp.center[0] + (0.5 / vp.width as f64 - 0.5) * half * aspect * 2.0;
    let im_start = vp.center[1] + (0.5 / vp.height as f64 - 0.5) * half * 2.0;
    PixelGrid { re_start, re_step, im_start, im_step }
}

/// Renders a full frame at fixed `max_iter`, one iteration count per pixel.
/// Rows are distributed across the rayon thread pool; each row scans its
/// pixels left to right against the precomputed [`PixelGrid`].
pub fn render(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    let w = vp.width as usize;
    let h = vp.height as usize;

    let pg = pixel_grid(vp);
    let mut buf = vec![0.0f32; w * h];

    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im = pg.im_start + y as f64 * pg.im_step;
        for x in 0..w {
            let re = pg.re_start + x as f64 * pg.re_step;
            row[x] = compute(fractal, re, im, julia_c, max_iter);
        }
    });

    buf
}

/// Computes one pixel against `cap`, then redoes it at `true_max` only if
/// it did not escape within `cap`. Escaping points are cheap regardless of
/// the ceiling, so this only pays a double-compute penalty on points that
/// are actually near/inside the set — the pixels [`render_neighbor_capped`]
/// is trying to shortcut around are elsewhere.
#[inline]
fn compute_capped(fractal: FractalType, re: f64, im: f64, julia_c: [f64; 2], cap: u32, true_max: u32) -> f32 {
    let guess = compute(fractal, re, im, julia_c, cap);
    if guess >= cap as f32 && cap < true_max {
        compute(fractal, re, im, julia_c, true_max)
    } else {
        guess
    }
}

/// Escape-time render that caps each pixel's iteration budget using its
/// left neighbor's result plus `slack`, instead of always iterating to
/// `max_iter`.
///
/// Escape time is spatially coherent almost everywhere except at the set
/// boundary: if pixel `x-1` escaped at iteration `n`, pixel `x` usually
/// escapes within a similar number of steps. Capping at `n + slack` lets
/// most of a row exit early; [`compute_capped`] falls back to the full
/// `max_iter` on the pixels where the guess undershoots (typically right
/// at the boundary, where escape time is not locally bounded). Each row is
/// independent so the cap resets to `max_iter` at the start of every row.
pub fn render_neighbor_capped(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32, slack: u32) -> IterBuf {
    let w = vp.width as usize;
    let h = vp.height as usize;

    let pg = pixel_grid(vp);
    let mut buf = vec![0.0f32; w * h];

    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im = pg.im_start + y as f64 * pg.im_step;
        let mut cap = max_iter;
        for x in 0..w {
            let re = pg.re_start + x as f64 * pg.re_step;
            let v = compute_capped(fractal, re, im, julia_c, cap, max_iter);
            row[x] = v;
            cap = ((v as u32).saturating_add(slack)).min(max_iter);
        }
    });

    buf
}

/// Computes a single pixel outside the parallel grid loop — used by probes
/// and tests that need one iteration count without rendering a full frame.
pub fn pixel(fractal: FractalType, re: f64, im: f64, julia_c: [f64; 2], max_iter: u32) -> f32 {
    compute(fractal, re, im, julia_c, max_iter)
}

/// Floating-point operations performed per iteration of each kernel's inner
/// loop, counting real multiplications/additions/divisions in the reference
/// scalar implementation. Used to convert measured iteration throughput
/// into FLOP/s for the benchmark reports.
pub const fn flops_per_iter(fractal: FractalType) -> u64 {
    match fractal {
        FractalType::Mandelbrot | FractalType::Julia => 8,
        FractalType::Newton => 25,
        FractalType::Nova => 27,
    }
}

pub(crate) fn compute(fractal: FractalType, re: f64, im: f64, julia_c: [f64; 2], max_iter: u32) -> f32 {
    match fractal {
        FractalType::Nova => nova(re, im, max_iter),
        FractalType::Newton => newton(re, im, max_iter),
        FractalType::Mandelbrot => mandelbrot(re, im, max_iter),
        FractalType::Julia => julia(re, im, julia_c[0], julia_c[1], max_iter),
    }
}

/// Continuous (fractional) iteration count from the discrete escape count
/// `iter` and the squared modulus `zn_sq = |z|^2` at escape.
///
/// Plain integer iteration counts produce visible banding, since every
/// pixel that escapes on the same step gets the same color regardless of
/// how far past the escape radius `z` actually landed. The correction term
/// `nu = log2(log2(|z|))` interpolates between bands using how much `z`
/// overshot the bailout radius, giving a continuous value suitable for
/// smooth palette lookup. Points that never escape are clamped to
/// `max_iter` and colored as in-set.
pub(crate) fn smooth_iter(iter: u32, zn_sq: f64, max_iter: u32) -> f32 {
    if iter >= max_iter {
        return max_iter as f32;
    }
    const INV_LN2: f64 = std::f64::consts::LOG2_E;
    let log_zn = zn_sq.ln() / 2.0;
    let nu = (log_zn * INV_LN2).ln() * INV_LN2;
    (iter as f64 + 1.0 - nu) as f32
}

/// Single-precision variant of [`smooth_iter`] for the f32 fast paths
/// (SIMD CPU kernels, CUDA `fractal_kernel_f32`, WGSL). Must stay
/// algebraically identical to the f64 version — divergence here shows up
/// as a visible coloring seam at [`F32_PRECISION_THRESHOLD`].
pub(crate) fn smooth_iter_f32(iter: u32, zn_sq: f32, max_iter: u32) -> f32 {
    if iter >= max_iter { return max_iter as f32; }
    const INV_LN2: f32 = std::f32::consts::LOG2_E;
    let log_zn = zn_sq.ln() / 2.0;
    let nu = (log_zn * INV_LN2).ln() * INV_LN2;
    iter as f32 + 1.0 - nu
}