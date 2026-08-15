use crate::fractal::fractal_type::FractalType;
use crate::fractal::mandelbrot::mandelbrot;
use crate::fractal::julia::julia;
use crate::fractal::newton::newton;
use crate::fractal::nova::nova;
use crate::gui::viewport::Viewport;
use rayon::prelude::*;

pub const ESCAPE_RADIUS_SQ: f64 = 4.0;
pub const ESCAPE_RADIUS_SQ_F32: f32 = 4.0;

pub const F32_PRECISION_THRESHOLD: f64 = 1e6;

pub type IterBuf = Vec<f32>;

#[derive(Clone, Copy, Debug)]
pub struct PixelGrid {
    pub re_start: f64,
    pub re_step: f64,
    pub im_start: f64,
    pub im_step: f64,
}

pub fn pixel_grid(vp: &Viewport) -> PixelGrid {
    let aspect = vp.width as f64 / vp.height as f64;
    let half = 2.0 / vp.zoom;
    let re_step = half * aspect * 2.0 / vp.width as f64;
    let im_step = half * 2.0 / vp.height as f64;
    let re_start = vp.center[0] + (0.5 / vp.width as f64 - 0.5) * half * aspect * 2.0;
    let im_start = vp.center[1] + (0.5 / vp.height as f64 - 0.5) * half * 2.0;
    PixelGrid { re_start, re_step, im_start, im_step }
}

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

#[inline]
fn compute_capped(fractal: FractalType, re: f64, im: f64, julia_c: [f64; 2], cap: u32, true_max: u32) -> f32 {
    let guess = compute(fractal, re, im, julia_c, cap);
    if guess >= cap as f32 && cap < true_max {
        compute(fractal, re, im, julia_c, true_max)
    } else {
        guess
    }
}

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

pub fn pixel(fractal: FractalType, re: f64, im: f64, julia_c: [f64; 2], max_iter: u32) -> f32 {
    compute(fractal, re, im, julia_c, max_iter)
}

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

pub(crate) fn smooth_iter(iter: u32, zn_sq: f64, max_iter: u32) -> f32 {
    if iter >= max_iter {
        return max_iter as f32;
    }
    const INV_LN2: f64 = std::f64::consts::LOG2_E;
    let log_zn = zn_sq.ln() / 2.0;
    let nu = (log_zn * INV_LN2).ln() * INV_LN2;
    (iter as f64 + 1.0 - nu) as f32
}

pub(crate) fn smooth_iter_f32(iter: u32, zn_sq: f32, max_iter: u32) -> f32 {
    if iter >= max_iter { return max_iter as f32; }
    const INV_LN2: f32 = std::f32::consts::LOG2_E;
    let log_zn = zn_sq.ln() / 2.0;
    let nu = (log_zn * INV_LN2).ln() * INV_LN2;
    iter as f32 + 1.0 - nu
}