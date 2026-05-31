use rayon::prelude::*;
use crate::fractal::fractal_type::FractalType;
use crate::gui::viewport::Viewport;

pub const ESCAPE_RADIUS_SQ: f64 = 256.0 * 256.0;

pub type IterBuf = Vec<f32>;

pub fn render(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    let w = vp.width as usize;
    let h = vp.height as usize;
    let mut buf = vec![0.0f32; w * h];

    buf.par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..w {
                let [re, im] = vp.pixel_to_complex(x as f64 + 0.5, y as f64 + 0.5);
                row[x] = compute(fractal, re, im, julia_c, max_iter);
            }
        });

    buf
}

fn compute(fractal: FractalType, re: f64, im: f64, julia_c: [f64; 2], max_iter: u32) -> f32 {
    match fractal {
        FractalType::Nova        => nova(re, im, max_iter),
        FractalType::Newton      => newton(re, im, max_iter),
        FractalType::Mandelbrot  => mandelbrot(re, im, max_iter),
        FractalType::Julia       => julia(re, im, julia_c[0], julia_c[1], max_iter),
    }
}

fn smooth_iter(iter: u32, zr: f64, zi: f64, max_iter: u32) -> f32 {
    if iter >= max_iter {
        return max_iter as f32;
    }
    let log_zn = (zr * zr + zi * zi).ln() / 2.0;
    let nu = (log_zn / std::f64::consts::LN_2).ln() / std::f64::consts::LN_2;
    (iter as f64 + 1.0 - nu) as f32
}

fn mandelbrot(cr: f64, ci: f64, max_iter: u32) -> f32 {
    let q = (cr - 0.25) * (cr - 0.25) + ci * ci;
    if q * (q + cr - 0.25) < 0.25 * ci * ci { return max_iter as f32; }
    if (cr + 1.0) * (cr + 1.0) + ci * ci < 0.0625 { return max_iter as f32; }

    let (mut zr, mut zi) = (0.0, 0.0);
    for i in 0..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        if zr2 + zi2 > ESCAPE_RADIUS_SQ {
            return smooth_iter(i, zr, zi, max_iter);
        }
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }
    max_iter as f32
}

fn julia(zr0: f64, zi0: f64, cr: f64, ci: f64, max_iter: u32) -> f32 {
    let (mut zr, mut zi) = (zr0, zi0);
    for i in 0..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        if zr2 + zi2 > ESCAPE_RADIUS_SQ {
            return smooth_iter(i, zr, zi, max_iter);
        }
        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
    }
    max_iter as f32
}

fn newton(cr: f64, ci: f64, max_iter: u32) -> f32 {
    let (mut zr, mut zi) = (cr, ci);
    for i in 0..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        let z3r = zr * (zr2 - 3.0 * zi2);
        let z3i = zi * (3.0 * zr2 - zi2);
        let fr = z3r - 1.0;
        let fi = z3i;
        let d_re = 3.0 * (zr2 - zi2);
        let d_im = 6.0 * zr * zi;
        let denom = d_re * d_re + d_im * d_im;
        if denom < 1e-20 { break; }
        let new_zr = zr - (fr * d_re + fi * d_im) / denom;
        let new_zi = zi - (fi * d_re - fr * d_im) / denom;
        let dr = new_zr - zr;
        let di = new_zi - zi;
        zr = new_zr;
        zi = new_zi;
        if dr * dr + di * di < 1e-12 {
            return i as f32;
        }
    }
    max_iter as f32
}

fn nova(cr: f64, ci: f64, max_iter: u32) -> f32 {
    let (mut zr, mut zi) = (1.0, 0.0);
    for i in 0..max_iter {
        let zr2 = zr * zr;
        let zi2 = zi * zi;
        let z3r = zr * (zr2 - 3.0 * zi2);
        let z3i = zi * (3.0 * zr2 - zi2);
        let fr = z3r - 1.0;
        let fi = z3i;
        let d_re = 3.0 * (zr2 - zi2);
        let d_im = 6.0 * zr * zi;
        let denom = d_re * d_re + d_im * d_im;
        if denom < 1e-20 { break; }
        let new_zr = zr - (fr * d_re + fi * d_im) / denom + cr;
        let new_zi = zi - (fi * d_re - fr * d_im) / denom + ci;
        let dr = new_zr - zr;
        let di = new_zi - zi;
        zr = new_zr;
        zi = new_zi;
        if dr * dr + di * di < 1e-12 {
            return i as f32;
        }
    }
    max_iter as f32
}