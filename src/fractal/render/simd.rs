use crate::fractal::fractal::{pixel_grid, IterBuf};
use crate::fractal::fractal_type::FractalType;
use crate::fractal::kernels::julia::{julia, julia_x4, julia_x8};
use crate::fractal::kernels::mandelbrot::{mandelbrot, mandelbrot_x4, mandelbrot_x8, mandelbrot_x8x2};
use crate::gui::viewport::Viewport;
use rayon::prelude::*;

pub fn render_simd(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    use wide::f64x4;

    let w = vp.width as usize;
    let h = vp.height as usize;

    let pg = pixel_grid(vp);
    let mut buf = vec![0.0f32; w * h];

    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im = pg.im_start + y as f64 * pg.im_step;
        let im4 = f64x4::splat(im);
        let mut x = 0usize;

        while x + 4 <= w {
            let re = f64x4::from([
                pg.re_start + x as f64       * pg.re_step,
                pg.re_start + (x + 1) as f64 * pg.re_step,
                pg.re_start + (x + 2) as f64 * pg.re_step,
                pg.re_start + (x + 3) as f64 * pg.re_step,
            ]);
            let lanes = match fractal {
                FractalType::Mandelbrot => mandelbrot_x4(re, im4, max_iter),
                FractalType::Julia => {
                    let cr4 = f64x4::splat(julia_c[0]);
                    let ci4 = f64x4::splat(julia_c[1]);
                    julia_x4(re, im4, cr4, ci4, max_iter)
                }
                _ => unreachable!(),
            };
            row[x..x+4].copy_from_slice(&lanes);
            x += 4;
        }

        for x in x..w {
            let re = pg.re_start + x as f64 * pg.re_step;
            row[x] = match fractal {
                FractalType::Mandelbrot => mandelbrot(re, im, max_iter),
                FractalType::Julia => julia(re, im, julia_c[0], julia_c[1], max_iter),
                _ => unreachable!(),
            };
        }
    });

    buf
}

pub fn render_simd_f32(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    use wide::f32x8;

    let w = vp.width as usize;
    let h = vp.height as usize;

    let pg = pixel_grid(vp);
    let mut buf = vec![0.0f32; w * h];

    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im = (pg.im_start + y as f64 * pg.im_step) as f32;
        let im8 = f32x8::splat(im);
        let mut x = 0usize;

        while x + 8 <= w {
            let re = f32x8::from([
                (pg.re_start + (x    ) as f64 * pg.re_step) as f32,
                (pg.re_start + (x + 1) as f64 * pg.re_step) as f32,
                (pg.re_start + (x + 2) as f64 * pg.re_step) as f32,
                (pg.re_start + (x + 3) as f64 * pg.re_step) as f32,
                (pg.re_start + (x + 4) as f64 * pg.re_step) as f32,
                (pg.re_start + (x + 5) as f64 * pg.re_step) as f32,
                (pg.re_start + (x + 6) as f64 * pg.re_step) as f32,
                (pg.re_start + (x + 7) as f64 * pg.re_step) as f32,
            ]);
            let lanes = match fractal {
                FractalType::Mandelbrot => mandelbrot_x8(re, im8, max_iter),
                FractalType::Julia => {
                    let cr8 = f32x8::splat(julia_c[0] as f32);
                    let ci8 = f32x8::splat(julia_c[1] as f32);
                    julia_x8(re, im8, cr8, ci8, max_iter)
                }
                _ => unreachable!(),
            };
            row[x..x + 8].copy_from_slice(&lanes);
            x += 8;
        }

        let im_f64 = pg.im_start + y as f64 * pg.im_step;
        for x in x..w {
            let re = pg.re_start + x as f64 * pg.re_step;
            row[x] = match fractal {
                FractalType::Mandelbrot => mandelbrot(re, im_f64, max_iter),
                FractalType::Julia => julia(re, im_f64, julia_c[0], julia_c[1], max_iter),
                _ => unreachable!(),
            };
        }
    });

    buf
}

pub fn render_simd_f32_ilp(vp: &Viewport, fractal: FractalType, julia_c: [f64; 2], max_iter: u32) -> IterBuf {
    use wide::f32x8;

    let w = vp.width as usize;
    let h = vp.height as usize;

    let pg = pixel_grid(vp);
    let mut buf = vec![0.0f32; w * h];

    buf.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let im = (pg.im_start + y as f64 * pg.im_step) as f32;
        let im8 = f32x8::splat(im);
        let mut x = 0usize;

        if fractal == FractalType::Mandelbrot {
            while x + 16 <= w {
                let re0 = f32x8::from(std::array::from_fn::<f32, 8, _>(|k| {
                    (pg.re_start + (x + k) as f64 * pg.re_step) as f32
                }));
                let re1 = f32x8::from(std::array::from_fn::<f32, 8, _>(|k| {
                    (pg.re_start + (x + 8 + k) as f64 * pg.re_step) as f32
                }));
                let (lanes0, lanes1) = mandelbrot_x8x2(re0, im8, re1, im8, max_iter);
                row[x..x + 8].copy_from_slice(&lanes0);
                row[x + 8..x + 16].copy_from_slice(&lanes1);
                x += 16;
            }
        }

        while x + 8 <= w {
            let re = f32x8::from(std::array::from_fn::<f32, 8, _>(|k| {
                (pg.re_start + (x + k) as f64 * pg.re_step) as f32
            }));
            let lanes = match fractal {
                FractalType::Mandelbrot => mandelbrot_x8(re, im8, max_iter),
                FractalType::Julia => {
                    let cr8 = f32x8::splat(julia_c[0] as f32);
                    let ci8 = f32x8::splat(julia_c[1] as f32);
                    julia_x8(re, im8, cr8, ci8, max_iter)
                }
                _ => unreachable!(),
            };
            row[x..x + 8].copy_from_slice(&lanes);
            x += 8;
        }

        let im_f64 = pg.im_start + y as f64 * pg.im_step;
        for x in x..w {
            let re = pg.re_start + x as f64 * pg.re_step;
            row[x] = match fractal {
                FractalType::Mandelbrot => mandelbrot(re, im_f64, max_iter),
                FractalType::Julia => julia(re, im_f64, julia_c[0], julia_c[1], max_iter),
                _ => unreachable!(),
            };
        }
    });

    buf
}
