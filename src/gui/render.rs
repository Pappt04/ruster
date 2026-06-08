use std::sync::mpsc::{self, Receiver, Sender};
use egui::ColorImage;
use crate::{
    fractal::fractal_type::FractalType,
    gui::color::{colorize, ColorScheme},
    gui::viewport::Viewport,
};

pub struct RenderRequest {
    pub vp: Viewport,
    pub fractal: FractalType,
    pub julia_c: [f64; 2],
    pub max_iter: u32,
    pub scheme: ColorScheme,
    pub use_ms: bool,
}

pub struct RenderWorker {
    tx: Sender<RenderRequest>,
    pub rx: Receiver<ColorImage>,
    pub busy: bool,
}

impl RenderWorker {
    pub fn new() -> Self {
        let (req_tx, req_rx) = mpsc::channel::<RenderRequest>();
        let (img_tx, img_rx) = mpsc::channel::<ColorImage>();

        std::thread::spawn(move || {
            #[cfg(feature = "cuda")]
            {
                use crate::gpu::cuda::CudaFractal;
                let mut compute: Option<CudaFractal> = None;
                let mut cached_size = (0u32, 0u32);

                while let Ok(mut req) = req_rx.recv() {
                    while let Ok(newer) = req_rx.try_recv() { req = newer; }

                    let sz = (req.vp.width, req.vp.height);
                    if compute.is_none() || cached_size != sz {
                        compute = Some(CudaFractal::new(sz.0, sz.1));
                        cached_size = sz;
                    }

                    let vp = &req.vp;
                    let aspect = vp.width as f64 / vp.height as f64;
                    let half   = 2.0 / vp.zoom;

                    let buf = compute.as_mut().unwrap().render(
                        vp.center[0] + (0.5 / vp.width  as f64 - 0.5) * half * aspect * 2.0,
                        vp.center[1] + (0.5 / vp.height as f64 - 0.5) * half * 2.0,
                        half * aspect * 2.0 / vp.width  as f64,
                        half * 2.0          / vp.height as f64,
                        req.julia_c[0], req.julia_c[1],
                        req.max_iter,
                        req.fractal.as_u32(),
                    );

                    let pixels = colorize(&buf, req.max_iter, req.scheme);
                    let image  = ColorImage::new([sz.0 as usize, sz.1 as usize], pixels);
                    if img_tx.send(image).is_err() { break; }
                }
            }

            #[cfg(not(feature = "cuda"))]
            {
                use crate::fractal::fractal::{render as cpu_render, render_mariani_silver};
                #[cfg(feature = "simd")]
                use crate::fractal::fractal::{render_simd, render_simd_f32, F32_PRECISION_THRESHOLD};
                #[cfg(feature = "simd")]
                use crate::fractal::fractal_type::FractalType;

                while let Ok(mut req) = req_rx.recv() {
                    while let Ok(newer) = req_rx.try_recv() { req = newer; }

                    let buf = if req.use_ms {
                        render_mariani_silver(&req.vp, req.fractal, req.julia_c, req.max_iter)
                    } else {
                        #[cfg(feature = "simd")]
                        {
                            match req.fractal {
                                FractalType::Mandelbrot | FractalType::Julia => {
                                    if req.vp.zoom < F32_PRECISION_THRESHOLD {
                                        render_simd_f32(&req.vp, req.fractal, req.julia_c, req.max_iter)
                                    } else {
                                        render_simd(&req.vp, req.fractal, req.julia_c, req.max_iter)
                                    }
                                }
                                _ =>
                                    cpu_render(&req.vp, req.fractal, req.julia_c, req.max_iter),
                            }
                        }
                        #[cfg(not(feature = "simd"))]
                        cpu_render(&req.vp, req.fractal, req.julia_c, req.max_iter)
                    };

                    let pixels = colorize(&buf, req.max_iter, req.scheme);
                    let w = req.vp.width as usize;
                    let h = req.vp.height as usize;
                    let image = ColorImage::new([w, h], pixels);
                    if img_tx.send(image).is_err() { break; }
                }
            }
        });

        Self { tx: req_tx, rx: img_rx, busy: false }
    }

    pub fn request(&mut self, req: RenderRequest) {
        let _ = self.tx.send(req);
        self.busy = true;
    }

    pub fn poll(&mut self) -> Option<ColorImage> {
        match self.rx.try_recv() {
            Ok(img)  => { self.busy = false; Some(img) }
            Err(_)   => None,
        }
    }
}
