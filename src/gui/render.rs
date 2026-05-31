use std::sync::mpsc::{self, Receiver, Sender};
use egui::ColorImage;
use crate::{
    fractal::fractal::render as compute_render,
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
            while let Ok(mut req) = req_rx.recv() {
                while let Ok(newer) = req_rx.try_recv() {
                    req = newer;
                }
                let buf = compute_render(&req.vp, req.fractal, req.julia_c, req.max_iter);
                let pixels = colorize(&buf, req.max_iter, req.scheme);
                let w = req.vp.width as usize;
                let h = req.vp.height as usize;
                let image = ColorImage::new([w, h], pixels);
                if img_tx.send(image).is_err() { break; }
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
            Ok(img) => { self.busy = false; Some(img) }
            Err(_) => None,
        }
    }
}
