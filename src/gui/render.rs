use std::sync::mpsc::{self, Receiver, Sender};
use egui::ColorImage;
use crate::{
    fractal::{fractal_type::FractalType, IterBuf},
    gui::color::{colorize, ColorScheme},
    gui::viewport::Viewport,
};

/// Slack added to a row-neighbor's escape count when computing the per-pixel
/// iteration cap for `render_neighbor_capped`. See CURSOR_OPTIMIZATIONS.md 1d.
const NEIGHBOR_CAP_SLACK: u32 = 16;

#[derive(Clone, PartialEq)]
struct CacheKey {
    vp: Viewport,
    fractal: FractalType,
    julia_c_bits: [u64; 2],
    max_iter: u32,
    use_ms: bool,
    use_perturbation: bool,
    use_sa: bool,
    use_neighbor_cap: bool,
    use_multiref: bool,
}

impl CacheKey {
    fn from(req: &RenderRequest) -> Self {
        Self {
            vp: req.vp.clone(),
            fractal: req.fractal,
            julia_c_bits: [req.julia_c[0].to_bits(), req.julia_c[1].to_bits()],
            max_iter: req.max_iter,
            use_ms: req.use_ms,
            use_perturbation: req.use_perturbation,
            use_sa: req.use_sa,
            use_neighbor_cap: req.use_neighbor_cap,
            use_multiref: req.use_multiref,
        }
    }

    /// True if every field except `vp` matches — i.e. the cached buffer was
    /// produced with the same fractal/params and none of the modes that are
    /// incompatible with incremental pan (MS/perturbation/SA/neighbor-cap all
    /// invalidate the "buffer values are independent of the rest of the frame"
    /// assumption incremental pan relies on). Used to gate `shift_and_fill`.
    fn pan_eligible_match(&self, other: &CacheKey) -> bool {
        self.fractal == other.fractal
            && self.julia_c_bits == other.julia_c_bits
            && self.max_iter == other.max_iter
            && !self.use_ms && !other.use_ms
            && !self.use_perturbation && !other.use_perturbation
            && !self.use_sa && !other.use_sa
            && !self.use_neighbor_cap && !other.use_neighbor_cap
    }
}

struct FrameCache {
    key: Option<CacheKey>,
    buf: IterBuf,
}

impl FrameCache {
    fn new() -> Self {
        Self { key: None, buf: vec![] }
    }
}

pub struct RenderRequest {
    pub vp: Viewport,
    pub fractal: FractalType,
    pub julia_c: [f64; 2],
    pub max_iter: u32,
    pub scheme: ColorScheme,
    pub use_ms: bool,
    pub use_perturbation: bool,
    pub use_sa: bool,
    pub use_neighbor_cap: bool,
    pub use_multiref: bool,
    /// CUDA builds only: route tiles between CPU/GPU via the adaptive
    /// prepass-guided scheduler instead of a single full-frame GPU dispatch.
    pub use_heterogeneous: bool,
}

/// Whether a `RenderResult` is a fast, lower-resolution preview (worker still busy,
/// a `Final` result will follow) or the completed full-resolution frame. See 3c in
/// CURSOR_OPTIMIZATIONS.md.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Quality {
    Preview,
    Final,
}

pub struct RenderResult {
    pub image: ColorImage,
    pub quality: Quality,
}

pub struct RenderWorker {
    tx: Sender<RenderRequest>,
    pub rx: Receiver<RenderResult>,
    pub busy: bool,
}

impl RenderWorker {
    pub fn new() -> Self {
        let (req_tx, req_rx) = mpsc::channel::<RenderRequest>();
        let (img_tx, img_rx) = mpsc::channel::<RenderResult>();

        std::thread::spawn(move || {
            #[cfg(feature = "cuda")]
            {
                use crate::gpu::cuda::CudaFractal;
                use crate::fractal::fractal::{compute_reference_orbit, compute_reference_orbit_f128, F128_ZOOM_THRESHOLD};
                use crate::fractal::fractal_type::FractalType;
                use crate::scheduler::{self, controller::ThresholdController, SchedulerConfig};
                let mut compute: Option<CudaFractal> = None;
                let mut cached_size = (0u32, 0u32);
                let mut het_controller = ThresholdController::new(50.0);
                let scheduler_cfg = SchedulerConfig::default();

                while let Ok(mut req) = req_rx.recv() {
                    while let Ok(newer) = req_rx.try_recv() { req = newer; }

                    let sz = (req.vp.width, req.vp.height);
                    if compute.is_none() || cached_size != sz {
                        compute = Some(CudaFractal::new(sz.0, sz.1));
                        cached_size = sz;
                    }

                    let vp     = &req.vp;
                    let aspect = vp.width as f64 / vp.height as f64;
                    let half   = 2.0 / vp.zoom;
                    let re_start = vp.center[0] + (0.5 / vp.width  as f64 - 0.5) * half * aspect * 2.0;
                    let im_start = vp.center[1] + (0.5 / vp.height as f64 - 0.5) * half * 2.0;
                    let re_step  = half * aspect * 2.0 / vp.width  as f64;
                    let im_step  = half * 2.0          / vp.height as f64;
                    let cuda     = compute.as_mut().unwrap();

                    let buf = if req.use_perturbation && req.fractal == FractalType::Mandelbrot {
                        let orbit = if vp.zoom > F128_ZOOM_THRESHOLD {
                            compute_reference_orbit_f128(vp.center[0], vp.center[1], req.max_iter)
                        } else {
                            compute_reference_orbit(vp.center[0], vp.center[1], req.max_iter)
                        };
                        cuda.render_perturbation(
                            &orbit,
                            re_start, im_start, re_step, im_step,
                            req.max_iter,
                        )
                    } else if req.use_heterogeneous {
                        scheduler::render_heterogeneous(
                            vp, req.fractal, req.julia_c, req.max_iter,
                            cuda, &mut het_controller, &scheduler_cfg,
                        ).buf
                    } else {
                        cuda.render(
                            re_start, im_start, re_step, im_step,
                            req.julia_c[0], req.julia_c[1],
                            req.max_iter,
                            req.fractal.as_u32(),
                        )
                    };

                    let pixels = colorize(&buf, req.max_iter, req.scheme);
                    let image  = ColorImage::new([sz.0 as usize, sz.1 as usize], pixels);
                    // CUDA path is CPU-cache-free and single-pass for now (progressive
                    // refinement is CPU-only in this iteration — see CURSOR_OPTIMIZATIONS.md 3c).
                    if img_tx.send(RenderResult { image, quality: Quality::Final }).is_err() { break; }
                }
            }

            #[cfg(not(feature = "cuda"))]
            {
                use crate::fractal::fractal::{render as cpu_render, render_mariani_silver, render_perturbation, render_perturbation_sa, render_neighbor_capped, shift_and_fill, render_perturbation_multiref};
                #[cfg(feature = "simd")]
                use crate::fractal::fractal::{render_simd, render_simd_f32, render_simd_f32_ilp, F32_PRECISION_THRESHOLD};
                #[cfg(feature = "simd")]
                use crate::fractal::fractal_type::FractalType;

                let mut cache = FrameCache::new();

                // Dispatches to the right kernel for an arbitrary target viewport
                // (may be a half-res preview viewport, not necessarily req.vp).
                let compute_buf = |req: &RenderRequest, target_vp: &Viewport| -> IterBuf {
                    if req.use_perturbation && req.use_sa {
                        render_perturbation_sa(target_vp, req.fractal, req.julia_c, req.max_iter)
                    } else if req.use_perturbation && req.use_multiref {
                        render_perturbation_multiref(target_vp, req.fractal, req.julia_c, req.max_iter)
                    } else if req.use_perturbation {
                        render_perturbation(target_vp, req.fractal, req.julia_c, req.max_iter)
                    } else if req.use_ms {
                        render_mariani_silver(target_vp, req.fractal, req.julia_c, req.max_iter)
                    } else if req.use_neighbor_cap {
                        render_neighbor_capped(target_vp, req.fractal, req.julia_c, req.max_iter, NEIGHBOR_CAP_SLACK)
                    } else {
                        #[cfg(feature = "simd")]
                        {
                            match req.fractal {
                                FractalType::Mandelbrot => {
                                    if target_vp.zoom < F32_PRECISION_THRESHOLD {
                                        // render_simd_f32_ilp is bit-identical to
                                        // render_simd_f32, only faster (2a ILP).
                                        render_simd_f32_ilp(target_vp, req.fractal, req.julia_c, req.max_iter)
                                    } else {
                                        render_simd(target_vp, req.fractal, req.julia_c, req.max_iter)
                                    }
                                }
                                FractalType::Julia => {
                                    if target_vp.zoom < F32_PRECISION_THRESHOLD {
                                        render_simd_f32(target_vp, req.fractal, req.julia_c, req.max_iter)
                                    } else {
                                        render_simd(target_vp, req.fractal, req.julia_c, req.max_iter)
                                    }
                                }
                                _ =>
                                    cpu_render(target_vp, req.fractal, req.julia_c, req.max_iter),
                            }
                        }
                        #[cfg(not(feature = "simd"))]
                        cpu_render(target_vp, req.fractal, req.julia_c, req.max_iter)
                    }
                };

                while let Ok(mut req) = req_rx.recv() {
                    while let Ok(newer) = req_rx.try_recv() { req = newer; }

                    let w = req.vp.width;
                    let h = req.vp.height;
                    let key = CacheKey::from(&req);

                    if cache.key.as_ref() == Some(&key) {
                        // Cache hit: recolor only, skip straight to a single Final
                        // result — no preview phase needed, this is already fast.
                        let pixels = colorize(&cache.buf, req.max_iter, req.scheme);
                        let image = ColorImage::new([w as usize, h as usize], pixels);
                        if img_tx.send(RenderResult { image, quality: Quality::Final }).is_err() { break; }
                        continue;
                    }

                    // Incremental pan: same fractal/params, axis-aligned pixel-only
                    // center shift, none of MS/perturbation/SA/neighbor-cap active.
                    // shift_and_fill mutates cache.buf in place — cheap enough to
                    // skip the preview phase and go straight to Final.
                    if let Some(prev_key) = cache.key.as_ref() {
                        if prev_key.pan_eligible_match(&key) {
                            if let Some((dx, dy)) = prev_key.vp.delta_pixels(&req.vp) {
                                let in_bounds = (dx.unsigned_abs() as u32) < w && (dy.unsigned_abs() as u32) < h;
                                let axis_aligned = dx == 0 || dy == 0;
                                let nonzero = dx != 0 || dy != 0;
                                if in_bounds && axis_aligned && nonzero {
                                    shift_and_fill(
                                        &mut cache.buf, w as usize, h as usize, dx, dy,
                                        &req.vp, req.fractal, req.julia_c, req.max_iter,
                                    );
                                    cache.key = Some(key);
                                    let pixels = colorize(&cache.buf, req.max_iter, req.scheme);
                                    let image = ColorImage::new([w as usize, h as usize], pixels);
                                    if img_tx.send(RenderResult { image, quality: Quality::Final }).is_err() { break; }
                                    continue;
                                }
                            }
                        }
                    }

                    // Cache miss: quick half-res preview first, then full-res final.
                    if w >= 2 && h >= 2 {
                        let preview_vp = req.vp.with_size((w / 2).max(1), (h / 2).max(1));
                        let preview_buf = compute_buf(&req, &preview_vp);
                        let preview_pixels = colorize(&preview_buf, req.max_iter, req.scheme);
                        let preview_image = ColorImage::new(
                            [preview_vp.width as usize, preview_vp.height as usize],
                            preview_pixels,
                        );
                        if img_tx.send(RenderResult { image: preview_image, quality: Quality::Preview }).is_err() {
                            break;
                        }
                    }

                    let new_buf = compute_buf(&req, &req.vp);
                    cache.key = Some(key);
                    cache.buf = new_buf.clone();

                    let pixels = colorize(&new_buf, req.max_iter, req.scheme);
                    let image = ColorImage::new([w as usize, h as usize], pixels);
                    if img_tx.send(RenderResult { image, quality: Quality::Final }).is_err() { break; }
                }
            }
        });

        Self { tx: req_tx, rx: img_rx, busy: false }
    }

    pub fn request(&mut self, req: RenderRequest) {
        let _ = self.tx.send(req);
        self.busy = true;
    }

    /// Returns the next available result. `busy` only clears on `Quality::Final` —
    /// a `Preview` result keeps the worker marked busy so the caller keeps polling
    /// for the final frame that follows.
    pub fn poll(&mut self) -> Option<RenderResult> {
        match self.rx.try_recv() {
            Ok(result) => {
                if result.quality == Quality::Final {
                    self.busy = false;
                }
                Some(result)
            }
            Err(_) => None,
        }
    }
}
