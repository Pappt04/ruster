//! The camera: complex-plane center, zoom, and output pixel dimensions.
//! [`crate::fractal::fractal::pixel_grid`] derives its per-frame pixel
//! stepping from a `Viewport`, and this module's `pixel_to_complex` is the
//! same affine mapping expressed as a one-off query rather than a
//! precomputed grid — used for UI interactions (zoom-to-cursor, pan) where
//! only one or two points need converting, not every pixel.

#[derive(Clone, Debug, PartialEq)]
pub struct Viewport {
    pub center: [f64; 2],
    pub zoom: f64,
    pub width: u32,
    pub height: u32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self { center: [-0.5, 0.0], zoom: 1.0, width: 800, height: 600 }
    }
}

impl Viewport {
    /// Same affine mapping as [`crate::fractal::fractal::pixel_grid`],
    /// evaluated at one point rather than expanded into a per-frame grid.
    pub fn pixel_to_complex(&self, px: f64, py: f64) -> [f64; 2] {
        let half = 2.0 / self.zoom;
        let re = self.center[0] + (px / self.width as f64 - 0.5) * half * self.aspect_ratio() * 2.0;
        let im = self.center[1] + (py / self.height as f64 - 0.5) * half * 2.0;
        [re, im]
    }

    /// Zooms by `factor` while keeping the complex-plane point under
    /// pixel `(px, py)` fixed on screen: converts that pixel to its
    /// complex coordinate before changing zoom, then shifts `center` by
    /// however far that same pixel now maps to after the zoom, so the
    /// point the cursor was over does not visibly drift.
    pub fn zoom_at(&mut self, px: f64, py: f64, factor: f64) {
        let [re, im] = self.pixel_to_complex(px, py);
        self.zoom *= factor;
        let [re2, im2] = self.pixel_to_complex(px, py);
        self.center[0] += re - re2;
        self.center[1] += im - im2;
    }

    /// Translates the camera by a screen-space drag of `(dpx, dpy)`
    /// pixels, converted to the equivalent complex-plane offset at the
    /// current zoom.
    pub fn pan(&mut self, dpx: f64, dpy: f64) {
        let aspect = self.aspect_ratio();
        let half = 2.0 / self.zoom;
        self.center[0] -= dpx / self.width as f64 * half * aspect * 2.0;
        self.center[1] -= dpy / self.height as f64 * half * 2.0;
    }

    /// Restores the default view for `fractal` (its own natural center,
    /// zoom reset to 1).
    pub fn reset(&mut self, fractal: crate::fractal::fractal_type::FractalType) {
        self.center = fractal.default_center();
        self.zoom = 1.0;
    }

    /// Same center/zoom, different pixel dimensions — used for progressive-refinement
    /// preview passes at a fraction of full resolution.
    pub fn with_size(&self, width: u32, height: u32) -> Viewport {
        Viewport { center: self.center, zoom: self.zoom, width, height }
    }

    /// `Some((dx_px, dy_px))` iff `self` -> `other` is a pure pixel-aligned
    /// translation at fixed zoom/size (used for incremental-pan strip recycling);
    /// `None` otherwise, which forces a full re-render. The reconstructed pixel
    /// delta must round-trip the actual center delta within a tight tolerance —
    /// sub-pixel drift must never silently misalign a recycled buffer.
    pub fn delta_pixels(&self, other: &Viewport) -> Option<(i32, i32)> {
        if self.zoom != other.zoom || self.width != other.width || self.height != other.height {
            return None;
        }
        let half = 2.0 / self.zoom;
        let aspect = self.aspect_ratio();
        let re_per_px = half * aspect * 2.0 / self.width as f64;
        let im_per_px = half * 2.0 / self.height as f64;

        let dre = other.center[0] - self.center[0];
        let dim = other.center[1] - self.center[1];
        // pan() moves the *viewport* by -delta relative to a screen drag, so the
        // pixel delta implied by a center delta is the negation of dividing by step.
        let dpx = -dre / re_per_px;
        let dpy = -dim / im_per_px;
        let dpx_round = dpx.round();
        let dpy_round = dpy.round();

        const TOL: f64 = 1e-6;
        if (dpx - dpx_round).abs() > TOL || (dpy - dpy_round).abs() > TOL {
            return None;
        }
        Some((dpx_round as i32, dpy_round as i32))
    }

    #[inline(always)]
    fn aspect_ratio(&self) -> f64 {
        self.width as f64 / self.height as f64
    }
}
