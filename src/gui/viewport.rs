use num_complex::Complex;

#[derive(Clone, Debug)]
pub struct Viewport {
    pub center: [f64; 2],
    pub zoom: f64,
    pub width: u32,
    pub height: u32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            center: [-0.5, 0.0],
            zoom: (1.0),
            width: (800),
            height: (600),
        }
    }
}

impl Viewport {
    pub fn pixel_to_complex(&self, px: f64, py: f64) -> Complex<f64> {
        let middle = 2.0 / self.zoom;
        let aspect_ratio = self.get_aspect_ratio();
        let re = self.center[0] + (px / self.width as f64 - 0.5) * middle * aspect_ratio * 2.0;
        let im = self.center[1] + (py / self.height as f64 - 0.5) * middle * 2.0;
        Complex { re, im }
    }

    pub fn zoom_at(&mut self, px: f64, py: f64, factor: f64) {
        let zi = self.pixel_to_complex(px, py);
        self.zoom *= factor;

        let zi2 = self.pixel_to_complex(px, py);
        self.center[0] += zi.re - zi2.re;
        self.center[1] += zi.im - zi2.im;
    }

    pub fn pan(&mut self, dpx: f64, dpy: f64) {
        let aspect = self.get_aspect_ratio();
        let middle = 2.0 / self.zoom;
        self.center[0] -= dpx / self.width as f64 * middle * aspect * 2.0;
        self.center[1] -= dpy / self.height as f64 * middle * 2.0;
    }

    pub fn reset(&mut self, fractal: crate::fractal::fractal_type::FractalType) {
        self.center = fractal.default_center();
        self.zoom= 1.0;
    }

    #[inline(always)]
    fn get_aspect_ratio(&self) -> f64 {
        self.width as f64 / self.height as f64
    }
}
