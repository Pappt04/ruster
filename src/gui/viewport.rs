#[derive(Clone, Debug)]
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
    pub fn pixel_to_complex(&self, px: f64, py: f64) -> [f64; 2] {
        let aspect = self.aspect_ratio();
        let half = 2.0 / self.zoom;
        let re = self.center[0] + (px / self.width as f64 - 0.5) * half * aspect * 2.0;
        let im = self.center[1] + (py / self.height as f64 - 0.5) * half * 2.0;
        [re, im]
    }

    pub fn zoom_at(&mut self, px: f64, py: f64, factor: f64) {
        let [re, im] = self.pixel_to_complex(px, py);
        self.zoom *= factor;
        let [re2, im2] = self.pixel_to_complex(px, py);
        self.center[0] += re - re2;
        self.center[1] += im - im2;
    }

    pub fn pan(&mut self, dpx: f64, dpy: f64) {
        let aspect = self.aspect_ratio(); 
        let half = 2.0 / self.zoom;
        self.center[0] -= dpx / self.width as f64 * half * aspect * 2.0;
        self.center[1] -= dpy / self.height as f64 * half * 2.0;
    }

    pub fn reset(&mut self, fractal: crate::fractal::fractal_type::FractalType) {
        self.center = fractal.default_center();
        self.zoom = 1.0;
    }

    #[inline(always)]
    fn aspect_ratio(&self) -> f64 {
        self.width as f64 / self.height as f64
    }
}
