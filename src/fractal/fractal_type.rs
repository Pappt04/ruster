#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum FractalType {
    #[default]
    Nova,
    Newton,
    Mandelbrot,
    Julia,
}

impl FractalType {
    pub const ALL: &'static [Self] = &[Self::Nova, Self::Newton, Self::Mandelbrot, Self::Julia];
    pub const fn name(self) -> &'static str {
        match self {
            Self::Mandelbrot => "Mandelbrot",
            Self::Julia => "Julia Set",
            Self::Newton => "Newton",
            Self::Nova => "Nova",
        }
    }

    pub const fn default_center(self) -> [f64; 2] {
        match self {
            Self::Mandelbrot => [-0.5, 0.0],
            Self::Julia => [0.0, 0.0],
            Self::Newton => [-0.4, -0.6],
            Self::Nova => [0.0, 0.0],
        }
    }
}
