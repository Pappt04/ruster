/// Identifies which escape-time or root-finding fractal a render targets.
///
/// The GPU kernels (CUDA and WGSL) dispatch on the same discriminant via
/// [`FractalType::as_u32`], so the two representations must be kept in sync.
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

    /// Discriminant passed to the GPU kernels. Must match the `FRACTAL_*`
    /// constants in `fractal.cu` and `fractal.wgsl`.
    pub const fn as_u32(self) -> u32 {
        match self {
            Self::Mandelbrot => 0,
            Self::Julia      => 1,
            Self::Newton     => 2,
            Self::Nova       => 3,
        }
    }

    /// Complex-plane point the viewport centers on when a fractal is first
    /// selected. Newton and Nova are root-finding maps with no natural
    /// "interesting" region, so they default to the origin.
    pub const fn default_center(self) -> [f64; 2] {
        match self {
            Self::Mandelbrot => [-0.5, 0.0],
            Self::Julia      => [0.0, 0.0],
            Self::Newton     => [0.0, 0.0],
            Self::Nova       => [0.0, 0.0],
        }
    }
}
