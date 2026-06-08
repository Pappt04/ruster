pub mod fractal;
pub mod fractal_type;

pub use fractal::{render, render_mariani_silver, pixel, flops_per_iter, IterBuf, F32_PRECISION_THRESHOLD};
#[cfg(feature = "simd")]
pub use fractal::render_simd_f32;
pub use fractal_type::FractalType;