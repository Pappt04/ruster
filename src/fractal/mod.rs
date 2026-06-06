pub mod fractal;
pub mod fractal_type;

pub use fractal::{render, pixel, flops_per_iter, IterBuf};
pub use fractal_type::FractalType;