pub mod fractal;
pub mod fractal_type;

pub use fractal::{render, render_mariani_silver, render_perturbation, render_perturbation_sa, compute_reference_orbit, compute_series_approx, RefOrbit, SeriesApprox, pixel, flops_per_iter, IterBuf, F32_PRECISION_THRESHOLD};
pub use fractal::{render_simd, render_simd_f32};
pub use fractal_type::FractalType;