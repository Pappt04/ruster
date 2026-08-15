//! Fractal math: pixel-grid setup, per-fractal iteration kernels, the CPU
//! render backends (scalar, SIMD, tiled, perturbation), and post-render
//! analysis. GPU backends live under `crate::gpu` and reuse the same
//! [`fractal_type::FractalType`] discriminant and iteration formulas.

pub mod fractal;
pub mod fractal_type;
pub mod analysis;
pub mod kernels;
pub mod render;
pub mod perturbation;

pub use fractal::{render, pixel, flops_per_iter, IterBuf, F32_PRECISION_THRESHOLD, PixelGrid, pixel_grid, render_neighbor_capped};
pub use render::simd::{render_simd, render_simd_f32, render_simd_f32_ilp};
pub use render::tile::{render_tile_exact, render_tile_exact_into, render_tile_exact_simd, render_cpu_tile, render_cpu_tile_into};
pub use render::mariani_silver::{render_mariani_silver, render_mariani_silver_dem};
pub use render::pan::shift_and_fill;
pub use render::hilbert::render_tiled;
pub use perturbation::perturbation_theory::{render_perturbation, render_perturbation_sa, compute_reference_orbit, compute_reference_orbit_f128, compute_series_approx, RefOrbit, SeriesApprox, F128_ZOOM_THRESHOLD, render_perturbation_multiref, RefOrbitSet, MAX_REFS, perturb_mandelbrot_flagged, render_perturbation_rebase};
pub use kernels::bulb_precheck::{in_period3_bulb, in_cardioid_or_period2};
pub use kernels::mandelbrot::{mandelbrot_dem, mandelbrot_ide, render_ide_biased};
pub use fractal_type::FractalType;
pub use analysis::{box_count_dimension, estimate_area, iteration_histogram, BoxCountResult};
