pub mod fractal;
pub mod fractal_type;
pub mod hilbert;
pub mod analysis;
pub mod julia;
pub mod mandelbrot;
pub mod nova;
pub mod newton;
pub mod bulb_precheck;
pub mod double_double;
pub mod perturburation_theory;

pub use fractal::{render, render_cpu_tile_into, render_tile_exact_into, render_mariani_silver, pixel, flops_per_iter, IterBuf, F32_PRECISION_THRESHOLD, PixelGrid, pixel_grid, render_neighbor_capped};
pub use fractal::{render_simd, render_simd_f32, render_simd_f32_ilp, render_tiled, shift_and_fill, render_mariani_silver_dem, mandelbrot_ide, render_ide_biased, render_tile_exact, render_tile_exact_simd, render_cpu_tile};
pub use perturburation_theory::{render_perturbation, render_perturbation_sa, compute_reference_orbit, compute_reference_orbit_f128, compute_series_approx, RefOrbit, SeriesApprox, F128_ZOOM_THRESHOLD, render_perturbation_multiref, RefOrbitSet, MAX_REFS, perturb_mandelbrot_flagged, render_perturbation_rebase};
pub use bulb_precheck::{in_period3_bulb, in_cardioid_or_period2};
pub use mandelbrot::mandelbrot_dem;
pub use fractal_type::FractalType;
pub use analysis::{box_count_dimension, estimate_area, iteration_histogram, BoxCountResult};