pub mod fractal;
pub mod fractal_type;
pub mod hilbert;
pub mod analysis;

pub use fractal::{render, render_mariani_silver, render_perturbation, render_perturbation_sa, compute_reference_orbit, compute_reference_orbit_f128, compute_series_approx, RefOrbit, SeriesApprox, pixel, flops_per_iter, IterBuf, F32_PRECISION_THRESHOLD, F128_ZOOM_THRESHOLD, PixelGrid, pixel_grid, render_neighbor_capped, in_period3_bulb};
pub use fractal::{render_simd, render_simd_f32, render_simd_f32_ilp, render_tiled, shift_and_fill, render_perturbation_multiref, RefOrbitSet, MAX_REFS, perturb_mandelbrot_flagged, render_perturbation_rebase, mandelbrot_dem, render_mariani_silver_dem, mandelbrot_ide, render_ide_biased, render_tile_exact};
pub use fractal_type::FractalType;
pub use analysis::{box_count_dimension, estimate_area, iteration_histogram, BoxCountResult};