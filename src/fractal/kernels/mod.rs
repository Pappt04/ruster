//! Per-pixel iteration kernels: scalar, SIMD, and distance/derivative
//! variants for each supported fractal. These are the algorithmic source
//! of truth — the CUDA (`fractal.cu`) and WGSL (`fractal.wgsl`) kernels
//! implement the same recurrences and must be kept in algebraic step with
//! them.

pub mod bulb_precheck;
pub mod julia;
pub mod mandelbrot;
pub mod newton;
pub mod nova;
