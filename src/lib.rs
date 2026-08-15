//! NovaFractal: an interactive fractal renderer comparing CPU (scalar,
//! SIMD, perturbation-theory deep zoom), wgpu, and CUDA render backends,
//! plus a heterogeneous CPU+GPU tile scheduler. `fractal` holds the math
//! and CPU backends, `gpu` the GPU backends, `scheduler` the heterogeneous
//! dispatcher (CUDA builds only), and `gui` the interactive application.

pub mod fractal;
pub mod gui;
pub mod gpu;
#[cfg(feature = "cuda")]
pub mod scheduler;
