//! GPU render backends: wgpu (portable, f32-only) always available, and an
//! optional CUDA backend targeting the discrete NVIDIA GPU directly, built
//! only when the `cuda` feature is enabled (`build.rs` compiles
//! `fractal.cu` to PTX for it).

pub mod wgpu;

#[cfg(feature = "cuda")]
pub mod cuda;
