//! wgpu compute-shader backend (`fractal.wgsl`): portable across GPU
//! vendors via Vulkan/Metal/DX12, but WGSL has no f64 type, so this
//! backend always renders in f32 regardless of zoom.

pub mod fractal_compute;
pub mod uniforms;
