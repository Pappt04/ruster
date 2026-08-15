/// Per-frame constants uploaded to the WGSL compute shader's uniform
/// buffer for direct escape-time rendering. Layout must match the
/// `Uniforms` struct declared in `fractal.wgsl` field-for-field — GPU
/// uniform buffers have no reflection, so a mismatch silently
/// misinterprets bytes rather than failing to compile or link.
/// `_pad` rounds the struct to WGSL's 16-byte uniform alignment.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    pub re_start : f32,
    pub im_start : f32,
    pub re_step  : f32,
    pub im_step  : f32,
    pub julia_cr : f32,
    pub julia_ci : f32,
    pub max_iter : u32,
    pub fractal  : u32,
    pub width    : u32,
    pub height   : u32,
    pub _pad     : [u32; 2],
}

/// Uniform buffer layout for the WGSL perturbation-rendering entry point:
/// carries the reference orbit's center and length instead of a Julia
/// constant, since the orbit itself (`RefOrbit`) is uploaded separately as
/// a storage buffer. Must match `fractal.wgsl`'s `PerturbUniforms`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PerturbUniforms {
    pub re_start  : f32,
    pub im_start  : f32,
    pub re_step   : f32,
    pub im_step   : f32,
    pub ref_re    : f32,
    pub ref_im    : f32,
    pub orbit_len : u32,
    pub max_iter  : u32,
    pub width     : u32,
    pub height    : u32,
    pub _pad      : [u32; 2],
}