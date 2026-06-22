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

/// Uniforms for the perturbation-theory compute shader (`fractal_perturb.wgsl`).
/// Mandelbrot only — no julia_c / fractal discriminant needed.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PerturbUniforms {
    pub re_start  : f32,
    pub im_start  : f32,
    pub re_step   : f32,
    pub im_step   : f32,
    pub ref_re    : f32,  // reference center real
    pub ref_im    : f32,  // reference center imag
    pub orbit_len : u32,  // actual orbit length (entries used)
    pub max_iter  : u32,
    pub width     : u32,
    pub height    : u32,
    pub _pad      : [u32; 2],
}