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