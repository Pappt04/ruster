use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use crate::fractal::fractal::RefOrbit;

pub struct CudaFractal {
    dev:    Arc<CudaDevice>,
    output: CudaSlice<f32>,
    width:  u32,
    height: u32,
}

impl CudaFractal {
    pub fn new(width: u32, height: u32) -> Self {
        let dev = CudaDevice::new(0).expect("no CUDA device");

        let ptx = Ptx::from_src(include_str!(env!("FRACTAL_PTX")));
        dev.load_ptx(ptx, "fractal", &["fractal_kernel", "fractal_perturb_kernel"]).unwrap();

        let n      = (width * height) as usize;
        let output = dev.alloc_zeros::<f32>(n).unwrap();

        Self { dev, output, width, height }
    }

    pub fn render(
        &mut self,
        re_start : f64,
        im_start : f64,
        re_step  : f64,
        im_step  : f64,
        julia_cr : f64,
        julia_ci : f64,
        max_iter : u32,
        fractal  : u32,
    ) -> Vec<f32> {
        let f   = self.dev.get_func("fractal", "fractal_kernel").unwrap();
        let cfg = self.launch_cfg();
        unsafe {
            f.launch(cfg, (
                &mut self.output,
                re_start, im_start, re_step, im_step,
                julia_cr, julia_ci,
                max_iter, fractal,
                self.width, self.height,
            ))
        }.unwrap();
        self.dev.dtoh_sync_copy(&self.output).unwrap()
    }

    /// Perturbation-theory render (Mandelbrot only).
    ///
    /// `orbit` is the reference orbit computed by `compute_reference_orbit()`.
    /// The orbit slice `[0..=orbit.len]` is uploaded to the device each call.
    pub fn render_perturbation(
        &mut self,
        orbit    : &RefOrbit,
        re_start : f64,
        im_start : f64,
        re_step  : f64,
        im_step  : f64,
        max_iter : u32,
    ) -> Vec<f32> {
        let orbit_slice  = orbit.len + 1;
        let orbit_re_dev = self.dev.htod_sync_copy(&orbit.zr[..orbit_slice]).unwrap();
        let orbit_im_dev = self.dev.htod_sync_copy(&orbit.zi[..orbit_slice]).unwrap();

        let f   = self.dev.get_func("fractal", "fractal_perturb_kernel").unwrap();
        let cfg = self.launch_cfg();
        unsafe {
            f.launch(cfg, (
                &mut self.output,
                &orbit_re_dev, &orbit_im_dev,
                orbit.len as u32,
                re_start, im_start, re_step, im_step,
                max_iter, self.width, self.height,
            ))
        }.unwrap();
        self.dev.dtoh_sync_copy(&self.output).unwrap()
    }

    fn launch_cfg(&self) -> LaunchConfig {
        LaunchConfig {
            block_dim: (16, 16, 1),
            grid_dim:  ((self.width + 15) / 16, (self.height + 15) / 16, 1),
            shared_mem_bytes: 0,
        }
    }
}
