use cudarc::driver::{CudaDevice, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use crate::fractal::fractal::RefOrbit;

pub struct CudaFractal {
    dev:            Arc<CudaDevice>,
    output:         CudaSlice<f32>,
    prepass_output: Option<CudaSlice<f32>>,
    width:          u32,
    height:         u32,
}

impl CudaFractal {
    pub fn new(width: u32, height: u32) -> Self {
        let dev = CudaDevice::new(0).expect("no CUDA device");

        let ptx = Ptx::from_src(include_str!(env!("FRACTAL_PTX")));
        dev.load_ptx(ptx, "fractal", &[
            "fractal_kernel",
            "fractal_perturb_kernel",
            "fractal_kernel_tiled",
        ]).unwrap();

        let n      = (width * height) as usize;
        let output = dev.alloc_zeros::<f32>(n).unwrap();

        Self { dev, output, prepass_output: None, width, height }
    }

    /// Full-frame render using Z-order (Morton) traversal for better L2 locality.
    ///
    /// Launched with a 1-D grid of 16×16 blocks; each block covers 256 consecutive
    /// Morton codes.  The output buffer is still row-major — only traversal changes.
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
        let cfg = self.morton_cfg(self.width, self.height);
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
    /// `orbit` is the reference orbit computed by `compute_reference_orbit[_f128]()`.
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
        // Perturbation kernel still uses the 2-D linear launch — linear access into
        // the orbit arrays already gives good coalescing.
        let cfg = LaunchConfig {
            block_dim: (16, 16, 1),
            grid_dim:  ((self.width + 15) / 16, (self.height + 15) / 16, 1),
            shared_mem_bytes: 0,
        };
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

    /// Render a coarse prepass at 1/8 resolution.
    ///
    /// Reuses `fractal_kernel` (Morton dispatch) with 8× coarser steps.
    /// The prepass output is used by the scheduler to classify tiles before
    /// dispatching full-resolution work to CPU or GPU.
    ///
    /// The internal prepass buffer is lazily allocated and grown as needed.
    pub fn render_prepass(
        &mut self,
        re_start  : f64,
        im_start  : f64,
        re_step   : f64,
        im_step   : f64,
        julia_cr  : f64,
        julia_ci  : f64,
        max_iter  : u32,
        fractal   : u32,
        prepass_w : u32,
        prepass_h : u32,
    ) -> Vec<f32> {
        let n = (prepass_w * prepass_h) as usize;

        let needs_alloc = self.prepass_output
            .as_ref()
            .map_or(true, |s| s.len() < n);
        if needs_alloc {
            self.prepass_output = Some(self.dev.alloc_zeros::<f32>(n).unwrap());
        }

        let f   = self.dev.get_func("fractal", "fractal_kernel").unwrap();
        let cfg = self.morton_cfg(prepass_w, prepass_h);
        unsafe {
            f.launch(cfg, (
                self.prepass_output.as_mut().unwrap(),
                re_start, im_start,
                re_step * 8.0, im_step * 8.0,  // 8× coarser pixel spacing
                julia_cr, julia_ci,
                max_iter, fractal,
                prepass_w, prepass_h,
            ))
        }.unwrap();

        self.dev.dtoh_sync_copy(self.prepass_output.as_ref().unwrap()).unwrap()
    }

    /// Render a subset of tiles using `fractal_kernel_tiled`.
    ///
    /// `tiles` is a slice of `[x0, y0, w, h]` descriptors.  Each tile is
    /// processed by a z-slice of the 3-D grid; threads outside the tile bounds
    /// exit early.  Results are written directly into the full output buffer at
    /// their correct row-major positions.
    ///
    /// Used by the heterogeneous scheduler (Stage 3) to send GPU-classified tiles
    /// to the GPU while the CPU handles boundary tiles concurrently.
    pub fn render_tiled(
        &mut self,
        tiles    : &[[u32; 4]],
        re_start : f64,
        im_start : f64,
        re_step  : f64,
        im_step  : f64,
        julia_cr : f64,
        julia_ci : f64,
        max_iter : u32,
        fractal  : u32,
    ) -> Vec<f32> {
        if tiles.is_empty() {
            return self.dev.dtoh_sync_copy(&self.output).unwrap();
        }

        // Flatten [x0,y0,w,h] descriptors to a u32 slice for the device.
        let flat: Vec<u32> = tiles.iter()
            .flat_map(|t| t.iter().copied())
            .collect();
        let tile_descs_dev = self.dev.htod_sync_copy(&flat).unwrap();

        // Grid must be large enough to cover the widest/tallest tile; bounds checks
        // inside the kernel discard threads beyond the actual tile extent.
        let max_tw = tiles.iter().map(|t| t[2]).max().unwrap_or(1);
        let max_th = tiles.iter().map(|t| t[3]).max().unwrap_or(1);
        let cfg = LaunchConfig {
            block_dim: (16, 16, 1),
            grid_dim:  (
                (max_tw + 15) / 16,
                (max_th + 15) / 16,
                tiles.len() as u32,
            ),
            shared_mem_bytes: 0,
        };

        let f = self.dev.get_func("fractal", "fractal_kernel_tiled").unwrap();
        unsafe {
            f.launch(cfg, (
                &mut self.output,
                &tile_descs_dev,
                re_start, im_start, re_step, im_step,
                julia_cr, julia_ci,
                max_iter, fractal,
                self.width,
            ))
        }.unwrap();

        self.dev.dtoh_sync_copy(&self.output).unwrap()
    }

    /// 1-D Morton launch config for a `w × h` image using 16×16 thread blocks.
    fn morton_cfg(&self, w: u32, h: u32) -> LaunchConfig {
        let total  = w * h;
        let blocks = (total + 255) / 256;
        LaunchConfig {
            block_dim: (16, 16, 1),
            grid_dim:  (blocks, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}
