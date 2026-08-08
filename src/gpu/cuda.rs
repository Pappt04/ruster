use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use crate::fractal::fractal::{RefOrbit, F32_PRECISION_THRESHOLD};

pub struct CudaFractal {
    dev:            Arc<CudaDevice>,
    output:         CudaSlice<f32>,
    prepass_output: Option<CudaSlice<f32>>,
    width:          u32,
    height:         u32,
    // Cached kernel handles — `get_func` takes an `RwLock` read + two
    // `BTreeMap` string lookups every call; cheap in isolation, but the
    // heterogeneous scheduler can call `dispatch_tiled[_f32]` twice per frame
    // (committed batch + steal mop-up), so caching removes it from what's
    // otherwise a per-frame hot path instead of a one-shot setup cost.
    fn_kernel:        CudaFunction,
    fn_kernel_f32:    CudaFunction,
    fn_perturb:       CudaFunction,
    fn_tiled:         CudaFunction,
    fn_tiled_f32:     CudaFunction,
}

impl CudaFractal {
    /// Creates a fresh CUDA context (`CudaDevice::new`) and loads the PTX
    /// module into it. Fine for the one-instance-per-process cases (the live
    /// app, `bench_runner`'s CLI subcommands) this was originally written for.
    ///
    /// A criterion run that constructs many `CudaFractal`s in one process
    /// (one per benchmark group/resolution — ~25+ across a full `cuda/`
    /// sweep) should use [`Self::from_device`] instead: repeatedly retaining
    /// a fresh primary context and reloading the PTX module that many times
    /// back-to-back was observed to destabilize the driver on this laptop's
    /// Optimus setup (a sweep hung indefinitely — 0% GPU utilization — deep
    /// into the run, while the same operations ran fine in isolation).
    pub fn new(width: u32, height: u32) -> Self {
        let dev = CudaDevice::new(0).expect("no CUDA device");
        Self::from_device(dev, width, height)
    }

    /// Like [`Self::new`], but reuses an already-constructed device/context
    /// instead of retaining a new one. The PTX module is only loaded once per
    /// device (guarded by `has_func`) — safe to call repeatedly with the same
    /// `dev` for different resolutions.
    pub fn from_device(dev: Arc<CudaDevice>, width: u32, height: u32) -> Self {
        if !dev.has_func("fractal", "fractal_kernel") {
            let ptx = Ptx::from_src(include_str!(env!("FRACTAL_PTX")));
            dev.load_ptx(ptx, "fractal", &[
                "fractal_kernel",
                "fractal_kernel_f32",
                "fractal_perturb_kernel",
                "fractal_kernel_tiled",
                "fractal_kernel_tiled_f32",
            ]).unwrap();
        }

        let n      = (width * height) as usize;
        let output = dev.alloc_zeros::<f32>(n).unwrap();

        let fn_kernel     = dev.get_func("fractal", "fractal_kernel").unwrap();
        let fn_kernel_f32 = dev.get_func("fractal", "fractal_kernel_f32").unwrap();
        let fn_perturb    = dev.get_func("fractal", "fractal_perturb_kernel").unwrap();
        let fn_tiled      = dev.get_func("fractal", "fractal_kernel_tiled").unwrap();
        let fn_tiled_f32  = dev.get_func("fractal", "fractal_kernel_tiled_f32").unwrap();

        Self {
            dev, output, prepass_output: None, width, height,
            fn_kernel, fn_kernel_f32, fn_perturb, fn_tiled, fn_tiled_f32,
        }
    }

    /// Full-frame render using Z-order (Morton) traversal for better L2 locality.
    ///
    /// Launched with a 1-D grid of 16×16 blocks; each block covers 256 consecutive
    /// Morton codes.  The output buffer is still row-major — only traversal changes.
    ///
    /// Dispatches to the f32 kernel for Mandelbrot/Julia below
    /// `F32_PRECISION_THRESHOLD` — mirroring `cpu_render_fastest`'s SIMD
    /// dispatch in bench_runner.rs. Consumer GPUs (this targets Ampere
    /// GeForce, see build.rs) run fp64 at a small fraction of fp32 throughput,
    /// so the f64 kernel below — kept for bit-exact agreement with the CPU at
    /// deep zoom, and reused by the heterogeneous scheduler's tiled dispatch —
    /// was making full-frame CUDA renders dramatically slower than both wgpu
    /// (always f32) and even the CPU at shallow zoom. Newton/Nova have no f32
    /// fast path in this codebase (same as `cpu_render_fastest`) so they
    /// always take the f64 kernel.
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
        let f32_capable = fractal == 0 || fractal == 1;
        if f32_capable {
            let zoom = 4.0 / (im_step.abs() * self.height as f64);
            if zoom < F32_PRECISION_THRESHOLD {
                return self.render_f32(re_start, im_start, re_step, im_step, julia_cr, julia_ci, max_iter, fractal);
            }
        }

        let f   = self.fn_kernel.clone();
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

    fn render_f32(
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
        let f   = self.fn_kernel_f32.clone();
        let cfg = self.morton_cfg(self.width, self.height);
        unsafe {
            f.launch(cfg, (
                &mut self.output,
                re_start as f32, im_start as f32, re_step as f32, im_step as f32,
                julia_cr as f32, julia_ci as f32,
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

        let f   = self.fn_perturb.clone();
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

        let f   = self.fn_kernel.clone();
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

        let mut out = self.dev.dtoh_sync_copy(self.prepass_output.as_ref().unwrap()).unwrap();
        out.truncate(n);
        out
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
    /// Launches `fractal_kernel_tiled` over `tiles` — launch only, no
    /// readback. No-op if `tiles` is empty (rather than leaving it to callers
    /// to remember not to call this with nothing to do).
    ///
    /// Split out from the old single-shot `render_tiled` so a caller that
    /// needs to dispatch more than once per frame (the heterogeneous
    /// scheduler's work-stealing mop-up, see `crate::scheduler`) can do so
    /// without paying `readback`'s full-frame `dtoh_sync_copy` more than once
    /// — doubling that cost would regress exactly the fixed-cost problem the
    /// scheduler rewrite exists to fix. TODO: `readback` itself still copies
    /// the *entire* frame back regardless of how many tiles were actually
    /// dispatched; a real partial/windowed copy would need either per-tile-row
    /// copies or a compacted device-side buffer — left as a follow-up, not
    /// required for the scheduler rewrite.
    ///
    /// That TODO is now measured, and it is *the* reason the heterogeneous
    /// scheduler cannot beat plain GPU rendering on this machine. At 1920x1080
    /// the full-frame copy is 1.33 ms against a ~0.28 ms kernel — **~82% of the
    /// frame** is fixed cost that shifting tiles to the CPU does not reduce
    /// (and the share grew, not shrank, when the kernel got 2x faster: see the
    /// fp64 bulb fix in fractal.cu). So the largest kernel saving the scheduler
    /// can possibly realise is ~0.01 ms at zoom 1e0 (CPU takes 3.1% of pixels)
    /// while its own machinery costs ~1.3 ms: two orders of magnitude
    /// underwater before any tuning. Fixing this copy is the single change that
    /// makes the scheduler's premise true. See results/summary.md §3.
    /// Uploads `[x0,y0,w,h]` tile descriptors and computes the launch grid
    /// shared by `dispatch_tiled`/`dispatch_tiled_f32`. Returns `None` if
    /// `tiles` is empty (nothing to launch).
    fn upload_tiles(&self, tiles: &[[u32; 4]]) -> Option<(CudaSlice<u32>, LaunchConfig)> {
        if tiles.is_empty() {
            return None;
        }
        let flat: Vec<u32> = tiles.iter().flat_map(|t| t.iter().copied()).collect();
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
        Some((tile_descs_dev, cfg))
    }

    pub fn dispatch_tiled(
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
    ) {
        let Some((tile_descs_dev, cfg)) = self.upload_tiles(tiles) else { return };

        let f = self.fn_tiled.clone();
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
    }

    /// Like [`Self::dispatch_tiled`] but uses `fractal_kernel_tiled_f32`
    /// (Mandelbrot/Julia only — see that kernel's doc comment). Callers
    /// (`crate::scheduler`, gated by `SchedulerConfig::gpu_tiles_f32`) are
    /// responsible for only calling this for fractal ids 0/1 below
    /// `F32_PRECISION_THRESHOLD`; this method doesn't re-check either
    /// condition, matching `dispatch_tiled`'s "caller decides" contract.
    pub fn dispatch_tiled_f32(
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
    ) {
        let Some((tile_descs_dev, cfg)) = self.upload_tiles(tiles) else { return };

        let f = self.fn_tiled_f32.clone();
        unsafe {
            f.launch(cfg, (
                &mut self.output,
                &tile_descs_dev,
                re_start as f32, im_start as f32, re_step as f32, im_step as f32,
                julia_cr as f32, julia_ci as f32,
                max_iter, fractal,
                self.width,
            ))
        }.unwrap();
    }

    /// Copies the persistent full-frame output buffer back to the host. Call
    /// once per frame, after all of that frame's `dispatch_tiled`/
    /// `dispatch_tiled_f32` calls.
    pub fn readback(&self) -> Vec<f32> {
        self.dev.dtoh_sync_copy(&self.output).unwrap()
    }

    /// Convenience wrapper equivalent to a single `dispatch_tiled` +
    /// `readback` — kept for any caller that only ever dispatches once.
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
        self.dispatch_tiled(tiles, re_start, im_start, re_step, im_step, julia_cr, julia_ci, max_iter, fractal);
        self.readback()
    }

    /// Full-res dimensions this instance was constructed with.
    pub fn width(&self) -> u32 { self.width }

    /// Full-res dimensions this instance was constructed with.
    pub fn height(&self) -> u32 { self.height }

    /// 1-D Morton launch config for a `w × h` image using 16×16 thread blocks.
    ///
    /// `morton_decode` interleaves a fixed 16 bits each for x and y, i.e. it
    /// bijects the codes `[0, dim*dim)` onto the square `[0,dim) x [0,dim)`
    /// for any power-of-two `dim`. The first `w*h` codes do NOT in general
    /// cover the `w x h` rectangle unless `w == h` and both are powers of
    /// two — for any other shape they land in some differently-shaped subset
    /// of the same area, silently leaving real pixels uncomputed (they keep
    /// whatever was in the output buffer, typically 0.0 from `alloc_zeros`).
    /// Padding to the smallest enclosing power-of-two *square* guarantees
    /// every pixel in the rectangle is visited exactly once (the in-kernel
    /// `x >= width || y >= height` check discards the rest of the square).
    ///
    /// The padding looks expensive and measurably isn't. At 1920x1080 it
    /// launches a 2048x2048 grid — 4,194,304 threads for 2,073,600 pixels,
    /// 2.02x oversubscription, against wgpu's 1.01x. Timed against the same
    /// `mandelbrot_f32` math launched on a plain 2-D grid (via
    /// `dispatch_tiled_f32` over one full-frame tile), the median difference
    /// over 7 paired runs is **+1.9%**, with two runs showing Morton *ahead* —
    /// the surplus threads land in one contiguous L-shaped dead region, so
    /// whole blocks fail the bounds check and retire immediately. Recorded
    /// here as a tested-and-rejected hypothesis so it doesn't get "optimized"
    /// again: the real cost in this path is the full-frame `dtoh_sync_copy`
    /// — 1.33 ms of a 1.61 ms frame, ~82% — now that the fp64 bulb check in
    /// `mandelbrot_f32` has been fixed. See results/summary.md §2.1 and
    /// `examples/gpu_probe.rs`.
    fn morton_cfg(&self, w: u32, h: u32) -> LaunchConfig {
        let dim    = w.max(h).max(1).next_power_of_two();
        let total  = dim as u64 * dim as u64;
        let blocks = ((total + 255) / 256) as u32;
        LaunchConfig {
            block_dim: (16, 16, 1),
            grid_dim:  (blocks, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}
