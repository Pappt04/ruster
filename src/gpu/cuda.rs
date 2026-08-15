use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use crate::fractal::fractal::{RefOrbit, F32_PRECISION_THRESHOLD};

/// A host buffer whose pages are locked (via `cuMemHostRegister`) so CUDA can
/// DMA into it directly rather than staging through the driver's own internal
/// pinned bounce buffer.
///
/// Registration is not free — it walks and locks the page table — so this is
/// worth it only for a buffer reused across many frames, which is exactly the
/// render-target case. Allocate one per resolution and keep it; do not create
/// one per frame.
///
/// Derefs to `[f32]`, so it drops into any `&mut [f32]` slot (e.g.
/// [`CudaFractal::readback_into`]). Unregisters on drop; if registration fails
/// (page-locked memory is a limited system resource) it degrades silently to a
/// plain heap buffer rather than failing the render, and [`Self::is_pinned`]
/// reports which happened.
pub struct PinnedBuf {
    buf: Vec<f32>,
    pinned: bool,
}

impl PinnedBuf {
    pub fn new(len: usize) -> Self {
        let mut buf = vec![0.0f32; len];
        let rc = unsafe {
            cudarc::driver::sys::lib().cuMemHostRegister_v2(
                buf.as_mut_ptr() as *mut std::ffi::c_void,
                len * std::mem::size_of::<f32>(),
                0,
            )
        };
        let pinned = rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS;
        Self { buf, pinned }
    }

    /// Whether page-locking actually succeeded. False means the buffer still
    /// works, just at pageable-transfer speed.
    pub fn is_pinned(&self) -> bool { self.pinned }

    /// Hands out the underlying data without unregistering — cloning here
    /// rather than moving keeps the registered allocation alive for reuse.
    pub fn to_vec(&self) -> Vec<f32> { self.buf.clone() }
}

impl std::ops::Deref for PinnedBuf {
    type Target = [f32];
    fn deref(&self) -> &[f32] { &self.buf }
}
impl std::ops::DerefMut for PinnedBuf {
    fn deref_mut(&mut self) -> &mut [f32] { &mut self.buf }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        if self.pinned {
            unsafe {
                cudarc::driver::sys::lib()
                    .cuMemHostUnregister(self.buf.as_mut_ptr() as *mut std::ffi::c_void);
            }
        }
    }
}

pub struct CudaFractal {
    dev:            Arc<CudaDevice>,
    output:         CudaSlice<f32>,
    prepass_output: Option<CudaSlice<f32>>,
    /// Compact scratch for `dispatch_tiled_compact`/`_f32_compact` — tiles
    /// are written back-to-back in dispatch order instead of at their
    /// row-major frame position, so a batch's readback only ever costs the
    /// pixels actually dispatched. Sized `width*height` once at construction
    /// (an upper bound: a batch can never exceed the whole frame), so unlike
    /// `prepass_output` it never needs to grow.
    compact_output:       CudaSlice<f32>,
    /// Persistent RGBA8 landing buffer for `colorize_into`'s second kernel
    /// pass — sized `width*height*4` bytes once at construction, same
    /// lifetime/reuse rationale as `output`.
    color_output:         CudaSlice<u8>,
    /// Host-side landing buffer for `readback_compact`, grown (never
    /// shrunk) to the largest batch seen so far — avoids a per-frame `Vec`
    /// allocation for what's usually a repeated, similarly-sized transfer.
    compact_host_scratch: Vec<f32>,
    /// Device-side palette LUT uploaded by `colorize_into`, keyed by the
    /// caller's scheme id so it's only re-uploaded when the scheme actually
    /// changes (`gui::color::lut_bytes` is a `LUT_SIZE * 4`-byte constant
    /// per scheme — trivial to re-send, but there's no reason to pay even
    /// that every frame while the user has one scheme selected).
    lut_cache:      Option<(u8, CudaSlice<u8>)>,
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
    fn_tiled_compact:     CudaFunction,
    fn_tiled_f32_compact: CudaFunction,
    fn_hist:              CudaFunction,
    fn_colorize:           CudaFunction,
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
                "fractal_kernel_tiled_compact",
                "fractal_kernel_tiled_f32_compact",
                "hist_kernel",
                "colorize_kernel",
            ]).unwrap();
        }

        let n              = (width * height) as usize;
        let output         = dev.alloc_zeros::<f32>(n).unwrap();
        let compact_output = dev.alloc_zeros::<f32>(n).unwrap();
        let color_output   = dev.alloc_zeros::<u8>(n * 4).unwrap();

        let fn_kernel     = dev.get_func("fractal", "fractal_kernel").unwrap();
        let fn_kernel_f32 = dev.get_func("fractal", "fractal_kernel_f32").unwrap();
        let fn_perturb    = dev.get_func("fractal", "fractal_perturb_kernel").unwrap();
        let fn_tiled      = dev.get_func("fractal", "fractal_kernel_tiled").unwrap();
        let fn_tiled_f32  = dev.get_func("fractal", "fractal_kernel_tiled_f32").unwrap();
        let fn_tiled_compact     = dev.get_func("fractal", "fractal_kernel_tiled_compact").unwrap();
        let fn_tiled_f32_compact = dev.get_func("fractal", "fractal_kernel_tiled_f32_compact").unwrap();
        let fn_hist       = dev.get_func("fractal", "hist_kernel").unwrap();
        let fn_colorize   = dev.get_func("fractal", "colorize_kernel").unwrap();

        Self {
            dev, output, prepass_output: None,
            compact_output, color_output, compact_host_scratch: Vec::new(), lut_cache: None,
            width, height,
            fn_kernel, fn_kernel_f32, fn_perturb, fn_tiled, fn_tiled_f32,
            fn_tiled_compact, fn_tiled_f32_compact, fn_hist, fn_colorize,
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
        self.dispatch_render(re_start, im_start, re_step, im_step, julia_cr, julia_ci, max_iter, fractal);
        self.dev.dtoh_sync_copy(&self.output).unwrap()
    }

    /// Like [`Self::render_and_colorize`], but launches this frame's kernel
    /// without reading anything back — split out so [`Self::render`] and
    /// [`Self::render_and_colorize`] can share the dispatch logic while
    /// disagreeing on what happens to `self.output` afterward (a raw D2H
    /// copy vs. an on-device colorize pass).
    fn dispatch_render(
        &mut self,
        re_start : f64,
        im_start : f64,
        re_step  : f64,
        im_step  : f64,
        julia_cr : f64,
        julia_ci : f64,
        max_iter : u32,
        fractal  : u32,
    ) {
        let f32_capable = fractal == 0 || fractal == 1;
        if f32_capable {
            let zoom = 4.0 / (im_step.abs() * self.height as f64);
            if zoom < F32_PRECISION_THRESHOLD {
                self.dispatch_render_f32(re_start, im_start, re_step, im_step, julia_cr, julia_ci, max_iter, fractal);
                return;
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
    }

    fn dispatch_render_f32(
        &mut self,
        re_start : f64,
        im_start : f64,
        re_step  : f64,
        im_step  : f64,
        julia_cr : f64,
        julia_ci : f64,
        max_iter : u32,
        fractal  : u32,
    ) {
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
    }

    /// Like [`Self::render`], but colorizes on-device (`colorize_into`)
    /// instead of returning raw escape values — skips the escape buffer's
    /// D2H copy entirely (nothing needs it once colorized) and moves the
    /// histogram/CDF/LUT work that `gui::color::colorize` would otherwise run
    /// on the CPU onto the GPU instead. Use this instead of `render` +
    /// `gui::color::colorize` whenever the caller only wants pixels.
    ///
    /// # Panics
    /// If `out.len() != width * height * 4`.
    #[allow(clippy::too_many_arguments)]
    pub fn render_and_colorize(
        &mut self,
        re_start  : f64,
        im_start  : f64,
        re_step   : f64,
        im_step   : f64,
        julia_cr  : f64,
        julia_ci  : f64,
        max_iter  : u32,
        fractal   : u32,
        scheme_id : u8,
        lut       : &[u8],
        out       : &mut [u8],
    ) {
        self.dispatch_render(re_start, im_start, re_step, im_step, julia_cr, julia_ci, max_iter, fractal);
        self.colorize_into(max_iter, scheme_id, lut, out);
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
        self.dispatch_render_perturbation(orbit, re_start, im_start, re_step, im_step, max_iter);
        self.dev.dtoh_sync_copy(&self.output).unwrap()
    }

    /// Split out from [`Self::render_perturbation`] the same way
    /// [`Self::dispatch_render`] is split from [`Self::render`] — shared by
    /// [`Self::render_perturbation`] and
    /// [`Self::render_perturbation_and_colorize`].
    fn dispatch_render_perturbation(
        &mut self,
        orbit    : &RefOrbit,
        re_start : f64,
        im_start : f64,
        re_step  : f64,
        im_step  : f64,
        max_iter : u32,
    ) {
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
    }

    /// Like [`Self::render_perturbation`], but colorizes on-device — same
    /// rationale as [`Self::render_and_colorize`].
    ///
    /// # Panics
    /// If `out.len() != width * height * 4`.
    pub fn render_perturbation_and_colorize(
        &mut self,
        orbit     : &RefOrbit,
        re_start  : f64,
        im_start  : f64,
        re_step   : f64,
        im_step   : f64,
        max_iter  : u32,
        scheme_id : u8,
        lut       : &[u8],
        out       : &mut [u8],
    ) {
        self.dispatch_render_perturbation(orbit, re_start, im_start, re_step, im_step, max_iter);
        self.colorize_into(max_iter, scheme_id, lut, out);
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
    /// scheduler rewrite exists to fix.
    ///
    /// `readback`/`readback_into` still copy the *entire* frame regardless of
    /// how many tiles were actually dispatched, which was measured as *the*
    /// reason the heterogeneous scheduler couldn't beat plain GPU rendering
    /// on this machine — at 1920x1080 the full-frame copy is 1.33 ms against
    /// a ~0.28 ms kernel, ~82% of the frame as fixed cost shifting tiles to
    /// the CPU does nothing to reduce (results/summary.md §3). A row-banded
    /// readback window doesn't fix it either: GPU-classified cells at the
    /// frame's own top/bottom edges pin the band to full height in practice
    /// (measured against this classifier). The actual fix is
    /// `dispatch_tiled_compact`/
    /// `dispatch_tiled_f32_compact` below, which write tiles contiguously
    /// into a compact buffer instead of at their frame position, so
    /// `readback_compact` only ever pays for pixels actually dispatched.
    /// `dispatch_tiled`/`dispatch_tiled_f32` (and `render_tiled`) are kept for
    /// any caller that wants the simpler full-frame-position contract (e.g. a
    /// one-shot single-tile render where compaction buys nothing).
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
    ///
    /// Allocates a fresh `Vec` per call. Prefer [`Self::readback_into`] with a
    /// [`PinnedBuf`] where the destination can be reused — that is ~7.5%
    /// faster, and on a readback-bound frame (§2 of results/summary.md, ~82%
    /// of a CUDA frame) that is the cheapest win available.
    pub fn readback(&self) -> Vec<f32> {
        self.dev.dtoh_sync_copy(&self.output).unwrap()
    }

    /// Copies the full-frame output buffer into a caller-owned destination.
    ///
    /// The transfer is measurably faster when `dst` is page-locked (see
    /// [`PinnedBuf`]): the driver can DMA straight into it instead of staging
    /// through its own internal pinned bounce buffer. Measured at 1920x1080
    /// (8.29 MB) over this machine's PCIe 3.0 x8 link: 1.336 ms into a
    /// pageable `Vec`, 1.235 ms into a page-locked one (-7.5%, 6.2 -> 6.7
    /// GB/s against a 7.9 GB/s theoretical ceiling). Reusing a *pageable*
    /// buffer is worth nothing on its own — the allocator keeps handing back
    /// the same mapped pages — so the win here is the pinning, not the reuse.
    ///
    /// # Panics
    /// If `dst.len()` is not `width * height`.
    pub fn readback_into(&self, dst: &mut [f32]) {
        assert_eq!(
            dst.len(), (self.width * self.height) as usize,
            "readback destination must be exactly width*height",
        );
        self.dev.dtoh_sync_copy_into(&self.output, dst).unwrap();
    }

    /// Uploads `[x0,y0,w,h,offset]` tile descriptors (5 × `uint32` per tile —
    /// see `fractal_kernel_tiled_compact`'s doc comment in `fractal.cu` for
    /// why `offset` has to travel with the tile rather than being derived in
    /// the kernel) and computes the shared launch grid, mirroring
    /// `upload_tiles` above. `offset` values are the caller's responsibility:
    /// `crate::scheduler` computes them as a running prefix sum over
    /// `tw*th` so the same numbers used to place each tile in the compact
    /// device buffer are reused, unchanged, to scatter it back out of the
    /// host readback afterward — the kernel and the scatter step must agree
    /// on where each tile landed, and computing that in one place (the
    /// caller) rather than twice is what keeps them from drifting apart.
    fn upload_tiles_compact(&self, tiles: &[[u32; 5]]) -> Option<(CudaSlice<u32>, LaunchConfig)> {
        if tiles.is_empty() {
            return None;
        }
        let flat: Vec<u32> = tiles.iter().flat_map(|t| t.iter().copied()).collect();
        let tile_descs_dev = self.dev.htod_sync_copy(&flat).unwrap();

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

    /// Like [`Self::dispatch_tiled`], but tiles land contiguously in
    /// dispatch order in a *compact* buffer (each tile's `offset` field)
    /// instead of at their row-major position in a full-frame buffer. Pair
    /// with [`Self::readback_compact`] to copy back only the pixels this
    /// batch actually computed, not the whole frame. No-op if `tiles` is
    /// empty. See `fractal_kernel_tiled_compact` in `fractal.cu`.
    pub fn dispatch_tiled_compact(
        &mut self,
        tiles    : &[[u32; 5]],
        re_start : f64,
        im_start : f64,
        re_step  : f64,
        im_step  : f64,
        julia_cr : f64,
        julia_ci : f64,
        max_iter : u32,
        fractal  : u32,
    ) {
        let Some((tile_descs_dev, cfg)) = self.upload_tiles_compact(tiles) else { return };

        let f = self.fn_tiled_compact.clone();
        unsafe {
            f.launch(cfg, (
                &mut self.compact_output,
                &tile_descs_dev,
                re_start, im_start, re_step, im_step,
                julia_cr, julia_ci,
                max_iter, fractal,
            ))
        }.unwrap();
    }

    /// f32 counterpart of [`Self::dispatch_tiled_compact`] — same
    /// Mandelbrot/Julia-only, caller-decides contract as
    /// [`Self::dispatch_tiled_f32`].
    pub fn dispatch_tiled_f32_compact(
        &mut self,
        tiles    : &[[u32; 5]],
        re_start : f64,
        im_start : f64,
        re_step  : f64,
        im_step  : f64,
        julia_cr : f64,
        julia_ci : f64,
        max_iter : u32,
        fractal  : u32,
    ) {
        let Some((tile_descs_dev, cfg)) = self.upload_tiles_compact(tiles) else { return };

        let f = self.fn_tiled_f32_compact.clone();
        unsafe {
            f.launch(cfg, (
                &mut self.compact_output,
                &tile_descs_dev,
                re_start as f32, im_start as f32, re_step as f32, im_step as f32,
                julia_cr as f32, julia_ci as f32,
                max_iter, fractal,
            ))
        }.unwrap();
    }

    /// Copies back exactly the first `total_pixels` entries of the compact
    /// buffer — i.e. everything written by this frame's
    /// `dispatch_tiled_compact`/`_f32_compact` calls, and nothing else — as
    /// one contiguous transfer. Returns a borrow of an internal, geometrically
    /// -grown host buffer (never shrinks) rather than allocating fresh every
    /// call, since this runs every frame the GPU has any tiles at all.
    ///
    /// # Panics
    /// If `total_pixels > width * height`.
    pub fn readback_compact(&mut self, total_pixels: u32) -> &[f32] {
        let n = total_pixels as usize;
        assert!(n <= (self.width * self.height) as usize, "batch exceeds frame size");
        if self.compact_host_scratch.len() < n {
            self.compact_host_scratch.resize(n, 0.0);
        }
        let view = self.compact_output.slice(0..n);
        self.dev.dtoh_sync_copy_into(&view, &mut self.compact_host_scratch[..n]).unwrap();
        &self.compact_host_scratch[..n]
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

    /// GPU-side histogram-equalization + palette-LUT colorization of
    /// `self.output` — the device-side counterpart of `gui::color::colorize`
    /// (same algorithm, see `hist_kernel`/`colorize_kernel` in `fractal.cu`).
    ///
    /// Only valid when `self.output` holds this frame's *entire* escape-time
    /// buffer, i.e. after `render`/`render_f32`/`render_perturbation` — the
    /// heterogeneous scheduler's tiled dispatch never fills `self.output` in
    /// full (CPU tiles never touch the GPU at all), so it cannot use this;
    /// its CPU-resident pixels still need `gui::color::colorize` on the host.
    /// For the plain full-frame render path, this replaces both the escape
    /// buffer's D2H copy *and* the CPU-side histogram/CDF/LUT pass with two
    /// small (bins-sized, not frame-sized) transfers plus one D2H copy of the
    /// final RGBA8 image — the same byte count the escape buffer would have
    /// cost, but with the histogram/colorize compute that used to run
    /// afterward on the CPU now overlapped into the GPU pipeline instead of
    /// serialized after a readback.
    ///
    /// `scheme_id` is an opaque cache key (`gui::color::ColorScheme::palette_index()`
    /// as `u8` is the intended caller) — the palette LUT is only re-uploaded
    /// when it changes. `lut` is `lut_bytes.len()/4` RGBA8 entries, sampled at
    /// t ∈ [0,1] exactly like `gui::color::PALETTES` (see
    /// `gui::color::lut_bytes`).
    ///
    /// # Panics
    /// If `out.len() != width * height * 4`.
    pub fn colorize_into(&mut self, max_iter: u32, scheme_id: u8, lut: &[u8], out: &mut [u8]) {
        let n = (self.width * self.height) as usize;
        assert_eq!(out.len(), n * 4, "colorize destination must be width*height RGBA8 bytes");
        assert_eq!(lut.len() % 4, 0, "lut must be RGBA8 (4 bytes/entry)");
        let bins = max_iter as usize + 1;
        let cfg  = Self::linear_cfg(n as u32);

        // Pass 1: histogram of escaped pixels' floor(escape value), entirely
        // on-device (`self.output` never leaves the GPU for this). `bins` is
        // tiny next to `n` (max_iter is in the thousands at most, `n` is in
        // the millions), so allocating fresh each call rather than caching
        // like `compact_output` isn't worth the bookkeeping.
        let mut hist_dev = self.dev.alloc_zeros::<u32>(bins).unwrap();
        unsafe {
            self.fn_hist.clone().launch(cfg, (&self.output, &mut hist_dev, n as u32, max_iter))
        }.unwrap();
        let hist_host: Vec<u32> = self.dev.dtoh_sync_copy(&hist_dev).unwrap();

        // CDF: identical sequential accumulation to `gui::color::colorize`.
        // `bins`-sized, not `n`-sized — cheap enough that a scan kernel isn't
        // worth it (see `hist_kernel`'s doc comment in fractal.cu).
        let total: f64 = hist_host.iter().map(|&c| c as f64).sum();
        let mut cdf = vec![0.0f32; bins];
        let mut running = 0.0f64;
        for (i, &c) in hist_host.iter().enumerate() {
            running += c as f64;
            cdf[i] = if total > 0.0 { (running / total) as f32 } else { 0.0 };
        }
        let cdf_dev = self.dev.htod_sync_copy(&cdf).unwrap();

        if self.lut_cache.as_ref().map(|(id, _)| *id) != Some(scheme_id) {
            let lut_dev = self.dev.htod_sync_copy(lut).unwrap();
            self.lut_cache = Some((scheme_id, lut_dev));
        }
        let lut_size = (lut.len() / 4) as u32;

        // Pass 2: per-pixel LUT lookup, entirely on-device.
        {
            let lut_dev = &self.lut_cache.as_ref().unwrap().1;
            unsafe {
                self.fn_colorize.clone().launch(cfg, (
                    &self.output, &cdf_dev, lut_dev, &mut self.color_output,
                    n as u32, max_iter, lut_size,
                ))
            }.unwrap();
        }

        self.dev.dtoh_sync_copy_into(&self.color_output, out).unwrap();
    }

    /// 1-D launch config for a flat `n`-thread kernel (256 threads/block —
    /// matches every other kernel's block size in this file).
    fn linear_cfg(n: u32) -> LaunchConfig {
        LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim:  ((n + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        }
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
