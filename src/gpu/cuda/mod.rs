//! CUDA backend targeting the discrete NVIDIA GPU directly (as opposed to
//! the portable wgpu backend). Kernels are precompiled to PTX by `build.rs`
//! from `fractal.cu` and loaded once via `nvrtc`; this module owns the
//! device buffers and launch configuration around them.
//!
//! Frame pixels are dispatched in Morton (Z-order) rather than row-major
//! order (see [`CudaFractal::morton_cfg`]) so that threads within a warp,
//! and blocks scheduled close together in time, cover a spatially compact
//! region of the image — escape-time cost is spatially correlated, so this
//! keeps warps more uniform in their iteration counts than a row-major
//! sweep would, reducing the amount of time fast-finishing threads in a
//! warp sit idle waiting for the slowest one.

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DeviceSlice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use crate::fractal::fractal::F32_PRECISION_THRESHOLD;
use crate::fractal::perturbation::perturbation_theory::RefOrbit;

/// Host-side pixel buffer registered as CUDA page-locked ("pinned")
/// memory, so [`CudaFractal::readback_into`] can DMA directly into it
/// instead of the driver first staging through an internal pinned buffer.
/// Falls back to ordinary (unpinned) memory if `cuMemHostRegister` fails —
/// still correct, just without the DMA speedup.
pub struct PinnedBuf {
    buf: Vec<f32>,
    pinned: bool,
}

impl PinnedBuf {
    /// Allocates `len` f32s and attempts to pin them via
    /// `cuMemHostRegister`. Registration can fail (e.g. no CUDA context,
    /// address not page-aligned in a way the driver accepts) without that
    /// being a fatal error for the caller — [`PinnedBuf::is_pinned`]
    /// reports whether it succeeded.
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

    pub fn is_pinned(&self) -> bool { self.pinned }

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

/// Owns one CUDA device context, its loaded kernel functions, and the
/// GPU-resident buffers reused across frames. `output` is the full-frame,
/// row-major buffer used by whole-frame and Morton-order rendering;
/// `compact_output` is a separate densely-packed buffer for
/// [`CudaFractal::dispatch_tiled_compact`], whose tiles are typically a
/// small subset of the frame gathered from scattered scheduler work
/// rather than covering it fully.
pub struct CudaFractal {
    dev:            Arc<CudaDevice>,
    output:         CudaSlice<f32>,
    prepass_output: Option<CudaSlice<f32>>,
    compact_output:       CudaSlice<f32>,
    color_output:         CudaSlice<u8>,
    compact_host_scratch: Vec<f32>,
    lut_cache:      Option<(u8, CudaSlice<u8>)>,
    width:          u32,
    height:         u32,
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
    /// Opens CUDA device 0 (the discrete GPU on this machine's topology;
    /// see [`crate::gpu::cuda`] module docs) and loads all kernels onto it.
    pub fn new(width: u32, height: u32) -> Self {
        let dev = CudaDevice::new(0).expect("no CUDA device");
        Self::from_device(dev, width, height)
    }

    /// Loads `fractal.cu`'s compiled PTX (embedded at build time via the
    /// `FRACTAL_PTX` env var `build.rs` sets) onto an already-open device
    /// and allocates the frame-sized output buffers. PTX loading is
    /// skipped if a module named `"fractal"` is already loaded on this
    /// device, so constructing multiple `CudaFractal`s sharing a device
    /// (e.g. one per scheduler worker) doesn't reload and relink the
    /// kernels each time.
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

    /// Renders a full frame in Morton order and reads it back synchronously.
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

    /// Chooses between the f32 and f64 CUDA kernels the same way the CPU
    /// backend does: Mandelbrot/Julia (`fractal` discriminants 0/1) below
    /// [`F32_PRECISION_THRESHOLD`] use [`CudaFractal::dispatch_render_f32`];
    /// everything else — deeper zoom, or Newton/Nova which have no f32
    /// kernel at all — uses the f64 path. Zoom is recovered from `im_step`
    /// rather than passed explicitly, since it is the same quantity
    /// [`crate::fractal::fractal::pixel_grid`] derives it from on the CPU
    /// side.
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

    /// Renders and immediately colorizes on the GPU, avoiding a
    /// device-to-host round trip of the raw iteration buffer between the
    /// two stages — only the final RGBA bytes cross the PCIe bus.
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

    /// Renders a full frame in perturbation mode against `orbit` and reads
    /// it back synchronously; see [`CudaFractal::dispatch_render_perturbation`].
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

    /// Uploads `orbit`'s `zr`/`zi` arrays fresh for every call — the
    /// reference orbit changes whenever the view center or zoom changes,
    /// so there is no cross-frame reuse to cache here, unlike the LUT
    /// caching in [`CudaFractal::colorize_into`]. Dispatches over a plain
    /// row-major 16x16-block grid rather than Morton order: the
    /// perturbation kernel's per-pixel cost already varies far less than
    /// escape-time iteration count does (most of its work is the fixed
    /// reference-orbit walk), so Morton ordering's warp-uniformity benefit
    /// does not apply here.
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

    /// Renders a coarse `prepass_w x prepass_h` grid at 8x the pixel
    /// stride of the full frame (`re_step * 8.0`, `im_step * 8.0`) — a
    /// cheap, low-resolution pass over the same view used to estimate
    /// per-region iteration cost before committing to a full-resolution
    /// render. The prepass buffer is only reallocated when a larger one is
    /// requested than currently held, so repeated prepasses at the same or
    /// smaller size reuse the existing GPU allocation.
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
                re_step * 8.0, im_step * 8.0,  
                julia_cr, julia_ci,
                max_iter, fractal,
                prepass_w, prepass_h,
            ))
        }.unwrap();

        let mut out = self.dev.dtoh_sync_copy(self.prepass_output.as_ref().unwrap()).unwrap();
        out.truncate(n);
        out
    }

    /// Uploads a batch of `[x0, y0, w, h]` tile descriptors and builds a
    /// launch grid sized to the largest tile, with one grid Z-layer per
    /// tile — shared by [`CudaFractal::dispatch_tiled`] and
    /// [`CudaFractal::dispatch_tiled_f32`]. Tiles write into `self.output`
    /// at their own frame-relative offset, so results land directly in the
    /// full-frame buffer without a separate compositing step.
    fn upload_tiles(&self, tiles: &[[u32; 4]]) -> Option<(CudaSlice<u32>, LaunchConfig)> {
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

    /// f64 tiled dispatch: writes tiles into `self.output` without reading
    /// back. Used by [`CudaFractal::render_tiled`] and by callers that
    /// batch several `dispatch_tiled*` calls before a single
    /// [`CudaFractal::readback`].
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

    /// f32 counterpart of [`CudaFractal::dispatch_tiled`], for callers that
    /// have already determined the current zoom is below
    /// [`F32_PRECISION_THRESHOLD`] for every tile in the batch.
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

    /// Allocating readback of the full-frame output buffer.
    pub fn readback(&self) -> Vec<f32> {
        self.dev.dtoh_sync_copy(&self.output).unwrap()
    }

    /// Copies the device output buffer into a caller-owned host buffer —
    /// pair `dst` with a [`PinnedBuf`] so this DMAs directly rather than
    /// staging through the driver's internal pinned memory. This
    /// device-to-host transfer is the dominant cost of a CUDA frame, so
    /// avoiding both the allocation in [`CudaFractal::readback`] and the
    /// staging copy here is the single largest win available in this
    /// backend's render loop.
    pub fn readback_into(&self, dst: &mut [f32]) {
        assert_eq!(
            dst.len(), (self.width * self.height) as usize,
            "readback destination must be exactly width*height",
        );
        self.dev.dtoh_sync_copy_into(&self.output, dst).unwrap();
    }

    /// Like [`CudaFractal::upload_tiles`], but for tile descriptors with a
    /// fifth field — a destination offset into the *compact* output
    /// buffer, since these tiles are not laid out at their own screen
    /// position within a full-frame buffer but packed contiguously
    /// (typically a scattered subset of the frame the scheduler batched
    /// for the GPU).
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

    /// f64 compact-tile dispatch: writes into `self.compact_output` at
    /// each tile's own destination offset rather than `self.output`. Read
    /// back with [`CudaFractal::readback_compact`].
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

    /// f32 counterpart of [`CudaFractal::dispatch_tiled_compact`].
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

    /// Reads back the first `total_pixels` of `compact_output` into a
    /// reused host-side scratch buffer (grown, never shrunk, across
    /// calls), returning a borrow rather than an owned `Vec` to avoid a
    /// per-call heap allocation on what is typically a per-batch hot path.
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

    /// Convenience wrapper: [`CudaFractal::dispatch_tiled`] followed by
    /// [`CudaFractal::readback`] for one-shot (non-batched) callers.
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

    /// GPU-side mirror of [`crate::gui::color::colorize`]'s
    /// histogram-equalization pipeline: a per-value histogram over the
    /// current `output` buffer, reduced on the host into a cumulative
    /// distribution function (`bins` small enough — at most `max_iter + 1`
    /// — that this reduction is not worth a second kernel), then a second
    /// kernel maps each pixel's CDF-equalized position through the palette
    /// LUT. The LUT upload is skipped and the cached device copy reused
    /// whenever `scheme_id` matches the previous call, since the same
    /// palette is typically used across many consecutive frames.
    pub fn colorize_into(&mut self, max_iter: u32, scheme_id: u8, lut: &[u8], out: &mut [u8]) {
        let n = (self.width * self.height) as usize;
        assert_eq!(out.len(), n * 4, "colorize destination must be width*height RGBA8 bytes");
        assert_eq!(lut.len() % 4, 0, "lut must be RGBA8 (4 bytes/entry)");
        let bins = max_iter as usize + 1;
        let cfg  = Self::linear_cfg(n as u32);

        let mut hist_dev = self.dev.alloc_zeros::<u32>(bins).unwrap();
        unsafe {
            self.fn_hist.clone().launch(cfg, (&self.output, &mut hist_dev, n as u32, max_iter))
        }.unwrap();
        let hist_host: Vec<u32> = self.dev.dtoh_sync_copy(&hist_dev).unwrap();

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

    /// Standard 1D launch grid for the histogram/colorize kernels, which
    /// operate on a flat pixel array with no 2D spatial meaning.
    fn linear_cfg(n: u32) -> LaunchConfig {
        LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim:  ((n + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    pub fn width(&self) -> u32 { self.width }

    pub fn height(&self) -> u32 { self.height }

    /// Launch grid for Morton-order dispatch (see the module-level
    /// documentation): the frame is conceptually padded up to a `dim x
    /// dim` square, `dim` the next power of two at or above
    /// `max(w, h)`, since a Z-order curve's bit-interleaving only maps
    /// cleanly onto power-of-two square regions. `blocks` covers that
    /// padded area in flat 256-thread (16x16) chunks via a 1D grid — the
    /// kernel itself decodes each thread's linear index back into `(x, y)`
    /// through the Morton bit-interleaving and discards threads that land
    /// outside the true `w x h` frame.
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
