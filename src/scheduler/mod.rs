//! Corner-sampling, work-stealing heterogeneous CPU+GPU scheduler (see
//! `TUTORIAL.md` Stage 3, and the rewrite notes in `results/summary.md`).
//! Classifies the frame into variable-size tiles via cheap corner sampling
//! (`classifier::partition_frame` — no GPU prepass dispatch needed), routes
//! divergent "boundary" tiles to the CPU (exact per-pixel fill) and coherent
//! tiles to the GPU (`fractal_kernel_tiled`), running both concurrently via
//! shared work-stealing queues so whichever side finishes early picks up the
//! other's leftover work, then composites into one full-resolution buffer.
//!
//! No tile is ever skipped or approximated — only *where* (and, by default,
//! in what precision — see `SchedulerConfig::simd_cpu_tiles`/`gpu_tiles_f32`)
//! it's computed changes. Set both of those `false` for a strict bit-exact
//! match to plain CPU `render()` (their default `true` trades a handful of
//! pixels' worth of GPU/CPU floating-point non-determinism for a large
//! throughput win — see each field's doc comment). See `classifier`'s module
//! doc for why a corner-sampling *routing* decision on its own can never
//! affect correctness, only performance.

pub mod classifier;
pub mod controller;

use std::collections::VecDeque;
use std::sync::Mutex;

use crate::fractal::{pixel_grid, render_cpu_tile_into, IterBuf, F32_PRECISION_THRESHOLD};
use crate::fractal::fractal_type::FractalType;
use crate::gpu::cuda::CudaFractal;
use crate::gpu::fractal_compute::FractalCompute;
use crate::gpu::unifroms::Uniforms;
use crate::gui::viewport::Viewport;
use controller::ThresholdController;

pub struct SchedulerConfig {
    /// Top-level partition tile size. Bigger than the old fixed grid (128 vs
    /// 32) by design — a uniform region terminates early as one big GPU tile
    /// instead of being forced into many small ones.
    pub max_tile_size: u32,
    /// Recursion floor; matches the 16x16 GPU block dim `fractal_kernel_tiled`
    /// uses.
    pub min_tile_size: u32,
    /// Fraction of the GPU-classified tiles held back as genuinely stealable
    /// by CPU workers, instead of being dispatched in the GPU's immediate
    /// batch. See `render_heterogeneous`'s doc comment on why this needs to
    /// be a real reservation, not just "CPU falls back to an empty queue."
    pub steal_reserve_frac: f32,
    /// Minimum number of leftover CPU-queue tiles that justifies GPU issuing
    /// a second (mop-up) dispatch for them.
    pub min_steal_tiles: u32,
    /// Render CPU tiles with the f32 SIMD path (`render_tile_exact_simd`)
    /// instead of the exact f64 scalar path, where applicable (Mandelbrot/
    /// Julia below `F32_PRECISION_THRESHOLD`). Defaults to `true` — NOT
    /// bit-identical to plain CPU `render()` (see `classifier`'s module doc),
    /// but the throughput win is large enough that this is what real usage
    /// (including the live app) wants; set `false` for a strict bit-exact
    /// comparison (e.g. validating other changes against `render()`).
    pub simd_cpu_tiles: bool,
    /// Dispatch GPU tiles with `fractal_kernel_tiled_f32` instead of the
    /// exact f64 `fractal_kernel_tiled`, where applicable (Mandelbrot/Julia
    /// below `F32_PRECISION_THRESHOLD`). Same bit-exactness-vs-speed tradeoff
    /// as `simd_cpu_tiles`, but the GPU-side impact is much larger in
    /// practice: `fractal_kernel_tiled` never had the f32 fast path plain
    /// `CudaFractal::render()` already has, so with this `false` the
    /// scheduler's GPU tiles pay this GPU tier's full fp64 tax while plain
    /// `cuda.render()` doesn't — measured ~6-7x slower GPU-tile throughput
    /// than plain GPU rendering at the same viewport, which was the
    /// difference between the hybrid scheduler losing badly to single-backend
    /// rendering and actually beating it. Defaults to `true` for the same
    /// reason as `simd_cpu_tiles`; set `false` for a strict bit-exact
    /// comparison.
    pub gpu_tiles_f32: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_tile_size: 128,
            min_tile_size: 16,
            steal_reserve_frac: 0.2,
            min_steal_tiles: 64,
            // Both default on: without them the scheduler's GPU tiles pay
            // this GPU tier's full fp64 tax while plain `cuda.render()`
            // doesn't (measured ~6-7x slower GPU-tile throughput), and CPU
            // tiles leave rayon's SIMD speedup on the table — together they
            // were the difference between the hybrid scheduler losing badly
            // to plain single-backend rendering and actually beating it. The
            // tradeoff (documented on each field, and in `classifier`'s
            // module doc) is a handful of pixels out of millions that can
            // differ from plain CPU `render()` on Mandelbrot/Julia, from
            // genuine GPU/CPU floating-point non-determinism on isolated
            // chaotic pixels — visually imperceptible, and the same category
            // of pre-existing imprecision this codebase already had (just
            // slightly more of it), not a new correctness class.
            simd_cpu_tiles: true,
            gpu_tiles_f32: true,
        }
    }
}

/// Per-frame telemetry from the heterogeneous scheduler, without the frame
/// buffer itself — returned by [`render_heterogeneous_into`], where the caller
/// owns the destination.
#[derive(Debug, Clone, Copy)]
pub struct HeterogeneousStats {
    pub gpu_ms: f32,
    /// Pixel-weighted (not tile-count-weighted) fraction of the frame CPU
    /// actually ended up computing, including anything it stole from the GPU
    /// reserve — reflects realized work, not the pre-steal partition split.
    pub cpu_ms: f32,
    pub cpu_tile_frac: f32,
    pub gpu_steal_dispatched: bool,
    pub cpu_stolen_tile_count: u32,
}

pub struct HeterogeneousResult {
    pub buf: IterBuf,
    pub gpu_ms: f32,
    pub cpu_ms: f32,
    /// Pixel-weighted (not tile-count-weighted) fraction of the frame CPU
    /// actually ended up computing, including anything it stole from the GPU
    /// reserve — reflects realized work, not the pre-steal partition split.
    pub cpu_tile_frac: f32,
    /// Whether the GPU issued a second dispatch to mop up leftover CPU-queue
    /// tiles this frame.
    pub gpu_steal_dispatched: bool,
    /// How many tiles CPU workers claimed from the GPU's reserved
    /// `steal_queue` (i.e. tiles originally classified GPU-cheap but
    /// rendered by CPU because CPU finished its own queue first).
    pub cpu_stolen_tile_count: u32,
}

/// Render one frame by corner-sampling the viewport into GPU/CPU tiles and
/// running both concurrently with fallback work-stealing in both directions.
/// `cuda` must already be sized for `vp`'s width/height (the caller is
/// responsible for recreating it on resize, same as the existing plain-CUDA
/// render path).
pub fn render_heterogeneous(
    vp: &Viewport,
    fractal: FractalType,
    julia_c: [f64; 2],
    max_iter: u32,
    cuda: &mut CudaFractal,
    controller: &mut ThresholdController,
    cfg: &SchedulerConfig,
) -> HeterogeneousResult {
    let mut buf = vec![0.0f32; (vp.width * vp.height) as usize];
    let s = render_heterogeneous_into(vp, fractal, julia_c, max_iter, cuda, controller, cfg, &mut buf);
    HeterogeneousResult {
        buf,
        gpu_ms: s.gpu_ms,
        cpu_ms: s.cpu_ms,
        cpu_tile_frac: s.cpu_tile_frac,
        gpu_steal_dispatched: s.gpu_steal_dispatched,
        cpu_stolen_tile_count: s.cpu_stolen_tile_count,
    }
}

/// Like [`render_heterogeneous`], but renders into a caller-owned `out`
/// instead of allocating a frame buffer per call.
///
/// This is the form to use in a render loop. Keeping one destination alive
/// across frames is what makes page-locking worthwhile: pass a
/// [`crate::gpu::cuda::PinnedBuf`] and the GPU readback — the dominant cost of
/// the whole frame — DMAs straight into it, ~7.5% faster than into pageable
/// memory. Registering a buffer is only worth it when amortized over many
/// frames, so this cannot be done inside a per-call API.
///
/// # Panics
/// If `out.len() != vp.width * vp.height`.
#[allow(clippy::too_many_arguments)]
pub fn render_heterogeneous_into(
    vp: &Viewport,
    fractal: FractalType,
    julia_c: [f64; 2],
    max_iter: u32,
    cuda: &mut CudaFractal,
    controller: &mut ThresholdController,
    cfg: &SchedulerConfig,
    out: &mut [f32],
) -> HeterogeneousStats {
    assert_eq!(
        out.len(), (vp.width * vp.height) as usize,
        "heterogeneous render destination must be exactly width*height",
    );
    let w = vp.width;
    let h = vp.height;
    debug_assert_eq!(w, cuda.width(), "CudaFractal not sized for this viewport");
    debug_assert_eq!(h, cuda.height(), "CudaFractal not sized for this viewport");

    let pg = pixel_grid(vp);

    // --- Partition: cheap corner-sampling recursive classification, no GPU
    // prepass dispatch at all. ---
    let (mut gpu_tiles, cpu_tiles) = classifier::partition_frame(
        &pg, fractal, julia_c, max_iter, w, h,
        cfg.max_tile_size, cfg.min_tile_size, controller.threshold,
    );

    // --- Reserve: hold back a fraction of the GPU's classified tiles as
    // genuinely stealable by CPU. Reserving up front (rather than letting CPU
    // fall back to an initially-full gpu queue) matters because
    // `dispatch_tiled`'s kernel launch enqueues asynchronously and returns
    // almost immediately — there's no real wall-clock window during which an
    // un-reserved queue would still have anything left in it by the time
    // freshly-spawned CPU threads get around to checking it. Reserving a
    // fraction gives CPU workers a real window: the reserve sits available
    // for as long as CPU is still busy on its own (likely larger) queue. ---
    let reserve_frac = cfg.steal_reserve_frac.clamp(0.0, 1.0);
    let reserve_count = ((gpu_tiles.len() as f32) * reserve_frac).round() as usize;
    let split_at = gpu_tiles.len() - reserve_count.min(gpu_tiles.len());
    let gpu_reserve: Vec<[u32; 4]> = gpu_tiles.split_off(split_at);
    let gpu_committed = gpu_tiles;

    let total_pixels = (w as u64) * (h as u64);
    let cpu_queue: Mutex<VecDeque<[u32; 4]>> = Mutex::new(cpu_tiles.into_iter().collect());
    let steal_queue: Mutex<VecDeque<[u32; 4]>> = Mutex::new(gpu_reserve.into_iter().collect());

    let use_simd = cfg.simd_cpu_tiles;
    let zoom = vp.zoom;
    let n_workers = rayon::current_num_threads().max(1);
    let use_gpu_f32 = cfg.gpu_tiles_f32
        && matches!(fractal, FractalType::Mandelbrot | FractalType::Julia)
        && zoom < F32_PRECISION_THRESHOLD;
    let dispatch = |cuda: &mut CudaFractal, batch: &[[u32; 4]]| {
        if use_gpu_f32 {
            cuda.dispatch_tiled_f32(
                batch, pg.re_start, pg.im_start, pg.re_step, pg.im_step,
                julia_c[0], julia_c[1], max_iter, fractal.as_u32(),
            );
        } else {
            cuda.dispatch_tiled(
                batch, pg.re_start, pg.im_start, pg.re_step, pg.im_step,
                julia_c[0], julia_c[1], max_iter, fractal.as_u32(),
            );
        }
    };

    // Results come back over a channel rather than `JoinHandle`s — rayon's
    // `Scope::spawn` (unlike `std::thread::Scope::spawn`) doesn't hand back a
    // handle, since the closures run on the pool's existing worker threads,
    // not new ones. Each message's third field is that worker's own elapsed
    // time from `cpu_t0` at the moment IT finished — `rayon::scope` can't
    // return until the GPU dispatch code below (running on the calling
    // thread) also finishes, so timing "cpu_ms" as `cpu_t0.elapsed()` taken
    // after the scope call (and after draining this channel) would silently
    // inflate it to match whichever of GPU/CPU was slower, rather than
    // reflecting when the CPU workers actually finished — which would feed
    // the adaptive threshold controller a distorted signal.
    // Each worker reports one flat scratch buffer holding all the tiles it
    // rendered, back to back, plus the (tile, offset) index needed to scatter
    // them. This replaces one `Vec<f32>` allocation *per tile* — measured at
    // 1.0-3.9 ms per frame, the single largest piece of scheduler overhead and
    // an order of magnitude above everything else (results/summary.md §3.4).
    // The cost was page faults on freshly-mapped pages, not the copying, so
    // one geometrically-grown buffer per worker removes essentially all of it.
    type WorkerOut = (Vec<f32>, Vec<([u32; 4], usize)>, u32, f32);
    let (tx, rx) = std::sync::mpsc::channel::<WorkerOut>();
    let cpu_t0 = std::time::Instant::now();

    let (gpu_ms, gpu_steal_dispatched) = rayon::scope(|s| {
        // --- CPU worker pool: spawned onto rayon's already-warm global pool
        // (`rayon::scope`, not `std::thread::scope`) — spawning fresh OS
        // threads every single frame was measured to eat most of the
        // scheduler's theoretical win at this frame-time scale (low
        // single-digit milliseconds). Each worker pulls from two shared,
        // mutex-guarded queues — rayon's own work-stealing has no primitive
        // for "steal from a queue owned by a different subsystem [the GPU
        // dispatch]", so plain `Mutex<VecDeque>` is still the right data
        // structure, just driven from pool threads instead of raw ones.
        // Workers drain `cpu_queue` first, then fall back to `steal_queue`
        // once it's empty. ---
        for _ in 0..n_workers {
            let tx = tx.clone();
            let pg = &pg;
            let cpu_queue = &cpu_queue;
            let steal_queue = &steal_queue;
            s.spawn(move |_| {
                let mut scratch: Vec<f32> = Vec::new();
                let mut index: Vec<([u32; 4], usize)> = Vec::new();
                let mut stolen = 0u32;
                loop {
                    let claim = {
                        let mut cq = cpu_queue.lock().unwrap();
                        if let Some(t) = cq.pop_front() {
                            Some((t, false))
                        } else {
                            drop(cq);
                            steal_queue.lock().unwrap().pop_front().map(|t| (t, true))
                        }
                    };
                    match claim {
                        Some((tile, is_steal)) => {
                            if is_steal { stolen += 1; }
                            let [_, _, tw, th] = tile;
                            let offset = scratch.len();
                            let n = (tw * th) as usize;
                            scratch.resize(offset + n, 0.0);
                            render_cpu_tile_into(
                                pg, fractal, julia_c, max_iter, tile, use_simd, zoom,
                                &mut scratch[offset..offset + n],
                            );
                            index.push((tile, offset));
                        }
                        None => break,
                    }
                }
                let finished_ms = cpu_t0.elapsed().as_secs_f32() * 1000.0;
                let _ = tx.send((scratch, index, stolen, finished_ms));
            });
        }

        // --- GPU side: coarse-grained, synchronous on this (calling) thread
        // (cudarc's copies block anyway) — runs concurrently with the CPU
        // workers above, which execute on rayon's pool threads. Dispatch the
        // committed batch immediately, then mop up whatever CPU hasn't
        // claimed from `steal_queue` if it's a big enough leftover to justify
        // a second launch. ---
        let gpu_t0 = std::time::Instant::now();
        dispatch(cuda, &gpu_committed);

        let leftover_len = steal_queue.lock().unwrap().len();
        let mut gpu_steal_dispatched = false;
        if leftover_len as u32 >= cfg.min_steal_tiles {
            let leftover: Vec<[u32; 4]> = std::mem::take(&mut *steal_queue.lock().unwrap()).into_iter().collect();
            if !leftover.is_empty() {
                dispatch(cuda, &leftover);
                gpu_steal_dispatched = true;
            }
        }

        let gpu_did_anything = !gpu_committed.is_empty() || gpu_steal_dispatched;
        // If GPU had nothing to do at all this frame (every tile ended up
        // CPU-side), don't read back — `self.output` would be stale from a
        // previous frame/viewport, and every pixel gets overwritten by the
        // CPU merge below regardless.
        // DMA straight into the caller's frame buffer. When that buffer is a
        // `PinnedBuf` the driver can write into it directly instead of staging
        // through its own bounce buffer — on a readback-bound frame (~82% of a
        // CUDA frame, §2) that is the cheapest win available.
        if gpu_did_anything { cuda.readback_into(out); } else { out.fill(0.0); }
        let gpu_ms = gpu_t0.elapsed().as_secs_f32() * 1000.0;

        (gpu_ms, gpu_steal_dispatched)
        // `rayon::scope` blocks here until every `s.spawn`ed worker above has
        // finished (and sent its result) before returning this tuple.
    });

    let mut cpu_pixels = 0u64;
    let mut cpu_stolen_tile_count = 0u32;
    let mut cpu_ms = 0.0f32;
    let buf = &mut *out;
    for _ in 0..n_workers {
        let (scratch, index, stolen, finished_ms) = rx.recv().unwrap();
        cpu_stolen_tile_count += stolen;
        // The slowest worker to finish is when CPU-side work was actually
        // done — not when we got around to draining the channel, which can
        // happen arbitrarily later if GPU (running concurrently on the
        // calling thread) was the one holding `rayon::scope` open.
        cpu_ms = cpu_ms.max(finished_ms);
        // Scatter this worker's tiles into the frame. Only CPU tiles are
        // copied here (3-23% of pixels in practice), so this is tens of
        // microseconds; the GPU's share arrived already in place via the
        // readback below/above.
        for ([x0, y0, tw, th], offset) in index {
            cpu_pixels += (tw as u64) * (th as u64);
            for row in 0..th {
                let dst = ((y0 + row) * w + x0) as usize;
                let src = offset + (row * tw) as usize;
                buf[dst..dst + tw as usize].copy_from_slice(&scratch[src..src + tw as usize]);
            }
        }
    }

    controller.update(gpu_ms, cpu_ms);

    HeterogeneousStats {
        gpu_ms,
        cpu_ms,
        cpu_tile_frac: if total_pixels > 0 { cpu_pixels as f32 / total_pixels as f32 } else { 0.0 },
        gpu_steal_dispatched,
        cpu_stolen_tile_count,
    }
}

/// Like [`render_heterogeneous`], but drives `FractalCompute` (wgpu) instead
/// of `CudaFractal`. Same corner-sampling partition, same reserve/queue/
/// work-stealing structure, same `SchedulerConfig`/`ThresholdController` —
/// only the GPU dispatch mechanics differ (a `Uniforms` struct + tile
/// descriptors over a wgpu compute pass, instead of discrete f64 args over a
/// CUDA kernel launch).
///
/// wgpu compute shaders have no f64 support at all — `main_tiled` is always
/// f32, for every fractal, unconditionally. There is no `gpu_tiles_f32`-style
/// choice to make on this backend: `SchedulerConfig::gpu_tiles_f32` is simply
/// not consulted here. `simd_cpu_tiles` still applies to the CPU side, same
/// as the CUDA path.
pub fn render_heterogeneous_wgpu(
    vp: &Viewport,
    fractal: FractalType,
    julia_c: [f64; 2],
    max_iter: u32,
    compute: &FractalCompute,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    controller: &mut ThresholdController,
    cfg: &SchedulerConfig,
) -> HeterogeneousResult {
    let w = vp.width;
    let h = vp.height;
    debug_assert_eq!(w, compute.width(), "FractalCompute not sized for this viewport");
    debug_assert_eq!(h, compute.height(), "FractalCompute not sized for this viewport");

    let pg = pixel_grid(vp);

    let (mut gpu_tiles, cpu_tiles) = classifier::partition_frame(
        &pg, fractal, julia_c, max_iter, w, h,
        cfg.max_tile_size, cfg.min_tile_size, controller.threshold,
    );

    let reserve_frac = cfg.steal_reserve_frac.clamp(0.0, 1.0);
    let reserve_count = ((gpu_tiles.len() as f32) * reserve_frac).round() as usize;
    let split_at = gpu_tiles.len() - reserve_count.min(gpu_tiles.len());
    let gpu_reserve: Vec<[u32; 4]> = gpu_tiles.split_off(split_at);
    let gpu_committed = gpu_tiles;

    let total_pixels = (w as u64) * (h as u64);
    let cpu_queue: Mutex<VecDeque<[u32; 4]>> = Mutex::new(cpu_tiles.into_iter().collect());
    let steal_queue: Mutex<VecDeque<[u32; 4]>> = Mutex::new(gpu_reserve.into_iter().collect());

    let use_simd = cfg.simd_cpu_tiles;
    let zoom = vp.zoom;
    let n_workers = rayon::current_num_threads().max(1);

    let uniforms = Uniforms {
        re_start: pg.re_start as f32, im_start: pg.im_start as f32,
        re_step: pg.re_step as f32, im_step: pg.im_step as f32,
        julia_cr: julia_c[0] as f32, julia_ci: julia_c[1] as f32,
        max_iter, fractal: fractal.as_u32(),
        width: w, height: h,
        _pad: [0; 2],
    };
    let dispatch = |batch: &[[u32; 4]]| {
        compute.dispatch_tiled(device, queue, batch, uniforms);
    };

    // Same flat per-worker scratch as `render_heterogeneous` — see the comment
    // there for why one `Vec` per tile was the scheduler's dominant cost.
    type WorkerOut = (Vec<f32>, Vec<([u32; 4], usize)>, u32, f32);
    let (tx, rx) = std::sync::mpsc::channel::<WorkerOut>();
    let cpu_t0 = std::time::Instant::now();

    let (gpu_buf, gpu_ms, gpu_steal_dispatched) = rayon::scope(|s| {
        for _ in 0..n_workers {
            let tx = tx.clone();
            let pg = &pg;
            let cpu_queue = &cpu_queue;
            let steal_queue = &steal_queue;
            s.spawn(move |_| {
                let mut scratch: Vec<f32> = Vec::new();
                let mut index: Vec<([u32; 4], usize)> = Vec::new();
                let mut stolen = 0u32;
                loop {
                    let claim = {
                        let mut cq = cpu_queue.lock().unwrap();
                        if let Some(t) = cq.pop_front() {
                            Some((t, false))
                        } else {
                            drop(cq);
                            steal_queue.lock().unwrap().pop_front().map(|t| (t, true))
                        }
                    };
                    match claim {
                        Some((tile, is_steal)) => {
                            if is_steal { stolen += 1; }
                            let [_, _, tw, th] = tile;
                            let offset = scratch.len();
                            let n = (tw * th) as usize;
                            scratch.resize(offset + n, 0.0);
                            render_cpu_tile_into(
                                pg, fractal, julia_c, max_iter, tile, use_simd, zoom,
                                &mut scratch[offset..offset + n],
                            );
                            index.push((tile, offset));
                        }
                        None => break,
                    }
                }
                let finished_ms = cpu_t0.elapsed().as_secs_f32() * 1000.0;
                let _ = tx.send((scratch, index, stolen, finished_ms));
            });
        }

        // --- GPU side: coarse-grained, synchronous on this (calling) thread —
        // wgpu's `device.poll(Wait)` inside `readback()` blocks the same way
        // cudarc's copies do. Runs concurrently with the CPU workers above. ---
        let gpu_t0 = std::time::Instant::now();
        dispatch(&gpu_committed);

        let leftover_len = steal_queue.lock().unwrap().len();
        let mut gpu_steal_dispatched = false;
        if leftover_len as u32 >= cfg.min_steal_tiles {
            let leftover: Vec<[u32; 4]> = std::mem::take(&mut *steal_queue.lock().unwrap()).into_iter().collect();
            if !leftover.is_empty() {
                dispatch(&leftover);
                gpu_steal_dispatched = true;
            }
        }

        let gpu_did_anything = !gpu_committed.is_empty() || gpu_steal_dispatched;
        let gpu_buf = if gpu_did_anything {
            compute.readback(device, queue)
        } else {
            vec![0.0f32; (w * h) as usize]
        };
        let gpu_ms = gpu_t0.elapsed().as_secs_f32() * 1000.0;

        (gpu_buf, gpu_ms, gpu_steal_dispatched)
    });

    let mut cpu_pixels = 0u64;
    let mut cpu_stolen_tile_count = 0u32;
    let mut cpu_ms = 0.0f32;
    let mut buf = gpu_buf;
    for _ in 0..n_workers {
        let (scratch, index, stolen, finished_ms) = rx.recv().unwrap();
        cpu_stolen_tile_count += stolen;
        cpu_ms = cpu_ms.max(finished_ms);
        for ([x0, y0, tw, th], offset) in index {
            cpu_pixels += (tw as u64) * (th as u64);
            for row in 0..th {
                let dst = ((y0 + row) * w + x0) as usize;
                let src = offset + (row * tw) as usize;
                buf[dst..dst + tw as usize].copy_from_slice(&scratch[src..src + tw as usize]);
            }
        }
    }

    controller.update(gpu_ms, cpu_ms);

    HeterogeneousResult {
        buf,
        gpu_ms,
        cpu_ms,
        cpu_tile_frac: if total_pixels > 0 { cpu_pixels as f32 / total_pixels as f32 } else { 0.0 },
        gpu_steal_dispatched,
        cpu_stolen_tile_count,
    }
}
