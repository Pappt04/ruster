
pub mod classifier;
pub mod controller;

use std::collections::VecDeque;
use std::sync::Mutex;

use crate::fractal::{pixel_grid, render_cpu_tile_into, IterBuf, F32_PRECISION_THRESHOLD};
use crate::fractal::fractal_type::FractalType;
use crate::gpu::cuda::CudaFractal;
use crate::gpu::wgpu::fractal_compute::FractalCompute;
use crate::gpu::wgpu::unifroms::Uniforms;
use crate::gui::viewport::Viewport;
use controller::ThresholdController;

pub struct SchedulerConfig {
    pub max_tile_size: u32,
    pub min_tile_size: u32,
    pub steal_reserve_frac: f32,
    pub min_steal_tiles: u32,
    pub simd_cpu_tiles: bool,
    pub gpu_tiles_f32: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_tile_size: 128,
            min_tile_size: 16,
            steal_reserve_frac: 0.2,
            min_steal_tiles: 64,
            simd_cpu_tiles: true,
            gpu_tiles_f32: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HeterogeneousStats {
    pub gpu_ms: f32,
    pub cpu_ms: f32,
    pub cpu_tile_frac: f32,
    pub gpu_steal_dispatched: bool,
    pub cpu_stolen_tile_count: u32,
}

pub struct HeterogeneousResult {
    pub buf: IterBuf,
    pub gpu_ms: f32,
    pub cpu_ms: f32,
    pub cpu_tile_frac: f32,
    pub gpu_steal_dispatched: bool,
    pub cpu_stolen_tile_count: u32,
}

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

    type WorkerOut = (Vec<f32>, Vec<([u32; 4], usize)>, u32, f32);
    let (tx, rx) = std::sync::mpsc::channel::<WorkerOut>();
    let cpu_t0 = std::time::Instant::now();

    let (gpu_ms, gpu_steal_dispatched) = rayon::scope(|s| {
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
        if gpu_did_anything { cuda.readback_into(out); } else { out.fill(0.0); }
        let gpu_ms = gpu_t0.elapsed().as_secs_f32() * 1000.0;

        (gpu_ms, gpu_steal_dispatched)
    });

    let mut cpu_pixels = 0u64;
    let mut cpu_stolen_tile_count = 0u32;
    let mut cpu_ms = 0.0f32;
    let buf = &mut *out;
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

    HeterogeneousStats {
        gpu_ms,
        cpu_ms,
        cpu_tile_frac: if total_pixels > 0 { cpu_pixels as f32 / total_pixels as f32 } else { 0.0 },
        gpu_steal_dispatched,
        cpu_stolen_tile_count,
    }
}

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
