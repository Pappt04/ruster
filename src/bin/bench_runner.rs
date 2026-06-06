/// bench_runner — deterministic workload for `perf stat` / `perf record`.
///
/// Usage:
///   bench_runner [options]
///
///   --fractal  mandelbrot|julia|newton|nova|all   (default: all)
///   --width    N                                  (default: 1920)
///   --height   N                                  (default: 1080)
///   --iters    N                                  (default: 1000)
///   --runs     N   warm-up + timed repetitions    (default: 5)
///   --threads  N   rayon threads, 0 = num_cpus    (default: 0)
///   --backend  cpu|wgpu|hybrid                    (default: cpu)
///   --scaling       run thread-scaling sweep      (flag)
///   --json          emit JSON to stdout            (flag)
///
/// Examples:
///   perf stat -e cycles,instructions,cache-misses \
///       ./target/release/bench_runner --fractal mandelbrot --runs 3
///
///   ./target/release/bench_runner --backend wgpu --width 3840 --height 2160
///
///   ./target/release/bench_runner --scaling --json > bench_results/scaling.json

use std::time::{Duration, Instant};
use novafractal::fractal::{render, flops_per_iter, FractalType};
use novafractal::gui::viewport::Viewport;

// ── CLI ───────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Args {
    fractals:   Vec<FractalType>,
    width:      u32,
    height:     u32,
    max_iter:   u32,
    runs:       usize,
    threads:    usize,
    backend:    Backend,
    scaling:    bool,
    json:       bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Backend { Cpu, Wgpu, Hybrid }

impl Default for Args {
    fn default() -> Self {
        Self {
            fractals:  FractalType::ALL.to_vec(),
            width:     1920,
            height:    1080,
            max_iter:  1000,
            runs:      5,
            threads:   0,
            backend:   Backend::Cpu,
            scaling:   false,
            json:      false,
        }
    }
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--fractal" => {
                i += 1;
                a.fractals = match raw[i].as_str() {
                    "mandelbrot" => vec![FractalType::Mandelbrot],
                    "julia"      => vec![FractalType::Julia],
                    "newton"     => vec![FractalType::Newton],
                    "nova"       => vec![FractalType::Nova],
                    "all"        => FractalType::ALL.to_vec(),
                    x => panic!("unknown fractal: {x}"),
                };
            }
            "--width"   => { i += 1; a.width    = raw[i].parse().unwrap(); }
            "--height"  => { i += 1; a.height   = raw[i].parse().unwrap(); }
            "--iters"   => { i += 1; a.max_iter = raw[i].parse().unwrap(); }
            "--runs"    => { i += 1; a.runs     = raw[i].parse().unwrap(); }
            "--threads" => { i += 1; a.threads  = raw[i].parse().unwrap(); }
            "--backend" => {
                i += 1;
                a.backend = match raw[i].as_str() {
                    "cpu"    => Backend::Cpu,
                    "wgpu"   => Backend::Wgpu,
                    "hybrid" => Backend::Hybrid,
                    x => panic!("unknown backend: {x}"),
                };
            }
            "--scaling" => a.scaling = true,
            "--json"    => a.json    = true,
            x => panic!("unknown flag: {x}"),
        }
        i += 1;
    }
    a
}

// ── measurement types ─────────────────────────────────────────────────────────

#[derive(Debug)]
struct Sample {
    fractal:        &'static str,
    backend:        &'static str,
    threads:        usize,
    width:          u32,
    height:         u32,
    max_iter:       u32,
    runs:           usize,
    median_ms:      f64,
    min_ms:         f64,
    max_ms:         f64,
    mpix_per_sec:   f64,
    total_iters:    u64,
    gflops:         f64,
    /// bytes read (iteration buffer) + bytes written (color buffer)
    mem_traffic_mb: f64,
}

// ── CPU benchmarking ──────────────────────────────────────────────────────────

fn bench_cpu(fractal: FractalType, vp: &Viewport, max_iter: u32, runs: usize) -> Sample {
    let pixels = (vp.width * vp.height) as u64;
    let julia_c = [-0.4f64, 0.6];

    // warm-up (not timed)
    let _ = render(vp, fractal, julia_c, max_iter);

    let mut times: Vec<Duration> = Vec::with_capacity(runs);
    let mut last_buf = Vec::new();

    for _ in 0..runs {
        let t = Instant::now();
        last_buf = render(vp, fractal, julia_c, max_iter);
        times.push(t.elapsed());
    }

    // analytical GFLOPS: sum actual iteration values from the last buf
    // (buf[i] ≈ float iteration count, clamped to max_iter for in-set pixels)
    let total_iters: u64 = last_buf.iter()
        .map(|&v| v.min(max_iter as f32) as u64)
        .sum();
    let fpi = flops_per_iter(fractal);

    times.sort();
    let med = times[runs / 2].as_secs_f64() * 1e3;
    let mn  = times[0].as_secs_f64()        * 1e3;
    let mx  = times[runs - 1].as_secs_f64() * 1e3;
    let med_s = times[runs / 2].as_secs_f64();

    // Memory traffic: read iter buf (f32) + write color buf (4 bytes/pixel).
    let mem_bytes = pixels * (4 + 4) as u64;

    Sample {
        fractal:        fractal.name(),
        backend:        "cpu",
        threads:        rayon::current_num_threads(),
        width:          vp.width,
        height:         vp.height,
        max_iter,
        runs,
        median_ms:      med,
        min_ms:         mn,
        max_ms:         mx,
        mpix_per_sec:   pixels as f64 / 1e6 / med_s,
        total_iters,
        gflops:         (total_iters * fpi) as f64 / med_s / 1e9,
        mem_traffic_mb: mem_bytes as f64 / 1e6,
    }
}

// ── wgpu benchmarking ─────────────────────────────────────────────────────────

fn init_wgpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    use wgpu::*;
    pollster::block_on(async {
        let instance = Instance::default();
        let adapter = instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await?;
        let (device, queue) = adapter.request_device(
            &DeviceDescriptor {
                label: Some("bench"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
                memory_hints: MemoryHints::Performance,
            },
            None,
        ).await.ok()?;
        eprintln!("[wgpu] adapter: {}", adapter.get_info().name);
        Some((device, queue))
    })
}

fn bench_wgpu(fractal: FractalType, vp: &Viewport, max_iter: u32, runs: usize) -> Option<Sample> {
    use novafractal::gpu::fractal_compute::FractalCompute;
    use novafractal::gpu::unifroms::Uniforms;

    let (device, queue) = init_wgpu()?;
    let compute = FractalCompute::new(&device, vp.width, vp.height);

    let aspect = vp.width as f64 / vp.height as f64;
    let half = 2.0 / vp.zoom;
    let uniforms = Uniforms {
        re_start: (vp.center[0] + (0.5 / vp.width  as f64 - 0.5) * half * aspect * 2.0) as f32,
        im_start: (vp.center[1] + (0.5 / vp.height as f64 - 0.5) * half * 2.0) as f32,
        re_step:  (half * aspect * 2.0 / vp.width  as f64) as f32,
        im_step:  (half * 2.0          / vp.height as f64) as f32,
        julia_cr: -0.4,
        julia_ci:  0.6,
        max_iter,
        fractal:  fractal_to_u32(fractal),
        width:    vp.width,
        height:   vp.height,
        _pad:     [0; 2],
    };

    // warm-up
    let _ = compute.render(&device, &queue, uniforms);

    let pixels = (vp.width * vp.height) as u64;
    let mut times: Vec<Duration> = Vec::with_capacity(runs);
    let mut last_buf = Vec::new();

    for _ in 0..runs {
        let t = Instant::now();
        last_buf = compute.render(&device, &queue, uniforms);
        times.push(t.elapsed());
    }

    let total_iters: u64 = last_buf.iter()
        .map(|&v| v.min(max_iter as f32) as u64)
        .sum();
    let fpi = flops_per_iter(fractal);

    times.sort();
    let med_s = times[runs / 2].as_secs_f64();

    Some(Sample {
        fractal:        fractal.name(),
        backend:        "wgpu",
        threads:        1,
        width:          vp.width,
        height:         vp.height,
        max_iter,
        runs,
        median_ms:      med_s * 1e3,
        min_ms:         times[0].as_secs_f64() * 1e3,
        max_ms:         times[runs - 1].as_secs_f64() * 1e3,
        mpix_per_sec:   pixels as f64 / 1e6 / med_s,
        total_iters,
        gflops:         (total_iters * fpi) as f64 / med_s / 1e9,
        mem_traffic_mb: pixels as f64 * 8.0 / 1e6,
    })
}

fn fractal_to_u32(f: FractalType) -> u32 {
    match f {
        FractalType::Mandelbrot => 0,
        FractalType::Julia      => 1,
        FractalType::Newton     => 2,
        FractalType::Nova       => 3,
    }
}

// ── hybrid benchmarking ───────────────────────────────────────────────────────
//
// Splits the image into top (CPU) and bottom (GPU) halves, renders both
// concurrently, and combines the buffers.  The wall time is max(t_cpu, t_gpu).

fn bench_hybrid(fractal: FractalType, vp: &Viewport, max_iter: u32, runs: usize) -> Option<Sample> {
    use novafractal::gpu::fractal_compute::FractalCompute;
    use novafractal::gpu::unifroms::Uniforms;

    let (device, queue) = init_wgpu()?;

    let h_top    = vp.height / 2;
    let h_bottom = vp.height - h_top;

    let vp_top = Viewport {
        center: [vp.center[0], vp.center[1]],
        zoom:   vp.zoom,
        width:  vp.width,
        height: h_top,
    };

    // Recompute im_start for the bottom half: it starts where the top half ends.
    let aspect    = vp.width as f64 / vp.height as f64;
    let half      = 2.0 / vp.zoom;
    let im_step   = half * 2.0 / vp.height as f64;
    let im_start  = vp.center[1] + (0.5 / vp.height as f64 - 0.5) * half * 2.0;
    let im_start_bottom = im_start + h_top as f64 * im_step;

    let compute_bottom = FractalCompute::new(&device, vp.width, h_bottom);
    let uniforms_bottom = Uniforms {
        re_start: (vp.center[0] + (0.5 / vp.width as f64 - 0.5) * half * aspect * 2.0) as f32,
        im_start: im_start_bottom as f32,
        re_step:  (half * aspect * 2.0 / vp.width as f64) as f32,
        im_step:  im_step as f32,
        julia_cr: -0.4,
        julia_ci:  0.6,
        max_iter,
        fractal:  fractal_to_u32(fractal),
        width:    vp.width,
        height:   h_bottom,
        _pad:     [0; 2],
    };

    let julia_c = [-0.4f64, 0.6f64];
    // warm-up
    let _ = render(&vp_top, fractal, julia_c, max_iter);
    let _ = compute_bottom.render(&device, &queue, uniforms_bottom);

    let pixels = (vp.width * vp.height) as u64;
    let mut wall_times: Vec<Duration> = Vec::with_capacity(runs);

    for _ in 0..runs {
        let t_wall = Instant::now();

        // CPU top half in a separate thread; GPU bottom on this thread.
        // std::thread::scope ensures both finish before we record wall time.
        std::thread::scope(|s| {
            let vp_top_ref  = &vp_top;
            s.spawn(move || render(vp_top_ref, fractal, julia_c, max_iter));
            let _ = compute_bottom.render(&device, &queue, uniforms_bottom);
        });

        wall_times.push(t_wall.elapsed());
    }

    wall_times.sort();
    let med_s = wall_times[runs / 2].as_secs_f64();

    Some(Sample {
        fractal:        fractal.name(),
        backend:        "hybrid(cpu+wgpu)",
        threads:        rayon::current_num_threads(),
        width:          vp.width,
        height:         vp.height,
        max_iter,
        runs,
        median_ms:      med_s * 1e3,
        min_ms:         wall_times[0].as_secs_f64() * 1e3,
        max_ms:         wall_times[runs - 1].as_secs_f64() * 1e3,
        mpix_per_sec:   pixels as f64 / 1e6 / med_s,
        total_iters:    0,   // not tracked in hybrid mode
        gflops:         0.0, // not tracked in hybrid mode
        mem_traffic_mb: pixels as f64 * 8.0 / 1e6,
    })
}

// ── output ────────────────────────────────────────────────────────────────────

fn print_table(samples: &[Sample]) {
    println!("\n{:-<100}", "");
    println!("{:<12} {:<8} {:>8} {:>10} {:>10} {:>10} {:>12} {:>12} {:>12}",
             "Fractal", "Backend", "Threads", "Mpix/s", "GFLOPs", "Med ms",
             "Min ms", "Max ms", "MemMB");
    println!("{:-<100}", "");
    for s in samples {
        println!("{:<12} {:<8} {:>8} {:>10.2} {:>10.3} {:>10.2} {:>12.2} {:>12.2} {:>12.1}",
                 s.fractal, s.backend, s.threads,
                 s.mpix_per_sec, s.gflops,
                 s.median_ms, s.min_ms, s.max_ms,
                 s.mem_traffic_mb);
    }
    println!("{:-<100}", "");
}

fn print_json(samples: &[Sample]) {
    println!("[");
    for (i, s) in samples.iter().enumerate() {
        let comma = if i + 1 < samples.len() { "," } else { "" };
        println!("  {{\"fractal\":\"{}\",\"backend\":\"{}\",\"threads\":{},\
                  \"width\":{},\"height\":{},\"max_iter\":{},\"runs\":{},\
                  \"median_ms\":{:.4},\"min_ms\":{:.4},\"max_ms\":{:.4},\
                  \"mpix_per_sec\":{:.4},\"total_iters\":{},\"gflops\":{:.4},\
                  \"mem_traffic_mb\":{:.2}}}{}",
                 s.fractal, s.backend, s.threads,
                 s.width, s.height, s.max_iter, s.runs,
                 s.median_ms, s.min_ms, s.max_ms,
                 s.mpix_per_sec, s.total_iters, s.gflops,
                 s.mem_traffic_mb, comma);
    }
    println!("]");
}

fn print_flop_budget(fractals: &[FractalType], max_iter: u32) {
    eprintln!("\n── Analytical FLOP budget (lower bound, no FMA fusion) ──────────");
    eprintln!("{:<12} {:>10} {:>16}", "Kernel", "FLOPs/iter", "FLOPs @ max_iter");
    for &f in fractals {
        let fpi = flops_per_iter(f);
        eprintln!("{:<12} {:>10} {:>16}", f.name(), fpi, fpi * max_iter as u64);
    }
    eprintln!("smooth_iter() (ln-based) adds ~6 FLOPs per escaped pixel.");
    eprintln!("────────────────────────────────────────────────────────────────\n");
}

fn git_hash() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
        .trim()
        .to_string()
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();

    // Configure rayon thread pool.
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .expect("rayon pool init");
    }

    let vp = Viewport {
        center: [-0.5, 0.0],
        zoom:   1.0,
        width:  args.width,
        height: args.height,
    };

    if !args.json {
        eprintln!("bench_runner  git={}", git_hash());
        eprintln!("  resolution : {}×{}", args.width, args.height);
        eprintln!("  max_iter   : {}", args.max_iter);
        eprintln!("  runs       : {}", args.runs);
        eprintln!("  backend    : {:?}", args.backend);
        eprintln!("  rayon cpus : {}", rayon::current_num_threads());
        print_flop_budget(&args.fractals, args.max_iter);
    }

    let mut samples: Vec<Sample> = Vec::new();

    // ── thread-scaling sweep ─────────────────────────────────────────────────
    if args.scaling {
        let ncpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let thread_counts: Vec<usize> = (0..=ncpus.trailing_zeros())
            .map(|e| 1usize << e)
            .collect();

        for &t in &thread_counts {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .expect("rayon pool");

            for &fractal in &args.fractals {
                let mut s = pool.install(|| bench_cpu(fractal, &vp, args.max_iter, args.runs));
                s.threads = t;
                samples.push(s);
            }
        }
    } else {
        // ── single-configuration run ─────────────────────────────────────────
        for &fractal in &args.fractals {
            match args.backend {
                Backend::Cpu => {
                    samples.push(bench_cpu(fractal, &vp, args.max_iter, args.runs));
                }
                Backend::Wgpu => {
                    if let Some(s) = bench_wgpu(fractal, &vp, args.max_iter, args.runs) {
                        samples.push(s);
                    } else {
                        eprintln!("[wgpu] no GPU available — skipping");
                    }
                }
                Backend::Hybrid => {
                    // CPU-only baseline for comparison
                    samples.push(bench_cpu(fractal, &vp, args.max_iter, args.runs));
                    // Hybrid (CPU top half + wgpu bottom half)
                    if let Some(s) = bench_hybrid(fractal, &vp, args.max_iter, args.runs) {
                        samples.push(s);
                    } else {
                        eprintln!("[hybrid] no GPU available — only CPU baseline shown");
                    }
                }
            }
        }
    }

    // ── output ───────────────────────────────────────────────────────────────
    if args.json {
        print_json(&samples);
    } else {
        print_table(&samples);
    }
}
