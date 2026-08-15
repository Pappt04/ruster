//! Times `fractal_kernel_f32` compiled four ways, to separate the cost of the
//! two self-inflicted handicaps in the CUDA path from everything else:
//!   * `--fmad=false` (build.rs, applied file-wide for CPU/GPU bit-exactness)
//!   * the fp64 `in_period3_bulb` call inside the otherwise-f32 kernel
//!
//! Kernel-only timing: launches without any device-to-host copy, syncing via
//! `synchronize()` so the measurement is pure compute.
//!
//! Run: cargo run --release --features cuda --example ptx_variants

#[cfg(not(feature = "cuda"))]
fn main() { println!("needs --features cuda"); }

#[cfg(feature = "cuda")]
fn main() {
    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
    use cudarc::nvrtc::Ptx;
    use novafractal::fractal::pixel_grid;
    use novafractal::gui::viewport::Viewport;
    use std::time::Instant;

    const W: u32 = 1920;
    const H: u32 = 1080;
    const MAX_ITER: u32 = 1000;
    const REPS: usize = 50;

    // Generate the four variants from the shipped fractal.cu, so this probe
    // stays reproducible instead of depending on files built by hand.
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/ptx_variants");
    std::fs::create_dir_all(&dir).unwrap();
    let stock = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/gpu/fractal.cu"),
    ).unwrap();
    // The line under test. `mandelbrot_f32` now uses the f32 predicate (that is
    // the fix); the comparison variant reverts it to the fp64 one so the
    // ablation that motivated the change stays reproducible from the tree.
    const F32_CALL: &str = "if (in_period3_bulb_f32(cr, ci)) return (float)max_iter;";
    assert_eq!(stock.matches(F32_CALL).count(), 1, "fractal.cu changed — update this probe");
    let f64bulb = stock.replace(
        F32_CALL,
        "if (in_period3_bulb((double)cr, (double)ci)) return (float)max_iter;",
    );
    std::fs::write(dir.join("stock.cu"), &stock).unwrap();
    std::fs::write(dir.join("f64bulb.cu"), &f64bulb).unwrap();
    for src in ["stock", "f64bulb"] {
        for fmad in ["false", "true"] {
            let st = std::process::Command::new("nvcc")
                .args(["--ptx", "--gpu-architecture=sm_86", "-O3", "--ftz=false", "--prec-div=true"])
                .arg(format!("--fmad={fmad}"))
                .arg("-o").arg(dir.join(format!("{src}_fmad{fmad}.ptx")))
                .arg(dir.join(format!("{src}.cu")))
                .status().expect("nvcc not found");
            assert!(st.success(), "nvcc failed for {src}/fmad={fmad}");
        }
    }
    let dir = dir.display().to_string();

    let vp = Viewport { center: [-0.5, 0.0], zoom: 1.0, width: W, height: H };
    let pg = pixel_grid(&vp);
    let dev = CudaDevice::new(0).unwrap();
    let mut out = dev.alloc_zeros::<f32>((W * H) as usize).unwrap();

    // Same padded-power-of-two-square Morton grid `CudaFractal::morton_cfg` uses.
    let dim = W.max(H).next_power_of_two();
    let blocks = ((dim as u64 * dim as u64 + 255) / 256) as u32;
    let cfg = LaunchConfig {
        block_dim: (16, 16, 1),
        grid_dim: (blocks, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("\n=== fractal_kernel_f32, {W}x{H}, max_iter={MAX_ITER}, zoom=1, kernel-only ===\n");
    let variants = [
        ("stock_fmadfalse", "SHIPPED: --fmad=false + f32  bulb check"),
        ("stock_fmadtrue", "--fmad=true  + f32  bulb check"),
        ("f64bulb_fmadfalse", "--fmad=false + fp64 bulb check (pre-fix)"),
        ("f64bulb_fmadtrue", "--fmad=true  + fp64 bulb check (pre-fix)"),
    ];

    let mut baseline = 0.0f64;
    for (i, (name, desc)) in variants.iter().enumerate() {
        let module = format!("m_{name}");
        let ptx = Ptx::from_src(std::fs::read_to_string(format!("{dir}/{name}.ptx")).unwrap());
        dev.load_ptx(ptx, &module, &["fractal_kernel_f32"]).unwrap();
        let f = dev.get_func(&module, "fractal_kernel_f32").unwrap();

        let mut run = || {
            let f = f.clone();
            unsafe {
                f.launch(cfg, (
                    &mut out,
                    pg.re_start as f32, pg.im_start as f32,
                    pg.re_step as f32, pg.im_step as f32,
                    -0.4f32, 0.6f32, MAX_ITER, 0u32, W, H,
                ))
            }.unwrap();
            dev.synchronize().unwrap();
        };
        for _ in 0..10 { run(); }
        let t0 = Instant::now();
        for _ in 0..REPS { run(); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
        if i == 0 { baseline = ms; }
        println!("  {desc:<40} {ms:7.3} ms   {:+6.1}%", (ms - baseline) / baseline * 100.0);
    }
    println!("\n  (row 1 is what the project ships; rows 3-4 are the pre-fix kernel)");

    // ── correctness: how many pixels does the f32 bulb check actually change? ──
    //
    // The period-3 check is an interior shortcut ("return max_iter"), so an f32
    // rounding difference can only reclassify pixels within ~1e-7 of the bulb
    // boundary. This measures that instead of assuming it.
    println!("\n=== pixel-level effect of the f32 period-3 check ===");
    println!("  {:<28} {:>10} {:>10} {:>12} {:>10}", "viewport", "differing", "of total", "% of frame", "max |Δ|");
    let render = |name: &str, vp: &Viewport| -> Vec<f32> {
        let pg = pixel_grid(vp);
        let f = dev.get_func(&format!("m_{name}"), "fractal_kernel_f32").unwrap();
        let mut buf = dev.alloc_zeros::<f32>((W * H) as usize).unwrap();
        unsafe {
            f.launch(cfg, (
                &mut buf,
                pg.re_start as f32, pg.im_start as f32,
                pg.re_step as f32, pg.im_step as f32,
                -0.4f32, 0.6f32, MAX_ITER, 0u32, W, H,
            ))
        }.unwrap();
        dev.dtoh_sync_copy(&buf).unwrap()
    };
    // Whole set, then two views centred on a period-3 bulb (where the check is
    // load-bearing), then a deep zoom.
    // The bulb radius is 0.07371484375, so a view centred on the nucleus only
    // contains the boundary while its half-width (2/zoom) exceeds that — hence
    // zoom 30 frames it and zoom 3000 sits entirely inside. To actually stress
    // the predicate we also centre ON the boundary circle (nucleus + radius),
    // where the arc crosses the frame at any zoom.
    const NUC: [f64; 2] = [-0.1225611668766536, 0.7448617666197442];
    const RAD: f64 = 0.07371484375;
    let views: &[(&str, [f64; 2], f64)] = &[
        ("whole set", [-0.5, 0.0], 1.0),
        ("period-3 bulb (boundary in view)", NUC, 30.0),
        ("bulb interior", NUC, 3000.0),
        ("ON boundary arc, zoom 1e2", [NUC[0] + RAD, NUC[1]], 1e2),
        ("ON boundary arc, zoom 1e4", [NUC[0] + RAD, NUC[1]], 1e4),
        ("ON boundary arc, zoom 1e6", [NUC[0] + RAD, NUC[1]], 1e6),
        ("seahorse valley", [-0.75, 0.1], 100.0),
    ];
    for &(label, center, zoom) in views {
        let vp = Viewport { center, zoom, width: W, height: H };
        let a = render("f64bulb_fmadfalse", &vp);
        let b = render("stock_fmadfalse", &vp);
        let mut diff = 0usize;
        let mut maxd = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            if x != y {
                diff += 1;
                maxd = maxd.max((x - y).abs());
            }
        }
        let total = a.len();
        println!("  {label:<28} {diff:>10} {total:>10} {:>11.5}% {maxd:>10.2}",
                 diff as f64 / total as f64 * 100.0);
    }
}
