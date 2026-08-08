//! Does the scheduler's tiled GPU dispatch cost more than one whole-frame
//! launch covering the same pixels? `render_heterogeneous` always goes through
//! `dispatch_tiled[_f32]`, so if per-tile launch geometry is expensive the
//! scheduler pays it even when the CPU is assigned nothing.
//!
//! Run: cargo run --release --features cuda --example tiled_dispatch_probe
#[cfg(not(feature = "cuda"))]
fn main() { println!("needs --features cuda"); }

#[cfg(feature = "cuda")]
fn main() {
    use novafractal::fractal::pixel_grid;
    use novafractal::gpu::cuda::CudaFractal;
    use novafractal::gui::viewport::Viewport;
    use std::time::Instant;
    const W: u32 = 1920; const H: u32 = 1080; const MI: u32 = 1000; const R: usize = 40;
    let vp = Viewport { center: [-0.5, 0.0], zoom: 1.0, width: W, height: H };
    let pg = pixel_grid(&vp);
    let mut cuda = CudaFractal::new(W, H);

    // Uniform tilings covering the identical frame — only launch geometry varies.
    let grid = |t: u32| { let mut v=Vec::new(); let mut y=0;
        while y<H { let th=t.min(H-y); let mut x=0;
            while x<W { let tw=t.min(W-x); v.push([x,y,tw,th]); x+=tw; } y+=th; } v };

    println!("\n=== GPU dispatch geometry, identical 1920x1080 frame ===");
    let mut base = 0.0;
    for (label, tiles) in [
        ("plain render() [Morton, no tiles]".to_string(), vec![]),
        ("1 tile (whole frame)".to_string(), vec![[0,0,W,H]]),
        (format!("{} tiles (128px)", grid(128).len()), grid(128)),
        (format!("{} tiles (64px)", grid(64).len()), grid(64)),
        (format!("{} tiles (32px)", grid(32).len()), grid(32)),
        (format!("{} tiles (16px)", grid(16).len()), grid(16)),
    ] {
        let mut run = || {
            if tiles.is_empty() {
                std::hint::black_box(cuda.render(pg.re_start,pg.im_start,pg.re_step,pg.im_step,-0.4,0.6,MI,0));
            } else {
                cuda.dispatch_tiled_f32(&tiles,pg.re_start,pg.im_start,pg.re_step,pg.im_step,-0.4,0.6,MI,0);
                std::hint::black_box(cuda.readback());
            }
        };
        for _ in 0..8 { run(); }
        let t0 = Instant::now();
        for _ in 0..R { run(); }
        let ms = t0.elapsed().as_secs_f64()*1000.0/R as f64;
        if base == 0.0 { base = ms; }
        println!("  {label:<34} {ms:7.3} ms   {:+7.1}% vs plain", (ms-base)/base*100.0);
    }
    println!("\n  readback is ~1.33 ms of every row above; the rest is kernel + launch.");
}
