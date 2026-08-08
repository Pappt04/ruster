//! Measures the ONE cost that decides whether the heterogeneous scheduler can
//! ever beat GPU-only rendering: the device-to-host readback.
//!
//! §3 of results/summary.md shows the scheduler is bounded by
//! `CudaFractal::readback`, a fixed full-frame `dtoh_sync_copy` that does not
//! shrink when tiles move to the CPU. This probe asks three questions:
//!
//!   1. Does copy time scale with copy SIZE? (If not, partial readback is
//!      pointless and the scheduler is unfixable on this axis.)
//!   2. How much does the per-frame `Vec` allocation in `dtoh_sync_copy` cost?
//!   3. How much does page-locking (pinning) the destination buy?
//!
//! Run: cargo run --release --features cuda --example readback_probe

#[cfg(not(feature = "cuda"))]
fn main() { println!("needs --features cuda"); }

#[cfg(feature = "cuda")]
fn main() {
    use cudarc::driver::CudaDevice;
    use std::time::Instant;

    const W: usize = 1920;
    const H: usize = 1080;
    const N: usize = W * H;
    const REPS: usize = 60;
    let mb = (N * 4) as f64 / 1e6;

    let dev = CudaDevice::new(0).unwrap();
    let src = dev.alloc_zeros::<f32>(N).unwrap();

    macro_rules! time {
        ($name:expr, $mbytes:expr, $body:block) => {{
            for _ in 0..10 { $body }
            let t0 = Instant::now();
            for _ in 0..REPS { $body }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
            let mbv: f64 = $mbytes;
            println!("  {:<46} {ms:7.3} ms   {:6.1} GB/s", $name, mbv / 1e3 / (ms / 1e3));
            ms
        }};
    }

    println!("\n=== full-frame readback strategies ({W}x{H} f32 = {mb:.2} MB) ===");

    // (a) exactly what CudaFractal::readback does today: fresh Vec every call.
    let cur = time!("dtoh_sync_copy  [fresh Vec, pageable] (current)", mb, {
        std::hint::black_box(dev.dtoh_sync_copy(&src).unwrap());
    });

    // (b) same transfer, destination allocated once and reused.
    let mut dst = vec![0.0f32; N];
    let reuse = time!("dtoh_sync_copy_into [reused Vec, pageable]", mb, {
        dev.dtoh_sync_copy_into(&src, &mut dst).unwrap();
    });

    // (c) reused buffer, page-locked so the driver DMAs straight into it
    // instead of staging through its own internal pinned bounce buffer.
    let mut pinned = vec![0.0f32; N];
    let reg = unsafe {
        cudarc::driver::sys::lib().cuMemHostRegister_v2(
            pinned.as_mut_ptr() as *mut std::ffi::c_void, N * 4, 0,
        )
    };
    let pin_ms = if reg == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        let ms = time!("dtoh_sync_copy_into [reused Vec, PINNED]", mb, {
            dev.dtoh_sync_copy_into(&src, &mut pinned).unwrap();
        });
        unsafe {
            cudarc::driver::sys::lib()
                .cuMemHostUnregister(pinned.as_mut_ptr() as *mut std::ffi::c_void);
        }
        Some(ms)
    } else {
        println!("  (cuMemHostRegister failed: {reg:?} — skipping pinned variant)");
        None
    };

    println!("\n  reused-vs-fresh Vec : {:+.1}%", (reuse - cur) / cur * 100.0);
    if let Some(p) = pin_ms {
        println!("  pinned-vs-pageable  : {:+.1}%", (p - reuse) / reuse * 100.0);
        println!("  pinned-vs-current   : {:+.1}%  <-- free win, no algorithm change",
                 (p - cur) / cur * 100.0);
    }

    // ── Q1: does copy time scale with copy size? ─────────────────────────────
    //
    // The decisive question. If a 25% copy costs ~25% of the time, a partial
    // readback proportional to the GPU's tile share is worth building. If it is
    // dominated by fixed per-call latency, the scheduler's core bottleneck
    // cannot be removed this way at all.
    println!("\n=== does readback time scale with size? (reused pageable dst) ===");
    println!("  {:<10} {:>9} {:>9} {:>10} {:>9}", "fraction", "MB", "ms", "vs full", "GB/s");
    for frac in [1.0f64, 0.75, 0.5, 0.25, 0.10, 0.05, 0.01] {
        let n = (((N as f64) * frac) as usize).max(1);
        let mut sub = vec![0.0f32; n];
        let view = src.slice(0..n);
        for _ in 0..10 { dev.dtoh_sync_copy_into(&view, &mut sub).unwrap(); }
        let t0 = Instant::now();
        for _ in 0..REPS { dev.dtoh_sync_copy_into(&view, &mut sub).unwrap(); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
        let sub_mb = (n * 4) as f64 / 1e6;
        println!("  {:<10} {sub_mb:>9.2} {ms:>9.3} {:>9.1}% {:>9.1}",
                 format!("{:.0}%", frac * 100.0), ms / reuse * 100.0,
                 sub_mb / 1e3 / (ms / 1e3));
    }

    // ── Q2: cost of many small copies (the naive per-tile readback) ──────────
    println!("\n=== per-call overhead: same total bytes split into N copies ===");
    println!("  {:<10} {:>12} {:>9} {:>12}", "copies", "bytes each", "ms", "vs 1 copy");
    for ncopies in [1usize, 8, 64, 135, 340, 825] {
        let chunk = N / ncopies;
        let t0 = Instant::now();
        for _ in 0..10 {
            for i in 0..ncopies {
                let lo = i * chunk;
                let view = src.slice(lo..lo + chunk);
                dev.dtoh_sync_copy_into(&view, &mut dst[lo..lo + chunk]).unwrap();
            }
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / 10.0;
        println!("  {ncopies:<10} {:>12} {ms:>9.3} {:>11.2}x", chunk * 4, ms / reuse);
    }
    println!("\n  (135 = GPU tile count at zoom 1e4, 340 at zoom 1e0, 825 at zoom 1e2)");
}
