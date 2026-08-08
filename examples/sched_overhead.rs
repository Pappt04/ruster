//! Breaks the heterogeneous scheduler's ~1.3 ms fixed per-frame overhead into
//! its components, by reproducing each piece of machinery with the actual
//! fractal work removed. Tells us how much of that overhead is recoverable and
//! which part to attack first.
//!
//! Mirrors `scheduler::render_heterogeneous`'s structure: rayon::scope with one
//! spawn per worker, two Mutex<VecDeque> tile queues, a Vec<f32> allocated per
//! CPU tile, an mpsc message per worker, and a serial row-by-row merge.
//!
//! Run: cargo run --release --example sched_overhead

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;

const W: u32 = 1920;
const H: u32 = 1080;
const REPS: usize = 200;

fn bench<F: FnMut()>(name: &str, mut f: F) -> f64 {
    for _ in 0..20 { f(); }
    let t0 = Instant::now();
    for _ in 0..REPS { f(); }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / REPS as f64;
    println!("  {name:<52} {ms:8.4} ms");
    ms
}

/// Same tiling `classifier::partition_frame` produces at its 128px top level,
/// then split down to `tile` px so tile counts match the real runs.
fn tiles(tile: u32) -> Vec<[u32; 4]> {
    let mut v = Vec::new();
    let mut y = 0;
    while y < H {
        let th = tile.min(H - y);
        let mut x = 0;
        while x < W {
            let tw = tile.min(W - x);
            v.push([x, y, tw, th]);
            x += tw;
        }
        y += th;
    }
    v
}

fn main() {
    let n_workers = rayon::current_num_threads().max(1);
    println!("\n=== scheduler fixed overhead, work removed (rayon threads = {n_workers}) ===");
    println!("  frame {W}x{H}\n");

    // 1. rayon::scope with one spawn per worker + an mpsc message each, queues
    //    empty. This is the floor the scheduler pays even when the CPU is
    //    assigned nothing at all.
    let scope_ms = bench("rayon::scope + N spawns + N channel msgs (no work)", || {
        let (tx, rx) = std::sync::mpsc::channel::<u32>();
        rayon::scope(|s| {
            for _ in 0..n_workers {
                let tx = tx.clone();
                s.spawn(move |_| { let _ = tx.send(0); });
            }
        });
        drop(tx);
        let mut n = 0u32;
        for _ in 0..n_workers { n += rx.recv().unwrap(); }
        std::hint::black_box(n);
    });

    // 2. Add the two mutex-guarded queues and drain them, still with no
    //    per-tile work — isolates lock contention at realistic tile counts.
    for tile in [128u32, 64, 32, 16] {
        let t = tiles(tile);
        let count = t.len();
        let label = format!("  + Mutex<VecDeque> drain, {count} tiles ({tile}px)");
        let ms = bench(&label, || {
            let q: Mutex<VecDeque<[u32; 4]>> = Mutex::new(t.iter().copied().collect());
            let steal: Mutex<VecDeque<[u32; 4]>> = Mutex::new(VecDeque::new());
            let (tx, rx) = std::sync::mpsc::channel::<u32>();
            rayon::scope(|s| {
                for _ in 0..n_workers {
                    let tx = tx.clone();
                    let (q, steal) = (&q, &steal);
                    s.spawn(move |_| {
                        let mut got = 0u32;
                        loop {
                            let claim = {
                                let mut g = q.lock().unwrap();
                                match g.pop_front() {
                                    Some(t) => Some(t),
                                    None => { drop(g); steal.lock().unwrap().pop_front() }
                                }
                            };
                            match claim { Some(_) => got += 1, None => break }
                        }
                        let _ = tx.send(got);
                    });
                }
            });
            drop(tx);
            let mut n = 0u32;
            for _ in 0..n_workers { n += rx.recv().unwrap(); }
            std::hint::black_box(n);
        });
        println!("      -> lock traffic alone: {:+.4} ms over the scope floor", ms - scope_ms);
    }

    // 3. The allocate-per-tile + serial row-by-row merge, on its own.
    println!();
    for tile in [128u32, 32, 16] {
        let t = tiles(tile);
        let count = t.len();
        let mut out = vec![0.0f32; (W * H) as usize];
        let label = format!("Vec-per-tile alloc + serial merge, {count} tiles ({tile}px)");
        bench(&label, || {
            let local: Vec<([u32; 4], Vec<f32>)> = t.iter()
                .map(|&d| (d, vec![0.0f32; (d[2] * d[3]) as usize]))
                .collect();
            for ([x0, y0, tw, th], buf) in &local {
                for row in 0..*th {
                    let dst = ((y0 + row) * W + x0) as usize;
                    let src = (row * tw) as usize;
                    out[dst..dst + *tw as usize]
                        .copy_from_slice(&buf[src..src + *tw as usize]);
                }
            }
            std::hint::black_box(&out);
        });
    }

    // 4. For scale: what a single full-frame memcpy of the same buffer costs.
    println!();
    let src = vec![1.0f32; (W * H) as usize];
    let mut dst = vec![0.0f32; (W * H) as usize];
    bench("(reference) one flat 8.29 MB host memcpy", || {
        dst.copy_from_slice(&src);
        std::hint::black_box(&dst);
    });
}
