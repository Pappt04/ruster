//! Dumps `scheduler::classifier::partition_frame`'s tile classification for a
//! given viewport to JSON — GPU tiles and CPU tiles as `[x0, y0, w, h]`
//! rectangles — so it can be rendered outside the app. Pure CPU corner
//! sampling, no GPU dispatch (`classifier`'s own module doc), but it lives
//! behind `crate::scheduler`, which `lib.rs` only compiles under
//! `--features cuda` — hence the feature gate below despite not touching
//! CUDA at all.
//!
//! Pair with `scripts/render_classification.py` to turn the JSON into a
//! red/green tile-map PNG (GPU = red, CPU = green) — see that script's own
//! doc comment for why the split renderer is a plain image-drawing script
//! rather than more Rust.
//!
//! Run: cargo run --release --features cuda --example classify_dump -- \
//!          [--width 1920] [--height 1080] [--zoom 1.0] [--center RE,IM]
//!          [--fractal mandelbrot|julia|newton|nova] [--julia-c RE,IM]
//!          [--max-iter 1000] [--max-tile-size 128] [--min-tile-size 16]
//!          [--threshold 0.02] [--out results/figures/classification.json]

#[cfg(not(feature = "cuda"))]
fn main() { println!("needs --features cuda (classifier lives behind the scheduler module, which is cuda-gated — see this file's doc comment)"); }

#[cfg(feature = "cuda")]
fn main() {
    use novafractal::fractal::{pixel_grid, FractalType};
    use novafractal::gui::viewport::Viewport;
    use novafractal::scheduler::classifier::partition_frame;
    use std::fmt::Write as _;

    struct Args {
        width: u32,
        height: u32,
        zoom: f64,
        center: [f64; 2],
        fractal: FractalType,
        julia_c: [f64; 2],
        max_iter: u32,
        max_tile_size: u32,
        min_tile_size: u32,
        threshold: f32,
        out: String,
    }

    fn parse_pair(s: &str) -> [f64; 2] {
        let parts: Vec<&str> = s.split(',').collect();
        assert_eq!(parts.len(), 2, "expected RE,IM, got {s}");
        [parts[0].parse().unwrap(), parts[1].parse().unwrap()]
    }

    let mut a = Args {
        width: 1920,
        height: 1080,
        zoom: 1.0,
        center: [-0.5, 0.0],
        fractal: FractalType::Mandelbrot,
        julia_c: [-0.4, 0.6],
        max_iter: 1000,
        max_tile_size: 128,
        min_tile_size: 16,
        // Matches `ThresholdController::new(0.02)`'s starting guess, used
        // identically across every other scheduler probe in this directory.
        threshold: 0.02,
        out: "results/figures/classification.json".to_string(),
    };

    let raw: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "--width"         => { i += 1; a.width = raw[i].parse().unwrap(); }
            "--height"        => { i += 1; a.height = raw[i].parse().unwrap(); }
            "--zoom"          => { i += 1; a.zoom = raw[i].parse().unwrap(); }
            "--center"        => { i += 1; a.center = parse_pair(&raw[i]); }
            "--julia-c"       => { i += 1; a.julia_c = parse_pair(&raw[i]); }
            "--max-iter"      => { i += 1; a.max_iter = raw[i].parse().unwrap(); }
            "--max-tile-size" => { i += 1; a.max_tile_size = raw[i].parse().unwrap(); }
            "--min-tile-size" => { i += 1; a.min_tile_size = raw[i].parse().unwrap(); }
            "--threshold"     => { i += 1; a.threshold = raw[i].parse().unwrap(); }
            "--out"           => { i += 1; a.out = raw[i].clone(); }
            "--fractal" => {
                i += 1;
                a.fractal = match raw[i].as_str() {
                    "mandelbrot" => FractalType::Mandelbrot,
                    "julia"      => FractalType::Julia,
                    "newton"     => FractalType::Newton,
                    "nova"       => FractalType::Nova,
                    x => panic!("unknown fractal: {x}"),
                };
            }
            x => panic!("unknown flag: {x}"),
        }
        i += 1;
    }

    let vp = Viewport { center: a.center, zoom: a.zoom, width: a.width, height: a.height };
    let pg = pixel_grid(&vp);

    // `partition_frame` already clamps max_tile_size >= min_tile_size
    // internally, so CLI input doesn't need guarding here.
    let (gpu_tiles, cpu_tiles) = partition_frame(
        &pg, a.fractal, a.julia_c, a.max_iter, a.width, a.height,
        a.max_tile_size, a.min_tile_size, a.threshold,
    );

    let gpu_pixels: u64 = gpu_tiles.iter().map(|&[_, _, w, h]| (w as u64) * (h as u64)).sum();
    let cpu_pixels: u64 = cpu_tiles.iter().map(|&[_, _, w, h]| (w as u64) * (h as u64)).sum();
    let total = (a.width as u64) * (a.height as u64);

    println!(
        "{} gpu tiles ({:.1}% of pixels), {} cpu tiles ({:.1}% of pixels)",
        gpu_tiles.len(), gpu_pixels as f64 / total as f64 * 100.0,
        cpu_tiles.len(), cpu_pixels as f64 / total as f64 * 100.0,
    );

    let mut json = String::new();
    write!(json, "{{\"width\":{},\"height\":{},\"gpu_tiles\":[", a.width, a.height).unwrap();
    for (i, &[x, y, w, h]) in gpu_tiles.iter().enumerate() {
        if i > 0 { json.push(','); }
        write!(json, "[{x},{y},{w},{h}]").unwrap();
    }
    json.push_str("],\"cpu_tiles\":[");
    for (i, &[x, y, w, h]) in cpu_tiles.iter().enumerate() {
        if i > 0 { json.push(','); }
        write!(json, "[{x},{y},{w},{h}]").unwrap();
    }
    json.push_str("]}");

    if let Some(parent) = std::path::Path::new(&a.out).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&a.out, json).unwrap();
    println!("wrote {}", a.out);
}
