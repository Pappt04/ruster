//! Lists every GPU adapter wgpu can see, and which one
//! `PowerPreference::HighPerformance` selects — i.e. which device a plain
//! wgpu benchmark in this project would run on by default.
//!
//! This exists because the answer was, for a long time, "not the RTX 3050":
//! the NVIDIA stack was installed compute-only, so there was no NVIDIA Vulkan
//! ICD and wgpu silently fell back to the integrated AMD Vega. See
//! results/summary.md §1.1.
//!
//! The Vega is no longer just a footgun to guard against — it's a real
//! second wgpu data point. `bench_runner --backend wgpu-igpu` and
//! `WGPU_ADAPTER=integrated cargo bench --bench fractal_bench` both select it
//! explicitly by `DeviceType`, same as this example does below.
//!
//! Run: cargo run --release --example adapters

fn main() {
    let instance = wgpu::Instance::default();

    println!("\n=== all adapters visible to wgpu ===");
    for a in instance.enumerate_adapters(wgpu::Backends::all()) {
        let i = a.get_info();
        println!(
            "  {:<8} | {:<52} | {:?}\n           driver: {} {}",
            format!("{:?}", i.backend), i.name, i.device_type, i.driver, i.driver_info
        );
    }

    let picked = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no adapter");
    let i = picked.get_info();
    println!("\n=== HighPerformance selects ===");
    println!("  {} ({:?}, {:?})", i.name, i.device_type, i.backend);
    if i.device_type == wgpu::DeviceType::DiscreteGpu {
        println!("  -> discrete GPU: wgpu and CUDA are on the same silicon, comparison is fair.");
    } else {
        println!("  -> NOT a discrete GPU: wgpu numbers are NOT comparable to CUDA numbers.");
    }
}
