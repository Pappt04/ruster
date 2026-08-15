fn main() {
    #[cfg(feature = "cuda")]
    compile_cuda();
}

/// Compiles `fractal.cu` to PTX via `nvcc` at build time and exposes its
/// path to the crate through the `FRACTAL_PTX` environment variable (read
/// back with `env!("FRACTAL_PTX")` in `gpu::cuda::CudaFractal::from_device`),
/// so the CUDA kernels ship embedded in the compiled binary rather than
/// needing `fractal.cu` present or nvrtc-compiled at runtime.
#[cfg(feature = "cuda")]
fn compile_cuda() {
    use std::path::PathBuf;
    use std::process::Command;

    let cu  = "src/gpu/cuda/fractal.cu";
    let ptx = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("fractal.ptx");

    let status = Command::new("nvcc")
        .args([
            "--ptx",
            "--gpu-architecture=sm_86", // Ampere (RTX 30xx) — matches the target RTX 3050
            "-O3",
            "--ftz=false",       // keep denormals (needed for smooth coloring accuracy)
            "--prec-div=true",   // full precision division
            // Fused multiply-add computes a*b+c with a single rounding step, which is
            // *more* accurate than the CPU's separate multiply-then-add — but that very
            // difference means the two backends compute a different last bit for the
            // same chaotic escape-time iteration, which can flip which iteration a
            // boundary pixel escapes on. Disabling FMA contraction makes CUDA's `+`/`*`
            // round exactly like Rust's, keeping CPU and GPU bit-identical.
            "--fmad=false",
            "-o", ptx.to_str().unwrap(),
            cu,
        ])
        .status()
        .expect("nvcc not found — install the CUDA toolkit");

    assert!(status.success(), "nvcc failed");

    println!("cargo:rerun-if-changed={cu}");
    println!("cargo:rustc-env=FRACTAL_PTX={}", ptx.display());
}
