fn main() {
    #[cfg(feature = "cuda")]
    compile_cuda();
}

#[cfg(feature = "cuda")]
fn compile_cuda() {
    use std::path::PathBuf;
    use std::process::Command;

    let cu  = "src/fractal/fractal.cu";
    let ptx = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("fractal.ptx");

    let status = Command::new("nvcc")
        .args([
            "--ptx",
            "--gpu-architecture=sm_86", // Ampere (RTX 30xx)
            "-O3",
            "--ftz=false",       // keep denormals (needed for smooth coloring accuracy)
            "--prec-div=true",   // full precision division
            "--fmad=true",       // allow fused multiply-add (free accuracy boost)
            "-o", ptx.to_str().unwrap(),
            cu,
        ])
        .status()
        .expect("nvcc not found — install the CUDA toolkit");

    assert!(status.success(), "nvcc failed");

    println!("cargo:rerun-if-changed={cu}");
    println!("cargo:rustc-env=FRACTAL_PTX={}", ptx.display());
}
