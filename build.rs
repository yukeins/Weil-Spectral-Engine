// build.rs: Automates NVIDIA CUDA Kernel compilation using nvcc with g++-12
use std::process::Command;
use std::env;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let nvcc_path = "/usr/local/cuda/bin/nvcc";
    let gxx_path = "/usr/bin/g++-12";

    if Path::new(nvcc_path).exists() && Path::new(gxx_path).exists() {
        println!("cargo:rerun-if-changed=src/cuda_kernel.cu");

        // Compile CUDA kernel into object file
        let status = Command::new(nvcc_path)
            .args(&[
                "-ccbin", gxx_path,
                "-O3",
                "-arch=sm_86", // Architecture for RTX 2050 (Ampere compute capability 8.6)
                "-Xcompiler", "-fPIC",
                "-c", "src/cuda_kernel.cu",
                "-o", &format!("{}/cuda_kernel.o", out_dir),
            ])
            .status();

        if let Ok(st) = status {
            if st.success() {
                // Archive into static library libcuda_kernel.a
                let _ = Command::new("ar")
                    .args(&["crus", &format!("{}/libcuda_kernel.a", out_dir), &format!("{}/cuda_kernel.o", out_dir)])
                    .status();

                println!("cargo:rustc-link-search=native={}", out_dir);
                println!("cargo:rustc-link-lib=static=cuda_kernel");
                println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/x86_64-linux/lib");
                println!("cargo:rustc-link-lib=dylib=cudart");
            }
        }
    }
}
