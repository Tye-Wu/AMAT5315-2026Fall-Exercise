fn main() {
    println!("cargo:rerun-if-changed=src/kernel.rs");
    if std::env::var_os("CARGO_FEATURE_ENZYME").is_some() {
        let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
        let status = std::process::Command::new(std::env::var_os("RUSTC").unwrap())
            .args([
                "--edition=2024",
                "--crate-name",
                "enzyme_kernel",
                "--crate-type",
                "staticlib",
                "-C",
                "opt-level=3",
                "-C",
                "lto=fat",
                "-C",
                "panic=abort",
                "-Zautodiff=Enable",
                "src/kernel.rs",
                "-o",
            ])
            .arg(out.join("libenzyme_kernel.a"))
            .status()
            .expect("compile Enzyme kernel with pinned rustc");
        assert!(status.success(), "Enzyme kernel compilation failed");
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=static=enzyme_kernel");
    }
}
