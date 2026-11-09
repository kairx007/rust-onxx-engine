use std::{env, path::PathBuf, process::Command};

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    let clang_output = Command::new("xcrun")
        .args(["--find", "clang"])
        .output()
        .expect("Xcode Command Line Tools are required to build the bundled ONNX Runtime on macOS");
    if !clang_output.status.success() {
        panic!("xcrun could not locate Apple Clang required by the bundled ONNX Runtime");
    }
    let clang = PathBuf::from(String::from_utf8_lossy(&clang_output.stdout).trim());

    let resource_output = Command::new(&clang)
        .arg("-print-resource-dir")
        .output()
        .expect("could not locate Apple Clang's compiler runtime directory");
    if !resource_output.status.success() {
        panic!("Apple Clang could not report its compiler runtime directory");
    }
    let runtime_dir =
        PathBuf::from(String::from_utf8_lossy(&resource_output.stdout).trim()).join("lib/darwin");
    if !runtime_dir.join("libclang_rt.osx.a").is_file() {
        panic!(
            "Apple Clang availability runtime was not found at {}",
            runtime_dir.join("libclang_rt.osx.a").display()
        );
    }

    println!("cargo:rustc-link-search=native={}", runtime_dir.display());
    println!("cargo:rustc-link-lib=static=clang_rt.osx");
    println!("cargo:rustc-link-arg=-mmacosx-version-min=13.4");
}
