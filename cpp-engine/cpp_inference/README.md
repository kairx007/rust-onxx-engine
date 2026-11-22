# DeepLOB C++ Inference Engine Guide

This directory contains the C++ implementation for running the TransformerLOB (and baseline DeepLOB) models via ONNX Runtime. This is intended to achieve sub-millisecond inference latency suitable for High-Frequency Trading (HFT) applications.

This guide will walk you through setting up the environment, compiling the C++ sources, and executing the inference engine safely and efficiently.

## Prerequisites

Before starting, ensure you have the following installed on your Windows system:
1. **PowerShell** (available by default on Windows).
2. **CMake** (version 3.16 or higher). If you have installed project dependencies in a virtual environment via `pip`, you can use the `cmake` available there (e.g., `.venv/Scripts/cmake.exe`).
3. **C++ Compiler**: A robust compiler such as **MinGW (GCC)** or **MSVC** (Visual Studio).

## Step 1: Install ONNX Runtime SDK

We provide a script to automatically download and extract the required ONNX Runtime SDK library for Windows (x64) into the `third_party/` directory.

Open a PowerShell terminal from the root of the project (or inside `cpp_inference/`) and run:
```powershell
cd cpp_inference
.\setup_onnxruntime.ps1
```

Once complete, the script will output the path to the extracted ONNX Runtime directory. It will look something like `third_party\onnxruntime-win-x64-1.24.4`.

## Step 2: Configure the CMake Build

Set an environment variable linking to the exact path of the installed SDK from the previous step, and use CMake to generate your project build files.

Run the following in PowerShell (assuming your terminal is at the project root):

```powershell
# Adjust the path below if the setup script downloaded a different version!
$env:ONNXRUNTIME_ROOT = "$PWD\cpp_inference\third_party\onnxruntime-win-x64-1.24.4"

# Generate build files using MinGW
cmake -S cpp_inference -B cpp_inference/build -G "MinGW Makefiles"
```

> **Note on Compilers**: If you are using Microsoft Visual Studio rather than MinGW, simply omit the `-G "MinGW Makefiles"` flag so CMake can natively target MSVC.

## Step 3: Build the Executables

Once configured, compile the inference engine and the benchmarking binaries. 

```powershell
# Build using all available cores
cmake --build cpp_inference/build -j
```

> **What this does**: The CMake script compiles three binaries: `inference.exe`, `benchmark.exe`, and `full_inference.exe`. Crucially, it also automatically copies the requisite `onnxruntime.dll` directly next to these `.exe` binaries so that you do not need to mess with your system `$env:PATH` variables.

## Step 4: Run the Inference Engine

You can safely run the compiled executables straight out of your build folder.

```powershell
cd cpp_inference/build

# 1. To run a basic, single-inference test (outputs top-1 predicted class):
.\inference.exe

# 2. To run the latency benchmark (1000 iter) and confirm sub-1-ms performance:
.\benchmark.exe

# 3. To run an end-to-end full evaluation inference over test sequences:
.\full_inference.exe
```

## Troubleshooting
- **Missing DLLs Error?** Ensure you didn't move the `.exe` files out of the `build/` directory! They strictly rely on `onnxruntime.dll` being in the exact same directory.
- **MinGW GCC Errors?** If CMake complains about unresolved linker errors with `.lib` files, don't worry. The `CMakeLists.txt` is already configured to invoke `gendef` and `dlltool` to automatically generate MinGW compatible import libraries `.dll.a` from the dynamic library. Just make sure those tools are on your System PATH (they come packaged with standard MinGW setups).
