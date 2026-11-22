# Phase 5 Benchmark Report

Generated: 2026-04-20 16:39:41

Model: `transformer.onnx`

## Latency Comparison

| Environment | Mean (ms) | Std (ms) | P99 (ms) |
|---|---:|---:|---:|
| Python (ONNX Runtime) | 2.0333 | 2.4195 | 8.8862 |
| C++ (ONNX Runtime) | 1.1590 | 2.0525 | 9.7863 |

Speedup (Python mean / C++ mean): **1.75x**

C++ target check (<1 ms mean): **FAIL**

## Full CSV Inference (C++)

| Metric | Value |
|---|---:|
| Total inference time (ms) | 545.0594 |
| Mean per window (ms) | 1.0901 |
| Throughput (windows/s) | 917.3312 |
| Accuracy (%) | 33.0000 |

## Commands Used

```text
C:\Users\mitta\OneDrive\Desktop\DL project\DeepLOB-HFT-Cpp\.venv\Scripts\python.exe src/benchmark_python.py --model transformer.onnx --warmup 20 --runs 200
cpp_inference\build-mingw\benchmark.exe transformer.onnx 20 200
cpp_inference\build-mingw\full_inference.exe transformer.onnx data/raw/FI2010_test.csv data/processed/mu.npy data/processed/sigma.npy 1 500 10
```
