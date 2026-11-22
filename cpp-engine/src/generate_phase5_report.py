"""
Generate a Phase 5-ready comparison report by running:
1) Python ONNX benchmark
2) C++ ONNX benchmark
3) Full-dataset C++ CSV inference

Usage:
    python src/generate_phase5_report.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, Tuple


def resolve_model_path(model_arg: str | None) -> str:
    if model_arg:
        return model_arg
    for candidate in ("transformer.onnx", "deeplob.onnx"):
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("No ONNX model found. Expected transformer.onnx or deeplob.onnx")


def run_command(cmd: list[str]) -> Tuple[int, str]:
    completed = subprocess.run(cmd, capture_output=True, text=True)
    output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
    return completed.returncode, output.strip()


def parse_latency_metrics(output: str) -> Dict[str, float]:
    patterns = {
        "mean_ms": r"Mean latency \(ms\):\s*([0-9.]+)",
        "std_ms": r"Std latency \(ms\)\s*:\s*([0-9.]+)",
        "p99_ms": r"P99 latency \(ms\)\s*:\s*([0-9.]+)",
    }
    metrics: Dict[str, float] = {}
    for key, pat in patterns.items():
        m = re.search(pat, output)
        if not m:
            raise ValueError(f"Could not parse {key} from benchmark output")
        metrics[key] = float(m.group(1))
    return metrics


def parse_full_inference_metrics(output: str) -> Dict[str, float]:
    patterns = {
        "total_ms": r"Total infer time\s*:\s*([0-9.]+)",
        "mean_window_ms": r"Mean per window\s*:\s*([0-9.]+)",
        "throughput": r"Throughput\s*:\s*([0-9.]+)",
        "accuracy_pct": r"Accuracy\s*:\s*([0-9.]+)%",
    }
    metrics: Dict[str, float] = {}
    for key, pat in patterns.items():
        m = re.search(pat, output)
        if not m:
            raise ValueError(f"Could not parse {key} from full_inference output")
        metrics[key] = float(m.group(1))
    return metrics


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 benchmark suite and generate markdown report")
    parser.add_argument("--model", default=None, help="ONNX model path (auto if omitted)")
    parser.add_argument("--cpp-benchmark", default="cpp_inference/build-mingw/benchmark.exe", help="Path to C++ benchmark executable")
    parser.add_argument("--cpp-full", default="cpp_inference/build-mingw/full_inference.exe", help="Path to C++ full inference executable")
    parser.add_argument("--csv", default="data/raw/FI2010_test.csv", help="CSV path for full C++ inference")
    parser.add_argument("--mu", default="data/processed/mu.npy", help="mu.npy path")
    parser.add_argument("--sigma", default="data/processed/sigma.npy", help="sigma.npy path")
    parser.add_argument("--py-warmup", type=int, default=100)
    parser.add_argument("--py-runs", type=int, default=1000)
    parser.add_argument("--cpp-warmup", type=int, default=100)
    parser.add_argument("--cpp-runs", type=int, default=1000)
    parser.add_argument("--full-batch-size", type=int, default=1)
    parser.add_argument("--full-max-windows", type=int, default=-1)
    parser.add_argument("--full-warmup", type=int, default=20)
    parser.add_argument("--out", default="logs/phase5_report.md", help="Output markdown report path")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)

    cpp_benchmark = Path(args.cpp_benchmark)
    cpp_full = Path(args.cpp_full)
    if not cpp_benchmark.exists():
        raise FileNotFoundError(f"C++ benchmark executable not found: {cpp_benchmark}")
    if not cpp_full.exists():
        raise FileNotFoundError(f"C++ full inference executable not found: {cpp_full}")

    py_cmd = [
        sys.executable,
        "src/benchmark_python.py",
        "--model",
        model_path,
        "--warmup",
        str(args.py_warmup),
        "--runs",
        str(args.py_runs),
    ]
    py_code, py_output = run_command(py_cmd)
    if py_code != 0:
        raise RuntimeError(f"Python benchmark failed:\n{py_output}")
    py_metrics = parse_latency_metrics(py_output)

    cpp_cmd = [
        str(cpp_benchmark),
        model_path,
        str(args.cpp_warmup),
        str(args.cpp_runs),
    ]
    cpp_code, cpp_output = run_command(cpp_cmd)
    if cpp_code != 0:
        raise RuntimeError(f"C++ benchmark failed:\n{cpp_output}")
    cpp_metrics = parse_latency_metrics(cpp_output)

    full_cmd = [
        str(cpp_full),
        model_path,
        args.csv,
        args.mu,
        args.sigma,
        str(args.full_batch_size),
        str(args.full_max_windows),
        str(args.full_warmup),
    ]
    full_code, full_output = run_command(full_cmd)
    if full_code != 0:
        raise RuntimeError(f"C++ full inference failed:\n{full_output}")
    full_metrics = parse_full_inference_metrics(full_output)

    speedup = py_metrics["mean_ms"] / cpp_metrics["mean_ms"] if cpp_metrics["mean_ms"] > 0 else float("inf")
    cpp_target = "PASS" if cpp_metrics["mean_ms"] < 1.0 else "FAIL"

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""# Phase 5 Benchmark Report

Generated: {now}

Model: `{model_path}`

## Latency Comparison

| Environment | Mean (ms) | Std (ms) | P99 (ms) |
|---|---:|---:|---:|
| Python (ONNX Runtime) | {py_metrics['mean_ms']:.4f} | {py_metrics['std_ms']:.4f} | {py_metrics['p99_ms']:.4f} |
| C++ (ONNX Runtime) | {cpp_metrics['mean_ms']:.4f} | {cpp_metrics['std_ms']:.4f} | {cpp_metrics['p99_ms']:.4f} |

Speedup (Python mean / C++ mean): **{speedup:.2f}x**

C++ target check (<1 ms mean): **{cpp_target}**

## Full CSV Inference (C++)

| Metric | Value |
|---|---:|
| Total inference time (ms) | {full_metrics['total_ms']:.4f} |
| Mean per window (ms) | {full_metrics['mean_window_ms']:.4f} |
| Throughput (windows/s) | {full_metrics['throughput']:.4f} |
| Accuracy (%) | {full_metrics['accuracy_pct']:.4f} |

## Commands Used

```text
{' '.join(py_cmd)}
{' '.join(cpp_cmd)}
{' '.join(full_cmd)}
```
"""

    out_path = Path(args.out)
    write_text(out_path, report)

    raw_dir = out_path.parent
    write_text(raw_dir / "phase5_python_output.txt", py_output + "\n")
    write_text(raw_dir / "phase5_cpp_benchmark_output.txt", cpp_output + "\n")
    write_text(raw_dir / "phase5_cpp_full_output.txt", full_output + "\n")

    print(f"Report written: {out_path}")
    print(f"Python mean latency (ms): {py_metrics['mean_ms']:.4f}")
    print(f"C++ mean latency (ms): {cpp_metrics['mean_ms']:.4f}")
    print(f"Speedup: {speedup:.2f}x")
    print(f"C++ target (<1 ms mean): {cpp_target}")


if __name__ == "__main__":
    main()
