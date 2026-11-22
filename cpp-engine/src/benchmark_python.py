"""
Benchmark ONNX inference latency in Python for direct comparison with C++ benchmark.

Usage:
    python src/benchmark_python.py --warmup 100 --runs 1000
"""

import argparse
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort


def percentile_99(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, 99))


def resolve_model_path(model_arg: str | None) -> str:
    if model_arg:
        return model_arg

    if Path("transformer.onnx").exists():
        return "transformer.onnx"
    if Path("deeplob.onnx").exists():
        return "deeplob.onnx"

    return "deeplob.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description="Python ONNX Runtime latency benchmark")
    parser.add_argument("--model", default=None, help="Path to ONNX model (auto-detected if omitted)")
    parser.add_argument("--warmup", type=int, default=100, help="Number of warm-up inferences")
    parser.add_argument("--runs", type=int, default=1000, help="Number of timed inferences")
    args = parser.parse_args()

    if args.runs <= 0:
        raise ValueError("--runs must be > 0")

    model_path = resolve_model_path(args.model)
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    x = np.zeros((1, 1, 100, 40), dtype=np.float32)
    x[0, 0, :, :] = np.linspace(0.0, 1.0, 4000, dtype=np.float32).reshape(100, 40)

    for _ in range(max(0, args.warmup)):
        sess.run([output_name], {input_name: x})

    latencies_ms = np.empty(args.runs, dtype=np.float64)
    for i in range(args.runs):
        t0 = time.perf_counter_ns()
        out = sess.run([output_name], {input_name: x})
        t1 = time.perf_counter_ns()
        latencies_ms[i] = (t1 - t0) / 1_000_000.0

    logits = out[0][0]
    pred = int(np.argmax(logits))

    mean = float(latencies_ms.mean())
    std = float(latencies_ms.std())
    p99 = percentile_99(latencies_ms)

    print(f"Model path       : {model_path}")
    print(f"Warmup runs      : {args.warmup}")
    print(f"Timed runs       : {args.runs}")
    print(f"Mean latency (ms): {mean:.4f}")
    print(f"Std latency (ms) : {std:.4f}")
    print(f"P99 latency (ms) : {p99:.4f}")
    print(f"Predicted class  : {pred} (0=down, 1=stationary, 2=up)")
    print(f"Target check     : {'PASS (<1 ms mean)' if mean < 1.0 else 'FAIL (>=1 ms mean)'}")


if __name__ == "__main__":
    main()
