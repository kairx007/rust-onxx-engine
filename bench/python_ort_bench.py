"""Python / onnxruntime baseline that mirrors the Rust `benchmark --mode inference` protocol.

Reads the exact tensors exported by `lob-inference export-windows`, so inputs are
bit-identical to what the Rust engine feeds ONNX Runtime. Writes a JSON report in
the Rust `BenchmarkReport` schema (src/benchmark/report.rs) so the same analysis
code handles Rust, Python, and (later) C++ reports.

Timed boundaries, per call (same as Rust):
  samples_ms              : full predict path = finite check + Session.run + output shape check + argmax
  session_run_samples_ms  : Session.run only

Modes:
  bench      : warm-up + timed calls in this process (one trial; isolation = one process per trial)
  logits     : run every window once and save logits to .npy (numerical reference)
  cold-start : load model + first prediction; prints JSON timings (process-level wall time is
               measured by the orchestrator)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import_start = time.perf_counter()
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

import_seconds = time.perf_counter() - import_start


def fnv1a64(path: Path) -> str:
    h = 0xCBF29CE484222325
    data = path.read_bytes()
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"fnv1a64:{h:016x}"


def stats(samples: list[float]) -> dict | None:
    """Nearest-rank percentiles and population std, identical to src/benchmark/statistics.rs."""
    if not samples or any((not math.isfinite(s)) or s < 0 for s in samples):
        return None
    ordered = sorted(samples)
    n = len(ordered)
    mean = sum(ordered) / n
    var = sum((s - mean) ** 2 for s in ordered) / n

    def pct(p: float) -> float:
        rank = max(1, math.ceil(p * n))
        return ordered[min(rank, n) - 1]

    return {
        "sample_count": n,
        "mean_ms": mean,
        "standard_deviation_ms": math.sqrt(var),
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
        "p50_ms": pct(0.50),
        "p90_ms": pct(0.90),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "p999_ms": pct(0.999),
    }


def make_session(model: Path, intra: int | None, graph_opt: str | None) -> tuple[ort.InferenceSession, str]:
    options = ort.SessionOptions()
    if intra is not None:
        options.intra_op_num_threads = intra
    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "level1": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "level2": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    if graph_opt is not None:
        options.graph_optimization_level = levels[graph_opt]
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    described = (
        f"intra_op_threads={intra if intra is not None else 'ort-default'}; inter_op_threads=ort-default; "
        f"graph_optimization={graph_opt or 'ort-default'}; intra_op_spinning=ort-default"
    )
    return session, described


def predict(session: ort.InferenceSession, input_name: str, window: np.ndarray) -> tuple[int, np.ndarray, float]:
    if not np.isfinite(window).all():
        raise ValueError("non-finite input")
    run_start = time.perf_counter()
    (scores,) = session.run(None, {input_name: window})
    run_ms = (time.perf_counter() - run_start) * 1000.0
    if scores.shape[-1] != 3 or scores.size != 3 or not np.isfinite(scores).all():
        raise ValueError(f"unexpected output {scores.shape}")
    flat = scores.reshape(3)
    return int(np.argmax(flat)), flat, run_ms


def command(*args: str) -> str | None:
    try:
        out = subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
        return out or None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bench", "logits", "cold-start"], default="bench")
    parser.add_argument("--model", choices=["deeplob", "transformerlob"], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True, help="windows.npy from export-windows")
    parser.add_argument("--intra-threads", type=int)
    parser.add_argument("--graph-opt", choices=["disable", "level1", "level2", "all"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--input-policy", choices=["first", "cycle"], default="first")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    windows = np.load(args.windows)
    assert windows.dtype == np.float32 and windows.ndim == 4 and windows.shape[1:] == (1, 100, 40)

    load_start = time.perf_counter()
    session, session_options = make_session(args.model_path, args.intra_threads, args.graph_opt)
    load_ms = (time.perf_counter() - load_start) * 1000.0
    input_name = session.get_inputs()[0].name

    if args.mode == "cold-start":
        first = np.ascontiguousarray(windows[0:1])
        t0 = time.perf_counter()
        predict(session, input_name, first)
        first_ms = (time.perf_counter() - t0) * 1000.0
        print(json.dumps({"import_ms": import_seconds * 1000, "model_load_ms": load_ms, "first_inference_ms": first_ms}))
        return

    if args.mode == "logits":
        out = np.empty((windows.shape[0], 3), dtype=np.float32)
        for i in range(windows.shape[0]):
            _, scores, _ = predict(session, input_name, np.ascontiguousarray(windows[i : i + 1]))
            out[i] = scores
        np.save(args.output, out)
        print(f"saved logits {out.shape} -> {args.output}")
        return

    inputs = [np.ascontiguousarray(windows[0:1])] if args.input_policy == "first" else [
        np.ascontiguousarray(windows[i : i + 1]) for i in range(windows.shape[0])
    ]
    for call in range(args.warmup):
        predict(session, input_name, inputs[call % len(inputs)])
    samples, run_samples = [], []
    trial_start = time.perf_counter()
    for call in range(args.iterations):
        x = inputs[call % len(inputs)]
        t0 = time.perf_counter()
        _, _, run_ms = predict(session, input_name, x)
        samples.append((time.perf_counter() - t0) * 1000.0)
        run_samples.append(run_ms)
    elapsed = time.perf_counter() - trial_start

    report = {
        "metadata": {
            "model": "DeepLOB" if args.model == "deeplob" else "TransformerLOB",
            "model_file": str(args.model_path),
            "model_file_hash": fnv1a64(args.model_path),
            "mode": "inference",
            "input": str(args.windows),
            "input_shape": "[1, 1, 100, 40]",
            "input_type": "float32",
            "execution_provider": "cpu",
            "runtime_initialization_ms": import_seconds * 1000.0,
            "initial_model_load_ms": load_ms,
            "warmup_iterations": args.warmup,
            "measured_iterations": args.iterations,
            "trial_count": 1,
            "runtime_version": f"onnxruntime-python {ort.__version__}",
            "rust_version": None,
            "build_mode": f"python {platform.python_version()}",
            "operating_system": f"{sys.platform} {platform.machine()}",
            "cpu": command("sysctl", "-n", "machdep.cpu.brand_string"),
            "ram_bytes": int(command("sysctl", "-n", "hw.memsize") or 0) or None,
            "thread_configuration": session_options,
            "git_commit": command("git", "rev-parse", "--short", "HEAD"),
            "timestamp_unix_seconds": int(time.time()),
            "memory": None,
            "cpu_utilization_percent": None,
            "cpu_peak_percent": None,
            "session_options": session_options,
            "runtime_build_info": f"onnxruntime {ort.__version__} (PyPI wheel); providers={ort.get_available_providers()}",
            "input_policy": args.input_policy,
            "windows_used": len(inputs),
            "load_average": command("sysctl", "-n", "vm.loadavg"),
        },
        "trials": [
            {
                "elapsed_seconds": elapsed,
                "completed_windows": args.iterations,
                "throughput_per_second": args.iterations / elapsed if elapsed > 0 else None,
                "samples_ms": samples,
                "statistics": stats(samples),
                "preprocessing_stages": None,
                "session_run_samples_ms": run_samples,
                "session_run_statistics": stats(run_samples),
            }
        ],
        "aggregate": stats(samples),
    }
    if args.output is None:
        json.dump(report, sys.stdout, indent=1)
    else:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
