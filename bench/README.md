# Phase 2 benchmark harness

Process-isolated, randomized benchmarks of the Rust engine against a version-matched
Python/ONNX Runtime baseline, with numerical-equivalence checks and paper figures.
A C++ engine plugs into the same pipeline (see "Adding the C++ engine").

## Quick start

```sh
# one-time: isolated Python environments (analysis + two ORT versions)
cd bench
uv venv --python 3.12 .venv          && uv pip install --python .venv/bin/python numpy matplotlib scipy onnxruntime onnx
uv venv --python 3.12 .venv-ort131   && uv pip install --python .venv-ort131/bin/python numpy onnxruntime==1.31.0
cd ..

# one-time: raw-scale FI-2010 test CSV for the Rust pipeline (see "Dataset")
bench/.venv/bin/python bench/prepare_dataset.py

# full suite (~15 min); refuses to start in macOS Low Power Mode
bench/.venv/bin/python -I bench/run_suite.py --reps 10
bench/.venv/bin/python -I bench/analyze.py results/phase2/<run_id>
```

`--quick` runs 2 repetitions × 50 calls to check the plumbing (allowed in Low Power Mode).

## What runs

| Engine key | What it is |
|---|---|
| `rust` | `lob-inference`, bundled static ONNX Runtime 1.31.0 (`ort` 2.0.0-rc.15), `--device-policy cpu` |
| `python` | PyPI `onnxruntime==1.31.0` — same ORT commit (`402979a`) as the Rust build |
| `rust-msdylib` | same Rust code built with `--features ort-dynamic`, loading Microsoft's `libonnxruntime.1.31.0.dylib` |
| `rust-coreml` | Rust with `--device-policy auto-efficiency` (the `ort` crate's implicit default; offloads to CoreML) |
| `python-ort132` | `onnxruntime==1.32.0`, numerics only |
| `cpp` | `bench/cpp/rustlob_cpp_bench`: adapter of `cpp-engine/cpp_inference` (ORT C++ API, CPU), Microsoft ONNX Runtime 1.31.0 C++ SDK |

Matrix (each cell = 10 fresh processes × (20 warm-up + 200 timed calls)):
model {DeepLOB, TransformerLOB} × intra-op threads {1, ORT default} × input policy
{`first` = one window reused, `cycle` = all 319 windows in order}; plus ablations for device policy,
ORT binary, and graph-optimization level; cold start (spawn → first prediction → exit);
Rust preprocessing and end-to-end; logits for every window from every engine.

### Isolation protocol (`run_suite.py`)
- one OS process per trial (no shared allocator, ORT arena, or thread pools between trials);
- all (configuration, repetition) pairs shuffled with a fixed seed and interleaved, so drift
  is spread across configurations; 1 s cool-down between processes; no resource sampler;
- `caffeinate` keeps the machine awake; environment snapshot (CPU, P/E cores, macOS, power,
  Low Power Mode, thermal, load average, top processes, git state, binary SHA-256, ORT build string)
  in `env.json`, plus load average before every run in `suite.json`;
- before measuring, ORT node placement is checked from trace logs; the suite aborts if a
  CPU-only engine placed any node on CoreML.

## Output layout — `results/phase2/<run_id>/`

```
env.json                 machine + software snapshot
suite.json               plan, randomized order, per-run wall time / return code / load, placement, numerics
inputs/windows.npy       [319,1,100,40] float32 — exact tensors the Rust engine feeds ORT
inputs/labels.npy        [319] uint8 (0=Up 1=Stationary 2=Down)
inputs/windows.json      manifest: source rows per window, normalization, hashes
raw/<engine>/*.json      one BenchmarkReport per process (schema: src/benchmark/report.rs)
numerics/                logits per engine/model/threads (Rust: predictions CSV, Python: .npy)
logs/                    stdout/stderr of every process, ORT placement traces
tables/*.csv             every number behind every figure
figures/*.png|pdf        paper figures
summary.md               tables + figures in one page
```

## Statistics (`analyze.py`)
- Unit of replication = one process. Engine comparisons use per-process medians:
  Mann–Whitney U (two-sided) and a bootstrap 95% CI of the ratio of medians (10,000 resamples).
- Pooled per-call P50/P90/P99/P99.9 use nearest rank, identical to `src/benchmark/statistics.rs`.
- "Host overhead" = per-call `predict` time minus `Session::run` time, both measured in the same call.

## Dataset
The repository has the FI-2010 test split only as z-scored arrays
(`data/normalization/FI2010_test_test_n50000_w100_z.npz`, 31,937 rows). `prepare_dataset.py`
inverts the training z-score (`raw = z·σ + μ`) and writes `bench/data/FI2010_test_reconstructed.csv`
(40 features + label). The Rust pipeline re-normalizes it; round-trip max error 4.8e-7
(float32 rounding), and the exported windows match the original z-scored windows within that bound.

## C++ engine (integrated)

`cpp-engine/` holds the original project (training code, Windows C++ sources, raw FI-2010 CSVs).
Two builds run on macOS:

- **Original binaries, unmodified** (`cpp-engine/cpp_inference/build-macos/{inference,benchmark,full_inference}`),
  built with the project's own `CMakeLists.txt`; used to reproduce the authors' Phase 5 numbers
  (`results/phase2-cpp/original/`).
- **Harness adapter** `bench/cpp/rustlob_cpp_bench.cpp` (same session setup as `benchmark.cpp`) that
  follows the contract below and runs inside `run_suite.py`'s randomized schedule as engine `cpp`.

ONNX Runtime C++ SDK: Microsoft's `Microsoft.ML.OnnxRuntime` **1.31.0** NuGet package (headers from
`build/native/include`, library from `runtimes/osx-arm64/native`), extracted to
`cpp-engine/cpp_inference/third_party/onnxruntime-osx-arm64-1.31.0/` (git-ignored there). The GitHub release
tarball was not reachable from this machine. Build:

```sh
R=$PWD/cpp-engine/cpp_inference/third_party/onnxruntime-osx-arm64-1.31.0
ln -sf libonnxruntime.dylib $R/lib/libonnxruntime.1.dylib
ONNXRUNTIME_ROOT=$R cmake -S bench/cpp -B bench/cpp/build -DCMAKE_BUILD_TYPE=Release && cmake --build bench/cpp/build -j
ONNXRUNTIME_ROOT=$R cmake -S cpp-engine/cpp_inference -B cpp-engine/cpp_inference/build-macos \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH=$R/lib && cmake --build cpp-engine/cpp_inference/build-macos -j
```

Data checks on the C++ project's raw data and artifacts: `bench/.venv-ort131/bin/python -I bench/data_checks.py`
→ `results/data_checks/report.md`.

## Contract for adding another engine

The C++ engine must consume the **same inputs** and emit the **same report schema**; then
`analyze.py` includes it automatically (it reads every `raw/<engine>/` directory; C++ is colored
as the third main engine).

**Inputs.** Read `results/phase2/<run_id>/inputs/windows.npy` (NPY v1, `<f4`, C order,
shape `[N,1,100,40]`). Window `i` is the contiguous `4000` floats at offset `i*4000`.

**Session setup to match the Rust/Python runs.**
- ONNX Runtime **1.31.0** (Microsoft release `onnxruntime-osx-arm64-1.31.0`), CPU execution
  provider only. Do not call `SetEpSelectionPolicy` / auto device selection; verify with
  verbose logging that no node lands on CoreML (`Node(s) placed on [...]`).
- Intra-op threads `1` and "ORT default" (do not set) — run both.
- Default graph optimization (do not set) unless reproducing the ablation.

**Timing protocol per process.** Load model → 20 warm-up calls → 200 timed calls with input
policy `first` (window 0 every call) or `cycle` (window `call % N`). For each timed call record:
- `samples_ms`: wall time of the full predict path (create input tensor view over the window,
  `Session::Run`, read 3 outputs, argmax) — `std::chrono::steady_clock`;
- `session_run_samples_ms`: wall time of `Session::Run` alone, inside the same call.

**Report JSON** → `results/phase2/<run_id>/raw/cpp/<anything>.json`, one file per process:

```json
{
  "metadata": {
    "model": "DeepLOB",                       // or "TransformerLOB"
    "mode": "inference",
    "model_file": "models/deeplob.onnx",
    "model_file_hash": null,                  // optional; FNV-1a 64 of the .onnx if you have it
    "input": "inputs/windows.npy",
    "input_shape": "[1, 1, 100, 40]", "input_type": "float32", "execution_provider": "cpu",
    "runtime_version": "onnxruntime C++ 1.31.0",
    "runtime_build_info": "<Ort::GetBuildInfoString()>",
    "session_options": "intra_op_threads=1; inter_op_threads=ort-default; graph_optimization=ort-default; intra_op_spinning=ort-default; device_policy=cpu",
    "input_policy": "cycle",                  // or "first"
    "windows_used": 319,
    "warmup_iterations": 20, "measured_iterations": 200, "trial_count": 1,
    "runtime_initialization_ms": 0.0, "initial_model_load_ms": 0.0,
    "rust_version": null, "build_mode": "release (-O3)", "operating_system": "macos aarch64",
    "cpu": "Apple M3", "ram_bytes": null, "thread_configuration": "<same as session_options>",
    "git_commit": null, "timestamp_unix_seconds": 0, "memory": null,
    "cpu_utilization_percent": null, "cpu_peak_percent": null, "load_average": null
  },
  "trials": [{
    "elapsed_seconds": 0.0, "completed_windows": 200, "throughput_per_second": null,
    "samples_ms": [...200 floats...], "session_run_samples_ms": [...200 floats...],
    "statistics": null, "session_run_statistics": null, "preprocessing_stages": null
  }],
  "aggregate": null
}
```

Use `intra_op_threads=ort-default` in `session_options` when the thread count is not set.
`statistics`/`aggregate` may be null — `analyze.py` recomputes everything from raw samples.

**Numerics.** Run every window once (1 thread and default threads) and save logits as
`results/phase2/<run_id>/numerics/cpp__deeplob__t1.npy` (`[N,3]` float32; also `__tdefault`).
They are compared against Python ORT 1.31 (1 thread) and should be bit-identical at 1 thread.

New engines should be added to `run_suite.py` so they run inside the same randomized, interleaved
schedule rather than as a separate session.
