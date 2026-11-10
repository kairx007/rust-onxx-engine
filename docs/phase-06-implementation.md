# Phase 6: Benchmarking and Performance Profiling

## What is available

`lob-inference benchmark` now supports:

```sh
cargo run --release -- benchmark --model deeplob --warmup 20 --iterations 200 --trials 5
cargo run --release -- benchmark --model transformerlob --mode inference --output results/deeplob.json
cargo run --release -- benchmark --model deeplob --mode preprocessing --profile-memory --profile-cpu --output results/preprocessing.csv
cargo run --release -- benchmark --model deeplob --mode end-to-end --output results/end-to-end.json
cargo run --release -- benchmark --model deeplob --mode load --trials 5 --output results/load.json
```

`--format json|csv` selects the report encoding. Without it, `.csv` selects CSV and other output extensions select JSON. Output files are created exclusively; existing reports are never replaced. Parent directories are created as needed.

Inference mode prepares a dataset tensor before timing, warms the loaded session, then times each prediction call using `Instant`. One valid tensor is reused to isolate model execution. CSV parsing, normalization, tensor creation, model loading, output serialization, and warm-up calls are excluded from those individual latency samples. Throughput is completed inference calls divided by measured loop time, aggregated across trials.

Preprocessing mode times CSV loading/feature extraction, normalization, window generation, and tensor construction separately, as well as the total preprocessing workload. End-to-end mode times preparation, each prediction, and serialization of prediction rows to an in-memory CSV buffer. It does not include physical disk writes. Load mode reports repeated session loading separately from runtime initialization.

Latency summaries include mean, population standard deviation, minimum, maximum, and P50/P90/P95/P99. Percentiles use nearest rank (`ceil(p * N)`, one-based). JSON retains every sample and each trial summary; CSV has one row per raw sample with trial-level summaries. Defaults are 20 warm-up calls, 200 measured calls, and 5 trials.

Optional process sampling runs in a parallel collector at 100 ms intervals using `ps`, reporting RSS and the available CPU percentage field. It reports missing values explicitly when sampling fails. Sampling invokes a subprocess and adds load to the machine, so resource-profiled runs should not be used as the cleanest inference-latency baseline. Results identify this method. CPU field semantics vary by OS; do not compare them across platforms without checking the platform's `ps` behavior.

Reports include model path and an FNV-1a 64-bit file fingerprint, input description/shape/type, provider, counts, timestamp, Rust and OS information, CPU/RAM where available, Git revision, build mode, thread configuration, raw samples, and summaries. Runtime/thread metadata reflects the project's documented ONNX Runtime 1.31 binary and default thread configuration; values that cannot be retrieved from the current API are called out instead of implied to be per-run measurements.

## Rust/C++ comparison

```sh
cargo run --release -- compare --model deeplob --rust-results rust.json --cpp-results cpp.json
```

The comparison reads JSON benchmark reports, requires inference mode, matching model name, provider, input shape/type, model fingerprint, and runtime-version metadata, then prints mean-latency improvement and throughput ratio. An optional `--output comparison.json` writes the comparison without overwriting an existing file. C++ measurement reports must be converted to this JSON schema with real raw samples and environment metadata. The Rust repository has no C++ source, so it cannot run the reference engine or validate C++ report provenance. Comparison numbers remain conditional on matching runtime version, tensor values, preprocessing, threading, build mode, and hardware.

## Limitations

- No ONNX models, normalization statistics, or real CSV inputs are checked in, so real DeepLOB/TransformerLOB measurements cannot be produced from this repository state.
- CSV loading and feature extraction share the current loader boundary and are reported together.
- End-to-end output serialization is in-memory rather than a physical prediction-file write.
- Runtime thread controls are not exposed by the current wrapper.
- Resource measurements are best-effort OS process samples and can perturb the workload.
- This phase does not establish numerical equivalence or claim a Rust performance advantage.
