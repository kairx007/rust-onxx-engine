# RustLOB Inference

RustLOB runs DeepLOB and TransformerLOB ONNX models on CPU. It provides model inspection, CSV preprocessing, predictions, and repeatable inference benchmarks.

## Requirements

- Rust stable 1.92 or later
- An ONNX model for DeepLOB or TransformerLOB
- For real-data predictions: a compatible CSV dataset and the model's training normalization arrays

The project uses ONNX Runtime 1.31 through `ort` 2.0.0-rc.15. The runtime is downloaded by the Cargo build; no separate ONNX Runtime installation is required. On macOS, Xcode Command Line Tools are required.

## Build

```sh
cargo build --release
```

Commands can be run as `cargo run --release -- <command>` or through the built `lob-inference` executable.

## Run a model

The default model paths are:

```text
models/deeplob.onnx
models/transformerlob.onnx
```

Put the ONNX file at its matching path, or pass `--model-path /path/to/model.onnx` to a model command. The current input contract is float32 `[1, 1, 100, 40]` with three output classes; TransformerLOB must accept this same shape.

Check the runtime and model before using a dataset:

```sh
cargo run --release -- runtime-info
cargo run --release -- inspect --model deeplob
cargo run --release -- validate --model deeplob
cargo run --release -- smoke-test --model deeplob
```

Use `transformerlob` in place of `deeplob` for TransformerLOB. The synthetic smoke test checks model loading and one inference call; it does not check real-data preprocessing or prediction accuracy.

### Real-data predictions

Copy `configs/default.toml` to a local config, then set the model path, CSV path, verified feature and label columns, header setting, and normalization paths. The default column mapping is only a candidate: verify it against your CSV before setting `data.schema_verified = true`.

The CSV must contain 40 selected features and the configured label column. Labels are currently required by the loader and must use supported FI-2010 values (1, 2, or 3). Provide the model's training `mu.npy` and `sigma.npy` arrays, each containing 40 values; the application does not calculate normalization statistics. The pipeline creates non-overlapping 100-row windows.

Run inference:

```sh
cargo run --release -- --config configs/local.toml predict \
  --model deeplob \
  --input data/test.csv \
  --output outputs/deeplob-predictions.csv
```

For ordered data segments, use `--segment path/to/first.csv --segment path/to/next.csv` instead of `--input`. Prediction output contains one record per complete window with the true class, predicted class/direction, and three scores. Existing output is preserved unless `--overwrite` is supplied.

### Benchmark inference

```sh
cargo run --release -- --config configs/local.toml benchmark \
  --model deeplob \
  --warmup 20 \
  --iterations 200 \
  --trials 5 \
  --output results/deeplob.json
```

Inference latency excludes model loading and preprocessing. Reports include raw samples, mean, standard deviation, P50/P90/P95/P99, throughput, and run metadata. Use `--mode preprocessing`, `--mode end-to-end`, or `--mode load` for other measurements. Add `--profile-memory` or `--profile-cpu` for best-effort resource sampling; sampling adds system overhead. Reports are JSON by default; use a `.csv` output path or `--format csv` for CSV. Existing reports are never overwritten.

Rust/C++ report comparison:

```sh
cargo run --release -- compare \
  --model deeplob \
  --rust-results results/deeplob.json \
  --cpp-results results/deeplob-cpp.json
```

The C++ engine is not included. Run it separately and compare only reports collected with matching models, inputs, preprocessing, runtime, provider, and thread settings.

## Documentation

- [Model setup and execution guide](docs/running-a-model.md)
- [Benchmark methodology and report details](docs/phase-06-implementation.md)

## Current limitations

No ONNX models, normalization arrays, or real datasets are included. Real-data inference and performance measurements require those assets. GPU inference, automatic input-layout conversion, and numerical-equivalence validation are not implemented.
