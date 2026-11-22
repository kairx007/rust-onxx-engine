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

The CSV must contain 40 selected features and the configured label column. Labels are currently required by the loader and must use supported FI-2010 values (1, 2, or 3). Provide the model's training `mu.npy` and `sigma.npy` arrays, each containing 40 values; the application does not calculate normalization statistics. Window *i* covers rows *i*…*i*+99 and is labelled with row *i*+100, the tick after it — the convention the models were trained with and the C++ engine uses. `preprocessing.stride` (default 1) sets the step between windows. The default column mapping matches the FI-2010 CSVs used for training: a header line, a row-index column 0, features in columns 1–40, and the label in column 145.

Run inference:

```sh
cargo run --release -- --config configs/local.toml predict \
  --model deeplob \
  --input data/test.csv \
  --output outputs/deeplob-predictions.csv
```

For ordered data segments, use `--segment path/to/first.csv --segment path/to/next.csv` instead of `--input`. Prediction output contains one record per complete window with the true class, predicted class/direction, and three scores (`class0_up_score`, `class1_stationary_score`, `class2_down_score`; FI-2010 labels 1/2/3 = Up/Stationary/Down map to classes 0/1/2). Existing output is preserved unless `--overwrite` is supplied.

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

### Runtime options

Global flags (or the matching `[runtime]` keys in the TOML config):

| Flag | Config key | Meaning |
|---|---|---|
| `--intra-threads N` | `intra_op_threads` | ONNX Runtime intra-op threads (omit for the ORT default) |
| `--inter-threads N` | `inter_op_threads` | ONNX Runtime inter-op threads |
| `--graph-opt LEVEL` | `graph_optimization` | `disable`, `level1`, `level2`, `level3`, `all` |
| `--intra-spinning BOOL` | `intra_op_spinning` | allow intra-op threads to spin |
| `--device-policy POLICY` | `device_policy` | `cpu` (default) or `auto-efficiency` |

`device_policy = "cpu"` is the default because the `ort` crate otherwise sets ONNX Runtime's
`MaxEfficiency` automatic device policy, which on Apple silicon places parts of both models on the
CoreML execution provider (reduced precision, slower for these models) even though only the CPU
provider is registered. `runtime-info` prints the linked ONNX Runtime build string and the
effective session options; benchmark reports record both.

`benchmark --input-policy cycle` feeds every prepared window in turn instead of reusing the first
one. Inference-mode reports carry per-call `session_run_samples_ms` (time inside `Session::run`)
next to the full `predict` latency.

`export-windows --output-dir DIR` writes the exact preprocessed tensors (`windows.npy`,
`labels.npy`, `windows.json`) so other engines can be fed identical inputs.

Build with `--no-default-features --features ort-dynamic` to load ONNX Runtime at run time
from `ORT_DYLIB_PATH` instead of linking the bundled binary.

The process-isolated benchmark suite, Python baseline, analysis, and C++ integration contract
are in [`bench/README.md`](bench/README.md).

## Results: Rust vs C++

A process-isolated benchmark (499 fresh processes, randomized interleaved order, Apple M3) compared this engine with
the C++ engine in `cpp-engine/` on identical models, inputs and ONNX Runtime 1.31.0:

| Model | Threads | C++ | Rust + Microsoft ORT | Rust (bundled ORT) |
|---|---|---|---|---|
| DeepLOB | 1 | 0.674 ms | 0.667 ms | 0.691 ms |
| DeepLOB | default | 0.275 ms | 0.275 ms | 0.318 ms |
| TransformerLOB | 1 | 0.897 ms | 0.890 ms | 0.914 ms |
| TransformerLOB | default | 0.336 ms | 0.338 ms | 0.520 ms |

- With the same ONNX Runtime library, Rust is within 1.1% of C++; outputs are bit-identical at one thread.
- The gap with the default build comes from the ONNX Runtime binary the `ort` crate bundles, not from Rust; build
  with `--no-default-features --features ort-dynamic` and set `ORT_DYLIB_PATH` to use another build.
- Rust's own per-call overhead is about 2 µs; cold start is 37–51 ms (C++ 42–55 ms).

Full results: [`research/08_final_results.md`](research/08_final_results.md); figures and tables:
[`results/phase2/20261122-final-rust-vs-cpp/`](results/phase2/20261122-final-rust-vs-cpp/summary.md); paper draft:
[`research/09_paper_draft.md`](research/09_paper_draft.md); data audit:
[`results/data_checks/report.md`](results/data_checks/report.md).

## Documentation

- [Model setup and execution guide](docs/running-a-model.md)
- [Benchmark harness, methodology and C++ integration](bench/README.md)
- [Research notes and final results](research/README.md)

## Current limitations

The repository includes `models/deeplob.onnx`, `models/transformerlob.onnx` (with its external-weights file `models/transformer.onnx.data`), and the FI-2010 training normalization arrays. The raw FI-2010 CSVs are not included; `bench/prepare_dataset.py` reconstructs a raw-scale test CSV from the committed z-scored test array. GPU inference, batching, and automatic input-layout conversion are not implemented. Numerical equivalence is checked against Python ONNX Runtime by the benchmark suite; equivalence against the C++ engine is pending.
