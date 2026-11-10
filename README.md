# RustLOB

RustLOB is a Rust implementation of the ONNX Runtime foundation and preprocessing pipeline for DeepLOB and TransformerLOB.

Phase 1 provides CPU runtime initialization, reusable model sessions, metadata inspection, and CLI diagnostics. Phase 2 loads segmented FI-2010-style CSV data, maps features and labels, applies supplied training normalization statistics, extracts non-overlapping windows, and constructs contiguous inference tensors. Phases 3 and 4 provide DeepLOB and TransformerLOB inference engines; Phase 2 itself does not run a model.

## Requirements

- Stable Rust 1.92 or later (required by `ort` 2.0.0-rc.15)
- ONNX model files supplied separately in `models/` for Phase 1 model inspection
- Xcode Command Line Tools on macOS; the bundled ONNX Runtime archive requires a macOS 13.4 deployment target

`ort` is pinned to `2.0.0-rc.15`, which wraps ONNX Runtime 1.31. The configured `download-binaries` feature provides the runtime without a separately installed system package. NPY normalization arrays are read with `npyz`; CSV data is read with `csv`. See the [ort crate docs](https://docs.rs/ort/2.0.0-rc.15/ort/), [npyz docs](https://docs.rs/npyz/0.9.1/npyz/), and [csv docs](https://docs.rs/csv/1.4.0/csv/).

## Phase 1 commands

```sh
cargo run -- --help
cargo run -- runtime-info
cargo run -- inspect --model deeplob
cargo run -- validate --model transformerlob
```

## Phase 2 preprocessing

The pipeline API is available from `PreprocessingPipeline`; the example reports prepared rows, windows, and tensor shape without running inference:

```sh
cargo run --example preprocessing_demo
```

Before using real data, inspect its format and update `configs/default.toml`. The current mapping (`feature_columns = 0..39`, `label_column = 144`, no header) is a candidate copied from the Phase 2 specification, not a verified property of any supplied dataset. The default `schema_verified = false` makes the pipeline refuse to start until the mapping has been checked. Set it to `true` only after confirming the actual file layout.

The expected normalization inputs are one-dimensional, C-order NPY vectors of length 40 at `data/normalization/mu.npy` and `sigma.npy`. The application only loads these values; it never computes or substitutes statistics. These files are not present in the repository.

A single CSV file can be given as `dataset_path`. For several instrument/day files, list them in `segment_paths` in chronological order. Each file remains a separate segment, so windows cannot cross file boundaries. If `dataset_path` names a directory containing multiple CSVs, the loader requires an explicit `segment_paths` list instead of guessing chronological order. Blank physical rows, malformed records, inconsistent field counts, invalid selected values, and invalid labels are rejected.

The pipeline retains normalized segment data and compact window references. It allocates a contiguous 4,000-value `f32` buffer only when a window is requested, with shape `[1, 1, 100, 40]`. `InputTensor::into_ort_tensor` transfers that owned buffer into an ONNX Runtime tensor; it does not execute a model.

## Phase 3 DeepLOB inference

`DeepLobEngine::load` loads one DeepLOB ONNX session and validates its metadata against the configured input and three-class output contract. `DeepLobEngine::predict` accepts a Phase 2 `InputTensor`, submits a borrowed tensor view under the input name discovered from the loaded ONNX graph, then validates and classifies its three output scores. The retained session is reused across calls. Prediction requires mutable access to the engine because ONNX Runtime's session run API takes `&mut self`.

No DeepLOB ONNX model is currently included under `models/`, so the actual graph's names, output shape, and output value semantics remain unverified. Synthetic tests cover metadata contracts, tensor adaptation, output validation, and classification, but do not establish that the real DeepLOB model loads or runs. Real-model integration testing remains required when the model asset is supplied.

## Phase 4 TransformerLOB inference

`TransformerLobEngine` reuses the common session runner and `Prediction` mapping from Phase 3. It reads tensor names and shapes from the loaded graph and currently accepts the Phase 2 `[1, 1, 100, 40]` float32 input without reshaping or reordering. It rejects any statically incompatible shape rather than guessing a layout transform. The engine retains one CPU session for repeated calls.

No TransformerLOB ONNX model is included. Its actual metadata and any model-specific input transform remain unverified, and successful TransformerLOB inference has not been demonstrated. Supply and inspect the model before treating this path as model-compatible.

## Phase 2 limitations

No FI-2010 CSV files, `mu.npy`, `sigma.npy`, or ONNX models are included. Synthetic fixtures verify the implementation path, but cannot verify the real dataset's headers, label location, segment ordering, separate-label-file format, training partition provenance, or model metadata. No real-data compatibility or performance claim is made.

See [`context/context.md`](context/context.md), [`context/phase-01.md`](context/phase-01.md), and [`context/phase-02.md`](context/phase-02.md) for the project and phase specifications.

## Phase 5 command-line workflows

Build the `lob-inference` executable with `cargo build --release`. The existing diagnostics remain available:

```sh
lob-inference --help
lob-inference runtime-info
lob-inference inspect --model deeplob
lob-inference validate --model transformerlob
```

Use `--config PATH` to select a TOML configuration and `--log-level error|warn|info|debug|trace` to override its logging level. The existing flat model path keys are retained. `data.output_path` defaults to `outputs/predictions.csv`; `--output` overrides it. `--model-path` can override the selected model path for a single command.

Run a smoke prediction with synthetic input, or request one prepared from a dataset:

```sh
lob-inference smoke-test --model deeplob
lob-inference smoke-test --model deeplob --input data/sample.csv
```

Run CSV inference and write window-level labels, predictions, and scores:

```sh
lob-inference predict --model deeplob --input data/test.csv --output outputs/deeplob.csv
lob-inference predict --model transformerlob --segment data/day-1.csv --segment data/day-2.csv --overwrite
```

The prediction CSV schema is `window_index,source_file,segment_index,start_row,true_class,predicted_class,predicted_direction,down_score,stationary_score,up_score`. Classes use the existing internal mapping 0=Down, 1=Stationary, 2=Up. Output is written to a temporary file and published only after all windows succeed. Existing output is preserved unless `--overwrite` is supplied.

Run the basic timing command on prepared input data:

```sh
lob-inference benchmark --model deeplob --input data/sample.csv --iterations 100
```

It loads the model once, prepares one valid tensor, then repeats inference on that tensor. It reports model-load time separately from inference time; it is a basic command-level measurement, not the Phase 6 benchmark suite.

The repository currently has no ONNX models, FI-2010 CSVs, or normalization arrays. Real model validation, smoke inference, CSV predictions, and benchmark execution require those assets. Dataset workflows also require verifying the configured columns and setting `data.schema_verified = true` after inspection. Synthetic smoke input does not establish model compatibility or data accuracy.
