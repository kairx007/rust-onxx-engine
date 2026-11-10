# Running a DeepLOB or TransformerLOB Model

This guide takes you from an ONNX model file to CSV predictions and benchmark reports. Follow the steps in order: confirming that the model loads is separate from confirming that your dataset and preprocessing match the model.

## 1. Confirm the model format and contract

The application loads **ONNX files**. It does not load training checkpoints such as `.pt`, `.pth`, `.h5`, or `.keras` directly. Export those to ONNX using the model's supported export process before continuing.

The current Rust inference contract is:

| Property | Expected value |
|---|---|
| Provider | CPU |
| Input type | `float32` |
| Input tensor shape | `[1, 1, 100, 40]` |
| Sequence length | 100 rows |
| Features per row | 40 |
| Output | 3 class scores |

The contract is enforced from the loaded ONNX graph metadata. The code does not guess a layout conversion. TransformerLOB currently also has to accept `[1, 1, 100, 40]`; an ONNX model that expects a different tensor layout will fail validation until an explicit adapter is implemented.

## 2. Place the ONNX file

The default configuration expects these paths:

```text
models/deeplob.onnx
models/transformerlob.onnx
```

Put your model in the matching location. Alternatively, keep it elsewhere and pass `--model-path /absolute/path/to/model.onnx` to `inspect`, `validate`, `smoke-test`, `predict`, or `benchmark`.

Build the optimized binary once:

```sh
cargo build --release
```

You can also run commands through Cargo with `cargo run --release -- ...` as shown below.

## 3. Configure the model and dataset

Copy the default configuration to a local file so the repository defaults remain untouched:

```sh
cp configs/default.toml configs/local.toml
```

Edit `configs/local.toml`. For example, the relevant fields could look like:

```toml
[models]
deeplob_path = "models/deeplob.onnx"
transformerlob_path = "models/transformerlob.onnx"

[data]
dataset_path = "data/raw/test.csv"
output_path = "outputs/predictions.csv"
segment_paths = []
feature_columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
label_column = 144
has_headers = false
schema_verified = true

[normalization]
mean_path = "data/normalization/mu.npy"
std_path = "data/normalization/sigma.npy"
```

These are examples, not verified dataset values. `feature_columns` and `label_column` are **zero-based column indices**. The default mapping assumes features at columns 0 through 39 and a label at column 144, but you must verify the actual file layout before setting `schema_verified = true`. If the CSV has a header row, set `has_headers = true` and check that the selected columns still correspond to the intended values.

The application currently expects labels in the input CSV during inference, even though predictions do not need labels mathematically. It uses those labels to populate the output's `true_class` field. Files without the configured label column are rejected by the current loader.

Normalization files must be the **training statistics used by the model**: two one-dimensional NPY arrays, each containing 40 values. The application loads these values; it does not calculate or infer them. Do not compute mean/std from the prediction dataset and substitute them for training statistics.

The preprocessing configuration is intentionally constrained to sequence length 100 and stride 100. The application produces non-overlapping windows and keeps files as separate segments, so a window never crosses a segment boundary.

## 4. Check runtime and model metadata

Run these from the repository root:

```sh
cargo run --release -- runtime-info
cargo run --release -- inspect --model deeplob
cargo run --release -- validate --model deeplob
```

For TransformerLOB, replace `deeplob` with `transformerlob`. To use a non-default model path, add `--model-path /absolute/path/to/model.onnx` to the `inspect` and `validate` commands.

`runtime-info` confirms that the CPU ONNX Runtime can initialize and prints configured model paths. `inspect` loads the selected ONNX file and prints its input/output names, element types, and dimensions. `validate` checks the selected graph against the configured shape and class contract. Resolve any metadata mismatch before moving on.

## 5. Run a synthetic smoke test

This checks that the selected model loads and can make one prediction from a synthetic all-zero tensor:

```sh
cargo run --release -- smoke-test --model deeplob
```

Synthetic smoke testing does **not** load the CSV, normalization files, or validate that model outputs are accurate for market data. It is a model/runtime check only.

## 6. Smoke test with a real dataset window

After configuring and verifying the CSV schema and normalization paths, prepare one real window and pass it through the model:

```sh
cargo run --release -- --config configs/local.toml smoke-test \
  --model deeplob \
  --input data/raw/test.csv \
  --mean-path data/normalization/mu.npy \
  --std-path data/normalization/sigma.npy
```

This checks CSV reading, feature extraction, normalization, window construction, and one inference call. A complete window requires at least 100 valid data rows. The current preprocessing pipeline rejects blank rows, malformed numeric values, invalid labels, inconsistent CSV widths, and missing selected columns.

## 7. Run inference over all windows

```sh
cargo run --release -- --config configs/local.toml predict \
  --model deeplob \
  --input data/raw/test.csv \
  --mean-path data/normalization/mu.npy \
  --std-path data/normalization/sigma.npy \
  --output outputs/deeplob-predictions.csv
```

For TransformerLOB, change the model name and its configured model path:

```sh
cargo run --release -- --config configs/local.toml predict \
  --model transformerlob \
  --input data/raw/test.csv \
  --output outputs/transformerlob-predictions.csv
```

The CLI overrides for the mean and standard deviation files are optional when `configs/local.toml` already contains the right paths. A successful run writes one CSV record per complete window, including the source, window location, true label, predicted class/direction, and three class scores. The internal output classes are 0=Down, 1=Stationary, and 2=Up. Output is first written to a temporary file and published after every prediction succeeds. Existing output is preserved unless you pass `--overwrite`.

For multiple chronological CSV segments, provide each with a repeated `--segment` option instead of `--input`:

```sh
cargo run --release -- --config configs/local.toml predict \
  --model deeplob \
  --segment data/raw/day-1.csv \
  --segment data/raw/day-2.csv \
  --output outputs/multiple-days.csv
```

Do not point `dataset_path` at a directory containing several CSV files and rely on file ordering. The loader requires explicit `segment_paths` or repeated `--segment` arguments when more than one CSV is found.

## 8. Benchmark the model

Once real-data inference succeeds, run the pure inference benchmark:

```sh
cargo run --release -- --config configs/local.toml benchmark \
  --model deeplob \
  --input data/raw/test.csv \
  --warmup 20 \
  --iterations 200 \
  --trials 5 \
  --output results/deeplob-inference.json
```

Inference mode loads the model once, prepares the first window tensor before timing, runs the configured warm-up, then measures each inference call. It reuses that same tensor to isolate model execution. The report includes raw latency samples, mean, population standard deviation, minimum, maximum, P50/P90/P95/P99, trial throughput, and runtime/machine metadata. Percentiles use nearest rank. Model and runtime initialization are reported separately from inference latency.

The default report format follows the output extension: `.csv` writes CSV; other extensions write JSON. You may specify `--format json` or `--format csv`. Existing report paths are never replaced. Use a new filename for each run.

Other benchmark modes:

```sh
# CSV/feature loading, normalization, window generation, and tensor construction
cargo run --release -- --config configs/local.toml benchmark --model deeplob \
  --mode preprocessing --output results/preprocessing.json

# Preprocessing, inference over every window, and in-memory prediction serialization
cargo run --release -- --config configs/local.toml benchmark --model deeplob \
  --mode end-to-end --output results/end-to-end.json

# Repeated ONNX session loading, with runtime initialization reported separately
cargo run --release -- --config configs/local.toml benchmark --model deeplob \
  --mode load --trials 5 --output results/model-load.json
```

To collect optional resource samples, add `--profile-memory`, `--profile-cpu`, or both. The collector samples process RSS and the OS-reported CPU percentage concurrently at 100 ms intervals. This is best-effort, platform-dependent, and adds some system load; treat resource-profiled runs separately from clean latency baselines. Benchmark results vary with CPU frequency, thermal state, background activity, memory pressure, OS scheduling, runtime version, and thread settings. Run release builds under similar, otherwise idle conditions and keep the generated reports.

## 9. Compare Rust and C++ measurements

The repository does not contain the C++ reference engine. Run it separately, collect real measurements, and convert them to the benchmark JSON report structure. Then compare:

```sh
cargo run --release -- compare \
  --model deeplob \
  --rust-results results/deeplob-inference.json \
  --cpp-results results/deeplob-cpp.json \
  --output results/deeplob-comparison.json
```

The comparison checks that both reports identify the requested model and inference mode and agree on provider, input shape/type, model fingerprint, and runtime-version metadata. It reports mean, P50, P95, and P99 latency, latency improvement percentage, throughput, and the Rust/C++ throughput ratio. For a meaningful comparison, also ensure that both runs used identical input tensor values, preprocessing, normalization, ONNX Runtime version, CPU provider, thread settings, and optimized builds. The report cannot prove that imported C++ measurements or those conditions are genuine; do not claim one implementation is faster unless the runs were controlled.

## 10. Common problems

| Message or symptom | What to check |
|---|---|
| Model file does not exist | Confirm `[models]` in the chosen TOML file or provide `--model-path`. |
| Metadata/input shape rejected | Check `inspect` output. The current app requires float32 `[1, 1, 100, 40]` input and three classes. |
| Schema is not verified | Inspect the actual CSV and confirm feature/label columns before setting `schema_verified = true`. |
| Missing normalization file | Supply the model's training `mu.npy` and `sigma.npy` through config or `--mean-path`/`--std-path`. |
| CSV row has too few columns | Verify the selected zero-based feature and label indices and whether the file has a header. |
| Invalid label | The configured label column must contain supported FI-2010 labels (1, 2, or 3). |
| No complete windows | The selected CSV segment needs at least 100 data rows after its header. |
| Several CSV files found in a directory | Supply their chronological order with repeated `--segment` options. |
| Output/report already exists | Choose a new path; prediction output can be replaced with `--overwrite`. Benchmark and comparison reports intentionally do not overwrite. |
| TransformerLOB graph rejected | Its actual model must accept the current `[1, 1, 100, 40]` contract; no automatic layout conversion is implemented. |

## 11. What this workflow does not establish

Successful model loading and prediction do not demonstrate numerical equivalence with the original C++ implementation or correctness of the model's learned behavior. Those require suitable reference outputs, real model/data assets, and the later numerical-equivalence validation phase. The repository does not include ONNX models, normalization files, or FI-2010 CSV data, so those must be supplied before the real-data steps can run.
