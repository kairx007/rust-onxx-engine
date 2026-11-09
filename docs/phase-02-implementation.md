# Phase 2 implementation report

## Implemented

- Extended the TOML configuration for data columns, segment ordering, normalization files, preprocessing, and tensor shape.
- Added segmented datasets with flat row-major `f32` features and aligned internal labels.
- Added configurable CSV loading, feature extraction, label mapping, field-count checks, malformed/blank-row rejection, and path/row-specific errors.
- Added 1D NPY vector loading for 40-element `f32` or `f64` means and standard deviations. Values are validated; statistics are never calculated or substituted.
- Added in-place feature-wise Z-score normalization with epsilon exactly `1e-8`.
- Added non-overlapping 100-row windows, final-row labels, and segment-boundary isolation.
- Added contiguous `[1, 1, 100, 40]` input buffers, structural/value checks, and ownership transfer into an ONNX Runtime tensor without model execution.
- Added `PreprocessingPipeline`, a prepared dataset/window API, and a preprocessing demo.
- Added synthetic tests for CSV, features, labels, NPY statistics, normalization, windows, tensor layout/ORT construction, and the full pipeline.

## Data mapping safety

The repository has no real FI-2010 CSVs or normalization files. The defaults `feature_columns = 0..39`, `label_column = 144`, and `has_headers = false` are candidate values from the phase specification only. `schema_verified` defaults to `false`; `PreprocessingPipeline::new` refuses to run until the user has inspected the actual dataset and explicitly marks the mapping verified. A directory with multiple CSV files also requires an explicit chronological `segment_paths` list. Each listed file is kept as its own segment.

The implementation assumes labels are in the same CSV as the selected features. If the actual FI-2010 distribution stores labels separately, the data loader/config must be extended after that format is inspected. Synthetic fixtures do not establish real-data compatibility or training-statistic provenance.

## Normalization and tensor behavior

The NPY reader accepts one-dimensional C-order `[40]` floating vectors encoded as `f32` or `f64`, converting values to `f32` and rejecting non-finite values. Negative standard deviations are rejected. Zero standard deviation is allowed because the specified `+ 1e-8` denominator handles it; non-finite normalized results are still rejected.

Normalized rows stay in their source segment. Window descriptors contain segment index and start row, and carry the label from the window's last row. Tensor data is copied only when requested into one contiguous 4,000-element row-major `Vec<f32>`, then can be moved into `ort::value::Tensor<f32>`. No inference is called.

## Validation

The synthetic suite covers valid/missing/empty/malformed CSVs, headers and inconsistent field widths, feature ordering, label mapping, blank-row rejection, path/segment ordering, NPY dtype/shape/value validation, normalization including zero standard deviation, non-overlapping windows and incomplete tails, boundary isolation, tensor shape/order/length/finiteness, ORT tensor construction, and a complete synthetic pipeline.

## Verification completed

- `cargo fmt --check` — passed.
- `cargo clippy --all-targets --all-features -- -D warnings` — passed.
- `cargo test` — passed, 36 tests across the unit, integration, and documentation test targets.
- `cargo build --release --quiet` — passed.
- `cargo run --quiet -- runtime-info` — passed; reports the CPU execution provider and `[1, 1, 100, 40]` input shape.
- `cargo run --quiet --example preprocessing_demo` — stopped with the intended `schema_verified is false` configuration error, since the repository contains no real dataset to verify the candidate mapping against.

Real-data validation remains pending until the FI-2010 and normalization assets are supplied and the column mapping is checked.
