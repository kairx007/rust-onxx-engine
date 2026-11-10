# Phase 5 implementation report

## Implemented

- Added the `lob-inference` binary with Clap subcommands for `runtime-info`, `inspect`, `validate`, `smoke-test`, `predict`, and `benchmark`.
- Preserved the existing `runtime-info`, `inspect --model`, and `validate --model` invocation patterns.
- Split argument definitions from command dispatch and added an application layer for model selection, inference workflows, and CSV output.
- Added a model factory that loads either DeepLOB or TransformerLOB once through the existing runtime/session infrastructure.
- Added CLI overrides for input/ordered segment paths, model path, normalization paths, output path, log level, and overwrite behavior. Existing TOML fields are preserved; `data.output_path` is optional for backward compatibility.
- Added runtime reporting, actual metadata inspection, and model-specific contract validation.
- Added synthetic or dataset-backed smoke test orchestration.
- Added CSV prediction over Phase 2 prepared windows, preserving window order and labels. Output is streamed to a temporary file and published only after all predictions succeed.
- Added a basic benchmark command that loads one model, prepares one tensor outside the timer, then repeats inference against that tensor. It reports engine load and inference timing separately; runtime initialization and preprocessing are not included in those timings.
- Added unit coverage for CLI parsing, configuration overrides, model selection errors, synthetic tensor creation, CSV schema, overwrite protection/replacement, and temporary-file cleanup.

## Output schema

```text
window_index,source_file,segment_index,start_row,true_class,predicted_class,predicted_direction,down_score,stationary_score,up_score
```

Classes use the existing internal mapping `0=Down`, `1=Stationary`, `2=Up`. Existing output files are preserved unless `--overwrite` is supplied.

## Model/data availability

The repository contains no ONNX models, FI-2010 CSVs, or normalization NPY files. The default preprocessing schema remains unverified. CLI help and runtime reporting can be exercised, and synthetic unit tests validate the application components, but actual model inspection/validation, smoke inference, CSV prediction, and benchmark execution cannot succeed until those assets and a verified data mapping are provided. Tests do not claim real-model inference.

## Verification

Quality gates passed: `cargo fmt --check`, `cargo check`, `cargo test` (59 passed, 0 failed), `cargo clippy --all-targets --all-features -- -D warnings`, `cargo build --release --quiet`, and `git diff --check`. Top-level, inspect, predict, benchmark, and smoke-test help commands were exercised; `runtime-info` initialized CPU ONNX Runtime and reported the API target and both model files as missing. A smoke-test invocation exited with the expected missing-DeepLOB-model error. No real inference, CSV prediction, or benchmark run was possible without the model/data/normalization assets.
