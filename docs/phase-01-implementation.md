# Phase 1 implementation report

## Delivered

- Rust 2024 Cargo package with a locked dependency graph and stable toolchain configuration.
- TOML configuration loading and validation, including CPU-only provider enforcement and configurable model paths.
- Structured `tracing` logging and domain-specific errors.
- Process-wide ONNX Runtime initialization through `ort`, with CPU execution provider selection and telemetry disabled.
- DeepLOB and TransformerLOB registry entries and missing-file detection.
- Reusable ONNX sessions with input/output names, element types, and dimensions extracted from the actual session metadata.
- Metadata checks for the configured `[1, 1, 100, 40]` input and three class output, accepting dynamic dimensions.
- `runtime-info`, `inspect`, and `validate` CLI commands.
- Tests for configuration, runtime initialization, model path resolution, invalid ONNX rejection, and metadata compatibility.

## Dependency choices

`ort` is pinned to `2.0.0-rc.15` and wraps ONNX Runtime 1.31. The version is pinned because the 2.0 API is still a release candidate. Resolved direct versions are: anyhow 1.0.104, clap 4.6.7, ort 2.0.0-rc.15, serde 1.0.229, thiserror 2.0.21, toml 0.9.12, tracing 0.1.44, and tracing-subscriber 0.3.23. Only the `ort` runtime, std, tracing, binary download/TLS, and API 31 features are enabled; ndarray support is disabled because this phase does not construct tensors. No alternate inference backend is included.

On macOS, the bundled static runtime requires a 13.4 deployment target and Apple's availability helper from Xcode. `build.rs` locates and links the compiler runtime from the active Xcode toolchain.

## Validation performed

- `cargo fmt --check` — passed.
- `cargo clippy --all-targets --all-features -- -D warnings` — passed.
- `cargo check` — passed.
- `cargo test` — passed (13 tests).
- `cargo build --release` — passed after adding the macOS linker configuration.
- `cargo run -- --help` — displayed all three commands.
- `cargo run -- runtime-info` — initialized ONNX Runtime and reported CPU.
- `cargo run -- inspect --model deeplob` — correctly returned exit code 1 and reported the missing model.

## Limitations

Neither ONNX model is present. Successful loading and metadata validation against the research models remain pending, as do architecture-specific compatibility checks. The test suite rejects an invalid temporary ONNX file but does not claim equivalence or successful loading for DeepLOB or TransformerLOB. GPU support and all Phase 2+ functionality remain out of scope.

## Git and PR

- Branch: `feat/runtime-foundation`.
- No GitHub remote is configured, per the local Git-only instruction. This report can serve as the Phase 1 pull request summary when a remote is added.
