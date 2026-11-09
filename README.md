# RustLOB

RustLOB is a Rust implementation of the ONNX Runtime foundation for DeepLOB and TransformerLOB.

Phase 1 provides configuration, CPU runtime initialization, reusable model sessions, model metadata inspection, and command-line diagnostics. Preprocessing, inference workflows, benchmarking, and CSV evaluation are outside this phase.

## Requirements

- Stable Rust 1.92 or later (required by `ort` 2.0.0-rc.15)
- ONNX model files supplied separately in `models/`
- Xcode Command Line Tools on macOS; the bundled ONNX Runtime archive requires a macOS 13.4 deployment target

`ort` is pinned to `2.0.0-rc.15`, which wraps ONNX Runtime 1.31. The configured `download-binaries` feature provides the runtime without a separately installed system package; unused `ndarray` support is disabled. See the [ort crate documentation](https://docs.rs/ort/2.0.0-rc.15/ort/) and its [environment API](https://docs.rs/ort/2.0.0-rc.15/ort/environment/index.html).

## Commands

```sh
cargo run -- --help
cargo run -- runtime-info
cargo run -- inspect --model deeplob
cargo run -- validate --model transformerlob
```

Use a custom configuration by passing `--config` before or after the command:

```sh
cargo run -- --config configs/custom.toml runtime-info
```

The default configuration expects `models/deeplob.onnx` and `models/transformerlob.onnx`. Model paths are interpreted relative to the current working directory. Inspection and validation exit with an error when a model is missing or its metadata does not match the configured input/output specifications.

## Phase 1 implementation status

The Cargo package, configuration validation, structured logging, CPU runtime, model registry, session loading, metadata inspection, and CLI diagnostics are implemented. Unit and integration tests cover configuration, runtime initialization, registry behavior, invalid model files, and metadata validation.

The research ONNX models are not included in this repository. Their real input/output metadata and successful DeepLOB/TransformerLOB session loading must be checked after the model files are provided. No inference or performance results are claimed.

See [`context/context.md`](context/context.md) for the overall project specification and [`context/phase-01.md`](context/phase-01.md) for detailed Phase 1 requirements.
