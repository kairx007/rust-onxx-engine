# RustLOB

RustLOB is a Rust implementation of the ONNX Runtime inference foundation for DeepLOB and TransformerLOB.

Phase 1 establishes configuration, runtime/session management, model metadata inspection, and command-line diagnostics. Preprocessing, inference workflows, benchmarking, and CSV evaluation are outside this phase.

## Requirements

- Stable Rust toolchain
- ONNX model files supplied separately in `models/`

See [`context/context.md`](context/context.md) for the project specification and [`context/phase-01.md`](context/phase-01.md) for Phase 1 requirements.
