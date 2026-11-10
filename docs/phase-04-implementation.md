# Phase 4 implementation report

## Implemented

- Added `TransformerLobContract`, resolving input/output names from session metadata and validating one float32 input, one float32 output, the Phase 2 `[1, 1, 100, 40]` layout (static or dynamic dimensions), and a three-class final output axis.
- Added a TransformerLOB input adapter that reuses the borrowed Phase 2 data without copying or changing order. It rejects incompatible static shapes; it does not guess a reshape, transpose, pad, or truncation.
- Added `TransformerLobEngine`, retaining a single `ModelSession` and delegating named ONNX execution/output handling to the shared inference runner.
- Refactored DeepLOB to use that same runner while preserving its model-specific metadata and input handling.
- Reused the shared `Prediction`, output validation, class mapping, errors, Phase 1 runtime, and Phase 2 `InputTensor`.
- Added unit tests for TransformerLOB metadata and input adapter behavior.

## Metadata and model availability

There is no `.onnx` file in the repository. Actual TransformerLOB metadata cannot be inspected. The implementation's candidate contract is therefore limited to a single float32 input compatible with the Phase 2 tensor, and a single float32 output with a final three-class axis. Input/output names are obtained from a future loaded model rather than hard-coded. No layout transformation is implemented because no model evidence specifies one. DeepLOB's actual model metadata is also unavailable, although its existing configured contract remains unchanged.

Consequently, no real TransformerLOB load, prediction, or repeated-session inference was possible. The added tests exercise contract validation and tensor adaptation only; they do not establish compatibility with a TransformerLOB ONNX model.

## Verification

Quality gates passed: `cargo fmt --check`, `cargo check`, `cargo test` (46 passed, 0 failed), `cargo clippy --all-targets --all-features -- -D warnings`, `cargo build --release --quiet`, and `git diff --check`. Real-model validation remains pending until the model is supplied.
