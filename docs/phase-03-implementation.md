# Phase 3 implementation report

## Implemented

- Added `DeepLobContract`, which validates the existing model metadata contract and resolves input/output names from the loaded graph instead of hard-coding them.
- Added a borrowed `TensorRef<f32>` adapter for Phase 2 `InputTensor` values. It validates the input and preserves the contiguous `[1, 1, 100, 40]` data without cloning its feature buffer.
- Added `DeepLobEngine`, which loads one reusable `ModelSession`, validates it during construction, runs named-input inference, finds the expected named output, and converts validated scores into a prediction.
- Added output validation against the dimensions declared by the loaded model, requiring a final class axis of three and singleton runtime prefix dimensions for one input, exactly three finite values, and preserved class ordering.
- Added `Direction` and `Prediction` types with Down/Stationary/Up mapping and lowest-index tie-breaking. The returned scores are unchanged; no softmax is applied.
- Added inference-specific error variants and mutable session access needed by the pinned `ort` session execution API.

## Model metadata status

There is no `.onnx` file in `models/` or elsewhere in the project tree. Actual DeepLOB input/output names, output layout, and whether the output represents logits or probabilities could not be inspected. At load time, names are taken from the actual session metadata. The implementation requires a float32 input compatible with configured `[1, 1, 100, 40]`, one float32 output whose final axis is three classes and whose prefix dimensions are singleton or dynamic in metadata, and exactly three finite output values with singleton runtime prefix dimensions. Real model compatibility remains unverified until the asset is supplied.

## Tests and checks

- Unit tests cover model metadata/name resolution, invalid output layouts, borrowed tensor shape and ordering, score shape and finiteness, three-class mapping, and deterministic ties.
- Existing Phase 1 and Phase 2 tests remain in the suite.
- Real-model load, inference, and repeated-session inference tests could not run because no DeepLOB ONNX model is available. Synthetic metadata and score tests are not evidence of model execution.

Quality gates passed: `cargo fmt --check`, `cargo check`, `cargo test` (43 passed, 0 failed), `cargo clippy --all-targets --all-features -- -D warnings`, and `cargo build --release --quiet`. A search of the project tree found no `.onnx` model files.
