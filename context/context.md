# RustLOB — Rust Alternative to DeepLOB-HFT-Cpp

## 1. Project Overview

**Project Name:** RustLOB  
**Objective:** Build a high-performance Rust alternative to the existing DeepLOB-HFT-Cpp inference engine.

The objective is to reproduce the inference functionality of the existing C++ implementation using Rust, maintaining equivalent model predictions, preprocessing behaviour, and benchmarking methodology.

The Rust implementation will use ONNX Runtime through the Rust `ort` crate.

This is primarily a systems programming and performance benchmarking project, not a model-training project.

### Primary Goals

1. Reimplement the existing C++ inference engine in Rust.
2. Support both DeepLOB and TransformerLOB architectures.
3. Maintain numerical equivalence with the existing ONNX/C++ implementation.
4. Implement standalone inference, benchmarking, and CSV evaluation workflows.
5. Measure inference latency, throughput, and memory consumption.
6. Compare Rust and C++ performance under identical experimental conditions.

---

## 2. Reference Implementation

The project must follow the existing research implementation:

**Reference:** DeepLOB-HFT-Cpp — A Production-Grade Deep Learning Pipeline for Limit Order Book Mid-Price Prediction with High-Performance C++ Inference.

The existing C++ implementation uses:

- ONNX Runtime C++ API.
- CMake build system.
- DeepLOB model.
- TransformerLOB model.
- FI-2010 dataset.
- Single-sample inference.
- Latency benchmarking.
- Full CSV inference and evaluation.

The Rust implementation should reproduce these capabilities rather than introduce a different inference methodology.

---

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Rust |
| Build system | Cargo |
| Inference engine | ONNX Runtime |
| Rust ONNX bindings | ort |
| Tensor representation | ndarray / contiguous buffers |
| CSV processing | csv |
| Serialization | serde |
| CLI | clap |
| Benchmark timing | std::time::Instant |
| Error handling | anyhow / thiserror |
| Logging | tracing |
| Testing | Rust built-in testing framework |

Use stable Rust and compatible crate versions.

Avoid unnecessary dependencies.

---

## 4. Model Specifications

The inference engine must support two architectures.

### DeepLOB

- CNN front-end: Three convolutional blocks.
- Temporal encoder: Single-layer LSTM.
- Hidden units: 64.
- Approximate parameters: 66K.
- Output: Three classification logits.

### TransformerLOB

- CNN front-end: Three convolutional blocks.
- Temporal encoder: Two Transformer encoder layers.
- Attention heads: 4.
- Model dimension: 64.
- Feed-forward dimension: 128.
- Approximate parameters: 114K.
- Output: Three classification logits.

### Common Input Specifications

```text
Input shape: [1, 1, 100, 40]
Batch size: 1
Time steps: 100
Features: 40
Output classes: 3
```

Class mapping:

```text
0 = Down
1 = Stationary
2 = Up
```

The Rust engine must load existing ONNX models. Do not implement or retrain the neural network architectures in Rust.

---

# 5. Seven Implementation Phases

## Phase 1 — Project Setup and ONNX Runtime Integration

**Objective:** Establish the Rust project and initialize ONNX Runtime.

Tasks:

- Create the Cargo project structure.
- Configure dependencies.
- Integrate the `ort` crate.
- Configure ONNX Runtime provider selection.
- Implement model loading.
- Implement runtime initialization and session management.
- Add structured error handling.
- Add logging.

Expected output:

- Working Rust project.
- Successfully loaded ONNX model.
- Runtime initialization tests.
- Model metadata inspection.

Acceptance criteria:

- Cargo project compiles.
- ONNX Runtime initializes successfully.
- Both model files can be loaded.
- Invalid model paths produce meaningful errors.

## Phase 2 — Tensor Construction and Preprocessing

**Objective:** Reproduce the input preprocessing behaviour of the C++ implementation.

Tasks:

- Implement input tensor construction.
- Load normalization parameters (`mu.npy`, `sigma.npy`).
- Implement feature-wise Z-score normalization.
- Implement 100-timestep window extraction.
- Preserve the original feature ordering.
- Ensure contiguous tensor memory.
- Validate tensor dimensions and input values.

Normalization:

```text
X_norm = (X - mu) / (sigma + epsilon)

epsilon = 1e-8
```

Input shape:

```text
[1, 1, 100, 40]
```

Acceptance criteria:

- Identical preprocessing semantics to the reference implementation.
- Correct tensor dimensions.
- No unintended copies in the inference hot path.
- Numerical preprocessing tests pass.

## Phase 3 — DeepLOB Inference

**Objective:** Implement DeepLOB inference using ONNX Runtime.

Tasks:

- Load DeepLOB ONNX model.
- Prepare the input tensor.
- Execute ONNX inference.
- Extract three output logits.
- Determine the predicted class.
- Measure individual inference execution time.
- Handle invalid inputs and runtime errors.

Acceptance criteria:

- DeepLOB inference executes successfully.
- Predictions match the reference ONNX/C++ implementation.
- Repeated inference works without memory errors.

## Phase 4 — TransformerLOB Inference

**Objective:** Implement TransformerLOB inference.

Tasks:

- Load TransformerLOB ONNX model.
- Reuse the common tensor pipeline.
- Execute model inference.
- Extract output logits.
- Determine predicted class.
- Measure execution latency.
- Validate output dimensions and numerical consistency.

Acceptance criteria:

- TransformerLOB inference executes successfully.
- Predictions match the reference implementation.
- Both models share the same inference abstractions wherever practical.

## Phase 5 — Inference Executables

**Objective:** Reproduce the three C++ executables as Rust CLI workflows.

### 1. Smoke Test

Equivalent to `inference.exe`.

Responsibilities:

- Load an ONNX model.
- Construct a sample input tensor.
- Execute one inference.
- Display predicted class and logits.
- Report execution status.

### 2. Benchmark Executable

Equivalent to `benchmark.exe`.

Responsibilities:

- Load the selected model.
- Execute warmup iterations.
- Execute timed inference iterations.
- Calculate latency statistics.
- Display benchmark results.

### 3. Full CSV Inference

Equivalent to `full_inference.exe`.

Responsibilities:

- Load FI-2010 CSV data.
- Load normalization parameters.
- Extract 100-timestep windows.
- Perform inference for each window.
- Compare predictions against ground-truth labels.
- Calculate classification accuracy.
- Calculate throughput and total execution time.

Acceptance criteria:

- All three workflows execute independently.
- Both model architectures are supported.
- Errors are handled gracefully.
- Output formats are consistent.

## Phase 6 — Benchmarking and Performance Profiling

**Objective:** Build a reproducible Rust performance benchmarking system.

Benchmark configuration:

```text
Warmup iterations: 20
Timed iterations: 200
Input shape: [1, 1, 100, 40]
Batch size: 1
Execution provider: CPU
```

Metrics:

- Mean latency.
- Median latency.
- P50 latency.
- P99 latency.
- P999 latency, where sample count permits meaningful estimation.
- Standard deviation.
- Throughput (windows/second).
- Peak memory consumption.

Benchmarking requirements:

- Use `std::time::Instant`.
- Separate model loading from inference timing.
- Separate cold-start measurements from warm inference.
- Avoid logging inside timed inference loops.
- Keep input preparation outside the model execution timer when measuring pure inference.
- Report end-to-end latency separately.
- Use consistent CPU execution settings.

Acceptance criteria:

- Reproducible benchmark execution.
- Structured benchmark output.
- Separate measurements for both architectures.
- No accidental inclusion of preprocessing or model-loading time in pure inference latency.

## Phase 7 — Numerical Equivalence and Final Validation

**Objective:** Verify that the Rust implementation reproduces the reference C++ inference behaviour.

Tasks:

- Compare Rust predictions with C++ predictions.
- Compare Rust output logits with ONNX reference outputs.
- Validate preprocessing consistency.
- Test multiple input windows.
- Test invalid inputs.
- Test repeated inference.
- Test both architectures.
- Run complete CSV evaluation.
- Document benchmark methodology and results.

Numerical tolerance:

```text
Absolute tolerance: 1e-5
```

Final acceptance criteria:

- Both architectures work correctly.
- Numerical equivalence tests pass.
- All three inference workflows work.
- Benchmark reports are generated.
- CSV evaluation completes successfully.
- The project builds from a clean checkout.

---

# 6. Proposed Project Architecture

```text
rust-lob/
│
├── Cargo.toml
├── Cargo.lock
├── README.md
├── PROJECT_CONTEXT.md
│
├── models/
│   ├── deeplob.onnx
│   └── transformerlob.onnx
│
├── configs/
│   └── default.toml
│
├── data/
│   └── normalization/
│       ├── mu.npy
│       └── sigma.npy
│
├── src/
│   ├── main.rs
│   │
│   ├── cli/
│   │   ├── mod.rs
│   │   ├── inference.rs
│   │   ├── benchmark.rs
│   │   └── evaluate.rs
│   │
│   ├── runtime/
│   │   ├── mod.rs
│   │   ├── session.rs
│   │   └── model.rs
│   │
│   ├── preprocessing/
│   │   ├── mod.rs
│   │   ├── normalization.rs
│   │   ├── window.rs
│   │   └── tensor.rs
│   │
│   ├── inference/
│   │   ├── mod.rs
│   │   ├── deeplob.rs
│   │   ├── transformerlob.rs
│   │   └── output.rs
│   │
│   ├── benchmark/
│   │   ├── mod.rs
│   │   ├── latency.rs
│   │   ├── statistics.rs
│   │   └── memory.rs
│   │
│   ├── evaluation/
│   │   ├── mod.rs
│   │   ├── csv_loader.rs
│   │   ├── metrics.rs
│   │   └── evaluator.rs
│   │
│   └── error.rs
│
├── tests/
│   ├── runtime_tests.rs
│   ├── preprocessing_tests.rs
│   ├── deeplob_tests.rs
│   ├── transformerlob_tests.rs
│   └── equivalence_tests.rs
│
└── results/
    └── benchmarks/
```

This is a proposed structure. Codex should adjust it when justified by actual implementation requirements.

---

# 7. Coding Standards and Constraints

The AI coding agent must follow these rules:

1. Use Rust as the primary implementation language.
2. Use ONNX Runtime through the `ort` crate.
3. Do not introduce Candle, Burn, or `tch` as additional inference backends.
4. Do not retrain or recreate the neural network models.
5. Do not change the model input shape.
6. Preserve the original preprocessing semantics.
7. Avoid unnecessary memory allocations in the inference hot path.
8. Prefer reusable abstractions over duplicated code.
9. Keep model loading outside the inference hot path.
10. Use explicit error handling instead of excessive `.unwrap()`.
11. Write unit and integration tests for every major module.
12. Do not fabricate benchmark results.
13. Never claim numerical equivalence without executing comparison tests.
14. Keep the implementation modular and maintainable.

---

# 8. Codex Execution Instructions

Treat this document as the project's primary implementation specification.

### Mandatory workflow

- Work on one phase at a time.
- Inspect existing files before writing code.
- Understand the existing architecture before modifying it.
- Never blindly overwrite working implementations.
- Explain the proposed implementation before major architectural changes.
- Write code in small, logical increments.
- Compile and test after each meaningful change.
- Fix errors before proceeding.
- Update project documentation as implementation progresses.
- Maintain a checklist of completed tasks.
- Do not silently skip requirements.

### Phase completion protocol

Before marking a phase complete, provide:

1. Files created or modified.
2. Features implemented.
3. Tests executed.
4. Test results.
5. Known limitations.
6. Remaining tasks.

Do not begin the next phase until the current phase meets its acceptance criteria.

### Initial instruction to Codex

Start with Phase 1 only.

First inspect the repository and available assets. Identify whether the ONNX models, normalization files, and reference implementation are available.

Then propose the Cargo workspace structure and dependency versions.

Do not implement Phase 2 or subsequent phases until Phase 1 is completed and validated.

---

# 9. Final Deliverables

The completed project should provide:

- A standalone Rust ONNX inference engine.
- DeepLOB inference support.
- TransformerLOB inference support.
- Single-sample inference CLI.
- Latency benchmark CLI.
- Full CSV evaluation CLI.
- Numerical equivalence test suite.
- Reproducible performance reports.
- Build and usage documentation.

### Estimated Scope

| Metric | Estimate |
|---|---:|
| Implementation phases | 7 |
| Production Rust LOC | 1,450–2,200 |
| Inference backends | 1 (ONNX Runtime) |
| Model architectures | 2 |
| CLI workflows | 3 |

The LOC estimate is indicative, not a code-generation target.

### Definition of Success

The project is successful when the Rust implementation provides the same functional inference capabilities as the reference C++ implementation, produces numerically equivalent predictions, and provides a reproducible performance comparison under controlled experimental conditions.

**Primary objective: A reliable, memory-efficient and high-performance Rust replacement for the existing C++ DeepLOB-HFT inference engine.**
