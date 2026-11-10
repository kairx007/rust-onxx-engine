# PHASE_07.md — Numerical Equivalence and Final Validation

## 1. Phase Overview

**Phase:** 07 — Numerical Equivalence and Final Validation  
**Branch:** `test/equivalence-validation`  
**Objective:** Validate the complete Rust inference engine against the original C++ implementation and establish that both implementations produce equivalent results using identical ONNX models, inputs, and preprocessing.

### Project context

The project is a Rust alternative to the existing `DeepLOB-HFT-Cpp` inference engine.

The Rust implementation supports:

- DeepLOB inference.
- TransformerLOB inference.
- ONNX Runtime CPU execution.
- FI-2010 preprocessing.
- CSV-based inference.
- Benchmarking and performance profiling.

This phase validates the complete system developed in Phases 1–6.

### Reference implementation

The original C++ engine is the reference implementation.

Use it to compare:

1. Preprocessed input tensors.
2. ONNX model outputs.
3. Predicted classes.
4. Full-dataset predictions.
5. Evaluation metrics.
6. End-to-end execution behavior.

The original paper also describes PyTorch-to-ONNX output verification with an absolute tolerance of `1e-5`. This is a reference validation criterion from the model export process, not an automatically guaranteed Rust-versus-C++ tolerance.

### Scope

**In scope:**
- Rust versus C++ input equivalence.
- Rust versus C++ numerical output comparison.
- DeepLOB validation.
- TransformerLOB validation.
- Prediction consistency.
- Full-dataset evaluation.
- Classification metrics.
- Error and edge-case validation.
- Final regression testing.
- Final project documentation.

**Out of scope:**
- New model architectures.
- Model retraining.
- ONNX model export changes.
- GPU support.
- New inference backends.
- Unverified performance optimization.
- Changes to the original C++ engine's behavior.

---

## 2. Validation Objectives

The validation must answer the following questions:

1. Does Rust preprocess the input data identically to C++?
2. Does Rust construct the same input tensors?
3. Does Rust pass the correct tensor dimensions and values to ONNX Runtime?
4. Do both implementations produce numerically equivalent model outputs?
5. Do both implementations predict the same classes?
6. Do both implementations produce equivalent results over the full test dataset?
7. Are accuracy and class-wise metrics consistent?
8. Are errors handled consistently?
9. Does the Rust implementation preserve the original inference behavior?
10. Are the performance results from Phase 6 reproducible?

A successful release must be supported by actual validation evidence.

---

## 3. Validation Architecture

Suggested structure:

```text
src/
├── validation/
│   ├── mod.rs
│   ├── config.rs
│   ├── input_comparison.rs
│   ├── output_comparison.rs
│   ├── prediction_comparison.rs
│   ├── metrics_comparison.rs
│   ├── dataset_validation.rs
│   ├── tolerance.rs
│   ├── report.rs
│   └── error.rs
│
├── inference/
├── preprocessing/
├── benchmark/
├── application/
├── runtime/
├── models/
└── tensor/
```

Suggested responsibilities:

| Module | Responsibility |
|---|---|
| `config.rs` | Validation configuration |
| `input_comparison.rs` | Compare preprocessed tensors |
| `output_comparison.rs` | Compare raw model scores |
| `prediction_comparison.rs` | Compare predicted classes |
| `metrics_comparison.rs` | Compare evaluation metrics |
| `dataset_validation.rs` | Run full-dataset validation |
| `tolerance.rs` | Numerical tolerance calculations |
| `report.rs` | Generate validation reports |
| `error.rs` | Validation errors |

Adapt this structure to the existing repository. Reuse existing preprocessing, inference, and benchmarking functionality.

---

## 4. Task 1 — Establish the Validation Environment

Before comparing results, establish that both implementations use equivalent environments.

Record:

- Rust version.
- C++ compiler and version.
- Operating system and version.
- CPU model and architecture.
- Available CPU cores.
- RAM.
- ONNX Runtime version.
- Execution provider.
- Runtime optimization settings.
- Thread configuration.
- Model file paths.
- Model file hashes.
- Dataset file hashes.
- Normalization parameter hashes.
- Build configuration.

### Environment consistency

The following should be identical:

| Factor | Rust | C++ |
|---|---|---|
| DeepLOB ONNX model | Same | Same |
| TransformerLOB ONNX model | Same | Same |
| Dataset | Same | Same |
| Normalization parameters | Same | Same |
| Input tensor shape | Same | Same |
| Input data type | float32 | float32 |
| ONNX Runtime version | Same | Same |
| Execution provider | CPU | CPU |
| Runtime optimization | Same settings | Same settings |
| Thread configuration | Same | Same |

If any setting differs, document it.

Do not claim strict equivalence if the execution environments are materially different.

### Deliverable

An environment manifest for both Rust and C++ implementations.

---

## 5. Task 2 — Validate Model and Artifact Identity

Before comparing predictions, verify that both implementations use the same model files.

For each model:

1. Calculate the ONNX file SHA-256 hash.
2. Compare Rust and C++ model hashes.
3. Inspect ONNX input metadata.
4. Inspect ONNX output metadata.
5. Verify tensor data types.
6. Verify input and output dimensions.
7. Verify the expected class count.
8. Verify class mapping.

Expected input contract from the paper:

| Property | Value |
|---|---|
| Shape | `[1, 1, 100, 40]` |
| Data type | float32 |
| Elements | 4,000 |

Expected output:
- Three class scores.
- Class 0: Down.
- Class 1: Stationary.
- Class 2: Up.

Use actual ONNX metadata to confirm these expectations for each model.

### Acceptance criteria

- Both implementations use identical ONNX artifacts.
- Model metadata matches.
- Normalization artifacts match.
- No unexpected model or preprocessing configuration differences exist.

---

## 6. Task 3 — Validate Preprocessing Equivalence

This is one of the most important stages.

Even if both implementations execute the same ONNX model, different preprocessing can produce different predictions.

Compare the preprocessing stages independently.

### A. CSV loading

Verify:

- Same input file.
- Same row count.
- Same feature columns.
- Same feature ordering.
- Same data types.
- Same chronological ordering.
- Same handling of invalid rows.
- Same day and instrument boundaries.

### B. Feature extraction

Verify:

- Exactly 40 expected features.
- Identical feature ordering.
- Identical source columns.
- Identical numerical values.
- No accidental inclusion of labels as input features.

### C. Normalization

The original paper specifies Z-score normalization using statistics computed exclusively from the training partition.

Validate:

\[
x' = \frac{x-\mu}{\sigma+\epsilon}
\]

Use the exact normalization convention implemented by the original project. In particular, verify whether epsilon is added to the standard deviation or applied through another documented convention.

The project specification uses:

\[
\epsilon = 10^{-8}
\]

Compare:
- Mean arrays.
- Standard deviation arrays.
- Normalization parameters.
- Normalized feature values.

Do not recalculate normalization parameters independently for Rust and C++.

Both must use the same saved training-derived parameters.

### D. Window extraction

Verify:

| Property | Expected |
|---|---|
| Window length | 100 |
| Stride | 100 |
| Overlap | None |
| Feature count | 40 |
| Input dimensions | `[1,1,100,40]` |

Compare:
- Number of generated windows.
- First window.
- Middle window.
- Last window.
- Window ordering.
- Day and instrument boundaries.
- Label alignment.

### E. Tensor comparison

For every selected test window, compare the complete 4,000-element tensor.

Report:

- Maximum absolute difference.
- Mean absolute difference.
- Root mean squared error.
- Number of unequal elements.
- Relative difference where meaningful.

Maximum absolute difference:

\[
D_{\max} = \max_i |x_i^{Rust}-x_i^{Cpp}|
\]

Mean absolute difference:

\[
MAE = \frac{1}{N}\sum_i |x_i^{Rust}-x_i^{Cpp}|
\]

Root mean squared error:

\[
RMSE = \sqrt{\frac{1}{N}\sum_i(x_i^{Rust}-x_i^{Cpp})^2}
\]

For identical preprocessing operations and inputs, investigate any unexplained differences rather than accepting them automatically.

### Acceptance criteria

- Input row counts match.
- Feature ordering matches.
- Normalization artifacts match.
- Window counts match.
- Tensor dimensions match.
- Tensor values match within the agreed numerical tolerance.
- No cross-boundary windows are introduced.

---

## 7. Task 4 — Validate DeepLOB Numerical Equivalence

Run DeepLOB inference in both implementations using identical input tensors.

### Test procedure

1. Load the same DeepLOB ONNX file in Rust.
2. Load the same DeepLOB ONNX file in C++.
3. Prepare identical input tensors.
4. Verify their dimensions and values.
5. Execute inference in Rust.
6. Execute inference in C++.
7. Extract raw output scores.
8. Compare the scores numerically.
9. Compare predicted class indices.
10. Save the comparison results.

### Output comparison

For each output class:

```text
Rust:
  Down       = ...
  Stationary = ...
  Up         = ...

C++:
  Down       = ...
  Stationary = ...
  Up         = ...
```

Calculate:

\[
D_i = |y_i^{Rust}-y_i^{Cpp}|
\]

Report:

- Maximum absolute difference.
- Mean absolute difference.
- RMSE.
- Maximum relative difference where meaningful.
- Number of outputs outside tolerance.
- Predicted class agreement.

### Numerical tolerance

Use the following as an initial validation policy:

\[
|y_{Rust}-y_{Cpp}| \leq 10^{-5}
\]

This is inspired by the tolerance specified in the original paper for PyTorch-to-ONNX verification.

However, do not assume that this tolerance is guaranteed for Rust-versus-C++ execution.

First establish whether both implementations use identical runtime versions, execution providers, optimization settings, and thread configurations.

If a difference exceeds tolerance:

1. Preserve the failing input tensor.
2. Preserve both output vectors.
3. Verify model identity.
4. Verify input tensor identity.
5. Verify runtime settings.
6. Investigate numerical differences.
7. Document the cause.
8. Define a justified tolerance only if strict equality is not technically achievable.

Do not increase tolerance merely to make tests pass.

### Acceptance criteria

- DeepLOB loads in both implementations.
- Both receive identical input tensors.
- Output scores satisfy the justified numerical tolerance.
- Predicted classes agree.
- All failures are documented.

---

## 8. Task 5 — Validate TransformerLOB Numerical Equivalence

Repeat the same process for TransformerLOB.

Pay particular attention to:

- Actual TransformerLOB input names.
- Input count.
- Input dimensions.
- Input layout.
- Output dimensions.
- Output interpretation.
- Model-specific adapters.

Do not assume TransformerLOB has identical ONNX metadata to DeepLOB.

### Test procedure

1. Verify model file identity.
2. Verify actual input/output metadata.
3. Prepare identical model-compatible input tensors.
4. Execute inference in Rust.
5. Execute inference in C++.
6. Compare raw output scores.
7. Compare predicted class indices.
8. Generate a numerical comparison report.

Use the same tolerance methodology as DeepLOB, with model-specific findings documented separately.

### Acceptance criteria

- TransformerLOB model identity matches.
- Input contracts match.
- Model outputs are numerically equivalent within the justified tolerance.
- Predicted classes agree.
- No model-specific adapter errors remain unresolved.

---

## 9. Task 6 — Validate Prediction Consistency

Numerical similarity and classification agreement are related but different.

Two implementations may produce slightly different scores while still predicting the same class.

For every validation sample, compare:

| Field | Rust | C++ |
|---|---|---|
| Window index | Value | Value |
| Down score | Value | Value |
| Stationary score | Value | Value |
| Up score | Value | Value |
| Predicted class | Value | Value |
| Predicted direction | Value | Value |

Calculate:

\[
Agreement =
\frac{\text{Number of matching predictions}}
{\text{Total predictions}}\times100
\]

Target:

\[
Agreement = 100\%
\]

Investigate every disagreement.

### Special cases

Test:
- Scores with clear winning classes.
- Scores close to decision boundaries.
- Equal or nearly equal class scores.
- Very large finite scores.
- Very small finite scores.
- Invalid NaN or infinite values.

Use deterministic argmax and the same tie-breaking convention in both implementations.

Do not treat a prediction match as proof of numerical equivalence.

---

## 10. Task 7 — Full Dataset Validation

After validating individual tensors, validate the complete test dataset.

Use the same test split and preprocessing artifacts as the original C++ implementation.

The paper reports the FI-2010 held-out test evaluation using Days 9–10, with 9,000 samples. Verify the actual dataset split and sample count from the original project artifacts before treating those figures as the expected test set.

### Execution

```text
Same FI-2010 test dataset
          |
          v
     Rust Pipeline
          |
          v
    Rust Predictions

Same FI-2010 test dataset
          |
          v
      C++ Pipeline
          |
          v
    C++ Predictions
          |
          v
    Compare Results
```

Compare:

- Number of input rows.
- Number of windows.
- Number of successful inferences.
- Number of failed inferences.
- Ground-truth labels.
- Prediction order.
- Predicted class distribution.
- Per-sample class agreement.

### Acceptance criteria

- Both implementations process the same samples.
- Both generate the same number of windows.
- Predictions correspond to the same window indices.
- No samples are silently skipped.
- Any differences are explained.

---

## 11. Task 8 — Validate Classification Metrics

Calculate the same classification metrics for both implementations.

### Required metrics

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Macro precision.
- Macro recall.
- Macro F1.
- Weighted F1.
- Confusion matrix.
- Per-class recall.

### Confusion matrix

Use the same class order:

```text
                Predicted
              Down  Stat  Up

Actual Down
       Stat
       Up
```

### Reference DeepLOB results

The original paper reports:

| Actual class | Recall |
|---|---:|
| Down | 38.52% |
| Stationary | 59.35% |
| Up | 34.38% |

These are reference figures from the paper, not guaranteed acceptance values for a new run.

If your Rust and C++ predictions are identical and use the same labels, their classification metrics should also match.

If the metrics differ from the paper, investigate:
- Dataset split.
- Preprocessing.
- Label mapping.
- Model checkpoint.
- Window alignment.
- Evaluation implementation.

Do not alter predictions or labels to force agreement with the published results.

### Acceptance criteria

- Metrics are calculated using identical definitions.
- Confusion matrices match when predictions and labels match.
- Class-wise metrics match.
- Any deviation from published reference metrics is documented.

---

## 12. Task 9 — Validate End-to-End Inference

Validate the complete Rust application built in Phase 5.

Run:

```bash
lob-inference predict \
  --model deeplob \
  --input data/test.csv \
  --output outputs/rust_predictions.csv
```

Run the equivalent C++ executable with the same input and model.

Compare:

1. CSV input handling.
2. Preprocessing.
3. Window generation.
4. Model execution.
5. Prediction extraction.
6. Prediction ordering.
7. Output serialization.
8. Final prediction files.

### Verify output CSV

Check:
- Same number of rows.
- Same window indices.
- Same class mapping.
- Same predicted classes.
- Numerically equivalent scores.
- Correct label alignment.
- No missing predictions.

### Acceptance criteria

The Rust application must successfully process the actual test data and produce a prediction file that can be compared against C++ output.

---

## 13. Task 10 — Regression and Edge-Case Validation

Validate that the project remains stable beyond the normal execution path.

### Input validation

Test:
- Missing CSV.
- Empty CSV.
- Malformed rows.
- Missing features.
- Incorrect feature count.
- Invalid numerical values.
- Missing normalization files.
- Invalid tensor dimensions.

### Model validation

Test:
- Missing ONNX model.
- Invalid ONNX model.
- Wrong model metadata.
- Missing input.
- Missing output.
- Unsupported data types.

### Inference validation

Test:
- Repeated inference.
- Session reuse.
- Invalid model outputs.
- NaN values.
- Infinite values.
- Runtime execution failures.

### Output validation

Test:
- Invalid output path.
- Missing output directory.
- Existing output file.
- Interrupted output generation.
- Empty prediction results.

### Acceptance criteria

Errors must be descriptive and must not silently produce incorrect predictions.

---

## 14. Task 11 — Validate Performance Results from Phase 6

Re-run selected Phase 6 benchmarks to verify that the reported performance is reproducible.

For each model:

1. Use the same benchmark configuration.
2. Use the same hardware.
3. Use the same release build.
4. Use the same ONNX Runtime configuration.
5. Use the same input tensors.
6. Repeat benchmark trials.
7. Compare against the Phase 6 reports.

Review:

- Mean latency.
- P50.
- P95.
- P99.
- Standard deviation.
- Throughput.
- Memory consumption.
- CPU utilization.

Do not require identical timings.

Instead, document variation and investigate unusually large deviations.

### Acceptance criteria

- Benchmark methodology is repeatable.
- Configuration is preserved.
- Performance results are reproducible within a documented range.
- No unexplained regression is introduced.

---

## 15. Task 12 — Generate Final Validation Reports

Generate machine-readable reports in JSON and CSV.

Suggested structure:

```text
validation-results/
├── environment/
│   └── environment-manifest.json
│
├── deeplob/
│   ├── input-comparison.csv
│   ├── output-comparison.csv
│   ├── prediction-comparison.csv
│   ├── metrics-comparison.json
│   └── summary.json
│
├── transformerlob/
│   ├── input-comparison.csv
│   ├── output-comparison.csv
│   ├── prediction-comparison.csv
│   ├── metrics-comparison.json
│   └── summary.json
│
├── performance/
│   └── benchmark-revalidation.json
│
└── final-validation-report.md
```

### Final report contents

1. Project overview.
2. Validation environment.
3. Model artifact hashes.
4. Dataset and normalization artifact hashes.
5. Preprocessing comparison.
6. Tensor comparison.
7. DeepLOB numerical comparison.
8. TransformerLOB numerical comparison.
9. Prediction agreement.
10. Full-dataset evaluation.
11. Classification metrics.
12. End-to-end validation.
13. Performance revalidation.
14. Failed tests.
15. Known limitations.
16. Final acceptance status.

Preserve raw data and failed comparison samples for reproducibility.

Do not commit large datasets, model binaries, or generated artifacts unless the repository explicitly requires them.

---

## 16. Task 13 — Final Project Documentation

Update the README and project documentation.

Include:

### Architecture

Explain:

- Rust application architecture.
- ONNX Runtime integration.
- DeepLOB inference.
- TransformerLOB inference.
- Preprocessing pipeline.
- CLI architecture.
- Benchmarking architecture.

### Setup

Document:

- Rust installation.
- ONNX Runtime setup.
- Build instructions.
- Model placement.
- Dataset placement.
- Configuration.

### Usage

Provide working examples for:

```bash
lob-inference runtime-info

lob-inference inspect --model deeplob

lob-inference validate --model deeplob

lob-inference smoke-test --model deeplob

lob-inference predict \
  --model deeplob \
  --input data/test.csv \
  --output outputs/predictions.csv

lob-inference benchmark \
  --model deeplob \
  --iterations 200
```

### Results

Include:
- Actual Rust benchmark results.
- Actual C++ benchmark results.
- Latency comparison.
- Throughput comparison.
- Numerical equivalence results.
- Classification metrics.
- Known limitations.

Never insert fabricated results.

---

## 17. Git Workflow

### Branch

Create:

```bash
git switch main
git pull --ff-only
git switch -c test/equivalence-validation
```

Before switching:
- Inspect Git status.
- Preserve unrelated changes.
- Do not discard uncommitted user work.

Never work directly on `main`.

### Commit strategy

Use atomic Conventional Commits.

| Checkpoint | Example commit |
|---|---|
| Validation architecture | `test(validation): add validation module structure` |
| Environment manifest | `feat(validation): record runtime environment metadata` |
| Input comparison | `test(validation): compare Rust and C++ input tensors` |
| Numerical comparison | `test(validation): compare ONNX inference outputs` |
| Prediction comparison | `test(validation): validate prediction consistency` |
| Dataset validation | `test(validation): add full dataset validation` |
| Metrics comparison | `test(validation): compare classification metrics` |
| End-to-end validation | `test(validation): validate complete inference pipeline` |
| Performance revalidation | `test(validation): revalidate benchmark results` |
| Reports | `feat(validation): generate final validation reports` |
| Documentation | `docs(project): document final validation results` |

These are suggested checkpoints. Adapt them to the actual repository and keep commits buildable.

Before every commit:

```bash
git status
git diff
cargo fmt --check
cargo test
```

### Pull request

When complete:

1. Review the entire diff.
2. Run all quality gates.
3. Push `test/equivalence-validation`.
4. Open a PR against `main`.
5. Include the validation report.
6. Include actual test results.
7. Document any deviations.
8. Do not merge automatically.

---

## 18. Quality Gates

Run:

```bash
cargo fmt --check

cargo check

cargo test

cargo clippy --all-targets --all-features -- -D warnings

cargo build --release
```

Also verify:

- DeepLOB numerical validation succeeds.
- TransformerLOB numerical validation succeeds.
- Input tensor comparison succeeds.
- Prediction agreement is measured.
- Full-dataset inference works.
- Classification metrics are consistent.
- End-to-end CSV comparison works.
- Phase 6 benchmark revalidation completes.
- All validation reports are generated.
- Phase 1–6 functionality remains intact.
- No model files or datasets are accidentally committed.

---

## 19. Phase 7 Acceptance Criteria

### Environment and artifacts

- [ ] Rust and C++ environments are documented.
- [ ] ONNX Runtime versions are verified.
- [ ] Model hashes match.
- [ ] Dataset hashes match.
- [ ] Normalization artifacts match.
- [ ] Runtime configuration is documented.

### Preprocessing

- [ ] CSV row counts match.
- [ ] Feature ordering matches.
- [ ] Normalization matches.
- [ ] Window generation matches.
- [ ] Tensor dimensions match.
- [ ] Tensor values match within the justified tolerance.

### DeepLOB

- [ ] Same ONNX model is used.
- [ ] Same input tensors are used.
- [ ] Output scores are compared.
- [ ] Numerical tolerance is satisfied or deviations are explained.
- [ ] Prediction classes agree.

### TransformerLOB

- [ ] Same ONNX model is used.
- [ ] Actual input/output metadata is verified.
- [ ] Same compatible input tensors are used.
- [ ] Output scores are compared.
- [ ] Numerical tolerance is satisfied or deviations are explained.
- [ ] Prediction classes agree.

### Full-system validation

- [ ] Full-dataset inference succeeds.
- [ ] Prediction counts match.
- [ ] Prediction ordering matches.
- [ ] Classification metrics match.
- [ ] End-to-end output comparison succeeds.
- [ ] Edge-case tests pass.
- [ ] Performance revalidation is completed.
- [ ] Final reports are generated.
- [ ] Documentation is complete.
- [ ] Quality gates pass.
- [ ] Changes are committed on `test/equivalence-validation`.
- [ ] Pull request against `main` is ready for review.

A failed comparison must be investigated and documented. Do not mark an acceptance criterion as passed merely because the test ran.

---

## 20. Instructions for Codex

You are implementing **Phase 7 only** of the Rust inference engine.

### Before implementation

1. Inspect the repository structure.
2. Read the completed Phase 1–6 implementations.
3. Inspect Git status and commit history.
4. Locate the original C++ implementation.
5. Inspect the Rust inference engines.
6. Inspect the original ONNX models.
7. Inspect the preprocessing artifacts.
8. Identify the exact dataset split used by C++.
9. Inspect the Phase 6 benchmark reports.
10. Prepare a validation plan before modifying files.

### Git requirements

- Create or switch to `test/equivalence-validation`.
- Do not include `phase-07` in the branch name.
- Never work directly on `main`.
- Preserve unrelated changes.
- Use atomic Conventional Commits.
- Do not force-push.
- Do not automatically merge the PR.

### Implementation requirements

- Use C++ as the reference implementation.
- Compare actual input tensors.
- Compare actual ONNX output scores.
- Compare predictions.
- Validate both models.
- Validate full-dataset execution.
- Validate classification metrics.
- Validate end-to-end inference.
- Revalidate selected Phase 6 benchmarks.
- Generate reproducible reports.
- Preserve failed comparison samples.
- Do not change preprocessing or model behavior to force matching results.

### Strict exclusions

Do not implement:

- New models.
- Model retraining.
- ONNX export changes.
- GPU execution.
- Candle, Burn, or tch.
- New preprocessing algorithms.
- Unverified optimizations.
- Changes to the original C++ engine's core behavior.

### Completion report

At completion, provide:

1. Final project architecture.
2. Environment comparison.
3. Model artifact hashes.
4. Dataset and normalization artifact hashes.
5. Preprocessing equivalence results.
6. DeepLOB numerical comparison.
7. TransformerLOB numerical comparison.
8. Prediction agreement percentages.
9. Full-dataset evaluation results.
10. Classification metrics.
11. End-to-end comparison results.
12. Performance revalidation results.
13. Generated validation reports.
14. Failed tests and known limitations.
15. Quality-gate results.
16. Git branch name.
17. Commit hashes and messages.
18. Pull request status.

**Never claim numerical equivalence without comparing actual model outputs from both implementations using identical inputs.**

The final objective is to establish that the Rust inference engine is a validated alternative to the original C++ engine, with reproducible evidence of correctness and performance.
