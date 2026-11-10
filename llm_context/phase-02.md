# PHASE 02 — Tensor Construction & Preprocessing Pipeline

**Project:** RustLOB  
**Phase:** 02 of 07  
**Language:** Rust  
**Previous Phase:** Project Foundation & ONNX Runtime Integration  
**Git Branch:** `feat/tensor-preprocessing`  
**Status:** Not Started

---

# 1. Phase Objective

Build a complete, reusable, memory-efficient preprocessing and tensor construction pipeline in Rust.

The purpose of this phase is to transform raw FI-2010 Limit Order Book (LOB) data into correctly normalized, structured tensors that can be consumed directly by the ONNX Runtime inference engine.

The pipeline must reproduce the preprocessing methodology used in the reference DeepLOB-HFT-Cpp implementation.

By the end of this phase, RustLOB must support:

1. Loading raw FI-2010 CSV data.
2. Extracting the 40 LOB features.
3. Loading precomputed normalization parameters.
4. Applying feature-wise Z-score normalization.
5. Extracting non-overlapping 100-timestep windows.
6. Preserving the original chronological ordering.
7. Extracting and encoding ground-truth labels.
8. Constructing contiguous tensor buffers.
9. Validating tensor dimensions and values.
10. Producing inference-ready input tensors.

**Important:** This phase implements preprocessing and tensor construction only. Actual ONNX model execution belongs to Phase 3 and Phase 4.

# 2. Reference Specifications

Follow the preprocessing methodology described in the DeepLOB-HFT-Cpp research paper.

### Dataset specifications

| Property | Specification |
|---|---|
| Dataset | FI-2010 |
| Number of instruments | 5 |
| Trading days | 10 |
| Features per snapshot | 40 |
| Prediction horizon | k = 10 |
| Sequence length | 100 |
| Window extraction | Non-overlapping |
| Number of classes | 3 |
| Normalization | Z-score |
| Normalization statistics | Training partition only |

### Input tensor specification

```text
[1, 1, 100, 40]
```

Where:

- Batch size = 1
- Channels = 1
- Time steps = 100
- Features = 40

### Feature specification

Each LOB snapshot contains 40 features representing:

- Bid prices.
- Bid volumes.
- Ask prices.
- Ask volumes.

These features represent ten levels of the Limit Order Book.

Preserve the exact original feature ordering.

### Label mapping

```text
Original FI-2010 label:
1 = Down
2 = Stationary
3 = Up

Internal label:
0 = Down
1 = Stationary
2 = Up
```

The label associated with each window must be taken from the final row of that window.

Do not include labels in the model input tensor.

---

# 3. Git Workflow — Mandatory

## Branch creation

Create the following branch:

```bash
git switch main
git pull origin main
git switch -c feat/tensor-preprocessing
```

Before switching branches:

- Inspect the current Git status.
- Preserve all uncommitted changes.
- Ensure Phase 1 has been integrated into main.
- Never overwrite unrelated changes.

Do not use `phase-02` or any phase number in the branch name.

## Commit convention

Follow Conventional Commits.

Format:

```text
<type>(<scope>): <description>
```

Suggested commit checkpoints:

| Milestone | Commit message |
|---|---|
| Preprocessing architecture | `feat(preprocessing): add preprocessing module structure` |
| CSV loading | `feat(data): implement FI-2010 CSV loader` |
| Feature extraction | `feat(data): extract LOB feature columns` |
| Normalization loading | `feat(preprocessing): load normalization parameters` |
| Z-score normalization | `feat(preprocessing): implement feature normalization` |
| Window extraction | `feat(preprocessing): implement sliding window extraction` |
| Label handling | `feat(data): implement label encoding and extraction` |
| Tensor construction | `feat(tensor): implement contiguous inference tensors` |
| Validation | `test(preprocessing): add pipeline validation tests` |
| Documentation | `docs(preprocessing): document data pipeline` |

Create commits after meaningful, tested milestones. Closely related tasks may share a commit.

Never create empty commits or commit broken functionality.

Before every commit:

```bash
git status
git diff
cargo fmt --check
cargo test
```

Stage only relevant files.

At the end, prepare a PR targeting `main`. Do not merge automatically.

---

# 4. Proposed Architecture

Extend the existing RustLOB project structure.

```text
rust-lob/
│
├── Cargo.toml
├── Cargo.lock
│
├── configs/
│   └── default.toml
│
├── data/
│   ├── raw/
│   └── normalization/
│       ├── mu.npy
│       └── sigma.npy
│
├── src/
│   │
│   ├── data/
│   │   ├── mod.rs
│   │   ├── csv_loader.rs
│   │   ├── feature_extractor.rs
│   │   └── labels.rs
│   │
│   ├── preprocessing/
│   │   ├── mod.rs
│   │   ├── normalization.rs
│   │   ├── window.rs
│   │   └── pipeline.rs
│   │
│   ├── tensor/
│   │   ├── mod.rs
│   │   ├── tensor.rs
│   │   ├── shape.rs
│   │   └── validation.rs
│   │
│   ├── runtime/
│   │   └── ...
│   │
│   ├── models/
│   │   └── ...
│   │
│   └── ...
│
├── tests/
│   ├── csv_loader_tests.rs
│   ├── feature_extraction_tests.rs
│   ├── normalization_tests.rs
│   ├── window_tests.rs
│   ├── tensor_tests.rs
│   └── pipeline_tests.rs
│
└── examples/
    └── preprocessing_demo.rs
```

The architecture should be modular, with independent components for loading, feature extraction, normalization, window extraction, and tensor construction.

Avoid putting the entire preprocessing pipeline in one file.

---

# 5. Detailed Implementation Tasks

## Task 1 — Preprocessing Architecture

Create the preprocessing and tensor modules.

Responsibilities:

- Define shared preprocessing data structures.
- Establish input and output contracts.
- Define error types.
- Establish module boundaries.
- Integrate the new modules with the existing Phase 1 application.

Design the pipeline around the following conceptual flow:

```text
Raw FI-2010 CSV
       |
       v
CSV Data Loader
       |
       v
Feature Extraction
       |
       v
Label Extraction
       |
       v
Normalization Parameters
       |
       v
Z-Score Normalization
       |
       v
100-Timestep Window Extraction
       |
       v
Tensor Construction
       |
       v
Tensor Validation
       |
       v
Inference-Ready Data
```

Define reusable data structures.

Conceptually:

```rust
pub struct LobDataset {
    pub features: Vec<f32>,
    pub labels: Vec<u8>,
    pub rows: usize,
}

pub struct NormalizationStats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

pub struct LobWindow {
    pub features: Vec<f32>,
    pub label: u8,
}

pub struct InputTensor {
    pub data: Vec<f32>,
    pub shape: [usize; 4],
}
```

These are conceptual structures. Adapt ownership and memory representation when necessary.

Avoid unnecessary cloning of large datasets.

Acceptance criteria:

- Module structure established.
- Shared data structures implemented.
- Modules integrate with existing application.
- Unit tests for basic data structures pass.

## Task 2 — FI-2010 CSV Loader

Create:

`src/data/csv_loader.rs`

Implement a robust CSV loading component.

Responsibilities:

1. Accept a configurable dataset path.
2. Open CSV files.
3. Parse numeric values.
4. Validate row structure.
5. Handle malformed records.
6. Detect missing or invalid values.
7. Report row counts.
8. Preserve original row ordering.
9. Return structured data.

### Dataset column handling

The reference paper identifies:

- Columns 0–39: 40 LOB features.
- Column 144: ternary movement label.

Do not assume this layout without inspecting the actual dataset files.

First inspect the provided FI-2010 data format and verify:

- Whether files have headers.
- Whether columns are zero-indexed as expected.
- Whether labels are stored in the same CSV or separate files.
- Whether there are additional target columns.
- Whether multiple instrument/day files need to be combined.

Implement configurable column mappings if necessary.

Do not silently skip malformed rows.

Provide useful errors containing file path and row information.

### Required functionality

Conceptually:

```rust
pub fn load_csv(path: &Path) -> Result<LobDataset>;
```

The actual interface may be adjusted to preserve labels and raw column information correctly.

Acceptance criteria:

- Valid FI-2010 files load successfully.
- Feature count is validated.
- Row order is preserved.
- Malformed rows are handled explicitly.
- Missing files produce descriptive errors.

## Task 3 — Feature Extraction

Create:

`src/data/feature_extractor.rs`

Extract exactly 40 LOB features from every snapshot.

Responsibilities:

- Extract the configured feature columns.
- Preserve the original feature order.
- Convert values into the chosen floating-point representation.
- Validate feature count.
- Detect NaN and infinity.
- Preserve chronological ordering.

Expected representation:

```text
Rows × 40 features
```

The output must be suitable for feature-wise normalization.

Important:

Do not calculate additional market microstructure indicators.

Do not introduce OFI, spread, imbalance, or other derived features.

The purpose is to reproduce the original 40-feature input representation.

Acceptance criteria:

- Exactly 40 features per row.
- Correct feature ordering.
- No unexpected data transformations.
- Numerical values validated.

## Task 4 — Normalization Parameter Loader

Create:

`src/preprocessing/normalization.rs`

Load the precomputed normalization parameters:

```text
data/normalization/
├── mu.npy
└── sigma.npy
```

Expected dimensions:

```text
mu:    [40]
sigma: [40]
```

Responsibilities:

- Load mean values.
- Load standard deviation values.
- Validate vector dimensions.
- Validate numeric values.
- Detect NaN and infinity.
- Validate standard deviations.
- Provide descriptive errors.

### Important data integrity rule

The normalization statistics must come exclusively from the training partition.

Never calculate new normalization statistics from validation or test data.

Do not silently recompute missing normalization parameters.

If the original normalization files are unavailable, report the dependency clearly.

For testing, synthetic normalization parameters may be generated as fixtures, but they must never be presented as the actual FI-2010 training statistics.

Acceptance criteria:

- Both parameter vectors load successfully.
- Dimensions are validated.
- Invalid statistics are rejected.
- Statistics can be reused across all model inputs.

## Task 5 — Z-Score Normalization

Implement feature-wise normalization.

Formula:

```text
X_norm(j) = (X(j) - mu(j)) / (sigma(j) + epsilon)
```

Where:

```text
epsilon = 1e-8
```

For every row:

- Normalize feature 0 using mean 0 and standard deviation 0.
- Normalize feature 1 using mean 1 and standard deviation 1.
- Continue through all 40 features.

Use the same normalization statistics for all datasets.

### Requirements

- Preserve row ordering.
- Preserve feature ordering.
- Avoid unnecessary memory allocation.
- Avoid recomputing normalization parameters.
- Detect invalid normalized outputs.
- Support processing large datasets.

Use a reusable normalizer rather than duplicating logic across model implementations.

Conceptual interface:

```rust
pub struct Normalizer {
    stats: NormalizationStats,
}

impl Normalizer {
    pub fn normalize(&self, dataset: &mut LobDataset) -> Result<()>;
}
```

Adapt this interface to the final data representation.

Acceptance criteria:

- Formula matches the reference.
- Epsilon is exactly 1e-8.
- Normalization is feature-wise.
- No data leakage.
- Numerical tests pass.

## Task 6 — Sliding Window Extraction

Create:

`src/preprocessing/window.rs`

Implement non-overlapping sliding-window extraction.

Reference specification:

```text
Window length = 100
Stride = 100
```

Given a sequence of 250 rows:

```text
Window 1: rows 0–99
Window 2: rows 100–199
Remaining rows: 200–249
```

The final incomplete window must be discarded.

Do not generate overlapping windows.

### Window requirements

Each window contains:

```text
100 timesteps × 40 features
```

The associated label must be taken from the final row of the corresponding window.

Conceptual structure:

```rust
pub struct WindowConfig {
    pub sequence_length: usize,
    pub stride: usize,
}
```

Default:

```rust
WindowConfig {
    sequence_length: 100,
    stride: 100,
}
```

### Important

Preserve chronological ordering.

Do not shuffle the windows.

Do not combine windows across different trading-day or instrument boundaries unless the reference data preparation explicitly establishes that they form one continuous sequence.

Verify the dataset segmentation before extraction.

The pipeline should support separate dataset segments so that windows cannot accidentally cross boundaries.

### Acceptance criteria

- Correct non-overlapping windows.
- Correct stride.
- Incomplete trailing windows discarded.
- Correct final-row labels.
- Chronological ordering preserved.
- No cross-segment windows.

## Task 7 — Label Extraction and Encoding

Create:

`src/data/labels.rs`

Implement label extraction and encoding.

Original labels:

```text
1 = Down
2 = Stationary
3 = Up
```

Internal labels:

```text
0 = Down
1 = Stationary
2 = Up
```

Requirements:

- Extract the correct label column.
- Validate supported label values.
- Convert original labels into internal labels.
- Associate the label with the final row of each window.
- Preserve label alignment with extracted windows.
- Reject invalid or missing labels.

Do not accidentally use labels as model input features.

Provide a reverse mapping for readable CLI output if needed.

Acceptance criteria:

- Correct class mapping.
- Invalid labels rejected.
- Window labels correctly aligned.
- No label leakage into input tensors.

## Task 8 — Tensor Construction

Create:

`src/tensor/tensor.rs`

This is the most important integration task in Phase 2.

Convert the normalized 100 × 40 window into the required four-dimensional input tensor.

Expected shape:

```text
[1, 1, 100, 40]
```

Expected memory layout:

```text
Batch
  └── Channel
        └── Time
              └── Features
```

For a single input window:

```text
100 × 40 = 4,000 floating-point values
```

The tensor should use a contiguous row-major buffer.

Conceptual structure:

```rust
pub struct InputTensor {
    pub data: Vec<f32>,
    pub shape: [usize; 4],
}
```

The tensor should contain exactly 4,000 values.

### Memory requirements

- Avoid unnecessary copies.
- Avoid unnecessary allocations.
- Use contiguous memory.
- Ensure the data lifetime is compatible with ONNX Runtime.
- Keep tensor construction independent from model execution.
- Do not introduce unsafe code unless necessary and justified.

### Multiple windows

The initial implementation must support batch-size-one tensors.

Do not introduce dynamic batching in this phase.

However, design the tensor abstraction so that future batching can be added without rewriting the entire preprocessing pipeline.

Acceptance criteria:

- Correct shape.
- Exactly 4,000 values.
- Correct row-major ordering.
- Contiguous buffer.
- Correct floating-point representation.

## Task 9 — Tensor Validation

Create:

`src/tensor/validation.rs`

Validate every tensor before it is passed to the inference engine.

Checks:

1. Expected dimensions.
2. Expected number of values.
3. Correct batch dimension.
4. Correct channel dimension.
5. Correct timestep count.
6. Correct feature count.
7. No NaN values.
8. No infinite values.
9. Valid buffer length.
10. Correct memory layout.

Expected shape:

```text
[1, 1, 100, 40]
```

Expected number of values:

```text
4000
```

Validation must produce descriptive errors.

Avoid expensive repeated validation inside the future latency-critical inference loop. Design it so validation can be performed at the preprocessing/inference boundary.

Acceptance criteria:

- Invalid tensors rejected.
- Valid tensors accepted.
- All structural checks tested.

## Task 10 — Unified Preprocessing Pipeline

Create:

`src/preprocessing/pipeline.rs`

Integrate all the previous components.

The pipeline should provide a single reusable entry point.

Conceptual flow:

```text
Dataset Path
     |
     v
CSV Loader
     |
     v
Feature Extraction
     |
     v
Label Extraction
     |
     v
Normalization Loader
     |
     v
Z-Score Normalizer
     |
     v
Window Extractor
     |
     v
Tensor Constructor
     |
     v
Tensor Validator
     |
     v
Inference-Ready Windows
```

The pipeline must not execute ONNX inference.

Provide reusable APIs for:

- Loading a dataset.
- Preparing normalized features.
- Extracting windows.
- Creating a tensor for a selected window.
- Retrieving the corresponding ground-truth label.

Do not force the entire dataset to be converted into thousands of independent tensor allocations upfront.

Prefer a design that stores normalized data efficiently and constructs inference tensors as required.

Acceptance criteria:

- All modules work together.
- Output is inference-ready.
- Labels remain aligned.
- Preprocessing is deterministic.
- No unnecessary data copies.

---

# 6. Configuration Updates

Extend `configs/default.toml`.

Suggested configuration:

```toml
[data]
dataset_path = "data/raw"
feature_count = 40
label_column = 144

[preprocessing]
sequence_length = 100
stride = 100
epsilon = 0.00000001

[normalization]
mean_path = "data/normalization/mu.npy"
std_path = "data/normalization/sigma.npy"

[tensor]
batch_size = 1
channels = 1
time_steps = 100
features = 40
dtype = "float32"
```

The label column must be configurable and verified against the actual dataset format.

Avoid hardcoding dataset paths throughout the codebase.

Validate configuration before starting preprocessing.

---

# 7. Testing Strategy

Implement comprehensive tests.

## CSV loader tests

- Valid CSV loading.
- Missing file.
- Empty CSV.
- Malformed records.
- Incorrect column count.
- Invalid numeric values.
- Row-order preservation.

## Feature extraction tests

- Exactly 40 features.
- Correct column selection.
- Correct ordering.
- NaN detection.
- Infinity detection.

## Normalization tests

Use synthetic test data.

Example:

```text
Input:
[10, 20]

Mean:
[5, 10]

Standard deviation:
[5, 10]
```

Expected normalized values should be approximately:

```text
[1.0, 1.0]
```

Also test:

- Zero standard deviation.
- Invalid standard deviation.
- NaN statistics.
- Infinite statistics.
- Incorrect vector dimensions.

## Window extraction tests

Test:

| Input rows | Expected windows |
|---:|---:|
| 50 | 0 |
| 100 | 1 |
| 150 | 1 |
| 200 | 2 |
| 250 | 2 |
| 300 | 3 |

Verify:

- Non-overlapping windows.
- Correct stride.
- Final incomplete window discarded.
- Correct label extraction.
- No cross-segment windows.

## Tensor tests

Verify:

- Shape `[1, 1, 100, 40]`.
- Buffer length 4000.
- Correct feature ordering.
- Correct row ordering.
- No NaN or infinite values.
- Correct contiguous layout.

## Pipeline integration tests

Create a small deterministic synthetic CSV fixture.

Run the entire pipeline and verify:

- CSV loading.
- Feature extraction.
- Label extraction.
- Normalization.
- Window extraction.
- Tensor construction.
- Tensor validation.

The tests must not require the real FI-2010 dataset or actual ONNX model files.

---

# 8. Performance and Memory Requirements

The preprocessing pipeline must be designed with low-latency inference in mind.

Requirements:

1. Avoid repeated parsing of the same CSV files.
2. Load normalization parameters once.
3. Reuse normalization statistics.
4. Avoid unnecessary cloning.
5. Prefer contiguous memory.
6. Avoid creating a separate allocation for every intermediate operation.
7. Keep preprocessing separate from model execution.
8. Support processing large datasets.
9. Preserve deterministic ordering.
10. Avoid premature parallelization.

Do not claim zero-copy operation unless the implementation actually achieves it.

Do not optimize based on assumptions. Measure performance in later benchmarking work.

---

# 9. Scope Restrictions

The following are out of scope:

- ONNX model execution.
- DeepLOB inference.
- TransformerLOB inference.
- Model training.
- ONNX export.
- Latency benchmarking.
- Throughput benchmarking.
- Memory profiling.
- Statistical performance analysis.
- C++ versus Rust comparison.
- GPU execution.
- Dynamic batching.
- Candle or Burn integration.

Phase 2 ends when correctly constructed inference-ready tensors can be produced and validated.

---

# 10. Phase 2 Definition of Done

### Dataset

- [ ] FI-2010 CSV loader implemented.
- [ ] Dataset format verified.
- [ ] 40 features extracted.
- [ ] Correct label column identified.
- [ ] Row ordering preserved.
- [ ] Invalid data handled.

### Normalization

- [ ] `mu.npy` loader implemented.
- [ ] `sigma.npy` loader implemented.
- [ ] Vector dimensions validated.
- [ ] Z-score normalization implemented.
- [ ] Epsilon set to 1e-8.
- [ ] Training-only statistics preserved.

### Window extraction

- [ ] Sequence length 100.
- [ ] Stride 100.
- [ ] Non-overlapping windows.
- [ ] Incomplete windows discarded.
- [ ] Correct final-row label.
- [ ] Segment boundaries preserved.

### Tensor construction

- [ ] Shape `[1, 1, 100, 40]`.
- [ ] Buffer length 4000.
- [ ] Contiguous row-major layout.
- [ ] Float32 representation.
- [ ] Tensor validation implemented.

### Integration

- [ ] Unified preprocessing pipeline.
- [ ] Configuration updated.
- [ ] Unit tests pass.
- [ ] Integration tests pass.
- [ ] Documentation updated.

### Git

- [ ] Correct feature branch created.
- [ ] Atomic commits created.
- [ ] No datasets or generated artifacts committed.
- [ ] Branch diff reviewed.
- [ ] Pull request prepared.
- [ ] No automatic merge.

---

# 11. Required Final Report from Codex

After completing this phase, provide:

1. Architecture overview.
2. Modules created and modified.
3. Dependency changes.
4. Dataset format and column mapping discovered.
5. Normalization implementation details.
6. Window extraction implementation.
7. Tensor memory layout explanation.
8. Sample tensor validation output.
9. Unit and integration test results.
10. Performance and memory observations, without unsupported claims.
11. Known limitations.
12. Completed acceptance checklist.
13. Git branch name.
14. Commit hashes and commit descriptions.
15. Pull request summary.

Never fabricate results.

If the real FI-2010 dataset or normalization files are unavailable, report the missing assets and use synthetic fixtures only for isolated tests.

---

# 12. Instructions to the AI Coding Agent

You are acting as a senior Rust systems engineer working on RustLOB.

Read the following documents before implementation:

- `PROJECT_CONTEXT.md`
- `PHASE_01.md`
- `PHASE_02.md`

Treat `PROJECT_CONTEXT.md` as the overall architecture specification and this file as the detailed implementation specification for the current phase.

### Mandatory execution sequence

1. Inspect the existing repository.
2. Review the Git status and current branch.
3. Verify Phase 1 has been integrated into main.
4. Create the branch `feat/tensor-preprocessing`.
5. Inspect the actual FI-2010 dataset format if available.
6. Verify the existing ONNX input specifications.
7. Review the current Rust module structure.
8. Propose the preprocessing architecture.
9. Implement one logical component at a time.
10. Write tests alongside implementation.
11. Run formatting, linting, and tests.
12. Create meaningful Git commits.
13. Update documentation.
14. Review the complete branch diff.
15. Prepare a PR against main without merging.

### Critical restrictions

- Do not implement Phase 3 or later.
- Do not modify the existing ONNX model files.
- Do not retrain models.
- Do not change the 40-feature representation.
- Do not change the sequence length of 100.
- Do not change the non-overlapping window methodology.
- Do not introduce data leakage.
- Do not shuffle the dataset.
- Do not fabricate normalization statistics.
- Do not silently skip malformed data.
- Do not overengineer the pipeline.
- Do not overwrite unrelated user changes.

### Initial instruction

Start by inspecting the repository, Phase 1 implementation, dataset availability, and Git history.

Then prepare a detailed implementation plan for the preprocessing pipeline.

Implement Phase 2 incrementally, validate each component, and commit after meaningful milestones.

Do not begin Phase 3.

**The final objective is a deterministic, memory-efficient Rust preprocessing pipeline that produces exactly compatible `[1, 1, 100, 40]` inference tensors for the DeepLOB and TransformerLOB ONNX models.**
