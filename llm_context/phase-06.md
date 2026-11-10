# PHASE_06.md — Benchmarking and Performance Profiling

## 1. Phase Overview

**Phase:** 06 — Benchmarking and Performance Profiling  
**Branch:** `feat/benchmarking`  
**Objective:** Build a reproducible benchmarking system to measure the performance of the Rust DeepLOB and TransformerLOB inference engines and compare Rust against the original C++ inference engine.

### Project context

The project is a Rust alternative to the existing `DeepLOB-HFT-Cpp` implementation.

The existing system uses ONNX Runtime for model execution.

The Rust implementation uses:
- Rust stable
- Cargo
- `ort` crate
- ONNX Runtime
- CPU Execution Provider
- DeepLOB ONNX model
- TransformerLOB ONNX model

### Reference benchmark from the original paper

The original C++ implementation reports:

| Metric | C++ reference |
|---|---:|
| Mean inference latency | 1.159 ms/window |
| Throughput | 917 windows/second |
| Warm-up iterations | 20 |
| Timed iterations | 200 |

The paper uses `std::chrono::high_resolution_clock` for timing and reports mean, standard deviation, and P99 latency.

These figures are reference results. The Rust implementation must be benchmarked independently under documented conditions before making any performance claims.

### Phase 6 scope

**In scope:**
- Formal benchmarking infrastructure.
- DeepLOB benchmarking.
- TransformerLOB benchmarking.
- Warm-up and timed execution.
- Latency statistics.
- Throughput measurements.
- Memory and CPU profiling.
- Preprocessing versus inference timing.
- End-to-end performance measurements.
- Rust versus C++ comparison.
- Benchmark report generation.
- Reproducibility and test coverage.

**Out of scope:**
- Numerical equivalence validation — Phase 7.
- Model training or export.
- GPU inference.
- Candle, Burn, or tch.
- Changing model architecture.
- Changing preprocessing semantics.
- Rewriting the C++ engine.
- Introducing new inference backends.

---

## 2. Benchmarking Objectives

The benchmark system must answer these questions:

1. How fast is Rust DeepLOB inference?
2. How fast is Rust TransformerLOB inference?
3. What is the average latency per window?
4. What are the P50, P90, P95, and P99 latency values?
5. What throughput can Rust sustain?
6. How much memory does the inference process consume?
7. What is the CPU utilization during inference?
8. How much time is spent in preprocessing?
9. How much time is spent inside ONNX Runtime?
10. How does Rust compare with the original C++ engine?
11. Does performance remain consistent across repeated benchmark runs?

The benchmark must separate model execution performance from preprocessing and end-to-end application performance.

---

## 3. Benchmarking Architecture

Extend the Phase 5 benchmark command rather than creating an unrelated benchmarking application.

Suggested structure:

```text
src/
├── benchmark/
│   ├── mod.rs
│   ├── config.rs
│   ├── runner.rs
│   ├── warmup.rs
│   ├── timer.rs
│   ├── statistics.rs
│   ├── latency.rs
│   ├── throughput.rs
│   ├── memory.rs
│   ├── cpu.rs
│   ├── comparison.rs
│   ├── report.rs
│   └── error.rs
│
├── cli/
│   └── handlers/
│       └── benchmark.rs
│
├── application/
├── runtime/
├── models/
├── inference/
├── preprocessing/
└── tensor/
```

This is a proposed structure. Follow existing repository conventions and reuse existing code.

### Module responsibilities

| Module | Responsibility |
|---|---|
| `config.rs` | Benchmark configuration |
| `runner.rs` | Benchmark orchestration |
| `warmup.rs` | Warm-up execution |
| `timer.rs` | High-resolution elapsed-time measurement |
| `statistics.rs` | Statistical calculations |
| `latency.rs` | Per-inference latency measurements |
| `throughput.rs` | Windows-per-second calculations |
| `memory.rs` | Memory measurement integration |
| `cpu.rs` | CPU utilization measurement integration |
| `comparison.rs` | Rust/C++ result comparison |
| `report.rs` | JSON and CSV report generation |
| `error.rs` | Benchmark-specific errors |

Do not duplicate inference engines or ONNX Runtime session management.

---

## 4. Task 1 — Define Benchmark Configuration

Create a reusable benchmark configuration.

Suggested conceptual structure:

```rust
pub struct BenchmarkConfig {
    pub model: ModelType,
    pub warmup_iterations: usize,
    pub measured_iterations: usize,
    pub trials: usize,
    pub execution_provider: ExecutionProvider,
    pub measure_preprocessing: bool,
    pub measure_memory: bool,
    pub measure_cpu: bool,
}
```

Adapt this to the existing project.

### Suggested defaults

| Parameter | Default |
|---|---:|
| Warm-up iterations | 20 |
| Timed iterations | 200 |
| Independent trials | 5 |
| Execution provider | CPU |
| Model loading in inference timer | No |
| Preprocessing in inference timer | No |

The 20 warm-ups and 200 timed iterations match the original paper's reported C++ benchmark protocol.

The additional independent trials are intended to improve the reliability of the Rust measurements.

Allow users to override benchmark settings through CLI arguments.

Validate:
- Iterations must be greater than zero.
- Warm-up count must not be negative.
- Trial count must be greater than zero.
- Model must be valid.
- Input data must be available when required.
- Execution provider must be supported.

---

## 5. Task 2 — Design Benchmark Categories

Implement separate benchmark modes.

### A. Model Loading Benchmark

Measure the time required to load an ONNX model and initialize its runtime session.

Measure separately:

- Runtime initialization.
- Model loading.
- Session creation.
- Metadata validation.

Do not include model loading in inference latency.

Report model initialization time independently.

### B. Pure Inference Benchmark

This is the most important benchmark for comparing Rust and C++ inference engines.

Measure only the model inference operation.

Execution:

```text
Load model once
      |
Prepare input tensors
      |
Warm-up inference
      |
Timed inference
      |
Calculate statistics
```

Requirements:

- Reuse one loaded model session.
- Reuse valid input tensors.
- Do not reload the model.
- Do not perform CSV parsing inside the timed section.
- Do not perform normalization inside the timed section.
- Do not write output files inside the timed section.
- Measure individual inference calls.

This benchmark measures inference latency per window.

### C. Preprocessing Benchmark

Measure:

- CSV loading.
- Feature extraction.
- Normalization.
- Window generation.
- Tensor construction.

Report each stage separately where practical.

Do not include ONNX inference in preprocessing measurements.

### D. End-to-End Benchmark

Measure the complete pipeline:

```text
CSV Input
    |
    v
Preprocessing
    |
    v
Tensor Construction
    |
    v
ONNX Inference
    |
    v
Prediction Collection
    |
    v
Output Generation
```

Report total execution time and end-to-end throughput.

### E. Memory and CPU Benchmark

Measure resource consumption while running inference.

Report:
- Peak resident memory, where supported.
- Average or sampled resident memory, where supported.
- CPU utilization.
- Runtime process resource usage.

Clearly document the measurement mechanism and platform.

Do not claim exact memory or CPU measurements when the operating system does not expose reliable values.

---

## 6. Task 3 — Implement Warm-Up Execution

Implement a dedicated warm-up stage.

Default:

```text
Warm-up iterations: 20
```

Warm-up execution must occur before timed measurements.

Purpose:
- Allow runtime initialization to complete.
- Stabilize CPU execution behavior.
- Reduce first-run effects.
- Allow relevant caches to warm up.

Requirements:

1. Load the model before warm-up.
2. Prepare the input tensor before warm-up.
3. Execute the configured number of warm-up inferences.
4. Exclude warm-up measurements from final latency statistics.
5. Validate warm-up outputs.
6. Report warm-up configuration.

Do not include session creation or model loading in warm-up inference latency.

If warm-up inference fails, abort the benchmark with a descriptive error.

---

## 7. Task 4 — Implement High-Resolution Timing

Use a monotonic timing source such as Rust's `std::time::Instant`.

Conceptual implementation:

```rust
let start = Instant::now();

let prediction = engine.predict(&input)?;

let elapsed = start.elapsed();
```

Record individual inference durations.

Use nanoseconds internally where practical and convert to milliseconds for reporting.

Requirements:

- Timer must measure the intended operation only.
- Avoid logging inside the timed section.
- Avoid unnecessary allocations inside the timed section.
- Do not include CSV writing in pure inference timing.
- Do not include model loading in pure inference timing.
- Do not use wall-clock timestamps for elapsed-duration measurements.

Document timer overhead and the limitations of very short measurements.

---

## 8. Task 5 — Implement Latency Statistics

Calculate the following metrics:

| Metric | Meaning |
|---|---|
| Mean | Average latency |
| Median / P50 | Middle latency |
| P90 | 90th percentile |
| P95 | 95th percentile |
| P99 | 99th percentile |
| Minimum | Lowest observed latency |
| Maximum | Highest observed latency |
| Standard deviation | Variation in latency |

### Calculations

For measured latency values:

\[
\text{Mean} = \frac{1}{N}\sum_{i=1}^{N}t_i
\]

For population standard deviation:

\[
\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(t_i-\mu)^2}
\]

Document whether standard deviation is population or sample standard deviation. Use the same convention for Rust and C++.

### Percentile calculation

Choose one percentile calculation method and document it.

Use the same method for all compared implementations.

Do not silently mix percentile definitions between reports.

### Important

P99 is a tail-latency metric. A small number of outliers can influence it substantially.

With only 200 timed iterations, P99 is based on very few observations. Report the sample count alongside percentiles and consider larger iteration counts for stronger tail-latency conclusions.

---

## 9. Task 6 — Implement Throughput Measurement

Calculate throughput using:

\[
\text{Throughput} =
\frac{\text{Number of completed windows}}
{\text{Total elapsed seconds}}
\]

Report windows per second.

For example:

```text
Completed windows: 200
Elapsed time: 0.24 seconds

Throughput: 833.33 windows/second
```

Numbers above are illustrative.

For pure inference throughput:
- Exclude model loading.
- Exclude preprocessing.
- Exclude output file writing.

For end-to-end throughput:
- Include the complete processing pipeline.
- Include the actual work required to process all input windows.

Clearly label both measurements.

Do not calculate throughput by simply taking the reciprocal of average latency when the measurement protocol is not equivalent.

---

## 10. Task 7 — Implement Memory Profiling

Measure the inference process's memory consumption.

Suggested metrics:

- Memory before model loading.
- Memory after model loading.
- Memory after warm-up.
- Average sampled memory during inference.
- Peak resident set size.

Prefer operating-system process metrics such as resident set size (RSS).

On macOS, Linux, and Windows, memory measurement APIs may differ. Isolate platform-specific implementations behind a common interface.

Suggested conceptual API:

```rust
pub struct MemoryMetrics {
    pub baseline_bytes: Option<u64>,
    pub post_model_load_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
    pub average_rss_bytes: Option<u64>,
}
```

Requirements:

- Clearly identify unavailable measurements.
- Distinguish process RSS from model memory.
- Do not imply RSS is identical to actual model weight memory.
- Avoid sampling overhead that materially changes inference timing.

Memory profiling should be optional.

---

## 11. Task 8 — Implement CPU Profiling

Measure CPU utilization during benchmark execution.

Report:
- Average CPU utilization.
- Peak CPU utilization, where available.
- CPU thread configuration.
- Number of benchmark iterations.
- Execution provider.

Document the sampling interval and measurement mechanism.

Keep CPU monitoring outside the pure inference timer where possible.

Do not compare CPU utilization numbers obtained using different sampling methods without disclosing the difference.

---

## 12. Task 9 — Benchmark DeepLOB and TransformerLOB

Both models must be benchmarked independently.

Example:

```bash
lob-inference benchmark \
  --model deeplob \
  --warmup 20 \
  --iterations 200 \
  --trials 5
```

TransformerLOB:

```bash
lob-inference benchmark \
  --model transformerlob \
  --warmup 20 \
  --iterations 200 \
  --trials 5
```

The benchmark system must produce separate results for each model.

Do not assume the two models have equivalent inference costs.

Report the actual model name and model file used.

---

## 13. Task 10 — Establish Fair Rust vs C++ Comparison

This is one of the most important tasks in Phase 6.

The comparison must be apples-to-apples.

### A. Keep the following identical

| Factor | Rust | C++ |
|---|---|---|
| ONNX model | Same file | Same file |
| Model architecture | Same | Same |
| Input tensor | Same values | Same values |
| Input shape | Same | Same |
| Input type | float32 | float32 |
| Preprocessing | Same procedure | Same procedure |
| Normalization | Same parameters | Same parameters |
| Sequence length | 100 | 100 |
| Stride | 100 | 100 |
| ONNX Runtime version | Same | Same |
| Execution provider | CPU | CPU |
| Runtime optimizations | Same settings | Same settings |
| Thread configuration | Same | Same |
| Build mode | Release | Optimized release |

If exact equality of any setting is impossible, record the difference and avoid claiming a fully controlled comparison.

### B. Keep hardware conditions consistent

Run both engines on the same machine.

Record:
- CPU model.
- CPU architecture.
- Core count.
- RAM.
- Operating system.
- OS version.
- Rust version.
- C++ compiler and version.
- ONNX Runtime version.
- Runtime execution provider.
- Thread settings.

Avoid running unrelated CPU-intensive workloads during benchmark execution.

### C. Separate the measurements

Compare:

1. Model loading time.
2. Pure inference latency.
3. Inference throughput.
4. Preprocessing latency.
5. End-to-end throughput.
6. Memory consumption.
7. CPU utilization.

Do not compare Rust end-to-end latency against C++ pure inference latency.

### D. Keep C++ unchanged

Treat the original C++ engine as the reference implementation.

Do not modify its core behavior to improve the Rust comparison.

If changes are necessary to make the benchmark protocol equivalent, document them separately and preserve the original reference.

---

## 14. Task 11 — Implement Rust vs C++ Comparison Reports

Create a comparison module that accepts benchmark results from both implementations.

Suggested report:

| Metric | C++ | Rust | Difference |
|---|---:|---:|---:|
| Mean latency | Measured | Measured | Calculated |
| P50 latency | Measured | Measured | Calculated |
| P95 latency | Measured | Measured | Calculated |
| P99 latency | Measured | Measured | Calculated |
| Throughput | Measured | Measured | Calculated |
| Peak RSS | Measured | Measured | Calculated |

### Relative performance calculation

Latency improvement:

\[
\text{Latency improvement (\%)} =
\frac{L_{C++}-L_{Rust}}{L_{C++}}\times100
\]

Throughput speedup:

\[
\text{Throughput ratio} =
\frac{T_{Rust}}{T_{C++}}
\]

Interpretation:

- Positive latency improvement means Rust has lower latency.
- Negative latency improvement means Rust has higher latency.
- Throughput ratio greater than 1 means Rust has higher throughput.
- Throughput ratio below 1 means C++ has higher throughput.

Use actual measurements.

Do not hardcode the paper's reported C++ numbers as if they were freshly measured results.

The comparison tool should accept imported C++ benchmark results or a manually prepared result file.

---

## 15. Task 12 — Implement Benchmark Report Generation

Generate machine-readable benchmark reports.

Support:

- JSON
- CSV

Suggested output structure:

```text
benchmark-results/
├── deeplob/
│   ├── run-001.json
│   ├── run-001.csv
│   └── summary.json
│
├── transformerlob/
│   ├── run-001.json
│   ├── run-001.csv
│   └── summary.json
│
└── comparisons/
    ├── deeplob-rust-vs-cpp.json
    └── transformerlob-rust-vs-cpp.json
```

Adapt file naming and output paths to the existing project.

### Benchmark metadata

Every report should include:

```text
model_name
model_file
model_file_hash
runtime_version
execution_provider
input_shape
input_type
warmup_iterations
measured_iterations
trial_count
hardware_information
operating_system
compiler_version
build_mode
thread_configuration
timestamp
benchmark_mode
```

Where available, also record:
- Git commit hash.
- Rust version.
- C++ compiler version.
- CPU information.
- RAM information.
- Benchmark configuration.

### Report integrity

- Never overwrite previous benchmark reports silently.
- Preserve raw measurements.
- Include summary statistics.
- Identify missing metrics explicitly.
- Do not fabricate values.
- Clearly distinguish reference numbers from newly measured results.

---

## 16. Task 13 — Extend Benchmark CLI

Extend the Phase 5 command.

### Basic benchmark

```bash
lob-inference benchmark \
  --model deeplob \
  --iterations 200
```

### Custom warm-up and trials

```bash
lob-inference benchmark \
  --model deeplob \
  --warmup 20 \
  --iterations 200 \
  --trials 5
```

### End-to-end benchmark

```bash
lob-inference benchmark \
  --model deeplob \
  --mode end-to-end \
  --input data/test.csv
```

### Pure inference benchmark

```bash
lob-inference benchmark \
  --model deeplob \
  --mode inference \
  --iterations 1000
```

### Resource profiling

```bash
lob-inference benchmark \
  --model deeplob \
  --profile-memory \
  --profile-cpu
```

### Report export

```bash
lob-inference benchmark \
  --model deeplob \
  --format json \
  --output benchmark-results/deeplob.json
```

### Rust/C++ comparison

```bash
lob-inference compare \
  --model deeplob \
  --rust-results benchmark-results/rust.json \
  --cpp-results benchmark-results/cpp.json
```

The `compare` command is a proposed extension. Use the existing CLI design and avoid unnecessary command complexity.

---

## 17. Task 14 — Benchmark Reliability and Reproducibility

Benchmark results can be distorted by environmental conditions.

Implement and document the following controls:

1. Use release builds.
2. Run warm-up iterations.
3. Use the same input tensors.
4. Reuse loaded sessions.
5. Use the same execution provider.
6. Use the same runtime version.
7. Use consistent thread settings.
8. Avoid logging inside timed loops.
9. Run multiple independent trials.
10. Preserve raw latency samples.
11. Record system information.
12. Record build and runtime configuration.

### Important environmental factors

Document:
- CPU frequency scaling.
- Thermal throttling.
- Background system activity.
- Memory pressure.
- Operating-system scheduling.
- Runtime thread configuration.

Do not claim perfect reproducibility. The goal is controlled, transparent measurement.

### Benchmark validity

A benchmark must fail or be marked invalid if:
- Model loading fails.
- Warm-up fails.
- Inference fails.
- The input tensor is invalid.
- The output is invalid.
- Required measurement data is missing.

---

## 18. Task 15 — Tests

### A. Statistics tests

Test:
- Mean.
- Median.
- P90.
- P95.
- P99.
- Minimum.
- Maximum.
- Standard deviation.
- Empty sample handling.
- Single-sample handling.

Use deterministic test data with known expected statistics.

### B. Timer tests

Verify:
- Timings are non-negative.
- Each iteration is recorded.
- Timing collection works across repeated runs.

Do not assert exact execution durations.

### C. Throughput tests

Verify:
- Correct windows-per-second calculation.
- Zero elapsed time is handled safely.
- Zero completed windows is handled safely.

### D. Warm-up tests

Verify:
- Correct number of warm-up calls.
- Warm-up samples are excluded from timed statistics.
- Warm-up errors abort execution.

### E. Report tests

Verify:
- JSON serialization.
- CSV serialization.
- Required metadata fields.
- Missing optional metrics.
- Existing report handling.
- Invalid report data.

### F. Comparison tests

Use synthetic benchmark results to verify:
- Latency difference calculations.
- Throughput ratios.
- Missing metrics.
- Invalid comparisons.

### G. Integration tests

When actual model assets are available:
- Run DeepLOB benchmark.
- Run TransformerLOB benchmark.
- Verify result generation.
- Verify session reuse.
- Verify warm-up and timed iteration counts.

Do not assert fixed benchmark performance numbers in automated tests.

---

## 19. Performance Engineering Constraints

1. Do not change preprocessing semantics.
2. Do not change model architecture.
3. Do not reload sessions per inference.
4. Do not include model loading in pure inference latency.
5. Avoid logging inside timed sections.
6. Avoid output serialization inside pure inference timing.
7. Avoid unnecessary allocations.
8. Avoid benchmark instrumentation that materially changes measured execution.
9. Keep raw samples for later analysis.
10. Do not optimize before establishing a baseline.

If optimizations are introduced later, measure each change against the original baseline.

Every optimization must preserve the inference contract.

---

## 20. Git Workflow

### Branch

```bash
git switch main
git pull --ff-only
git switch -c feat/benchmarking
```

Before switching branches:
- Inspect `git status`.
- Preserve uncommitted changes.
- Do not discard unrelated user work.

Never work directly on `main`.

### Commit strategy

Use atomic Conventional Commits.

| Checkpoint | Example commit |
|---|---|
| Benchmark architecture | `feat(benchmark): add benchmark module structure` |
| Configuration | `feat(benchmark): add benchmark configuration` |
| Warm-up | `feat(benchmark): implement warmup execution` |
| Timing | `feat(benchmark): implement inference timing` |
| Statistics | `feat(benchmark): calculate latency statistics` |
| Throughput | `feat(benchmark): calculate inference throughput` |
| Memory profiling | `feat(benchmark): add memory profiling` |
| CPU profiling | `feat(benchmark): add CPU utilization monitoring` |
| Comparison | `feat(benchmark): add Rust and C++ comparison` |
| Reporting | `feat(benchmark): export benchmark reports` |
| Tests | `test(benchmark): add benchmarking tests` |
| Documentation | `docs(benchmark): document benchmark methodology` |

These are suggested checkpoints. Keep commits meaningful, atomic, and buildable.

Before each commit:

```bash
git status
git diff
cargo fmt --check
cargo test
```

### Pull request

When complete:

1. Review the full diff.
2. Run all quality gates.
3. Push `feat/benchmarking`.
4. Open a PR against `main`.
5. Include benchmark methodology.
6. Include measured results only.
7. Include hardware and runtime metadata.
8. Document limitations.
9. Do not merge automatically.

---

## 21. Quality Gates

Run:

```bash
cargo fmt --check

cargo check

cargo test

cargo clippy --all-targets --all-features -- -D warnings

cargo build --release
```

Additionally verify:

- DeepLOB benchmark works.
- TransformerLOB benchmark works.
- Warm-up iterations are excluded.
- Timed iterations are recorded.
- Latency statistics are correct.
- Throughput calculations are correct.
- Memory profiling works where supported.
- CPU profiling works where supported.
- JSON reports are valid.
- CSV reports are valid.
- C++ comparison works with imported measurements.
- Benchmark configuration is recorded.
- Phase 1–5 functionality remains intact.
- No datasets, ONNX binaries, generated outputs, or build artifacts are accidentally committed.

---

## 22. Phase 6 Acceptance Criteria

- [ ] Benchmark architecture is implemented.
- [ ] Benchmark configuration is configurable.
- [ ] DeepLOB can be benchmarked.
- [ ] TransformerLOB can be benchmarked.
- [ ] Warm-up execution works.
- [ ] Model loading is excluded from pure inference latency.
- [ ] Individual inference durations are measured.
- [ ] Mean latency is calculated.
- [ ] Standard deviation is calculated.
- [ ] P50, P90, P95, and P99 are calculated.
- [ ] Minimum and maximum latency are calculated.
- [ ] Throughput is calculated.
- [ ] Preprocessing is measured separately.
- [ ] End-to-end performance is measured.
- [ ] Memory profiling is implemented where supported.
- [ ] CPU profiling is implemented where supported.
- [ ] Multiple benchmark trials are supported.
- [ ] Raw benchmark samples are preserved.
- [ ] JSON and CSV report generation works.
- [ ] Hardware and runtime metadata are recorded.
- [ ] Rust/C++ comparison is implemented.
- [ ] Fair comparison methodology is documented.
- [ ] Unit and integration tests pass.
- [ ] Phase 1–5 functionality remains intact.
- [ ] Quality gates pass.
- [ ] Changes are committed on `feat/benchmarking`.
- [ ] Pull request against `main` is ready for review.

---

## 23. Instructions for Codex

You are implementing **Phase 6 only** of the Rust inference engine.

### Before implementation

1. Inspect the repository structure.
2. Read completed Phase 1–5 implementations.
3. Inspect the Git status and commit history.
4. Inspect the Phase 5 benchmark command.
5. Inspect DeepLOB and TransformerLOB inference APIs.
6. Inspect the original C++ benchmark implementation.
7. Identify how C++ records latency and throughput.
8. Verify the available ONNX Runtime version.
9. Identify available system metrics on the target platform.
10. Prepare an implementation plan before modifying files.

### Git requirements

- Create or switch to `feat/benchmarking`.
- Do not include `phase-06` in the branch name.
- Never work directly on `main`.
- Preserve unrelated user changes.
- Use atomic Conventional Commits.
- Do not force-push.
- Do not automatically merge the PR.

### Implementation requirements

- Extend the Phase 5 benchmark command.
- Reuse the existing inference engines.
- Implement reproducible warm-up and timing.
- Implement latency statistics.
- Implement throughput calculations.
- Separate model loading, preprocessing, inference, and end-to-end measurements.
- Add resource profiling where supported.
- Support both models.
- Implement JSON and CSV reports.
- Implement Rust/C++ comparison.
- Record hardware and runtime metadata.
- Write meaningful tests.
- Preserve raw measurement samples.

### Strict exclusions

Do not implement:

- New model architectures.
- Model training or export.
- GPU execution.
- Candle, Burn, or tch.
- New preprocessing algorithms.
- Changes to C++ inference behavior.
- Phase 7 numerical equivalence validation.
- Unverified performance optimizations.

### Completion report

At completion, provide:

1. Final benchmark architecture.
2. Benchmark commands.
3. Warm-up and timing methodology.
4. Latency statistics implementation.
5. Throughput calculation methodology.
6. Memory profiling implementation.
7. CPU profiling implementation.
8. DeepLOB benchmark results.
9. TransformerLOB benchmark results.
10. Rust/C++ comparison results, if C++ measurements are available.
11. Hardware and runtime metadata.
12. Generated report locations.
13. Tests and quality-gate results.
14. Git branch name.
15. Commit hashes and messages.
16. Pull request status.
17. Known limitations.

**Never fabricate benchmark results or claim Rust is faster than C++ without controlled measurements.**

The final objective is a reproducible benchmark suite that establishes the real performance characteristics of both Rust inference engines and enables a fair comparison with the original C++ implementation.
