# Paper draft — Rust vs C++ inference engines for limit-order-book models

> Working draft built only from measured results in this repository (`research/08_final_results.md`,
> `results/phase2/20261122-final-rust-vs-cpp/`, `results/data_checks/report.md`). `[CITE]` marks places where the
> literature search (Phase 3) must supply references; nothing has been cited yet. Venue and author list: [TODO].

**Working title:** *Same Runtime, Same Speed? A Controlled Rust vs C++ Comparison of ONNX Runtime Inference for
Limit-Order-Book Mid-Price Prediction*

## Abstract

Deep models for limit-order-book (LOB) mid-price prediction are usually trained in Python and deployed through a
native inference runtime, and Rust is increasingly proposed as a memory-safe alternative to C++ for that deployment
path [CITE]. We compare a Rust inference engine (RustLOB, `ort` bindings) with an existing C++ engine
(DeepLOB-HFT-Cpp [CITE]) on two published architectures, DeepLOB and a Transformer variant, using identical ONNX
models, identical preprocessed FI-2010 inputs, and the same ONNX Runtime release. Measurements use one fresh process
per trial in a seed-randomized, interleaved schedule (499 processes) and are compared on per-process medians with
Mann–Whitney tests and bootstrap confidence intervals. When both engines load the same vendor-built runtime library,
Rust and C++ latency differs by at most 1.1% (0.667 vs 0.674 ms for DeepLOB and 0.890 vs 0.897 ms for TransformerLOB
at one thread), outputs are bit-identical, and the language's own per-call overhead is 0.3 µs (C++) versus 1.8–2.2 µs
(Rust). The differences that do appear come from the deployment stack, not the language: the runtime build bundled by
the Rust binding is up to 55% slower with multi-threading, and the binding's default device policy silently offloads
part of each model to Apple's CoreML, changing outputs by up to 3×10⁻³ and slowing inference by up to 2×. We release
the benchmark harness, which detects both effects.

## 1. Introduction

- Context: LOB mid-price prediction with deep models (DeepLOB, Transformer variants) [CITE]; deployment latency
  budgets in electronic trading [CITE]; ONNX Runtime as the common deployment runtime [CITE].
- Question: does replacing a C++ inference engine with Rust cost latency, accuracy or reproducibility?
- Why existing comparisons are hard to interpret: language comparisons often mix runtime versions, thread settings,
  inputs and timer boundaries (our own earlier deck compared ORT 1.24.4 single-threaded C++ with ORT 1.31 default-thread
  Rust on different inputs; `research/01_docs_digest.md`).
- Contributions:
  1. A controlled Rust vs C++ comparison on two LOB architectures with identical models, inputs and runtime release,
     including a same-binary control that isolates the language.
  2. Evidence that the remaining gaps are deployment-stack effects: the binding's bundled runtime build and its default
     automatic device policy (silent CoreML offload), with ONNX Runtime node-placement traces.
  3. An open, process-isolated benchmark harness with numerical-equivalence and placement checks, plus a data audit
     that aligned both engines' windowing and labels with the training code.

## 2. Background and related work

- LOB prediction and FI-2010 [CITE Ntakaris et al.; Zhang et al. DeepLOB; Transformer LOB models].
- ONNX Runtime execution providers, threading and graph optimization [CITE].
- Rust for ML inference and systems (`ort`, Candle, Burn; prior Rust vs Python/C++ measurements) [CITE].
- Benchmarking methodology: warm-up, process isolation, tail latency, non-parametric comparison [CITE].

## 3. Systems under test

| Engine | Language / API | ONNX Runtime 1.31.0 (commit 402979a) |
|---|---|---|
| C++ | C++17, ORT C++ API (`cpp-engine/cpp_inference`, harness adapter `bench/cpp`) | Microsoft build (NuGet, osx-arm64) |
| Rust | Rust 2024, `ort` 2.0.0-rc.15, `lob-inference` | `ort`-bundled build, statically linked |
| Rust + MS ORT (control) | same Rust code, `ort-dynamic` feature | Microsoft build (PyPI wheel dylib) |
| Rust, default device policy (pitfall) | same Rust code, `ort` default `MaxEfficiency` policy | `ort`-bundled build |

Both engines: CPU execution provider, batch size 1, float32 input `[1, 1, 100, 40]`, zero-copy input tensors
(C++ `CreateTensor` over the window buffer; Rust borrowed `TensorRef`), argmax over three logits.

Models: DeepLOB (CNN + LSTM, 69,507 parameters, ONNX opset 14) and TransformerLOB (CNN + 2-layer Transformer
encoder, 119,911 parameters, opset 18), exported from PyTorch; identical files for both engines.

## 4. Methodology

**Data.** FI-2010 test split (31,937 rows), z-scored with the training statistics shipped with the models. Window *i*
covers rows *i*…*i*+99 and is labelled with row *i*+100 (training convention). For latency, 319 windows at stride 100;
for accuracy and equivalence, all 31,837 stride-1 windows. Both engines read the same exported tensor file.

**Data audit** (`bench/data_checks.py`). Raw CSV layout, normalization provenance, label semantics (FI-2010 label 1 =
Up, verified against mid-price moves), window/label alignment and artifact identity were checked; the Rust engine's
windowing, label alignment and class names were corrected to match training before measurement.

**Protocol** (`bench/run_suite.py`). One OS process per trial (20 warm-up + 200 timed calls); 15 processes per
configuration; all (configuration, repetition) pairs shuffled with a fixed seed and interleaved; 1 s cool-down;
resource sampling off; environment snapshot. Factors: engine × model × intra-op threads {1, ORT default} × input
policy {one window reused, all windows cycled}. Per call we record the full predict path and `Session::run` alone.
Cold start = wall time from process spawn to first prediction and exit. Before measuring, ORT trace logs confirm that
CPU-only engines place no node on another execution provider.

**Statistics** (`bench/analyze.py`). Unit of replication = process. Per-process medians are compared with two-sided
Mann–Whitney U tests and 95% bootstrap CIs of the ratio of medians (10,000 resamples). Tails use pooled nearest-rank
P99/P99.9. A bimodal "slow mode" seen in earlier runs (per-process medians ~1.7× apart, equally frequent for every
engine) is detected and reported separately; it did not occur in the final run.

**Machine.** Apple M3 (4 performance + 4 efficiency cores), 8 GiB, macOS 26.5.2, battery power, Low Power Mode off.

## 5. Results

### 5.1 Latency (Fig. `fig11_rust_vs_cpp_headline`, `fig2_process_medians`)

| Model | Threads | C++ | Rust + MS ORT | Rust (bundled ORT) |
|---|---|---|---|---|
| DeepLOB | 1 | 0.674 ms | 0.667 (×0.989) | 0.691 (×1.026) |
| DeepLOB | ORT default | 0.275 | 0.275 (×0.999) | 0.318 (×1.155) |
| TransformerLOB | 1 | 0.897 | 0.890 (×0.993) | 0.914 (×1.020) |
| TransformerLOB | ORT default | 0.336 | 0.338 (×1.006) | 0.520 (×1.551) |

Median of 15 per-process medians; × = ratio to C++. Same-binary comparison (Rust + MS ORT vs C++): ratios
0.989 [0.971, 0.993], 0.999 [0.995, 1.002], 0.993 [0.992, 0.996], 1.006 [1.003, 1.008]; p = 9e-6, 0.25, 9e-4, 0.004.
Rust with the bundled runtime vs C++: all eight comparisons p ≤ 6e-4.

### 5.2 Where the time goes (Fig. `fig3_host_overhead`)

Per-call time outside `Session::run`: C++ 0.3–0.4 µs; Rust 1.8–2.2 µs (input validation, borrowed-view creation,
output checks, argmax) — under 0.7% of a call. More than 99% of latency is inside ONNX Runtime for both engines.

### 5.3 Tails and throughput (Fig. `fig1_latency_ecdf`)

Pooled P99 / P99.9 (ms), 1 thread: DeepLOB C++ 0.843 / 1.153, Rust 0.743 / 0.845; TransformerLOB C++ 0.958 / 0.995,
Rust 0.998 / 1.468. Default threads: DeepLOB C++ 0.368 / 0.510, Rust 0.349 / 0.423; TransformerLOB C++ 0.534 / 0.777,
Rust 0.543 / 0.560. Neither engine has systematically heavier tails. Throughput (windows/s, 1 thread / default):
DeepLOB C++ 1,467 / 3,488 vs Rust 1,444 / 3,139; TransformerLOB C++ 1,112 / 2,924 vs Rust 1,092 / 1,919.

### 5.4 Cold start (Fig. `fig5_cold_start`)

Spawn → first prediction → exit: Rust 36.9 ms (DeepLOB) / 50.7 ms (TransformerLOB); C++ 42.4 / 55.4 ms. The Rust
binary links ONNX Runtime statically; the C++ binary loads a 45 MB shared library.

### 5.5 Numerical equivalence (Fig. `fig8_numerical_agreement`)

On all 319 benchmark windows × 3 logits, Rust (both runtime builds) and C++ are bit-identical at one thread; at
default threads all CPU engines deviate identically by ≤ 5.4×10⁻⁶ from the one-thread result (reduction order),
inside the 1×10⁻⁵ tolerance. On the full test set both engines evaluate 31,837 windows and produce identical correct
counts (DeepLOB 18,908; TransformerLOB 7,311).

### 5.6 Pitfall: implicit device policy (Fig. `fig10_device_binary_ablation`)

With the `ort` binding's default automatic device policy (`MaxEfficiency`), ONNX Runtime assigned 2 fused CoreML
partitions (of 7 post-partitioning DeepLOB nodes; the convolution blocks) and 14 (of 51 TransformerLOB nodes) to the
CoreML execution provider although only the CPU provider was registered.
Logits changed by up to 3.0×10⁻³ (DeepLOB) and 9.5×10⁻⁴ (TransformerLOB); latency rose to 0.797 ms and 1.770 ms at one
thread (×1.18, ×1.97 vs C++); cold start rose to 117 ms and 442 ms. Benchmark reports still labelled the provider
"cpu". The C++ API applies no such policy. Forcing `PreferCPU` restores bit-identical outputs and C++-level latency.

### 5.7 Secondary measurements

Graph optimization (Rust, 1 thread): disabling it costs +10% (DeepLOB 0.763 vs 0.691 ms; TransformerLOB 1.011 vs
0.914 ms); `all` equals the default. Rust preprocessing of the full raw test CSV (31,937 rows × 150 columns): CSV
parsing 95.9 ms dominates; z-scoring 0.58 ms; windowing 0.008 ms; tensor construction for 319 windows 0.48 ms.

## 6. Discussion

- **Language is not the bottleneck.** For models of this size (~0.3–0.9 ms per inference), both engines spend more
  than 99% of each call inside the same runtime; the host language contributes microseconds. With an identical
  runtime binary the engines are indistinguishable for practical purposes (≤ 1.1%).
- **The deployment stack is.** Two binding-level defaults dominate the observed differences: which runtime binary is
  linked (a 2–55% effect here, largest with multi-threading) and which devices the runtime may use (a silent change of
  hardware and precision). Neither is visible in source code; both were found only through same-binary controls and
  node-placement traces.
- **Practical guidance.** Pin and record the runtime binary (not just its version string); force the intended
  execution provider and verify placement; compare engines on the same binary; isolate processes and report
  per-process statistics.
- **Rust-specific benefits** (memory safety, single static binary, faster cold start here) come at no measurable
  inference cost once the runtime is controlled. [Discuss with CITE on Rust safety in production systems.]

## 7. Threats to validity

- One machine (Apple M3, 8 GiB), macOS, CPU only, batch size 1, battery power. A second platform (x86-64 Linux) is
  needed before generalizing. [TODO]
- The Rust and C++ Microsoft runtimes are separately packaged builds (PyPI vs NuGet) of the same release and commit.
- Two small models (<120k parameters); larger models may shift the language/runtime balance further toward the runtime.
- Model quality is not a subject of this study: both engines run the same files (DeepLOB 59.4%, TransformerLOB 23.0%
  accuracy on the full test set; the normalization statistics shipped with the models differ from those used in
  training, and the TransformerLOB export behaves like an untrained model — `results/data_checks/report.md`).
- Earlier runs showed an engine-independent bimodal process state on this machine; the final run did not, and all
  comparisons are within one interleaved schedule.

## 8. Conclusion

Under controlled conditions — same models, inputs, runtime release and binary — a Rust ONNX Runtime engine matches a
C++ engine within 1.1% on DeepLOB and TransformerLOB inference, with bit-identical outputs and microsecond-level
language overhead. The performance and correctness risks lie in the deployment stack: the runtime binary a binding
ships and the device policy it applies by default. The released harness makes both measurable.

## Artifacts

- Code: `src/` (Rust engine), `cpp-engine/` (C++ engine and training pipeline), `bench/` (harness, analysis, data checks).
- Results: `results/phase2/20261122-final-rust-vs-cpp/` (summary, tables, figures), `results/data_checks/report.md`.
- Final numbers: `research/08_final_results.md`.

## Open items before submission

1. Literature search and citations (Phase 3).
2. Repeat the final run on a second machine (x86-64 Linux) and on AC power.
3. Decide venue and format; convert to the venue template.
4. Optional: verify the TransformerLOB export against its PyTorch checkpoint (only affects the threats section).
