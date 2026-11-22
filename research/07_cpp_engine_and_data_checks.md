# 07 — C++ engine and data checks

Sources: `cpp-engine/` (C++ engine, Python training code, raw FI-2010 CSVs), benchmark run
`results/phase2/20261122-full-cpp/` (Rust, Python, C++ in one randomized, interleaved schedule; 415 processes,
0 failures; battery power, Low Power Mode off), original C++ binaries `results/phase2-cpp/original/`,
data report `results/data_checks/report.md` (regenerate with `bench/data_checks.py`).

## C++ engine
- Code: `cpp_inference/{inference,benchmark,full_inference}.cpp`, ORT C++ API, CPU, `ORT_ENABLE_ALL`, 1 intra-op thread,
  zero-copy `CreateTensor`. Written for Windows (MinGW, ORT 1.24.4). Built unmodified on macOS against Microsoft's
  ORT **1.31.0** C++ SDK (NuGet osx-arm64; GitHub release unreachable from this machine).
- Reproduces the authors' Phase 5 result exactly: TransformerLOB accuracy **33.0000% (165/500)**.
  On this M3: TransformerLOB 0.89–0.94 ms mean, DeepLOB 0.67–0.70 ms (1 thread; their Windows report: 1.159 ms).
- Their `benchmark.cpp` times `Session::Run` only, on synthetic input, one thread; `full_inference` uses
  stride-1 windows and label at row i+100 (matches training).
- Harness adapter `bench/cpp/rustlob_cpp_bench.cpp` (same setup) runs in the fair schedule as engine `cpp`.

## Three-engine results (fast-mode per-process medians, cycled inputs)

| Model | Threads | Rust | Python | C++ |
|---|---|---|---|---|
| DeepLOB | 1 | 0.672 ms | 0.675 ms | 0.670 ms |
| DeepLOB | ORT default | 0.312 | 0.280 | 0.275 |
| TransformerLOB | 1 | 0.892 | 0.898 | 0.901 |
| TransformerLOB | ORT default | 0.514 | 0.344 | 0.336 |

- 1 thread: all three within ~1%; ratios vs Rust 0.997–1.014, all p > 0.05 except one Python case (p = 0.021, +0.7%).
- Default threads: C++ 0.879× (DeepLOB) and 0.653× (TransformerLOB) of Rust, p ≤ 0.003 — the same gap as Python and
  as Rust loading Microsoft's dylib, i.e. the `ort`-bundled ORT binary, not Rust code.
- Overhead outside `Session::run`: C++ 0.3–0.4 µs, Rust 1.9–2.2 µs, Python 4.1–4.6 µs.
- Cold start (spawn → first prediction → exit): Rust 39 / 51 ms, C++ 51 / 61 ms, Python 111 / 118 ms.
- Numerics: C++ bit-identical to Rust and Python at 1 thread; same ≤ 5.7e-6 drift at default threads.

## Data findings
1. All models and processed arrays in `cpp-engine/` are byte-identical to the Rust repo's.
2. Raw CSV: header line + index column 0; features raw cols 1–40, labels raw cols 145–149. The Rust default config
   (cols 0–39, label 144) is off by one for the raw file.
3. **Normalization skew**: shipped `mu/sigma` = stats of the first 50k training rows; `train.py` trains on 200k rows
   (σ differs up to 118%) and never saves its stats. Effect on accuracy is small (DeepLOB 64.3% → 65.4%).
4. **Label names flipped**: data show FI-2010 label 1 = Up, 3 = Down ⇒ class 0 = Up, 2 = Down; Rust's `Direction`
   and the C++ printouts say 0 = Down.
5. **Rust label off-by-one**: Rust labels each window with its last row (and stride 100); training uses the next row
   (stride 1). The labels differ for 28.2% of Rust's 319 test windows.
6. Model quality (days 9–10, training convention): DeepLOB 64–65% (majority baseline 71%, macro-F1 0.35);
   TransformerLOB 19%. On its own training windows TransformerLOB scores 31% ⇒ the exported ONNX does not behave
   like a trained model. Confirming needs `transformer_checkpoint.pt` (not provided).
7. The Phase 2 reconstructed CSV matches the real raw test CSV within 7.1e-7 ⇒ Phase 2 results stand.

## Fixes applied to the Rust engine (after this review)
- Windows: window *i* = rows *i*…*i*+99, label = row *i*+100 (training convention); `stride` configurable, default 1
  (`src/preprocessing/window.rs`, `src/tensor/input.rs`, `src/config/settings.rs`).
- Directions: class 0 = Up, 1 = Stationary, 2 = Down (`src/inference/prediction.rs`, `src/data/labels.rs`);
  prediction CSV columns renamed `class0_up_score`, `class1_stationary_score`, `class2_down_score`.
- Default config matches the raw FI-2010 CSV: header line, features in columns 1–40, label in column 145.
- Verification: on the real test CSV the Rust engine evaluates the same 31,837 windows as C++ `full_inference` with
  identical correct counts (DeepLOB 18,908; TransformerLOB 7,311). Tests: 70 pass, including a raw-layout and a
  stride-1 training-convention test. Latency results are unaffected (labels play no part in timing).
- Model quality (normalization stats, TransformerLOB export) deliberately left as is: both engines run the same
  model files, so the Rust vs C++ comparison does not depend on it.
