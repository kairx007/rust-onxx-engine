# 08 — Final results: Rust vs C++ inference for DeepLOB / TransformerLOB

Supersedes the interim numbers in `06_phase2_results.md` and `07_cpp_engine_and_data_checks.md`.

**Run:** `results/phase2/20261122-final-rust-vs-cpp/` (summary `summary.md`, figures `figures/`, tables `tables/`).
**Machine:** Apple M3 (4 performance + 4 efficiency cores), 8 GiB, macOS 26.5.2; battery power, Low Power Mode off,
1-min load 1.3–2.8.
**Protocol:** 499 isolated benchmark processes, 0 failures. Each measurement is a fresh process (20 warm-up + 200 timed calls);
15 processes per configuration; all configurations shuffled with a fixed seed and interleaved; 1 s cool-down.
No slow-mode processes occurred in this run (0/15 in every configuration).
**Inputs:** 319 windows from the real FI-2010 test CSV, normalized with the shipped training statistics; identical
bytes for every engine (`inputs/windows.npy`).
**Models:** `models/deeplob.onnx`, `models/transformerlob.onnx`, byte-identical to `cpp-engine/`.

## Engines

| Key | Engine | ONNX Runtime |
|---|---|---|
| `cpp` | C++ (`bench/cpp` adapter of `cpp-engine/cpp_inference`), ORT C++ API | Microsoft 1.31.0 (NuGet osx-arm64) |
| `rust` | Rust `lob-inference`, `ort` 2.0.0-rc.15, CPU-only device policy | `ort`-bundled 1.31.0 (static) |
| `rust-msdylib` | same Rust code, `--features ort-dynamic` | Microsoft 1.31.0 (PyPI wheel dylib) |
| `rust-coreml` | Rust with the `ort` crate's implicit `MaxEfficiency` device policy | `ort`-bundled 1.31.0 |

All four use ONNX Runtime 1.31.0 built from commit `402979a`. The two Microsoft libraries are separately packaged
builds (NuGet vs PyPI, different file hashes) of the same release.

## Headline: latency (median of per-process medians, 95% bootstrap CI; cycled inputs)

| Model | Threads | C++ | Rust + Microsoft ORT | Rust (bundled ORT) |
|---|---|---|---|---|
| DeepLOB | 1 | 0.674 ms [0.673, 0.675] | 0.667 (×0.989) | 0.691 (×1.026) |
| DeepLOB | ORT default | 0.275 [0.275, 0.276] | 0.275 (×0.999) | 0.318 (×1.155) |
| TransformerLOB | 1 | 0.897 [0.895, 0.897] | 0.890 (×0.993) | 0.914 (×1.020) |
| TransformerLOB | ORT default | 0.336 [0.335, 0.337] | 0.338 (×1.006) | 0.520 (×1.551) |

(× = ratio to C++; figure `fig11_rust_vs_cpp_headline`; tables `headline_rust_vs_cpp.csv`,
`same_binary_rust_vs_cpp.csv`, `engine_comparison.csv`.)

1. **Same runtime library ⇒ Rust ≈ C++.** Rust + Microsoft ORT vs C++ + Microsoft ORT: ratios 0.989–1.006
   (Mann–Whitney U on per-process medians: DeepLOB 1 thread p = 9e-6 in Rust's favour; default threads p = 0.25;
   TransformerLOB 1 thread p = 9e-4 in Rust's favour; default threads p = 0.004 in C++'s favour). All differences ≤ 1.1%.
2. **The bundled ORT binary is the gap, not Rust.** With the ORT build the `ort` crate downloads, Rust is 2–3% slower
   than C++ at 1 thread and 15% (DeepLOB) / 55% (TransformerLOB) slower with ORT's default thread pool
   (p ≤ 6e-4 for all eight Rust-vs-C++ comparisons). Swapping in Microsoft's library removes the gap entirely, so the
   cause is how that binary was built (kernels/threading), not the Rust code.
3. **Host-language overhead is negligible.** Time per call outside `Session::run`: C++ 0.3–0.4 µs, Rust 1.8–2.2 µs —
   under 0.7% of a call (`fig3_host_overhead`).
4. **Tails.** Pooled P99 / P99.9 (ms): DeepLOB 1 thread C++ 0.843 / 1.153, Rust 0.743 / 0.845; default threads C++
   0.368 / 0.510, Rust 0.349 / 0.423. TransformerLOB 1 thread C++ 0.958 / 0.995, Rust 0.998 / 1.468; default threads
   C++ 0.534 / 0.777, Rust 0.543 / 0.560. No engine has systematically heavier tails (`fig1_latency_ecdf`).
5. **Throughput** (windows/s, median process): DeepLOB C++ 1,467 / 3,488 vs Rust 1,444 / 3,139; TransformerLOB C++
   1,112 / 2,924 vs Rust 1,092 / 1,919 (1 thread / default threads).

## Cold start (spawn → runtime init → model load → first prediction → exit)

| Model | Rust | C++ | Rust with CoreML policy |
|---|---|---|---|
| DeepLOB | 36.9 ms [29.4, 40.5] | 42.4 [38.2, 49.6] | 117.2 |
| TransformerLOB | 50.7 [43.8, 51.9] | 55.4 [45.7, 61.3] | 442.2 |

Rust starts ~5 ms faster than C++ here; the C++ figure includes dynamic loading of the 45 MB ORT dylib, while the Rust
binary links ORT statically. (`fig5_cold_start`, `tables/cold_start.csv`.)

## Numerical equivalence (all 319 windows × 3 logits)

- Rust (bundled ORT), Rust + Microsoft ORT and C++ are **bit-identical at 1 intra-op thread**.
- At default threads every engine shows the same ≤ 5.4e-6 deviation from the 1-thread result (parallel reduction
  order), inside the 1e-5 tolerance; argmax agreement 100%.
- On the full test CSV (31,837 stride-1 windows, training label convention) Rust and C++ produce identical correct
  counts: DeepLOB 18,908 and TransformerLOB 7,311 (`results/phase2-cpp/rust_vs_cpp_eval/`).

## Pitfall found: silent CoreML offload in the Rust binding

`ort` 2.0.0-rc.15 sets ONNX Runtime's `MaxEfficiency` automatic device policy on every session. On Apple silicon ORT
then places 2 DeepLOB and 14 TransformerLOB nodes on the CoreML execution provider (ORT trace logs,
`tables/node_placement.csv`) although only the CPU provider is registered. Effects: logits off by up to 3.0e-3
(300× the tolerance), 1.2–2.0× slower inference (DeepLOB 0.797 vs 0.674 ms; TransformerLOB 1.770 vs 0.897 ms at
1 thread), and 3–8× slower cold start. The C++ API applies no such policy. Fix: `device_policy = "cpu"` (default),
which sets `PreferCPU`; regression test in `tests/model_loading_tests.rs`.

## Data alignment fixes (Rust)

The Rust engine originally labelled each window with its last row and used stride 100; training and the C++ engine use
the row after the window and stride 1 (labels differed for 28% of windows), and it named class 0 "Down" although
FI-2010 label 1 (class 0) is Up. Both are fixed; see `07_cpp_engine_and_data_checks.md` and
`results/data_checks/report.md`.

## Not in scope / caveats

- Model quality is outside this comparison (both engines run the same model files): DeepLOB 59.4% and TransformerLOB
  23.0% accuracy on the full test set; normalization statistics shipped with the models differ from the ones used in
  training. Details in `results/data_checks/report.md`.
- One machine (Apple M3, 8 GiB), macOS, CPU only, batch size 1, on battery power.
- The Rust and C++ Microsoft libraries are different packagings of the same ORT release.
- Python was measured in earlier runs (bit-identical outputs, same latency as C++ within ~2%) and is omitted here by
  choice.

## Reproduce

```sh
bench/.venv/bin/python -I bench/run_suite.py --reps 15 --engines rust,cpp,rust-msdylib,rust-coreml
bench/.venv/bin/python -I bench/analyze.py results/phase2/<run_id> --reference cpp
bench/.venv-ort131/bin/python -I bench/data_checks.py
```

Figures in `results/phase2/20261122-final-rust-vs-cpp/figures/`:
`fig11_rust_vs_cpp_headline` (headline), `fig2_process_medians`, `fig1_latency_ecdf`, `fig3_host_overhead`,
`fig5_cold_start`, `fig8_numerical_agreement`, `fig10_device_binary_ablation`, `fig4_drift`, `fig6_graph_opt`,
`fig7_preprocessing`, `fig9_confusion` (PNG and PDF).
