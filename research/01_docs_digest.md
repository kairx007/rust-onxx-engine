# 01 — Docs digest

Files under `docs/` in the working tree: `running-a-model.md`, `RustLOB.pdf`, `DeepLOB_HFT_IEEE_Paper (1) - PRIYANSHU MITTAL IIIT Dharwad (1).pdf`. The six `phase-0N-implementation.md` files are deleted in the working tree but present at `HEAD`; digested here from git. I also digest `README.md`, the spec files in `llm_context/`, and `ppt/index.html`, because they carry claims and numbers the paper will lean on. PDF text was extracted with a small zlib script (no PDF tooling installed): whitespace is lost but content is readable; table cell alignment may be imperfect → treat PDF numbers as [needs eyeball check against the PDF].

---

## `README.md` (91 lines)
- **Purpose:** quickstart. "RustLOB runs DeepLOB and TransformerLOB ONNX models on CPU. It provides model inspection, CSV preprocessing, predictions, and repeatable inference benchmarks."
- **Key facts:** Rust ≥1.92; ONNX Runtime "1.31 through `ort` 2.0.0-rc.15"; model contract float32 `[1,1,100,40]` → 3 classes; non-overlapping 100-row windows; outputs never overwritten without `--overwrite`; benchmark reports never overwritten.
- **Stated non-goals / limitations (§Current limitations):** "No ONNX models, normalization arrays, or real datasets are included… GPU inference, automatic input-layout conversion, and numerical-equivalence validation are not implemented." → **stale in part**: models and normalization arrays are now committed (M-2).
- **Numbers:** none.

## `docs/running-a-model.md` (217 lines)
- **Purpose:** step-by-step user guide, ONNX file → predictions → benchmark → Rust/C++ comparison.
- **Design decisions stated:**
  - Loads **only ONNX**; no `.pt/.pth/.h5/.keras` (§1).
  - Contract "enforced from the loaded ONNX graph metadata. The code does not guess a layout conversion" (§1).
  - Window preprocessing "intentionally constrained to sequence length 100 and stride 100"; each CSV is its own segment so no window crosses a file boundary (§3).
  - Normalization = *training* `mu/sigma` loaded, never computed from the prediction set (§3).
  - Output written to a temp file and published only after all predictions succeed (§7).
  - Inference benchmark "reuses that same tensor to isolate model execution"; percentiles nearest-rank; population std (§8).
  - Compare checks model, mode, provider, shape/type, model fingerprint, runtime-version string match; "cannot prove that imported C++ measurements … are genuine" (§9).
- **Non-goals (§11):** does not establish numerical equivalence with the C++ implementation or model correctness.
- **Numbers:** none (defaults 20 warm-up / 200 iterations / 5 trials; RSS/CPU sampling every 100 ms).
- **Reusable for the paper:** the §8 timed-boundary taxonomy (inference / preprocessing / end-to-end / load) → a figure; the §9 comparability checklist → a threats-to-validity table.

## `docs/phase-01-implementation.md` (HEAD, 39 lines) — runtime foundation
- Delivered: TOML config + validation, `tracing`, process-wide ORT init with CPU EP and telemetry off, DeepLOB/TransformerLOB registry, sessions whose I/O names/types/dims come from the actual graph, `runtime-info`/`inspect`/`validate`.
- Decisions: `ort` pinned `=2.0.0-rc.15` (RC API); only `std, tracing, download-binaries, tls-native, api-31` features; ndarray support disabled "because this phase does not construct tensors"; macOS needs 13.4 target + Xcode availability helper (`build.rs`).
- Numbers: resolved versions (anyhow 1.0.104, clap 4.6.7, ort 2.0.0-rc.15, serde 1.0.229, thiserror 2.0.21, toml 0.9.12, tracing 0.1.44, tracing-subscriber 0.3.23); **13 tests passed**.
- Non-goals: GPU; Phase 2+.
- Limitation then: no model files → "Successful loading … pending."

## `docs/phase-02-implementation.md` (HEAD, 40 lines) — preprocessing
- Delivered: segmented dataset (flat row-major f32 features + aligned `u8` labels), CSV loader with blank-row/width/numeric/label checks, 1-D NPY (f32 or f64 → f32) stats loader, in-place z-score with ε = 1e-8, non-overlapping 100-row windows with label of the **last** row, contiguous 4,000-element tensor.
- Safety decision: `schema_verified` defaults false; candidate FI-2010 mapping cols 0–39 / label col 144 "from the phase specification only".
- Assumption flagged: labels are in the same CSV as features.
- Numbers: **36 tests**. Zero std allowed (ε handles it); negative std rejected.
- Reusable: window-vs-segment boundary rule; the "refuse to run until the schema is verified" gate.

## `docs/phase-03-implementation.md` (HEAD, 22 lines) — DeepLOB engine
- Delivered: `DeepLobContract` (names from graph), borrowed `TensorRef<f32>` adapter ("preserves the contiguous data without cloning its feature buffer"), `DeepLobEngine` (one reusable session), output validation, `Direction`/`Prediction` with lowest-index tie-break; "**no softmax is applied**" — returned scores unchanged; unknown whether outputs are logits or probabilities.
- Numbers: **43 tests**. Limitation: no real model run.

## `docs/phase-04-implementation.md` (HEAD, 20 lines) — TransformerLOB engine
- Delivered: `TransformerLobContract`, adapter that "does not guess a reshape, transpose, pad, or truncation", `TransformerLobEngine`; DeepLOB refactored onto a **shared runner** (`src/inference/runner.rs`).
- Numbers: **46 tests**. No real model run.

## `docs/phase-05-implementation.md` (HEAD, 30 lines) — CLI
- Delivered: six subcommands, model factory, CSV prediction with temp-file-then-publish, basic benchmark. Output schema: `window_index,source_file,segment_index,start_row,true_class,predicted_class,predicted_direction,down_score,stationary_score,up_score`.
- Numbers: **59 tests**. No real inference possible then.

## `docs/phase-06-implementation.md` (HEAD, 42 lines) — benchmarking
- Four benchmark modes; reports (JSON retains every sample; CSV one row per sample); FNV-1a 64 model fingerprint; optional `ps`-based RSS/CPU sampling at 100 ms ("adds load … should not be used as the cleanest baseline"); Rust/C++ `compare`.
- Limitations: no models/data to produce real numbers; CSV load + feature extraction reported together; end-to-end serialization is in-memory only; **thread controls not exposed**; "does not establish numerical equivalence or claim a Rust performance advantage."
- Reusable: the explicit list of what each timer includes/excludes.

## `docs/RustLOB.pdf` — *Project proposal* "RustInferenceLOB: A Cross-Runtime Benchmark of Rust, Python, and C++ Inference Engines for Limit Order Book Mid-Price Prediction" (Ujwal Akotkar, IIIT Dharwad, supervisor Prof. Shirshendu Layek, dated 11 Oct 2026)
- **Goals:** first systematic cross-runtime benchmark; six configurations — Python/PyTorch, Python/ONNX, C++/ONNX, Rust/ONNX (`ort`), Rust/Candle, Rust/Burn (config table lists `tch` in the timeline); metrics: latency (mean/P50/P99/P999), cold start, peak memory, allocation count, throughput, numerical equivalence (tolerance 10⁻⁵ abs), accuracy (macro-F1, recall); Mann–Whitney U, 95% CIs, process isolation, 20 warm-up + 200+ timed runs; decision framework; open-source Rust workspace with Criterion benches; target ACM ICAIF 2026 (deadline 2 Aug 2026 — already passed relative to 2026-11-22; fallback listed: ICAIF workshop Oct 2026, IEEE CIFEr).
- **RQs:** RQ1 does Rust-via-ORT beat Python/C++ significantly; RQ2 pure-Rust frameworks vs ORT at batch 1; RQ3 cold start/memory/allocation/numerical reproducibility.
- **Stated assumptions/numbers:** ONNX opset 11, fixed input `[1,1,100,40]`; training: Adam lr 1e-4, wd 1e-5, OneCycleLR, batch 32, seed 42, inverse-frequency class weights; chronological split train days 1–7 / val day 8 / test days 9–10; k = 10 horizon. Cites "Oxidizing Cross-Encoders" (Rust ORT only 1.1× faster than Python on a small model).
- **Reusable:** RQ framing, threats-to-validity list, the 6-config matrix.
- **Gap vs code:** only the Rust/ORT leg exists (M-7); no training/export scripts, no Python/C++ code, no Criterion.

## `docs/DeepLOB_HFT_IEEE_Paper (1) - …pdf` — upstream C++ paper "DeepLOB-HFT-Cpp" (IIIT Dharwad, five authors)
- **System:** DeepLOB (CNN+LSTM ≈ 66K params) and TransformerLOB (CNN + 2-layer, 4-head Transformer ≈ 114K params) trained on FI-2010; PyTorch→ONNX→C++ ONNX Runtime executables (single-sample, latency benchmark, full CSV).
- **Numbers (quote exactly; source: that PDF, Sec. VII):**
  - Table III (TransformerLOB, 200 timed runs, input `[1,1,100,40]`): Python(ONNX) mean 2.033 ms, std 2.420, P99 8.886, speedup 1.00×; C++(ONNX) mean 1.159 ms, std 2.053, P99 9.786, **1.75×**.
  - Table IV (end-to-end C++, 500 windows): total 545.06 ms; mean 1.090 ms/window; throughput 917.33 win/s; **top-1 accuracy 33.00% (165/500) on a 15-epoch model**.
  - Table II (DeepLOB per-class recall, n = 9,000): Down 38.52%, Stationary 59.35%, Up 34.38%.
  - The paper notes P99 is *higher* for C++ than Python (9.8 vs 8.9 ms) and attributes it to OS effects (not tested).
  - Hardware/OS/ORT version/thread config of the C++ runs: **[UNCLEAR — not found in the text I extracted; check the PDF]**.
- **Relevance:** this is the baseline the Rust engine is compared against (`context/context.md` §2, `ppt/` slide 6).

## `llm_context/context.md` + `phase-01…07.md` (specs, 514 + ~5,500 lines) — *not in docs/, but the intent document*
- Defines goals: reproduce the C++ pipeline in Rust, numerical equivalence (abs tol **1e-5**), Rust-vs-C++ benchmark "under identical experimental conditions", seven phases; stack table lists "Tensor representation: ndarray / contiguous buffers" (code uses plain `Vec<f32>`); proposes a module tree (`cli/{inference,benchmark,evaluate}.rs`, `evaluation/{metrics,evaluator}.rs`, `tests/equivalence_tests.rs`, `results/benchmarks/`) — actual tree differs and has no `evaluation/` module or equivalence tests.
- **Phase 7 (numerical equivalence + final validation) has no implementation and no doc**; grep for `equivalen` in `src/` is empty.
- Model specs stated: DeepLOB 3 conv blocks + 1-layer LSTM (64 hidden) ≈ 66K params; TransformerLOB 3 conv blocks + 2 encoder layers, 4 heads, d_model 64, FFN 128 ≈ 114K params. These are *claims about the ONNX files*; I did not inspect graph internals beyond I/O metadata → [TODO: dump ONNX op list to confirm].

## `ppt/index.html` (8 slides, "Neo-Grid Bold" theme) — *numbers the paper may be tempted to reuse; see M-6*
- Slide 5 "Rust inference on Apple M3": mean latency **DeepLOB 0.799 ms; TransformerLOB 1.785 ms**; P99 **0.915 ms / 1.913 ms**; 1,000 timed samples each; **1,251 / 560 windows/s**; "Release build · CPU · ONNX Runtime 1.31 · 20 warm-ups · 5 × 200 timed calls … Recorded benchmark reports · inference-only timing; input tensor reused".
- Slide 6 "C++ run and published reference" (TransformerLOB mean, ms): fresh C++ **0.854** (Apple M3; P99 0.939; ORT **1.24.4**; **one thread**; **synthetic input**; times `Session.Run`) vs Rust **1.785** (ORT 1.31; **default threads**; CSV-derived input; times Rust `predict` path) vs paper C++ **1.159**. The deck itself says: "Promising, not yet apples-to-apples … does not isolate a language speed advantage."
- Slide 7: "Claim supported today: working inference + preliminary performance data"; lists what's missing (identical tensors, score/class comparison, aligned runtime/threads/boundaries).
- Reusable ideas: the 4-stage pipeline strip (slide 4); the comparability-caveat framing (slide 6).
- **Problem:** I could not find the benchmark reports behind slide 5/6 (only one JSON exists and its numbers differ) — see `04_paper_evidence.md`.
