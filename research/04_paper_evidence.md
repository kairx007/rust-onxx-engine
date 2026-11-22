# 04 — Evidence for the paper

Framing caveat: the code is a **thin, validation-heavy Rust harness around ONNX Runtime** for two LOB classifiers. Performance of the network is ORT's. A paper must therefore be about *deployment/measurement methodology and engineering around ORT* (or a cross-runtime comparison, as the proposal intends), not about a new inference engine/kernels. Ratings below follow that reality.

## 1. Candidate contributions

| # | Claim (draft wording) | Evidence in repo | Strength |
|---|---|---|---|
| C1 | **A contract-first Rust inference harness**: model I/O is validated against the *loaded ONNX graph* (dtype, rank, static/dynamic dims, class axis) and tensor names are discovered rather than hard-coded; incompatible models fail at load time with a typed error and the engine refuses to guess layout transforms. | `src/runtime/metadata.rs:44-133`; `src/models/deeplob.rs:20-101`; `src/models/transformerlob.rs:19-139`; docs: `running-a-model.md#1`, `phase-04` (HEAD: "does not guess a reshape, transpose, pad, or truncation"); 63 passing tests incl. fabricated-metadata cases. | **Needs benchmarks / weak as a research claim.** True and well-engineered, but it is software-engineering hygiene; novelty vs. just using `ort` directly is low. Becomes meaningful only if paired with a *measured* failure-mode study (e.g., how many mis-exported models are caught). |
| C2 | **Memory-safe-by-construction wrapper**: zero `unsafe` in the crate (`forbid`), typed errors with file/row/column context, no panics in the inference path. | `src/lib.rs:2`; `Cargo.toml:30-31`; 42-variant `AppError` (`src/error.rs`); 17 non-test `unwrap/expect`, none in `inference/ runtime/ tensor/ models/ data/ preprocessing/` (my count, `03_subsystems.md` §8). | **Strong as a factual property, weak as a contribution.** Must be scoped to "this crate"; all FFI `unsafe` is inside `ort`. Must be benchmarked against *safety cost* (is there any overhead vs. C++?) to be interesting. |
| C3 | **Reproducible, comparability-checked benchmark harness**: four timed boundaries (inference / preprocessing / end-to-end / load), raw samples retained, nearest-rank P50/90/95/99, population std, run metadata (CPU, RAM, rustc, git hash, model FNV-1a fingerprint), no-overwrite reports, and a `compare` command that *refuses* to compare reports whose model hash/EP/shape/runtime-version differ. | `src/benchmark/*`; `src/application/inference_service.rs:139-421,355-395`; `comparison.rs:26-45`; docs: `phase-06` (HEAD), `running-a-model.md#8-9`. | **Needs benchmarks.** Design is sound and aligns with the proposal's "apples-to-apples" aim, but weaknesses exist (literal runtime-version/thread strings `inference_service.rs:373,384`; same tensor reused `:191`; no process isolation; `ps` sampler). Strengthen by fixing these and showing the harness *detects* an apples-to-oranges comparison (the existing deck already contains one: ORT 1.24.4/1 thread vs 1.31/default threads — M-6). |
| C4 | **Cross-runtime benchmark of LOB inference (Rust/ORT vs C++/ORT vs Python/ORT [vs Candle/Burn])** — the proposal's headline. | Proposal PDF `docs/RustLOB.pdf` §1,4,7; C++ numbers in the IEEE PDF Table III; Rust harness above. Only Rust/ORT runs here. | **Needs benchmarks (most of it unbuilt).** Only the Rust leg exists; C++ leg is external and unreproducible from this repo; Python/Candle/Burn absent (M-7). The "first systematic benchmark" claim is also **unverified** (no literature search yet — Phase 3). |
| C5 | **Finding: language wrapper adds little when the same ORT does the work** (hypothesis from the proposal citing a 1.1× Qdrant result) — to be tested by measuring Rust-side overhead vs `session.run` alone and vs C++. | Structure: per-call Rust work is a 16 KB copy + 2 finite scans + output parse (`tensor/input.rs:39`, `validation.rs:18-26`, `output.rs:5-49`). No measurement exists. | **Needs benchmarks.** Potentially the most defensible *scientific* claim, because it is likely true and cheap to test, but it's currently a hypothesis. |
| C6 | **Atomic, no-clobber, crash-safe prediction output** (temp file → fsync → hard-link/rename; Drop cleanup). | `src/application/output_writer.rs:66-168`; 4 tests. | **Weak** (good engineering, not research). Fine for a "system description" paragraph only. |

**Not claimable** (no evidence): speedup over C++/Python by Rust; numerical equivalence with PyTorch/C++; accuracy of the models (not evaluated by this code — it writes predictions and true labels but computes no metrics); GPU/SIMD/quantization/KV-cache/attention work; "pluggable backends".

Recommendation (my judgment): lead with **C3 + C5 (+C4 if the other runtimes get built)** and describe C1/C2/C6 as system design. If the aim is a *full* paper, the experimental work in §6 is mandatory.

## 2. Existing performance data (quoted exactly with source)

### 2a. The only run artifact on disk — `outputs/benchmarks/presentation-translob-next.json` (untracked, gitignored; file mtime 2026-11-20 17:09 IST)
Metadata (fields verbatim): `model: "TransformerLOB"`, `model_file: "models/transformerlob.onnx"`, `model_file_hash: "fnv1a64:e44e5de52db2cf20"`, `mode: "inference"`, `input: "data/raw/FI2010_test.csv"`, `execution_provider: "cpu"`, `runtime_initialization_ms: 12.838624999999999`, `initial_model_load_ms: 711.1558329999999`, `warmup_iterations: 20`, `measured_iterations: 200`, `trial_count: 5`, `runtime_version: "ONNX Runtime 1.31; ort crate 2.0.0-rc.15"`, `rust_version: "rustc 1.97.1 (8bab26f4f 2026-07-14)"`, `build_mode: "release"`, `operating_system: "macos aarch64"`, `cpu: "Apple M3"`, `ram_bytes: 8589934592`, `thread_configuration: "ONNX Runtime defaults; custom thread counts are not exposed"`, `git_commit: "bdd807f"`, `memory/cpu_*: null` (no profiling).

Aggregate (1,000 samples), verbatim: `mean_ms 7.799348764000003`, `standard_deviation_ms 9.65097457695395`, `minimum_ms 1.713791`, `maximum_ms 43.895916`, `p50_ms 2.142166`, `p90_ms 22.013291000000002`, `p95_ms 29.297792`, `p99_ms 39.522083`.

Per-trial (verbatim from the JSON):

| Trial | elapsed_seconds | throughput/s | mean_ms | p50_ms | p99_ms | max_ms |
|---|---|---|---|---|---|---|
| 1 | 0.79047325 | 253.01 | 3.9523 | 2.0423 | 20.5379 | 20.73 |
| 2 | 2.142483 | 93.35 | 10.7123 | 4.5628 | 42.5163 | 43.90 |
| 3 | 1.108158083 | 180.48 | 5.5407 | 1.9858 | 35.0453 | 38.51 |
| 4 | 3.270834375 | 61.15 | 16.3539 | 15.1804 | 40.7122 | 41.71 |
| 5 | 0.487531334 | 410.23 | 2.4376 | 1.8613 | 8.4471 | 8.85 |

Derived by me (labelled as such): total 1,000 calls / 7.799 s ≈ **128 windows/s** aggregate; between-trial mean spans 2.44–16.35 ms (6.7×). The distribution is bimodal/very noisy relative to the quiet floor (~1.7–2.1 ms). Cause **[UNCLEAR]** — candidates (not tested): background load while presenting, thermal/power state on an 8 GB M3, ORT default thread oversubscription, OS scheduling. This run **cannot** support a latency claim.

### 2b. `outputs/deeplob-predictions.csv` (319 rows; input `data/raw/FI2010_test.csv`, not in repo)
Derived by me from the CSV (not produced by any repo code): 187/319 = **58.6 %** windows with `predicted_class == true_class`; true-class counts {Down 46, Stationary 215, Up 58}; predicted {Down 44, Stationary 248, Up 27}. Predicting "Stationary" always would score 215/319 = 67.4 % on this slice, i.e. the model is *below the majority baseline* on this metric; macro-F1/per-class recall have not been computed [TODO]. Rows 319 × 100 = 31,900 ≈ the 31,937 test rows quoted by the IEEE paper (consistent but unverified). The CSV's mtime is 10 Nov 15:27 and the commit that added the model files (`bdd807f`) is 10 Nov 15:57, so whether this CSV was produced by the committed `deeplob.onnx` is [UNCLEAR].

### 2c. `ppt/index.html` (untracked deck) — quoted as-is, **source reports not found**
- "Mean latency · lower is faster **0.799 ms** DeepLOB; **1.785 ms** TransformerLOB … DeepLOB · P99 **0.915 ms** · 1,000 timed samples · **1,251 windows/s**; TransformerLOB · P99 **1.913 ms** · 1,000 timed samples · **560 windows/s** … Release build · CPU · ONNX Runtime 1.31 · 20 warm-ups · 5 × 200 timed calls … Recorded benchmark reports · inference-only timing; input tensor reused" (slide 5).
- "**0.854** C++ run Apple M3 … **1.785** Rust run Apple M3 … **1.159** Paper C++ Table III … Fresh C++: 0.854 ms mean, 0.939 ms P99 · ORT 1.24.4 · one thread · synthetic input · times Session.Run. Rust: 1.785 ms mean, 1.913 ms P99 · ORT 1.31 · default threads · CSV-derived input · times Rust predict path." (slide 6).
- **Conflict:** the only Rust report on disk for the same model/commit gives mean 7.80 ms / P99 39.5 ms, not 1.785 / 1.913; no report for DeepLOB (0.799 ms) exists in the repo. 1,251/s = 1/0.799 ms and 560/s = 1/1.785 ms, i.e. the throughput figures are reciprocals of the mean, not measured loop throughput. → **Do not cite slide numbers until a result file is produced and committed.** [TODO: ask the user where the deck's source JSONs are.]

### 2d. Upstream C++ paper (`docs/DeepLOB_HFT_IEEE_Paper….pdf`)
See `01_docs_digest.md` (Table III 1.159 ms / 2.033 ms / 1.75×; Table IV 1.090 ms, 917.33 win/s, accuracy 33.00 %). Hardware and ORT version unknown to me.

## 3. Limitations and gaps (all grounded)
- **Scope:** CPU only; batch = 1; fixed 100×40 float32; non-overlapping windows only; two models; no GPU/quantization/threads configuration (`settings.rs:156,178,232`; `inference_service.rs:384`).
- **Not implemented vs. own spec:** Phase 7 numerical equivalence (abs tol 1e-5) — none; evaluation metrics (accuracy/F1/confusion) — none (the `predict` CSV carries `true_class` but nothing scores it); spec's `evaluation/` module and `tests/equivalence_tests.rs` don't exist.
- **Labels required** at inference time — the loader errors on a missing label column even though prediction doesn't need it (`running-a-model.md#3`; `labels.rs:28-43`).
- **`schema_verified` gate** means the default config cannot run; column 144 / 0–39 mapping is a "candidate" and was never verified in-repo against real FI-2010 CSV (absent).
- **TODO/FIXME markers:** `grep -rnE "TODO|FIXME|XXX|HACK|unimplemented|todo!" src tests examples` → **none**. Gaps are documented in prose instead (docs "Limitations" sections).
- **Hard-coded/misleading strings:** `"Failed predictions: 0"` literal (`commands.rs:86`); runtime version and thread-config literals (`inference_service.rs:373,384`); error text "Phase 1 supports only 'cpu'" (`error.rs:23`); "Phase 2" wording in validation errors (`settings.rs:180,234,257`; `window.rs:16`) — phase-numbered messages leak into user-facing output.
- **Benchmark validity risks:** same tensor reused every call (cache/branch-predictor-friendly, not data-dependent); no input diversity; no process isolation; no pinning/frequency control; in-process trials share allocator/ORT arena state; `ps` sampler perturbs; macOS-only evidence so far (Apple M3, 8 GB).
- **Provenance gaps:** how `data/normalization/{mu,sigma}.npy` and the `.onnx` files were produced (training seed, epochs, split) isn't in this repo; the C++ paper says the DeepLOB model was a 15-epoch model with 33 % top-1 on a 500-window slice.
- **Single-run evidence:** exactly one benchmark JSON exists and it is noisy.
- **Dependency risk:** `ort =2.0.0-rc.15` (release candidate API); ORT binary downloaded at build time (needs network; reproducibility of the exact ORT build is not recorded beyond the literal string).

## 4. Docs-vs-code (and docs-vs-docs) mismatches

| ID | Statement | Where | What the code/files show |
|---|---|---|---|
| M-1 | Brief: "Rust LLM inference engine" (GGUF, KV cache, SIMD, sampling …) | task prompt | It's a LOB classifier harness over ORT; none of that exists (see `00_inventory.md`). |
| M-2 | "No ONNX models, normalization arrays, or real datasets are included." | `README.md` (Current limitations); also phases 1–6 docs (HEAD) | `models/{deeplob,transformerlob}.onnx`, `transformer.onnx.data`, `data/normalization/{mu,sigma}.npy` etc. are tracked since `bdd807f` and both models load and run (`inspect`/`smoke-test`, my runs). Raw FI-2010 CSV is still absent. |
| M-3 | README links "Benchmark methodology: docs/phase-06-implementation.md" | `README.md` (Documentation) | File deleted in the working tree (still in HEAD). |
| M-4 | Phase docs report 13/36/43/46/59 tests | phase 1–5 docs (HEAD) | Current suite is 63 (27+36). Not a contradiction per se (snapshots), but no doc states the current count. |
| M-5 | `.gitignore` ignores `*.onnx` and `/data/normalization/*.npy` | `.gitignore` | Those files are tracked (committed before the ignore rule; `git ls-files`). Fresh clones get them; newly produced ones would not be committed. |
| M-6 | Deck slide 5: Rust TransformerLOB mean 1.785 ms, P99 1.913 ms; DeepLOB 0.799 ms | `ppt/index.html` | Only report on disk: mean 7.80 ms, P99 39.5 ms (min 1.714 ms). No DeepLOB report. The deck's own slide 6 also flags ORT 1.24.4/1 thread (C++) vs 1.31/default (Rust). |
| M-7 | Proposal: six runtimes (PyTorch, ORT-Python, ORT-C++, ORT-Rust, Candle, Burn), Criterion benches, training + ONNX-export scripts, 10⁻⁵ equivalence, P999, cold-start, allocation count, Mann–Whitney | `docs/RustLOB.pdf` §7 | Repo has only Rust/ORT, no Criterion, no training/export, no equivalence test, no P999/allocation/statistical-test code (percentiles stop at P99, `statistics.rs:40-43`). Cold-start ≈ the `load` mode (`inference_service.rs:311-329`) only. |
| M-8 | Proposal: "exported with opset 11 and fixed input shape [1,1,100,40]" | `docs/RustLOB.pdf` §7.5 | Both committed ONNX models report a dynamic batch dim (`[-1,1,100,40]`) (`inspect`). Opset not checked [TODO]. |
| M-9 | Spec: model files `deeplob.onnx`, `transformerlob.onnx` only | `llm_context/context.md` §6 | A third file `transformer.onnx.data` is required by `transformerlob.onnx` (external weights); undocumented, and not matched by `.gitignore`. |
| M-10 | Spec tech stack: "Tensor representation: ndarray / contiguous buffers" | `llm_context/context.md` §3 | Contiguous `Vec<f32>` only; ndarray not used (phase-01 doc says ndarray support disabled). Consistent with "or", but ndarray isn't present. |
| M-11 | Output scores called "class scores"; user docs list "three scores" | `running-a-model.md#7`, `README.md` | They are raw model outputs (logits per observed negatives), no softmax (phase-03 doc says so; README doesn't). Do not call them probabilities. |
| M-12 | Proposal timeline: submit to ICAIF 2026 by 2 Aug 2026 | `docs/RustLOB.pdf` §9 | Today is 2026-11-22; deadline passed; fallbacks named (ICAIF workshop Oct 2026, IEEE CIFEr) also look past. [UNCLEAR: current target venue.] |
| M-13 | README: ONNX Runtime "1.31" | `README.md` | Value is hard-coded as a string (`inference_service.rs:373`) and the build only selects `api-31`; I did not verify the downloaded binary's version [TODO: print `ort::info()` or equivalent]. Deck says C++ used 1.24.4. |

## 5. Experiments needed in Phase 2 (per claim)

| Claim | Metric(s) | Models | Baselines / comparators | Notes (from §3 weaknesses) |
|---|---|---|---|---|
| C5 / C4 (Rust overhead, cross-runtime) | `session.run`-only latency vs full `predict()` latency (isolates Rust overhead); mean/P50/P95/P99/P999, std, 95 % CI (bootstrap), Mann–Whitney U; throughput; cold start (process start→first prediction); peak RSS; allocations/call | DeepLOB, TransformerLOB | (a) C++/ORT *same ORT version, same thread count, same input tensors*; (b) Python/ORT; (c) optional Candle/Burn; (d) PyTorch eager | Fix first: print real ORT version; expose `intra_op_threads`; run ≥30 trials in **separate processes**; vary inputs (many windows, not one); pin/record power mode; ≥2 machines (the deck's M3 8 GB + an x86 Linux box) |
| C3 (harness quality) | Detection test: feed deliberately mismatched reports to `compare` and show refusal; variance decomposition (within vs between trial); effect of `--profile-*` sampler on latency | both | Harness with sampler off vs on; with/without warm-up | Needs ≥1 clean baseline first (current run is noisy, §2a) |
| C1 (contract validation) | Detection rate on a set of intentionally broken exports (wrong dtype, rank, layout NHWC, class count, fixed batch>1) | synthetic + 2 real | Naive `ort` usage (no validation) | Cheap: generate mutated ONNX via `onnx` Python |
| C2 (safety) | Clippy/`cargo-geiger` for dependency `unsafe` count; fuzz CSV loader/NPY reader (`cargo fuzz`); panic-freedom on malformed input | n/a | C++ reference loader (ASan/UBSan findings) | Scope claim to crate boundary |
| Numerical equivalence (gates any comparison) | max/mean abs diff of logits vs PyTorch and vs C++ on N windows (spec tolerance 1e-5); argmax agreement % | both | PyTorch reference + C++ ORT | Spec Phase 7; currently zero evidence. Hardest prerequisite |
| Model quality (context only) | accuracy, macro-F1, per-class recall, confusion matrix on full test set | both | Paper's DeepLOB recall (Table II), majority class | Current 58.6 % vs 67.4 % majority on a 319-window slice (§2b) |
| Batching (optional extension) | latency/throughput vs batch ∈ {1,8,32} (models are dynamic-batch) | both | batch=1 | Requires relaxing the hard-coded `[1,1,100,40]` guards (≥8 places, `03_subsystems.md` §8) |
