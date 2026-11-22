# 00 — Inventory

> **Scope correction (read first).** The Phase-1 brief describes an "LLM inference engine" (GGUF/safetensors, tokenizer, KV cache, SIMD kernels, quantization, sampling). **This repository is not that.** It is **RustLOB**: a CLI (`lob-inference`) that runs two *pre-exported ONNX classifiers* (DeepLOB, TransformerLOB) on limit-order-book windows from the FI-2010 dataset, through the `ort` crate (ONNX Runtime bindings), CPU only. All tensor math happens inside ONNX Runtime; no kernels, attention, or quantization exist in the Rust code. Evidence of absence: a case-insensitive grep over `src/ tests/ examples/ Cargo.toml build.rs configs/` for `gguf|safetensors|simd|avx|neon|cuda|metal|coreml|kvcache|quantiz|int8|fp16|tokeniz|rayon|mmap|flash|rope|attention|sampling|temperature` returns only two test lines that assert `"cuda"` is *rejected* (`tests/runtime_tests.rs:15`, `tests/config_tests.rs:39`). The brief's LLM-specific subsystem questions are therefore answered "not present" in `03_subsystems.md`; the paper framing needs to be decided (see `04_paper_evidence.md` and README open questions).

Snapshot: branch `main`, HEAD `bdd807f` (2026-11-10). Working tree has uncommitted changes (see §7).

## 1. Package / build

| Item | Value | Evidence |
|---|---|---|
| Package | `rust-engine` 0.1.0, edition 2024, MIT; "Rust ONNX Runtime foundation for DeepLOB and TransformerLOB" | `Cargo.toml:1-6` |
| Workspace | **None** — single package, single crate (lib `rust_engine` + bin) | `Cargo.toml` has no `[workspace]` |
| Binary | `lob-inference` → `src/main.rs` | `Cargo.toml:8-10` |
| Library | `rust_engine` (`src/lib.rs`), `#![forbid(unsafe_code)]`, `#![deny(missing_debug_implementations)]` | `src/lib.rs:2-3` |
| Example | `examples/preprocessing_demo.rs` (23 lines) | file |
| Toolchain | `channel = "stable"`, clippy + rustfmt | `rust-toolchain.toml` |
| README requirement | "Rust stable 1.92 or later" | `README.md` (Requirements) |
| Release profile | `opt-level=3`, `lto="thin"`, `codegen-units=1` | `Cargo.toml:25-28` |
| Lints | `unsafe_code = "forbid"` (also in manifest) | `Cargo.toml:30-31` |
| Feature flags | **None defined** (`[features]` absent). Only `ort`'s features are selected. | `Cargo.toml` |
| Build targets | macOS gets a custom `build.rs` that locates Apple Clang's `libclang_rt.osx.a` and links it with `-mmacosx-version-min=13.4`, because the downloaded static ONNX Runtime needs Apple's availability helper. Other OSes: `build.rs` returns immediately. | `build.rs:4-37`; `docs/phase-01-implementation.md#dependency-choices` (HEAD) |
| Tests / benches | 63 tests pass (27 lib-unit + 36 integration; `cargo test --offline`, run by me 2026-11-22). **No `benches/` directory, no Criterion.** | see `03_subsystems.md` §9 |

## 2. Dependencies (and why each matters)

From `Cargo.toml:12-23`; resolved versions from `Cargo.lock`.

| Crate | Req. | Resolved | Role in this project |
|---|---|---|---|
| `ort` | `=2.0.0-rc.15`, `default-features=false`, features `std, tracing, download-binaries, tls-native, api-31` | 2.0.0-rc.15 (+ `ort-sys` same) | **The inference backend.** Rust bindings to ONNX Runtime; downloads a prebuilt ORT binary at build time. Pinned exactly because 2.0 is an RC (`phase-01` doc). `ndarray` is a transitive dep of `ort` in `Cargo.lock` but ndarray *support* is not used by the project's own code. |
| `csv` | 1.4 | 1.4.0 | FI-2010 CSV parsing (`src/data/csv_loader.rs`) and prediction/benchmark CSV output. |
| `npyz` | 0.9 | 0.9.1 | Reads `mu.npy` / `sigma.npy` normalization vectors (`src/preprocessing/normalization.rs:85`). |
| `clap` | 4.5 (derive) | 4.6.7 | CLI (`src/cli/args.rs`). |
| `serde`, `serde_json` | 1.0 | — | TOML config (`deny_unknown_fields`) and JSON benchmark reports. |
| `toml` | 0.9 | 0.9.12 | Config parsing. |
| `thiserror` / `anyhow` | 2.0 / 1.0 | — | Typed `AppError` (`src/error.rs`) in the library; `anyhow` only at the CLI boundary (`src/cli/commands.rs`). |
| `tracing`, `tracing-subscriber` | 0.1 / 0.3 (`env-filter`,`fmt`) | — | Logging (`src/logging.rs`, 12 lines). |

**Not dependencies** (the brief expected them): `rayon`, `half`, `memmap2`, any CUDA/Metal/CoreML crate, `candle`, `burn`, `tch`, `criterion`. The project proposal PDF (`docs/RustLOB.pdf`) lists Candle, Burn and `tch` as planned runtimes; none are in the code (see mismatch M-7).

Execution providers: only CPU. `Runtime::initialize` rejects anything but `"cpu"` (`src/runtime/environment.rs:26-28`); config validation does too (`src/config/settings.rs:156-160`). Thread counts: not configured anywhere (`grep intra|inter_op|with_.*threads` in `src/` → no hits); the benchmark metadata literally records `"ONNX Runtime defaults; custom thread counts are not exposed"` (`src/application/inference_service.rs:384`).

## 3. Crates / modules

One crate; modules under `src/` (`src/lib.rs:5-16`):

| Module | Purpose | LOC (`wc -l`, all lines incl. tests) |
|---|---|---|
| `application` | Workflows: predict / smoke-test / benchmark, model factory, atomic CSV writer, diagnostics | 1110 |
| `benchmark` | Latency statistics, report (JSON/CSV), `ps`-based resource sampler, Rust-vs-C++ comparison | 681 |
| `cli` | `clap` args + command dispatch | 673 |
| `config` | TOML `Settings` + validation | 282 |
| `data` | CSV loader, feature extractor, label codec, `LobDataset`/`LobSegment` | 398 |
| `inference` | `DeepLobEngine`, `TransformerLobEngine`, shared `runner`, input adapter, output parser, `Prediction` | 432 |
| `models` | `ModelType`, `ModelRegistry`, `DeepLobContract`, `TransformerLobContract` | 435 |
| `preprocessing` | Z-score `Normalizer`, `WindowExtractor`, `PreprocessingPipeline` | 373 |
| `runtime` | ORT `Environment` singleton, `ModelSession`, tensor-metadata extraction/validation | 350 |
| `tensor` | `InputTensor` (`[1,1,100,40]` f32) + validation | 104 |
| top-level (`main.rs`, `lib.rs`, `error.rs`, `logging.rs`) | | 204 |
| **`src/` total** | | **5042** (of which ≈772 are inline `#[cfg(test)]` modules) |
| `tests/` | 10 integration files + `common/mod.rs` | 826 |

(`tokei` is not installed; counts are `/usr/bin/wc -l`, which counts blanks/comments. Note: in this shell `wc` is shadowed by an alias to another tool, so I invoked `/usr/bin/wc`.)

## 4. CLI surface (`src/cli/args.rs:22-114`)

`runtime-info`, `inspect`, `validate`, `smoke-test`, `predict`, `benchmark` (modes: `inference | preprocessing | end-to-end | load`), `compare` (Rust-vs-C++ JSON reports). Global: `--config` (default `configs/default.toml`), `--log-level`.

## 5. Configuration (`configs/default.toml`, `src/config/settings.rs`)

`deny_unknown_fields` everywhere. Validation **hard-codes the model contract**: input shape must be `[1,1,100,40]` (`settings.rs:178`), classes = 3 (`:183`), window length = stride = 100 (`:232`), epsilon exactly `1e-8` (`:237`), tensor dtype `float32` (`:260`). The "configurable" shape fields exist but cannot take any other value. Default feature columns 0..39, label column 144, `has_headers=false`, `schema_verified=false` (gate: preprocessing refuses to run until the user flips it, `src/preprocessing/pipeline.rs:21`).

## 6. Assets in the repo

| Path | Size | Tracked? | Notes |
|---|---|---|---|
| `models/deeplob.onnx` | 281 KB | tracked (commit `bdd807f`) although `.gitignore` now has `*.onnx` | Loads; input `[-1,1,100,40]` f32 named `input`, output `[-1,3]` named `output` (my `inspect` run, 2026-11-22) |
| `models/transformerlob.onnx` | 200 KB | tracked | Loads; input `[-1,1,100,40]`, output `[1,3]` (static!) |
| `models/transformer.onnx.data` | 476 KB | tracked | External-weights file; `transformerlob.onnx` contains the string `transformer.onnx.data` as its external-data `location`. File name does not match the model name. |
| `data/normalization/{mu,sigma}.npy` | 288 B each | tracked (despite `.gitignore` `/data/normalization/*.npy`) | 40-vectors |
| `data/normalization/{X,Y}.npy`, `FI2010_{train,test}_..._n50000_w100_z.npz`, `zscore_stats.npz` | 8.0 MB / 400 KB / 1.5 MB / 733 KB / 717 B | tracked | Pre-windowed, z-scored FI-2010 arrays. **Not read by any Rust code** (grep: only `mu.npy`/`sigma.npy` are loaded). Provenance undocumented → [UNCLEAR]. |
| `data/raw/` | `.gitkeep` only | — | The FI-2010 CSV that the README/output CSV refer to (`data/raw/FI2010_test.csv`) is **not in the repo**. |
| `outputs/deeplob-predictions.csv`, `outputs/benchmarks/presentation-translob-next.json` | 27 KB / 25 KB | **untracked & gitignored** (`/outputs/`) | The only real run artifacts on disk; see `04_paper_evidence.md` §2. |
| `context/`, `llm_context/` | spec docs (7 phase specs + context) | tracked (`llm_context/` has its own `.gitignore` containing `context`) | Authoring specs (prompts for a code-generation agent). `context/*` and `llm_context/*` are byte-identical for `context.md` and `phase-01.md` (`diff -q`). |
| `ppt/index.html` | 23 KB | untracked | 8-slide HTML deck with benchmark numbers (digested in `01_docs_digest.md`). |
| `docs/RustLOB.pdf`, `docs/DeepLOB_HFT_IEEE_Paper … .pdf` | 183 KB / 984 KB | untracked | Project proposal; the upstream C++ paper. |

## 7. Working-tree state (affects what "docs/" means)

`git status` at session start: `.gitignore` modified; **`docs/phase-01…06-implementation.md` deleted (uncommitted)**; the two PDFs, `docs/` additions and `ppt/` untracked. Those six phase docs exist at `HEAD` and I read them via `git show HEAD:docs/phase-0N-implementation.md`; citations to them below are marked "(HEAD)". `README.md` still links `docs/phase-06-implementation.md`, which is absent from the working tree (mismatch M-3). I did not modify any existing file; I did run `cargo build --offline`/`cargo test --offline`, which only write to the already-ignored `target/`.

## 8. Git history (52 commits)

Phase-wise feature branches merged one by one: runtime foundation → preprocessing → DeepLOB engine → TransformerLOB engine → CLI → benchmark suite → docs/spec → README refresh → `fix(data): accept integral decimal labels` (`2422326`) → `chore: add model and normalized benchmark data` (`bdd807f`). No release tags examined.
