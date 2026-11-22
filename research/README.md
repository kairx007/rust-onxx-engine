# research/ — Phase 1 findings (paper drafting)

Generated 2026-11-22 from commit `bdd807f` (+ uncommitted working tree). Only files under `research/` were created; no existing source/docs file was modified (I ran `cargo build/test --offline` and the `lob-inference inspect/smoke-test/runtime-info` commands, which write only to the git-ignored `target/`).

> **The repo is not an LLM inference engine.** It is *RustLOB*: a Rust CLI that runs pre-exported DeepLOB / TransformerLOB **ONNX** classifiers on FI-2010 limit-order-book windows via ONNX Runtime (`ort`), CPU only. There is no GGUF/safetensors, tokenizer, KV cache, SIMD/GPU kernel, quantization, attention code, or sampler. Those parts of the brief are recorded as "not present" with evidence.

| File | Contents |
|---|---|
| [`00_inventory.md`](00_inventory.md) | Scope correction, manifest, dependencies and why they matter, modules + LOC, assets, working-tree state |
| [`01_docs_digest.md`](01_docs_digest.md) | Digest of README, `running-a-model.md`, phase docs 1–6 (from git HEAD), proposal PDF, C++ IEEE paper PDF, spec files, slide deck |
| [`02_architecture.md`](02_architecture.md) | Module-architecture and request-flow Mermaid diagrams; step-by-step trace with `file:line` |
| [`03_subsystems.md`](03_subsystems.md) | Subsystem deep dives (present / delegated-to-ORT / absent), unsafe count, hot-path allocations, tests, benchmark harness |
| [`04_paper_evidence.md`](04_paper_evidence.md) | Candidate contributions with strength ratings, all existing perf data quoted, limitations, 13 docs-vs-code mismatches, Phase-2 experiment plan |
| [`05_visual_candidates.md`](05_visual_candidates.md) | Figure/table candidates with drafts |
| [`06_phase2_results.md`](06_phase2_results.md) | Phase 2 benchmark results: CoreML-offload bug, numerical equivalence, overhead, binary/threads, cold start, caveats |
| [`07_cpp_engine_and_data_checks.md`](07_cpp_engine_and_data_checks.md) | C++ engine build/reproduction, three-engine results, data and artifact checks |
| [`08_final_results.md`](08_final_results.md) | **Final Rust vs C++ results** (supersedes 06/07 numbers): latency, same-binary comparison, cold start, numerics, CoreML pitfall |

Conventions: `path:line` = code (verified against the working tree); `docs/<file>#section` = docs; "(HEAD)" = file deleted in the working tree but read from git; `[TODO: …]` = work to do; `[UNCLEAR: …]` = could not determine; `M-n` = mismatch IDs in `04_paper_evidence.md` §4. Numbers marked "derived by me" were computed from repo artifacts and are not repo-produced.
