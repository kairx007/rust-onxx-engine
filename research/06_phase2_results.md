# 06 — Phase 2 results (benchmarks)

Run: `results/phase2/20261122-full-ac/` (summary: `summary.md`; every number below is in `tables/*.csv`).
Machine: Apple M3 (4P+4E), 8 GiB, macOS 26.5.2, AC power, Low Power Mode **off**, 1-min load ≈ 2.5–3 (desktop apps open).
Protocol: 335 processes, one trial per fresh process, seed-randomized interleaved order, 1 s cool-down,
20 warm-up + 200 timed calls per process, 10 processes per configuration. 0 failures.
Harness and reproduction: `bench/README.md`.

## Headline findings

1. **The "CPU-only" Rust engine was not CPU-only (bug, fixed).** `ort` 2.0.0-rc.15 (api-22+) silently sets
   ONNX Runtime's `MaxEfficiency` auto-device policy on every session. On Apple silicon, ORT then placed
   2 DeepLOB and 14 TransformerLOB nodes on `CoreMLExecutionProvider` (ORT trace logs,
   `tables/node_placement.csv`), even though only the CPU provider was registered and reports said `cpu`.
   Effects: logits off by up to **3.0e-3** (DeepLOB) / **9.5e-4** (TransformerLOB) — 300× / 95× over the
   1e-5 spec — and **1.19–3.44× slower** inference, **4–14× slower cold start**. Fix: `device_policy = "cpu"`
   (default) → `AutoDevicePolicy::PreferCPU` (`src/runtime/environment.rs`), regression test
   `tests/model_loading_tests.rs`. Consequence: all pre-fix Rust numbers (incl. the deck's 1.785 ms) measured
   a CoreML-offloaded configuration.

2. **Numerical equivalence holds (after the fix).** On all 319 windows × 3 logits, Rust (bundled ORT),
   Rust + Microsoft ORT dylib, Python onnxruntime 1.31.0 and 1.32.0 are **bit-identical at 1 intra-op thread**.
   At default threads every CPU engine drifts by the *same* ≤ 5.7e-6 (parallel-reduction order), inside 1e-5.
   Argmax agreement 100% everywhere (`tables/numerical_equivalence.csv`, fig 8).

3. **Language overhead is negligible.** Time outside `Session::run` per call: Rust 1.8–2.8 µs, Python
   4.2–5.3 µs (P50) — < 1% of a 0.3–0.9 ms inference (fig 3). At 1 thread, Rust and Python fast-mode medians
   are equal within ±1.4% (DeepLOB 0.670 vs 0.680 ms; TransformerLOB 0.894 vs 0.897 ms).

4. **At default threads, the *binary* matters, not the language.** The `ort`-bundled ORT 1.31.0 build is slower
   multi-threaded than Microsoft's build of the same commit (402979a): DeepLOB 0.312 vs 0.280 ms
   (Python 0.897×, p = 4.6e-5), TransformerLOB 0.511 vs 0.341 ms (0.668×, p = 0.0025). The **same Rust code**
   loading Microsoft's dylib matches Python (0.275 / 0.337 ms), so this is the bundled binary's threading,
   not Rust (fig 10). Tail: bundled Rust has the tightest DeepLOB P99 (0.353 ms vs Python 1.321 ms).

5. **Cold start: Rust 3× faster.** Spawn → first prediction → exit: Rust 34 / 42 ms vs Python 112 / 114 ms
   (DeepLOB / TransformerLOB), dominated by interpreter + import (fig 5, `tables/cold_start.csv`).

6. **Bimodal machine state.** In every configuration and engine, per-process medians split into two tight
   clusters ~1.7× apart ("slow mode", ~25% of processes; Rust 26%, Python 25%, Fisher p = 1.0). Not E-core
   QoS (`taskpolicy -c utility` stays fast; `-b` gives 4.4 ms), correlated with *lower* system load
   (p = 2e-7) → likely CPU-frequency state; not reproduced in a 12-run follow-up. Primary statistics therefore use
   fast-mode medians; all-process figures are reported alongside. Methodological point for the paper: with
   n = 10 processes, an engine-independent bimodality alone produced an *apparent* 1.7× "Rust is slower"
   gap (all-process median 1.166 vs 0.680 ms).

7. **Model quality is a data/model issue, identical across engines.** On the 319 test windows:
   DeepLOB accuracy 58.6% (macro-F1 0.362) — below the 67.4% majority-class baseline; **TransformerLOB 18.2%**
   (macro-F1 0.189), far below chance — strongly suggests the TransformerLOB ONNX export or its expected
   preprocessing/label order does not match this pipeline. [TODO: verify with the training code.]

## Other measurements
- Graph optimization (Rust, 1 thread): disabling it costs +10% (DeepLOB 0.738 vs 0.670 ms; TransformerLOB
  0.989 vs 0.894); `all` ≈ ORT default (fig 6).
- Input policy (`first` vs `cycle`): no material difference at 1 thread; data-dependent timing is not a factor
  for these models.
- Rust preprocessing of the full test set (31,937 rows): CSV parse 42.3 ms ≫ z-score 0.70 ms, tensors 0.49 ms,
  windows 0.007 ms (fig 7). End-to-end CSV → 319 predictions: 143 ms (DeepLOB), 206 ms (TransformerLOB).

## Caveats
- One machine (M3, 8 GiB) with desktop apps running; slow-mode cause undetermined.
- `rust-msdylib` reports in this run carry the bundled runtime-version *string* (fixed afterwards in code);
  their `runtime_build_info` correctly shows Microsoft's build (`git-branch=HEAD`).
- The test CSV is reconstructed from z-scored arrays (round-trip error ≤ 4.8e-7); the raw FI-2010 file is not in the repo.
- C++ engine not yet included.

## Claims now supportable (update to `04_paper_evidence.md`)
- C5 (language overhead small when the same ORT does the work): **strong** — µs-level overhead, 1-thread parity, bit-identical logits.
- New: hidden device-policy defaults in a Rust binding silently change hardware, precision, and latency;
  a placement check must be part of any cross-runtime benchmark — **strong**, with ORT trace evidence.
- New: binding-bundled vs vendor ORT binaries differ by up to 1.5× multi-threaded — **strong** on this machine; needs a second machine.
- C3 (harness): now demonstrated — it caught the CoreML offload and the bimodality.
