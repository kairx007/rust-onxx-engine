# PHASE 01 — Project Foundation & ONNX Runtime Integration

**Project:** RustLOB  
**Phase:** 01 of 07  
**Language:** Rust  
**Inference Backend:** ONNX Runtime  
**Rust Binding:** `ort`  
**Status:** Not Started

---

# 1. Phase Objective

Establish the foundational architecture of RustLOB and successfully integrate ONNX Runtime through the Rust `ort` crate.

By the end of this phase, the project must support:

1. Cargo workspace and project initialization.
2. ONNX Runtime initialization.
3. DeepLOB and TransformerLOB model loading.
4. Model input/output metadata inspection.
5. Model session management.
6. Configuration management.
7. Structured logging and error handling.
8. CLI commands for runtime and model inspection.
9. Unit and integration tests.
10. A clean, traceable Git commit history.

This phase focuses exclusively on infrastructure and runtime integration.

Do not implement actual inference preprocessing, tensor construction, benchmarking, CSV processing, or model evaluation.

---

# 2. Reference Implementation

The project is a Rust replacement for the C++ inference engine described in:

**DeepLOB-HFT-Cpp: A Production-Grade Deep Learning Pipeline for Limit Order Book Mid-Price Prediction with High-Performance C++ Inference.**

The C++ implementation uses ONNX Runtime and provides three executables:

- `inference.exe`
- `benchmark.exe`
- `full_inference.exe`

Our eventual Rust implementation will reproduce these workflows.

For Phase 1, implement the underlying runtime initialization, model loading, and metadata inspection capabilities.

---

# 3. Technology Stack

| Technology | Purpose |
|---|---|
| Rust stable | Primary language |
| Cargo | Build system |
| ort | ONNX Runtime bindings |
| ONNX Runtime | Inference backend |
| clap | CLI argument parsing |
| serde | Serialization |
| toml | Configuration |
| anyhow | Application-level errors |
| thiserror | Custom errors |
| tracing | Structured logging |
| tracing-subscriber | Logging output |

Requirements:

- Verify compatibility between `ort` and ONNX Runtime.
- Use stable Rust.
- Check actual crate APIs before implementation.
- Avoid outdated API examples.
- Commit `Cargo.lock`.
- Avoid unnecessary dependencies.
- Document dependency choices.

---

# 4. Git Workflow — Mandatory

Git version control is mandatory throughout the project.

Codex must maintain a clean and professional Git history.

## 4.1 Branch Naming Convention

Use the following naming convention:

```text
<type>/<phase>-<short-description>
```

Supported types:

| Type | Purpose |
|---|---|
| `feat/` | New functionality |
| `fix/` | Bug fixes |
| `refactor/` | Code restructuring |
| `test/` | Testing improvements |
| `docs/` | Documentation |
| `chore/` | Configuration and maintenance |

### Phase 1 branch

Create:

```text
feat/phase-01-runtime-foundation
```

All Phase 1 implementation work must happen on this branch.

Do not work directly on `main`.

## 4.2 Commit Message Convention

Follow Conventional Commits:

```text
<type>(<scope>): <description>
```

Examples:

```text
chore(project): initialize Rust project structure

build(deps): integrate ONNX Runtime dependencies

feat(config): implement TOML configuration loader

feat(runtime): initialize ONNX Runtime environment

feat(models): add DeepLOB and TransformerLOB registry

feat(runtime): implement ONNX session management

feat(runtime): add model metadata inspection

feat(cli): add runtime information command

feat(cli): implement model inspection and validation

test(runtime): add ONNX integration tests

docs(phase-01): document runtime architecture
```

Commit messages must:

- Use lowercase descriptions.
- Be concise and meaningful.
- Describe the actual changes.
- Avoid generic messages such as `update`, `changes`, or `final code`.
- Never claim functionality that hasn't been implemented.

## 4.3 Mandatory Commit Checkpoints

Make commits incrementally after completing these milestones:

| Checkpoint | Commit |
|---|---|
| Project initialization | `chore(project): initialize Rust project structure` |
| Dependencies and toolchain | `build(deps): configure ONNX Runtime dependencies` |
| Configuration and errors | `feat(config): implement configuration and error handling` |
| Logging | `feat(logging): add structured logging infrastructure` |
| Runtime initialization | `feat(runtime): initialize ONNX Runtime environment` |
| Model registry | `feat(models): implement model registry` |
| Session management | `feat(runtime): implement ONNX session management` |
| Metadata inspection | `feat(runtime): add ONNX metadata inspection` |
| CLI | `feat(cli): implement runtime and model commands` |
| Testing | `test(phase-01): add runtime and configuration tests` |
| Documentation | `docs(phase-01): document implementation and usage` |

These are suggested checkpoints. If two closely related tasks are completed together, they may share a commit.

Never create empty commits or commit incomplete, broken code just to follow the table.

## 4.4 Commit Discipline

After every meaningful implementation milestone:

1. Inspect modified files.
2. Review the Git diff.
3. Remove debugging code and accidental files.
4. Run relevant tests.
5. Run `cargo fmt --check`.
6. Run `cargo clippy --all-targets --all-features -- -D warnings` where applicable.
7. Stage only relevant files.
8. Create a descriptive commit.
9. Verify the commit was created successfully.

Use:

```bash
git status
git diff
git diff --cached
git log --oneline
```

Never use:

```bash
git add .
git commit -am "update"
git push --force
```

unless explicitly justified and approved.

Do not commit:

- ONNX model binaries.
- Dataset files.
- Generated benchmark outputs.
- Local environment files.
- API keys or credentials.
- Build artifacts.
- Temporary debugging files.

Update `.gitignore` accordingly.

## 4.5 Pull Request Workflow

At the end of Phase 1:

- Ensure all commits are on `feat/phase-01-runtime-foundation`.
- Ensure the branch builds successfully.
- Ensure tests pass.
- Review the complete branch diff.
- Prepare a pull request targeting `main`.

Suggested PR title:

```text
feat: implement Phase 1 RustLOB runtime foundation
```

The PR description must include:

- Implementation summary.
- Architecture overview.
- Dependencies introduced.
- Testing results.
- Screenshots or terminal output where useful.
- Known limitations.
- Phase 1 acceptance checklist.

Do not merge the PR automatically. Leave merging to the project owner.

---

# 5. Project Architecture

Create the following initial architecture:

```text
rust-lob/
│
├── Cargo.toml
├── Cargo.lock
├── rust-toolchain.toml
├── .gitignore
├── README.md
├── PROJECT_CONTEXT.md
├── PHASE_01.md
│
├── configs/
│   └── default.toml
│
├── models/
│   └── .gitkeep
│
├── src/
│   ├── main.rs
│   ├── lib.rs
│   │
│   ├── cli/
│   │   ├── mod.rs
│   │   └── commands.rs
│   │
│   ├── config/
│   │   ├── mod.rs
│   │   └── settings.rs
│   │
│   ├── runtime/
│   │   ├── mod.rs
│   │   ├── environment.rs
│   │   ├── session.rs
│   │   └── metadata.rs
│   │
│   ├── models/
│   │   ├── mod.rs
│   │   └── registry.rs
│   │
│   ├── error.rs
│   └── logging.rs
│
├── tests/
│   ├── config_tests.rs
│   ├── runtime_tests.rs
│   └── model_loading_tests.rs
│
└── assets/
    └── .gitkeep
```

Adjust this structure when justified by actual implementation requirements.

Keep the architecture modular. Avoid putting everything inside `main.rs`.

---

# 6. Implementation Tasks

## Task 1 — Rust Project Initialization

- Initialize the Cargo project.
- Choose Rust 2021 or 2024 based on dependency compatibility.
- Create the source structure.
- Configure library and binary entry points.
- Configure debug and release profiles.
- Enable appropriate compiler warnings.

Suggested release configuration:

```toml
[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
```

Verify:

```bash
cargo check
cargo build
cargo test
cargo build --release
```

**Acceptance criteria:**

- Cargo project initializes.
- Dependencies resolve.
- Debug and release builds succeed.
- No unnecessary dependencies.

Create the initial Git commit after successful validation.

## Task 2 — ONNX Runtime Integration

Integrate ONNX Runtime through `ort`.

Implement:

- Runtime initialization.
- CPU execution provider configuration.
- Runtime lifecycle management.
- Session initialization infrastructure.
- Runtime error handling.
- Compatibility checks.

Investigate the selected `ort` version's actual lifecycle and initialization API.

Avoid unnecessary global mutable state or repeated initialization.

**Acceptance criteria:**

- ONNX Runtime initializes successfully.
- CPU execution provider is available.
- Runtime errors are propagated.
- Runtime resources are managed safely.

Commit after successful runtime validation.

## Task 3 — Model Registry

Create `src/models/registry.rs`.

Support:

```rust
pub enum ModelType {
    DeepLOB,
    TransformerLOB,
}
```

Responsibilities:

- Model identification.
- Model path resolution.
- Model file existence checks.
- Model-specific metadata expectations.
- Missing model error handling.

Default paths:

```text
models/deeplob.onnx
models/transformerlob.onnx
```

Do not hardcode machine-specific paths.

The application must compile even when model binaries are unavailable.

**Acceptance criteria:**

- Both models are registered.
- Paths are configurable.
- Missing models produce useful errors.

Commit after implementation and tests.

## Task 4 — Session Management

Create `src/runtime/session.rs`.

Implement a reusable model session abstraction.

Responsibilities:

- Load ONNX models.
- Associate sessions with model identifiers.
- Retain runtime resources as required by the selected API.
- Expose controlled metadata access.
- Propagate model-loading errors.
- Avoid unnecessary session recreation.

Conceptual structure:

```rust
pub struct ModelSession {
    // Model identifier
    // ONNX Runtime session
    // Associated metadata
}
```

Follow the actual ownership and lifetime requirements of `ort`.

Lifecycle:

```text
Application Startup
       |
       v
Runtime Initialization
       |
       v
Model Registry
       |
       v
Model Path Resolution
       |
       v
ONNX Model Loading
       |
       v
Metadata Inspection
       |
       v
Session Ready
       |
       v
Application Shutdown
```

**Acceptance criteria:**

- Sessions can be created.
- Both architectures can be loaded independently.
- Errors propagate correctly.
- Sessions are reusable.

Commit after validation.

## Task 5 — Model Metadata Inspection

Create `src/runtime/metadata.rs`.

Retrieve:

- Input names.
- Output names.
- Input dimensions.
- Output dimensions.
- Input element types.
- Output element types.

Expected specifications from the reference paper:

```text
Input: [1, 1, 100, 40]
Output: 3 classification logits
```

Class mapping:

```text
0 = Down
1 = Stationary
2 = Up
```

Implement metadata validation.

Do not assume the actual ONNX models match the paper. Inspect their real metadata and report discrepancies.

Support dynamic dimensions if present.

**Acceptance criteria:**

- Input and output metadata can be inspected.
- Expected dimensions are validated.
- Incompatible models produce descriptive errors.

Commit after successful tests.

## Task 6 — Configuration Management

Create `configs/default.toml`.

```toml
[application]
name = "RustLOB"
environment = "development"

[runtime]
execution_provider = "cpu"

[models]
deeplob_path = "models/deeplob.onnx"
transformerlob_path = "models/transformerlob.onnx"

[model_specs]
batch_size = 1
channels = 1
time_steps = 100
features = 40
output_classes = 3

[logging]
level = "info"
```

Implement:

- TOML loading.
- Configuration validation.
- Configurable model paths.
- CLI overrides where appropriate.
- Descriptive configuration errors.

**Acceptance criteria:**

- Default configuration loads.
- Invalid configuration is detected.
- Model paths can be changed without source modifications.

Commit after testing.

## Task 7 — CLI Foundation

Implement CLI using `clap`.

### Runtime information

```bash
cargo run -- runtime-info
```

Display:

- RustLOB version.
- ONNX Runtime initialization status.
- Selected execution provider.
- Runtime configuration.

### Model inspection

```bash
cargo run -- inspect --model deeplob
```

```bash
cargo run -- inspect --model transformerlob
```

Display:

- Model name.
- Model path.
- File availability.
- Input metadata.
- Output metadata.
- Validation status.

### Model validation

```bash
cargo run -- validate --model deeplob
```

```bash
cargo run -- validate --model transformerlob
```

Validate model loading and metadata compatibility.

Do not report success when the model is missing.

Support:

```bash
cargo run -- --help
```

And custom configuration:

```bash
cargo run -- --config configs/custom.toml runtime-info
```

**Acceptance criteria:**

- All commands work.
- Exit codes are meaningful.
- Errors are readable.
- CLI logic is separated from runtime internals.

Commit after validation.

## Task 8 — Error Handling

Create `src/error.rs`.

Handle:

- Runtime initialization failure.
- Missing model files.
- Invalid ONNX models.
- Session creation failures.
- Metadata extraction failures.
- Metadata validation failures.
- Configuration errors.
- Filesystem errors.
- Unsupported execution providers.

Use `thiserror` for domain-specific errors and `anyhow` for application-level propagation where appropriate.

Avoid excessive `.unwrap()` and `.expect()`.

**Acceptance criteria:**

- Errors are descriptive.
- Errors propagate correctly.
- No unnecessary application panics.

Commit with related infrastructure changes.

## Task 9 — Logging

Create `src/logging.rs`.

Use `tracing` and `tracing-subscriber`.

Support:

- ERROR
- WARN
- INFO
- DEBUG
- TRACE

Log important runtime events.

Do not add unnecessary logging inside future inference hot paths.

**Acceptance criteria:**

- Logging initializes.
- Log levels are configurable.
- Runtime events are observable.

Commit with the logging implementation.

## Task 10 — Testing and Integration

Implement:

### Configuration tests

- Valid configuration.
- Missing configuration.
- Invalid TOML.
- Invalid paths.

### Runtime tests

- Runtime initialization.
- CPU provider configuration.
- Initialization lifecycle.
- Error handling.

### Registry tests

- DeepLOB registration.
- TransformerLOB registration.
- Path resolution.
- Missing model detection.

### Model loading tests

- DeepLOB loading.
- TransformerLOB loading.
- Invalid ONNX rejection.
- Missing model rejection.

Use small deterministic ONNX fixtures when necessary.

Do not make the entire test suite dependent on large model binaries.

### Metadata tests

- Input metadata extraction.
- Output metadata extraction.
- Shape validation.
- Data type validation.
- Incompatible metadata detection.

Run:

```bash
cargo fmt --check

cargo clippy --all-targets --all-features -- -D warnings

cargo check

cargo test

cargo build --release
```

Resolve failures before completion.

Commit the final test suite separately.

---

# 7. Performance and Resource Considerations

Prepare the architecture for future low-latency inference.

- Initialize runtime resources outside future hot paths.
- Load models once and reuse sessions.
- Avoid unnecessary heap allocations.
- Keep preprocessing separate from inference.
- Keep model loading separate from execution.
- Prefer contiguous memory for future tensor inputs.
- Avoid premature concurrency.
- Do not optimize based on assumptions.

Performance measurements belong to Phase 6.

---

# 8. Scope Restrictions

Explicitly out of scope:

- DeepLOB tensor preprocessing.
- TransformerLOB tensor preprocessing.
- Actual inference execution.
- Custom tensor construction.
- Z-score normalization.
- Sliding-window extraction.
- CSV reading and evaluation.
- Benchmarking.
- Memory profiling.
- Statistical analysis.
- C++ versus Rust comparison.
- Model training.
- ONNX export.
- GPU execution.
- Candle, Burn and tch integration.

---

# 9. Phase 1 Definition of Done

### Infrastructure

- [ ] Cargo project initialized.
- [ ] Modular architecture established.
- [ ] Dependencies configured.
- [ ] Cargo.lock committed.
- [ ] Configuration implemented.
- [ ] Logging implemented.
- [ ] Error handling implemented.

### Runtime

- [ ] ONNX Runtime integrated.
- [ ] CPU execution configured.
- [ ] Runtime lifecycle implemented.
- [ ] Session management implemented.
- [ ] Metadata inspection implemented.

### Models

- [ ] DeepLOB registered.
- [ ] TransformerLOB registered.
- [ ] Configurable paths supported.
- [ ] Missing models handled.
- [ ] Metadata validation implemented.

### CLI

- [ ] Help command works.
- [ ] Runtime information works.
- [ ] Model inspection works.
- [ ] Model validation works.

### Quality

- [ ] Unit tests pass.
- [ ] Integration tests pass where applicable.
- [ ] cargo fmt passes.
- [ ] cargo clippy passes.
- [ ] cargo check passes.
- [ ] cargo test passes.
- [ ] Release build passes.
- [ ] README updated.

### Git

- [ ] Correct feature branch created.
- [ ] Meaningful atomic commits created.
- [ ] No generated binaries or datasets committed.
- [ ] Git working tree clean.
- [ ] Branch diff reviewed.
- [ ] Pull request prepared against main.
- [ ] No automatic merge.

---

# 10. Required Final Report

Codex must provide:

1. Implementation summary.
2. Architecture and module descriptions.
3. Dependency versions and rationale.
4. ONNX Runtime integration details.
5. CLI examples with actual output.
6. Test results.
7. Known limitations.
8. Completed acceptance checklist.
9. Git branch name.
10. Commit history with commit hashes.
11. Pull request summary.

Do not mark Phase 1 complete if functionality is stubbed or mocked.

If ONNX models are unavailable, report which runtime tests were validated independently and which model-specific checks remain pending.

---

# 11. Instructions to the AI Coding Agent

You are acting as a senior Rust systems engineer working on RustLOB.

Treat `PROJECT_CONTEXT.md` as the overall project specification and this file as the detailed Phase 1 implementation specification.

### Mandatory execution sequence

1. Inspect the repository.
2. Read `PROJECT_CONTEXT.md` and `PHASE_01.md`.
3. Inspect available model and reference assets.
4. Check the Rust toolchain.
5. Verify `ort` API and ONNX Runtime compatibility.
6. Check the current Git branch and working tree.
7. Create `feat/phase-01-runtime-foundation`.
8. Propose the final implementation architecture.
9. Implement one logical task at a time.
10. Test each meaningful milestone.
11. Create atomic Git commits.
12. Update documentation.
13. Review the entire branch.
14. Produce the final implementation report.
15. Prepare a pull request targeting `main`, without merging it.

### Git-specific instructions

- Never work directly on `main`.
- Never overwrite unrelated user changes.
- Never discard uncommitted user changes.
- Never force-push.
- Never commit secrets, datasets, ONNX models, or build artifacts.
- Never create commits for unimplemented functionality.
- Do not amend existing commits unless explicitly requested.
- Use meaningful Conventional Commit messages.
- Commit only after relevant checks pass.
- Report commit hashes and branch status.

### Critical restrictions

- Do not implement Phase 2 or later.
- Do not generate placeholder implementations.
- Do not fabricate ONNX Runtime results.
- Do not silently ignore errors.
- Do not change research specifications.
- Do not overengineer the architecture.

**Start by inspecting the repository and Git status. Then create the Phase 1 feature branch and begin implementation.**
