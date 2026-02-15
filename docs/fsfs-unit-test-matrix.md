# fsfs Living Unit-Test Matrix v1

Issue: `bd-2hz.10.1`  
Parent: `bd-2hz.10`

## Goal

Define a living, module-level unit-test coverage contract for `fsfs` so quality work starts before full feature completion and remains traceable as modules evolve.

This matrix is normative for:

- happy-path coverage
- edge/boundary coverage
- error/failure coverage
- cancellation/cancellation-correctness coverage (when async or queued behavior exists)
- degraded-mode coverage
- deterministic reason-code and structured logging assertions

## Group Legend

- `H`: happy path
- `E`: edge and boundary
- `ER`: error/failure
- `C`: cancellation/cancellation-correctness
- `D`: degraded/safe-mode fallback behavior

Example group id: `CFG-ER` = config module error coverage lane.

## Deterministic Assertion Baseline (Required in all ER/C/D lanes)

1. Assert reason-code determinism:
   - reason codes are stable across identical inputs
   - reason codes are validated against canonical registry (`ALL_FSFS_REASON_CODES`) when applicable
2. Assert structured diagnostics:
   - every failure/degraded/cancel path emits machine-readable fields required by the governing contract
3. Assert replay-safety:
   - diagnostics include enough stable metadata to replay/triage (trace id, config identity, or manifest handles, depending on contract)

## Module Coverage Matrix

| Module | Contract anchors | Current unit coverage | Required groups | Deterministic reason/log assertions |
|---|---|---|---|---|
| `crates/frankensearch-fsfs/src/lib.rs` | `docs/fsfs-dual-mode-contract.md` | no direct `#[cfg(test)]` module | `LIB-H`, `LIB-E`, `LIB-ER` | Re-export/API surface drift must fail with deterministic diagnostics. |
| `crates/frankensearch-fsfs/src/adapters/mod.rs` | `docs/fsfs-dual-mode-contract.md` | no direct `#[cfg(test)]` module | `ADP-H`, `ADP-E`, `ADP-ER` | Adapter dispatch failures must include stable mode/action metadata. |
| `crates/frankensearch-fsfs/src/adapters/cli.rs` | `docs/fsfs-dual-mode-contract.md`, `docs/fsfs-config-contract.md` | `31` tests (`src/adapters/cli.rs::tests`) | `CLI-H`, `CLI-E`, `CLI-ER`, `CLI-C`, `CLI-D` | Parse/dispatch errors must emit stable reason codes and machine output mode metadata. |
| `crates/frankensearch-fsfs/src/adapters/tui.rs` | `docs/fsfs-dual-mode-contract.md` | `4` tests (`src/adapters/tui.rs::tests`) | `TUI-H`, `TUI-E`, `TUI-ER`, `TUI-D` | TUI-mode failures must map to same canonical reason-code family as CLI. |
| `crates/frankensearch-fsfs/src/catalog.rs` | `docs/fsfs-incremental-change-detection-contract.md`, `docs/fsfs-determinism-contract.md` | `6` tests (`src/catalog.rs::tests`) | `CAT-H`, `CAT-E`, `CAT-ER`, `CAT-C`, `CAT-D` | Replay/reconcile decisions must emit deterministic queue action + reason fields. |
| `crates/frankensearch-fsfs/src/concurrency.rs` | `docs/fsfs-determinism-contract.md` | `33` tests (`src/concurrency.rs::tests`) | `CONC-H`, `CONC-E`, `CONC-ER`, `CONC-C`, `CONC-D` | Lock/scheduler failures must preserve deterministic ordering and reason metadata. |
| `crates/frankensearch-fsfs/src/config.rs` | `docs/fsfs-config-contract.md`, `docs/fsfs-dual-mode-contract.md` | `18` tests (`src/config.rs::tests`) | `CFG-H`, `CFG-E`, `CFG-ER`, `CFG-C`, `CFG-D` | `config_loaded` payload fields and warning reason codes must be deterministic. |
| `crates/frankensearch-fsfs/src/evidence.rs` | `docs/fsfs-expected-loss-contract.md`, `docs/fsfs-scope-privacy-contract.md`, `docs/fsfs-dual-mode-contract.md` | `27` tests (`src/evidence.rs::tests`) | `EVD-H`, `EVD-E`, `EVD-ER`, `EVD-C`, `EVD-D` | Evidence validation failures must include stable `reason_code`, family, and trace linkage. |
| `crates/frankensearch-fsfs/src/explanation_payload.rs` | `docs/fsfs-dual-mode-contract.md`, `docs/fsfs-explanation-payload-contract.md` | `7` tests (`src/explanation_payload.rs::tests`) | `EXP-H`, `EXP-E`, `EXP-ER`, `EXP-D` | Ranking and policy decision exports must preserve stable enum labels, reason codes, and confidence bounds across JSON/TOON/TUI projections. |
| `crates/frankensearch-fsfs/src/lexical_pipeline.rs` | `docs/fsfs-dual-mode-contract.md`, `docs/fsfs-high-cost-artifact-detectors-contract.md` | `6` tests (`src/lexical_pipeline.rs::tests`) | `LEX-H`, `LEX-E`, `LEX-ER`, `LEX-C`, `LEX-D` | Incremental lexical fallback behavior must emit deterministic mutation/action diagnostics. |
| `crates/frankensearch-fsfs/src/lifecycle.rs` | `docs/fsfs-determinism-contract.md`, `docs/fsfs-expected-loss-contract.md` | `34` tests (`src/lifecycle.rs::tests`) | `LIFE-H`, `LIFE-E`, `LIFE-ER`, `LIFE-C`, `LIFE-D` | Disk/daemon/health transitions must preserve stable phase/action reason fields. |
| `crates/frankensearch-fsfs/src/main.rs` | `docs/fsfs-dual-mode-contract.md`, `docs/fsfs-config-contract.md` | `4` tests (`src/main.rs::tests`) | `MAIN-H`, `MAIN-E`, `MAIN-ER`, `MAIN-D` | CLI bootstrap failures must expose canonical reason categories and exit semantics. |
| `crates/frankensearch-fsfs/src/mount_info.rs` | `docs/fsfs-root-discovery-contract.md` | `24` tests (`src/mount_info.rs::tests`) | `MNT-H`, `MNT-E`, `MNT-ER`, `MNT-D` | Root/mount guard outcomes must carry deterministic decision fields (`path`, mode, final decision, reason). |
| `crates/frankensearch-fsfs/src/orchestration.rs` | `docs/fsfs-expected-loss-contract.md`, `docs/fsfs-determinism-contract.md` | `9` tests (`src/orchestration.rs::tests`) | `ORCH-H`, `ORCH-E`, `ORCH-ER`, `ORCH-C`, `ORCH-D` | Queue/scheduler actions must emit stable policy mode, fallback action, and reason code. |
| `crates/frankensearch-fsfs/src/pressure.rs` | `docs/fsfs-expected-loss-contract.md`, `docs/fsfs-determinism-contract.md` | `8` tests (`src/pressure.rs::tests`) | `PRS-H`, `PRS-E`, `PRS-ER`, `PRS-C`, `PRS-D` | Pressure transitions must include deterministic state, threshold, and fallback reason metadata. |
| `crates/frankensearch-fsfs/src/pressure_sensing.rs` | `docs/fsfs-expected-loss-contract.md`, `docs/fsfs-determinism-contract.md` | `17` tests (`src/pressure_sensing.rs::tests`) | `SENSE-H`, `SENSE-E`, `SENSE-ER`, `SENSE-C`, `SENSE-D` | Sensor anti-flap and boundary crossings must produce stable transition diagnostics. |
| `crates/frankensearch-fsfs/src/query_execution.rs` | `docs/fsfs-dual-mode-contract.md`, `docs/fsfs-expected-loss-contract.md`, `docs/fsfs-determinism-contract.md` | `6` tests (`src/query_execution.rs::tests`) | `QEX-H`, `QEX-E`, `QEX-ER`, `QEX-C`, `QEX-D` | Cancellation and degraded retrieval paths must emit deterministic fallback reason codes. |
| `crates/frankensearch-fsfs/src/query_planning.rs` | `docs/fsfs-dual-mode-contract.md`, `docs/fsfs-expected-loss-contract.md` | `12` tests (`src/query_planning.rs::tests`) | `QPLAN-H`, `QPLAN-E`, `QPLAN-ER`, `QPLAN-C`, `QPLAN-D` | Intent/fallback decisions must include stable class, budget profile, and reason metadata. |
| `crates/frankensearch-fsfs/src/redaction.rs` | `docs/fsfs-scope-privacy-contract.md` | `48` tests (`src/redaction.rs::tests`) | `RED-H`, `RED-E`, `RED-ER`, `RED-C`, `RED-D` | Redaction/deny outputs must always include deterministic policy version and reason code. |
| `crates/frankensearch-fsfs/src/repro.rs` | `docs/fsfs-determinism-contract.md`, `docs/fsfs-provenance-attestation-contract.md` | `36` tests (`src/repro.rs::tests`) | `REPRO-H`, `REPRO-E`, `REPRO-ER`, `REPRO-C`, `REPRO-D` | Manifest/attestation mismatches must emit deterministic startup action and reason codes. |
| `crates/frankensearch-fsfs/src/runtime.rs` | `docs/fsfs-dual-mode-contract.md`, `docs/fsfs-determinism-contract.md`, `docs/fsfs-expected-loss-contract.md` | `12` tests (`src/runtime.rs::tests`) | `RUN-H`, `RUN-E`, `RUN-ER`, `RUN-C`, `RUN-D` | Runtime orchestration outputs must preserve stable mode/action/reason fields. |
| `crates/frankensearch-fsfs/src/shutdown.rs` | `docs/fsfs-determinism-contract.md` | `8` tests (`src/shutdown.rs::tests`) | `SHUT-H`, `SHUT-E`, `SHUT-ER`, `SHUT-C`, `SHUT-D` | Signal-handling transitions must be deterministic with stable shutdown reason semantics. |
| `crates/frankensearch-fsfs/tests/benchmark_baseline_matrix.rs` | `docs/fsfs-determinism-contract.md` | `13` integration tests | `BENCH-H`, `BENCH-E`, `BENCH-ER`, `BENCH-D` | Benchmark artifacts must include replay command + deterministic `dataset_sha256`/`matrix_sha256`/`samples_sha256` identity fields. |

`BENCH-*` is the reproducible evidence lane for `bd-2hz.11.5` (demo/benchmark showcase): run `cargo test -p frankensearch-fsfs --test benchmark_baseline_matrix -- --nocapture` to emit the artifact bundle (`benchmark_manifest.json`, `benchmark_matrix.json`, `samples.jsonl`) with fixed replay metadata and stable hashes suitable for audit and CI comparison.

## Contract-to-Group Traceability Index

- `docs/fsfs-dual-mode-contract.md` -> `LIB-*`, `ADP-*`, `CLI-*`, `TUI-*`, `MAIN-*`, `QPLAN-*`, `QEX-*`, `RUN-*`, `EXP-*`
- `docs/fsfs-explanation-payload-contract.md` -> `EXP-*`, `QPLAN-*`, `QEX-*`, `RUN-*`
- `docs/fsfs-config-contract.md` -> `CFG-*`, `CLI-*`, `MAIN-*`, `RUN-*`
- `docs/fsfs-root-discovery-contract.md` -> `MNT-*`, `RUN-*`
- `docs/fsfs-file-classification-contract.md` -> `CAT-*`, `RUN-*`
- `docs/fsfs-high-cost-artifact-detectors-contract.md` -> `LEX-*`, `CAT-*`, `RUN-*`
- `docs/fsfs-incremental-change-detection-contract.md` -> `CAT-*`, `RUN-*`, `ORCH-*`
- `docs/fsfs-expected-loss-contract.md` -> `ORCH-*`, `PRS-*`, `SENSE-*`, `QPLAN-*`, `QEX-*`, `RUN-*`, `EVD-*`
- `docs/fsfs-determinism-contract.md` -> `CONC-*`, `ORCH-*`, `PRS-*`, `SENSE-*`, `REPRO-*`, `SHUT-*`, `BENCH-*`, `RUN-*`
- `docs/fsfs-scope-privacy-contract.md` -> `RED-*`, `EVD-*`, `CLI-*`, `TUI-*`
- `docs/fsfs-provenance-attestation-contract.md` -> `REPRO-*`, `RUN-*`, `EVD-*`

## Explicit Backlog Hooks from `bd-2hz.10.1` Review Notes

These areas must stay represented in matrix updates as their implementation beads evolve:

- calibration guards (`bd-2hz.4.4`) -> `PRS-D`, `SENSE-D`, `ORCH-D`
- disk budget management (`bd-2hz.3.7`) -> `LIFE-D`, `RUN-D`
- daemon lifecycle (`bd-2hz.3.8`) -> `LIFE-*`, `SHUT-*`
- configuration validation (`bd-2hz.13`) -> `CFG-*`
- shared TUI framework (`bd-2hz.12`) -> `TUI-*`
- filesystem watcher (`bd-2hz.14`) -> add `WATCH-*` groups once module lands
- ranking priors (`bd-2hz.5.5`) -> `QPLAN-*`, `QEX-*`

## Update Protocol (Living Document Rules)

1. Every new `fsfs` module MUST add a row before merge.
2. Any contract change in `docs/fsfs-*-contract.md` MUST update:
   - linked module rows
   - traceability index entries
   - reason/log assertion text for affected groups
3. Any test lane completion (`bd-2hz.10.*`) MUST annotate this matrix by:
   - marking covered groups in bead evidence notes
   - adding concrete test file/function references where coverage landed
4. Missing coverage is tracked as explicit gaps; do not silently infer coverage from adjacent modules.
