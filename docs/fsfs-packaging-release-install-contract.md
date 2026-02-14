# fsfs Packaging, Release, and Install Workflow Contract v1

Issue: `bd-2hz.11.1`  
Parent: `bd-2hz.11`

## Goal

Define a deterministic packaging/release/install workflow for `fsfs` that covers:

- cross-platform binary targets and artifact naming
- checksum/signature integrity expectations
- install and upgrade UX behavior, including rollback expectations

## Normative Terms

- `MUST`: hard requirement
- `SHOULD`: expected default unless explicitly justified
- `MUST NOT`: forbidden behavior

## Target Platform Matrix (Required)

Release artifacts MUST be produced for exactly these targets:

- `x86_64-unknown-linux-musl`
- `aarch64-unknown-linux-musl`
- `x86_64-apple-darwin`
- `aarch64-apple-darwin`

Linux MUSL targets MUST be built with `cargo-zigbuild`; macOS targets MUST be built with Cargo target builds.

## Artifact Contract

Each release target MUST publish:

- archive: `fsfs-<tag>-<target>.tar.xz`
- checksum file: `fsfs-<tag>-<target>.tar.xz.sha256`
- metadata file: `fsfs-<tag>-<target>.metadata.json`
- optional signing files:
  - `fsfs-<tag>-<target>.tar.xz.sig`
  - `fsfs-<tag>-<target>.tar.xz.pem`

Metadata MUST include:

- `tag`
- `target`
- `binary`
- `build_timestamp_utc`
- `rustc`

## Integrity and Signature Policy

- All released archives MUST include SHA-256 checksums in `sha256:<64 hex>` form.
- Installer default behavior MUST always verify checksum before replacing a binary.
- Signature verification SHOULD be supported through `--verify` mode and Sigstore/Cosign blob signatures.
- Missing signatures in optional-signing mode MUST emit a deterministic warning reason code, not silent success.

## Install UX Expectations

Supported install entrypoints:

- end-user installer:
  - `curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/frankensearch/main/install.sh | bash -s -- --easy-mode`
- developer path:
  - `cargo +nightly install --path crates/frankensearch-fsfs`

Installer preflight checks MUST include:

- platform/target support
- disk-space floor
- destination write permissions
- release endpoint reachability (unless `--offline`)
- existing-install detection

Installer flags MUST include:

- `--version`
- `--dest`
- `--verify`
- `--checksum`
- `--easy-mode`
- `--force`

Non-root install MUST be default behavior.

## Upgrade UX Expectations

- Upgrade version resolution order MUST be:
  1. explicit `--version` (pinned)
  2. latest available release tag
- Supported upgrade expectations MUST include:
  - fresh install
  - `N-1 -> N`
  - `N-2 -> N`
- On checksum/signature verification failure, installer MUST abort without replacing active binary.
- On post-install validation failure, installer SHOULD restore previous binary when available and emit rollback reason code.

## Install/Upgrade Scenario Playbooks

### Playbook 1: Installer preflight fails due to insufficient disk

- Symptoms: install aborts before download/apply with `install.preflight.disk_space_low`.
- Diagnose:
  - verify target destination and free-space floor,
  - confirm whether large previous artifacts are still present.
- Recovery:
  - free disk space or change `--dest` to a larger volume,
  - rerun installer preflight before fetching release assets.
- Exit criteria: preflight passes and install proceeds to verification stage.

### Playbook 2: Checksum/signature verification failure during install or upgrade

- Symptoms: verification step fails with `release.package.checksum_failed` or `install.verify.signature_missing`.
- Diagnose:
  - recompute SHA-256 on downloaded archive and compare with manifest checksum file,
  - confirm asset tag/target alignment (no mixed target archive),
  - if signature mode enabled, confirm signature sidecar availability.
- Recovery:
  - discard suspect artifact bundle and redownload from release source,
  - rerun with explicit `--verify` and pinned `--version`,
  - do not continue with replace/apply while verification fails.
- Exit criteria: checksum (and signature when requested) validation passes deterministically.

### Playbook 3: Upgrade rollback triggered after apply

- Symptoms: upgrade starts, then reverts with `upgrade.apply.rollback_triggered`.
- Diagnose:
  - inspect post-install validation output and metadata file for failing check,
  - verify upgrade path validity (`N-1 -> N` or `N-2 -> N` supported),
  - confirm destination permissions and executable health.
- Recovery:
  - keep rolled-back binary active,
  - address failing validation condition, then rerun upgrade with pinned version and verify mode,
  - if path unsupported, stage through an intermediate supported version.
- Exit criteria: upgraded binary validates successfully without rollback.

## Host Migration Playbooks (Priority Projects)

This section defines the required migration playbooks for first-party host adopters replacing bespoke search layers with `fsfs`.

### Shared migration gates (all host projects)

- Baseline capture (before cutover):
  - collect `p50/p95/p99` query latency, failure rate, and representative relevance checks from incumbent search path.
- Shadow phase:
  - run `fsfs` in shadow mode while incumbent remains source of truth,
  - capture deterministic artifacts (`manifest.json`, `events.jsonl`, replay command) for every migration stage.
- Cutover gate:
  - no unresolved fatal errors,
  - measurable parity/quality checks pass for host-specific query sets,
  - rollback command validated in a dry-run path.
- Post-cutover gate:
  - monitor at least one sustained window (for example 24h) before decommissioning legacy path.

### Playbook: `/dp/coding_agent_session_search` (cass)

- Cutover checklist:
  - map existing retrieval knobs to `docs/fsfs-config-contract.md`,
  - execute shadow query corpus covering identifier, short keyword, and natural language requests,
  - verify result-shape parity for machine consumers (`jsonl`/`toon` where applicable).
- Rollback criteria:
  - persistent latency regression beyond agreed budget window,
  - repeated fallback or refinement-failure behavior that breaches acceptance thresholds.
- Post-cutover monitoring:
  - track query throughput, phase mix (`Initial` vs `Refined`), and failure category distribution,
  - confirm no drift in contract fields consumed by downstream automation.

### Playbook: `/dp/xf`

- Cutover checklist:
  - validate ingestion/index scope mappings for project roots and exclusions,
  - run shadow comparisons for high-volume corpus queries and recency-heavy prompts,
  - verify explainability payloads preserve stable reason-code semantics.
- Rollback criteria:
  - missing/incorrect top-ranked matches in agreed validation suites,
  - instability in streaming output contract used by agent workflows.
- Post-cutover monitoring:
  - monitor search latency percentiles and degraded-mode entry frequency,
  - track evidence ledger integrity and replay success for sampled requests.

### Playbook: `/dp/mcp_agent_mail_rust`

- Cutover checklist:
  - validate all thread/message retrieval paths against incumbent behavior,
  - verify JSON/TOON output contracts for agent-facing search tooling,
  - run migration-specific e2e scenarios with deterministic artifact bundles.
- Rollback criteria:
  - contract-breaking output deltas for agent consumers,
  - unresolved retrieval correctness regressions in thread/history queries.
- Post-cutover monitoring:
  - monitor automation failure rate attributable to search output differences,
  - monitor reason-code distribution for fallback/degraded transitions.

### Playbook: `/dp/frankenterm`

- Cutover checklist:
  - validate interactive latency and incremental indexing behavior under terminal-driven workflows,
  - confirm explain and streaming surfaces remain actionable in TUI loops,
  - run shadow acceptance with representative terminal session traces.
- Rollback criteria:
  - noticeable interactive lag regressions beyond budget,
  - operator-critical explain/diagnostic surfaces missing required fields.
- Post-cutover monitoring:
  - monitor interactive response-time SLO and pressure-driven degradation transitions,
  - monitor operator incident count linked to search migration.

### Migration artifact and evidence requirements

Each host migration run MUST publish:

- `migration_manifest.json` (host, version, baseline metrics, gate decisions)
- `migration_events.jsonl` (structured events with reason codes and timestamps)
- `migration_replay_command.txt` (deterministic reproduction command)
- `migration_validation_report.md` (pass/fail summary with explicit rollback decision)

## Staged Rollout and Deterministic Fallback Protocol

This rollout protocol is mandatory for project-by-project fsfs adoption.

### Phase 0: Shadow

- Scope:
  - fsfs runs in parallel with incumbent search path; incumbent remains user-facing authority.
- Required gates:
  - contract outputs are schema-valid for all exercised machine interfaces,
  - no unresolved fatal error category in shadow run artifacts,
  - relevance and latency deltas stay inside predeclared migration budget.
- Deterministic failure triggers:
  - machine contract violation,
  - repeated parse/serialization failures for agent-facing outputs,
  - reproducible correctness regressions on migration query set.

### Phase 1: Canary

- Scope:
  - route a bounded cohort to fsfs (project-level or traffic-slice gate), keep rollback path hot.
- Required gates:
  - error budget remains within phase target,
  - latency budget (`p95/p99`) remains inside migration envelope,
  - fallback/degraded reason-code rate does not exceed declared ceiling.
- Deterministic failure triggers:
  - error-budget breach,
  - sustained latency regression beyond threshold window,
  - missing/incorrect results in canary verification suite.

### Phase 2: Default

- Scope:
  - promote fsfs to default path for host project.
- Required gates:
  - canary phase stability window completes successfully,
  - incident review confirms no unresolved rollout blockers,
  - rollback procedure remains executable and tested.
- Deterministic failure triggers:
  - critical incident attributable to rollout deltas,
  - contract-breaking output regression after promotion,
  - inability to execute rollback/restore path.

### Deterministic rollback procedure

When any phase trigger trips, execute the following in order:

1. Freeze rollout progression for the affected host project.
2. Re-pin the last known good fsfs version (or incumbent path) and revert rollout routing.
3. Re-apply prior config profile and verify startup/health checks.
4. Re-run the migration validation corpus and confirm baseline parity restoration.
5. Emit rollback artifact bundle and explicit operator decision note.

Rollback completion criteria:

- affected traffic returns to known-good path,
- key baseline checks pass,
- incident ticket includes deterministic replay handles and root-cause hypothesis.

### Rollout artifact requirements

Each rollout phase MUST publish:

- `rollout_phase_manifest.json` (`phase`, gates, thresholds, pass/fail decision)
- `rollout_phase_events.jsonl` (events + reason-code timeline)
- `rollout_phase_replay_command.txt` (deterministic reproduction)
- `rollout_phase_summary.md` (phase outcome + go/no-go decision)

## CI Quality Gate Matrix Guidance (Pre-Merge vs Nightly)

CI validation operates in two profiles:

1. `premerge` (pull_request, push, manual dispatch):
   - optimized for fast merge safety checks.
2. `nightly` (scheduled):
   - full validation sweep for deeper drift detection.

Required gate categories:

- unit
- integration
- snapshot
- e2e
- perf
- fault
- soak
- contract/schema/conformance

Artifact publishing requirements on every CI run:

- `ci_artifacts/quality_gate_matrix.json`
- `ci_artifacts/rollout_host_checklist.json`
- existing e2e contract artifacts and retention policy outputs

Failure runs MUST include deterministic replay pointers and gate metadata artifacts in summary output.

## Upgrade and Migration Compatibility Verification Strategy

This strategy is normative for all release candidates and required for `bd-2hz.11.6`.

### Version-path matrix (required)

| Path | Expected behavior | Verification requirement |
|---|---|---|
| `N-2 -> N` | automatic migration or deterministic hard-fail with explicit recovery guidance | Full migration suite + result stability + rollback attempt |
| `N-1 -> N` | automatic migration with no data loss and stable behavior | Full migration suite + result stability + rollback attempt |
| `N -> N` (fresh install) | no migration required, baseline behavior retained | Baseline compatibility checks |
| `N -> N-1` (rollback) | rollback succeeds or fails with deterministic reason code and safe state | Rollback verification suite |

### Index/storage compatibility coverage (required)

1. FSVI format compatibility:
   - open golden FSVI snapshots from each supported prior version,
   - verify header/version parsing and segment traversal,
   - verify idempotent migration behavior where migration is required.
2. FrankenSQLite schema migration:
   - run migrations on populated databases (not empty-only fixtures),
   - validate post-migration schema and data invariants,
   - verify repeated migration invocation is idempotent.
3. Tantivy index compatibility:
   - verify existing lexical index opens and query path remains functional,
   - if incompatible, fail with deterministic reason code and explicit rebuild guidance.
4. Configuration evolution:
   - old configs with deprecated keys MUST continue to work with warnings,
   - unknown/deprecated keys MUST emit deterministic warning reason codes,
   - no silent semantic reinterpretation without migration note.

### Golden snapshot strategy (required)

- For each release, produce deterministic golden artifacts for a fixed-seed corpus:
  - FSVI index snapshot,
  - FrankenSQLite snapshot,
  - Tantivy snapshot,
  - effective config snapshot.
- Golden snapshots MUST be replayable and checksum-protected.
- Migration test runs MUST publish artifacts proving:
  - snapshot source version,
  - migration steps executed,
  - post-migration integrity checks.

### Result-stability and quality gates (required)

- Use a fixed golden query set across versions and paths.
- Acceptable quality drift threshold:
  - `NDCG` delta MUST be `< 0.01` for migration paths (`N-2 -> N`, `N-1 -> N`).
- Regression failure MUST emit deterministic reason code and block rollout progression.

### Rollback verification (required)

- Every migration test cycle MUST attempt rollback validation (`N -> N-1`) after upgrade validation.
- Rollback acceptance:
  - runtime starts in safe mode,
  - no silent corruption of migrated artifacts,
  - deterministic operator guidance if full rollback is unsupported.

### Large-corpus migration soak (required)

- Execute at least one multi-GB corpus migration soak run per release cycle.
- Capture:
  - migration duration,
  - peak memory usage,
  - post-migration correctness checks,
  - replay command and artifact manifest.
- Soak failures MUST block rollout promotion until resolved or explicitly waived.

### CI gate requirements (required)

- PRs that touch format-sensitive surfaces MUST run migration compatibility lanes in CI.
- CI MUST block merge when any required migration lane fails.
- Required outputs:
  - migration matrix pass/fail summary,
  - per-path reason-code report,
  - replay handles for failing lanes.

### Required migration artifacts

Each migration compatibility run MUST publish:

- `migration_matrix_report.json`
- `migration_invariants_report.json`
- `migration_quality_regression.json`
- `migration_soak_metrics.json` (when soak lane runs)
- `migration_replay_command.txt`

## CI Release Workflow Mapping

Workflow alignment MUST map to:

- `release-build` (per-target archive/checksum/signature generation)
- `release-publish` (GitHub release publishing)
- `publish-crates` (optional crates.io publish gate)

## Required Reason Codes

- `release.build.missing_target`
- `release.package.checksum_failed`
- `release.publish.asset_upload_failed`
- `install.preflight.disk_space_low`
- `install.verify.signature_missing`
- `upgrade.apply.unsupported_path`
- `upgrade.apply.rollback_triggered`
- `upgrade.migration.matrix_failed`
- `upgrade.migration.invariant_violation`
- `upgrade.migration.quality_regression`
- `upgrade.migration.rollback_verification_failed`
- `upgrade.migration.soak_budget_exceeded`

## Validation Artifacts

- `schemas/fsfs-packaging-release-install-v1.schema.json`
- `schemas/fixtures/fsfs-packaging-release-install-contract-v1.json`
- `schemas/fixtures/fsfs-packaging-release-install-release-manifest-v1.json`
- `schemas/fixtures/fsfs-packaging-release-install-upgrade-plan-v1.json`
- `schemas/fixtures-invalid/fsfs-packaging-release-install-invalid-*.json`
- `scripts/check_fsfs_packaging_release_install_contract.sh`

## Validation Command

```bash
scripts/check_fsfs_packaging_release_install_contract.sh --mode all
```
