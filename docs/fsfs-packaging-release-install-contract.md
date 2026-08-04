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

## Target and Capability Matrices (Required)

The release has two distinct profiles. Their names and capability metadata MUST
make the distinction machine-readable; a lite artifact is never a substitute
for a full embedded artifact.

The full embedded profile MUST be produced for exactly these targets:

- `aarch64-apple-darwin`
- `x86_64-pc-windows-msvc`

The explicit model-free lite profile MUST be produced for exactly these targets:

- `x86_64-unknown-linux-musl`
- `aarch64-unknown-linux-musl`
- `x86_64-apple-darwin`
- `aarch64-apple-darwin`

Linux MUSL lite targets MUST be built with `cargo-zigbuild`; macOS and Windows
targets MUST be built with Cargo target builds. Apple Silicon therefore has both
a full artifact and a separately named lite artifact. Intel macOS and Linux MUSL
have only explicit lite artifacts until a loader-capable backend is supported
and exercised in CI.

## Model Feature and Build-Input Boundary

The `frankensearch-fsfs` crate default feature set MUST compile the registered
Model2Vec and FastEmbed loaders without embedding large model artifacts. Normal
workspace checks, source builds, crates.io packaging, dry runs, and publication
MUST NOT require local Potion or MiniLM files at compile time. A plain default
build MUST become a real semantic binary after the operator runs the documented
`fsfs download-models` commands; no Cargo feature flag or rebuild is permitted
in that recovery path.

Default-profile downloads MUST use revision-pinned manifests, verify every
declared length and SHA-256, and atomically promote the complete installation.
Missing, corrupt, or offline-blocked models MUST produce typed actionable
failures. Production indexing and search MUST NOT admit hash control embeddings
as semantic results.

A `.verified` receipt MAY avoid re-hashing a previously verified cache only when
it binds the exact canonical production manifest (including pinned revision and
every file name, size, URL, and SHA-256) plus the complete current file-state
set. The receipt is a same-UID performance optimization, not an authenticated
trust root: an actor that can rewrite both model bytes and the user-owned receipt
can forge them together. Protecting against that actor requires an OS-backed
trust boundary such as fs-verity or privileged immutable ownership. Explicit
`download-models --verify` requests, download promotion, and every bundled-model
promotion MUST perform full size-and-SHA verification before reporting success
or writing a new receipt. Receipt construction and publication MUST remain
private to that full-verification boundary: no public raw writer may mint a
current receipt from file metadata alone. The boundary MUST compare complete
file state before and after hashing, atomically replace and sync the receipt,
and fail rather than publish when a file changes during verification.

`fsfs download-models MODEL --verify` MUST return a typed nonzero error when the
registered installation is missing, incomplete, or checksum-mismatched; a
diagnostic success envelope for any of those states is forbidden. On success it
MUST refresh the current receipt. `fsfs status`
MUST classify registered caches as `missing`, `incomplete`, `mismatch`, or
`verified` by checking the matching manifest rather than directory existence,
but it MUST remain observational and MUST NOT create or refresh a receipt.
Status MUST NOT claim that a verified cache is loadable. `fsfs doctor` MUST
exercise the compiled loader for every manifest-verified configured semantic
model and MUST fail when a loader rejects the cache. Doctor MUST NOT probe
writability by creating and removing files in model or index directories;
permission metadata MAY be reported, but effective writability MUST be labeled
unknown/not tested because proving it would violate the observation-only
boundary. Any hard doctor check MUST
produce exactly one rendered report, return a nonzero exit status, and use an
`ok: false` machine envelope with the established stable `subsystem_error` code
and the failing checks in its context. Warning-only
and passing reports return zero.

The explicit `--no-default-features` profile is the model-free lite binary. It
enables acquisition and verification commands but MUST NOT compile or advertise
Model2Vec or FastEmbed execution. Downloading model files alone MUST NOT be
presented as enabling semantic execution in this deliberately stripped binary.
An explicitly authorized lexical-only fallback MAY return lexical results, but
MUST report zero admitted semantic scores and stable degradation/recovery
metadata.

The shipping CLI's `Full` and `FastOnly` search modes MUST fail before Initial
when the vector generation or matching real fast embedder is missing, corrupt,
or inadmissible. They MUST NOT silently reinterpret that failure as
lexical-only. `LexicalOnly` remains an explicit operator choice. A failure
isolated to quality initialization or inference MUST preserve the admitted
Initial payload and emit a sanitized, actionable `RefinementFailed` phase.

The full GitHub release lane is the explicit exception:

- `release-build` MUST provision the revision-pinned model inputs with
  `scripts/rch-ensure-deps.sh --models-only`.
- It MUST verify the provisioned byte lengths and SHA-256 digests with
  `scripts/rch-ensure-deps.sh --models-only --check` before compilation.
- It MUST build with `--no-default-features --features embedded-models`.
- Cargo build scripts MUST remain network-free; network admission belongs to
  the release provisioning step.

The `release-build-lite` lane, including every Linux MUSL target, MUST retain
`--no-default-features` and MUST NOT provision or enable semantic loaders or
embedded models.

The end-user installer's ordinary profile MUST NOT silently substitute one of
those deliberately stripped lite artifacts. It MAY install a full embedded
release artifact. If no full artifact exists for the selected platform, it
MUST build the loader-capable Cargo default from source. Only an explicit
`install.sh --lite` request may select the model-free source profile.

Intel macOS (`x86_64-apple-darwin`) is a declared temporary exception to that
source fallback: the pinned ONNX Runtime distribution has no supported Intel
Darwin binary, so the default semantic source build is not routable. Until a
loader-capable Intel backend is built and exercised in CI, ordinary installation
MUST fail with the stable `unsupported_platform` outcome and actionable `--lite`
guidance. It MUST NOT attempt the known-unbuildable default source command or
silently install the lite artifact.

## Artifact Contract

For a `v<version>` release tag, every full embedded target MUST publish:

- versioned archive: `fsfs-<version>-<target>.<ext>`
- checksum: `fsfs-<version>-<target>.<ext>.sha256`
- metadata: `fsfs-<version>-<target>.metadata.json`
- installer alias: `fsfs-<target>.<ext>` plus its `.sha256` sidecar

`<ext>` is `tar.xz` for `aarch64-apple-darwin` and `zip` for
`x86_64-pc-windows-msvc`.

Every lite target MUST instead publish:

- archive: `fsfs-lite-<version>-<target>.tar.xz`
- checksum: `fsfs-lite-<version>-<target>.tar.xz.sha256`
- metadata: `fsfs-lite-<version>-<target>.metadata.json`

Optional signing files append `.sig` and `.pem` to the archive name. Lite and
full names MUST remain distinct even when both profiles exist for one target.

Metadata MUST include:

- `tag`
- `target`
- `binary`
- `build_timestamp_utc`
- `rustc`
- `profile` (`embedded` or `lite`)
- `semantic_loaders` (boolean)
- `embedded_models` (boolean)

## Integrity and Signature Policy

- All released archives MUST include SHA-256 checksums in `sha256:<64 hex>` form.
- Installer default behavior MUST always verify checksum before replacing a binary.
- A missing checksum manifest entry or sidecar, a malformed digest, an absent
  `sha256sum`/`shasum` implementation, or a checksum mismatch MUST fail closed.
  No skip token, warning-only bypass, or unverified replacement is permitted.
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
Ordinary artifact-download failure MUST fall back to the loader-capable default
source build on supported semantic targets. Loader-free fallback is forbidden
unless `--lite` was supplied; Intel macOS follows the typed unsupported-platform
exception above.

The user-facing Quick Start MUST provision and explicitly verify Potion and
MiniLM before its first `fsfs index` command whenever the installed profile does
not embed those artifacts. Documentation MUST NOT promise a fixed 60-second
setup for a path that downloads roughly 621 MB of pinned model data.

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

### Host migration validation command matrix (required)

All cargo-heavy migration checks MUST run through `rch exec -- ...`.
For nightly lanes, prefer `RUSTUP_TOOLCHAIN=nightly rch exec -- cargo ...`
instead of `cargo +nightly` inside `rch` commands.

| Host project | Bead/thread | Toolchain requirement | Required validation lanes (minimum) |
|---|---|---|---|
| `/data/projects/xf` | `bd-3un.35` / `br-3un.35` | Nightly required for migration feature paths that pull `asupersync` nightly surfaces | `RUSTUP_TOOLCHAIN=nightly cargo check --features frankensearch-migration`, targeted migration tests, host-specific hybrid regression checks |
| `/data/projects/coding_agent_session_search` | `bd-3un.36` / `br-3un.36` | Nightly required for migration feature paths; validate local-path dependency availability on worker before long lanes | `RUSTUP_TOOLCHAIN=nightly cargo check --features frankensearch-migration`, targeted search module tests, bakeoff parity lane |
| `/data/projects/mcp_agent_mail_rust` | `bd-3un.37` / `br-3un.37` | Nightly recommended for consistency with shared dependency graph | `cargo check -p mcp-agent-mail-search-core --features hybrid`, targeted bridge/tests, db-planner compatibility checks |

Reference commands:

```bash
# xf (bd-3un.35)
RUSTUP_TOOLCHAIN=nightly rch exec -- cargo check --features frankensearch-migration
RUSTUP_TOOLCHAIN=nightly rch exec -- cargo test --features frankensearch-migration hybrid::tests -- --nocapture

# cass (bd-3un.36)
RUSTUP_TOOLCHAIN=nightly rch exec -- cargo check --features frankensearch-migration
RUSTUP_TOOLCHAIN=nightly rch exec -- cargo test --features frankensearch-migration search::tests -- --nocapture

# mcp_agent_mail_rust (bd-3un.37)
rch exec -- cargo check -p mcp-agent-mail-search-core --all-targets --features hybrid
rch exec -- cargo test -p mcp-agent-mail-search-core --features hybrid fs_bridge::tests -- --nocapture
rch exec -- cargo check -p mcp-agent-mail-db --all-targets --features hybrid
```

Remote worker bootstrap and parity preflight (mandatory):

```bash
# 1) Fleet health
rch doctor
rch workers probe --all
rch status
rch queue

# 2) Worker nightly capability sanity (all configured hosts)
awk -F'"' '/host = / {print $2}' ~/.config/rch/workers.toml | while read -r host; do
  ssh -i ~/.ssh/contabo_vps_ed25519 -o StrictHostKeyChecking=accept-new -o BatchMode=yes "ubuntu@$host" \
    'cargo +nightly --version >/dev/null && rustup run nightly cargo --version >/dev/null'
done
```

Remote failure signatures and required handling:

- `no such command: +nightly` in remote logs:
  rerun with `RUSTUP_TOOLCHAIN=nightly rch exec -- cargo ...` and record worker id.
- `failed to select a version for the requirement asupersync`:
  classify as worker dependency-source skew; run local-circuit fallback and record full error text.
- `Project sync failed: rsync failed` (code `255`):
  retry once after `rch queue`; if repeated, run local-circuit fallback and mark migration run as infra-degraded.

If remote workers cannot resolve required local-path dependencies, run the same
command through `rch` local-circuit mode and record that mode explicitly in
the migration report (`execution_mode=local-circuit`, `remote_failure_signature=<text>`).

### Migration artifact and evidence requirements

Each host migration run MUST publish:

- `migration_manifest.json` (host, version, baseline metrics, gate decisions)
- `migration_events.jsonl` (structured events with reason codes and timestamps)
- `migration_replay_command.txt` (deterministic reproduction command)
- `migration_validation_report.md` (pass/fail summary with explicit rollback decision)

When multiple host migrations run in parallel, artifacts MUST be emitted under
host-scoped directories to avoid collisions:

- `artifacts/xf/`
- `artifacts/coding_agent_session_search/`
- `artifacts/mcp_agent_mail_rust/`
- `artifacts/frankenterm/`

## Post-Migration Dead-Code Decommission Template (Required)

Every host migration MUST create and complete a dedicated decommission child bead
to remove now-dead legacy search code after cutover sign-off.

Instantiation rule:

1. Create a `post-cutover cleanup` child bead when migration parity is near completion.
2. Use this section as the bead template body.
3. Keep cleanup bead open until all checklist items and validation evidence are complete.

### Start gate checklist (MUST all pass before edits)

- Parent migration bead is functionally complete with parity evidence.
- Required quality lanes are green on the frankensearch-backed path.
- Rollback command path is documented and dry-run validated.
- Active-agent ownership is deconflicted (file reservations + thread notice).

### Mandatory cleanup surfaces

- Code:
  - remove legacy retrieval/fusion/embedder implementations replaced by frankensearch.
  - remove transitional probes/adapters that are no longer needed.
- Configuration:
  - remove obsolete feature flags, env vars, and config branches for retired paths.
- Tests and benches:
  - remove duplicate legacy-only coverage and retain frankensearch-backed assertions.
- Documentation and runbooks:
  - remove dual-path language and update operator guidance to single-owner path.
- Telemetry contracts:
  - remove stale legacy reason-code references and keep canonical frankensearch taxonomy only.

### Validation matrix (required before closure)

All cargo-heavy commands MUST run through `rch`.

| Gate | Command pattern | Pass criteria |
|---|---|---|
| Compile | `rch exec -- cargo check --all-targets [--features ...]` | zero errors |
| Lint | `rch exec -- cargo clippy --all-targets [--features ...] -- -D warnings` | zero warnings |
| Tests | `rch exec -- cargo test [scope] -- --nocapture` | required suites pass |
| Integration | host-specific migration regression suite | parity or documented improvement |
| Static bug scan | `ubs <changed-files>` | exit code `0` |

If remote workers fail, run the same command via local-circuit mode and record:
`execution_mode=local-circuit` plus `remote_failure_signature=<text>`.

### Communication and audit checklist

- Reserve edited files with `reason=<bead-id>`.
- Send start message in Agent Mail thread `<br-###>` with scope + reserved paths.
- Post completion message with exact command evidence and changed-file summary.
- Add bead comment with:
  - removed/retained surfaces,
  - validation command list,
  - residual risks or explicit `none`.
- Release file reservations after completion.

### Required decommission artifact bundle

- `decommission_manifest.json` (host, removed surfaces, retained surfaces, sign-off metadata)
- `decommission_validation_report.md` (gate-by-gate pass/fail and command evidence)
- `decommission_replay_command.txt` (deterministic repro command set)

### Copy/Paste bead template

Use this skeleton for future host cleanup beads:

```text
Title: Post-cutover cleanup: remove legacy search stack from <host> after frankensearch parity sign-off

Start gate (MUST):
1) Parent migration bead complete with parity evidence.
2) Strict check/clippy/tests (+ host integration lane) green.
3) Rollback command validated.

Scope:
- Remove legacy search code replaced by frankensearch.
- Remove obsolete config/feature toggles.
- Remove legacy-only tests/docs and keep frankensearch-backed coverage.

Closure evidence:
- rch check/clippy/test command outputs
- ubs scan on changed files
- Agent Mail completion note + bead comment with changed surfaces
```

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

`release-publish` MUST require successful completion of both `release-build` and
`release-build-lite`; an OR condition is forbidden. Before uploading, it MUST
admit exactly the six profile/target pairs above, validate each metadata truth
table (`profile`, `semantic_loaders`, and `embedded_models`), require the
platform-correct `tar.xz` or `zip` archive and `.sha256` sidecar, and recompute
every checksum. Missing, duplicate, or extra versioned pairs MUST fail closed.

### Crates.io publish lane policy (required when enabled)

When `publish-crates` is enabled in CI:

- Trigger constraints:
  - tag MUST match `v*`,
  - prerelease tags containing `-` MUST be skipped,
  - publishing MUST run only on the canonical repository owner.
- Secrets/variables:
  - `CARGO_REGISTRY_TOKEN` secret is required.
  - `ENABLE_CRATES_PUBLISH=true` repository variable is required to opt into crates.io publishing.
- Version alignment:
  - the top-level `frankensearch` crate MUST match the tag version (`vX.Y.Z` -> crate version `X.Y.Z`).
- Ordered publish behavior:
  - publish order MUST be `frankensearch-core`, `frankensearch-embed`, `frankensearch-index`, `frankensearch-lexical`, `frankensearch-fusion`, `frankensearch-storage`, then `frankensearch`,
  - for each crate in sequence:
    - run `cargo publish --dry-run` and retry briefly to absorb crates.io index propagation lag,
    - publish only after dry-run success,
    - treat an already-published crate version as idempotent success,
    - wait between publishes to reduce index race failures.

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

CI can enforce only the dependency-free model boundary without installing the
JSON Schema CLI:

```bash
scripts/check_fsfs_packaging_release_install_contract.sh --mode model-features
```
