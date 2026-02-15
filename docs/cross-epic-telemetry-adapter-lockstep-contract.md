# Cross-Epic Telemetry Schema + Adapter Lockstep Contract

Issue: `bd-2ugv`

## Purpose

Define a single compatibility contract and validation workflow that keeps telemetry schema evolution and host adapter behavior in lockstep across core, fsfs, and ops workstreams.

This contract binds:

- `bd-2yu.2.1` (canonical telemetry schema + taxonomy)
- `bd-2yu.5.8` (host adapter SDK + conformance harness)
- host integration execution (`bd-2yu.5.9` and dependent adapter tasks)

## Cross-Epic Invariants

1. Adapter identity must declare `adapter_id`, `adapter_version`, `host_project`, `telemetry_schema_version`, and `redaction_policy_version`.
2. Adapter envelopes must match the canonical schema version unless explicitly allowed by the lag window policy.
3. Compatibility window is explicit and bounded by `MAX_SCHEMA_VERSION_LAG` in `crates/frankensearch-core/src/contract_sanity.rs`.
4. Versions older than `(core - lag_window)` are deprecated and fail rollout gates.
5. Versions newer than core are rejected (`TooNew`) until core is upgraded.
6. Canonical host/adapters must preserve identity pairing (`host_project` <-> `adapter_id`) for known first-class hosts.
7. Redaction policy mismatches are always hard failures.
8. Every failure path must emit deterministic diagnostics containing reason code and replay command.

## Version Lifecycle and Rollback Rules

Compatibility status categories:

- `Exact`: adapter schema equals core schema.
- `Compatible { lag }`: adapter lags within allowed window; rollout may continue with warning.
- `Deprecated { lag }`: adapter is too old; rollout is blocked.
- `TooNew { ahead }`: adapter is ahead of core; rollout is blocked until core catches up.

Rollback behavior:

1. If rollout introduces `TooNew`, pin adapter deployment to the previous version and re-run conformance.
2. If rollout introduces `Deprecated`, roll adapter forward or temporarily pin core until adapter updates land.
3. If redaction mismatch appears, stop rollout immediately; this is a policy violation, not a soft compatibility issue.

## Validation Workflow

## 1) Unit Compatibility Checks (core contract logic)

```bash
cargo test -p frankensearch-core contract_sanity::tests -- --nocapture
```

Coverage includes exact/lagging/deprecated/too-new classification and deterministic diagnostic/replay-command generation.

## 2) Integration Conformance Checks (adapter SDK harness)

```bash
cargo test -p frankensearch-core host_adapter::tests -- --nocapture
```

Coverage includes identity, envelope checks, lifecycle hooks, redaction policy, and fixture-driven conformance behavior.

## 3) E2E Drift Scenario Replay (deterministic)

```bash
cargo test -p frankensearch-core contract_sanity::tests::two_host_adapter_drift_scenario_emits_actionable_diagnostics -- --nocapture
cargo test -p frankensearch-core contract_sanity::tests::classify_version_against_supports_drift_simulation -- --nocapture
```

These tests validate two-host lockstep behavior and deterministic drift classification across simulated core version changes.

## 4) Host Adapter E2E Lanes (deterministic scripts)

Mandatory host lanes for `bd-2yu.5.9` live under `scripts/e2e/`:

```bash
scripts/e2e/telemetry_adapter_cass.sh --mode all
scripts/e2e/telemetry_adapter_xf.sh --mode all
scripts/e2e/telemetry_adapter_agent_mail.sh --mode all
```

Fast contract smoke path (no remote compile, deterministic artifact emission):

```bash
scripts/e2e/telemetry_adapter_cass.sh --mode unit --dry-run
scripts/e2e/telemetry_adapter_xf.sh --mode unit --dry-run
scripts/e2e/telemetry_adapter_agent_mail.sh --mode unit --dry-run
```

Dry/mock verification (no external host repository dependency):

```bash
scripts/e2e/telemetry_adapter_cass.sh --mode all --dry-run
scripts/e2e/telemetry_adapter_xf.sh --mode all --dry-run
scripts/e2e/telemetry_adapter_agent_mail.sh --mode all --dry-run
```

Contract for each lane:

- uses `rch exec -- ...` for every cargo-heavy command
- accepts `--execution live|dry` (`--dry-run` is an alias of `--execution dry`)
- emits deterministic artifacts under:
  - `test_logs/telemetry_adapters/<host>-<mode>-<timestamp>/`
  - `structured_events.jsonl` (schema: `telemetry-adapter-e2e-event-v1`)
  - `terminal_transcript.txt`
  - `replay_command.txt`
  - `summary.json`
  - `summary.md`
  - `manifest.json` (schema: `telemetry-adapter-e2e-manifest-v1`)
- guarantees summary/manifest emission on both success and failure via exit-trap finalization
- includes machine-stable `reason_code` values in event lines and lane summaries
- guarantees interrupted runs (`INT`/`TERM`/`HUP`) finalize as `status=fail` with
  `reason_code=telemetry_adapter.session.interrupted` and the active stage name
- includes host-specific migration check command against:
  - `/data/projects/coding_agent_session_search`
  - `/data/projects/xf`
  - `/data/projects/mcp_agent_mail_rust`

Artifact interpretation quick map:

- `structured_events.jsonl`: canonical stage timeline and status transitions (`started`, `ok`, `fail`, `skipped_dry_run`).
- `terminal_transcript.txt`: full command transcript with reproducible command headers.
- `replay_command.txt`: deterministic replay command for rerunning the exact lane.
- `summary.md` / `summary.json`: human + machine lane summary including `status`, `reason_code`, and `execution_mode`.
- `manifest.json`: run envelope with artifact inventory and stable metadata for downstream tooling.

## Reason Codes and Replay Commands

Primary reason codes:

- `contract.schema.lagging` (warning)
- `contract.schema.deprecated` (error)
- `contract.schema.too_new` (error)
- `adapter.identity.schema_version_mismatch` (warning only when in compatibility window; otherwise error)
- `adapter.identity.canonical_pair_mismatch` (error)
- `adapter.identity.redaction_policy_mismatch` (error)
- `adapter.hook.error` (error)

Replay command mapping:

- Schema/compatibility drift violations:
  - `FRANKENSEARCH_HOST_ADAPTER=<adapter_id> cargo test -p frankensearch-core contract_sanity::tests -- --nocapture`
- Canonical identity-pair violations (`adapter.identity.canonical_pair_mismatch`):
  - `FRANKENSEARCH_HOST_ADAPTER=<adapter_id> cargo test -p frankensearch-core contract_sanity::tests -- --nocapture`
- Adapter identity/envelope/redaction/hook violations:
  - `FRANKENSEARCH_HOST_ADAPTER=<adapter_id> cargo test -p frankensearch-core host_adapter::tests -- --nocapture`

## Upgrade Choreography

1. Land schema updates (`bd-2yu.2.1` lineage) and regenerate fixtures/schemas.
2. Update adapter SDK expectations (`bd-2yu.5.8` lineage).
3. Run core contract + adapter harness tests.
4. Roll host adapters in waves (canary -> partial -> full).
5. Require zero hard violations in `ContractSanityReport::diagnostics()` before full rollout.
6. Archive diagnostic artifacts for release sign-off.

## Host Integration Guide (Adapter SDK + Conformance)

This guide is the required host-onboarding path for known and future projects.

### Step 1: Define adapter identity

Every host adapter MUST declare:

- `adapter_id`
- `adapter_version`
- `host_project`
- `telemetry_schema_version`
- `redaction_policy_version`

These fields are mandatory inputs to conformance and rollout gates.

### Step 2: Implement telemetry envelope mapping

Map host-native telemetry into canonical envelope fields:

- lifecycle and state transitions
- decision/alert/degradation/transition/replay-marker events
- deterministic reason codes
- replay command handles

Do not emit host-specific ad hoc payloads without a canonical mapping.

### Step 3: Run conformance harness (required)

Run these checks before any rollout:

```bash
cargo test -p frankensearch-core contract_sanity::tests -- --nocapture
cargo test -p frankensearch-core host_adapter::tests -- --nocapture
```

If running under load or CI offload policy, execute heavy cargo commands via:

```bash
rch exec -- cargo test -p frankensearch-core contract_sanity::tests -- --nocapture
rch exec -- cargo test -p frankensearch-core host_adapter::tests -- --nocapture
```

### Step 4: Execute rollout verification

1. Shadow host traffic with adapter enabled.
2. Validate compatibility status is `Exact` or `Compatible`.
3. Confirm zero hard failures (`Deprecated`, `TooNew`, redaction mismatch).
4. Promote canary only after deterministic replay checks pass.

### Step 5: Define rollback pin

Before default rollout, document:

- previous adapter version pin,
- previous core version pin (if needed),
- deterministic rollback command sequence,
- owner/on-call escalation target.

### Known host integration map

| Host project | Integration expectation | Minimum verification |
|---|---|---|
| `cass` (`/dp/coding_agent_session_search`) | Preserve agent-facing search telemetry semantics and replay handles. | Contract sanity + adapter conformance + shadow verification evidence. |
| `xf` (`/dp/xf`) | Preserve high-volume search stream fidelity and reason-code stability. | Contract sanity + adapter conformance + canary error/latency gate pass. |
| `mcp_agent_mail_rust` (`/dp/mcp_agent_mail_rust`) | Preserve thread/message retrieval telemetry correctness for agent workflows. | Contract sanity + adapter conformance + deterministic replay validation. |
| `frankenterm` (`/dp/frankenterm`) | Preserve interactive session telemetry and degradation transitions. | Contract sanity + adapter conformance + interactive canary verification. |

### Future host template

For any new host:

1. Register adapter identity and version policy.
2. Provide fixture-backed conformance examples for core event categories.
3. Run contract + adapter test commands.
4. Publish rollout and rollback plan with explicit reason-code ownership.
5. Add host entry to this guide before full rollout.

## Sign-Off Checklist

- [ ] No `Deprecated` or `TooNew` adapters in the latest contract report.
- [ ] No redaction-policy mismatch violations.
- [ ] Unit + integration + drift replay commands pass on CI and locally.
- [ ] Host adapter rollout order and rollback pin versions documented.
- [ ] Release notes include schema version, compatibility window, and impacted adapters.
