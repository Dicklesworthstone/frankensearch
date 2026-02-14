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
6. Redaction policy mismatches are always hard failures.
7. Every failure path must emit deterministic diagnostics containing reason code and replay command.

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

## Reason Codes and Replay Commands

Primary reason codes:

- `contract.schema.lagging` (warning)
- `contract.schema.deprecated` (error)
- `contract.schema.too_new` (error)
- `adapter.identity.schema_version_mismatch` (warning only when in compatibility window; otherwise error)
- `adapter.identity.redaction_policy_mismatch` (error)
- `adapter.hook.error` (error)

Replay command mapping:

- Schema/compatibility drift violations:
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
