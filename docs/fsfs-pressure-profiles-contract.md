# fsfs Pressure Profile and Override Contract v1

Issue: `bd-2hz.4.5`  
Parent: `bd-2hz.4`

## Goal

Define strict/performance/degraded operating profiles with deterministic override precedence and migration-safe evolution rules.

This contract is normative for:

- profile defaults and capability boundaries
- profile-vs-override resolution order
- conflict handling and reason-coded diagnostics
- profile version evolution without silent drift

## Normative Terms

- `MUST`: hard requirement
- `SHOULD`: expected default unless explicitly justified
- `MUST NOT`: forbidden behavior

## Profile Definitions (Required)

`fsfs` MUST provide exactly these profile IDs:

- `strict`
- `performance`
- `degraded`

Each profile MUST define:

- scheduler mode
- embed/index concurrency limits
- quality-tier enablement boundary
- background indexing allowance
- pressure enter/exit thresholds
- explicit override policy (overridable vs locked fields)

## Capability Boundaries

## strict

- safety-first
- conservative concurrency limits
- aggressive fallback entry under pressure
- locked fields for unsafe cost expansion

## performance

- balanced throughput/latency
- moderate concurrency and quality access
- controlled fallback behavior

## degraded

- correctness-preserving minimal mode
- reduced features and bounded resource spend
- explicit recovery requirements before escalation back to richer profiles

## Deterministic Override Precedence

Resolution MUST apply in this exact order:

1. hard safety guards
2. CLI overrides
3. environment overrides
4. config-file overrides
5. selected profile defaults

If an override targets a locked field, the override MUST be rejected with deterministic reason code and no partial application.

## Conflict and Safety Semantics

- conflict detection MUST be explicit and machine-readable
- all conflicts MUST emit stable reason codes
- safety clamps MUST be visible in diagnostics
- effective profile state MUST be reconstructable from emitted fields

## Required Diagnostics Fields

Profile-resolution diagnostics MUST include:

- `event`
- `trace_id`
- `selected_profile`
- `precedence_chain`
- `overrides[]` (source, field, applied/rejected, reason)
- `safety_clamps[]`
- `conflict_detected`
- `reason_code`
- `effective_profile_version`

## Scenario Playbooks (Pressure + Recovery)

### Playbook 1: Rapid pressure escalation into degraded states

- Symptoms: frequent transitions toward `EmbedDeferred`, `LexicalOnly`, `MetadataOnly`, or `Paused`.
- Diagnose:
  - inspect transition reason codes: `degrade.transition.escalated`, `degrade.transition.calibration_fallback`,
  - inspect profile diagnostics: `profile.resolution.ok` or `profile.resolution.conflict`,
  - verify whether safety clamps were applied (`safety.clamp.hard_pause.quality_enabled`, `safety.clamp.hard_pause.allow_background_indexing`).
- Recovery:
  - reduce load and confirm profile-appropriate ceilings (`cpu`, `memory`, `io`, `load`),
  - clear accidental hard-pause inputs if present,
  - prefer profile changes over ad-hoc override churn.
- Exit criteria: transitions stabilize with repeated `degrade.state.stable` under expected workload.

### Playbook 2: Recovery appears stuck after host pressure drops

- Symptoms: system remains in degraded mode despite healthy host metrics.
- Diagnose:
  - verify recovery-gate telemetry: `degrade.transition.recovery_pending`,
  - verify eventual recovery emission: `degrade.transition.recovered`,
  - check anti-flap/readiness controls (`consecutive_healthy_required`) and override mode.
- Recovery:
  - keep pressure in healthy band long enough to satisfy recovery gate,
  - remove manual holds/forced overrides that pin degraded mode,
  - avoid forcing direct jumps that violate stepwise recovery policy.
- Exit criteria: deterministic stepwise recovery and stable non-degraded operation.

### Playbook 3: Operator override rejected or silently ineffective

- Symptoms: requested profile override does not apply.
- Diagnose:
  - inspect `overrides[]` entries (`applied=false`) and `override.rejected.locked_field`,
  - inspect `precedence_chain` to ensure request source is actually winning,
  - inspect conflict signal (`conflict_detected=true`).
- Recovery:
  - move change to an allowed field for the selected profile,
  - if lock is intentional, switch profile first, then reapply override,
  - document manual intervention in incident notes.
- Exit criteria: override outcomes are explicit and reproducible with matching reason codes.

## Degraded-Mode UX and Override Control Contract

When runtime enters any non-`normal` query mode, the UI layer MUST provide:

- a deterministic banner string describing the active mode impact
- a transition context classification (`pressure_escalation|pressure_recovery|manual_override_transition|manual_override_hold|state_stable`)
- a controls hint string listing currently valid override actions

Transition context MUST remain stable for equivalent `(from, to, changed, override_mode)` inputs so operator replay logs remain auditable.

## Override Guardrails and Manual Intervention Visibility

Manual overrides (`override_mode != auto`) MUST be:

- explicitly marked as manual intervention in policy/explanation metadata
- accompanied by guardrail text for the resulting target mode
- emitted with reason-coded events that distinguish automatic transitions from operator-driven transitions

At minimum, degradation policy artifacts MUST expose:

- `manual_intervention` (`true|false`)
- `transition_context`
- `override_guardrail`
- `override_mode`

This metadata is mandatory for operator timeline/audit surfaces and post-incident replay.

## Migration-Safe Evolution Strategy

1. Profile schema version is explicit and required.
2. Semantic profile changes MUST increment profile revision.
3. New profile fields MUST provide deterministic defaults for all profiles.
4. Removing or repurposing profile fields MUST NOT happen without explicit migration metadata.
5. Existing profile IDs (`strict|performance|degraded`) MUST remain stable across minor revisions.

## Validation Artifacts

- `schemas/fsfs-pressure-profiles-v1.schema.json`
- `schemas/fixtures/fsfs-pressure-profiles-contract-v1.json`
- `schemas/fixtures/fsfs-pressure-profiles-decision-v1.json`
- `schemas/fixtures-invalid/fsfs-pressure-profiles-invalid-*.json`
- `scripts/check_fsfs_pressure_profiles_contract.sh`

## Validation Command

```bash
scripts/check_fsfs_pressure_profiles_contract.sh --mode all
```
