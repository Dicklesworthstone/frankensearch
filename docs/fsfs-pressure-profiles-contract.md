# fsfs Pressure Profile Contract: Schema v1, Profile Revision 2

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
- quality is permitted by default; explicit fast-only requests can disable it

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

For quality work, resolve the highest-precedence explicit `search.fast_only`
request before applying the selected profile's capability ceiling. Project
config overrides user config within the file layer. Superseded quality requests
are recorded as `override.superseded.higher_precedence`; they MUST NOT cause a
rejection or change the effective request.
An open quality circuit (`pressure.quality_circuit_open`) disables quality work
the same way as a hard pause, recorded as
`safety.clamp.quality_circuit_open.quality_enabled`.

`pressure.degradation_override` applies its stage's query contract
(`pressure.rs`) to every search entry point (CLI, `--stream`, `--expand`, the
query daemon, `fsfs serve` and the dashboard): `force_lexical_only` runs
searches lexical-only, and `force_metadata_only` and `force_paused` refuse
search with `SearchError::InvalidConfig` on `pressure.degradation_override`.
`force_paused` also refuses `index`, `watch`, `append-batch`, `delete`,
`compact`, `flush` and the compaction daemon; `daemon --stop` stays allowed.
The query daemon's socket name includes the override, so clients with
different overrides never share a daemon. `force_lexical_only` also builds
lexical-only generations without loading or provisioning either semantic model;
watch updates and appends to those generations remain lexical-only. Appending to
an existing semantic generation under that override is refused until `fsfs index`
rebuilds it as lexical-only. On builds with semantic support, returning to ordinary
indexing rebuilds the semantic tiers instead of reusing the lexical-only
completion receipt.

`force_metadata_only` indexes each file admitted by discovery using its path,
file name and filesystem metadata. It does not read, hash or decode file bodies,
load or provision semantic models, or create vector tiers. Existing discovery
exclusions still apply. Metadata records contain no content snippets; their
checkpoint evidence describes the path and metadata, not a content digest.
Rebuilding under this override replaces existing content-bearing rows with
metadata-only rows. Returning to ordinary indexing rebuilds the full text and
eligible semantic tiers, even when the source file stamps have not changed;
metadata-only completion receipts cannot stand in for completed content indexing.
`append-batch` and retained content upserts are refused while this override is
active, so a supplied body cannot bypass the metadata-only policy. Search remains
refused under the override; status and doctor remain available.

`force_embed_deferred` keeps full-text lexical indexing and fast semantic
indexing/search available while deferring the expensive quality tier. It does
not inspect, provision, load or run the quality model during indexing or search,
and it suppresses cross-encoder reranking. An explicit quality-only search is
refused; ordinary hybrid searches run their lexical and fast stages. The effective
profile reports quality disabled with
`safety.clamp.embed_deferred.quality_enabled` when the override clamps a quality
request, including an explicit `fast_only=false` request.

A rebuild under this override publishes a complete fast-tier generation and
records `vector.quality_tier.deferred.pressure_override` in its sentinel. It
reports `quality_deferred_files` separately from failed or incomplete fast
semantic work, with a `quality_index_deferred` completion warning. Status derives
the pending count from the persisted policy and current file membership. It
retires any old quality tier so stale quality rows cannot be searched. Appending
or applying retained content upserts to a generation that still carries a
quality tier is refused before mutation; rebuild with `fsfs index` under the
override first. Fast-tier appends and watch updates then continue normally.
Removing the override and running ordinary `fsfs index` rebuilds the quality
tier when its configured model is available, including for unchanged sources.
This mode defers quality work until that later indexing run; it does not create
an automatic background embedding queue.

Background-indexing permission is enforced independently of watch intent. The
`strict` and `degraded` profiles, or `hard_pause_requested=true`, refuse source
watching and the legacy background compaction daemon. This applies to `watch`,
`index --watch`, configured watch mode, and direct retained-watch API calls,
before initial indexing, index-directory creation or source notification
registration. `force_full` cannot override these restrictions. The refusal
identifies `pressure.profile` or `pressure.hard_pause_requested` and gives a
one-shot or profile-selection remedy.

The `performance` profile permits an explicit watch request without turning
ordinary indexing into a watcher. CLI watch flags participate in normal override
order, so `index --watch --no-watch-mode` remains a one-shot index request.
`daemon --stop` remains available, and the complete-generation read-only query
daemon does not require background-indexing permission.

Ordinary indexing reads `cpu_ceiling_pct`, `io_ceiling_bytes_per_sec` and
`load_ceiling_per_mille` at cooperative admission boundaries. The existing
pressure controller applies the configured profile thresholds, EWMA and
consecutive-reading hysteresis to each signal relative to its ceiling. CPU is
host utilization; load is the one-minute host load per available CPU. Linux
process disk-I/O rates use independent read/write counter deltas divided by
actual elapsed time. Other platforms currently have no disk-I/O counter in this
sampler; CPU and load admission still apply. Shadow-oracle sampling remains
separate.

New model loads and bounded indexing, append and retained-mutation inference
wait while the controller is degraded or in emergency. Complete watching also
pauses before consuming dirty hints or allocating a candidate; a later quiet
sample resumes owed work without requiring another filesystem event. Existing
selected generations and pinned readers remain available. Waiting yields in
sleeps of at most 100 ms and checks cancellation on each iteration. These are cooperative controls:
already running synchronous work can overshoot a threshold. The separate RSS
boundary continues to return its typed memory-pressure error.

The `config.setting.no_effect` warning still covers a `strict` or `degraded`
profile because its resolved concurrency and scheduler limits are not read yet;
its quality and background-indexing ceilings do apply. Automatic durable
background resumption of deferred quality embedding remains unimplemented.

The `performance` profile allows either `fast_only` value. The `strict` and
`degraded` profiles disable quality: `fast_only=true` is accepted, while a winning
`fast_only=false` MUST return `SearchError::InvalidConfig` with field
`search.fast_only`, value `false`, and a reason naming the profile. No quality
request overrides a hard pause; its existing safety-clamp diagnostics explain
the effective disabled state.

Other locked profile fields retain their deterministic
`override.rejected.locked_field` diagnostics and cannot be partially applied.

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

Profile revision 2 makes `performance.quality_enabled` overridable and applies
source precedence before checking quality capability. Current defaults and
resolution diagnostics emit revision 2. The JSON schema envelope remains v1;
the named v1 fixtures below preserve historical revision 1 serialization and
do not establish current runtime behavior.

## Validation Artifacts

- `schemas/fsfs-pressure-profiles-v1.schema.json`
- `schemas/fixtures/fsfs-pressure-profiles-contract-v1.json`
- `schemas/fixtures/fsfs-pressure-profiles-decision-v1.json`
- `schemas/fixtures-invalid/fsfs-pressure-profiles-invalid-*.json`
- `scripts/check_fsfs_pressure_profiles_contract.sh`

## Validation Command

This command validates the historical JSON fixtures. Current profile policy is
covered by the inline configuration and pressure-profile contract tests, plus
the real-model CLI fast-only checks in the executable quickstart test.

```bash
scripts/check_fsfs_pressure_profiles_contract.sh --mode all
```
