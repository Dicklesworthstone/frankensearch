# fsfs Dual-Mode Product Contract (Agent CLI + Deluxe TUI)

Issue: `bd-2hz.1.1`  
Parent: `bd-2hz.1`

## Contract Goal

Define a single semantic contract for `fsfs` that is shared by:

1. Agent-first CLI mode (`fsfs ... --json` / `--toon`)
2. Deluxe interactive TUI mode

Both modes MUST expose the same core search/index/explain semantics.  
Mode differences are allowed only where explicitly listed in this document.

## Normative Terms

- `MUST`: required for conformance
- `SHOULD`: strong default; deviations need explicit rationale
- `MAY`: optional

## Canonical Semantic Surface (Parity Required)

The following capabilities MUST be semantically identical in CLI and TUI.

### 1. Query + Search Semantics

- Query canonicalization MUST be identical (same normalization, truncation, and filtering pipeline).
- Query classification MUST be identical (same class outputs and budgeting behavior for equal input).
- Hybrid retrieval/fusion logic MUST be identical for equal inputs/config.
- Progressive phases MUST mean the same thing in both modes:
  - `Initial`: fast-tier answer set
  - `Refined`: quality-tier upgraded answer set
  - `RefinementFailed`: fast-tier retained with explicit reason

### 2. Result Set Semantics

- Ranking order MUST be identical for equal query/config/index snapshot.
- Score semantics MUST be identical (same source score fields, same blend semantics).
- Pagination windowing (`limit`, `offset`/cursor) MUST return the same ordered subset.
- Filtering semantics MUST be identical (same include/exclude behavior and error handling).

### 3. Indexing + Status Semantics

- Crawl/index eligibility rules MUST be identical (same exclusions, size limits, and content policies).
- Queue state semantics MUST be identical (`pending`, `running`, `blocked`, `failed`, `complete`).
- Staleness detection semantics MUST be identical (same trigger conditions and reason codes).

### 4. Explainability Semantics

- Explanation payload meaning MUST be identical (same component names and interpretation).
- Explanation payload exports MUST conform to `schemas/fsfs-explanation-payload-v1.schema.json`
  and `docs/fsfs-explanation-payload-contract.md`.
- Rank movement semantics MUST be identical (`promoted`, `demoted`, `stable` with same thresholds).
- Decision/fallback reasons MUST be emitted from the same canonical reason-code set.

### 5. Trace Query + Replay Semantics

- Trace query/filter semantics MUST be identical in CLI and TUI:
  - same selector fields (`trace_id`, `root_request_id`, `project_key`, `instance_id`)
  - same event-type vocabulary (`decision|alert|degradation|transition|replay_marker`)
  - same frame-range behavior (`since_frame_seq`/`until_frame_seq`, inclusive)
  - same ordering vocabulary (`oldest_first|newest_first`)
  - same hard limits (bounded `limit`, same failure behavior)
- Replay-by-trace-id semantics MUST be identical:
  - same source resolution rules (`manifest_path`/`artifact_root`)
  - same partial replay window behavior (`start_frame_seq`/`end_frame_seq`)
  - same strictness behavior for unknown reason codes
- Validation failures MUST map to the same canonical reason-code set in both modes:
  - `trace.query.*`
  - `trace.replay.*`

## Intentional Divergence Policy (Allowed Differences)

The table below defines the ONLY sanctioned CLI/TUI divergences.

| Area | CLI Behavior | TUI Behavior | Rationale | Constraint |
|---|---|---|---|---|
| Interaction model | Single-shot command execution | Long-lived interactive session | Different UX envelopes | Underlying operations remain semantically identical |
| Progressive display | Emits phase records/events | In-place visual update across phases | Human readability in TUI | Phase payload content parity preserved |
| Output encoding | JSON/TOON machine payload | Rendered panes/widgets + optional export | Operator ergonomics | Exported machine payload matches CLI schema |
| Recovery affordances | Exit code + stderr + retry guidance | Inline error panel + retry action + status indicators | Faster human recovery loops | Same canonical error/reason codes |
| Discoverability | `--help`, subcommand docs, examples | command palette, hotkeys, contextual help | Mode-appropriate navigation | Same command capability set discoverable in both modes |

Any divergence not listed above is non-conformant.

## Output Stability + Versioning Commitments

### 1. Machine Contract Versioning

All machine-readable outputs MUST include:

- `contract_version` (semver-like, e.g., `1.0`)
- `schema_version` (per-payload schema tag)
- `mode` (`cli` or `tui`)
- `generated_at` (RFC3339 timestamp)

### 2. Compatibility Rules

- Patch/minor updates MAY add optional fields only.
- Existing fields MUST NOT change meaning in patch/minor versions.
- Field removal, rename, or semantic reinterpretation REQUIRES major version bump.
- `--json` and `--toon` MUST remain stable across equivalent contract versions.

### 3. Error + Exit Contract

CLI mode MUST define deterministic exit behavior:

- `0`: success (including degraded-but-valid result paths)
- non-zero: contractually invalid config, I/O/index failure, or unrecoverable runtime failure

TUI mode MUST surface equivalent outcome state with:

- same canonical error/reason code
- same severity level
- same suggested remediation class

## Human Discoverability + Recovery Requirements

### 1. Discoverability Minimums

CLI MUST provide:

- complete `--help` tree
- at least one minimal and one advanced example per top-level command
- explicit mention of machine-output mode (`--json`/`--toon`)

TUI MUST provide:

- command palette discoverability for all primary actions
- visible keybinding/help overlay
- contextual action hints at failure/recovery boundaries

### 2. Recovery Minimums

Both modes MUST provide:

- clear root-cause category (`config`, `index`, `resource`, `model`, `io`, `internal`)
- canonical `reason_code`
- deterministic next-step guidance
- replay/diagnostic handle where available

### 3. Degraded Operation

- If quality phase is unavailable, both modes MUST continue with fast-phase semantics.
- If lexical or semantic source is missing, both modes MUST degrade with explicit source-loss reason code.
- Degraded behavior MUST preserve deterministic ordering guarantees.

## Conformance Checklist (Implementation Gate)

Downstream implementation beads MUST prove:

1. **Parity tests**: same input/query/config/index snapshot yields equivalent semantic outputs across CLI and TUI export mode.
2. **Divergence tests**: only sanctioned differences appear, and only at presentation/interaction layer.
3. **Version tests**: payloads include required version fields and respect compatibility policy.
4. **Recovery tests**: representative failures surface correct canonical reason codes and guidance.
5. **Discoverability tests**: help/palette paths cover all primary command surfaces.
6. **Matrix alignment**: module-level unit coverage stays synced with `docs/fsfs-unit-test-matrix.md` (`bd-2hz.10.1`), including reason-code and structured-log assertions for ER/C/D lanes.

## Required Logging/Artifact Fields

To support auditability and replay, both modes SHOULD emit:

- `mode`
- `command_or_action`
- `contract_version`
- `reason_code` (when degraded/error)
- `fallback_applied` (bool)
- `replay_handle` (if available)

## Downstream Beads Unblocked

This contract is authoritative input for:

- `bd-2hz.1.2`
- `bd-2hz.1.3`
- `bd-2hz.1.5`
- `bd-2hz.3.1`
- `bd-2hz.6.1`
- `bd-2hz.7.1`
- `bd-2hz.10.1`
- `bd-2hz.13`
