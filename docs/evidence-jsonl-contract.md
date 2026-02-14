# Evidence JSONL Contract + Redaction Policy v1

Issue: `bd-2yu.2.3`

## Purpose

Define a safe-by-default evidence logging contract that preserves postmortem and explainability value while enforcing deterministic replay metadata and privacy boundaries.

Artifacts:

- Schema: `schemas/evidence-jsonl-v1.schema.json`
- Valid fixtures: `schemas/fixtures/evidence-*.json`
- Negative fixtures: `schemas/fixtures-invalid/evidence-*.json`

## JSONL Line Shape

Each JSONL line is a standalone envelope:

```json
{
  "v": 1,
  "ts": "2026-02-14T00:00:00Z",
  "event": {
    "...": "evidence payload"
  }
}
```

## Required Evidence Payload Fields

- identity:
  - `event_id` (ULID)
  - `project_key`
  - `instance_id` (ULID)
- trace:
  - `root_request_id` (ULID)
  - `parent_event_id` (nullable ULID)
- event classification:
  - `type` (`decision`, `alert`, `degradation`, `transition`, `replay_marker`)
  - `reason.code` (machine-stable code)
  - `reason.human` (operator-facing explanation)
  - `reason.severity` (`info`, `warn`, `error`)
- replay metadata:
  - `replay.mode` (`live`, `deterministic`)
  - if `deterministic`, all required:
    - `seed`
    - `tick_ms`
    - `frame_seq`
- redaction metadata:
  - `redaction.policy_version`
  - `redaction.transforms_applied[]`
  - `redaction.contains_sensitive_source` (boolean)
- payload summary:
  - optional sanitized fields only (strictly enumerated by schema)

## Deterministic Replay Requirements

For replay-capable incidents, logs must be sufficient to reconstruct ordering and timing:

- stable `frame_seq` ordering
- fixed `tick_ms`
- explicit `seed`

Deterministic logs missing these fields are invalid by schema.

## Trace Query + Replay Tooling Interface

Issue linkage: `bd-2hz.8.5`

Tooling MUST support deterministic trace lookup and replay by trace ID across CLI and TUI flows.

### Trace query/filter model

A query request MUST support:

- exact match selectors:
  - `trace_id`
  - `root_request_id`
  - `project_key`
  - `instance_id`
- prefix selector:
  - `reason_code_prefix`
- event type selector:
  - `decision | alert | degradation | transition | replay_marker`
- frame-sequence bounds:
  - `since_frame_seq` (inclusive)
  - `until_frame_seq` (inclusive)
- bounded pagination and ordering:
  - `limit` in `[1, 1000]`
  - `sort` in `oldest_first | newest_first`

Validation failures MUST emit machine-stable reason codes:

- `trace.query.limit.zero`
- `trace.query.limit.too_large`
- `trace.query.trace_id.empty`
- `trace.query.root_request_id.empty`
- `trace.query.project_key.empty`
- `trace.query.instance_id.empty`
- `trace.query.reason_code_prefix.empty`
- `trace.query.frame_seq_range.invalid`

### Replay entrypoint semantics

A replay request MUST include:

- `trace_id` (non-empty)
- at least one replay source:
  - `manifest_path`
  - `artifact_root`
- optional frame window:
  - `start_frame_seq`
  - `end_frame_seq` (`start <= end` when both present)
- strictness mode:
  - `strict_reason_codes` (boolean)

Replay validation failures MUST emit machine-stable reason codes:

- `trace.replay.trace_id.empty`
- `trace.replay.manifest_path.empty`
- `trace.replay.artifact_root.empty`
- `trace.replay.source.missing`
- `trace.replay.frame_seq_range.invalid`

## Reason-Code Semantics

Reason code format:

- pattern: `namespace.subject.detail`
- examples:
  - `search.phase.refinement_failed`
  - `control.backpressure.dropping`
  - `slo.latency.p95_exceeded`

Rules:

- code must be machine-stable and used for aggregation.
- `human` text can evolve but should remain concise.

## Sensitive Data Classification + Redaction Rules

| Class | Examples | Default Transform | Allowed Output |
|---|---|---|---|
| `query_content` | user query text | `hash_sha256`, `truncate_preview` | `query_hash`, `query_preview` (max 120 chars) |
| `filesystem_path` | absolute paths | `path_tokenize` | project-relative tokenized path only |
| `identifier` | user IDs, email-like IDs | `hash_sha256` | one-way hash |
| `document_text` | full content snippets | `drop` or `truncate_preview` | no raw full text |

Policy constraints:

- Redaction is mandatory and on by default.
- Raw sensitive fields are forbidden in evidence payload.
- Evidence records must declare transforms actually applied.

## Compatibility Policy

- Breaking changes bump envelope `v`.
- Additive optional fields at same version are allowed only if explicitly added to schema.
- Unknown extra fields are rejected in strict validation mode.

## Validation Strategy (E2E/CI)

## Positive validation

```bash
for f in schemas/fixtures/evidence-*.json; do
  jsonschema -i "$f" schemas/evidence-jsonl-v1.schema.json
done
```

## Negative validation (must fail)

```bash
for f in schemas/fixtures-invalid/evidence-*.json; do
  if jsonschema -i "$f" schemas/evidence-jsonl-v1.schema.json; then
    echo "unexpected pass: $f" && exit 1
  fi
done
```

Contract gate:

- CI fails if any positive fixture fails or any negative fixture passes.
