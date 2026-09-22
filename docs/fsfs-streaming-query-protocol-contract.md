# fsfs Streaming Query Protocol Contract (NDJSON + TOON)

Issue: `bd-2hz.6.4`  
Parent: `bd-2hz.6`

## Goal

Define deterministic machine protocol semantics for `fsfs search --stream` in both NDJSON and TOON formats.

## Event Taxonomy (Normative)

Every stream frame MUST be one of:

- `started`: first frame of a stream (`seq = 0`), carrying `stream_id`, `query`,
  `format`, and, when a readable vector generation exists,
  `vector_generation_id`, `vector_generation_is_hash` and `semantic_admitted`
- `progress`: pipeline progress update
- `result`: ranked search result item
- `explain`: explainability payload for a result
- `warning`: non-fatal warning
- `terminal`: final stream outcome + retry/exit policy

These names are stable and versioned by `schema_version = "fsfs.stream.query.v1"`.

### Phase progress reason codes

Each delivered result phase is announced by one `progress` frame before its
`result` frames. Its `payload.reason_code` identifies the phase:

| `reason_code` | `payload.stage` | Meaning |
|---|---|---|
| `query.stream.initial_ready` | `retrieve.fast` (`retrieve.hash_control` for a hash control generation) | Initial results follow |
| `query.stream.refined_ready` | `retrieve.quality` | Quality-refined results follow |
| `query.stream.refinement_failed` | `retrieve.refinement_failed` | Refinement failed or timed out; the Initial results stand |

A daemon-served stream that replays a cached answer emits a progress frame
with `stage = cache` and `reason_code = daemon_cache_hit` before the replayed
phases.

## Frame Envelope

Each frame MUST include:

- `v` (integer protocol version)
- `schema_version` (stable string)
- `stream_id` (stable stream correlation id)
- `seq` (monotonic per-stream sequence)
- `ts` (RFC3339 UTC)
- `command` (`search`)
- event payload (`started|progress|result|explain|warning|terminal`)

## Transport Semantics

### NDJSON mode (`--format jsonl`)

- One JSON object per line
- Each emitted frame ends with a single newline (`\n`)
- Line boundaries are frame boundaries

### TOON mode (`--format toon`)

- Frames are emitted with record-separator framing:
  - byte `0x1E` (RS) prefix
  - TOON payload bytes
  - trailing newline (`\n`)
- RS is the canonical frame boundary marker (TOON payload may be multi-line)

## Terminal + Retry Semantics

Terminal payload MUST include deterministic process-level semantics:

- `status`: `completed | failed | cancelled`
- `exit_code`: deterministic mapping from canonical error mapping
- `failure_category` (for failed/cancelled):
  - `config | index | model | resource | io | internal`
- `retry`:
  - `none`
  - `retry_after_ms { delay_ms, next_attempt, max_attempts }`
  - `retry_exhausted { exhausted_after }`

Rules:

- `completed` MUST use `exit_code = 0`, and MUST NOT carry retry guidance
- `cancelled` MUST use interrupted exit code (`130`) and retry `none`
- `failed` MUST use non-zero exit code and include failure category + error payload

## Exit/Fault Categorization Policy

`SearchError` classes map deterministically to categories and retryability:

- `config`: invalid config, query parse
- `index`: index not found/corrupt/version/dimension mismatch
- `model`: embedder/reranker/model availability and inference failures
- `resource`: timeout, insufficient responses, queue pressure, cancellation
- `io`: io/hash/durability-disabled
- `internal`: subsystem errors

Retry guidance is allowed only for retryable classes and bounded by retry budget.

## CLI Policy

- `--stream` is valid only for the `search` command
- `--stream` defaults to NDJSON if format is unspecified
- `--stream` only supports `jsonl` and `toon` formats

## Conformance Expectations

Implementations MUST prove with tests:

1. NDJSON frame roundtrip correctness
2. TOON frame roundtrip correctness (including ambiguous string tokens)
3. Terminal invariant validation
4. Deterministic failure category + retry guidance mapping
5. CLI stream-mode format/command guardrails
