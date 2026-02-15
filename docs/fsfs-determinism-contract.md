# fsfs Determinism and Reproducibility Contract v1

Issue: `bd-2hz.1.5`  
Parent: `bd-2hz.1`

## Goal

Define one deterministic contract for `fsfs` so all downstream workstreams share the same reproducibility guarantees, validation thresholds, and diagnostics format.

## Normative Terms

- `MUST`: hard requirement
- `SHOULD`: default expectation unless explicitly justified
- `MUST NOT`: forbidden behavior

## Determinism Tiers

## Tier 1: Bit-Exact Reproducibility

Given identical inputs, state, and configuration, outputs MUST be bit-identical.

Required surfaces:
- ranked search output ordering and scores
- degradation-state transitions for identical pressure signals

Required policy:
- deterministic tie-break (`doc_id` lexical ordering)
- deterministic clock mode (`frozen` or `simulated`)
- deterministic seed present in manifest/logs

## Tier 2: Semantic Equivalence

Outputs MAY differ in non-semantic fields but MUST remain functionally equivalent.

Required surfaces:
- explain output payloads
- evidence ledger records where ordering is non-semantic

Allowed variance examples:
- ordering of equal-weight annotations
- non-functional presentation ordering

## Tier 3: Statistical Reproducibility

Outputs MAY vary within explicit tolerances and reproducible bounds.

Required surfaces:
- embedding similarity comparisons with floating-point sensitivity
- performance/latency metrics and benchmark summaries

Required policy:
- tolerance policy MUST be explicit (`metric`, `max_delta`, window)
- seed/config/artifact metadata MUST still be captured

Benchmark reproducibility anchor for this tier:
- `crates/frankensearch-fsfs/tests/benchmark_baseline_matrix.rs` MUST be treated as the canonical statistical reproducibility lane.
- Each run MUST preserve the emitted artifact bundle (`benchmark_manifest.json`, `benchmark_matrix.json`, `samples.jsonl`) and replay command (`cargo test -p frankensearch-fsfs --test benchmark_baseline_matrix -- --nocapture`).
- The manifest MUST carry stable digest fields for dataset and generated artifacts (`dataset_sha256`, `matrix_sha256`, `samples_sha256`).

## Non-Determinism Sources and Required Mitigations

`fsfs` MUST define and enforce mitigations for:

- float arithmetic variation:
  use canonical rounding or explicit epsilon/ULP policy at comparison boundaries
- thread scheduling variation:
  use stable tie-break ordering and deterministic reduction/order steps
- filesystem traversal order variation:
  canonicalize by sorted path traversal before downstream processing
- timestamp/clock variation:
  injectable clock for tests/replay (`LabRuntime`-compatible simulated time)
- randomness variation:
  explicit seed control, logged in every reproducibility artifact

## Reproducibility Manifest Contract

Every deterministic replay/check artifact MUST include:

- `run_id`
- `determinism_tier`
- `seed`
- `config_hash` (`sha256:<64 hex>`)
- `index_version`
- `model_versions[]` (name/version/digest)
- `platform` (os/arch/rustc)
- `clock_mode`
- `tie_break_policy`
- `float_policy`
- `query_fingerprint`
- `evidence_bundle` (artifact paths + manifest hash)

Tier-1 manifests MUST NOT use `realtime` clock mode.

## Testing Contract

Minimum deterministic validation requirements:

- Unit:
  deterministic ranked output, degradation-state transitions, and serialization/normalization paths
- Integration:
  repeat identical scenarios at least twice and assert stable outputs/artifacts
- E2E:
  replay bundle execution from manifest and verify deterministic pass/fail with diagnostics

For benchmark-focused Tier-3 checks, replay validation MUST include hash equality checks for the generated matrix and sample artifacts against manifest digests so benchmark claims remain auditable.

All determinism checks MUST emit machine-readable result artifacts with:
- scenario id
- tier
- comparison mode
- pass/fail
- mismatch diagnostics (reason-coded and field-addressable)

## Logging and Mismatch Diagnostics

Structured logs for determinism checks MUST include:

- `seed`
- `config_hash`
- `determinism_tier`
- `comparison_mode`
- `run_id`
- `manifest_ref`
- mismatch reason codes and field paths (when failures occur)

If a check fails, diagnostics MUST include:
- `reason_code`
- `field_path`
- `lhs` and `rhs` values (or redacted hashes when sensitive)

## Validation Artifacts

- `schemas/fsfs-determinism-v1.schema.json`
- `schemas/fixtures/fsfs-determinism-contract-v1.json`
- `schemas/fixtures/fsfs-determinism-manifest-v1.json`
- `schemas/fixtures/fsfs-determinism-check-result-v1.json`
- `schemas/fixtures-invalid/fsfs-determinism-invalid-*.json`

## Validation Commands

```bash
for f in schemas/fixtures/fsfs-determinism-*.json; do
  jsonschema -i "$f" schemas/fsfs-determinism-v1.schema.json
done

for f in schemas/fixtures-invalid/fsfs-determinism-invalid-*.json; do
  if jsonschema -i "$f" schemas/fsfs-determinism-v1.schema.json; then
    echo "unexpected pass: $f" && exit 1
  fi
done
```
