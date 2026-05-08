# fsfs Model Cache Diagnostics v1

Issue: `bd-pkl0.11`

## Goal

Make quality-model cache state observable without forcing tests to download
models or exposing host-specific cache paths. The diagnostics contract reports
whether the quality tier is warm, cold, missing, or unknown, and attaches
operator advice for each degraded state.

## Contract Rules

Every v1 report must:

- set `raw_paths_present = false`
- set `network_required = false`
- report model identity with a `sha256:*` digest and a redacted cache directory
- include actionable advice with a `model_cache.*` reason code
- mention the operator knobs `FRANKENSEARCH_MODEL_DIR`, `FSFS_DOWNLOAD_MODE`,
  `indexing.model_dir`, or `search.fast_only` where relevant

The contract definition requires four stable state rules: `warm`, `cold`,
`missing`, and `unknown`.

## Offline Fixtures

The missing/offline fixture proves that diagnostics can be validated without
network access. It reports the unavailable quality model, marks downloads as
offline, and records the deterministic fallback path to the hash embedder.

## Validation

Replay the schema-level checks with:

```bash
scripts/check_fsfs_model_cache_diagnostics.sh --mode all
```

The workspace schema conformance suite also round-trips the fixtures through
Rust validation and golden JSON artifacts.

## Validation Artifacts

- `schemas/fsfs-model-cache-diagnostics-v1.schema.json`
- `schemas/fixtures/fsfs-model-cache-diagnostics-contract-v1.json`
- `schemas/fixtures/fsfs-model-cache-diagnostics-warm-v1.json`
- `schemas/fixtures/fsfs-model-cache-diagnostics-cold-v1.json`
- `schemas/fixtures/fsfs-model-cache-diagnostics-missing-offline-v1.json`
- `schemas/fixtures-invalid/fsfs-model-cache-diagnostics-invalid-*.json`
- `crates/frankensearch-fsfs/tests/golden/fsfs_model_cache_diagnostics_*_roundtrip_v1.golden.json`
