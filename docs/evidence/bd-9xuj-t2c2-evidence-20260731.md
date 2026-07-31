# bd-9xuj T2-C2 — identity retention through persisted/admitted owners

Evidence card for the C2 slice of the typed-identity campaign.
All file:line claims below are verified against the final code-commit tree.

## Commits

| role | sha (40-hex) |
|---|---|
| branch | `codex/sandygrove-t2c2-20260731` |
| base (frozen C1r2 chain tip) | `7b12c026b3f6c0ee3c30065a1e4085c6fa9dae4d` |
| C2 code commit | `e5693265b11c31cb98ba4864a6fa8ff21524a643` |
| C2 code tree | `fe1b73c8ac01e104860358f5e35b2beb5a155d61` |

Ordered chain: `02c1c783` (T2-C1r2, core admission) → `890e64c4` (T2-C3,
in-memory replay) → `19357aa0` (readiness-map corrections) → `7b12c026`
(C1r2 evidence) → `e5693265` (this slice, C2).

Files changed by the code commit (3, all reserved for this slice):
`crates/frankensearch-index/src/in_memory.rs`,
`crates/frankensearch-index/src/two_tier.rs`,
`frankensearch/src/index_builder.rs`.
The leased `crates/frankensearch-index/src/lib.rs` was READ but not touched;
see "Deferred lib.rs hunks" below.

## What was retained, where (final-tree file:line)

### 1. in_memory.rs — embedder id/revision no longer destroyed at load

- Fields `embedder_id` / `embedder_revision: Option<String>` — `in_memory.rs:97/:102`.
- Shared load path captures both header strings verbatim before the backing
  `VectorIndex` drops — `in_memory.rs:345-346` (`from_open_index`, serving
  both `from_fsvi` v1 and `from_admitted_v2`). A v1 header's empty revision
  is kept as `Some("")`, distinct from the `None` of headerless sources.
- `from_vectors_with_identity` captures `logical_model_id` /
  `immutable_revision` from the validated declared space —
  `in_memory.rs:267-268` — the same rule the production `create_v2` writer
  uses (`lib.rs:1823-1834`, read-only citation).
- Accessors `embedder_id()` / `embedder_revision()` — `in_memory.rs:401/:410`;
  `InMemoryTwoTierIndex` per-tier passthroughs — `in_memory.rs:1387-1416`.
- `from_vectors` stays typed-absent (`None`); nothing is fabricated.
- Diagnostics only: compatibility joins exclusively on the C3
  `space_fingerprint_hex`.

### 2. two_tier.rs — per-tier space identity on the persistent TwoTierIndex

- Fields on `TwoTierIndex` — `two_tier.rs:428-442` (fingerprints + declared
  bundles).
- `open_with_paths` fills fingerprints from each tier's validated FSVI v2
  identity header — `two_tier.rs:745-752`. `VectorIndex::open` is v1-only
  today, so header-sourced fingerprints are structurally `None` for every
  production artifact: typed `LegacyUnidentified`, never synthesized from
  id/revision strings.
- Accessors: `fast_space_fingerprint_hex()` `:1242`,
  `quality_space_fingerprint_hex()` `:1252`, `fast_declared_identity()`
  `:1267`, `quality_declared_identity()` `:1273`,
  `space_fingerprint_hex_for_tier()` `:1282`.
- `semantic_vector_for_doc_id` (`:1391`) keeps its silent quality→fast
  fallback BY DESIGN (C4 owns changing it); it now delegates to the new
  `semantic_vector_with_tier_for_doc_id` (`:1412`), which returns
  `(SemanticVectorTier, Vec<f32>)` (`SemanticVectorTier` at `:1491`) so the
  fallback is observable and joinable to a space.
- Builder: `set_fast_identity` `:1576`, `set_quality_identity` `:1593`
  (validate-at-set, revision string bound, id string deliberately
  untouched); `finish()` `:1739` checks the declared space dimension against
  the written vectors (`ensure_identity_describes_tier` `:1820`), writes
  headers via `create_with_revision` (byte-identical to the former `create`
  call when no identity was declared), and attaches the declared identities
  to the returned index. A declared quality identity is dropped unless THIS
  build wrote the quality tier (stale on-disk artifact protection).

### 3. index_builder.rs — the comment's identity binding is now performed

- `IndexBuilder::build` threads `fast_embedder.identity()` /
  `qe.identity()` into the builder — `index_builder.rs:334-346/:351-364`.
- Identity-less (legacy) embedders keep building byte-identically: empty
  revision, typed-absent identity, debug log — absence routed, never
  fabricated. Operational id strings are unchanged (`fsfs/src/runtime.rs`
  compares them to `embedder.id()` at `:678/:11203`; the typed bundle, not
  a string, is the compatibility authority).
- **NEW production failure mode** (named per review #8151): threading
  identities through `build` made it able to fail where it previously
  succeeded — an embedder whose declared identity's space dimension does
  not describe the vectors it actually emits is rejected at
  `TwoTierIndexBuilder::finish` with typed `InvalidConfig`
  (`fast_identity.space.dimension` / `quality_identity.space.dimension`,
  via `ensure_identity_describes_tier`) — never a panic, and never an
  index written that carries an identity lying about its vectors. Pinned
  end-to-end through the facade by
  `build_fails_typed_when_declared_identity_dimension_mismatches_vectors`
  (follow-up commit on `codex/sandygrove-t2c2f-20260731`).

## Identity representation decisions

1. Join key stays the lowercase hex SHA-256 space fingerprint (`&str`),
   matching C1r2's `verify_space_identity` and C3's in-memory field — no
   parallel representation invented.
2. `TwoTierIndex` additionally retains the full
   `EmbeddingIdentityBundleV1` per tier when builder-declared: that is the
   `expected` side C4 needs for `verify_producer_conformance`. Its storage
   component describes the PRODUCER's output contract, not the index's
   persisted encoding (per C1r2's index-seam doc).
3. Retention is process-local for v1 artifacts BY DESIGN: write-side
   `create_v2` conversion is explicitly out of C2 scope, so a reopen from
   disk is typed `LegacyUnidentified`. Tests pin both directions (retained
   at `finish()`; `None` + surviving id/revision strings on reopen).
4. Embedder id/revision strings are diagnostics everywhere; they never
   admit, reject, or synthesize a space identity.

## Tests (all in-tree, red-proofed)

| test | file | red proof |
|---|---|---|
| `from_fsvi_preserves_embedder_identity_strings` | in_memory.rs | 1 |
| `admitted_v2_load_preserves_embedder_identity_strings` | in_memory.rs | 1 |
| `constructor_embedder_identity_rules` | in_memory.rs | 2 |
| `in_memory_two_tier_exposes_per_tier_embedder_identity` | in_memory.rs | 2 |
| `builder_finish_retains_declared_typed_identity` | two_tier.rs | 3, 4 |
| `builder_finish_rejects_identity_not_describing_vectors` | two_tier.rs | — (negative-path pin) |
| `builder_quality_identity_without_quality_tier_is_dropped` | two_tier.rs | — (fail-closed pin) |
| `builder_without_identity_stays_legacy_unidentified` | two_tier.rs | — (byte-compat pin) |
| `semantic_vector_with_tier_reports_serving_tier` | two_tier.rs | 3, 5 |
| `build_threads_typed_identity_into_persisted_headers` | index_builder.rs | 4, 6 |
| `build_without_identity_bundles_stays_legacy_unidentified` | index_builder.rs | — (legacy-arm pin) |
| `build_fails_typed_when_declared_identity_dimension_mismatches_vectors` | index_builder.rs | — (new-failure-mode pin; review #8151 follow-up commit) |

Red-proof transcripts (each: mutation applied at `e5693265`, targeted test
FAILING with `EXIT_STATUS=101` under `set -o pipefail`, mutation reverted,
`git diff --stat` empty = tree byte-identical, re-green):
`docs/evidence/bd-9xuj-t2c2-red-proofs-20260731/red-proof-{1..6}-*.txt`.

## Verification (commands run in the worktree at `e5693265`, wrapper
`cargo-local.sh` = `RCH_DISABLE=1`, repo-canonical `CARGO_TARGET_DIR`,
scratchpad `TMPDIR`)

| command | result |
|---|---|
| `cargo test -p frankensearch-core -p frankensearch-index -p frankensearch -- --test-threads=4` | 17/17 suites ok, 0 failed, `EXIT_STATUS=0` |
| `cargo clippy --no-deps -p frankensearch-index -p frankensearch --all-targets` | `EXIT_STATUS=0`; remaining warnings are pre-existing (`simd.rs` `mul_widen` deprecations ×6, fusion `TELEMETRY_TIMESTAMP_FALLBACK_RFC3339`) — zero on C2 lines |
| `cargo fmt --check -p frankensearch-index -p frankensearch` | `EXIT_STATUS=0` |
| `ubs` on the 3 changed files, base `7b12c026` vs head | criticals 4 → 4 (all pre-existing); zero warning/critical findings on C2-added lines; 2 info-level "clone() usages — audit" notes on `in_memory.rs:267-268`, audited necessary (borrowed `&EmbeddingSpaceIdentityV1` → owned fields, once per construction, no loop) |

The facade crate was NOT model-gated in this environment: the full
`-p frankensearch` suite ran green on default features (no
`--no-default-features` fallback needed).

## Deferred lib.rs hunks (leased to FoggyPrairie — NOT applied)

C2 compiles and tests green without either hunk. Both are additive
conveniences for whoever holds the lib.rs lease next:

1. Root re-export of the new tier type (today reachable as
   `frankensearch_index::two_tier::SemanticVectorTier`):

```diff
 pub use two_tier::{
-    TwoTierIndex, TwoTierIndexBuilder, TwoTierIndexPaths, VECTOR_INDEX_FALLBACK_FILENAME,
-    VECTOR_INDEX_FAST_FILENAME, VECTOR_INDEX_QUALITY_FILENAME,
+    SemanticVectorTier, TwoTierIndex, TwoTierIndexBuilder, TwoTierIndexPaths,
+    VECTOR_INDEX_FALLBACK_FILENAME, VECTOR_INDEX_FAST_FILENAME, VECTOR_INDEX_QUALITY_FILENAME,
 };
```

2. `VectorIndex` space-fingerprint convenience (the readiness map's original
   C2 wish; TwoTierIndex derives it from `identity_v2()` +
   `crate::fingerprint_hex` directly, so it is not required), placed next to
   `identity_v2()` (`lib.rs:1992`):

```rust
    /// Lowercase hex SHA-256 fingerprint of the embedding space recorded in
    /// this index's FSVI v2 identity header, when present (bd-9xuj T2-C2).
    ///
    /// `None` is the typed legacy-unidentified state for v1 files — never
    /// fabricated from the id/revision strings.
    #[must_use]
    pub fn space_fingerprint_hex(&self) -> Option<String> {
        self.identity_v2()
            .map(|identity| fingerprint_hex(&identity.space_fingerprint))
    }
```

## Out-of-scope (deliberately untouched)

Consumer search-API retirement (C4), `BoundQueryEmbedding`/core verifiers
(C1r2 done), write-side `create_v2` conversion, fusion/fsfs crates, goldens.
`fusion/src/refresh.rs:584` (carries a previous index's id string forward)
is the readiness map's same-class site for a later slice; fusion is out of
C2's file budget.
