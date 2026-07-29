# bd-9xuj T2 readiness map — typed embedding-identity wiring

**Author:** MossyPine (read-only audit, requested by YellowSparrow msg #5595) · **Date:** 2026-07-29
**Basis:** committed `origin/main` content only (line numbers from `git show origin/main:<path>`); working tree ignored. Lease snapshot 2026-07-29T06:10Z. Load-bearing claims in §0.1 independently re-verified before landing.

## 0. Baseline corrections to the premise

`RetrievalTopology` already has consumers (throughout `core/src/recovery_plan.rs`). The genuinely zero-call-site surfaces are `BoundQueryEmbedding`, `TieredQueryEmbeddings`, `IdentityBoundEmbedding`, and `Embedder::embed_bound`/`embed_batch_bound`/`SyncEmbed::embed_bound_sync`.

### 0.1 BLOCKING type-level impedance mismatch (T2's first commit must fix this)

`BoundQueryEmbedding::verify_space` (`core/src/types.rs:201`) compares the **full-bundle** fingerprint (`EmbeddingIdentityBundleV1::fingerprint()` = SHA-256 over space ‖ producer ‖ input ‖ storage, `generation.rs:995-1006`). But `IdentityBoundEmbedding::validate()` (`traits.rs:66-125`) requires query-side storage `format.starts_with("in-memory-")` + F32, while the index side binds storage `format == "fsvi-v2"` little-endian (`index/src/lib.rs:256`, `:534-540`). **A query-side bundle fingerprint can therefore never equal an index-side bundle fingerprint** — `verify_space` as written fails closed on all legitimate traffic at every index seam. The correct join key is the **space** fingerprint: query `identity().space.fingerprint()` (`generation.rs:423`) vs index `FsviV2IdentityMetadata::space_fingerprint` (`index/src/lib.rs:506`). T2-C1 adds `verify_space_identity(expected_space_fp, tier)` alongside the existing method (kept for embedder↔embedder comparison).

## 1. Bypass inventory (guard legend: NONE / DIM = dimension-only / ID-STR = embedder_id string / SPACE = fingerprint)

### 1.1 fusion/src/sync_searcher.rs — worst offender; NO identity evidence held at all
- `:393` `search_collect(&self, query_vec: &[f32], k)` NONE; `:407` `search_collect_with_filter` NONE; `:426` `search_iter` NONE; `:432` `search_iter_with_filter` NONE; `:446` `search_internal` NONE.
- `:465` one raw vec into the FAST space (DIM downstream); **`:549` the SAME raw vec into the QUALITY space** via `quality_scores_for_hits` (DIM downstream) — the canonical bd-9xuj defect verbatim.
- `:480` semantic vector handed to `SyncLexicalSearch::search_sync` (impls ignore it; vestigial param, `:30/:39`).
- `SyncTwoTierSearcher` (`:198`, `new()` `:230`) holds only index+config: no embedder, no id, no dimension pairing. Publicly exported (`fusion/src/lib.rs:107`); this is the CASS surface.
- Placeholder telemetry identities `"sync-fast-query"`/`"sync-quality-query"` at `:527`/`:625`.

### 1.2 index/src/two_tier.rs (persistent TwoTierIndex)
- `:750` `search_fast` DIM; `:764` `search_fast_with_params` DIM (ANN `hnsw.rs:814`, MRL `mrl.rs:265`); `:846` `search_fast_classified` DIM explicit `:851-856`; `:951` `resolve_fast_ann_and_wal` DIM.
- **`:1054` `quality_scores_for_hits` — DIM ONLY (`:1063-1068`)**; doc comment `:1049-1053` advertises dimension as the contract; `quality_index.embedder_id()/identity_v2()` in scope, unconsulted.
- `:1225/:1245/:1278` doc vectors escape as untagged `Vec<f32>` (feed PRF/MMR).
- Identity accessors are strings only (`:1159-1177`); **no `identity_v2()`/space-fingerprint passthrough exists on TwoTierIndex** — must be added (C2).

### 1.3 index/src/in_memory.rs — identity destroyed at load
- `InMemoryVectorIndex` (`:43`) has **no identity field at all**; `from_fsvi` (`:203`) opens a `VectorIndex` then discards id/revision/identity_v2, keeping `dimension` only (`:237-241`); `from_vectors` (`:150`) never receives identity.
- `search_top_k*` family (`:286/:342/:382/:441/:459/:575/:592`) DIM; `scores_for_hits` (`:1015`) DIM; `InMemoryTwoTierIndex` (`:1044`) none; **`:1149` `quality_scores_for_hits` DIM ONLY (`:1152-1157`)**.

### 1.4 index leaf scorers (search.rs / mrl.rs / hnsw.rs / lib.rs)
- Legitimately dimension-scoped; do NOT convert: `search.rs:192/:227/:351/:514/:571/:876` (guard `ensure_query_dimension` `:1602-1610`), `mrl.rs:241/:257`, `hnsw.rs:718/:749/:1180/:1200`. Exception to note: `lib.rs:2082` `dot_query_at` has NO length check on the fused path.
- Positive precedent: `native_hnsw.rs:445-540` already carries `embedding_identity_fingerprint` from `FrozenEmbeddingIdentityBundleV1` — the only fingerprint-bound artifact in the index crate.

### 1.5 fusion/src/searcher.rs (async TwoTierSearcher) — identity available, thrown away
- `:1342/:1363` fast `embed()` raw (identity available, unused); `:1777` quality `embed()` raw; `:1504/:1508` DIM index calls; **`:1883-1885` `quality_scores_for_hits(&quality_vec, …)` DIM only**.
- `:1845-1852` `prf_expand` mixes index-sourced vectors into the query vector with no space check → re-bind after expansion.
- `new` (`:214`) / `with_quality_embedder` (`:288`) validate nothing against the index; `:230-240` only warns on non-semantic fast embedder. Metrics capture embedder ids for telemetry only (`:809/:929`).

### 1.6 fusion/src/daemon_fallback.rs — the TEMPLATE, no action needed
Already SPACE-bound: `identity()` `:895` returns the attested `EmbeddingIdentityBundleV1`; `validate_local_fallback` `:425-450` is a true fingerprint guard (`:443`). Adopt `embed_bound_sync` and copy this shape.

### 1.7 fusion/src/federated.rs — adjacent hazard, deferred
No raw-vector API, but `fuse_weighted` (`:410`) blends cosine magnitudes across shards whose spaces are never compared (`FederatedIndex` `:84` carries name/searcher/weight only). Distinct from the raw-vec bypass; later: topology-tagged shard receipts.

### 1.8 fsfs/src/runtime.rs — strongest existing guard (ID-STR+DIM), still string-equality
- One `VectorIndex` serves both tiers (`SearchExecutionResources` `:537-548`): fast embed `:7198` → `:7205` DIM; **quality embed `:7343` → `:7345` same index, DIM only**; DimensionMismatch is the only backstop (`:7385-7392`).
- `:663` quality gate vs hardcoded 384 (`:416`); `:667` ID-STR; `:11592-11648` HashEmbedder synthesized by `"fnv1a-"` prefix-match on the id string + ID-STR `:11637` + DIM `:11647`; `:11670-11700` quality prep ID-STR+DIM; `:9956-9958` reuse gate ID-STR+DIM.
- Write side discards identity: `:9975/:10054/:10132/:10966` `replace_with_empty(path, id, dim)` → **v1, identity_v2=None**. (`Embedder::id()` doc explicitly disclaims space compatibility, `traits.rs:318-320`.)

### 1.9 facade & write-side root cause
- `frankensearch/src/lib.rs`: pure re-exports, no bypass. `index_builder.rs:322/:324` write id strings while `identity()` is in hand; `:299-309` comment asserts identity binding the code does not perform. Same-class site: `fusion/src/refresh.rs:543/:545/:584` (carries the previous index's id forward).
- **`TwoTierIndexBuilder::finish` (`two_tier.rs:1510`) → `create`/`create_with_revision` → `identity_v2: None` (`lib.rs:1066`). `VectorIndex::create_v2` (`lib.rs:936`) has ZERO call sites outside lib.rs ⇒ every production index today is v1 with no identity bundle.** Any verify path needs an explicit legacy-v1 fallback (reuse `FsviInspection::ReindexRequired(LegacyUnidentified)`, `lib.rs:578`) or T2 bricks every existing index.

## 2. Smallest wiring sequence (each commit compile-atomic, independently green)

- **C1 (core, SAFE-NOW):** `space_fingerprint()` + `verify_space_identity()` on `BoundQueryEmbedding`; negative test pinning §0.1 (in-memory-vs-fsvi-v2 same space verifies OK space-scoped, fails full-bundle). Keep `verify_space`.
- **C2 (index lib.rs, WAIT):** expose `VectorIndex::space_fingerprint_hex()` + `TwoTierIndex::{fast,quality}_space_fingerprint` passthroughs. Additive only.
- **C3 (in_memory.rs, SAFE-NOW):** add optional space-fingerprint/identity field; populate in `from_fsvi` before the source `VectorIndex` drops; `from_vectors_with_identity` beside the old ctor.
- **C4 (two_tier.rs + in_memory.rs, mostly SAFE-NOW):** additive `_bound` seams — `quality_scores_for_hits_bound` / `search_fast_bound` / `search_fast_classified_bound` verifying space when present, DIM+`warn!(LegacyUnidentified)` when v1. Old methods `#[deprecated]`. Enforcement lives at the index boundary so no caller can forge it.
- **C5 (sync_searcher.rs, WAIT):** `search_collect_tiered(&TieredQueryEmbeddings, k)` family routing fast()/quality() through `_bound`; topology into metrics via `supported_topology()` narrowed by index coverage; raw methods become legacy shims; replace placeholder telemetry ids.
- **C6 (searcher.rs, SAFE-NOW):** `embed` → `embed_bound` at `:1342/:1363/:1777`; `_bound` index calls at `:1504/:1508/:1883`; re-bind after `prf_expand`; graceful raw fallback when a stub embedder's `identity()` errors (fail-closed default, `traits.rs:303-310`).
- **C7 (searcher.rs, SAFE-NOW):** fallible `try_new`/`try_with_quality_embedder` comparing embedder space fingerprint vs index space fingerprint; infallible ctors remain for legacy-v1.
- **C8 (two_tier.rs SAFE-NOW + index_builder.rs WAIT):** `set_fast_identity`/`set_quality_identity` on `TwoTierIndexBuilder`; `finish()` routes through `create_v2` when supplied; `IndexBuilder::build` passes `identity()`. **C8 is what makes C4–C7 bite on fresh indexes.**
- **C9 (fsfs runtime.rs, WAIT-hard):** space-fingerprint compares replacing ID-STR at `:667/:11637/:11685/:9956` (legacy path retained); `_bound` calls at `:7205/:7345`; kill the 384-constant gate; emit `RetrievalTopology` in payloads.
- **C10 (cleanup, after B2):** remove deprecated raw methods; drop `SyncLexicalSearch::search_sync`'s vestigial vector param.

**Recommended first landing slice: C1 + C3 + C6 + C7** (all SAFE-NOW; closes the async searcher's cross-space bypass end-to-end). C4-persistent + C8-index can land in parallel on clear `two_tier.rs`; C2/C5/C8-facade/C9/C10 queue behind the lanes in §4.

## 3. Tests & goldens

### 3.1 Existing pins to touch or sibling
- `quality_scores_for_hits`: `two_tier.rs:1940/:1981/:2359/:3264/:3959/:4000/:4140`; `index/tests/in_memory_tests.rs:262/:273/:279`; `frankensearch/tests/cross_component.rs:220`; `index/benches/quality_rescore_hasher_ab.rs`.
- Sync raw-vector surface: `fusion/tests/sync_searcher_tests.rs` (13 refs; fixtures `rank_flip_index` `:18`, `clustered_sync_index` `:69` must gain identities) + 18 inline refs in `sync_searcher.rs` (fixture `make_index` `:983`); benches `collect_limit_all.rs`/`progressive_replay.rs`/`sync_int8_fetch.rs` must stay green.
- Async text-based `search_collect` (17 refs `frankensearch/tests/integration.rs` etc.): unaffected by the raw-vec conversion; C7 gates them only once C8 lands (their indexes are v1 → legacy path until then).
- Test stubs needing `identity()` or the C6 fallback (~22 across searcher.rs/federated.rs/refresh.rs/storage/fsfs; list in the audit). Helper: `EmbeddingIdentityBundleV1::explicit_test_model` (`generation.rs:1083`, public).

### 3.2 Goldens (all 112 checked)
No golden serializes a search payload today (`SearchPayload` `output_schema.rs:436-446` has no topology/identity field) ⇒ pure type migration changes NO golden. Goldens change only where T2 adds fields: `telemetry_embedding_roundtrip_v1` (+identity fingerprints), `telemetry_search_roundtrip_v1` (+requested/realized topology, coverage_ppm), the 3 explanation-payload goldens (identity-derived model names), `cli_e2e_*` manifests (`index_version: fsvi-v1 → fsvi-v2` once C8 lands), degraded-incident suites (+cross-space-rejection scenario), query-plan metamorphic + replay/determinism contracts (pin query space).

### 3.3 New negative tests (wrong space, SAME dimension, per converted seam)
(1) core: §0.1 pin — in-memory vs fsvi-v2 same-space verifies space-scoped, fails full-bundle; (2) two_tier: `quality_scores_for_hits_bound` rejects same-dim wrong-space (create_v2 fixtures) + accepts legacy-v1 with warning; (3) in_memory mirror + `from_fsvi_preserves_space_identity`; (4) `search_fast_bound` rejects quality-space vector; (5) sync: tiered rejects fast-vec-in-quality-tier; topology reporting tests; (6) async: `try_new` rejects wrong-space embedder; PRF stays in quality space; (7) end-to-end phase-2 rejection; (8) fsfs: quality disabled on space mismatch despite matching dimension; hash-synthesis requires space match; (9) facade: persisted restart with model B at same dimension → typed rejection (bd-9xuj's restart clause); (10) property: `explicit_test_model(a,d)` vs `(b,d)`, a≠b always rejects.

## 4. Conflict map (lease snapshot 2026-07-29T06:10Z + branch divergence vs origin/main)

**WAIT:** `fusion/src/sync_searcher.rs` (YellowSparrow exclusive → 07:38Z; also B2's LexicalSearch-removal target — two lanes, one file) blocks C5/C10 · `index/src/lib.rs` (YellowSparrow exclusive → 07:44Z; +2968 on integrate branch; IcyMouse receipt mandate) blocks C2 · `fsfs/src/runtime.rs` (YellowSparrow exclusive → 07:38Z; +3982 rewrite in flight) blocks C9 · `frankensearch/src/index_builder.rs` (FoggyPrairie exclusive → 08:02Z) blocks C8-facade.

**SAFE-NOW:** `core/src/types.rs` (C1; keep the `core/src/lib.rs` export edit to one added name — that file has +4/-4 on semantic branches) · `index/src/in_memory.rs` (C3, C4-half) · `index/src/two_tier.rs` (C4-persistent, C8-index — best index-crate landing zone) · `fusion/src/searcher.rs` (C6, C7 — best landing zone overall) · `daemon_fallback.rs`/`federated.rs` (template/deferred) · test files.

**Non-blocking adjacents:** gxwy (gauntlet-only), fk04a (quill grimoire/index/keeper — no T2 overlap; quill contended by semantic branches independently), IcyMouse receipt-v2 (helps C2; coordinate, don't wait).
