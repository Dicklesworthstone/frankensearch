# bd-9xuj T2 C4-write — typed identity-bound refresh merge (evidence card)

Date: 2026-07-31. Branch: `codex/sandygrove-c4write-20260731`.
Implements accepted design #8187 + reviewer additions #8199, with two
mid-run fleet-audit findings folded in as acceptance-blocking dispositions
(see "Audit dispositions" below).

## Provenance chain (40-hex, oldest first)

| Commit | Role |
| --- | --- |
| `02c1c7834bfa866b2897f4107d8f5cccbdaaddb5` | T2-C1r2: `BoundQueryEmbedding::verify_space_identity` / `verify_producer_conformance` + `SpaceIdentityAdmission` |
| `890e64c44a559ece8a7a26995a384614471ff7ef` | T2-C3: in-memory typed space identity |
| `7b12c026b3f6c0ee3c30065a1e4085c6fa9dae4d` | T2-C1r2 evidence + banked transcripts |
| `e5693265b11c31cb98ba4864a6fa8ff21524a643` | T2-C2: identity retention (per-tier fingerprints, declared bundles, builder `set_*_identity`) |
| `318f7d0fcbeadfc17d688d364239a40c59560da3` | T2-C2 evidence + banked transcripts |
| `52f3f1c8b54bf05ca484ed240d550c3d0ec08669` | T2-C2f: review #8151 cheap-now items (incl. `#[non_exhaustive]` on `SpaceIdentityAdmission`) — frozen reviewed GO base of this branch |
| `2c32a31a11b1fb8c020b16c80ff36f52fb67bb8a` | C4-write prep: byte-identical adoption of origin/main's containment `refresh.rs` (replicates `5386b39ee35e24911d7a4fffe2d83dfe3d842503` + `3f86ea57` doc fix; origin/main tip at adoption: `f1a6c53a97f7556d04747b37f6c60d1a8d68b677`) |
| `c73dbe8eabbd39774b0d096a24fe274401c8e4ea` | C4-write (ii): attested-vs-declared discriminator + `TwoTierIndex::open_admitted_v2_with_paths` |
| `aaf24321ec783799b8ce990eaea3a587aebc9ac3` | C4-write (iii): the typed identity-bound refresh merge slice |
| (this commit) | C4-write (iv): evidence card + banked transcripts |

## Containment-sync decision

The chain's `refresh.rs` was the 58726e26-lineage version (pre-containment).
Commit `2c32a31a` adopts origin/main's file byte-identically
(sha256 `5fe6e2ae60b06ffbb6e9d1d562498ee7aeba58900c74a006d54ad553aa4043b6`)
so C4-write modifies the CURRENT (contained) semantics and the final replay
cannot silently revert the landed refusal.

## Audit dispositions (mid-run fleet audit, acceptance-blocking)

1. **`VectorIndex::open` is strictly v1** (`parse_header` rejects v2,
   `crates/frankensearch-index/src/lib.rs:3757`). Every v2 entry in this
   slice goes through `VectorIndex::inspect` +
   `VectorIndex::open_admitted_v2` (read side, staged-proof side, and the
   `TwoTierIndex` admitted open). The pre-slice tree's
   `identity-bound-republication-unavailable` branch was UNREACHABLE for
   on-disk v2 — the plain-open probe failed first with
   `IndexVersionMismatch { expected: 1, found: 2 }`; the parent-red
   transcript pins this. A v2 pre-drain zero-side-effect test was added
   (`live_v2_same_producer_cycle_refuses_composite_authority_with_zero_side_effects`):
   admission/inspection alone drains nothing, embeds nothing, writes
   nothing, stages nothing, charges no retry.
2. **Split-generation canonical publication is gated.**
   `TwoTierIndexBuilder::finish` publishes fast-then-quality with no atomic
   pair authority; this slice does NOT invent a pair-atomicity primitive.
   The identity-bound v2 replacement is produced, admitted, and proven
   NON-canonically (`v2-staged/` under the index dir); canonical
   installation is refused typed
   (`refresh.canonical_publication` /
   `composite-generation-authority-unavailable`, naming bd-xomn.1/.3),
   PRE-DRAIN in `run_cycle` and again at
   `RefreshWorker::publish_staged_canonical` (pinned by
   `publish_staged_canonical_refuses_split_generation`). Because queue
   consumption must imply canonical publication, the staged merge takes
   caller-supplied jobs and never touches the queue; `run_cycle` never
   drains while the guard (a currently-permanent condition) holds, so no
   retry budget is consumed and no document can be dropped.

## The slice (verified derivation points, file:line at `aaf24321`)

**1. Bound carriers** — `RefreshRecord`
(`crates/frankensearch-fusion/src/refresh.rs:199`): both tiers are
`BoundQueryEmbedding`. Harvest sites call `Embedder::embed_batch_bound`
(first production consumer): canonical lane `refresh.rs:943` (fast) /
`refresh.rs:978` (quality); staging lane `refresh.rs:1293` / `refresh.rs:1307`.
Conversion re-validates at bind time (`into_bound_query`). No raw vector
enters a record; identity-less embedders fail typed (`embedder.identity`).

**2. Attested-only admission (guards 2+8)** — the attested bit derives from
WHERE the identity came from, structurally:

- `TwoTierIndex::fast_identity_is_attested`
  (`crates/frankensearch-index/src/two_tier.rs:1371`, quality `:1380`):
  true iff the tier's `VectorIndex` carries `identity_v2()` metadata, which
  only the v2 header parse inside exact admission can produce
  (`open_admitted_v2_with_paths` `two_tier.rs:545` → `admit_v2_tier`
  `two_tier.rs:1957`; plain `VectorIndex::open` is strictly v1). Builder
  declarations (C2 `set_*_identity`) never set it.
- `InMemoryVectorIndex::space_identity_is_attested`
  (`crates/frankensearch-index/src/in_memory.rs:428`; stored bit set only on
  the `from_open_index` admitted path `in_memory.rs:364`; declared path
  explicitly false `in_memory.rs:286`). `InMemoryTwoTierIndex` passthroughs
  `in_memory.rs:1430`/`:1441`.
- Refresh merge seam: existing tiers classified via `VectorIndex::inspect`
  (`inspect_tier`), gated by `admit_attested_tier` (`refresh.rs:551`)
  against the artifact's OWN header fingerprints (never caller-supplied),
  admitted exactly via `admit_existing_tier` (`refresh.rs:581`) with a
  binding reconstructed from the executing identity and matched against the
  header full-bundle fingerprint (`reconstruct_admission_binding`
  `refresh.rs:440`). Per-embedding C1r2 verifiers run in
  `stage_identity_bound_generation` (`refresh.rs:1377`):
  `verify_space_identity` against the attested space hex and
  `verify_producer_conformance` via `require_same_producer`
  (`refresh.rs:500`). Declared-only or v1 tiers keep the landed typed
  refusal (`identityless-fsvi-v1` → LegacyUnidentified/RecoveryPlan
  routing).

**3. Producer conformance at strictest (guard 7)** — `SameProducer` only.
`ConformanceCompatibleProducer` is telemetry-logged (both fingerprints) and
refused as `executing-producer-attestation-unavailable`
(`attestation_unavailable_refusal` `refresh.rs:332`, doc comment documents
the deliberate narrowing and the reopening condition: executing-producer
attestation running in code). The `#[non_exhaustive]` admission enum fails
closed on unknown variants.

**4. v2 republication — first production `create_v2` call sites** — staged
fast and quality tiers are written via `VectorIndex::create_v2` with the
complete identity bundle (executing embedder identity with canonical
`fsvi-v2`/F16/little-endian storage), generation sequence = attested prior
+ 1, OS-entropy nonce, then re-admitted through
`TwoTierIndex::open_admitted_v2_with_paths` — attested by construction.
Reviewer caution honored: the proof reads the fingerprint OUT OF THE
ARTIFACT'S OWN HEADER BYTES via `open_admitted_v2`
(`staging_same_producer_merges_and_republishes_attested_v2`), never only
the in-memory bundle.

## Red proofs (transcripts in `bd-9xuj-c4write-red-proofs-20260731/`)

All transcripts carry date, tree/sha256 identification, and `EXIT_STATUS`.
Mutations were applied to the FINAL bytes
(`refresh.rs` sha256 `33d9e386e5210a572381505071e0b31bc2007ac4ef49520696a2393be1466fc8`)
and restored byte-identically (hash re-verified after each restore).

| Proof | Test / transcript | Result |
| --- | --- | --- |
| (a) positive re-enablement | `staging_same_producer_merges_and_republishes_attested_v2` — merge succeeds; staged header space fingerprint == producing space, read from artifact bytes via `open_admitted_v2`; sequence 7→8; carried row byte-stable, updated row changed; canonical untouched; queue untouched | green (`red-proofs-green.txt`, EXIT_STATUS=0) |
| (a) parent-red | same fixture through `run_cycle` on the pre-slice tree (containment refresh.rs at `c73dbe8e`) refuses with `IndexVersionMismatch{expected:1,found:2}` before any drain/embed/write | `parent-red-a.txt`, EXIT_STATUS=0 (refusal asserted) |
| (b) cross-space | `cross_space_attempt_rejects_with_typed_space_identity` — NEW typed `refresh.fast_space_identity` (value = executing space fingerprint, reason carries attested fingerprint), canonical + staging | green; mutation M3 red (`mutation-m3-red.txt`, EXIT_STATUS=101) |
| (c) declared-not-attested | `bootstrap_publishes_declared_identity_never_attested` (declared retained, attested false, merge still refused `identityless-fsvi-v1`) + `builder_declared_identity_is_never_attested` and in-memory discriminator pins (index crate, commit `c73dbe8e`) | green |
| (d) ConformanceCompatibleProducer | `certified_sibling_producer_refused_attestation_unavailable` (fixture asserts `is_conformance_compatible_with` both ways; refusal value `executing-producer-attestation-unavailable`; reason carries BOTH producer fingerprints) + per-record seam variant `two_faced_embedder_refused_at_bound_record_seam` | green; mutations M1 (`mutation-m1-red.txt`) and M2 (`mutation-m2-red.txt`) red, EXIT_STATUS=101 |
| (e) copied certificate (reviewer addition) | `copied_certificate_foreign_producer_rejects` — same space, different producer, byte-identical `golden_vectors` copy → REJECTS (attestation-unavailable); the regression guard for the future compatible lane | green; covered by mutation M2 red |
| finding-1 pre-drain | `live_v2_same_producer_cycle_refuses_composite_authority_with_zero_side_effects` — zero queue/embed/write/staging side effects, retry_count 0 | green |
| finding-2 pin | `publish_staged_canonical_refuses_split_generation` | green |
| supporting | `staging_never_resurrects_tombstoned_rows`, `staging_quality_tier_carries_and_binds`, `live_v2_quality_without_quality_embedder_refuses_republication`, all containment-era operational pins (retry-budget zero, byte-preservation, WAL/tombstone fleet pins) kept green | green |

## Verification (exit statuses)

- `cargo test -p frankensearch-core -p frankensearch-index -p frankensearch-fusion`
  → **exit 0** (core 1002; index 531 lib + all suites incl. the C1r2
  `space_identity_roundtrip`; fusion 929 lib + composition/interaction/
  sync-searcher suites; doctests). No model-dependent failures — no `--lib`
  scoping was needed. Transcript: `verify-three-crate-tests.txt`.
- `cargo clippy --no-deps -p frankensearch-index --all-targets` and
  `-p frankensearch-fusion --all-targets` → **exit 0**; my lines clean.
  Remaining warnings are pre-existing: 5× simd `mul_widen` deprecations
  (index) and the `TELEMETRY_TIMESTAMP_FALLBACK_RFC3339` dead-code const
  (fusion searcher.rs, present on the base tree).
- `cargo fmt -p frankensearch-index -p frankensearch-fusion -- --check` →
  clean.
- UBS: `refresh.rs` criticals **0** (baseline 0); `two_tier.rs` +
  `in_memory.rs` criticals **4 → 4** (all pre-existing `panic!` sites in
  old tests; zero new criticals). Warning deltas are `expect()` idiom in
  added test code.

## Deliberate scope boundaries (for the next slices)

- Canonical publication of v2 replacements awaits composite generation
  authority (bd-xomn.1/.3); until then `run_cycle` refuses admissible v2
  replacements pre-drain and the staged merge is the proven write path.
- The serving/read side (IndexCache/daemon open of a v2 canonical
  generation) is the read-slice's surface; nothing in this slice installs a
  v2 canonical generation, so no serving path regresses.
- The bootstrap lane still publishes v1 (with declared identity retained
  process-locally, C2); an identity-persisting v1→v2 bootstrap conversion
  is out of scope here and becomes trivial once composite authority lands.
