# QG-6 R1 evidence card — all-grimoire validate-once on the rebind path (validated TERMDICT metadata reused across tombstone-only rebinds)

- Author: SandyGrove (delegated by FoggyPrairie #7268, frozen design #6660).
  Round-3 corrections per integration reviewer YellowSparrow #7397.
- Date: 2026-07-31 (America/New_York)
- Class: **causal SELF-SPEEDUP / NoClaim, unsealed artifacts.** Quill-vs-Quill
  only; **no incumbent leg** (Tantivy deferred to R0's fastest-Tantivy
  tournament), **no QG activation**, no competitive claim. The terminal
  KEEP/REJECT belongs to bd-quill-gauntlet-qg6-cache-termdict-gwd4 (still
  blocked on gxwy + x4e4.5.4).
- **No-claim statement (binding for every use of the numbers below): the
  3.67x figure (median 3.673x) is a mechanism/diagnostic measurement on the
  tombstone-rebind path only. It is NOT a public QG speed claim and NOT
  gate-activation evidence. The measured scope is "all-grimoire
  validate-once on the rebind path" — the self-time family it collapses is
  the sum over ALL `frankensearch_quill::grimoire::` symbols on the rebind
  path, not an isolated validation-only family, and no query-side or
  whole-benchmark speedup is asserted.**
- Machine class: local-5975wx-32c (host thinkstation1, AMD Threadripper PRO
  5975WX, 64 hw threads; CPU frequency scaling active — mitigated by
  interleaved pairing + both A/A nulls). Timed runs pinned to core 8;
  builds on cores 32-63. All runs foreground, `set -euo pipefail`.

## Supersession chain (ordered; every stated SHA full 40-hex)

The candidate lives on topic branch `codex/sandygrove-qg6r1-20260731`. The
ACTUAL ordered chain, first to last:

1. `c08b8fc265a0de97d8cb4a412c262bdf4e050b1d` — chain base: the s1rc1
   Arc-backed `EncodedSegment` commit (bd-s1rc1) on which the R1 candidate
   was cut. It is the first commit of the branch chain above origin/main
   `504fa185c6a392f8e9e48a8a28e70f1a235a8361`; it is not itself an R1
   review round.
2. `793de7a5bf2459b571c349321486a148bc57ae67` — the R1 candidate as first
   reviewed (round 1). Review verdict: NO-GO with two narrow blockers,
   design accepted. This commit is immutable; it was superseded, not
   amended.
3. `dce0b3c3989dbc3702d41abff6ebcec14f8f77f1` — round-2 successor,
   produced by the round-1 review to fix its two blockers: (a) the
   owned-only rebind trust gate (mapped backings now rejected with a typed
   reopen-required `InvalidTransition`), (b) honest estimated
   payload-bytes accounting for validated TERMDICT metadata. The round-2
   review (YellowSparrow #7397) found three residual issues in it: a
   test-side `let-else`/`panic!` introduced in that round, the evidence
   card not yet committed with the correct chain, and insufficiently
   tightened no-claim wording.
4. **This successor** — round-3 successor, produced by review round 2
   (#7397): removes the `let-else`/`panic!` from
   `mapped_segment_rebind_is_rejected_with_reopen_required_transition_error`
   (the check is folded into the existing `assert!(matches!(..))` over
   `Result::err`, which reports `{error:?}` instead of panicking via
   `let-else`; zero new panic paths, zero new unwrap/expect sites),
   commits this corrected card, and tightens the no-claim wording.
   No production-code change. A commit cannot embed its own hash: the
   fourth element's SHA is the tip of `codex/sandygrove-qg6r1-20260731`
   carrying this card, recorded in the #7397 review thread.

No history was rewritten at any round: each successor is a new commit on
the same chain; every reviewed SHA above remains reachable and immutable.

## Design (what changed and why it is safe)

Base `c08b8fc265a0de97d8cb4a412c262bdf4e050b1d` (s1rc1 Arc-backed
EncodedSegment; fk04a landed) already validates TERMDICT metadata once per
**binding** and serves queries from borrowed views. The residual: every
snapshot successor transition that retains a segment
(`KeeperSnapshot::publish_owned_segments` → `RecoveredSegment::rebind` →
`bind_shared`) re-ran the complete TERMDICT parse/validation for each
retained segment although the backing bytes are the same immutable `Arc`
allocation. Tombstones live in the MANIFEST, not the FSLX image.

R1 (commit `793de7a5bf2459b571c349321486a148bc57ae67`): `bind_shared`
accepts a predecessor (`reuse_from`); the predecessor's
`Arc<ValidatedTermDictionaryMetadata>` is reused only when ALL
content-identity witnesses hold:

1. `Arc::ptr_eq` of the shared `RecoveredSegmentBacking` (exact-Arc fast path),
2. trailer-verified whole-file xxh3 recorded at validation time equals the
   reader's (`term_dictionary_file_xxh3` — checksum witness; the new
   manifest's `file_xxh3` is additionally re-checked by the retained
   `validate_witnesses` call in `rebind`),
3. schema equality,
4. exact address/length binding of the live TERMDICT slice
   (`TermDictionary::from_validated_metadata` gate, warm `OnceLock`
   section-checksum — no bytes re-hashed).

Any witness mismatch falls back to one complete fresh validation
(pinned by test), so reuse can never weaken admission. Fresh bytes, cold
reopen, and durable reload keep the existing validate-fresh path and the
landed fk04a error taxonomy verbatim (taxonomy tests untouched and green).

Byte accounting (bd-enf6z seam): **estimated payload bytes**, not an exact
memory claim. `ValidatedTermDictionaryMetadata::payload_bytes` (struct +
block/restart directory capacities; flat records) surfaced as
`RecoveredSegment::term_dictionary_metadata_payload_bytes` and
`KeeperSnapshot::term_dictionary_metadata_payload_bytes`. Documented
exclusions: `Arc` control-block overhead, allocation alignment, allocator
slack. Shared-Arc semantics: each concurrently held snapshot generation
reports the FULL shared payload, so summing across snapshots
double-counts; within one snapshot the sum never double-counts. **No QG-7
exact-memory claim is made.** Wiring into the cross-crate
`SegmentStats`/output schema is deliberately left to the QG-7 accounting
owner (bd-enf6z).

Trust boundary (round-1 blocker 1, fixed in
`dce0b3c3989dbc3702d41abff6ebcec14f8f77f1`): the reuse chokepoint is
**owned-only**. Any binding that names a predecessor (`rebind`) REJECTS a
`RecoveredSegmentBacking::Mapped` backing with a typed
`KeeperError::InvalidTransition` naming the segment and requiring a
durable reopen — a warm mapped reader's stored trailer hash and
once-checked section gates vouch for the ORIGINAL bytes, so neither reuse
nor an in-place re-validation may be labeled "fresh" there. Pinned by
`mapped_segment_rebind_is_rejected_with_reopen_required_transition_error`
(typed error + rejected rebind neither validates nor reuses; as of the
round-3 successor the pin is a single `assert!(matches!(..))` over
`Result::err` — no `let-else`/`panic!`). No production path change:
mapped snapshots never rebound in place.

## Ladder

1. **Full suite**: `cargo test -p frankensearch-quill` — 493 pass / 0
   fail / 1 ignored at `793de7a5`; 494 pass / 0 fail / 1 ignored at
   `dce0b3c3` and at the round-3 successor (includes all landed fk04a
   corruption + taxonomy tests, reopen pins, and the b61bbf89/5ee393fe
   binding-taxonomy pins, all unmodified).
2. **Result equality**: the new index-level test pins EXACT equality of
   the (global_docid, score-bit) sequences of ranked hits across a
   tombstone-only delete commit — a true element-wise comparison
   (`tombstone_only_delete_commit_reuses_termdict_metadata_and_preserves_results`).
   The harness additionally checks FNV-1a digest equality over the
   (docid, score-bits) result streams for 200 warm + 200 post-delete
   queries: digests matched on both arms in every one of 72 invocations
   (`8d3651c6b098cc0d` warm, `e794dbd80fbbbc41` post). **An FNV-1a
   fingerprint match is NOT byte-identity**: it is digest equality over
   the extracted (docid, score-bits) stream, with the usual fingerprint
   collision caveat, and it does not compare index bytes or full result
   envelopes. The element-wise index-level test above is the identity
   witness; the harness digests are corroborating fingerprints only.
3. **RED-proof**: with ONLY the lever disabled (rebind passes `None`),
   `term_dictionary_metadata_is_validated_once_per_bound_segment_and_reused_concurrently`
   fails at the exact-Arc assertion and
   `tombstone_only_delete_commit_reuses_termdict_metadata_and_preserves_results`
   fails at the reuse-count assertion (transcript in report). A fresh
   validation is the only way to mint a new metadata allocation, so
   `Arc::ptr_eq` + `full_validations == 1` is a sound short-circuit witness.
4. **Corrupt-after-caching honest observable**
   (`cached_termdict_views_do_not_re_read_bytes_while_fresh_open_fails_closed`):
   corrupting durable TERMDICT bytes after open leaves the live snapshot's
   cached views untouched (no re-read, `full_validations` stays 1) while a
   cold open of the same directory fails closed
   (`KeeperError::SegmentOpen` naming the exact segment path).
   Structural note: `publish_owned_segments` is in-memory-only; the durable
   writer publishes via full cold reopen (`open_snapshot_blocking`), so the
   reuse path can never see mutable bytes — in-memory backings are
   immutable shared allocations.

## Mechanism (before/after, honest attribution — diagnostic only)

Frozen-profile framing: the campaign's >=90.34% query-self-time /
18 MB-per-term-query TERMDICT numbers are from the pre-fk04a profile at
8b35790e. On THIS base those query-side reparses are already collapsed by
the landed fk04a lever — inherited, not claimed here. Census confirms:
warm-query allocations are byte-identical across arms
(6,008,577 B / 43,025 blocks for 200 queries; ~59 µs/query both arms).

R1's own delta is the rebind path (scope: all-grimoire validate-once on
the rebind path). 100k docs, 30k vocab, tier-merged to 4 sealed segments,
200 tombstone-only delete publications (transcripts/census-100k.txt):

| metric (tombstone_deletes phase)   | base (c08b8fc2) | lever (+R1) | delta |
|------------------------------------|-----------------|-------------|-------|
| wall per delete publication        | 11.75 ms        | 3.26 ms     | 3.61x |
| alloc bytes per phase (200 dels)   | 382,477,264     | 1,246,864   | 307x  |
| alloc blocks per phase             | 30,851          | 13,451      | 2.3x  |

Count-to-time conversion: base transient metadata alloc is 1.91 MB per
delete; the 8.5 ms/delete saved matches the validation-family share below.

Arm-scoped perf (F=999 self-time, delete-weighted run QUERIES=0
DELETES=400, whole-run samples incl. build):

- **All `frankensearch_quill::grimoire::` symbols summed: 57.74% (base) →
  3.60% (lever)**. Scope statement: this is the sum over EVERY symbol whose
  name contains `grimoire` — it includes `SliceReader` vint decoding and
  any non-validation grimoire helpers, not an isolated
  validation-only family (hence "all-grimoire"). The five named top
  symbols alone sum to 55.59% on base (decode_entry 23.09%,
  read_vint_multibyte 16.51%, validate_block 9.97%,
  validate_metadata_basic 3.13%, copy_bytes 2.89%).
- Lever-arm residual top: quiver::parse_id_hash / IdMapSection re-parse on
  rebind (route-next candidate, OUT of frozen R1 scope).

## A/B (interleaved paired + both nulls, Foggy terminal contract)

12 A/B rounds randomized order + 12 base/base + 12 lever/lever nulls;
100k docs, 200 queries, 200 deletes, seed 24301; timing binaries (census
allocator compiled OUT); core 8 (transcripts/ab-results.tsv).

| phase              | base/lever median | min–max       | rounds base slower | null bands (b/b, l/l)      |
|--------------------|-------------------|---------------|--------------------|-----------------------------|
| tombstone_deletes  | **3.673x**        | 3.404–3.967   | 12/12              | 0.950–1.014, 0.992–1.007    |
| build              | 1.040x            | 0.938–1.121   | 10/12              | 0.961–1.060, 0.982–1.021    |
| warm_queries       | 0.990x            | 0.893–1.025   | 3/12               | 0.954–1.128, 0.962–1.014    |
| post_queries       | 0.969x            | 0.876–1.029   | 3/12               | 0.958–1.042, 0.978–1.031    |

Verdict per phase: tombstone_deletes is a decisive self-speedup far
outside both nulls; build is suggestive but overlaps the null band — NOT
claimed; query phases are NULL — exactly the no-regression contract.
Restated per the no-claim statement above: the 3.673x median is a
mechanism/diagnostic measurement on the tombstone-rebind path only — it
is not a public QG speed claim and not gate-activation evidence.

## QG guard notes (diagnostic observations, not gate evidence)

- **QG-3 watch-mode reopen**: unchanged by construction — durable publish
  is a full cold reopen; reuse never crosses reopen. Reopen pin test
  ("each durable reload performs one fresh validation") green, unmodified.
- **QG-4 commit latency**: the 3.67x tombstone-only publication figure is
  a diagnostic on this card's fixture, not QG-4 gate evidence; retained-
  segment rebinds inside normal commits also stop re-validating. No
  commit-path regression measured.
- **QG-7 RSS/accounting**: persistent metadata now payload-accounted
  (estimated payload bytes; per-segment and per-snapshot accessors with
  documented exclusions and shared-Arc double-count semantics — no
  exact-memory claim). Retained payload is unchanged vs base (base minted
  an equal-sized allocation per rebind and dropped the old one); transient
  allocation shrinks 307x on the delete path. SegmentStats wiring deferred
  to bd-enf6z (cross-crate goldens).
- **QG-9 cold open**: untouched code path; fresh validation always.

## Round-2 successor verification tuple (as recorded at dce0b3c3)

- Base: `c08b8fc265a0de97d8cb4a412c262bdf4e050b1d`
- Reviewed (immutable, round-1 NO-GO): `793de7a5bf2459b571c349321486a148bc57ae67`
- Round-2 successor: `dce0b3c3989dbc3702d41abff6ebcec14f8f77f1`
- Tip tree at dce0b3c3: `721dbf5f890b055ad6acbdac858707a63ced3804`
- Successor patch-id (stable): `dfc5c8c404a6827b6358147bc13c4c2da09416f4`
- Full base..dce0b3c3 patch-id (stable): `ef5af856268b2dd65ab7365c17fc2073c915d9aa`
- Full base..dce0b3c3 diff sha256: `f6fa23a33c1319c9d25e611f3256cde017286fa2ddc72d163acffbb0dfd143ff`
- Exact-commit suite at dce0b3c3: 494 pass / 0 fail / 1 ignored
- Exact-file UBS at dce0b3c3 (keeper.rs, grimoire.rs, index.rs): exit
  nonzero with the inherited whole-file/test inventory — 312 critical
  heuristics / 5065 warnings / 1033 info (same scale as fk04a's recorded
  304/4820/1000 on these files). The one flagged dce0b3c3-round addition
  (a `panic!` inside a #[cfg(test)] let-else arm) is REMOVED by the
  round-3 successor.
- Timing caveat: the census/perf/A-B numbers above were measured with
  793de7a5-built binaries. dce0b3c3 adds one enum-discriminant check per
  predecessor-naming bind and renames accessors; the round-3 successor
  changes one test assertion and docs only; no measured path changes. The
  gwd4 terminal campaign reprofiles the exact landed SHA by contract.

## Provenance

- Base: `c08b8fc265a0de97d8cb4a412c262bdf4e050b1d` (origin/codex/sandygrove-s1rc1-20260730 tip; contains landed fk04a 8b864cb7+df9c22dd and taxonomy fixes 5ee393fe/b61bbf89 and origin/main 504fa185)
- Lever commit (round-1 reviewed): `793de7a5bf2459b571c349321486a148bc57ae67`
- Harness source (identical both arms, sha256): 961350a50d9e5f90a4cbcecdf877905a1e1ab847ff2ac7e45f4bae5a7f951c3b
- ELF sha256:
  - base-census  896f23c90b3938b6c341e1caf6fd55d2289dec0dbb855366671b544ad10465fe
  - base-timing  74631e64685a5e4e45aa41de7cd9207a19207fc5348b58a02f1a3bb9570950d5
  - lever-census cda4623a57146c34fc849a16fac0b1705e1f97f8be8d0eb7d9f18012e256b65b
  - lever-timing a68fb26ebfb72d16c141a4379ac42dde9b2e53895f6ef26daf7550cb06f82e41
- Transcripts: scratchpad/r1/transcripts/{census-100k.txt, ab-results.tsv, perf-run-{base,lever}.txt}; perf data: scratchpad/r1/perf-{base,lever}.data

## Repro (scripts in scratchpad/r1/, strict pipefail)

1. Suite: `bash cargo_local.sh test -p frankensearch-quill` (493/0/1 at 793de7a5).
2. RED-proof: change `rebind` to pass `None` instead of `Some(self)`; the
   two reuse tests fail at their reuse assertions; restore; suite green.
3. Binaries: `bash harness_build.sh` (4 ELFs, sha256s printed).
4. Census: `ARM=<arm> taskset -c 8 bin/r1_harness-<arm>-census` with
   DOCS=100000 SEGMENTS=10 VOCAB=30000 QUERIES=200 DELETES=200 SEED=24301.
5. Perf: `bash perf_record.sh`; then `perf report --stdio --sort symbol`.
6. A/B: `bash ab_driver.sh`; stats: `python3 ab_stats.py`.

## Rebase notes for the lander

- Code content is c08b8fc2..dce0b3c3 (quill-crate only: grimoire.rs,
  keeper.rs, index.rs) plus the round-3 test-assertion fix and this card —
  built directly on s1rc1's tip `c08b8fc2`; landing after s1rc1 is a
  clean cherry-pick of the chain.
- Deliberate behavior changes, both pinned by tests: (1) the rebind tail of
  `term_dictionary_metadata_is_validated_once_per_bound_segment_and_reused_concurrently`
  now pins reuse instead of fresh validation (frozen R1 design); (2) as of
  dce0b3c3, a mapped-backing rebind returns the typed reopen-required
  `InvalidTransition` instead of silently re-validating (previously
  unreachable in production). All other fk04a/taxonomy/reopen pins are
  untouched and green.
- The rebind reuse fires only on the in-memory backend today
  (`publish_owned_segments` is in-memory-only; durable publish cold-reopens
  — that IS the QG-3/QG-9 contract). The QG fixtures' Quill arm is
  in-memory, so the lever is live where QG-4/QG-6 measure.
- Landing checklist per the plumbing-clobber memory: assert worktree HEAD
  == expected base, `git diff --numstat` of exactly these files vs the
  landing parent, re-materialize on drift. Do not remove the worktrees
  destructively.

## Appendix — DRAFT ledger rows (do not paste into docs/ without gwd4 owner review)

These are drafts for the gwd4 verdict owner. The lever commits themselves
make no performance claim (fk04a protocol). Classification: causal
SELF-SPEEDUP / NoClaim with unsealed artifacts — no incumbent (Tantivy leg
deferred to R0's fastest-Tantivy tournament), no QG activation. The 3.67x
number in these rows is a mechanism/diagnostic measurement on the
tombstone-rebind path only ("all-grimoire validate-once on the rebind
path") — NOT a public QG speed claim and NOT gate-activation evidence.

### PERF_LEDGER.md draft row (if gwd4 admits the self-speedup as a maintenance entry)

| date | lever | class | workload | result | nulls | provenance |
|------|-------|-------|----------|--------|-------|------------|
| 2026-07-31 | quill: all-grimoire validate-once on the rebind path — reuse validated TERMDICT metadata across tombstone-only rebinds (bind_shared reuse_from; owned-only — mapped rebind rejected with typed reopen-required InvalidTransition; Arc::ptr_eq + file_xxh3 + schema + slice-binding witnesses; fallback = fresh validation of immutable owned bytes) | causal SELF-SPEEDUP / NoClaim, unsealed artifacts (Quill-vs-Quill; NO incumbent — Tantivy leg deferred to R0 tournament; NO QG activation; mechanism/diagnostic measurement on the tombstone-rebind path only — NOT a public QG speed claim, NOT gate-activation evidence) | in-memory 100k docs / 30k vocab / tier-merged 4 segments / 200 tombstone-only delete publications; core 8, local-5975wx-32c, interleaved paired | tombstone-only publication: median 3.673x (3.404–3.967, 12/12 rounds) — rebind-path diagnostic only; alloc bytes on delete phase 382.5MB -> 1.25MB (307x); self-time summed over ALL grimoire:: symbols (incl. SliceReader vint decode, not validation-isolated) 57.74% -> 3.60%; query phases NULL by design (0.99x/0.97x inside null band); result agreement = index test pins EXACT (docid, score-bit) sequence equality across a tombstone-only commit; harness adds FNV-1a digest equality over the (docid, score-bits) result streams across all 72 invocations — an FNV fingerprint match is NOT byte-identity | A/A base/base 0.950–1.014, A/A lever/lever 0.992–1.007 (both straddle 1.0) | src base c08b8fc265a0de97d8cb4a412c262bdf4e050b1d, round-1 reviewed 793de7a5bf2459b571c349321486a148bc57ae67, round-2 successor dce0b3c3989dbc3702d41abff6ebcec14f8f77f1, round-3 successor = tip of codex/sandygrove-qg6r1-20260731 carrying the QG-6 R1 card; ELF sha256 base-timing 74631e64…, lever-timing a68fb26e… (793de7a5-built; later successors change unmeasured paths only); transcripts scratchpad/r1/transcripts/ |

### NEGATIVE_EVIDENCE.md draft rows (the honest nulls found on the way)

1. Warm-query latency: NULL. On this base (fk04a landed) R1 does not move
   term-query latency or allocation (byte-identical census: 6,008,577 B /
   43,025 blocks per 200 queries both arms; ratios 0.893–1.029 inside the
   A/A band). The campaign's 18MB/term-query collapse belongs to landed
   fk04a, not to R1. Retry predicate: none — do not re-run a query-side
   A/B for this lever; the query-side lever above validated metadata is
   bd-e8h-w1-termdict-snapshot-cache-h0eq (blocked on gwd4).
2. Build/commit (new-segment) path: median 1.040x, 10/12, but min 0.938
   overlaps the base A/A null band (0.961–1.060) — NOT claimable at n=12.
   Retry predicate: a commit-heavy fixture with >=8 retained segments and
   >=50 commits per invocation would separate it if anyone needs the claim.
3. Route-next (unclaimed residual): lever-arm rebind residual is dominated
   by quiver::parse_id_hash + IdMapSection re-parse (23.2%+9.6% self-time
   on the delete-weighted run) — IDMAP/IDHASH lookup plans are rebuilt per
   rebind the same way TERMDICT metadata used to be. Out of frozen R1
   scope; candidate for a future bead with the same witness pattern.
