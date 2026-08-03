# bd-9xuj T2 C4-write r2 — NO-GO repair: real pre-drain admission, read-only classification, retained owners (evidence card)

Date: 2026-07-31. Branch: `codex/sandygrove-c4write-r2-20260731`.
Successor chain — NOT a rewrite: the r1 branch `codex/sandygrove-c4write-20260731`
stays frozen at its NO-GO'd tip; this branch is cut off exactly that tip.

## NO-GO chain (40-hex)

| Commit | Role |
| --- | --- |
| `868c0801b52b556c9a6991b6ca1a98b4802d30e9` | Frozen r1 tip (evidence card + transcripts). NO-GO'd by SwiftBass audit #8325, acceptance FoggyPrairie #8326. The split-publication guard itself (composite-generation-authority refusal) was GO; the three items below blocked. |
| `b1056dd9` | r2 (i): index-crate owner retention (`TierSource`, whole-owner `admit_v2_tier`, `fast_admitted_owner`/`quality_admitted_owner`) + read-only `two_tier::observe_tier` + builder write-side WAL hygiene + unit tests |
| `02e17c97` | r2 (ii): fusion pre-drain repair (`admit_existing_generation` performs full admission; `inspect_tier` via `observe_tier` only; staging consumes the SAME retained owners) + integration tests |
| `afcae1db` | r2 (iii): panic-free test diagnostics (UBS zero-new-criticals vs the 868c0801 baseline) |
| (this commit) | r2 (iv): this card + banked red-proof/verification transcripts |

## The audit findings (quoted intent, line-cited against 868c0801)

1. **Pre-drain claims admission it doesn't perform** — `run_cycle`'s
   `ensure_canonical_cycle_admissible` (refresh.rs:818-835 at 868c0801) only
   reached `VectorIndex::inspect` (header parse, lib.rs:1603-1617): it never
   called `open_admitted_v2`, never recomputed content/docset digests, never
   retained `ValidatedFsviBytes`. A header-valid/content-corrupt v2 artifact
   sailed past to the WRONG refusal (composite authority — which asserts
   admissibility).
2. **`inspect` is not side-effect-free** — `read_header_for_inspection` uses
   ordinary `File::open` (atime, lib.rs:3841-3890); worse, `inspect_tier`
   routed v1/ReindexRequired tiers through MUTABLE `VectorIndex::open`
   (refresh.rs:244-250 at 868c0801), which deletes stale WAL sidecars and
   truncates corrupt WAL trailers (lib.rs:1668-1674, 1726-1777) — a mixed
   v2+v1 generation could MUTATE during classification, before refusal.
3. **The admitted owner is peeled** — `admit_v2_tier` called
   `open_admitted_v2` then extracted `validated.index`
   (two_tier.rs:1957-1965 at 868c0801), dropping the `ValidatedFsviBytes`
   capability, complete witness, and publication state — contradicting the
   owner contract at lib.rs:804-816 ("no conversion into a
   mutable/path-opened `VectorIndex`").

## Option-A choice (repair A): FULL admission at pre-drain — and why

Chosen: perform real `open_admitted_v2` admission during pre-drain
classification (SwiftBass's stated preference), NOT the format-only
renaming alternative.

- Cost basis: admission is one artifact copy + SHA-256/digest recomputation;
  a refresh cycle batch-embeds documents through ML models. Admission cost is
  second-order, and the admission-bearing path only runs when the queue is
  non-empty (idle polls return before any classification).
- Correctness basis: it makes every published claim literally true — the
  composite-authority refusal now fires only for a generation that ACTUALLY
  admitted (content digests recomputed), and a content-corrupt artifact gets
  its own typed `IndexCorrupted` digest error (required test iii).
- The rejected alternative (format-only classification) would have preserved
  the wrong-refusal hazard as a documented gap and killed the "admission"
  vocabulary everywhere; strictly worse when the full check is this cheap.
- Single admission, not double: `stage_identity_bound_generation`'s former
  separate admission step was deleted; staging consumes the SAME retained
  owners the classification produced (refresh.rs:1244 `admit_existing_generation`,
  staging step "1+3" comment).

## What each repair is (r2 derivation points)

| Repair | Where (r2 tip) |
| --- | --- |
| A. Pre-drain full admission, owners retained through the check | `refresh.rs:1244` `admit_existing_generation` → `refresh.rs:624` `admit_existing_tier` (binding reconstruction + `open_admitted_v2`) → `refresh.rs:612` `AdmittedCanonicalTier { owner, attested_space_hex }`; `refresh.rs:667` `ExistingGenerationClass::AttestedV2 { fast, quality }`; gate at `refresh.rs:1322` `ensure_canonical_cycle_admissible` |
| B. No mutable v1 open during classification | `refresh.rs:272` `inspect_tier` routes ALL recognition through `frankensearch_index::two_tier::observe_tier` (`two_tier.rs:2211`), read-only: same crate parsers (`parse_header`/`parse_v2_header`), read-only `wal::read_wal`, same staleness predicate as `VectorIndex::open` (`two_tier.rs:2282` `observe_v1_wal`) — no deletion, no truncation, no writable mapping. Opens prefer the crate's `O_NOATIME\|O_NOFOLLOW\|O_CLOEXEC` opener (the exact-admission opener), falling back to an ordinary read-only open where denied/unsupported — never weaker than `VectorIndex::inspect`'s unconditional ordinary open. |
| C. Sealed owner retained | `two_tier.rs:422` `TierSource::{PathOpened, AdmittedV2(ValidatedFsviBytes)}` (whole owner, never peeled); `two_tier.rs:2069` `admit_v2_tier` returns the owner whole; accessors `two_tier.rs:945/954` `fast_admitted_owner`/`quality_admitted_owner`; `refresh.rs:721` `StagedIdentityBoundGeneration::{fast,quality}_admitted_owner` delegate to the retained owners |
| Write-side WAL hygiene (consequence of B) | `two_tier.rs:1984` (+ quality twin): `TwoTierIndexBuilder::finish` removes the adjacent WAL sidecar of each tier it just rewrote — classification no longer deletes stale sidecars, so the WRITE path owns that cleanup (mod-256 generation wraparound could otherwise resurrect foreign rows) |

## Claim corrections (repair E) — what is ACTUALLY verified now

The r1 code and card used "admissible"/"identity gates PASSED" for a check
that was header-only. Corrected on this card and in code:

- r1 `composite_authority_refusal` reason claimed "the identity-bound v2
  replacement ... is admissible" after only a header parse. r2 reason
  (refresh.rs:334) states: "the existing generation ... was fully admitted
  (header identity gates plus recomputed content/docset digests via exact v2
  admission)" — and that is now literally what happened before the refusal.
- r1 card's "admission/inspection alone drains nothing, embeds nothing,
  writes nothing" (zero-side-effect claim) was TRUE for the v2 fixture it
  tested but FALSE as a general claim: v1 classification could delete/truncate
  WAL sidecars. r2 makes the general claim true and pins it with
  byte/mtime/directory-listing invariance tests across v2, v1-stale-WAL,
  corrupt-trailer-v1, and mixed v2+v1 fixtures.
- "All identity gates passed" now additionally means: content and docset
  digests recomputed byte-for-byte over the artifact's bytes, sealed owner
  retained. A header-valid/content-corrupt artifact can no longer be called
  anything but corrupt.

## Deliberate fail-closed narrowing (documented behavior change)

Read-only v1 observation cannot see record flags without a record-table
inspector, so `FsviV1Observation::retains_content` counts tombstoned rows
conservatively. Consequence: an ALL-TOMBSTONED v1 tier (record_count > 0,
live 0, no active WAL) now takes the `identityless-fsvi-v1` refusal instead
of classifying bootstrap-replaceable as on r1. Empty-seed bootstrap
(record_count == 0) is behavior-identical to r1. Pinned by
`all_tombstoned_v1_fails_closed_as_retaining_content` (parent bootstraps
there — transcript pass A shows the parent FAILING this pin by succeeding).

## Deferred lib.rs seam (NOT built here — GoldThrush observational-open train)

To restore flag-level v1 precision and give `inspect` a no-atime spine, the
index crate root needs (exact proposed signature):

```rust
/// Read-only, no-atime, record-level observation of a legacy FSVI v1
/// artifact. Never deletes or truncates the WAL sidecar; never maps
/// writable; O_NOATIME|O_NOFOLLOW|O_CLOEXEC with a typed rejection where
/// unsupported.
pub fn VectorIndex::observe_v1(path: &Path) -> SearchResult<FsviV1DetailedObservation>

pub struct FsviV1DetailedObservation {
    pub metadata: VectorMetadata,   // parsed v1 header (record_count, gen, ...)
    pub live_record_count: usize,   // record-table flags actually read
    pub tombstone_count: usize,
    pub active_wal_records: usize,  // generation-matched, deduped
    pub wal_state: FsviV1WalSidecarState, // Absent|Empty|Active|Stale|CorruptTrailer{valid_len}
}
```

When it lands, `two_tier::observe_tier` delegates its v1 arm to it and the
all-tombstoned narrowing above reverts to exact classification.

## Tests + red proofs

Red proofs run against the FROZEN tip via `git archive 868c0801 | tar -x`
(no worktree ops; frozen branch untouched), appending ONLY the new tests.
Transcripts in `bd-9xuj-c4write-r2-red-proofs-20260731/` with commands and
pipefail EXIT_STATUS lines.

| Required test | Name | Parent (868c0801) | r2 |
| --- | --- | --- | --- |
| (i) v1-stale-WAL invariance | `classification_never_deletes_a_stale_wal_v1` | FAILED — WAL deleted during classification (pass A, EXIT_STATUS=101; live WARN "discarding stale/mismatched WAL entries and removing file" fired from the classify path) | ok |
| (i) corrupt-trailer-v1 invariance | `classification_never_truncates_a_corrupt_wal_trailer_v1` | FAILED — trailer truncated (pass A; live WARN "truncating corrupted WAL trailer") | ok |
| (i) v2 directory invariance | `pre_drain_full_admission_leaves_canonical_directory_invariant` | passed (v2 inspection was already byte-invariant; invariance HOLD, not a repair pin) | ok |
| (ii) mixed v2+v1 | `mixed_v2_fast_v1_quality_classifies_without_mutating_either` | FAILED — quality tier's stale WAL deleted before the refusal (pass A) | ok |
| (iii) content-corrupt v2 → correct refusal | `header_valid_content_corrupt_v2_refuses_admission_not_composite` | FAILED — parent returned the composite-authority refusal, i.e. certified a corrupt artifact admissible (pass A transcript quotes it verbatim) | ok — typed `IndexCorrupted` "digest mismatch" |
| (iv) retained-owner path replacement | `staged_generation_retains_admission_owners` (integration) + `admitted_owner_reads_survive_path_replacement`, `admitted_v2_open_retains_sealed_owners_in_full` (unit) | compile-RED: E0599 `fast_admitted_owner` does not exist (pass B EXIT_STATUS=101; pass C EXIT_STATUS=101 adds E0425/E0433 for `observe_tier`/`FsviTierObservation`) | ok |
| narrowing pin | `all_tombstoned_v1_fails_closed_as_retaining_content` | FAILED — parent bootstraps over an all-tombstoned v1 (pass A) | ok |
| flow guard | `bootstrap_over_empty_seed_with_stale_wal_does_not_resurrect_foreign_rows` | passed (parent got the same end state via the delete-at-classify path; r2 gets it via write-side hygiene) | ok |
| observation unit tests | `observe_tier_classifies_v1_seed_live_and_v2_without_mutation`, `observe_tier_never_deletes_a_stale_wal_sidecar`, `observe_tier_never_truncates_a_corrupt_wal_trailer`, `observe_tier_reports_upgrade_required_for_future_schema`, `plain_v1_open_has_no_admitted_owners`, `builder_finish_removes_adjacent_wal_sidecars` | API absent (covered by pass C compile-red) | ok |

Invariance evidence, what is byte-compared: full `std::fs::read` compare of
every artifact and WAL sidecar before/after the refused cycle, plus a sorted
directory manifest (file name, byte length, mtime) of the canonical index
dir (`dir_manifest`). Fixture staleness is proven inside each test by a
CONTROL ARM: a throwaway copy of the same main+WAL pair is mutably opened and
the WAL provably vanishes/shrinks — the hazard demonstrated live, then shown
absent on the classification path. atime is deliberately not asserted
(relatime; the no-atime open is best-effort by platform and documented as
such — the strict no-atime v1 inspector is the deferred seam above).

## Verification (r2 tip, transcripts banked)

| Check | Result |
| --- | --- |
| `cargo test -p frankensearch-index -p frankensearch-fusion -p frankensearch-core` (all targets + doc-tests) | ok — 1002+937(+4 ignored)+539 lib tests plus all integration/doc suites, 0 failed, EXIT_STATUS=0 (`verify-r2-tests.txt`) |
| `cargo clippy --no-deps -p frankensearch-index -p frankensearch-fusion --all-targets` | EXIT_STATUS=0; remaining warnings are pre-existing and not on r2 lines: 6x `wide::i16x8::mul_widen` deprecations (simd.rs, untouched) + 1 dead const (searcher.rs, untouched) (`verify-r2-lints.txt`) |
| `cargo fmt -p frankensearch-index -p frankensearch-fusion -- --check` | EXIT_STATUS=0 |
| UBS on the two changed files | Critical: 4 vs parent baseline 4 — identical set (the parent's own pre-existing mounted-filesystem test `panic!` block, line-shifted). ZERO new critical findings. Warning-count delta is proportional to ~1,400 added test lines and consists of the same classes as the parent baseline (expect/unwrap in tests etc.). |

## File manifests (sha256)

| File | 868c0801 (parent) | r2 tip |
| --- | --- | --- |
| `crates/frankensearch-index/src/two_tier.rs` | `b4db6971a2c06adbd019b915c5d4e47aa175574f1bf7b71a1888fc0c01741794` | `9cce5d44a0e01ec701b28a352ea0ca0f8738ec844e13d008e5518f829361c73e` |
| `crates/frankensearch-fusion/src/refresh.rs` | `33d9e386e5210a572381505071e0b31bc2007ac4ef49520696a2393be1466fc8` | `51b46765ffb6e81837029f6264b9aa6c249eb1f00ce780d04ad43e5c1a52a0b2` |

`crates/frankensearch-index/src/lib.rs` is byte-untouched on this branch
(not this lane's file); `in_memory.rs` required no change —
`from_admitted_v2(&ValidatedFsviBytes)` already borrows the owner without
consuming it.
