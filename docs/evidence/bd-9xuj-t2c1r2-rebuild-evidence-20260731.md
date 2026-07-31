# bd-9xuj T2-C1r2 rebuild + C3 replay — evidence card

**Date:** 2026-07-31 · **Branch:** `codex/sandygrove-t2c1r2-20260731`
**Acceptance criteria:** YellowSparrow integration review, messages #7388/#7397
(six verbatim requirements; all addressed below).

## 1. Base and commits

Base (protected tip, exact):
`58726e262af643403373cd7475387639dd2abe99` (`origin/main`,
"fix(index,fusion): three fleet-review data-loss bugs"), tree
`5706d2498b8afb16f773dfab675a4a55d7a01220`.

Ordered commits on the branch:

| # | Commit | Tree | Content |
|---|--------|------|---------|
| 1 | `02c1c7834bfa866b2897f4107d8f5cccbdaaddb5` | `68c6964b6267da82ae8e42119d2a1665c2ebd9a7` | T2-C1r2: rebuilt typed space-identity admission (`core/src/types.rs`, `core/src/lib.rs` one-name export, `index/tests/space_identity_roundtrip.rs`) |
| 2 | `890e64c44a559ece8a7a26995a384614471ff7ef` | `36dcc01d1cdfe4b88642ebb0d0af9a1fc74538c1` | T2-C3 replay: in-memory space identity (`index/src/in_memory.rs` only) |
| 3 | `19357aa0761744e458475428fad838377f2cc1a3` | `c9398266ceaadf9a71cb73a92d5ab7c75b63edd2` | docs-only readiness-map corrections (§5.1–§5.4) |
| 4 | (this card + committed red-proof transcripts, docs-only) | — | see `bd-9xuj-t2c1r2-red-proofs-20260731/` |

## 2. Supersession chain (ordered, truthful)

1. `8cb3c3e7` — first C1 (branch `codex/sandygrove-t2c1-20260731`, based on
   `7d448933`). **NO-GO** by integration review: no bind-time validation;
   obsolete same-space-is-sufficient admission law; synthetic FSVI evidence;
   new UBS-critical `panic!`. Never pushed to a protected ref.
2. `548a6177` — original C3 (branch `t2-c3`, parented on `8cb3c3e7`). Frozen
   content; never NO-GO'd, but stranded on the superseded parent.
3. `e57f7676` / `1a36c676` — intermediate rebuild pair on this branch,
   superseded by an in-branch fold of a clippy/rustfmt fix into the C1 slice
   (content otherwise identical; red proofs were re-banked against the final
   SHAs, not these).
4. `02c1c783` (C1r2) and `890e64c4` (C3 replay) — the current, final slices.

C3 replay deviations from frozen `548a6177` (both confined to test code in
`in_memory.rs`, documented in the commit message):
- the two `other => panic!` match arms became let-else pins in
  Result-returning tests (requirement #4: no new panic paths in this train);
- the admitted-v2 fixture moved into an isolated directory because the new
  base's admission snapshots the containing directory
  (`DirectoryChangedDuringRead`); with the shared test temp dir the test was
  flaky (reproduced 3/9 at `--test-threads=4`; 8/8 green after isolation).

## 3. Requirement-by-requirement

1. **Constructor validates identity.** `BoundQueryEmbedding::new`
   (`crates/frankensearch-core/src/types.rs:167`) calls
   `EmbeddingIdentityBundleV1::validate()` (the real, pre-existing validate
   API in `core/src/generation.rs`) before binding and surfaces its typed
   `SearchError::InvalidConfig`. Red proof 1.
2. **Same-space-is-sufficient dropped.** `verify_space_identity`
   (`types.rs:288`) is documented and tested as the space JOIN — necessary,
   never sufficient for a foreign producer. The corrected admission law lives
   in `verify_producer_conformance` (`types.rs:331`): same producer →
   `SpaceIdentityAdmission::SameProducer`; different producer admitted only
   via `is_conformance_compatible_with`'s pinned golden-vector certificate →
   `SpaceIdentityAdmission::ConformanceCompatibleProducer` (typed telemetry,
   enum at `types.rs:371`, stable codes at `types.rs:388`); otherwise typed
   rejection at `query_embedding.<tier>.producer_conformance`. Red proof 3
   witnesses the resurrected obsolete law being caught.
3. **Producer-conformance certification/telemetry + real create_v2
   roundtrip.** `crates/frankensearch-index/tests/space_identity_roundtrip.rs`
   writes a real FSVI v2 artifact through the production
   `VectorIndex::create_v2` writer (`index/src/lib.rs:1819`), admits it via
   `VectorIndex::open_admitted_v2` (`index/src/lib.rs:1647`), pins
   header-hex == core-side `space.fingerprint()`, then joins
   (`verify_space_identity`), certifies (`SameProducer` and
   `ConformanceCompatibleProducer` against the artifact's frozen bundle), and
   rejects a wrong model at equal dimension — all against the artifact's own
   header bytes. Not synthetic.
4. **No new panic paths.** The `8cb3c3e7` test `panic!` is gone; all new
   error-variant pins use let-else with `Result<(), String>` tests. UBS
   confirms: CRITICAL `panic!` count on the changed files is 1 → 1
   (base → head); the single finding is pre-existing on `origin/main`
   (`types.rs:678` at base `58726e26`, shifted to `types.rs:858` at head —
   the pre-C1 `verify_space` test), not introduced by this train.
5. **LegacyUnidentified boundary documented.** Doc section on
   `verify_space_identity` (`types.rs:262`–`287` region): expected
   fingerprints come only from identity-bearing sources; `create_v2` has
   zero production call sites at `58726e26` (only in-crate and
   `native_hnsw.rs` test fixtures; `TwoTierIndexBuilder::finish` →
   v1 constructors, `identity_v2: None`), so every production index is
   identity-less v1 and must be routed as typed `LegacyUnidentified`
   (`FsviReindexReason::LegacyUnidentified` → `RecoveryPlan` reindex) — never
   fabricated, never dimension-only, never warn-and-proceed. Mirrored in the
   roundtrip test's module docs and `in_memory.rs` field docs
   (`in_memory.rs:89`, accessor `:359`).
6. **Readiness-map correction is its own docs-only commit.** `19357aa0`
   (map §5.1–§5.4), separate from both code slices; truthful about the
   NO-GO history.

## 4. Verification (exact commands, exit statuses)

All cargo runs via a wrapper that sets `RCH_DISABLE=1` inside the script
body (`scratchpad/cargo-local.sh`; no `[RCH] remote` lines observed), with
`CARGO_TARGET_DIR=/data/projects/frankensearch/target` and a scratchpad
`TMPDIR`; `--test-threads=4`.

| Command | Result |
|---|---|
| `cargo test -p frankensearch-core -- --test-threads=4` | ok — 1001 + 1 + 2 passed, 0 failed; `EXIT_STATUS=0` |
| `cargo test -p frankensearch-index -- --test-threads=4` | ok — lib 518, fsvi_roundtrip 25, in_memory_tests 7, ivf_recall 1, space_identity_roundtrip 1, zero_signal 3, doctests 2(+1 ignored), (+5 bench-registered) all passed, 0 failed; `EXIT_STATUS=0` |
| `cargo clippy --no-deps -p frankensearch-core --all-targets` | zero warnings; exit 0 |
| `cargo clippy --no-deps -p frankensearch-index --all-targets` | only pre-existing `simd.rs` `mul_widen` deprecations (5, +5 dup) and one pre-existing bench warning; zero on changed files; exit 0 |
| `cargo fmt -p frankensearch-core -p frankensearch-index -- --check` | clean; exit 0 |
| in_memory flake probe (post-fix) | 8/8 runs `EXIT_STATUS=0` |

**UBS on changed files** (`ubs types.rs lib.rs in_memory.rs
space_identity_roundtrip.rs`, per-check counts diffed against the same files
extracted at base `58726e26`): CRITICAL 1 → 1 (**+0**, the pre-existing
panic above). Zero new findings on production lines. All positive deltas are
inventory-style counts over the added *test* code (assert inventory +48,
`unwrap()/expect()` usage +138, `expect_err` inventory +11, `std::fs` usage
+9, clone-in-test-loops +8, etc.) — the identical house style the base
already carries on these same files (172 asserts / 136 unwraps at base).

## 5. Red proofs (mutation-red, banked against final SHAs)

Transcripts committed beside this card in
`docs/evidence/bd-9xuj-t2c1r2-red-proofs-20260731/` (also at
`scratchpad/red-proof-*.txt`). Every run states the mutation, the exact
HEAD (`890e64c4…`), the `git diff --stat` of the mutation, and ends with an
explicit `EXIT_STATUS` line. After each proof the mutation was reverted by
editing back; `git status`/`git diff HEAD` confirmed byte-identical to HEAD
and the suites re-ran green.

| Proof | Mutation | Failing tests (all `EXIT_STATUS=101`) |
|---|---|---|
| `red-proof-1-bind-validation.txt` | `identity.validate()?` removed from `new()` | `types::tests::bind_time_validation_rejects_incoherent_identity_bundle` |
| `red-proof-2-space-join.txt` | `verify_space_identity` → unconditional `Ok(())` | core: `same_dimension_wrong_space_is_rejected_by_space_fingerprint`, `distinct_models_at_equal_dimension_always_reject_space_scoped`; index: `in_memory::tests::bound_query_joins_in_memory_space_identity_through_verifier` (C3's banked red proof, reproduced), `create_v2_roundtrip_joins_and_certifies_through_real_artifact` |
| `red-proof-3-producer-conformance.txt` | uncertified-producer rejection replaced with unconditional admit (obsolete law resurrected) | `types::tests::foreign_producer_admission_requires_conformance_certificate` (fails with "uncertified foreign producer must be rejected: SameProducer") |

## 6. File:line claims (verified against tree `c9398266…`, final code tree `36dcc01d…`)

- `crates/frankensearch-core/src/types.rs:167` `BoundQueryEmbedding::new`
  (validate-first); `:214` `space_fingerprint()`; `:233` `verify_space`
  (full-bundle, embedder↔embedder); `:288` `verify_space_identity` (space
  join + LegacyUnidentified boundary docs); `:331`
  `verify_producer_conformance`; `:371` `enum SpaceIdentityAdmission`;
  `:388` `code()`.
- `crates/frankensearch-core/src/lib.rs:185` — the single added export name
  (`SpaceIdentityAdmission`).
- `crates/frankensearch-index/src/in_memory.rs:89`
  `space_fingerprint_hex: Option<String>`; `:230`
  `from_vectors_with_identity`; `:278` `from_admitted_v2`; `:287`
  `from_open_index`; `:359` `space_fingerprint_hex()`; `:1316`/`:1327`
  per-tier passthroughs.
- `crates/frankensearch-index/tests/space_identity_roundtrip.rs:67` the real
  create_v2 roundtrip test.
- Pre-existing (NOT this train): `types.rs:858` `other => panic!` in the
  pre-C1 `verify_space` test (present at base `58726e26` line 678).
