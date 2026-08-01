# bd-9xuj T2 C4-write r3 — NO-GO repair: durable WAL retirement BEFORE tier publication (evidence card)

Date: 2026-07-31. Branch: `codex/sandygrove-c4write-r3-20260731`.
Successor chain — NOT a rewrite: the r2 branch
`codex/sandygrove-c4write-r2-20260731` stays frozen at its NO-GO'd tip
`7095b1da`; this branch is cut off exactly that tip. Scope is EXACTLY the
dual-audit acceptance contract; the r2 admission and owner-retention work
both auditors ruled GO is untouched.

## NO-GO chain (40-hex)

| Commit | Role |
| --- | --- |
| `868c0801b52b556c9a6991b6ca1a98b4802d30e9` | Frozen r1 tip. NO-GO'd by SwiftBass #8325 / FoggyPrairie #8326. |
| `7095b1da60fe84f6dbd39a4bdc5ada4329580c90` | Frozen r2 tip. All three r1 blockers ruled repaired (GO), but NO-GO'd overall by SwiftBass #8366 + FoggyPrairie #8367 for the new crash-consistency blocker below. |
| `8944a29d` | r3 (i): durable WAL retirement BEFORE publication in `TwoTierIndexBuilder::finish` + reopen non-silence + three fault/crash-point tests (two_tier.rs only) |
| `316b6dd9` | r3 (ii): counting embedder spy asserted on every refusal path + refresh-seam crash-state fail-closed test + claim precision (refresh.rs + two_tier.rs docs) |
| `34b88062` | r3 (iii): panic-free crash-state test diagnostics (UBS zero-new-criticals, same discipline as r2's afcae1db) |
| (this commit) | r3 (iv): this card + banked red-proof/verification transcripts |

## The audit finding (quoted, line-cited against 7095b1da)

SwiftBass #8366: "`TwoTierIndexBuilder::finish` publishes each new canonical
legacy main first and only afterward performs best-effort WAL deletion …
`fast_writer.finish()?` completes, then `let _ = fs::remove_file(wal_path)`
ignores every failure" (fast: two_tier.rs:1961-1984; quality: :1986-2000).
"Writer publication is already durable before cleanup (lib.rs:3630-3717) …
New legacy main starts at `compaction_gen = 1` (lib.rs:1920-1950);
`next_generation(1) == 2` (lib.rs:5798-5800); observer/reopen treats a
generation-2 WAL as active against that new main (two_tier.rs:2291-2300).
Therefore a crash after rename/sync but before removal, a removal error, or
a crash before a post-removal parent-directory sync can leave/resurrect
foreign WAL rows on reopen. That falsifies the evidence-card claim that
write-side cleanup closes generation-collision/wraparound resurrection."

FoggyPrairie #8367 set the fail-closed acceptance contract: (1) durable
retirement before publication following lib.rs:1847-1911; (2) fault tests
for removal failure, crash after rename, directory-sync failure, and
generation-2 collision reopen; (3) a counting embedder spy; (4) narrowed
absolute claims.

**Retraction:** the r2 card's claim that the write-side cleanup "closes
wraparound resurrection" was FALSE AS WRITTEN — the r2 ordering
(publish-then-best-effort-delete, errors swallowed) recreated the same
resurrection class through its own crash window. This card supersedes it.

## Protocol choice (contract item 1) — ordering route, and why

Chosen: durable retirement strictly BEFORE publication, following the
stronger existing protocol `VectorIndex::replace_with_empty` /
`install_replacement` (lib.rs:1847-1911). NOT chosen: the allowed
alternative of a main generation "provably incompatible with any observed
WAL." Justification: FSVI v1's compaction generation is a mod-255 wrapping
u8 and WAL activeness is `wal_gen == next_generation(main_gen)`; every u8
value collides with some possible sidecar history, so any generation choice
is a convention, not a structural incompatibility — exactly what the
contract excludes. The ordering route reuses the already-audited crate-root
protocol verbatim.

Per tier, `write_tier_with_durable_wal_retirement` (two_tier.rs):

1. writes + durably publishes the replacement at a sibling
   `temporary_output_path` staging name (never the canonical name);
2. `retire_wal_sidecar_durably`: `wal::remove_wal` errors PROPAGATE typed —
   never `let _ =` — then `sync_parent_directory`;
3. only then `VectorIndex::install_replacement` (validates the staged
   artifact, idempotently re-retires, renames atomically, syncs parent).

Crash algebra, pinned at every observable point: {old main + old WAL} →
{old main, no WAL} → {new main, no WAL}. The hazard state
{new main + old WAL} is unreachable through `finish()`: publication
strictly follows durable retirement (successful run: WAL absent; removal
failure: typed error, NOTHING published, old main byte-identical, sidecar
as found; dir-sync failure: typed error, nothing published, safe
intermediate {old main, no WAL}). A crash can strand a staging file; its
`.tmp.<pid>.<nanos>` name resolves through no open/discovery path.
`index/src/lib.rs` is byte-untouched (not editable in this train); only
existing in-crate APIs are called.

## Fault/crash-point tests (contract item 2) + red proofs

| Test | Mechanism | Parent @ 7095b1da |
| --- | --- | --- |
| (i) `finish_propagates_wal_removal_failure_without_publishing_replacement` (two_tier.rs) | REAL fs injection — a directory occupying the sidecar name makes unlink fail (EISDIR class); typed `SearchError::Io`, canonical bytes byte-identical, obstruction left as found, staging cleaned, old generation reopens | **BEHAVIORAL RED, EXIT_STATUS=101** (`parent-red-passA.txt`): r2 published the replacement before attempting removal — the transcript shows the canonical bytes flipped to the new generation ("new-doc" header) despite the late error. This is the in-process projection of the crash window: the same publish-before-retire ordering that leaves {new main + old WAL} on a crash between rename and removal. |
| (ii) `crash_after_rename_state_is_pinned_and_unconstructable_through_finish` (two_tier.rs) | constructs the exact r2 crash-window state (fresh gen-1 main renamed over canonical, old gen-2 WAL adjacent; collision proven via `read_wal` gen == `next_generation(1)`); pins the observer (sidecar present, rows ACTIVE, `retains_content`), pins the frozen v1 reopen exactly (`wal_records` == ["foreign-doc"]; resurrection through the SEARCH surface, doc-id surface main-slab-only), then proves `finish()` over the WAL-bearing dir exits {new main, no WAL} | GREEN on parent by design: it pins observer/reopen behavior that already exists at r2 and pins unconstructability — the ordering half of unconstructability is the RED in (i)/(iii). Run in passA alongside (i). |
| (iii) `finish_propagates_wal_directory_sync_failure_without_publishing` (two_tier.rs) | documented one-shot test seam `builder_fault_injection` BETWEEN the unlink and the real `sync_parent_directory` (a genuine directory-fsync failure is not constructible on ordinary filesystems); typed Io error carrying the seam marker; post-state is the safe intermediate {old main intact, condemned WAL gone}; nothing published | **COMPILE-RED, EXIT_STATUS=101** (`parent-red-passB.txt`): `builder_fault_injection` does not exist at r2 — the dir-sync fault ordering cannot even be expressed there. |
| (iv) `legacy_crash_state_generation_collision_refuses_pre_drain_without_embedding` (refresh.rs) + the observer/reopen pins in (ii) | the observation-driven admission seam fails CLOSED on the crash state: typed `identityless-fsvi-v1` refusal before any drain/embedding, crash state byte-identical (main + WAL + dir manifest), zero embed invocations (direct spy), no foreign row in the served generation | **COMPILE-RED, EXIT_STATUS=101** (`parent-red-passC.txt`): `embed_invocations` and the 3-tuple spy-bearing `make_worker` do not exist at r2 — the direct zero-embed proof is inexpressible there. The refusal itself is green at r2 (it is r2's own containment refusal); what r3 adds is the exact-state pin + the direct spy. |

**Pinned residual (honest):** on a hand-constructed legacy crash state
(possible only from pre-r3 crashes or external manipulation — `finish()`
can no longer manufacture it), the frozen v1 `VectorIndex::open` STILL
replays the generation-colliding WAL rows: in the v1 format they are
byte-indistinguishable from legitimate incremental appends, and lib.rs is
not editable in this train. This is pinned exactly (test ii) rather than
hidden; the two-tier reopen path now emits a `warn_if_wal_rows_replayed`
warning (never silent), and the refresh admission seam refuses the state
typed (test iv). Full closure would need a v1 format change or a lib.rs
seam — out of this train's scope.

## Counting embedder spy (contract item 3)

`StubEmbedder` carries `embed_calls`/`embed_batch_calls` counters,
incremented synchronously at invocation (an un-awaited call still counts);
the trait's `embed_bound`/`embed_batch_bound` defaults delegate into the
counted methods, so `embed_invocations() == 0` is a direct proof that no
embed-class method started inference. Every refusal-path test asserts it
across the refusing call via `assert_no_embed_invocations_since`
(identityless v1 canonical+staging in all its variants, composite
authority, cross-space, producer attestation, quality-without-embedder,
content-corrupt v2, v2 full-admission invariance, all-tombstoned, and the
new crash-state test). Deliberately NOT asserted zero:
`two_faced_embedder_refused_at_bound_record_seam` — that refusal verifies
records the embedder must first produce, so zero embedding is structurally
the wrong claim there.

## Claim precision (contract item 4) — the exact guarantee

The absolute wordings ("WITHOUT mutating anything", two_tier.rs:2188 at
7095b1da; "zero side effects", refresh.rs:3942-3945 at 7095b1da) are
narrowed, here and in the code docs, to:

- **Guaranteed:** no writes, no truncation, no deletion, no writable opens
  or mappings; bytes, directory entries, sizes, and mtimes invariant across
  observation/classification and across every refused cycle.
- **Not guaranteed (stated, best-effort):** access metadata. The preferred
  opener is the `O_NOATIME | O_NOFOLLOW | O_CLOEXEC` fast path (the same
  opener exact v2 admission uses); where that open is denied or unsupported
  the documented fallback is an ordinary read-only `File::open`, and the v1
  WAL sidecar read is an ordinary read — either may update atime and follow
  symlinks. This is why `dir_manifest` never asserts atime.

## Verification (r3 tip)

- `cargo test -p frankensearch-index -p frankensearch-fusion`: 15 suites,
  0 failed (index lib 542 incl. the 3 new fault tests; fusion lib 938
  incl. the new crash-state test; all integration + doctests green),
  EXIT_STATUS=0 (`verify-r3-tests.txt`). frankensearch-core untouched.
- `clippy --no-deps` both crates `--all-targets`: warning set BYTE-CLASS
  IDENTICAL to the r2 banked baseline (pre-existing deprecated
  `wide::mul_widen` ×6 and the pre-existing dead-code constant in
  searcher.rs); zero warnings on changed lines; EXIT_STATUS=0. fmt clean
  (`verify-r3-lints.txt`).
- UBS on the two changed files: criticals 4 → 4, the identical
  pre-existing panic block (line-shifted); finding classes identical
  parent vs head (`ubs-changed-files.txt`).

Per-lane wrapper (`c4r3-cargo.sh` / `c4r3-parent-cargo.sh`): RCH_DISABLE=1,
the one canonical `CARGO_TARGET_DIR=/data/projects/frankensearch/target`,
scratchpad TMPDIR, foreground waits only. Reservations held on two_tier.rs
+ refresh.rs for the duration.

## Tracker disposition

`bd-5fsy6` and `bd-xomn.3` stay OPEN (unchanged from the audit's ruling):
C4 remains staged/noncanonical, composite generation authority and the
facade/fsfs/storage/cache/search migrations remain absent. r3 closes ONLY
the WAL-durability blocker and the two instrumentation/claim items.
