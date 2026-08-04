# Evidence card — bd-s1rc1: EncodedSegment per-commit deep-clone elimination (Arc<Vec<u8>> backing) — REV 4 (REBASE pass, tip base)

REV 4 supersedes the REV 3 overlay card (session scratchpad,
`p5/p5-s1rc1-DRAFT-card.md`): the same GO'd lever, re-derived and fully
re-measured on the CURRENT protected tip instead of the 3684-era overlay.
The commit carrying this card is PREPARED IN A WORKTREE and NOT pushed;
the final landing awaits the user ruling gate
(`bd-s1rc1-ubs-user-ruling-gate-82rpt`).

- Bead: bd-s1rc1. Pass: s1rc1 REBASE, 2026-07-30.
- **Base: `504fa185c6a392f8e9e48a8a28e70f1a235a8361`** (origin/main at pass
  start; detached worktree, no shared-tree edits).
- Patch provenance: `p5-s1rc1-arc-backing.patch`, sha256
  `a4714ab54ead03c9dfbf385dedcfbe378d8a6fa93671a095c03b1e7f81b01c6e`,
  git-format with full-index blob metadata against base `3684b147`.
- Blessed design: FoggyPrairie #6267 + #6255 with MossyPine refinement —
  unchanged; the rebase introduced NO semantic redesign.

## Rebase mechanics (what moved under the patch)

File drift 3684b147 → 504fa185:

| file | base blob | tip blob | drift |
|---|---|---|---|
| segment.rs | db4393b3 | db4393b3 | IDENTICAL |
| keeper.rs | 8c1d74dd | 8bdb7cfd | TERMDICT cache + error-taxonomy train (8b864cb7, 5ee393fe, b61bbf89) |
| index.rs | 908e5146 | 8d9c31f6 | same train + bd-zljy publisher-lock test isolation |

`git apply --3way` (base blobs present in-clone): **all three files applied
cleanly, zero conflict markers, zero manual hunk surgery.**

- segment.rs: base identical ⇒ merged result is byte-exactly the patch's
  new-side blob `eb1479ade9d1fb805fde68647af37c5095904c65`.
- keeper.rs: the two production hunks landed at shifted offsets
  (`RecoveredSegmentBacking::Owned` 1885→1897,
  `bind_owned`→`from_encoded` 2132→2265); train lines untouched.
- index.rs: doc-comment hunk unshifted (1581); the test-block hunk shifted
  +50 lines (12712→12762) below the train's earlier test additions.
  395 insertions, 0 deletions.

## Patch-local UBS remediation (net-new in REV 4, closes the +2 critical delta)

The REV 3 patch's two net-new test `panic!` sites became typed
expect-style assertions (test-only lines, inside the patch's own
additions; no pre-existing panics touched):

1. `durable_delta_seal_retains_pending_on_manifest_write_failure_then_retries`
   fixture-backend guard: match now yields `Option<&mut KeeperWriter>` and
   `.expect("fixture must use a durable Keeper")`.
2. Same test's failure extraction: `let Err(..) else { panic! }` became
   `.err().expect("read-only MANIFEST target must fail publication")`
   (`QuillSearchSnapshot` has no `Debug` impl, so `.expect_err()` cannot
   compile; `.err().expect(...)` is the typed equivalent without a Debug
   bound on the Ok side).

`panic!(` occurrence counts per file are now IDENTICAL across arms
(index.rs 56, segment.rs 1, keeper.rs 13): the patch adds ZERO panic
sites — two fewer criticals than the REV 3 patch state.

## Acceptance ladder at the tip (all strict pipefail transcripts in `s1rc1-rebase/transcripts/`)

### 1. Gates — GREEN

- `cargo check -p frankensearch-quill --all-targets`: EXIT_STATUS=0.
- `cargo clippy --no-deps -p frankensearch-quill --all-targets`:
  EXIT_STATUS=0 on BOTH pristine-tip and lever trees; extracted warning
  sets are both EMPTY, diff exit 0 (`clippy-warning-diff.txt`).
- `cargo fmt -p frankensearch-quill -- --check`: EXIT_STATUS=0 both trees.
- `cargo test -p frankensearch-quill -- --test-threads=4` (scratchpad
  TMPDIR): EXIT_STATUS=0 — 490 lib tests (489 passed, 1 ignored),
  3 integration, 2 doctests, 0 failed. All five patch-borne behavior
  tests green by name:
  `memory_delta_seal_retains_pending_on_prepublication_cancellation_then_retries`,
  `durable_delta_seal_retains_pending_on_manifest_write_failure_then_retries`,
  `published_memory_snapshot_bytes_survive_successor_publication`,
  `encoded_segment_clone_shares_backing_and_unique_extraction_is_zero_copy`,
  `from_encoded_reader_retains_backing_and_matches_from_owned`.

### 2. Byte/hash identity — PASS (cross-ELF, 4 runs, full manifests, tip-built)

Probe `p5_seal_probe.rs` (source sha256
`ada03e97ea21b34995199028fce9714c7835a6c9381379f3eb034c18267d9cff`,
byte-identical to the REV 3 probe; overlay-only, does NOT land) rebuilt
`--release` from THIS worktree (lever) and the pristine-tip archive tree
(base):

- Probe ELF sha256 (base): `1a486879a1c61bfb1f7ddb4bf89bcc9c58bbc884ba0819aeecb50af92dcbb223`
- Probe ELF sha256 (lever): `83f08aa58991c6dd61991c789457397425c56182b8f46aca6dccbfc4e9d7df4a`

All four per-run `MANIFEST.sha256` files (10 artifacts each, xxh3
witnesses + IDMAP included) are byte-identical; combined manifest has 40
rows and canonical digest

    sha256(probe-manifests-combined.sha256) =
    8678387786cb0cbc1e8473bba34641393f7bd57016adc32d89d60f598211b98a

— IDENTICAL to the REV 3 overlay digest: the tip train changed no emitted
byte, and neither does the lever.

### 3. Mechanism — DHAT (valgrind-3.25.1, load-insensitive; `dhat_family.py` v2, selftest 4/4 PASS)

Exact banked env block; ELFs built from the tip with
`--profile release-perf --features perf-harness`,
RUSTFLAGS `-C force-frame-pointers=yes`:

- BEFORE ELF sha256 (pristine tip): `39c4ecabd787e3ca4f4fdcd62c5af3381bf1309362c8937324b1182fe7c74ab4`
- AFTER ELF sha256 (lever): `40de3a53a6886af16096f608556b04fc868e3cf730a297c1c27ca8b397187311`

50k docs (memory child, 1 thread, positions on):

| | banked BEFORE (3684-era) | fresh BEFORE (tip) | AFTER (tip lever) |
|---|---|---|---|
| total allocated | 1,146,905,996 | 1,171,626,962 | 1,085,915,139 (−7.32% vs fresh) |
| **v2 payload-site (exact claim)** | 88,280,448 (2 pps) | 87,884,436 (2 pps) | **0 (0 pps)** |
| v1 diagnostic family | 88,294,528 (6 pps) | 87,897,636 (6 pps) | 13,440 (3 pps) |
| child completion | OK | `quill-perf-child 264028160 0` | `quill-perf-child 257789952 0` |

200k docs (same command, COUNT=200000):

| | fresh BEFORE (tip) | AFTER (tip lever) |
|---|---|---|
| total allocated | 7,275,372,547 | 6,906,437,248 (−5.07%) |
| **v2 payload-site (exact claim)** | 353,951,118 (2 pps) | **0 (0 pps)** |
| payload-site bytes/doc | 1,769.8 | 0 |
| v1 diagnostic family | 354,009,638 (6 pps) | 60,976 (7 pps) |
| child completion | `quill-perf-child 1052307456 0` | `quill-perf-child 1028395008 0` (peak −2.3%, observed only) |

**The exact claim is payload-site disappearance: → 0 bytes / 0 pps at
BOTH scales, against BOTH the banked 3684-era baseline and the fresh
tip baseline.** Family-level v1 figures are diagnostic only. The fresh
tip BEFORE differs from the banked BEFORE by <0.5% in the payload family
(the train's index/keeper changes shifted the pp split, not the mass).

### 4. UBS three-file census (v5.3.7, both arms at tip) — exact numbers

| arm | critical | warning | info | raw exit |
|---|---|---|---|---|
| pristine tip | 310 | 4,879 | 981 | 1 (inherited, pre-existing findings) |
| lever | 310 | 5,055 | 988 | 1 (inherited) |

- **Critical delta vs pristine tip: 0 new** (REV 3 was +2; the panic
  remediation above removed exactly those two — the patch is now 2
  criticals BETTER than its REV 3 form).
- Warning +176 / info +7 are the `.expect()` density of ~390 added
  TEST-only lines, same finding classes both arms (REV 3 attribution
  unchanged).
- The raw exit 1 on BOTH arms is inherited from pre-existing findings in
  these three files and is not introduced by this patch.

### 5. Timing

None. Zero wall-clock measurements in this pass (mechanism lever; DHAT
only). Any timing claim requires the separate same-worker interleaved
paired A/B and its own ledger discipline.

## Landing state

- Commit prepared in the detached worktree at base `504fa185`; NOT pushed.
- Push awaits the user ruling gate `bd-s1rc1-ubs-user-ruling-gate-82rpt`.
- The probe example and wrapper scripts do not land; raw artifacts (DHAT
  out-files, probe outputs, transcripts) remain in the session scratchpad
  `s1rc1-rebase/`.
