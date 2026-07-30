# E8-H P6 — TermInterner bucket-map identity-hasher swap, LOCAL QG-2 ingest (measured 2026-07-29, landed 2026-07-30)

**Bead:** `bd-e8h-w2-u64-hasher-swap-vcfft`. **Status: LANDED** through the
serialized post-UNION integration train rooted at
`a61456eca99c2a19394e37fe5d826094989982d6` (source commit `bd16d35d` in that
train). The lever was prepared and revalidated on the exact `928a16ba`
landing base, measured under a publication freeze as
`scratchpad/p6/p6-hasher-swap.patch` (81-line unified diff against the
continuity base's `crates/frankensearch-quill/src/scribe.rs`), and published
here together with this card.

## Verdict

**KEEP.** QG-2 smoke memory-child ingest, 200k docs, single thread:
paired docs/s ratio lever/base **1.0334 mean / 1.0340 median**
(95% t-CI **[1.0296, 1.0372]**, n=32 pairs, 32/32 pairs favor the lever).
In the ledger's time-ratio convention (new/old, <1 = speedup):
**0.9677 point / CI [0.9641, 0.9713]** — the point estimate and median clear
the [0.97, 1.03] wash band; the CI's upper bound overlaps the band edge by
~0.001, but the A/A null CI ([0.9962, 1.0057]) and the A/B CI are disjoint,
so the effect is unambiguous. Emitted segment bytes are proven identical.

- **Comparison class: SELF-SPEEDUP** (maintenance evidence; both arms are
  frankensearch; not campaign output, no incumbent arm anywhere in this card).
- **Machine class: local-5975wx-32c** (diagnostic-only class per the P1 card;
  not ratchet-admissible).

## The lever

`TermInterner.buckets: HashMap<u64, Bucket>` (scribe.rs, `use
std::collections::HashMap` = SipHash13) is keyed by an ALREADY-FINALIZED
64-bit hash: the only producers of the key are `find`/`intern_accounted`,
which call `hash_parts` (generic `S: BuildHasher`, default
`ahash::RandomState`, over the composite `(field_ord, term)` key) immediately
before every `get`/`entry`. Re-hashing that u64 through SipHash on every
token of every document was pure waste — the P1 profile card attributed
`hash_one::<&u64>` 2.77% + sip `Hasher::write` 1.49% ≈ 4.3% of the quill arm
to this family (this pass's own baseline sample: 3.11% + 1.40%).

**Design chosen: (a) local identity hasher** (`PrehashedKeyHasher`, a
14-line private `std::hash::Hasher` whose `write_u64` stores the key), NOT
(b) `AHashMap` à la delta.rs:

- The key is **provably always a finalized hash** — `buckets` has exactly
  seven use sites (decl, ctor, `get`, `entry`, `values()` sum, `capacity`,
  `clear`); both key-producing sites compute the key via `hash_parts` on the
  line above. The invariant is stated in a doc comment on the type and on
  the field, with an explicit "do NOT reuse for non-hash keys" warning.
- ahash would still re-hash the u64 (cheaper than sip, but nonzero);
  identity is free. Zero new dependency either way (quill already deps
  ahash for `hash_parts`).
- Degenerate-hasher tests stay meaningful: under the test-injected
  `ConstHasher`, every term collapses into ONE `Bucket::Many` map entry, so
  the collision-verification path is exercised regardless of the map hasher.

**Durable-hash rule distinction (workspace Cargo.toml comment):**
"process-seeded ahash must never substitute durable xxh3 hashing" governs
**on-disk/durable hashes**. `buckets` is a transient in-memory map hasher —
the same category as delta.rs's `ahash::AHashMap` precedent — and this pass
changes only the MAP's internal slot hasher, not `hash_parts` (still ahash)
and not any durable xxh3 (content hashes, segment ids, FSLX witnesses are
untouched; byte-identity below proves it).

## Banked-evidence provenance (revision-bound clause)

The A/B measurement ran 2026-07-29 on ELFs built from the continuity base
**`3684b1477`** (P1/P2 lineage), under the publication freeze. It is applied
to the 928a16ba landing base under the revision-bound clause of the landing
protocol: `crates/frankensearch-quill/src/scribe.rs` is **blob-identical**
between the two revisions — git blob `0d66bba20f4fd988c4b0f4a13cc757b274c63af2`
verified at `928a16baed6d997fb5f63827387eb51cb3f4f4fa` immediately before
landing — so the measured lever site is byte-for-byte the code the patch
landed on. The byte-identity probe was additionally re-run fresh at the
landing base (ladder below) and reproduced the banked artifact hashes
exactly, tying the two bases together end-to-end.

Measurement provenance (unchanged from the banked draft):

- Source: `3684b1477` exported via `git archive` into
  `scratchpad/p6/overlay/frankensearch`; `../fast_cmaes` satisfied by
  symlink; Cargo.lock copied from the pass-2 overlay for identical
  dependency resolution.
- Deviation from the pass brief, deliberate: zero working-tree files were
  overlaid. Inspection showed the pass-2 base ELF was built from pristine
  `3684b147` (+ probe example only), and that day's working tree (local HEAD
  `57e3d85b` lineage) was *behind* origin/main at the lever site (missing
  `running_collision_bytes_reserved`); overlaying it would have destroyed
  Arm-A/Arm-B source parity. Overlay = pristine base + the scribe.rs lever +
  overlay-only probe example.
- Build: `cargo bench -p frankensearch-quill-gauntlet --features
  perf-harness --profile release-perf --bench perf_matrix --no-run`,
  `RUSTFLAGS="-C force-frame-pointers=yes"`, `RCH_DISABLE=1` wrapper
  (`p6/cargo-p6.sh`), isolated `CARGO_TARGET_DIR=/data/tmp/cargo-target-p6-hasher`.
  Toolchain identical to P1/P2: rustc 1.99.0-nightly (9f36de775 2026-07-19).
- **Arm A ELF** (base, pass-2 stash, SHA re-verified before use):
  `perf_matrix_base_iso` =
  `9c3cacf0fa0ab66b46b9fb9482c1b8e858985a02b4e7775ef47dec574f22078b`
- **Arm B ELF** (lever): `p6/elfs/perf_matrix_p6_lever` =
  `454205fd818de191ceca5d69700c51f5c6b441e1e4c39e574a65ed38e66e1bd9`
- Environment note: an unrelated franken_networkx rch perf job (pinned CPU 9,
  `-j8` build phase) ran on this host earlier in that session; benches ran
  with load ≈3.2/64 threads, no competing pinned process on core 8, and the
  A/A null bounds whatever bleed remained.

## Verification ladder — original (3684b1477 overlay, 2026-07-29)

1. **Tests** — `cargo test -p frankensearch-quill --lib`, pristine overlay:
   **474 passed / 1 failed / 1 ignored**; the single failure is the known
   flake `keeper::tests::labruntime_serializes_concurrent_publishers_across_a_late_symlink_alias`
   (assert at keeper.rs:14933), re-verified failing on PRISTINE baseline in
   that exact environment before exempting. With the lever: **identical
   474 / 1 (same flake) / 1**.
2. **Byte identity** — overlay-only probe
   (`examples/p6_ingest_probe.rs`, banked in `p6/`): 3 flush cycles × 3,000
   docs, seeded-xorshift Zipf-ish corpus (~20k distinct terms/cycle, both
   map-hit and map-miss paths), full tokenizer → `TermInterner` →
   `ColumnarAccumulator` → `flush_accumulator` → FSLX segment bytes,
   `TermInterner::reset` exercised between cycles. All four artifacts
   SHA-256-identical across arms AND across repeated runs of each arm:
   `p6_segment_0.fslx d3b2382d…`, `p6_segment_1.fslx 1a42f42e…`,
   `p6_segment_2.fslx dc861ef5…`, `p6_summary.txt dd4257a8…`
   (full hashes in `p6/probe-base-shas.txt` = `p6/probe-lever-shas.txt`).
   Structural reason: TERMDICT order is `sorted_ids()` (a byte sort over
   composite keys); map iteration order never reaches emission (`values()`
   is only summed).
3. **RED-ability** — the ordering seam is pinned by
   `scribe::tests::sorted_ids_match_composite_byte_order_and_field_grouping`;
   with a deliberately reversed `sorted_ids` sort (throwaway build) that
   test goes RED **and** the probe fails hard with
   `TermDictionary(NonAscendingInput { index: 1 })` — the TERMDICT encoder
   structurally rejects unsorted emission, so an ordering leak cannot even
   produce bytes silently. Overlay verified byte-equal to the banked patch
   after revert.
4. **A/B** — interleaved paired, external wall, QG-2 smoke memory child
   (`QUILL_PERF_CHILD_MODE=memory ENGINE=quill COUNT=200000 HEAP=50000000
   THREADS=1 POSITIONS=true SCALE=smoke`), taskset core 8, one untimed
   warmup per arm per batch (`scratchpad/ab_run.sh`):
   - **A/A null** (Arm A vs itself, n=16): median 1.0012, mean 1.0010,
     95% t-CI [0.9962, 1.0057], min 0.9802 / max 1.0184; per-arm
     throughput spread ≈0.5–0.6%.
   - **A/B** (n=32 pairs, two independent 16-pair batches, medians 1.0345
     and 1.0335): mean **1.0334**, median **1.0340**, 95% t-CI
     **[1.0296, 1.0372]**, min 1.0076, max 1.0506, **32/32 pairs > 1**.
     Arm A ≈ 38.8k docs/s → Arm B ≈ 40.1k docs/s.
5. **Mechanism** — arm-scoped perf (P1 method: dwarf, F=1997, cycles:P,
   ~10k samples/arm), self-time:

   | frame | base | lever |
   |---|---|---|
   | `RandomState::hash_one::<&u64>` | 3.11% | **absent** (<0.1% floor) |
   | sip `Hasher::write` (u64-key instantiation) | 1.40% | **absent** (remaining sip rows 0.33%+0.31% are String-keyed maps elsewhere, out of scope) |
   | `TermInterner::find_in_bucket` | 2.08% | 1.81% |
   | `TermInterner::matches` | 1.47% | 0.78% |
   | `TermInterner::intern_accounted` | 0.86% | 1.14% (map-op cost now inlined here) |
   | `TermInterner::hash_parts` (must stay) | 0.93% | 0.99% |
   | interner+hash family total | 10.87% | 5.84% |

   The two targeted frames vanish; the ~5-point family shrink is consistent
   with the measured ~3.3% wall gain after the small inlined-probe offset.

## Verification ladder — fresh at the landing base (928a16ba, 2026-07-30)

Run in a clean detached worktree at exact
`928a16baed6d997fb5f63827387eb51cb3f4f4fa` (`RCH_DISABLE=1` wrapper,
isolated `CARGO_TARGET_DIR=/data/tmp/cargo-target-e8h-landing`, scratchpad
TMPDIR, taskset cores 16–31, 14 build jobs; strict-pipefail transcripts with
exit trailers banked in `scratchpad/landing-evidence/`):

1. **Precondition** — `git ls-tree` and `git hash-object` both report
   scribe.rs blob `0d66bba20f4fd988c4b0f4a13cc757b274c63af2` at the landing
   base; the banked patch applies clean (`git apply --check`, then applied:
   +48/−3, scribe.rs only).
2. **Tests** — `cargo test -p frankensearch-quill` (full crate, not just
   --lib): **pristine 928a16ba = 482 passed / 0 failed / 1 ignored** (lib)
   plus green `cancellation_contract` integration suites, exit 0; **patched
   = identical 482 / 0 / 1**, exit 0. Test count differs from the 3684-era
   ladder (474) because the train landed new work (TERMDICT-cache et al.);
   the `labruntime_serializes_concurrent_publishers_across_a_late_symlink_alias`
   flake did NOT reproduce at the landing base — no exemption needed.
3. **Byte identity, re-proven at the landing base** — the same probe source
   rebuilt from THIS worktree on both arms (`release-perf`): pristine ELF
   `aa314dee3898bcac0d6fd6af235f786248024a7798dbd95f3026d3eeb3bb744c8`,
   patched ELF
   `e3e86843421cf7ea624222884272e45f9d4b8e31da0794845bb88b20f6aeb0f3`.
   All four artifacts byte-identical across arms and across two runs per
   arm, AND identical to the banked 3684-era hashes
   (`d3b2382d…`, `1a42f42e…`, `dc861ef5…`, `dd4257a8…`) — the train's
   TERMDICT-cache work did not change emitted segment bytes for this corpus,
   which ties the banked A/B to the landing base end-to-end.
4. **fmt** — `cargo fmt -p frankensearch-quill --check` clean (exit 0).
5. **clippy** — `cargo clippy --no-deps -p frankensearch-quill`: pristine
   warning set EMPTY; first patched run surfaced exactly one new pedantic
   warning (`clippy::doc_markdown`, "SipHash" missing backticks in the new
   `PrehashedKeyHasher` doc comment). Fixed by backticking the word — the
   ONLY deviation from the banked patch, doc-text-only, zero codegen effect
   (probe artifacts re-verified identical after the fix). Final warning-set
   diff vs pristine: EMPTY (exit 0).
6. **UBS** — `ubs crates/frankensearch-quill/src/scribe.rs`, both arms:
   honest exit **1** on BOTH pristine and patched (scribe.rs carries
   inherited scanner noise, same situation as the s1rc1 files), totals
   **47 critical / 1408 warning / 163 info on BOTH arms**, and the 22-row
   per-class census (unwrap/expect, panic-family, timing-safe-equality,
   indexing-panic-surface, etc.) is byte-identical between arms —
   **delta = zero new findings in every class**.

## Adjacent sites noted, NOT levered (one family per pass)

- scribe.rs contains **no other non-test std HashMap/HashSet**; the
  `HashSet` near :8280 is inside `mod tests`
  (`lease_disjointness_under_concurrent_sessions_property`) — test-only.
- Residual SipHash after the lever: `hash_one::<&String>` 0.22% + two sip
  `write` rows ≈0.6% — String-keyed maps outside scribe.rs's interner;
  separate family, separate pass if ever worth it (small).

## Repro

```bash
# overlay build (wrapper sets RCH_DISABLE=1, isolated target, frame pointers)
scratchpad/p6/cargo-p6.sh test -p frankensearch-quill --lib
scratchpad/p6/cargo-p6.sh bench -p frankensearch-quill-gauntlet \
  --features perf-harness --profile release-perf --bench perf_matrix --no-run
# byte identity
PROBE_OUT=<dir> taskset -c 8 scratchpad/p6/elfs/p6_ingest_probe_{base,lever}
# A/B (exact pass-2/3 method)
scratchpad/ab_run.sh scratchpad/elfs/perf_matrix_base_iso \
  scratchpad/p6/elfs/perf_matrix_p6_lever out.tsv 16 8
python3 scratchpad/ab_stats.py out.tsv
# mechanism
scratchpad/perf_sample.sh <elf> out.perf.data
```

Artifacts: `scratchpad/p6/` — `p6-hasher-swap.patch`, `elfs/`, `aa_null.tsv`,
`ab_main.tsv`, `ab_main2.tsv`, `probe-{base,lever}-shas.txt`, `elf-shas.txt`,
`perf/p6-{base,lever}.perf.data`, `ci_stats.py`. Landing-base ladder
transcripts: `scratchpad/landing-evidence/` (build/test/clippy/fmt/UBS logs
with EXIT_STATUS trailers, probe SHA files, UBS class censuses).
