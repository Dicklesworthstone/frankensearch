# E8-H P12 — canonical-encode direct preimage encoder (bd-e8h-w2-canonical-encode-si8mk takeover), local-5975wx-32c (2026-07-30)

**Publication provenance:** integrated after `a61456ec` in evidence-set
commit `5b91f680`; measurements are diagnostic/NoClaim per card scope.

**Task:** implement and adjudicate the rank-1 QG-2 lever from the P9
generator-corrected attribution
(`docs/evidence/e8h/p9-local-qg2-generator-corrected-attribution.md`):
replace the serde_json-built canonical document preimage
(`canonical_document_preimage` + `canonical_metadata`,
`crates/frankensearch-quill/src/index.rs`) with a direct encoder producing
BYTE-IDENTICAL preimage bytes, under the registry-1.0.2 IDMAP `content_hash`
contract (xxh3 over exactly those bytes). Sanctioned takeover of the silent
W2.5/W2 lanes (IndigoOsprey/RusticDeer salvage, ScarletPelican retainer).
The measurement freeze constraints held: the pass made zero shared-tree
edits, commits, or tracker calls; it was overlay-only on pristine
`git archive 3684b147`, with raw session artifacts in `scratchpad/p12/`.

## W2.5 negative-result disposition (the brief said it was LOST — it is NOT)

The W2.5 lane's "naive direct-encoder falsified" result WAS ledgered, but only
on a non-main lineage when this pass was adjudicated: commit `217e73c1`
("perf(quill): bank contaminated W2.5 encoder diagnostic", YellowSparrow,
2026-07-29, then reachable from `codex/*` branches but not `origin/main`)
appends full NEGATIVE_EVIDENCE.md + PERF_LEDGER.md rows. Contents that bind
this pass:

- Falsified form: a **scalar per-byte hand-written canonical JSON emitter**
  replacing `serde_json::to_vec` (their overlay diff sha
  `77f95402…`, base commit `37c42ed4` whose index.rs canonical fns are
  IDENTICAL to `3684b147`'s — the oracle set transfers).
- Correctness PASSED 15/15 (hostile unicode/control bytes, every u8
  digit-width boundary, metadata insertion-order permutations, canonical byte
  + content-hash equality, whole-FSLX-segment identity).
- Timing: apparent **1.133x REGRESSION** (direct 6,791 ns/doc vs serde
  6,085 ns/doc) ruled INVALID-EXTERNAL-INTERFERENCE (a four-slot job
  overlapped ~203 s of the measured window on the same worker); decision
  CONSERVATIVE NO-SHIP.
- Retry predicate: never resample the same scalar per-byte emitter; a retry
  must use a materially different, profile-named mechanism — **block escape
  scanning** or a **single-pass serializer/hash/IDMAP sink** — and repeat the
  correctness oracles first.

**This pass's design is routed through that predicate**, and treats the
contaminated 1.133x as directionally informative anyway: serde_json's own
`format_escaped_str_contents` already does table-driven block escape scanning
with bulk clean-run copies, so a per-byte push loop plausibly LOSES to it —
the mechanism, not the contamination, is the likely source of their adverse
diagnostic.

## The byte contract (read from serde_json 1.0.151, the locked version)

Preimage = compact JSON array `["<id>","<content>","<title-or-empty>",[<metadata bytes as decimal u8 integers>]]`
where metadata = compact JSON object over byte-lexicographic key order
(`BTreeMap` iteration). Facts that bound the design:

- **No floats exist anywhere in the preimage** (`IndexableDocument` is
  {String, String, Option<String>, HashMap<String,String>}); the only
  numbers are the u8 integers of the double-encoded metadata bytes. The
  ryu/grisu float-formatting trap the brief warned about is structurally
  absent — proven, not assumed.
- Escapes (serde_json `ESCAPE` table, ser.rs): bytes `0x00..=0x1F`
  (short `\b \t \n \f \r` for 08/09/0A/0C/0D, else `\u00xx` LOWERCASE hex —
  including 0x0B), plus `"` and `\`. `0x7F` and all non-ASCII UTF-8 pass
  through raw. Integers via itoa (plain decimal, no padding).
- The preimage bytes are RETAINED per doc (`PendingIdentity.canonical_content`)
  and consumed at flush by (a) per-doc IDMAP xxh3
  (`IdMapEntryInput::from_canonical_content`) and (b) the segment-id batch
  digest, which streams `(len_le, bytes)` per identity
  (`derive_segment_id`, index.rs). The metadata bytes are ALSO persisted as
  the METADATA_FIELD stored value. So the bytes must exist; a
  hash-only streaming path would have to re-derive the segment-id batch
  digest incrementally across flush/rollback seams for a hash family that is
  only ~2.8% of the child and explicitly out of lever scope.

## Design (chosen: direct encoder with block escape scanning + exact-size single-pass sink)

`canonical_metadata` and `canonical_document_preimage` keep their exact
signatures (zero call-site churn; `serde_json::Error` retained, now
unreachable). New shape, per document:

1. **Exact-size pre-pass**: `json_escaped_len` (per-byte extra-length table,
   identical coverage to serde_json's ESCAPE) + u8 decimal widths → ONE
   `Vec::with_capacity(exact)` per buffer, ZERO growth reallocation, and
   retained capacity ≤ the serde path's doubling-grown capacity (RSS
   never worse; the buffers live until flush).
2. **Block escape scanning emit**: clean runs appended with one bulk
   `extend_from_slice` per run; escapes take the cold arm
   (`JSON_ESCAPE_EXTRA` / `JSON_ESCAPE_SHORT` 256-entry tables, `\u00xx`
   lowercase hex). This is the named W2.5 retry mechanism, NOT their
   falsified per-byte emitter.
3. **Metadata object**: `Vec<(&str,&str)>` + `sort_unstable_by` byte order
   replaces the per-entry-allocating `BTreeMap` (same order by construction:
   String's `Ord` is byte-lexicographic; duplicate keys impossible from
   HashMap). Net allocation change per doc: −(n BTree nodes) +1 Vec.
4. **u8 decimal emission**: branch ladder on value bands (identical bytes to
   itoa), no per-element serializer dispatch.
5. `debug_assert_eq!(out.len(), pre_pass_len)` pins the two passes together
   in every debug/test build.

**Rejected alternatives** (documented per brief):
- *Streaming xxh3 without materialization*: bytes are structurally required
  (segment-id batch digest + retained identities; see contract section);
  xxh3 is ~2.8% child and untouchable; restructuring PendingIdentity for a
  non-addressable family is risk without basis.
- *Single-pass emit with worst-case reserve* (skip the pre-pass): worst-case
  6x for strings / 4x for the u8 array would be RETAINED until flush across
  200k docs — an RSS regression risk the exact-size pre-pass eliminates for
  one extra linear scan.
- *Scalar per-byte emitter*: falsified by W2.5 (see disposition); not
  resampled.

## Attribution correction discovered en route (binding on successors)

P9's `canonical_core` = 16.23% ENG includes `stable_digit_scatter` at 2.41%
of child (546M cycles) via a NAME-match in the classifier
(`sre[canonical_core] = …|stable_digit_scatter|…`). That function is the
**scribe.rs radix-sort digit scatter of the SEAL path** (term-row ordering),
not JSON digit emission — no canonical-encode lever can touch it. Corrected
addressable family: **canonical core ≈ 12.9% ENG ≈ 9.4% child** (+2.8%
untouchable xxh3@ingest). The realistic recovery estimate shrinks from the
brief's 5-8% of child to ≈ **2.5-5%** — preregistered before the A/B:
expected paired ratio 1.02-1.05 against the hard ≥1.03 bar, i.e. this pass
was at risk of an honest sub-bar landing from the start.

## Ladder

| step | status | evidence |
|---|---|---|
| overlay `cargo test -p frankensearch-quill` | 477/478 pass; the 1 failure is `keeper::tests::labruntime_serializes_concurrent_publishers_across_a_late_symlink_alias` (`saw_two_waiters`) | flake is PRE-EXISTING: identical failure recorded in the previous session's `scratchpad/test-lever-full.log` (different lever, same 3684b147 base), fails under full-suite load on this host (3/3, also at `--test-threads=4`), passes 5/5 isolated; keeper.rs is disjoint from this lever (index.rs only). Pristine re-verify: CONFIRMED — the identical test fails with the identical `saw_two_waiters` assertion on untouched 3684b147 (474/476 vs my 477/479; the +3 are exactly this pass's property tests) |
| property tests (encoder vs serde reference oracle) | GREEN | `canonical_encoder_matches_serde_reference_on_curated_adversarial_documents` (controls singly + torture strings + unicode/combining/BOM/RTL + 10KB boundary-planted strings), `…_on_seeded_fuzz_documents` (2,000 seeded-PRNG docs, adversarial char pool), `canonical_preimage_u8_metadata_array_matches_serde_for_every_byte_value` (all 256 values + width boundaries + 600-len run); each case also asserts `indexable_document_content_hash` == xxh3(reference bytes) |
| RED-ability | PROVEN | mutation `table[0x0B] = b'v'` (plausible wrong short escape): both string-path property tests FAIL, u8-array test correctly unaffected; reverted, re-GREEN |
| corpus-wide hash equality | **PASSED** | canon_probe (overlay-only example, feature-symmetric builds): 200k pinned QG-2 corpus docs (seed 0x5155_494c_4c50_4552, Zipf S11, vocab 8192, max 4096B; content 174,699,940 bytes total), per-doc IDMAP content_hash files byte-identical (`cmp` clean) + order-sensitive fold equal = `0c0143d44daf6595` both ELFs |
| A/B + A/A | **DONE — sub-bar** | see A/B result below |
| mechanism | **DONE** | see Mechanism below |

## Provenance

- Base arm A ELF: `scratchpad/elfs/perf_matrix_base_iso`, sha256 re-verified
  this pass = `9c3cacf0fa0ab66b46b9fb9482c1b8e858985a02b4e7775ef47dec574f22078b`
  (pristine 3684b147, P1/P2/P6/P8/P9 lineage).
- Lever ELF: `scratchpad/p12/perf_matrix_lever_p12`, sha256 =
  `aeeac40f52037903666aadcbcb6e30e04e5d45e9b2f4991d06f9857e58cf53bb`
  (overlay = 3684b147 + index.rs only; patch
  `scratchpad/p12/p12-canonical-encode.patch`, git-diff --full-index against
  blob `908e5146…`, +386/−8 lines, two hunks: encoder + tests).
- Toolchain: rustc 1.99.0-nightly (9f36de775 2026-07-19), pinned by
  rust-toolchain.toml `nightly-2026-07-20` — same as the base ELF.
- Cargo.lock: fresh resolve is byte-identical to the base ELF's
  (`scratchpad/src-main/Cargo.lock`); serde_json locked at **1.0.151** (the
  property-test oracle compiles against exactly the replaced code).
- Build: release-perf (thin LTO, line-tables), `-C force-frame-pointers=yes`,
  `--features perf-harness`, isolated CARGO_TARGET_DIR, RCH-disabled wrapper.

## Run receipts (asymmetry rule: CPU/wall per arm for every timed context)

Host load during the timed window: 1-min ~4 (far quieter than P9's ~11-44).
Governor powersave on core 8 — relative paired ratios only. All runs
`taskset -c 8`, `/usr/bin/time` user+sys+maxrss per run, external wall.

| context | arm | median wall s | median CPU s | CPU/wall | median peak RSS MiB |
|---|---|---:|---:|---:|---:|
| A/A | base #1 | 5.256 | 5.240 | 0.996 | 510 |
| A/A | base #2 | 5.270 | 5.245 | 0.996 | 503 |
| A/B batch1 | base | 5.270 | 5.250 | 0.996 | 515 |
| A/B batch1 | lever | 5.205 | 5.180 | 0.996 | 478 |
| A/B batch2 | base | 5.339 | 5.320 | 0.996 | 536 |
| A/B batch2 | lever | 5.248 | 5.225 | 0.996 | 478 |

Arm relative spreads 0.7-1.7% (wall); every arm CPU/wall = 1.00 (single
thread, no parallelism asymmetry in this fixture's quill arm).

## A/B result

- **A/A null** (base ELF vs itself, 16 interleaved pairs): paired wall ratio
  median **0.9956** [p5 0.9890, p95 1.0101] — clean, straddles 1.0, ±1% band.
- **A/B batch1** (16 pairs): wall **1.0143** [1.0002, 1.0285]; CPU 1.0153.
- **A/B batch2** (16 pairs): wall **1.0221** [0.9973, 1.0600]; CPU 1.0222.
- **POOLED n=32: wall median 1.0175 [p5 0.9972, p95 1.0476]; CPU median
  1.0183 [0.9976, 1.0482]**; 7/32 pairs ≥1.03.
- Secondary (no QG claim): lever peak RSS median −37 MiB (515→478,
  **−7.2%**), consistent across all 32 pairs; user CPU 4.94→4.81 s mean, sys
  0.37→0.40 s (the win is user-side compute, not page-fault/kernel).

## Mechanism (same-day arm-scoped dwarf profiles; cross-day comparison is INVALID — see caveat)

Method: fresh traced runs of BOTH arms today (base `base-200k-dwarf-p12.perf.data`
total 20.76B cycles; lever `lever-200k-dwarf.perf.data` total 20.52B cycles —
traced totals ratio 1.012, coherent with the untraced A/B 1.0175), P9
classifier, 3 replicates per arm.

**Caveat that changes the P9 numbers:** comparing today's lever profile
against P9's YESTERDAY base profile (load ~11-44 vs today ~4, DVFS state
different) inflates encode-leaf shares and is not usable; and even same-day,
family attribution is unstable across ELFs/replicates (`stable_digit_scatter`
swung 69M ↔ 546M cycles between same-fixture runs; base canonical family
replicate spread 1.90-2.22B = 16%). Function-precise allocator leaves are the
attribution-stable signal.

| signal (same-day, rep1 vs rep1) | base | lever | delta |
|---|---:|---:|---:|
| `_int_malloc` (whole child) | 456M | 356M | **−100M** |
| `_int_realloc` | 114M | 21M | **−93M (−81%)** — growth-realloc elimination as designed |
| `_int_free` | 351M | 263M | **−88M** |
| `__memset` | 84M | 74M | −10M |
| allocator family total | ~1.005B | ~0.713B | **−292M ≈ −1.4% of child** |
| canonical_core family (3-rep range) | 1.90-2.22B | 2.76-2.81B | **+~0.7B — see reading** |

**Reading:** the lever's win is ALLOCATION-SHAPED, not dispatch-shaped. The
−292M allocator cycles ≈ the entire measured +1.75% wall win. The canonical
family's apparent growth is partly concentration (the base scatters serde
frames into poll-body/[unknown]/other families — base `doc_ingest_body`
1.57B vs lever 0.93B is the counter-shift — while the lever inlines the whole
emitter into `canonical_document_preimage` self), but even reading
generously, direct-emitter compute (exact-size pre-pass double scan +
push-based u8-decimal emission) is FLAT-TO-WORSE vs serde_json's tuned
single-pass serializer. serde dispatch was never the bottleneck; the
allocation pattern was. This retroactively corroborates the DIRECTION of
W2.5's contaminated 1.133x adverse diagnostic for pure emitter swaps, and
**falsifies the P9 rank-1 scoring** (Impact 6.0 assumed 5-8% child
recoverable from serializer indirection; the actually-recoverable slice was
the ~1.4% allocator traffic).

## Verdict

**REJECT under the hard point-estimate rule** (pooled paired wall median
1.0175 < 1.03; no CI arguments). Real, reproducible-in-two-batches ~+1.8%
CPU/wall improvement with −7% peak RSS and full byte-identity — but the bar
is the bar. Overlay edits reverted after profiling; the patch + all evidence
banked in `scratchpad/p12/`.

**Route-next (for the ledger row):** (a) an alloc-only hybrid — keep
serde_json's emitter, serialize into a REUSED scratch buffer, then one
exact-size copy into the retained Vec (targets the proven −1.4% allocator
slice without emitter-compute risk) — cannot clear 1.03 alone on this
evidence; bundle with another lever or drop; (b) any future canonical-encode
attempt must present a counted receipt that emitter compute (not allocation)
dominates — this pass measured the split and it does not; (c) re-rank the
P9 lever table with canonical-encode's Impact cut to ≤2 (the family is
floor-bound: xxh3 2.8% untouchable + emitter parity + allocator 1.4%
captured-and-rejected).

## Incident note (patch artifact)

The first `p12-canonical-encode.patch` was generated through
`git diff | tee <file> | head -5` — `head`'s exit SIGPIPE-killed `tee`
mid-write and silently truncated the patch at line 417 (caught by
`git apply --check`). The lever source was then reconstructed from the
session's exact edit blocks and verified byte-identical to the original by
blob hash (`git hash-object` = `a2ff0578…`, the b/ anchor recorded in the
truncated patch's header — i.e., provably the same bytes the lever ELF was
built from), and the patch regenerated with `git diff --output=`. Roundtrip
proof: pristine blob `908e5146…` + patch → `a2ff0578…`. Same family as the
s1rc1 piped-transcript trap: never let a `head`/filter sit downstream of a
`tee` that is producing an artifact.

## Repro

```bash
# overlay: git archive 3684b147 → scratchpad/p12/overlay, apply p12-canonical-encode.patch
# tests:   scratchpad/p12/cargo.sh cargo test -p frankensearch-quill --lib -- canonical
# probe:   PROBE_OUT=<file> PROBE_COUNT=200000 taskset -c 8 <elf-dir>/examples/canon_probe
# A/B:     scratchpad/p12/ab_run.sh <baseELF> <leverELF> <tsv> 16 8; python3 scratchpad/p12/ab_stats.py <tsv>
```
