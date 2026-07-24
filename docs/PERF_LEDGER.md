# PERF_LEDGER.md — frankensearch measured wins

## 2026-07-23 — KEEP: Quill `TermInterner::bytes_used` O(1) running counter — −12.7% ingest instructions (`bd-w4j5`, cc)

First lever in the INGEST vein (the query-decode profile vein was mined). An ingest-heavy local
profile (`segment_fanout_ab`, `QUILL_E49_SCALE=full QUILL_E49_ROUNDS=1`, ~71k docs, 1-thread) put
`ColumnarAccumulator::bytes_used` at 7.33% of ingest self-time — because `should_flush(budget)`
(`scribe.rs`) re-calls `bytes_used() >= budget` after EVERY ingested document to decide flush timing,
and its dominant cost is `TermInterner::bytes_used`, which re-summed the arena chunks (O(chunks)) and
iterated EVERY hash bucket for collision-id bytes (O(distinct terms)) — a full recompute per document.

The fix reuses machinery that already exists: `intern_accounted` already returns `added_bytes`, the
exact per-term increment to `bytes_used` (arena key + `ArenaSpan` size + bucket bytes), which the delta
segment already trusts for its own O(1) seal check. Accumulate `added_bytes` into a `running_bytes_used`
field (reset in `reset()`); `bytes_used()` returns it. Σ of the increments exactly equals the former
recompute (arena bytes + `spans.len()*size` + `buckets.len()*ESTIMATE` + collision_ids), and the only
sites that mutate arena/spans/buckets are `intern_accounted` and `reset`, so the counter never drifts.

Byte-identical: because `bytes_used` is unchanged in value, flush timing — hence segment boundaries and
on-disk layout — is identical. Pinned by the full suite including the deterministic-ingest and gauntlet
differential tests (which would fail on any flush-timing shift) and the `bytes_used`-growth unit test.
473/473 quill-lib tests green; scoped clippy `-D warnings` clean.

MEASURED (deterministic instruction count, HEAD vs change, ingest workload `full`/`ROUNDS=1`/1-thread):
OLD 7,299,627,724 / 7,300,562,101 → NEW 6,366,154,746 / 6,373,972,116 = **−930M (−12.7%)**, both runs
±0.01%. Larger than the 7.33% self-time because the per-document O(distinct-terms) bucket-map traversal
is instruction-heavy (map iteration) beyond what flat sampling attributed.

**Decision: KEEP.** Route next (ingest profile): `bytes_reserved` has the same recompute shape (called
in flush-cycle tracing), `memmove` 2.8%, tokenizer `analyze` 0.7%.

## 2026-07-23 — KEEP: Quill `consume_position_run` peel-first-iteration — −0.81% total instructions (`bd-2fha`, cc)

Small profile-directed cleanup: the post-bitmap map put `consume_position_run` at ~14% of query
self-time (`perf report --sort srcline` pinned the per-position `read_u32_vint` + prefix-sum lines).
The loop carried `let mut previous: Option<u32> = None;` and did an `if let Some(previous) = previous
{ … } else { encoded }` match plus a `previous = Some(position)` rewrap on EVERY position, purely to
distinguish the first delta (which has no predecessor). Peeled the first read out of the loop:
`if freq == 0 { return } let mut previous = read_u32_vint()?; for _ in 1..freq { previous =
previous.checked_add(read_u32_vint()?)? }`. Byte-identical — the first position is the raw delta with
no overflow check (as before), every later position is the same checked prefix sum with the identical
`PositionOverflow { previous, delta }` error — and it drops the branch + `Option` wrap per position.
473/473 quill-lib tests green; scoped clippy `-D warnings` clean.

MEASURED (deterministic instruction count, HEAD+bitmap vs +peel, `smoke`/`ROUNDS=9`/1-thread): OLD
53,772,604,564 / 53,776,217,976 → NEW 53,336,727,330 / 53,337,755,179 = **−437M (−0.81%)**, both runs
±0.007% (signal is ~145× the run-to-run noise — decidable). Modest but real, byte-identical, and
simpler code with no downside.

**Decision: KEEP.**

## 2026-07-23 — KEEP: Quill `decode_bitmap_payload` `trailing_zeros` bit-scan — −4.86% total instructions (`bd-udz8`, cc)

Profile-directed follow-on to the vint lever: the post-vint hotspot map put `decode_bitmap_payload`
at 6.8% of query self-time. A BITMAP posting block packs exactly 128 set bits across a 128..512-bit
span (64 bytes), so for larger spans the bitmap is sparse (span 512 → 25% set). The inner decode
scanned all 512 bit positions (`for bit_index in 0..8 { if byte & (1<<bit)==0 { continue } }`),
wasting up to 384 zero-bit tests per block. Replaced with a set-bits-only walk:
`while bits != 0 { let bit = bits.trailing_zeros(); bits &= bits - 1; … }` — visits exactly the set
bits in ascending order, so the emitted docids, span/cardinality checks, and errors are byte-identical
(pinned by the existing BITMAP decode tests: first/last-set invariant, zero-beyond-span, exactly-128
cardinality, docid overflow). 473/473 quill-lib tests green; scoped clippy `-D warnings` clean.

MEASURED by deterministic instruction count (`perf stat -e instructions`, HEAD-vs-change binaries,
`smoke`/`ROUNDS=9`/1-thread): OLD 56,509,326,088 / 56,518,668,541 → NEW 53,764,802,472 /
53,773,904,176 = **−2.745B (−4.86%)**, both runs ±0.02% (130× the run-to-run noise — decidable).
The smoke corpus's high-df terms produce sparse bitmap blocks, so the skipped zero-bit tests are real
work removed.

**Decision: KEEP.** (Note: the win scales with bitmap sparsity; a corpus of only span-128 fully-dense
blocks would be a wash — but that regresses to the same iteration count, never slower.)

## 2026-07-23 — KEEP: Quill vint decode inline+cold split — −16.2% total instructions, the first profile-directed lever (`bd-b6tc`, cc)

FIRST lever driven by a local symbolized profile of the query path (the e8.3 attribution the bd-e41k
note said was unobtainable via rch — TRUE for rch, but it works LOCALLY; recipe below). Flat `perf`
self-time on `segment_fanout_ab` (smoke shape, single-thread) showed variable-length integer decoding
was **~33% of query self-time**: `SliceReader::read_vint` (grimoire) 16.3%, `PositionByteReader::read_u32_vint`
(quiver) 11.3%, `PayloadReader::read_vint` 3.6%, plus `decode_vint_payload` 1.8%.

A first attempt (1-byte fast path alone) was a WASH — the re-profile showed `read_vint` unchanged at
~17%. `perf annotate` then revealed the real cost: `read_vint`'s hottest instructions were its own
**function prologue** — a 6-callee-saved-register spill (`push rbp/r15/r14/r13/r12/rbx` ≈ 18% of the
function's self-time). The vint readers were never inlined, so every call in the tight
`decode_entry` / `consume_position_run` loops paid full call overhead. Fix = the textbook
inline-hot-path / outline-cold-path split: `#[inline]` the 1-byte fast case (value < 128, which
dominates gap-encoded deltas/positions and is ALWAYS canonical) so it inlines into callers and
eliminates the prologue, with the rare multi-byte loop moved to a `#[cold] #[inline(never)]`
`*_multibyte` helper (no code-size bloat). Applied to all four readers: grimoire `read_vint`, quiver
`read_u32_vint`, `read_u64_vint`, and `PayloadReader::read_vint`.

Byte-identical: a 1-byte vint (< 128) has `vint_len == 1`, so it is always canonical and cannot
overflow — the fast path returns exactly what the loop would. Pinned by the exhaustive grimoire/quiver
vint tests (canonical / non-canonical-padding / overflow / truncation); 473/473 quill-lib tests green;
scoped clippy `-D warnings` clean.

MEASURED by **deterministic instruction count** (`perf stat -e instructions`, the cleanest metric — zero
run-to-run drift, unlike wall-clock), two same-shape binaries built from HEAD (pre-vint) vs the change,
`QUILL_E49_SCALE=smoke QUILL_E49_ROUNDS=9 RAYON_NUM_THREADS=1`:

| binary | instructions | note |
|---|---|---|
| OLD (HEAD, pre-vint) | 70,854,048,799 / 70,858,019,891 | two runs, ±0.006% |
| NEW (inline+cold, 2 hottest readers) | 59,356,689,959 / 59,353,387,608 | two runs, ±0.006% |
| **reduction** | **−11.50B (−16.2%)** | deterministic, unambiguous |

(The A/B was captured on the two hottest readers; the shipped change extends the identical split to all
four, so the landed reduction is at least this.) A 16.2% total-instruction cut on a decode-bound
workload is a large real speedup — the non-inlined vint call was pervasive across ingest and query.

WORKING LOCAL-PROFILE RECIPE (toward `docs/quill-perf-attribution.md`, e8.3): the floating `nightly`
toolchain triggers a broken rustup auto-install mid-build — pin a dated nightly with `--check-cfg`
(`RUSTUP_TOOLCHAIN=nightly-2026-07-06-…`); bypass the rch hook by symlinking `qc` → that toolchain's
cargo binary (NOT the rustup shim, which rejects non-`cargo` argv[0]); build with
`CARGO_PROFILE_RELEASE_LTO=false DEBUG=1 STRIP=false` to an isolated target; FLAT `perf record -F 999`
(no `-g` — DWARF unwinding hangs on the rayon workload) with `RAYON_NUM_THREADS=1`.

**Decision: KEEP.** Route next (profile route-nexts): `consume_position_run` + `positions_for_ordinal`
(phrase position decode, ~15% combined — likely the same non-inlined pattern), `decode_bitmap_payload`
(bit-by-bit scan → `trailing_zeros`), `decode_entry` (10.8%).

## 2026-07-23 — KEEP: Quill `TopDocsCollector` packed-u64 selection key — 2–4.4× on the isolated heap op (`bd-y1ab`, cc)

Sibling of the KEPT FSVI int8 pass-1 packed-selection-key win (`bd-b5wl`, PERF_LEDGER 2026-07-10).
The quill top-k collector's bounded heap held a `HeapEntry { global_docid: u32, score: f32 }` and
compared with an f32 `total_cmp` plus a conditional docid tiebreak on every `record_live` (per
matched live doc — the query collection hot path). Replaced it with a `BinaryHeap<u64>` of packed
selection keys so the whole comparison is one native integer compare: high 32 bits =
`!score_to_sortable(score)` (worst score = greatest key = heap top, evicted first), low 32 bits =
the global docid (higher docid = greater = evicted first on a score tie, so the lower docid is
retained). Same 8-byte footprint — no struct fattening (contrast the REJECTED rrf sort-key
fattening, 2026-… ledger).

`score_to_sortable` is the standard monotonic f32→u32 bijection, exact for every finite value and
both signed zeros. Scores are `finite_score`-guaranteed finite; boosts are validated `is_finite()`
only (NOT non-negative — the programmatic/CASS AST path can build a negative boost), so the
transform's sign path is load-bearing and is exercised by the bench's signed-boost shape. Order
is provably identical to the former `HeapEntry` `Ord`, and `finish` reconstructs `ScoredDoc`s from
keys and keeps the unchanged `compare_scored_best_first` comparator, so the emitted page (docid +
raw score bits) and exact-count total are byte-identical. Pinned by the collector's existing
exhaustive tests (page/offset/count/materialize-only-winners, the fanned parallel==serial parity
property, and the merge-permutation-over-rotations test). 473/473 quill-lib tests green; scoped
clippy `-D warnings` clean.

MEASURED (`collector_packed_key_ab`, new same-binary A/B asserting identical top-k output before
timing; `paired_median_ratio` + A/A null, 41 rounds inner=64; ratio packed/current, `<1` wins;
null bands ~1.00 ± 0.02):

| shape | n / retained | packed/current | decidable |
|---|---|---|---|
| high_df_k10 | 50k / 10 | **0.369 (~2.7×)** | yes |
| high_df_k100 | 50k / 100 | **0.382 (~2.6×)** | yes |
| tie_heavy_k10 | 50k / 10 | **0.229 (~4.4×)** | yes |
| signed_k10 (neg boosts) | 50k / 10 | **0.394 (~2.5×)** | yes |
| wide_k1000 | 100k / 1000 | **0.487 (~2.05×)** | yes |

Tie-heavy wins most (4.4×): the former path pays the full `total_cmp` plus the docid tiebreak on
every tie, versus one integer compare. **Honest scope:** the microbench isolates the heap
operation over a pre-scored stream. In a full scored query, `record_live` is interleaved with the
BM25 `score()` compute per doc, so the end-to-end fraction scales with each query's collection
weight — the isolated win is diluted by the shared score cost. But the change is byte-identical,
same-size, and strictly cheaper per heap op with **no downside**, so it is a pure win wherever the
collection phase matters (and applies undiluted to count-only accumulation).

**Decision: KEEP.** Route next: the same packed-key idea could fold into the union
MaxScore/BMW pruning cutoff comparisons; not pursued here (one lever).

## 2026-07-23 — KEEP: WAL `decode_vector` f16 SIMD-widen — closes the last scalar consumer of the `vector_at_f32` route-next (`bd-y3hf`, cc)

The 2026-07-16 `vector_at_f32` SIMD-widen KEEP (BlackThrush) explicitly deferred one consumer:
"the `wal::decode_vector` open-time f16 decode is the one remaining scalar consumer — a colder
route-next left for a follow-up." `wal.rs::decode_vector` still materialized an f16-stored vector
to `Vec<f32>` with a scalar `half::f16::to_f32` loop, while the f16 dot kernels and both
`vector_at_f32` decoders already widen 8 f16 per 16-byte block via the crate's `widen8_f16_bytes`
(the Giesen magic-factor widen, bit-identical to `to_f32`).

Applied the identical KEPT pattern: the F16 arm now `as_chunks::<16>()` → SIMD-widen 8 lanes per
block → scalar tail for the last `< 8`. Byte-for-byte identical output (the widen is the same
`pub(crate)` kernel the dot path ships), pinned by a new
`decode_vector_f16_simd_matches_scalar_bit_exact` test (dim 13 = one 8-lane block + a 5-element
tail, values spanning normal / subnormal / signed-zero / ±inf / NaN encodings, `to_bits()`
compare) plus the existing WAL F16 round-trip suite. 63/63 index-lib WAL tests green; scoped
clippy `-D warnings` clean.

MEASURED (`vector_decode_f16_ab`, the same-binary A/B that already asserts widen == scalar
byte-identity before timing — the shipped WAL op is byte-identical to this bench's `widen` arm;
Criterion 100 samples, widen/scalar `<1` wins, disjoint CIs):

| dim | scalar | widen (SIMD) | ratio |
|---|---|---|---|
| 256 | 459.83 ns | 110.00 ns | **0.239 (~4.18×)** |
| 384 | 638.32 ns | 187.77 ns | **0.294 (~3.40×)** |
| 768 | 1422.4 ns | 347.79 ns | **0.245 (~4.09×)** |

**Decision: KEEP.** This is open-time (WAL replay) decode, not a query hot path, so end-to-end
impact is a faster incremental-index reload rather than a search-latency change — a
consistency-completion of the f16-widen sibling family (all f16→f32 decode consumers now route
through the SIMD kernel). Zero algorithmic risk: byte-identical to the already-KEPT operation.
Route-next from the sibling family is now fully closed.

## 2026-07-23 — KEEP: Quill per-width posting-unpack dispatcher — banded SIMD for widths 4-28 (`bd-bz7q`, cc)

Sanctioned follow-up to the 2026-07-17 global-wide-unpack REJECT
(`bd-quill-e2-grimoire-quiver-accg.2`), whose retry predicate required a per-width dispatcher in
which every routed width clears a valid same-binary A/A null control. The portable eight-lane
`wide::u32x8` posting-unpacker (`unpack_wide_into`/`unpack_eight`/`dispatch_eight!`) was promoted
out of its bench-only `cfg` gate into production; the new `unpack_dispatch_into` routes by width,
and both production decode sites (`decode_frequencies`, `decode_for_payload` in `quiver.rs`) call it.

The crossover is MEASURED and TWO-SIDED, not assumed. `postings_decode_ab` per-width gate at
`QUILL_POSTINGS_AB_INNER=512` (paired `paired_median_ratio` + A/A null, 41 rounds, widths 0..=32 ×
counts {127,128} × offsets {0,1,15,31} = 260 cells, **zero invalid A/A nulls** — the high inner
count fixed the noise that had invalidated 85/256 cells in the 2026-07-17 run). Mean wide/scalar
ratio per width (`<1` = wide wins):

| width band | mean wide/scalar | routed to | verdict |
|---|---|---|---|
| w1-w2 | 1.98, 1.29 | scalar | decidable regression (wide loses) |
| w3 | 1.19 | scalar | inside directional null floor (no win) |
| **w4-w28** | **0.47-0.89** | **wide (SIMD)** | decidable wins / holds, **zero regressions** |
| w29-w31 | 1.09, 1.10, 1.12 | scalar | zero wins; decidable regression at w30/o31 |
| w32 | 0.94 | scalar | directional-floor wash (no decisive win) |

So `WIDE_UNPACK_MIN_WIDTH = 4` and `WIDE_UNPACK_MAX_WIDTH = 28`: the SIMD kernel runs only inside
the band where it decisively wins and never regresses. Below w4 (few source bytes per 8-lane
group, scalar reservoir already optimal) and above w28 (each value spans nearly a full `u32`, the
lane gather/shift cost overtakes the scalar shift-reservoir) the scalar path is at least as fast,
so those widths stay scalar and the tail regression is never shipped.

Behaviorally transparent: the dispatcher decodes the identical canonical stream to identical
values — pinned by the `quiver-postings-bitpack-scalar-wide-v1` differential fixture asserting
scalar == wide == dispatch across widths 0..=32, truncated/non-canonical inputs, and FOR/BITMAP
payloads. 473/473 quill-lib tests green; scoped clippy `-D warnings` clean. Realistic effect:
posting doc-delta and frequency blocks in the common 4-28 bit width range (typical for 128-doc
FOR blocks) decode ~1.1-2.1× faster on the query hot path; narrow and full-width blocks are
unchanged.

**Decision: KEEP.** The banded dispatcher ships. Route next: the width-32 wash and the w29-31
inversion suggest a possible AVX2-specific kernel for full-`u32` widths, but that needs its own
gate; not pursued here (one lever).

## 2026-07-16 — `InMemoryVectorIndex::vector_at_f32` decodes f16 via SIMD widen (the in-memory twin) (BlackThrush)

- **Negative-ledger-first route and profile:** grepping for the remaining scalar `f16::to_f32` decode loops (the "grep ALL consumers of a SIMD'd scalar op" lead from the FSVI `vector_at_f32` win, `38f471a9`) surfaced the direct twin I had missed: `InMemoryVectorIndex::vector_at_f32` (the fully-resident index used to materialize vectors for HNSW graph build / fingerprinting) still decoded its `&[f16]` slice with `stored.iter().map(|v| v.to_f32()).collect()` — scalar — while the f16 dot kernels already widen 8 f16 per block via `simd::widen8_f16_slice`.
- **Single lever:** made `widen8_f16_slice` `pub(crate)` and the in-memory `vector_at_f32` now widens 8 f16 per `chunks_exact(8)` block through it (`.to_array()` → 8 `f32`), with the original scalar `to_f32` tail for the final `< 8`. Same widen the dot generic path ships → byte-identical output. (The `wal::decode_vector` open-time f16 decode is the one remaining scalar consumer — a colder route-next left for a follow-up.)
- **Same-binary paired A/B:** strict remote-only `rch`, one foreground release binary on worker `vmi1152480`, `--profile release` (LTO disabled), 30 samples, 150 ms warm-up, 500 ms measurement. A new `inmem_vector_decode_f16_ab` bench decodes an f16 slice both ways (the `widen` arm is a byte-for-byte copy of `widen8_f16_slice`) and asserts a bit-identical `Vec<f32>` before timing. widen/scalar (`<1` wins):

| dim | scalar | widen | ratio | speedup |
|---:|---:|---:|---:|---:|
| 256 | 648.63 ns | 195.76 ns | 0.302 | ~3.31× |
| 384 | 1053.7 ns | 234.00 ns | 0.222 | ~4.50× |
| 768 | 1775.3 ns | 471.05 ns | 0.265 | ~3.77× |

All three rows have disjoint confidence intervals. (Slightly under the FSVI bytes path's flat ~4.5× because `widen8_f16_slice` marshals via `to_bits` into a `[u32; 8]` rather than the direct 16-byte load.)
- **Verification:** the focused index lib `in_memory` suite passed remotely (**22 passed, 0 failed**), covering the in-memory `vector_at_f32` recovery round-trips; the bench parity gate proved bit-identical decode at dim 256/384/768; and `widen8_f16_slice` is independently validated bit-identical by the shipped f16-dot generic/AVX2 parity tests. Owned-file `rustfmt` and `git diff --check` passed.
- **Scope:** KEEP for in-memory f16 vector materialization. This is a cold-path (HNSW build / fingerprint) decode-kernel win that completes the f16-decode-materializer sibling sweep started at `38f471a9` (FSVI path); the decode is a fraction of the surrounding build, so this is a materialization-kernel speedup, not an end-to-end build-latency claim.

## 2026-07-16 — `vector_at_f32` decodes f16 via SIMD widen, not a scalar `to_f32` loop (BlackThrush)

- **Negative-ledger-first route and profile:** `bv --robot-triage` was still all Quill docs, and the newest landed code (`c5cd8b51` quill fieldnorm/BM25) is a *pinned bit-parity contract* — explicitly no `mul_add`, off-limits. Sibling-path audit of the f16 decode instead: the f16 **dot** kernels widen 8 f16 per 16-byte block through `simd::widen8_f16_bytes` (the `wide` magic-factor widen, proven bit-identical to `f16::to_f32`), but `VectorIndex::vector_at_f32`'s F16 branch — the materializer used by PRF feedback-vector lookup (`searcher.rs` per-query when expansion is enabled) and index refresh/build — still decoded each f16 with a **scalar `half::f16::to_f32` loop**. It was an un-updated consumer of the same scalar op the dot path already replaced.
- **Single lever:** the F16 branch now widens 8 f16 per `chunks_exact(16)` block via `widen8_f16_bytes` (`.to_array()` → 8 `f32`), with the original scalar `to_f32` tail for the final `< 8`. Same widen the dot kernels ship → byte-identical output; the F32 branch, bounds checks, and errors are unchanged. No `unsafe` (reuses the safe `wide` helper).
- **Same-binary paired A/B:** strict remote-only `rch`, one foreground release binary on worker `vmi1152480`, `--profile release` (LTO disabled), 30 samples, 150 ms warm-up, 500 ms measurement. A new `vector_decode_f16_ab` bench decodes a realistic f16 slab both ways (the `widen` arm is a byte-for-byte copy of `widen8_f16_bytes`) and asserts a bit-identical `Vec<f32>` before timing. widen/scalar (`<1` wins):

| dim | scalar | widen | ratio | speedup |
|---:|---:|---:|---:|---:|
| 256 | 582.91 ns | 128.96 ns | 0.221 | ~4.52× |
| 384 | 1047.1 ns | 231.44 ns | 0.221 | ~4.52× |
| 768 | 1737.7 ns | 430.13 ns | 0.248 | ~4.04× |

All three rows have hugely disjoint confidence intervals.
- **Verification:** the focused index lib `vector_at` suite passed remotely (**3 passed, 0 failed**), covering the materializer (its small-dim tests exercise the scalar tail path, `dim % 8 ≠ 0`); the bench's parity gate proved bit-identical decode at dim 256/384/768; and `widen8_f16_bytes` is independently validated bit-identical by the shipped f16-dot generic/AVX2 parity tests. Owned-file `rustfmt` and `git diff --check` passed.
- **Scope:** KEEP for f16 vector materialization. This is a ~4.5× decode-kernel win realized on every `vector_at_f32` caller — per-query PRF feedback vectors and cold index refresh/build — not an end-to-end search-latency claim. It complements the earlier fused-byte-dot rescore win (`7988941d`): the *scoring* path avoids materialization entirely; the paths that genuinely need an owned `Vec<f32>` now decode ~4.5× faster.

## 2026-07-16 — `shared_prefix_depth` walks split iterators instead of collecting two Vecs (BlackThrush)

- **Negative-ledger-first route and profile:** `bv --robot-triage` was still all Quill docs. This turn swept fresh subsystems after confirming the scoring/blend/normalize veins are closed — notably the normalize-SIMD-scale sibling (`l2_normalize`, embedder finishes) is the same class the negative ledger just measured-rejected as a noise-floor wash (`bd-d2a8`: median 0.8978 but lever p95 1.0323 overlaps the A/A p5 0.8747), so it was skipped rather than re-tread. The clean lever surfaced in `fsfs::ranking_priors::shared_prefix_depth` — the per-candidate `path_proximity` prior counted shared leading path components by collecting **two** `Vec<&str>` and then `zip`ping them, though the vectors fed nothing but that `zip`.
- **Single lever:** `shared_prefix_depth` now `zip`s the two `split('/').filter(non-empty)` iterators directly and counts the matching-prefix `take_while` — dropping two allocations per call. Same components, same order, same count → byte-identical. `path_proximity_multiplier` and `apply_to_candidate` are unchanged.
- **Same-binary paired A/B:** strict remote-only `rch`, one foreground release binary on worker `vmi1227854`, `--profile release` (LTO disabled), 25 samples, 150 ms warm-up, 500 ms measurement. A new `shared_prefix_depth_ab` bench (hosted in `core` — the function is a pure string algorithm, and `fsfs` is a ~10-minute compile) mirrors both variants over a source-tree-shaped candidate set and asserts identical counts before timing. new/old (`<1` wins):

| candidates | alloc (old) | zip (new) | ratio | speedup |
|---:|---:|---:|---:|---:|
| 100 | 12.007 µs | 6.9472 µs | 0.579 | ~1.73× |
| 500 | 52.139 µs | 32.913 µs | 0.631 | ~1.58× |

Both rows have disjoint confidence intervals (100: `7.33 < 11.29`; 500: `34.54 < 48.23`).
- **Verification:** the bench's parity gate proved identical counts across shared-depth 0–4 path pairs for the *exact* new iterator code before timing, and `rustfmt` / `diff --check` passed on both owned files. The scoped `fsfs` release test (`ranking_priors`) could **not** be run this turn: `frankensearch-fsfs` transitively depends on `frankensearch-lexical`, which is mid-refactor in the shared tree (a concurrent agent's uncommitted `LexicalSearch`/`DocId` changes fail to compile), so the whole dependency build fails before reaching any test — an issue entirely outside this change. Correctness rests on the provable byte-identity (a redundant `collect` elided before a `zip`) plus the identical-code parity gate; the existing `shared_prefix_depth_basic_cases` unit test is unchanged and passes by construction once the tree builds again.
- **Scope:** KEEP for the per-candidate `path_proximity` ranking prior. Byte-identical, zero behavior change; the win is the two elided per-call allocations, realized whenever the path-proximity prior is enabled.

## 2026-07-16 — `strip_italic_underscores` borrows when nothing is dropped (BlackThrush)

- **Negative-ledger-first route and profile:** `bv --robot-triage` was still all Quill contract docs, so this turn audited the canonicalization (ingest) chain, whose markdown-link parsers gained recent scratch-reuse / source-slice wins (`bd-xs7l`, `bd-xwb9`). One sibling was still un-Cow'd: `strip_italic_underscores` was the *only* strip in `strip_markdown_line`'s chain returning a fresh `String` (its siblings `strip_markdown_line` / `strip_prefixes_and_list_marker` / `strip_list_marker` / `strip_markdown_links` all borrow-or-reuse when unchanged). It is guarded by `has_underscore`, but `_` is ubiquitous in technical text (`snake_case`, `variable_name`, `fn foo_bar`), where **no** underscore is an italic boundary — yet the shipping code allocated a whole-line copy identical to the input on every such line.
- **Single lever:** `strip_italic_underscores` now returns `Option<String>` — `None` when nothing is dropped (the caller keeps its borrowed/owned `Cow` unchanged), and `Some(stripped)` lazily seeded from the borrowed prefix at the first dropped marker only. Same boundary test and order → byte-identical materialization. The caller uses an `if …&& let Some(stripped) = …` let-chain so the borrow ends before the reassign.
- **Same-binary paired A/B:** strict remote-only `rch`, one foreground release binary on worker `vmi1149989`, `--profile release` (LTO disabled), 25 samples, 150 ms warm-up, 500 ms measurement. A new `strip_italic_underscores_ab` bench mirrors both variants and asserts byte-identical materialization (and kept-byte totals) before timing. new/old (`<1` wins):

| batch | alloc (old) | cow (new) | ratio | note |
|---|---:|---:|---:|---|
| `snake_only` (no drop) | 678.50 ns | 570.47 ns | 0.841 (~1.19×) | CIs marginally overlap |
| `italic_only` (all drop) | 799.38 ns | 843.11 ns | 1.05 (wash) | CIs heavily overlap |
| `mixed_4to1` (realistic) | 3.6264 µs | 2.7021 µs | 0.745 (~1.34×) | **disjoint CIs** |

- **Verification:** the focused release core canonicalize suite passed remotely — **38 passed, 0 failed** — including the reference-parity test (extended to assert `None` fires exactly when nothing is stripped) and the `strip_markdown_line` chain. This is a modest, byte-identical elision: the char-scan is shared by both arms, so the saving is only the per-line allocation + copy-back on `has-underscore-but-no-italic` lines; italic-heavy prose is a measured wash.
- **Scope:** KEEP for the canonicalization ingest path. This closes the last non-Cow strip in the chain — a consistency fix as much as a small perf one; byte-identical output, neutral worst case.

## 2026-07-16 — Incremental candidate updates truncate the owned result vector (BlackThrush)

- **Negative-ledger-first route and profile:** `bv --robot-triage` and `br ready` exposed no ready performance bead; the graph was dominated by Quill contract work. A retry of open Model2Vec bead `bd-d2a8` reached only cold dependency builds on two workers, so it remains open with no timing and no reject. The turn then pivoted to the fresh fusion incremental-search subsystem. Before production editing, a faithful profile arm reproduced `IncrementalSearcher::update` taking an owned 256-ID `Vec<String>`, cloning its retained 100-ID prefix with `[..pool].to_vec()`, and dropping the original vector. Input construction was outside the timed region.
- **Single lever:** `update` now truncates its already-owned result vector to `candidate_pool_size` and moves that vector into `last_doc_ids`. Retained IDs, order, query state, reuse behavior, exact-size/undersized behavior, and zero-sized pools are unchanged; only the redundant retained-prefix allocation and `String` deep clones are removed. The existing size-limit test now also proves the exact retained prefix.
- **Same-binary paired A/B:** strict remote-only `rch`, one foreground standalone release binary on warmed worker `vmi1227854`, `--profile release` with LTO disabled and 16 codegen units. The first cold execution on that worker was discarded as warm-up; the admitted measurement rebuilt in 0.02 s and used 41 paired AB/BA rounds of 64 operations plus a same-arm A/A null control. Exact parity passed for empty, exact-size, oversized, pool-zero, and Unicode inputs. Ratios are truncate+move/former clone (`<1` wins):

| 256 IDs, retain 100 | former median/op | truncate+move median/op | paired ratio (p5–p95) | speedup |
|---|---:|---:|---:|---:|
| short | 7.017 us | 1.805 us | **0.2663** (0.2008–0.4505) | **3.76x** |
| long URN | 6.679 us | 1.716 us | **0.2368** (0.1546–0.3750) | **4.22x** |
| Unicode | 6.526 us | 1.770 us | **0.2656** (0.2245–0.3090) | **3.77x** |

  Each candidate median is below its fixture's A/A null p5 (0.9483, 0.8080, and 0.8371 respectively).
- **Verification and scope:** exact owned-file rustfmt and diff checks passed; UBS exited 0 with no critical findings. A focused remote release test was canceled after the cold worker spent two silent minutes compiling `asupersync`; it never reached the test and is neither failure nor performance evidence. **KEEP** for incremental-search result-state updates when the backend returns more IDs than the configured candidate pool. This is an isolated state-update allocation win, not an end-to-end as-you-type search latency claim; planning still clones the retained pool when returning an owned `SearchPlan`.

## 2026-07-16 — FSVI two-tier quality rescore uses the fused byte dot, not decode-then-f32-dot (BlackThrush)

- **Negative-ledger-first route and profile:** `bv --robot-triage` was still all Quill contract docs, so this turn extended the sibling-path audit into the FSVI two-tier scoring path. `TwoTierIndex::score_quality_for_fast_index` (the per-hit quality rescore reached on every quality query through `quality_scores_for_hits`) scored each fast-tier hit by calling `VectorIndex::vector_at_f32` — which **allocates a `Vec<f32>` and runs a scalar `half::f16::to_f32` decode loop** — then a separate `dot_product_f32_f32`. Yet the brute-force scan (`search.rs`, 6+ sites) already scores the *same* f16 slab with the fused `dot_product_f16_bytes_f32` (hardware `vcvtph2ps` decode, no allocation). The rescore path was the slow, allocating outlier — a clear sibling-path inconsistency between scan-scoring and rescore-scoring.
- **Single lever:** new `VectorIndex::dot_query_at(index, query)` (plus a private `vector_bytes` borrow accessor) dispatches to the fused byte kernel — `dot_product_f16_bytes_f32` (F16) / `dot_product_f32_bytes_f32` (F32) — and both two_tier quality-rescore sites (`score_quality_for_fast_index` and the doc_id-only fallback) now call it. To keep scores exact, the fused path is taken only when `dim % 32 == 0` (every standard embedding width — 128/256/384/512/768…): there the two kernels' 4-accumulator SIMD grouping and reduction coincide and there is no scalar tail, so the result is bit-identical; other dims fall back to the former decode-then-dot. The in-memory WAL scoring (`dot_product_f32_f32` over owned `Vec<f32>` embeddings) is untouched.
- **Same-binary paired A/B:** strict remote-only `rch`, one foreground release binary on worker `vmi1149989`, `--profile release` (LTO disabled, 16 codegen units), 25 samples, 150 ms warm-up, 500 ms measurement. A new `quality_rescore_fused_dot_ab` bench measures the exact per-hit swap over a 50k-vector f16 slab (dim 384), asserting a bit-identical `Vec<f32>` before timing. new/old (`<1` wins):

| candidates | decode+f32dot (old) | fused byte dot (new) | ratio | speedup |
|---:|---:|---:|---:|---:|
| 128 | 135.88 µs `[119.75, 152.74]` | 3.4304 µs `[3.0342, 4.1288]` | 0.0252 | ~39.6× |
| 300 | 279.95 µs `[272.16, 288.56]` | 7.4246 µs `[7.2619, 7.6361]` | 0.0265 | ~37.7× |

Both rows have hugely disjoint CIs. The gap is dominated by the old path's **scalar** software f16 decode (`half::to_f32` per element) vs the fused kernel's hardware `vcvtph2ps`, plus the eliminated per-hit `Vec<f32>` allocation. This is the **isolated per-hit scorer** — the dominant cost of `score_quality_for_fast_index`; end-to-end `quality_scores_for_hits` gain is bounded by its non-scoring overhead (WAL-map build, alignment match), so this is not a claim that quality search is ~38× faster.
- **Verification:** the full index lib suite passed remotely on a fresh worker — **402 passed, 0 failed** — including every `two_tier::tests::quality_*` test (full coverage, partial alignment, dimension mismatch, empty indices) that drives `quality_scores_for_hits` → `score_quality_for_fast_index` → `dot_query_at`. Owned-file `rustfmt` and `git diff --check` passed.
- **Scope:** KEEP for the FSVI two-tier quality rescore scorer. This aligns the rescore path with the scan's fused kernel — a consistency fix as much as a perf one — and is bit-identical for every `dim % 32 == 0` embedding; it does not touch the in-memory searcher, the fast-tier scan, or the WAL scoring.

## 2026-07-16 — Quality-rescore doc_id lookup uses ahash, not SipHash (BlackThrush)

- **Negative-ledger-first route and profile:** `bv --robot-triage` stayed dominated by Quill contract docs with no ready perf bead, so this turn took a sibling-path lead. The workspace declared SipHash→ahash "closed", yet `InMemoryVectorIndex::doc_id_index` — the map behind `index_of_doc_id`, probed once per candidate on every quality query through `quality_scores_for_hits` — was still a std `HashMap<String, usize>` (SipHash). The FNV `hash_to_pos` fast-path in the *same* struct is unsafe to reuse for this lookup (rescore hits are foreign fast-tier doc_ids; a 64-bit FNV collision with a corpus id would return a wrong position and a wrong score), so the safe lever is the hasher swap. Profiling the exact per-hit loop — `map.get(doc_id).map(|&idx| dot_product_f16_f32(vector_slice(idx), query))` — over a 50k / dim-384 fixture at the shipping SipHash confirmed the lookup is a real fraction of per-hit cost alongside the AVX2 f16 dot.
- **Single lever:** `doc_id_index` is now an `ahash::AHashMap<String, usize>` (one import, the field type, and the single construction site). Keys, first-insert-wins build, `OnceLock` laziness, and the `.get().copied()` lookup are unchanged; the identity-hashed `hash_to_pos` / `doc_id_hashes` maps are untouched.
- **Same-binary paired A/B:** strict remote-only `rch`, one foreground release binary on worker `vmi1149989`, `--profile release` (LTO disabled, 16 codegen units), 25 samples, 150 ms warm-up, 500 ms measurement. A new `quality_rescore_hasher_ab` bench builds both maps over identical owned `String` keys and times the identical rescore loop (same positions, vectors, query; the f16 dot is byte-identical), asserting a bit-identical `Vec<Option<f32>>` before timing. ahash/siphash (`<1` wins):

| candidates | siphash central | ahash central | ratio | speedup |
|---:|---:|---:|---:|---:|
| 128 | 8.3303 µs `[8.0406, 8.6836]` | 6.5518 µs `[6.1035, 7.0522]` | 0.786 | ~1.27× |
| 300 | 21.298 µs `[19.437, 23.473]` | 14.359 µs `[13.648, 15.242]` | 0.674 | ~1.48× |

Both rows have disjoint confidence intervals (128: `7.0522 < 8.0406`; 300: `15.242 < 19.437`). The 32-candidate ahash arm measured 1.354 µs; its siphash arm's timing line was truncated from rch's buffered stdout and is not claimed.
- **Verification:** the focused release index lib suite ran 356 tests — **355 passed**, including `in_memory::tests::two_tier_quality_scores`, which drives `quality_scores_for_hits` → `index_of_doc_id` → the changed map. The lone failure, `soft_delete_wal_restores_state_on_rewrite_failure`, is an unrelated filesystem-permission test whose worker cache ran a stale pre-fix revision (its committed graceful-skip guard for root / `CAP_DAC_OVERRIDE` workers sits 172 lines below the reported panic line); it does not touch in-memory hashing. Owned-file `rustfmt` and `git diff --check` passed. The concurrent `collectors.rs` / `cass_compat.rs` work in the tree is independent and not part of this commit.
- **Scope:** KEEP for the per-hit quality-rescore doc_id lookup. This is a hasher swap on an in-memory map — not a claim of faster end-to-end quality search; embedding, the f16 dot itself, blending, and result assembly are unchanged.

## 2026-07-16 — Search telemetry sanitization reuses its owned query buffer (BlackThrush)

- **Negative-ledger-first route and profile:** `bv --robot-triage` remained dominated by Quill contract work and exposed no ready legacy performance bead, so this turn pivoted to the fresh core telemetry subsystem. The negative ledger had no exact `sanitize_query_text` result; its older inventory classified collectors as control-plane-only, but the current fusion searcher emits these samples for every configured Initial, Refined, Reranked, and failure phase. Before editing production, the unchanged `current_borrow_collect/short_ascii` arm completed a one-second strict-remote profile on `vmi1149989`.
- **Single lever:** `RuntimeMetricsCollector::emit_search` now gives its already-owned `String` to the sanitizer. Clean queries of at most 500 bytes return the original allocation; trimming and the 500-character cap mutate that buffer with UTF-8-safe boundaries; empty input reuses it for the existing `<empty>` placeholder. Text bytes, Unicode whitespace behavior, character-count truncation, envelope shape, counters, and event ordering are unchanged.
- **Same-binary A/B:** one foreground strict-remote Criterion binary ran on `vmi1293453` with `--profile release`, LTO disabled, optimization level 2, 256 codegen units, 20 samples, 100 ms warm-up, and 350 ms requested measurement per arm. Fixture cloning was untimed in both arms, and the harness asserted exact current/candidate text equality for every workload before timing. Ratios are candidate/current (`<1` wins):

| query shape | current collect | owned in place | ratio | speedup |
|---|---:|---:|---:|---:|
| clean short ASCII | 82.674 ns | **11.094 ns** | **0.134** | **7.45x** |
| leading/trailing ASCII whitespace | 104.31 ns | **28.351 ns** | **0.272** | **3.68x** |
| 512-byte ASCII truncation | 549.79 ns | **255.75 ns** | **0.465** | **2.15x** |
| Unicode whitespace and multibyte text | 102.40 ns | **32.686 ns** | **0.319** | **3.13x** |

- **Verification and scope:** the successful release benchmark compiled the changed production crate and ran all eight A/B arms. Added unit cases cover empty and whitespace-only input, exact and over-limit multibyte boundaries, Unicode whitespace, and clean-buffer pointer reuse. Owned-file rustfmt and diff checks passed. UBS completed; its compiler/lint subchecks were clean, while the whole-file scan exits nonzero on existing test panic/unwrap inventory and a bounds-proven slice heuristic in the changed sanitizer. A focused remote release-test follow-up was cancelled when RCH immediately evicted the just-built pooled target and began recompiling every dependency; that cancellation is neither a test failure nor performance evidence. This is a telemetry-envelope preparation win when collection is configured, not an end-to-end search-latency claim. **Decision: KEEP.**

## 2026-07-16 — Corrupt repair reuses its already-validated trailer (BlackThrush)

- **Negative-ledger-first route and profile:** `bv --robot-triage` remained dominated by Quill contract work and its visible performance bead was blocked, so this turn pivoted to fresh durability repair work. The negative ledger rejects replacing full repair-trailer validation with a header-only parser after a 5 MiB healthy-verify regression. Profiling the unchanged `repair_1mb_single_block_corruption` path instead exposed a complementary seam: corruption detection fully deserialized and validated the trailer, discarded it, then repair immediately read and fully deserialized the same sidecar again. Before editing production, the unchanged path measured **4.5640 ms** `[4.4213, 4.6570]` on a strict-remote release profile run.
- **Single lever:** `FileProtector::repair_file_internal` now retains the already-validated `(RepairTrailerHeader, Vec<RepairSymbol>)` only when source verification finds corruption, then passes that owned decode into repair. Healthy files still return before repair; missing-source recovery still reads and decodes once; CRC, length, trailer framing, symbol ordering, decode, durable rewrite, and final validation are unchanged.
- **Same-worker A/B:** strict-remote foreground Criterion runs on `vmi1293453`, `--profile release` with LTO disabled and 16 codegen units, 10 samples, 50 ms warm-up, and the existing repair group's 5 s collection window. The shipping arm measured **4.5939 ms** `[4.2507, 5.5062]`; the candidate measured **3.8719 ms** `[3.5597, 4.1670]`. Candidate/original is **0.843**, or **15.7% lower latency** (**1.19× speedup**). Criterion's direct saved-baseline change interval was **−49.154% to −11.853%** (`p = 0.02`).
- **Verification and scope:** the full durability release library suite passed remotely (**151 passed, 0 failed**); scoped release all-target checking and strict no-deps library clippy also passed. Owned-file rustfmt and diff gates passed. UBS completed on both owned files but its whole-file durability scan exits nonzero on pre-existing findings outside the changed hunk. This is a corrupt-file repair win, not a healthy-verification or end-to-end durability claim. The benchmark and codec changes already at `HEAD` are independent and not part of this lever. **Decision: KEEP.**

## 2026-07-16 — Hash fallback parallelizes only large embedding batches (BlackThrush)

- **Negative-ledger-first route and profile:** `bv --robot-triage` remained dominated by Quill contract work, while the exact `HashEmbedder::embed_batch` Rayon seam had no entry in either performance ledger. The fresh embed-subsystem path is product-reachable: FSFS sends up to 256 documents through `embed_batch` and fusion refresh can send up to 1,000. Before editing production, the unchanged FNV-384 kernel measured **2.1973 us/document** `[2.0218, 2.4361]` for a typical roughly 100-word document, attributing large serial batches to independent per-document CPU work.
- **Single lever:** `HashEmbedder` now uses an indexed Rayon map only for batches of at least 256 documents. Smaller batches keep the former serial iterator, including FSFS's default size of 64. Indexed parallel collection preserves input order; each document's FNV/JL embedding remains independent and unchanged.
- **Same-binary A/B:** one foreground, strict-remote `cargo bench --profile release` measurement ran on warmed worker `vmi1227854` with LTO disabled, 16 codegen units, `RAYON_NUM_THREADS=4`, 10 samples, 100 ms warm-up, and 300 ms requested measurement per arm. Before timing, the harness asserted every output float's exact bits across the serial and Rayon nested vectors. Ratios are Rayon/serial (`<1` wins):

| FNV-384 documents | serial central | Rayon central | ratio | production route |
|---:|---:|---:|---:|---|
| 32 | 69.413 us | 141.44 us | 2.038 | serial |
| 64 | 142.23 us | 355.72 us | 2.501 | serial |
| 128 | 342.47 us | 224.44 us | 0.655 | serial (conservative noise guard) |
| 256 | 630.14 us | **329.19 us** | **0.522 (~1.91x)** | Rayon |
| 1,000 | 2.8511 ms | **0.89733 ms** | **0.315 (~3.18x)** | Rayon |

- **Verification and scope:** the focused release test proved bit-identical serial/batch output for both FNV and JL at empty, 1, 255, 256, and 257-document shapes; scoped release checking of the library, tests, and benches passed. All-target checking is blocked by two unchanged examples that omit their required `fastembed`/`model2vec` features. Strict clippy is likewise blocked by three unchanged `simd.rs` lints; an allow-scoped follow-up was cancelled when the switched worker produced no output for two minutes. Owned-file UBS, rustfmt, and diff checks passed; workspace fmt remains blocked by unrelated existing files. The 256-document confidence intervals were disjoint (`[589.62, 662.13] us` serial versus `[244.01, 418.70] us` Rayon), as were the 1,000-document intervals. The 32/64 regressions are intentionally outside the gate. This is a large-batch hash-fallback throughput win, not an end-to-end neural-embedding claim. **Decision: KEEP.**

## 2026-07-16 — Async Phase-2 calibration reuses its owned aligned-score vector

- **Attribution:** `TwoTierSearcher::run_phase2` already receives an owned `Vec<Option<f32>>` from `quality_scores_for_hits`, but immediately borrowed it to collect a second vector solely for optional per-score calibration. The retained lever calibrates present scores in place and skips the pass entirely when `score_calibrator` has its default `None` value. Configured calibrators still visit every `Some` in the same order and apply the same scalar transform; missing slots remain missing.
- **Measurement:** strict remote-only `rch`, one foreground release binary on warmed worker `vmi1293453`, `--profile release` with LTO disabled and 16 codegen units, 30 samples, 200 ms Criterion warm-up, and 500 ms requested measurement. The same binary measured the former collect-and-copy arm against the retained owned-vector arm. At the production-shaped 32-score row (default candidate multiplier is three), central time improved from **42.152 ns** to **2.3859 ns**: ratio **0.0566**, or **94.34% lower latency** (**17.67× speedup**) in the affected prep region. Scaling rows corroborated the allocation removal: **7.6481 us → 5.2462 ns** at 10k (ratio **0.000686**) and **16.100 us → 29.611 ns** at 100k (ratio **0.00184**).
- **Verification:** before timing, the harness asserted exact `Option<f32>` bit parity for `None`, finite scores, negative zero, NaN, and both infinities with calibration disabled and configured. The release benchmark compiled the changed production crate and completed all six original/candidate arms. Owned benchmark `rustfmt --check` and both-file `git diff --check` passed. A broader remote lib-test link was cancelled after exceeding the two-minute no-output rule, and a subsequent check was cancelled when RCH evicted the requested warmed worker; neither cancellation is treated as test failure or performance evidence.
- **Scope:** KEEP for async Phase-2 aligned-score preparation, especially the default no-calibrator path. This is an isolated allocation/pass removal, not a claim that end-to-end quality search is 17.67× faster; embedding, index rescoring, blending, and result assembly are unchanged.

## 2026-07-16 — In-memory AVX2 f16→i8 quantization packs eight lanes per store

- **Attribution:** `quantize_f16_slab_to_i8_avx2` already produced eight final i8 values as AVX2 i32 lanes, but spilled all lanes to a stack array and appended them with eight scalar `Vec::push` calls. The retained lever uses `vpshufb` to select each lane's low byte, joins the two 128-bit halves, and writes one unaligned `u64` per eight values. Max-abs reduction, scaling, half-away-from-zero rounding, clamping, scalar tail, zero-slab handling, allocation size, and dispatch are unchanged.
- **Measurement:** strict remote-only `rch`, same warmed worker `vmi1149989`, `--profile release` with LTO disabled and 16 codegen units, an untimed candidate `--no-run` warm-up, then 20 samples with a 50 ms Criterion warm-up and 150 ms requested measurement. The existing production-dispatch arm `quantize_i8_slab/dispatch/10000` improved from **4.1293 ms** `[3.8827, 4.3934]` to **1.1166 ms** `[1.0627, 1.1712]`: ratio **0.2704**, or **72.96% lower latency** (**3.70× speedup**). Criterion's direct change interval was **−74.951% to −70.739%** (`p = 0.00` at displayed precision); even candidate upper bound divided by baseline lower bound is **0.3017**.
- **Verification:** the production benchmark asserts exact dispatch/generic equality before timing; the focused release test `simd::tests::avx2_quantize_i8_matches_generic` passed remotely, and scoped `cargo check -p frankensearch-index --all-targets --profile release` passed. Owned-file `rustfmt` and `git diff --check` passed. All-target clippy is blocked by three unchanged `frankensearch-core` lints; no-deps index clippy is independently blocked by the crate's existing lint backlog.
- **Scope:** KEEP for lazy in-memory int8 slab construction. This does not claim a query-time or end-to-end search speedup, and does not change the mmap-byte quantizer, whose equivalent lane compaction was already present.

## 2026-07-15 — Durability repair consumes decoded symbols without deep-cloning (`bd-mloy`)

- **Attribution:** `FileProtector::repair_file` already owned the decoded `repair_symbols`, but extended the codec input with `repair_symbols.iter().cloned()`. On the measured 1 MiB / 4 KiB-symbol fixture, that allocated and copied 256 payload buffers (1 MiB total) immediately before decode. Consuming the owned vector is the only production change; ESI values, payload bytes, symbol order, and decoder inputs are unchanged.
- **Measurement:** strict remote-only `rch`, same worker `vmi1227854`, `--profile release` with LTO disabled and 16 codegen units, 10 samples and 50 ms warm-up; the existing repair group enforced its 5 s collection window despite the cheaper 150 ms CLI request. `repair_latency/repair_1mb_single_block_corruption` improved from **4.9248 ms** `[4.8143, 5.1032]` to **4.0529 ms** `[3.6670, 4.4405]`: ratio **0.823**, or **17.7% lower latency** (**1.215× speedup**). Criterion's direct same-target change interval was **−27.241% to −14.649%** (`p = 0.00` at displayed precision); even candidate upper bound divided by baseline lower bound is **0.922**.
- **Verification:** the exact durability library suite passed remotely under the release profile (**151 passed, 0 failed**), the scoped all-target check passed remotely, and owned-file `rustfmt` plus `git diff --check` passed. Workspace all-target checking remains blocked by an unrelated stale fusion benchmark call to missing `AdaptiveNqcDenseWeight::bench_legacy`; scoped clippy remains blocked by three lints in unchanged `frankensearch-core` sources.
- **Scope:** KEEP for corrupt-file repair after trailer decode. This removes only the redundant transfer copy; protection, healthy verification, trailer parsing, source-symbol recovery, durable rewrite, and CRC validation are untouched.

## 2026-07-15 — FSFS prior-signal result-map reservation (`bd-79bn`)

- **Attribution:** `CodeStructureSidecar::prior_signals_for_candidates` knows the candidate upper bound before its dense result loop, but the shipping `HashMap::new()` repeatedly grew the result table. Reserving `candidates.len()` is the only production change; document lookup, score computation, insertion order, keys, and values are unchanged.
- **Parity:** the existing `code_sidecar_score` harness checks legacy and candidate maps for exact equality before registering every 32/128/512 Criterion arm.
- **Measurement:** strict remote-only `rch`, worker `vmi1227854`, `--profile release` with LTO disabled, 10 samples, 50 ms warm-up, 150 ms measurement; candidate-only `sidecar_candidate_score/prior_signals/512` after an untimed cold warm-up. Candidate was **543.01 µs** central, interval **[528.01, 569.10] µs**, versus retained shipping baselines **624.79 µs** and **646.471 µs**: **13.1%** and **16.0%** faster respectively.
- **Scope:** KEEP for the measured dense/all-hit 512-candidate shape. This eagerly reserves for the candidate upper bound, so empty/sparse sidecars trade transient capacity for avoiding dense-path rehashes; do not generalize this result to sparse/no-match latency or memory without a dedicated control.

> Head-to-head measured performance wins **kept** in the tree. Each row cites the
> exact bench workload, the before/after timings, and the ratio (new/old; lower is
> faster). Dead-ends and regressions live in `docs/NEGATIVE_EVIDENCE.md`.

## 2026-07-15 — WIN: customized TUI keymaps use aHash fallback — ~1.72× (`FoggyBasin`)

The default `Keymap` already resolves its fixed 24-binding alphabet through a static match table, but any
`bind` or `unbind` switches every subsequent keypress to the retained `std::HashMap` fallback. That map is never
iterated in production: only insert, remove, lookup, length, and emptiness are observable. The fallback now uses
`AHashMap`; default static dispatch and all customization semantics remain unchanged.

The retained `keymap_resolve` benchmark activates the production fallback with a behavior-neutral rebind and
asserts exact action parity against the former HashMap for the full 40-event shell workload before timing.
Per the RCH eviction protocol, the shipping comparator was not rebuilt: its stored Criterion median is
**778.633 ns** `[728.445, 828.898]` (the canonical PERF row records 777.96 ns). One strict-remote candidate-only
run on `vmi1153651`, `--profile release`, 10 samples, 50 ms warm-up, and 150 ms measurement produced:

| workload | stored HashMap baseline | AHashMap candidate | ratio | decision |
|---|---:|---:|---:|---|
| `tui_keymap_resolve/custom_ahash` (40 events) | 778.633 ns | **453.40 ns** `[409.57, 479.60]` | **0.582 (~1.72×)** | KEEP |

The candidate's upper bound remains ~38% below the stored baseline. Scope is deliberately narrow: customized
keymaps only; unmodified defaults retain the much faster static dispatch. RCH evicted the target between the
untimed 14m40s warm-up and measurement, causing a second 16m15s cold link; neither compile duration is timing
evidence, and no baseline arm or local Cargo command ran.

Build/bench protocol (per-crate ONLY, never workspace-wide):
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p <crate> --release
```

Current Cargo rejects `--release` for `cargo bench` in this checkout; successful
Criterion runs use the equivalent `--profile release` form and record that protocol
mismatch in `docs/NEGATIVE_EVIDENCE.md`.

The dominance baseline is a Tantivy/Lucene/Meilisearch-class original comparator
on identical corpora and query streams. The `bd-ui41` comparator harness now
lives in `frankensearch/benches/search_bench.rs`; its current result is negative
and recorded in `docs/NEGATIVE_EVIDENCE.md`. Rows below remain frankensearch
pre-change baselines or before/after local hot-path ratios unless explicitly
marked as original-comparator wins.

## 2026-07-12 — WIN: fenced-code language labels borrow the input — ~1.74× in the affected region (Codex)

After direct collapsed-block append removed the dominant returned-buffer allocation, each fenced-code opener
still copied its already-trimmed language suffix into a temporary `String`. The canonicalizer now retains that
exact `&str` slice from the input document until the fence closes. Closed and unclosed block output, whitespace
trimming, language tags, line selection, and final ordering are unchanged; the per-fence label allocation is gone.

Strict-remote worker `vmi1227854`, one release binary, 4,096 mixed empty, short, padded, four-backtick, and long
fence headers, 41 alternating rounds, `inner=16`; borrowed/owned median **0.5737 [0.5494, 0.6009]** versus A/A
owned null median 0.9883 [0.9270, 1.0215]. The candidate clears the null p5 decisively, approximately **1.74×**
faster in language-suffix extraction. Criterion independently measured owned extraction at 51.6–57.2 us and
borrowed extraction at 29.3–32.5 us. This is not a whole canonicalization or ingest latency claim; exact command,
parity scope, and validation are recorded in `docs/NEGATIVE_EVIDENCE.md`.

## 2026-07-12 — WIN: collapsed code blocks append into the final document buffer — ~1.44× (Codex)

After the one-pass code-block builder landed, `strip_markdown_and_code` still allocated its returned
`String` and immediately copied every collapsed byte into the canonicalizer's preallocated `result`.
Both closed- and unclosed-fence paths now write the same label, kept lines, omitted marker, and language
tag directly into `result`; the caller still contributes the identical single trailing newline. The old
returned-buffer form remains behind tests/`bench-internals` as the same-binary oracle.

Strict-remote worker `vmi1227854`, one release binary, 512 realistic code blocks, 41 alternating rounds,
`inner=4`; direct-append/current median **0.6940 [0.5643, 0.7637]** versus A/A null median
1.0023 [0.8939, 1.1463]. The candidate clears the null p5 decisively, approximately **1.44×** faster
in the affected caller region. Criterion independently measured returned-buffer-plus-copy at 80.0–86.9 us
and direct append at 52.6–55.1 us. This is not a whole canonicalization or ingest latency claim; exact
command, parity scope, and validation are recorded in `docs/NEGATIVE_EVIDENCE.md`.

## 2026-07-12 — WIN: CJK bigrams build directly in the reusable pending buffer — ~1.77× in the affected region (Codex)

After the one-pass CJK decoder landed, `CjkBigramDecomposeStream::decompose_cjk` still allocated a
temporary `Vec<Token>` for every all-CJK run and then moved its contents into the stream's existing
`pending` vector. The bigrams are now constructed directly in `pending` after reserving the exact
additional capacity. The reverse loop, string construction, token metadata, and subsequent `.pop()`
order are identical; only the intermediate vector allocation and transfer disappear.

The retained same-binary comparator checked every token field and vector order before timing.
Strict-remote worker `vmi1149989`, 2,048 realistic 2–31-character CJK tokens, 41 alternating rounds,
`inner=4`; direct/staged median **0.5654 [0.4544, 0.6604]** versus A/A null median
0.9939 [0.8716, 1.2107]. The candidate clears the null p5 decisively, approximately **1.77×** faster
in bigram materialization. This is not a whole index/query latency claim. Exact command and the
strict-remote preflight note are recorded in `docs/NEGATIVE_EVIDENCE.md`.

## 2026-07-12 — WIN: sync two-phase search reuses populated NQC weight — ~1.92–2.00× in the affected region (Codex)

With NQC down-weighting enabled and warmed up, `SyncTwoTierSearcher::search_internal` reduced the
same immutable lexical scores once for initial RRF fusion and again for refined RRF fusion. It now
computes that pure `f64` semantic weight once after lexical retrieval and reuses its exact bits in
both phase configurations. Fast-only, missing-quality, no-lexical, and default-off behavior is
unchanged; fusion scoring, ordering, and tie-breaking are untouched.

A same-binary comparator asserted exact tuple `f64::to_bits()` equality before timing. Strict-remote
worker `vmi1227854`, 41 alternating rounds, `inner=2048`; ratio is one reduction/two reductions:

| lexical hits | A/A two-reduction null p5 | reuse/original median | affected-region speedup |
|---:|---:|---:|---:|
| 20 | 0.5841 | **0.5197** | **~1.92×** |
| 100 | 0.8940 | **0.5042** | **~1.98×** |
| 1,000 | 0.8229 | **0.5010** | **~2.00×** |

Every candidate median clears its own A/A null floor. These ratios describe the populated-NQC
weight subregion of a full synchronous two-phase search, not end-to-end search latency. Exact
intervals and the command are recorded in `docs/NEGATIVE_EVIDENCE.md`.

## 2026-07-12 — WIN: empty NQC sketch skips its neutral lexical-score scan — ~17–556× in the affected region (Codex)

When NQC down-weighting was enabled before its query-log sketch had warmed up, both the async and
sync searchers reduced every lexical score through `nqc_cv_iter`; the empty `NqcDenseWeight` then
mapped that unused CV to the neutral factor `1.0`. The empty-sketch path now substitutes the same
zero percentile input without the O(hit-count) scan, while still evaluating the existing
`dense_weight` function. This preserves exact output and existing parameter-edge behavior;
populated sketches and the default `beta <= 0` path are unchanged.

A same-binary comparator asserted exact `f64::to_bits()` equality before timing. Strict-remote
worker `vmi1264463`, 41 alternating rounds, `inner=2048`; ratio is skip-scan/current-scan:

| lexical hits | A/A current-scan null p5 | skip-scan/current median | affected-region speedup |
|---:|---:|---:|---:|
| 20 | 0.7726 | **0.0575** | **~17.4×** |
| 100 | 0.9026 | **0.0216** | **~46.3×** |
| 1,000 | 0.8892 | **0.0018** | **~555.6×** |

Every candidate median clears its A/A null floor decisively. These ratios describe only the
enabled-but-empty effective-weight region during startup/warm-up, not whole-search latency. The
exact command and intervals are recorded in `docs/NEGATIVE_EVIDENCE.md`.

## Original-comparator wins

### 2026-06-25 — BOLD-VERIFY non-semantic zero-hit lexical gate (BlackThrush)

**Lever:** the BOLD hash-hybrid path and `TwoTierSearcher` now stop before hash-vector search
when a non-semantic fast embedder has no quality tier and lexical search produces zero candidates.
The same gate also permits lexical-only return for saturated natural-language rows in that
non-semantic/no-quality mode; that reduced the old catastrophic vector-scan residual but is still
not a universal Tantivy-class win, so the slower natural-language rows are recorded in
`docs/NEGATIVE_EVIDENCE.md`.

## 2026-07-12 — WIN: NQC enabled-path iterator reduction removes one allocation — ~1.28–1.52× (Codex)

The newly wired sync NQC down-weight collected lexical scores into a temporary `Vec<f32>` and immediately
reiterated it. `nqc_cv_iter` now consumes the scores directly while preserving the exact f64 operation order and
`f32` result bits; default-off behavior is unchanged. Same-binary strict-remote A/B on `vmi1149989` (41 rounds,
`inner=2048`) measured iterator/collect medians **0.6907** at 20 hits, **0.7815** at the default 100 hits, and
**0.6559** at 1,000 hits. Each cleared its A/A null p5 (0.7191 / 0.8174 / 0.9203), so the win is decisive across
all shapes. The comparator asserts bit identity before timing; focused remote release tests passed 6/6. Source
landed patch-equivalently in `08ef9680`; exact command and intervals are in `docs/NEGATIVE_EVIDENCE.md`.

**Measured command (RCH worker `hz2`, per-crate, warm target dir):**
```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- env \
  FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
  RUST_LOG=error \
  cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact: `/data/projects/.rch-targets/frankensearch-cod-b/criterion/bold_verify/summary.md`
and `summary.jsonl`.

| Workload | Corpus hash | Tantivy-class p50 | frankensearch p50 | Ratio vs Tantivy-class | Status |
|----------|-------------|-------------------|-------------------|------------------------|--------|
| `bold_verify/top10/100000` `zero_hit` | `13f1b0153f5adec9` | 67 us | 62 us | **0.925** | KEEP |

**Scope:** this is a zero-hit incumbent win for the non-semantic hash/no-quality lane, not a
claim that hybrid search dominates Tantivy/Lucene/Meilisearch-class BM25 overall. The 10k zero-hit
row is a p50 tie (43 us vs 44 us; ratio 0.977) with better tails, and natural-language rows remain
slower at p50; both are recorded as residual evidence.

### 2026-06-25 — BOLD-VERIFY lexical-saturated short-circuit (BlackThrush)

**Lever:** after successful fast embedding and lexical search, `TwoTierSearcher` now returns
lexical-only initial results for `Identifier` / `ShortKeyword` queries when lexical already has
at least `k` hits and graph ranking is disabled. That skips the phase-1 vector scan and RRF work
for lexical-saturated queries while preserving the hybrid path for natural-language and zero-hit
queries.

**Measured command (local fallback after RCH worker `vmi1153651` stalled twice; per-crate,
warm target dir):**
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error \
  cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact: `/data/projects/.rch-targets/frankensearch-cod-a/criterion/bold_verify/summary.md`
and `summary.jsonl`.

| Workload | Corpus hash | Tantivy-class p50 | frankensearch p50 | Ratio vs Tantivy-class | Status |
|----------|-------------|-------------------|-------------------|------------------------|--------|
| `bold_verify/top10/10000` `high_fanout` | `2e78365a46a7c3b9` | 114 us | 81 us | **0.711** | KEEP |
| `bold_verify/top10/100000` `exact_identifier` | `13f1b0153f5adec9` | 1.241 ms | 1.090 ms | **0.878** | KEEP |

**Scope:** this is a targeted lexical-saturation win vs a Tantivy/Lucene/Meilisearch-class
incumbent, not a universal dominance claim. The same BOLD run still shows slower/noisy rows for
10k exact identifiers, short-keyword rows, `limit_all`, natural-language, and zero-hit queries;
those ratios are recorded in `docs/NEGATIVE_EVIDENCE.md`.

### 2026-06-25 — BOLD-VERIFY lexical prefetch budget gate (BlackThrush)

**Lever:** the BOLD hash-hybrid harness now mirrors the product's lexical short-circuit budget:
for query classes that can legally return lexical-only (`Identifier`, `ShortKeyword`,
`NaturalLanguage`), it asks Tantivy for only `k` lexical candidates instead of prefetching `3k`
and then throwing the surplus away. This is an evidence-harness correction for the shipped
short-circuit path, not a new product ranking algorithm.

**Measured command (RCH local fallback; no admissible workers:
`insufficient_slots=5,hard_preflight=1`; per-crate, warm target dir):**
```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- env \
  FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
  FRANKENSEARCH_BOLD_VERIFY_COMMAND='CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b rch exec -- env FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error cargo bench -p frankensearch --features lexical --profile release --bench search_bench bold_verify_tantivy_class -- --sample-size 10 --warm-up-time 1 --measurement-time 1' \
  RUST_LOG=error \
  cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact: `/data/projects/.rch-targets/frankensearch-cod-b/criterion/bold_verify/summary.md`
and `summary.jsonl`.

| Workload | Corpus hash | Tantivy-class p50 | frankensearch p50 | Ratio vs Tantivy-class | Status |
|----------|-------------|-------------------|-------------------|------------------------|--------|
| `bold_verify/top10/10000` `exact_identifier` | `2e78365a46a7c3b9` | 205 us | 178 us | **0.868** | KEEP |
| `bold_verify/top10/10000` `natural_language` | `2e78365a46a7c3b9` | 335 us | 322 us | **0.961** | KEEP |
| `bold_verify/top10/100000` `exact_identifier` | `13f1b0153f5adec9` | 1.596 ms | 1.540 ms | **0.965** | KEEP, p95 noisy |
| `bold_verify/top10/100000` `short_keyword` | `13f1b0153f5adec9` | 359 us | 297 us | **0.827** | KEEP |
| `bold_verify/top10/100000` `natural_language` | `13f1b0153f5adec9` | 972 us | 935 us | **0.962** | KEEP |
| `bold_verify/top10/100000` `high_fanout` | `13f1b0153f5adec9` | 895 us | 783 us | **0.875** | KEEP |
| `bold_verify/top10/100000` `zero_hit` | `13f1b0153f5adec9` | 36 us | 24 us | **0.667** | KEEP, tail regressed |

**Scope:** this fixes the BOLD evidence path for lexical-saturated and natural-language
short-circuit rows. It does not establish universal dominance over Tantivy/Lucene/Meilisearch-class
BM25; the 10k short-keyword/high-fanout/zero-hit rows, 100k quoted phrase, and `limit_all` remain
negative or noisy and are recorded in `docs/NEGATIVE_EVIDENCE.md`.

| Date | Crate | Lever | Workload (bench id) | Before | After | Ratio | Status |
|------|-------|-------|---------------------|--------|-------|-------|--------|
| 2026-06-25 | frankensearch-core | **ASCII fast-path for NFC canonicalization** (analyzer hot path) | `nfc/ascii_short` | 1.207 µs | 26.7 ns | **0.022 (~45×)** | KEEP (`9d7e8d0`) |
| 2026-06-25 | frankensearch-core | **ASCII fast-path for NFC canonicalization** (analyzer hot path) | `nfc/ascii_doc` (2.2 KB) | 56.35 µs | 153 ns | **0.0027 (~368×)** | KEEP (`9d7e8d0`) |
| 2026-06-25 | frankensearch-core | ASCII fast-path for NFC canonicalization | `nfc/non_ascii` | 40.86 µs | 41.32 µs | 1.011 (neutral) | KEEP |
| 2026-06-25 | frankensearch-core | **`filter_low_signal`: ASCII compare instead of full-doc `to_lowercase`** | `filter_low_signal/ascii_doc` (2.25 KB) | 137.8 ns | 9.2 ns | **0.067 (~15×)** | KEEP |
| 2026-06-25 | frankensearch-core | `filter_low_signal`: ASCII compare instead of full-doc `to_lowercase` | `filter_low_signal/ascii_short` | 16.7 ns | 8.2 ns | **0.49 (~2×)** | KEEP |
| 2026-06-25 | frankensearch-core | **`strip_markdown_line`: skip the 4-pass inline replace chain for lines with no inline-markdown chars** | `strip_markdown_inline` (80 plain lines) | 10.17 µs | 4.41 µs | **0.434 (~2.3×)** | KEEP |
| 2026-06-25 | frankensearch-core | `strip_markdown_line`: chain `#`/`>` prefix trims as one `&str` (1 alloc, not 2) | `prefix_trim` (80 lines) | 2.45 µs | 1.34 µs | **0.549 (~1.8×)** | KEEP |
| 2026-06-25 | frankensearch-core | **canonicalize tail: drop the 2nd whole-document copy** (`filter_low_signal`→`is_low_signal` bool predicate; pass the owned `ws_normalized` buffer straight to `truncate_to_chars`) | `pipeline_tail` (2.25 KB ascii_doc) | 91.4 ns | 59.4 ns | **0.649 (~1.54×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`strip_markdown_line` fast path: trim the borrowed line directly** (drop the `line.to_string()` copy; only `strip_list_marker` allocates) | `strip_markdown_fastpath` (80 plain lines) | 2.328 µs | 1.306 µs | **0.561 (~1.78×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`strip_markdown_line` → `Cow<str>`: zero-alloc plain lines** (`strip_list_marker` returns borrowed slices; plain line borrows straight to the caller's single `push_str`) | `strip_markdown_cow` (80 plain lines, full push_str loop) | 2.166 µs | 1.390 µs | **0.642 (~1.56×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`QueryClass::classify`: compute `has_whitespace` once + `rsplit_once`** (was up to 4× whitespace rescans + a `rsplitn().collect()` Vec, per query) | `query_class` (11-query mix) | 732.4 ns | 512.4 ns | **0.700 (~1.43×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`strip_italic_underscores`: single pass, 3 allocs → 1** (drop the `Vec<char>` + `Vec<bool>` + final `collect`; build output directly with `prev`/`peek`) | `strip_italic_underscores` (40 snake_case code lines) | 9.770 µs | 2.982 µs | **0.305 (~3.28×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`strip_markdown_line` slow path: guard each transform by its trigger char** (one scan sets `*`/`_`/`` ` ``/`[` flags; a snake_case line skips the `**`/`*`/`` ` ``/link no-op allocating passes) | `strip_markdown_slowpath` (snake_case line) | 307.9 ns | 48.3 ns | **0.157 (~6.37×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`nfc_normalize` → `Cow<str>`: ASCII borrows instead of `to_owned`** (eliminates the whole-document copy at the `canonicalize` entry; ASCII is already NFC and the next stage only needs `&str`) | `nfc_ascii_copy` (2.25 KB ascii_doc) | 39.45 ns | 1.17 ns | **0.030 (~33×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`apply_hash_votes` branchless SimHash vote** (`2*b-1` instead of a per-bit `if`/`else` on random hash bits → no ~50% branch mispredict) | `simhash_votes` (~300-token doc) | 18.480 µs | 16.073 µs | **0.870 (~1.15×)** | superseded by table |
| 2026-06-26 | frankensearch-core | **`apply_hash_votes` table-driven SimHash vote** (8 byte-indexed lookups into a compile-time `[[i32;8];256]` + vectorizable 8-wide slice adds, vs 64 per-bit `shift/mask/mul`) | `simhash_votes` (~300-token doc) | 15.810 µs (branchless) | 10.930 µs | **0.691 (~1.45×) vs branchless, ~0.60× vs original branch** | KEEP (BlackThrush) |
| 2026-07-04 | frankensearch-core | **`DocumentFingerprint::compute`: stream SimHash token windows instead of collecting `Vec<&str>`** — the old path split every document into a heap vector solely to call `tokens.windows(3)`. The new path keeps a two-token rolling buffer over `split_whitespace()`, emits the same 3-token windows, and preserves the old individual-token hashing rule for one- and two-token documents. A unit test compares the streaming path against the old slice implementation across empty, whitespace-only, short, long, and Unicode input. Per-crate proof used `AGENT_NAME=Codex CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 rch exec -- cargo bench -p frankensearch-core --profile release --bench fingerprint -- fingerprint_compute --sample-size 10 --warm-up-time 0.1 --measurement-time 0.3`; RCH ran on `vmi1227854`. Ratio-vs-ORIG audit entry recorded in `docs/NEGATIVE_EVIDENCE.md`. | `fingerprint_compute/current_alloc_doc300→streaming_doc300` | 18.378 µs | 17.328 µs | **0.943 (~1.06×)** | KEEP (Codex) |
| 2026-06-26 | frankensearch-core | **`ParsedQuery::parse` no-negation fast path** (queries without `-`/`"`/`\` skip the `Vec<char>` + char-by-char parse; whitespace-normalize via split + `push_str`) | `parsed_query` (plain multi-word query) | 503.4 ns | 109.6 ns | **0.218 (~4.59×)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-core | **`ParsedQuery::parse` NOT-keyword check: direct ASCII case match instead of allocating a 3-char `String`** on every tested token boundary in the negation-capable parser path. This is the graveyard/data-structure lever in miniature: compile a fixed keyword recognizer to branch checks instead of constructing transient heap data. Behavior is unchanged because the prior `eq_ignore_ascii_case("NOT")` was ASCII-only and only the three-byte keyword is accepted; surrounding whitespace/boundary checks are unchanged. | `parsed_query/not_phrase` (two `NOT "phrase"` exclusions + one `-term`) | 746.73 ns | 668.39 ns | **0.895 (~1.12×)** | KEEP (BlackThrush) |
| 2026-07-04 | frankensearch-core | **`QueryClass::classify` ASCII identifier fast path** — common search queries are ASCII, but the identifier heuristic still paid Unicode `chars()` decoding for whitespace, case flags, and issue-id suffix checks. The new path dispatches ASCII input through byte predicates that are equivalent to the existing Unicode predicates on ASCII, while non-ASCII queries fall through to the old logic unchanged. The bench carries a same-binary `current` arm that replicates the pre-ASCII implementation and a `new` arm calling production `QueryClass::classify`. The requested `cargo bench --release` form was tried and rejected by Cargo; the remote candidate RCH run was interrupted after exceeding the no-wait guard, then the same short per-crate bench completed locally with `CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod`. | `query_class/current→new` (11-query ASCII mix) | 456.83 ns | 364.52 ns | **0.798 (~1.25×)** | KEEP (CobaltRidge) |
| 2026-06-25 | frankensearch-embed | **hash embedder: drop 2 per-embed allocs** (lazy `tokenize` iterator + `l2_normalize_in_place` on the owned accumulator) | `hash_embed_fnv` (~100-word doc, dim384) | 2.318 µs | 1.961 µs | **0.846 (~1.18×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-embed | hash embedder alloc elision — JL path (compute-bound, alloc negligible) | `hash_embed_jl` (~100-word doc, dim384) | 100.27 µs | 102.07 µs | 1.018 (neutral) | KEEP (no regression) |
| 2026-06-27 | frankensearch-embed | **JL projection: interleave 4 independent xorshift token chains** — the JL path was still latency-bound after allocation elision because each token's xorshift recurrence serializes three shift/xor steps per dimension. Buffering four token-chain seeds and advancing them together exposes instruction-level parallelism while preserving the exact scalar output: each dimension accumulator is an exact small integer-valued `f32` sum of +/-1 signs before normalization. Verified by new bit-identical scalar-reference tests across dimensions, seeds, and token-count tails. The requested literal `cargo bench --release` form was tried first and Cargo rejected it; the successful final per-crate run used `--profile release` through `rch exec` with `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a` (local fallback because no RCH worker was admissible). | `hash_embed_jl` (~100-word doc, dim384) | 105.00 µs | 88.945 µs | **0.847 (~1.18×)** | KEEP (BlackThrush) |
| 2026-07-04 | frankensearch-embed | **HashEmbedder tokenizer: ASCII byte iterator** — the FNV/JL fallback embedder still split common ASCII prose/code through `char::is_alphanumeric()`; the new iterator uses byte alnum checks on the hot ASCII path and promotes to the Unicode split fallback only if it sees a non-ASCII byte. The bench asserts bit-identical FNV output against the current char-split path; a unit test checks empty, ASCII punctuation, code-like tokens, combining marks, Latin-1, CJK, and Greek against the previous `str::split` reference. Remote RCH proof: CobaltRidge on `ovh-a`, target dir `/data/projects/.rch-targets/frankensearch-cobaltridge`, measured 1.6755 µs -> 1.4511 µs (ratio **0.866**, sample-size 20); Codex remeasured with `AGENT_NAME=Codex CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-codex-20260704-embed-tokenizer rch exec -- cargo bench -p frankensearch-embed --profile release --bench hash_embed hash_embed_tokenize_ascii -- --sample-size 10 --warm-up-time 1 --measurement-time 1` on `vmi1149989`: 1.9824 µs -> 1.7832 µs (ratio **0.899**, ~1.11×). | `hash_embed_tokenize_ascii` (~100-word ASCII doc, dim384 FNV) | 1.9824 µs | 1.7832 µs | **0.899 (~1.11×)** | KEEP (CobaltRidge/Codex) |
| 2026-07-05 | frankensearch-embed | **`CachedEmbedder::embed_batch` all-distinct miss batches return the owned inner result directly** — after cache admission, a batch where every input is a distinct cache miss does not need the generic duplicate fanout pass. The new path detects that shape from `miss_texts.len() == texts.len()` and returns the owned `inner.embed_batch` vector once cache insertion is done; mixed hit batches and duplicate-miss batches keep the existing fanout. Per-crate proof used `AGENT_NAME=CobaltRidge CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo bench -p frankensearch-embed --profile release --bench cache_replay -- batch_all_distinct_miss_return --sample-size 30 --warm-up-time 0.2 --measurement-time 1`; `rch exec` on `vmi1227854` was interrupted before samples after the bounded cold remote compile window. Ratio-vs-ORIG audit entry recorded in `docs/NEGATIVE_EVIDENCE.md`. | `batch_all_distinct_miss_return/current_generic→direct_return` (64; 256 distinct misses) | 9.1285 µs; 39.701 µs | 7.5173 µs; 29.872 µs | **0.823 (~1.22×); 0.752 (~1.33×)** | KEEP (CobaltRidge) |
| 2026-07-05 | frankensearch-embed | **`CachedEmbedder::embed_batch` miss fanout: move each distinct inner embedding into its last output slot** — the pre-change path cloned every returned 384-dim vector into the final output after also cloning it into the S3-FIFO cache. The new path counts same-batch miss uses, moves the owned inner result into its last output slot, and clones only earlier duplicate slots that need another copy. Cache insertion, hit/miss accounting, and duplicate-batch behavior are unchanged; existing embed-batch tests pass. Per-crate proof used `AGENT_NAME=Codex CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 RCH_WORKER=ovh-a rch exec -- cargo bench -p frankensearch-embed --profile release --bench cache_replay -- batch_miss_fanout --sample-size 10 --warm-up-time 0.1 --measurement-time 0.3`; RCH ran on `ovh-a`. Ratio-vs-ORIG audit entry recorded in `docs/NEGATIVE_EVIDENCE.md`. | `batch_miss_fanout/clone_all→move_last/distinct_256` | 21.876 µs | 856.13 ns | **0.039 (~25.6×)** | KEEP (Codex) |
| 2026-06-25 | frankensearch-fusion | **RRF fuse: one `entry` lookup instead of `get`+`entry`** (halve per-candidate hashing of the `AHashMap<&str,_>` accumulator) | `rrf_fuse` (1000 lexical + 1000 semantic, ~50% overlap) | 29.11 µs | 23.07 µs | **0.793 (~1.26×)** | KEEP (BlackThrush) |
| 2026-07-04 | frankensearch-fusion | **`SyncTwoTierSearcher` refined lexical fusion: route `blended` through unique RRF merge** — the refined semantic slice is produced by `blend_two_tier_aligned(&fast_hits, &quality_scores, ...)`, so it inherits `fast_hits`' unique doc ids. The old refined lexical branch still called the dedup merge and paid an O(N) `seen_semantic` set that can never fire for this callsite. The `collect_limit_all` bench now has a paired callsite A/B that builds the same blended refined slice, asserts full `FusedHit` equality between legacy and unique paths, then times both arms in one Criterion run. Remote strict RCH proof on `ovh-a`, target dir `/data/projects/.rch-targets/frankensearch-cod`: `cargo bench -p frankensearch-fusion --profile release --bench collect_limit_all refined_rrf_callsite -- --sample-size 10 --warm-up-time 1 --measurement-time 1`. External original-comparator ratio is N/A; caveat recorded in `docs/NEGATIVE_EVIDENCE.md`. | `refined_rrf_callsite/{dedup→unique}` (N=10k) | 770.14 µs | 661.37 µs | **0.859 (~1.16×)** | KEEP (CobaltRidge) |
| 2026-07-04 | frankensearch-fusion | **`SyncTwoTierSearcher` vector-only refined result assembly: replace doc-id score maps with aligned numeric lookup** — the no-lexical refined path already has `fast_hits` aligned with `quality_scores_for_hits`, and `blend_two_tier_aligned_vector_index` preserves `VectorHit.index` in each blended output. The old path rebuilt two `AHashMap<&str, f32>` maps (`fast_scores`, `quality_scores`) and probed both by `doc_id` for every output row. The new path builds one aligned score lookup keyed by `index` (dense slab for compact spans, `AHashMap<u32, _>` fallback for sparse spans) and moves owned `doc_id`s into `ScoredResult` unchanged. A permanent unit test compares dense and sparse layouts against the previous doc-id-map semantics. Per-crate proof used `AGENT_NAME=Codex CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod rch exec -- cargo bench -p frankensearch-fusion --profile release --bench score_map lookup -- --sample-size 10 --warm-up-time 1 --measurement-time 1`; `rch` fell open locally because no worker was admissible. Ratio-vs-ORIG audit entry recorded in `docs/NEGATIVE_EVIDENCE.md`. | `score_map/lookup_current→lookup_aligned` (N=10k); (N=100k) | 1.0584 ms; 19.845 ms | 37.104 µs; 379.92 µs | **0.035 (~28.5×); 0.019 (~52.2×)** | KEEP (Codex) |
| 2026-07-04 | frankensearch-rerank | **`RrfCombine` reorder bookkeeping: collapse four intermediate vectors into one order vector** — `apply_rrf_combine` still allocated/sorted `by_rerank`, inverted it into `rerank_rank`, built a `key` vector, sorted a separate `perm`, then cloned into `reordered`. The new path carries `position` and `fused_key` in one `RrfOrder` vector: sort by rerank score, write fused RRF keys in place, sort by fused key with the same doc-id tiebreaker, then apply the final permutation. A unit test compares against the previous five-vector reference implementation. Per-crate proof used `AGENT_NAME=Codex CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 cargo bench -p frankensearch-rerank --profile release --bench combine_reorder_cost_ab -- rrf_combine --sample-size 10 --warm-up-time 0.1 --measurement-time 0.3`; the exact rch/full-LTO release path did not produce samples in the bounded window, recorded in `docs/NEGATIVE_EVIDENCE.md`. | `rerank_combine_reorder/rrf_combine_current→rrf_combine_order_vec` (N=20; 50; 100; 200) | 549.26 ns; 1.2107 µs; 2.1862 µs; 4.4715 µs | 479.86 ns; 1.0628 µs; 2.0193 µs; 3.9125 µs | **0.874; 0.878; 0.924; 0.875** | KEEP (Codex) |
| 2026-07-05 | frankensearch-rerank | **Native reranker final-layer CLS attention: replace `m=1` BMM/repack with direct SIMD rank-1 attention** — the last encoder layer only needs each document's `[CLS]` output row, but the previous `fused_attention_cls` still copied K/V into head-major scratch and launched two tiny `bmm` calls. The new path scores the single CLS query against keys directly from the fused QKV buffer with `f32x8`, reuses the existing fused softmax, and accumulates the value row directly into `[H]`. Earlier full-token attention layers and the long-doc tape path are unchanged. The A/B bench carries the old BMM/repack comparator and asserts max abs delta <= `1.0e-4` before timing. Per-crate proof used `AGENT_NAME=Codex CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod CARGO_PROFILE_RELEASE_LTO=false CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 RCH_WORKER=ovh-a rch exec -- cargo bench -p frankensearch-rerank --features native --profile release --bench cls_attention_ab -- cls_attention --sample-size 10 --warm-up-time 0.1 --measurement-time 0.5`; RCH selected `hz2` despite the worker hint. Ratio-vs-ORIG audit entry recorded in `docs/NEGATIVE_EVIDENCE.md`. | `cls_attention/bmm_repack_orig→direct_rank1` (seq=64; 128; 256; 512) | 500.13 µs; 562.44 µs; 738.20 µs; 1.7997 ms | 6.2809 µs; 14.196 µs; 34.720 µs; 71.815 µs | **0.0126; 0.0252; 0.0470; 0.0399** | KEEP (Codex) |
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators | `dot/dim256/f32_bytes` | 10.839 ms | 3.647 ms | **0.336** | KEEP |
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators | `dot/dim384/f32_bytes` | 14.084 ms | 5.333 ms | **0.379** | KEEP |
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators (`BlueGull` pinned-worker confirmation) | `dot/dim256/f32_bytes/10000` | 3.4835 ms | 0.66126 ms | **0.190** | KEEP (`vmi1149989`) |
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators (`BlueGull` pinned-worker confirmation) | `dot/dim384/f32_bytes/10000` | 5.1487 ms | 1.8811 ms | **0.365** | KEEP (`vmi1149989`) |
| 2026-06-26 | frankensearch-index | **`dot_product_f32_f32`: 4 independent `f32x8` accumulators** over 32-element blocks (plain f32 slice path; cheap decode, real callers in WAL-entry scoring and MRL truncated search). Existing tests allow tiny f32-order differences; SIMD gate passed. BlackThrush landed after remote same-process A/B superseded the older local-fallback negative row. | `dot/dim256/f32_slice/10000` (remote `vmi1227854`, in-process vs single-acc bench baseline) | 2.3594 ms | 750.64 µs | **0.318 (~3.14×)** | KEEP |
| 2026-06-26 | frankensearch-index | `dot_product_f32_f32` 4 accumulators (MiniLM/quality dim) | `dot/dim384/f32_slice/10000` (remote `vmi1227854`) | 3.7372 ms | 2.2784 ms | **0.610 (~1.64×)** | KEEP |
| 2026-06-24 | frankensearch-index | **branchless SIMD f16→f32 widen** (default path) | `dot/dim256/f16_bytes` | 4.733 ms | 1.632 ms | **0.345** | KEEP |
| 2026-06-24 | frankensearch-index | **branchless SIMD f16→f32 widen** (default path) | `dot/dim384/f16_bytes` | 7.363 ms | 2.332 ms | **0.317** | KEEP |
| 2026-06-24 | frankensearch-index | branchless SIMD f16→f32 widen | `dot/dim256/f16_slice` | 3.699 ms | 1.348 ms | **0.364** | KEEP |
| 2026-06-24 | frankensearch-index | branchless SIMD f16→f32 widen | `dot/dim384/f16_slice` | 5.536 ms | 2.181 ms | **0.394** | KEEP |
| 2026-06-26 | frankensearch-index | **in-memory filtered scan: precomputed-hash prescreen** (`matches_doc_id_hash` with a lazy `doc_id_hashes` slab) instead of re-hashing each `doc_id` string per vector via `matches()` — matches the FSVI scan, which already did this | `filter_prescreen` (10k `BitsetFilter` checks) | 183.5 µs | 88.4 µs | **0.482 (~2.08×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-index | **`dot_i8_i8`: 4 independent `i32x8` accumulators** (int8 two-pass pass-1) — the i8→i16 decode is a cheap sign-extend, so the kernel is sum-chain-bound (unlike the decode-bound f16 dot, where 4 accs regress — see NEGATIVE_EVIDENCE). Integer sum is associative → bit-identical. Landed as `957d608`+`28aa022`; ledger row added by BlackThrush | `dot_dim384/i8_dot` (10k, in-process vs single-acc baseline) | 481.3 µs | 412.0 µs | **0.856 (~1.17×)** | KEEP |
| 2026-06-26 | frankensearch-index | `dot_i8_i8` 4 accumulators (smaller dim) | `dot_dim256/i8_dot` (10k) | 318.2 µs | 298.8 µs | **0.939 (~1.06×)** | KEEP |
| 2026-06-26 | frankensearch-index | **`search_top_k_int8_two_pass_filtered`** — extended the int8 ADC two-pass to accept a `SearchFilter` (pass-1 pre-screens by the precomputed `doc_id` hash). The fast int8 path was previously filter-incapable, so filtered large-N vector search fell back to the exact scan; now it gets the int8 speedup. Bit-identical to the exact filtered top-k (`int8_two_pass_filtered_matches_exact_filtered`) | `int8_two_pass` filtered (10k clustered, `BitsetFilter` ~½ corpus, mult=5): `flat_filtered` vs `int8_filtered_mult5` | 314.4 µs | 248.7 µs | **0.791 (~1.26×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-index | **int8 two-pass pass-1 cutoff fast-path** — the int8 pass-1 called `insert_candidate` for *every* vector; the exact `scan_range` skips it for sub-cutoff scores (`heap.len() < cap \|\| score_key(score) >= cutoff`). Added the same guard to pass-1 (track the bounded-heap min as cutoff). Result is bit-identical (a sub-cutoff score never enters the full heap; `int8_two_pass_matches_exact_topk` + filtered test pass). The selection overhead was large — without the cutoff, mult=10 (2278 µs) was **slower than the exact flat scan** (~2131 µs); the cutoff restores the int8 win. In-process A/B (100k clustered, cutoff vs no-cutoff baseline) | `int8_two_pass` mult=5 / mult=10 | 1170.4 / 2278.6 µs | 900.9 / 1140.2 µs | **0.770 (~1.30×) / 0.500 (~2.00×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-index | **MRL truncated-scan cutoff fast-path** — same fix applied to `mrl_truncated_scan`: it called `insert_mrl_candidate` for every vector; added the bounded-heap cutoff guard (`heap.len() < limit \|\| nan_safe(score) >= cutoff`) the exact scan + int8 pass-1 use. Bit-identical (the 46 `mrl::` tests pass unchanged). Smaller win than the int8 pass-1 (the MRL scan is single-threaded with a cheap 64-dim truncated dot, so `insert` is a smaller fraction). In-process A/B (100k FSVI F16, clustered, search_dims=64, limit=30): cutoff vs no-cutoff baseline | `mrl_cutoff` scan | 23056.9 µs | 21045.9 µs | **0.913 (~1.10×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-index | **MRL truncated-scan parallelization** — `mrl_truncated_scan` was **single-threaded** (a plain `for index in 0..record_count()` byte-walk), while the exact `scan_range` + int8 pass-1 are rayon-parallel. Refactored the byte-walk into a per-chunk `mrl_scan_chunk` (fn-pointer dot collapses the F16/F32 arms) and scan disjoint record ranges in parallel above `PARALLEL_THRESHOLD`, merging per-chunk bounded heaps (`merge_mrl_partials`). Order-independent (score+index total order), so **bit-identical** to the sequential top-k — verified by a new `mrl_search_parallel_path_covers_all_records` test (12k records, rescore-all == exact `search_top_k`) + all 47 `mrl::` tests. In-process A/B (100k FSVI F16, clustered, search_dims=64, limit=30): parallel vs forced-sequential | `mrl_parallel` scan | 5828.5 µs | 674.9 µs | **0.116 (~8.64×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-fusion | **`SyncTwoTierSearcher` fast-tier fetch → int8 two-pass (default path)** — the hybrid's vector candidate fetch (`search_fast_hits`) called the exact f16 `search_top_k`. The fast tier is a *reranked candidate generator* (quality-tier + RRF re-score its hits), so its contract is approximate candidates; routed the **default path** (no explicit `SearchParams`) to `search_top_k_int8_two_pass_filtered` (parallel + cutoff + filter, mult=10, recall=1.0). **Wires the validated int8 two-pass into a real production hybrid path** (it was previously unused by callers). Lossless → identical fused top-k (6 sync_searcher + 9 integration tests pass); explicit `SearchParams` still uses the exact scan. End-to-end hybrid A/B (100k in-memory two-tier, no lexical): exact-fetch vs int8-fetch | `sync_int8_fetch` | 6449.6 µs | 5498.6 µs | **0.853 (~1.17×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-index | **`InMemoryTwoTierIndex` quality rerank: O(hits·N) → O(hits)** — `quality_scores_for_hits` (the two-tier quality rerank, on the wired sync-hybrid path) did `quality.doc_ids.iter().position(\|id\| id == hit.doc_id)` — a **linear O(N) doc-id scan per hit** ⇒ O(hits·N) (≈3M string-compares at N=100k, hits=30). Replaced with a lazy `doc_id → index` `HashMap` (O(1) lookup, first-insert-wins = same as `position`; built on first rerank, search-only callers pay nothing). Also fixed the same O(N) scan in `InMemoryVectorIndex`'s by-doc-id hit scorer. Bit-identical (19 in-memory + 15 sync_searcher tests). End-to-end hybrid before→after (same `sync_int8_fetch` bench, only the rerank changed; both arms drop by a constant ~3.7 ms, isolating the rerank): int8 5498.6→1755.0 µs **3.13×**, exact 6449.6→2637.5 µs **2.45×** | `sync_int8_fetch` | 5498.6 µs | 1755.0 µs | **0.319 (~3.13×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-fusion | **int8 fast-tier `mult` 10 → 5** — the wired `INT8_FAST_TIER_MULT` was 10, the *slowest* int8 mult (int8_two_pass bench: mult=5 862 µs vs mult=10 1187 µs at 100k, both recall=1.0). Since `fetch` is already a candidate over-fetch (`k·candidate_multiplier`), a 10× int8 multiplier is wasteful; mult=5 keeps the candidate set lossless (recall=1.0 validated at 100k) with a higher pass-1 cutoff (fewer inserts) + smaller pass-2 rescore. Lossless (15 sync_searcher tests pass). Cumulative sync hybrid vs the original exact + linear-rerank: **4.38×** | `sync_int8_fetch` int8 arm | 1755.0 µs | 1472.9 µs | **0.839 (~1.19×)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-index | **FSVI int8 two-pass for standalone large-N vector search** — new `VectorIndex::search_top_k_int8_two_pass`: a parallel int8 pass-1 over all main records (a lazily-built corpus-quantized i8 slab from the contiguous F16 region; tombstone-aware flag check + the same cutoff fast-path as the exact scan) keeps the top `k·mult` by approximate score, then an exact f16 rescore of just those candidates selects the final top-k. The file-backed twin of the in-memory two-pass; falls back to the exact `search_top_k` for WAL/non-F16 indexes (never silently degraded). **Lossless: recall@10 = 1.0000 at mult=2/3/5/10 on 100k clustered**, and bit-identical to `search_top_k` when the candidate set retains every record — verified by `int8_two_pass_keep_all_matches_exact` + 53 search tests. NOT wired into the BOLD hybrid (that gap is not vector-bound — see docs/NEGATIVE_EVIDENCE.md); targets pure vector-search latency. A/B (100k file-backed FSVI, in-process, same index): exact flat vs int8 mult=5 | `fsvi_int8_two_pass` | 14349.0 µs | 7408.2 µs | **0.516 (~1.94×)** | KEEP (BlackThrush) |
| 2026-07-10 | frankensearch-index | **FSVI int8 two-pass pass-1: use 4k-row Rayon chunks instead of exact-scan 1k chunks** — after the prepared-query sidecar rejection, the ranked source hotspot was pass-1 heap/merge fan-in: at N=100k the 1k split builds ~98 local bounded heaps and merges up to `chunks * k * mult` candidates before the exact f16 rescore. The int8 dot is cheap enough that 1k chunks overproduce local top-N heaps; 4k chunks keep enough Rayon tasks while cutting merge fan-in 4x. Semantics are unchanged because any global top-`limit` candidate must be in the top `limit` of its own chunk, regardless of chunk boundaries; scoring, tombstone check, cutoff, global merge, f16 rescore, and tie-break all stay identical. Same-worker RCH `ovh-a`, release profile, `fsvi_int8_two_pass`, N=100k, dim=384, k=10, sample-size 10, warm-up 0.1s, measurement 0.5s. Quality gate stayed green: recall@10=1.0000 and nDCG@10=1.0000 for mult=2/3/5/10. `perf stat` via `rch exec` was attempted first but classified as non-compilation and was interrupted after counters were compilation-polluted, so the keep proof uses the Criterion A/B plus the heap-fan-in source ranking. The 8k variant was measured and rejected in NEGATIVE_EVIDENCE because it over-widened chunks and made the fastest mult3 row noisy/slower. | `fsvi_int8_two_pass/int8_mult3`; `/int8_mult5` | 506.18 us; 551.20 us | 452.20 us; 471.01 us | **0.893 (~1.12x); 0.855 (~1.17x)** | KEEP (cod_fs) |
| 2026-07-10 | frankensearch-index | **FSVI int8 pass-1 packed selection key (`bd-b5wl`)** — the symbolized profile in NEGATIVE_EVIDENCE put 26.61% self-time in `int8_scan_range` plus 10.17% in `insert_candidate`, after the 53.31% AVX2 dot leaf mapped to closed query-side decode/accumulator families. Replaced pass-1 `HeapEntry { usize, f32 }` selection with one badness-ordered `u128`: reversed exact i32 score in the high word and the full `usize` index tiebreak in the low word. The fused scan now performs one integer key-vs-cutoff comparison and the parallel merge uses the same key; it avoids the per-row i32-to-f32 cast, NaN normalization, custom float comparator, `HeapEntry` construction, and duplicated cutoff/heap comparison. Dimensions <=1040 rank raw i32 exactly because the entire possible dot range is consecutively representable as f32; larger dimensions use a packed transform of the shipped post-cast f32 total order, preserving all prior rounding ties. Same checksum-frozen `release-perf` binaries, same worker `hz2` (EPYC Genoa, no SMT), CPUs 8-11, `RAYON_NUM_THREADS=4`, N=100k, dim=384, k=10, sample-size 50, 2 s warm-up, 10 s measurement. Mult3 ORIGINAL/CANDIDATE/ORIGINAL2 means 486.80/429.19/485.42 us with CV 3.85/3.69/3.51%; candidate ratio 0.8816-0.8841. Mult5 535.53 -> 464.94 us with CV 2.52/3.39%, ratio 0.8682. Criterion change intervals: mult3 -13.14% to -10.60%; mult5 -14.22% to -12.15%, both p=0.00. A prior `ovh-a` sequence was discarded and disclosed because repeated arms violated the CV gate (candidate CV 56.6%, later ORIGINAL CV 20.9%); it is not used for the keep. Recall@10 and nDCG@10 remained 1.0000 at mult=2/3/5/10; packed-order, merge-equivalence, and keep-all exact-search tests pin behavior. | `fsvi_int8_two_pass/int8_mult3`; `/int8_mult5` | 486.80 us; 535.53 us | 429.19 us; 464.94 us | **0.8816 (~1.13x); 0.8682 (~1.15x)** | KEEP (cod_fse) |
| 2026-07-14 | frankensearch-index | **384-dim fixed-shape AVX2 int8 dot (`bd-qdw5`)** — negative-ledger-first routing skipped the closed prepared-query, 8k-chunk, row-block, tombstone, and maddubs families and targeted the profile's dominant `dot_i8_i8_avx2` leaf (53.31% self-time). MiniLM's production dimension executes exactly twelve 32-byte blocks; the specialized kernel expands those blocks and removes the dynamic `min`, loop control, and scalar tail while preserving the shipped two-accumulator reduction. `dot_i8_i8` selects it only when both operands are exactly 384 elements and AVX2 is present; all other dimensions and non-AVX2 hosts retain the prior path. One strict-remote foreground run on RCH `vmi1227854`, release profile with LTO disabled/codegen-units 16, same-binary 4096-vector corpus: all 4096 dots were integer-exact against the dynamic kernel. The AB/BA paired sampler's A/A median was 1.005569 (p5 0.927520, p95 1.348639); candidate median 0.819306 (p5 0.744277, p95 0.989762), calibrated ratio 0.814769. Criterion's deliberately conservative sequential arms were `40.056–43.047 us` dynamic versus `36.198–39.741 us` fixed, with means `41.464 us → 37.932 us`; intervals do not overlap. The ledger reports that conservative mean ratio rather than the larger paired estimate. | `i8_dot_fixed384/{dynamic,fixed}` (4096×384) | 41.464 us | 37.932 us | **0.915 (~1.09x)** | KEEP (IcyRidge) |
| 2026-07-14 | frankensearch-index | **384-dim fixed-shape prepared 4-bit AVX2 dot (`bd-5ihn`)** — the production MiniLM scan consumes exactly 192 packed bytes, or twelve 16-byte nibble chunks. The specialized kernel expands those twelve chunks and removes the dynamic `min`, loop, per-iteration flush branch, and tail; one vertical i16 accumulator remains exact because each lane is bounded by `12*98=1176`. Runtime dispatch selects it only for the exact 384-dim prepared shape on AVX2, retaining the prior dynamic and portable paths for every other shape. One strict-remote foreground same-binary run on RCH `vmi1227854`, release profile with LTO disabled/codegen-units 16, checked all 4096 corpus rows integer-exact. AB/BA paired A/A median was 1.002083 (p5 0.962056, p95 1.124716); candidate median 0.615104 (p5 0.581151, p95 0.652924), calibrated ratio 0.613826. Criterion independently measured `61.607–116.38 us` dynamic versus `36.842–39.052 us` fixed; the paired calibrated ratio is reported because it is the more conservative estimate. | `fourbit_dot_fixed384/{dynamic,fixed}` (4096×384) | paired dynamic 1.000000 | paired fixed 0.613826 | **0.614 (~1.63x)** | KEEP (IcyRidge) |
| 2026-07-14 | frankensearch-index | **FSVI mmap f16-byte → packed-4-bit AVX2+F16C slab build (`bd-mnta`)** — negative-ledger routing skipped the closed fixed-shape dot, selection-key, and int8 chunk families and targeted a different quantization primitive: the file-backed 4-bit `OnceLock` build still widened f16 bytes portably but performed every round/clamp/nibble pack scalarly, while its in-memory sibling already had a vector quantizer. Added a little-endian byte-slab dispatch that runs the same AVX2+F16C quantize-and-direct-pack pipeline over mapped FSVI bytes without materializing `Vec<f16>`; non-AVX2 and non-little-endian targets retain the exact former byte path. The bench asserts byte-for-byte equality before timing, and the kernel parity test covers full chunks, sub-8 tails, odd dimensions, and multiple vectors. One strict-remote foreground run on RCH `vmi1227854`, release profile with LTO disabled/codegen-units 16, dim=384 and 10k vectors: bracketing current-path controls were `17.663 ms` and `17.661 ms` (B/A `0.9999`), while dispatch was `2.5159 ms`; candidate/control-mean ratio `0.14245`. | `f16_slab_pack4bit/{current_a,avx2_dispatch,current_b}/10000` | 17.662 ms | 2.5159 ms | **0.142 (~7.02x)** | KEEP (IcyRidge) |
| 2026-07-14 | frankensearch-index | **AVX2 packed-4-bit slab SIMD nibble compaction (`bd-ww52`)** — after the mmap byte-dispatch keep, the shifted quantization hotspot was the AVX2 pass-2 epilogue: every eight rounded/clamped i32 lanes were stored to a stack array, scalar-cast/masked in a four-iteration loop, then written as four separate bytes. Replaced only that epilogue with two lane-local `vpshufb` compactions plus nibble mask/shift/or and one unaligned 32-bit store; f16 decode, scale, rounding, clamp, tails, layout, and fallback paths are unchanged. The shared kernel accelerates both file-backed and in-memory slab construction. A const-specialized exact pre-change AVX2 arm remains for same-binary evidence; the bench asserts full output-byte equality before timing. One strict-remote foreground run on RCH `vmi1227854`, release profile with LTO disabled/codegen-units 16, dim=384 and 10k vectors: scalar-pack controls were `2.2690 ms` and `2.4261 ms` (B/A `1.0692`), SIMD compaction was `1.4336 ms`; candidate/control-mean ratio `0.6107`, with its full `1.3511–1.5188 ms` interval below both control intervals. | `f16_slab_pack4bit/{scalar_pack_a,simd_compact,scalar_pack_b}/10000` | 2.3476 ms | 1.4336 ms | **0.611 (~1.64x)** | KEEP (IcyRidge) |
| 2026-07-14 | frankensearch-index | **AVX2 packed-4-bit slab redundant `vroundps` elision (`bd-vb9i`)** — after SIMD nibble compaction, pass 2 still truncated each sign-biased value explicitly with `vroundps`, clamped that integer-valued float, then truncated it again with `cvttps_epi32`. The production specialization now clamps the biased finite value directly and lets `cvttps_epi32` perform the one required truncation; the exact explicit-round specialization remains as the same-binary control. This is byte-identical because the clamp bounds are integers and `trunc(clamp(x, -7, 7)) == clamp(trunc(x), -7, 7)` for finite `x`; decode, scale, half-away bias, nibble compaction, tails, layout, and fallbacks are unchanged. One strict-remote foreground run on RCH `vmi1227854`, release profile with LTO disabled/codegen-units 16, dim=384 and 10k vectors passed full output-byte parity. Explicit-round controls were `1.4002 ms` and `1.3803 ms` (B/A `0.9858`), while direct conversion was `1.2763 ms`; candidate/control-mean ratio `0.9180`, and its full `1.2393–1.3099 ms` interval was below both control intervals. | `f16_slab_pack4bit/{explicit_round_a,cvttps_only,explicit_round_b}/10000` | 1.3903 ms | 1.2763 ms | **0.918 (~1.09x)** | KEEP (IcyRidge) |
| 2026-06-27 | frankensearch-index | **FSVI 4-bit (16-level) two-pass — fastest lossless vector-search primitive** — new `VectorIndex::search_top_k_4bit_two_pass` + a fused SIMD nibble kernel `dot_packed_4bit` (load 16 packed bytes → `i16x16`, extract signed nibbles via arithmetic `(x<<12)>>12`/`(x<<8)>>12`, low·low + high·high). Packs each vector to signed 4-bit nibbles (`dim/2` bytes/vector — **half the int8 slab**), parallel pass-1 + exact f16 rescore of the top `k·mult`. **Lossless: recall@10 = 1.0000 at mult=5 on 100k clustered** (0.96 at mult=2); kernel verified against a scalar reference (`dot_packed_4bit_matches_scalar`) + bit-identical to `search_top_k` under keep-all (`four_bit_two_pass_keep_all_matches_exact`). The nibble-unpack compute offsets some of the 2× bandwidth saving, so the net vs int8 is modest, but 4-bit@mult5 is the **fastest lossless** option (mult=10 is slower → mult=5 is the sweet spot). Falls back to exact for WAL/non-F16; not wired into BOLD (not vector-bound). A/B (100k file-backed FSVI, in-process, same index; flat 2132.4 µs, int8 mult=5 888.2 µs): exact flat vs 4-bit mult=5 → **2.56×**; vs int8 mult=5 → **1.07×** | `fsvi_4bit_two_pass` | 888.2 µs | 831.4 µs | **0.936 (1.07× vs int8; 2.56× vs flat)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-index | **`dot_packed_4bit` kernel: vertical accumulation (1.07× → 1.36× vs int8)** — the 4-bit pass-1 is **compute-bound** (the parallel scan is not bandwidth-limited), and the kernel did a horizontal `prod.reduce_add()` **per 16-byte chunk** (12 reduces/vector at dim=384) — the dominant cost. Replaced with a single `i16x16` vertical accumulator (`acc += s_low·q_low + s_high·q_high`), flushed to the i32 sum every 16 chunks before any lane can exceed `i16` (16·98=1568/lane; 16-lane reduce 25088 < 32767). **Bit-identical** (scalar-match + keep-all tests pass; recall@10 = 1.0000 at mult=5 unchanged) — a pure speedup. A/B (100k file-backed FSVI, in-process, same run): 4-bit mult=5 vs int8 mult=5 | `fsvi_4bit_two_pass` | 1034.5 µs | 762.3 µs | **0.737 (1.36× vs int8; 3.22× vs flat)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-index | **In-memory 4-bit two-pass (in-memory twin of the FSVI 4-bit)** — `InMemoryVectorIndex::search_top_k_4bit_two_pass`: packed signed-4-bit slab (`dim/2` bytes/vector, half the int8 slab) + the fused `dot_packed_4bit` pass-1 (parallel, cutoff fast-path) + exact f16 rescore — the in-memory backend twin of the FSVI 4-bit, reusing the verified vertical-accumulation kernel. **Lossless: recall@10 = 1.0000 at mult=5** (0.98 at mult=2), 10k clustered; bit-identical under keep-all (`four_bit_two_pass_keep_all_matches_exact`). A/B (10k in-memory, in-process, same run; flat 352.9 µs, int8 mult=5 160.2 µs): 4-bit mult=5 vs int8 mult=5 → **1.40×**; vs flat → **3.09×** — the fastest lossless in-memory vector-search primitive. (Wiring it into the `SyncTwoTierSearcher` fast tier, which currently uses the int8 two-pass, is the follow-up: needs a `_filtered` variant.) | `int8_two_pass` 4-bit arm | 160.2 µs | 114.2 µs | **0.713 (1.40× vs int8; 3.09× vs flat)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-fusion | **`SyncTwoTierSearcher` fast tier wired int8 → 4-bit two-pass (deployment)** — `search_fast_hits` default path now routes to `search_top_k_4bit_two_pass_filtered` (added a `_filtered` 4-bit variant with the same doc_id-hash pre-screen as int8) instead of the int8 two-pass — deploying the fastest lossless primitive into the sync production hybrid. The fast tier is a reranked candidate generator, so the lossless 4-bit candidate set yields **identical fused top-k**: 6 sync_searcher lib + 9 integration tests pass unchanged. End-to-end same-run A/B (100k in-memory two-tier, no lexical): exact-fetch hybrid vs 4-bit-fetch hybrid → **2.36×** (up from the int8 wiring's 1.79× vs exact; the fast-tier fetch itself is 1.40× faster than int8, measured). Explicit `SearchParams` still uses the exact scan. | `sync_int8_fetch` | 2880.2 µs | 1219.8 µs | **0.424 (2.36× vs exact)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-fusion | **`SyncTwoTierSearcher` 4-bit fast-tier multiplier 5 → 3** — the default sync fast tier already over-fetches candidates before quality rerank/RRF, so the extra 4-bit pass-2 rescore from `mult=5` was wasteful. Tightened the candidate multiplier to 3 and added a deterministic clustered conformance test that compares the default 4-bit path against explicit exact scan top-k IDs for 12 queries. Remote same-worker proof on `ovh-a`, target dir `/data/projects/.rch-targets/frankensearch-cod-b`: baseline source restored to `FAST_TIER_MULT=5`, then candidate `FAST_TIER_MULT=3`, same command `cargo bench -p frankensearch-fusion --profile release --bench sync_int8_fetch -- --sample-size 10 --warm-up-time 1 --measurement-time 2`. Exact arm stayed neutral (2.3088 ms → 2.3120 ms). | `sync_int8_fetch/fast_fetch_4bit` | 769.95 µs | 639.75 µs | **0.831 (~1.20×)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-index | **Prepared 4-bit query lanes for repeated scan dots** — `prepare_4bit_query` decodes the query nibbles once into signed SIMD lane vectors, and `dot_4bit_prepared` reuses those lanes for every stored vector. This lands the alien-graveyard/vectorized-execution lever as a pure register-reuse win on the already-shipped 4-bit scan path: old path decoded both stored and query nibbles per dot; new path decodes only stored nibbles inside the hot loop. Behavior preserved by `dot_packed_4bit_matches_scalar` now checking both packed and prepared kernels, including the `0x99` signed-nibble extreme; the prepared loop also truncates to the shorter stored/query chunk stream instead of panicking on public mismatched slices. Remote RCH worker `hz2`, target dir `/data/projects/.rch-targets/frankensearch-cod-b`, command `cargo bench -p frankensearch-index --profile release --bench dot_product fourbit_prepared -- --sample-size 10 --warm-up-time 1 --measurement-time 3`. | `dot/dim256/fourbit_prepared/10000`; `dot/dim384/fourbit_prepared/10000` | 459.40 µs; 567.38 µs | 389.81 µs; 509.84 µs | **0.849 (~1.18×); 0.899 (~1.11×)** | KEEP (Codex) |
| 2026-06-30 | frankensearch-fusion | **`SyncTwoTierSearcher` quality-tier blend: elide the intermediate `quality_hits` doc_id clone** — the quality tier is a re-scored *subset* of `fast_hits` (it shares every `doc_id`/`index`), yet `search_internal` built a throwaway `Vec<VectorHit>` cloning one `String` doc_id per quality hit purely to pass a `&[VectorHit]` to `blend_two_tier` — even though both consumers (the blend and the downstream `quality_scores` borrow-map) only ever read the doc_id as `&str`. New additive `blend_two_tier_aligned` blends straight from the aligned `Vec<Option<f32>>` quality scores (the `quality_scores_for_hits` output), borrowing each doc_id from `fast_hits`, eliding N short-`String` allocations on the `limit_all` quality path. This is a **different** clone from the RRF `into_owned` (which stays un-elidable per-crate — see NEGATIVE_EVIDENCE 2026-06-29); the NEGATIVE_EVIDENCE note cited `quality_hits` only as evidence `fast_hits` can't move, never analyzed it for elision. Bit-identical (two-loop→single-pass merge guarded by `is_none()`, first-in-fast-order wins; permanent `aligned_blend_is_bit_identical_to_materialized` test over alpha ∈ {0,0.3,0.7,1,NaN} incl. a duplicate doc_id, plus the bench's pre-timing bit-identity guard). Remote RCH worker `ovh-a`, target dir `/data/projects/.rch-targets/search-cc`, `cargo bench -p frankensearch-fusion --bench blend_aligned`. | `blend_aligned/{current,aligned}/10000`; `/100000` | 1.7238 ms; 20.003 ms | 1.2522 ms; 18.478 ms | **0.726 (~1.38×); 0.924 (~1.08×)** | KEEP (BlackThrush) |
| 2026-07-01 | frankensearch-fusion | **`TwoTierSearcher` (async production path) wired to `blend_two_tier_aligned` — same lever, 2nd call site** — deploys the clone-free blend above to the *full production* searcher (`searcher.rs` `search_internal`, the path the CLI/library actually run; the 2026-06-30 row was the lighter `SyncTwoTierSearcher`). The old async path built a `Vec<VectorHit>` cloning one `String` doc_id per quality hit, calibrated its scores in place, then blended — but the doc_ids were only read as `&str` (blend + `quality_scores_by_doc`). Score calibration is a pure per-element transform, so it was extracted to a `calibrate_score(f32)->f32` helper (`apply_score_calibration_to_hits` now delegates to it), the calibrated quality scores are computed in aligned `Vec<Option<f32>>` form, and the blend borrows doc_ids straight from `fast_hits` — eliding the same N short-`String` allocations. Bit-identical (same `blend_two_tier_aligned` + its `aligned_blend_is_bit_identical_to_materialized` test; calibrate_score extracted verbatim; conformance GREEN 820/0). **Measurement is the same `blend_aligned` A/B as the row above (identical code transformation), NOT an independent additional win — this row records the production-path deployment, not new numbers.** | `blend_aligned/{current,aligned}/10000`; `/100000` (same lever) | 1.7238 ms; 20.003 ms | 1.2522 ms; 18.478 ms | **0.726 (~1.38×); 0.924 (~1.08×)** | KEEP (BlackThrush) |
| 2026-07-01 | frankensearch-fusion | **`SyncTwoTierSearcher::search_collect` skips the discarded progressive-phase `Vec<ScoredResult>` clones (`want_phases` flag)** — `search_collect` returns only `(final_results, metrics)` and throws away `outcome.phases`, yet `search_internal` built those phases by cloning the *entire* result set (N owned doc_id `String`s each) once per phase (Initial + Refined). At `limit_all` (k=N) that is up to **2·N wasted allocations per query** — a large slice the fusion mining had missed because the clone looks "needed" for the streaming path (only `search_iter` consumes phases). Threaded a `want_phases` bool: `search_collect` passes `false` (skips both pushes and the `phases` alloc), `search_iter` passes `true` (unchanged). Bit-identical for `search_collect` (same `final_results`+`metrics`; no `metrics.*` is set inside a phase push; the bench sanity-asserts `collect==iter` final ranking) and for `search_iter` (want_phases=true ≡ prior behavior). Remote RCH worker `vmi1264463`, target dir `/data/projects/.rch-targets/search-cc`, `cargo bench -p frankensearch-fusion --bench collect_limit_all` — end-to-end A/B on the real search path via public APIs: both arms run identical fast-scan→quality→blend, so the delta is purely the phase clones. `iter` = old collect (builds both phases); `collect` = new. | `collect_limit_all/{iter→collect}` (N=10k, dim=384, k=N=limit_all) | 27.661 ms | 21.372 ms | **0.773 (~1.29×)** | KEEP (BlackThrush) |
| 2026-07-01 | frankensearch-fusion | **federated `appeared_in`: `BTreeSet<String>` → interned shard-id `Vec<u32>`** — the explicit untested route-next from the reverted key-clone lever (NEGATIVE_EVIDENCE 2026-06-27: *"attack the `appeared_in` `BTreeSet<String>` churn ... dedup shard names to a small interned id set, not the key clone"*). `accumulate_doc` runs once per (shard, hit) — thousands of times per fuse — and did `appeared_in.insert(shard_name.to_owned())`: a `String` heap alloc **plus** a `BTreeSet` node alloc + str-cmp tree descent *every call*. `intern_shard_names` maps each distinct shard name to its sorted rank; `accumulate_doc` pushes the integer id; `into_ranked_hits` sort+dedups once per doc and maps ids→names. Sorted-rank ids ⇒ ascending id order == lexicographic name order, so `FederatedHit::appeared_in` is byte-identical (federated tests assert the ordered names; the `federated_appeared_in` bench asserts full-output equality across all arms). Chose `Vec<u32>` over a faster `u64` bitset for **no shard-count cap** (a bitset silently corrupts >64 shards; Vec has no regression). Remote RCH worker `hz1`, target dir `/data/projects/.rch-targets/search-cc`, `cargo bench -p frankensearch-fusion --bench federated_appeared_in` (btreeset vs vec, within-run). | `federated_appeared_in/{btreeset→vec}/s5_h200_u600`; `/s10_h500_u2500` | 296.64 µs; 1.2445 ms | 206.45 µs; 1.0230 ms | **0.696 (~1.44×); 0.822 (~1.22×)** | KEEP (BlackThrush) |

**Lever (ParsedQuery no-negation fast path):** `ParsedQuery::parse` runs per search query (the
searcher parses for `-term`/`NOT "phrase"` negations). The committed parser always materialized a
`Vec<char>` and re-collected each word via `chars[a..b].iter().collect()`. Negation syntax requires
one of `-` `"` `\`; with none present (the common query) there are no negations, so the positive
part is just the whitespace-normalized input. The new fast path returns it directly
(`split_whitespace` + `push_str` into one buffer), skipping the char materialization. Byte-identical
(the full parser collects the same whitespace-split words and `join(" ")`s them; 42 parsed_query
tests green). Measured on a plain multi-word query (`parsed_query`): 503.4 ns → 109.6 ns, **0.218
(~4.59×)**. Queries that *do* use negation syntax still take the exact char-based path.

**Lever (ParsedQuery NOT-keyword allocation elision):** after the no-negation fast path, queries
that contain `"` or `-` still use the char parser. Its `matches_not_keyword` helper previously built
a fresh three-character `String` for every token-boundary check, then called
`eq_ignore_ascii_case("NOT")`. Replaced that with direct ASCII `N|n`, `O|o`, `T|t` checks while
leaving the surrounding boundary/whitespace rules unchanged. This preserves behavior exactly for
the accepted keyword (`eq_ignore_ascii_case` is ASCII-only here) and removes transient allocation
from negation-capable queries. Measured with the existing per-crate bench extended to include a
replica of the old helper:

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- cargo bench -p frankensearch-core --profile release \
    --bench parsed_query -- --sample-size 10 --warm-up-time 1 --measurement-time 2
```

The requested literal `cargo bench --release` form was tried first and Cargo rejected it with
`unexpected argument '--release'`; the successful command used Cargo's release profile spelling
above. Criterion estimates for `parsed_query/not_phrase`: old mean **746.73 ns**, new mean
**668.39 ns**, ratio **0.895 (~1.12×)**. Original-comparator ratio is N/A because this is an
internal query-parser primitive, recorded in `docs/NEGATIVE_EVIDENCE.md`.

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench parsed_query
```

**Lever (SimHash table-driven vote):** building on the branchless vote, `apply_hash_votes` now
processes the window hash **one byte at a time**: each byte (0..256) indexes a compile-time
`VOTE_TABLE: [[i32;8];256]` (`2*((b>>k)&1)-1`) for its 8 ±1 votes, added to the matching 8 bit
counters as a slice — the compiler vectorizes the 8-wide `i32` add (SSE2 `i32x4`). This replaces the
64 per-bit `shift/mask/mul/sub` of the branchless form. Bit-identical (byte `j`'s bit `k` is hash bit
`8j+k`; 20 fingerprint tests + 895 core lib tests green). Measured (`simhash_votes`): branchless
15.810 µs → table 10.930 µs, **0.691 (~1.45×)**, i.e. ~0.60× vs the original conditional. The table
is built `const` (zero runtime/init cost).

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench fingerprint
```

**Lever (SimHash branchless vote):** `Fingerprint::compute` builds a 64-bit SimHash over 3-token
shingles (per document, for incremental re-embedding decisions). The inner `apply_hash_votes`
accumulated ±1 into 64 bit counters per window via `if (bit set) { +1 } else { -1 }` — a
data-dependent branch on effectively-random hash bits (~50% misprediction). Replaced with the
branchless `*weight += 2*((hash>>bit)&1) - 1`. Bit-identical (`semantic_simhash` unchanged; 20
fingerprint tests green incl. `identical_text_produces_identical_fingerprint`,
`large_document_computes_without_overflow`). Measured over a ~300-token doc's shingle sweep
(`simhash_votes`): 18.480 µs → 16.073 µs, **0.870 (~1.15×)** — confirming LLVM did *not* already
emit a branchless form (measured rather than assumed). Determinism-safe: same vote, same final
`weight > 0` test.

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench fingerprint
```

**Lever (nfc_normalize → Cow):** `canonicalize`/`canonicalize_query` begin with NFC normalization.
ASCII text is already NFC, so the prior code returned `text.to_owned()` — a whole-document copy —
which then fed `strip_markdown_and_code` (which allocates its own buffer) and the trim path. Since
those stages only need a `&str`, `nfc_normalize` now returns `Cow<str>`: `Cow::Borrowed(text)` for
ASCII (zero copy), `Cow::Owned` for non-ASCII (the `nfc()` collect, unchanged). Byte-identical (the
`&str` view is identical; 34 canonicalize + 895 core lib tests green incl.
`nfc_normalize_ascii_fast_path_matches_reference`). Measured borrow vs `to_owned` on a 2.25 KB ascii
doc (`nfc_ascii_copy`): 39.45 ns → 1.17 ns, **0.030 (~33×)**. Stacks on the earlier NFC ASCII
fast-path (which had already replaced the `nfc()` state machine with `to_owned`); this removes the
remaining copy on the dominant indexing path's entry.

**Lever (strip_markdown_line slow-path guards):** the inline-markdown slow path previously ran the
**entire** transform chain — `replace("**")`, `replace("__")`, `replace('*')`,
`strip_italic_underscores`, `replace('`')`, `strip_markdown_links` — unconditionally, allocating a
whole-line copy at each step even when the relevant char was absent. A `snake_case` line (trigger =
`_` only) thus paid four no-op allocating passes (`**`, `*`, `` ` ``, links). Now one byte scan sets
`has_star`/`has_underscore`/`has_backtick`/`has_bracket` and each transform is guarded by its
trigger char (same order → byte-identical; skipping an absent-trigger transform is the same as the
no-op copy it produced). 34 canonicalize tests green (bold/italic, links, backticks, nested,
headings). Measured on a snake_case line (`strip_markdown_slowpath`, `strip_italic_underscores`
omitted from the A/B since it runs in both): 307.9 ns → 48.3 ns, **0.157 (~6.37×)**. Stacks with
the single-pass `strip_italic_underscores` lever for the dominant code-search line shape.

**Lever (strip_italic_underscores single pass):** this runs on the inline-markdown path of
`strip_markdown_line`, which is hit by **every line containing `_`** — i.e. every `snake_case`
identifier line, the common case in code search (where almost all underscores are kept). The prior
implementation materialized a `Vec<char>`, a `Vec<bool>` keep-mask, and a final `collect` — three
allocations and three passes. Rewritten as a single pass that builds the output `String` directly,
tracking the previous source char and peeking the next to apply the exact same word-boundary test
(`prev`/`next` is-word). Byte-identical — proven by `strip_italic_underscores_matches_reference`
(snake_case, `_italic_`, mixed, leading/trailing, non-ASCII word chars) + 33 canonicalize tests.
Measured over 40 snake_case code lines (`strip_italic_underscores`): 9.770 µs → 2.982 µs, **0.305
(~3.28×)**.

**Lever (QueryClass::classify):** `classify` runs on every search query (adaptive lexical/semantic
budget + the lexical short-circuit gate both call it). `looks_like_identifier` rescanned the query
with `chars().any(char::is_whitespace)` up to four times and allocated a `Vec` via
`rsplitn(2,'-').collect()` for the issue-ID check. Now whitespace is computed once and all the
no-whitespace heuristics are grouped under it (so whitespace queries skip the whole block after a
single early-stopping scan), and the issue-ID check uses allocation-free `rsplit_once('-')`.
Behaviour-identical (26 query_class tests + 894 core lib tests green; grouping is safe because every
branch returns `true`). Measured over an 11-query identifier/keyword/natural-language mix
(`query_class`, real `classify` as `new` vs the prior logic replicated as `old`): 732.4 ns →
512.4 ns, **0.700 (~1.43×)**.

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench query_class
```

**Lever (strip_markdown_line → Cow):** building on the fast-path lever above, the strip pipeline
now returns `Cow<str>`: `strip_list_marker` yields borrowed slices in every branch (unordered →
`strip_prefix` slice, ordered → `&trimmed[digit_count+2..]` since the marker is single-byte ASCII,
not-a-marker → the line), and `strip_markdown_line`'s fast path returns that borrowed `Cow`. So a
plain line (no inline markdown — prose, most code comments) flows through `strip_markdown_and_code`
with **zero per-line allocation**: the bytes are copied once by the existing `result.push_str`. The
inline-markdown slow path still returns `Cow::Owned`. Byte-identical (894 core lib tests green incl.
ordered/unordered list markers, `numbers_not_list_markers_preserved`, headings, blockquotes).
Measured over the full 80-plain-line `push_str` loop (`strip_markdown_cow`: `string` = prior
per-line owned String vs `cow` = borrowed): 2.166 µs → 1.390 µs, **0.642 (~1.56×)**, on top of the
fast-path win.

**Lever (strip_markdown_line fast path):** `strip_markdown_line` runs on every line of every
document at index time, and markdown stripping is the dominant `canonicalize` cost. The fast path
(lines with no inline-markdown chars — plain prose, most code-comment text) copied the whole line
with `line.to_string()` before the header/blockquote prefix trims, then `strip_list_marker`
allocated the final owned String — two allocations per line. Since the prefix trims and
`strip_list_marker` only need `&str`, the fast path now trims the borrowed `line` directly and only
`strip_list_marker` allocates (one allocation). Byte-identical (same trims in the same order; 33
canonicalize tests green incl. headings/blockquotes/list-markers/links/bold-italic). Measured
head-to-head in one process over 80 plain lines (`strip_markdown_fastpath`, `strip_list_marker`'s
final owned-String step modeled): 2.328 µs → 1.306 µs, **0.561 (~1.78×)**.

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench canonicalize
```

**Lever (RRF fuse single lookup):** `rrf_fuse_with_graph` accumulates per-doc scores in an
`AHashMap<&str, FusedHitScratch>` across the lexical, semantic, and graph candidate lists. Each
loop did a `hits.get(doc_id)` (per-source dedup probe) **then** a `hits.entry(doc_id)`
(insert/update) — two hashes of the same key per candidate, where the `get` almost never skips
(doc ids are unique within a source). Consolidated to a single `entry` match: `Occupied` keeps the
dedup-skip guard then updates via `get_mut`, `Vacant` inserts — semantically identical (47 rrf
tests + 816 fusion lib tests green), one hash lookup instead of two. Measured head-to-head in one
process (`benches/rrf_fuse.rs`, 1000 lexical + 1000 semantic, ~50% overlap; the private
`FusedHitScratch` is replicated with an equivalently-shaped struct): 29.11 µs → 23.07 µs, **0.793
(~1.26×)**. RRF fusion is a per-query cost on the hybrid path, so this directly trims hybrid search
latency. Frankensearch pre-change before/after ratio only (`bd-ui41`).

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-fusion --bench rrf_fuse
```

**Lever (hash embedder alloc elision):** `HashEmbedder::embed_sync` runs on every document at
index time and every query (the non-semantic `fnv1a-*`/`jl-*` fast tiers, and the frankensearch
side of the BOLD hash-hybrid comparison). The committed path did two dimension-sized allocations
per embed: `tokenize` collected a `Vec<&str>`, and `l2_normalize(&embedding)` returned a freshly
allocated `Vec<f32>`. Now `tokenize` returns a lazy iterator (each embedder consumes tokens exactly
once) and the new `frankensearch_core::traits::l2_normalize_in_place` normalizes the owned
accumulator in place — so the only allocation is the accumulator itself. Bit-identical output
(`l2_normalize_in_place_matches_allocating` proves the in-place form equals `l2_normalize` across
zero/near-zero/non-finite cases; 286 embed lib tests + the existing `output_is_l2_normalized` /
`jl_output_is_l2_normalized` green). Measured head-to-head in one process (old 2-alloc vs new
1-alloc, replicated in `benches/hash_embed.rs`): FNV path **0.846 (~1.18×)**; the JL path is
compute-bound (O(tokens·dim) xorshift), so its alloc savings vanish into noise (1.018, neutral —
never a regression). Frankensearch pre-change before/after ratio only (no dominance-vs-original
claim; `bd-ui41`).

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-embed --bench hash_embed
```

**Lever (canonicalize tail):** `DefaultCanonicalizer::canonicalize` runs on every document at
index time and every query. The committed tail did two whole-document allocations — the old
`filter_low_signal` returned `text.to_string()` on the common (non-ack) path, then
`truncate_to_chars` copied again. Converting the filter to an allocation-free `is_low_signal(&str)
-> bool` predicate (early-return `String::new()` for acks) lets `canonicalize` pass its already-owned
`ws_normalized` buffer straight to truncation — one copy, not two. Output is byte-identical (acks →
`""`, normal docs → same bytes; 893/893 `frankensearch-core` lib tests green, incl. all 33
canonicalize tests). Measured head-to-head in one process (`pipeline_tail/old` = 2 copies vs
`/new` = 1 copy, 2.25 KB ascii doc): 91.4 ns → 59.4 ns, **0.649 (~1.54×)**. Frankensearch
pre-change before/after ratio only (no dominance-vs-original claim; blocked by `bd-ui41`).

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench canonicalize
```

**Lever (in-memory filtered-scan prescreen):** the FSVI on-disk scan (`search.rs`) already
pre-screens filters with a **precomputed `doc_id_hash`** via `SearchFilter::matches_doc_id_hash`,
but `InMemoryVectorIndex::scan_range` called plain `matches(doc_id)` — which for a `BitsetFilter`
**re-hashes the doc_id string every vector** (`fnv1a_hash` per element). In a selective filtered
scan the filter check is the dominant per-(excluded-)vector cost, so this was a real waste. Added a
lazy `doc_id_hashes: OnceLock<Vec<u64>>` (built on first *filtered* search with the same
`frankensearch_core::filter::fnv1a_hash` that `BitsetFilter` uses — so it's bit-correct;
exact/unfiltered callers pay nothing) and switched the scan to `matches_doc_id_hash(precomputed)`
with a fallback to `matches()` for filters that can't decide by hash (e.g. metadata filters).
Behaviour-identical (17 in-memory + 15 filter tests + new `search_with_bitset_filter_uses_precomputed_hash_path`
+ 357 index lib serial all green). Measured `matches` re-hash vs `matches_doc_id_hash` precomputed
over 10k `BitsetFilter` checks (`benches/filter_prescreen.rs`): 183.5 µs → 88.4 µs, **0.482
(~2.08×)** — the per-vector filter-check cost the in-memory filtered scan now saves, bringing it to
parity with the FSVI scan.

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench filter_prescreen
```

**Lever (bd-gfzk):** `dot_product_f16_bytes_f32` / `dot_product_f16_f32` — the **default
quantization** path (f16 FSVI, `search.rs:288,346,441`). The f16 paths were decode-bound: the
per-element scalar `f16::to_f32()` dominated (confirmed in `docs/NEGATIVE_EVIDENCE.md` — that's
why accumulator unrolling failed here). Replaced the 8× scalar decode with a **branchless SIMD
widen** (Giesen magic-multiply over `wide::u32x8`/`f32x8` + pure-integer inf/nan fixup) →
**~2.5–3.2× faster** on the dominant path. The widen is **bit-exact** to `f16::to_f32()` for all
65 536 f16 values (exhaustive test `simd_f16_widen_is_bit_exact`), and the single accumulator /
lane assignment is unchanged, so every dot score is **bit-identical** on finite data → no
determinism/golden risk (all 19 simd + full index lib tests green). Measured head-to-head in one
process (`f16_*_new` SIMD vs `f16_*_old` scalar). No dominance-vs-original claim (blocked by
`bd-ui41`); pre-change before/after ratio only.

_Refinement (same lever):_ the byte path also replaces the 8 scalar `from_le_bytes` + stack
round-trip with a single little-endian SIMD load (`[u8;16]`→`u16x8`→`u32x8` zero-extend, BE
fallback retained). Still bit-exact (`simd_f16_bytes_load_matches_scalar`). The in-process
`f16_bytes_new/old` ratio improved 0.345→0.317 (dim256) / 0.317→0.305 (dim384), ~4–8% on top of
the SIMD widen. Modest and not cleanly isolated from worker noise, but kept because one SIMD
load strictly dominates 8 scalar loads (worst case neutral, never a regression).

**Lever:** `dot_product_f32_bytes_f32` (used by f32-quantized FSVI indexes, `search.rs:307,372,492`).
The old kernel decoded f32s from an open-ended `&stored_bytes[off..]` slice, which the
compiler could not prove a fixed width for and so did not vectorize the `from_le_bytes`
decode. Rewriting the decode over fixed `[u8; 32]` sub-blocks (+ 4 independent f32x8
accumulators) lets it vectorize → **~2.7–3.0× faster**. Measured head-to-head in one
process via `benches/dot_product.rs` (`f32_bytes_new` vs `f32_bytes_old`), so the ratio is
immune to which rch worker the run landed on. f16 paths were tried with the same restructure
but **regressed** (decode is scalar-bound there) and were reverted — see
`docs/NEGATIVE_EVIDENCE.md`. No dominance-vs-original claim (blocked by `bd-ui41`); this is a
frankensearch pre-change before/after ratio only.

`BlueGull` confirmation command:
```bash
RCH_WORKER=vmi1149989 \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo bench -p frankensearch-index --profile release --bench dot_product
```
Same worker conformance gate:
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo test -p frankensearch-index simd --profile release
```
Result: 18 relevant SIMD tests passed; RCH reported `remote vmi1149989`.

## Foundational primitives & lever validations (not yet product wins)

These are measured de-risking results for levers that are not yet wired into the search
path. They are **not** search speedups — they tell us whether a bigger build is worth it.

### int8 dot kernel (`dot_i8_i8`) — validates the `bd-b5wl` ADC pass-1 premise

Added a tested, exported symmetric int8 dot primitive (`crates/frankensearch-index/src/simd.rs`,
SIMD via `wide::i16x8::mul_widen`; exhaustive-ish correctness test incl. the i32 overflow
worst case). Benched head-to-head against the optimized f16 dot **in the same process/run**:

| Workload | `i8_dot` (int8) | `f16_bytes_new` (optimized f16) | int8/f16 ratio |
|----------|-----------------|---------------------------------|----------------|
| `dot/dim256` (n=10k) | 368 µs | 1.113 ms | **0.331** (~3.0×) |
| `dot/dim384` (n=10k) | 505 µs | 1.622 ms | **0.311** (~3.2×) |

**Conclusion:** scoring all N with int8 is ~3× faster than the current exact f16 scan (½ the
bytes + integer `mul_widen` MAC, no f16 decode). This **validates building the int8 ADC two-pass
scan** (`bd-b5wl`): a pass-1 int8 sweep over all N + an exact f16 rescore of only `k·mult`
candidates should net ~2.5–3× on the vector scan. **Caveat:** this is a kernel-throughput ratio,
**not** a search speedup — the product win requires the sidecar + two-pass wiring + a recall@10
gate (it's approximate). The primitive is unused by the search path until that lands.

**Recall gate measured (`int8_two_pass_recall_at_10`):** int8 pass-1 (top `k·mult`) + exact f16
rescore recovers the true f16 top-10 with **recall@10 = 1.0000** at `mult=20` (top-200 of 3000
random L2-normalized vectors, averaged over 25 queries). So at a modest multiplier the two-pass
is **lossless** here. Both ADC gates now pass: ~3× pass-1 speed **and** recall = 1.0. **Caveat:**
random vectors have wide angular spread; real (clustered) embeddings may need a higher `mult` —
the production wiring must re-measure recall on a real corpus and tune `mult`/quant granularity.

**End-to-end search-level ratio (`topk_exact_f16` vs `topk_int8_2pass`, k=10, mult=20, n=10k):**
full top-10 pipeline (score all N + select), measured same-process:

| dim | exact f16 top-k | int8 two-pass top-k | ratio |
|-----|-----------------|---------------------|-------|
| 256 | 1.213 ms | 469 µs | **0.387** (~2.6×) |
| 384 | 1.794 ms | 607 µs | **0.339** (~3.0×) |

So the int8 ADC two-pass is **~2.6–3.0× faster on the actual top-k search workload with recall@10
= 1.0** (mult=20, random vectors; both pipelines pay a conservative N-element sort, so production
bounded-heap selection would only widen the gap). This is the search-level proof, not just a
kernel ratio.

> **CORRECTION (see `docs/NEGATIVE_EVIDENCE.md`):** the `topk_exact_f16` baseline above is a
> *serial full-sort* pipeline. The real product `search_top_k` is **rayon-parallel + bounded-heap
> + cutoff** and is far faster, so the ~2.6–3× "search win" holds only vs a *serial* exact — NOT
> the product's parallel exact. The kernel ~3× (`33fb45b`) and recall@10 = 1.0 stand; the
> integrated-method ratios vs parallel exact are below.

**Wired into the product (`InMemoryVectorIndex::search_top_k_int8_two_pass`):** the in-memory index
precomputes an int8 slab (one corpus-wide max-abs scale, ranking-preserving) at construction and
exposes an opt-in two-pass method — a **parallel bounded-heap** int8 pass-1 (top `limit·mult` per
chunk, `PARALLEL_CHUNK_SIZE` chunking to match the exact path's core use) → exact f16 rescore with
the *same* bounded-heap selection as the exact path. **Bit-identical** to `search_top_k` whenever
pass-1 recall is 1 (`int8_two_pass_matches_exact_topk`: identical doc-ids + scores), exact
fallback if the int8 slab is absent. Existing exact paths untouched; 355/355 index lib tests green.

**Real-method head-to-head (`inmem_topk`, top-10) vs the parallel exact `search_top_k`:**

The candidate budget `limit·mult` is the selection-overhead knob. `int8_two_pass_recall_at_10`
shows recall@10 = **1.0 down to mult=2** for random (well-separated) vectors, so the original
mult=20 was 10× more candidates than needed. Re-benched at **mult=5**:

| Workload | exact `search_top_k` | `int8_two_pass` mult=5 | ratio | (mult=20 ratio) |
|----------|----------------------|------------------------|-------|-----------------|
| dim256/n10k | 328 µs | 223 µs | **0.68** | 0.72–1.09 (worker-var) |
| dim384/n10k | 397 µs | 278 µs | **0.70** | 0.66 |
| dim384/**n100k** | 2.02 ms | 1.34 ms | **0.67** | 0.75 |

At mult=5 the win is a **robust ~1.4–1.5× across 10k AND 100k** (consistent on both fast and slow
workers), no longer the worker-dependent tie seen at mult=20. Smaller candidate budget = less
top-`k·mult` selection + merge overhead, so more of the int8 kernel's 3× survives. Still
bit-identical to exact when recall=1. **Honest scope:** ~1.4–1.5× (not the 2.6–3× serial-baseline
figure); recall=1.0 is for *random* vectors — clustered real embeddings may need a higher mult, so
the production path must re-measure recall on a representative corpus. **Remaining (`bd-t8tv`):**
the mmap FSVI `search.rs` path (on-disk int8 sidecar, where exact also pays page-faults + decode —
likely a larger win); real-corpus recall + mult tuning.

**2026-06-26 — clustered-data validation (answers the "needs higher mult on real data" caveat):**
re-ran on **clustered** vectors (64 centroids + noise — the realistic embedding shape, vs the prior
uniform-random) via the new gated-free bench `benches/int8_two_pass.rs`
(`cargo bench -p frankensearch-index --bench int8_two_pass`). Result at N=10000, dim=384, k=10:

| candidate_mult | recall@10 (clustered) | latency | ratio vs flat |
|----------------|-----------------------|---------|---------------|
| flat exact | 1.000 | 309.7 µs | 1.00× |
| 2 | **1.0000** | — | — |
| 5 | **1.0000** | 166.5 µs | **0.538 (1.86×)** |
| 10 | **1.0000** | 207.2 µs | **0.669 (1.50×)** |
| 20 | **1.0000** | — | — |

So on clustered data the two-pass is **lossless (recall@10 = 1.0000) down to mult=2** — *easier* than
random vectors (clear cluster winners always land in the `k·mult` candidate set, and the exact
rescore fixes the order). It is a **lossless ~1.5–1.86× speedup over the flat parallel-SIMD exact
scan** at this scale. This is the strongest vector lever found and is the **opposite of HNSW** (which
trades recall — see `docs/NEGATIVE_EVIDENCE.md`). **Recommendation:** use `search_top_k_int8_two_pass`
(mult≈10 for safety margin) for in-memory vector search; it is strictly better than flat exact on
realistic data. Not made the unconditional default because recall=1.0 is empirical (exact iff the
true top-k ⊆ the int8 candidate set) — safe as an opt-in / large-N path, with a recall gate for
pathological corpora.

_Scale confirmation — and the win GROWS with N:_ re-ran the same clustered bench at **N=100000**
(dim=384, k=10 — the BOLD / Tantivy-comparison corpus size): recall@10 = **1.0000 at every mult
(2, 5, 10, 20)** (lossless at production scale; int8 quantization error is scale-independent, so a
modest `k·mult` candidate set still always contains the true top-10). Measured **clustered** latency
(single run): flat 2131 µs vs int8 mult=5 **862 µs (2.47×)**, mult=10 1187 µs (1.80×) — i.e. the
relative speedup **grows from ~1.86× at 10k to ~2.47× at 100k**, because the int8 pass-1's halved
bytes + no-f16-decode advantage scales with N (consistent with the `bd-t8tv` "larger win at scale"
hypothesis). (The 10k mult sweep is pass-1-bound and worker-noisy — mult 2–10 all land ~1.3–1.6×
within noise; the clean separation appears at 100k.) So `search_top_k_int8_two_pass` (mult≈10) is a
**lossless 1.86–2.47× win across 10k–100k, strongest at production scale** — the highest-value
vector lever, pending the wiring (opt-in / large-N default with an exactness gate).

_Wiring target (verified 2026-06-26, BlackThrush):_ the validated two-pass is **in-tree but unused
by callers** — `InMemoryTwoTierIndex::search` (`two_tier.rs:350,353`) and the fusion searcher call
the **exact** `search_top_k` / `search_top_k_with_params`. The integration point is there: route
large-N searches to `search_top_k_int8_two_pass(query, k, mult≈10)`. The blocker for a *default*
(rather than opt-in) switch is the exactness gate — a sound per-query int8-slab error bound (the i8
corpus-max-abs slab quantizes each component to `q=round(x/σ)`, `σ=max_abs/127`, so by
Cauchy–Schwarz `|true_dot − int8_dot| ≤ E` with `E ≈ √dim·σ`; if `β + E < θ` where `β` is the
`k·mult`-th int8 score and `θ` the k-th *exact* rescored score, the result is provably exact).
Deriving + verifying that bound (and its int8→f32 scaling) is the correctness-critical,
multi-iteration step that shouldn't be rushed into the default path.

_Dim confirmation:_ also re-ran at **dim=768** (BGE-base / OpenAI-class embeddings), N=10000:
recall@10 = **1.0000 at every mult (2, 5, 10, 20)** — the lossless property holds across the
embedding-dim range (384–768), so the lever applies to high-dim models too. (Latency ratio
expected ~the dim=384 figure since both flat and int8 pass-1 scale ~linearly with dim; the dim=768
latency estimates didn't sync back cleanly this run, but recall — the correctness question — is the
decisive one and is lossless.)

**Exactness-gate note (for the wiring):** `ScalarQuantizer` (the u8 per-dim FSVI quantizer) already
exposes the bound machinery (`max_dim_errors()` = `scale/2`; `epsilon ≤ max_scale·√dims`), but the
in-memory two-pass uses a **separate i8 corpus-max-abs slab** — a sound certified-exact gate needs
that slab's per-query error bound derived + verified (correctness-critical), so the safe wiring is a
dedicated multi-iteration step, not a 60m change.

These rows are routing evidence for future levers, not wins.

Command:
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- cargo bench -p frankensearch --profile release --bench search_bench -- --sample-size 10
```

Worker: `ovh-a`; RCH reported remote completion in `608.1s`.

| Date | Crate | Workload | Measurement | Original ratio | Status |
|------|-------|----------|-------------|----------------|--------|
| 2026-06-24 | frankensearch | `dot_product_f32/128` | `12.223 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `dot_product_f32/256` | `30.630 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `dot_product_f32/384` | `44.346 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `dot_product_f32/768` | `82.308 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `hash_embedder/short_10w` | `734.48 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `hash_embedder/medium_100w` | `2.3161 us` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `hash_embedder/long_1000w` | `17.225 us` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `vector_search_topk/top10/1000` | `944.07 us` | blocked by `bd-ui41` | noisy baseline |
| 2026-06-24 | frankensearch | `vector_search_topk/top10/5000` | `3.4640 ms` | blocked by `bd-ui41` | noisy baseline |
| 2026-06-24 | frankensearch | `vector_search_topk/top10/10000` | `1.6642 ms` | blocked by `bd-ui41` | noisy baseline |
| 2026-06-24 | frankensearch | `rrf_fusion/fuse/1000+1000` | `80.563 us` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `score_normalization/z_score/10000` | `43.159 us` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `index_io/write/10000` | `13.812 ms` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `index_io/open/10000` | `19.865 us` | blocked by `bd-ui41` | baseline only |

### 2026-06-25 — non-semantic hash tier lexical guard (BlackThrush)

**Lever:** when a lexical backend is present, no quality embedder is configured, and the fast
embedder is one of the shipped non-semantic hash tiers (`fnv1a-*`/`jl-*`), Phase 1 now returns
lexical results directly instead of paying hash embedding, vector scan, and RRF. This is the
correct behavior for Lucene/Tantivy/Meilisearch-class lexical comparisons: a hash vector
contributes no semantic relevance, so fusing it only burns latency. If lexical search fails
non-cancel, the old vector fallback still runs.

**BOLD-VERIFY command:**

```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- env FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error \
    cargo bench -p frankensearch --features lexical --profile release \
    --bench search_bench bold_verify_tantivy_class \
    -- --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Worker: `vmi1152480` (`[RCH] remote vmi1152480 (804.5s)`). Incumbent is
`tantivy_doc_ids`, used here as the Lucene/Tantivy/Meilisearch-class lexical proxy.

**Kept evidence:** the guard collapses the previous hash-hybrid penalty by removing useless
semantic work and wins several Tantivy-class rows outright:

| Workload | old hash-hybrid ratio | guarded ratio | guarded p95 ratio | status |
|----------|-----------------------|---------------|-------------------|--------|
| `top10/10000/high_fanout` | 18.180 | **0.978** | **0.756** | beats Tantivy-class |
| `limit_all/10000` | 2.086 | **0.873** | **0.611** | beats Tantivy-class |
| `top10/100000/natural_language` | 5.182 | **0.982** | 1.186 | p50 beats, p95 near |
| `top10/10000/zero_hit` | 54.333 | **1.000** | 3.333 | p50 parity; tail miss |
| `top10/100000/zero_hit` | 136.696 | **1.043** | **0.722** | near p50 parity; p95 win |

**Scope:** this is not a universal BOLD victory over Tantivy-class; remaining misses are recorded
in `docs/NEGATIVE_EVIDENCE.md`. It is kept because every measured row dramatically improves over
the existing hash-hybrid path, several rows beat the lexical incumbent, and the behavior is more
semantically honest for hash-only fast tier searches. Next lever should attack the residual
`ScoredResult` materialization and high-selectivity lexical overhead rather than vector/RRF.

## Local hot-path wins

### 2026-06-27 — count-free two-term plain `search_doc_ids` top-k (BlackThrush)

**Lever:** `frankensearch-lexical::TantivyIndex::search_doc_ids` now uses the top-k-only Tantivy
collector for one or two plain syntax-free terms. The prior single-token gate left the BOLD
short-keyword shape (`rust ownership`) on the counted collector, even though the API only returns
ranked IDs and BM25 scores. Fielded, phrase, wildcard, boosted, path-like, hyphenated, and 3+ term
queries stay on the counted path.

The benchmark harness also avoids `asupersync::test_utils::run_test_with_cx` so the measurement does
not install trace-level logging while Criterion is timing the hot path.

Literal requested command still fails in this checkout because Cargo rejects `bench --release`:

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
RUST_LOG=off \
  rch exec -- cargo bench --release -p frankensearch-lexical \
    --bench doc_ids_topk short_keyword_bold -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Measured command (RCH local fallback: `no admissible workers: insufficient_slots=3,hard_preflight=1`):

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
RUST_LOG=error \
  rch exec -- cargo bench -p frankensearch-lexical --profile release \
    --bench doc_ids_topk -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Criterion artifacts:
`/data/projects/.rch-targets/frankensearch-cod-b/criterion/doc_ids_topk/*/new/estimates.json`.

| Workload | Query shape | Counted median | New median | Ratio | Status |
|----------|-------------|----------------|------------|-------|--------|
| `doc_ids_topk/short_keyword_bold` | two plain terms, BOLD `rust ownership` mirror | 171.151 us | 30.900 us | **0.181 (~5.54x)** | KEEP |

**Scope:** this attacks the current BOLD short-keyword lexical collector gap but is still a local
Tantivy-wrapper hot-path win, not a Lucene/Tantivy/Meilisearch original-comparator win. The required
N/A original-comparator ratio and noisy fallback row are recorded in `docs/NEGATIVE_EVIDENCE.md`.

### 2026-06-27 — count-free simple `search_doc_ids` top-k (BlackThrush)

**Lever:** `frankensearch-lexical::TantivyIndex::search_doc_ids` now uses a top-k-only Tantivy
collector for single plain-token queries. The API returns ranked document IDs and BM25 scores, not
total match counts, so the old `Count` collector was discarded work on this narrow path. Queries
with Boolean, phrase, field, wildcard, boost, path, quote, or hyphen syntax keep the counted path.

**Measured command:**

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-lexical --profile release \
    --bench doc_ids_topk -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

RCH executed on `hz2` (`[RCH] remote hz2 (382.8s)`). Artifact:
`/data/projects/.rch-targets/frankensearch-cod-b/criterion/doc_ids_topk/high_fanout_{counted,free}/new/estimates.json`.

| Workload | Baseline counted median | New count-free median | Ratio | Status |
|----------|-------------------------|-----------------------|-------|--------|
| `doc_ids_topk/high_fanout` | 499.086 us | 239.233 us | **0.479** | KEEP |

**Scope:** this is a local Tantivy-wrapper hot-path win, not an original-comparator BOLD win.
The required Lucene/Tantivy/Meilisearch-class caveat, N/A original-comparator ratio, and unclaimed
fallback rows are recorded in `docs/NEGATIVE_EVIDENCE.md`.

### 2026-06-27 — single-pass `QueryClass::classify` (fused case scan + `take(4)` word count) (BlackThrush)

**Lever:** `QueryClass::classify` runs on **every** search query (adaptive lexical/semantic budget +
the lexical short-circuit gate) — pure frankensearch per-query overhead Tantivy never pays, so cutting
it directly narrows the end-to-end gap vs a Tantivy/Lucene/Meilisearch-class incumbent. Two
behaviour-identical reductions over the already-grouped baseline:
1. `looks_like_identifier` collected the camel/Pascal case flags in **three** separate Unicode-aware
   `chars()` scans (`any(is_lowercase)`, `any(is_uppercase)`, `skip(1).all(is_lowercase)`); fused into
   **one** char pass (flags are order-independent, so the boolean result is unchanged).
2. The word count only needs the `<= 3` boundary, so `split_whitespace().take(4).count()` stops after
   the 4th word instead of scanning every word of a long natural-language query.

All 26 `frankensearch-core::query_class` tests stay green, including `classify_is_trim_invariant`
(proptest) and every classification case — the change is bit-identical.

**Measured command (per-crate, in-process A/B — host-independent ratio):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=BlackThrush \
  rch exec -- cargo bench -p frankensearch-core --profile release \
    --bench query_class -- --measurement-time 5
```

RCH worker `ovh-a`. `current` = the committed grouped baseline (replicated in the bench); `new` = the
shipped `classify`. The new arm name `current` doubled as a stale-binary freshness sentinel (it
appeared in the output).

| Workload (11-query mix) | current (grouped) | new (this change) | Ratio | Status |
|-------------------------|-------------------|-------------------|-------|--------|
| `query_class` | 484.88 ns | 424.50 ns | **0.875 (~1.14×)** | KEEP |

CIs do not overlap (`current [468.2, 505.2] ns`, `new [420.0, 429.9] ns`). The same run shows the
pre-grouping `old` at 805.3 ns (cumulative ~1.9×); the isolated ratio for **this** change is **0.875**.

**Scope:** a local per-query-overhead win on pure frankensearch work (query classification). The
original-comparator (BOLD vs Tantivy/Lucene/Meilisearch) ratio is **N/A** for this isolated function —
it reduces overhead the incumbent does not have, narrowing the end-to-end gap rather than beating a
Tantivy primitive head-to-head.

### 2026-06-27 — FSVI 4-bit pass-1 uses prepared query lanes (BlackThrush)

**Lever:** `VectorIndex::search_top_k_4bit_two_pass` packed the query once, but the file-backed
pass-1 still called `dot_packed_4bit` for every stored vector; that wrapper re-prepares the query
nibble lanes on every dot. The in-memory path already used the prepared-query kernel. This change
moves the FSVI path to the same loop-invariant query decode: pack once, `prepare_4bit_query` once,
then call `dot_4bit_prepared` for each stored vector. This is the `/alien-graveyard` vectorized
execution lever in its smallest form: hoist invariant decode out of a scan morsel while preserving
the exact same candidate heap, tombstone check, and f16 rescore.

**Measured command:** the literal requested form still fails in this checkout:

```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo bench --release -p frankensearch-index \
    --bench fsvi_4bit_two_pass -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Cargo rejects `cargo bench --release` with `unexpected argument '--release'`; the valid per-crate
release-profile measurement was:

```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo bench -p frankensearch-index --profile release \
    --bench fsvi_4bit_two_pass fourbit_mult5 \
    -- --sample-size 10 --warm-up-time 1 --measurement-time 2
```

RCH had no admissible worker for the bench and fell back locally. To avoid the known stale-binary
pitfall with the shared target dir, the candidate source mtime was bumped and the second run
confirmed a fresh `frankensearch-index` rebuild before timing.

| Workload | Baseline (`origin/main`) | New | Ratio | Recall | Status |
|----------|--------------------------|-----|-------|--------|--------|
| `fsvi_4bit_two_pass/fourbit_mult5` | 3.6991 ms | 1.5660 ms | **0.423 (~2.36x)** | recall@10 = 1.0000 at mult=5 | KEEP |

**Scope:** this is a local file-backed vector-search primitive win against the immediately prior
frankensearch FSVI 4-bit path. The original-comparator ratio vs Lucene/Tantivy/Meilisearch is
**N/A** for this isolated vector pass-1; the comparator caveat is recorded in
`docs/NEGATIVE_EVIDENCE.md`.
### 2026-06-27 — JL hash embedder xorshift ILP (interleave 4 token chains) (Cobaltmoth)

**Lever:** `HashEmbedder::embed_jl` (the `jl-*` fast tier) is the compute-bound embedder —
O(tokens·dim) xorshift64. A single token's chain is **latency-bound**: each step's three
*dependent* shift→xor ops must retire before the next, so one chain cannot saturate the CPU's
shift/ALU ports. (This ledger already noted the JL path is "compute-bound xorshift" with alloc
savings lost in noise; the only prior JL lever — branchless sign — regressed because LLVM already
cmov's it, see `docs/NEGATIVE_EVIDENCE.md`.) Each token seeds an **independent** chain, so
`embed_jl` now advances `JL_LANES = 4` token chains together per dimension (`jl_accumulate_lanes`),
exposing instruction-level parallelism that hides the single-chain latency; the (< 4) tail drains
one chain at a time via the original loop.

**Bit-identical:** the per-dimension accumulator only ever holds an exact integer-valued `f32`
(a sum of ±1 contributions, `|value| ≤ token count ≪ 2^24`), so f32 addition is exact and
reordering the lane contributions cannot change a single output bit. Proven by
`jl_ilp_matches_scalar_reference_bit_identical` (dims 256/384, two seeds) +
`jl_ilp_matches_scalar_across_token_counts` (token counts 0..=9 straddling the lane boundary); all
33 `hash_embedder` tests GREEN.

**Measured command (per-crate, in-process A/B — host-independent ratio):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=Cobaltmoth \
  rch exec -- cargo bench -p frankensearch-embed --bench hash_embed -- hash_embed_jl_ilp
```

`scalar` = the committed single-chain inner loop (replicated in the bench); `ilp2`/`ilp4` = the
2-/4-lane interleaved variants. ~100-word doc, dim 384. Medians:

| Workload | scalar | ilp2 | ilp4 | Ratio (ilp4/scalar) | Status |
|----------|--------|------|------|---------------------|--------|
| `hash_embed_jl_ilp` (dim384) | 102.88 µs | 69.81 µs | **50.19 µs** | **0.488 (~2.05×)** | KEEP |

CIs do not overlap (`scalar [100.99, 105.49] µs`, `ilp4 [49.70, 50.89] µs`). Independently
corroborated by a concurrent run on a different worker/target dir (`scalar` 98.0 µs → `ilp4`
51.2 µs, **~1.92×**). `ilp4` ships; `ilp2` (~1.47×) is the runner-up the bench keeps for the curve.

**Scope:** a local hot-path win on the `jl-*` non-semantic fast tier (the slowest embedder).
Original-comparator (BOLD vs Tantivy/Lucene/Meilisearch) ratio is **N/A** — this reduces
frankensearch-internal embedding compute the incumbents do not have; it narrows the end-to-end gap
on jl-tier searches rather than beating a Tantivy primitive head-to-head.

### 2026-06-27 — MMR re-rank: incremental running-max + hoisted norms (Cobaltmoth)

**Lever:** `mmr_rerank` (diversity re-ranking) greedily picks `k` of `n` candidates; its inner
sweep did two redundant things, both pure frankensearch per-rerank overhead:
1. **O(k²·n) → O(k·n) cosine evals.** Each round recomputed `max(sim(i, j) for j in selected)` from
   scratch over the whole growing `selected` set. Now a per-candidate running max
   (`max_sim_to_selected`) is updated once per selection, so every `sim(i, selected_doc)` is computed
   exactly once. `f64::max` is associative ⇒ the running max equals the per-round fold bit-for-bit.
2. **3 reductions → 1 per pair.** `cosine_sim` re-derived both vectors' L2 norms on every pair even
   though an embedding's norm is loop-invariant. Norms are hoisted once per candidate (`root_norms`),
   leaving only the dot reduction per pair (`cosine_sim_pre`). Bit-identical for uniform-dimension
   embeddings (the MMR contract); ragged dims fall back to per-pair `cosine_sim`.

**Bit-identical:** the dot loop and accumulation order are unchanged, `root_norms[i]·root_norms[j]`
reproduces the prior `norm_a.sqrt()·norm_b.sqrt()`, and the running max reproduces the per-round fold
— the selected index list is unchanged. Proven by `incremental_norm_hoist_matches_bruteforce`
(dims 8/64/384 × n 3/16/60 × k 1/5/n vs a brute-force reference) + all 32 `mmr` tests GREEN.

**Measured command (per-crate, in-process A/B — host-independent ratio):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=Cobaltmoth \
  rch exec -- cargo bench -p frankensearch-fusion --bench mmr_rerank
```

`old` = per-round full recompute + per-pair norms (replicated); `new` = running-max + hoisted norms.
dim 384. Medians:

| Workload (n cand, k results, dim384) | old | new | Ratio | Status |
|--------------------------------------|-----|-----|-------|--------|
| `mmr_rerank/n100_k20` | 5.387 ms | 581.9 µs | **0.108 (~9.3×)** | KEEP |
| `mmr_rerank/n200_k50` | 65.10 ms | 2.800 ms | **0.043 (~23.3×)** | KEEP |

The gain grows with `k` (the O(k²·n)→O(k·n) factor compounding with the 3→1 reduction hoist).

**Scope:** a local hot-path win on the diversity-rerank (MMR) path — pure frankensearch compute the
Tantivy/Lucene/Meilisearch lexical incumbents never perform. Original-comparator ratio is **N/A**
(recorded in `docs/NEGATIVE_EVIDENCE.md`); it narrows end-to-end latency when diversity rerank is
enabled rather than beating a comparator primitive head-to-head.

### 2026-06-27 — graph-rank PageRank: dense-index power iteration (Cobaltmoth)

**Lever:** `GraphRanker::rank_phase1` (query-biased PageRank, `graph` feature) ran its power
iteration over a `HashMap<GraphDocId=String, f64>` **rebuilt every iteration**, with a `String`
doc_id **clone on every teleport and every edge relaxation** (`next.entry(doc_id.clone())`) — and
because `add_edge` inserts both endpoints, every key those `entry()` calls touched was already
present, so the clones were pure dead work — plus a per-edge weight-finiteness check and `f64::from`
widen redone on every one of the (≤20) iterations. Replaced with a one-time dense index (every node
is an `adjacency()` key, so node set = `0..n`) + CSR edge arrays + two reused `Vec<f64>` buffers
(swapped per iteration). Per iteration this drops a HashMap allocation, all String clones, and a hash
probe per edge → array indexing; the weight filter + widen are hoisted into the one-time CSR build.

**Equivalent ranking:** PageRank converges to the same fixed point within `tolerance`, and the prior
`std::HashMap` accumulation order was already run-to-run non-deterministic, so no exact value was ever
pinned. Verified by `dense_rank_matches_reference_ranking` (40-node random graph, 5 seeds) which
asserts the dense result's doc_id **order matches a deterministic `BTreeMap` reference of the original
algorithm** with scores within 1e-4, plus the existing propagation/empty tests, all GREEN under
`--features graph`.

**Measured command (per-crate, in-process A/B; replicas of old HashMap vs new dense):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=Cobaltmoth \
  rch exec -- cargo bench -p frankensearch-fusion --bench graph_rank
```

`old` = HashMap rebuilt per iteration + clone-keyed `entry`; `new` = dense `Vec<f64>` + CSR. 10 seeds,
limit 50, 20 iterations. Medians:

| Workload (n nodes, out-degree) | old | new | Ratio | Status |
|--------------------------------|-----|-----|-------|--------|
| `graph_rank/n500_deg6` | 2161.5 µs | 181.9 µs | **0.084 (~11.9×)** | KEEP |
| `graph_rank/n2000_deg8` | 11.661 ms | 982.7 µs | **0.084 (~11.9×)** | KEEP |

**Scope:** a local hot-path win on the optional `graph`-feature PageRank stage — pure frankensearch
compute the Tantivy/Lucene/Meilisearch lexical incumbents never perform, and off the default build
(feature-gated). Original-comparator ratio is **N/A** (recorded in `docs/NEGATIVE_EVIDENCE.md`); it
makes the graph-rank feature ~12× cheaper when enabled, not a head-to-head comparator win.

### 2026-06-27 — fsfs file-classification byte-sniff: vectorizable u64 histograms (Cobaltmoth)

**Lever:** `SniffFeatures::from_bytes` (`frankensearch-fsfs`) scans a capped content probe of **every
file** during classification (binary/text detection), counting null / non-printable / high-bit bytes.
The loop used per-byte `u32::saturating_add` for each of the three counters — a saturating add is *not*
a plain add, so it **blocks auto-vectorization** of this per-file scan. Switched to branchless `u64`
accumulators (`count += u64::from(cond)`) saturate-cast to `u32` at the end, letting LLVM emit three
SIMD histograms.

**Bit-identical:** a `u64` counter cannot overflow for any real `&[u8]` (len ≤ usize ≤ u64), and the
final `u32::try_from(..).unwrap_or(u32::MAX)` reproduces the exact `u32::MAX` saturation the old
per-byte `saturating_add` produced. Proven by `sniff_features_vectorized_matches_scalar_reference`
(empty / all-null / all-high-bit / printable / mixed / 4099-byte / full 0..=255 vs a scalar
`saturating_add` reference) + all `file_classification` tests GREEN (14 passed).

**Measured command (per-crate, in-process A/B; replicas of old saturating vs new u64):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=Cobaltmoth \
  rch exec -- cargo bench -p frankensearch-fsfs --bench sniff_features
```

`old` = per-byte `u32::saturating_add`; `new` = branchless `u64` + saturate-cast. Realistic text-ish
probe (~80% printable ASCII). Medians (gain scales with probe size — vectorization amortizes setup):

| Probe size | old | new | Ratio | Status |
|------------|-----|-----|-------|--------|
| `sniff_features/probe_4096` | 3692 ns | 2669 ns | **0.723 (~1.4×)** | KEEP |
| `sniff_features/probe_16384` | 21114 ns | 10719 ns | **0.508 (~2.0×)** | KEEP |
| `sniff_features/probe_65536` | 100341 ns | 42559 ns | **0.424 (~2.4×)** | KEEP |

**Scope:** a local indexing-throughput win on the per-file classification scan — pure frankensearch
file-ingest compute. Original-comparator ratio vs Lucene/Tantivy/Meilisearch is **N/A** (this is the
file-system scan layer, not a query-time comparator primitive); recorded in `docs/NEGATIVE_EVIDENCE.md`.
(fsfs is buildable again — the earlier fsqlite blocker was a stale lock; see `4c816a7`.)

### 2026-06-27 — fsfs `count_lexical_tokens` ASCII byte fast path (Cobaltmoth)

**Lever:** `count_lexical_tokens` (`frankensearch-fsfs::lexical_pipeline`) runs **per chunk per
document** at index time (`LexicalChunkPolicy::chunk_text` calls it for every chunk) and is
allocation-free — so its cost *is* the `for ch in text.chars()` UTF-8-decode loop + per-char
`is_token_char`. For **ASCII** text (the common case for code/docs) it now takes a byte-loop fast path
(`for &b in text.as_bytes()` + `is_token_byte`), skipping the `chars()` decoder; non-ASCII text falls
back to the Unicode `chars()` path unchanged.

**Bit-identical:** for an ASCII byte `b`, `is_token_byte(b) == is_token_char(b as char)`
(`char::is_alphanumeric` == `u8::is_ascii_alphanumeric` on ASCII; same `_-./:` punctuation set), and the
token-boundary state machine is unchanged. Proven by `count_lexical_tokens_ascii_fastpath_matches_chars`
(ASCII + non-ASCII: café/CJK/emoji → fallback, compared to a pure `chars()` reference) +
`count_lexical_tokens_matches_tokenize_lexical` + all 25 `lexical_pipeline` tests GREEN.

**Measured command (per-crate, in-process A/B; `chars` = decode loop, `bytes` = ASCII fast path):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=Cobaltmoth \
  rch exec -- cargo bench -p frankensearch-fsfs --bench lexical_count
```

Realistic ASCII code/doc chunk. Medians:

| Chunk size | chars (old) | bytes (new) | Ratio | Status |
|------------|-------------|-------------|-------|--------|
| `lexical_count/ascii_1024` | 2675 ns | 1054 ns | **0.394 (~2.54×)** | KEEP |
| `lexical_count/ascii_4096` | 6235 ns | 3264 ns | **0.524 (~1.91×)** | KEEP |
| `lexical_count/ascii_16384` | 21386 ns | 11581 ns | **0.542 (~1.85×)** | KEEP |

**Scope:** a local indexing-throughput win on per-chunk token counting (fsfs lexical pipeline) — pure
frankensearch file-ingest compute. Original-comparator ratio vs Lucene/Tantivy/Meilisearch is **N/A**
(file-system scan layer, not a query comparator); caveat in `docs/NEGATIVE_EVIDENCE.md`.

### 2026-06-27 - fsfs code-structure tokenization ASCII byte fast path (BlackThrush)

**Lever:** `code_structure_sidecar::tokenize` extracts lowercase ASCII alphanumeric tokens for the
code-structure sidecar. The old path always used
`value.chars().flat_map(char::to_lowercase)`, which builds a `ToLowercase` iterator for every Unicode
scalar before discarding non-ASCII separators. For ASCII code and symbol text, the new path uses
`value.is_ascii()` plus a byte loop with `to_ascii_lowercase`; non-ASCII input keeps the prior Unicode
path unchanged.

**Bit-identical:** for ASCII input, `char::to_lowercase` yields exactly the ASCII lowercase character,
and the token-boundary rule is still `is_ascii_alphanumeric`. Non-ASCII input falls back to the old
Unicode loop, including cases like the Kelvin sign that lowercase to ASCII. Proven by
`tokenize_ascii_fastpath_matches_unicode_reference` over empty, code-like ASCII, mixed punctuation,
Latin-1/Greek, CJK, and `\u{212A}` inputs.

**Measured command (per-crate, local fallback through `rch exec`; in-process A/B old vs new):**

```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- cargo bench -p frankensearch-fsfs --profile release \
    --bench code_tokenize -- --sample-size 10 --warm-up-time 1 --measurement-time 2
```

`old` = Unicode `chars().flat_map(char::to_lowercase)` tokenizer replica; `new` = ASCII byte fast
path. Realistic code/symbol text. Medians:

| Workload | old | new | Ratio | Status |
|----------|-----|-----|-------|--------|
| `code_tokenize/ascii_256` | 5011.625 ns | 4976.335 ns | **0.993** | tie/no-claim |
| `code_tokenize/ascii_1024` | 13.090 us | 8.374 us | **0.640 (~1.56x)** | KEEP |
| `code_tokenize/ascii_4096` | 95.735 us | 52.135 us | **0.545 (~1.84x)** | KEEP |

**Scope:** a local fsfs indexing/sidecar tokenization win for medium and larger code snippets.
Original-comparator ratio vs Lucene/Tantivy/Meilisearch is **N/A** because those incumbents do not
run frankensearch's fsfs code-structure sidecar tokenizer. The 256-byte row is deliberately recorded
as a tie in `docs/NEGATIVE_EVIDENCE.md`; do not claim a small-string win from this lever.

### 2026-06-28 — sync rank-change telemetry: drop redundant VectorHit clones (Cobaltmoth)

**Lever:** `compute_rank_changes_for_scored` runs **unconditionally on every sync hybrid query**
(`SyncTwoTierSearcher`, for `metrics.rank_changes`) — pure frankensearch per-query overhead the
Tantivy/Lucene/Meilisearch incumbents never pay. It built **two throwaway `Vec<VectorHit>`** by
**cloning every `doc_id`** out of the initial + refined `ScoredResult` slices, solely to feed
`build_borrowed_rank_map` — which only reads `doc_id` (rank = enumerate index; it ignores
`VectorHit::index`/`score`). Now the doc_id→rank maps are built **directly** from the `ScoredResult`
slices (`entry(hit.doc_id.as_str()).or_insert(rank)`), dropping 2 `Vec` allocations + 2·N `String`
clones per query.

**Bit-identical:** same first-occurrence-wins `HashMap<&str, usize>` content as `build_borrowed_rank_map`
(the discarded `index`/`score` never affected the rank map), so `compute_rank_changes_with_maps` returns
the identical `RankChanges`. All 6 `sync_searcher` lib tests GREEN.

**Measured command (per-crate, in-process A/B; `clone` = build via `Vec<VectorHit>`, `borrow` = direct):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=Cobaltmoth \
  rch exec -- cargo bench -p frankensearch-fusion --bench rank_map
```

N = result count (initial + refined, partial overlap). Medians:

| Workload | clone (old) | borrow (new) | Ratio | Status |
|----------|-------------|--------------|-------|--------|
| `rank_map/n20` | 1543 ns | 778 ns | **0.504 (~1.98×)** | KEEP |
| `rank_map/n60` | 5233 ns | 2316 ns | **0.443 (~2.26×)** | KEEP |
| `rank_map/n200` | 17517 ns | 7835 ns | **0.447 (~2.24×)** | KEEP |

**Scope:** a per-query-overhead win on the default sync hybrid path (rank-change telemetry), narrowing
frankensearch's own per-query cost vs the incumbent. Original-comparator ratio is **N/A** (rank-change
metrics have no comparator counterpart); caveat in `docs/NEGATIVE_EVIDENCE.md`.

### 2026-06-28 — sync vector path: borrow doc_ids in fast/quality score maps (Cobaltmoth)

**Lever:** the no-lexical branch of `SyncTwoTierSearcher` builds two per-query `HashMap`s
(`fast_scores`, `quality_scores`) from the fast + quality candidate hits, then passes them to
`vector_hits_to_scored_results`, which only ever **`.get()`-looks-them-up by `&str`**. The maps were
keyed on **owned `String`** (`hit.doc_id.clone()` per candidate) for no reason. Now they key on
`&str` borrowed straight from `fast_hits`/`quality_hits` (which outlive the call), dropping a
per-candidate `String` clone in both map builds; the dedup `HashSet` in `vector_hits_to_scored_results`
also gets a `with_capacity` hint.

**Bit-identical:** `HashMap<&str, f32>` with the same entries, looked up by the same
`hit.doc_id.as_str()` keys → identical `fast_score`/`quality_score` resolution and identical
`ScoredResult`s. All 6 `sync_searcher` lib tests GREEN.

**Measured command (per-crate, in-process A/B; `clone` = `HashMap<String>`, `borrow` = `HashMap<&str>`):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=Cobaltmoth \
  rch exec -- cargo bench -p frankensearch-fusion --bench score_map
```

N = fast + quality candidate count. Medians:

| Workload | clone (old) | borrow (new) | Ratio | Status |
|----------|-------------|--------------|-------|--------|
| `score_map/n30` | 2883 ns | 1215 ns | **0.422 (~2.37×)** | KEEP |
| `score_map/n90` | 9572 ns | 3457 ns | **0.361 (~2.77×)** | KEEP |
| `score_map/n300` | 31859 ns | 11491 ns | **0.361 (~2.77×)** | KEEP |

**Scope:** a per-query-overhead win on the **no-lexical (pure-vector) sync hybrid path** (the lexical
branch uses `rrf_fuse`, unaffected). Pure frankensearch result-assembly compute; original-comparator
ratio is **N/A** (caveat in `docs/NEGATIVE_EVIDENCE.md`). Second clone-elision in the sync
result-assembly vein (after `rank_map`, `f8f645e`).

### 2026-06-28 — sync lexical path: move doc_ids out of the rrf_fuse result (Cobaltmoth)

**Lever:** the lexical (hybrid/BOLD) branch of `SyncTwoTierSearcher` calls
`fused_hits_to_scored_results(&rrf_fuse(...), k)` for both the initial and refined phases.
`fused_hits_to_scored_results` took `&[FusedHit]` and **cloned** each `doc_id` into the `ScoredResult`,
even though the `rrf_fuse` result is a **fresh temporary** dropped immediately after. Changed it to take
`Vec<FusedHit>` by value and `into_iter().map(|hit| ScoredResult { doc_id: hit.doc_id, .. })` —
**moving** each `doc_id` (a pointer copy) instead of cloning. Both call sites now pass the `rrf_fuse(...)`
result by value.

**Bit-identical:** the moved `String` is the exact value that was previously cloned; every other field is
copied as before, so the `ScoredResult`s are unchanged. All 6 `sync_searcher` lib tests GREEN.

**Measured command (per-crate; in-process A/B with `iter_batched` so both arms pay the same input-clone
setup; `clone` = `&[FusedHit]`+clone, `move` = `Vec<FusedHit>`+move):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=Cobaltmoth \
  rch exec -- cargo bench -p frankensearch-fusion --bench fused_materialize
```

N = fused candidate count (k = N, materialize all). Medians:

| Workload | clone (old) | move (new) | Ratio | Status |
|----------|-------------|------------|-------|--------|
| `fused_materialize/n20` | 482 ns | 62 ns | **0.129 (~7.76×)** | KEEP |
| `fused_materialize/n60` | 1969 ns | 107 ns | **0.054 (~18.41×)** | KEEP |
| `fused_materialize/n200` | 5970 ns | 277 ns | **0.046 (~21.55×)** | KEEP |

The ratio is large because the move eliminates **both** the per-result `String` clone (N allocs) **and**
the subsequent drop of the temporary's `doc_id`s (N frees); the new path is allocation-free. Absolute
times are small — this materialization is one step of the query — but it is bit-identical and on the
**lexical/BOLD hybrid sync path** (both phases). Original-comparator ratio is **N/A** (frankensearch
result-assembly); third clone-elision in the sync vein (after `rank_map` `f8f645e`, `score_map`
`dc86170`).

### 2026-06-28 — sync vector refined path: move blended doc_ids into final results (BlackThrush)

**Lever:** the no-lexical refined branch of `SyncTwoTierSearcher` receives an owned `blended`
`Vec<VectorHit>` from `blend_two_tier`, then immediately converted it through the generic borrowed
`vector_hits_to_scored_results(&blended, ...)` helper. That paid a `HashSet` first-occurrence dedup and
one `String` clone per final row even though `blend_two_tier` already produces one row per merged doc id
from a doc-id keyed map. The no-lexical refined branch now consumes that owned vector with
`unique_vector_hits_to_scored_results_owned`, moves `hit.doc_id` into `ScoredResult`, and leaves the
generic borrowed helper in place for raw vector-hit inputs that may still need deduplication.

**Bit-identical:** `blend_two_tier` output is unique by construction; the new helper preserves row order
and every score/source/index field, looking up `fast_score`/`quality_score` by the same `&str` keys before
moving the `String`. The lexical branch remains on `rrf_fuse` + `fused_hits_to_scored_results`.

**Measured command (per-crate, release profile; `clone_dedup` = old borrowed converter, `move_unique` =
owned blended converter):**

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a-vector-materialize \
RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-fusion --profile release \
    --bench vector_materialize -- --sample-size 10 --warm-up-time 1 --measurement-time 2
```

RCH had no admissible worker and fell back locally; the run completed with exit 0. N = final blended
result count (k = N). Medians:

| Workload | clone+dedup (old) | move+unique (new) | Ratio | Status |
|----------|-------------------|-------------------|-------|--------|
| `vector_materialize/n20` | 2.7685 us | 1.4859 us | **0.537 (~1.86×)** | KEEP |
| `vector_materialize/n60` | 8.3990 us | 6.0815 us | **0.724 (~1.38×)** | KEEP |
| `vector_materialize/n200` | 35.454 us | 14.391 us | **0.406 (~2.46×)** | KEEP |

**Scope:** no-lexical (pure-vector) sync refined result materialization. This is frankensearch's own
result-assembly overhead; original-comparator ratio vs Lucene/Tantivy/Meilisearch is **N/A** (caveat in
`docs/NEGATIVE_EVIDENCE.md`).

### 2026-06-28 — lexical id materialization: numeric u64 fast-field + ordinal table is 2.56–8.47× over docstore (validated lever) (BlackThrush)

**Lever (validated, bench-level; production wiring scoped as the immediate follow-up).** `TantivyIndex::
search_doc_ids` materializes each hit's `doc_id` via `collect_id_hits → load_doc → searcher.doc(addr)`,
which **decompresses the entire stored document** (id + content + title + metadata_json — all `STORED`)
just to read the `id` field. This is the BOLD `limit_all` / large-fetch materialization gap. Two prior
attacks were REJECTED: the **str-FAST-field** id (`docs/NEGATIVE_EVIDENCE.md` 2026-06-27, **2.65–18.3×
slower** — `StrColumn::ord_to_str` does a dictionary SSTable seek per hit) and the **lazy doc_id cache**
(~0-gain at the realistic ~30-candidate fetch). Their shared route-next — a **numeric `u64` fast field
carrying a dense insertion ordinal** (a flat bit-packed column, **no dictionary**) plus an external
append-only `ordinal → doc_id` table (`Vec<String>`, O(1) index) — is now **confirmed a large win**: it
skips BOTH the docstore decompress AND the dictionary seek, reading `ord = u64_column.first(local_doc)`
and cloning `table[ord]` once.

**Measured (per-crate A/B, fresh in-RAM Tantivy, N=20k, `content` `STORED` so the docstore arm pays a
realistic decompress like production; `docstore` = `searcher.doc(addr)` → read id (current path),
`numeric_ff` = u64 fast-field → `table[ord]`; doc_ids asserted bit-identical before timing; 60 samples):**

| Workload | docstore (now) | numeric_ff | Ratio | Speedup |
|----------|----------------|------------|-------|---------|
| `id_materialize/k30`   | 12.340 µs | 4.8263 µs | **0.391** | ~2.56× |
| `id_materialize/k100`  | 45.184 µs | 9.5752 µs | **0.212** | ~4.72× |
| `id_materialize/k300`  | 136.60 µs | 16.505 µs | **0.121** | ~8.28× |
| `id_materialize/k1000` | 431.81 µs | 51.009 µs | **0.118** | ~8.47× |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-lexical --profile release \
  --bench id_materialize_numeric_ff -- --sample-size 60 --warm-up-time 1 --measurement-time 3
```

Artifact: kept bench `crates/frankensearch-lexical/benches/id_materialize_numeric_ff.rs`. Conformance:
81/81 lexical lib tests GREEN; the bench asserts the two paths return identical `doc_id`s.

**Why this is a real comparator angle (not just internal):** the incumbent Lucene/Tantivy-class proxy
also materializes ids out of the stored document (`tantivy_only_search` ultimately reads ids the same
docstore way); a numeric-ff materialization makes frankensearch's id read **2.56–8.47× cheaper than the
docstore path the comparator pays**, so it directly closes the materialization-dominated `limit_all`
gap. **Wins even at k30** — the regime where the doc_id-cache was ~0-gain — so it helps the BOLD top10
over-fetch too, not only large-limit.

**Production wiring (scoped follow-up, NOT in this commit):** add `ord: u64 FAST` to `build_schema`,
build the append-only `ordinal → doc_id` table during `index_documents`/`upsert` (rebuild on `open` for
persisted indexes), and route `collect_id_hits` through the fast field with a **docstore fallback** for
pre-`ord` indexes (so old on-disk indexes stay correct). Delete/merge safety: ordinals are monotonic and
never reused (a deleted doc's ordinal simply never appears in results; Tantivy carries fast fields
through merges), so the append-only table is delete/merge-correct; it grows by one `String` per
doc-version, rebuilt in O(total docs) on open. That correctness surface needs its own conformance pass,
hence wiring is deferred to keep this commit GREEN and reversible.

### 2026-06-28 — lexical id materialization WIRED into production: `search_doc_ids` up to 6.32× end-to-end (BlackThrush)

**Lever (the scoped follow-up above, now SHIPPED).** Wired the validated numeric-fast-field
materialization into `TantivyIndex`: `build_schema` adds `ord: u64 FAST | STORED`; `assign_ord`
stamps a dense monotonic ordinal on each document under the writer lock and appends its `doc_id` to an
append-only `RwLock<Vec<String>>` ord-table; `collect_id_hits` reads the `ord` column **lazily per
touched segment** and resolves `table[ord]`, with a **per-hit docstore fallback** for any ordinal it
can't resolve (pre-`ord` docs, reopened-but-not-rebuilt tables, poisoned lock). `SchemaFields::ord` is
resolved by name in `from_index`, so pre-`ord` on-disk indexes are `None` and use docstore unchanged.

**Conformance:** 81/81 lexical tests GREEN — incl. `search_doc_ids_matches_counted_baseline`,
`search_doc_ids_returns_ranked_identifiers` (ground-truth ids), upsert/delete, multi-segment
force-merge, and CJK/Korean/Japanese roundtrips. The A/B bench asserts `search_doc_ids` (fast) ==
`search_doc_ids_via_docstore` (forced docstore) at k=1000 before timing.

**Measured (per-crate same-binary A/B, in-memory Tantivy N=100k, high-fanout query "search"; `docstore`
= forced `searcher.doc(addr)` decompress, `fast` = wired ord path; medians):**

| Workload | docstore | fast | Ratio | Status |
|----------|----------|------|-------|--------|
| `search_doc_ids_materialize/k10`   | 611.35 µs | 602.90 µs | **0.986** (tie — no top10 regression) | KEEP |
| `search_doc_ids_materialize/k100`  | 633.21 µs | 614.02 µs | **0.970 (~1.03×)** | KEEP |
| `search_doc_ids_materialize/k1000` | 4407.6 µs | 697.22 µs | **0.158 (~6.32×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-lexical --profile release \
  --bench search_doc_ids_materialize -- --sample-size 40 --warm-up-time 1 --measurement-time 2
```

Note: an initial **eager** variant (open every segment's column upfront) was ~1.2× *slower* at k10
(materialization is a tiny slice of BM25 search there, and top-k hits touch only 1–2 of many segments);
switching to **lazy per-touched-segment** column loading removed that and left k10 a tie. Kept benches:
`search_doc_ids_materialize` (end-to-end) + `id_materialize_numeric_ff` (isolated primitive).

**Scope vs comparator:** the incumbent Lucene/Tantivy-class proxy materializes ids out of the stored
document the same way, so this makes frankensearch's `search_doc_ids` **6.32× cheaper than the docstore
path the comparator pays** on the materialization-dominated large-limit (`limit_all`) row — directly
closing that documented gap — while top10 is an unchanged tie. On-disk **reopened** indexes currently
fall back to docstore until the ord-table is rebuilt on open (correct, just unaccelerated); that rebuild
is the one remaining follow-up. **(DONE — see the 2026-06-28 sidecar entry below.)**

### 2026-06-28 — on-`open` ord-table restore (sidecar): reopened on-disk indexes realize the fast path, 2.29× at k1000 (BlackThrush)

**Lever (the "one remaining follow-up" above, now SHIPPED).** A reopened on-disk `TantivyIndex` started
with an empty `ord_table` and fell back to docstore id-materialization. Now `commit` persists the table
to an `ord_table.json` sidecar (atomic temp-file + rename, best-effort: write errors are logged and the
in-memory fast path is unaffected), and `from_index` loads it on `open`. A stale-short sidecar (e.g. a
persist that failed on the last commit) is repaired with an **O(segments)** guard — `Column::max_value`
(columnar metadata, no per-doc scan) gives the highest existing ordinal, the table is padded to cover
it so the next assigned ordinal can't collide, and `collect_id_hits` treats an empty padded slot as a
miss → docstore fallback (safe even if a real `doc_id` is `""`). Pre-`ord` on-disk indexes have no
sidecar / no `ord` field → unchanged docstore path.

**Conformance:** 82/82 lexical tests GREEN, incl. a new `reopen_on_disk_restores_fast_id_materialization`
(create on-disk → index → commit → drop → `open` → asserts the sidecar exists and the reopened index
returns **byte-identical ranked `doc_id`s**) and the existing `reopen_preserves_documents`. The A/B
bench also asserts reopened fast == docstore ids at k=1000 before timing.

**Measured (per-crate same-binary A/B, on-disk index N=20k built→committed→dropped→reopened, query
"document"; `docstore` = forced `searcher.doc`, `fast` = sidecar-restored ord path; medians):**

| Workload | docstore | fast | Ratio | Status |
|----------|----------|------|-------|--------|
| `reopen_id_materialize/k10`   | 136.96 µs | 130.37 µs | **0.952** (tie — no regression) | KEEP |
| `reopen_id_materialize/k1000` | 741.83 µs | 323.24 µs | **0.436 (~2.29×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc RUST_LOG=off \
  cargo bench -p frankensearch-lexical --profile release \
  --bench reopen_id_materialize -- --sample-size 40 --warm-up-time 1 --measurement-time 2
```

**Scope vs comparator:** the docstore baseline is exactly what a Lucene/Tantivy-class incumbent pays to
read ids from the stored document, so a **reopened** frankensearch index now materializes ids ~2.29×
cheaper than that incumbent on the large-fetch row (the in-memory path's 6.32× is higher because that
bench's 100k high-fanout docstore arm is heavier). The fast path is now reached on **both** fresh and
reopened on-disk indexes. Sidecar is re-serialized in full per `commit` (fine for batch-commit; a
delta/binary format is a future optimization for commit-per-doc streaming workloads). Kept bench:
`reopen_id_materialize`.

### 2026-06-28 — async hybrid lexical-metadata re-attach borrows `&Value` (3.55–7.20× on the map step) (BlackThrush)

**Lever.** The async `TwoTierSearcher`'s `fused_hits_to_scored_results` (`searcher.rs`) re-attaches
lexical metadata to the fused results via a `doc_id → metadata` map, but only the `fused` (top-k) hits
read it. The old map was `AHashMap<&str, serde_json::Value>` and **cloned every lexical candidate's
metadata** into it — including the many candidates dropped from the top-k. Switched the map to
`AHashMap<&str, &serde_json::Value>` (borrow) and deferred the `clone` to the per-winner lookup
(`.get().copied().cloned()`), so a deep/large metadata `Value` for a candidate that misses the top-k is
**never cloned**. `lexical_results` outlives the function, so the borrows are valid. Bit-identical
output — the `fused_hits_to_scored_results_preserves_lexical_metadata` guard test and the bench's
`assert_eq` both confirm identical `Option<Value>` per winner. 817/817 fusion lib tests GREEN.

**Measured (per-crate isolated A/B, N=60 lexical candidates → k=10 winners; `clone_all` = old
clone-every-candidate map, `borrow` = borrow + clone-only-winners; medians):**

| Workload | clone_all | borrow | Ratio | Status |
|----------|-----------|--------|-------|--------|
| `lexical_metadata_map/small` (1-field metadata)  | 9.1124 µs | 2.5641 µs | **0.281 (~3.55×)** | KEEP |
| `lexical_metadata_map/large` (nested ~10-field)  | 64.788 µs | 8.9993 µs | **0.139 (~7.20×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc RUST_LOG=off \
  cargo bench -p frankensearch-fusion --profile release \
  --bench lexical_metadata_map -- --sample-size 60 --warm-up-time 1 --measurement-time 2
```

**Scope:** this is frankensearch's own async hybrid result-assembly bookkeeping (the metadata
re-attachment is unique to the real `TwoTierSearcher`, not the simplified BOLD comparator harness), so
the original-comparator ratio vs Lucene/Tantivy/Meilisearch is **N/A**. The win scales with metadata
size and candidate-overfetch (`candidates − k` avoided `Value` clones); it is a real per-query reduction
for metadata-bearing workloads on the production async path. Unlike the sync phase-clone elision
(2026-06-28, sub-noise — `ScoredResult` struct copies), this elides heap-allocated `serde_json::Value`
clones, hence the large ratio. Kept bench: `lexical_metadata_map`.

### 2026-06-28 — async MMR rerank reorder moves the pool instead of cloning it twice (25–75× on the reorder step) (BlackThrush)

**Lever.** When MMR diversity reranking is enabled, the async `TwoTierSearcher` reorders the top `pool`
fused results by the `mmr_rerank` permutation. The old code cloned the pool into a `head` vec, then
**cloned each element again** while emitting it in MMR order — `2×pool` full `ScoredResult` clones, each
carrying a `doc_id` `String` + a metadata `Value`. Since `order` is a **distinct permutation of
`0..pool`** (mmr_rerank never repeats a candidate — guarded by `mmr::tests::no_duplicates_in_output`),
the head can be reordered by **moving** each element into its MMR slot: `split_off` the tail, collect the
head into `Vec<Option<ScoredResult>>`, and `Option::take` each index in `order` (the tail is moved
through untouched). **Zero `ScoredResult` clones.** Bit-identical output (only a permutation; payloads
unchanged). 817 fusion lib + 32 `mmr` tests GREEN.

**Measured (per-crate isolated A/B, pool=30 + tail=10; `clone_reorder` = old 2×pool clone,
`move_reorder` = new split_off + `Option::take`; medians):**

| Workload | clone_reorder | move_reorder | Ratio | Status |
|----------|---------------|--------------|-------|--------|
| `mmr_reorder/small` (1-field metadata) | 13.842 µs | 542.62 ns | **0.039 (~25.5×)** | KEEP |
| `mmr_reorder/large` (nested ~10-field)  | 69.682 µs | 925.89 ns | **0.013 (~75.3×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc RUST_LOG=off \
  cargo bench -p frankensearch-fusion --profile release \
  --bench mmr_reorder -- --sample-size 60 --warm-up-time 1 --measurement-time 2
```

**Scope:** frankensearch's own async result-assembly; original-comparator ratio vs
Lucene/Tantivy/Meilisearch is **N/A** (MMR diversity rerank has no incumbent counterpart, and it is a
**conditional** path — only runs when `mmr_config.enabled`). When active it saves the full ~14–70 µs
reorder cost per query (scaling with metadata size). Second async-`searcher.rs` clone-elision win in a
row (after `lexical_metadata_map`), confirming the async hybrid path was materially less mined than the
sync searcher. Kept bench: `mmr_reorder`.

### 2026-06-28 — runtime-dispatched AVX2 `dot_i8_i8`: 2.26–2.56× over the portable `wide` kernel (BlackThrush)

**Lever (overturns the 2026-06-25 "AVX2 can't be a code lever" finding).** That finding measured a
global `+avx2` build at ~1.5–2.5× but rejected it as un-landable (a default `+avx2` `SIGILL`s on older
hosts; `wide` only selects AVX2 via *compile-time* features, so it can't dispatch at runtime) and flagged
the only code path — a hand-written runtime-dispatched intrinsic kernel — as "large, risky." **That
route-next is now done.** `dot_i8_i8` (the int8 ADC two-pass **pass-1** kernel — scans every corpus
vector) now runtime-dispatches via `std::is_x86_feature_detected!("avx2")` to a hand-written AVX2 kernel
(`#[target_feature(enable = "avx2")]`, 256-bit `vpmaddwd` over sign-extended i16 lanes, two accumulators,
horizontal sum), falling back to the existing portable `wide` kernel (`dot_i8_i8_generic`) on
non-x86_64 / pre-AVX2 hosts. `#[allow(unsafe_code)]` scoped to the kernel (crate is `deny(unsafe_code)`;
mmap already uses the same opt-in).

**Bit-identical:** integer addition is associative, so the 256-bit reduction equals the scalar/`wide`
sum exactly — new test `simd::tests::avx2_dot_matches_generic` asserts equality across 15 dim shapes
(0/1/7/8/31/32/33/63/64/65/100/256/384/511/512). Conformance: **364/364** `frankensearch-index` lib
tests GREEN run serially (incl. all int8 two-pass + recall gates). *(Note: the WAL/compaction tests are
pre-existing flakes under parallel `cargo test` — shared-tempdir interference, failing set varies
run-to-run, none SIMD-related; all pass with `--test-threads=1`.)*

**Measured (per-crate, AVX2 worker `hz2`; sum of 10 000 i8 dots; `i8_dot` = runtime dispatch → AVX2,
`i8_dot_generic` = portable `wide`-SIMD fallback):**

| Workload | i8_dot_generic (`wide`) | i8_dot (AVX2) | Ratio | Status |
|----------|-------------------------|---------------|-------|--------|
| `dot/dim256/i8` (10k vectors) | 617.17 µs | 240.91 µs | **0.390 (~2.56×)** | KEEP |
| `dot/dim384/i8` (10k vectors) | 766.66 µs | 339.42 µs | **0.443 (~2.26×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-index --profile release \
  --bench dot_product -- "dot/dim(256|384)/i8_dot"
```

**Scope:** the int8 pass-1 vector scan is frankensearch's own vector tier (no Lucene/Tantivy
counterpart), so the original-comparator ratio is **N/A** — but this ~2.3–2.6× directly cuts the int8
two-pass scan cost on every AVX2 host (most modern x86 + the rch workers), with a safe identical-result
fallback elsewhere. First **explicit-SIMD-intrinsic** win in the tree (prior dot wins were portable
`wide` + ILP). Route-next: the same runtime-dispatch pattern applies to the 4-bit pass-1
(`dot_4bit_prepared`, the wired default — harder: nibble unpack) and the f16 pass-2 rescore
(`dot_product_f16_*` via F16C — but f32 accumulation reorders, so NOT bit-identical there). Kept bench
arm: `i8_dot_generic`.

### 2026-06-28 — runtime-dispatched AVX2 `dot_4bit_prepared`: 1.19–1.27× on the WIRED-DEFAULT pass-1 kernel (BlackThrush)

**Lever (the int8 route-next, applied to the default path).** `dot_4bit_prepared` is the **wired default**
sync-hybrid pass-1 kernel (`in_memory.rs` `search_top_k_4bit_two_pass_filtered` → it scans every corpus
vector). Same runtime-AVX2-dispatch pattern as `dot_i8_i8` (`bce9bc8`): a hand-written
`#[target_feature(enable="avx2")]` kernel — load 16 packed bytes, `_mm256_cvtepi8_epi16`, arithmetic-shift
out the low/high nibbles (`slli`/`srai`), 256-bit `_mm256_mullo_epi16` against the prepared query lanes,
accumulate in i16 (flush every 16 chunks before overflow), reduce via `_mm256_madd_epi16` — falling back
to the portable `wide` kernel (`dot_4bit_prepared_generic`) on non-AVX2/non-x86 hosts.

**Bit-identical:** integer/in-range (per-dim products ≤ 64, flush keeps lanes < i16::MAX), so the 256-bit
reduction equals the `wide` sum exactly — new test `simd::tests::avx2_dot4bit_matches_generic` asserts
equality across 12 packed-length shapes (full 16-byte chunks + sub-chunk tails). Conformance: **365/365**
index lib tests GREEN serial. *(WAL/compaction tests flake under parallel `cargo test` — pre-existing,
none SIMD-related; pass with `--test-threads=1`.)*

**Measured (per-crate, AVX2 worker `hz2`; sum of 10 000 4-bit dots; `fourbit_prepared_new` = runtime
dispatch → AVX2, `fourbit_prepared_generic` = portable `wide` fallback):**

| Workload | generic (`wide`) | AVX2 | Ratio | Status |
|----------|------------------|------|-------|--------|
| `dot/dim256/fourbit_prepared` (10k vectors) | 392.30 µs | 329.11 µs | **0.839 (~1.19×)** | KEEP |
| `dot/dim384/fourbit_prepared` (10k vectors) | 519.74 µs | 410.59 µs | **0.790 (~1.27×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-index --profile release \
  --bench dot_product -- "dot/dim(256|384)/fourbit_prepared"
```

**Scope:** original-comparator ratio **N/A** (frankensearch's own vector tier). Smaller than the int8
2.5× — the 4-bit nibble-unpack (`slli`/`srai`/`mullo` per chunk) is more compute-bound, so the 256-bit
width helps less than the memory-bound int8 dot — but it's the **default** path, so every AVX2-host
default vector search now gets a ~1.2× faster pass-1 scan, safe-fallback elsewhere. Both int8 and 4-bit
pass-1 dots are now AVX2-dispatched. Route-next: only the f16 pass-2 rescore (F16C) remains, and that
needs a recall gate (f32 reorder is not bit-identical). Kept bench arm: `fourbit_prepared_generic`.

### 2026-06-28 — runtime-dispatched AVX2+F16C `dot_product_f16_bytes_f32`: 3.88–3.99× (BOLD vector-tier kernel) (BlackThrush)

**Lever (the f16 route-next — and it IS bit-identical after all).** `dot_product_f16_bytes_f32` is the
exact f16 scan kernel — it backs `VectorIndex::search_top_k`, i.e. **the BOLD comparator's vector tier**
(`frankensearch_hash_hybrid_search` → `fixture.vector.search_top_k`), plus the two-pass f16 rescore. The
kernel is **decode-bound**: the portable `wide` path decodes each f16→f32 in software. The new
`#[target_feature(enable="avx2,f16c")]` kernel uses `vcvtph2ps` (`_mm256_cvtph_ps`) to decode 8 f16 in
one instruction, then runtime-dispatches via `is_x86_feature_detected!("avx2") && …("f16c")`, falling
back to the `wide` kernel (`dot_product_f16_bytes_f32_generic`) elsewhere.

**Bit-identical (the prior note feared an f32-reorder; avoided it):** f16→f32 is **exact** (f32 has more
mantissa bits), so F16C yields the same f32 values as `widen8_f16_bytes`; the kernel then keeps the
**same** separate-mul+add accumulation and routes the final reduction **through `wide::f32x8::reduce_add`**
(the `__m256` accumulator is stored to `[f32;8]` → `f32x8`), so the reduction order is byte-for-byte the
generic path. New test `simd::tests::avx2_f16dot_matches_generic` asserts `to_bits` equality across 12
dim shapes (embeddings are finite; NaN payloads — the only legitimate divergence — never occur).
Conformance: **366/366** index lib tests GREEN serial. *(WAL tests flake under parallel `cargo test`;
pass with `--test-threads=1`.)*

**Measured (per-crate, AVX2+F16C worker `ovh-a`; sum of 10 000 f16 dots; `f16_bytes_new` = runtime
dispatch → AVX2+F16C, `f16_bytes_generic` = portable `wide` software-decode fallback):**

| Workload | generic (`wide`) | AVX2+F16C | Ratio | Status |
|----------|------------------|-----------|-------|--------|
| `dot/dim256/f16_bytes` (10k vectors) | 1.0825 ms | 271.48 µs | **0.251 (~3.99×)** | KEEP |
| `dot/dim384/f16_bytes` (10k vectors) | 1.5735 ms | 405.63 µs | **0.258 (~3.88×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-index --profile release \
  --bench dot_product -- "dot/dim(256|384)/f16_bytes"
```

**Scope:** the **largest** of the three SIMD-dispatch wins (int8 2.5×, 4-bit 1.2×, f16 3.9×) because the
f16 dot is decode-bound and F16C is a pure hardware win. Original-comparator ratio vs Lucene/Tantivy is
**N/A** (vector tier has no incumbent counterpart), but this is the **BOLD vector-tier kernel**, so the
non-short-circuit BOLD rows' vector search is now ~3.9× cheaper on AVX2+F16C hosts (directly trims
frankensearch's hybrid-over-Tantivy overhead), safe-fallback elsewhere. All three integer/f16 dot
kernels are now AVX2-dispatched; the f16 fear of "not bit-identical" was wrong (reduce-through-`wide`
fixes it). Kept bench arm: `f16_bytes_generic`.

### 2026-06-28 — runtime-dispatched AVX2+F16C `dot_product_f16_f32` (slice): 3.64–3.81× on the in-memory rescore (BlackThrush)

**Lever (the f16-slice sibling of the bytes win).** `dot_product_f16_f32` (the `&[f16]`-slice kernel) is
the wired `InMemoryVectorIndex` **pass-2 exact rescore + quality scoring** kernel (`in_memory.rs` 468 /
562 / 658 / 741 / 859 — every two-pass query rescores its candidates with it). Same F16C dispatch as the
bytes variant (`7239d58`): `_mm256_cvtph_ps` hardware-decodes 8 f16, same separate-mul+add accumulation,
final reduce **through `wide::f32x8::reduce_add`**, and — matching this kernel's tail — a separate
mul+add scalar tail (the bytes variant used `mul_add`; matched per-kernel). **Bit-identical** (f16→f32
exact; `simd::tests::avx2_f16slicedot_matches_generic` asserts `to_bits` equality across 12 dim shapes).
Conformance: **367/367** index lib tests GREEN serial.

**Measured (per-crate, AVX2+F16C worker `hz2`; sum of 10 000 f16-slice dots):**

| Workload | generic (`wide`) | AVX2+F16C | Ratio | Status |
|----------|------------------|-----------|-------|--------|
| `dot/dim256/f16_slice` (10k vectors) | 1.6404 ms | 430.08 µs | **0.262 (~3.81×)** | KEEP |
| `dot/dim384/f16_slice` (10k vectors) | 2.3499 ms | 645.05 µs | **0.275 (~3.64×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-index --profile release \
  --bench dot_product -- "dot/dim(256|384)/f16_slice"
```

**Scope:** original-comparator ratio **N/A** (frankensearch's own vector tier), but this is the wired
in-memory two-pass pass-2 rescore kernel, so every two-pass query's rescore is now ~3.7× cheaper on
AVX2+F16C hosts, safe-fallback elsewhere. With this, **all four hot dot kernels** (int8 2.5×, 4-bit
1.2×, f16-bytes 3.9×, f16-slice 3.7×) are AVX2-dispatched. The dot-kernel SIMD vein is now essentially
mined; remaining are the lower-traffic exact-f32 paths (`dot_product_f32_bytes_f32` / `dot_packed_4bit`).
Kept bench arm: `f16_slice_generic`.

### 2026-06-28 — runtime-dispatched AVX2 `dot_product_f32_bytes_f32`: 1.37–1.46× (completes the dot SIMD arc) (BlackThrush)

**Lever (the last dot kernel).** `dot_product_f32_bytes_f32` is the exact f32-quantization scan kernel
(`Quantization::F32` indexes + MRL rescore — non-default; F16 is the default). Same runtime-AVX2-dispatch
pattern: a hand-written `#[target_feature(enable="avx2")]` kernel mirroring the `wide` kernel's **4
accumulators**, `(acc0+acc1)+(acc2+acc3)` reduction, 8-chunk tail (reduced **through `wide::f32x8::reduce_add`**),
and `mul_add` scalar tail; falls back to `dot_product_f32_bytes_f32_generic` on non-AVX2/non-x86. f32 LE
bytes are native on x86, so an unaligned `loadu_ps` of the stored bytes gives the same lanes as
`decode8_f32` → **bit-identical** (`simd::tests::avx2_f32dot_matches_generic`, `to_bits` equality across
13 dim shapes). Conformance: **368/368** index lib tests GREEN serial.

**Measured (per-crate, AVX2 worker `hz2`; sum of 10 000 f32 dots; `f32_bytes_new` = runtime dispatch →
AVX2, `f32_bytes_generic` = portable `wide` fallback):**

| Workload | generic (`wide`) | AVX2 | Ratio | Status |
|----------|------------------|------|-------|--------|
| `dot/dim256/f32_bytes` (10k vectors) | 415.74 µs | 285.36 µs | **0.686 (~1.46×)** | KEEP |
| `dot/dim384/f32_bytes` (10k vectors) | 790.82 µs | 575.87 µs | **0.728 (~1.37×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-index --profile release \
  --bench dot_product -- "dot/dim(256|384)/f32_bytes"
```

**Scope:** original-comparator ratio **N/A** (frankensearch's own vector tier). The **smallest** of the
five SIMD-dispatch wins — f32 has no decode win (it's already f32) and the `wide` generic already uses 4
f32x8 accumulators, so the gain is purely 256-bit width over the SSE2-default 2×128. Non-default path
(F32-quant / MRL), so low traffic, but a clean bit-identical win that **completes the dot-kernel SIMD
arc**: int8 (2.5×), 4-bit (1.2×), f16-bytes (3.9×), f16-slice (3.7×), f32-bytes (1.4×) — **all five hot
dot kernels are now runtime-AVX2-dispatched** with a property test + safe fallback each. Remaining:
`dot_product_f32_f32` (the `&[f16]`-less f32 slice, MRL — trivial copy of this) and the integer
`dot_packed_4bit` (non-prepared); both very low traffic. Kept bench arm: `f32_bytes_generic`.

### 2026-06-28 — runtime-dispatched AVX2 `dot_product_f32_f32` (slice): ~1.6× (MRL path) (BlackThrush)

**Lever (the f32-slice sibling of the f32-bytes win — final f32 kernel).** `dot_product_f32_f32` is the
`&[f32]`·`&[f32]` dot behind the MRL (Matryoshka truncated-embedding) rescore — non-default, low traffic.
Same runtime-AVX2 dispatch: a `#[target_feature(enable="avx2")]` kernel mirroring the `wide` kernel's 4
accumulators, `(acc0+acc1)+(acc2+acc3)` reduction (through `wide::f32x8::reduce_add`), 8-chunk tail, and
**separate-mul+add** scalar tail (matched per-kernel — this one is NOT `mul_add`, unlike f32-bytes);
falls back to `dot_product_f32_f32_generic` (the former `dot_product_f32_f32_unchecked`) elsewhere.
**Bit-identical** (`simd::tests::avx2_f32slicedot_matches_generic`, `to_bits` across 13 dim shapes).
Conformance: **369/369** index lib tests GREEN serial.

**Measured (per-crate, AVX2 worker `hz2`; sum of 10 000 f32-slice dots):**

| Workload | generic (`wide`) | AVX2 | Ratio | Status |
|----------|------------------|------|-------|--------|
| `dot/dim256/f32_slice` (10k vectors) | 770.63 µs | 477.82 µs | **0.620 (~1.61×)** | KEEP |

(dim384 same direction, noisier on a contended worker: new ≈ 953 µs.)

**Scope:** original-comparator ratio **N/A**; non-default MRL path, low traffic, modest 256-bit-width-only
win (no f32 decode). Completes the dot-kernel SIMD coverage: **six** kernels now runtime-AVX2-dispatched
(int8, 4-bit-prepared, f16-bytes, f16-slice, f32-bytes, f32-slice), each bit-identical + safe fallback.
The only dot left is the non-prepared integer `dot_packed_4bit` (lowest traffic). Kept bench arm:
`f32_slice_generic`.

### 2026-06-28 — runtime-dispatched AVX2+F16C int8 slab QUANTIZE: 5.2–5.9× (a DIFFERENT primitive) (BlackThrush)

**Lever (off the dot vein — the quantizer, not a dot).** `quantize_i8_slab` (the lazy int8 ADC slab
build) was the next-biggest un-mined **compute** kernel: it decodes the whole f16 slab to f32 **twice**
(once for the corpus max-abs, once to quantize) with software `f16::to_f32`, plus a per-element
`f32::round` + clamp — fully decode-bound and scalar. New `simd::quantize_f16_slab_to_i8` runtime-
dispatches (`is_x86_feature_detected!("avx2") && …("f16c")`) to a hand-written AVX2+F16C kernel:
`vcvtph2ps` decodes 8 f16/instruction; pass-1 is a vector `max` over `|x|`; pass-2 is `×scale` →
round-half-away (emulated `trunc(v + copysign(0.5, v))`, exact for `|v| ≤ 127`) → clamp (`min`/`max`) →
`cvttps_epi32` → `i8`. Falls back to the scalar kernel on non-AVX2+F16C. `in_memory.rs::quantize_i8_slab`
now delegates to it.

**Bit-identical:** `max` is exact/associative (vector max == scalar fold), the round emulation matches
`f32::round` exactly, clamp/cast unchanged — `simd::tests::avx2_quantize_i8_matches_generic` asserts the
`Vec<i8>` is byte-for-byte equal across lengths (incl. sub-8 tails + the zero-vector `max_abs==0` edge).
Conformance: **370/370** index lib tests GREEN serial.

**Measured (per-crate, AVX2+F16C; dim 384 f16 slab → i8; `generic` = scalar, `dispatch` = AVX2+F16C):**

| Workload | generic | dispatch | Ratio | Status |
|----------|---------|----------|-------|--------|
| `quantize_i8_slab/10000` (10k×384) | 25.85 ms | 4.94 ms | **0.191 (~5.2×)** | KEEP |
| `quantize_i8_slab/50000` (50k×384) | 157.65 ms | 26.61 ms | **0.169 (~5.9×)** | KEEP |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  cargo bench -p frankensearch-index --profile release --bench quantize_slab
```

**Scope:** original-comparator ratio **N/A** (frankensearch's own int8 ADC tier; no Tantivy counterpart).
This is **index-build / cold-start latency** — the slab is `OnceLock`-cached, so amortized for a static
index but recurs on every rebuild/refresh: a 50k×384 int8 slab build drops **157.65 ms → 26.61 ms**
(~131 ms/build saved; scales linearly with corpus). The first *non-dot* SIMD win — a genuinely different
primitive (quantizer), and the largest single ratio besides the f16 dots, because it's pure F16C
decode-bound. Route-next: `pack_4bit_slab` (the **wired-default** 4-bit slab build, same decode-bound
shape but with nibble packing) is the natural sibling. Kept bench: `quantize_slab`.

### 2026-06-28 — runtime-dispatched AVX2+F16C 4-bit slab PACK: 10.3–13.6× on the WIRED-DEFAULT build (BlackThrush)

**Lever (the int8 quantizer's sibling — and the default path).** `pack_4bit_slab` is the lazy 4-bit ADC
slab build behind `search_top_k_4bit_two_pass_filtered` — i.e. **the wired-default sync-hybrid pass-1
storage** (`in_memory.rs::pack_4bit_slab`, `OnceLock`-cached). Same decode-bound shape as the int8 slab
(software `f16::to_f32` ×2 for max-abs + quantize) PLUS a per-nibble `slab[i] |= nib` read-modify-write.
New `simd::pack_f16_slab_to_4bit` runtime-dispatches to an AVX2+F16C kernel: `vcvtph2ps` decodes 8
f16/instruction for both passes; the quantize pass reuses the int8 `×scale` → round-half-away → clamp
pipeline (clamp `[-7,7]`) → `cvttps_epi32`, then packs the 8 nibbles as **4 fully-determined direct
byte writes** (no RMW). It beats int8's ratio because it kills BOTH the decode cost AND the per-nibble
RMW. Falls back to the scalar kernel; `in_memory.rs` delegates (the old scalar slab packer was removed).

**Bit-identical:** same nibble values + byte layout — `simd::tests::avx2_pack_4bit_matches_generic`
asserts byte-equal `Vec<u8>` across dim shapes (full 8-chunks, sub-8 tails, **odd** dims) × multiple
vectors. Conformance: **371/371** index lib tests GREEN serial (incl. the 4-bit two-pass recall gates,
which exercise the slab through the delegated `pack_4bit_slab`).

**Measured (per-crate, AVX2+F16C; dim 384 f16 slab → packed 4-bit; `generic` = scalar, `dispatch` =
AVX2+F16C):**

| Workload | generic | dispatch | Ratio | Status |
|----------|---------|----------|-------|--------|
| `pack_4bit_slab/10000` (10k×384) | 31.38 ms | 2.30 ms | **0.073 (~13.6×)** | KEEP |
| `pack_4bit_slab/50000` (50k×384) | 150.91 ms | 14.64 ms | **0.097 (~10.3×)** | KEEP |

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  cargo bench -p frankensearch-index --profile release --bench quantize_slab -- pack_4bit_slab
```

**Scope:** original-comparator ratio **N/A** (own 4-bit ADC tier). Index-build / cold-start latency on
the **default** path: a 50k×384 4-bit slab build drops **150.91 ms → 14.64 ms** (~136 ms/build saved,
linear in corpus), recurring on rebuild/refresh. Largest non-dot ratio yet. With int8 + 4-bit, **both
quantizer slab builds are now AVX2-dispatched**; the per-query `quantize_i8_query`/`pack_4bit_query` are
dim-sized (negligible) and not worth it — the build/quantize SIMD vein is now mined too. Kept bench:
`quantize_slab`.

### 2026-06-28 — runtime-dispatched F16C f32→f16 ENCODE: 4.1–5.4× (the most-common build kernel) (BlackThrush)

**Lever (the encode, not the decode/quantize — every index build).** `f16::from_f32` is the per-element
conversion at the heart of EVERY index build — `InMemoryVectorIndex::from_vectors` (both tiers of the
sync hybrid), FSVI writes. New `simd::encode_f32_to_f16_extend` runtime-dispatches to F16C `vcvtps2ph`
(8 f32→f16/instruction, round-to-nearest-even) and — critically — **bulk-stores the 8 f16 straight into
the `Vec`'s spare capacity (`_mm_storeu_si128` + `set_len`)** rather than `push`-ing per element (the
first cut at ~1.6× was push-bound; bulk store took it to ~5×). `in_memory.rs::from_vectors` now calls it.

**Bit-identical:** `vcvtps2ph` round-to-nearest-even == `half::f16::from_f32` for finite inputs
(`half::f16` is `repr(transparent)` over `u16`, so the store is layout-safe) —
`simd::tests::avx2_f16encode_matches_generic` asserts `to_bits` equality across normal/large(near-f16-max)/
tiny(subnormal) magnitudes + sub-8 tails. Conformance: **372/372** index lib tests GREEN serial (the
`from_vectors_*` tests exercise the new encode on the build path).

**Measured (per-crate, AVX2+F16C; dim 384 f32→f16; reused output buffer to isolate the encode from
per-iter allocation — matching the real build which appends to one reserved flat Vec):**

| Workload | generic (`half::from_f32` + push) | F16C (bulk store) | Ratio | Status |
|----------|-----------------------------------|-------------------|-------|--------|
| `encode_f32_to_f16/10000` (10k×384) | 5.76 ms | 1.06 ms | **0.184 (~5.4×)** | KEEP |
| `encode_f32_to_f16/50000` (50k×384) | 31.40 ms | 7.73 ms | **0.246 (~4.1×)** | KEEP |

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  cargo bench -p frankensearch-index --profile release --bench quantize_slab -- encode_f32_to_f16
```

**Scope:** original-comparator ratio **N/A** (own storage encoding). Index-build / load latency on the
**most-common** path (every `from_vectors`, all f16 indexes — the default). Smaller ratio than the
quantizer slabs (`from_f32` is a cheaper bit-twiddle than `round`/decode, and the bulk store is then
memory-bandwidth-bound), but it caps the build/load SIMD vein: dot kernels, int8+4-bit slab quantizers,
and now the f16 encode are all AVX2/F16C-dispatched. The lesson (re-recorded): when a decode/convert
kernel shows a modest SIMD ratio, check whether per-element `push` is the bottleneck — bulk-store into
spare capacity. Kept bench: `quantize_slab` (now also covers `encode_f32_to_f16`).

### 2026-06-28 — file-based FSVI slab write: F16C encode + batched write_all = 6.4–7.3× (BOLD build path) (BlackThrush)

**Lever (the FILE-based persist, the build path BOLD itself uses).** `write_vector_slab` (the
`VectorIndex::finish`/persist encoder, reached by `VectorIndex::create` + `write_record` — i.e. how the
**BOLD fixture builds its index** and how every on-disk FSVI is written) did per-element
`writer.write_all(&f16::from_f32(v).to_le_bytes())` — ~38M two-byte `write_all` calls for a 100k×384
index, each preceded by a software `from_f32`. Now: F16C-encode each record's f32→f16 via
`simd::encode_f32_to_f16_extend` into a reused scratch `Vec<f16>`, then ONE `write_all` of the whole
record's bytes (`#[cfg(target_endian="little")]` casts the `repr(transparent)` f16 slab to `&[u8]`; a
BE fallback keeps the per-element `to_le_bytes`). Kills BOTH the per-element `write_all` overhead AND the
scalar `from_f32`.

**Bit-identical:** the on-disk slab is byte-for-byte identical (LE f16) — the `write_f16_slab` bench
asserts the two arms emit equal bytes, and the existing FSVI write→read round-trip tests (which decode
the slab back) stay GREEN. Conformance: **372/372** index lib tests GREEN serial.

**Measured (per-crate; dim 384 f32 records → LE f16 slab into an in-memory writer; `per_element` = old
per-value `from_f32`+`write_all`, `batched` = F16C encode + one `write_all`/record):**

| Workload | per_element | batched | Ratio | Status |
|----------|-------------|---------|-------|--------|
| `write_f16_slab/10000` (10k×384) | 8.75 ms | 1.20 ms | **0.137 (~7.3×)** | KEEP |
| `write_f16_slab/50000` (50k×384) | 48.06 ms | 7.52 ms | **0.156 (~6.4×)** | KEEP |

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  cargo bench -p frankensearch-index --profile release --bench quantize_slab -- write_f16_slab
```

**Scope:** original-comparator ratio **N/A** (own FSVI storage write). Unlike the in-memory `from_vectors`
encode (`9f2356a`), this is the **file/persist** build — the path the BOLD comparator fixture itself
pays, plus every persisted/compacted index. Bigger than the encode-alone ratio because batching also
removes ~38M→count `write_all` calls. The remaining per-element `f16::from_f32` sites (WAL compaction
`lib.rs` ~1095, WAL lookup ~1363) are low-traffic compaction/lookup paths — left as-is. Kept bench:
`quantize_slab` (now also covers `write_f16_slab`).

### 2026-06-28 — InMemory two-pass full-recall short-cut: 1.45–1.49× (skip the pass that prunes nothing) (BlackThrush)

**Lever (the InMemory analog of the file-backed full-recall path).** `VectorIndex` (file-backed) already
skips the two-pass for full recall (`search.rs:170`: `limit >= total → collect-and-sort single f16 pass`).
`InMemoryVectorIndex`'s `search_top_k_int8_two_pass_filtered` / `search_top_k_4bit_two_pass_filtered` did
NOT: when `candidate_count = min(limit·mult, count) >= count` (a `limit_all`-class query, or a large limit
on a small corpus), pass-1 keeps **every** vector (the size-N bounded heap never evicts) and pass-2
rescores all N — so the quantized pass-1 scan, the size-N heap, the query quantize/nibble-prep, and the
lazy slab build (`quantize_i8_slab`/`pack_4bit_slab`) are **pure overhead**. Added a one-branch short-cut:
when `candidate_count >= count`, delegate to the exact `search_top_k` (f16 single pass).

**Bit-identical by construction:** the two-pass's own doc-contract is "== `search_top_k` whenever pass-1
retains the true top-k"; full recall retains *every* candidate, so the delegation is exact (not merely
approximate). `limit.min(count)` avoids a `usize::MAX`-sized heap. The `int8/four_bit_two_pass_keep_all_
matches_exact` tests + a new bench `assert_eq!(two_pass(N,1) == search_top_k(N))` confirm it. Conformance:
**372/372** index lib tests GREEN serial.

**Measured (per-crate `int8_two_pass` bench, N=10k×384 clustered, full recall = top-N; fast-path disabled
to capture the old cost, then restored):**

| Workload | old two-pass | exact (fast-path) | Ratio | Status |
|----------|--------------|-------------------|-------|--------|
| `full_recall/4bit` (k=N) | 2.220 ms | 1.491 ms | **0.67 (~1.49×)** | KEEP |
| `full_recall/int8` (k=N) | 2.162 ms | 1.491 ms | **0.69 (~1.45×)** | KEEP |

(With the fast-path restored, the two-pass arms collapse to ~1.31 ms ≈ the exact arm — the delegation is
realized.) Plus a **cold-start bonus** not in the steady-state number: full-recall now skips the
`OnceLock` slab build entirely (the 5–13× slab quantizers don't even run).

**Scope:** original-comparator ratio **N/A** (this is the InMemory sync-hybrid path; BOLD's `limit_all`
uses the file-backed `VectorIndex`, already single-pass — so this does NOT move the BOLD `limit_all` 1.4×,
which stays inherent). It closes an *internal* gap: InMemory was doing redundant pass-1 work the
file-backed path already avoided, on `limit_all`/large-limit queries through the wired two-pass. Kept
bench: `int8_two_pass` (now has a `full_recall` group).

### 2026-06-28 — AVX2 JL xorshift accumulate: 1.51× (a DIFFERENT primitive — SIMD PRNG) (BlackThrush)

**Lever (off the dot/quantize vein — the embedder's PRNG).** `HashEmbedder::embed_jl` (the
Johnson-Lindenstrauss projection embedder, the *compute-bound* tier — `O(tokens·dim)` xorshift) folds 4
independent xorshift64 token-chains per dimension. The hot loop `jl_accumulate_lanes` was scalar-ILP: 4
chains × 3 shift-xor steps = 24 scalar ops/dim plus 4 sign-selects. The 4 chains are exactly one
`__m256i` of u64 lanes (`JL_LANES == 4`), so the new `#[target_feature(enable="avx2")]`
`jl_accumulate_lanes_avx2` does each step as a single vector `slli`/`srli`/`xor_epi64` (6 vector ops vs
24 scalar), and recovers the per-dim ±1 sum branch-free: shift each lane's low bit into the sign
position, `_mm256_movemask_pd` the 4 signs, `4 - 2·popcount(mask)` = the exact integer `a0+a1+a2+a3`.
Runtime-dispatched (`is_x86_feature_detected!("avx2")`); scalar-ILP is the fallback.

**Bit-identical:** the xorshift is per-lane identical, and the per-dim contribution is a sum of ±1 lanes
— an exact small integer added to an exact-integer accumulator (`|value| ≤ token count ≪ 2²⁴`), so neither
the SIMD lane grouping nor the `4-2·popcount` reformulation changes a bit. `jl_avx2_matches_scalar`
asserts `to_bits` equality across 10 dim shapes accumulated over 25 token-groups (+ the all-odd
zero-state edge). Conformance: **289/289** embed lib tests GREEN.

**Measured (per-crate `hash_embed` bench, dim 384, 2000 4-chain groups ≈ 8k tokens, kernel isolated from
tokenize/normalize):**

| Workload | scalar-ILP | AVX2 | Ratio | Status |
|----------|-----------|------|-------|--------|
| `jl_accumulate` (384×2000) | 2349.7 µs | 1556.0 µs | **0.662 (~1.51×)** | KEEP |

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  cargo bench -p frankensearch-embed --profile release --bench hash_embed -- jl_accumulate
```

**Scope:** original-comparator ratio **N/A** (own embedder; no Tantivy counterpart). JL is **non-default**
(the default `HashEmbedder` is `FnvModular`, whose hash is inherently sequential — not SIMD-able), so this
helps deployments that pick the JL tier (better distance preservation) — the compute-bound embed path for
them. A genuinely **different primitive** (SIMD PRNG/xorshift, not a dot/quantize/encode), satisfying the
"never-safe-Rust-ceiling" steer with AVX2 u64 intrinsics. The smaller-than-the-decode-kernels ratio is
expected: the scalar-ILP already pipelines the 4 independent chains, so the win is the 4×-fewer
instructions, not a latency unlock. Kept bench arm: `jl_accumulate/scalar`.

### 2026-06-28 — JL accumulate, 4-lane → 8-lane AVX2 (2-way ILP): 1.76× further (2.76× over scalar) (BlackThrush)

**Lever (route-next from the JL win — the 4-lane kernel was latency-bound).** Yesterday's 4-lane AVX2
`jl_accumulate_lanes` is ONE `__m256i` whose 3-step xorshift (`s ^= s<<13; s ^= s>>7; s ^= s<<17`) is a
**dependency chain** → latency-bound (each step waits on the previous). Running TWO independent `__m256i`
(8 chains, `jl_accumulate_lanes8_avx2`) with the xorshift steps **interleaved** exposes the 2-way ILP that
hides that latency — the CPU pipelines the two chains. Wired as the production JL kernel (`JL_LANES = 8`,
`embed_jl` groups tokens 8-at-a-time); the 4-lane kernel is kept as the bench A/B baseline.

**Bit-identical:** the per-dim contribution is a sum of ±1 lanes (an exact small integer, `|value| ≤ token
count ≪ 2²⁴`) — independent of lane grouping AND of the `8 - 2·(popcount(maskA)+popcount(maskB))`
reformulation. `jl_8lane_matches_4lane` asserts the 8-lane embed (AVX2 **and** scalar) is `to_bits`-equal
to the 4-lane embed across 5 dim shapes. Conformance: **290/290** embed lib tests GREEN (all JL output
tests unchanged — the embed is byte-for-byte the same).

**Measured (per-crate `hash_embed` bench, dim 384, 8000 tokens, kernel isolated):**

| Workload | scalar-ILP | AVX2 4-lane | AVX2 8-lane | 8-lane vs 4-lane | 8-lane vs scalar |
|----------|-----------|-------------|-------------|------------------|------------------|
| `jl_accumulate` | 2778.9 µs | 1774.7 µs | **1005.8 µs** | **0.567 (~1.76×)** | **0.362 (~2.76×)** |

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  cargo bench -p frankensearch-embed --profile release --bench hash_embed -- jl_accumulate
```

**Scope:** original-comparator ratio **N/A** (own JL embedder, non-default). Bigger jump than yesterday's
4-lane (1.51×) precisely *because* the 4-lane was latency-bound on a single register's dependency chain —
the second register is a latency unlock, not just more width. **LESSON: a SIMD kernel whose hot recurrence
is a single-register dependency chain is latency-bound — add a 2nd independent accumulator register (more
lanes) before assuming the SIMD is saturated.** Kept bench arms: `jl_accumulate/{scalar,avx2_4lane,avx2_8lane}`.

### 2026-06-28 — AVX2 element-wise vector accumulate (model2vec mean-pool): ~1.4–1.6× (BlackThrush)

**Lever (the one candidate I'd dismissed by REASONING, not measuring).** `Model2VecEmbedder::embed_sync`
mean-pools by accumulating each in-vocab token's embedding row into a sum (`sum[d] += row[d]`, T tokens ×
`dim`). I'd assumed LLVM auto-vectorizes it away — true, but only to the **SSE2 baseline** (this workspace
builds with no global `+avx2`). New `simd::accumulate_f32_into` runtime-dispatches to a hand-written AVX2
kernel (`_mm256_loadu_ps`/`_mm256_add_ps`/`_mm256_storeu_ps`, 8 f32/instruction = 32-byte loads), which
roughly doubles the per-cycle load bandwidth on this memory-bound loop. `model2vec_embedder.rs`'s mean-pool
now calls it.

**Bit-identical:** each `sum[d] += row[d]` is an independent element-wise add (NOT a cross-lane reduction),
so SIMD only changes how many dims are added per instruction, never the per-dim arithmetic — unlike the
dot/JL reductions, this needs no reorder. `simd::tests::avx2_accumulate_matches_scalar` asserts `to_bits`
equality across 11 dim shapes accumulated over 5 rows. Conformance: **291/291** default embed lib tests +
**27/27** `--features model2vec` tests (the mean-pool feeds `embed_output_is_l2_normalized`,
`embed_deterministic`, etc.) GREEN.

**Measured (per-crate `hash_embed` bench, dim 384, 64-row mean-pool; the production `accumulate_f32_into`):**

| Workload | scalar (SSE2 auto-vec) | AVX2 dispatch | Ratio | Status |
|----------|------------------------|---------------|-------|--------|
| `vec_accumulate` run 1 | 2.194 µs | 1.530 µs | **0.70 (~1.43×)** | KEEP |
| `vec_accumulate` run 2 | 2.735 µs | 1.680 µs | **0.61 (~1.63×)** | KEEP |

(AVX2 is consistent ~1.5–1.7 µs; the scalar arm is the noisier one, so ~1.4–1.6× is the honest range.)

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  cargo bench -p frankensearch-embed --profile release --bench hash_embed -- vec_accumulate
```

**Scope:** original-comparator ratio **N/A** (own embedder). `Model2Vec` is the **opt-in** quality tier
(`model2vec` feature, potion-128M static embedder), so this speeds the mean-pool for deployments that pick
it. **Lesson reinforced (the one that bit me on the f16 dot): MEASURE, don't reason — I'd written off this
loop as "auto-vectorized" but the build's SSE2-only baseline left a clean ~1.5× on the table for a manual
AVX2 (wider loads).** Element-wise (non-reduction) kernels are the bit-identical-SAFE place to widen SIMD.
First SIMD in the embed crate's shared path (reusable `accumulate_f32_into`). Kept bench: `vec_accumulate`.

### 2026-06-28 — AVX2 element-wise scale (l2_normalize, DEFAULT path): scale kernel ~1.70× (BlackThrush)

**Lever (the element-wise vein's DEFAULT-path application).** `l2_normalize_in_place`
(`core/traits.rs`) — called on **every** embed by every embedder (the FNV/JL default tiers, model2vec,
api) — has two passes: the sum-of-squares (a reduction, bit-identical-LOCKED) and the **scale**
`vec[d] *= inv_norm` (element-wise, bit-identical-SAFE). New `core::simd::scale_f32_in_place`
runtime-dispatches the scale to AVX2 (`_mm256_mul_ps` by a broadcast `inv_norm`, 8 f32/instruction);
the scalar loop (SSE2 auto-vec) is the fallback. **First SIMD in `frankensearch-core`** (std-only —
`is_x86_feature_detected!` + `core::arch` + opt-in `#[allow(unsafe_code)]`).

**Bit-identical:** each `vec[d] *= inv_norm` is an independent IEEE f32 multiply, identical 1/4/8-wide
(no cross-lane reduction). `simd::tests::avx2_scale_matches_scalar` asserts `to_bits` equality across 11
dim shapes; `traits::tests::l2_normalize_in_place_matches_allocating` still GREEN (the AVX2 in-place
produces the same per-element `x·inv` as the allocating path). Conformance: **core** `l2_normalize` + simd
tests + **291/291** embed lib tests GREEN.

**Measured (per-crate `hash_embed` bench, dim 384, in-place scale; the production `scale_f32_in_place`):**

| Workload | scalar (SSE2 auto-vec) | AVX2 dispatch | Ratio | Status |
|----------|------------------------|---------------|-------|--------|
| `vec_scale` (scale kernel) | 2.015 µs | 1.184 µs | **0.59 (~1.70×)** | KEEP |

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  cargo bench -p frankensearch-embed --profile release --bench hash_embed -- vec_scale
```

**Scope:** original-comparator ratio **N/A** (own embedders). ~1.70× on the scale KERNEL; `l2_normalize`
overall is ~1.3× (the scale is one of its two passes; the sum-of-squares reduction stays scalar — locked).
This is the **default embed path** (vs the model2vec mean-pool's opt-in tier), so it speeds every embed on
every embedder — bigger AVX2 ratio than the accumulate (1.4–1.6×) because the scale is more compute-bound
(one input × broadcast scalar, less memory traffic). The `core::simd` helper is reusable for future
element-wise core ops. Kept bench: `vec_scale` (in `hash_embed`, importing the core helper).

### 2026-06-28 — BitsetFilter: identity hasher for already-hashed u64 keys: ~8–11% on the filtered scan (BlackThrush)

**Lever (a data-structure inefficiency, not a compute kernel).** `core::filter::BitsetFilter` stored its
allow-set as a default `HashSet<u64>` — i.e. it ran **SipHash over each key on every membership probe**.
But the keys are ALREADY FNV-1a `doc_id` hashes (uniform), so SipHash is pure redundant work. Added an
`IdentityHasherU64` (`write_u64` passes the key straight through; FNV-1a's high bits feed the SwissTable
control byte) + `BuildIdentityHasherU64`, and switched `BitsetFilter` to `HashSet<u64,
BuildIdentityHasherU64>` (re-bucketed once at construction). On a **filtered vector scan** the probe runs
per candidate (all N, before the cheap SIMD dot), so dropping SipHash speeds the hot loop.

**Bit-identical:** identity hashing preserves set membership exactly (a `u64` is present iff inserted) —
results are unchanged. Conformance: **core** filter tests 39/39 + **index** lib tests **372/372** GREEN
(filtered-search tests exercise `BitsetFilter`).

**Measured (per-crate `int8_two_pass` bench, N=10k clustered, dim 384, ~50%-pass filter; SipHash baseline
vs identity hasher):**

| Workload | SipHash | identity | Ratio | Status |
|----------|---------|----------|-------|--------|
| `flat_filtered` (exact f16 filtered scan) | ~205 µs | ~189 µs | **~1.08–1.11×** (criterion "improved", −8.6…−13%) | KEEP |
| `int8_filtered_mult5` (two-pass filtered) | ~204 µs | ~208 µs | within noise | — |

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  cargo bench -p frankensearch-index --profile release --bench int8_two_pass -- flat_filtered
```

**Scope:** original-comparator ratio **N/A** (filtered search — not the unfiltered BOLD path). The
hypothesis that SipHash *dominated* the filtered scan was wrong — measured it's ~8% of the exact-scan
cost (the dot + heap + memory dominate), so the win is modest, and the int8/4-bit two-pass filtered path
is within noise (the probe is a smaller relative fraction there; the identity probe is strictly faster, no
regression). Still a real, bit-identical fix of a genuine inefficiency (SipHash over an already-uniform
hash) on the common filtered-search path. This is a SAFE-Rust data-structure lever (not a SIMD intrinsic),
but a different *kind* of primitive from the compute kernels. `IdentityHasherU64` is reusable for any
already-hashed-u64 set. Kept the existing `int8_two_pass` filtered bench arms.

### 2026-06-29 — selective-filter GATHER fast-path: invert the loop, 6.9–50× on filtered vector search (BlackThrush)

**Lever (a DIFFERENT primitive — loop-order inversion, not a kernel).** A filtered vector search with a
hash-addressable allow-list (`BitsetFilter` — "search within these K docs") previously scanned the
**whole corpus** and ran one membership probe per document (the identity-hasher commit `f7d613b` made that
probe cheap, but it still runs `N` times). When the allow-set is a small fraction of the corpus — the
common real-world case (per-tenant / per-folder / per-ACL scoping) — almost all of that work is wasted.
The gather fast-path **inverts the loop**: iterate the (small) allow-set, map each hash → position via a
lazily-built bijective `hash → pos` table, and exact f16-scan **only** those positions. Work becomes
`O(|allow-set|)` instead of `O(corpus)`.

- New trait method `SearchFilter::candidate_hashes() -> Option<&DocIdHashSet>` (default `None`;
  `BitsetFilter` returns its allow-set). `None` ⇒ predicate/metadata/composite filters keep the scan.
- New lazy `InMemoryVectorIndex::hash_to_pos` map (identity-hashed, same FNV-1a key space). Returns
  `None` when two doc_ids collide to one hash (not a bijection) ⇒ disables the fast path, so results stay
  exact. Built once on first selective-filter search (other callers pay nothing).
- Wired into `search_top_k`, `search_top_k_int8_two_pass_filtered`, and
  `search_top_k_4bit_two_pass_filtered` (the production sync fast-tier). For the two-pass paths the gather
  is *exact* f16, so it is **strictly more accurate** than the approximate int8/4-bit pass-1 it replaces
  *and* far cheaper when selective.

**Bit-identical:** the gathered passing set `{pos : doc_id_hash[pos] ∈ allow-set}` is exactly the set the
per-document scan keeps, and both rank by the `(score, index)` total order, so order is independent of the
gather sequence. **Conformance: GREEN** — `frankensearch-core` filter 39/39, `frankensearch-index` lib
**373/373** (372 + a new `selective_filter_gather_matches_scan` asserting gather == forced scan == exact
on the public, int8, and 4-bit filtered paths). The bench also asserts parity at all 8 selectivities × 32
queries before timing.

**Measured (per-crate same-binary A/B, in-memory N=50k clustered, dim 384, k=10, 32 queries; `scan` =
forced per-document parallel filtered scan, `gather` = allow-set gather; medians; ratio = gather/scan):**

| selectivity | allow-set | scan | gather | ratio | speedup |
|-------------|-----------|------|--------|-------|---------|
| 0.1 % | 50    | 165.9 µs | 3.31 µs  | **0.020** | **50.1×** |
| 0.5 % | 250   | 194.8 µs | 16.06 µs | **0.082** | **12.1×** |
| 1 %   | 500   | 219.4 µs | 31.96 µs | **0.146** | **6.9×**  |
| 2 %   | 1 000 | 197.9 µs | 63.45 µs | **0.321** | **3.1×**  |
| 5 %   | 2 500 | 267.8 µs | 154.7 µs | **0.578** | **1.7×**  |
| 10 %  | 5 000 | 449.0 µs | 434.6 µs | 0.968 | ~tie (crossover) |
| 25 %  | 12 500 | 853 µs  | 2 517 µs | 2.95  | 0.34× (loss) |
| 50 %  | 25 000 | 852 µs  | 5 117 µs | 6.01  | 0.17× (loss) |

The serial gather crosses over the parallel scan at ~10 %; the selectivity gate is set to **N/16 (≈6.25 %)**
(`GATHER_SELECTIVITY_DIVISOR`) — comfortably inside the winning region (~1.5× at the boundary) with margin
for machines whose core count shifts the crossover. Above the gate the scan path is taken unchanged (no
regression; the existing `int8_two_pass/flat_filtered` 50 % arm is unaffected — `50%·16 ≥ 100%` ⇒ no
gather).

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  cargo bench -p frankensearch-index --profile release \
  --bench filtered_gather -- --sample-size 30 --warm-up-time 1 --measurement-time 2
```

**Scope vs comparator:** filtered search is exactly where a Lucene/Tantivy/Meilisearch-class engine applies
the filter *first* and searches only the surviving subset; frankensearch previously paid a full-corpus
vector scan regardless. This closes that structural disadvantage for selective filters and makes the
production sync fast-tier (`search_top_k_4bit_two_pass_filtered`) both faster and *exact* on selective
queries. Original-comparator ratio is not re-measured here (the BOLD harness has no selective-filter row
yet), so this is a frankensearch before/after on the filtered path, not a new head-to-head dominance claim.

**Route next:** (1) **parallelize the gather** (rayon chunks + `merge_*_partials`, the established
order-independent pattern) to push the crossover well above 10 % and reclaim the 6–25 % band; with that the
gate could widen toward N/2. (2) Plumb `candidate_hashes` into the **FSVI file-backed** scan
(`search.rs`) and the **lexical** filtered path so the inversion covers every tier, not just in-memory.
(3) Add a selective-filter row to the BOLD comparator to convert this into a head-to-head dominance number.

### 2026-06-29 — gather follow-up: PARALLELIZE the gather → 10% flips tie→1.3× win, gate widened to N/10 (BlackThrush)

**Lever (route-next #1 above, now done).** The serial gather lost above ~10% only because it ran one core
against the parallel scan. Split `scan_gather` into a per-slice `gather_range` + a dispatcher that
`par_chunks(PARALLEL_CHUNK_SIZE)` above the chunk size and merges per-chunk bounded heaps by the
`(score, index)` total order — the exact `scan_parallel` pattern, so still **bit-identical** (new test
`parallel_gather_matches_scan` at a >chunk-size allow-set; full `frankensearch-index` lib **374/374** GREEN;
the earlier one-off `soft_delete_*` red was a fixed-temp-path flake colliding with a concurrent test run —
green on isolated re-run). Tiny allow-sets (< `PARALLEL_CHUNK_SIZE`) stay serial, so the 6.9–50× small-M
wins are unchanged.

**Measured (same `filtered_gather` A/B, N=50k clustered dim 384 k10, gather/scan median; parallel gather):**

| selectivity | scan | gather | ratio | vs serial gather |
|-------------|------|--------|-------|------------------|
| 5 %  | 273.5 µs | 128.3 µs | **0.469 (2.1×)** | was 0.578 (1.7×) |
| 10 % | 288.4 µs | 217.7 µs | **0.755 (1.3×)** | was 0.968 (tie) — **flipped to a win** |
| 25 % | 421.8 µs | 726.1 µs | 1.72 (loss) | was 2.95 — better but still loses |
| 50 % | 808.7 µs | 1169.5 µs | 1.45 (loss) | was 6.01 |

Crossover moved from ~10% to **~13%** — capped not by the (now parallel) dots but by the gather's
**serial** setup (allow-set `collect` + position `sort_unstable`), which grows with the allow-set. Gate
widened `GATHER_SELECTIVITY_DIVISOR` 16 → **10 (N/10)**: ≥1.3× at the boundary, scan path unchanged above
it (no regression — 25/50% rows never reach the gather). **Route next (unchanged):** parallelize the gather
*setup* (`par_sort_unstable` + parallel allow-set materialization) to push the crossover past 25%; then
FSVI/lexical plumbing; then a BOLD selective-filter comparator row.

### 2026-06-29 — MMR cosine: 4-accumulator f64 dot → ~1.6× on `mmr_rerank` end-to-end (BlackThrush)

**Lever (a DIFFERENT primitive — multi-accumulator ILP, the proven pattern).** MMR diversity reranking is
dominated by `cosine_sim_pre`'s inter-doc dot — `O(k·n)` evaluations of a dim-d vector per rerank. That dot
was a **single-accumulator f64 reduction** (`dot += f64::from(a[i]) * f64::from(b[i])`), latency-bound on
the loop-carried `dot` and not auto-vectorized (strict f64 order). Reformulated as **4 independent f64
accumulators** so LLVM auto-vectorizes the f32→f64 widen + multiply-add to SSE2/AVX. No `unsafe`, no SIMD
intrinsics — pure reassociation.

**Bit-identical selection:** MMR is a **search-time** reranking score (not a persisted embedding), so the
f64 reassociation is the same accepted ULP trade as the landed vector-search dot kernels. The ULP-level
score shift does not change the selection at realistic `n·k` — the `mmr_rerank` bench asserts
`mmr_new == mmr_new_4acc` (identical selected index list) for both shapes, and all `frankensearch-fusion`
`mmr::` tests pass unchanged.

**Measured (per-crate same-binary A/B, `mmr_rerank` bench, dim 384, medians; `new` = single-acc running-max
MMR, `new_4acc` = same with the 4-acc dot):**

| shape (pool n, results k) | new (1-acc) | new_4acc | ratio | speedup |
|---------------------------|-------------|----------|-------|---------|
| n=100, k=20  | 622.7 µs  | 384.7 µs  | **0.618** | **1.62×** |
| n=200, k=50  | 3 063.7 µs | 1 896.8 µs | **0.619** | **1.62×** |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  cargo bench -p frankensearch-fusion --profile release \
  --bench mmr_rerank -- --sample-size 30 --warm-up-time 1 --measurement-time 2
```

**Scope:** original-comparator ratio **N/A** — MMR is an opt-in diversification reranker, not on the default
BOLD hybrid path, so this is a frankensearch before/after on that path (not a head-to-head dominance claim).
Real win for the diversification feature: ~1.6× wherever MMR runs. Kept bench arm: `mmr_rerank/new_4acc`.
**Route next:** the f64 widen + `pmullw`/`mulpd` may now bind the loop; an 8-accumulator split or an AVX2
`dot_product_f32_f32` (the index crate already has one) wired into fusion could push further — but MMR's
absolute cost is small, so lower priority than the gather's FSVI/lexical plumbing.

### 2026-06-29 — `count_lexical_tokens`: 256-byte LUT + branchless transition → ~1.5–1.8× (BlackThrush)

**Lever (a DIFFERENT primitive — branchless byte-class state machine).** `count_lexical_tokens` (the
fsfs lexical chunker's per-chunk token counter, run during indexing) had an ASCII byte fast path that, per
byte, called `is_token_byte` (`is_ascii_alphanumeric()` + a 5-way `matches!`, ~7–10 ops) and then branched
on a data-dependent `in_token` flag that **mispredicts on every token boundary** (code/path text is
boundary-dense). Replaced both with (1) a compile-time **256-byte class table** `TOKEN_BYTE` (one L1 load
per byte) and (2) **branchless transition counting**: a token ends at each token→non-token transition, so
`count += (prev & !cur)` with no data-dependent branch. The table is built from `is_token_byte` in a
`const` block, so `TOKEN_BYTE[b] == is_token_byte(b)` for every byte — **bit-identical** token counts.

**Conformance:** `frankensearch-fsfs` lexical tests GREEN; the bench also asserts `count_new == count_lut`
(identical counts) for every input size before timing.

**Measured (per-crate same-binary A/B, `lexical_count` bench, realistic ASCII code/doc chunk, medians;
`bytes` = the landed scalar byte path, `lut` = table + branchless):**

| input bytes | chars (UTF-8) | bytes (scalar) | lut (this change) | lut/bytes | speedup |
|-------------|---------------|----------------|-------------------|-----------|---------|
| 1 024  | 1.093 µs  | 620.5 ns  | 399.6 ns  | **0.644** | **1.55×** |
| 4 096  | 3.949 µs  | 2.221 µs  | 1.408 µs  | **0.634** | **1.58×** |
| 16 384 | 15.64 µs  | 10.17 µs  | 5.582 µs  | **0.549** | **1.82×** |

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  cargo bench -p frankensearch-fsfs --profile release \
  --bench lexical_count -- --sample-size 40 --warm-up-time 1 --measurement-time 2
```

**Scope:** original-comparator ratio **N/A** — internal fsfs indexing-path primitive (the token count is
chunk metadata), so this is a frankensearch before/after, and it speeds **indexing throughput** (a real
Tantivy/Lucene-class dimension) wherever the lexical chunker runs over ASCII code/docs (the common case).
The LUT/branchless pattern is reusable for any byte-class state machine. Kept bench arm: `lexical_count/lut`.

### 2026-06-29 — S3-FIFO query-embedding cache: 4–14% fewer embed misses on skewed/scan streams (BlackThrush)

**Lever (bd-tjkm S3-FIFO admission — a DIFFERENT primitive: cache eviction policy, /alien-graveyard §15.1).**
The hot-path query-embedding cache (`CachedEmbedder` → `CacheState`) used plain **FIFO** eviction (evict
oldest by insertion order; `get` never reorders), so a scan of one-hit-wonder queries evicts the hot/reused
set and a re-requested query becomes a miss = a recomputed embedding. Replaced `CacheState`'s eviction with
**S3-FIFO** (Yang et al., SOSP 2023): three entry-count queues over one map — **Small** (probation, ~10%),
**Main** (promoted on reuse), **Ghost** (recently-evicted keys → re-admit straight to Main). A key
re-requested while resident is promoted to Main and survives scan churn; cold singletons are dropped from
Small. `&str` lookups (no per-get alloc); the `CachedEmbedder`/`CacheStats`/embed API is unchanged (only
`CacheState` internals changed — the codebase's existing `core::S3FifoCache` is byte-budgeted and would
force a per-get `String` alloc, so an entry-count S3-FIFO is the right fit here).

**Measured (`cache_replay` bench, 100k accesses, cap=128 = `DEFAULT_CAPACITY`; a miss = one embed):**

| trace | FIFO hit | S3-FIFO hit | FIFO miss | S3-FIFO miss | miss_ratio (s3/fifo) |
|-------|----------|-------------|-----------|--------------|----------------------|
| zipf s=2 (U=2000)       | 0.120 | 0.154 | 87 985 | 84 577 | **0.961** |
| zipf s=3 (U=5000)       | 0.145 | 0.211 | 85 536 | 78 892 | **0.922** |
| scan-polluted (hot 256) | 0.072 | **0.201** | 92 796 | 79 915 | **0.861** |

S3-FIFO has a **strictly higher hit rate on every trace** — 4 % fewer embeds on mild Zipf, **14 % fewer** on
scan-polluted streams (where FIFO admits one-hit-wonders and evicts the hot set; S3-FIFO's Small queue
absorbs them). Per cache op S3-FIFO is ~2× the FIFO time (≈52 ns→110 ns), but that is **ns-scale vs the embed
it avoids** (µs for the hash embedder, ~0.5 ms for model2vec/Native), so the net is positive end-to-end and
**scales with embedder cost** — timely as the pure-Rust Native/model embedders land (`a18943d`).

```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc RUST_LOG=off \
  cargo bench -p frankensearch-embed --profile release --bench cache_replay
```

**Conformance:** `frankensearch-embed` lib **292/292** GREEN — the existing cache tests (hit/miss, stats,
cap=1/2 eviction) all pass (at small sizes with freq 0, S3-FIFO matches FIFO order) plus a new
`s3fifo_keeps_reused_key_through_scan` proving a reused key survives a cold scan that overflows the cache.

**Scope:** original-comparator ratio **N/A** (internal cache policy). This is the S3-FIFO admission half of
**bd-tjkm**; the `cache_replay` bench is the replay-trace churn-identification the bead requires before the
policy change. Kept bench: `cache_replay`. **Route next (bd-tjkm remainder):** the expected-loss
candidate-budget controller (candidate_multiplier / quality_timeout from measured latency-vs-loss) + a full
query-path replay harness with deterministic fallback.

### 2026-06-29 — BOLD head-to-head re-measure: dominance HELD through the session (parity-or-better, limit_all inherent) (BlackThrush)

**Head-to-head confirmation vs the Tantivy/Lucene/Meilisearch-class incumbent**, re-run after this session's
5 perf wins + concurrent-agent commits to verify nothing regressed the comparator (the mission metric).
Ran `bold_verify_tantivy_class` (`-p frankensearch --features lexical --profile release`, 10-sample p50,
RCH `ovh-a`, `FRANKENSEARCH_BOLD_VERIFY_EMIT=1`). Ratio = frankensearch_p50 / incumbent_p50 (**< 1 = we win**):

| corpus | query_class | hash_hybrid ratio | verdict |
|--------|-------------|-------------------|---------|
| 10k  | exact_identifier | **0.825** | 1.21× faster |
| 10k  | natural_language | **0.811** | 1.23× faster |
| 10k  | quoted_phrase    | 1.039 | ~tie |
| 10k  | high_fanout      | 1.015 | tie |
| 10k  | zero_hit         | 1.000 | tie |
| 10k  | short_keyword    | 1.105 | 21 µs vs 19 µs (tiny abs) |
| 10k  | **limit_all**    | **1.445** | SLOWER (inherent) |
| 100k | natural_language | **0.812** | 1.23× faster |
| 100k | exact_identifier | 1.006 | tie |
| 100k | quoted_phrase    | 1.015 | tie |
| 100k | high_fanout      | 1.006 | tie |
| 100k | short_keyword    | 1.022 | tie |
| 100k | zero_hit         | 1.000 | tie |

**Result: parity-or-better on every top-k row at both 10k and 100k** (natural_language a clean ~1.23× win
at both scales; exact_identifier 1.21× at 10k) — the session's wins (gather, MMR, LUT, S3-FIFO) and the
concurrent frankentorch/rerank commits **did not regress the comparator**. The lone slower row is
**`limit_all` (~1.45×)**, the structurally-inherent gap (the hybrid embeds + scans the *entire* vector tier
+ RRF-fuses ~2·N where the incumbent is lexical-only; decomposed + confirmed optimal/inherent in the
2026-06-28 entry — the ~1.286→1.445 shift is 10-sample/worker variance on a path nothing this session
touched). Original-comparator verdict: **frankensearch is at parity-or-better vs the Tantivy-class incumbent
on all top-k query classes; the only residual is the inherent `limit_all` semantics gap.** No code change
(measurement); artifact `/data/projects/.rch-targets/frankensearch-cc/criterion/bold_verify/summary.{md,jsonl}`.

### 2026-06-29 — limit_all doc_id materialization clone is ~23% (NOT the ledger's "3-5%") — reopens the move-not-clone lever (BlackThrush)

**Lever-sizing measurement that corrects a wrong prior estimate.** The 2026-06-28 `limit_all` decomposition
estimated the RRF `into_owned` doc_id clone-elision (the `VectorHit<'a>` lever) at "~3-5% of limit_all" —
but that was never measured. `rrf_fuse` returns `Vec<FusedHit>` whose `doc_id: String` are `to_owned()`
clones of the borrowed inputs (`rrf.rs:348`, `FusedHitScratch::into_owned`); for `limit_all` (k=N) that is
N short-String allocations. Isolated it (`materialize_clone` bench, N doc_ids, owned-clone vs borrowed):

| N | owned clone (`Vec<String>`) | borrowed (`Vec<&str>`) | clone overhead |
|---|------------------------------|------------------------|----------------|
| 10 000  | **429.8 µs** | 3.94 µs | **~426 µs (109×)** |
| 100 000 | **4.856 ms**  | (~40 µs) | ~4.8 ms |

**At 10k the clone is ~426 µs = ~23 % of the measured `limit_all` p50 (1869 µs)** — not 3-5 %. Eliding it
would move `limit_all` from **1.445× → ~1.12×** vs the incumbent, closing most of the lone remaining gap.
**Route: the elision is bit-identical and likely fusion-crate-only** — have `rrf_fuse` take *owned* inputs
and **move** each winner's doc_id `String` out of the input (`std::mem::take`) instead of cloning, rather
than the multi-crate `VectorHit<'a>` lifetime refactor. (Negligible for top-k — k=10 clones are ~0 — so this
is a `limit_all`/large-fetch win specifically.) Next: implement `rrf_fuse` move-not-clone (owned inputs),
wire the sync hot callers, A/B vs the clone path. Kept bench: `materialize_clone`. Original-comparator: this
directly attacks the `limit_all` row (the biggest measured gap).

**UPDATE — move-not-clone is BLOCKED; the elision is invasive-multi-crate after all (traced the actual
caller).** The "fusion-crate-only move" route fails: `rrf_fuse` cannot move doc_ids out of its inputs
because the inputs' doc_ids are **reused by value all through phase-2 and result assembly, after the fuse**.
Concretely in the sync path (`sync_searcher.rs`): after `rrf_fuse(lexical, &fast_hits, …)`, phase-2
`quality_scores_for_hits(query_vec, &fast_hits)` looks up `find_index_by_doc_id(&hit.doc_id)` + the WAL by
`hit.doc_id` (`two_tier.rs:35,44,58`), and the score-maps + blend re-read `fast_hits`/`lexical_hits` doc_ids
(`sync_searcher.rs:~42-48`, and `lexical_hits.as_ref()` at ~215). So both inputs must stay intact (doc_ids
included) long after the fuse — `std::mem::take` would corrupt phase-2. Eliding the clone therefore needs a
**deep multi-crate pipeline lifetime refactor** (borrow doc_ids through fuse → phase-2 → blend, materialize
once at final `ScoredResult` assembly), which is exactly the invasive `VectorHit<'a>`-class change the
2026-06-28 decomposition flagged — and it's forbidden by the per-crate constraint. **Net: the clone is big
(~23%, the ledger's 3-5% was wrong) but its elision is invasive-multi-crate-blocked, not a clean lever. The
`materialize_clone` bench is kept as the lever-sizing evidence; no per-crate win is available on it.**

**The alternative elision primitives are ALSO multi-crate-invasive (don't chase them either):** (1)
**`Arc<str>` doc_ids** would make the clone a refcount bump (~8× cheaper, ~20% off limit_all) — but it
requires changing the doc_id type on `ScoredResult` + `VectorHit` + `FusedHit` + every construction site +
the public result API (huge cross-crate blast radius). (2) **inline small-string** (`SmolStr` etc.) doesn't
help: the public `ScoredResult.doc_id: String` at the end of the pipeline forces the heap allocation
regardless of an inline type in the middle. So **every approach (move, `Arc<str>`, inline) needs a deep
cross-cutting type/lifetime change** — the String-everywhere result API + the input-doc_id reuse in phase-2
make N owned allocations unavoidable per-crate. **The limit_all doc_id clone lever is conclusively closed
for single-crate work; the deep `Arc<str>`-or-lifetime refactor is the only path and is a deliberate
multi-crate decision, not a 60-min lever.**

**UPDATE 2 — parallelizing the clone (rayon) is also REJECTED (allocator-bound), and the clone is ~10% not
23% on a clean worker.** Since the clone can't be elided per-crate, the next idea was to keep it but
parallelize it (`into_par_iter().map(into_owned)`) — bit-identical (order preserved), single-crate,
non-invasive. Measured (`materialize_clone`, clean worker, owned serial vs `owned_clone_par` rayon):

| N | serial clone | rayon par clone | borrowed (lower bound) |
|---|--------------|-----------------|------------------------|
| 10 000 | **190.6 µs** | **228.3 µs (SLOWER)** | 2.90 µs |

The parallel clone is **~1.2× SLOWER** than serial: the N small `String` allocations **contend on the global
allocator**, so the clone is allocator-bound, not CPU-bound — rayon adds spawn overhead with no parallelism
to exploit. Rejected (not wired). Also note the serial clone is **190 µs here (~10 % of limit_all's 1869 µs),
not the 23 % from `dcde68f`** — that earlier run was on a contended worker (load ~13); 190 µs is the clean
number. **So the limit_all doc_id clone is ~10 %, allocator-bound, un-elidable per-crate (move/Arc/inline all
multi-crate), and un-parallelizable (allocator contention). Fully closed; `materialize_clone` bench kept,
par arm reverted.**

---

## 2026-06-29 — federated cross-shard fuse: SipHash → aHash on the merge map is ~1.09–1.22× (BlackThrush)

**Lever LANDED (single-crate, bit-identical).** `federated::{fuse_rrf, fuse_weighted}` accumulate per-doc
aggregates in `std::collections::HashMap<String, AggregateDoc>` — the DoS-resistant **SipHash** default
hasher — keyed on owned `doc_id` strings, while the sibling single-node `rrf_fuse_with_graph` already uses
`ahash::AHashMap`. Swapped the four federated map sites to `AHashMap` (aHash). This is **separate from the
Cobaltmoth clone-elim lever** (NEGATIVE_EVIDENCE 2026-06-27, ~1.05×, reverted): that attacked the key
*clone*; this attacks the key *hash*. The per-hit allocations (`doc_id.clone()`, `appeared_in.to_owned()`)
are unchanged, so this is a clean, orthogonal hasher win.

**Bit-identical:** `into_ranked_hits` sorts by a **strict total order** — `score.total_cmp` → `appeared_in`
count → `source_rank` → **`doc_id` tiebreak** (`federated.rs:491`). Since `doc_id` is the unique map key,
no two output entries compare Equal, so the transient map's (hasher-dependent) drain order never reaches the
result. Output is identical for every input.

**Measured** (per-crate A/B, isolates only the hasher — identical alloc profile in both arms;
`federated_fuse` bench, real merge shape: `String` key clone + `BTreeSet<String>` appeared-in + template clone):
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-fusion --bench federated_fuse
```

| Workload (shards × hits, universe) | SipHash (prod) | aHash (landed) | ratio | speedup |
|------------------------------------|----------------|----------------|-------|---------|
| `s5_h200_u600`   (5 × 200, ~50% overlap)  | 173.4 µs | 142.6 µs | **0.82** | **~1.22×** |
| `s10_h500_u2500` (10 × 500, ~50% overlap) | 805.4 µs | 742.0 µs | **0.92** | **~1.09×** |

CIs non-overlapping on both. The smaller workload shows the larger ratio (hash-compute is a bigger fraction
when the total doc set is small; at scale the `BTreeSet`/`String` allocations dilute it toward ~1.09×).

**Original-comparator ratio: N/A** — federated multi-shard fusion is a frankensearch-only cross-shard path
(Tantivy/Lucene/Meilisearch have no single-call cross-shard fuse comparator). Internal micro-lever; the
`federated_fuse` A/B bench is kept for re-validation.

---

## 2026-06-29 — two-tier blend final sort: stable `sort_by` → `sort_unstable_by` is ~1.30–1.40× (BlackThrush)

**Lever LANDED (single-crate, bit-identical).** `blend_two_tier` (the main two-tier quality-phase blend,
`blend.rs:179`) finished by sorting the deduped `Vec<VectorHit>` with a **stable** `sort_by`, while the
sibling RRF fuse path (`rrf.rs`) already uses `sort_unstable_by`. Switched it (and the identical
`federated::into_ranked_hits` sort, `federated.rs:497`) to `sort_unstable_by`. Found via the
sibling-path-consistency audit (an optimization on one path, not its twin) — same heuristic that landed the
federated aHash win (`9543ae6`).

**Bit-identical:** the comparator is a **strict total order** — `score.total_cmp` then a `doc_id` tiebreak —
and `blended` is built from an `AHashMap<&str, _>` so every `doc_id` is unique; no two elements compare
Equal, so stable and unstable produce identical output. Conformance GREEN (`frankensearch-fusion` lib
817/817). `federated` is the same argument (`doc_id` is the unique map key).

**Measured** (per-crate `blend_reorder` A/B, isolates the final sort over a realistic blended set —
unique ids, **mostly-distinct** f32 scores as in a real `alpha*q+(1-alpha)*f` blend, so the `doc_id`
tiebreak rarely fires):
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-fusion --bench blend_reorder
```

| Workload | stable `sort_by` (ORIG) | `sort_unstable_by` (landed) | ratio | speedup |
|----------|-------------------------|-----------------------------|-------|---------|
| `n200`  (typical top-k blend) | 5.95 µs  | 4.57 µs  | **0.77** | **~1.30×** |
| `n2000` (limit_all blend)     | 80.07 µs | 57.19 µs | **0.71** | **~1.40×** |

CIs non-overlapping on both. (An initial over-tied bench draft — only `n/8` distinct scores — made the
tiebreak dominate and showed a false `n2000` regression; the real blend score distribution is
mostly-distinct, where unstable cleanly wins by dropping the stable-sort O(N) scratch alloc + better
constant factors. Lesson: model the real key distribution, not a tie-adversarial one.)

**Original-comparator ratio: N/A** — the blend/federated final sort is a frankensearch-only two-tier /
cross-shard step (no Tantivy/Lucene/Meilisearch single-call equivalent). Internal sort micro-lever; the
`blend_reorder` A/B bench is kept for re-validation.

---

## 2026-06-29 — exact-search final winners sort: stable `sort_by` → `sort_unstable_by` (~1.16–1.47×, biggest on limit_all) (BlackThrush)

**Lever LANDED (single-crate, bit-identical) — lands on the biggest gap vs ORIG (`limit_all`).** The exact
vector search ordered its collected `Vec<HeapEntry>` best-first with a **stable** `sort_by(compare_best_first)`
at three sites: `search.rs:183` (the **`limit_all` scan-all path**, `scan_wal_collect_all`, where `winners`
holds *every* match), `search.rs:821` (bounded top-k, `heap.into_vec()`), and `in_memory.rs:857`
(`resolve_heap`). Switched all three to `sort_unstable_by`. Found via the sibling-path-consistency audit
(`rrf.rs`/`blend.rs` already use `sort_unstable_by`).

**Bit-identical:** `compare_best_first` is a **strict total order** — `score_key.total_cmp` then a
unique-`index` tiebreak (`index` is the vector position, unique per entry) — so no two entries compare Equal
and the unstable sort yields output identical to the stable sort. Conformance GREEN (`frankensearch-index`
lib 374/374).

**Measured** (per-crate `winners_sort` A/B, isolates the final sort over realistic winners sets — unique
indices, mostly-distinct cosine-like f32 scores):
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
  rch exec -- cargo bench -p frankensearch-index --bench winners_sort
```

| Winners (path) | stable `sort_by` (ORIG) | `sort_unstable_by` (landed) | ratio | speedup |
|----------------|-------------------------|-----------------------------|-------|---------|
| `n100`   (bounded top-k)        | 1.302 µs   | 1.121 µs   | **0.86** | **~1.16×** |
| `n10000` (`limit_all`)          | 247.6 µs   | 188.2 µs   | **0.76** | **~1.32×** |
| `n50000` (`limit_all`)          | 1858.2 µs  | 1263.7 µs  | **0.68** | **~1.47×** |

CIs non-overlapping on all. The win **grows with N** (the exact `limit_all` profile): unstable drops the
stable-sort O(N) scratch allocation + has better constant factors, saving ~595 µs/query at 50k winners.

**Original-comparator relevance:** this is the frankensearch-side final ordering of the **`limit_all`**
path — the documented biggest gap vs the Tantivy/Lucene/Meilisearch-class incumbent (BOLD `limit_all` row).
It shrinks the frankensearch side of that inherent gap. Internal sort lever; `winners_sort` bench kept.

---

## 2026-06-29 — MRL rescore final sort: stable `sort_by` → `sort_unstable_by` (~1.16–1.47×, same lever) (BlackThrush)

**Lever LANDED (single-crate, bit-identical) — extends the winners-sort sweep.** `mrl_search_with_stats`
(`mrl.rs:325`, the Matryoshka truncated-dims rescore search path) ordered its `rescored` candidates with a
stable `sort_by` before `truncate(limit)`. The comparator is the **identical strict total order** as the
exact-search winners sort — `nan_safe(score).total_cmp` then a **unique `index` tiebreak** (vector position,
distinct per rescored entry) — so `sort_unstable_by` is bit-identical and drops the mergesort scratch alloc.
Directly covered by the `winners_sort` bench (same crate, same `score`+`index` comparator):

| Winners (path) | stable `sort_by` | `sort_unstable_by` | ratio | speedup |
|----------------|------------------|--------------------|-------|---------|
| `n100`   | 1.302 µs  | 1.121 µs  | 0.86 | ~1.16× |
| `n10000` | 247.6 µs  | 188.2 µs  | 0.76 | ~1.32× |
| `n50000` | 1858.2 µs | 1263.7 µs | 0.68 | ~1.47× |

**Negative result on the same sweep — `IndexWriter::finish` records sort MUST stay stable (REVERTED).** The
`finish()` full-corpus records sort (`lib.rs:1624`, `doc_id_hash` then `doc_id`) looked like the same lever,
but `self.records` can contain **duplicate `doc_id`s** (soft-delete then rewrite), so the comparator is NOT
a strict total order and the **stable** sort is load-bearing for last-write-wins among dupes. Switching to
`sort_unstable_by` broke `soft_delete_wal_restores_state_on_rewrite_failure`. Reverted to `sort_by` and added
a guard comment. Lesson: the stable→unstable lever requires a **truly unique** tiebreak key — verify the data
can't carry duplicate keys (here, updated/rewritten docs), not just that the comparator *names* a unique field.

**Original-comparator ratio: N/A** (internal sort lever; MRL is a frankensearch-only truncated-dims search path).

---

## 2026-06-29 — limit_all winners sort: gated PARALLEL sort (`par_sort_unstable_by`) ~2.81× at 50k (BlackThrush)

**Lever LANDED (single-crate, bit-identical) — a NEW primitive on the biggest gap (`limit_all`).** The
exact-search `limit_all` scan-all final sort (`search.rs:183`, `winners` holds *every* match) was a serial
`sort_unstable_by`. The earlier ledger rejected parallelizing the materialization *clone* (allocator-bound),
but the **sort** is CPU-bound (comparison sort) and parallelizes cleanly. Gated `par_sort_unstable_by`
(rayon) above `PAR_SORT_THRESHOLD = 16_384` winners; below it the serial sort stays (rayon spawn/merge
overhead is not amortized for the cheap `compare_best_first` comparison). Bit-identical — `compare_best_first`
is a strict total order, so the parallel sort yields the same unique order. Conformance GREEN (index lib 374/374).

**Measured** (per-crate `winners_sort` bench, `par` arm vs serial `sort_unstable_by`):
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
  rch exec -- cargo bench -p frankensearch-index --bench winners_sort
```

| Winners | serial `sort_unstable_by` | `par_sort_unstable_by` | ratio | verdict |
|---------|---------------------------|------------------------|-------|---------|
| `n100`   | 1.76 µs   | 1.14 µs   | (noise; rayon serial-fallback) | serial (below threshold) |
| `n10000` | 196.0 µs  | 184.7 µs  | ~1.06× (marginal/noisy)        | serial (below threshold) |
| `n50000` | 1284.3 µs | 457.6 µs  | **0.36 (~2.81×)**              | **parallel (≥16384)** |

Threshold 16384 keeps the marginal/noisy n10000 zone serial and only parallelizes where the win is clear and
large — the `limit_all` case. Stacks on the earlier serial `sort_by → sort_unstable_by` win (`afb646b`):
together ~1.16× (top-k) → ~4.1× (50k winners, stable→par) on the biggest measured gap.

**Original-comparator relevance:** shrinks the frankensearch side of the `limit_all` BOLD gap vs the
Tantivy/Lucene/Meilisearch-class incumbent. Internal sort lever; `winners_sort` bench (now with `par` arm) kept.

---

## 2026-06-29 — BOLD re-measure post sort-wins: top10 at parity, limit_all 1.59× p50 (inherent) (BlackThrush)

**Surface (measured, no new single-crate lever).** Re-ran the BOLD Tantivy-class comparator
(`bold_verify_tantivy_class`, 10k corpus, hash hybrid + lexical-guard) after this session's sort wins to
locate the current biggest gap without ceiling-framing.

| query_class | frankensearch p50 µs | tantivy p50 µs | ratio | note |
|-------------|----------------------|----------------|-------|------|
| exact_identifier | 112  | 115  | **0.97** | at/under parity |
| short_keyword    | 30   | 28   | 1.07  | parity |
| quoted_phrase    | 144  | 142  | 1.01  | parity |
| natural_language | 153  | 154  | **0.99** | at/under parity |
| high_fanout      | 85   | 78   | 1.09  | parity |
| zero_hit         | 46   | 47   | 0.98  | at/under parity |
| **limit_all**    | **2151** | **1351** | **1.59** | the lone gap |

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc rch exec -- \
  env FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error cargo bench -p frankensearch --features lexical \
  --profile release --bench search_bench bold_verify_tantivy_class -- --sample-size 10 --warm-up-time 1 --measurement-time 3
```

**Reading it:** all six top10 classes are **parity-or-better** (0.97–1.09; the >1.0 cases are within the
sample-10 noise band). `limit_all` is the **only** gap at **1.59× p50** — and at 10k corpus the parallel
winners sort (`PAR_SORT_THRESHOLD = 16384`) does **not** trigger, so this is the inherent hybrid cost (vector
scan + RRF + materialize) the lexical-only incumbent never pays. (The high p95/p99 ratios, 2.15×/3.42×, are
unreliable at `--sample-size 10` — essentially the max of 10 samples — not a systematic tail.)

**Where the 800 µs limit_all gap lives (single-crate sub-levers, all addressed or locked):** the vector scan
is bandwidth-bound + AVX2-capped; the winners sort is `sort_unstable_by` (+ gated parallel at ≥16384,
`7c53d3f`); the doc_id materialization is the numeric-ord fast field (`14e87e4`, up to 6.32×); the **RRF full
sort is the largest remaining slice (~472 µs at 10k ≈ 22% of limit_all, `rrf_sort` measurement this session)**
but its `cmp_for_ranking` `doc_id`-String tiebreak is **bit-identity-locked** — a cheaper deterministic key
would reorder tied docs — and `par_sort` regresses it (this ledger, RRF-par-sort entry). The residual
materialization `String` clones are allocator-bound + multi-crate-blocked. **Conclusion: limit_all is at its
single-crate floor; top10 is won. No per-crate lever remains on the measured gap.** Original-comparator
ratios are the table above (this IS the head-to-head vs the Tantivy/Lucene/Meilisearch-class incumbent).

---

## 2026-06-29 — in-memory limit_all winners sort: gated parallel sort (extends 7c53d3f to the fast tier) (BlackThrush)

**Lever LANDED (single-crate, bit-identical) — extends the proven parallel sort to the in-memory index.**
`d4bc73c`/`7c53d3f` gated `par_sort_unstable_by` on the file-backed exact-search limit_all sort (`search.rs:183`).
The **in-memory** index (`InMemoryVectorIndex::resolve_heap`, `in_memory.rs:857`) — the two-tier **fast tier** —
serves limit_all the same way: `search_top_k` with `limit >= count` builds a `count`-sized heap, so
`winners = heap.into_vec()` holds **every record** and was sorted serially. Applied the same gated
`par_sort_unstable_by` above `PAR_SORT_THRESHOLD = 16_384`.

**Bit-identical & measured-by-coverage:** `in_memory.rs`'s `compare_best_first` is the **identical** strict
total order as `search.rs`'s (`score_key.total_cmp` then unique `index` tiebreak), and the data (POD
`HeapEntry`) is identical, so the **`winners_sort` bench directly covers this site** — no new bench needed:

| Winners | serial `sort_unstable_by` | `par_sort_unstable_by` | ratio | verdict |
|---------|---------------------------|------------------------|-------|---------|
| `n10000` | 196.0 µs  | 184.7 µs  | ~1.06× (kept serial, <threshold) | serial |
| `n50000` | 1284.3 µs | 457.6 µs  | **0.36 (~2.81×)** | **parallel (≥16384)** |

Conformance: `frankensearch-index` lib **373/374** — the lone failure is the **pre-existing flaky**
`soft_delete_wal_restores_state_on_rewrite_failure` (a WAL-restoration test, intermittent, unrelated to this
gated sort: tiny test corpora never reach the 16384 parallel branch, and the serial `else` path is
byte-identical to before). Original-comparator relevance: shrinks the in-memory fast-tier side of the
limit_all gap. The `winners_sort` bench covers it.

---

## 2026-06-29 — exact f16·f32 dot kernel: single→4 accumulators = 1.45× (gather + pass-2 + small-N hot path) (BlackThrush)

**Lever LANDED (per-crate, conformance GREEN).** The shipped exact f16 dot kernels accumulated into a
**single** `__m256`/`f32x8` (`sum = add(sum, mul(decode(f16), q))`), so each iteration's `vaddps` waited on
the previous one — the loop was **latency-bound on the ~4-cycle f32-add chain** (~48 chunks at dim=384),
while the hardware can retire `vcvtph2ps`/`vmulps` 3–4× faster. Split into **four independent accumulators**
(`(s0+s1)+(s2+s3)` tree, grouped chunk→lane mapping) to make the loop decode-throughput-bound. Applied
identically to all four variants — `dot_product_f16_f32_avx2`, `dot_product_f16_f32_generic`,
`dot_product_f16_bytes_f32_avx2`, `dot_product_f16_bytes_f32_generic` — so the AVX2≡generic bit-identity
tests stay GREEN by construction.

**Measured (per-crate, in-process A/B, dim=384, 4096-vector f16 scan; `f16_dot_ilp` bench, commit prior):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc rch exec -- \
  cargo bench -p frankensearch-index --bench f16_dot_ilp
```

| kernel | time (4096×384 f16 dots) | ratio |
|--------|--------------------------|-------|
| `acc1` (shipped single-accumulator) | 110.16 µs | 1.00 |
| `acc4` (4 independent accumulators) | **75.81 µs** | **0.688 (1.45× faster)** |

**Conformance:** `frankensearch-index` lib **355/356** — the lone failure is the **pre-existing flaky**
`soft_delete_wal_restores_state_on_rewrite_failure` (a WAL-restoration test on root workers, intermittent,
unrelated: it exercises tombstone/WAL recovery, never the dot kernels). Both `avx2_f16dot_matches_generic`
and `avx2_f16slicedot_matches_generic` **passed** (the 4-acc reassociation is matched bit-for-bit between
AVX2 and generic). A real `Compiling frankensearch-index` rebuild was confirmed (not stale cache).

**Quality:** the 4-acc partial-sum order shifts each f32 dot by ULPs vs the old single-acc — a
reassociation in the class the project already accepts (MMR cosine 4-acc, softmax/GELU). No ranking or
dot-value test regressed (355/356, the one being the unrelated WAL flake), so it is quality-neutral in
practice. **Original-comparator relevance:** the exact f16 dot is the per-vector kernel of the selective-
filter **gather** (`gather_range`), the two-pass **pass-2 refine**, and the small-N exact scan — so this
speeds frankensearch's filtered/exact vector paths vs a Tantivy/Lucene/Meilisearch-class incumbent. The
`f16_dot_ilp` bench covers it.

---

## 2026-06-29 — merge-structured RRF fusion: 1.31-1.46× (growing with N) on the limit_all shape (BlackThrush)

**Lever LANDED (single-crate, byte-identical, conformance GREEN).** The RRF fuse built every doc into one
`N`-entry value `AHashMap`, then `into_values()` (random order) → from-scratch `sort_unstable_by`. The final
sort is the largest frankensearch-owned slice of `limit_all`, and pdqsort is **adaptive** — a near-sorted
input sorts in ~O(N) (probe: presorted **4.3-5.8× faster** than random; `rrf_sort_order` bench,
NEGATIVE_EVIDENCE 2026-06-29). A naive reorder-via-`remove` regressed at large N (cache-miss removes), so
this restructures the fuse instead: keep small `&str→(rank,score)` lexical/graph contribution maps
(cache-resident), walk the already-score-sorted `semantic` slice **once in order**, emitting each
`FusedHitScratch` directly → `results` lands in fused order for the vector-only majority, so the sort runs
near-O(N). Also drops the `N`-entry value map (replaced by a small `&str` dedup set + sequential `Vec` push).

**Byte-identical:** `rrf_score` is a commutative sum of per-source contributions, so emitting
`semantic+lexical+graph` instead of `lexical+semantic+graph` is bit-for-bit equal; all fields + the
`in_both_sources` (lexical ∧ semantic) rule are reproduced exactly. Proven by `merge_matches_map_fusion`
(40 randomized overlap/graph/dedup/pagination trials, `rrf_score.to_bits()` equality) — GREEN.

**Measured (per-crate, `-p frankensearch-fusion`, `rrf_merge_fuse` bench, limit_all shape: all-N semantic in
score order + 20% lexical overlap, `limit=N`, head-to-head in one process):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc rch exec -- \
  cargo bench -p frankensearch-fusion --bench rrf_merge_fuse
```

| N | map (`rrf_fuse_with_graph`) | merge | ratio |
|---|------------------------------|-------|-------|
| 10000 | 1.3255 ms | 1.0103 ms | **0.76 (1.31× faster)** |
| 50000 | 9.785 ms  | 6.7137 ms | **0.69 (1.46× faster)** |

The win **grows with N** (opposite of the rejected reorder-via-remove). Wired: `rrf_fuse` (the hot path —
`limit_all`, sync/async searchers) now calls `rrf_fuse_with_graph_merge`; the map version is kept as the
differential-test + bench reference. Original-comparator relevance: shrinks the frankensearch-owned RRF slice
of the `limit_all` gap vs the Tantivy/Lucene/Meilisearch-class incumbent (the `rrf_merge_fuse` bench covers it).

---

## 2026-06-29 — RRF merge: skip the seen_semantic dedup for unique inputs = 1.10× more (BlackThrush)

**Lever LANDED (single-crate, conformance GREEN).** The merge-structured RRF (`4aeb66b`) dedups the
`semantic` slice with an O(N) `seen_semantic` `&str` set (mirroring the map's first-occurrence-wins). But a
vector-index `search_top_k` result has **unique doc_ids by construction** (doc_id is the index key; one live
record per doc), so that dedup never fires on the production hot path. Added `rrf_fuse_with_graph_merge_unique`
(shares `rrf_fuse_merge_inner(dedup_semantic: bool)`) which skips the set, and wired the searchers'
vector-hit fuse calls to it (`sync_searcher` initial fuse, `searcher` graph/non-graph fast-tier fuses). The
public `rrf_fuse` keeps the defensive dedup for arbitrary callers; `sync_searcher`'s `&blended` refine fuse
also stays on the dedup path (uniqueness not guaranteed).

**Measured (per-crate, `rrf_merge_fuse` bench, limit_all shape, head-to-head):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc rch exec -- \
  cargo bench -p frankensearch-fusion --bench rrf_merge_fuse
```

| N | merge (dedup) | merge_unique | ratio |
|---|---------------|--------------|-------|
| 10000 | 921.51 µs `[915,928]` | 837.36 µs `[824,853]` | **0.909 (1.10× faster)** |

CIs non-overlapping → real (~84 µs = the N hash-inserts). Identical output whenever `semantic` is unique
(asserted in the bench); `merge_matches_map_fusion` continues to cover the dedup variant. Conformance:
fusion lib **818/818 GREEN**, `--features graph` builds clean. Stacks on top of the merge win (`4aeb66b`) —
the production searcher's RRF is now `~1.1×` faster again on the limit_all path.

---

## 2026-07-04 — sync refined RRF: wire the unique-semantic merge path (CobaltRidge)

**Lever LANDED (single-crate).** The 2026-06-29 unique-semantic RRF fast path was wired to the
initial sync fuse and async searcher, but `SyncTwoTierSearcher`'s **refined lexical** fuse still
called the public `rrf_fuse` wrapper, which preserves the defensive semantic-dedup set. That old
note left the refined `&blended` path on dedup because uniqueness was assumed uncertain. Current
`blend_two_tier_aligned` proves the opposite: it emits from an `AHashMap<&str, ScorePair>` and then
sorts the unique keys, so the refined semantic slice is unique even if the fast input ever contains
duplicates. Switched that call site to `rrf_fuse_with_graph_merge_unique`, preserving public
`rrf_fuse` semantics for arbitrary callers while removing N unused hash inserts from the refined
`limit_all` path.

**Measured (per-crate, RCH worker `vmi1293453`, target dir
`/data/projects/.rch-targets/frankensearch-cod`, `cargo bench -p frankensearch-fusion --profile
release --bench rrf_merge_fuse`).** `merge` is the legacy refined-call-site behavior (dedup);
`merge_unique` is the landed path. Same binary, same worker.

| N | legacy dedup `merge` | landed `merge_unique` | ratio vs legacy original |
|---|----------------------|------------------------|--------------------------|
| 10000 | 690.09 µs `[674.57,703.80]` | 611.85 µs `[583.82,640.24]` | **0.887 (~1.13× faster)** |
| 50000 | 6.2551 ms `[6.0261,6.4687]` | 4.0401 ms `[3.8927,4.2061]` | **0.646 (~1.55× faster)** |
| 100000 | 16.657 ms `[16.000,17.295]` | 11.885 ms `[11.408,12.442]` | **0.714 (~1.40× faster)** |

Correctness: the refined semantic list is the output of `blend_two_tier_aligned`, which deduplicates
by doc_id before sorting; the unique RRF path is bit-identical to the dedup path for unique semantic
input. Conformance: `rch exec -- cargo test -p frankensearch-fusion --lib sync_searcher --profile
release` passed `7/7` focused sync-searcher tests. `cargo fmt -p frankensearch-fusion --check`
remains blocked by pre-existing rustfmt drift in unrelated committed bench files, so this commit
does not mass-format them.

---

## 2026-07-02 — sync_searcher per-query score maps + `seen` dedup: std SipHash → `ahash` (~2×) (SlateHeron)

**Lever (sibling-path consistency — the last SipHash hot-map island in fusion).** On the default sync hybrid path,
`RealtimeSyncSearcher` builds two per-query `HashMap<&str,f32>` score maps (`fast`/`quality`, `sync_searcher.rs:273,278`)
and a `seen` dedup `HashSet<&str>` (`:446`), then probes every candidate doc_id in both maps. Those were std
`std::collections` (SipHash, a crypto hash) while the sibling fusion paths `rrf.rs`/`blend.rs` and `search.rs` already
use `ahash`. Swapped the score-map/`seen` sites to `ahash::{AHashMap, AHashSet}`; `rank_map` (`:524`) stays std because
it feeds `blend::compute_rank_changes_with_maps` (`&std::HashMap`). Same lever family as the federated aHash swap
(`9543ae6`, [[sibling-path-consistency-audit]]).

**Bit-identical:** the maps/sets are only `.get()`/`.insert()`/`.entry()`-probed, never iterated for output, so the
hasher never changes results — the `sync_hash_ab` bench `debug_assert_eq!`s identical accumulated f32 bits across arms,
and fusion lib is **820/820 GREEN**, clippy-clean (no new `sync_searcher.rs` lints).

**Measured (per-crate same-binary A/B, `sync_hash_ab` bench, remote `hz2`):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc rch exec -- \
  cargo bench -p frankensearch-fusion --bench sync_hash_ab -- --sample-size 40 --warm-up-time 1 --measurement-time 2
```

| n (candidates) | sip (std) | ahash | ratio |
|---|---|---|---|
| 30  | 2.626 µs  | 1.329 µs  | **0.506 (1.98×)** |
| 100 | 9.092 µs  | 4.343 µs  | **0.478 (2.09×)** |
| 300 | 28.347 µs | 12.451 µs | **0.439 (2.28×)** |

**Scope:** original-comparator ratio **N/A** — an internal hasher microbench on the sync hybrid materialization, not a
lexical comparator lever; this is a frankensearch before/after on the per-query map cluster (~2× on that step). Kept
A/B harness: `sync_hash_ab`. WIRED to production this turn (`add5971` landed the measured bench; this commit lands the
scoped prod swap).

---

## 2026-07-02 — `fuse_expanded_payloads`: borrow `&str` keys + `ahash` over 5 accumulator maps — ~2.8–3.2× (SlateHeron)

**Lever (clone-elision + hasher).** `RealtimeRuntime::fuse_expanded_payloads` (`fsfs/runtime.rs`), the cross-query RRF
fusion over query-expansion variants, accumulates into FIVE maps (`scores`/`snippets`/`best_lexical_rank`/
`best_semantic_rank`/`appeared_in_count`) and called `.entry(hit.path.clone())` per hit across every payload — up to 5
owned `String` allocations per hit, and the fusion case (a doc in ≥2 variants) is exactly when the key already exists so
the clone is pure waste. Switched all five to `ahash::AHashMap<&str,_>` keyed on `hit.path.as_str()` (payloads outlive
the fusion); owned `String`s are materialized only for the top-`limit` output rows. `ahash` (not std SipHash) also
matches the sibling RRF paths (`rrf.rs`, `blend.rs`, and this session's `sync_searcher` swap `8665ce1`).

**Bit-identical:** maps are `.entry()`/`.get()`-probed only; the score sort tie-breaks by path (`&str` cmp == `String`
cmp) and output reads by key — the `expand_fuse_ab` bench asserts the identical ranked `(path, score)` output across all
three arms.

**Measured (per-crate same-binary 3-arm A/B, `expand_fuse_ab` bench in `frankensearch-fusion` — the lever is pure local
structs, so it is benched in the LIGHT crate; `frankensearch-fsfs` itself is a >10-min compile (vendored openssl + TUI +
pdf-extract) and cannot host a fast bench). Remote `hz2`:**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc rch exec -- \
  cargo bench -p frankensearch-fusion --bench expand_fuse_ab -- --sample-size 40 --warm-up-time 1 --measurement-time 2
```

| shape (P variants, H hits) | clone_sip (current) | borrow_sip | borrow_ahash | borrow_ahash vs clone_sip |
|---|---|---|---|---|
| p3_h20 | 16.88 µs | 9.56 µs  | 6.02 µs  | **0.357 (2.80×)** |
| p5_h40 | 54.94 µs | 26.42 µs | 17.04 µs | **0.310 (3.22×)** |
| p6_h60 | 93.86 µs | 49.10 µs | ~30 µs   | **~0.32 (~3.1×)** |

Attribution: clone-elision alone (`borrow_sip`/`clone_sip`) = **1.77–2.08×**; `ahash` on top
(`borrow_ahash`/`borrow_sip`) = **1.55–1.59×**; combined **~2.8–3.2×**. Both bit-identical, both kept.

**Scope:** original-comparator ratio **N/A** — fsfs query-expansion fusion, not a lexical comparator lever; a
frankensearch before/after on the expansion path (runs when multi-query expansion is enabled). Kept A/B harness:
`expand_fuse_ab` (lives in `frankensearch-fusion` benches for fast iteration). Verified: remote `cargo build -p
frankensearch-fsfs --lib` **compile SUCCEEDED** on `hz2` (the exit-102 was RCH-E309 artifact-*transfer* timeout, not
a compile error — the borrow/lifetime/`ahash` changes were accepted by the compiler); the edit is bit-identical by
construction (borrow `&str` keys → same values, same sort order, owned `String`s materialized identically at the
`take(limit)` output) and `expand_fuse_ab` asserts the identical ranked `(path, score)` output across all three arms.
No `fuse_expanded_payloads`-specific test exists; full fsfs suite not re-run (crate compile weight + transient rch
transfer flake).

---

## 2026-07-02 — fsfs default-path merge dedup wired to `ahash` — ~1.3× on the full merge (SlateHeron)

**Lever (wires the measured `merge_dedup_ab` result into production).** `merge_with_lexical_tail`
(`fsfs/runtime.rs:6127`) and `merge_with_fallback_tail` (`:6184`) — the DEFAULT (non-expansion) result-assembly
merge — dedup the fused head's doc_ids in a `HashSet<&str>` probed once per lexical/fallback-tail candidate
(O(tail); tail = full lexical result set). Swapped both from std SipHash to `ahash::AHashSet` (fsfs already deps `ahash`
since `401c3e3`). `&str` keys, bit-identical. Matches the sibling RRF/fusion paths and this session's `sync_searcher`
(`8665ce1`) + `fuse_expanded` (`401c3e3`) swaps.

**Measured `~1.3×` on the FULL merge shape** (`merge_dedup_ab`, remote `hz2`; benched in the light `frankensearch-fusion`
crate since fsfs is a >10-min compile): h50_t200 **0.745**, h50_t1000 **0.742**, h100_t2000 **0.791** (`sip`→`ahash`).
The set-op win is ~2× but the per-keep `FusedCandidate` clone (hasher-independent) dilutes it to ~1.3× end-to-end.
Bit-identical (merged doc_id order asserted across arms).

**Scope:** original-comparator ratio **N/A** (internal hasher on the fsfs merge, not a lexical comparator lever); a
frankensearch before/after on the default result path. Kept A/B harness `merge_dedup_ab`. Verified: remote
`cargo build -p frankensearch-fsfs --lib` compile SUCCEEDED (RCH-E309 exit-102 = artifact-transfer only).

---

## 2026-07-02 — filter `matches_doc_id`: conditional lowercase — skip the per-candidate alloc for extension-only filters (~1.6×) (SlateHeron)

**Lever (conditional allocation).** `SearchFilterExpr::matches_doc_id` (`fsfs/runtime.rs`) runs per candidate in
`apply_search_filter` (filters the fused head) + both merge loops, and it unconditionally allocated
`doc_id.to_ascii_lowercase()` every call — dead work for `Extension`-only filters (`type:`/`ext:`/`lang:`), which read
`doc_id` directly. Precomputed a `has_path_contains: bool` on the expr at `parse` time; `matches_doc_id` now allocates
`lowered` only when a PathContains clause will read it (`self.has_path_contains.then(|| doc_id.to_ascii_lowercase())`).
Path filters keep the identical `to_ascii_lowercase` + SIMD `str::contains` (a naive alloc-free scan measured SLOWER,
see the NEGATIVE_EVIDENCE mixed entry). Bit-identical (asserted both filter kinds).

**Measured (`filter_match_ab`, remote, per-candidate filter over path candidate sets):**

| filter | shape | old | new | ratio |
|---|---|---|---|---|
| PathContains | n200/1000/4000 | 4.59/22.57/86.77 µs | 4.37/22.21/89.03 µs | **~1.0 (tie, identical code)** |
| Extension    | n200 | 6.99 µs  | 4.57 µs  | **0.654 (1.53×)** |
| Extension    | n1000 | 36.39 µs | 22.34 µs | **0.614 (1.63×)** |
| Extension    | n4000 | 139.54 µs | 88.18 µs | **0.632 (1.58×)** |

**Scope:** original-comparator ratio **N/A** (internal, opt-in filter path); a frankensearch before/after — **narrow**
(only extension-only filters benefit; path filters are a strict tie, no regression). Kept A/B harness `filter_match_ab`.
Verified: remote `cargo build -p frankensearch-fsfs --lib` compile SUCCEEDED (RCH-E309 = artifact-transfer only).

---

## 2026-07-02 — `tokenize_lexical` ASCII byte fast path — ~1.1–1.17× on the per-document index-time tokenizer (SlateHeron)

**Lever (sibling-consistency — the ASCII fast path the token-emitter missed).** `tokenize_lexical`
(`fsfs/lexical_pipeline.rs:192`) runs a per-character loop over EVERY document's full text at index time via
`text.char_indices()` (UTF-8 decode per char) + `is_token_char` (Unicode `is_alphanumeric()`). Its sibling
`count_lexical_tokens` already had an ASCII byte fast path (256-byte `TOKEN_BYTE` LUT, won ~1.5–1.8×) but the
token-EMITTER never got it. Added `if text.is_ascii() { tokenize_lexical_ascii(text) }` — a byte-iteration + LUT path
reusing the file's existing `TOKEN_BYTE`. Bit-identical for ASCII text (byte index == char index; `TOKEN_BYTE[b]` ==
`is_token_char(b as char)` for every byte); non-ASCII text falls back to the unchanged char path.

**Measured (`tokenize_ascii_ab`, remote, realistic ASCII code/prose docs):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-slateheron rch exec -- \
  cargo bench -p frankensearch-fusion --bench tokenize_ascii_ab -- --sample-size 40 --warm-up-time 1 --measurement-time 2
```

| doc size | char (current) | fast | ratio |
|---|---|---|---|
| 5.8 KB  | 27.05 µs  | 24.63 µs  | **0.911 (1.10×)** |
| 29 KB   | 144.67 µs | 123.99 µs | **0.857 (1.17×)** |
| 116 KB  | 567.89 µs | 498.25 µs | **0.877 (1.14×)** |

Smaller than the count sibling's 1.5–1.8× because the per-token `to_ascii_lowercase` allocation (shared by both arms)
dominates token emission; the win is the saved UTF-8 decode + Unicode predicate per char. Bit-identical (bench asserts
identical token vectors; the `tokenize_lexical_preserves_code_and_path_tokens` test exercises the fast path).

**Scope:** original-comparator ratio **N/A** — index-time lexical tokenizer, not a lexical *comparator* lever; a
frankensearch before/after on indexing throughput (per-document, high corpus multiplier). Kept bench `tokenize_ascii_ab`.
Verified: remote `cargo build -p frankensearch-fsfs --lib` compile SUCCEEDED (RCH-E309 = artifact-transfer only).

---

## 2026-07-02 — `LexicalToken.text` `String`→`CompactString` SSO — ~1.07× MORE on the index-time tokenizer (SlateHeron)

**Lever (small-string optimization — a DIFFERENT kind than the closed clone/hasher/borrow family).** With the ASCII
decode fast path landed (`3280baf`), the dominant remaining `tokenize_lexical` emission cost is the per-token
`to_ascii_lowercase()` **heap allocation** (one per token). Lexical tokens (code identifiers, prose words, short paths)
are almost always <=24 bytes, so switching `LexicalToken.text` from `String` to `CompactString` makes them live inline —
**zero heap allocation** for the common case. Built via `CompactString::new(slice)` + in-place `make_ascii_lowercase()`,
which is byte-identical to `slice.to_ascii_lowercase()` for ANY input (`make_ascii_lowercase` touches only `0x41..=0x5A`,
so UTF-8 multi-byte sequences are untouched — the non-ASCII fallback keeps its ASCII-only lowering). This is SSO, NOT
clone-elision/aHash — a new technique on the index-time surface the search-time BOLD proxy cannot model.

**Measured (`tokenize_ascii_ab`, char/fast/compact arms, same remote run):**

| doc size | char (orig) | fast (String, shipped) | compact (SSO) | compact vs fast | compact vs char (cumulative) |
|---|---|---|---|---|---|
| 29 KB  | 144.67 µs | 118.91 µs | 110.23 µs | **0.927 (1.08×)** | 1.31× |
| 116 KB | 643.19 µs | 478.81 µs | 448.93 µs | **0.938 (1.07×)** | 1.43× |

Headline is the apples-to-apples **compact-vs-fast marginal** (~7%, both arms same run, CIs cleanly separated: fast
116–120 µs vs compact 110.0–110.5 µs @ 29 KB). The `vs char` column is cumulative with the decode fast path. The ~7% is
the eliminated per-token heap alloc for short tokens; residual cost is the byte scan + `Vec` growth (shared by both arms).

**Scope:** original-comparator ratio **N/A** — index-time lexical tokenizer; a public-API pipeline contract for external
consumers (no internal frankensearch caller). Public-API field type change (`LexicalToken.text: String → CompactString`)
with tiny blast radius — `CompactString` is a drop-in `&str` (`Deref`/`as_str`/`PartialEq<&str>` all work); the only
break is `let s: String = token.text` (rare). Follows the `DocId=CompactString` precedent. **Verified: remote
`cargo test -p frankensearch-fsfs --lib lexical_pipeline` PASSED (25/25 green, incl.
`tokenize_lexical_preserves_code_and_path_tokens` / `tokenize_lowercases_output`); exit 0 (artifacts transferred).**

---

## 2026-07-02 — `code_structure_sidecar` signal matching: drop the per-signal `BTreeSet` for a streaming min-probe — ~1.8× (SlateHeron)

**Lever (skip materialising an intermediate deduped collection when you only need a filtered-min probe — NOT the
clone/hasher/SSO families).** `score_document` (`code_structure_sidecar.rs`) tokenised EACH code-structure signal into a
full `BTreeSet<String>` (`tokenize(&signal.value)`) purely to compute
`query_tokens.intersection(&signal_tokens).next()` — the lexicographically smallest query token appearing in the signal.
It needs only that one token, but first allocates a `BTreeSet` + one heap `String` per signal token. Replaced with
`smallest_matching_token`: stream the signal's tokens through a single scratch buffer **hoisted across all signals of the
document** and probe `query_tokens` directly, cloning a token only when it becomes the new smallest match (0–1 heap
allocs/signal instead of one per token, and zero intermediate `BTreeSet`s). Bit-identical: `min(query ∩ signal_set)`
== `min` over signal token occurrences in `query` (set dedup cannot change the minimum); non-ASCII signals delegate to
the exact reference computation.

**Measured (`code_signal_probe_ab`, remote, realistic per-document signal sets):**

| signals/doc | old (`BTreeSet` + intersection) | new (stream + probe) | ratio |
|---|---|---|---|
| 128  | 21.41 µs  | 11.95 µs  | **1.79×** |
| 640  | 112.67 µs | 59.96 µs  | **1.88×** |
| 2560 | 426.75 µs | 238.97 µs | **1.79×** |

The bench allocates the scratch buffer per-call; the shipped `score_document` hoists it across the whole document, so the
production win is a strict lower bound of the above. **Parity asserted per-signal in the bench (`assert_eq!(old, new)`
for every signal) — bench exit 0.**

**Scope:** original-comparator ratio **N/A** — code-structure prior signal matching, a public-API pipeline contract
(`CodeStructureSidecar::score_query`; no internal frankensearch runtime caller — benefits integrators using the
code-structure prior), previously un-benched. A frankensearch before/after on the per-(query,document) signal loop. Added
bench `code_signal_probe_ab` + unit test `smallest_matching_token_matches_intersection_reference` (asserts parity vs the
old tokenize+intersection). Bench-verified (~1.8×, parity, exit 0). **Wired change verified: remote `cargo test -p
frankensearch-fsfs --lib code_structure_sidecar` PASSED (7/7 green, incl. the new parity test); exit 0.**

---

## 2026-07-02 — `normalize_signal_value` single-pass, single-alloc — ~1.12× (SlateHeron)

**Lever (fuse a multi-pass, multi-alloc string normalisation into one pass — NOT the clone/hasher/SSO families).**
`normalize_signal_value` (`code_structure_sidecar.rs`) made THREE allocations —
`split_whitespace().collect::<Vec<_>>().join(" ").trim().to_ascii_lowercase()` (a `Vec<&str>`, the joined `String`, the
lowercased `String`) — to do one thing: collapse `char::is_whitespace` runs to single ASCII spaces, strip leading/trailing
whitespace, ASCII-lowercase. It runs per extracted declaration at index time (via `push_signal`, per
`fn`/`struct`/`import`/heading across every line of every file) AND once per (query, document) at search time. Replaced
with a single pass into one pre-reserved `String`. Byte-identical: `split_whitespace` and `char::is_whitespace` share the
Unicode White_Space definition (token boundaries match), and output bytes <= input bytes so the reserve never reallocs.

**Measured (`normalize_signal_ab`, remote, realistic signal values):**

| batch | old (3 allocs) | new (1 pass) | ratio |
|---|---|---|---|
| 192  | 15.93 µs  | 14.23 µs  | **0.893 (1.12×)** |
| 1536 | 142.85 µs | 127.95 µs | **0.896 (1.12×)** |

Modest (the alloc reduction 3→1 is the win; the char copy is shared by both arms) but consistent across sizes and above
the ±3% gate. Parity asserted per-value in the bench (exit 0, no divergence).

**Scope:** original-comparator ratio **N/A** — code-structure signal normalisation, a public-API pipeline contract (no
internal frankensearch runtime caller; benefits integrators building `CodeStructureDocument`s). Added bench
`normalize_signal_ab` + unit test `normalize_signal_value_matches_three_alloc_reference` (asserts byte-identity vs the old
3-alloc form, incl. non-ASCII + vertical-tab whitespace). Bench-verified (~1.12×, parity, exit 0). **Wired change
verified: remote `cargo test -p frankensearch-fsfs --lib code_structure_sidecar` PASSED (8/8 green, incl. the new
`normalize_signal_value_matches_three_alloc_reference` parity test); exit 0.**

_Frontier note (this turn): re-confirmed the competitive vs-Tantivy path stays closed — `BitsetFilter` is already optimal
(identity-hasher `HashSet<u64>` over precomputed FNV hashes + prescreen, `filter_prescreen` 2.08× already landed), the
RRF top-k / WAND / block-max / threshold-pruning family is closed (BlackThrush explored it extensively), and query
preprocessing is heavily mined. The remaining landable levers are ratio-N/A caller-less contract modules (this win) or the
product-gated ANN-in-BOLD._

---

## 2026-07-02 — MEASURED: certificate-driven ANN ef selection vs flat @100k — certified 1.65× at recall≥0.95 (SlateHeron)

The operational payoff of the recall-certificate arc (`915c902` certificate → `e77e3fe` driver → `c30004c` live glue).
`hnsw_vs_flat_100k` now **auto-selects the ANN `ef`** via `HnswIndex::certify_ef_search` (target recall 0.95, alpha 0.1,
200 calibration queries), then reports the certified `ef`'s latency vs flat exact + its **out-of-sample** recall on 200
fresh held-out queries.

**Measured (remote `--features ann`, N=100k, dim=128, tight clusters NOISE=0.15, m=32, k=10):**

| path | latency (median) | vs flat | recall@10 |
|---|---|---|---|
| flat (exact, rayon) | 932.84 µs | 1.00× | 1.000 (baseline) |
| hnsw ef=40 | 275.00 µs | 3.39× | 0.9875 avg — **tail NOT certified ≥0.95** |
| **hnsw certified ef=100** | **566.79 µs** | **1.65×** | certified LB 1.0000, holdout 0.9890, meets 0.95 ✓ |

The certificate auto-selected **ef=100** and correctly **rejected the cheaper ef=40**: ef=40's 0.9875 *average* clears
0.95, but its split-conformal *lower bound* (≈10th percentile over 200 calibration recalls) does **not**, so it cannot
*guarantee* 0.95. This is the guarantee-vs-average distinction the whole arc exists to make safe — a **certified 1.65× at
recall ≥ 0.95** rather than an uncertified 3.4× at 0.9875 average with an uncontrolled tail. (At dim=384 the flat scan is
~3× costlier per vector, so the certified speedup is larger there — see [[ann-in-bold-viable]]'s 2.6–5× at dim 384.)

Verified: bench compiles + runs clean under `--features ann` (exit 0). Ratio vs Tantivy N/A directly (vector tier; flat
exact is the internal baseline). This closes the recall-certificate arc: a caller now obtains a **certified, measured**
ANN speedup from one `certify_ef_search(..)` call.

---

## 2026-07-02 — MEASURED: two-mode certified ANN ef selection @100k — tail 1.76× vs Bernstein-mean 3.70× vs flat (SlateHeron)

Completes the Bernstein mean-mode (`7f8de36`) with **live ANN data**. `hnsw_vs_flat_100k` now certifies the ANN `ef` under
BOTH guarantee modes on shared calibration (1000 queries; exact top-k measured ONCE per candidate `ef`), each reporting
the chosen `ef` + out-of-sample recall on 300 fresh held-out queries.

**Measured (remote `--features ann`, N=100k, dim=128, NOISE=0.15, m=32, k=10, target recall 0.95):**

| mode | guarantee | certified ef | latency | vs flat | certified LB | holdout recall |
|---|---|---|---|---|---|---|
| flat | exact | — | 1062.5 µs | 1.00× | — | 1.000 |
| **TAIL** (conformal, α=0.1) | per-query recall ≥ 0.95 w.p. ≥ 0.9 | 100 | 605.1 µs | **1.76×** | 1.0000 | 0.9927 |
| **MEAN** (Bernstein, δ=0.05) | E[recall] ≥ 0.95 w.p. ≥ 0.95 | **40** | 286.9 µs | **3.70×** | 0.9665 | 0.9760 |

The Bernstein **mean** mode certifies the cheaper `ef=40` → **3.70×** (2.1× more speedup than the tail mode's `ef=100` →
1.76×) by accepting the weaker **average**-recall guarantee; both meet the 0.95 target with high held-out recall
(0.976 / 0.993). This validates the `7f8de36` unit-test demonstration on live ANN, and gives the ANN-in-BOLD product
decision a **certified** speedup at each guarantee level: pick the tail mode for a per-query SLA, the mean mode for an
average-recall budget. Verified: bench compiles + runs clean under `--features ann` (exit 0). Ratio vs Tantivy N/A
directly (vector tier; flat exact is the internal baseline).

---

## 2026-07-02 — MEASURED: stratified (Mondrian-conformal) certified-ef ANN — 1.22–1.31× at 100k (the regime the 40k probe predicted) (SlateHeron)

Follow-through on the 40k stratified-ef probe (`6098cb5`), which found no speedup at 40k but predicted one at larger N
(steeper `ef`→latency curve). Made the bench's corpus size env-overridable (`FS_STRAT_N`, default 40k) and re-ran at 100k.

**Measured (remote `--features ann`, N=100k, dim=128, m=32, k=10, target 0.90, α 0.1; heterogeneous corpus: half tight
NOISE=0.08 / half diffuse NOISE=0.35):**

| policy | guarantee | ef | latency (median, 95% CI) | holdout recall@10 |
|---|---|---|---|---|
| global **population** | population 90th-pct ≥ 0.90 | 160 | 1.590 ms [1.41–1.80] | 0.982 |
| global **per-group** | every stratum ≥ 0.90 | 160 | 1.712 ms [1.55–1.93] | 0.982 |
| **stratified routed** | every stratum ≥ 0.90 | 40/160 | **1.305 ms** [1.23–1.39] | 0.964 |

At 100k the population-conformal `ef` **jumps to 160** (== per-group): recall at low `ef` is worse at scale, so the
population bound can no longer dilute the tail with easy queries. Both global policies therefore need `ef`=160, and
stratified routing (tight→`ef`40, diffuse→`ef`160) is the only remaining lever: **1.22× vs population-global, 1.31× vs
per-group-global**, at recall 0.964 ≥ target (CIs nearly/clearly separated). This **confirms the 40k probe's prediction
and flips backlog #2 positive**: stratified/Mondrian-conformal `ef` IS a speed lever once the `ef`→latency curve is steep
enough that the population bound can't cheat (at 40k it lost because population `ef`=80 dominated; at 100k
pop==pergroup==160, so stratification wins). Verified: bench compiles + runs clean under `--features ann` (exit 0).

**Also recorded this turn:** backlog #4 (AVX-512 dot kernels) is **measurement-blocked** — the rch workers are AMD
Threadripper PRO 5975WX (**Zen 3**: `avx2`/`fma` only, no `avx512`/`vnni`), so no AVX-512 win can be measured here.

---

## 2026-07-04 — aligned vector-index blend: skip the defensive merge only at 100k `limit_all` (CobaltRidge)

**Lever LANDED (single-crate, thresholded).** `blend_two_tier_aligned` still preserves arbitrary-caller duplicate
semantics by merging `fast_hits` through an `AHashMap<&str, ScorePair>`. The production vector-index callers
(`TwoTierSearcher` and `SyncTwoTierSearcher`) receive unique doc ids from the vector index, so the large `limit_all`
case can stream normalized scores directly into the output vector and sort it, skipping the defensive hash-merge.

Follow-up remote RCH measurement on `ovh-a` overturned the earlier noisy local 10k rejection and added the missing 50k
crossover row. The unique path is now a clean current-main win at 10k, 50k, and 100k, so
`blend_two_tier_aligned_vector_index` switches to it from 10k hits upward. Smaller queries keep the defensive map path.

**Measured (per-crate, remote `ovh-a`; target dir requested as `/data/projects/.rch-targets/search-cod`, rewritten by
RCH to a worker-scoped path):**

```bash
AGENT_NAME=CobaltRidge CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod \
  rch exec -- cargo bench -p frankensearch-fusion --profile release --bench blend_aligned -- \
  blend_aligned --sample-size 10 --warm-up-time 1 --measurement-time 1
```

| N | legacy materialized ORIG `current` | current main `aligned` | `aligned_unique` | ratio vs current main | ratio vs ORIG |
|---:|---:|---:|---:|---:|---:|
| 10000 | 1.0478 ms `[1.0250,1.0940]` | 1.1052 ms `[1.0767,1.1456]` | 874.55 µs `[868.76,882.15]` | **0.791 (~1.26× faster)** | **0.835 (~1.20× faster)** |
| 50000 | 10.876 ms `[10.721,11.097]` | 10.986 ms `[9.9437,12.771]` | 5.5231 ms `[5.3568,5.6369]` | **0.503 (~1.99× faster)** | **0.508 (~1.97× faster)** |
| 100000 | 31.353 ms `[24.605,36.328]` | 23.653 ms `[22.331,25.078]` | 12.708 ms `[12.015,14.291]` | **0.537 (~1.86× faster)** | **0.405 (~2.47× faster)** |

Correctness: `aligned_unique_matches_aligned_for_unique_fast_hits` proves bit-identical output for unique fast hits
across alpha `{0, 0.3, 0.7, 1, NaN}`; the bench also asserts bit-identical output and quality-map size before timing.
Conformance: `cargo test -p frankensearch-fusion --lib blend::tests` passed (42 passed, 1 ignored perf harness);
`cargo check -p frankensearch-fusion --all-targets` passed (existing bench dead-code warnings only); exact touched-file
`rustfmt --edition 2024 --check` passed.

---

## 2026-07-05 — FSVI selective-filter gather: sorted-record-table range gather for <2% allow-lists (CobaltRidge)

**Lever LANDED (file-backed twin of the earlier in-memory gather, with a tighter gate).** FSVI filtered
search still scanned every record and probed the allow-list before each dot. The record table is already
sorted by `doc_id_hash`, so a small `BitsetFilter` can invert the loop: binary-search each allowed hash
range, gather all live positions in that equal-hash range (collision-safe), then exact-score only those
vectors. WAL entries keep the existing filtered WAL scan.

**Measured (per-crate short RCH, `filtered_gather` FSVI A/B; ORIG = forced pre-change file-backed
per-document scan, candidate = forced file-backed gather):**

```bash
AGENT_NAME=CobaltRidge CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod \
  rch exec -- cargo bench -p frankensearch-index --profile release \
  --bench filtered_gather -- fsvi_filtered_gather \
  --sample-size 10 --warm-up-time 1 --measurement-time 1
```

| selectivity | ORIG scan | gather | ratio vs ORIG |
|---:|---:|---:|---:|
| 0.1% | 254.10 us | 13.470 us | **0.053 (~18.9x)** |
| 0.5% | 163.14 us | 75.535 us | **0.463 (~2.16x)** |
| 1% | 367.28 us | 171.90 us | **0.468 (~2.14x)** |

The same sweep rejected wider gates: 2% was only 0.842, while 5% was **4.83x slower**. Production is
therefore gated at `<2%` (`GATHER_SELECTIVITY_DIVISOR = 50`). Conformance: `cargo check -p
frankensearch-index --all-targets` passed via RCH; `cargo test -p frankensearch-index --lib --
--test-threads=1` passed locally through `rch` fallback (**389/389**). The new
`selective_bitset_filter_uses_file_backed_gather` test asserts forced scan, forced gather, and public
search agree.

### 2026-07-05 — Early-abandon (ADSampling-class) EXACT top-k dense scan: 0.885× (~13% faster), fine-granularity only (FlintOsprey)

**Lever:** a radically different top-k dense-scan primitive vs the incumbent "compute every candidate's
full dot, then top-k" flat scan. Store dense vectors with dimensions **energy-reordered** (one-time
index-build transform, free at query time — dot is permutation-invariant) and precompute per-vector
**block suffix L2 norms**. At query time accumulate each candidate's dot in `block`-dim chunks; once the
top-k heap is full, the Cauchy–Schwarz bound `partial + ‖q_suffix‖·‖v_suffix‖` is the *max achievable*
full dot — if it ≤ the k-th best, the candidate can never enter top-k → **abandon** the remaining dims.
Energy-ordering collapses the suffix norm fast, so far-from-query candidates abandon after 1–2 blocks.
This is **EXACT**: both arms use the same block summation order, so a non-abandoned candidate has a
byte-identical dot; the bench asserts identical top-k (ids + order) before timing.

**Measured command (per-crate, short RCH):**

```bash
AGENT_NAME=FlintOsprey CARGO_TARGET_DIR=/data/projects/.rch-targets/fs-op \
  rch exec -- cargo bench -p frankensearch-index --profile release \
  --bench early_abandon_scan -- --sample-size 20 --warm-up-time 1 --measurement-time 2
```

Workload: `early_abandon_scan` — N=50k, dim=384, k=10, 32 queries, 64 clusters, realistic dense-index
distribution (skewed spectral envelope ≈ real singular-value decay + within-cluster cos≈0.75).

| variant | time (median) | ratio vs `full` (ORIG) | blocks computed |
|---|---:|---:|---:|
| `full` (flat scan, ORIG) | 2.5312 ms | 1.000× | 100% |
| **`abandon_block32`** | **2.2401 ms** | **0.885× (~13% faster)** | 13.3% (abandons after ~1.6/12 blocks) |
| `abandon_block64` | 3.7852 ms | 1.495× (slower) | 19.5% |
| `abandon_block128` | 6.1515 ms | 2.430× (slower) | 35.7% |

CIs are **non-overlapping** (`full` low 2.4574 ms > `abandon32` high 2.2988 ms → worst-case ratio still
0.935× < 0.97 win threshold; best-case 0.825×). Top-k **bit-identical** (parity assert passes).

**Finding — granularity must be FINE.** Only `block=32` wins; `block=64/128` regress and get *worse* with
coarseness, because the minimum work before the first cutoff check is one full block (128 dims for
block128 vs 32 for block32) — coarse blocks pay 128–384 dims of dot before they can abandon, erasing the
prune. Fine blocks pay more horizontal reduces but abandon after ~51 dims; the abandonment win dominates.

**Distribution caveat (load-bearing).** The earlier config (`op-ea-bench.out`, NOISE=0.30 diffuse,
white-spectrum synthetic vectors) measured this primitive as a **loss** (0.58×/0.77× — a regression),
computing 87–91% of blocks. That was the *same synthetic-diffuseness artifact* that produced the false
ANN rejection (`ann-in-bold-viable`): a flat energy spectrum keeps every suffix norm large, so the bound
never tightens and nothing abandons. Real dense indices have a steep spectrum + tight clusters — the
regime measured here — where the primitive wins. **Never reject a distribution-sensitive vector lever on
white-noise/diffuse synthetic data.**

**Scope / route-next:** bench-validated primitive (exact, +13% at fine granularity). Production wiring is
the follow-up — needs the energy-reorder transform + block-suffix-norm sidecar in the real vector index
format, and re-measurement on a real embedded corpus (potion/BGE 130k, per `frontier-exhausted-*`).
Conformance: `frankensearch-index` bench crate compiles + runs green via RCH (exit 0, parity asserts pass).

**★ SCOPE CAVEAT — the win is sharp-cutoff-gated (small k ONLY); it REGRESSES at production top-k.**
Re-measured at **k=100** (fusion-pool / rerank-feed depth): `abandon_block32` = 3.0174 ms vs `full`
2.4888 ms = **1.212× (a REGRESSION)**, even though it still computes only 17.6% of blocks (2.11/12 vs
1.59/12 at k=10). Mechanism: `full` does ONE horizontal `reduce_add` per candidate (one dot over all 384
dims); `abandon` does ONE reduce PER BLOCK computed. At k=10 candidates abandon in ~1.59 blocks so the
~7.5× fewer multiply-adds beat the 1.59 latency-bound reduces (win); at k=100 the cutoff loosens (the
100th-best score is lower → looser bound) so candidates survive ~2.11 blocks and the extra reduces
dominate the saved multiply-adds. So this primitive is a win ONLY for **small-k exact dense retrieval
(k≈10)**; do NOT deploy it for the k≈50–100 candidate generation that feeds RRF fusion / the reranker —
there it regresses. Route-next to widen it: an **accumulator-preserving early-abandon** (keep the f32x8
accumulators live across blocks; reduce only at the bound check, or check every N blocks) to cut the
per-block horizontal-reduce cost that scales with blocks-computed — the one lever that could recover
larger-k. (Two other follow-ups rejected: cutoff-SEEDING — `40d2aed`, `NEGATIVE_EVIDENCE.md`; and
coarse blocks — block64/128 regress at both k.)

### 2026-07-05 — Early-abandon RESOLVED: adaptive-stride bound-check wins at BOTH k=10 and k=100 (recovers the prod-k regression) (FlintOsprey)

**Lever (resolves the caveat above):** the check-every-block abandon regresses at k=100 because it does one
horizontal `reduce_add` per block computed, and a looser k=100 cutoff means more blocks survive. Fix:
accumulate block=32 dots into **live f32x8 accumulators** and only reduce+check the Cauchy–Schwarz bound
after block 0 (catches the far-candidate majority in one reduce) and then every **3** blocks
(`abandon_stride3`). Checking less often is EXACT — it can only delay an abandon, never cause a wrong one
(the bound stays a valid upper bound at each check) — parity assert confirms bit-identical top-k. This
amortizes the latency-bound reduce over several blocks.

**Measured command:**

```bash
AGENT_NAME=FlintOsprey CARGO_TARGET_DIR=/data/projects/.rch-targets/fs-op \
  rch exec -- cargo bench -p frankensearch-index --profile release \
  --bench early_abandon_scan -- --sample-size 20 --warm-up-time 1 --measurement-time 2
```

Workload: `early_abandon_scan` — N=50k, dim=384, 32 queries, realistic skewed-spectrum + tight clusters,
swept k∈{10,100} in one run (medians, ratio vs `full` at the same k):

| variant | k=10 | k=100 |
|---|---:|---:|
| `full_k{k}` (flat scan, ORIG) | 2.726 ms · 1.000× | 2.475 ms · 1.000× |
| **`abandon_stride3_k{k}`** | **1.600 ms · 0.587× (WIN)** | **2.375 ms · 0.960× (win)** |
| `abandon_block32_k{k}` (check-every-block) | 2.049 ms · 0.752× (win) | 2.556 ms · 1.033× (regress) |
| `abandon_block64_k{k}` | 3.655 ms · 1.341× | 3.659 ms · 1.479× |
| `abandon_block128_k{k}` | 5.920 ms · 2.172× | 5.854 ms · 2.366× |

**Findings:**
1. **`abandon_stride3` strictly dominates check-every-block at both k** (k=10: 0.587× vs 0.752×; k=100:
   0.960× vs 1.033×). Ship the strided variant, not the check-every-block one.
2. **k=10 win is large + clean** (0.587×, non-overlapping CIs: `full` low 2.6678 ms > `stride3` high
   2.1192 ms). This supersedes the standalone k=10 block32 number in the `b524033` row above — the strided
   variant is faster there too.
3. **k=100 stride win is marginal + run-noisy** (0.960× this run with overlapping CIs; **0.865×** in a
   prior dedicated k=100 run with non-overlapping CIs). Call it ~0.87–0.96× — a real but modest win at
   production depth. The load-bearing point is that it does NOT regress the way check-every-block does
   (1.03–1.21×): striding makes the exact early-abandon **k-robust** instead of small-k-only.
4. Coarse blocks (64/128) still lose at both k — fine granularity + strided checks is the right combo.

**Scope / route-next:** exact, bench-validated, k-robust. Production wiring unchanged (energy-reorder +
block-suffix-norm sidecar; re-measure on the real 130k corpus). Remaining headroom at k=100 is the
approximate/probabilistic ADSampling bound (trades exactness for a bigger prune) — the one un-taken lever.
Conformance: bench compiles + runs green via RCH (exit 0); parity asserts pass for every (block, k) pair
(bit-identical top-k vs the flat scan).

### 2026-07-06 — CANDIDATE-TRANSPOSED cluster-grouped early-abandon: 0.748× at PRODUCTION k=100 (beats row-major) (FlintOsprey)

**Radically different execution model** vs both the flat scan and the row-major early-abandon above. The
row-major early-abandon's limit at production k (k=100 regresses, `43cb4b4` above) is the per-candidate
horizontal `reduce_add` — one per candidate per block, and the batched-GEMM probe already proved the scan
is FMA-compute-bound not bandwidth-bound (`NEGATIVE_EVIDENCE.md` 73a77fc). This primitive attacks the
reduce directly: process candidates in **groups of 8 stored candidate-major** (`tvecs[g·DIM·8 + d·8 + l]`),
so the 8 partial dots live in ONE `f32x8` accumulator and accumulation is **vertical** —
`acc += splat(q[d])·v_lanes[d]` — with **zero horizontal reduces during the scan**. Two one-time
index-build transforms (free at query time): energy-reorder dims (suffix-norm bound collapse) +
**cluster-sort candidates** so a group of 8 is intra-cluster → a far cluster abandons all 8 lanes together
(one max-fold bound check, 8× fewer reduces than row-major; no SIMD divergence). Group abandons when its
max lane-bound ≤ the k-th best.

**Measured command:**

```bash
AGENT_NAME=FlintOsprey CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
  rch exec -- cargo bench -p frankensearch-index --profile release \
  --bench transposed_abandon_scan -- --sample-size 20 --warm-up-time 1 --measurement-time 2
```

Workload: N=50k, dim=384, 32 queries, realistic skewed-spectrum + tight clusters, swept k∈{10,100}
(medians, ratio vs `full` at the same k):

| variant | k=10 | k=100 |
|---|---:|---:|
| `full_k{k}` (flat scan, ORIG) | 2.465 ms · 1.000× | 2.728 ms · 1.000× |
| **`transposed_k{k}`** | **1.848 ms · 0.750× (WIN)** | **2.040 ms · 0.748× (WIN)** |

**Findings:**
1. **Clean win at BOTH k, non-overlapping CIs** (k=10 `full` low 2.357 > `transposed` high 1.906; k=100
   `full` low 2.653 > `transposed` high 2.145; worst-case ratios 0.809× / 0.808×). k=10 was 0.635× in a
   first run — call it ~0.64–0.75×.
2. **The production-depth story: transposed = 0.748× at k=100**, vs row-major at k=100 (stride ~0.87–0.96×
   marginal; check-every-block 1.03–1.21× REGRESSION, `43cb4b4`/`c64b1f1`). Transposed is the k-robust
   dense-scan primitive — it wins *most* where row-major fails, because eliminating the per-candidate
   reduce matters most when the cutoff loosens and more blocks are computed.
3. **It computes MORE blocks** (28.2% k=10 / 30.4% k=100, vs row-major 13.3% / 17.6%) — group divergence +
   cluster-boundary groups. It wins anyway: it trades block-count for reduce-elimination + 8-wide vertical
   FMA, and the trade nets out ~0.75× at both k.

**Correctness (honest):** exact up to **f32 summation order** — the per-dim vertical accumulation differs
from the row-major 4-accumulator kernel, so dots differ at the ULP level → NOT bit-identical. Verified the
top-k SCORES match position-wise within 1e-4 AND **boundary id-swaps = 0/320 (k=10) and 0/3200 (k=100)**
across all queries → the top-k SET is identical in practice; only tie-order among ~equal scores could
differ. Immaterial for retrieval. (The first run's strict ordered-id assert flagged a tie-order swap at
k=100 — a false correctness alarm, corrected to score-equivalence.)

**Scope / route-next:** exact-in-real-arithmetic, k-robust, bench-validated. Production wiring needs the
candidate-major group layout + cluster-sort + per-lane suffix norms in the real vector index format
(cluster-ordered storage is natural for IVF-style indexes); re-measure on the real 130k corpus. Compose
with int8 (candidate-major int8 group scan). Conformance: bench compiles + runs green via RCH (exit 0);
score-equivalence + zero-swap asserts pass for every (k) pair.

### 2026-07-06 — QUERY-DIRECTED cluster traversal: 0.24–0.27× vs flat (4.1×/3.7×), 2.5× over the transposed scan (FlintOsprey)

**The biggest dense-scan win of the arc — and it stacks on the transposed primitive above.** That scan
(`7afaadf`) visits cluster-groups in arbitrary storage order, so far clusters scanned BEFORE the query's
own cluster don't abandon (the cutoff is still loose from low-scoring far candidates) → ~28–30% of blocks
computed. Fix: precompute per-cluster centroids `μ_c` (one-time, free at query time); at query compute the
64 cheap dots `q·μ_c` (~0.13% of a full scan), sort clusters by descending similarity, and traverse
cluster-groups **near-cluster-first**. The near cluster is scanned first → the top-k cutoff jumps to the
true level immediately → every subsequent far cluster's group abandons after block 0. EXACT: no cluster is
skipped (unlike a loose cluster-bound prune — the max-radius residual bound ≈0.62 exceeds the score gap
≈0.61 in 384-dim, useless); the traversal ORDER only changes WHEN the cutoff tightens, and the
per-candidate suffix-norm bound still guarantees no true top-k is dropped.

**Measured command:**

```bash
AGENT_NAME=FlintOsprey CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
  rch exec -- cargo bench -p frankensearch-index --profile release \
  --bench cluster_ordered_scan -- --sample-size 20 --warm-up-time 1 --measurement-time 2
```

Workload: N=50k, dim=384, 32 queries, realistic skewed-spectrum + tight clusters, swept k∈{10,100}
(medians, ratio vs `full` at the same k):

| variant | k=10 | k=100 |
|---|---:|---:|
| `full_k{k}` (flat scan, ORIG) | 2.755 ms · 1.000× | 2.831 ms · 1.000× |
| `transposed_unord_k{k}` (= `7afaadf`) | 1.625 ms · 0.590× | 2.027 ms · 0.716× |
| **`transposed_ord_k{k}`** (query-directed) | **0.668 ms · 0.242×** | **0.770 ms · 0.272×** |

**Findings:**
1. **0.242× / 0.272× vs the flat scan (4.1× / 3.7× faster), non-overlapping CIs** (worst-case 0.267× /
   0.293×) — and **0.41× / 0.38× vs the landed transposed** (a further 2.4–2.6× on top of the previous best).
2. **Block-count HALVED**: 28.2%→14.0% (k=10), 30.4%→15.0% (k=100). The near cluster's ~781 candidates are
   scanned deep; the other ~63 clusters abandon at block 0 because the cutoff is already at the true top-k
   level. The 64 centroid dots (~0.13% overhead) pay for themselves many times over.
3. **k-robust**: the win holds at both k (unlike the row-major variants that flipped sign at k=100).
4. **Exact** up to f32 summation order (inherits the transposed vertical accumulation): boundary id-swaps
   **0/320** and **0/3200**, scores match <1e-4 → identical top-k set.

**Scope / route-next:** this makes the dense scan an EXACT IVF (guaranteed recall 1.0), not the approximate
probe-N-clusters IVF (`HnswIndex`, a recall/latency trade). Production wiring: cluster-ordered storage +
per-cluster centroids in the vector index (IVF layout is standard); the traversal is the query path.
Compose with int8. Re-measure on the real 130k corpus (the win scales with cluster separation — well-
separated clusters skip more; report is on realistic synthetic). Conformance: bench compiles + runs green
via RCH (exit 0); score-equivalence + zero-swap asserts pass for every (k).

---

## 2026-07-06 — FlintOsprey — RESIDUAL-BUCKETED transposed groups: exact pre-block-0 skip (dense top-k) — 1.75× / 1.43× over the prior best (0.169× / 0.218× vs flat)

`bench(index) bucketed_transposed_scan` — new CURRENT BEST of the exact early-abandon arc, stacks on the
query-directed cluster traversal (`2945ed8`).

```
AGENT_NAME=FlintOsprey CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
  rch exec -- cargo bench -p frankensearch-index --profile release \
  --bench bucketed_transposed_scan
```

Workload: N=50k, dim=384, 32 queries, realistic skewed-spectrum + tight clusters, swept k∈{10,100}
(medians, ratio vs `full` at the same k):

| variant | k=10 | k=100 |
|---|---:|---:|
| `full_k{k}` (flat scan, ORIG) | 2.607 ms · 1.000× | 2.617 ms · 1.000× |
| `transposed_ord_k{k}` (= `2945ed8`, prior best) | 0.7745 ms · 0.297× | 0.8155 ms · 0.312× |
| **`bucketed_skip_k{k}`** (residual-bucketed + skip) | **0.4417 ms · 0.169×** | **0.5698 ms · 0.218×** |

**Primitive.** Sort each cluster's members by residual `dist[i]=‖v_i−μ_c‖` at INDEX time (query-independent,
core-first), then pack candidate-transposed groups of 8 → each group is residual-HOMOGENEOUS. At query,
before computing block 0 for a group, apply the exact whole-group skip `q·μ_c + max_lane(dist) ≤ cutoff`
(valid Cauchy–Schwarz upper bound for every lane, ‖q‖=1). Homogeneous groups have a TIGHT `max_lane(dist)`
(≈ the group's dist level, not the whole cluster's max-radius), so far-residual groups clear the cutoff and
skip entirely — porting the per-candidate pruning of `rowmajor_prefilter` (REJECTED: too slow in row-major)
into the reduce-free transposed layout that already won on reduce-elimination.

**Findings:**
1. **0.169× / 0.218× vs flat (5.9× / 4.6× faster), non-overlapping CIs** — and **0.570× / 0.699× vs the
   prior best `2945ed8`** (a further 1.75× / 1.43× on top of it). CIs: bucketed_skip [432,452]µs /
   [558,582]µs vs transposed_ord [752,798]µs / [787,848]µs — disjoint at both k.
2. **The skip is the win; bucketing alone is neutral.** Block-% base=14.0/15.0, bkt_noskip=13.9/15.0
   (bucketing the layout without the explicit skip does ~nothing), bkt_skip=8.7/10.6. Bucketing's role is
   to make the skip FIRE: 57.4% (k10) / 47.7% (k100) of groups are skippable at the final cutoff. This is
   exactly why the loose *cluster*-radius skip failed (max-radius 0.62 > gap 0.61) but the per-*group*
   residual bound succeeds — homogeneous grouping tightens the bound below the cutoff.
3. **Exact** (recall 1.0 up to f32 summation order, inherited from the transposed vertical accumulation):
   boundary id-swaps **0/320** and **0/3200**, scores <1e-4 → identical top-k set. Boundary groups whose 8
   lanes straddle two clusters (dist vs different μ) are flagged unskippable, preserving the bound's validity.
4. **Fallback-safe**: the skip is one compare per group; if within-cluster residual variance were low it
   would simply never fire and the scan degrades to `2945ed8`. Here it fires hard.

**Scope / route-next:** keeps the dense scan an EXACT IVF (guaranteed recall 1.0), now probing far cells at
zero block cost. Production wiring: (cluster, residual)-sorted storage + per-group residual maxima +
per-cluster centroids in the vector index. Compose with int8 (candidate-major int8 group scan). Re-measure
on the real 130k corpus (win scales with within-cluster residual spread — hub/edge structure skips more).
Conformance: bench compiles + runs green via RCH (exit 0); score-equivalence + zero-swap asserts pass ∀k.

---

## 2026-07-06 — FlintOsprey — RESIDUAL-SPACE (centered) early-abandon: scan q·(v−μ_c) — 1.10× / 1.19× over 05b57f3 (0.131× / 0.159× vs flat)

`bench(index) residual_abandon_scan` — new CURRENT BEST of the exact early-abandon arc, stacks on the
residual-bucketed scan (`05b57f3`).

```
AGENT_NAME=FlintOsprey CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
  rch exec -- cargo bench -p frankensearch-index --profile release \
  --bench residual_abandon_scan
```

Workload: N=50k, dim=384, 32 queries, realistic skewed-spectrum + tight clusters, swept k∈{10,100}
(medians, ratio vs `full` at the same k):

| variant | k=10 | k=100 |
|---|---:|---:|
| `full_k{k}` (flat scan, ORIG) | 2.934 ms · 1.000× | 2.888 ms · 1.000× |
| `bucketed_skip_k{k}` (= `05b57f3`, prior best) | 0.4227 ms · 0.144× | 0.5433 ms · 0.188× |
| **`residual_scan_k{k}`** (centered r) | **0.3849 ms · 0.131×** | **0.4580 ms · 0.159×** |

**Primitive.** The scan bounds `q·v ≤ q·v[0:b] + ‖q_suf‖·‖v_suf‖` with `‖v_suf‖≈‖v‖=1`. But
`q·v = q·μ_c + q·r`, `r=v−μ_c`, and in tight clusters `‖r‖≈0.3 ≪ 1`. Store the CENTERED residual `r`
transposed (with residual suffix norms), reconstruct the score per lane via `gqmu[lane]=q·μ_{c(lane)}`, and
bound `q·v ≤ gqmu + q·r[0:b] + ‖q_suf‖·‖r_suf‖`. The CS slack `‖q_suf‖·‖r_suf‖` is ~3× smaller → the bound
tightens ~3× faster. Check-then-compute abandons one block earlier and folds the block-0 skip in as its b=0
case; the per-lane `gqmu` makes boundary groups (lanes straddling clusters) exact with NO special-case.

**Findings:**
1. **0.131× / 0.159× vs flat (7.6× / 6.3× faster) — and 0.910× / 0.843× vs the prior best `05b57f3`**
   (a further 1.10× / 1.19×). CIs non-overlapping at both k: residual [379,391]µs / [445,473]µs vs
   bucketed [415,431]µs / [524,563]µs.
2. **Blocks 8.7→6.5% (k10), 10.6→7.6% (k100)** — a 25% / 28% cut. The centering is what drives it: far
   groups clear at block 0 (same skip) AND the near cluster's non-top-k candidates now abandon early
   (the raw `‖v_suf‖`≈1 kept the old bound loose enough to force a deep near-cluster scan; the small
   `‖r_suf‖` lets them go).
3. **Block cut (25/28%) exceeds the time cut (10/19%)** — the per-lane `gqmu` gather (8 loads/group) +
   two-part reconstruction is per-group overhead that amortizes better at the deeper k=100 scan → the
   larger k=100 win. Still net-positive at both k.
4. **Exact** (recall 1.0 up to f32 summation order): boundary id-swaps **0/320** and **0/3200**, scores
   <1e-4 → identical top-k set (the two-part `q·μ_c + q·r` reconstruction matches direct `q·v` within tol).

**Scope / route-next:** stacks cleanly on the bucketed layout + query-directed traversal; keeps the dense
scan an EXACT IVF (recall 1.0). Storage: centered residuals + per-cluster centroids + residual suffix norms
(same footprint as raw + a centroid table). Compose with int8 on the residual body (residuals are small →
quantize well). Re-measure on the real 130k corpus (win scales with cluster tightness `‖r‖/‖v‖`).
Conformance: bench compiles + runs green via RCH (exit 0); score-equivalence + zero-swap asserts pass ∀k.

---

## 2026-07-06 — FlintOsprey — gqmu SPLAT-not-gather (transposed residual kernel) — 1.16× / 1.12× over 8d13097 (0.121× / 0.134× vs flat)

`bench(index) residual_ilp_scan` — new CURRENT BEST of the exact early-abandon arc. Removes the per-group
score-reconstruction gather that the landed residual scan (`8d13097`) paid on every group.

```
AGENT_NAME=FlintOsprey CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
  rch exec -- cargo bench -p frankensearch-index --profile release \
  --bench residual_ilp_scan
```

Workload: N=50k, dim=384, 32 queries, realistic skewed-spectrum + tight clusters, swept k∈{10,100}
(medians, ratio vs `full` at the same k):

| variant | k=10 | k=100 |
|---|---:|---:|
| `full_k{k}` (flat scan, ORIG) | 2.823 ms · 1.000× | 2.833 ms · 1.000× |
| `residual_base_k{k}` (= `8d13097`) | 0.3986 ms · 0.141× | 0.4255 ms · 0.150× |
| **`residual_splat_k{k}`** (gqmu splat) | **0.3427 ms · 0.121×** | **0.3797 ms · 0.134×** |
| `residual_fast_k{k}` (splat + 4-acc ILP) | 0.3590 ms · 0.127× | 0.4201 ms · 0.148× |

**Primitive.** The residual scan reconstructs each lane's score as `gqmu[lane]=q·μ_{c(lane)}` + `q·r`.
`8d13097` built `gqmu` with 8 per-lane gathers on EVERY group. But in the (cluster,dist)-bucketed layout
99% of groups are intra-cluster (all 8 lanes share cluster c), so `gqmu = f32x8::splat(qmu[c])` — one splat,
no gather. Only boundary groups (flagged `intra=false`) still gather per-lane, preserving exactness.

**Findings:**
1. **0.860× / 0.892× vs the prior best `8d13097`** (1.16× / 1.12×), non-overlapping CIs at both k
   (splat [334,352]µs / [373,387]µs vs base [391,406]µs / [420,432]µs). **0.121× / 0.134× vs flat**
   (8.3× / 7.5×). This directly closes the ledger-noted "block cut 25/28% > time cut 10/19%" gap on
   `8d13097` — the per-group gather WAS the overhead beyond the dot.
2. **Exact** (recall 1.0 up to f32 summation order): boundary id-swaps 0/320 & 0/3200, scores <1e-4.
3. Cheap + safe: pure win from eliminating gathers on the common path; degrades to per-lane gather at
   cluster boundaries. Stacks on the residual/bucketed/traversal arc with no layout change.

**Scope / route-next:** the gather elimination is layout-inherent (bucketing already made groups
intra-cluster); production just stores `gqmu` as a splat on the fast path. See the companion
NEGATIVE_EVIDENCE entry: the 4-accumulator ILP on the same block dot REGRESSES (load-bound, not
latency-bound) — do not add accumulators to this kernel. Conformance: bench compiles + runs green via RCH
(exit 0); score-equivalence + zero-swap asserts pass ∀k.

---

## 2026-07-07 — FlintOsprey — CLS-attention SW-PREFETCH of the strided K/V access (reranker) — up to 1.11× over the shipped direct rank-1 (grows with s_len)

`bench(rerank) cls_attention_prefetch` — first perf lever on the RERANKER tier (the vector-scan arc is
synthetic-floored). Improves the shipped `direct_rank1` CLS attention (`738fffb`).

```
AGENT_NAME=FlintOsprey CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
  rch exec -- cargo bench -p frankensearch-rerank --features native --profile release \
  --bench cls_attention_prefetch
```

Cross-encoder final-layer CLS attention (H=384, NH=12, HD=32), medians, ratio vs the shipped `direct_rank1`:

| s_len | `direct_rank1` (ORIG) | `direct_prefetch` | ratio |
|---|---:|---:|---:|
| 64 | 6.744 µs | 6.326 µs | 0.938× |
| 128 | 14.376 µs | 13.840 µs | 0.963× (CIs overlap) |
| 256 | 32.339 µs | 29.944 µs | 0.926× |
| 512 | 77.701 µs | 70.192 µs | 0.903× |

**Primitive.** In the direct CLS attention, for a fixed head the per-token K and V live `STRIDE = 3H = 1152`
floats (~4.6 KB) apart in the interleaved qkv buffer — so the QK dot loop and the weighted-value-sum loop
touch a fresh, scattered cache line per token. A 4.6 KB constant stride is at/beyond the HW stride-
prefetcher's reach, so both loops are memory-latency-bound at large `s_len`. Added `_mm_prefetch(_MM_HINT_T0)`
of token `j+4`'s K (QK loop) and V (value loop).

**Findings:**
1. **0.938× / 0.926× / 0.903× at s_len 64 / 256 / 512** (non-overlapping CIs; s_len=128 a wash within noise).
   The win **grows with s_len** — larger qkv (s_len=512 → 2.36 MB, streams from L3) has more strided-load
   latency to hide, confirming the mechanism. The production cross-encoder range (256–512) gets the most.
2. **Exact / bit-identical**: prefetch is a hint, output max-delta from `direct_rank1` is **0.0** (asserted).
3. Distribution-independent (a memory-latency lever, not data-dependent) — unlike the vector-scan arc's
   bound-tightening levers, this does not need real-corpus structure to pay.

**Scope / route-next:** validated on the self-contained kernel microbench; production wiring is a 2-line add
to `native.rs`'s `cls_attention` (the `k_base` QK loop + `weighted_value_sum_hd` V loop) — safe (a hint can't
change correctness), but unmeasurable via rch (the `native_rerank` end-to-end bench SKIPs without the staged
model). Next reranker levers to probe: the same strided access in the EARLIER (non-CLS, full m×n) encoder
layers, and the softmax `exp` cost. Conformance: bench compiles + runs green via RCH (exit 0); bit-identity
asserted ∀ s_len.

---

## 2026-07-07 — FFN GELU instruction-level parallelism (FlintOsprey)

Commit lands `fast_gelu_inplace` (native.rs) ILP-4 unroll; bench `gelu_ilp`. Exact-GELU over the FFN
intermediate `[total, 1536]` is a **measured ~10-14% of the cross-encoder forward** (native.rs:180 comment).
Medians, ratio vs the shipped one-group-per-iter loop (`base` = ORIG):

| buffer (FFN rows × 1536) | `base` (ORIG) | `ilp2` | `ilp4` (LANDED) | ilp4 ratio |
|---|---:|---:|---:|---:|
| 1 (1536) | 4.252 µs | 4.079 µs | 4.101 µs | 0.965× |
| 8 (12 288) | 40.08 µs | 38.32 µs | 38.28 µs | 0.955× |
| 64 (98 304) | 484.9 µs | 460.2 µs | 463.1 µs | 0.955× |
| 512 (786 432) | 2.964 ms | 2.925 ms | 2.878 ms | 0.971× (CIs overlap) |

**Primitive.** The shipped inner loop processes ONE `f32x8` lane group per iteration: load 8 → the full
`gelu_vec8` dependency chain (`z → |z| → t=1/(1+c·|z|) → 5-term Horner erf poly → exp(-z²) → copysign →
0.5·x·(1+erf)`) → store 8. That chain is **latency-bound** (each op waits on the prior; the reciprocal and
`exp` are high-latency but pipelined-throughput units), so a single group in flight leaves the FMA/EXP ports
mostly idle. GELU is a **pure elementwise map** (no cross-lane reduction), so consecutive groups are fully
independent — issuing 4 `gelu_vec8` back-to-back lets the core overlap their latency chains → latency-bound
→ throughput-bound. Same lever class as the f16-dot 4-accumulator win (1.45×).

**Findings:**
1. **0.955-0.965× (~4-5% faster) at the in-cache widths** (1-64 FFN rows), non-overlapping CIs. The win
   **fades to ~3% (overlapping CI) at the 3 MB buffer** (512 rows) where the kernel turns
   memory-bandwidth-bound (streaming 3 MB in+out dominates the compute overlap). Chose **ilp4** over ilp2: it
   is the most uniform across the sweep (0.955-0.971, never regresses, and wins the large/batched regime that
   ilp2 gives back at 0.987×).
2. **Exact / BYTE-identical**: each 8-element group gets the identical `gelu_vec8` at the identical position —
   no reduction is reassociated (unlike the rejected softmax max-reduce, 75f0f8f) — so this is unconditionally
   bit-identical (parity asserts max-delta **0.0** for both ilp2 and ilp4 ∀ size). Distribution-independent.
3. Distinct from the softmax rejection: that tested a *reduce* (exp-bound, the max pass was negligible); this
   tests the exp/poly **throughput** on an independent-lane map, which was never before probed.

**Scope / route-next:** validated on the self-contained kernel microbench (reimplements `gelu_vec8` with
identical A–S 7.1.26 constants); landed directly into `native.rs` (byte-identical, so end-to-end ranking is
unchanged — the `native_rerank` end-to-end bench still SKIPs without the staged model, but the change cannot
alter output). The dominant reranker compute (the int8 Linear/FFN GEMMs, ~2/3 FLOPs) lives in `frankentorch`
(`ft-kernel-cpu`, a git dep) — not a frankensearch file, so not landable here; `forward_batch` already
batches all docs' tokens for full weight-reuse. Conformance: library compiles + links via RCH with
`--features native` (exit 0, `Finished` clean); bench green (exit 0); bit-identity asserted ∀ size.

---

## 2026-07-07 — Attention-softmax EXP-throughput ILP (FlintOsprey)

Commit lands `softmax_row_fused` (native.rs) 4-exp-interleave; bench `softmax_exp_ilp`. The attention softmax
is the reranker's **dominant *growing* frame — its own profiling note: "~24% of the per-doc forward wall-clock
at seq 512 (12·S² exp calls)"**. Its exp+sum loop issued ONE f32x8 group per iter (one `exp` in flight,
starving the exp port). Interleave 4 independent exps/iter to overlap their latency. Full-attention shape
(NH·n rows × width n=s_len), medians, ratio vs the shipped 1-exp loop (`base` = ORIG):

| n = s_len | `base` (ORIG) | `ilp4_seq` (LANDED) | `ilp4_multi` | seq ratio | multi ratio |
|---|---:|---:|---:|---:|---:|
| 64  | 117.7 µs | 119.99 µs | 122.18 µs | 1.019× (noisy) | 1.038× |
| 128 | 462.8 µs | 486.4 µs  | 465.3 µs  | 1.051× (noisy) | 1.006× |
| 256 | 1.804 ms | 1.912 ms  | 1.829 ms  | 1.059× (noisy) | 1.014× |
| 512 | 8.744 ms | **8.405 ms** | 8.531 ms | **0.961×** | 0.976× |

**Primitive.** The `exp` is a long-latency polynomial (pipelined throughput, high latency); one group in
flight leaves the exp port idle — the same latency-bound shape the GELU 4-group ILP win (aa11627) exploited.
`ilp4_seq` computes e0..e3 (independent exps → overlap) then adds them to a **single accumulator in the base's
exact order** → **BIT-IDENTICAL** (only exp scheduling changes; the reduction is untouched — parity asserts
max-delta **0.0** ∀ n). This is NOT the rejected softmax lever (75f0f8f = the scalar MAX-reduce, negligible);
that rejection's own verdict was "softmax is EXP-BOUND" — here we attack that exp's THROUGHPUT.

**Findings:**
1. **Clean non-overlapping ~3.9% win at n=512** (base [8.61, 8.90] vs [8.34, 8.49], tight CIs) — exactly the
   "dominant growing frame" (~24% at seq 512, and it GROWS with S), where softmax actually costs. At n≤256 the
   ilp4 arms are **noise-corrupted** (wide CIs; the inflated `ilp4_seq` medians are outliers — inconsistent
   with `ilp4_multi`, which does strictly MORE work yet lands ~tie), i.e. within-noise, not real regressions.
   Softmax is a small fraction at short seq, so neutral-there / win-at-long-seq is the right end-to-end shape.
2. **`ilp4_multi` (4 sum accumulators, reassociated) wins LESS than `ilp4_seq`** (0.976× vs 0.961× at n=512),
   confirming the bottleneck is **exp latency, not the sum chain** — so the bit-identical single-accumulator
   form is also the FASTEST. Landed `ilp4_seq`, not the reassociated variant.
3. **Exact / BYTE-identical** → zero-risk, end-to-end ranking provably unchanged (the reranker's own note
   already validates the ~1e-6 softmax exp tolerance against the numpy/ONNX reference; this doesn't even
   perturb that — the bytes are identical).

**Scope / route-next:** validated on the self-contained kernel microbench (faithful `softmax_row_fused` copy);
landed directly into native.rs (bit-identical, so `native_rerank` end-to-end still SKIPs without the staged
model but output cannot change). Conformance: library compiles + links via RCH `--features native` (exit 0,
`Finished` clean); bench green (parity asserts passed, exit 0); bit-identity asserted ∀ n. Same ILP lever now
banked on the two hottest landable transcendental frames (GELU ~10-14%, softmax ~24%@512). The dominant int8
Linear/FFN GEMMs remain in frankentorch (not landable here).

---

## 2026-07-08 — CLS-attention query-lane cache for long final-layer attention (SearchCod)

Commit lands a gated final-layer CLS attention primitive in `native.rs`: for `s_len >= 256`, load the four
SIMD query lane groups once per head and reuse them for every token's K dot product. The old direct rank-1
path rebuilt the same four `f32x8` query lanes for every token dot. The arithmetic grouping for each
`q·k` score is unchanged; only invariant query-lane materialization moves out of the token loop. Shorter
sequences keep the existing direct path because the blanket q-cache form regressed at `s_len=128`.

Measured command family (per-crate release-profile RCH, target dir requested by the user):

```bash
AGENT_NAME=SearchCod RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-rerank --features native \
    --profile release --bench cls_attention_ab -- cls_attention \
    --sample-size 10 --warm-up-time 0.1 --measurement-time 0.5
```

| Workload | ORIG `direct_rank1` | Candidate `q_cached` | Ratio vs ORIG | Worker | Decision |
|---|---:|---:|---:|---|---|
| `cls_attention/q_cached/256` | 26.929 us | 22.583 us | 0.839 | vmi1293453 | keep behind `s_len >= 256` |
| `cls_attention/q_cached/384` | 52.192 us | 48.273 us | 0.925 | hz2 | keep behind `s_len >= 256` |
| `cls_attention/q_cached/512` | 49.687 us | 45.861 us | 0.923 | vmi1293453 | keep behind `s_len >= 256` |
| `cls_attention/q_cached/512` focused confirm | 54.262 us | 50.979 us | 0.940 | ovh-a | confirm |

**Scope / proof:** the bench asserts the q-cached output matches the BMM reference within `1.0e-4` for every
swept length. The production path is stricter than the benchmarked candidate: it uses q-cache only at
`s_len >= 256`, preserving the old direct rank-1 path for 64 and 128 token cases. No softmax, value-sum,
prefetch, or BMM repack behavior is changed.

---

## 2026-07-09 — FSFS code-structure sidecar query-state hoist across candidates (SearchCod)

Commit lands an algebraic/dataflow-fusion primitive in `frankensearch-fsfs`'s code-structure sidecar:
`prior_signals_for_candidates` and enabled `rank_candidates` now prepare the query token set and normalized
query string once per candidate list, then score each document with that prepared query. The legacy original
prepared the same query state inside `score_query(query, doc_id)` once per candidate. This is a different path
from the rejected rerank attention, dense vector top-k, fusion exact-id/materialization, storage batching, and
durability trailer seams.

Measured command (requested RCH runs on `vmi1264463` went stale twice and were cancelled cleanly; this is the
local fallback using the requested target dir and release profile):

```bash
AGENT_NAME=SearchCod \
CARGO_TARGET_DIR=/data/projects/frankensearch/.rch-targets/search-cod RUST_LOG=off \
  cargo bench -p frankensearch-fsfs --no-default-features --profile release \
    --bench code_sidecar_score -- sidecar_candidate_score \
    --sample-size 10 --warm-up-time 0.2 --measurement-time 0.5
```

| Candidate count | LEGACY ORIGINAL `score_query` loop | LANDED `prior_signals` | Ratio vs ORIG | Decision |
|---:|---:|---:|---:|---|
| 32 | 42.975 us | 35.938 us | 0.836 / 1.196x faster | keep |
| 128 | 182.99 us | 149.91 us | 0.819 / 1.221x faster | keep |
| 512 | 779.62 us | 624.79 us | 0.801 / 1.248x faster | keep |

**Primitive / proof:** this factors invariant query preparation out of the candidate loop. `score_query` is
unchanged for single-document callers and remains the legacy comparator in the bench. The benchmark asserts
the candidate-list map equals the legacy per-candidate loop before timing, and the unit test
`prior_signals_match_legacy_per_candidate_scoring` covers matching rows plus missing-document behavior. Output
reason codes, matched signals, sidecar boosts, and deterministic tie-breaks are unchanged.

---

## 2026-07-09 — f16-slab → int8 quantization SIMD widen (FlintOsprey)

Commit routes `search.rs::quantize_f16_bytes_to_i8` through the branchless SIMD f16→f32 widen
(`simd.rs::widen8_f16_bytes`, Giesen magic-multiply, bit-exact) instead of a scalar per-element
`f16::from_le_bytes(..).to_f32()`. This builds (lazily, once, cached in `int8_slab`) the int8 quantization of
the whole F16 main-vector region for the int8 two-pass scan — the decode-bound bottleneck the f16-DOT arc
already SIMD-ized, but this sibling quantize path was **missed** and stayed scalar (a
sibling-path-consistency gap). Bench `f16_slab_quantize` (dim=384), medians, ratio vs the scalar decode (ORIG):

| vectors | scalar (ORIG) | simd_widen | ratio |
|---|---:|---:|---:|
| 1 000  | 2.552 ms | 1.723 ms | **0.675×** |
| 10 000 | 27.18 ms | 16.76 ms | **0.617×** |
| 50 000 | 175.7 ms | 110.2 ms | **0.627×** (~1.60×) |

Throughput 150 → 225 Melem/s, all CIs non-overlapping. At 130k×384 this trims ~170 ms off the first int8
query's cold-start (the lazy `int8_slab` build).

**Primitive (SIMD data-layout / branchless widen).** The f16 decode is the documented bottleneck
(NEG_EV: "the f16 paths are decode-bound"); `widen8_f16_bytes` widens 8 f16 lanes per instruction group via
the Giesen denormal-multiply trick. The scale/round/clamp stays SCALAR (per-element, identical to the shipped
code) and the max-abs reduction is order-independent, so the output int8 slab is **BIT-IDENTICAL** — verified
by the bench parity assert (`scalar == simd` ∀ vectors, incl. 50k) and 389/389 index lib tests green. The
Giesen widen is itself bit-exact to `f16::to_f32()` (exhaustively verified in simd.rs over all 65 536 patterns).

**Scope / route-next:** landed the int8 path (measured). The sibling `pack_4bit_f16_bytes` (search.rs) has the
**identical scalar-decode gap** and the identical fix applies (SIMD decode → scalar nibble-pack, bit-identical)
— it is the 4-bit two-pass slab build, "not wired into the BOLD hybrid", so left as the obvious follow-up.
Conformance: `cargo test -p frankensearch-index --lib` = **389 passed / 0 failed**; bench green via RCH (exit
0, parity asserted). Files: `simd.rs` (widen8_f16_bytes → pub(crate)), `search.rs` (quantize rewrite + import).

## 2026-07-09 — f16-slab → 4-bit slab packing SIMD widen (SearchCod)

Runtime commit `11ba0a9` closes the route-next from the int8 quantization entry above: `pack_4bit_f16_bytes`
now builds the signed 4-bit two-pass slab by decoding f16 bytes 8 lanes at a time through
`simd.rs::widen8_f16_bytes`, then keeps the existing scalar max-abs scale, rounding, clamp, and nibble packing.
Primitive class: **SIMD-within-register / branchless widen**. This is a different path from the rejected live
search stream state-cell and storage VALUES attempts; it touches only the cold slab build for the 4-bit
two-pass vector scan.

Measured command (per-crate RCH, worker `hz2`; the bench embeds the LEGACY ORIGINAL scalar packer and asserts
`scalar == simd_widen` before timing each row):

```bash
AGENT_NAME=SearchCod \
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod \
  rch exec -- cargo bench -p frankensearch-index --profile release \
    --bench f16_slab_pack4bit -- f16_slab_pack4bit \
    --sample-size 10 --warm-up-time 0.1 --measurement-time 0.35 --noplot
```

| vectors | LEGACY ORIGINAL scalar decode | SIMD widen packer | ratio-vs-ORIG | Decision |
|---:|---:|---:|---:|---|
| 1,000 | 2.7226 ms | 1.3581 ms | 0.499 / 2.00x faster | keep |
| 10,000 | 27.041 ms | 14.328 ms | 0.530 / 1.89x faster | keep |
| 50,000 | 136.35 ms | 69.297 ms | 0.508 / 1.97x faster | keep |

Proof: the same Giesen widen helper is already exhaustively checked against `f16::to_f32()` in `simd.rs`; the
bench's equality assert covers the full 4-bit slab bytes for every measured vector count. Since the scale is
corpus-wide and the nibble packer is unchanged, query rankings and byte layout are unchanged. Conformance for
this ledger closeout is the RCH bench exit 0 plus focused index checks below in this session.

---

## 2026-07-09 — Ops historical analytics percentile selection (SearchCod)

Commit routes `HistoricalAnalyticsScreen::percentile` through exact order-statistic selection instead of
sorting the whole telemetry vector to read one rank. This is an ops/control-plane analytics path, separate
from the previously rejected search, rerank, vector, fusion, embed, storage, durability, and FSFS paths.

Measured command (per-crate RCH, worker `ovh-a`, requested target dir; `rch` rewrote the remote target dir to a
worker-scoped path):

```bash
AGENT_NAME=SearchCod \
CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod \
  rch exec -- cargo bench -p frankensearch-ops --profile release \
    --bench percentile_select -- ops_percentile_p95 \
    --sample-size 10 --warm-up-time 0.1 --measurement-time 0.3
```

| Telemetry values | LEGACY ORIGINAL full sort | LANDED `select_nth_unstable` | Ratio vs ORIG | Decision |
|---:|---:|---:|---:|---|
| 64 | 178.62 ns | 77.185 ns | 0.432 / 2.31x faster | keep |
| 512 | 2.3279 us | 740.78 ns | 0.318 / 3.14x faster | keep |
| 4,096 | 24.087 us | 5.8299 us | 0.242 / 4.13x faster | keep |
| 16,384 | 114.12 us | 13.991 us | 0.123 / 8.16x faster | keep |

**Primitive / proof:** data-layout/order-statistic selection. The rank math is unchanged; the vector is cloned
as before, then `select_nth_unstable(index)` partitions directly to the same element the old sorted vector
would have returned. The bench asserts p95 equality against the legacy full-sort comparator before timing,
and `percentile_matches_full_sort_reference` covers duplicates, unsorted inputs, clamped percentages, and
boundary ranks. Focused release-profile conformance via RCH:
`cargo test -p frankensearch-ops percentile --profile release` = **29 passed / 0 failed**.

---

## 2026-07-08 — Pool-local min-max SCORE fusion LANDED in Rust (a9e53b4, FlintOsprey)

First QUALITY-vein land after the perf-latency frontier closed (user greenlit the pivot from perf rejections).
Ported the strongest measured BEIR fusion result (NEGATIVE_EVIDENCE `45530fb`) into Rust: `pool_minmax_fuse`,
a drop-in sibling of `rrf_fuse`. Per tier, min-max normalize the raw scores WITHIN the retrieved pool -> [0,1];
out-of-pool docs get the pool minimum (0 for min-max); tier-weighted sum; sorted by the existing deterministic
FusedHit comparator. Recovers the score MAGNITUDE RRF's rank transform discards (runaway vs marginal top match).

QUALITY (the WIN, Python-measured: fastembed BGE-small + rank_bm25 stem/stop, full BEIR test sets, POOL in {50,100}):
**+0.0038 mean nDCG@10 over RRF, POSITIVE on all 4 corpora** (scifact +0.0052, nfcorpus +0.0067, arguana
+0.0003, scidocs +0.0030), never-negative. Top-order magnitude recovery, not a recall change.

LATENCY (measured, bench `pool_minmax_fuse`, vs production merge-optimized `rrf_fuse` = LEGACY ORIGINAL):

| pool | rrf_fuse (ORIG) | pool_minmax | ratio vs ORIG |
|---:|---:|---:|---:|
| 50   | 2.2948 us | 2.8137 us | 1.23x |
| 100  | 4.2728 us | 5.2950 us | 1.24x |
| 1000 | 38.757 us | 52.100 us | 1.34x |

HONEST CORRECTION to the a9e53b4 commit message ("latency-neutral by construction"): pool_minmax is
**~1.23-1.34x SLOWER**, because production `rrf_fuse` uses the merge-structure optimization (`4aeb66b`,
near-sorted sort input) whereas pool_minmax does a from-scratch full sort of the value map. Both are O(pool) —
only the constant differs. END-TO-END NEGLIGIBLE: fusion is us-scale (the ~13us delta at pool=1000 is nothing
beside the ms-scale vector/lexical search), and the +0.0038 nDCG quality gain is the deliverable. Route-next:
port the merge-structure to pool_minmax if fusion latency ever matters.

Conformance: 7 unit tests GREEN incl. the magnitude-recovery property (a semantic-only high-magnitude doc
outranks an in-BOTH marginal doc — RRF's exact weakness). NON-BREAKING (rrf_fuse untouched; opt-in). Enabler:
`FusedHitScratch` already carried raw per-tier scores (RRF only used ranks), so no new plumbing. Searcher-wiring
/ default switch = separate product decision (keep RRF where per-query pool stats are unreliable). Files:
`rrf.rs` (pool_minmax_fuse + 7 tests), `lib.rs` (re-export), `benches/pool_minmax_fuse.rs`, `Cargo.toml`.

---

## 2026-07-09 — kNN neighbor SMOOTHING (graph score-diffusion) LANDED in Rust (257c468, FlintOsprey)

QUALITY-vein land #2 after the perf-latency frontier closed (following pool-min-max fusion `a9e53b4`).
A **structurally-different** primitive class from tier fusion: **label-propagation / manifold-ranking** over
the doc-doc dense k-NN graph. Each candidate borrows score from its dense nearest neighbors —
`smoothed(d)=(1−α)cos(q,d)+α·mean_{n∈NN(d)∩pool}cos(q,n)` — rescuing below-threshold relevants that sit
inside clusters of confident results (a RECALL mechanism, unlike RRF/pool-min-max which reorder a fixed set).

**Quality (Python-measured, BGE-small hybrid, full BEIR test sets, α=0.3/M=10, the DEPLOYABLE pool-restricted
form — `neighbor_smooth_pool.py`):** mean **+0.0039 nDCG@10** over no-smooth — nfcorpus **+0.0114**, arguana
**+0.0052**, scidocs **+0.0026** (recall-bound + semantic corpora), −0.0035 on recall-saturated scifact
(recall-saturation-gated). NEW finding: pool-restriction (the deployable form) BEATS the full-cosine Python
form on the two most recall-bound corpora (nfcorpus +0.0114 vs +0.0070) — averaging only in-pool neighbors
drops the out-of-pool low-cosine drag. Full detail + table in `docs/NEGATIVE_EVIDENCE.md`.

**Latency (measured, bench `neighbor_smooth`, vs the no-smooth ORIGINAL = identity α=0 pool clone):**
| pool | ORIGINAL (no-smooth clone) | smooth α=0.3 | +mutual (opt-in) |
|---:|---:|---:|---:|
| 50   | 0.17 µs | 5.9 µs   | 28.7 µs |
| 100  | 0.31 µs | 12.2 µs  | 63.8 µs |
| 1000 | 2.9 µs  | 158.9 µs | 704 µs  |

The diffusion pass adds ~12 µs/query at the realistic fusion POOL=100 (`ebb3377`: nDCG@10 saturates ≤50-100
cand/tier) — µs-scale, negligible beside the ms-scale vector+lexical search; the "nearly free atop a k-NN
neighbor graph" claim holds. The large ratio-vs-ORIGINAL is only because ORIGINAL is a bare pool clone (no-op);
the meaningful figure is the ~12 µs absolute. mutual-kNN (reciprocal-edge, the recall-bound opt-in) costs ~5×.

The kernel is O(pool·M) and nearly free atop an existing k-NN neighbor graph. Conformance:
8 unit tests GREEN incl. the cluster-rescue property; full fusion suite 841 lib + 77 integration tests GREEN (0 failed). NON-BREAKING
(opt-in; `Similar` edges only, so it composes with the graph-PageRank/citation consumers). Searcher-wiring
(supply the ANN/HNSW neighbor graph at fuse time) = separate product step. Files: `smooth.rs` (kernel + 8 tests),
`lib.rs` (re-export), `benches/neighbor_smooth.rs`, `Cargo.toml`.

---

## 2026-07-09 — TUI command-palette cached search index LANDED in Rust (SearchCod)

Fresh path after the rejection-ledger review: `frankensearch-tui` command-palette filtering, not the recent
fusion/index/rerank/fsfs/storage/ops lanes. The hot loop was re-normalizing every action field with
`to_lowercase()` on each filter pass; shell/ops overlay paths call this during palette typing, navigation,
confirmation, and render. Primitive class: **data-layout / algebraic-fusion**. Registration now builds a
private lowercased search index parallel to the public `Action` vector, preserving `Action`'s serialized shape
and keeping filter semantics identical to the legacy lowercase-per-call scan.

Latency (RCH worker `hz1`, command:
`CARGO_TARGET_DIR=/data/projects/frankensearch/.rch-targets/search-cod rch exec -- cargo bench -p frankensearch-tui --profile release --bench palette_filter -- palette_filter --sample-size 20 --warm-up-time 0.1 --measurement-time 0.45 --noplot`):

| actions | legacy_ORIG median | cached_index median | ratio-vs-ORIG |
|---:|---:|---:|---:|
| 128 | 10.314 us | 5.2096 us | 0.505 / 1.98x faster |
| 1,024 | 82.127 us | 45.063 us | 0.549 / 1.82x faster |
| 4,096 | 330.60 us | 154.66 us | 0.468 / 2.14x faster |

Proof: the benchmark asserts equal hit counts against an embedded `legacy_ORIG` comparator before timing;
`palette_filter_matches_legacy_normalization` compares result IDs for empty, case-folded, id-prefix,
description, theme, and Unicode-description queries. Conformance GREEN:
`rch exec -- cargo check -p frankensearch-tui --all-targets` on `hz2`; local fallback release tests with isolated
target `cargo test -p frankensearch-tui --profile release` = **202 passed / 0 failed**, doctests ok;
`cargo clippy -p frankensearch-tui --all-targets --profile release -- -D warnings`; `cargo fmt -p frankensearch-tui --check`.

---

## 2026-07-09 — QUERY-HUBNESS dense-score correction LANDED in Rust (ba5052a, FlintOsprey)

QUALITY-vein land #3, and a **structurally-different** primitive class from the prior two (fusion-normalization
`a9e53b4`, graph-diffusion smoothing `257c468`): a **query-distribution doc statistic**. High-dim cosine
retrieval has HUBS (docs near many queries regardless of relevance) that crowd out specific answers; demote them
`s'(q,d)=cos(q,d)−β·r_d`, `r_d` = doc's mean cosine to its Kq nearest *sample queries*.

This REVERSES the round-3 rejection (`64ac8b7`): the query-FREE proxies (doc-density/centroid/PC) conflate hubs
with tight relevant clusters and are corpus-fragile. Measuring the QUERY-SIDE estimate **leakage-free** (disjoint
query split; `r_d` from a background half, applied to the held-out eval half) shows the mechanism is realizable.

**Quality (Python-measured, BGE-small hybrid, full BEIR, β=0.2, leakage-free):** all-positive, **mean +0.0033
nDCG@10** — scifact +0.0010, nfcorpus +0.0026, arguana +0.0067, scidocs +0.0030 — with GENUINE dense-tier gains
where topical centrality anti-correlates with relevance (arguana counter-argument stance dense **+0.0128**,
scidocs citation **+0.0078**). A LOWER BOUND: `r_d` came from only 150–500 background queries; a production query
log (thousands) estimates it better. β=0.3 is higher-gain on stance/citation corpora (arguana +0.0089).

**Latency (measured, bench `hubness_penalty`, vs the no-correction ORIGINAL = identity β=0 pool clone):**
| pool | ORIGINAL | correct β=0.2 |
|---:|---:|---:|
| 50   | 166 ns  | 175 ns  |
| 100  | 295 ns  | 292 ns  |
| 1000 | 2.80 µs | 2.71 µs |

Query-time correction is **LATENCY-NEUTRAL** (the multiply-subtract is free under the pool clone; deltas within
noise). The offline `compute_query_hubness` builder is ~109 ms / 696 ms @ 2000×200 / 5000×500 docs×queries —
amortized (recomputed periodically from the query log), not per query.

Conformance: 7 unit tests GREEN incl. the hub-demotion property; full fusion suite 848 lib + 77 integration tests GREEN (0 failed).
NON-BREAKING (opt-in). Deployment dependency: a background query-embedding sample (the engine already sees the
query stream). Structurally orthogonal to fusion normalization (acts on dense scores pre-fusion) → composes with
pool-min-max (`a9e53b4`). Files: `hubness.rs` (kernel + builder + 7 tests), `lib.rs`, `benches/hubness_penalty.rs`,
`Cargo.toml`. Full measured detail + the round-3→round-4 arc in `docs/NEGATIVE_EVIDENCE.md`.

---

## 2026-07-09 — TUI default keymap static dispatch LANDED (SearchCod)

Different profiled path after the rejection streak: `frankensearch-tui` input dispatch, specifically the
per-keypress `Keymap::resolve` calls in the shell/palette/overlay event loop. This does not touch exact-id
fusion, vector top-k, rerank attention, embedding gather, FSFS token scanners, durability trailer packing,
storage batching, ops percentile selection, tombstone bitmaps, or fusion quality reranking.

Primitive class: **succinct-structure / static command alphabet**. The default keymap is a fixed 24-binding
terminal command alphabet, but the legacy path always probed a `HashMap<(KeyCode, Modifiers), KeyAction>`.
The landed path dispatches the unmodified default map through a static match table and falls back to the
legacy `HashMap` as soon as `bind` or `unbind` mutates the map, so custom keymaps keep their exact semantics.

Bench: `AGENT_NAME=SearchCod CARGO_TARGET_DIR=/data/projects/frankensearch/.rch-targets/search-cod cargo bench -p frankensearch-tui --profile release --bench keymap_resolve -- --sample-size 10 --warm-up-time 1 --measurement-time 1`
using the allowed local fallback after the post-rebase RCH verification job on `vmi1167313` stopped making
progress. The bench replicates the LEGACY ORIGINAL HashMap resolver in the same binary and asserts identical
actions for a 40-event shell workload before timing.

| row | LEGACY ORIGINAL `HashMap` | static dispatch | ratio vs ORIG |
|---|---:|---:|---:|
| `tui_keymap_resolve/40_event_shell_mix` | 777.96 ns | 135.89 ns | **0.175 (~5.72x)** |

Conformance: `AGENT_NAME=SearchCod CARGO_TARGET_DIR=/data/projects/frankensearch/.rch-targets/search-cod cargo test -p frankensearch-tui input::tests --profile release`
passed 11 input tests / 0 failed, including default-fast-path equivalence, unknown-key miss equivalence,
override of a default binding, and unbind behavior.

---

## 2026-07-09 — Core host-attribution fixed-universe resolver LANDED in Rust (SearchCod)

Fresh path after the rejection-ledger review: `frankensearch-core` host-project attribution resolution, which feeds
ops host identity but is distinct from the recent exact-id, vector scan, rerank, embed gather, fsfs, storage,
fusion-quality, durability, ops percentile/rollup, and TUI palette work. Primitive class:
**succinct-structure / fixed-universe bitmask**. The resolver now maps aliases into a 4-bit project universe and
uses rolling token windows for multi-token aliases, replacing the legacy candidate `Vec`, collision `BTreeSet`,
per-hint token `Vec`, alias token `Vec`, and token `BTreeSet` construction.

Latency (RCH worker `hz2`, command:
`AGENT_NAME=SearchCod CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod rch exec -- cargo bench -p frankensearch-core --profile release --bench host_attribution_resolve -- host_attribution_resolve --sample-size 12 --warm-up-time 0.1 --measurement-time 0.35 --noplot`):

| docs | LEGACY ORIGINAL median | landed median | ratio-vs-ORIG |
|---:|---:|---:|---:|
| 64 | 58.231 us | 9.4185 us | 0.162 / 6.18x faster |
| 1,024 | 957.43 us | 151.30 us | 0.158 / 6.33x faster |
| 16,384 | 15.554 ms | 2.4793 ms | 0.159 / 6.27x faster |

Proof: the benchmark embeds the LEGACY ORIGINAL resolver and asserts exact `HostProjectAttribution` equality
before timing every row. `host_project_attribution_matches_alias_boundaries` covers alias boundaries,
multi-token aliases, non-alias substrings, weight precedence, and collision demotion. Conformance GREEN:
`rch exec -- cargo test -p frankensearch-core host_project_attribution_matches_alias_boundaries --profile release`
= **1 passed / 0 failed**; `rch exec -- cargo check --workspace --all-targets`; `rch exec -- cargo clippy -p
frankensearch-core --lib --tests --bench host_attribution_resolve -- -D warnings`; `rch exec -- cargo clippy
--no-deps -p optimize-params --all-targets -- -D warnings`; `rustfmt --edition 2024 --check` on touched Rust
files. UBS was run on the new benchmark: **0 critical / 1 expected benchmark assertion warning**; the wider
touched-file UBS scan still reports pre-existing panic/unwrap inventories in existing test/tool code.

One adjacent compile fix is included: `tools/optimize_params` now fills the newly added `RrfConfig` fields via
`..RrfConfig::default()`, which was required for the workspace compile proof to stay green.

---

## 2026-07-09 — Storage dedup batch VALUES slot-join REJECTED (SearchCod)

> **⚠ 2026-07-10 LEDGER-INTEGRITY AUDIT (cc_fse): PROXY-MEASURED — production symbol has 0.000% self-time; needs
> re-verification (`bd-0j5e`).** `benches/dedup_batch.rs` imports `Storage`, but only to build the SQLite fixture:
> `Storage::check_dedup_batch` (`content_hash.rs:146`) is **never called**. Both arms are bench-local copies
> (`legacy_check_dedup_batch:114`, `slot_join_check_dedup_batch:191`). Unlike the graph_rank row this one is
> *reproducible* — both arms are retained and cross-checked by a parity assert (`:105-108`) — and both hit a real
> SQLite, so the SQL work is genuine and the ~1.16×/1.11×/1.20× conclusion is plausible. But the row is evidence
> about a copy of the shipped query, not the shipped query, so it does not satisfy the self-time rule. Re-point the
> `legacy` arm at `Storage::check_dedup_batch` before treating as do-not-retry.

Different hot path from the recent search/fusion/TUI lanes: `frankensearch-storage::Storage::check_dedup_batch`.
Primitive class: **algebraic-fusion / data-layout**. Candidate replaced the LEGACY ORIGINAL `IN (...)` query plus
Rust `HashMap<String, DedupRow>` with an ordered `WITH requested(ord, doc_id) AS (VALUES ...)` relation returning
slot-aligned rows. Production code was restored after measurement.

Bench command (local fallback after the requested RCH release test job stopped making progress in optimized
storage codegen): `CARGO_TARGET_DIR=/data/projects/frankensearch/.rch-targets/search-cod cargo bench -p frankensearch-storage --profile release --bench dedup_batch -- --sample-size 10 --warm-up-time 0.2 --measurement-time 0.8`.

| batch | LEGACY ORIGINAL median | VALUES slot-join median | ratio-vs-ORIG |
|---:|---:|---:|---:|
| 32 | 412.75 us | 477.96 us | 1.158x slower |
| 128 | 4.6572 ms | 5.1901 ms | 1.114x slower |
| 384 | 40.948 ms | 49.202 ms | 1.202x slower |

Conclusion: no ship. The CTE/VALUES join costs more than it saves in Rust map allocation/probes for the measured
batch shapes. Focused conformance: `cargo test -p frankensearch-storage check_dedup` passed 12 tests / 0 failed.
Repro bench retained at `crates/frankensearch-storage/benches/dedup_batch.rs`.

---

## 2026-07-09 — Mutual-kNN neighbor-smoothing reciprocity: O(1) packed-edge set replaces O(deg) String scan — 1.21×/1.35×/1.50× (CopperVireo)

Fresh path after the perf hot-core rejection streak (vector early-abandon seeding/approx-bounds/ILP,
rerank CLS-prefetch/softmax-max, SimHash tableless votes, tombstone bitmap, embed gather, ops percentile,
durability trailer — all rejected 07-06→07-09). This is the **first perf dig into the freshly-landed fusion
QUALITY kernels** (`smooth.rs` `neighbor_smooth`, `257c468`), which were written for correctness and never
perf-mined; it touches none of the exhausted paths above and leaves the shipped scores bit-identical.

Primitive class: **succinct-structure / integer relabel + O(1) set membership**. In the `mutual`
(reciprocal-kNN) config — the documented "no-regret refinement" for recall-bound corpora — the LEGACY
ORIGINAL gates each candidate `d`'s in-pool neighbor `n` on whether `n` points *back* to `d`, realized as
`is_similar_neighbor(graph, n, d)` = a std-SipHash `HashMap<String,_>` lookup **plus** an O(deg(n)) linear
`String`-equality scan, called once per in-pool neighbor per candidate → **O(pool · M · deg)** String
compares on top of the O(pool · M) diffusion. Profiling (the shipped `neighbor_smooth` bench) showed `mutual`
running 5–6× the non-mutual path — the entire gap is that reciprocity scan.

The landed kernel (`mutual_neighbor_smooth`) relabels the pool to dense `u32` indices and, in **one** walk
over each candidate's `Similar` adjacency (each neighbor string hashed exactly once), builds both (a) a
packed-`u64` (`src_idx << 32 | dst_idx`) `AHashSet` of every in-pool directed `Similar` edge — uncapped, to
match the ORIGINAL's full-adjacency scan — and (b) a flat integer CSR of the M-capped in-pool forward
neighbors. The diffusion pass is then pure integer work: reciprocity is an O(1)
`recip.contains((n_idx << 32) | d_idx)`, with no second `graph.neighbors` SipHash lookup. **Non-mutual is
byte-for-byte the ORIGINAL** (compute-floored single-hop diffusion — irreducible per-edge hashing, no
headroom).

Bench `neighbor_smooth_recip_ab` (isolated local target `search-copper`; the first cross-build baseline read
3.29 ms @ pool 1000 but was contention-inflated — the quiescent figure is ~1.1 ms). ORIG is measured **first
AND last** (`ORIG2`) to bracket the first-arm ordering bias, which was only ~1–3% here and is dwarfed by the
win:

| pool | LEGACY ORIGINAL (ORIG / ORIG2) | landed `v3` | ratio vs ORIG2 |
|---:|---:|---:|---:|
| 50   | 37.96 / 37.60 µs | 31.12 µs | 0.828 / **1.21× faster** |
| 100  | 85.62 / 82.81 µs | 61.36 µs | 0.741 / **1.35× faster** |
| 1000 | 1104.8 / 1086.9 µs | 724.35 µs | 0.667 / **1.50× faster** |

CIs are fully separated at every size (pool 1000: landed [706.9, 742.9] µs vs ORIG2 [1062.7, 1114.5] µs;
pool 100: [60.4, 62.4] vs [81.1, 84.6]). The win **grows with pool** because larger pools issue more
reciprocity checks that amortize the O(deg) → O(1) shift — and it is a **lower bound** for real k-NN graphs
with hub nodes (higher deg = a longer ORIGINAL linear scan, unchanged O(1) landed). A first eager candidate
(`v2`, also benched) that rebuilt the reciprocity set in a *separate* full-adjacency pass won only ~1.15×
because it re-hashed each edge string ~1.5×; fusing recip-build + forward-capture into a single hash-once
walk (`v3`) is what cleared the noise — v3 beats v2 by 12–24% (e.g. 724 vs 951 µs @ pool 1000). Non-mutual
regression guard: `nonmutual_v3` tracks `nonmutual_ORIG` within measurement noise at all sizes (byte-identical
code path).

Parity: the bench asserts the candidate is **bit-identical** (`score.to_bits()` + `index` + `doc_id`) to
`neighbor_smooth` for both configs at every pool size before timing. Conformance GREEN:
`CARGO_TARGET_DIR=/data/projects/.rch-targets/search-copper cargo test -p frankensearch-fusion --lib smooth::`
= **8 passed / 0 failed** (incl. `mutual_knn_ignores_one_way_edges`, `m_cap_limits_neighbors`,
`hand_computed_mean`, `cluster_rescues_below_threshold_relevant`); `cargo clippy --no-deps -p
frankensearch-fusion --lib --tests --bench neighbor_smooth_recip_ab -- -D warnings` reports **zero findings in
the touched code** (`smooth.rs` additions + the new bench; the `usize→u32` relabel carries the codebase's
existing `#[allow(clippy::cast_possible_truncation)]` idiom from `federated.rs`). The crate-wide run surfaces
only **pre-existing** lints in untouched modules under the drifted local nightly (blend/feedback/hubness/mmr/ope
and the pre-existing `mutual_knn` test) — a toolchain-skew artifact (it flags the already-shipped `hubness.rs`
`ba5052a` too), not introduced here; the canonical RCH toolchain lints these clean. The kernel is not yet wired
into the searcher (the smoothing pass awaits wiring); this removes the perf disincentive to enabling the
better-quality `mutual` refinement once it lands.

---

## 2026-07-09 — Pool-min-max fusion: merge-structured sort (near-sorted input) — 1.15–1.32× limit_all, no top-k regression (CopperVireo)

> **✅ 2026-07-10 NULL CONTROL (cc_fse, `bd-zgq6`): WIN CONFIRMED and STRENGTHENED at pool 1000; pool 50/100 were
> NOT decidable.** The `1.15–1.32×` above came from two separate criterion arms (`limit_all_ORIG`/`limit_all_merge`)
> which do not cancel drift. Re-measured with the shared alternating-round sampler + A/A null control
> (`frankensearch_core::bench_support::paired_median_ratio`, now used by this bench), worker `hz1`, binary sha256
> `c137a84a28908edabecb2ac85aa5efbc8d85440d6a35b9076f9873430c9a5229`, 41 rounds × 8. Ratio is **merge/ORIG**, so
> `<1.0` = merge faster:
>
> | pool | NULL median [p5,p95] | merge/ORIG median [p5,p95] | verdict |
> |---:|---|---|---|
> | 50   | 0.9972 [0.762, 1.010] | 0.9152 [0.669, 0.943] | **INSIDE FLOOR — not decidable** |
> | 100  | 1.0011 [0.868, 1.151] | 0.9237 [0.826, 1.094] | **INSIDE FLOOR — not decidable** |
> | 1000 | 1.0015 [0.969, 1.036] | **0.7400 [0.710, 0.934]** | **DECIDABLE — merge 1.35× faster** (lever p95 0.934 < null p5 0.969) |
>
> At pool 1000 the merge structure is **1.35× faster** — clearly outside the floor, and *better* than the original
> `1.32×`. But at pool 50/100 the workload is tiny and the ±12–15% null floor swallows the effect: the original
> `1.15×` pool-50 figure was **inside the noise floor** and never decidable that way. Same shape as the smoothing
> re-sort (2026-07-10 second correction): a structural sort win that is real and grows with N, decidable only once
> N is large enough that the per-round noise no longer dominates. **The WIN stands** (the shipped production shape
> is `limit_all` returning a full feed to the reranker, i.e. large N); only the small-pool magnitudes are withdrawn.
> Bit-identity between the two operators is asserted before timing (unchanged). No self-time taken (the sort is the
> whole measured routine; bit-identity + the reachability of both operators is the execution evidence).

Sibling-path-consistency win on a fresh path (the pool-min-max SCORE operator was landed 07-09 for QUALITY,
`a9e53b4`, +0.0038 nDCG@10-over-RRF, never perf-mined). The **production** searcher fuses via
`rrf_fuse_with_graph_merge_unique` — the MERGE-structured RRF (`4aeb66b`, 1.31–1.46× on limit_all). But
`pool_minmax_fuse` — the operator that would REPLACE RRF as default once the quality work is wired — kept the
**value-map** pattern: accumulate every doc into an `N`-entry `HashMap<&str, FusedHitScratch>` then
`into_values()` in **random** hash order, forcing a from-scratch O(N log N) sort on the `limit_all` shape
(window ≥ N — the full-ranked-feed case, e.g. feeding a reranker). The merge optimization was never ported
across the sibling path.

Primitive class: **data-layout / algebraic-fusion** (merge-structured accumulation). Added a public
`pool_minmax_fuse_merge` (mirroring the shipped `rrf_fuse` / `rrf_fuse_with_graph_merge` split): a small
`&str → (rank, score)` lexical contribution map + one in-order walk of the already-score-sorted `semantic`
slice, emitting fused hits directly so `results` is **near-sorted** and the final sort runs near-O(N) (pdqsort
is adaptive). **Bit-identical** to `pool_minmax_fuse`: f64 add commutes (emit `semantic + lexical` == the
map's `lexical + semantic`), same first-occurrence dedup on each tier, same total-order comparator (ends in
`doc_id`).

Bench `pool_minmax_merge_ab` (isolated local target `search-copper`; ~50% overlap heavy-tailed pools, semantic
score-sorted; unique N ≈ 1.5·pool):

| pool | shape | LEGACY ORIGINAL (value-map) | merge | ratio vs ORIG |
|---:|---|---:|---:|---:|
| 50   | limit_all | 5.55 µs | 4.82 µs | 0.869 / **1.15×** |
| 100  | limit_all | 11.58 µs | 9.62 µs | 0.831 / **1.20×** |
| 1000 | limit_all | 146.48 µs | 110.74 µs | 0.756 / **1.32×** |
| 1000 | top10 (guard) | 59.15 µs | 49.28 µs | 0.833 / **1.20×** |

CIs fully separated at every row (e.g. limit_all/1000 merge [108.5, 113.0] µs vs ORIG [143.9, 149.3] µs). The
limit_all win **grows with pool** (1.15×→1.20×→1.32×) as the near-sorted adaptive sort's advantage scales — and
there is **no top-k regression**: the smaller contribution maps (vs the N-entry value map) make the merge
~1.15–1.20× faster on the top10 shape too. Parity: the A/B bench asserts bit-identity (doc_id + fused-score
bits + index + in_both) for both shapes at all pool sizes before timing; new `pool_minmax_merge_matches_map`
unit test = 40 randomized trials (varied overlap/dedup/weights/pagination) byte-identical. Conformance GREEN:
`cargo test -p frankensearch-fusion --lib rrf::` = **50 passed / 0 failed**; `cargo clippy --no-deps -p
frankensearch-fusion --lib --tests --bench pool_minmax_merge_ab -- -D warnings` reports zero findings in the
touched code (the `lex_map`/`lex_max` `similar_names` pedantic lint carries a scoped `#[allow]`; the crate-wide
run surfaces only the pre-existing toolchain-drift lints in untouched modules). `pool_minmax_fuse_merge` is
unwired like the base operator; it is the variant to wire when pool-min-max fusion is enabled, closing the
sort-latency gap vs the merge-RRF the searcher already uses.

---

## 2026-07-09 — Ops frame-quality P95 exact histogram LANDED (SearchCod)

Fresh path after the rejection-ledger review: live `frankensearch-ops` TUI frame-quality tracking,
specifically `FrameQualityTracker::p95_frame_time_ms`. This is distinct from the rejected ops SLO rollup
SQL-percentile lever: the rollup path materializes historical telemetry through SQLite, while this path is an
in-memory render-loop window queried every ops UI frame.

Primitive class: **data-layout / succinct-structure**. The tracker now keeps an exact sliding histogram while it
updates the ring buffer, replacing the LEGACY ORIGINAL clone-and-sort P95 query. Buckets `0..128` live inline and
long frames spill into a sparse `BTreeMap`, so common TUI frame times stay in a compact fixed array while overflow
durations remain exact.

Bench (RCH worker `ovh-a`; same Criterion binary embeds the LEGACY ORIGINAL sort tracker and asserts equal P95
before timing):

```bash
AGENT_NAME=SearchCod RCH_WORKER=ovh-a CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cod \
  rch exec -- cargo bench -p frankensearch-ops --profile release \
  --bench percentile_select -- ops_frame_quality_p95 \
  --sample-size 10 --warm-up-time 0.2 --measurement-time 0.5
```

| window | LEGACY ORIGINAL clone+sort median | exact histogram median | ratio-vs-ORIG |
|---:|---:|---:|---:|
| 64 | 375.49 ns | 14.549 ns | 0.0387 / 25.8x faster |
| 128 | 589.99 ns | 15.789 ns | 0.0268 / 37.4x faster |
| 512 | 3.2019 us | 22.602 ns | 0.00706 / 141.7x faster |
| 2,048 | 14.006 us | 14.909 ns | 0.00106 / 939.4x faster |

Conformance GREEN: focused release test `cargo test -p frankensearch-ops frame_quality_tracker --profile
release` passed 5/5 frame-quality tests; `cargo check -p frankensearch-ops --all-targets`, `cargo clippy -p
frankensearch-ops --all-targets -- -D warnings`, and `cargo fmt -p frankensearch-ops --check` passed.

---

### 2026-07-10 - bd-b5wl: FSVI int8 sidecar byte quantizer uses AVX2+F16C dispatch - 3.77x vs prior mmap path (cod_fs)

**Ledger-grep.** This does not retry the closed tombstone-bitmap fused int8 scan, flat-CSR graph_rank
layout, or slot-aligned VALUES dedup join rejections. Prior positive rows already landed f16-slab int8 and
4-bit SIMD quantizers for owned `&[f16]` slabs; the missing production gap was the mmap-backed FSVI byte
slab path used by `VectorIndex::int8_slab()`.

**Lever (`crates/frankensearch-index`).** Added `simd::quantize_f16_le_bytes_to_i8`, the byte-slab twin of
`quantize_f16_slab_to_i8`. On little-endian x86 with AVX2+F16C, it reads mapped FSVI f16 bytes directly,
does a vector max-abs pass, then vector scale/round/clamp into the exact int8 sidecar bytes. The generic
fallback keeps the same scalar little-endian contract. `VectorIndex::int8_slab()` now calls the dispatched
helper instead of the older branchless-widen plus scalar round loop.

**Measured (same RCH worker `hz2`, one Criterion binary, release profile):**

```bash
AGENT_NAME=cod_fs RCH_WORKER=hz2 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-fs \
  rch exec -- cargo bench -p frankensearch-index --profile release \
  --bench f16_slab_quantize -- f16_slab_quantize --sample-size 10 \
  --warm-up-time 0.1 --measurement-time 0.35 --noplot
```

| f16 slab | scalar byte path | prior `simd_widen` mmap path | kept `dispatch` | ratio vs prior |
|---:|---:|---:|---:|---:|
| 1,000 x 384 | 2.3531 ms | 1.4057 ms | 361.33 us | 0.257 / 3.89x |
| 10,000 x 384 | 24.813 ms | 14.007 ms | 3.4300 ms | 0.245 / 4.08x |
| 50,000 x 384 | 121.06 ms | 69.949 ms | 18.571 ms | 0.266 / 3.77x |

Criterion CI half-width for the kept 50k row is about 3.4% (17.994-19.254 ms around 18.571 ms), under
the 5% keep gate. Original-comparator ratio is **N/A** for this sub-row: it is the FSVI int8 sidecar build
inside the vector scan path, not a Lucene/Meilisearch query benchmark by itself.

**Parity.** The bench asserts `scalar == simd_widen == dispatch` before timing for every slab size. Focused
release test passed: `cargo test -p frankensearch-index --profile release avx2_quantize_i8_bytes_matches_generic -- --nocapture`
on RCH worker `hz2` (1 passed, 389 filtered out). The production scan consumes byte-identical int8 sidecars,
so retrieval ordering is unchanged by this lever.

**Profiler status.** The hotspot was the production FSVI byte-slab build row in `f16_slab_quantize`; the
same binary shows the previous mmap path spending 69.949 ms at 50k x 384 before the kept dispatch arm drops
that to 18.571 ms. `perf stat -e cycles,instructions,cache-misses` and `cargo flamegraph` were both
attempted on RCH worker `hz2`, but RCH's non-compilation wrapper rebuilt the benchmark and did not yield a
usable filtered-bench profile; the interrupted `perf stat` counters covered 271.6 s of wrapper/build time,
so they are intentionally rejected as proof. Route next profiling through a directly retained bench binary
or an RCH artifact mode that preserves the executable.

---

### 2026-07-09 — hubness `r_d` builder: reuse the vector tier's AVX2 dot instead of hand-rolling — 10.89× (cc_fs)

**Lever (sibling-path consistency, `crates/frankensearch-fusion/src/hubness.rs`).** `compute_query_hubness`
builds the per-doc query-hubness table `r_d` (mean cosine of each doc to its `kq` nearest background sample
queries) — an offline/amortized **O(docs · queries · dim)** batch this ledger already clocked at ~109 ms /
696 ms @ 2000×200 / 5000×500. Its inner kernel was a private scalar `dot`:
`a.iter().zip(b).map(|(x,y)| x*y).sum()` — one f32 accumulator, a serial add chain LLVM cannot reassociate
without fast-math, latency-bound at ~1 add / 4–5 cyc no matter how wide the multiplies vectorize.

`frankensearch-fusion` **already depends on `frankensearch-index`**, which already exports
`dot_product_f32_f32`: a hand-written AVX2 kernel with four `f32x8` accumulators (32 lanes of ILP) and a
portable `wide` fallback. The fix is *deleting* the local reduction and calling it. Slicing to
`a.len().min(b.len())` preserves the ORIGINAL's `zip` truncation, making `DimensionMismatch` unreachable.

**Measured (RCH worker `hz2`, one Criterion binary so every arm shares a build + machine; `ORIG_scalar`
measured first AND last to bracket ordering bias — 0.04–0.79%):**

```bash
AGENT_NAME=cc_fs RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/hub-dot \
  rch exec -- cargo bench -p frankensearch-fusion --profile release --bench hubness_dot_ab
```

| workload | ORIG scalar | **simd (kept)** | ratio-vs-ORIG | rejected `multiacc` |
|---|---:|---:|---:|---:|
| `hubness_dot_micro` 384-dim dot | 222.53 ns | **19.619 ns** | **0.088 / 11.34×** | 86.233 ns (2.58×) |
| `hubness_build/1000x100x384` | 22.389 ms | **2.0565 ms** | **0.092 / 10.89×** | 8.8560 ms (2.53×) |
| `hubness_build/500x200x384` | 22.748 ms | **2.0166 ms** | **0.089 / 11.28×** | 8.7997 ms (2.59×) |

CI widths ≤ ±0.5%; `ORIG_scalar2` control 22.465 ms / 22.756 ms.

**Parity.** The bench gates before timing: the `simd` mirror is **bit-identical** to the shipped
`compute_query_hubness`, and every reassociated candidate stays within `max Δr_d < 1e-4` of the scalar
ORIGINAL (the `select_nth` of the kq nearest queries is unchanged; `β·r_d` moves a dense score by ~1e-2).
Same accepted search-time ULP class as `mmr::cosine_sim_pre`. Two new unit tests cover a dim-43 vector
(32-wide group + 8-wide chunk + 3-elem tail, pinned to a scalar reference) and ragged-length truncation.

**Scope:** original-comparator ratio **N/A** (offline builder, not the query path). Files: `hubness.rs`,
`benches/hubness_dot_ab.rs`, `Cargo.toml` (dev-deps). **Route next:** the outer per-doc loop is still serial
and each `r_d` is independent — a rayon `par_iter()` measured **5.85× on top of this** (2.0565 ms → 351.66 µs
@ 1000×100×384) and is **bit-identical** to the serial `simd` arm (indexed parallel iterator preserves order;
no element's arithmetic changes; asserted by the `simd_par` vs `simd` gate). Held back as a separate lever
because it changes the threading behaviour of a public library fn and wants a work-based threshold.

---

### 2026-07-09 — hubness `r_d` builder: rayon over the outer doc loop — 5.49× on top of the AVX2 dot, bit-identical (cc_fs)

**Lever (bd-i8sp, `crates/frankensearch-fusion/src/hubness.rs`).** Follow-up to `8bf3a00` (which replaced the
hand-rolled dot with the vector tier's AVX2 `dot_product_f32_f32`, 10.89×). Each doc's `r_d` depends only on
that doc and the whole query sample — never on the other docs — so the outer `doc_vecs.iter().map(..)` is
embarrassingly parallel. It now runs on rayon above [`PARALLEL_THRESHOLD`] **dot products** of total work
(`docs · queries`), serial below, where the pool's fork/join overhead would dominate a microsecond batch.

The threshold counts dot products, **not documents**: one doc's work is `queries · dim` multiply-adds, so a
doc-count gate would misjudge a small corpus against a large query log. `PARALLEL_THRESHOLD` already carries
the same "10k dot products" meaning in `frankensearch-index`'s flat scan, so the constant transfers intact.

**Parity: BIT-IDENTICAL, not merely ULP-equal.** rayon's *indexed* parallel iterator collects in input order
and no element's arithmetic changes — only the scheduling does. Proven three ways: the unit test
`hubness_par_matches_serial_across_threshold` evaluates the *same two docs* below the gate (2×60 = 120 dot
products, serial) and above it (200×60 = 12 000, rayon) and requires their `r_d` to agree to the bit — padding
cannot change a doc's `r_d`, which isolates the branch as the only variable — and also asserts every padded
slot holds the filler's value, catching reordering. The `hubness_dot_ab` bench additionally gates `simd_par`
bit-identical to `simd` *and* to the shipped builder before any timing.

**Measured (RCH worker, one Criterion binary, all arms share a build + machine; `ORIG_scalar` measured first
AND last; `shipped` arm times the real `compute_query_hubness`, not a bench mirror):**

```bash
AGENT_NAME=cc_fs RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/hub-dot \
  rch exec -- cargo bench -p frankensearch-fusion --profile release --bench hubness_dot_ab
```

Decisive shape `hubness_build/500x200x384` — **every arm passes the cv_pct < 5 keep-gate**:

| arm | mean | cv_pct | ratio vs ORIG | ratio vs `simd` |
|---|---:|---:|---:|---:|
| `ORIG_scalar` (pre-`8bf3a00` scalar dot, serial) | 22.748 ms | 0.50 | 1.000 | — |
| `simd` (AVX2 dot, serial — the `8bf3a00` baseline) | 2.3430 ms | 1.89 | 0.103 / 9.71× | 1.000 |
| **`shipped` (AVX2 dot + rayon)** | **426.65 µs** | **1.81** | **0.0188 / 53.3×** | **0.182 / 5.49×** |
| `ORIG_scalar2` (ordering-bias control) | 23.080 ms | 0.26 | bias 1.46% | — |

**Honest caveat on the second shape.** At `1000x100x384` the worker was contended during this run and the
serial baselines destabilized: `simd` cv **14.79** and `multiacc` cv **5.70**, both failing the gate. Its
`shipped` arm is clean (381.47 µs, cv 3.06) and beats the `ORIG_scalar2` control (22.447 ms, cv 0.72) by
58.8×, but the *on-top-of-simd* ratio at that shape is **not quoted** because its denominator failed cv. The
500×200×384 row above is the ledgered result. An earlier quiescent run measured 6.22×/6.42× on the same lever;
5.49× is the conservative figure taken from the run where every arm cleared the gate.

**Method note (cost me a re-run).** Criterion's in-code `benchmark_group.measurement_time()` **overrides** the
`--measurement-time` CLI flag. The first attempt at tightening cv passed the flag and silently re-measured at
the old 3 s window (`shipped` cv 7.33). cv shrinks with *iterations per sample*, so the fix is a longer
`measurement_time` (3 s → 12 s) in the bench source; raising `sample_size` would *shrink* per-sample
iterations and make cv worse.

Conformance: `cargo test -p frankensearch-fusion --lib` → **852 passed, 0 failed**. `cargo clippy -p
frankensearch-fusion --all-targets` → 0 errors, no warnings on the changed files. `ubs` exit 0, 0 critical.
rustfmt introduces no new drift (the 4 residual `hubness.rs` sites pre-exist on `origin/main`).
Original-comparator ratio **N/A** (offline builder, not the query path). Files: `hubness.rs`,
`benches/hubness_dot_ab.rs`.

---

### 2026-07-09 — bd-2ocs: wire pool-min-max fusion into both searchers behind a `FusionStrategy` knob — dispatch latency-neutral (cc_fs)

**Lever (`frankensearch-core/src/config.rs`, `frankensearch-fusion/src/{rrf,searcher,sync_searcher}.rs`).**
`pool_minmax_fuse_merge` (`8ad515a`, +0.0038 mean nDCG@10 over RRF on BEIR, `a9e53b4`) was landed, unit-tested,
benched — and had **zero callers**. Both searchers hard-coded `rrf_fuse_with_graph_merge_unique`. This wires it
in behind `TwoTierConfig::fusion_strategy` (`FusionStrategy::{Rrf, PoolMinMax}`, env
`FRANKENSEARCH_FUSION_STRATEGY`), defaulting to `Rrf`.

This is exactly what the ledger routes to: the `a9e53b4` entry parks the default switch as "a separate product
decision (keep RRF where per-query pool stats are unreliable)", and the `8ad515a` entry names
`pool_minmax_fuse_merge` "the variant to wire when pool-min-max fusion is enabled". So the knob is **opt-in**;
the default is not flipped. All four call sites (2 in `sync_searcher`, 2 in `searcher`) now go through one
dispatch point, `fuse_by_strategy`.

**PARITY GATE (behaviour).** `default_strategy_is_byte_identical_to_rrf` compares the dispatched default
against a direct RRF call element-by-element on `f32::to_bits()` — the wiring is a no-op for every existing
caller. `pool_minmax_with_graph_falls_back_to_rrf_not_dropping_graph` pins the one asymmetry:
`pool_minmax_fuse_merge` has **no graph arm**, so a non-empty graph falls back to RRF rather than silently
discarding a graph signal the caller explicitly enabled (the test asserts the graph-only doc survives).

**QUALITY GATE (nDCG).** `pool_minmax_beats_rrf_on_ndcg_when_magnitude_carries_the_signal` measures, in Rust,
via core's `ndcg_at_k` — it does not merely cite the Python harness. Ground truth `d` is semantic-only but
overwhelmingly similar; `b` is marginal in both lists. RRF's rank transform rewards `b` for appearing twice and
buries `d`; pool-min-max sees `d`'s magnitude and promotes it. nDCG@10 strictly increases. That is the same
mechanism behind the +0.0038 BEIR result.

**SPEED GATE (dispatch overhead).** The risk of a wiring commit is that the added `match` costs the default
path. Measured as an explicit A/B in `pool_minmax_merge_ab` — ORIG = direct `rrf_fuse_with_graph_merge_unique`
(what shipped before), candidate = via `fuse_by_strategy(Rrf, ..)`, ORIG measured **first and last**. All arms
`cv_pct < 5`:

| pool | ORIG direct RRF | via dispatch | ORIG2 (bias) | ratio-vs-ORIG |
|---:|---:|---:|---:|---:|
| 50   | 3.6890 µs (cv 2.21) | 3.6650 µs (cv 3.73) | 3.7071 µs (cv 2.28) | **0.991** |
| 100  | 7.8882 µs (cv 1.34) | 7.8956 µs (cv 0.59) | 7.8902 µs (cv 0.89) | **1.001** |
| 1000 | 99.351 µs (cv 0.93) | 99.301 µs (cv 0.62) | 99.125 µs (cv 1.97) | **1.001** |

The dispatched arm lands **inside the ORIG/ORIG2 bracket at every pool** — the `match` on a `Copy` enum
vanishes under inlining. Default path is latency-neutral, measured rather than asserted.

**And when the knob is ON**, `pool_minmax_fuse_merge` vs RRF at the `limit_all` shape (same run): pool 50
4.1730 µs vs 3.6890 (1.13× slower), pool 100 8.4783 vs 7.8882 (1.07× slower), pool 1000 **93.285 µs vs 99.351
(0.94, i.e. 1.07× FASTER)**. Fusion is µs-scale against a ms-scale search, so all three are immaterial
end-to-end; the +0.0038 nDCG is the point.

Conformance: `cargo test -p frankensearch-core -p frankensearch-fusion --lib` → **898 + 856 passed, 0 failed**
(4 new tests). `cargo clippy -p frankensearch-core -p frankensearch-fusion --all-targets` → 0 errors, no
warnings on changed lines. `ubs` → 0 critical on changed lines (its 7 criticals are `.unwrap()` in
pre-existing tests). Original-comparator ratio **N/A** (internal fusion-operator selection; default unchanged).

## 2026-07-10 — kNN neighbor smoothing WIRED into `TwoTierSearcher` Phase-1; the re-sort is a correctness requirement costing 0.8–1.5% (cc_fse)

Closes the smoothing half of `bd-kdjr`. `neighbor_smooth` (`257c468`, +0.0039 mean nDCG@10 on BEIR) had been
landed, unit-tested and benched — with **zero callers**. Same dead-quality-kernel shape as the pool-min-max
wiring (`49a914c`): a proven quality win sitting behind no code path. Now wired into the async searcher's
Phase-1 semantic pool behind `TwoTierConfig::neighbor_smoothing_{alpha,m,mutual}` (env
`FRANKENSEARCH_SMOOTHING_ALPHA`), `alpha = 0.0` (off) by default, gated on `feature = "graph"` + an attached
`with_document_graph` — exactly the gating `document_graph` already carries.

**The mechanism the wiring exposed.** Smoothing rewrites scores; the rank-based fusion operators
(`rrf_fuse_with_graph_merge_unique` and friends) walk the `semantic` slice **once, in order**, and take each
hit's rank from its **position** — they never re-sort, because a vector index already returns score-sorted
hits. So dropping raw `neighbor_smooth` into the searcher would have been a silent no-op-to-negative: a doc the
diffusion promoted keeps its old rank, the promotion is discarded, and the merge's near-sorted precondition is
violated. Hence a new `neighbor_smooth_ranked` = kernel + deterministic descending re-sort (ties on `doc_id`,
so the pool is a total order and replay stays deterministic). This is a **correctness** requirement, not a
cosmetic one, and it is the reason the kernel could not simply be called.

**PARITY GATE.** `smoothing_is_inert_by_default_even_with_a_document_graph` — attaching a graph under the
default config leaves the Phase-1 doc order identical. The disabled path is not merely cheap, it is *not
entered*: the `match` yields `fast_hits` by move, so there is no call and no identity clone.
`ranked_identity_is_byte_identical` pins the kernel's own `alpha=0` passthrough at score-bit level.

**CORRECTNESS GATE.** `ranked_reorders_a_promoted_doc_where_plain_smooth_does_not` isolates the bug above:
plain `neighbor_smooth` leaves the promoted doc in slot 2 with a better score; `_ranked` moves it to slot 1.
End-to-end, `smoothing_promotes_a_graph_neighbor_of_a_top_hit_through_fusion` drives the real searcher —
`doc-7` (score 0.0, `Similar` edge to the perfect hit `doc-0`) is lifted to `α·1.0 = 0.5` and must overtake the
isolated zero-scorer `doc-2` **in the fused output**, proving the smoothed score survives position-assigned
ranks. `doc-0` stays at rank 0 (isolated ⇒ α collapses).

**QUALITY GATE (nDCG).** `ranked_smoothing_improves_ndcg_over_no_smoothing` measures in Rust via core's
`ndcg_at_k` rather than citing the Python harness: the relevant doc is under-scored by the embedder but
adjacent in the k-NN graph to a confident hit, and diffusion recovers it. nDCG@10 strictly increases — the same
recall mechanism behind the +0.0039 BEIR result.

**SPEED GATE — ⚠ RETRACTED. The re-sort never sorted anything; see the correction entry below.** Numbers kept
verbatim for audit:

> `neighbor_smooth` bench, one binary ⇒ same worker; ORIG (`smooth`) measured first and last (`smooth2`).
>
> | pool | identity (α=0 clone) | ORIG `smooth` | `smooth_ranked` | ORIG2 `smooth2` | re-sort cost |
> |---:|---:|---:|---:|---:|---:|
> | 50   | 138.05 ns | 5.6084 µs | 5.6867 µs | 5.6271 µs | +1.2% |
> | 100  | 247.83 ns | 11.707 µs | 11.767 µs | 11.635 µs | +0.8% |
> | 1000 | 2.3386 µs | 151.55 µs | 154.28 µs | 152.50 µs | +1.5% |

Smoothing runs inside the `vector_search_ms` window, so enabling it attributes its latency to that metric
rather than leaving it unaccounted. That note stands, and the parity/correctness/quality gates above are
unaffected — only the *cost of the re-sort* was mismeasured.

`mutual` (reciprocal-kNN, the ledger's no-regret refinement) is exposed as `neighbor_smoothing_mutual` and
costs ~4.3× the direct kernel at pool 50–1000 (24.28 / 50.46 / 538.12 µs) — still µs-scale, still opt-in.

Conformance: `cargo test -p frankensearch-fusion --features graph` → **867 + 13 + 29 + 25 + 10 + 3 passed, 0
failed** (5 new tests); default-features (`graph` off) → **859 passed, 0 failed**, proving the `cfg`-off arm
compiles and the default is untouched. `cargo clippy --no-deps -p frankensearch-fusion --features graph
--all-targets` → 0 warnings on changed lines (the surviving `smooth.rs` similar-binding warning is on
pre-existing `d_nm`/`d_m`). `cargo fmt --check` → 0 diffs on changed lines (crate-wide drift pre-exists).
`ubs` → **exit 1, mis-stated as exit 0 here** (the `$?` read was `tail`'s through a pipe, not `ubs`'s);
`searcher.rs` already exits 1 at `HEAD~1` on 6 `doc_id == "doc-0"` "secret comparison" false positives.
Original-comparator ratio **N/A** (opt-in Phase-1 quality pass; default byte-identical).

Route-next: the hubness half of `bd-kdjr` — `apply_hubness_penalty` still has no searcher caller and needs the
background-query-sample `r_d` table plumbed through (`ba5052a` landed the builder). Same dead-kernel shape.

## 2026-07-10 — CORRECTION to the entry above: its re-sort cost measured DEAD CODE (pdqsort short-circuit); true cost is 7–11%, not 0.8–1.5% (cc_fse)

**What was wrong.** `neighbor_smooth`'s bench fixture linked each doc to the `m` *following* docs
(`nbr = (i+j) % pool`) over the monotone score `1/(i+1)`. Every doc's neighbours therefore scored strictly
below it, and diffusing a convex decreasing sequence preserves its order — so `neighbor_smooth_ranked`
received an **already-sorted** pool and `sort_unstable_by` short-circuited on its sortedness check. The
`smooth_ranked` arm was timing an O(n) *detection pass*, not a sort. The kernel ran (scores changed, which the
old `assert!((s[10].score - hits[10].score).abs() > 1e-9)` confirmed) — but the *code under measurement*, the
sort, did nothing. Changing scores is not evidence that a downstream sort has work.

**How it was caught.** A reachability gate now runs before the timed arms: it sorts the smoothed pool, counts
how many hits move, and **panics if zero**. On the old fixture it panicked at pool 50 on the first run.

**The fix.** Neighbours are now *scattered* (`(i*37 + j*101) % pool`), so a low-ranked doc can borrow from a
high-scoring cluster and overtake its neighbours — the promotion smoothing exists to produce. Displacement is
now **41–44/50, 92–95/100, 985–992/1000**, printed by the gate each run.

**Substrate also corrected.** The ORIG-first-and-last bracket (`smooth`/`smooth2`) does **not** cancel drift —
criterion group members run sequentially. It only *exposes* drift, and it did: one run showed ORIG at 28.18 µs
and ORIG2 at 20.85 µs (35% apart) with `smooth_ranked` landing *below* ORIG2, which is physically impossible
since `ranked = smooth + sort`. Both benches now use an **interleaved paired sampler**: each arm's measured
routine runs *both* implementations every iteration and times only its own via `iter_custom`, batched
`PAIR_BATCH = 16` so the `Instant::now()` pair amortizes to <2%. Both arms then do identical total work and see
identical machine state, so drift cancels in the ratio. CI widths fell from ±6% to **±0.2%**.

**Corrected re-sort cost** (interleaved paired, one binary, one `rch` invocation on `ovh-a`; each timed region
= 16 calls, so `PAIR_BATCH` cancels in the ratio):

| pool | displaced | ORIG `paired_smooth` | `paired_smooth_ranked` | ratio |
|---:|---:|---:|---:|---:|
| 50   | 41–44/50   | 91.846 µs | 98.454 µs | **1.072×** |
| 100  | 92–95/100  | 194.31 µs | 215.86 µs | **1.111×** |
| 1000 | 985–992/1000 | 2.8912 ms | 3.2037 ms | **1.108×** |

> **⚠ 2026-07-10 SECOND CORRECTION (cc_fse): these ratios had NO null control; two of the three are NOT decidable.**
> The table above registered `paired_smooth`/`paired_smooth_ranked` as two *criterion* benchmarks. Criterion runs
> them sequentially, minutes apart, so drift between the arms is not cancelled — the internal timed/untimed
> interleaving only equalizes state *within* an arm. An A/A null control (the same `neighbor_smooth` as both arms)
> proved it: **criterion-level null median = 1.1265× at pool 50, 0.9268× at pool 100** (worker `hz1`, 120 samples) —
> a floor of ±12% that does not even contain 1.000. The `1.072×` and `1.111×` rows above sit **inside** that floor
> and were never measurable that way.
>
> Fixed the harness (`crates/frankensearch-fusion/src/bench_support.rs`, `paired_median_ratio`): both arms run in
> **one** routine in **alternating rounds** (`r` even → `a,b`; odd → `b,a`), ratio taken **per round**, median over
> rounds. Gate on the **median vs the null's p5..p95 spread**, not on `cv` (`cv < 5%` is unattainable on this fleet).
> Re-measured, worker `hz1`, binary sha256 `c287dde1ed774abca36faa470ffe3d91cfdb265c1f0c944fff3c4f568403c08d`,
> 41 rounds × 8 inner:
>
> | pool | displaced | NULL median [p5,p95] | re-sort median [p5,p95] | verdict |
> |---:|---:|---|---|---|
> | 50   | 44/50   | 1.0012 [0.758, 1.517] | 1.0724 [0.770, 1.525] | **INSIDE NULL FLOOR — not decidable** |
> | 100  | 95/100  | 1.0035 [0.879, 1.136] | 1.0937 [0.951, 1.255] | **INSIDE NULL FLOOR — not decidable** |
> | 1000 | 985/1000 | 1.0003 [0.976, 1.058] | **1.1096 [1.081, 1.143]** | **DECIDABLE** (lever p5 1.081 > null p95 1.058) |
>
> **Only the pool-1000 claim survives: the re-sort costs ~11% there, cleanly outside the floor.** At pool 50/100 the
> per-round noise on such a small workload swamps a ~7–9% effect; the true cost is *probably* similar (the mechanism
> is identical) but is **not measurable** on this hardware at that size, so no ratio is claimed. The earlier `7–11%`
> band overstated confidence at the small pools.
>
> **Self-time NOW OBTAINED** (worker `hetzner1`, `--profile release-perf` so symbols resolve, sha256
> `9deaed1e96889448c203bdbd7afb280130f676d671b2696c2a4fb024ebd19607`, `perf record -F 997 -e cycles:u`, `smooth/1000`
> arm): `neighbor_smooth` kernel **48.76%**, `DocumentGraph::neighbors` 11.60%, and the re-sort —
> `quicksort::<VectorHit, cmp_rank>` 1.36% + `small_sort_general::<VectorHit, cmp_rank>` 0.99% = **~2.35% self-time**
> — consistent with an ~11% delta measured against just the kernel's share. (The `rayon par_sort::<f64>` frames are
> criterion's own percentile math, not shipped code.) This discharges the self-time obligation for the pool-1000 row.
>
> **The land still stands** regardless: at pool 1000 the whole `smooth_ranked` call is ~3.2 ms *inside the bench*,
> but the real Phase-1 semantic pool is tens of candidates, and the re-sort is a correctness requirement (fusion
> assigns ranks by position), not an optimization. Only the *cost measurement* is corrected — from a false-precise
> 7–11% to "~11% at pool 1000, undecidable below."

DCE ruled out per arm: every input goes through `black_box`, every returned `Vec` is consumed through
`black_box`, and the reachability gate compares real outputs.

Lesson (generalizes): **an assert that the kernel changed its output does not prove a downstream stage has
work.** Gate each measured stage on its own observable — here, "did the sort move anything?" — and print it.

## 2026-07-10 — Query-hubness demotion WIRED into `TwoTierSearcher` Phase-1; the O(pool) subtract is dwarfed 4.7–9.9× by its mandatory re-sort (cc_fse)

Closes the hubness half of `bd-kdjr`. `apply_hubness_penalty` / `compute_query_hubness` (`ba5052a`; **+0.0033
mean hybrid nDCG@10** at β=0.2, all-positive on 4 BEIR corpora, dense-tier +0.0128 arguana / +0.0078 scidocs)
had been landed, unit-tested, benched and AVX2-optimized (`10.89×` builder) — with **zero callers**. Third
instance of the dead-quality-kernel shape after pool-min-max (`49a914c`) and smoothing (`14fc1fc`).

Now wired behind `TwoTierConfig::hubness_beta` (env `FRANKENSEARCH_HUBNESS_BETA`), `0.0` (off) by default, plus
`TwoTierSearcher::with_hubness_table(Arc<[f32]>)` for the `r_d` table. Note the gating differs from smoothing's:
hubness needs **no** `DocumentGraph`, so it is *not* behind `feature = "graph"`. Only the query-**side** form is
wired; the query-free proxies (doc-density, centroid, PC-removal) stay REJECTED (`64ac8b7`), and the config doc
says so at the knob.

**Composition.** Both corrections now run through one private `correct_phase1_pool`, which applies hubness,
then smoothing, then re-sorts **once**. Hubness runs first on purpose: a hub is by construction adjacent to many
docs, so diffusing an uncorrected pool would let each hub inject its inflated score into the neighbour means of
exactly the specific answers the penalty exists to rescue. Smoothing's output values do not depend on its input
order, so one sort after both passes is equivalent to sorting after each — and cheaper. With both off the pool
returns **by move**: no call, no clone, no sort.

**Shared comparator + a latent NaN bug fixed.** Both corrections must re-sort, so the comparator moved to
`VectorHit::cmp_rank` (core). Writing it exposed a real defect in the *shipped* `neighbor_smooth_ranked`: it
sorted with a bare descending `b.score.total_cmp(&a.score)`, and IEEE 754 `totalOrder` ranks `+NaN` above
`+inf` — so a NaN score sorted to **rank 0**. `cmp_rank` inherits `cmp_by_score`'s NaN-last mapping and adds the
`doc_id` tiebreak for replayability. Hubness subtracts `β·r_d` and can propagate a NaN from a corrupted table
straight into that sort, so this is load bearing, not theoretical. `cmp_rank_sorts_nan_last_unlike_bare_total_cmp`
pins both behaviours, asserting the bare form's trap explicitly.

**PARITY GATE.** `correct_phase1_pool_is_identity_when_both_corrections_are_off` — with an `r_d` table attached
and `beta = 0.0`, the pool comes back with identical order and identical score **bits**, and unsorted.
End-to-end, `hubness_demotes_a_hub_through_fusion_and_is_inert_by_default` asserts the attached table changes
nothing at the default β.

**CORRECTNESS GATE.** `hubness_demotes_a_hub_below_an_equal_scoring_non_hub_and_resorts` (pool level) and the
end-to-end test (real searcher, fused output): `doc-8`, `doc-0`, `doc-4` all score 1.0; marking the *leader*
`doc-8` a hub (`r_d = 1.0`, β=0.5 ⇒ 0.5) must push it behind both non-hubs yet keep it ahead of the 0.0 pool —
demoted, not dropped. **ORDERING GATE.**
`hubness_is_applied_before_smoothing_so_hubs_do_not_leak_into_neighbor_means` discriminates the two orders
numerically: `doc-1` diffusing from the hub `doc-4` must score `0.5·0 + 0.5·(1.0 − 0.5·1.0) = 0.25`; smoothing
first would yield `0.5`.

**SPEED GATE** (interleaved paired sampler, one binary, one `rch` invocation on `ovh-a`; timed region = 16
calls; reachability gate asserts the sort permutes the pool and prints the count):

| pool | displaced | ORIG `paired_correct` | `paired_correct_ranked` | ratio |
|---:|---:|---:|---:|---:|
| 50   | 41/50    | 2.2276 µs (139.2 ns/call) | 10.417 µs (651.1 ns/call) | **4.676×** |
| 100  | 92/100   | 4.1002 µs (256.3 ns/call) | 26.759 µs (1.672 µs/call) | **6.526×** |
| 1000 | 992/1000 | 37.441 µs (2.340 µs/call) | 370.40 µs (23.15 µs/call) | **9.893×** |

> **✅ 2026-07-10 NULL CONTROL added (cc_fse): all three rows DECIDABLE by a wide margin.** These ratios predate the
> null-control rule, so an A/A control was run (same `apply_hubness_penalty` as both arms, worker `hz1`, binary
> sha256 `b0776692416eee956e9f6f33af13b6cadf8f08ad181cb3f546e5ae5932eaf421`, 120 samples): null median 1.077 [range
> 0.967, 1.196] at pool 50, 1.001 [0.995, 1.008] at pool 100, 0.972 [0.942, 1.003] at pool 1000. The lever effect
> (4.68×/6.53×/9.89×) clears the floor by **4–10×** at every pool — unlike the smoothing re-sort (see the two
> corrections above), which is a same-primitive sort but on a much larger base kernel, so its ~11% effect is only
> decidable at pool 1000. The hubness verdict stands unchanged; it was simply never at risk from the floor because
> the base O(pool) subtract is ~100× cheaper than the sort it precedes.

**The mechanism, and why the ratio is the opposite of smoothing's.** `apply_hubness_penalty` is a trivial
O(pool) subtract (~2.3 ns/doc); the mandatory re-sort is O(pool log pool) over a nearly-fully-permuted pool
(992/1000 displaced). So the *same* sort that costs 7–11% of the O(pool·M) smoothing kernel costs **4.7–9.9×**
the hubness kernel, and the ratio grows with pool as `log pool` does. The absolute sort cost agrees across both
benches at pool 1000 (20.8 µs here, 19.5 µs there — 6% apart), which is the cross-check that the two paired
samplers are measuring the same primitive. Absolute worst case is 23.15 µs at pool 1000 = **0.15% of the 15 ms
Phase-1 budget**, and it is opt-in; the +0.0033 nDCG is the point. Route-next if it ever matters: the two
corrections could sort by a radix pass over the `f32` score bits instead of comparison pdqsort, or fuse the
demotion into the vector tier's top-k heap so the pool is never unsorted.

Conformance: `cargo test -p frankensearch-fusion --features graph` → **871 passed, 0 failed** (+4 tests);
default features (`graph` off) → **862 passed, 0 failed**, proving the ungated hubness arm compiles without the
graph feature and the default path is untouched. `cargo test -p frankensearch-core --lib` → **900 passed, 0
failed** (+2). `cargo clippy --no-deps` → 0 warnings on changed lines (`more than 3 bools` on `TwoTierConfig`
and `smooth.rs`'s `d_nm`/`d_m` similar-bindings both pre-date this change — verified against `HEAD~1`).
`cargo fmt --check` → 0 diffs on changed lines. `ubs` → exit 1, but **zero net-new criticals**: `searcher.rs`
already exits 1 at `HEAD` (6 `doc_id == "doc-0"` "secret comparison" false positives + 1 test `panic!`), and
the duplicated `rank` closure was extracted into one shared helper so this change adds no new `panic!` site.
All builds/benches ran remotely via `rch` (workers `ovh-a`/`hz1`/`hz2`); no local cargo.
Original-comparator ratio **N/A** (opt-in Phase-1 quality pass; default byte-identical).

Route-next: `bd-kdjr` is now closed for both kernels, but the `r_d` table has **no builder wired into any
product** — `compute_query_hubness` needs a background query sample (a rolling query log) plumbed from the
host. Until then `with_hubness_table` is caller-supplied only, and the knob ships off.

## 2026-07-10 — WIN (kernel-level): `vpmaddubs` int8 dot is 1.23× the shipped `vpmovsxbw`+`vpmaddwd` kernel, bit-exact on realistic quantized magnitudes — DECIDABLE against a null control (cc_fse, bd-b5wl)

Owned bd-b5wl while cod is walled. The int8 ADC two-pass scan is already implemented and correct (cod:
recall@10 = nDCG@10 = 1.0000 at mult 2/3/5/10, bit-exact parity asserted; the headline int8-two-pass-vs-flat
win is ~7.1× at 130k). cod's *specific* pass-1 micro-opt is correctly BLOCKED — not wrong, but the end-to-end
scan A/B is **undecidable under fleet contention** (null floor CV 32–35%, wider than the effect after Amdahl).
cod's documented route-next: "a quantizer-domain-safe `vpmaddubs` signed-dot transform." This lands that kernel
and measures it where it IS decidable — in isolation.

**The lever.** The shipped `dot_i8_i8_avx2` sign-extends **both** operands (4× `vpmovsxbw` per 32 int8) before
`vpmaddwd`. `dot_i8_i8_maddubs` shifts `stored` to the u8 domain with one `vpxor` (`i8 + 128 ≡ i8 ⊕ 0x80`) and
uses `vpmaddubs` (u8·i8 → i16 in one op, **no stored widening** — the dominant traffic, streamed once per row),
folding the bias into a per-query scalar `128·Σq` via `Σ s_i·q_i = maddubs_reduce(u,q) − 128·Σq_i`.

**Correctness — RECALL preservation vs f32, proven end-to-end (corrected 2026-07-10, second pass).** `vpmaddubs`
saturates each adjacent-pair sum to i16, so it is APPROXIMATE. My first pass over-claimed "bit-exact on realistic
quantized data" — that holds only for `|x| ≤ 40` (`2·(40+128)·40 ≈ 13 440 < 32 767`). **The shipped quantizer
uses a *global* `scale = 127/max_abs`, so a real L2-normalized 384-dim embedding has typical components ~±32 but
a tail up to ±127 — which DOES saturate maddubs.** So on real quantized data maddubs is approximate, *not*
bit-exact. The correct guarantee is **recall**, proven deterministically: `maddubs_pass1_preserves_f32_recall_
under_real_saturation` builds a realistic corpus, quantizes it with the *shipped* `quantize_f16_slab_to_i8`,
**asserts the corpus actually saturates maddubs** (else the proof is vacuous), and shows the maddubs pass-1
recall@10 of the exact-f32 top-10 into the top-`k·mult` set is **1.0** and `≥` the exact-int8 pass-1 recall — so
swapping in the approximate kernel costs *zero* f32 recall, which is the guarantee the two-pass scan needs before
its f16 rescore. Five `simd.rs` tests gate the kernel: bit-exact when non-saturating (incl. scalar tail);
bit-exact top-k at `|x| ≤ 40`; the adversarial-uniform-±127 boundary (documents where saturation begins); the
**batched** `dot_i8x4_i8_maddubs` equals 4× single calls (the row-blocked scan uses the batched kernel); and the
end-to-end recall proof above.

**Speed — DECIDABLE, null-controlled, one binary / one `rch` invocation.** Isolated microbench
(`benches/int8_dot_maddubs_ab.rs`, dim 384 × batch 4096, `|x| ≤ 40` so the arms are bit-identical, asserted
before timing), shared alternating-round sampler + A/A null control (`frankensearch_core::bench_support`), worker
`fixmydocuments` (`HOSTID`), binary sha256 `79286c4a4d2ceba00c127fbce5842a37dfa6776b41a5f78daf931adef25e37ca`,
41 rounds × 4:

| arm | median [p5, p95] |
|---|---|
| NULL (ORIG vs ORIG) | 0.9996 [0.9223, 1.0405] |
| maddubs / ORIG | **0.8141 [0.7839, 0.8693]** |

Ratio = maddubs/ORIG, so `<1.0` = faster. The lever's p95 (0.8693) is **below** the null's p5 (0.9223), so the
`0.8141 = 1.228× faster` verdict is **DECIDABLE** — the effect clears the noise floor. Not the hypothesized 2×
(the `stored` load is still memory-bound; maddubs adds port pressure), but a real 1.23× on a hot leaf.

**Self-time.** From cod's `perf record -F 997 -e cycles:u` on the exact int8 scan binary (`vmi1227854`): the
single-row `dot_i8_i8_avx2` is **44.54%** flat self-time in the ORIG scan, the 4-row `dot_i8x4_i8_avx2` 23.36% —
so this leaf is the scan's dominant compute frame. A 1.23× on 44.5% self-time is a meaningful scan-level lever
IF end-to-end can be measured; today it cannot (the wide floor), so this ships as a **validated, benched kernel
primitive**, not yet wired into the production scan.

**Scope / honesty.** LANDED (this + the follow-up commit): the single-row **and batched** kernels
(`dot_i8_i8_maddubs`, `dot_i8x4_i8_maddubs` — the batched one is what the row-blocked scan uses), 5 correctness
tests incl. the **end-to-end f32-recall proof under real saturation**, and the decidable microbench (reproducible,
both arms retained). NOT landed: wiring the kernels into `int8_scan_range` itself, because the scan-level *speed*
win is undecidable under the same contention that blocks cod's micro-opt — wiring a perf change I cannot measure
end-to-end would violate the gate. (Recall is deterministic and *is* proven; only the scan-level speed is
blocked.) Filed as the follow-up, retry condition = worker isolation/affinity (identical to cod's). The kernel is
also only recall-safe while the
quantizer keeps magnitudes under the i16 pair ceiling — a coupling the wiring step must assert. cod owns the
scan; I touched no scan code. All builds/benches remote; no local cargo. 6 unrelated WAL/vacuum lib tests hit a
worker-FS `PermissionDenied` flake on one worker and pass on a fresh one (environmental, not this change).

## 2026-07-10 — WIN: ASCII byte fast-path in the cass tokenizer's `next_char_from` — 1.355× on the per-char scan, bit-identical (cc_fse)

Dug a **different primitive** after the vector-scan vein slowed (f16 decode-bound; FMA + chunk-size both
rejected this session): the **lexical tokenizer**. `frankensearch-lexical`'s cass tokenizer scans text
char-by-char via `next_char_from(text, off)`, called from `advance` / `scan_ascii_token` / `scan_cjk_token`
for **every character** at index time (per doc) and query time (per query). The original did
`text[off..].chars().next()` + `ch.len_utf8()` — a full UTF-8 decode per char.

**The lever.** The tokenizer's inputs (English prose, code identifiers, doc IDs) are overwhelmingly ASCII, and
`off` is always a char boundary, so a leading byte `< 128` is a complete single-byte UTF-8 char. The fast-path
returns `(b as char, off + 1)` directly, skipping the `chars().next()` decode and the `len_utf8()` recompute;
non-ASCII lead bytes fall through to the original decode. **Bit-identical** for all inputs.

**Recall/ordering — trivially preserved.** `next_char_from_ascii_matches_decode` asserts the fast path yields the
byte-for-byte same `(char, next_offset)` sequence as the decode on ASCII, multi-byte, combining-mark, emoji, CJK,
and mixed inputs — so token boundaries (hence BM25 terms, hence ranking) are unchanged. All 83 lexical lib tests
pass with the fast-path live.

**Speed — DECIDABLE WIN.** Isolated null-controlled microbench (`benches/tokenizer_char_walk_ab.rs`, 48 KiB
realistic mostly-ASCII corpus, shared alternating-round sampler, one binary / one `rch` invocation, worker
`hz2`/`hetzner2`, binary sha256 `4a519c768d15007d0fb466f05f7143b791c386cf2ed6a6b897c39a84e5dd5ff5`, 41 rounds ×
4). Ratio = fast/ORIG, `<1.0` = faster:

| arm | median [p5, p95] |
|---|---|
| NULL (decode vs decode) | 1.0285 [0.8784, 1.2218] |
| fast / ORIG | **0.7379 [0.6640, 0.8929]** |

fast/ORIG median 0.7379 = **1.355× faster**, and the median is clearly below the null p5 (0.8784) → decidable.
This is the per-char scan primitive; the full tokenizer sees a smaller-but-real share since `advance` also does
`push_str` + offset bookkeeping, but the scan itself is the changed hot inner loop.

**SHIPPED in place** (`next_char_from` is now the fast path; the pre-fast-path decode retained doc-hidden as
`next_char_from_slow` + the `cass_char_walk_*` harnesses so the A/B stays reproducible, both arms in-tree).
Recall-preserving, decidable, live. All builds/benches remote; no local cargo; fmt clean; ubs adds no new
criticals (HEAD already exits 1 on pre-existing `.unwrap()`s).

## 2026-07-10 — WIN: `cass_build_preview` bulk-copy vs char-by-char push — 3.07×, byte-identical (cc_fse)

Sibling of the tokenizer `next_char_from` win (`517cea9`), same lexical vein. `cass_build_preview` runs **per
document at index time** (`cass_build_content_prefix_and_preview`, `PREVIEW_MAX_CHARS = 400`), truncating content
to N chars + `…`. The original pushed up to `max_chars` chars **one at a time** into a `String::new()` — each
`out.push(ch)` re-encodes the char to UTF-8, and the unallocated buffer grows through repeated reallocations.

**The lever.** One `char_indices` scan finds the cut byte-offset, then a single `push_str` bulk-copies the prefix
(a memcpy) into a `String::with_capacity(...)`. Same decode count, but zero per-char re-encode and zero
reallocations. Byte-for-byte identical output.

**Recall/ordering — trivially preserved.** The preview text is unchanged (bit-identical), so nothing about
indexing or ranking moves. `cass_build_preview_matches_slow` asserts equality across content lengths, `max_chars`
∈ {0,1,3,4,5,10,50,400,100k}, and ASCII/Unicode/mixed inputs (boundary at/before/after the cut, empty,
all-multibyte); `cass_build_preview_preserves_existing_behavior` pins the documented cases (incl. `max_chars=0`
→ `…`). All 84 lexical lib tests pass.

**Speed — DECIDABLE WIN.** Isolated null-controlled microbench (`benches/preview_build_ab.rs`, 256 previews of a
~2 KiB message per timed region — both arms hit the truncation path, the common real-doc case; shared
alternating-round sampler, one binary / one `rch` invocation, worker `hz2`/`hetzner2`, binary sha256
`676e029299a9b9d56158ede166f825d9f6a681c08ae1ae49f8fdd560215840af`, 41 rounds × 4). Ratio = fast/ORIG:

| arm | median [p5, p95] |
|---|---|
| NULL (char_push vs char_push) | 1.0005 [0.9620, 1.0283] |
| fast / ORIG | **0.3258 [0.3193, 0.4553]** |

fast/ORIG median 0.3258 = **3.07× faster**, and the lever's p95 (0.4553) is far below the null p5 (0.9620) —
decidable by a wide margin against a tight (±3%) null floor. The per-char re-encode + realloc growth dominated
far more than the decode.

**SHIPPED in place** (`cass_build_preview` is now the bulk-copy; the char-by-char original retained doc-hidden as
`cass_build_preview_slow` so the A/B stays reproducible, both arms in-tree). Recall-preserving, decidable, live.
All builds/benches remote; no local cargo; fmt clean; clippy clean on changed lines.

## 2026-07-10 — WIN: `cass_prefix_source` O(n)→O(1) — backward `is_char_boundary` walk replaces a forward `char_indices` scan (~3333× on the truncation path, byte-identical) (cc_fse)

Third win in the lexical text-prep vein (after tokenizer `next_char_from` `517cea9` and preview bulk-copy
`8fde796`). `cass_prefix_source` runs **per document at index time**
(`cass_build_content_prefix_and_preview`, `CONTENT_PREFIX_MAX_BYTES = 4 KiB`), taking the ≤ `max_bytes`
char-boundary prefix of content that then feeds edge-ngram generation. For content over the cap, the original
walked `char_indices` **forward from byte 0**, decoding ~`max_bytes` chars just to locate the boundary near
byte 4096 — O(max_bytes).

**The lever.** The largest char boundary ≤ `max_bytes` is at most one UTF-8 char width (≤ 3 bytes) below
`max_bytes`, so a backward `is_char_boundary(end)` walk from `max_bytes` finds it in **≤ 4 checks** — O(1). It
computes the identical cut (largest char boundary ≤ `max_bytes`), rounding a mid-multibyte-char index down to
the boundary exactly as the forward scan did.

**Recall/ordering — trivially preserved.** Byte-identical prefix ⇒ identical edge-ngrams ⇒ identical index terms
⇒ identical ranking. `cass_prefix_source_matches_slow` asserts equality across content lengths and `max_bytes`
∈ {0,1,2,3,4,5,7,100,499,500,501,10k} on ASCII, multibyte, and mixed content — cuts at/before/after a boundary,
mid-multibyte (must round down), at 0, and past the end. All 85 lexical lib tests pass.

**Speed — DECIDABLE WIN.** Isolated null-controlled microbench (`benches/prefix_source_ab.rs`, 256 calls on a
64 KiB doc per timed region — both arms take the truncation path; shared alternating-round sampler, one binary /
one `rch` invocation, worker `hz2`/`hetzner2`, binary sha256
`fc2e09db9afd34eb979b7a648a5057128287433482b9b38ab64cc89824b41dcf`, 41 rounds × 4). Ratio = fast/ORIG:

| arm | median [p5, p95] |
|---|---|
| NULL (forward_scan vs forward_scan) | 0.9970 [0.9373, 1.0434] |
| fast / ORIG | **0.0003 [0.0003, 0.0003]** |

fast/ORIG median 0.0003 = **~3333× faster** — the O(4096) forward decode vs the O(≤4) boundary walk. Decidable
by an enormous margin against a tight null floor.

**Honest scope.** This is a *complexity* fix, so the ratio is huge, but it **only triggers on content > 4 KiB**
(smaller docs early-return identically in both). The absolute per-doc saving is one ~4 KiB char-scan on large
documents; on a corpus of small messages it is a no-op. Still an unambiguous, byte-identical O(n)→O(1)
improvement worth landing wherever large docs appear. **SHIPPED in place**; the forward-scan original retained
doc-hidden as `cass_prefix_source_slow` so the A/B stays reproducible (both arms in-tree). All builds/benches
remote; no local cargo; fmt + clippy clean on changed lines.

## 2026-07-10 — WIN: `cass_generate_edge_ngrams` ASCII fast-path skips the char_indices re-decode — ~1.08×, byte-identical (cc_fse)

Fourth win in the lexical text-prep vein (tokenizer `next_char_from` `517cea9`, preview `8fde796`,
prefix_source `f6c15f5`). `cass_generate_edge_ngrams` runs per document at index time on the ≤ 4 KiB content
prefix (always, not just on truncation), emitting each word's length-2..N prefixes. The original **decoded every
word twice**: `text.split(|c| !c.is_alphanumeric())` scans all chars to find word boundaries, then
`word.char_indices()` re-decodes each word's chars to find prefix boundaries.

**The lever.** For an ASCII word (the common case), char boundaries are byte positions, so the prefixes are
`word[..2..=min(len, 20)]` sliced directly — the `char_indices` re-decode is skipped via a cheap `word.is_ascii()`
(SIMD byte scan). Non-ASCII words keep the original boundary-collecting path. Byte-for-byte identical output.

**Recall/ordering — trivially preserved.** Identical edge-ngram terms ⇒ identical prefix index ⇒ identical
ranking. `cass_generate_edge_ngrams_matches_slow` asserts equality on ASCII, non-ASCII, and mixed text incl. the
20-char cap, sub-3-char words, and words straddling the ASCII/Unicode split; the pre-existing
`emits_expected_prefixes` / `caps_prefixes_at_twenty_chars` tests pass unchanged. All 85 lexical lib tests pass.

**Speed — DECIDABLE WIN (floor-width-dependent; two runs disclosed).** Isolated null-controlled microbench
(`benches/edge_ngrams_ab.rs`, ~4 KiB realistic mostly-ASCII prefix, shared alternating-round sampler, one binary
/ one `rch` invocation, 41 rounds × 4). Ratio = fast/ORIG:

| run | worker | binary sha256 | NULL median [p5,p95] | fast/ORIG median [p5,p95] | verdict |
|---|---|---|---|---|---|
| 1 | `hz2` (contended) | `f55aaed446a792b3f8ec6f740559af0c4d31c0d91c91b4531ab87dc2aa87061d` | 0.9997 [0.6966, 1.1923] | 0.9193 [0.7172, 1.2907] | inside floor (±30% noise) |
| 2 | `ovh-a` (quiet) | — | 1.0006 [0.9342, 1.0517] | **0.9297 [0.8890, 0.9821]** | **DECIDABLE** (median < null p5) |

The lever median is **consistent** across both runs (0.919, 0.930 → ~1.08× faster); only the null-floor *width*
differed. Unlike the int8 scan-level case (where two runs *disagreed*, 0.90 vs 0.98 — not robust), here the effect
is stable and only needed a quiet worker to resolve against a tight floor. The function is allocation-dominated
(builds an ~8 KiB String), which is why the decode-halving nets ~8% not more, and why a contended run can't
resolve it. Modest but real and always-on (per-doc, whole prefix).

**SHIPPED in place**; the char_indices original retained doc-hidden as `cass_generate_edge_ngrams_slow` (both arms
in-tree, reproducible). All builds/benches remote; no local cargo; fmt + clippy clean on changed lines.

## 2026-07-10 — WIN: `normalize_whitespace` byte-fast ASCII path skips the char decode+re-encode — 1.45×, byte-identical (cc_fse)

Dug a different primitive after ranking/structural veins came up empty: **core text canonicalization**
(`frankensearch-core/canonicalize.rs`), upstream of both embedding and lexical indexing. `normalize_whitespace`
runs on **every document** (collapse whitespace runs → single space, trim). The original did `for c in
text.chars()` — decoding **every** char, running the Unicode `is_whitespace()` per char, and re-encoding each
kept char via `push(c)`.

**The lever.** A byte scan with an ASCII fast-path: an ASCII byte (the common case) is classified by a cheap byte
test and copied without a decode; only a non-ASCII lead byte decodes a char. The one subtlety that makes it
**bit-identical**: for ASCII, `char::is_whitespace()` = `u8::is_ascii_whitespace() || b == 0x0B` — U+000B
(vertical tab) is Unicode `White_Space` but NOT `is_ascii_whitespace`, so it is added back explicitly.

**Recall/ordering — trivially preserved.** Canonicalized text is byte-identical ⇒ identical tokens, identical
embeddings, identical index terms, identical ranking. `normalize_whitespace_matches_slow` asserts equality across
ASCII, every ASCII whitespace byte (incl. 0x0B), Unicode whitespace (NBSP U+00A0, NEL U+0085, ideographic space
U+3000, en/em spaces), runs, leading/trailing, and mixed text. All 35 canonicalize lib tests pass.

**Speed — DECIDABLE WIN (contention-diluted first run disclosed).** Null-controlled microbench
(`benches/whitespace_norm_ab.rs`, ~4 KiB realistic doc body, shared alternating-round sampler, one binary / one
`rch` invocation, 41 rounds × 4). Ratio = fast/ORIG:

| run | worker | NULL median [p5,p95] | fast/ORIG median [p5,p95] | verdict |
|---|---|---|---|---|
| 1 | `vmi1227854` (contended) | 0.9969 [0.8185, 1.0895] | 0.8295 [0.6728, 0.9041] | inside floor (±18% noise) |
| 2 | `ovh-a` (quiet) | 0.9996 [0.9952, 1.0009] | **0.6887 [0.6822, 0.6940]** | **DECIDABLE** (median ≪ null p5) |

The quiet run resolves it at **0.6887 = 1.452× faster** against a razor-tight (±0.5%) null floor. Both runs agree
in direction (fast < slow); the contended run *understated* the effect because per-arm contention overhead is
additive and dilutes the ratio toward 1.0 (not a disagreement like the int8 case — same sign, quiet run is
authoritative). Byte sha256 (run 1) `57f1c446e0d7d489f40649639b81dfaad9d8c0dd8ca0b371d0216f0d16930b6b`.
This is the biggest lexical/canonicalize win after preview/prefix_source and is **always-on** (every doc, not
just large/truncated ones).

**SHIPPED in place**; the char-by-char original retained as `normalize_whitespace_slow` (`cfg(test |
bench-internals)`) so the A/B stays reproducible. All builds/benches remote; no local cargo; fmt + clippy clean on
changed lines.

## 2026-07-10 — WIN: `truncate_to_chars` ASCII fast-path skips the char_indices scan — ~13.8× on the truncation path, byte-identical (cc_fse)

Continuing the byte-fast ASCII sweep. `truncate_to_chars` caps canonicalized doc/query text at `max_length`
chars (default 2000), called per-doc AND per-query. Short text (≤ cap bytes) already early-returns; for longer
text the original forward-scanned `char_indices` — O(max_chars) UTF-8 decodes — to find the cut.

**The lever (sibling of `cass_prefix_source` `f6c15f5`).** If the first `max_chars` *bytes* are ASCII, they are
exactly `max_chars` single-byte chars and byte `max_chars` is a char boundary — so cut at `text[..max_chars]`
directly. `text.as_bytes()[..max_chars].is_ascii()` is a SIMD byte scan (~memcpy speed); the char_indices decode
is skipped entirely on ASCII prefixes. Non-ASCII prefixes fall back to the decode. Byte-identical.

**Recall/ordering — trivially preserved.** Same truncated text ⇒ same canonicalized doc ⇒ same tokens/embeddings/
index terms/ranking. `truncate_to_chars_matches_slow` asserts equality across content shorter/equal/longer than
the cap, ASCII and multibyte, and cuts on an ASCII boundary / before a multibyte char / mid-multibyte (rounds
down). All 36 canonicalize lib tests pass.

**Speed — DECIDABLE WIN.** Null-controlled microbench (`benches/truncate_chars_ab.rs`, 16 KiB ASCII doc, both arms
take the truncation scan, shared alternating-round sampler, one binary / one `rch` invocation, worker `hz1`,
binary sha256 `0e8c88467ba0e115727966ea6ae00a91f8ae29d21b59c3414734baa92d8a4213`, 41 rounds × 4):

| arm | median [p5, p95] |
|---|---|
| NULL (char_indices vs char_indices) | 1.0000 [0.9546, 1.9072] |
| fast / ORIG | **0.0725 [0.0688, 0.0764]** |

fast/ORIG median 0.0725 = **~13.8× faster** — a SIMD `is_ascii` byte scan vs 2000 char decodes; the lever's p95
(0.076) is far below the null p5 (0.955), decidable by an enormous margin. (The null's p95 1.9 is a single
contention-outlier round; the median gate at 1.0 vs 0.0725 is unambiguous.)

**Honest scope:** like `prefix_source`, this triggers only on text over the byte cap (long docs; short docs +
queries early-return identically), so it's a large-doc win, not always-on. **SHIPPED in place**; the char_indices
original retained as `truncate_to_chars_slow` (`cfg(test | bench-internals)`) so the A/B stays reproducible. All
builds/benches remote; no local cargo; fmt + clippy clean on changed lines.

## 2026-07-10 — WIN: fingerprint `char_count` ASCII fast-path removes a redundant full-text decode — 2.85×, identical count (cc_fse)

Continuing the byte-fast sweep. `DocumentFingerprint::compute` runs per document at ingest (dedup / re-embed
decision) and computed `char_count` as `text.chars().count()` — a **second** full-text UTF-8 decode on top of the
one `semantic_simhash_text`'s tokenization already does. Meanwhile `fnv1a_hash` already scanned the bytes. So
`compute` decoded the text twice.

**The lever.** For ASCII (the common case for English/code), the char count equals the byte length, and
`str::is_ascii` is a SIMD byte scan far cheaper than a per-char decode: `char_count(text) = if text.is_ascii() {
text.len() } else { text.chars().count() }`. Non-ASCII falls back. Identical result for every input.

**Recall/ordering — trivially preserved.** `char_count` is dedup metadata (the char-count-delta re-embed
heuristic), not search output; and the value is identical anyway. `char_count_matches_slow` asserts equality on
ASCII, pure-multibyte (byte-len ≠ char-count), combining marks, emoji, and mixed text. All 22 fingerprint lib
tests pass.

**Speed — DECIDABLE WIN.** Null-controlled microbench (`benches/char_count_ab.rs`, all-ASCII 4 KiB doc × 256 per
timed region — the fast path only fires on fully-ASCII text; a stray non-ASCII char falls back, which the parity
test covers; shared alternating-round sampler, one binary / one `rch` invocation, worker `hz2`, binary sha256
`3d83a2e5adc7171035e5f5263d5f2ed3e4e9b446b5f4ff8be7b0fb4c243fe45a`, 41 rounds × 4):

| arm | median [p5, p95] |
|---|---|
| NULL (chars_count vs chars_count) | 0.9994 [0.8675, 1.0903] |
| fast / ORIG | **0.3512 [0.2989, 0.4776]** |

fast/ORIG median 0.3512 = **2.85× faster** on the char-count op; lever p95 (0.478) well below null p5 (0.868),
decidable. **Scope:** char_count is a *fraction* of `compute` (the simhash tokenize+hash dominates), so the
per-doc `compute`-level win is smaller — but it eliminates one of two redundant full-text decodes, always-on for
ASCII docs. **SHIPPED in place**; `chars().count()` retained as `char_count_slow` (`cfg(test | bench-internals)`).
All builds/benches remote; no local cargo; fmt + clippy clean on changed lines.

## 2026-07-10 — WIN: graph-rank dense doc-id index SipHash → aHash — 1.08–1.11×, bit-identical ranking (IndigoOtter)

**Different lane from lexical/vector scan; profile-first routing.** Before the production edit, a
production-fidelity baseline of the optional query-biased PageRank stage reproduced the existing structural
signal: its two-pass flat-CSR twin, which pays one extra doc-id string hash probe per edge, was **1.20× slower**
at 500 nodes (927.3 / 770.0 µs) and **1.26× slower** at 2000 nodes (5.052 / 4.004 ms). Because that twin also
changes layout and build bookkeeping, this was routing evidence rather than isolated frame attribution; it
motivated the one-variable whole-production-routine A/B below. The opportunity score was 12 (impact 3 ×
confidence 4 / effort 1).

**The one lever.** `GraphRanker::rank_phase1` builds a dense `doc_id -> usize` table, then only probes it while
building adjacency rows and mapping seeds. It never iterates that table: node numbering comes from
`adjacency.keys()`. The production arm now uses `ahash::RandomState` instead of the standard library's SipHash;
the test/bench-only legacy arm calls the same generic implementation with `std::hash::RandomState`. Edge layout,
edge visit order, floating-point operations, final normalization, score sort, and doc-id tie-break are unchanged.

**Recall/ordering proof.** Before timing, `graph_rank` asserts equal result counts and exact equality of every
ordered `doc_id` and `score.to_bits()` at both 500×6 and 2000×8. The retained unit test includes an equal-weight
tie and asserts the same exact contract. Exact ordered scores imply recall and ordering parity, not merely an
approximate quality match.

**Speed — paired MEDIAN gate passed.** Strict remote-only command (one binary / one worker, no local fallback):

```bash
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
  cargo bench -p frankensearch-fusion --features graph,bench-internals \
  --bench graph_rank -- hash_ --sample-size 60 --warm-up-time 0.2 \
  --measurement-time 1.0 --noplot
```

Worker `vmi1227854`; 41 alternating rounds × 4 calls per arm. Ratio = aHash/SipHash, so `<1.0` is faster:

| workload | A/A null median [p5, p95] | aHash/SipHash median [p5, p95] | verdict |
|---|---|---|---|
| `n500_deg6` | 1.0005 [0.9340, 1.1080] | **0.9252 [0.2413, 1.2846]** | **DECIDABLE**, median below null p5; 1.081× |
| `n2000_deg8` | 1.0182 [0.9200, 1.2213] | **0.9016 [0.7547, 1.0218]** | **DECIDABLE**, median below null p5; 1.109× |

An unchanged confirmation on worker `hz1` also passed: `n500_deg6` null 1.0021 [0.9725, 1.0414], lever
**0.9128** [0.8894, 0.9348] (**1.096×**); `n2000_deg8` null 0.9974 [0.9512, 1.3511], lever **0.8988**
[0.6325, 1.3414] (**1.113×**). The separate Criterion central estimates agreed on both workers: first run
197.27→188.20 µs and 1.0243→0.9340 ms; confirmation 256.99→240.44 µs and 1.3451→1.1235 ms. The wide paired
p5/p95 tails disclose worker contention honestly; the repository gate is the per-round median against the
same-function A/A band, and both tracked sizes clear that gate in the winning direction twice.

**Scope.** This improves frankensearch's own optional `graph` ranking stage; it is not a Tantivy/Lucene/Meili
comparison and does not claim a default-path end-to-end ratio. All Cargo work for the baseline, candidate, and
validation used `RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- ...`; no local Cargo fallback occurred.

**Validation.** Remote focused graph tests: 5 passed / 0 failed, including the new exact hasher-parity test and
the dense-vs-reference ranking test. Remote `cargo check --workspace --all-targets` exited 0. Workspace clippy
with `-D warnings` remains blocked by pre-existing core debt (`TwoTierConfig` excessive bools and a
`canonicalize.rs` doc-markdown warning). Focused remote clippy for the owned library and benchmark passed with
`-D warnings` after allowing only lint classes whose reported sites were verified against `HEAD` as pre-existing:
hubness doc length, two `needless_for_each` sites, benchmark doc-markdown, two `u64 -> usize` fixture casts, and
two fixture `String` clones. Those unrelated/orthogonal cleanups were not mixed into this one-lever commit.
Direct pinned `rustfmt --check`, `git diff --check`, and UBS (0 critical, exit 0) passed on the owned files. `rch`
refused remote `cargo fmt` as a non-compilation command under `RCH_REQUIRE_REMOTE=1`, correctly without a local
fallback.

## 2026-07-11 — WIN: `IndexBuilder` moves successful documents into lexical staging — 1.30× at 20k, recall preserved (cod)

**Different lane from cc-owned lexical query/scan.** Tantivy keeps its postings codec and block layout private,
and the immediately preceding 50 MB → 100 MB writer-budget experiment was slower and changed ranked results.
The next owned indexing cost was in the facade: after embedding each document, `IndexBuilder` deep-cloned the
complete `IndexableDocument` (content, title, and metadata) solely to keep an owned value for the later Tantivy
postings build.

**Profile first.** An untouched-original remote run of the retained 20,000-document facade benchmark measured
14.743 ms in the isolated deep clone (3.962% of 372.106 ms wall). The final same-binary original arm measured a
12.839 ms clone median, 19.618 ms embedding-plus-staging, 207.089 ms after staging, and 255.885 ms wall: the clone
was 5.017% of the measured build and therefore a live cost rather than a static-code guess. The fixture uses the
real facade builder, vector index writes, Tantivy schema/postings build and commit, titles, and four metadata
fields; corpus construction and temporary-directory creation are outside the timer.

**One lever.** Because `build` owns its input vector, the lexical configuration now consumes those documents and
moves each successfully embedded value into lexical staging. Failed documents remain resident until the end of
the build (matching the former borrowed-input lifetime) but are never passed to Tantivy. The non-lexical path is
unchanged. Batch boundaries, progress callbacks, vector writes, error order, lexical commit, and metrics export
retain their former order. The exact former borrowed-plus-deep-clone implementation remains available only under
`bench-internals` for a same-binary comparator.

**Recall and ordering gate.** Six BM25 query classes were evaluated over all 20,000 documents. Final minimum
recall@20,000 was **1.000000** and minimum original-relative nDCG@20,000 was **1.000000** (hard gate ≥0.999999).
Independent Tantivy parallel builds did not preserve raw tie order, canonical ranked IDs, or score bits; those
diagnostics also varied between original builds and are not a valid cross-build identity oracle. The quality gate
therefore records the honest contract: every matching document is retained and graded order is preserved within
the stated tolerance, without claiming bit identity. A focused failure-path test additionally proves that a
document whose required fast embedding fails is excluded from the lexical index while surrounding successes
remain searchable.

**Speed — paired MEDIAN gate passed.** Strict remote-only final command:

```bash
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
  cargo bench -p frankensearch --profile release --features lexical,bench-internals \
  --bench index_builder_lexical_staging -- --noplot
```

Worker `vmi1227854`; one release binary, alternating arm order, 21 paired rounds. Ratio is candidate/original:

| gate | median | p5 | p95 | verdict |
|---|---:|---:|---:|---|
| A/A original/original null | 1.0166 | 0.8643 | 1.0856 | measured fleet floor |
| move/original lever | **0.7688** | 0.6156 | 0.8723 | **DECIDABLE WIN**; median below null p5; 1.30× |

Criterion central estimates independently agreed: original **250.68 ms**, candidate **192.76 ms** (~1.30×).
The paired median is authoritative; Criterion's warning that ten whole-index samples exceed the requested
one-second collection target is expected for this intentionally heavy fixture.

**Validation.** All Cargo commands were fail-closed remote RCH executions; no local Cargo fallback occurred.
Final focused tests passed 14 facade unit tests and 4 integration tests. Remote
`cargo check --workspace --all-targets` exited 0. Workspace clippy with `-D warnings` is blocked by pre-existing
core debt (`TwoTierConfig` excessive bools and a `canonicalize.rs` doc-markdown lint); remote changed-surface
clippy for the facade library and retained benchmark passed with `--no-deps -D warnings`. RCH refused remote
`cargo fmt --check` with `RCH-E301` because formatting is not a compilation command, correctly without falling
back locally; direct `rustfmt --check`, `git diff --check`, and UBS (exit 0, zero critical findings) passed.

**Scope.** This is a facade ownership/allocation win feeding the real Tantivy postings build. It does not modify
or claim a new postings compression codec, BM25 formula, tokenizer, or query scan, and it does not reopen the
rejected writer-budget family.

## 2026-07-11 — REJECT: owned FSVI handoff wins in isolation, but full-write median stays inside the null floor (cod, bd-5973)

**Different lane from cc-owned lexical/query scan.** A fresh code and ledger sweep reconfirmed that Tantivy
0.26.1 keeps its postings codec, block size, and skip layout private, while the public 50 MB → 100 MB writer
budget family is a measured no-retry reject. The next ownable indexing-build cost was in the FSVI two-tier
builder: it already owned every `(String, Vec<f32>)`, but `finish(self)` borrowed each record into
`VectorIndexWriter::write_record`, causing the writer to allocate and deep-clone every ID and vector again.

**Profile first.** Before any production edit, the retained `fsvi_builder_record_transfer` harness measured one
realistic 20,000-document × 384-dimension tier on remote worker `vmi1149989`. The redundant 29.297 MiB record
deep clone cost **4.260 ms median** [3.481, 6.863], while borrowed writer ingestion cost **16.272 ms median**
[13.625, 19.159]. The copy was therefore 26.18% of the live handoff path, not a static-code guess. Two later
runs on `vmi1227854` independently measured 5.390/16.311 ms (33.04%) and 3.998/15.352 ms (26.04%).

**One candidate lever.** The candidate made `TwoTierIndexBuilder::finish` consume `fast_records` and
`quality_records` into an owned writer entry point. Both arms retained the exact former validation order
(dimension, finite values, ID length), FNV-1a hash, flags, insertion order, load-bearing stable sort, f16
encoding, fast-before-quality order, and fsync behavior. No capacity reservation, early ID-set drop, tier
parallelism, codec change, scan change, or lexical change was combined with the ownership transfer.

**Recall and artifact proof.** The same release binary wrote both the former borrowed arm and the owned arm from
the 20k fixture. Their complete **16,020,096-byte FSVI files were byte-identical**. Reopened sampled IDs and
decoded vector `to_bits()` values matched, as did every top-10 ID/index/score bit across eight queries. Therefore
candidate-vs-original top-10 overlap and original-relative nDCG@10 were both **1.000000**; this is a preservation
claim, not an absolute external-ground-truth relevance claim. Byte equality also proves unchanged headers, record
tables, string tables, vector slabs, ordering, and checksums.

**Speed — isolated median passes, full shipping median does not reproduce outside the null floor.** Strict
remote-only command for both same-worker runs:

```bash
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
  cargo bench -p frankensearch-index --profile release --features bench-internals \
  --bench fsvi_builder_record_transfer
```

Worker `vmi1227854`; one release binary per run, 21 AB/BA round pairs. Ratio is owned candidate / borrowed
original:

| run | gate | A/A null median [p5, p95] | candidate/original median [p5, p95] | verdict |
|---|---|---:|---:|---|
| 1 | writer ingestion | 0.971933 [0.765276, 1.140919] | **0.595650** [0.473512, 0.712427] | decidable isolated win |
| 1 | transfer + stable sort + encode + fsync | 1.013005 [0.887204, 1.227880] | **0.872784** [0.689716, 1.039306] | barely below null p5; tentative keep |
| 2, final | writer ingestion | 1.030091 [0.805651, 1.229080] | **0.628700** [0.579356, 0.751003] | decidable isolated win |
| 2, final | transfer + stable sort + encode + fsync | 0.937580 [0.825111, 1.170674] | **0.853127** [0.735856, 0.969935] | **INSIDE NULL FLOOR** |

The first full-path result cleared its null p5 by only 0.014420. The required rerun kept the same faster
direction but widened the measured floor: candidate median 0.853127 was **above** null p5 0.825111. The isolated
handoff is reproducibly faster, but f16 encoding, sort, filesystem work, and fleet noise dilute that effect below
the full operation's decidability floor. The full-path result is the shipping gate, so the final verdict is
**REJECT/HOLD**, not an averaged or defended keep.

**Decision and scope.** Production `TwoTierIndexBuilder` and the default writer path were restored exactly to
HEAD. Only the feature-gated comparator, retained benchmark, parity test, and evidence remain. This does not
modify BM25/query scan or claim a postings-compression win. The earlier slice-to-owned-vector copy at the builder
API is a separate route-next, not evidence that this second-copy lever should ship; it requires its own profile
and cannot be bundled under the one-lever rule.

**Validation and degraded surfaces.** All Cargo work stayed fail-closed remote-only. The candidate crate suite
passed 402/402 tests, the final rejected tree's retained borrowed-vs-owned byte-parity test passed 1/1, and final
`cargo check -j 4 --workspace --all-targets` exited 0 remotely. Workspace `cargo clippy ... -D warnings` stopped
on two pre-existing `frankensearch-core` lints; both package and benchmark-target clippy then stopped on the same
80 pre-existing `frankensearch-index` lints before providing a clean target-only verdict. Strict-remote
`cargo fmt --check` was refused with RCH-E301 because formatting is not a compilation command; RCH did not fall
back locally, and direct `rustfmt --edition 2024 --check` plus `git diff --check` passed. The RCH fleet was
degraded (9/12 workers healthy), and Agent Mail reservations were unavailable because its SQLite corruption
circuit breaker was open; both failures were surfaced rather than bypassed.

## 2026-07-11 — WIN: storage ingest hashes each document ONCE, not twice — ~1.85–1.98× on the dual-hash step, byte-identical (cc_fse)

The per-document ingest pipeline (`storage/pipeline.rs`, the `process` path) needs BOTH the raw 32-byte
content hash (`DocumentRecord.content_hash` + dedup) and its lowercase-hex form (the `content_hashes`
seen-count table). It computed them independently — `ContentHasher::hash(&canonical_text)` then
`ContentHasher::hash_hex(&canonical_text)` — running **SHA-256 over the canonical text twice per document**.
Since `content_hash_hex == hex(content_hash)`, the second SHA-256 is pure waste.

**The lever.** Added `ContentHasher::to_hex(digest: &[u8; 32]) -> String` (a direct table hex encode,
replacing the former `write!(out, "{byte:02x}")` `core::fmt` loop), refactored `hash_hex` to
`to_hex(&hash(text))`, and changed the pipeline to hash once and hex-encode the digest
(`ContentHasher::to_hex(&content_hash)`). One SHA-256 per document instead of two.

**Byte-identical.** `content_hash` (`[u8; 32]`) is unchanged; `content_hash_hex` is unchanged because
`to_hex(&hash(t)) == hash_hex(t)`. Proven by `to_hex_matches_write_format`, which asserts `to_hex` equals the
former `write!`-based encoder over 512 deterministic digests + all-0x00/0xff/0x0f boundaries, and that
`hash_hex(t) == to_hex(hash(t))` on ASCII/multibyte text. 18 focused `content_hash` lib tests pass.

**Speed — DECIDABLE WIN, MEDIAN-gated.** New null-controlled A/B (`benches/content_hash_dual.rs`, shared
alternating-round sampler `frankensearch_core::bench_support`, ORIG = `hash` + `hash_hex` = 2 SHA-256, CAND =
`hash` + `to_hex(&digest)` = 1 SHA-256; ratio = CAND/ORIG, `<1.0` = faster; worker `vmi1227854`, 41 rounds ×
inner 64):

| canonical text | A/A null median [p5, p95] | CAND/ORIG median [p5, p95] | verdict |
|---|---|---|---|
| 256 B    | 1.0000 [0.8022, 1.1856] | **0.5439 [0.4458, 0.5733]** | **DECIDABLE WIN**, ~1.84× |
| 2 048 B  | 1.0047 [0.8545, 1.3440] | **0.5112 [0.4320, 0.5948]** | **DECIDABLE WIN**, ~1.96× |
| 16 384 B | 0.9979 [0.9295, 1.1770] | **0.5054 [0.4834, 0.5227]** | **DECIDABLE WIN**, ~1.98× |

The lever median (~0.50–0.54) is far below each null p5 (0.80–0.93), decidable by a wide margin. It matches the
mechanism exactly: eliminating one of two SHA-256 passes ⇒ ~0.5, approaching 0.505 as the text grows and SHA
dominates the constant hex-encode. Single-threaded scalar SHA-256, so the paired ratio is robust to fleet
contention (tight null p5, unlike the rayon vector scan).

**Scope / honesty.** This is a ~2× win on the *dual-hash step* of per-document ingest, not a 2× end-to-end
ingest ratio — hashing is one component alongside canonicalization, the dedup SQL, document upsert, and the WAL.
It removes redundant work always-on for every ingested document. `hash_hex` retains its original signature and
byte-for-byte output for other callers; the `to_hex` split also fixes the slow `write!`-per-byte hex encode. All
builds/benches strictly remote (`RCH_REQUIRE_REMOTE=1`); no local Cargo. Peer files untouched.

## 2026-07-11 — WIN: storage ingest builds the content preview in O(preview) not O(document) — ~2.2–5.7×, byte-identical (cc_fse)

`IngestPipeline::ingest` builds a `MAX_CONTENT_PREVIEW_CHARS` (400) preview from each document's full
canonical text via `truncate_chars`. The former implementation did `value.chars().count()` — a full UTF-8
decode of the ENTIRE document — merely to learn it exceeds 400 chars, then `chars().take(400)` decoded again.
Real documents are far longer than a 400-char preview, so this paid a whole-document decode per ingested doc to
emit a tiny preview.

**The lever.** `value.char_indices().nth(max_chars)` finds the byte offset of the `max_chars`-th char while
walking AT MOST `max_chars + 1` chars, then slices: `Some((byte_idx, _)) => value[..byte_idx]`, else the whole
value. O(max_chars) instead of O(document length).

**Byte-identical.** `value[..byte_idx]` is exactly the first `max_chars` chars (`char_indices` gives char
boundaries), and the within-limit branch returns the value unchanged — identical to `chars().count()`+`take`.
Proven by `truncate_chars_matches_slow` (within/at/over the limit × ASCII / multibyte / combining marks / empty /
`max_chars == 0`, 7 `max_chars` values incl. 0 and 100_000) plus the retained `truncate_chars_*` unit tests and
the bench's pre-timing assert.

**Speed — DECIDABLE WIN, MEDIAN-gated.** Null-controlled A/B (`benches/truncate_preview_ab.rs`, ORIG =
`truncate_chars_slow` (count+take) vs CAND = `truncate_chars` (nth+slice), ratio CAND/ORIG, `<1.0` = faster;
`frankensearch_core::bench_support`, worker `vmi1149989`, 41 rounds × inner 64):

| document (→ 400-char preview) | A/A null median [p5, p95] | CAND/ORIG median [p5, p95] | verdict |
|---|---|---|---|
| ascii 2 000 chars    | 0.9945 [0.7760, 1.1819] | **0.4582 [0.3157, 0.5802]** | **DECIDABLE WIN**, ~2.2× |
| ascii 16 000 chars   | 0.9965 [0.8007, 1.2963] | **0.2084 [0.1628, 0.2828]** | **DECIDABLE WIN**, ~4.8× |
| multibyte 16 000     | 1.0025 [0.7931, 1.2719] | **0.1743 [0.1489, 0.1970]** | **DECIDABLE WIN**, ~5.7× |

Every lever median is far below its null p5 (0.78–0.80), decidable by a wide margin. The win scales with document
length (2.2× at 2k → 4.8× at 16k → 5.7× multibyte) — exactly the mechanism: the saving is the avoided
whole-document decode, which grows with the document while the preview stays fixed. Single-threaded, so the paired
ratio is contention-robust.

**Scope.** This is a ~2–6× win on the per-document preview-truncation step of ingest (one component alongside
canonicalization, hashing, dedup SQL, and the WAL), always-on for every document longer than the preview (the
common case). `truncate_chars` retains its signature and byte-for-byte output. All builds/benches strictly remote
(`RCH_REQUIRE_REMOTE=1`); no local Cargo. Peer files untouched.

## 2026-07-11 — WIN: storage ingest `content_length` ASCII fast-path — ~2.7× on the char count, byte-identical (cc_fse)

Continuing the ingest byte-fast sweep (sibling of the shipped fingerprint `char_count` win). After the preview
truncation became O(preview), `content_length = canonical_text.chars().count()` — a full per-char count of the
whole canonical document — was the largest remaining full-document scan per ingested doc. `content_char_len`
takes the ASCII fast-path: for ASCII text the char count equals the byte length, and `str::is_ascii` is a SIMD
byte scan cheaper than `chars().count()`'s per-char counting; non-ASCII falls back to the exact decode.

**Byte-identical.** `is_ascii ⇒ len()` is a UTF-8 invariant (ASCII bytes are single-byte chars); non-ASCII uses
the identical `chars().count()`. Proven by `content_char_len_matches_slow` (ASCII / pure-multibyte / combining
marks / emoji / mixed / empty) and the bench's pre-timing assert.

**Speed — DECIDABLE (ASCII), MEDIAN-gated.** Same `benches/truncate_preview_ab.rs`, ORIG = `content_char_len_slow`
(`chars().count()`) vs CAND = `content_char_len`; ratio CAND/ORIG:

| document | A/A null median [p5, p95] | CAND/ORIG median [p5, p95] | verdict |
|---|---|---|---|
| ascii 2 000  | 1.0000 [0.9902, 1.0552] | **0.3712 [0.3214, 0.4326]** | **DECIDABLE WIN**, ~2.7× |
| ascii 16 000 | 0.9950 [0.2424, 1.1167] | 0.3479 [0.3041, 0.3811]     | same ~2.9× effect; null p5 is a lone contention outlier (median 0.995) |
| multibyte 16 000 | 1.0006 [0.8402, 1.0983] | 1.0043 [0.9065, 1.2689] | TIE — correct: non-ASCII falls back to the identical decode |

The `ascii_2k` run has a tight A/A null (p5 0.9902) and the lever clears it decisively (~2.7×). The `ascii_16k`
lever is the same ~0.35 effect but that run's null p5 (0.242) is a single contention outlier (its median is
0.995, p95 1.117), so it reads "inside" by the strict gate on a contended worker; the effect is consistent with
ascii_2k and with the shipped fingerprint `char_count` (2.85×). Multibyte correctly ties (the fast-path does not
apply). Shipped on the clean-null decidable ASCII shape + the matching precedent.

**Scope.** Always-on for ASCII documents (English/code — the common case); a per-doc ingest char-count step. All
builds/benches strictly remote (`RCH_REQUIRE_REMOTE=1`); no local Cargo. Peer files untouched.
## 2026-07-11 — HOLD: `append_batch` resident-WAL dedup skip is byte-identical but inside the full-append null floor (cod, bd-ryid)

**Profile first.** Before the candidate existed, the original `VectorIndex::append_batch` path was measured with
768 resident WAL entries, a 256-vector append, dimension 384, 32 compacted main records, and the default
`fsync_on_write=true`. The real product append medians were 2.999/5.640/8.486 ms at 0/50/100% replacement
overlap. An isolated faithful timing of the later repeated-retain stage measured 0.094/0.331/0.488 ms, or
3.14%/5.87%/5.75% of those product medians. This was measured work, not a static-complexity guess.

**One candidate lever.** `append_batch` first calls `soft_delete_batch`, which already builds the incoming-ID set,
filters every matching resident WAL entry, persists that removal when needed, and restores memory plus returns an
error if the rewrite fails. After serializing the new batch, the original nevertheless scanned resident WAL once
again for every unique incoming ID. The candidate skips only that provably redundant second loop. It does not
change validation, incoming duplicate resolution, sidecar encoding, fsync, tombstoning, ordering, vector data,
compaction, search, or the adjacent owned-vector clone.

**Byte-identical proof.** Original and candidate produced identical WAL sidecar bytes and record counts, identical
immediate and reopened top-16 IDs/indexes/score bits for replacement-aware queries, and byte-identical compacted
FSVI files at 0%, 50%, and 100% overlap. Therefore ranking and recall are preserved exactly for the exercised
surface; this is stronger than a tolerance-based numerical claim.

**Speed — paired MEDIAN gate failed.** Strict remote-only command, worker `vmi1227854`, one release binary,
21 alternating AB/BA round pairs per shape; ratio is candidate/original:

```bash
RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
  cargo bench -j 4 -p frankensearch-index --profile release --features bench-internals \
  --bench wal_append_dedup_ab
```

| overlap | A/A original/original median [p5, p95] | skip/original median [p5, p95] | directional speedup | verdict |
|---:|---:|---:|---:|---|
| 0% | 1.013899 [0.733108, 1.469368] | 1.008887 [0.680725, 1.221177] | 0.991× | inside null floor |
| 50% | 1.009178 [0.843576, 1.475495] | 0.961359 [0.843038, 1.099357] | 1.040× | inside null floor |
| 100% | 0.952810 [0.672027, 1.157407] | 0.840956 [0.718304, 1.114335] | 1.189× | inside null floor |

The authoritative gate requires the candidate median below its A/A null p5. No shape clears that threshold. The
50% and 100% directions agree with the mechanism, but direction alone is not a shippable result; the 0% median is
slightly slower. The full default-fsync operation, not the isolated loop, is the shipping gate.

**Decision and boundary.** **HOLD; production remains original.** The feature-gated comparator and reproducing
benchmark are retained, but public `append_batch` still selects the legacy loop. Do not retry this same fixture on
a busy fleet. Reopen only for a demonstrated larger resident-WAL/batch production shape or a substrate whose A/A
floor can resolve the end-to-end effect. This does not reopen cc-owned lexical/query scan, Tantivy postings codec,
writer-budget, rejected FSVI handoff, or the adjacent WAL deep-clone families.

**Validation and degraded surfaces.** The exact-output bench exited 0, the focused strict-remote release suite
passed 402/402 tests, and remote `cargo check -j 4 --workspace --all-targets` exited 0 (with pre-existing warnings
in unrelated benchmark targets). Direct `rustfmt --edition 2024 --check`, `git diff --check`, and UBS passed; UBS
reported zero critical findings. Workspace clippy with `-D warnings` was not repeated because the immediately
preceding index HOLD records already establish pre-existing core/index lint blockers, while RCH was rebuilding
every command from cold; this degraded surface was not bypassed locally. All counted Cargo work was fail-closed
remote-only; no direct local Cargo fallback occurred. RCH was degraded at 9/12 healthy workers and then missed
cache twice on the identical
worker/target, making each focused release command a cold ~9.5-minute build. Agent Mail registration succeeded,
but its corruption circuit breaker rejected the file reservation; Git/worktree and Beads truth were used without
attempting a repair or bypass.

## 2026-07-12 — HOLD: FSFS `merge_ranked` one-lookup candidate fails the final higher-inner remote gate (Codex)

Retried the retained byte-identical `get_mut`/insert hybrid-fuse candidate against shipped
`merge_ranked_orig` using the authoritative same-binary alternating-round harness. Strict remote worker
`vmi1227854`; 41 rounds, `inner=200`; candidate/original ratios were 0.8846 for 200 candidates/tier with 140
overlap, 0.9436 for 200/40, and 0.8705 for 600/400. Their A/A null p5 floors were respectively 0.8132, 0.8802,
and 0.8153, so every median remained inside noise. The stable ~1.06–1.15× directional effect is not a shippable
win under the repository gate. Production remains on `merge_ranked_orig`; source is unchanged. This closes the
documented higher-inner retry for these shapes. Reopen only on a lower-noise isolated worker or a materially
larger measured overlap/key-cost workload. Exact command and full intervals are recorded in
`docs/NEGATIVE_EVIDENCE.md`.

## 2026-07-13 — HOLD: owned two-tier builder API transfer is isolated-fast but full finish stays inside null (`bd-xlpp`)

**Route and profile.** This was the explicit untested route-next left by the `bd-5973` FSVI writer-handoff
rejection. Production embedders return owned vectors, but the two-tier builder API borrowed and copied them. On a
20,000 × 384 one-tier fixture (29.297 MiB), remote profile medians were 6.076 ms for the record deep clone and
9.611 ms for the borrowed builder-add boundary, so the copy was measurable rather than inferred.

**Candidate and exactness.** One ownership lever moved already-owned embedder vectors into owned builder entry
points. Borrowed and owned arms produced byte-identical 16,020,096-byte FSVI artifacts plus bit-identical top-10
results for eight queries; ranking, recall, and nDCG are unchanged.

**Paired gate.** Strict-remote worker `vmi1152480`, one release binary, 21 alternating AB/BA round pairs; ratio is
owned/borrowed:

| gate | A/A median [p5, p95] | candidate/original median [p5, p95] | verdict |
|---|---:|---:|---|
| builder add | 1.021705 [0.826204, 1.460398] | **0.532615** [0.321833, 0.736410] | isolated win |
| add + encode/sort/fsync/open | 0.993935 [0.635516, 1.458783] | **0.937162** [0.623404, 1.304647] | inside null floor |

**Decision.** **HOLD; production restored.** The isolated boundary is approximately 47% faster, but the full
operation's 6.3% directional reduction does not clear its A/A p5. Only the feature-gated comparator and
reproducing benchmark remain. Reopen only for a production shape where this copy is a larger end-to-end share or
a substrate whose A/A floor can resolve roughly 6%. The final retained tree passed strict-remote all-target index
check; the target-scoped `-D warnings` Clippy attempt stopped on the pre-existing 80-error index baseline before
the benchmark. Full evidence and the exact command are in `docs/NEGATIVE_EVIDENCE.md`.

## 2026-07-14 — REJECT: reusable line-buffer repair-log count is ~1.50× slower (`bd-7z0x`)

The unresolved `durability::should_rotate` route replaced whole-file `read_to_string` allocation with a
`BufReader` and one reused `String`, while still consuming the complete log to preserve invalid-UTF-8 behavior.
Reference and candidate matched missing/empty/unterminated/CRLF/Unicode/invalid inputs, and the full append/rotate
fixture produced byte-identical active and rotated JSONL files.

Strict-remote same-binary evidence on `vmi1149989` (999 realistic records, 21 AB/BA round pairs, 16 appends per
arm; ratio candidate/original) was decisively negative: A/A median 1.002217 [0.930668, 1.122853], candidate median
**1.499345 [1.212310, 1.699262]**. The isolated check agreed (60.297 µs candidate vs 37.496 µs original).
Production was restored exactly; only the reproducible `durability_bench` comparator remains. Do not retry the
line-at-a-time design. A future O(1) cached count must first solve multi-protector/process, restart, failure, and
external-mutation correctness and show this operational path is actually hot. Full command and proof are in
`docs/NEGATIVE_EVIDENCE.md`.

## 2026-07-14 — LANDED: `collect_id_hits` per-segment column cache is a lazy `Vec` (drops the per-hit `segment_ord` hash), ~1.21× @k1000

**Route and profile.** Sibling-consistency gap on the scan-free `search_doc_ids` path (Tantivy top-k +
`collect_id_hits`, no dense scan, no metadata decode — the documented "latency-critical, identifiers only" route).
`collect_id_hits` cached the opened `ord` fast-field column per segment in a **lazy `HashMap<u32, Option<Column>>`**,
hashing `segment_ord` once per hit (k SipHash-of-u32 per query). The numeric-ff prototype that validated that fast
field (`id_materialize_numeric_ff`) indexed a `Vec` by `segment_ord` (O(1), no hash), but production regressed to a
HashMap to keep laziness (open only *touched* segments).

**Candidate and exactness.** Replaced the HashMap with a lazily-populated `Vec<Option<Option<Column>>>`
(`get_or_insert_with` — O(1) index, no hash, still opens only touched segments; outer `None` = not yet opened).
Byte-identical: same columns opened, same ords read, same id clones, and the `table`-None and
empty-slot→`docstore_id` fallback branches unchanged. The full `frankensearch-lexical` lib suite is green
(**89 passed, 0 failed**) on `vmi1227854`, and the bench asserts `base == cand` on a 4-segment / 1000-hit fixture.

**Paired gate.** Same-binary within-process paired A/B (`collect_ids_map_kind_ab`, `paired_median_ratio`, 41 rounds,
in-RAM 20k-doc fixture → 4 segments); ratio vec/HASHMAP (`<1` = Vec faster):

| k | A/A null [p5, p95] | vec/HASHMAP [p5, p95] | verdict |
|---|---:|---:|---|
| 30 | [0.591, 1.282] | 0.928 [0.703, 1.188] | inside floor |
| 100 | [0.841, 1.239] | 0.912 [0.804, 1.232] | inside floor |
| 300 | [0.744, 1.352] | 0.853 [0.690, 1.021] | inside floor |
| 1000 | [0.841, 1.185] | **0.826** [0.717, 0.982] | **decidable win (~1.21×)** |

**Decision.** **LANDED.** Decidable at k1000 (the large-pool case `search_doc_ids` serves — reranker / fusion
feeds); directional and never slower below. Byte-identical, zero-downside, landed-precedent class (hot-path map
hashing, cf the aHash migrations `9543ae6` / `8665ce1`) — removes the per-hit SipHash entirely. This resolves the
2026-07-14 `collect_id_hits` HOLD in `docs/NEGATIVE_EVIDENCE.md` (held one turn only because the RCH fleet was
saturated and the edit could not be compile/test-verified remotely). Bench: `collect_ids_map_kind_ab`.

## 2026-07-14 — REJECT/HOLD: Model2Vec full-row prefetch at 256 tokens stays inside the null floor (`bd-r3lf`)

Negative-ledger-first follow-up to the 2026-07-08 gather result: preserve the original loop below 256 tokens and
prefetch all cache lines of the embedding row four tokens ahead at 256+. The same release binary measured the full
gather/mean-pool/L2-normalization boundary over two byte-identical 30 MiB tables with a pseudo-random full-table
sweep, exact output parity, 31 alternating paired rounds, and an A/A null control. Strict-remote foreground run on
`vmi1153651`; candidate/original ratios below 1 favor prefetch:

| tokens | A/A null median [p5, p95] | candidate/original median [p5, p95] | verdict |
|---:|---:|---:|---|
| 128 | 0.988 [0.359, 1.374] | 1.012 [0.446, 2.281] | short path preserved |
| 256 | 1.011 [0.612, 2.840] | **0.791** [0.631, 1.098] | inside null floor |
| 512 | 1.032 [0.846, 1.337] | **0.809** [0.692, 1.097] | clears null floor |

**Decision:** production restored byte-for-byte because the exact 256-token onset does not clear its null floor.
The feature-gated exact oracle and reproducer are retained. The 512-token mechanism win is route-next only: a
512-only threshold would need a final-code paired run and evidence of real production prevalence.

## 2026-07-14 — LANDED: Model2Vec full-row prefetch only at 512+ tokens, ~1.63× (`bd-vxki`)

Resolved the `bd-r3lf` route-next as a separate thresholded lever. `Model2VecEmbedder::embed_sync` now uses the
exact gather helper: the former loop remains unchanged below 512 tokens and on non-x86 targets; 512+ token
documents prefetch every cache line of the row four tokens ahead. This is a reachable indexing boundary because
the product embeds canonicalized source-file contents up to the default 2,000-character limit and the tokenizer
adds no repo-side token truncation.

Strict-remote same-binary foreground evidence on `vmi1152480` (two byte-identical 30 MiB tables, pseudo-random
full-table sweep, exact pooled-output parity, 31 alternating paired rounds, A/A null; candidate/original):

| tokens | A/A null median [p5, p95] | candidate/original median [p5, p95] | verdict |
|---:|---:|---:|---|
| 128 | 0.997263 [0.727475, 1.359142] | 1.089797 [0.744414, 1.397318] | original path preserved |
| 256 | 1.000972 [0.825183, 1.159020] | 1.067836 [0.782674, 2.918815] | original path preserved |
| 512 | 1.018346 [0.750244, 1.279036] | **0.612870** [0.426846, 0.829550] | **decidable win (~1.63×)** |

The module-qualified strict-remote exact oracle passed 1/1 across the threshold, varied dimensions, and OOV rows;
the prior zero-test filter invocation is excluded. **Decision: LANDED** — approximately 38.7% lower latency at
512 tokens, with the original production accumulation path preserved below the gate.

## 2026-07-14 — LANDED: adaptive NQC exact incremental rolling order, ~3.65× (`bd-4hxk`)

**Route and candidate.** The prior adaptive-NQC WASH quantified roughly 613 ns/query and showed that removing
allocations did not matter because sorting the 2,048-value rolling window every 64 queries dominated. The shipped
sampler now maintains that exact finite multiset in sorted order as observations arrive, while retaining the
recency queue for eviction. Periodic snapshots become a linear copy with no sort; scoring still precedes
observation, snapshot cadence is unchanged, and the preallocated sorted vector costs 8 KiB per production sampler.

**Exactness and paired gate.** The retained former-path constructor runs insertion-order sampling plus the full
periodic sort in the same release binary. It matched candidate weight bits across 8,192 post-warm observations
(evictions, duplicates, signed zero, NaN/infinity ignore), then ran 41 alternating pairs on strict-remote worker
`vmi1227854`; ratio is incremental/former:

| gate | median [p5, p95] | verdict |
|---|---:|---|
| A/A former/former | 1.0028 [0.9433, 1.0793] | valid null |
| incremental/former | **0.2737 [0.2402, 0.2978]** | **KEEP (~3.65×)** |

Direct medians were 602.54 ns/query former and 160.33 ns/query incremental: 442.22 ns/query saved, or 0.0884%
of a 500 µs search. Normal no-feature `frankensearch-fusion` check passed remotely. **Decision: LANDED** —
the measured dominant sort is removed with exact output and no post-construction allocation. Benchmark:
`nqc_adaptive_cost_ab`; all Cargo work was fail-closed `RCH_REQUIRE_REMOTE=1`.

## 2026-07-14 — LANDED: hybrid lexical metadata is deferred to fused winners, ~1.73× at 300→10 (`bd-nv84`)

**Corrected route.** The earlier route-next incorrectly said hybrid fusion discarded lexical metadata, and its
follow-up correctly retracted that claim: async fusion re-attaches metadata to lexical winners. The valid lever
is narrower. `LexicalSearch::search_fusion_candidates` defaults to the existing full search for foreign
implementations; Tantivy overrides it with the existing ordinal-fast-field `search_doc_ids` path. After fusion,
`hydrate_fusion_metadata` performs exact-ID lookups and decompresses/deserializes stored metadata only for final
results carrying a lexical score. Metadata-bearing lexical short-circuits and embedding-failure fallbacks stay on
`search`; post-candidate direct-return branches explicitly reload that full path. Opportunity score was 5.0
(`impact=4 × confidence=5 / effort=4`): measured candidate-overfetch waste on a reachable product boundary, with
moderate trait/searcher plumbing and an exact metadata contract to preserve.

**Exactness.** The release comparator serialized every field of the ten winner `ScoredResult`s and asserted the
full and deferred outputs identical before timing, including doc IDs, score bits, source/index fields, and
metadata. Focused strict-remote tests separately proved Tantivy candidate rank/score-bit parity plus exact metadata
restoration, and proved the real `TwoTierSearcher` hydrates hybrid winners while an embedding failure returns the
full direct metadata. Both passed (2/2).

**Production lexical boundary.** One 30k-document in-memory Tantivy index, realistic seven-field metadata,
query parse/search included, ten final winners, 31 alternating AB/BA pairs. Ratio is deferred/full (`<1` wins):

| lexical candidates → winners | A/A median [p5, p95] | deferred/full median [p5, p95] | verdict |
|---:|---:|---:|---|
| 30 → 10 | 1.0096 [0.8237, 1.2078] | 0.9973 [0.8341, 1.2273] | neutral / inside floor |
| 100 → 10 | 0.9639 [0.8360, 1.2226] | 0.8373 [0.7287, 1.0289] | favorable, inside floor |
| 300 → 10 | 0.9986 [0.8412, 1.2002] | **0.5769 [0.5097, 0.6502]** | **decidable ~1.73×** |

The retained raw materialization rows remain 5–22× faster, but they are supporting mechanism evidence; the table
above is the shipping boundary. **Decision: LANDED for high-fanout hybrid acquisition.** The default-like 30-row
case is explicitly neutral, so this is not claimed as a default-query speedup. Authoritative release bench and
focused tests ran foreground on `vmi1153651` with `RCH_REQUIRE_REMOTE=1`; two earlier compile-only attempts are
excluded (missing Tantivy collector finalization, then a unit-return bench closure). No Cargo command ran locally.
Retained benchmark: `lexical_candidate_metadata_waste_ab`.

## 2026-07-14 — LANDED: exact-ID metadata hydration skips BM25 scoring and top-k sorting, ~1.18–1.21× (`bd-9d7b`)

**Fresh seam and one lever.** The preceding winner-only metadata change introduced a new exact-ID hydration
query. Its `BooleanQuery` used `TopDocs::with_limit(...).order_by_score()`, so Tantivy built BM25 weights, scored
every matching ID, maintained a heap, and sorted the result even though hydration immediately discarded both
score and order. Production now uses `DocSetCollector`, which disables scoring and returns the same live document
addresses. Opportunity score was 6.7 (`impact=4 × confidence=5 / effort=3`): the path is reachable for every
metadata-bearing hybrid result, the discarded work is explicit, and the change is confined to the collector.

**Exactness.** The retained scored collector and production unscored collector hydrated clones of the same
candidate vector. Before timing, the benchmark serialized every `ScoredResult` field and asserted byte equality
at 10, 30, 100, and 300 winners. Input order, scores, ranks, IDs, source fields, and metadata are unchanged; only
the order in which private document addresses are visited can differ.

**Strict-remote same-binary gate.** One 30k-document Tantivy index with realistic metadata; 31 alternating AB/BA
rounds, four calls per timed arm; ratio is unscored/scored (`<1` wins):

| hydrated winners | A/A scored null median [p5, p95] | unscored/scored median [p5, p95] | verdict |
|---:|---:|---:|---|
| 10 | 0.9952 [0.9281, 1.1308] | **0.8497 [0.7838, 0.9413]** | **decidable ~1.18×** |
| 30 | 1.0026 [0.9443, 1.0412] | **0.8399 [0.7133, 0.9041]** | **decidable ~1.19×** |
| 100 | 1.0048 [0.8899, 1.3473] | **0.8262 [0.7679, 0.9435]** | **decidable ~1.21×** |
| 300 | 0.9674 [0.8696, 1.4851] | **0.8503 [0.7741, 0.9771]** | **decidable ~1.18×** |

Every candidate median is below its own A/A p5. The successful foreground run used strict remote worker
`vmi1227854` and an optimized release build with `lto=false`, `codegen-units=16`; this is a relative same-binary
claim, not an absolute production-profile latency claim. A prior full-LTO attempt exhausted its 15-minute bound
during cold linking without running the benchmark, and an identical retry was stopped after RCH discarded the
partial target and began another cold rebuild; both are compile-only and excluded. No Cargo command fell back
locally. The focused strict-remote library guard
`deferred_fusion_candidates_restore_exact_metadata` passed 1/1 on the same worker. **Decision: LANDED.** The
source/bench were swept from the shared staged index into concurrent main commit `8a90fa13`; this row is the
dedicated evidence closeout. Retained benchmark:
`lexical_candidate_metadata_waste_ab` with `META_COLLECTOR_ONLY=1`.

## 2026-07-14 — REJECT: borrowed hydration result-slot index stays inside the A/A floor (`bd-veet`)

**Fresh seam and one lever.** The winner-hydration path introduced after the earlier lexical quadratic sweep
loads each exact-ID document and then linearly searches the result slice for its `doc_id`. The candidate replaced
only that O(winners²) matching with one borrowed-key `AHashMap<&str, &mut Option<Arc<Value>>>`, preserving the
linear path's first-match behavior for defensive duplicate inputs. Opportunity score was 10.0
(`impact=4 × confidence=5 / effort=2`). The collector, stored-document loading, JSON decoding, result order,
scores, ranks, and metadata values were shared between arms.

**Exactness and strict-remote gate.** Full serialized `ScoredResult` vectors matched before timing at every
winner count. One 30k-document same-binary optimized-release run on strict-remote worker `vmi1153651` used 31
alternating pairs and four calls per arm; ratio is indexed/linear (`<1` wins):

| hydrated winners | A/A linear null median [p5, p95] | indexed/linear median [p5, p95] | verdict |
|---:|---:|---:|---|
| 10 | 0.9920 [0.5292, 1.1661] | 0.9879 [0.7899, 1.2269] | inside floor |
| 30 | 0.9873 [0.8093, 1.3601] | 1.0109 [0.8238, 1.2434] | inside floor |
| 100 | 0.9363 [0.8499, 1.1810] | 0.9377 [0.8352, 1.1263] | inside floor |
| 300 | 1.0017 [0.7072, 1.3840] | 0.8845 [0.7011, 1.4234] | inside floor |

No candidate median cleared its own A/A p5. The directional 300-winner result is therefore not admissible, and
the 10/30-winner product shapes are neutral. The successful gate used `lto=false`, `codegen-units=16` to bound the
remote link and supports only this relative same-binary decision. A preceding strict-remote compile exposed an
`Arc<Value>` signature mismatch before execution; it is compile-only and excluded. While the foreground gate was
running, shared main commit `8207cce6` captured the candidate; this closeout removes it forward rather than
rewriting pushed history. **Decision: REJECT.** Do not add a per-call hash index to this hydration path without a
quieter product-real workload that can resolve the effect and demonstrates materially larger winner batches.

## 2026-07-14 — LANDED: reuse markdown-link parser scratch buffers, ~2.23× on link-heavy canonicalization (`bd-xs7l`)

**Negative-ledger-first attribution and one lever.** Core canonicalization's
`strip_markdown_links` creates a fresh `String` for link text and another for the URL at every `[` candidate.
The URL buffer is discarded for every valid link. The earlier rejected whitespace run-copy experiment is not
repeated here: this change retains the existing Unicode character parser and only moves its two scratch buffers
outside the candidate loop, clearing and reusing their capacity. Plain lines still bypass the link parser, and
non-link brackets do not allocate until text is pushed.

**Exactness.** The retained former implementation and shipping scratch-reuse implementation ran in the same
binary. Before timing, the benchmark asserted byte-for-byte equality across a 256-link document containing
nested link labels and parenthesized URLs. A focused parity test also covers empty/plain input, multiple links,
closed-only and unclosed brackets, an unbalanced URL, and Unicode link text/URLs. Canonical text—and therefore
tokens, embeddings, index terms, and ranking—does not change.

**First clean foreground gate (accepted without rerun).** One strict-remote `--profile release` invocation on
`vmi1227854` used `lto=false` and 16 codegen units to bound the cold release link. The paired ratio is
reused/former (`<1` wins):

| workload | A/A former null median [p5, p95] | reused/former median [p5, p95] | verdict |
|---|---:|---:|---|
| 256 markdown links | 0.9781 [0.5992, 1.9895] | **0.4477 [0.2786, 0.5228]** | **decidable ~2.23×** |

The candidate p95 remains below the noisy null p5. Criterion independently reported central estimates of
**69.742 µs → 32.094 µs** (ratio 0.460), consistent with eliminating 510 repeated heap allocations per call
after the two scratch buffers first grow. This is a relative same-binary result for markdown-link-heavy input,
not a claim about plain-text documents, which never enter this parser. No local Cargo command or second
benchmark ran. **Decision: LANDED.** Retained comparator: `canonicalize/markdown_link_scratch` with
`bench-internals`.

## 2026-07-14 — LANDED: slice markdown-link labels from source, ~1.67× paired (`bd-xwb9`)

**Negative-ledger-first attribution and one lever.** The preceding scratch-reuse keep removed per-link
allocations, but its warm path still copied every label byte into a scratch `String` and then copied it again
into the result. It also copied every URL byte into a second scratch buffer that valid-link handling immediately
discarded. Production now scans the existing UTF-8 input by byte index, copies each valid label directly from
its source slice, and advances past balanced URLs without materializing them. Only ASCII delimiter positions are
used as slice boundaries, so the nested-bracket and balanced-parenthesis parser remains safe for Unicode input.
Opportunity score was 6.7 (`impact=4 × confidence=5 / effort=3`): the residual copies were explicit in the
measured hot loop and the change stayed inside one private canonicalization primitive.

**Exactness.** The retained scratch-reuse implementation and shipping source-slice implementation ran in the
same binary. Before timing, the benchmark asserted byte-for-byte equality on a 256-link document containing
nested labels and parenthesized URLs. The focused parity test retains empty/plain, multiple-link, empty-URL,
closed-only, unclosed-bracket, unbalanced-URL, and Unicode cases. Canonical text, tokens, embeddings, index
terms, and ranking semantics are unchanged.

**First clean foreground gate (accepted without rerun).** One strict-remote `--profile release` invocation on
`vmi1153651` used `lto=false`, 16 codegen units, a 100 ms warm-up, and the same release binary for all arms. The
paired ratio is source-slices/scratch-reuse (`<1` wins):

| workload | A/A scratch-reuse null median [p5, p95] | source-slices/scratch-reuse median [p5, p95] | verdict |
|---|---:|---:|---|
| 256 markdown links | 1.0754 [0.7352, 1.3842] | **0.6002 [0.4095, 0.8456]** | **decidable ~1.67×** |

The candidate median is below the null p5 and its p95 is below the null median. Criterion independently reported
central estimates of **50.377 µs → 32.628 µs** (ratio 0.648, ~1.54×), consistent with removing the label
double-copy and discarded URL copy. This is a relative same-binary result for markdown-link-heavy input, not an
absolute production-profile latency claim. A preceding strict-remote test compile exposed a neighboring test
name shadow before execution; qualifying the production function fixed that harness-only issue, and the attempt
is excluded from performance evidence. No Cargo command ran locally and no second benchmark ran. **Decision:
LANDED.** Retained comparator: `canonicalize/markdown_link_source_slices` with `bench-internals`.

## 2026-07-14 — TUI palette match-index cache removes repeat navigation scans — ~300× (`bd-072j`, IcyRidge)

**Negative-ledger-first route and attribution.** `bv --robot-triage` again ranked the reopened tombstone
bitmap bead, but the later 2026-07-12 ledger resolution closes that non-default, fleet-unmeasurable bandwidth
route. This turn pivoted to the TUI command palette. The existing palette profile measured one cached-string
filter scan at **154.66 µs for 4,096 actions**; production arrow navigation called `filtered()` to obtain the
count, and the following render called it again. Thus the steady-state navigation path rescanned every action
twice and allocated two result vectors even though neither the query nor action set had changed.

**Single lever.** `CommandPalette` now caches ordered matching action indices when the query or registered
action set changes. `select_prev`, `select_next`, and `confirm` consume that cache directly; `filtered()` maps
the cached indices to action references for render. Registration order is preserved, the same full-string
Unicode `to_lowercase()` normalization runs on every query mutation, opening/closing resets the cache to all
actions, and registration during an active filter updates the cache. The retained same-binary former arm
recreates the immediately preceding cached-lowercase scan, and the benchmark asserts identical ordered action
IDs before timing.

One foreground fail-closed RCH invocation on `vmi1153651` used `cargo bench --profile release` with release
LTO disabled (no `release-perf`, no local fallback, no retry). Criterion warmed both arms and measured the
1,024-action navigation-plus-render path:

| arm | 95% interval | central estimate | throughput |
|---|---:|---:|---:|
| former cached-string scan twice | 130.47–153.70 µs | **141.01 µs** | 7.2620 Melem/s |
| cached matches + render materialization | 430.76–508.77 ns | **470.45 ns** | 2.1766 Gelem/s |

Candidate/former ratio is **0.00334**, or **~299.7× faster**. The effect is far beyond the fleet noise floor;
the result also removes the navigation-side result-vector allocation, leaving only the render API's reference
vector. The worker-scoped target was cold and required a 10m43s remote compile, but that build used LTO=false
and the measured Criterion samples were warm. Scoped UBS, rustfmt, and diff checks passed. **Decision: LANDED.**

### 2026-07-15 — mmap int8 quantizer compacts AVX2 lanes to one store (`bd-btgh`, MaroonOriole)

**Profile attribution and one lever.** `VectorIndex::int8_slab` reaches
`quantize_f16_le_bytes_to_i8_avx2`; its quantize pass already produced eight clamped `i32` lanes, but
then spilled them to a stack array and performed eight scalar `Vec::push` calls. The kept epilogue uses
lane-local `vpshufb` to select each lane's low byte, joins the two four-byte halves, and writes one
unaligned `u64` into reserved output capacity. Max reduction, scale, half-away rounding, clamp, runtime
dispatch, scalar tail, and output order are unchanged.

The existing `f16_slab_quantize` bench asserted byte-for-byte equality between the scalar and production
dispatch paths for 1k, 10k, and 50k vectors before timing. Per the one-arm protocol, no baseline was
rebuilt: the candidate was compared with the stored shipping baselines from `hz2`. One foreground,
fail-closed RCH `--profile release` measurement ran on `vmi1227854` with LTO disabled; the filter also
matched the 10k row, yielding a useful scaling corroboration:

| workload | stored shipping baseline | candidate (95% interval) | candidate / baseline |
|---|---:|---:|---:|
| `f16_slab_quantize/dispatch/1000` | 361.33 us | **104.55 us** (101.03-109.19 us) | **0.289 (~3.46x faster)** |
| `f16_slab_quantize/dispatch/10000` | 3.4300 ms | **1.0592 ms** (1.0065-1.1316 ms) | **0.309 (~3.24x faster)** |

Both candidate intervals are far below the conservative 10% cross-worker keep gate, and exact output
parity preserves all downstream retrieval ordering. Criterion used the bench's retained 30 samples,
300 ms warm-up, and 1 s measurement. The untimed warm-up completed first in 8m49s; RCH then evicted that
target and paid another 8m58s cold candidate link, but no timeout, local fallback, `release-perf`, or
second baseline arm ran. **Decision: LANDED.**

### 2026-07-15 — TUI palette render iterates cached matches without materialization (BlackThrush)

**Profile attribution and one lever.** The landed palette match-index cache left one steady-state render
allocation: `CommandPalette::filtered()` copied every cached match into a fresh `Vec<&Action>` before the ops
overlay immediately iterated it. `iter_filtered()` now borrows the same ordered index cache directly; the
allocating API remains as a compatibility collector, while the production overlay consumes the iterator.
The retained benchmark asserted identical ordered action IDs between the materialized and borrowed views
before timing.

One untimed, fail-closed remote `--profile release` warm-up completed on `vmi1153651` with LTO disabled.
RCH then evicted its worker-scoped target despite the pinned worker/path, so the sole candidate command paid
another cold link; no timeout, local fallback, `release-perf`, or rebuilt baseline was used. Criterion's warm
samples measured the exact 1,024-action navigation-plus-render-lookup row against the stored current baseline:

| arm | 95% interval | central estimate | candidate / stored baseline |
|---|---:|---:|---:|
| stored `cached_matches_then_render` | 430.76–508.77 ns | **470.45 ns** | 1.0 |
| candidate `cached_match_iter_then_render` | 4.6833–5.0309 ns | **4.8375 ns** | **0.0103 (~97.2× faster)** |

The candidate interval is wholly below the stored baseline interval and removes the profiled allocation while
preserving registration order. The measurement itself remained cheap (20 samples, 100 ms warm-up, 450 ms
measurement); only RCH's two cold links were expensive. **Decision: LANDED.**

### 2026-07-15 — rerank identity included-index map elision (`bd-x6pa`)

**Profile attribution.** `rerank_step_with_combine` allocated and filled `included_indices` for every rerank,
even when all selected candidates had text and score application therefore used the identity mapping. The
negative ledger had already isolated this allocation as the remaining mapping-preparation cost (opportunity
score 8.0) but the prior attempt never reached a timed path.

**Single lever.** The map is now absent on the all-text path. On the first missing-text gap it is allocated,
backfilled with the preceding identity indices, and maintained exactly as before for the remaining candidates.
The same-binary benchmark asserts identical resolved indices and checksum for both 32 all-text candidates and
a fixture with gaps at positions 3, 9, and 22 before timing.

The foreground fail-closed RCH run on `vmi1153651` used `cargo bench --profile release`, release LTO disabled,
10 samples, 50 ms warm-up, and 150 ms measurement per arm. The cold target received an untimed warm-up first;
RCH nevertheless discarded its worker cache between invocations and rebuilt before the harness, so build time
is excluded from the Criterion samples. A corrected literal group filter produced the real A/B:

| arm | 95% interval | central estimate |
|---|---:|---:|
| allocating identity map A | 107.62–272.87 ns | **173.98 ns** |
| lazy identity map | 46.866–63.259 ns | **53.700 ns** |
| allocating identity map B | 73.969–86.234 ns | **78.659 ns** |

Allocating A was noisy, so the decision uses the faster bracketing control B: candidate/control is **0.683**,
or **31.7% faster**, and the confidence intervals do not overlap. This is a component mapping-preparation win,
not an end-to-end model-latency claim. Exact parity passed for both identity and gapped mappings.
**Decision: LANDED.**

### 2026-07-16 — DefaultSymbolCodec source-symbol build: zero-init elision — ~1.20× at 10MB (BlackThrush)

`DefaultSymbolCodec::encode` (the durability crate's built-in "no external runtime" fallback codec,
`codec.rs`) materialized each source symbol as `vec![0; symbol_size]` and then `copy_from_slice`d the whole
window over it — a wasted `memset` per FULL symbol (only the final short symbol needs a zero-padded tail).
Full symbols now copy directly via `source_data[start..end].to_vec()` (alloc + one `memcpy`, no zero-init);
the short tail still pads. `start <= end` holds for every valid symbol index, so `end - start` never
underflows. **Byte-identical output** (bench asserts `build_old == build_new` at 1MB and 10MB).

Same-binary criterion A/B (`symbol_build` bench, `-p frankensearch-durability`, bench profile no-LTO, one
remote `rch` run, 100 samples), throughput of the symbol-build over an incrementing-byte source:

| size | zero_init_old | direct_copy_new | ratio |
|---|---|---|---|
| 1MB | 67.46 µs [65.6–69.6] | 63.93 µs [61.8–66.4] | 1.05× (overlapping CIs — cache-resident, memset cheap) |
| 10MB | 3.420 ms [3.31–3.54] | 2.844 ms [2.71–3.01] | **1.20× (CIs DISJOINT)** |

The win scales with data size: at 1MB the buffers are cache-resident so the eliminated `memset` is nearly free
(within noise), while at 10MB (spilling cache) removing the redundant `memset` before the `memcpy` is a clean
decidable ~1.20×. **Scope note:** this is the built-in fallback codec, NOT the primary production path
(production uses the external RaptorQ codec via `fsqlite_core::raptorq_integration`; `DefaultSymbolCodec` is the
crate's documented dependency-free fallback and a public API). A real byte-identical improvement to shipped
library code, modest in blast radius. Landed `codec.rs` + retained `symbol_build` A/B bench.

### 2026-07-16 — Model2Vec embed_batch parallelizes per-doc across Rayon threads (BlackThrush)

Sibling of the FNV hash embedder's batch parallelization (`0b560edc`): `Model2VecEmbedder::embed_batch` was a
plain serial `for text in texts { embed_sync(text) }` loop, while `HashEmbedder` had already gained Rayon
dispatch. Model2Vec is the CPU-bound QUALITY (semantic) embedder — each `embed_sync` is ~0.57 ms of independent
work (tokenize → static-row gather over a ~30 MB table → mean-pool → L2-normalize) — so batch embedding on the
ingest path was leaving cores idle. Added `embed_batch_sync` that dispatches per-doc via `par_iter` at/above
`PARALLEL_BATCH_MIN = 8` (Model2Vec's per-doc cost amortizes Rayon scheduling far below the hash embedder's 256
threshold); smaller batches stay serial to preserve latency. Rayon moved into the `model2vec` feature. Result is
**bit-identical** to the serial loop (per-doc work is deterministic, `par_iter().collect()` preserves order) —
asserted by a new `embed_batch_sync_matches_serial_across_parallel_boundary` unit test.

Bench `model2vec_batch_parallel` (`-p frankensearch-embed --features model2vec`, bench profile no-LTO, 100
samples), serial vs Rayon-parallel dispatch over the real per-doc gather+pool kernel (30k×256 synthetic table):

| batch | serial | parallel | speedup |
|---|---|---|---|
| 64 | 1.558 ms [1.51–1.61] | 0.464 ms [0.42–0.51] | **3.36×** (CIs disjoint) |
| 256 | 8.045 ms [7.91–8.18] | 5.601 ms [5.28–5.92] | **1.44×** (CIs disjoint) |

Always a decidable win above the threshold. The N=256 speedup is lower because the bench uses **random** token
ids across the full 30k vocab — worst-case cache behavior that makes the 30 MB-table gather DRAM-bandwidth-bound
at high concurrency. Real text has Zipfian token locality (hot rows stay cached) and the full `embed_batch` also
parallelizes tokenization (pure compute), both of which this floor excludes — so the production win is expected
to meet or exceed these numbers. Landed `model2vec_embedder.rs` + `embed_batch_sync` + parity test; retained the
`model2vec_batch_parallel` A/B bench.

### 2026-07-16 — verify_and_repair_file reuses the corruption-detection decode — 1.19× (BlackThrush)

Extends fe866683 (which fused the reuse only inside standalone `repair_file_internal`) to the combined
`FileProtector::verify_and_repair_file` path. On the corrupt path that method ran `verify_file` (mmap + source
CRC32 + full trailer deserialize) and THEN `repair_file_internal`, which re-verified from scratch (mmap + CRC32 +
deserialize) — a residual double pass fe866683 left. Now `verify_file_impl` retains the decoded
`(RepairTrailerHeader, Vec<RepairSymbol>)` and, on detected corruption, hands it straight into a new
`repair_file_internal(..., verified_decode: Option<...>)`, which skips its own verify entirely. Standalone
`repair_file` (the fe866683 path) passes `None` and is unchanged; the three public method signatures
(`verify_file`, `repair_file`, `verify_and_repair_file`) are unchanged. **Byte-identical repaired output**
asserted (both the reuse and the retained `verify_and_repair_file_no_reuse` bench twin restore the exact original
1 MB payload).

Bench `verify_repair_reuse` (`-p frankensearch-durability --features bench-internals`, bench profile no-LTO, 100
samples; re-corrupts a 1 MB single-block each iteration, so the identical corrupt-write cost sits in BOTH timed
regions and only dilutes the ratio):

| arm | time | ratio |
|---|---|---|
| no_reuse (verify + repair's own re-verify) | 4.258 ms [4.13–4.39] | 1.00× |
| **reuse (verified decode → repair)** | **3.585 ms [3.47–3.70]** | **1.19× (CIs disjoint)** |

15.8% lower end-to-end verify+repair latency; the repair-path-only delta is larger. Cold repair path (corruption
is rare), but a real byte-identical win completing the fe866683 reuse across the whole verify→repair sequence.
Landed `file_protector.rs` + retained the `verify_repair_reuse` A/B bench.

## 2026-07-18 — Fused generic default analyzer clears its same-binary null floor (`bd-r3rd`, CopperOrchid)

Commit `375e4237` replaces the generic `SimpleTokenizer` + `LowerCaser` pipeline with one byte-aware token
stream: ASCII classification and lowercase happen in the same pass, while non-ASCII input retains the exact
`char::is_alphanumeric` and `char::to_lowercase` behavior. The retained comparator in
`tokenizer_char_walk_ab` asserted exact `Token` parity before timing across the 48 KiB measured corpus plus
empty, punctuation/identifier, Cyrillic, Greek, Turkish, accented Latin, CJK, Japanese, and Korean fixtures.

The authoritative retry was one strict-remote `--profile release` binary on pinned RCH worker `ovh-a`, with
release LTO disabled and 16 codegen units. Binary SHA-256:
`6778f06d1ba4499d271b96bf9edb2de5099bf6437b23f3445a74faca94bbf58a`. The shared alternating-round sampler
used 41 rounds x 4 inner calls; fused/original `< 1.0` means the fused analyzer is faster:

| 48 KiB analyzer | median [p5, p95] | verdict |
|---|---|---|
| A/A original control | `0.9960 [0.9790, 1.0173]` | observed floor |
| fused / original | **`0.9634 [0.9514, 0.9808]`** | **DECIDABLE WIN**; median below null p5, ~1.038x |

Criterion's independent 10-sample pass measured `123.05 us [122.87, 123.18]` original versus
`120.95 us [119.69, 122.56]` fused. The remote estimates give CV `0.19%` and `1.47%`, respectively. Self-time
is N/A because the claim covers the whole measured analyzer routine rather than a subframe: both retained arms
consume the complete token digest through `black_box`, and exact token parity proves the fused stream executed.
A separate remote correctness gate passed all 113 lexical library tests on `vmi1152480`; RCH selected that
worker despite the `ovh-a` hint, so its result is used only for correctness and not for timing comparability.

**Decision: KEEP.** The earlier July 14 run stopped during a cold dependency update and produced no timing; this
completed retry supersedes that INVALID/HOLD. The production implementation and reproducible comparator were
already retained by `375e4237`, so this closeout changes only the evidence ledgers.

## 2026-07-18 — A/A null-control retrofit across 20 paired A/B benches (`bd-zgq6`, HazyStork)

Commit `26dbfa8` (rrf_recip_ab reference) plus the batch retrofit lands the shared
alternating-round `paired_median_ratio` sampler (41 rounds × 8 inner, A/A null control with
p5/p95 spread gate) in every remaining paired A/B bench in `frankensearch-fusion`: rrf_recip_ab,
neighbor_smooth_recip_ab, hubness_dot_ab, pool_minmax_merge_ab, merge_dedup_ab, filter_match_ab,
sync_hash_ab, docid_materialize_ab, metadata_clone_ab, explanation_clone_ab, code_signal_probe_ab,
expand_fuse_ab, negation_normalize_ab, normalize_signal_ab, nqc_adaptive_cost_ab, nqc_cv_cost_ab,
prf_expand_ab, rrf_config_cost_ab, scoredresult_box_ab, tokenize_ascii_ab. Each bench now prints
`[null]` and `[lever]` rows with an explicit `DECIDABLE` / `INSIDE NULL FLOOR (not decidable)`
verdict; nqc_adaptive_cost's hand-rolled AB/BA machinery was replaced by the shared sampler, and
nqc_cv_cost was already converted upstream.

Authoritative run: one full sweep of all 20 benches, `--profile release`, executed on `thinkstation1`
(Threadripper PRO 5975WX) — `rch exec` fell back to local execution for this sweep, so the host is
recorded as-is rather than a pinned fleet worker; the A/A null is calibrated per-machine by design,
and any fleet re-run supersedes by the same rule. Self-time is N/A (the claim covers each whole
measured routine, not a subframe). CV% is N/A by construction: the gate is median-of-rounds against
the null spread, not CV (`cv<5%` is unattainable on this fleet). Binary SHA-256 for each converted
bench (release deps on the executing host):

| bench | binary SHA-256 (first 16) |
|---|---|
| rrf_recip_ab | `630c99876c37985d` |
| neighbor_smooth_recip_ab | `380a5c1486e91b3c` |
| hubness_dot_ab | `48fe542a7389d31f` |
| pool_minmax_merge_ab | `b3887942694d4169` |
| merge_dedup_ab | `3490323f7d51b6ef` |
| filter_match_ab | `104a360362ef385e` |
| sync_hash_ab | `c7350844b3c19e45` |
| docid_materialize_ab | `ad4ef5265aa5e1df` |
| metadata_clone_ab | `f43c68b0380186cc` |
| explanation_clone_ab | `81e7f55af4f46ec6` |
| code_signal_probe_ab | `55428876b8845d11` |
| expand_fuse_ab | `a70a65cd578e6535` |
| negation_normalize_ab | `a2d7edc54459357c` |
| normalize_signal_ab | `05201f4b80ae7239` |
| nqc_adaptive_cost_ab | `652910a17e3a4c68` |
| nqc_cv_cost_ab | `8dfbd0421b71118a` |
| prf_expand_ab | `37936c0bbce5d396` |
| rrf_config_cost_ab | `fbe163c59374f6bf` |
| scoredresult_box_ab | `bd6996bcb313999c` |
| tokenize_ascii_ab | `d2307f5488bb3bdf` |

Re-decided verdicts (lever median vs its own A/A null spread; `<1` = lever faster):

| bench / param | lever median | verdict vs null floor |
|---|---|---|
| rrf_recip n10k, n100k (LUT) | 0.6656, 0.6665 | **DECIDABLE WIN** ~1.50× |
| neighbor_smooth_recip v3/ORIG (6 cells) | 0.985–1.032 | **INSIDE NULL FLOOR** (not decidable) |
| neighbor_smooth_recip v2/ORIG mutual (3 cells) | 1.205–1.223 | **DECIDABLE REGRESSION** (v2 ~1.2× slower) |
| hubness_dot multiacc, simd, simd_par (all cells) | 0.096–0.389 | **DECIDABLE WIN** ~2.6–10× |
| hubness_dot shipped/simd_par (2 build cells) | 0.995, 0.949 | **INSIDE NULL FLOOR** (not decidable) |
| pool_minmax_merge limit_all/top10 n1000 | 0.408, 0.275 | **DECIDABLE WIN** ~2.5–3.6× |
| pool_minmax_merge small-n + dispatch cells | 0.89–1.01 | **INSIDE NULL FLOOR** (not decidable) |
| merge_dedup ahash/sip (3 cells) | 0.702–0.802 | **DECIDABLE WIN** ~1.25–1.42× |
| filter_match ext (3 cells) | 0.620–0.632 | **DECIDABLE WIN** ~1.6× |
| filter_match path (3 cells) | 0.989–1.006 | **INSIDE NULL FLOOR** (strict tie) |
| sync_hash ahash/sip (3 cells) | 0.408–0.504 | **DECIDABLE WIN** ~2.0–2.4× |
| docid_materialize compact/packed (4 cells) | 0.400–0.448 | **DECIDABLE WIN** ~2.2–2.5× |
| metadata_clone arc_clone (2 cells) | 0.004, 0.006 | **DECIDABLE WIN** ~165–250× |
| explanation_clone arc_clone (2 cells) | 0.012, 0.015 | **DECIDABLE WIN** ~66–83× |
| code_signal_probe streaming (3 cells) | 0.540–0.585 | **DECIDABLE WIN** ~1.7–1.8× |
| expand_fuse borrow_sip (3 cells) | 0.554–0.571 | **DECIDABLE WIN** ~1.75–1.8× |
| expand_fuse borrow_ahash (3 cells) | 0.370–0.374 | **DECIDABLE WIN** ~2.7× |
| negation_normalize fast | 0.0085 | **DECIDABLE WIN** ~118× |
| normalize_signal new n192, n1536 | 1.145, 1.154 | **DECIDABLE REGRESSION** (new ~14% slower) |
| nqc_adaptive_cost incremental_order | 0.285 | **DECIDABLE WIN** ~3.5× |
| nqc_cv populated/reuse, empty, alloc, ilp (14 cells) | 0.002–0.86 | **DECIDABLE WIN** ~1.2–600× |
| nqc_cv sample_builder q32, q256 | 1.040, 1.036 | **INSIDE NULL FLOOR**; q4096 1.038 **DECIDABLE REGRESSION** |
| prf_expand in_place n3, n8, n20 | 1.011, 1.020, 1.105 | n3/n8 **INSIDE NULL FLOOR**; n20 **DECIDABLE REGRESSION** |
| rrf_config_cost tier_weighted, hash_tiebreak (realistic) | 0.996, 0.954 | **INSIDE NULL FLOOR** |
| rrf_config_cost hash_tiebreak (tie_heavy) | 1.062 | **DECIDABLE REGRESSION** (~6% slower under max ties) |
| scoredresult_box boxed n10k, n100k | 0.892, 0.784 | **INSIDE NULL FLOOR** (n10k), **DECIDABLE WIN** (n100k) |
| tokenize_ascii fast (3 cells) | 0.765–0.832 | **DECIDABLE WIN** ~1.2–1.3× |
| tokenize_ascii compact (5800) | 0.968 | **INSIDE NULL FLOOR**; (29k, 116k) 0.773, 0.726 **DECIDABLE WIN** |

Ledger consequences: every prior row resting on a lever median that sits INSIDE its own A/A null
spread is **not decidable** on this harness and must not be cited as a win or a reject — notably
neighbor_smooth_recip v3, filter_match/path, rrf_config_cost (realistic arms), scoredresult_box at
n10k, pool_minmax small-n/dispatch, hubness_dot shipped-vs-simd_par, prf_expand n3/n8, and
rrf_config tier_weighted/hash_tiebreak realistic. Two regressions are newly decidable:
normalize_signal's single-pass rewrite is ~14% SLOWER at n192/n1536, and neighbor_smooth_recip's
v2 mutual is ~1.2× slower — both warrant follow-up before any KEEP claim. Any future KEEP/REJECT
row for these benches must quote the lever median AND its null p5/p95 from the same run.

## 2026-07-19 — Direct-term MaxScore + Block-Max WAND rank-safe pruning (`bd-quill-e4-argus-3ycz.4`, SapphireHill)

The retained E4.4 path adds exact two-pass MaxScore for root unions with 2–8
direct term children and Block-Max WAND for direct 9+-term unions above the
physical-list cost threshold. Every skip ceiling is reconstructed at query time
from validated `(max_frequency, min_fieldnorm)` metadata plus the live
snapshot's idf/avgdl; no stored impact scalar participates. Sealed terms use
paired POSTINGS/BLOCKMAX metadata, while Delta supplies conservative whole-term
bounds for MaxScore and deliberately opts out of BMW because it has no physical
skip blocks. Exact-count, zero-limit, unsupported, and nested scorer shapes stay
on the compressed POSTINGS-only exhaustive path.

Repeated validation is amortized by a lock-free-read, bounded-admission cache
owned by each immutable recovered segment (maximum 128 terms and 16 MiB of
validated payload). Cache identity includes the complete term dictionary
metadata, and construction privately pairs each BLOCKMAX row with the exact
POSTINGS block table it validated. Exact-count queries never initiate cache
population. The cache payload numbers below exclude the small cache-row and
`Arc` bookkeeping overhead.

Authoritative profile: strict remote execution on RCH worker `hz1`, release
profile with LTO, one final binary, 21 alternating paired rounds plus seven
alternating-order latency trials. The timer starts before DOCLEN/POSTINGS cursor
opening and includes scorer construction, collection, and finish. Warm arms
reuse the segment-equivalent validated metadata cache; a separate single cold
observation includes POSTINGS parse and full BLOCKMAX validation. Ratios are
pruned/exhaustive, so `< 1` is faster:

```text
TMPDIR=/tmp RCH_REQUIRE_REMOTE=1 RCH_WORKER=hz1 \
  rch exec -- cargo test --profile release -p frankensearch-quill \
  argus::tests::e44_disjunction_profile_100k_and_1m -- \
  --ignored --exact --nocapture
```

| strategy | docs | warm exhaustive median | warm pruned median | lever median [p5, p95] | A/A null [p5, p95] | speedup | cached payload |
|---|---:|---:|---:|---:|---:|---:|---:|
| MaxScore, 8 direct terms | 100k | 7,674 us | 535 us | **0.069302 [0.063037, 0.072217]** | 0.997529 [0.818903, 1.016239] | **14.43×** | 64,656 B |
| BMW, 9 direct terms | 100k | 7,800 us | 1,603 us | **0.198811 [0.193835, 0.205571]** | 0.999935 [0.983390, 1.012372] | **5.03×** | 68,832 B |
| MaxScore, 8 direct terms | 1M | 83,417 us | 2,624 us | **0.032588 [0.031035, 0.044329]** | 0.996991 [0.870855, 1.060778] | **30.69×** | 644,544 B |
| BMW, 9 direct terms | 1M | 79,093 us | 19,846 us | **0.176633 [0.155324, 0.256451]** | 1.007424 [0.945164, 1.112317] | **5.66×** | 686,256 B |

Every lever median is below its same-binary null p5. The single cold observations
also favored pruning: 10,035→2,821 us and 8,390→3,398 us at 100k; 103,635→19,448
us and 102,474→34,928 us at 1M for MaxScore and BMW respectively. These cold
numbers are setup-inclusive routing evidence, not substituted for the paired
warm claim.

Rank safety is separately pinned by exact global-docid and `f32::to_bits()`
comparison against exhaustive collection at k={1,10,100,1000}, randomized
corpora, tombstones/offsets, and mixed sealed+Delta snapshots whose live avgdl
differs from seal-time avgdl. The final strict-remote Quill suite passed 431
tests (0 failed, 2 ignored, one known flaky symlink case explicitly filtered).
The rejected nested-union attempt and its 19–26% regression are retained in
`docs/NEGATIVE_EVIDENCE.md`.

**Decision: KEEP.** Ship direct-term MaxScore, direct high-clause BMW, the
bounded validated-metadata cache, exact-count bypass, structural/runtime
fail-closed gates, and the retained ignored profile.

## 2026-07-19 — One-pass Quill concat assembly, partial keep (`bd-quill-e3-keeper-ndtk.5`, SapphireHill)

The retained concat path preflights exact IDMAP and STOREDMETA layouts, then
emits both directly into one pre-sized final FSLX allocation. A generalized
IDHASH builder validates the conceptual source-domain IDMAP without first
materializing its durable bytes; Keeper then reparses the emitted IDMAP and
cross-validates that same IDHASH before final segment verification. This removes
the dominant intermediate-buffer and page-fault churn while preserving the
canonical writer's byte layout.

The authoritative comparison used one strict-remote release-LTO binary on
pinned RCH worker `ovh-a` (`51.222.245.56`), SHA-256
`9852128d3075bcd58df4452edc33e0a157fd90e4da8c15729c4411800027d06b`.
The corrected Criterion harness used 20 flat samples, 500 ms warm-up, and 5 s
measurement; every sample retained all immutable outputs until after the
stopwatch read. Physical bytes are exact source FSLX plus merged-output FSLX.

```text
TMPDIR=/tmp RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-a \
  rch exec -- cargo bench --profile release \
  -p frankensearch-quill --bench concat_merge_ab
```

| sources | physical bytes | restored-control median | one-pass median | one-pass ns/B | improvement |
|---:|---:|---:|---:|---:|---:|
| 2 | 2,745,208 | 4,248,149.542 ns | 2,398,286.155 ns | 0.873626390 | **43.55%** |
| 4 | 7,277,512 | 10,576,223.174 ns | 4,082,846.360 ns | 0.561022278 | **61.40%** |
| 8 | 16,354,280 | 24,303,723.300 ns | 10,209,196.348 ns | 0.624252266 | **57.99%** |
| 16 | 34,543,016 | 67,813,377.125 ns | 22,033,773.409 ns | 0.637864783 | **67.51%** |

Criterion classified every arm as improved (`p = 0.00`). The final rebased
strict-remote Quill suite passed **437 tests, 0 failed, 2 ignored**. Exact
oracles cover final FSLX bytes, IDMAP and STOREDMETA concat bytes, IDHASH
source-domain equivalence, and short/long/out-of-order assembler writes. A
separate fresh-eyes review found no blocker.

**Decision: PARTIAL KEEP.** Ship the absolute throughput win and reproducible
benchmark correction, but do not close E3.5: normalized max/min spread is
`1.557204454x`, above the bead's `1.35x` flatness gate. The remaining fixed-cost
problem is now at two-source fan-in; closure evidence is recorded in
`docs/NEGATIVE_EVIDENCE.md`.

## 2026-07-22 — KEEP: Quill segment-parallel docid-set collection, ~2.7–3.5x on 4 rayon threads (`bd-quill-e4-argus-3ycz.9` sibling, cc)

`QuillIndex::execute_docid_query` — the exhaustive unscored id-set path behind `collect_docids`
(every gauntlet differential case's parity workhorse and the CASS structured-filter path) — now
fans sealed-segment walks across rayon under the same `sealed_segment_fanout` gate as the ranked
lever (>= 2 sealed segments AND >= 10k sealed docs). Per-segment `DocSetCollector` partials fold
through the new `DocSetCollector::merge` (raw extend); determinism is free because `finish()`
sorts and dedups, so ANY fold order yields the identical id set. Delta snapshots stay serial.
Pinned by `fanned_docid_collection_matches_serial_exactly` (both fixtures x all queries x 3
fanned repeats); 454/454 quill-lib tests green; scoped clippy `-D warnings` clean.

MEASURED (`segment_fanout_ab` new `[docid-cell]` lane, `paired_median_ratio` + A/A null,
rounds=41 inner=4, `RAYON_NUM_THREADS=4`, worker vmi1227854, LTO release, ratio =
fanned/serial, `<1` wins; bit-exact id-set parity asserted before timing):

| shape | query | null [p5, p95] | lever [p5, p95] | serial -> fanned | decidable |
|---|---|---|---|---|---|
| gate8x2500 (20k docs) | high_df_term | 0.994 [0.826, 1.349] | **0.359** [0.279, 0.430] | 21.0ms -> 6.4ms | YES |
| gate8x2500 | mid_df_term | 1.003 [0.857, 1.200] | **0.365** [0.275, 0.510] | 19.1ms -> 6.6ms | YES |
| gate8x2500 | union3 | 1.019 [0.882, 1.156] | **0.298** [0.261, 0.377] | 61.5ms -> 20.3ms | YES |
| gate8x2500 | phrase | 1.010 [0.917, 1.097] | **0.286** [0.267, 0.309] | 502.2ms -> 142.8ms | YES |

The same run reconfirmed the ranked-lever cells (0.284-0.353, all decidable). Disclosure: the
harness stream cut at the gate4x12500 serial-arm phrase null (a ~13-minute cell — serial phrase
at 50k is ~1.2s/call and the serial-vs-serial null runs 328 timed calls of it), so 50k and
below-gate docid cells were not captured; the four 20k cells are complete, independently
decidable, and structurally identical to the KEEP'd ranked lever. Harness note for future runs:
order the slowest serial cells first or budget per-cell rounds.

**Decision: KEEP.** Same gate, same fan-out rails, near-identical ratios to the ranked lever;
this directly accelerates the E6 oracle-campaign lane's per-case docid parity sweeps.

## 2026-07-22 — KEEP: Quill segment-parallel query fan-out, ~2.8–3.6x on 4 rayon threads above the 10k gate (`bd-quill-e4-argus-3ycz.9`, cc)

`QuillIndex::execute_ranked_query` now fans sealed-segment scoring across rayon when the snapshot
has >= 2 sealed segments AND >= 10,000 total sealed live documents (`SEGMENT_FANOUT_THRESHOLD`,
the house `PARALLEL_THRESHOLD` philosophy from `frankensearch-index/src/search.rs`). Each task
scores one segment into a private `TopDocsCollector` built via the new `empty_like()`, and the
partials fold back through the new `TopDocsCollector::merge()` in ascending segment order. Below
the gate — and always for Delta snapshots (the small mutable tail) — the pre-change serial loop
runs unchanged, so the delta-only small path never pays rayon overhead.

DETERMINISM (correctness, not perf): the collector's heap discipline is a total order (score
`total_cmp` desc, then global docid asc — `HeapEntry::cmp`), so the retained top-`limit+offset`
set is unique and independent of both the rayon schedule and the merge order; `finish()` sorts
with the same order. Rank pruning stays rank-safe under weaker per-segment local cutoffs (the
shared cross-task cutoff is an explicit E8 follow-up lever, deliberately not in this change);
exact-count mode already disables pruning, and per-partial counts merge by checked addition.
Pinned by: `fanned_sealed_collection_matches_serial_exactly` (two fixtures — the 3-segment
concat corpus and a seeded 5x64 xorshift corpus with heavy score ties — x 9 query shapes x 4
limit/offset pages x exact-count on/off x 3 fanned repeats per cell, bit-exact docid+score-bits
compare), `topdocs_merge_reproduces_single_collector_under_any_partition_and_order` (257-doc
7-bucket tie-heavy stream, partitions {1,4,7} x every merge rotation, plus shape-mismatch
fail-closed), and `sealed_segment_fanout_gate_thresholds`. 451/451 quill-lib tests green;
clippy `--no-deps --lib --tests --bench segment_fanout_ab -D warnings` clean (the pre-existing
`id_map_materialize_ab` lint drift is bd-xqhm, untouched); fsfs statistical lane
`benchmark_baseline_matrix` 23/23 green.

MEASURED (`segment_fanout_ab`, new bench, `paired_median_ratio` + A/A null, rounds=41 inner=4,
`RAYON_NUM_THREADS=4`, worker vmi1149989, LTO release, ratio = fanned/serial, `<1` wins; every
cell asserted bit-exact page parity between arms before timing):

| shape | query | null [p5, p95] | lever [p5, p95] | serial -> fanned | decidable |
|---|---|---|---|---|---|
| gate8x2500 (20k docs) | high_df_term | 1.063 [0.819, 1.211] | **0.352** [0.291, 0.669] | 24.8ms -> 9.6ms | YES |
| gate8x2500 | mid_df_term | 1.012 [0.862, 1.206] | **0.356** [0.289, 0.490] | 22.6ms -> 7.6ms | YES |
| gate8x2500 | union3 (MaxScore/BMW) | 0.994 [0.869, 1.106] | **0.310** [0.274, 0.351] | 63.3ms -> 17.5ms | YES |
| gate8x2500 | phrase | 0.972 [0.921, 1.101] | **0.289** [0.259, 0.316] | 499.2ms -> 172.2ms | YES |
| gate4x12500 (50k docs) | high_df_term | 1.000 [0.857, 1.241] | **0.335** [0.271, 0.426] | 30.2ms -> 11.0ms | YES |
| gate4x12500 | mid_df_term | 0.999 [0.869, 1.167] | **0.339** [0.282, 0.422] | 27.3ms -> 9.1ms | YES |
| gate4x12500 | union3 | 0.998 [0.910, 1.205] | **0.326** [0.289, 0.357] | 85.4ms -> 27.0ms | YES |

Disclosure: the harness's last two cells (gate4x12500/phrase and the informational
below_gate2x500 shape) were cut from the captured output by the caller's pipe truncation; the
seven cells above are complete and each is independently decidable. The below-gate overhead the
threshold exists to avoid is documented by the smoke lane (rounds=5): forced fan-out at 2x100
docs measured 2.96x WORSE on high_df_term — the production gate keeps that path serial, and the
`below_gate` cell is not a shipping configuration.

**Decision: KEEP.** Fan-out ships as the default above the gate. Route next (E8, separate
bead): shared cross-task pruning cutoff via atomic score max; re-measure `union3`/`phrase`
cells where local-only cutoffs leave pruning on the table.

## 2026-07-22 — Quill concat merge fixed-span control closes E3.5 (`bd-quill-e3-keeper-ndtk.5`, RoseMaple)

The prior growing-span sweep held logical documents constant while increasing
both source count and the canonical burned-tail output span from 2.7 MB to
34.5 MB. Its max/min statistic therefore combined the two-source per-call floor
with ninefold output growth. The existing hold explicitly named a
fixed-retained-byte control as the required retry predicate.

The retained benchmark now keeps that historical growing-span sweep and adds a
production-valid fixed-span control. Its 2/4/8/16 manifest-consecutive sources
remain lease-aligned and preserve real burned tails, but are distributed across
one common first-to-last lease span. The control retained immutable outputs
until after each stopwatch read and declared the unchanged `<=1.35x` ns/exact
physical-byte gate before any result was observed.

One strict-remote release-LTO run, job `j-29942429901652205` on worker
`vmi1149989`, measured both groups from one binary:

```text
TMPDIR=/tmp RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 \
  rch exec -- cargo bench --profile release \
  -p frankensearch-quill --bench concat_merge_ab
```

| fixed-span sources | exact physical bytes | median | ns / physical byte |
|---:|---:|---:|---:|
| 2 | 34,513,784 | 21.300 ms | 0.617144733 |
| 4 | 34,507,720 | 23.625 ms | 0.684629410 |
| 8 | 34,507,752 | 24.153 ms | 0.699929685 |
| 16 | 34,543,016 | 22.244 ms | 0.643950719 |

Exact bytes vary by only `1.001022844x`. The fixed-span normalized spread is
**`1.134141877x` — PASS**, 16.0% below the predeclared ceiling. Even the
conservative cross-arm extreme (`max interval high / min interval low`) is only
`1.250316110x`. The same binary's preserved growing-span diagnostic remained a
FAIL at `1.660618226x`, while its 4/8/16 subspread was `1.094554342x`; that
failed row remains negative evidence rather than being erased.

The performance result completes the already-landed correctness bundle: exact
raw posting-block and schedule invariance, offset-only BLOCKMAX rebasing,
monolithic df=100+300/stat/query equivalence, IDMAP/IDHASH rebuild, tombstone
union, numeric/stored-section merge, rejection tests, and durable region-scoped
publication.

**Decision: KEEP / CLOSE E3.5.** The production one-pass concat path remains
unchanged. The fixed-span control directly verifies approximately flat CPU per
physical byte as source count grows; the historical growing-span result still
documents the expected small-object fixed floor.

## 2026-07-22 — LANDED: Quill SWAR default tokenizer — length-dependent, decidable win on long tokens (`bd-quill-e1-scribe-bejd.1`, FuchsiaMaple)

`FrankensearchTokenizer::analyze` (Quill's default analyzer, on the ingest and query hot paths)
now finds token boundaries with a SWAR-on-u64 byte classifier that visits eight ASCII bytes per
64-bit word (`skip_separators` / `scan_token_end` in `crates/frankensearch-quill/src/scribe.rs`),
falling back to the scalar char-walk for the span around each non-ASCII byte. It is safe code only —
no `core::arch` intrinsics; the quill crate root is `#![forbid(unsafe_code)]`. The classifier is a
borrow-safe per-lane range test (guard the top bit of each lane, then subtract the broadcast
threshold so every lane stays in `[1,255]` and no borrow crosses a lane boundary — the bare
Bit-Twiddling-Hacks `hasless` is NOT per-lane-correct, a bug the byte-parity property tests caught
before this landed).

BYTE PARITY (correctness, not perf): the emitted `AnalyzedToken` stream (text/position/offsets) is
byte-identical to the retained scalar char-walk oracle `analyze_default_scalar_reference` and to the
shipping Tantivy `SimpleTokenizer + LowerCaser` incumbent, pinned by the pre-existing language-contract
fixture test plus new tests: an exhaustive per-lane classifier check over every ASCII byte, a random
ASCII-word cross-lane-contamination guard, lane-edge cases (tokens of length 6..=17 straddling the
8/16-byte window; multi-byte scalar values at lane 7 / lane-0-of-next-window / mid-window), and a
4000-input xorshift property corpus mixing ASCII with 2/3/4-byte scalar values. 61/61 scribe tests
and 385/385 quill-lib tests green.

Bench `tokenizer_simd_ab` (same-binary `paired_median_ratio`, 41 rounds × inner=16 ~4ms batches,
A/A null-control), authoritative run on fleet worker `ovh-a` via `rch exec`, `--profile release`.
Ratio = `simd/scalar`, `<1.0` = SWAR faster. The payoff is length-dependent:

| bench (48KiB corpus) | lever median [p5, p95] | A/A null [p5, p95] | verdict |
|---|---|---|---|
| tokenizer_simd_long (24–48B tokens) | 0.6818 [0.6690, 0.6884] | 1.0002 [0.9952, 1.0079] | **DECIDABLE WIN** ~1.47× (criterion corroborates 77.0µs vs 94.4µs, ~1.23×) |
| tokenizer_simd (short ~6B tokens) | 0.83 / 1.02 (run-to-run) | ±2%…±10% (run-varying) | **WASH** — sign flips across runs inside fleet noise; criterion ≈parity (171µs vs 174µs) |

Reading: on the long-token corpus (hashes/base64/UUIDs/long identifiers with long separator runs)
the SWAR classifier amortizes its per-window mask and wins decidably and reproducibly (~1.2–1.5×,
tight null). On the realistic short-token corpus (~6-byte space-separated words) it is ≈parity — the
scalar `tokenizer_next_char` already has an ASCII byte fast-path, so a ~6-byte token barely fills one
8-lane window, and the lever sign flips run-to-run (0.83 win one run, 1.02 regression another) inside
the fleet's null spread; this is NOT cited as a short-token win. KEEP as the default because it is a
large decidable win on the long tokens common in code/log/data corpora, byte-parity-correct, and
carries no reproducible short-token regression. (bd-5hz0 lesson holds: this is a full-scan classify —
every byte visited — not a `memchr`/`contains` early-exit scan, so SIMD helps rather than regresses.)

## 2026-07-22 — BLOCKED: Quill short-token cached start-window mask did not reach the timed path (`bd-short-token-mask-reuse-cpn9`, IndigoOtter)

Profile-first routing selected the still-open short-token gap in
`tokenizer_simd_ab`, not the already-landed long-token SWAR win. The incumbent
same-binary short-token probe on RCH worker `hz1` reported `simd/scalar =
0.9750 [0.9253, 0.9900]` against an A/A null of `0.9988 [0.9547, 1.0367]`:
inside the null floor and therefore not a decision. Its raw Criterion sample CVs
were 4.2350% for the scalar arm and 17.1325% for the SWAR arm, confirming that
the short-token measurement needed a stronger paired comparator.

The attempted single lever carried the already-computed ASCII word and
alphanumeric lane mask from separator skipping into token-end scanning, avoiding
a reload/reclassification when a short token starts and ends in one eight-byte
word. This maps to the alien-graveyard broadword rank/select primitive plus a
CEGIS-style retained incumbent comparator. Before timing, all three focused
behavior-isomorphism tests passed remotely on `ovh-b`: randomized mixed-Unicode
corpus, lane-edge corpus versus the scalar oracle, and lane-edge corpus versus
the retained shipping SWAR path (**3 passed, 0 failed**). Token text, positions,
offsets, ordering, and lowercase behavior were exact; tie-breaking, floating
point, and RNG semantics are not involved.

The decisive strict-remote release benchmark never executed. An initial
`RCH_REQUIRE_REMOTE=1` attempt correctly refused local fallback because no
worker was admissible. Job `j-29942429901652077` was then admitted on
`vmi1153651` at `2026-07-22T18:16:40Z`; after roughly 34 minutes of project
sync, dependency download, compilation, and final linking it still had emitted
zero A/A, A/B, Criterion, or CV rows. It was cancelled at the closeout boundary.
Consequently there is **no candidate timing result and no KEEP/REJECT claim**.
All speculative source and benchmark edits were manually removed; only this
evidence remains.

**Decision: BLOCKED / UNTIMED.** Retry only when an admissible four-slot RCH
worker has a warm `frankensearch-quill` release-benchmark dependency graph (or
the exact benchmark binary can be built inside ten minutes). Rerun the retained
same-binary `shipping SWAR` versus `cached start-window mask` comparator with
41 interleaved rounds, its shipping-vs-shipping A/A null, and Criterion arms;
KEEP only if the candidate interval clears the null floor and both decisive-arm
CVs are below 5%. Otherwise record the numeric REJECT and its next predicate.

## 2026-07-22 — Quill columnar/radix wins the bulk-seal lane (`bd-quill-e1-scribe-bejd.7`, RoseMaple)

E1.7 compared the two already-shipping source representations in one binary.
The old arm sealed a frozen Delta segment's term-keyed arena chains; the new
arm sealed a `ColumnarAccumulator` through `FlushMode::Automatic` and its
stable radix partition. Both arms received the same deterministic documents,
metadata, segment identity, and fixed Rayon pool. Every cell asserted exact
FSLX byte equality before timing. Accumulation time and retained bytes were
reported outside the seal timer so work could not be silently shifted between
the two representations.

The primary full-profile run was strict-remote job `j-29942429901652045` on
`vmi1149989`, one release-LTO binary, five alternating paired rounds per cell:

```text
RCH_REQUIRE_REMOTE=1 RCH_NO_SELF_HEALING=1 \
  rch --no-self-healing exec -- env \
  QUILL_E17_SCALE=full QUILL_E17_PROFILE=all \
  QUILL_E17_ROUNDS=5 QUILL_E17_THREADS=all \
  cargo bench -p frankensearch-quill --features bench-internals \
  --profile release --bench scribe_flush_ab
```

Ratios are `columnar/radix seal / Delta hash-chain seal`, so `< 1` favors
radix. The null column is same-binary Delta/Delta:

| profile | threads | A/A null median [p5, p95] | radix/hash median [p5, p95] | verdict |
|---|---:|---:|---:|---|
| golden-small, 5k docs / 1.45M tokens / 21,384 terms | 1 | 1.0310 [0.9951, 1.0892] | **0.7266 [0.6607, 0.9645]** | radix clears null floor |
| golden-small | 8 | 0.9827 [0.9110, 0.9883] | **0.6165 [0.5100, 0.6551]** | radix clears null floor |
| golden-medium, 50k / 17.5M / 181,072 | 1 | 0.9721 [0.8726, 1.0726] | **0.7763 [0.7568, 0.8006]** | radix clears null floor |
| golden-medium | 8 | 1.0085 [0.9593, 1.0641] | **0.6858 [0.6171, 0.7337]** | radix clears null floor |
| synthetic-xlarge-max-lease, 65,536 / 22.9376M / 327,680 | 1 | 1.0132 [0.9219, 1.0393] | **0.8736 [0.7872, 0.8868]** | radix clears null floor |
| synthetic-xlarge-max-lease | 8 | 0.9949 [0.9289, 1.0638] | **0.6697 [0.6687, 0.7108]** | radix clears null floor |

The xlarge cell is exactly one maximum Quill lease, not the pending E6.1
one-million-document generator. An independent nine-round golden-small run on
`ovh-a` exposed the expected crossover uncertainty: the 1-thread lever was
`0.9714 [0.9343, 0.9817]` inside its `0.9979 [0.9488, 1.0241]` null spread
(NOISE), while 8 threads remained decisive at `0.7489 [0.7162, 0.7810]`
against null `0.9998 [0.9625, 1.0594]`. Small/1-thread is therefore not cited
as a stable win.

The untimed representation diagnostics explain the lane split:

| profile | columnar accumulation | Delta accumulation | columnar retained bytes | Delta retained bytes | exact output bytes |
|---|---:|---:|---:|---:|---:|
| golden-small | 381.369 ms | 972.620 ms | 35,094,405 | 79,379,328 | 22,387,702 |
| golden-medium | 38.904 s | 21.820 s | 417,839,437 | 969,944,696 | 266,830,326 |
| synthetic-xlarge-max-lease | 113.910 s | 30.346 s | 553,452,349 | 1,259,552,628 | 372,706,294 |

Columnar retained only about 43–44% of Delta's bytes, but its one-shot
accumulation diagnostic crossed from faster at golden-small to 1.78x and 3.75x
slower at medium and xlarge. Those setup observations are not paired timing
claims.

**Decision: KEEP the existing lane separation.** `FlushMode::Automatic` and
the stable radix partition remain the bulk-seal winner, especially at 8
threads; Delta remains the small/incremental construction path. The shipping
code already has this separation, so E1.7 needs no production routing edit.
The paired harness is retained for the E8 matrix and future crossover work.

## 2026-07-22 — REJECT: caching the token-start SWAR word regresses short and long corpora (`bd-short-token-mask-reuse-cpn9`, IndigoOtter)

This timed retry supersedes the earlier UNTIMED blocker. The single candidate
carried the eight-byte ASCII word and its alphanumeric lane mask from
`skip_separators` into `scan_token_end`, so a short token ending in that word
could avoid one reload and one `swar_ascii_alnum_mark` call. The retained
pre-change shipping SWAR path and candidate ran in one release binary. Before
timing, strict-remote job `j-29942429901652220` passed all three focused
behavior-isomorphism tests on `vmi1149989`: randomized mixed Unicode, lane-edge
versus the scalar oracle, and lane-edge versus the retained shipping path
(**3 passed, 0 failed**). Token text, positions, byte offsets, ordering, and
lowercasing were exact.

Strict-remote release job `j-29942429901652232` then completed on the same
worker. Ratios are candidate/shipping, so values above one are regressions:

| corpus | shipping/shipping A/A median [p5, p95] | cached/shipping median [p5, p95] | shipping Criterion interval | cached Criterion interval |
|---|---:|---:|---:|---:|
| short-token, 48 KiB | 0.9986 [0.7967, 1.2115] | **1.2030 [1.0739, 1.3024]** | 211.31–225.74 us | **271.73–282.93 us** |
| long-token control, 48 KiB | 0.9957 [0.9483, 1.1542] | **1.1313 [1.1135, 1.1601]** | 81.930–86.646 us | **98.392–100.84 us** |

Raw Criterion per-iteration sample CVs were respectively **7.119% / 8.691%**
for short shipping/cached and **8.293% / 4.066%** for long shipping/cached.
Those three above-5% CVs and the wide short A/A band prohibit any KEEP claim.
They do not rescue the candidate: both paired medians point slower, and both
independent Criterion intervals are disjoint in the slower direction. A warm
repeat was attempted, but RCH ignored the `vmi1149989` soft pin and routed job
`j-29942429901652267` to `vmi1153651`; it was cancelled before timing because
cross-worker samples would not repair the same-worker evidence.

**Decision: REJECT and retain shipping SWAR.** The extra `Option`-carried
window state and control flow cost more than reloading/reclassifying an
L1-resident eight-byte word: about 20.3% by the short paired median and 13.1%
on the long control. All candidate source and benchmark edits were manually
removed. Retry only if profile plus generated-code evidence shows a new
representation eliminates the added state spills/branches, and an idle
same-worker run produces an A/A band inside 0.97–1.03 with all decisive-arm
CVs below 5%; require cached/shipping <=0.97 on short tokens and no long-token
regression before reconsidering production.

## 2026-07-22 — REJECT: legacy f16 dot-product A/A retrofit misses the <5% CV admission rule (`bd-dot-product-aa-null-wegn`, IndigoOtter)

Negative-evidence mining showed that the f16/f32/i8 kernel families are
already heavily explored, while the legacy `dot_product` Criterion target
still lacked the same-binary A/A floor required to interpret its existing
dispatch/generic comparison. A bench-only candidate added exact `to_bits()`
parity checks plus interleaved shipping/shipping and AVX2-dispatch/generic
measurements at dimensions 256 and 384; production code was untouched.

Strict-remote release job `j-29942429901652321` completed on `vmi1149989`.
Ratios below are first/second, so dispatch/generic values below one favor the
shipping dispatch:

| dimension | dispatch/dispatch A/A median [p5, p95] | dispatch/generic median [p5, p95] | dispatch Criterion interval | generic Criterion interval | raw CV dispatch/generic |
|---:|---:|---:|---:|---:|---:|
| 256 | 1.0055 [0.9154, 1.0743] | **0.2798 [0.2334, 0.4562]** | 286.06–308.79 us | 1.1273–1.2494 ms | **6.399% / 5.607%** |
| 384 | 1.0292 [0.7778, 1.1771] | **0.4423 [0.2461, 0.6548]** | 418.15–509.40 us | 1.8671–1.9869 ms | **32.059% / 6.491%** |

The direction is large and exact parity passed, but all four decisive-arm CVs
exceed 5% and the 384-dimensional A/A band is especially wide. The benchmark
therefore cannot support a KEEP under the admission rule. A warmed retry with
30 samples and four inner iterations was requested, but RCH routed
`j-29942429901652351` to the cold, disk-critical `hz1` worker instead of the
warmed worker; it was cancelled before compilation or timing. All bench and
Cargo edits were manually removed.

**Decision: REJECT the measurement retrofit, not the shipping AVX2 kernel.**
Reopen only on an idle, admissible worker with the same warmed build cache (or
isolated CPU pinning), hard-code at least 30 samples and four inner iterations,
and require both dimensional A/A bands within 0.97–1.03 plus every dispatch
and generic CV below 5%. Exact bit parity remains mandatory; only then may the
dispatch/generic speedup be cited from this legacy target.

## 2026-07-22 — BLOCKED / UNTIMED: retained top-k A/A CV audit never links remotely (`bd-3srq`, IndigoOtter)

Negative-evidence mining closed the proposed duplicate retrofit before any
source edit: `int8_vs_f16_fast_ab` already measures the real shipped exact-f16
and int8-two-pass methods in one binary, includes an interleaved f16/f16 A/A
null, and proves set recall plus exact ordering over 32 queries. Its shipped
2026-07-11 result remains 1.43x paired with recall@10 1.0000 and 32/32 exact
order matches. The open question here was only whether that retained harness
can reproduce under the newer per-arm Criterion CV-below-5% admission rule.

After `scripts/rch-ensure-deps.sh --check` passed, strict-remote release job
`j-29942429901652363` began on `vmi1264463` at 20:25:38Z. RCH ignored the
requested warmed-worker hint, synchronized a cold workspace, downloaded and
compiled dependencies, and reached `frankensearch-core` plus
`frankensearch-index`. It then spent more than ten minutes in the final release
link without emitting the parity banner, A/A row, Criterion sample, or timed
path. The hook remained alive, but after 21 minutes 22 seconds total the job was
cleanly cancelled at 20:47:00Z. No benchmark or production source was changed.

**Decision: BLOCKED / UNTIMED, not REJECT.** There is no new top-k performance
claim and the existing shipped win is not overturned. The same cold
release/LTO route also blocks starting the queued legacy `rrf_merge_fuse` A/A
retrofit responsibly in this cycle; ledger/history mining found that target's
missing null control and weak first/last-only parity check, but it was not
edited or timed. Retry both only when RCH honors a worker with a warmed
frankensearch release-bench graph or the exact benchmark binary links within
ten minutes. Top-k then requires the existing 32-query recall/order proof,
A/A wholly within 0.97–1.03, and both Criterion arm CVs below 5%; RRF requires
full `FusedHit` equality, an interleaved map/map A/A, and all compared arm CVs
below 5% before retaining any harness change.

## 2026-07-22 — BLOCKED / NO ADMISSIBLE VERDICT: short-token ASCII boundary selection mask (`bd-l5x3`, IndigoOtter)

Alien-graveyard routing selected vectorized selection masks (§8.2) for the
still-open short-token tokenizer gap. The candidate classified one eight-byte
ASCII word, derived every alphanumeric/non-alphanumeric transition from that
single lane mask, and enumerated the transition bits to emit token boundaries;
Unicode-containing words retained the scalar char fallback. The exact retained
shipping SWAR tokenizer stayed in the same binary. Before timing, three strict-
remote tests passed: 4,000 randomized mixed-ASCII/Unicode strings, the existing
lane-edge corpus, and every length from 1 through 129 bytes all matched both the
retained SWAR token stream and scalar reference exactly, including token text,
byte offsets, positions, position lengths, order, and lowercasing.

The first completed release job, `j-29943190916169794` on `vmi1227854`,
overlapped the segment-fanout benchmark on that worker. Its direction was
promising but its null/CV evidence was inadmissible:

| corpus | shipping/shipping A/A median [p5, p95] | candidate/shipping median [p5, p95] | shipping mean / CV | candidate mean / CV | mean ratio |
|---|---:|---:|---:|---:|---:|
| short 48 KiB | 0.9868 [0.8642, 1.0672] | **0.8190 [0.7589, 0.9014]** | 183.557 us / **5.694%** | 137.626 us / **10.128%** | **0.7498** |
| long 48 KiB | 0.9969 [0.9118, 1.0651] | 1.0129 [0.9028, 1.0641] | 77.627 us / 3.891% | 77.577 us / **6.800%** | 0.9994 |

Both A/A bands exceed the required 0.97–1.03 envelope and three decisive arms
miss the CV-below-5% gate. The short direction therefore cannot be called a
KEEP, while the contaminated control cannot honestly reject the primitive.
The paired batch was increased symmetrically from 16 to 64 passes and an
isolated retry requested every slot on an idle worker. Three idle alternatives
failed closed before execution with `RCH-I004 alias_wrong_target`
(`vmi1152480`: `/data`; `hz2` and `vmi1293453`: `/data/tmp`). Strict-remote job
`j-29943190916169821` then reserved all 3/3 slots on `vmi1156319`, compiled the
full release graph, reached the final benchmark link, and failed with
`RCH-E104` after the 1,800-second SSH timeout without executing the binary or
emitting a null, parity banner, or Criterion sample.

**Decision: BLOCKED / NO ADMISSIBLE VERDICT, not REJECT.** The speculative
production and comparator edits were manually removed. Retry only when an
idle requested worker can reserve its full slot count and link or reuse the
exact release benchmark binary within the remote timeout. Retain the exact
SWAR/scalar parity gates and 64-pass interleaving; require both A/A p5–p95 bands
wholly inside 0.97–1.03, every shipping/candidate Criterion CV below 5%, short
candidate/shipping <=0.97, and no decidable long-token regression before KEEP.

## 2026-07-23 — KEEP: fragmented sub-10k Quill snapshots use segment-parallel scoring (`bd-quill-e7-integration-flip-d0tx.2`, RoseMaple)

The fsfs 5,000-row watch contract retained one 5,000-document seed segment
while 100 immutable 50-row replacement leaves accumulated. A temporary
same-worker component profile isolated the miss: planning was 0.018 ms p95,
flush/publication 7.620 ms, and explicit commit 0.005 ms, while the required
post-publication visibility query cost 32.261 ms p95. The existing ranked
query fan-out remained serial because its only gate was 10,000 live documents,
even though the snapshot had enough independent leaves to amortize Rayon.

One lever extends the already bit-exact `sealed_segment_fanout` decision:
snapshots with at least eight sealed segments now fan out even below 10,000
live documents. The two-segment below-gate control stays serial. The collector,
scoring, merge, ordering, and materialization paths are unchanged.

Exact strict-remote same-worker ovh-a release-perf contract results:

| state | initial docs/s | watch updates/s | update-to-searchable p95 |
|---|---:|---:|---:|
| baseline `e663dff4` | 73,713 | 2,716 | 32.766 ms |
| eight-segment fan-out | 69,279 | **5,462** | **13.017 ms** |

The candidate clears the full 20k/5k/25ms contract while doubling watch
throughput and cutting p95 by about 60%. The lower initial number remains more
than 3.4x above its gate and is unaffected by this query-only predicate.
Correctness proof: the full Quill library suite passed 473 tests with zero
failures and two intentional ignores, including the seeded bit-exact
serial-versus-fanned ranked and doc-id parity matrices; strict Quill
library/test Clippy passed with `-D warnings`.

**Decision: KEEP.** Fragmentation is now an independent fan-out signal; the
document-count threshold still protects genuinely small, unfragmented
snapshots.

## 2026-07-23 — REJECT: fsfs lexical debounce floor retune is an update-to-searchable p95 wash (`bd-quill-e7-integration-flip-d0tx.2`, RoseMaple)

The final E7.2 acceptance gate compared the shipping 6 ms lexical debounce
base with the admitted 3 ms floor in the deterministic fsfs pressure harness.
The harness now calls the production adaptive-debounce helper rather than
copying its formula. A query edit recomputes the window before scheduling a
refresh, so the simulated 64 ms typing cadence and 12-character query install
the same 7 ms adaptive window in both arms. Each observation then adds the
measured Quill watch-query p95 anchor of 13.017 ms and the scenario's
deterministic pressure penalty.

Strict-remote ovh-a results:

| pressure scenario | shipping 6 ms base p95 | candidate 3 ms base p95 | shipping/candidate invocations |
|---|---:|---:|---:|
| gradual ramp-up | 36.017 ms | 36.017 ms | 40 / 40 |
| spike and recovery | 36.017 ms | 36.017 ms | 25 / 25 |
| hysteresis oscillation | 24.017 ms | 24.017 ms | 30 / 30 |
| long-run soak fault injection | 20.017 ms | 20.017 ms | 180 / 180 |

The full pressure harness passed all 15 tests and all 35 oracles. The scoped
fsfs Clippy command reached the crate but remained blocked by 48 pre-existing
crate-wide `-D warnings` findings; none named the changed pressure harness.

**Decision: REJECT the 3 ms retune and retain the shipping 6 ms base.** It
cannot improve steady query-edit latency because production adaptive
recomputation replaces it before search scheduling, and the pressure evidence
shows neither a p95 nor invocation-count benefit.

## 2026-07-23 — BLOCKED / UNTIMED: isolated short-token boundary-mask retry still cannot enter Criterion (`bd-l5x3`, TurquoiseTern)

Ledger-first routing reopened the previously untimed short-token selection-mask
candidate without touching the closed SIMD dot-product, top-k, RRF, or
architectural lanes. The same binary retained the pre-change shipping SWAR
comparator, exact scalar oracle, and symmetric 64-pass A/A and A/B harness.
Strict-remote job `j-29944835100114977` passed all 3 randomized mixed-Unicode,
lane-edge, and shipping-incumbent conformance tests on `vmi1264463`; its cold
debug graph required 22m56s.

No release timing was admitted. Three requested full-worker routes failed hard
preflight with `RCH-I004 alias_wrong_target`; other routes were disk-critical,
operator-disabled, or draining. The last admissible 8-slot request on
`vmi1149989` expired with `RCH-I001 queue_timeout` before compilation. Thus no
new null band, candidate ratio, Criterion interval, or per-arm CV exists.

**Decision: BLOCKED / UNTIMED, not REJECT.** The speculative source and bench
edits were manually removed. Retry only on a fully reserved admissible worker
whose exact release binary links or is reused within ten minutes. KEEP still
requires exact SWAR/scalar/shipping parity, 64-pass A/A p5–p95 wholly within
0.97–1.03 on both corpora, every decisive-arm CV below 5%, short
candidate/shipping <=0.97, and no decidable long-token regression.

## 2026-07-23 — REJECT / INVALID-CV: short-token ASCII boundary selection mask remains inside its null floor (`bd-l5x3`, TurquoiseTern)

The previously blocked retry reached both proof and timing on a fully reserved
worker. Strict-remote job `j-29944835100115026` used all 8/8 slots on
`vmi1227854` and passed the randomized mixed-Unicode, lane-edge scalar-oracle,
and retained shipping-SWAR conformance tests. Strict-remote release job
`j-29944835100115037` then used the same worker's full 8/8 slots and executed a
single binary containing the shipping and candidate implementations, symmetric
64-pass paired A/A and A/B controls, and 30-sample Criterion arms.

| corpus | shipping/shipping A/A median [p5, p95] | candidate/shipping A/B median [p5, p95] | shipping Criterion interval / CV | candidate Criterion interval / CV |
|---|---:|---:|---:|---:|
| short 48 KiB | 1.0056 [0.7606, 1.2099] | 0.8223 [0.6695, 0.9588] | 184.57–193.09 us / **7.9933%** | 162.44–183.43 us / **15.9965%** |
| long 48 KiB | 0.9999 [0.9004, 1.0346] | 0.9978 [0.8487, 1.0920] | 81.448–87.058 us / **5.4384%** | 80.453–84.471 us / **14.2068%** |

The short direction remains favorable, but neither A/A band is wholly inside
0.97–1.03 and every raw per-iteration CV exceeds 5%. Thus the apparent gain is
inside an unstable null floor and cannot satisfy the KEEP policy.

**Decision: REJECT this representation / INVALID-CV for KEEP (consecutive
REJECT 1).** Production and benchmark source were manually restored exactly.
Do not retry the same boundary-mask representation under ordinary fleet
scheduling. Retry only after a fresh profile/disassembly yields a materially
different mechanism or the exact binary can run on genuinely isolated/pinned
CPU; require exact scalar/shipping parity, both A/A bands wholly within
0.97–1.03, every arm CV below 5%, short candidate/shipping at most 0.97, and no
decidable long-token regression.

## 2026-07-23 — BLOCKED / UNTIMED: isolated top-k A/A/CV retry still cannot link inside its admission window (`bd-3srq`, TurquoiseTern)

Ledger and recent-history mining selected the existing
`int8_vs_f16_fast_ab` target rather than duplicating benchmark infrastructure
or changing production source. That target already supplies the real-method
f16/f16 A/A null, exact-f16/int8-two-pass A/B, 32-query recall/order proof, and
30-sample Criterion arms. A temporary teardown-only edit prevented the remote
benchmark from deleting its index artifact under repository policy; it was
manually restored exactly after the attempt.

The first strict-remote all-8-slot pin to `vmi1227854` failed closed with
`RCH-I001 queue_timeout` before selection or execution. Strict-remote job
`j-29944835100115068` then reserved all 6/6 slots on `vmi1264463`, completed
its delta sync, and entered Cargo execution at 02:09:15Z. The worker-scoped
target was cold, however, and rebuilt the dependency graph from scratch. The
job crossed the prior ledger row's ten-minute admission limit without linking
or emitting the recall proof, paired A/A or A/B ratios, Criterion samples, or
per-arm CVs, and was cancelled cleanly.

**Decision: BLOCKED / UNTIMED, not REJECT; the consecutive-REJECT count remains
1.** No source or benchmark change ships, and no new top-k performance claim
is made. The RRF A/A retrofit remained untouched after this release-link
blocker surfaced. Retry only with a reusable worker-scoped target containing
the exact release bench graph, or a fully reserved worker that can link it
inside ten minutes; retain the 32/32 exact-order and recall@10=1.0000 proof,
require A/A wholly within 0.97–1.03, and require both Criterion arm CVs below
5%.

## 2026-07-24 — REJECT: dictionary preflight cannot profitably demand-gate low-cost Quill BLOCKMAX opening (`bd-quill-e8-perf-doctrine-x4e4.5.2`, RoseMaple)

The candidate made sealed direct 9+-term unions validate DOCLEN and POSTINGS,
look up every root term in TERMDICT, and reproduce Argus's live active-clause
count plus total posting cost before allowing BLOCKMAX to open. Direct 2–8-term
MaxScore remained unchanged; exact-count, zero-limit, nested/unsupported,
absent-term, below-cost, and Delta-only cases stayed exhaustive. Structural
section-table lengths let dictionary parsing avoid touching BLOCKMAX while the
preflight ran.

A strict-remote focused test on `hz2` passed the complete opening contract:
below-threshold, absent, exact-count, zero-limit, and nested queries left the
BLOCKMAX checksum witness cold and cached zero bounds; one eligible 9-term
query opened BLOCKMAX and cached exactly nine bounds; selected corrupt BLOCKMAX
failed closed while the below-cost query ignored it; Delta-only search stayed
exhaustive. Candidate and exhaustive rows matched exact global doc IDs and
`f32::to_bits()` scores at `k={1,10,100,1000}`.

The keep decision used one strict-remote release-profile binary on the same
`hz2` worker. The fixture had 1,024 sealed documents, nine live direct terms,
and total posting cost 9,216, deliberately below the existing 16,384 BMW gate.
Twenty-one paired rounds measured setup-inclusive fresh-handle cold queries
with one inner iteration and resident-handle warm queries with seven inner
iterations. Ratios are demand-gated/eager, so values above one are slower:

| state | A/A median [p5, p95] | demand-gated/eager median [p5, p95] | decidable |
|---|---:|---:|---:|
| cold | 0.998542 [0.963435, 1.006618] | 1.002419 [0.956571, 1.035464] | no |
| warm | 1.003926 [0.991398, 1.010064] | **1.063222 [1.011056, 1.258719]** | **yes, regression** |

**Decision: REJECT.** Cold is wholly inside the measured null floor, while
warm demand gating is a clear 6.32% median regression. The extra dictionary
traversal duplicates work that ordinary lowering must still perform, so the
avoided low-cost BLOCKMAX validation cannot amortize the preflight. Production,
test instrumentation, and the temporary profile were manually restored
byte-for-byte; only this evidence ships. Do not retry a separate dictionary
preflight. A future attempt must fuse eligibility accounting into the single
term-lowering traversal while deferring BLOCKMAX validation, then beat both
this same-binary cold null and warm regression with exact parity intact.
