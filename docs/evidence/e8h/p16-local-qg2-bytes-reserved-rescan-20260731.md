# E8-H P16 — the QG-2 ingest gap was 26% telemetry bookkeeping; O(1) `bytes_reserved` is a 2.76x KEEP (2026-07-31)

**Task:** stop working the hand-written lever list and instead profile the
*whole job* against the live incumbent, rank top self-time entries, and for each
ask the only question that matters — **does Tantivy pay this same cost?**

**Answer.** The single largest self-time entry in Quill's QG-2 ingest was not
tokenization, not the seal, not memmove. It was
`<ColumnarAccumulator>::bytes_reserved` at **26.17% of total profile
self-time** — a *telemetry high-water accessor* that rescanned the whole term
interner and arena **once per document**. Tantivy maintains no such per-document
figure, so it is work the incumbent does not do at all. Making it O(1) is a
**2.76x speedup on the Quill arm** and narrows the Tantivy gap from **7.60x to
2.82x**.

## Disposition

`VALID-AB / DIAGNOSTIC-CLASS / INCUMBENT-RATIO`. Host `thinkstation1` is the
local diagnostic class `local-5975wx-32c`, **not** a registered campaign class:
this activates no gate and moves no ratchet. The Quill-vs-Tantivy ratio within
each run is same-invocation and paired; the baseline-vs-lever comparison is
**cross-run** (see "Honest limits").

## Provenance

| axis | value |
|---|---|
| Repo state | local `main` at `c989466f`; branch is 311 behind / 30+ ahead of `origin/main` |
| Host | `thinkstation1`, Threadripper PRO 5975WX, 32 physical / 64 SMT |
| Toolchain | rustc 1.99.0-nightly (9f36de775 2026-07-19) |
| Harness | `perf_matrix`, `--features perf-harness --profile release-perf`, gate `QG-2`, fixture `bulk/medium/1/positions_on` (50,000 docs, threads=1, positions on), Zipf S11 / 8,192-term vocabulary / `max_document_bytes` 4,096 |
| Incumbent | genuine Tantivy, validated in-process by `assert_incumbent_is_genuine_tantivy()` before any cell is timed |
| Baseline ELF | `d39d8c0277ab4cdc56933ed9243e624395ac660b34c4f3f59be38caae809c4e0` |
| Lever ELF | `47b4d704bfab9ac730d71fc4819ee73e81f8dd27179bc4711717cdc3a0648195` |
| Profile | `perf record -F 999` (flat, no call graph), same cell |

## The profile that redirected the work

Top self-time entries, whole process, both arms:

| share | symbol | incumbent pays it? |
|---|---|---|
| **26.17%** | `<frankensearch_quill::scribe::ColumnarAccumulator>::bytes_reserved` | **NO** |
| 4.13% | `__memmove_avx_unaligned_erms` | yes — closed as a lever previously |
| 4.00% | `__ieee754_exp_fma` | yes (BM25 both sides) |
| 3.81% | `FrankensearchTokenStream::advance` | yes (this is the *Tantivy* arm's tokenizer) |
| 3.78% | `SyntheticCorpus::document_at` | **harness generator, charged to BOTH arms** |
| 2.63% / 2.39% | tantivy `SpecializedPostingsWriter` | incumbent-side |
| 2.10% | `FrankensearchTokenizer::analyze` | yes |

Only the top entry answers "no". Everything else is either shared, incumbent-
side, or generator overhead (`document_at` + `core::fmt::write` +
`String::write_char/write_str` + `pad_integral` + `u32::fmt` ≈ 12% of profile is
the *fixture generator*, consistent with the previously recorded ~21%-of-both-
arms figure).

## Mechanism

`bytes_reserved()` was computed unconditionally per document inside
`add_document_with_values` (for the returned `DocumentAccumulation` and its
tracing fields); `index.rs` folds it into an arena high-water mark. It summed
two O(n) scans:

- `TermInterner::bytes_reserved` iterated **every bucket** to total the
  collision-id vector capacities — O(distinct terms). At the pinned QG
  vocabulary of 8,192 that is ~8,192 bucket visits per document, **~409 million
  visits across a 50,000-document run**.
- `ByteArena::bytes_reserved` summed **every chunk's** capacity.

**This is a sibling-path miss.** `bd-w4j5` already gave `bytes_used` a running
counter for precisely this reason — *"`should_flush` re-called `bytes_used` once
per ingested document"* — and left its `bytes_reserved` twin scanning.

## Fix, and why it is exact rather than approximate

- Arena capacity changes **only** at `chunks.push`: the `needs_new` guard means
  `extend_from_slice` never grows an existing chunk. A running sum is exact.
- Collision-id capacity changes **only** at the two `Bucket::Many` sites, which
  now report their own capacity delta.
- `reset()` recomputes instead of tracking `retain`'s drops — once per flush
  cycle, off the hot path.
- Both accessors retain a `debug_assert_eq!` against the original scan, so drift
  fails loudly in every test build. **No `debug_assert` fired across the 473-test
  suite**, including the scribe accounting tests that assert exact
  `bytes_reserved` values.

## Result

Identical criterion settings both runs (`--sample-size 20 --warm-up-time 2
--measurement-time 10`):

| metric | baseline `d39d8c02` | lever `47b4d704` |
|---|---|---|
| quill docs/s | 19,402.68 (cv 1.46%) | **53,512.27** (cv 2.72%) |
| tantivy docs/s | 146,346.48 (cv 10.23%) | 151,229.40 (cv 8.21%) |
| **quill/tantivy** | **0.13161** (cv 11.48%) | **0.35408** (cv 9.54%) |
| A/A null (tantivy/tantivy) | 1.00246 — admissible | 0.99679 — admissible |

- **Quill arm: 2.76x faster** — 51.5 → **18.7 µs/doc**.
- **Incumbent gap: 7.60x → 2.82x slower.**
- The two Tantivy arms agree within **3.3%** across runs, and both A/A nulls are
  admissible, so the 2.76x is roughly **80x the cross-run drift**.

**Verdict: KEEP.**

### Side effect: a prior discrepancy resolves

An earlier reading of this cell put Quill at 64.2 µs/doc while P13's scale curve
reported 17.3 µs/doc — a 3.7x disagreement on the same arm between harnesses,
flagged as unexplained. With the rescan removed Quill sits at **18.7 µs/doc**,
consistent with P13. The gap was this bookkeeping, whose cost scales with
*vocabulary and arena size*, not with the quantities P13 varied.

## Honest limits

- **Cross-run A/B.** Baseline and lever are two ELFs in two invocations, not one
  interleaved paired run. The controls are the incumbent arm (3.3% agreement)
  and two admissible A/A nulls, not a same-invocation A/B of the two Quill
  builds. A campaign-class claim needs the interleaved form.
- **DIAGNOSTIC-CLASS**, so no gate/ratchet movement.
- The remaining **2.82x** gap is unexplained by this lever and needs its own
  profile — the top entry is now something else.
- QG-2 pins `threads=1`; nothing here says anything about thread scaling.

## Route next

1. **Re-profile.** The 26% entry is gone; rank the new top self-time entries and
   re-ask "does the incumbent pay this?" for each.
2. **Audit the whole `bytes_*` family for the same defect** — this was a
   sibling-path miss, and sibling-path misses cluster. Any other accessor whose
   cost scales with vocabulary/arena and is called per document is suspect.
3. **Shard fan-out (P15, 1b)** remains the largest *architectural* lever (9.27x
   phase ceiling) but its Amdahl bound is now smaller: the serial per-document
   cost this commit removed was inflating the parallelizable share.
