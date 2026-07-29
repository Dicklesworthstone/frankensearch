# Ingest-side allocation census — supplementary evidence for bd-6oiq / W2 rows

**Agent:** MossyPine · **Date:** 2026-07-29 (UTC) · **Status:** attribution evidence only —
no paired A/B, no perf claim, no W2 row edits (those belong to their lane owners).
**Source:** origin/main `f174befb` extract; same scratchpad harness family as
`docs/evidence/e8h/w13-collector-alloc-census-20260728/` (see its ANALYSIS.md for method
and host). dhat profiles ONLY `index_documents` + `commit` (corpus built outside the
window): 100,000 docs, 20 commits, `QuillIndex::in_memory`, `deterministic_ingest`,
single thread.

## Headline

**2.71 GB / 9.71 M blocks allocated to ingest 100k docs = 27,070 bytes and 97.1
allocator blocks per document** (QG-1 `bulk/medium/1`-like shape).

## Top allocating sites (by bytes; full data in `dhat-ingest-hundredk.symbolized.json`)

| % bytes | bytes | blocks | site |
|---:|---:|---:|---|
| 15.28 | 413.7 MB | 300,000 | `index::canonical_document_preimage` (3 blocks/doc) |
| 11.01 | 297.9 MB | 40 | `scribe::FlushTokenRow` reserve (batch-level, benign) |
| 8.10 | 219.2 MB | 2 | `segment::SegmentAssembler::new` |
| 7.19 | 194.7 MB | 20 | `segment::EncodedSegment::encode_with_limits_impl` |
| **7.19** | **194.7 MB** | **20** | **`segment::EncodedSegment::clone` — one full ~9.7 MB segment clone per commit** |
| 5.66 | 153.2 MB | 100 | `keeper::append_concat_bytes` |
| 5.15 | 139.4 MB | 20 | `quiver::EncodedStoredMetaSection::encode_accumulator_with_limits` |
| 4.77 | 129.1 MB | 860 | `scribe::append_span` |
| 3.95 | 106.9 MB | 225 | `grimoire::OwnedTerm` vec growth |
| 3.62 | 97.9 MB | **742,055** | `quiver::PostingBlockMeta` vec growth |
| 3.59 | 97.2 MB | **381,995** | `quiver::Posting` vec reserve |
| 2.04 | 55.2 MB | **1,086,298** | `quiver::write_vint` |
| 1.82 | 49.3 MB | 281,995 | `scribe::build_term_rows` |
| 1.32 | 35.8 MB | **844,734** | `quiver::EncodedPositionList::encode_with_limits` |
| 1.27 | 34.4 MB | **635,357** | `quiver::append_block` |

## Readings (hypothesis-relevant, not verdicts)

1. **W2.2 (postings accumulation growth policy) is supported on the block axis:** the
   quiver posting/position/vint paths account for ~3.7 M of 9.7 M total blocks —
   exactly the small-alloc churn its chunked-arena hypothesis targets.
2. **New unlisted candidate:** `EncodedSegment::clone` — a full encoded-segment clone
   per commit (7.2% of ingest bytes in 20 allocations). Worth a "why is this cloned"
   reading pass before any W2 lever lands; if it is avoidable, it is nearly free.
3. **`canonical_document_preimage` is the single largest byte site (15.3%, 3
   blocks/doc)** — a per-document canonicalization buffer that looks reusable across
   documents within a batch.
4. Unlike the query side (98% one frame), ingest allocation is SPREAD — no single-lever
   kill shot; consistent with the E8-H premise that the QG-1/QG-2 deficit needs several
   attributed levers.

Bytes/blocks here are allocator traffic, not wall-time; only paired A/B runs under the
QG harness can convert any of this into KEEP/REJECT verdicts.

## Appendix (2026-07-29): per-site lifetime / peak-retention table

Extracted from the same `dhat-ingest-hundredk.symbolized.json` (dhat v2 `tl`/`mb`/`gb`
fields; time unit µs; run length 24.06 s). Requested by YellowSparrow (msg 5585) to
complete the W2.2 profile-first gate.

| % bytes | total MB | blocks | avg B/blk | avg lifetime | peak-ret MB | @heap-max MB | site |
|---:|---:|---:|---:|---:|---:|---:|---|
| 15.28 | 413.7 | 300,000 | 1,379 | 238 ms | 13.4 | 0 | `index::canonical_document_preimage` |
| 11.01 | 297.9 | 40 | 7.4 MB | 330 ms | 15.0 | 0 | `scribe::FlushTokenRow` reserve |
| 8.10 | 219.2 | 2 | 109.6 MB | 7.26 s | 219.2 | 219.2 | `segment::SegmentAssembler::new` |
| 7.19 | 194.7 | 20 | 9.7 MB | **11.9 ms** | 9.8 | 0 | `EncodedSegment::encode_with_limits_impl` |
| 7.19 | 194.7 | 20 | 9.7 MB | **5.46 s** | 77.9 | 68.1 | `EncodedSegment::clone` |
| 5.66 | 153.2 | 100 | 1.5 MB | 82.8 ms | 50.8 | 50.8 | `keeper::append_concat_bytes` |
| 5.15 | 139.4 | 20 | 7.0 MB | 138 ms | 7.0 | 0 | `quiver::EncodedStoredMetaSection::encode…` |
| 4.77 | 129.1 | 860 | 150 KB | 22.9 ms | 3.3 | 0 | `scribe::append_span` |
| 3.95 | 106.9 | 225 | 475 KB | 285 ms | 28.3 | 28.3 | `grimoire::OwnedTerm` growth |
| 3.62 | 97.9 | 742,055 | 132 | **29.5 µs** | 0.02 | 0 | `quiver::PostingBlockMeta` growth |
| 3.59 | 97.2 | 381,995 | 255 | **23.4 µs** | 0.03 | 0 | `quiver::Posting` reserve |
| 2.04 | 55.2 | 1,086,298 | 51 | **1.1 µs** | ~0 | 0 | `quiver::write_vint` |
| 1.82 | 49.3 | 281,995 | 175 | 24.7 µs | 0.03 | 0 | `scribe::build_term_rows` |
| 1.72 | 46.6 | 22 | 2.1 MB | 628 ms | 9.7 | 8.0 | `grimoire::TermInput` reserve |
| 1.32 | 35.8 | 844,734 | 42 | **1.0 µs** | 0.01 | 0 | `quiver::EncodedPositionList::encode…` |
| 1.27 | 34.4 | 635,357 | 54 | **6.5 µs** | 0.01 | 0 | `quiver::append_block` |

Lifetime readings:
- **W2.2 (chunked-arena) is confirmed on all three axes:** the quiver posting/vint/
  position sites are ~3.7 M blocks of 42–255 bytes living 1–30 µs — allocate-use-free
  churn with near-zero retention, the exact shape an arena/chunk policy removes.
- **`EncodedSegment::clone` is retention, not churn:** the clone (avg life 5.46 s, 68 MB
  resident at global heap max) outlives its 12 ms encode-side original by ~450x — two
  full encoded segments coexist for the whole commit pipeline. Tracked separately as
  `bd-s1rc1`; primary payoff axes are allocator bytes + QG-7 peak RSS.
- `SegmentAssembler::new` (2 × 109.6 MB, 7.26 s, fully resident at heap max) is the
  dominant peak-RSS contributor in this in-memory fixture — worth a bounded-capacity
  reading pass when QG-7 work starts, but batch-level, not per-doc churn.
