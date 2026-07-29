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
