# E8-H Round-0 profile card — QG-2 cell on trj (partial P1 deliverable)

**Bead:** `bd-e8h-p1-profile-trj-ykx4` (Round-0 slice: QG-2 single-worker cell only;
QG-1 multi-thread and QG-6 100k cards remain).
**Method:** `perf record -F 499 -g --call-graph dwarf` over the full perf_matrix
process, QG-2 `bulk/medium/1/positions_on`, warmup 1 + 10 runs, ELF at source
`5433c45e` (contains the `ebd91757` terminal-join fairness fix), release-perf,
governor=performance, run receipt `trj-zen3-64c/20260728T234926Z-p1-qg2-perf-record`,
perf.data (219 MB) at `trj:~/.frankensearch-perf-runs/p1-trj-qg2.perf.data`.
**Scope caveat (do not skip):** process-wide samples, NOT timed-window-scoped —
harness corpus generation shares the main thread with the Quill arm. Generator
pollution is measured and subtracted narratively below; a dhat census plus a
window-scoped rerun refine before any KEEP cites this card.

## Namespace aggregates (self-time, whole process, 145 frames ≥0.05%)

| namespace | self-time |
|---|---|
| `tantivy::` + `frankensearch_lexical::` (tantivy arm, own threads `thrd-tantivy-in` + `docstore-compre`) | **32.7%** |
| `frankensearch_quill::` | **9.2%** |
| `frankensearch_quill_gauntlet::generator::` (harness corpus, main thread) | 5.6% |
| everything else (libc alloc/memmove, core::fmt, serde_json, hashbrown, std) | 41.7% |

## The inverted-attribution finding

Both arms index identical document streams and Quill measures ~2.9x slower per
document (quarant-adjacent trj artifact 0.350; post-fix m4 diagnostic 0.374) —
so Quill's timed work should DOMINATE the profile. It does not. Quill's cycles
sit in non-namespaced generic-runtime frames on its thread:

- **allocator self-time on the main (Quill+harness) thread: ~7.6%** —
  `_int_malloc` 2.80 + `malloc_consolidate` 1.49 + `_int_free_chunk` 1.49 +
  `unlink_chunk` 0.66 + `__libc_malloc2` 0.64 + `cfree` 0.52. The Tantivy
  indexing threads show almost no allocator frames (arena-based recording).
- **`__memmove_avx_unaligned_erms` on the main thread: 3.96%** (copies).
- `hashbrown::HashMap<String, String>::insert` 1.08% +
  `RandomState::hash_one::<&u64>` 0.80% (SipHash on u64 keys — default-hasher
  map in the ingest path or harness doc structures; identify exact owner).
- `core::fmt` block (~13.4%) + part of serde_json: **traced to the harness
  generator** (`SyntheticCorpus::document_at → write_fmt<String>` — caller
  graph verified), NOT to Quill. Excluded from lever hypotheses.

Quill-namespaced frames ≥0.1% (the visible algorithm tips):

| frame | self% |
|---|---|
| `scribe::FrankensearchTokenizer::analyze` | 2.26 |
| `index::canonical_document_preimage` | 0.96 |
| `scribe::TermInterner::find_in_bucket` | 0.55 |
| `scribe::append_canonical_term` | 0.55 |
| `quiver::EncodedPositionList::encode_with_limits` | 0.53 |
| `scribe::stable_digit_scatter` | 0.53 |

Canonicalization family (`canonical_document_preimage` + `append_canonical_term`
+ `stable_digit_scatter`) ≈ 2% — worth a row of its own: Quill canonicalizes
document content at ingest in ways Tantivy's pipeline does not.

## Consequences for the workstreams

1. **W2.2 (postings-accumulation / allocation churn) is now the top-ranked
   lever**: the 7.6% allocator + 4% memmove signature on Quill's thread against
   near-zero on Tantivy's arena threads is the strongest measured signal in the
   card. dhat census next to name the allocation sites (per the bead's
   profile-first gate — this card + dhat together satisfy it).
2. **W2.1 (interner)**: `find_in_bucket` 0.55% is visible but modest;
   the `hash_one::<&u64>` SipHash frame may also be interner-adjacent —
   owner identification folded into the dhat pass.
3. **New row candidate: ingest canonicalization cost** (~2% family) — file
   after dhat confirms it isn't generator-shadowed.
4. W2.4 already resolved for this fixture (zero IO both arms, in-memory by
   construction).

Cross-check note: the Tantivy tokenizer adapter
(`FrankensearchTokenStream::advance`, 7.07%) is the single hottest frame in
the process — on the TANTIVY arm. Both engines pay a tokenizer; Quill's
`analyze` shows only 2.26% — consistent with Quill's cost being downstream of
tokenization, in per-term/per-doc memory traffic.
