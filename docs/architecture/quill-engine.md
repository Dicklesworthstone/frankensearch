# Quill Engine — Architecture Summary

Maintained summary of the native lexical engine (`crates/frankensearch-quill`,
`crates/frankensearch-quill-gauntlet`). Distilled from the crate sources and
committed evidence; the per-episode design records remain the design of record
where they conflict. Every number below cites a committed receipt or the perf
ledger — nothing from memory.

Status: Quill is the default lexical backend everywhere. The facade flip is
`d117ce1f` (`lexical = ['quill']`), guarded by the CI QG-10 conformance guard
(`19fdf98a`), with runtime identity pins (`72a6ade8`, `9d145a14`) and the
published `frankensearch 0.4.x` line. `lexical-tantivy` is the pinned Tantivy
oracle/interop lane, never a runtime fallback. `cass-compat` is the foreign
schema-v8 carve-out. Owner ruling 2026-09-01 (recorded on
bd-quill-e7-integration-flip-d0tx).

## Module Map

| Module | Role |
|---|---|
| `schema`, `config`, `contract` | Shared schema, scoring, and budget contracts (scoring bound math lives in `contract.rs`) |
| `scribe` | Ingest-side encoding: accumulators, term-stream partitioning, FSLX section assembly for fresh and delta segments |
| `quiver` | Posting-list codec: fixed-size posting blocks plus per-block block-max entries |
| `grimoire` | FSLX framing: section kinds, TERMDICT encode/decode, per-term reference validation |
| `keeper` | Segment lifecycle: admission, compaction (re-encode), concat-merge, durability |
| `delta` | Delta segments: upsert/delete chains, tombstones, live-row visibility, conservative term bounds |
| `argus` | Query execution: sealed cursors, BM25 scoring, block-max/MaxScore pruning tiers |
| `index` | Orchestration: seal/flush lifecycle, open/recovery, clause-shape lowering, blue-green CURRENT resolution |
| `segment`, `stats`, `snippet`, `query`, `cass` | Segment handles, field statistics, snippet extraction, query AST, CASS interop |
| `tracing_conventions`, `error` | Span naming and typed errors |

## On-Disk Format (FSLX)

A segment is one framed file of named sections: `TERMDICT`, `POSTINGS`,
`POSITIONS` (optional), `BLOCKMAX`, `DOCLEN`, `IDMAP`, `IDHASH`. The TERMDICT
carries per-term spans into the other sections, and every decoded term is
contiguity-checked against the running section cursors
(`grimoire::ReferenceValidator`): a span must begin exactly where the previous
term ended and the section must end exactly at its planned length. Open
admission re-validates section lengths before any term is served
(`keeper` admission path).

## Write Path

Ingest accumulates postings per field, partitions terms deterministically
(stable radix partition), and encodes ordered term streams — postings,
optional positions, block-max bytes, TERMDICT inputs — into an FSLX segment
(`scribe::flush_accumulator` and the staged variant). Updates land in delta
segments: per-term chains of inserts/tombstones with generation-resolved
visibility. A later upsert or delete can only widen (never tighten unsafely)
the retained bounds used for pruning.

## Read Path and Pruning

`argus` opens sealed cursors over TERMDICT entries. Cursor flavor follows the
metadata: docs-only, positions, or block-max-carrying. Query shapes lower to
union kinds with distinct pruning budgets (`index.rs`):

- Direct-term unions: term-granular `MaxScore` for 2–8 clauses, block-max WAND
  (BMW) for 9+.
- Grouped unions (multi-field term groups): deferred grouped `MaxScore` only —
  a group owns no physical block-max list, so wider grouped roots stay
  POSTINGS-only rather than paying for metadata nothing reads.

Delta-visible terms resolve live doc freq and a conservative
`DeltaBlockMax` against the merged view of base plus delta.

## Where Block-Max Is Computed

The single answer the docs pass is tuned for: **per-block block-max entries are
minted at posting-encode time in `quiver::EncodedPostingList::encode_with_block_max`**
(one entry per `POSTINGS_PER_BLOCK` posting block), and the score bound math is
**`contract::block_max_score` / `block_max_tf_factor` / `block_max_frequency_to_code`**
(frequency encoded exactly up to 254, `u8::MAX` meaning "≥255", monotonically
conservative on decode). Delta segments carry the same shape as
`delta::DeltaBlockMax`, and `argus::SealedCursor` consumes the entries via
`supports_block_max` / `current_block_score_upper_bound` during WAND-style
pruning. Compaction re-derives entries from surviving rows
(`keeper::compact_terms` → `encode_with_block_max`); concat-merge copies
same-shape streams without re-encoding.

## Lifecycle, Durability, Visibility

Segments live in blue-green engine directories under an atomically swapped
`CURRENT` pointer; readers resolve the active root per open and foreign or
damaged layouts are typed errors. `fsfs` rebuilds the lexical index from
canonical storage on detection of a legacy/foreign layout (rebuild-on-detect,
quill-e7.5) — there is no on-disk format migration. Compaction preserves
per-segment RaptorQ-style repair sidecars and re-protection is part of the
compaction contract (see `fsfs doctor`'s `durability.*` checks).

## Conformance and Performance Evidence

`frankensearch-quill-gauntlet` is the differential witness: identical corpora
through Quill and the pinned `lexical-tantivy` oracle, compared on full result
contracts (ordering, metadata, snippets) plus perf evidence assembly and a
ratchet (`runner.rs`, `perf.rs`, `perf_ratchet.rs`, `perf_evidence.rs`). The
crate is excluded from the workspace lib-test lane and runs its own binaries; the quality gate
exercises the Quill-consumer lanes against it.

Committed performance evidence for lexical work on real corpora:
`docs/evidence/perf/fsfs-latency-20260903-thinkstation1.json` (Quill lexical
search p50 0.23 ms on a 1,000-document corpus; `fsfs index` 1,000 files in
14.1 s including both vector tiers; watch-mode event-to-applied p50 725 ms) and
`docs/evidence/perf/library-two-tier-latency-20260903-thinkstation1.json`.
Attribution detail: `docs/quill-perf-attribution.md`; ledger:
`docs/PERF_LEDGER.md`.
