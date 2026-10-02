# Q1: The Docid-Range Discipline (Merge = Concat)

**Status:** Normative. **Owning bead:** `bd-quill-e0-contracts-j53p.3`. **Design of record:** `COMPREHENSIVE_PLAN_FOR_THE_QUILL_LEXICAL_ENGINE.md` §7 (as amended). **Companion:** `docs/contracts/fslx-format-registry.md` (encodings this discipline relies on).

Q1 is the structural bet that makes Quill's segment merges near-free: if docids are allocated so that merging segments never requires renumbering, merge collapses to a streaming concatenation — killing the decode→re-encode merge tax that dominates tantivy's write amplification (and the entire `force_merge_bounded`/cooldown apparatus that grew around it).

---

## 1. The invariant

**Q1 (docid-range discipline).** Every segment S carries a half-open global docid interval `range(S) = [docid_lo, docid_hi)` in its FSLX header. At all times:

- **Q1-a (interval disjointness):** the covering intervals of a manifest generation's live segments are pairwise disjoint.
- **Q1-b (containment):** every posting, tombstone, DOCLEN/IDMAP/STOREDMETA entry, and NUMERIC pair in S refers only to docids in `range(S)`.
- **Q1-c (internal order):** within any posting list, docids are strictly ascending.
- **Q1-d (uniqueness):** a docid, once allocated, is never allocated again — whether used, burned, or tombstone-folded.

Holes *inside* an interval (burned lease tails §2, compaction-dropped docs §5) are legal: they are simply docids that no artifact references.

## 2. Allocation

- **Keeper owns a monotone allocator.** The high-watermark persists in the MANIFEST (`docid_high_watermark`); a writer session loads it at `open_writer()` and never hands out a docid below it.
- **Session leases.** Each ingest shard leases contiguous docids inside one **65,536-docid block** (one lease block = one `chunk_id` in the tombstone encoding — deliberate alignment; lease ordinals are block-relative). A lease starts at the watermark — inside its block when the watermark is unaligned — or at the next block boundary (below). Leases belong to a shard **session**; watch-mode batches within a session reuse the shard's current lease.
- **Frontier reservation.** The most recently granted lease (the *frontier* lease) reserves only the docids it issues, so the watermark is the exact allocation frontier. Before another shard's lease may start, the frontier lease claims the rest of its block (the watermark jumps to the block end) and the new lease starts at that boundary, so concurrently live leases are disjoint. bd-k07zw / #41: short sessions — one per fsfs watch publication, delete, or append, or an incremental library indexer — used to reserve a whole block each, which merges stored as dense DOCLEN/IDMAP holes.
- **Burn on end.** When a session ends (or a shard is retired), the claimed but unissued tail of each live lease other than the frontier lease is **burned**: the watermark does not roll back, the docids are never reused (Q1-d). A session that writes through one shard burns nothing.
- **Upserts allocate new docids.** An upsert = allocate a fresh docid from the shard's lease + tombstone the old docid. IDHASH probes visit current segments in descending `seal_seq`, verify exact IDMAP bytes, and continue after a tombstoned representative; a merge republishes older rows under a fresh `seal_seq`, so container order is not physical-update recency. At most one live docid per `DocId` string (the upsert invariant).
- **Docid width:** u32 in all payloads (u64 in headers/manifest for future-proofing). Consumption is one docid per indexed row plus burned tails; a session that spreads batches over k shards can burn up to k − 1 blocks, while a session through one shard consumes only what it indexes. A renumbering **deep compaction** is reserved (format-registry note) as the u32-exhaustion escape hatch; it is never expected to run and is not implemented at 1.0.

## 3. The two rules that keep intervals disjoint (R1, R2)

Shard sessions acquire lease blocks alternately, so one shard's later leases interleave with other shards' blocks in docid space. Two rules prevent that interleaving from ever producing overlapping covering intervals:

- **R1 — seal at lease boundaries.** A sealed segment's docid range is always a subinterval of a **single lease block**. Lease exhaustion forces a segment cut (columnar flush *and* delta seal — both paths) even if the arena/delta budget is not reached. Consequence: mini-segments are ≤ 65,536 docs, which the arena budget almost always enforces earlier anyway.
- **R2 — merge only bound-consecutive runs.** The tier policy sorts live segments by `docid_lo` and merges only **consecutive runs** in that order. Cross-shard merges are the normal case and cost the same (concat is concat). The merged segment's covering interval is exactly the union-hull of an uninterrupted run, so it cannot contain any other live segment's interval.

*Why both are needed:* without R1, a segment spanning two leases of shard A could straddle shard B's interleaved block; without R2, merging A's two segments while skipping B's interleaved one would produce a hull containing B's interval. Either way Q1-a dies. With both, interval disjointness is preserved for the life of the index by induction over seals, merges, and compactions.

## 4. The theorem (merge is concatenation)

**Claim.** For live segments S₁…Sₙ forming a bound-consecutive run sorted by `docid_lo` (R2 precondition), and any term t: `postings_M(t) = postings_S₁(t) ⧺ … ⧺ postings_Sₙ(t)` is already docid-sorted, and M with `range(M) = [lo₁, hiₙ)` satisfies Q1.

**Proof sketch.** By Q1-c each `postings_Sᵢ(t)` is ascending; by Q1-a + the run's ordering, every docid in Sᵢ is strictly less than every docid in Sᵢ₊₁; concatenation of ascending sequences over ordered disjoint ranges is ascending (Q1-c for M). Q1-b holds since `range(M)` contains each `range(Sᵢ)`. Q1-a holds by R2 (the hull contains no other live interval). Q1-d is untouched (no allocation). ∎

**What merge actually does** (all streaming, no posting decode):

1. K-way merge the n TERMDICTs by composite key bytes (galloping over prefix blocks).
2. Per term: copy posting-block bytes **verbatim and in input order** (FOR blocks store `first_doc` as an absolute u32 — no rebase exists to do). Every FSLX posting block is self-delimiting; a source stream's partial VINT block therefore remains legal at an interior segment seam. Ordinary merge never shifts later 128-posting boundaries.
3. BLOCKMAX: re-emit entries in the same block order, adjusting each `block_offset` relative to the merged term's posting stream while preserving `first_doc`, `max_freq_q`, and `min_fieldnorm`. No seam entry is synthesized because no posting block is rewritten.
4. DOCLEN/IDMAP/STOREDMETA: concatenate into the merged positional span (the gap between `hiᵢ` and `loᵢ₊₁` materializes as holes; see §4.1). IDMAP remains a physical positional witness and may therefore retain duplicate DocId bytes for an old tombstoned row and its replacement.
5. NUMERIC: k-way merge each field's `(value, docid)` pairs by their canonical value-then-docid order, preserving absolute docids. Raw segment-order concatenation is forbidden because disjoint docid ranges do not imply disjoint or ordered value ranges.
6. STATS: sum. Tombstones: union (carried, not folded — folding is compaction §5). IDHASH: rebuild rather than concatenate. For each exact DocId, apply the merged tombstones and select its unique live physical row; multiple live rows violate the upsert invariant and fail merge. If every row for that ID is tombstoned, retain the lowest global docid as a deterministic completeness representative. Translate selected rows into merged segment-relative ordinals, sort those ordinals ascending, and insert them into the canonical linear-probe table. This live-first rule is mandatory for merge-of-merges: a newly published high-`seal_seq` container may carry an older dead row while a lower-`seal_seq` source carries the live upsert. The rebuild never treats container `seal_seq` or global docid order as live-row recency.

Expected cost: sequential read + sequential write; I/O-bound. The e3.5 bench asserts CPU/byte stays ~flat as segment count grows.

**Why interior partial blocks are intentional.** A tail-only canonical grammar
cannot satisfy verbatim concat. For example, `df(A)=100` and `df(B)=300` would
group monolithically as `128/128/128/16`; filling A's tail consumes 28 postings
from B and shifts every later B boundary. FSLX instead preserves
`partial(100)/full(128)/full(128)/partial(44)`. Decoded postings, stats, and
queries are identical to the monolithic build, while every copied input block
is byte-identical and ordered. Each freshly sealed leaf segment contributes at
most one partial block per term. A merge output preserves exactly the leaf
partials already present in its inputs; it may contain multiple partial blocks,
but repeated merge schedules create none. Optional reblocking is reserved for explicit deep compaction, where
the CPU/storage trade is measured rather than hidden in ordinary merges.

### 4.1 Positional sections across run gaps

A bound-consecutive run may still have docid gaps *between* segments (burned tails within the same lease-block sequence). Merged positional sections (DOCLEN/IDMAP/STOREDMETA) index by `docid − lo₁` over the merged span and therefore materialize those gaps as holes (FSLX §5.5/§5.6 hole conventions). Space cost is bounded: gaps inside a run come only from burned tails of non-frontier leases, which R1 caps at sub-lease size per seam; a session through one shard leaves none. The tier policy MAY additionally decline runs whose hole ratio exceeds a threshold (`merge_max_hole_ratio`, default 0.5) — a policy knob, not a correctness requirement.

## 5. Compaction (the only re-encoding path)

Compaction rewrites one segment to fold its tombstones: surviving docids are **preserved** (no renumbering — gaps are fine, FOR deltas absorb them), dropped docids become holes. Triggered per-segment at tombstone density > `compaction_tombstone_density` (default 0.20). Obligation Q1-OB4 (§6). Deep compaction (renumbering) remains reserved and unimplemented (§2).

## 6. Obligations (each becomes a gauntlet fixture)

| ID | Obligation | Where enforced/tested |
|---|---|---|
| **Q1-OB1** | Manifest validation rejects overlapping covering intervals; `merge()` asserts bound-consecutive inputs (R2); seal asserts single-lease spans (R1) | e3.2/e3.3 validation + e3.5 assertion + e3.9 crash matrix |
| **Q1-OB2a** | Decoded postings, aggregate stats, and query results from `merge(S₁…Sₙ)` equal monolithic indexing of the concatenated document stream with the same docid assignments | e3.5 fixture, including `df=100 + df=300` (bridges through G1a's direct accumulator-to-seal reference path) |
| **Q1-OB2b** | Every input posting block appears byte-identically and in source order in the merged term stream; merge creates no new posting fragments | e3.5 raw-block witness + arbitrary merge-schedule property test |
| **Q1-OB2c** | Merged IDMAP is the gap-preserving positional concat, while IDHASH contains exactly one representative per exact DocId: the unique live row after tombstones, or the lowest global docid when all rows are dead. Sorting happens only after that choice; a high-`seal_seq` merge container carrying a stale dead row cannot mask a lower-`seal_seq` live row | e2.6 external-representative codec regression + e3.5 two-stage merge/reopen fixture |
| **Q1-OB3** | Query results are invariant under ANY merge schedule (BM25 stats are snapshot-level aggregates) | e3.5 random-merge-order property test; e6.3 metamorphic suite |
| **Q1-OB4** | Compaction preserves surviving docids; compacted segment is query-identical to the tombstone-paired original | e3.6 fixture (measured at realistic densities: 5/20/50% — never 0%) |
| **Q1-OB5** | Lease disjointness under concurrent sessions; burn-on-end accounting; R1 cut on lease exhaustion (both flush and delta-seal paths) | e1.6 property tests + e5.4 note |
| **Q1-OB6** | After arbitrary interleaved multi-shard ingest + random policy merges, intervals remain pairwise disjoint | e3.7 property test |

## 7. Interactions and non-interactions

- **Tombstones** live in the manifest (FSLX §6.3), keyed by absolute docid; merge unions them; Q1 untouched.
- **Upserts** never mutate segments; they tombstone + re-add under a fresh docid (§2), so Q1-c/Q1-d hold trivially.
- **The delta segment** allocates from the same shard lease; R1 applies to its seals identically (bead e5.4 note).
- **Resumable bulk builds** (bd-quill-duel-resumable-bulk) take *fresh* leases on resume, starting at the persisted watermark; skipped (already-present) docs keep their existing docids; Q1-d is preserved because nothing re-allocates. Result-level equivalence to an uninterrupted build holds by Q1-OB3 logic (docids differ by burn; scores/results don't).
- **Cross-process writers** (bd-quill-duel-writer-lock): the single-writer lock makes the allocator single-owner per generation; the publish CAS prevents a stale writer from publishing a manifest whose watermark/leases raced. Q1 needs *exactly one live allocator* — the lock provides it.
- **BM25 scores** never depend on docid values (only on per-term stats and per-doc norms), which is why burn-gaps and merge schedules are score-invisible (Q1-OB3's foundation).

## 8. Failure modes this discipline forecloses (design rationale)

1. **Merge renumbering** (tantivy's tax): impossible by construction — nothing ever renumbers.
2. **Sort-order corruption from interleaved shards:** R1+R2 induction (§3).
3. **Docid reuse after crash:** the watermark persists in the manifest and only moves forward; a crashed session's burned tail stays burned (recovery never rolls the watermark back — validated by the e3.9 kill-point matrix).
4. **Two allocators after writer takeover:** LOCK + generation-claim CAS (FSLX §7.1–7.2) serialize allocator ownership across processes.

*Changes to this document follow the format-registry discipline: amend, version the change in the FSLX registry when encodings are affected, and land the fixture in the same commit.*
