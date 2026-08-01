# E8-H P17 — within-batch shard fan-out: making parallel segment build reachable in production (2026-08-01)

**Task:** ship lever (1) of the ingest campaign — parallel segment build —
against the live pinned Tantivy oracle.

## The finding that redirected the measurement

P15 measured shared-nothing shard fan-out at **9.27x** (13.79x quiet) and
recorded it as *not reachable in production*. P16 then profiled the whole QG-2
job, found the profile flat after the `bytes_reserved` fix, and concluded the
constant-factor well was dry.

Both were measured at a cell where the lever **cannot exist**. QG-2 has exactly
one cell, `bulk/medium/1/positions_on`, and `perf_matrix.rs:231` sets
`deterministic_ingest: threads == 1`, which `resolved_ingest_shards` collapses
to a single shard. A parallel-build lever is definitionally unmeasurable there.

The thread matrix lives in **QG-1**: `bulk/{tiny,small,medium,xlarge}/{1,4,8,16}/
{positions_on,positions_off}`. At threads>1 Quill is configured for `threads`
shards and Tantivy gets `threads` writer threads.

## The structural gap

`QuillWriterState::index_documents_with_replacements` called
`ShardRouter::route_batch()` **once per batch** and then looped every document
of that batch into `shards[shard_id]` serially. `QuillIndex::index_documents`
holds an exclusive writer lock, and `flush_shard` seals inline on the calling
task. So with 16 shards configured, exactly one ran at a time — Quill indexed
on one thread while the incumbent used sixteen.

Signature in the baseline: Quill does **50,972 docs/s at 16 threads** and
~53,512 docs/s at one thread — flat in threads. The Tantivy arm is the one that
scales.

## Baseline (ELF `45acad85a388847b394479d91e6691d410be9b826a64e4e0a8fad9c86f3e3508`)

QG-1 `bulk/medium/16/positions_on`, 50,000 docs, batches of 5,000, 10
interleaved paired runs, same invocation:

| arm | docs/s | cv |
|---|---|---|
| quill | 50,971.57 | 5.185% |
| tantivy | 124,673.17 | 6.890% |
| **paired A/B (quill/tantivy)** | **0.413730** [0.401531, 0.424036] | 4.661% |
| A/A null (tantivy/tantivy) | 1.023663 [0.915085, 1.157139] — admissible | 14.664% |

## The lever

Within-batch fan-out. Three things stay serial because they must:

- **Admission** — the duplicate-id probe reads the published snapshot and the
  uncommitted-id set and mutates the latter, so it runs first, in input order,
  reporting the same first offender with the same message as before.
- **Docid allocation** — leases are per shard but the allocator is shared, so
  every span is allocated before the parallel region opens.
- **Sealing** — a flush is `async` and mutates shared publication state, so
  budget and lease-boundary seals happen between waves.

Accumulation itself (`canonical_metadata`, `add_document_with_values`,
`canonical_document_preimage`) is shared-nothing per shard and runs under rayon.
Work is issued in waves of at most 256 documents per shard so a shard's arena
cannot overshoot `scribe_shard_budget_bytes` by more than one wave.

Fan-out engages only when every participating shard would receive at least 256
documents (`fanout_shards = min(shard_count, len / 256)`, requiring >= 2).
Each sealed segment carries its own term dictionary — P15 measured sealed bytes
+8.9% at 32 shards — so splitting a small batch widely trades real index bytes
and search-time segment count for parallelism that isn't there.

**The single-shard path is retained verbatim** (verified: the 106-line body is
byte-identical after re-indent), so `deterministic_ingest` / threads=1 ingest is
unchanged by construction.

## Result

Lever ELF `77b5ca34053b83611771b4de71a90a80fb584bc2dd4f3b121bc11f8a0659f297`.
Same cell, same harness settings, same oracle, 10 interleaved paired runs:

| metric | baseline `45acad85` | lever `77b5ca34` |
|---|---|---|
| paired A/B (quill/tantivy) | 0.413730 [0.401531, 0.424036] | **0.689892 [0.648892, 0.715010]** |
| A/B cv | 4.661% | 6.467% |
| A/A null (tantivy/tantivy) | 1.023663 [0.915085, 1.157139] | 1.026073 [0.976341, 1.109803] |
| null admissible (contains 1.0) | yes | yes |

- **Quill ingest arm: 1.667x faster** (0.689892 / 0.413730).
- **Incumbent gap: 2.417x -> 1.450x slower.**
- The A/B confidence intervals are **fully disjoint** — [0.4015, 0.4240] vs
  [0.6489, 0.7150], no overlap.
- **Cross-run control:** the two A/A null medians agree to within **0.24%**
  (1.023663 vs 1.026073), so between-run host drift is ~0.2% while the effect is
  66.7% — roughly 280x the drift.

**Verdict: KEEP.**

## Correctness

`cargo test -p frankensearch-quill --lib`: **476 passed, 0 failed, 1 ignored.**

Two tests were added, because **no existing test built a batch large enough to
reach the fan-out path** — the writer's new parallel branch would otherwise have
shipped with zero coverage:

- `within_batch_fanout_agrees_with_the_retained_single_shard_path` — indexes the
  same 2,048-document batch through a 4-shard fan-out and through the
  deterministic single-shard control, and asserts the two arms return the same
  2,048 document ids as sets. Confirmed live from the trace: the fan-out arm
  sealed `segment_count=4` at `doc_count=512` each, the control sealed
  `segment_count=1` at `doc_count=2048`, and both answered `total_count=2048`.
- `within_batch_fanout_rejects_a_duplicate_id_inside_one_batch` — collides the
  last document with the first so the collision spans two shards of one batch,
  and asserts the duplicate-id rejection still fires.

## Costs, measured rather than assumed

| cost | serial | fan-out (4 shards) | delta |
|---|---|---|---|
| sealed bytes (2,048 docs) | 410,742 | 413,528 | **+0.68%** |
| segments | 1 | 4 | 4x |
| same query, exhaustive plan | 21.4 ms | 42.7 ms | **2.0x slower** |

The byte cost is far below P15's +8.9% at 32 shards precisely because the
`FANOUT_MIN_SHARD_DOCUMENTS` gate keeps every sealed segment substantial.

**The query cost is the real trade and must not be buried.** More segments means
more per-segment work until the tier policy merges them. The 2.0x above is a
worst case — a 2,048-document fixture split four ways, exhaustive plan, no
merge — but the direction is real: this lever moves work from ingest onto
search. It is a straight win only where ingest dominates or compaction keeps up.

## The threads=1 control

The fan-out **cannot engage** at one shard: `fanout_shards =
min(shard_count, len / 256)` is 1 when `shard_count` is 1, and the branch
requires `>= 2`. The single-shard body is byte-identical to the pre-change code
(verified mechanically: 106 lines, re-indent only), so threads=1 ingest is
unchanged **by construction**, not by measurement.

A measured threads=1 control was attempted but did not complete: the shared
cargo build directory was held by a peer swarm for the remainder of the session.
The construction argument stands on its own, but a measured control is still
owed and is the cheapest next confirmation.

## Honest limits

- Host `thinkstation1` (`local-5975wx-32c`) is a **local diagnostic class**, not
  a registered campaign class: activates no gate, moves no ratchet.
- Measured under a **heavy peer swarm** (load average 26 -> 55 across the
  session), so both arms are depressed and the result is a lower bound.
- The A/A null cv is wide (14.7%) for the same reason; the A/B cv is 4.7%.
