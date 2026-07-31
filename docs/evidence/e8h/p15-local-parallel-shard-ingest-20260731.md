# E8-H P15 — shard fan-out is a multiples-class ingest lever (KEEP); the parallel seal is a REJECT (2026-07-31)

**Task:** score the campaign's rank-1 architectural ingest lever — turning
Quill's shard partition from a *memory* partition into a *compute* partition —
against the serial shard fill that ships today.

**Answer.** Shared-nothing shard fan-out is the largest single ingest lever
measured in this campaign: **9.27x on the accumulate+seal phase at the
production shard count**, with an admissible A/A null, byte-identical sealed
output, and CPU/wall confirming the parallelism is real rather than a peer-load
artifact. The adjacent "free" lever — flipping the existing parallel radix seal
on — is a **REJECT**: it is a wash at low shard counts and a *regression* at
high ones.

## Disposition

`VALID-AB / DIAGNOSTIC-CLASS / SELF-SPEEDUP`.

Host `thinkstation1` is the local diagnostic class `local-5975wx-32c`, **not** a
registered campaign class: nothing here may activate a gate, move a ratchet, or
be quoted as a competitive ratio.

**This is a SELF-SPEEDUP (Quill vs Quill), not a competitive claim.** It does
*not* convert to the QG-2 Tantivy deficit and must never be quoted as closing
it. See "What this does NOT show" below.

## Provenance

| axis | value |
|---|---|
| Repo state | local `main` **`e2120608`** (after the P15 build fix `5116b352`) |
| ⚠ Branch divergence | this branch is **311 commits behind / 30 ahead of `origin/main`**. Every number here is measured on the stale local branch, not upstream. |
| Host | `thinkstation1`, AMD Ryzen Threadripper PRO 5975WX, **32 physical cores / 64 SMT threads** (`nproc`=64 is threads, not cores) |
| Toolchain | rustc 1.99.0-nightly (9f36de775 2026-07-19), pinned by `rust-toolchain.toml` |
| Cargo.lock | sha256 `73c06c36defef12eb5e1188881ce717127636d4c733d0dba7c9f38d9c4784257` |
| Bench ELF (run 1, scaling sweep) | sha256 `fc68601bfd39efe6793146ceb9cd23975b72a2ae5dc5ef76031c7c3986151778`, 811,104 B |
| Bench ELF (run 3, verdict cell) | sha256 `e889c5dcb40df315ee460fcda9aedb59012943123598cae166a27ea24cb5789d`, 811,384 B |
| Why two ELFs | run 3 followed commit `e2120608`, which changed only the bench's doc comment, default round count, and a new assert. **No measurement logic differs between them.** |
| Harness | `crates/frankensearch-quill/benches/parallel_shard_ingest_ab.rs`, 50,000 docs, 350 tokens/doc, fixed 8,192-term vocabulary, batch 250 |
| Engine source changed | **none.** Both levers are exercised through existing public `scribe` APIs. |

## Result — verdict cell (admissible null)

`QUILL_PSI_DOCS=50000 QUILL_PSI_SHARDS=32 QUILL_PSI_ROUNDS=11`, host loadavg
**60.18** (heavy peer load):

| arm | median | median CI95 | verdict |
|---|---|---|---|
| A/A null | 0.9862 | [0.9711, 1.0119] | **admissible** (contains 1.0, 11 rounds) |
| seal-automatic (1a) | 1.0361 | [1.0291, 1.0471] | **REJECT** — CI entirely above 1.0, 3.6% slower |
| shard-fanout (1b) | **0.1079** | [0.1035, 0.1247] | **KEEP — 9.27x, `PARALLEL_WINS`** |

CPU/wall: serial 6.131 s wall / 5.940 s cpu (0.97x) | parallel 0.634 s wall /
10.200 s cpu (**16.08x**) | cpu_overhead 1.717x.

The 16.08x CPU/wall is the load-independent witness that the lever genuinely
parallelises; the wall ratio alone could not distinguish that from a peer
releasing cores.

## Result — scaling sweep (run 1, loadavg 6.44)

⚠ Every `[decision]` line in this run printed **NOISE** because it was invoked
with `QUILL_PSI_ROUNDS=9`, below the `rounds >= 10` floor that
`PairedRatio::is_admissible_null` enforces. The *ratios* below are valid
paired measurements; their *verdicts* were not computable. This defect was in
the bench's own documented invocation and is fixed in `e2120608`.

| shards | shard-fanout ratio | wall speedup | CPU/wall | cpu_overhead | sealed bytes |
|---|---|---|---|---|---|
| 1 | 0.9941 | 1.01x | 0.98x | 1.012x | 249,092,150 |
| 2 | 0.5226 | 1.91x | 1.93x | 1.001x | 249,693,676 |
| 4 | 0.2711 | 3.69x | 3.35x | 0.986x | 250,904,472 |
| 8 | 0.1750 | 5.71x | 6.00x | 1.155x | 253,463,728 |
| 16 | 0.1137 | 8.80x | 12.23x | 1.447x | 258,735,584 |
| 32 | **0.0725** | **13.79x** | 26.11x | 1.993x | 271,194,688 |
| 64 | 0.0687 | 14.56x | 43.66x | 2.884x | 279,765,696 |

seal-automatic (1a) across the same sweep: 0.9706 / 0.9165 / 1.0191 / 0.9724 /
1.0036 / **1.0598** / **1.1178**. It degrades monotonically as shards grow,
because smaller per-shard row counts leave the parallel radix's Rayon overhead
dominating. **Reject it; do not flip `prepare_shard_flush` to `Automatic`.**

### Two costs the speedup does not pay for

1. **Total CPU grows.** cpu_overhead reaches 1.99x at 32 shards and 2.88x at
   64. Beyond the 32 physical cores the machine only has SMT threads left, so
   64 shards buys 5% more wall (14.56x vs 13.79x) for 45% more CPU. **The knee
   is at physical core count**, which coincides with today's
   `max_ingest_shards: 32` default.
2. **The index gets bigger.** Sealed bytes grow monotonically with shard count,
   **+8.9% at 32 shards and +12.3% at 64** vs a single shard, because each
   segment carries its own term dictionary. This is a real, permanent
   on-disk cost and must be weighed against the ingest win.

## What this does NOT show

- **Not end-to-end ingest.** The bench measures accumulate+seal in isolation
  through the `scribe` API. It does **not** include the exclusive writer lock,
  the per-document duplicate probe, doc-ord allocation, identity sidecar
  ordering, Delta/Keeper work, manifest staging, or commit. It is a **phase
  ceiling**, and Amdahl's law over that serial residue governs what production
  can actually realise.
- **Not a Tantivy comparison.** No incumbent arm ran. This cannot be quoted
  against the QG-2 deficit (2.07x pinned / 2.96x unpinned).
- **Not yet reachable in production.** `ShardRouter::route_batch` routes a
  *whole batch* to one shard, and `QuillIndex::index_documents` holds an
  exclusive writer lock with a serial per-document loop. Today the shards
  cannot run concurrently at all. Realising this lever requires routing
  *within* a batch and reconciling the serial residue — that is the
  implementation task this card justifies, not something it delivers.

## Route next

1. **Land 1b** — within-batch routing + parallel shard accumulation, keeping
   `FlushMode::Scalar`. Correctness obligations: doc-ord allocation, identity
   sidecar order, duplicate detection across shards, cancellation, and
   byte-identical sealed output (the bench already gates the last one).
2. **Measure the serial residue first.** The phase ceiling is 9-14x; the
   end-to-end win is whatever Amdahl leaves. Profile the residue before
   promising a number.
3. **`resolve_document_id` is O(S) per document** — `resolve_document_id_in`
   (keeper.rs:2985) walks *every* sealed segment with an IDMAP+IDHASH probe and
   never early-exits, because it must detect multiple-live ids. Tantivy has no
   unique-id constraint and does none of this. Measured at only **1.53%** of
   quill-arm self-time at N=50,000 (bead `bd-e8h-w2-u64-hasher-swap-vcfft`), so
   it is a **scaling** lever, not a constant-factor one — consistent with
   Quill's per-doc cost bending upward only above N≈128,000 (P13). Score it at
   `Xlarge`/QG-7, never against `medium`.

## Open question raised by this work

The gauntlet score-order lock from `f6b6864a`
(`harvested_14_score_bits_preserve_ranked_and_counted_orders`) **fails on this
branch** with a one-ULP divergence (left `1074094023`, right `1074094024`) —
the exact drift that commit existed to fix. The lock is absent from
`origin/main` under any name. It was deliberately **not** landed (see
`5116b352`) rather than landing a red test. Whether this branch's ranked
scoring is bit-divergent from the pinned Tantivy oracle, or the lock's expected
bits are stale, is unresolved and worth a dedicated pass.

## Reproduce

```bash
cargo bench -p frankensearch-quill --features bench-internals \
  --bench parallel_shard_ingest_ab
# defaults: docs=50000 batch=250 rounds=11 shards=[2,4,8,16,cores]

QUILL_PSI_DOCS=50000 QUILL_PSI_SHARDS=32 QUILL_PSI_ROUNDS=11 \
  cargo bench -p frankensearch-quill --features bench-internals \
    --bench parallel_shard_ingest_ab
```

`QUILL_PSI_ROUNDS` below 10 now asserts rather than silently reporting NOISE.
