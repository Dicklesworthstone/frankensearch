# E8-H P14 — the per-document identity path, measured on BOTH arms: at QG-2 it points the WRONG WAY (incumbent pays 11.6x more); at Xlarge it is the ×37 regime-2 driver (2026-07-30)

**Task:** P13 ranked "per-document eager identity resolution on the append-only
path" as suspect #1 and said of `validate_staged_manifest` that it *"has never
appeared in any profile card's frame list: it is unmeasured, not
measured-small."* This card measures it — and, per the discipline that correctly
rejected memmove, measures **the incumbent's equivalent path in the same
session, from the same ELF, over the same corpus**.

**Verdict: REJECT at QG-2's fixture with the sign inverted — and CONFIRMED as
the regime-2 driver at `Xlarge`.** The two halves must travel together; neither
one alone is the answer.

**Half 1 — at N = 50,000 (QG-2's own fixture): REJECT, and the sign is inverted.** Quill's entire per-document
identity probe — `KeeperSnapshot::delete_document`, i.e. `validate_staged_manifest`
+ the all-segments IDMAP probe — costs **0.318%** of process cycles
(**0.087 µs/doc**). Tantivy's equivalent per-document identity work costs
**3.858%** (**1.011 µs/doc**). **The incumbent pays 11.6x MORE on this path than
Quill does.** Deleting Quill's probe entirely would recover **0.94% of the QG-2
per-document gap**. `validate_staged_manifest` specifically is **0.030%**
(7 samples of 21,427) — it was unmeasured, and it is now measured, and it is
small.

**Half 2 — at N = 512,000: the same probe is 11.9% of process cycles, a ×37
growth, while every other per-document component stays flat.** P13's
O(live-segments) prediction is confirmed by measurement:
`resolve_document_id_in` grows ×41 and `validate_*` grows **×110** — the latter
because `validate_segment_transitions` does **two allocations and two sorts of
the whole segment list per document**. The lever is real; it lives above ~10⁵
documents, in `Xlarge`/QG-7/QG-9 scope, **not** at QG-2.

**The mechanism, stated plainly:** Tantivy's `delete_term` is *not* a free
opstamp enqueue. It calls `delete_query`, which constructs a `TermQuery`,
clones the term bytes, and materialises a `Weight` — **1.125%** eagerly per
document — and the deferral then costs it **2.257%** more in
`tantivy::indexer::index_writer::apply_deletes` on the indexing worker. Quill
resolves eagerly against sealed segments and is done. **Deferring this work is
what makes it expensive, not what makes it cheap.**

**Second finding, larger than the first, and it is the actual answer to
"what does Quill do per document that Tantivy does not":** measured
arm-scoped, the whole per-document identity family is **12.17% for Quill vs
4.12% for Tantivy = +2.26 µs/doc, ~24% of the QG-2 per-document gap** — and
**all** of that excess is canonicalization, none of it is the probe:

| per-document component | Quill | Tantivy counterpart | Quill − Tantivy |
|---|---:|---:|---:|
| content hash `canonical_document_preimage`+`canonical_metadata` | **2.710 µs/doc** | *none exists* — **0.000** | **+2.710** |
| duplicate set `uncommitted_ids: BTreeSet<String>` (contains+insert+`String` clone) | **0.518** | `assign_ord`→`ord_table.push` **0.069** | **+0.449** |
| `PendingIdentity` construction | 0.019 | — | +0.019 |
| **identity probe (this card's suspect)** | **0.087** | **1.011** | **−0.955** |
| **family total** | **3.334** | **1.079** | **+2.255** |

## Disposition

`VALID-ATTRIBUTION / DIAGNOSTIC-CLASS`. Host `thinkstation1` is the local
diagnostic class `local-5975wx-32c`, **not** a registered campaign class:
nothing here may activate a gate, move a ratchet, or be quoted as a competitive
ratio. **No engine source was changed, so there is no A/B and no null gate to
satisfy** — this is a self-time/inclusive attribution card. **No QG-2 re-run is
reported because no lever was landed**: re-running an unmodified ELF would
reproduce P13's `0.4840 [0.4648, 0.5008]` and is an A/A, not evidence.

## Provenance

| axis | value |
|---|---|
| Source | `git archive origin/main` = **`0355d4ad43c464c7a0478c2cb29dd052c4d5eed6`** (P13's export, reused read-only) |
| Bench ELF | `perf_matrix-ad95d11065ec2143`, sha256 **`c0ff90b9809987f06d75362263c051ec8975da8c6b0c07e7ec4f15c7b0595e9e`**, **81,682,848 B** — byte-identical to P13's "+ measurement overlay" ELF |
| **Overlay is provably inert here** | the overlay adds `QUILL_PERF_DOC_COUNT_OVERRIDE` **only** inside `MatrixScale::document_count`. This card's code path is `run_memory_child` → `child_env::<u64>("QUILL_PERF_CHILD_COUNT")` → `corpus_for(count)` → `synthetic_spec(count)`, which **never calls `document_count`**. The variable was never set. Measured behavior = upstream `origin/main`. |
| Cargo.lock | sha256 `fd99abcdb07b164f123f4acd6522e748c3b56763672652b6f166116b3d1de98b` (P13's fresh resolve) |
| Toolchain | rustc 1.99.0-nightly (9f36de775 2026-07-19), pinned `nightly-2026-07-20`; `RUSTFLAGS="-C force-frame-pointers=yes"`, `--profile release-perf` |
| CPU / kernel | AMD Ryzen Threadripper PRO 5975WX (Zen 3, 32c/64t); Linux 6.17.0-35-generic; `amd-pstate-epp`, governor **powersave**; THP `madvise` |
| perf | 6.17.13, `perf_event_paranoid=-1`, `perf_event_max_sample_rate=7000` |
| Affinity | `taskset -c 8` — **one logical CPU for both arms**, so CPU/wall is 0.993 (Quill) vs 0.998 (Tantivy): unlike the unpinned case, this pin **is** symmetric here and no thread-count artifact is in play |
| Host load | shared with peer `codex`/`rustc` sessions; 1-min load 7.8 at session start. Relative shares only. |

## Instrument

The blessed single-arm attribution harness (`QUILL_PERF_CHILD_MODE=memory`),
the same one the incumbent-copy card used — one arm per process, so every
sample is unambiguously owned:

```bash
env QUILL_PERF_CHILD_MODE=memory QUILL_PERF_CHILD_ENGINE=<tantivy|quill> \
    QUILL_PERF_CHILD_COUNT=50000 QUILL_PERF_CHILD_HEAP=50000000 \
    QUILL_PERF_CHILD_THREADS=1 QUILL_PERF_CHILD_POSITIONS=true \
    QUILL_PERF_SCALE=full \
  perf record -F 4999 -g --call-graph dwarf,32768 -o <tag>.perf.data -- \
  taskset -c 8 <ELF>
```

`QUILL_PERF_SCALE=full` ⇒ **batch 5,000**, and `CHILD_COUNT=50000` ⇒ **N = 50,000**:
**this is QG-2's own fixture shape** (`bulk/medium/1/positions_on`, positions ON,
writer heap 50 MB, threads=1). Both arms run `LexicalWrite::index_documents`,
which is per-document upsert semantics on **both** sides —
Quill `upsert_documents` → `snapshot().delete_document(&mut manifest, &id)`
(`index.rs:3111-3120`), Tantivy → `Term::from_field_text` + `writer.delete_term(term)`
(`frankensearch-lexical/src/lib.rs:2242-2244`). The comparison is like-for-like
by construction.

**What this instrument is and is not.** It is an *attribution* instrument, not
a *ratio* instrument: it runs corpus generation inline (~11.5% of each arm) and
includes each engine's commit, so its wall ratio (Quill 1.380 s vs Tantivy
1.313 s) is **not** the QG-2 ratio and is not quoted as one. The QG-2 ratio
comes from P13.

### Run receipts (untraced, `/usr/bin/time`, 3 interleaved reps, `taskset -c 8`)

| arm | wall (3 reps) | CPU=user+sys | CPU/wall | peak RSS | index bytes |
|---|---|---:|---:|---:|---:|
| Tantivy 0.26.1 | 1.27 / 1.32 / 1.35 s | **1.310 s** | 0.998 | 176.3 MB | 30,468,000 |
| Quill | 1.35 / 1.36 / 1.43 s | **1.370 s** | 0.993 | 291.7 MB | 0 (in-memory; `managed_disk_bytes` hard-zero, known) |

Conversion used throughout: **1% of process cycles = 0.274 µs/doc (Quill),
0.262 µs/doc (Tantivy)** at N = 50,000.

Traced: 3 dwarf runs per arm, interleaved (Quill 7,534 / 6,965 / 6,928 samples;
Tantivy 6,447 / 6,372 / 6,492) plus one frame-pointer run per arm. **Sample-loss
disclosure:** five of the six dwarf runs report zero loss; `n50k-tantivy-r3`
reports `lost 8 chunks` of 8,414 events (~0.1%) — its `delete_term` figure
(1.210%) is in line with r1/r2, so the headline is unaffected. Every callchain
resolves to `_start`; **no truncation**.

## Attribution method, and the mutation check that makes it non-vacuous

Both suspects are **inlined** (`validate_staged_manifest` into `delete_document`;
`delete_term` into the lexical adapter), so *self*-time cannot see them. This
card therefore measures **cycles-weighted INCLUSIVE share**: a sample counts for
a path if any frame of its callchain — including `perf script --inline`
expansions — names that path.

`resolve_document_id_in` is reachable **only** from `delete_document`
(`keeper.rs:3138`), but in 23 of 70 probe samples the dwarf unwinder elided the
intervening `delete_document` frame. The headline therefore uses the
**deliberately over-inclusive union** `delete_document | resolve_document_id_in |
lookup_document_id | validate_staged_manifest` — over-inclusive because
`validate_staged_manifest` also has publish-path callers (`keeper.rs:2913/2951/2976/2980`).
**An inflated Quill selector that still rejects is a stronger reject.**

**Selector specificity (mutation check): every selector was run against the
WRONG arm.** All ten fire at exactly **0.000%**, and the one *shared* selector
fires almost equally on both — which validates "same corpus, same generator"
structurally instead of asserting it:

| selector | own arm | **other arm** |
|---|---:|---:|
| `delete_document` | 0.207% (47/21427) | **0.000% (0/19311)** |
| `resolve_document_id_in` | 0.199% (43) | **0.000% (0)** |
| `validate_staged_manifest` | 0.030% (7) | **0.000% (0)** |
| `canonical_document_preimage\|canonical_metadata` | 9.952% (2105) | **0.000% (0)** |
| `SetValZST` (BTreeSet internals) | 4.051% (864) | **0.000% (0)** |
| `PendingIdentity` | 1.262% (259) | **0.000% (0)** |
| `delete_term` | 1.126% (214/19311) | **0.000% (0/21427)** |
| `apply_deletes` | 2.257% (401) | **0.000% (0)** |
| `delete_queue\|DeleteQueue\|DeleteOperation` | 0.706% (133) | **0.000% (0)** |
| `assign_ord\|ord_table` | 0.262% (50) | **0.000% (0)** |
| **shared:** harness generator `document_at` | **11.718%** (Quill) | **11.251%** (Tantivy) — ratio 0.96 |

**Independent-unwinder cross-check (frame pointer vs dwarf):**

| figure | dwarf (pooled 3 runs) | fp (1 run) |
|---|---:|---:|
| Quill `delete_document` | 0.207% | 0.259% |
| Quill `resolve_document_id_in` | 0.199% | 0.260% |
| Tantivy `apply_deletes` | 2.257% | 2.634% |
| Tantivy `TermQuery` (all) | 3.250% | 4.255% |

Both unwinders agree on every headline: Quill's probe is a fraction of a
percent, Tantivy's deferred apply is over 2%.

## THE TABLE — per-document identity path, both arms, one ELF, one host, one corpus

Cycles-weighted inclusive share of whole-process cycles, per run and pooled;
N = 50,000, batch 5,000, positions ON, heap 50 MB, threads=1, `taskset -c 8`.

**QUILL — per-document path, separated from flush/commit path**

| component | r1 | r2 | r3 | mean | µs/doc |
|---|---:|---:|---:|---:|---:|
| **identity probe (union selector)** | 0.378% | 0.316% | 0.254% | **0.318%** | **0.087** |
|   – `validate_staged_manifest` alone | — | — | — | 0.030% (7 smp) | 0.008 |
|   – `resolve_document_id_in` ∩ `delete_document` | — | — | — | 0.088% (20 smp) | 0.024 |
| `canonical_document_preimage`+`canonical_metadata` | 9.841% | 9.922% | 9.906% | **9.890%** | **2.710** |
| `uncommitted_ids: BTreeSet<String>` (per-doc) | 2.073% | 1.771% | 1.830% | **1.891%** | **0.518** |
| `PendingIdentity` (per-doc construction) | 0.066% | 0.074% | 0.070% | **0.070%** | 0.019 |
| **PER-DOCUMENT IDENTITY FAMILY** | | | | **12.169%** | **3.334** |
| *(flush path: `PendingIdentity` drop)* | *1.669%* | *1.705%* | *0.165%* | *bimodal* | — |
| *(commit path: `uncommitted_ids` drain)* | *0.241%* | *6.187%* | *0.116%* | ***bimodal*** | — |

**TANTIVY 0.26.1 — the same per-document identity work**

| component | r1 | r2 | r3 | mean | µs/doc |
|---|---:|---:|---:|---:|---:|
| **EAGER** `delete_term` → `delete_query` → `TermQuery::weight` (+term clone, queue push) | 1.062% | 1.104% | 1.210% | **1.125%** | **0.295** |
| **DEFERRED** `index_writer::apply_deletes` (on `thrd-tantivy-in`) | 2.275% | 2.302% | 2.194% | **2.257%** | **0.591** |
| delete-queue teardown (`drop_slow<[DeleteOperation]>`, `SegmentEntry` drop) | 0.740% | 0.638% | 0.049% | **0.476%** | 0.125 |
| `assign_ord` → `ord_table.push` | 0.206% | 0.192% | 0.388% | **0.262%** | 0.069 |
| **IDENTITY FAMILY** | 4.282% | 4.237% | 3.842% | **4.120%** | **1.079** |

Thread split (pinned, r1): Tantivy `thrd-tantivy-in` 47.19% / main 42.86% /
`docstore-compre` 9.80%; Quill main 99.95%.

### Reading

1. **The named suspect is 0.318% and the incumbent's equivalent is 3.858%**
   (1.125 eager + 2.257 deferred + 0.476 teardown). **Tantivy pays 11.6x more**,
   even against a Quill selector deliberately inflated with publish-path
   `validate_staged_manifest` callers. Removing Quill's probe entirely buys
   **0.087 µs/doc = 0.50% of Quill's 17.3 µs/doc and 0.94% of the ~9.3 µs/doc
   QG-2 gap.**
2. **`validate_staged_manifest` is now measured: 0.030%, 7 samples of 21,427.**
   P13's "unmeasured, not measured-small" is resolved — it is measured-small.
   The reason is visible in the source: on an all-new-id bulk load
   `resolve_document_id_in` returns `None`, so `delete_document` **returns
   before** `proposed.clone()` and before the second `validate_staged_manifest`
   (`keeper.rs:3138-3139`). The expensive half of the function never runs.
3. **The deferral asymmetry is real and inverted.** P13's code inventory read
   Tantivy's side as *"one opstamped enqueue; resolution deferred to
   merge/reader"*. That is code-accurate and cost-wrong: the enqueue builds a
   `TermQuery` and a `Weight` per document (1.125%), and `apply_deletes` then
   re-resolves the queue on the indexing worker (2.257%). **On a bulk load with
   all-new ids, every one of Tantivy's deletes misses too — and its miss costs
   more than Quill's.**
4. **What actually makes Quill's per-document identity work expensive is
   canonicalization, 2.710 µs/doc, against an incumbent counterpart that does
   not exist** (selector 0/19,311 samples). This is the arm-scoped re-derivation
   the incumbent-copy card demanded before any lever was spent on that family:
   Round-1's "~12-15% family" was inflated by *shared* generator `serde`/`fmt`
   self-time, but the **inclusive share of Quill's own canonical functions is
   9.890%**, reproducing to ±1.3% across three runs. It is the largest
   Quill-only per-document cost on this fixture.
5. **`uncommitted_ids: BTreeSet<String>` is a real second-order cost with a
   caveat.** Its *per-document* half (`contains` at `index.rs:2584` + `insert`
   with a `String` clone at `index.rs:2629`) is stable at **1.891%**. Its
   *commit-time* drain (`index.rs:2424/2813-2815`) is **bimodal — 0.241%,
   6.187%, 0.116% across three otherwise identical runs**. That 26x swing is
   itself an unexplained commit-path cliff and is flagged, not averaged away.

## GROWTH CHECK — does the probe become significant where regime 2 lives?

P13 located superlinearity above ~10⁵ documents and predicted
`resolve_document_id_in`'s O(live segments) term as its driver. The same
instrument at **N = 512,000**, 2 dwarf runs per arm, F = 1997:

**P13's prediction is CONFIRMED, and it flips this card's verdict from flat to
conditional.**

| figure (share of that arm's process cycles) | **N = 50,000** (QG-2's fixture) | **N = 512,000** (regime 2) | growth |
|---|---:|---:|---:|
| **Quill identity probe (union selector)** | **0.318%** (0.254–0.378) | **11.906%** (11.579–12.233) | **×37.4** |
|   – `resolve_document_id_in` | 0.199% | **8.201%** (8.198–8.203) | ×41 |
|   – of which `lookup_document_id` (per-segment IDMAP hash probe) | 0.070% | **3.902%** (3.861–3.942) | ×56 |
|   – `validate_*` (`staged`/`successor`/`segment_transitions`) | 0.030% | **3.294%** (3.276–3.312) | **×110** |
| Quill `canonical_*` content hash | 9.890% | 8.115% (8.003–8.226) | ×0.82 — **flat** |
| Quill `uncommitted_ids` (per-doc) | 1.891% | 1.692% (1.635–1.748) | ×0.89 — **flat** |
| Tantivy identity family | 4.120% (3.84–4.28) | 3.299% (2.162–4.435) — **bimodal** | ×0.80 |
| *generator anchor (Quill / Tantivy)* | *11.718% / 11.251%* | *9.203% / 15.344%* | — |

**Reading the growth check:**

- **The probe is the regime-2 driver, exactly as P13 predicted.** Its share
  grows **37x** between the two fixtures while *every other* per-document
  component stays flat or shrinks. `resolve_document_id_in` is O(live segments)
  per document and `lookup_document_id` — the IDMAP hash probe issued against
  **each** live segment — grows 56x.
- **`validate_staged_manifest` grows fastest of all (×110)** and this card can
  name why: `validate_segment_transitions` (`keeper.rs:10809-10833`) performs
  **two heap allocations and two `sort_unstable_by_key` passes over the whole
  segment list, per document**. At 50,000 documents the live-segment count is
  low enough that both are ~free; at 512,000 they are 3.29% of the process.
- **The generator anchor carries the normalization.** Corpus generation is
  identical work in both arms, so its share ratio measures the arms' process
  CPU ratio from the traces themselves: at N=512,000, Quill's process CPU is
  ≈ **15.344/9.203 = 1.67x** Tantivy's. Quill's probe alone (11.9% × 1.67 ≈ 19.9
  Tantivy-CPU-equivalent %) is therefore **~6x the incumbent's entire identity
  family** at this fixture. At N=50,000 the same anchor gives 0.96x — the arms
  are the same size, and the probe is a rounding error.
- **Honesty on the N=512,000 receipts.** They are too noisy for an absolute
  µs/doc budget and are not used as one: untraced CPU came out 25.80 s and
  15.11 s for the *same* Quill workload under peer load. `n512k-quill-r1` also
  reports **`lost 7.07%` of samples**; its probe figure (12.233%) agrees with
  the clean r2 (11.579%), so the direction is safe but the magnitude is
  directional only. **Shares only at this N; no absolute conversion.**
- Tantivy's `apply_deletes` is **bimodal at this scale** (1.830% vs 0.195%),
  which is why its family is reported as a range, not a mean.



## Consequences for the campaign

1. **Suspect #1 from P13 splits by regime, and each half is now measured.**
   *At QG-2 it is CLOSED as a REJECT* — 0.318%, a *credit* against the
   incumbent rather than a debit, and worth 0.94% of the gap. *Above ~10⁵
   documents it is PROMOTED to rank 1* — 11.9% of process cycles at N=512,000,
   ~6x the incumbent's whole identity family once normalized by the generator
   anchor. Chase it in `Xlarge`/QG-7/QG-9 scope. No code change is proposed here
   and none was made.
   The named sub-levers for that scope, in measured order:
   **(a)** `validate_segment_transitions`' two per-document allocations + two
   sorts (`keeper.rs:10809-10833`) — ×110 growth, and the only one that is pure
   overhead on a bulk load with no tombstones to compare;
   **(b)** hoisting the whole probe out of the all-new-id case in
   `upsert_documents` (`index.rs:3111-3120`), where every probe misses by
   construction;
   **(c)** `lookup_document_id`'s per-segment IDMAP probe (×56) — a segment-level
   membership filter would cut it without changing semantics.
2. **The re-ranked #1 is canonicalization, on measured arm-scoped grounds** —
   2.710 µs/doc, zero incumbent counterpart, ~29% of the ~9.3 µs/doc gap on its
   own. Note this is **not** the already-rejected "canonical-encode fast path"
   (WASH 1.0175 < 1.03): that lever shaved the encode; the measurement here says
   the whole per-document `serde_json` preimage is the cost, so the only lever
   with the right shape is **removing or deferring the class**, not speeding it up.
3. **Three QG-2 suspects are now eliminated by the same discipline** — memmove
   (incumbent pays the same toll), allocator family (DEAD on this class),
   per-document identity resolution (incumbent pays 11.6x more *at this
   fixture*). In every case the elimination required measuring the incumbent;
   in every case the Quill-only number alone would have looked actionable. The
   converse also holds and is why the fixture must always be named: the same
   probe that is a credit at N=50,000 is the single largest Quill-only
   per-document cost at N=512,000. **A suspect is not "rejected" or "confirmed"
   — a suspect is rejected or confirmed at a stated N.**
4. **The commit-path `uncommitted_ids` bimodality (0.24% → 6.19%, same binary,
   same fixture) is an open defect-shaped observation**, not a perf lever. It
   should be reproduced with more runs before anyone reasons about commit cost.

## Retry predicate

**For QG-2 / `medium` / N ≤ 128,000: settled false, no retry.** The whole probe
is under 1% of the gap and the incumbent pays 11.6x more; re-litigating it at
`medium` is a banned re-dig. A rerun cannot revive it.

**For N > 10⁵: open, and this card supplies the entry bar it must clear.** The
probe is admissible as a lever on any fixture where its measured inclusive
share on the Quill arm exceeds **1% of process cycles** — a bar it clears by
12x at N=512,000 and misses by 3x at N=50,000. Any such attempt must publish
the incumbent's identity family beside it (this card's Tantivy column is the
template) and must re-anchor with the generator selector rather than trusting
`/usr/bin/time` under peer load.

## Repro

```bash
# both arms, one ELF, interleaved; see Instrument above for the env block
bash run_p14.sh                       # N=50000 TAG=n50k FREQ=4999 REPS=3
bash mkscripts.sh                     # perf script --inline -> text
bash final.sh                         # per-run per-document vs flush/commit split
bash selector_null.sh                 # selector specificity (wrong-arm nulls)
bash fpcheck.sh                       # frame-pointer cross-check
```

Artifacts (session scratchpad, machine-local):
`n50k-{quill,tantivy}-r{1,2,3}.perf.data`, `n50k-{quill,tantivy}-fp.perf.data`,
`n512k-*`, `receipts-n50k.txt`, and the analysis scripts above under
`/data/tmp/claude-1000/-data-projects-frankensearch/8194bbd4-44a0-4308-9330-fc004b73200c/scratchpad/perf/`.
