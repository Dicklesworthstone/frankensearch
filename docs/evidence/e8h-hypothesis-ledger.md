# E8-H Hypothesis Ledger

**Contract:** `docs/contracts/quill-hyperopt-campaign.md` § Phase 2. Seeded 2026-07-28
(ScarletPelican, `bd-e8h-p2-hypothesis-ledger-s8t6`). Terminal outcomes still flow to
`docs/PERF_LEDGER.md` (keeps) and `docs/NEGATIVE_EVIDENCE.md` (rejects, via its
null-control commit gate); this file is the working ledger of open hypotheses.

**Rule:** no W-lane commit merges without its row updated results-inline. Rejects
require a retry-condition predicate — never "later".

## Row template

```
### <bead-id> — <short name>
Hypothesis:        <falsifiable statement naming a frame/cost>
Minimal repro:     <smallest invocation that shows the cost>
Expected signal:   <percent of the class's ceiling gap (per ceiling.json when M1 lands; raw estimate until then)>
Falsified if:      <what observation kills it>
Invocation:        <one-line command>
Machine classes:   <which classes the claim targets>
Results (inline):  <PENDING | numbers + artifact paths>
Retry predicate:   <filled only on REJECT>
```

## Machine-class fingerprint corrections (authoritative over class-ID labels)

- The trj classes follow the committed-baseline convention **`trj-zen3-<width>c`**
  (first artifact: `QG-2.trj-zen3-16c.latest.json`, activated on 2026-07-28
  with a measured 0.350 [0.345–0.356] MISS vs the ≥1.5x target — Quill
  59.8k vs Tantivy 171.2k docs/s single-thread, 30 paired runs, clean A/A
  null). That pre-fix artifact is now **quarantined**, not an active baseline:
  the Tantivy arm constructed and dropped an unused replacement writer after
  its measured worker join. Although replacement construction was excluded
  from `join_elapsed_ns`, its post-sample work and resource churn could bleed
  into later paired samples. Commit `ebd91757` replaces that path with a
  terminal join whose receipt says `writer_rearmed=false`; only its fresh
  candidate plus immediate same-worker reproduction can establish the new
  ratio. The direction and magnitude are deliberately not predicted, and
  0.35 must not be used as a floor or current performance claim. The machine
  is a Threadripper PRO **5995WX: Zen 3, 64 cores / 128 threads**, single
  NUMA node (NPS1), 512 GB, governor=performance, SMT on
  (`docs/evidence/e8h/fingerprints/trj-zen-128c-20260728/` — directory name
  predates the convention; contents authoritative). Consequences: **no AVX-512
  in silicon** (Zen 4+ only); per-CCD L3 partitioning still applies (8 CCDs).
- `m4-macos` is a Mac mini **M4 Pro, 14 cores (10P+4E), 64 GB, 16 KiB pages**
  (`docs/evidence/e8h/fingerprints/m4-macos-20260728/`).
- `m5-macos`: no reachable host as of 2026-07-28 (mmini/mmini-legacy asleep on the
  tailnet). Class stays declared; beads must not block on it (see p1-m5 bead body).

## First m4-macos diagnostic (2026-07-28, NON-CLAIM label, run receipt m4-macos/20260728T233512Z-qg2-diag-w5-r30-v2)

QG-2 cell `bulk/medium/1/positions_on`, warmup 5, 30 paired runs, ELF built at
`f9c6c57e` (the aarch64 fix commit — this cell was UNRUNNABLE on ARM before it).
FAIRNESS STATUS (corrected after ancestry verification): the ELF commit
`f9c6c57e` CONTAINS `ebd91757`'s terminal-join replacement
(`git merge-base --is-ancestor` verified), so this run executed the FIXED
Tantivy path — these are the first post-fix cross-engine numbers on any
class. Caveat that keeps them diagnostic: the run set no QUILL_PERF_RUN_ID,
so the harness persisted no attempt bundle and there is no
`writer_rearmed=false` lifecycle receipt on disk — a receipt-bearing rerun
(RUN_ID set; perf-runner now defaults it) upgrades them.

| arm | p50 docs/s | median CI95 |
|---|---|---|
| quill | 82,229.6 | [81,164.4, 83,185.6] |
| tantivy | 212,316.3 | [174,742.3, 225,244.3] |
| paired_ab | 0.3742 | [0.3609, 0.4810] |
| paired_null | 1.0395 | [0.9108, 1.1377] — contains 1.0, admissible |

**Observation 1 — gap structure looks architecture-invariant.** The m4
pre-fix diagnostic (0.374) has the same shape as the quarantined pre-fix trj
artifact (0.350): a large single-thread per-document cost, present on both
ARM64 and x86. Both numbers share the same fairness caveat, so the structural
reading (W2 is the deficit's home; W4 SIMD cannot close it) stands while the
exact magnitudes await post-`ebd91757` reruns. No cross-class ratio claim.

**Observation 2 — m4-macos null dispersion is ~4x wider than trj's** (null CI
±11%). P0.3's m4 bands must be derived at this width; tightening levers
(P-core QoS pinning, thermal-window gating, larger run counts) are explicit
P0.3/P1-m4 scope before any KEEP/Block cites an m4 cell.

## Open rows

### bd-e8h-w1-termdict-snapshot-cache-h0eq — decoded TERMDICT block cache
Hypothesis:        After gwd4, residual QG-6 fixed cost is re-DECODING termdict blocks per query; a snapshot-scoped decoded-block cache removes it.
Minimal repro:     QG-6 smoke cell, repeat-query lane, gwd4-landed build vs cache build.
Expected signal:   large fraction of the residual post-gwd4 QG-6 gap on m4-macos (quantify % once the m4 P1 card + ceiling land).
Falsified if:      post-gwd4 profile shows decode frames <0.1% self-time.
Invocation:        scripts/perf-runner.sh --class m4-macos -- cargo bench -p frankensearch-quill-gauntlet --bench perf_matrix --features perf-harness --profile release-perf (QG-6 fixture narrowed)
Machine classes:   m4-macos primary; x86-vps-ovh, trj secondary.
Results (inline):  PENDING (blocked: gwd4 landing + m4 P1 card).
Retry predicate:   n/a

### bd-e8h-w1-verify-once-checksums-d06f — verify-once checksum memoization
Hypothesis:        Per-query section checksum re-verification is a measurable QG-6 frame; verify-once per (snapshot, section) removes it without weakening the open-time contract.
Minimal repro:     QG-6 repeat-query lane; count checksum frames in samply.
Expected signal:   small-to-moderate; only pursued if frames >=0.1% after W1.1.
Falsified if:      validation frames <0.1% post-W1.1.
Invocation:        same as W1.1, sequenced after it.
Machine classes:   m4-macos, x86-vps-ovh.
Results (inline):  PENDING.
Retry predicate:   n/a

### bd-e8h-w1-collector-alloc-tc4q0 — collector allocation churn
Hypothesis:        Argus collectors allocate per query; lazy materialization (collect_id_hits shape) reduces QG-6 latency.
Minimal repro:     dhat census, QG-6 smoke + 100k.
Expected signal:   modest; NoClaim is a valid outcome.
Falsified if:      dhat shows no per-query collector allocs >=0.1%.
Invocation:        dhat-instrumented QG-6 run, local.
Machine classes:   all (claims per class).
Results (inline):  **FALSIFIED → NoClaim** (2026-07-28, MossyPine). Counting-allocator +
                   symbolized dhat census at 500 and 100k docs, 1,600 measured queries:
                   collector-owned frames = 0.011% of bytes / 0.27% of blocks; largest
                   single collector frame 0.0027% bytes. Query-phase allocation is
                   instead 98.28% grimoire::RestartMeta/BlockMeta vec growth — the
                   per-term × per-segment × per-query TERMDICT metadata reparse (18 MB
                   allocated per single-term query at 100k docs; 90 MB for mixed5).
                   Allocation-axis confirmation of gwd4's >83% self-time attribution;
                   consistent with the fsync-census finding that the deficit is pure
                   compute/allocation.
                   Artifacts: docs/evidence/e8h/w13-collector-alloc-census-20260728/
                   (ANALYSIS.md, alloc-census.{smoke,hundredk}.json, symbolized dhat).
Retry predicate:   re-census collectors only if a landed TERMDICT lever drops grimoire
                   frames below ~10% of query-phase bytes AND QG-6 still misses.

### bd-e8h-w2-interner-arena-x9s38 — scribe interner hashing + arena
Hypothesis:        Interner hashing + per-field re-hash + string alloc is a top-3 frame in the QG-1 thread=1 cell.
Minimal repro:     bd-6oiq flamegraph on bulk/medium/1.
Expected signal:   material share of the 8.9x single-thread gap (state as % of ceiling gap when M1 lands).
Falsified if:      interner frames <0.1% in the bd-6oiq card.
Invocation:        per bd-6oiq card; then paired A/B thread=1 and 16.
Machine classes:   x86-vps-ovh + trj primary.
Results (inline):  PENDING (blocked on bd-6oiq card).
Retry predicate:   n/a

### bd-e8h-w2-postings-accumulation-3onsu — postings accumulation growth policy
Hypothesis:        Per-term Vec realloc churn is a top-5 dhat frame at QG-1 medium; safe chunked-arena lists (expull shape) remove it.
Minimal repro:     dhat census on bulk/medium/1.
Expected signal:   material; quantify from census.
Falsified if:      realloc/memmove frames <0.1%.
Invocation:        per bd-6oiq card; then paired A/B.
Machine classes:   x86-vps-ovh + trj primary.
Results (inline):  PENDING (blocked on bd-6oiq card).
Retry predicate:   n/a

### bd-e8h-w2-seal-checksum-audit-ivh69 — seal-time checksum cost (AUDIT)
Hypothesis:        Section checksum computation is >5% of QG-1 seal wall-time at medium scale.
Minimal repro:     flamegraph attribution over the seal phase.
Expected signal:   decision row only (proceed to format-registry lever vs NoClaim).
Falsified if:      checksum frames <5% of bulk wall-time.
Invocation:        local flamegraph lane, bulk/medium.
Machine classes:   x86-vps-ovh, trj.
Results (inline):  PENDING.
Retry predicate:   n/a

### bd-e8h-w2-fsync-audit-ru7jc — commit-path sync-count census (AUDIT)
Hypothesis:        Quill issues materially more fsync/dirsync per commit than Tantivy on the same fixture.
Minimal repro:     strace both arms, one commit cell.
Expected signal:   count table; batching lever filed only if counts differ materially.
Falsified if:      counts are comparable.
Invocation:        strace -f -y -ttt -e trace=fsync,fdatasync,sync_file_range,msync,sync,syncfs,renameat,renameat2 around perf_matrix cells.
Machine classes:   trj-zen3-64c executed; macOS lane needs Law-7 attestation.
Results (inline):  QG-2 CENSUS COMPLETE (trj, 2026-07-28, receipts
                   trj-zen3-64c/20260728T233926Z + 234453Z-wide): across
                   warmup+10 runs of bulk/medium/1/positions_on, BOTH arms
                   issued ZERO durability syscalls (fsync/fdatasync/msync/
                   sync_file_range/syncfs/renameat) — the only 4 fsyncs were
                   harness artifact publication. Explained by construction:
                   perf_matrix builds both arms in memory (quill_in_memory /
                   tantivy_in_memory, perf_matrix.rs:272-283,480). So (a) the
                   QG-2 pairing is IO-symmetric and fair, (b) the 0.35-0.37x
                   single-thread deficit is PURE compute/allocation — no IO
                   lever exists for it, (c) this audit is INAPPLICABLE to
                   in-memory cells by construction.
Retry predicate:   re-run the census when an on-disk commit-bearing cell
                   exists (FoggySquirrel's QG-3 visibility / QG-5 on-disk
                   compact rework, or any gate whose arms leave
                   quill_in_memory), and on macOS with F_FULLFSYNC tracing
                   for Law 7.

## Banked priors (do NOT re-dig without meeting the retry predicate)

| Prior | Verdict | Retry predicate |
|---|---|---|
| Grouped MaxScore activation (`46a475ac`, x4e4.5.1) | REJECT | per its NEGATIVE_EVIDENCE row (query-shape distribution changes materially, e.g. nested pure-term unions dominate a real workload) |
| `core::simd` + `#[target_feature]` (bd-7zjk) | REJECT (packaging, not perf) | workspace ships an authorized non-default nightly feature, or policy authorizes an unsafe-exception kernel crate |
| SIMD posting-unpack dispatch bands (widths 4–28 win; narrow + full-u32 regress) | BANDED | bands are microarchitecture-specific: re-derive per class before any reuse; x86 bands never transfer to NEON |
| Count-free WAND gate extension | REJECT (fixture artifact) | only with fixtures spanning saturating + mid-IDF + rare term classes |
| SWAR tokenizer | KEEP (length-dependent) | any tokenizer lever must bench long AND short corpora |
| Tombstone bitmap @ ~1% density | WASH (inside A/A null) | density regime changes (>10%) or layout A/B with interleaved-paired + A/A null |
| **AVX-512 kernels on trj** | **BLOCKED TWICE** | trj silicon is Zen 3 (no AVX-512 at all) AND bd-7zjk packaging predicate; revisit only on new Zen4+ hardware AND packaging change. Recorded so nobody burns a week. |

## Not-selected math families (revival predicates)

| Family | Why not now | Revived if |
|---|---|---|
| Optimal transport / distribution matching | no measured failure signature | a ranking-quality workstream needs distribution-level comparison of score populations |
| Topological data analysis | no signature | persistent structural anomalies appear in index-graph analysis (e.g., HNSW defect taxonomies outgrow typed checks) |
| Control-theoretic compaction scheduling | static policy not yet shown insufficient | QG-6 foreground-latency-during-maintenance cells show oscillation or sustained >15% loss under the static merge planner |
