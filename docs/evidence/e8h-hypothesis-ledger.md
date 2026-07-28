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
  (first baseline: `QG-2.trj-zen3-16c.latest.json`, ACTIVATED 2026-07-28 as an
  honest 0.350 [0.345–0.356] MISS vs the ≥1.5x target — Quill 59.8k vs Tantivy
  171.2k docs/s single-thread, 30 paired runs, clean A/A null). The machine is a
  Threadripper PRO **5995WX: Zen 3, 64 cores / 128 threads**, single NUMA node
  (NPS1), 512 GB, governor=performance, SMT on
  (`docs/evidence/e8h/fingerprints/trj-zen-128c-20260728/` — directory name
  predates the convention; contents authoritative). Consequences: **no AVX-512
  in silicon** (Zen 4+ only); per-CCD L3 partitioning still applies (8 CCDs).
  Known QG-2 fairness item: a rearmed-writer construct/drop sits inside Quill's
  timed window that Tantivy's arm does not pay — the rerun will improve Quill's
  ratio; W2 rows should treat 0.35x as a floor, not the exact deficit.
- `m4-macos` is a Mac mini **M4 Pro, 14 cores (10P+4E), 64 GB, 16 KiB pages**
  (`docs/evidence/e8h/fingerprints/m4-macos-20260728/`).
- `m5-macos`: no reachable host as of 2026-07-28 (mmini/mmini-legacy asleep on the
  tailnet). Class stays declared; beads must not block on it (see p1-m5 bead body).

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
Results (inline):  PENDING.
Retry predicate:   n/a

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
Minimal repro:     strace -c both arms, one commit cell.
Expected signal:   count table; batching lever filed only if counts differ materially.
Falsified if:      counts are comparable.
Invocation:        strace -c (Linux) / fs_usage (macOS) around perf_matrix commit cells.
Machine classes:   x86-vps-ovh first (READY NOW); macOS lane needs Law-7 attestation.
Results (inline):  PENDING.
Retry predicate:   n/a

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
