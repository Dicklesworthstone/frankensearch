# E8-H P11 — GATE-CELL-SHAPED counter battery, QG-2 bulk/medium, local-5975wx-32c (2026-07-30) — DRAFT (BANKED, publication freeze)

**Landing provenance:** landed at 928a16ba-successor; measurements are diagnostic/NoClaim per card scope.

**NoClaim. Diagnostic attribution evidence only. This battery NEVER activates a
QG and produces NO speed claim** (prereg exclusion, quoted verbatim below).
Executed against the FROZEN pre-registration
`scratchpad/p11-gate-cell-counter-battery-PREREG.md` (registered before any
collection; both collection gates satisfied: P9 reported, timed slot free).
Measurement-only pass: zero shared-tree edits, zero commits, zero tracker calls.

## HEADLINE (estimand 1, primary)

**Engine-only instructions q/t on the QG-2 gate-cell shape = 1.41**
(share-scaled point estimate 1.412; robustness band 1.37–1.47 across UNCLASS
allocations and the record-internal flavor — all DERIVED numbers, see labels).
Raw whole-child instructions q/t = **1.253** (n=5 medians; per-pair
1.243–1.273) vs the seam's raw **1.272** (P10). On a like-for-like basis the
seam evidence GENERALIZES to the gate cell; see "Which interpretation fired".

## DEVIATIONS (all of them, prominently)

1. **The prereg's "ON-DISK index directory" is UNPRODUCIBLE — the QG-2 gate
   cell is in-memory.** Source-verified at THREE revisions: the 8.7x baseline
   commit `351f5c6d` (bulk_metric_unpooled → `quill_in_memory` /
   `tantivy_in_memory`), the ELF continuity base `3684b147` (same, and its
   qg-commit-parity log line prints `durability=in_memory`), and the current
   worktree. The gates manifest fixture line (`quill-perf-gates.toml` [gate.QG-2])
   says "medium; positions ON; threads = 1; commit included" — no disk. The
   on-disk premise traces to the P1 card's scope-correction paragraph
   ("a different fixture (medium corpus, on-disk, commit inside the timed
   window)"), which mischaracterized the fixture; the prereg inherited that
   error. Resolution: measured the gate's ACTUAL shape (in-memory, commit
   included). No on-disk variant was fabricated (none exists for any gate).
2. **Single-arm invocation = the `QUILL_PERF_CHILD_MODE=memory` child**, used
   as "the smallest gate-runner invocation that can run one arm at a time"
   (the prereg's own fallback clause). Verified in the ELF-rev source: the
   child runs the same `index_batches` + terminal `commit` body as
   `bulk_metric_unpooled`, with gate batching (5000 docs/batch under
   `QUILL_PERF_SCALE=full`). Deltas vs the gate cell body, all negligible in
   instructions: child adds `segment_stats()`+RSS readout (quill) /
   `benchmark_index_layout()`+RSS (tantivy); child omits the gate's
   `benchmark_join_workers` terminal fence (a join wait). No interleaved-cell
   perf-stat + share-derivation fallback was needed.
3. **Generator correction is the P9 call-chain split** (prereg's "otherwise"
   branch: the child has no seam to hoist generation out of the process).
   Engine-only numbers are therefore DERIVED (counter medians x classifier
   shares), labeled throughout. Raw AND engine-only are both reported (prereg
   GENERATOR CORRECTION clause).
4. **Classifier**: byte-identical copy of P9 `classify.awk` except ONE regex
   token widened (` +cycles` → ` +(cycles|instructions)`) so the identical
   call-chain split can weight by the sampled event. P9's file untouched.
5. **`--per-thread` is not supported** by perf 6.17 for launched workloads
   (only -p/-t/-a) — prereg's "where supported" clause. Asymmetry covered by
   task-clock/duration_time (CPU/wall) + context-switches per arm.
6. **Both arms pinned `taskset -c 8`** (parent constraint; method-match with
   P10, whose seam counter card is the comparison anchor). P9's rationale for
   leaving tantivy unpinned (its nominal threads=1 still runs ~4 threads)
   is acknowledged: under the pin, CPU/wall≈1.0 by construction for BOTH arms
   and the thread asymmetry surfaces as context-switches (28 vs 8,597).
   Tantivy's per-instruction miss rates are measured under single-core
   timesharing — the SAME bias as P10's anchor, so gate-vs-seam comparisons
   are internally consistent.
7. **tantivy-cycles perf.data lost 13 chunks** (~0.1% of events). Affects only
   the cycles-weighted comparability shares, not the primary (instruction
   captures were loss-free).
8. n=5 per arm achieved as pre-registered (no shortfall). Interleaved,
   alternating leader: T Q | Q T | T Q | Q T | T Q, after one discarded
   warmup pair.

## Machine-class fingerprint (Law 6: NOT comparable to trj-* or m4-*; those
executions route through FoggyPrairie's windows per the prereg)

| axis | value |
|---|---|
| Host | thinkstation1 — class `local-5975wx-32c` (dev host; diagnostic only) |
| CPU | AMD Ryzen Threadripper PRO 5975WX (Zen 3), 32c/64t, SMT on |
| Kernel / governor | Linux 6.17.0-35-generic; cpu8 governor **powersave**; observed ~4.17–4.26 GHz during runs |
| Load | ~2.9–3.8/64 background at battery start; both arms pinned core 8 |
| perf | 6.17.13; perf_event_paranoid=1; max_sample_rate=7000 |

## Provenance

- ELF: `scratchpad/p7/elfs/perf_matrix_G`, SHA-256 **re-verified in-script
  before the battery** =
  `ae96a2acb95efdd09af87b4e3ed2457f69529eda76f26d36459d414aa4c40aa4`
  (P7 generic build: pristine `git archive` of continuity base `3684b147`
  [ancestor of origin/main — verified], rustc 1.99.0-nightly 9f36de775,
  release-perf, perf-harness, RUSTFLAGS `-C force-frame-pointers=yes`).
  SAME binary, both arms; SAME ELF as P10's seam counter card → the
  gate-vs-seam counter comparison is same-binary by construction. No rebuild
  performed; no cargo invoked anywhere in this pass.
- Workload env (gate-cell shape, both arms):
  `QUILL_PERF_CHILD_MODE=memory QUILL_PERF_CHILD_ENGINE={quill|tantivy}
  QUILL_PERF_CHILD_COUNT=50000 QUILL_PERF_CHILD_HEAP=50000000
  QUILL_PERF_CHILD_THREADS=1 QUILL_PERF_CHILD_POSITIONS=true
  QUILL_PERF_SCALE=full` — i.e. the manifest QG-2 cell
  `bulk/medium/1/positions_on` (50k docs, heap 50 MB = perf_writer_heap_bytes(1),
  batch 5000, positions ON, terminal commit inside the process).
- Tantivy arm config (prereg screening clause): the gate-PINNED benchmark
  config (`in_memory_with_benchmark_config(50MB, 1, positions_on)`), recorded
  here as the frozen config. No tantivy-arm tuning was performed (prereg
  exclusion). The QG-6 fastest-equivalent screening rule concerns query-side
  configs; for QG-2 the manifest pins the ingest config.
- Corpus: seed-pinned synthetic Zipf S11, vocab 8192, doc 4096B, seed baked
  into the binary — identical for both arms by construction.

## Method (commands verbatim → `p11/run_stat_battery.sh`, `p11/run_records.sh`, `p11/run_classify.sh`)

```bash
# counters: n=5 interleaved per arm (+1 discarded warmup pair), both arms core 8
perf stat -x, -o stat-<arm>-<i>.csv -e cycles,instructions,branches,branch-misses,\
cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,\
l2_cache_accesses_from_dc_misses,task-clock,context-switches,duration_time \
  -- taskset -c 8 perf_matrix_G          # with the gate-cell child env above
# traced runs (separate from counter runs): per arm x {cycles,instructions}
perf record -F 6997 -e <event> -g --call-graph dwarf,32768 -o <arm>-50k-<event>.perf.data \
  -- taskset -c 8 perf_matrix_G
# split: 3 replicates per perf.data (perf script inline expansion is
# nondeterministic — P9 binding finding)
perf script -i <data> | gawk -f p11/classify.awk > split-<arm>-<event>-r<k>.txt
```

Event battery = the prereg's list exactly, plus `duration_time` (software
event; needed for CPU/wall). 9 HW events multiplex at ~55% coverage each —
same multiplexing profile as P10's anchor battery. Sample counts: quill
10,744 (cycles) / 9,816 (instructions); tantivy 9,388 / 8,811 — all above the
P9 8k floor. Class-level splits were REPLICATE-STABLE (quill byte-identical
across 3 reps; tantivy ENG jitter <=0.03% — the P9 jitter is family-level).

## Raw counter table (medians of n=5; whole child process, RAW = includes generator)

| counter | quill | tantivy | q/t | q spread% | t spread% |
|---|---:|---:|---|---|---|
| cycles | 5,799,957,949 | 5,221,854,794 | 1.111 | 3.40 | 4.52 |
| **instructions** | **13,020,668,728** | **10,390,083,902** | **1.253** | 2.54 | 1.46 |
| branches | 2,252,427,007 | 1,924,841,651 | 1.170 | 3.51 | 2.12 |
| branch-misses | 28,180,648 | 29,185,731 | 0.966 | 6.19 | 3.01 |
| cache-references | 214,838,950 | 153,129,772 | 1.403 | 5.26 | 3.48 |
| cache-misses (LLC) | 25,652,435 | 28,480,427 | 0.901 | 5.99 | 6.28 |
| L1-dcache-loads | 3,914,736,549 | 3,304,584,208 | 1.185 | 2.01 | 1.80 |
| L1-dcache-load-misses | 114,025,932 | 72,484,336 | 1.573 | 10.82 | 9.44 |
| l2_cache_accesses_from_dc_misses | 120,912,051 | 78,570,757 | 1.539 | 11.37 | 5.17 |
| task-clock (ms) | 1,361 | 1,254 | 1.086 | 3.41 | 4.57 |
| context-switches | 28 | 8,597 | — | 25.0 | 4.69 |
| duration_time (ms) | 1,369 | 1,265 | 1.082 | 3.15 | 4.43 |

Per-pair raw instruction ratios: 1.2431, 1.2709, 1.2508, 1.2552, 1.2733 →
median **1.2552**. Derived: IPC quill **2.245** / tantivy **1.990**;
branch-miss 1.25% vs 1.52%; L1d-miss-of-loads 2.91% vs 2.19%.

## Call-chain split (P9 method; shares of the whole child)

| partition | quill instr-weighted | tantivy instr-weighted | quill cycles-weighted | tantivy cycles-weighted |
|---|---|---|---|---|
| ENG | **71.41%** | **63.35%** (63.33–63.36) | 71.48% | 62.91% |
| GEN | 25.69% | 31.29% | 21.62% | 26.58% |
| LOOP | 1.74% | 3.34% | 4.09% | 5.99% |
| UNCLASS | 1.16% | 2.02% | 2.80% | 4.52% |

Cycles-weighted shares cohere with P9's seam split (ENG 72.67%/62.34%, GEN
20.34%/23.80% at 200k/batch-250): the gate shape's shares move <2 points.
**Same-work validity check:** harness toll (GEN+LOOP) instruction cost,
scaled to counter totals: quill 3.572e9 vs tantivy 3.598e9 — **equal within
0.7%** (identical corpus by construction; the split passes its own
invariant). GEN/ENG overlap 0.06%/0.31% (unwind noise, counted into GEN =
conservative for engine claims).

## PRE-DECLARED ESTIMANDS (prereg wording → measured values)

1. **"Engine-only instructions q/t per class (primary)"** — local-5975wx-32c:
   **1.412** [DERIVED: stat medians x instruction-weighted ENG shares =
   13.0207e9x0.7141 / 10.3901e9x0.6335 = 9.298e9 / 6.582e9]. Robustness:
   record-internal ENG-sum flavor 1.465; UNCLASS-reallocation extremes
   1.369–1.435 → **band 1.37–1.47**. Engine-only CYCLES q/t (context):
   1.262 [cycles-weighted shares], vs P9's seam engine-only CPU 1.23.
2. **"CPU/wall per arm per class (asymmetry context, mandatory)"** — under the
   core-8 pin: quill task/wall = 1361/1369 = **0.994**; tantivy 1254/1265 =
   **0.991**. The pin forces CPU/wall≈1; the threads=1 asymmetry appears as
   **context-switches 28 (quill) vs 8,597 (tantivy)** — tantivy's ~4 threads
   timesharing the pinned core. Unpinned-shape context (P9 receipts, seam):
   tantivy CPU/wall 1.77x, quill 1.00x. Never quote a wall ratio from this
   fixture without this row.
3. **"L1d-miss and LLC-miss per retired instruction, per arm per class"** —
   RAW process scope (misses cannot be namespace-split by perf stat):
   | per 1k retired instructions | quill | tantivy | q/t |
   |---|---|---|---|
   | L1-dcache-load-misses | 8.757 | 6.976 | 1.255 |
   | LLC (cache-misses) | 1.970 | 2.741 | **0.719** |
   | l2_cache_accesses_from_dc_misses | 9.286 | 7.562 | 1.228 |
   Tantivy's LLC-miss-per-instruction is HIGHER; quill's L1d/L2 pressure is
   higher. Caveat: tantivy measured under single-core timesharing (deviation
   6); same bias as the P10 anchor.

## Pre-declared interpretations (VERBATIM from the prereg) and which fired

> - If gate-cell instructions q/t >> seam's (>1.5): the seam materially
>   understates the gate excess-work problem; ingest levers must be re-ranked on
>   gate-cell attribution.
> - If gate-cell q/t ~= seam's (~1.2-1.3): seam evidence generalizes; current
>   lever queue stands.
> - If per-class q/t differs materially (e.g. x86 >> m4): the class split is
>   workload-mix/IO-shape dependent, not compute-shape — route to on-disk/commit
>   path profiling.
> - This battery NEVER activates a QG, never produces a public speed claim; it
>   is diagnostic attribution evidence (NoClaim).

**Fired: the second ("seam evidence generalizes; current lever queue
stands") — on like-for-like bases.** Raw-vs-raw: gate 1.253 vs seam 1.272
(P10) — same band, no understatement. Engine-only-vs-engine-only: gate 1.41
[this card] vs seam ~1.48 [DERIVED, mixed-method: P10 raw instructions x
P9 cycles-weighted shares] — same band within method error. The third
interpretation is out of scope this pass (single class; trj/m4 route through
FoggyPrairie's windows).

**Flag, not an amendment:** applied LITERALLY to the primary estimand, the
measured 1.41 falls BETWEEN the prereg's two bands (">1.5" vs "~1.2-1.3"),
because the prereg anchored "the seam's" number at its RAW/CPU headline
(1.2-1.3) while the seam's engine-only INSTRUCTION ratio was never separately
pre-computed (it derives to ~1.48). No post-hoc interpretation is invented
here; the like-basis reading above is reported, and the discrepancy between
the prereg's anchor and the derived like-basis anchor is left to the
orchestrator. The substantive new fact either way: **the raw 1.25-1.27
headline UNDERSTATES the engine-only instruction excess (~1.4-1.5x) in BOTH
shapes** — the generator contaminates the tantivy denominator proportionally
more (31.3% of its instructions vs quill's 25.7%). Lever ranking is
unaffected (P9's engine-only table remains the ranking basis; nothing here
re-orders it), and the instruction-count-reduction strategy frame (P10) is
STRENGTHENED: the true engine-side excess-work target is ~41% [DERIVED], not
~25%.

## What the gate cell actually measures (source-verified) + class scoping of the 8.7x

- In the gate runner, corpus generation sits OUTSIDE the timed window
  (`generated_batch` precedes the `Instant::now()` around `index_documents`;
  verified at `3684b147` and worktree) — the QG-1 ledger claim is confirmed.
  The timed window is index_documents + terminal commit. In THIS battery's
  whole-process counters, generation is inside the process and removed by the
  call-chain split instead.
- The `351f5c6d` QG-2 baseline (quill 20,579 vs tantivy 179,027 docs/s =
  8.7x) is a WALL-throughput window ratio from machine class
  `linux-x86_64-16cpu-AMD_Ryzen_7_5800X` — a DIFFERENT class; Law 6 forbids
  pooling and this card makes no throughput statement about it (prereg
  exclusion: counters only). What this card CAN say: on local-5975wx-32c the
  gate-cell instruction gap is 1.25x raw / 1.41x engine-only [DERIVED] and the
  cycle gap is 1.11x raw — an 8.7x wall gap is not an instruction-count
  phenomenon on this class; the known mechanism direction is the threads=1
  pin asymmetry (P1: tantivy gets ~1.8x thread parallelism at nominal
  threads=1), visible here as the 28-vs-8,597 context-switch asymmetry.

## Repro

```bash
sha256sum scratchpad/p7/elfs/perf_matrix_G   # must be ae96a2ac...
scratchpad/p11/run_stat_battery.sh           # 12 runs (2 warmup + 5 pairs), ~40 s
scratchpad/p11/run_records.sh                # 4 dwarf captures, ~90 s
scratchpad/p11/run_classify.sh               # 12 classifier replicates, ~5 min
python3 scratchpad/p11/aggregate.py          # medians / ratios / derived rates
```

Artifacts (session scratchpad, machine-local), all under `scratchpad/p11/`:
`stat-{quill,tantivy}-{warmup,1..5}.csv`, `child-*-{out,err}`,
`{quill,tantivy}-50k-{cycles,instructions}.perf.data` (277–335 MB each),
`split-{arm}-{event}-r{1..3}.txt`, `classify.awk` (adapted copy),
`aggregate.py`, `run_{stat_battery,records,classify}.sh`, `provenance.log`,
`p11-elf-src-perf_matrix.rs` (ELF-rev source extract, one level up).
