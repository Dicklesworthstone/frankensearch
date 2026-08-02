# E8-H P7 — QG-2 ingest codegen sensitivity (target-cpu G/N/Z), LOCAL

**Publication provenance:** integrated after `a61456ec` in evidence-set
commit `5b91f680`; measurements are diagnostic/NoClaim per card scope.

**Status: PUBLISHED NULL RESULT, hypothesis evidence only.** The measurement
pass itself made zero shared-tree edits, commits, or tracker mutations; this
publication adds the card and its explicit SURVEY row. Raw session artifacts
remain under `scratchpad/p7/`.

**Machine class: local-5975wx-32c** (AMD Ryzen Threadripper PRO 5975WX, Zen 3,
Linux 6.17.0-35-generic). Diagnostic-only class. **Every number in this card is
seam-diagnostic, never gate evidence** — the QG-2 smoke memory-child external-wall
seam is NOT the gate cell, and no certified/current gate numbers exist (the
"QG-2 activated 0.35" brief was quarantined; see campaign-status memory note).

## Question

The QG-2 single-thread ingest deficit vs the Tantivy oracle splits enormously by
machine class (~9x on the x86 5800X diagnostic, ~2.9x on trj-zen3-16c, ~1.9x on
an M4 invalid-null attempt). Pass 3 killed the allocator explanation. Surviving
hypothesis: per-uarch codegen — the portable build may leave more on the table
for quill than for tantivy on some microarchitectures. This pass tests codegen
sensitivity of BOTH arms on the local Zen 3 class by building the SAME source at
three codegen levels and A/B-ing all cells.

## Verdict

**NULL — the QG-2 memory-child ingest seam is codegen-INSENSITIVE for both
engines on Zen 3.** `-C target-cpu=native` and `-C target-cpu=znver3` move quill
by ≤0.4% (median-paired) and tantivy by ≤0.6%, all inside the A/A null band. The
quill/tantivy seam gap is FLAT across codegen levels (fast-tail 0.9027 / 0.9027 /
0.9010). This is a load-bearing null: the flags demonstrably rewrote the code
(ymm sites x74, 42% of all text symbols changed size, including the interner,
tokenizer, xxh3, and quiver SIMD-unpack families in quill AND 42% of tantivy's
symbols) and throughput did not respond. On this class, QG-2 ingest for both
arms is not autovectorization/ISA-width-bound — consistent with the W13 census
(ingest cost is SPREAD across memory/allocation-bound families).

- **Comparison class:** build-flag A/B on both arms; the quill-vs-tantivy seam
  columns are incumbent-adjacent but diagnostic-only at this seam/scale.
- **Mechanism perf-sample trigger (>=3% quill move): NOT met** — no arm-scoped
  perf pass was run, by protocol.

## Provenance / builds

- Source: pristine `git archive 3684b147` ("feat(perf): record observed thread
  provenance", P1/P2/P6 continuity base) into `scratchpad/p7/overlay/frankensearch`;
  `../fast_cmaes` sibling symlink; **Cargo.lock copied from the pass-2 overlay**
  (sha256 `9755f71a…`). Deliberately NO working-tree overlay (P6 deviation note
  honored: the local tree is behind origin/main at relevant sites).
- Toolchain: rustc 1.99.0-nightly (9f36de775 2026-07-19), pinned by the archive's
  `rust-toolchain.toml` (nightly-2026-07-20) — identical to P1/P2/P6.
- Build command (per level, isolated CARGO_TARGET_DIR, `RCH_DISABLE=1` wrapper
  `p7/cargo-p7-{g,n,z}.sh`):
  `cargo bench -p frankensearch-quill-gauntlet --features perf-harness --profile release-perf --bench perf_matrix --no-run`
- Levels and full-ELF SHA-256 (details + .text-only SHAs in `p7/elf-shas.txt`):
  - **G** `-C force-frame-pointers=yes` → `ae96a2ac…`
  - **N** `… -C target-cpu=native` → `4a66593a…`
  - **Z** `… -C target-cpu=znver3` → `a819b006…`
- **Pass-2 SHA check:** G does NOT reproduce the pass-2 stash `9c3cacf0…`
  byte-for-byte despite identical flags/toolchain/lock — the per-pass overlay
  path and CARGO_TARGET_DIR are baked into debuginfo. Per the pass brief, P7's
  own G was used for all arms (flag-parity exact across levels). Behaviorally
  G matches the P2/P3/P6 baseline (GQ ≈ 37-38k docs/s vs P6's ≈ 38.8k; P3's QG
  walls 5.17-5.69s vs P7 GQ walls centered 5.36s).
- **N vs Z:** .text sections differ (`669170ca…` vs `4dfb762a…`) → native
  detection is not byte-equivalent to explicit znver3; the matrix was NOT
  collapsed (6 cells kept).
- Flag-took-effect proof: ymm-instruction lines G=1,085 / N=80,052 / Z=78,316;
  per-symbol size diff G→N: 4,977/11,815 (42.1%) common text symbols changed,
  incl. quill 1161/2486, interner 190/415, tokenizer 41/107, xxh 7/8, quiver
  (banded wide::u32x8 unpack) 115/199, tantivy 1429/3410. NOTE: the banded
  u32x8 dispatcher (widths 4-28) was tuned on generic codegen and its code DID
  change under N/Z — with zero measurable wall effect at this seam.

## Method

QG-2 smoke memory child (P3 method, `p7/p7_matrix.sh`): `QUILL_PERF_CHILD_MODE=memory
COUNT=200000 HEAP=50000000 THREADS=1 POSITIONS=true SCALE=smoke`, engine via
`QUILL_PERF_CHILD_ENGINE={quill,tantivy}`, `taskset -c 8`, external wall time,
docs/s = 200000/wall. 7 cells: {G,N,Z}x{quill,tantivy} + **GQ2** (A/A twin of
GQ, same ELF+env). One untimed warmup per cell, then rotated round-robin,
**n=28 rounds/cell** (196 timed runs, 4 foreground chunks). Tantivy compiles
from the same workspace build, so target-cpu applies to both arms identically —
flag-parity within a level is automatic.

**Environment caveat:** ambient load from unrelated agent jobs spiked mid-series
(15-min load up to 8.6 on 64 threads; 10/196 walls >5.9s, clustered in rounds
4-8). Nothing else was pinned to core 8. Full-series means are noise-widened;
medians and the fast-tail estimator (below) are spike-robust, since
contamination only ever slows runs.

## Cells (docs/s, n=28 each)

| cell | median | p5 | p95 |
|---|---|---|---|
| G-quill (GQ) | 37,338 | 31,277 | 38,516 |
| G-quill twin (GQ2) | 37,752 | 32,681 | 38,135 |
| G-tantivy (GT) | 41,965 | 38,088 | 42,690 |
| N-quill (NQ) | 37,836 | 33,859 | 38,289 |
| N-tantivy (NT) | 41,740 | 38,344 | 42,503 |
| Z-quill (ZQ) | 37,772 | 32,570 | 38,392 |
| Z-tantivy (ZT) | 41,273 | 34,961 | 42,441 |

## Ratios (paired-by-round docs/s, >1 = numerator faster; n=28)

| ratio | mean [95% t-CI] | median | fast-tail* |
|---|---|---|---|
| A/A null GQ2/GQ | 1.0002 [0.9780, 1.0224] | 0.9985 | 0.9927 |
| **quill native N/G** | 1.0056 [0.9592, 1.0520] | **0.9998** | **0.9965** |
| **quill znver3 Z/G** | 1.0049 [0.9765, 1.0332] | **0.9976** | **0.9971** |
| tantivy native N/G | 1.0045 [0.9823, 1.0266] | 0.9961 | 0.9966 |
| tantivy znver3 Z/G | 0.9861 [0.9673, 1.0050] | 0.9939 | 0.9990 |
| seam G quill/tantivy | 0.8888 [0.8666, 0.9109] | 0.8936 | 0.9027 |
| seam N quill/tantivy | 0.8903 [0.8513, 0.9292] | 0.9105 | 0.9027 |
| seam Z quill/tantivy | 0.9033 [0.8894, 0.9172] | 0.9040 | 0.9010 |

\* fast-tail = ratio of medians of each cell's 10 fastest runs — contamination-
immune upper-bound estimator (load spikes only slow runs). Method precision is
bounded by the A/A twin's own fast-tail 0.9927 (~0.7%).

Readings:
- Quill codegen sensitivity: **zero within ~1% method precision** (medians
  0.9998/0.9976; fast-tails 0.9965/0.9971 vs A/A twin 0.9927). A true >=3%
  effect is excluded by both robust estimators independently.
- Oracle sensitivity: same null (0.9961/0.9939 medians). ZT/GT's full-series
  mean 0.9861 is inside the A/A band and contradicted by its own fast-tail
  0.9990 — noise, not a znver3 regression.
- Seam gap: ~0.90 at every level (tantivy ~1.11x faster at this external-wall
  smoke seam; the certified-style 8.7x headline lives at a different
  scale/method and is unaffected by anything here). Codegen does not move the
  seam on Zen 3.

## Ledger pre-flight (adjacent rows, cited + distinguished)

- **2026-06-25 (BlackThrush)** AVX2-build dot-kernel row: `-Ctarget-cpu`/
  `target-feature` cannot ship as a published-library default (SIGILL on
  non-AVX2 hosts); deploy-time flag recommendation only. Covers DENSE dot
  kernels, not lexical ingest — P7 extends the map to ingest and finds no gain
  even as a deploy flag on Zen 3.
- **2026-07-10 (cc_fse)** ISA-baseline row: dense hot kernels are runtime-
  dispatched (`#[target_feature]` + `is_x86_feature_detected!`), so target-cpu
  buys ~0 there. Different mechanism from P7's finding: quill ingest has no
  such dispatch — its codegen genuinely changed under N/Z — and STILL didn't
  speed up; the seam is simply not ISA-width-bound on this class.
- **2026-07-16 (BlackThrush)** bd-yt8m pair: x86-64-v3 is 3.4-4.9x at
  wide::f32x8 SITES but a wash on the dense production path. Same shape as P7:
  large local codegen deltas, no end-to-end movement.
- **bd-7zjk (2026-07-27)** core::simd `#[target_feature]` REJECTED ON
  PACKAGING (nightly `portable_simd`). That rejection covers a SOURCE change;
  it does NOT cover `-C target-cpu` BUILD flags (no source/feature change) —
  distinguished, and P7 now supplies the measurement that the build-flag form
  is value-zero on Zen 3 ingest anyway.

## Mechanism

Not sampled — the pre-declared trigger (quill moves >=3%) was not met (moved
<0.5% on both robust estimators). The structural evidence (ymm census +
per-symbol diff) shows the codegen lever pulled hard on exactly the suspected
families (interner probing, tokenizer, xxh3, quiver banded unpack) with zero
wall response, i.e., those families are bound by memory/branch/alloc behavior,
not by instruction selection — consistent with the W13 alloc census (quill
ingest cost is SPREAD; EncodedSegment::clone per commit) and the P6 finding
that the win there came from removing hash WORK, not from wider vectors.

## Routing recommendation (deliverable = hypothesis evidence, not a KEEP)

1. **Class-split mechanism:** per-uarch CODEGEN of the portable build is now
   disconfirmed as the explanation on the one class where it could be tested
   cheaply (Zen 3). The 9x-vs-2.9x split must come from something codegen
   flags don't touch: uarch runtime behavior (cache hierarchy, branch
   prediction, store-queue behavior against quill's memory-bound families),
   machine state (DRAM config), or per-class measurement-method drift.
   **Route next:** replicate this exact 3-level matrix on the 5800X diagnostic
   class where the 9x lives — if quill IS codegen-sensitive there, the lever is
   class-specific; if not (expected, given this null), pivot the class-split
   investigation to perf-counter comparison across classes (IPC, L1d/L2/LLC
   misses, branch misses on the same G binary).
2. **Build-flag lever:** dead on Zen 3 for QG-2 ingest, both arms. Do not add
   `-C target-cpu` to perf-runner profiles expecting ingest movement on this
   class. (The packaging reality stands regardless: a `target-cpu=native`
   default cannot ship on crates.io; the only shippable forms would be
   per-class CI/perf-runner build profiles — now evidence-free on Zen 3 — or
   `#[target_feature]` dispatch in source, which re-opens the bd-7zjk
   packaging question and now also lacks a demonstrated ingest-side payoff.)
3. **The published NEGATIVE_EVIDENCE row remains a SURVEY/NULL.** Do not
   promote this diagnostic pass into a lever verdict or QG claim.

## Repro

```bash
# builds (wrappers set RCH_DISABLE=1, isolated targets, per-level RUSTFLAGS)
scratchpad/p7/cargo-p7-g.sh bench -p frankensearch-quill-gauntlet \
  --features perf-harness --profile release-perf --bench perf_matrix --no-run
scratchpad/p7/cargo-p7-n.sh bench ... # same args
scratchpad/p7/cargo-p7-z.sh bench ... # same args
# matrix (chunkable; warmups fire when start==0)
scratchpad/p7/p7_matrix.sh 0 7 && scratchpad/p7/p7_matrix.sh 7 14 \
  && scratchpad/p7/p7_matrix.sh 14 21 && scratchpad/p7/p7_matrix.sh 21 28
# stats + structural checks
python3 scratchpad/p7/p7_stats.py scratchpad/p7/matrix.tsv
python3 scratchpad/p7/p7_symdiff.py
```

Artifacts: `scratchpad/p7/` — `matrix.tsv` (196 rows: cell, round, wall_s),
`elf-shas.txt`, `elfs/perf_matrix_{G,N,Z}`, `cargo-p7-{g,n,z}.sh`,
`p7_matrix.sh`, `p7_stats.py`, `p7_symdiff.py`, this card.
