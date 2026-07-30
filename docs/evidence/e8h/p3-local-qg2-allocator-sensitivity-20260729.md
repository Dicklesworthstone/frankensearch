# P3 — Allocator/THP symmetric sensitivity survey (local-5975wx-32c)

Date: 2026-07-29. Author: SandyGrove (coordinator, E8-H loop pass 3 — run by the
orchestrator directly after repeated server-side subagent terminations).
Bead lineage: follows P1 (`6cb219d9`) and bd-w8dut P2 (`88264af1`).

## Scope and honesty caveat (read first)

This survey uses the **memory-mode child seam** (`QUILL_PERF_CHILD_MODE=memory`,
200k docs, threads=1, positions on, smoke scale) — the same arm-scoped
diagnostic denominator as the P1 profile and the P2 A/B. It is NOT the QG-2
bulk gate cell: absolute docs/s below are seam-relative, and the
quill-vs-tantivy columns are **diagnostic only** — they must never be quoted
as gate numbers. Facade-level QG-2 remains inactive; the prior 0.1113 value
was a superseded, pre-current Ryzen 7 5800X pass-A diagnostic from commit
`351f5c6d`, and its stale latest pointer is excluded. Within-arm allocator
ratios (same ELF, same workload, env-only change) are the survey's valid
product. Machine class **local-5975wx-32c**; Law 6: not transferable.

## Fingerprint

- Host: AMD Threadripper PRO 5975WX 32c/64t (Zen 3), NPS1, governor powersave
  (amd-pstate-epp), kernel 6.17.0-35-generic, glibc 2.42 (Ubuntu 2.42-0ubuntu3.1).
- THP: `always [madvise] never` (madvise; READ ONLY — unchanged by this survey).
- Allocators: system glibc; jemalloc `/lib/x86_64-linux-gnu/libjemalloc.so.2`
  via LD_PRELOAD. mimalloc NOT present on host (fallback per plan: glibc
  tunable cell `glibc.malloc.hugetlb=1` instead).
- ELF (single, all cells): `perf_matrix_base_iso`, SHA-256
  `9c3cacf0fa0ab66b46b9fb9482c1b8e858985a02b4e7775ef47dec574f22078b`
  (identical to the P2 A/B baseline ELF; origin/main 3684b147 overlay,
  release-perf, force-frame-pointers).
- Method: 7 cells x (1 untimed warmup + 10 timed runs), rotated round-robin
  order per round, taskset core 8, external wall time, RCH_DISABLE=1,
  runner `scratchpad/p3/p3_matrix.sh`, raw data `scratchpad/p3/matrix.tsv`.

## Cells and per-cell results (docs/s = 200000 / external wall)

| cell | engine | allocator env | median | p5 | p95 | n |
|---|---|---|---:|---:|---:|---|
| QG  | quill   | glibc default | 37,665 | 31,459 | 38,592 | 10 |
| QG2 | quill   | glibc default (null arm) | 37,722 | 35,703 | 38,108 | 10 |
| QJ  | quill   | LD_PRELOAD=jemalloc | 35,374 | 33,884 | 36,563 | 10 |
| QH  | quill   | GLIBC_TUNABLES=glibc.malloc.hugetlb=1 | 37,327 | 32,487 | 37,745 | 10 |
| TG  | tantivy | glibc default | 41,984 | 39,995 | 42,463 | 10 |
| TJ  | tantivy | LD_PRELOAD=jemalloc | 44,793 | 43,806 | 45,117 | 10 |
| TH  | tantivy | GLIBC_TUNABLES=glibc.malloc.hugetlb=1 | 41,970 | 40,627 | 42,862 | 10 |

Paired per-round ratios (same round index across cells):

| ratio | median [p5, p95] | reading |
|---|---|---|
| A/A null QG/QG2 | 0.9983 [0.8808, 1.0248] | admissible (contains 1.0; one slow round widens p5) |
| quill jemalloc/glibc | **0.9563 [0.9050, 1.1431]** | straddles 1.0 — no quill-arm win; point estimate mildly negative |
| quill hugetlb/glibc | 0.9916 [0.8577, 1.2010] | wash |
| tantivy jemalloc/glibc | **1.0678 [1.0471, 1.1084]** | real incumbent-arm improvement, CI clear of 1.0 |
| tantivy hugetlb/glibc | 1.0070 [0.9828, 1.0349] | wash |

Diagnostic-only seam gap rows (NOT gate numbers, see caveat): quill/tantivy on
this seam is ~0.90 under glibc and ~0.80 under jemalloc — this seam's workload
is dominated by shared child overhead and does not reproduce the bulk gate gap.

## Verdict

**The allocator lever family on the QUILL arm is DEAD on this class and seam.**
Neither jemalloc substitution nor the THP malloc tunable moves the quill arm
(both CIs straddle 1.0; jemalloc's point estimate is negative). This is
consistent with P1's attribution: quill's memmove family is data-copy-shaped
(~5.2%) with only ~0.45% allocation-growth, and glibc allocator self-time
(~4.8%) is evidently not elastic to allocator choice at this workload.

**The TANTIVY arm IS allocator-sensitive (+6.8% under jemalloc).** Not
actionable for the campaign (gate law: shipped defaults on both arms), but
diagnostically important: the incumbent's remaining allocator appetite means
allocator-based levers would, if anything, WIDEN the gap when applied
symmetrically.

**Routing-only class contrast:** the superseded 0.1113 Ryzen diagnostic and
the retained M4 `invalid_null` attempt
`m4-macos-qg2-receipt-w5-r30-20260729T021107Z` at
`0.528360 [0.404084, 0.544504]` never formed a certified class split. The
available diagnostic contrast is further decoupled from "glibc allocator
hurts quill on Linux." Remaining suspects: per-uarch codegen of the compute
families
(canonicalization/identity, interner probing) and incumbent-side platform
behavior. The next discriminating probe belongs to the per-class profile
lanes (FoggyPrairie: trj/m4).

**Retry predicate:** reopen the allocator family only if (a) a bulk gate-cell
allocator A/B on a certified class moves the quill arm's median ratio by >3%
with an admissible null, or (b) a certified-class arm-scoped profile
attributes >8% of quill self-time to allocator frames.

## Repro

```
scratchpad/p3/p3_matrix.sh 10        # writes scratchpad/p3/matrix.tsv
# stats: python block in session log; medians/percentiles over 200000/wall
# cells: {QG,QG2,QJ,QH,TG,TJ,TH} as defined in p3_matrix.sh cell_env()
```
