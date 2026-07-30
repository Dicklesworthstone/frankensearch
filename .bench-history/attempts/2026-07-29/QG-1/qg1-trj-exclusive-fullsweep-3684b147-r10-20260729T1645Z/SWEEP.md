# QG-1 exclusive trj xlarge thread sweep

This target slice ran under the exclusive `trj-booking` claim from
2026-07-29T16:45:32Z through 2026-07-29T21:23:28Z. It supersedes the earlier
co-tenant sweep for thread-scaling attribution.

## Provenance

- Source revision: `3684b147797c5babdad4a5568e993db40ed90da5`, clean.
- Build profile: `release-perf`, features `perf-harness` and
  `tantivy-oracle`.
- Self-reporting ELF SHA-256:
  `90bf6c4cd69606def56fd2b526a07f398a40aacad8ac7a73e77bf2653c51ed1a`
  (77,945,432 bytes).
- Subject identity: Quill 0.2.1. Incumbent identity: Tantivy 0.26.1.
  Both engines are statically linked into the same self-hashed ELF.
- Host: `threadripperje`, AMD Ryzen Threadripper PRO 5995WX, 64 physical
  cores, 128 logical threads, 536,069,869,568 bytes RAM, one NUMA node.
- Runtime ISA: AVX2, FMA, BMI2, AES, VAES.
- Governor: `performance`; affinity: `0-127`; no cpuset cap.
- Host load across row boundaries stayed between 0.67 and 3.37 on 128
  logical CPUs.
- Fixture per row: deterministic one-million-document `xlarge`,
  positions enabled, in-memory durability, one excluded warmup, ten paired
  blocks, 1,000 ms commit cadence, and one terminal commit.
- Each row used one invocation for Tantivy/Tantivy A/A, Quill/Quill A/A,
  and Quill/Tantivy A/B. Bootstrap median-CI and the predeclared null laws
  decide admission; CV is provenance only.
- The untimed worker probe samples Linux `/proc/self/task/*/schedstat`.
  `active IDs` is the count of distinct new worker thread IDs that crossed
  the 1 ms CPU threshold during the exact fixture. `peak workers` is the
  maximum simultaneous new-worker high-water and must not be confused with
  the distinct-ID count.
- All 72 files copied from trj matched their remote SHA-256 byte for byte.

## Results

The target is Quill/Tantivy `>=3.0x`. Parenthesized ratios are diagnostic
only because an A/A null failed.

| requested | Quill active IDs / peak workers | Tantivy active IDs / peak workers | Quill docs/s | Tantivy docs/s | Quill/Tantivy bootstrap median-CI | row verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 / 1 | 39 / 8 | 35,371.870 | 139,186.055 | `0.257438 [0.250622, 0.262109]` | **MISS** |
| 2 | 1 / 2 | 85 / 10 | 37,252.012 | 196,637.500 | `0.189423 [0.183838, 0.193821]` | **MISS** |
| 4 | 1 / 4 | 504 / 16 | 35,577.625 | 162,203.108 | `(0.219794 [0.214862, 0.221369])` | **UNSCORED**: Tantivy null width, dispersion, order, and drift |
| 8 | 1 / 8 | 595 / 25 | 35,854.674 | 144,720.506 | `(0.249382 [0.231247, 0.252693])` | **UNSCORED**: Quill null center |
| 16 | 1 / 16 | 720 / 39 | 32,733.036 | 132,850.797 | `(0.242964 [0.234366, 0.263901])` | **UNSCORED**: Tantivy null width |
| 32 | 1 / 32 | 984 / 75 | 27,429.559 | 125,514.970 | `0.218916 [0.209276, 0.227571]` | **MISS** |
| 64 | 1 / 64 | 1,827 / 141 | 27,422.719 | 122,979.244 | `0.219276 [0.217200, 0.226471]` | **MISS** |
| 96 | 1 / 96 | 2,759 / 193 | 29,268.189 | 113,641.286 | `(0.256653 [0.253520, 0.260727])` | **UNSCORED**: Quill null center |
| 128 | 1 / 128 | 3,556 / 264 | 27,130.469 | 109,413.724 | `0.249378 [0.236671, 0.258318]` | **MISS** |

Five rows are scoreable MISSes and four are UNSCORED. No row passes. The
scoreable high-width ratios remain about `0.219x` to `0.249x`, while Quill
uses one observed CPU-active worker at every requested width. Quill
throughput falls from 35,854.674 docs/s at requested 8 to 27,130.469 docs/s
at requested 128; the extra requested workers create overhead without
putting the measured ingest path onto multiple CPU-active workers. Tantivy
also oversubscribes heavily, but remains 4.0x to 5.3x faster in every
scoreable row.

## Decision

**TARGET-SLICE REJECT / NO QG-1 PROMOTION.** This is not a complete
normative QG-1 gate or immediate reproduction, so every evidence artifact
correctly has `laws_attested=false` and the gate remains inactive. The
partial target slice cannot replace `QG-1.unmeasured.latest.json`.

Retry only after a counted mechanism changes one of the two facts exposed
here: either reduce Quill copy bytes/calls per document on identical input,
or prove that the ingest hot path actually uses more than one CPU-active
worker. Preserve the same exact-ELF, same-invocation incumbent, dual-null,
bootstrap median-CI, topology, corpus, and affinity contract. Do not
blind-resample invalid-null rows, weaken a null law, select a favorable
width, or gate on CV.
