# W1.3 collector allocation census — NoClaim (collectors), redirect to TERMDICT

**Bead:** `bd-e8h-w1-collector-alloc-tc4q0` · **Agent:** MossyPine · **Date:** 2026-07-28
**Source:** origin/main `f174befb74c5949f7ef0a7629328957912a61f2c` (clean git-archive extract; scratchpad harness, no repo code touched)
**Host:** AMD Ryzen Threadripper PRO 5975WX 32c/64t, Linux x86_64 (dev host — *not* a registered E8-H timing class; this census reports **allocation counts and frame attribution only**, which are microarchitecture-independent; no latency claim is made for any machine class).

## Method

Two harnesses over the identical deterministic workload (seeded-LCG corpus, 8k vocab,
quadratic skew; 80–160 body words + title; `QuillIndex::in_memory`,
`deterministic_ingest`, commit-sealed segments):

1. **Counting `GlobalAlloc`** (wraps `System`, atomics for alloc/realloc/bytes):
   per-query deltas over 200 measured queries after 50 warmups, per class
   (`term1`, `union3`, `phrase2`, `mixed5`) × k∈{10,100}, `search_paginated`,
   `exact_count=false`. Query strings pre-built outside the measured window.
2. **dhat frame attribution** (`dhat::Alloc`, profiler started *after* indexing
   and warmup): same workload, symbolized release build
   (`CARGO_PROFILE_RELEASE_DEBUG=1 STRIP=none`).

Scales: smoke = 500 docs / 1 commit; hundredk = 100,000 docs / 20 commits
(≤20 sealed segments; keeper tiering may merge).

## Results

Counting census (full tables in `alloc-census.{smoke,hundredk}.json`):

| scale | class | k=10 allocs/q | k=10 bytes/q |
|---|---|---:|---:|
| smoke | term1 | 96 | 301 KB |
| smoke | mixed5 | 407 | 1.52 MB |
| 100k | term1 | 1,049 | **18.2 MB** |
| 100k | union3 | 2,763 | **54.2 MB** |
| 100k | phrase2 | 11,714 | **36.3 MB** |
| 100k | mixed5 | 4,566 | **90.1 MB** |

Two structural facts: bytes/query scale **per query term** (~18 MB/term at 100k) and are
**k-independent** (k=10→100 moves allocs by <2%), i.e. the churn is in per-term
open/decode paths, not in result-window machinery.

dhat attribution, hundredk, 1,600 measured queries, 79.49 GB / 12.46 M blocks total
(`dhat-hundredk.symbolized.json`):

| attribution (by allocating frame) | bytes | blocks |
|---|---:|---:|
| `grimoire::RestartMeta`/`BlockMeta`/`IndexRecord` vec growth (TERMDICT metadata reparse) | **98.28 %** | 58.34 % |
| collector-owned (`TopDocsCollector::build`, per-segment collector vec) | **0.011 %** | 0.27 % |
| largest single collector-owned frame | 0.0027 % | 0.04 % |
| runner-up non-TERMDICT: `argus::BufferedUnionScorer::new` (aggregate) | ~0.9 % | — |

## Verdict

Per the bead's own falsifier ("dhat shows no per-query collector allocs >=0.1%"):
**FALSIFIED → NoClaim.** Collectors already allocate O(1) small per query
(`try_reserve_exact` heap + winners vec); no single collector frame reaches 0.1% on
either metric. Lazy materialization of collectors cannot move a ~50 MB/query workload.

**Redirect (quantified):** the census independently confirms
`bd-quill-gauntlet-qg6-cache-termdict-gwd4` from the allocation axis — ~98% of all
query-phase allocated bytes rebuild immutable TERMDICT metadata
(`RestartMeta`/`BlockMeta` vectors) per term × per segment × per query (doc-freq open +
sealed-cursor open both re-parse). At 100k docs this is 18 MB allocated per single-term
query. This is the same >83% self-time attribution the exact QG-6 profile found at
frozen `8b35790e`, now with an allocation-side magnitude. gwd4's owned-metadata cache
should eliminate ~98% of query-phase allocation traffic in one lever.

**Retry predicate for W1.3:** re-census collectors only if a landed TERMDICT lever drops
grimoire frames below ~10% of query-phase bytes AND QG-6 still misses; only then can
collector churn be a material residual.
