# QG-1 trj 5995WX full-SMT scaling tranche

## Decision

**PARTIAL TARGET-SLICE MISS: two rows scored, seven rows UNSCORED.**
The 1-thread and 96-thread rows passed both same-invocation A/A controls and
missed QG-1's `>=3.0x` Tantivy target. The other seven rows are retained as
diagnostics only because the Tantivy/Tantivy control failed a predeclared null
law. Every Quill/Quill control passed.

This sweep selected only the positions-on xlarge scaling tranche, not QG-1's
complete normative matrix or an immediate reproduction. It therefore does not
activate QG-1, does not replace `QG-1.unmeasured.latest.json`, and does not
support a competitive claim.

## Execution provenance

- Source revision:
  `bb31daa04ee58f9c38c9a0d6e42b5a125e6f02ae` (clean).
- Executing `release-perf` ELF SHA-256:
  `a466d5a64a67843a8f2acd4b7add23c25b2015241c64ea35b385ae59431b8c12`
  (76,985,272 bytes).
- Linked incumbent: Tantivy 0.26.1, index format v7.
- Artifact/evidence schemas: `quill-perf-artifact-v4` and
  `quill-perf-evidence-v2`.
- Host: `threadripperje`, AMD Ryzen Threadripper PRO 5995WX, 64 physical
  cores, 128 logical threads, one NUMA node, 499 GiB RAM.
- Runtime ISA: AVX2, FMA, BMI2, AES, and VAES; no AVX-512 was detected.
- Effective affinity: CPUs `0-127`; no affinity or cpuset cap. Every row
  records its actual requested thread count separately.
- Fixture: 1,000,000 deterministic synthetic documents, positions on,
  5,000-document batches, in-memory indexes, one excluded warmup, and ten
  paired blocks.
- Corpus recipe: seed `5860671082139239762`,
  `synthetic-zipf-s11-vocab8192-doc4096-v1`. The evidence corpus digest is
  cell-scoped and includes the fixture name, so it differs by thread count;
  the generator seed, recipe, count, generated order, and document bytes are
  otherwise identical across the sweep.
- Peak process RSS was 22,031,704,064 bytes at 128 threads, well below
  available RAM. Swap usage stayed effectively unchanged, and both indexes
  were in memory.
- The first host receipt recorded load `129.46`; the last recorded `4.65`.
  Host load was not used as an admission gate. Each row's own same-invocation
  A/A controls adjudicated noise.

## Fairness audit

| Surface | Verdict | Evidence |
|---|---|---|
| Analyzer | FAIR | Both arms use the shipping Frankensearch default analyzer semantics: Unicode-alphanumeric token boundaries plus lowercase, without stemming or stop-word removal. |
| Schema | FAIR | Both arms use `frankensearch-default-v1`, positions enabled, and the same stored/indexed field contract; every fixture preflight passed. |
| Commit policy | FAIR configuration | Both arms use the shipping 1,000 ms visibility cadence, in-memory durability, and one terminal commit inside the timed operation. Realized periodic commits differ because Quill takes longer; logs retain every count. |
| Heap budget | FAIR | Equal total heap at each point: `max(50,000,000, 15,000,000 * threads)` bytes for both arms. |
| Corpus | FAIR | Same deterministic one-million-document recipe, seed, order, and 5,000-document batches in both arms and across thread points. Corpus generation is outside the engine interval. |
| Build profile | FAIR | Both arms are linked into and executed from the same self-reporting `release-perf` ELF. |

Realized periodic-commit ranges across the 33 measured/null/warmup operations
per point were:

| Threads | Quill commits | Tantivy commits |
|---:|---:|---:|
| 1 | 24-27 | 10-13 |
| 2 | 26-27 | 11-12 |
| 4 | 24-25 | 11-12 |
| 8 | 23-24 | 11-12 |
| 16 | 25-27 | 12-13 |
| 32 | 29-34 | 12-13 |
| 64 | 28-31 | 12-14 |
| 96 | 28-33 | 13-14 |
| 128 | 28-31 | 13-14 |

## Results

Ratios and confidence intervals below come from the paired evidence estimator,
not from CV. `T null` is Tantivy/Tantivy and `Q null` is Quill/Quill.
UNSCORED rows show their raw diagnostic A/B magnitude in parentheses.

| Threads | Equal heap bytes | Quill docs/s | Tantivy docs/s | Quill/Tantivy median-CI | T null median-CI | Q null median-CI | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 50,000,000 | 34,896.923 | 135,684.166 | `0.256083 [0.250163, 0.258549]` | `1.009191 [0.973456, 1.036773]` | `0.995811 [0.982944, 1.028535]` | **MISS** |
| 2 | 50,000,000 | 34,543.200 | 165,817.834 | `(0.208493 [0.195344, 0.225094])` | invalid: dispersion + order + drift | `0.996405 [0.987534, 1.006858]` | **UNSCORED** |
| 4 | 60,000,000 | 35,589.081 | 173,165.385 | `(0.205067 [0.191886, 0.209096])` | invalid: width + dispersion + drift | `0.994270 [0.988540, 1.008535]` | **UNSCORED** |
| 8 | 120,000,000 | 33,770.710 | 142,833.746 | `(0.237273 [0.231981, 0.245118])` | invalid: width + order | `1.004559 [0.997818, 1.010668]` | **UNSCORED** |
| 16 | 240,000,000 | 32,791.136 | 136,259.514 | `(0.240605 [0.235825, 0.248195])` | invalid: width + dispersion | `1.003461 [0.993342, 1.006587]` | **UNSCORED** |
| 32 | 480,000,000 | 25,142.024 | 117,388.872 | `(0.209419 [0.199148, 0.216683])` | invalid: dispersion | `1.006079 [0.989295, 1.013863]` | **UNSCORED** |
| 64 | 960,000,000 | 28,562.221 | 126,486.750 | `(0.226495 [0.221930, 0.231632])` | invalid: order + drift | `1.005064 [0.999291, 1.012646]` | **UNSCORED** |
| 96 | 1,440,000,000 | 26,570.020 | 112,658.279 | `0.236489 [0.233668, 0.239445]` | `0.982554 [0.956492, 1.026133]` | `0.994896 [0.985826, 1.004710]` | **MISS** |
| 128 | 1,920,000,000 | 28,211.911 | 108,901.704 | `(0.254452 [0.248195, 0.263064])` | invalid: dispersion + order + drift | `0.999793 [0.994249, 1.005438]` | **UNSCORED** |

The scoreable comparison does not narrow at high thread count:
`0.256083` at one thread versus `0.236489` at 96 threads. The complete raw
diagnostic curve stays between `0.205067` and `0.256083`, with endpoints
`0.256083` and `0.254452`. That is evidence of no manifested high-thread
concat-merge advantage in this tranche, but it is not a certified nine-point
scaling curve because seven incumbent nulls failed.

Evidence JSON file SHA-256 values in thread order 1, 2, 4, 8, 16, 32, 64, 96,
128 are:

`ad016ef3ca00afbd3f203a16c84d4b2dd6471d0106eab9fef85fe24a889dbd9e`,
`33bd8fa56142087df9e148dbcdb0c264b69042e04daa51c757b263a35dd9829d`,
`f3778bf4966ba4f7f8f7fd9670346f65d7e89cdb562eb7a71d3490d986d58c48`,
`8a38737ffad3dc93f2f9dc04e0887843f0802850a69d1e31e5e631e26b594576`,
`16638cc7297969fbd7e1a329aa0c30726bab2bacc3a9beb45c94d7cbc7066d62`,
`12bd3cd4851d81345ffa845f1d415f1db33b0f9bb54ce11fe969f549b7ec8087`,
`93830d883341867119c8c1f0b2e9800be83eef2de197ac54a7c47e1a36829b1a`,
`9f71d9546f55e3feb555fb2f3c97ce131c4170807097fe17160ce13d964a239e`,
and
`b9448dc8fd6100d34169fd1ecdd261fb6529386cd9c6dd04fb971222bbf3e04a`.

## Retry predicate

- For the valid 1- and 96-thread MISS rows, retry only after a
  profile-attributed Quill hot-path change, with the same corpus, policy,
  topology provenance, exact-ELF self-report, dual A/A controls, and
  median-CI decision rule.
- For the seven invalid rows, do not blind-resample this unchanged route.
  Retry on a distinct eligible 64-core/128-thread machine or after a counted
  mechanism removes the Tantivy null's commit-boundary/order/drift failure.
  Require both arm nulls to pass before interpreting the A/B magnitude.
- Any QG-1 activation still requires the complete normative matrix plus an
  immediate same-ELF reproduction within the fixed reproduction bound. Never
  weaken a null law, select a favorable row, or gate on CV.
