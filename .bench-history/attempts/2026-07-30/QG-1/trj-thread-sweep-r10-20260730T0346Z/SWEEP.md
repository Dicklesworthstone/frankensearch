# QG-1 current-producer `trj` xlarge thread sweep

## Decision

**DIAGNOSTIC DIRECTION: Tantivy's observed throughput peaks at two workers and
then declines; Quill is still 4.0x to 5.5x slower across the raw sweep.**

This is the requested full `1, 2, 4, 8, 16, 32, 64, 96, 128` worker-width
sweep of the QG-1 xlarge positions-on tranche. It is not the complete
normative QG-1 matrix or its immediate reproduction. Every artifact therefore
has `laws_attested=false`, and every otherwise-admissible cell has
`admission_no_claim.code=evidence.incomplete_gate_selection`. The results do
not activate QG-1, replace the unmeasured placeholder, support promotion, or
certify a competitive claim.

Widths 1, 64, and 96 have valid Tantivy/Tantivy and Quill/Quill controls and
clear the required 2x worst-null bootstrap-median-CI floor. The other six
widths are UNSCORED because at least one predeclared A/A law failed. CV was
recorded in the raw artifacts but was never used for admission or
interpretation.

## Booking and execution provenance

- Agent Mail identity: `MaroonJay`.
- Booking thread: `trj-booking`; claim message `6493`, subject
  `[trj] CLAIM frankensearch`, expected duration up to six hours.
- Measurement interval:
  `2026-07-30T03:48:36Z` through `2026-07-30T08:35:49Z`.
- `trj` release: `2026-07-30T08:36:03Z`, 14 seconds after the successful
  sweep exit; message `6747`, subject `[trj] RELEASE frankensearch`.
- Source revision:
  `544ffeb19b519d2e6c849f68334a3eabefb3573a`, clean tree
  `659e7152e667329952975baabda10a60ab5675a4`. This was the committed current
  producer revision on `origin/codex/final-yellow-reconcile-20260729`, not
  the older `origin/main` revision.
- Build: strict-remote RCH job `j-29953680971137050` on `vmi1264463`
  (`38.242.209.154`), using exact base, clean overlay, and no local overlay:
  `RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec --base
  544ffeb19b519d2e6c849f68334a3eabefb3573a --clean-overlay --no-overlay --
  cargo build --profile release-perf -p frankensearch-quill-gauntlet
  --features perf-harness,tantivy-oracle --bench perf_matrix`.
- No local Cargo build was run. The final benchmark ELF was copied back from
  the RCH worker under the local-perf-binary policy.
- Executing ELF SHA-256:
  `e0dc6ba3c3c651e25e5693c12e053c1f77e829f38aac603f692266d8e7306ba1`
  (78,029,032 bytes).
- Toolchain: `nightly-2026-07-20`, rustc
  `1.99.0-nightly (9f36de775 2026-07-19)`, LLVM 22.1.8.
- `Cargo.lock` SHA-256:
  `393f80fd462ed19fd4c0b6ad4174aa3580bdff53d2999d610d3098f728800022`.
- Actual legacy incumbent: Tantivy 0.26.1, index format v7, lexical revision
  `062a5e5b2d41653b1c8b07888eda1a765e421f49`. Quill and Tantivy were
  statically linked into and executed by the same ELF.
- Measurement host: `threadripperje`, AMD Ryzen Threadripper PRO 5995WX,
  64 physical cores, 128 logical threads, one NUMA node, affinity `0-127`,
  performance governor. Sealed fingerprint:
  `linux-x86_64-threadripperje-128thread-AMD_Ryzen_Threadripper_PRO_5995WX_64-Cores`.
- Fixture per row: the identical deterministic one-million-document xlarge
  corpus, positions enabled, in-memory indexes, 5,000-document batches, one
  excluded warmup, ten paired blocks, 1,000 ms visibility cadence, and one
  terminal commit.
- Estimator: paired log ratio with 2,000 bootstrap resamples and seed
  `5860671082138523204`. Each invocation contained the Quill/Tantivy effect,
  Tantivy/Tantivy null, and Quill/Quill null.

## Observed workers and throughput

The worker columns below are observations, not requested-width assertions.
Each engine was observed 33 times per row. Quill used
`rayon_current_pool_width`; Tantivy used `tantivy_writer_construction`.
Every sealed observation has `min == max == configured width`.

Parentheses around a Quill/Tantivy ratio mean a null law failed and the
magnitude is diagnostic only.

| host identity | requested | Quill observed workers (`n`, min-max) | Tantivy observed workers (`n`, min-max) | Quill docs/s bootstrap median-CI | Tantivy docs/s bootstrap median-CI | Quill/Tantivy bootstrap median-CI |
|---|---:|---:|---:|---:|---:|---:|
| `threadripperje` | 1 | `33, 1-1` | `33, 1-1` | `33,381.135 [33,177.078, 33,921.241]` | `133,744.012 [131,089.807, 136,869.397]` | `0.249111 [0.245709, 0.256117]` |
| `threadripperje` | 2 | `33, 2-2` | `33, 2-2` | `35,254.033 [35,128.516, 35,815.711]` | `194,518.425 [185,201.393, 201,206.398]` | `(0.181479 [0.175571, 0.192343])` |
| `threadripperje` | 4 | `33, 4-4` | `33, 4-4` | `33,548.584 [33,075.787, 33,860.813]` | `163,632.892 [157,999.584, 167,737.859]` | `(0.203995 [0.199033, 0.212938])` |
| `threadripperje` | 8 | `33, 8-8` | `33, 8-8` | `33,896.044 [33,663.030, 33,948.887]` | `149,192.426 [146,292.697, 152,707.721]` | `(0.226381 [0.221486, 0.233467])` |
| `threadripperje` | 16 | `33, 16-16` | `33, 16-16` | `29,464.768 [29,357.674, 29,632.492]` | `129,896.195 [126,428.818, 138,316.043]` | `(0.226288 [0.214720, 0.234257])` |
| `threadripperje` | 32 | `33, 32-32` | `33, 32-32` | `25,293.486 [23,861.953, 25,330.733]` | `127,936.433 [120,622.433, 131,845.362]` | `(0.197659 [0.183384, 0.210235])` |
| `threadripperje` | 64 | `33, 64-64` | `33, 64-64` | `25,029.379 [24,753.672, 25,328.516]` | `124,060.934 [121,210.083, 124,620.657]` | `0.201267 [0.198519, 0.206829]` |
| `threadripperje` | 96 | `33, 96-96` | `33, 96-96` | `26,665.837 [26,615.804, 26,737.072]` | `114,905.833 [112,767.508, 118,482.142]` | `0.232032 [0.225772, 0.236904]` |
| `threadripperje` | 128 | `33, 128-128` | `33, 128-128` | `26,125.237 [25,012.248, 26,905.390]` | `111,044.317 [109,660.527, 113,872.262]` | `(0.230974 [0.221999, 0.241340])` |

## Independent-null admission and 2x margin

For QG-1, the null floor is the greatest absolute distance from identity
among both independent A/A bootstrap-median-CI endpoints. The required floor
is twice that value. The effect distance is
`abs(Quill/Tantivy median - 1)`. A row is interpretable only when both nulls
pass all predeclared laws, the effect median lies outside the combined null
interval, and its effect distance clears the 2x floor.

| workers | Tantivy/Tantivy median-CI | Quill/Quill median-CI | 2x worst-null floor | effect distance | admission |
|---:|---:|---:|---:|---:|---|
| 1 | `1.001443 [0.970495, 1.026785]` | `1.013170 [0.979475, 1.023041]` | `0.059010` | `0.750889` | **CLEAR**, partial-gate `no_decision` |
| 2 | `1.034992 [0.982923, 1.102447]` | `0.984835 [0.977828, 1.004762]` | `0.204894` | `0.818521` | **UNSCORED**: Tantivy width, dispersion, order, drift |
| 4 | `1.061696 [0.969460, 1.179161]` | `1.000817 [0.997241, 1.003667]` | `0.358321` | `0.796005` | **UNSCORED**: Tantivy center, width, dispersion, drift |
| 8 | `0.999866 [0.938410, 1.033428]` | `1.000471 [0.997305, 1.011666]` | `0.123179` | `0.773619` | **UNSCORED**: Tantivy drift |
| 16 | `0.977309 [0.932177, 1.019452]` | `1.000044 [0.984881, 1.005655]` | `0.135646` | `0.773712` | **UNSCORED**: Tantivy drift |
| 32 | `1.033538 [0.959830, 1.073998]` | `1.006201 [1.002004, 1.030996]` | `0.147995` | `0.802341` | **UNSCORED**: Quill center |
| 64 | `1.021125 [0.994087, 1.054955]` | `1.006177 [0.995918, 1.024535]` | `0.109911` | `0.798733` | **CLEAR**, partial-gate `no_decision` |
| 96 | `0.975105 [0.944234, 1.012092]` | `1.000283 [0.995590, 1.004703]` | `0.111533` | `0.767968` | **CLEAR**, partial-gate `no_decision` |
| 128 | `1.008719 [0.951635, 1.042144]` | `0.984563 [0.966841, 1.006051]` | `0.096731` | `0.769026` | **UNSCORED**: Tantivy order |

## Scaling breakpoint

Relative to the one-worker median:

| observed workers | Tantivy scaling | Quill scaling |
|---:|---:|---:|
| 1 | `1.0000x` | `1.0000x` |
| 2 | `1.4544x` | `1.0561x` |
| 4 | `1.2235x` | `1.0050x` |
| 8 | `1.1155x` | `1.0154x` |
| 16 | `0.9712x` | `0.8827x` |
| 32 | `0.9566x` | `0.7577x` |
| 64 | `0.9276x` | `0.7498x` |
| 96 | `0.8591x` | `0.7988x` |
| 128 | `0.8303x` | `0.7826x` |

Tantivy's directional median peaks at two observed workers
(`194,518.425` docs/s) and declines at every larger width through 128.
The two-worker lower CI (`185,201.393`) is above the four-worker upper CI
(`167,737.859`), but the two- and four-worker Tantivy nulls are invalid.
Therefore **two workers is the directional scaling stop, not a
decision-valid certified breakpoint**. The both-null-valid widths establish
only that no late scaling recovery occurs: Tantivy falls from
`133,744.012` docs/s at one worker to `124,060.934` at 64 and `114,905.833`
at 96.

Quill gains only 5.6% at two workers, is effectively flat through eight, and
falls below its one-worker throughput from 16 onward. The prior flat
Quill/Tantivy gap does not conceal a favorable high-width crossover.

## Target consequence and retry predicate

The raw Quill/Tantivy medians span `0.181479x` to `0.249111x`: Tantivy is
4.0x to 5.5x faster. On the three both-null-valid rows, Tantivy is 4.01x,
4.97x, and 4.31x faster at 1, 64, and 96 workers, respectively. Reaching a
Quill/Tantivy target of 5x from this surface requires roughly a 20x to 28x
Quill improvement; reaching 10x requires roughly 40x to 55x.

**The 5-10x-faster indexing target is decisively unmet on this tranche.**
This sweep did not measure search, so it makes no search-speed claim and the
search target remains unmeasured here.

Do not resample the unchanged implementation merely to seek favorable nulls.
The next QG-1 sweep requires a counted ingest mechanism change, the same
actual Tantivy incumbent and exact-ELF invocation, both independent A/A
controls, bootstrap median-CIs with the 2x floor, sealed observed worker
widths, the complete normative matrix, and an immediate same-ELF
reproduction. The previously measured 8.1587x copy-byte amplification gives
the strongest current mechanism route; repeat the identical-input copy
counter after a mapped copy-site change before consuming another full sweep.
