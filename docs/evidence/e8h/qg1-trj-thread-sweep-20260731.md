# QG-1 `trj` 1-to-128 thread sweep — Tantivy scaling stops at width 4

**Decision: STRUCTURAL / NO-CLAIM.** The continuous, receipt-bound sweep
completed all requested widths `1,2,4,8,16,32,64,96,128` against pinned
Tantivy 0.26.1. It establishes a raw single-host routing fact: Tantivy peaks
at 132,936 docs/s at requested width 4, then loses throughput at every wider
setting. It does not activate or adjudicate QG-1 because all nine row
manifests report `laws_attested=false`.

## Execution identity

- Measured source:
  `ccc37c8e611cd313201108ffe9260376a977b4bd`
- Full executable SHA-256, self-reported from inside the benchmark process
  and matched in every terminal row:
  `53ab4c0975f0ad2148e37f35641dfd56e78acd8048d01cdb8b1194aa8ab9b637`
- Exact build used `rch exec --base
  ccc37c8e611cd313201108ffe9260376a977b4bd --clean-overlay --no-overlay`;
  RCH selected `ovh-b`, synchronized zero overlay files, and produced project
  hash `885db21f6567d913`.
- Sweep runner SHA-256:
  `ccd37127d9eaab1a58f25591e97ff892c4c20e4ea1430b9e98d67e6098f3c8bb`
- Host for every row: `threadripperje`, boot
  `b107a2c6-9fac-40df-a637-c3a772b0ad57`, AMD Ryzen Threadripper PRO
  5995WX 64-Cores, 128 logical CPUs.
- Fixture: deterministic one-million-document `bulk/xlarge`, positions on,
  one warmup, ten paired blocks, continuous timing, work receipts on.
- Every row exited 0 with 66/66 receipts, zero H1/H2 wall mismatches, zero
  terminal failures, matching evidence identity, and equal final H1/H2
  walls.

Raw evidence:
`.bench-history/attempts/qg1-trj-h1h2-ccc37c8e-clean-r10-20260731T0349Z/`.
The consolidated `aggregate.json` SHA-256 is
`a1d4323d69d5c587ecefa99f85032615659f45d22b2f56ee3d77606995058e68`;
the 111-entry raw manifest SHA-256 is
`5a29364d552f848b09322854de6d537d756f6f3ea5eedde9a6390335257c2c2d`.

## Rows

The corrected null gate requires: effect median-CI excludes 1;
`abs(effect_median - 1) > 2 * max(T/T null half-width, Q/Q null
half-width)`; and each null median is within 2% of 1. CV is provenance only.

| width | host / boot ID | Quill docs/s | Tantivy docs/s | Q/T median [95% CI] | T/T | Q/Q | corrected null |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | 35,551.521 | 71,930.065 | `0.499882 [0.487438, 0.511549]` | 1.009576 | 1.001974 | PASS |
| 2 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | 36,505.685 | 116,424.790 | `0.312842 [0.306422, 0.321050]` | 1.038377 | 1.012091 | FAIL |
| 4 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | 37,270.193 | 132,936.059 | `0.277612 [0.273338, 0.284961]` | 0.973066 | 0.995543 | FAIL |
| 8 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | 35,409.346 | 126,257.903 | `0.281265 [0.269878, 0.286662]` | 0.998773 | 0.998907 | PASS |
| 16 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | 33,473.596 | 117,564.256 | `0.285757 [0.270176, 0.300316]` | 1.072212 | 0.996756 | FAIL |
| 32 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | 25,712.146 | 109,380.264 | `0.230809 [0.210160, 0.254111]` | 0.971548 | 0.996132 | FAIL |
| 64 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | 22,417.172 | 89,304.888 | `0.248270 [0.233913, 0.269171]` | 1.019307 | 1.019208 | PASS |
| 96 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | 24,465.701 | 64,092.430 | `0.382646 [0.344567, 0.413863]` | 0.999471 | 1.004594 | PASS |
| 128 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | 22,408.810 | 79,804.620 | `0.275095 [0.267489, 0.311525]` | 0.975137 | 1.001677 | FAIL |

Widths 2, 4, 16, 32, and 128 fail because the Tantivy null median is more
than 2% from 1. These are corrected diagnostic labels, not QG verdicts.

## Actual activity

Active equivalents are receipt-window process CPU integrals. Positive
dedicated workers are live dedicated threads with at least one observed CPU
tick; their resolution is 10 ms, and work from threads that exit before the
final census may be unattributed.

| width | host / boot ID | Quill active eq min/med/max | Tantivy active eq min/med/max | Quill positive workers min/med/max | Tantivy positive workers min/med/max |
|---:|---|---:|---:|---:|---:|
| 1 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | `0.999/0.999/1.001` | `1.654/1.704/1.757` | `0/0/0` | `2/5/5` |
| 2 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | `0.999/0.999/1.001` | `2.526/2.690/2.955` | `0/0/0` | `2/6/7` |
| 4 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | `0.999/0.999/1.000` | `4.405/4.639/4.915` | `0/0/0` | `8/9/10` |
| 8 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | `0.999/0.999/1.001` | `5.576/6.169/6.547` | `0/0/0` | `6/13/14` |
| 16 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | `1.000/1.001/1.002` | `5.684/6.235/7.599` | `0/0/0` | `5/21/22` |
| 32 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | `1.001/1.001/1.002` | `5.459/6.950/8.336` | `0/0/0` | `19/37/37` |
| 64 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | `1.007/1.009/1.160` | `6.659/7.160/8.066` | `0/0/62` | `5/69/70` |
| 96 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | `1.012/1.050/1.354` | `5.525/7.147/7.778` | `0/36/86` | `5/101/102` |
| 128 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | `1.012/1.052/1.238` | `7.148/9.132/10.051` | `0/38/74` | `5/130/136` |

The live census separately sees `requested - 1` Quill Rayon workers and
`2 * requested` Tantivy index workers plus support threads. That is
liveness/configuration, not productive concurrency: at width 128, Tantivy
has a median 130 dedicated workers with a positive tick but only 9.132
aggregate active equivalents.

## Structural conclusion and retry boundary

Tantivy scales `1.000x, 1.619x, 1.848x` relative to width 1 at widths 1, 2,
and 4. Relative to its width-4 apex, widths 8, 16, 32, 64, 96, and 128 are
`0.950x, 0.884x, 0.823x, 0.672x, 0.482x, 0.600x`. Its observed scaling
stops at requested width 4. The structural search/index opportunity is to
beat the incumbent around this low productive-concurrency ceiling rather
than reproduce its worker explosion.

The effect block remains sequential and disjoint from both null blocks, and
effect order/drift is not separately gated. That caveat, incomplete
normative selection, and `laws_attested=false` make the tranche no-claim.
QG-1 remains inactive and its unmeasured placeholder remains authoritative.

Retry only with the complete normative 74-cell bundle and immediate
same-ELF reproduction. Interleave null and effect blocks or bind effect
order/drift explicitly; ratchet-bind continuous timing and work-receipt
modes; preserve exact host, executable, wall, and actual-activity receipts.
