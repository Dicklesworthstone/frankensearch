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
  walls. A post-run collector audit described below invalidates the
  CPU-derived fields inside those receipts; it does not change the recorded
  host, worker census, wall, throughput, or executable identity.

Raw evidence:
`.bench-history/attempts/qg1-trj-h1h2-ccc37c8e-clean-r10-20260731T0349Z/`.
The consolidated `aggregate.json` SHA-256 is
`a1d4323d69d5c587ecefa99f85032615659f45d22b2f56ee3d77606995058e68`;
the 111-entry raw manifest SHA-256 is
`5a29364d552f848b09322854de6d537d756f6f3ea5eedde9a6390335257c2c2d`.

## Post-run receipt integrity correction

The measured collector at `ccc37c8e` replaced the independently observed
process CPU delta with `max(process_cpu, role_cpu_sum)` when non-atomic
`/proc` reads disagreed. The original pre-floor value is not retained.
Across all 594 receipts, process CPU, the derived active-concurrency
integral and mean, role/unattributed CPU decomposition, and positive-CPU-tick
worker counts are therefore **CONTAMINATED / VOID FOR INFERENCE**. In
particular, the earlier `4.639 -> 9.132` Tantivy active-equivalent statement
and the resulting "productive-concurrency ceiling" interpretation are
retracted.

The raw receipt bytes remain immutable. The independent thread names/roles
in each sampled census remain usable as liveness observations because the
collector did not synthesize or floor thread membership. Throughput and wall
measurements are also retained as source-bound diagnostics, not QG evidence.

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

## Actual observed worker census

These are thread-membership observations, not CPU-activity estimates.
`Quill all` counts the benchmark caller executing the synchronous write plus
dedicated Rayon workers. `Tantivy support` counts segment-updater, merge, and
docstore-compressor threads. Each min/median/max is over the 33 receipts for
that engine at the named width.

| width | host / boot ID | Quill all min/med/max | Quill Rayon min/med/max | Tantivy index min/med/max | Tantivy support min/med/max | Tantivy all min/med/max |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | `1/1/1` | `0/0/0` | `2/2/2` | `5/6/6` | `7/8/8` |
| 2 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | `2/2/2` | `1/1/1` | `4/4/4` | `6/8/9` | `10/12/13` |
| 4 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | `4/4/4` | `3/3/3` | `8/8/8` | `6/11/13` | `14/19/21` |
| 8 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | `8/8/8` | `7/7/7` | `16/16/16` | `7/15/16` | `23/31/32` |
| 16 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | `16/16/16` | `15/15/15` | `32/32/32` | `5/22/25` | `37/54/57` |
| 32 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | `32/32/32` | `31/31/31` | `64/64/64` | `31/38/41` | `95/102/105` |
| 64 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | `64/64/64` | `63/63/63` | `128/128/128` | `5/70/74` | `133/198/202` |
| 96 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | `96/96/96` | `95/95/95` | `192/192/192` | `7/103/107` | `199/295/299` |
| 128 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | `128/128/128` | `127/127/127` | `256/256/256` | `5/136/140` | `261/392/396` |

The census proves that the requested Quill width and Tantivy's
two-index-workers-per-requested-thread topology were actually instantiated.
It does not prove how many threads were simultaneously productive.

## Structural conclusion and retry boundary

Tantivy scales `1.000x, 1.619x, 1.848x` relative to width 1 at widths 1, 2,
and 4. Relative to its width-4 apex, widths 8, 16, 32, 64, 96, and 128 are
`0.950x, 0.884x, 0.823x, 0.672x, 0.482x, 0.600x`. Its observed scaling
stops at requested width 4. The usable structural fact is therefore the
combination of declining throughput after width 4 and a rapidly expanding
observed worker census. This sweep cannot distinguish a contention ceiling
from coordination, memory, or other per-worker costs because its CPU-derived
activity evidence is void.

The effect block remains sequential and disjoint from both null blocks, and
effect order/drift is not separately gated. That caveat, incomplete
normative selection, and `laws_attested=false` make the tranche no-claim.
QG-1 remains inactive and its unmeasured placeholder remains authoritative.

Retry only with the repaired collector that never rewrites process CPU and
never clips measured-call totals, the complete normative 74-cell bundle,
and immediate same-ELF reproduction. Interleave null and effect blocks or
bind effect order/drift explicitly; ratchet-bind continuous timing and
work-receipt modes; preserve exact host, executable, wall, and worker-census
receipts.
