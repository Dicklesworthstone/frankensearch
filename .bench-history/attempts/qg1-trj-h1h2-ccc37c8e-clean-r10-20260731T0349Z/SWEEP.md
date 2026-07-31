# QG-1 `trj` continuous/receipted thread sweep

## Outcome

**STRUCTURAL / NO-CLAIM.** The complete requested-width sweep finished
successfully, but every row has `laws_attested=false`, so this tranche does
not activate QG-1 and does not support a QG PASS, MISS, KEEP, or REJECT.
The raw throughput curve does answer the routing question: pinned Tantivy
0.26.1 reaches its observed apex at requested width 4 and declines at every
wider setting. The observed worker census expands from 8 median Tantivy
workers at width 1 to 392 at width 128, including exactly 256 index workers.

The usable structural target is therefore the incumbent's widening
worker-overhead regime after width 4. A post-run receipt audit invalidated
the CPU-derived activity fields, so this sweep does not identify a
productive-concurrency or contention ceiling.

## Provenance

- Sweep ID:
  `qg1-trj-h1h2-ccc37c8e-clean-r10-20260731T0349Z`
- Measured source:
  `ccc37c8e611cd313201108ffe9260376a977b4bd`
- Actual incumbent: pinned Tantivy `0.26.1`, linked into and executed beside
  Quill by the same benchmark process.
- Executable SHA-256:
  `53ab4c0975f0ad2148e37f35641dfd56e78acd8048d01cdb8b1194aa8ab9b637`
  (full digest reported from inside the process and matched by all nine
  terminal row receipts).
- Exact build:

  ```text
  RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec \
    --base ccc37c8e611cd313201108ffe9260376a977b4bd \
    --clean-overlay --no-overlay -- \
    cargo build -j 2 --profile release-perf \
      -p frankensearch-quill-gauntlet \
      --features perf-harness,tantivy-oracle \
      --bench perf_matrix \
      --message-format=json-render-diagnostics
  ```

  RCH selected `ovh-b`, synchronized zero overlay files, and reported project
  hash `885db21f6567d913`.
- Runner SHA-256:
  `ccd37127d9eaab1a58f25591e97ff892c4c20e4ea1430b9e98d67e6098f3c8bb`.
- Host: `threadripperje`, boot
  `b107a2c6-9fac-40df-a637-c3a772b0ad57`, Linux x86-64, AMD Ryzen
  Threadripper PRO 5995WX 64-Cores, 128 logical CPUs.
- Fixture: deterministic one-million-document `bulk/xlarge`, positions on,
  in-memory durability, one excluded warmup, ten paired blocks, continuous
  timing, and work receipts enabled.
- Terminal sweep status: exit 0 after `13,223,713,655,361 ns`. Every width
  produced 66/66 receipts, zero H1/H2 wall mismatches, zero terminal
  failures, matching evidence identity, and equal last H1/H2 walls.

The raw bundle contains 112 files before this adjudication. The 111-entry
checksum manifest verifies every other raw file:

- `aggregate.json`:
  `a1d4323d69d5c587ecefa99f85032615659f45d22b2f56ee3d77606995058e68`
- `runner.sh`:
  `ccd37127d9eaab1a58f25591e97ff892c4c20e4ea1430b9e98d67e6098f3c8bb`
- `sweep-status.txt`:
  `c85b23f788da1575c5cd21697ff066722ea17b4e51b325939aafa111ea90627c`
- `artifact-sha256.txt`:
  `5a29364d552f848b09322854de6d537d756f6f3ea5eedde9a6390335257c2c2d`

## Post-run CPU receipt invalidation

The `ccc37c8e` collector replaced the independently observed process CPU
delta with `max(process_cpu, role_cpu_sum)` when non-atomic `/proc` reads
disagreed. The original process value is not retained. Process CPU,
active-concurrency integrals/means, role/unattributed CPU decomposition, and
positive-tick worker counts in all 594 receipts are therefore
**CONTAMINATED / VOID FOR INFERENCE**. The prior `4.639 -> 9.132` active
equivalents and "low productive-concurrency ceiling" conclusion are
retracted.

No raw file was rewritten. Thread-membership census rows remain valid
liveness observations because thread names and roles were sampled
independently of the process-CPU floor. H1 wall and throughput remain
source-bound diagnostics.

## Corrected null gate and throughput

The corrected per-row null gate requires all three conditions:

1. the paired Quill/Tantivy bootstrap median-CI excludes 1;
2. `abs(effect_median - 1)` is greater than twice the larger A/A null
   half-width, where `half_width = max(abs(ci_low - 1), abs(ci_high - 1))`;
3. both A/A null medians are within 2% of 1.

CV is provenance only and decides no row. `2x floor` below is the threshold
from condition 2.

| requested | host / boot ID | Quill docs/s | Tantivy docs/s | Quill/Tantivy median [95% CI] | T/T null median | Q/Q null median | 2x floor | corrected null |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | 35,551.521 | 71,930.065 | `0.499882 [0.487438, 0.511549]` | 1.009576 | 1.001974 | 0.070120 | PASS |
| 2 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | 36,505.685 | 116,424.790 | `0.312842 [0.306422, 0.321050]` | 1.038377 | 1.012091 | 0.121926 | FAIL: T/T center |
| 4 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | 37,270.193 | 132,936.059 | `0.277612 [0.273338, 0.284961]` | 0.973066 | 0.995543 | 0.119941 | FAIL: T/T center |
| 8 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | 35,409.346 | 126,257.903 | `0.281265 [0.269878, 0.286662]` | 0.998773 | 0.998907 | 0.193757 | PASS |
| 16 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | 33,473.596 | 117,564.256 | `0.285757 [0.270176, 0.300316]` | 1.072212 | 0.996756 | 0.315396 | FAIL: T/T center |
| 32 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | 25,712.146 | 109,380.264 | `0.230809 [0.210160, 0.254111]` | 0.971548 | 0.996132 | 0.115736 | FAIL: T/T center |
| 64 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | 22,417.172 | 89,304.888 | `0.248270 [0.233913, 0.269171]` | 1.019307 | 1.019208 | 0.117623 | PASS |
| 96 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | 24,465.701 | 64,092.430 | `0.382646 [0.344567, 0.413863]` | 0.999471 | 1.004594 | 0.122632 | PASS |
| 128 | `threadripperje` / `b107a2c6-9fac-40df-a637-c3a772b0ad57` | 22,408.810 | 79,804.620 | `0.275095 [0.267489, 0.311525]` | 0.975137 | 1.001677 | 0.273005 | FAIL: T/T center |

The corrected gate passes widths 1, 8, 64, and 96. It fails widths 2, 4,
16, 32, and 128 solely because the Tantivy A/A median is more than 2% from
1. These labels are diagnostic: incomplete selection and
`laws_attested=false` prohibit a normative QG verdict.

## Actual observed worker census

These are thread-membership observations, not CPU activity. `Quill all`
counts the benchmark caller plus dedicated Rayon workers. `Tantivy support`
counts segment-updater, merge, and docstore-compressor threads. Each
min/median/max covers 33 receipts for that engine.

| requested | host / boot ID | Quill all min/med/max | Quill Rayon min/med/max | Tantivy index min/med/max | Tantivy support min/med/max | Tantivy all min/med/max |
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

The census proves the requested Quill topology and Tantivy's
two-index-workers-per-requested-thread topology were instantiated. It does
not prove simultaneous productive work.

## Structural interpretation and limitation

Tantivy scales from 71,930 docs/s at width 1 to 116,425 at width 2 and peaks
at 132,936 at width 4 (`1.848x` width 1). Relative to that raw peak it falls
to `0.950x`, `0.884x`, `0.823x`, `0.672x`, `0.482x`, and `0.600x` at
widths 8, 16, 32, 64, 96, and 128. Its observed throughput scaling therefore
stops at requested width 4 on this host and fixture.

The expanding census plus declining throughput is structural routing
evidence, but the void CPU fields cannot distinguish contention,
coordination, memory, or other per-worker costs. The null blocks also remain
sequential and disjoint from the effect block, and the harness does not bind
effect-block order or drift to those nulls. Together with incomplete
normative selection and `laws_attested=false`, that prevents promotion of
the raw curve to a certified QG decision. QG-1 remains inactive; the
checked-in unmeasured placeholder remains authoritative.
