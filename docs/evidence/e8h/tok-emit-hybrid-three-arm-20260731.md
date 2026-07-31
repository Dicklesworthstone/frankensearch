# Tokenizer ASCII-emit lever — three-arm A/B (BASE / PURE / HYBRID), 2026-07-31

**Status: DRAFT.** Author: SandyGrove (tok-hybrid lane). Settles the frozen
pure fused-emit candidate's row and records the length-dispatched hybrid's
first full measurement. **The pre-registered all-arm null gate FAILED on this
run (see Nulls), so no KEEP row is claimed from it**; effects larger than
every observed null width are recorded as such, and the retry bar is stated.

## Arms

Single worktree, three checkouts, `cargo build --profile release-perf
-p frankensearch-quill --features bench-internals --bin tokenizer_simd_ab`,
shared repo target dir, `RCH_DISABLE=1` wrapper.

| arm | tree state | ELF sha256 |
|---|---|---|
| BASE | `148ca11a` (origin/main tip at branch time) | `7ce38d9a1bac3fa8029f09d747d6d156a956a81ce0c7557b2d78a09dd121c280` |
| PURE | `60235b13` (cherry-pick of frozen `84335939`) | `91bcf1403aa1d5573c164cff83817e6a5072e893d89e49f322d4eca10af1dece` |
| HYBRID | `67cd775b` (length dispatch, threshold 16) | `2806975660b553eed31fb355ec1c50f6980706598b792b44597eefaaaf504570` |

BASE and PURE ELF hashes are **bit-identical to the banked 2026-07-31 run's**
(`base=7ce38d9a lever=91bcf140` in `84335939`'s message): the rebase onto
`148ca11a` reproduces the exact binaries, so this run and the banked run
measure the same artifacts.

## Method

`run_ab3.sh`: n=16 rounds, arm order rotated per round
(base/pure/hybrid -> pure/hybrid/base -> ...), `taskset -c 37`, external wall
per invocation, plus each invocation's in-process `[null]` (shipping vs
shipping) and `[lever]` (BoundaryMask candidate vs shipping) paired-median
lines on both fixture corpora (short word-mix ~2-15B tokens; long 24-48B+
tokens). Shipping-arm speed compares through the unchanged BoundaryMask
yardstick: `speed(X vs Y) = r_X / r_Y` per round, `r = candidate/shipping
median`; medians across rounds with binomial-order-statistic 95% CIs.

Load discipline: measurement started at 1-min loadavg **9.94** (bar: under
~10) after a monitored wait; loadavg was **42.38** by run end — an
agent-fleet burst landed mid-run and is visible in the null tails.

## Nulls (same-invocation A/A, per arm x corpus; bar: p5..p95 in [0.97,1.03])

| arm | corpus | median-across-runs p5..p95 | worst run | runs in band | gate |
|---|---|---|---|---|---|
| base | short | [0.9873, 1.0096] | [0.8637, 1.1737] | 12/16 | ADMISSIBLE |
| base | long | [0.9891, 1.0108] | [0.9247, 1.1457] | 11/16 | ADMISSIBLE |
| pure | short | [0.9556, 1.0378] | [0.9029, 1.1320] | 2/16 | **OUT-OF-BAND** |
| pure | long | [0.9474, 1.0144] | [0.9264, 1.0595] | 0/16 | **OUT-OF-BAND** |
| hybrid | short | [0.9687, 1.0219] | [0.8076, 1.2497] | 3/16 | **OUT-OF-BAND** |
| hybrid | long | [0.9731, 1.0370] | [0.8913, 1.0825] | 6/16 | **OUT-OF-BAND** |

**Gate verdict: VOID for KEEP purposes** (same failure mode as the banked
run: fused-emit-arm nulls wide while base nulls stay clean, aggravated here
by the mid-run load burst). Effects below are reported with that caveat;
only effects far outside every observed null width are treated as settled.

## Results (in-process, paired per round, n=16)

Per-arm lever medians `r` (higher = faster shipping tokenizer):

| arm | short r (CI95) | long r (CI95) |
|---|---|---|
| base | 0.8883 [0.8865, 0.8923] | 0.9990 [0.9971, 1.0090] |
| pure | 0.9177 [0.9144, 0.9203] | 0.7069 [0.7008, 0.7089] |
| hybrid | 0.9133 [0.9103, 0.9169] | 0.9814 [0.9790, 0.9854] |

Arm-pair shipping speedups (`>1` = first arm faster):

| pair | corpus | median | CI95 | reading |
|---|---|---|---|---|
| PURE vs BASE | short | 1.0318 | [1.0259, 1.0361] | at the 1.03 bar, CI not clear of it |
| PURE vs BASE | long | **0.7073** | [0.6983, 0.7106] | **1.41x regression — settled** (far outside every null band; banked run 0.697, midday 6-run check 0.700) |
| HYBRID vs BASE | short | 1.0288 | [1.0206, 1.0318] | WASH by the bar (0.1% under 1.03); retains ~all of PURE's short win |
| HYBRID vs BASE | long | 0.9832 | [0.9711, 0.9877] | WASH; inside the hybrid-long null band — residual cost not distinguishable from noise on this run |
| HYBRID vs PURE | long | 1.3896 | [1.3839, 1.4030] | hybrid recovers the pure regression in full |
| HYBRID vs PURE | short | 0.9954 | [0.9911, 1.0027] | no measurable dispatch cost |

External wall (whole binary, both corpora + startup; mid-run load caveat):
base median 2.941s, pure 3.140s, hybrid 2.914s; paired wall pure/base
1.0700 [1.0659, 1.0829]; hybrid/base 0.9952 [0.9874, 1.0061].

## Dispositions

* **PURE (frozen `84335939` candidate): REJECT.** The long-corpus 1.41x
  regression is ~10x every observed null half-width, reproduced three
  independent times (banked run under load, midday 6-run check, this run),
  and fails the KEEP bar ("median >= 1.03 on BOTH corpora") outright. The
  ~1.03x short-corpus win is real but cannot carry the row.
* **HYBRID vs BASE: WASH (formally VOID for KEEP).** Median 1.0288 short /
  0.9832 long. The designed outcome — keep the short win, neutralize the
  long loss — is achieved directionally (short retained at 99.5% of PURE's
  win; long recovered 1.39x vs PURE), but the short median lands 0.1% under
  the 1.03 KEEP bar and the long residual sits inside the failed null band.
  No KEEP row may be minted from this run.
* **Retry bar (pre-registered, unchanged):** same three-arm protocol at a
  window where ALL six arm x corpus same-invocation nulls hold p5..p95
  within [0.97,1.03] for the majority of runs and loadavg stays under ~10
  for the full duration. KEEP bar for HYBRID: median >= 1.03 on BOTH corpora
  with CI clear of the null.

## Threshold-selection evidence (recorded in `67cd775b`'s message)

Fixed-width in-situ sweep (widths 6-32, single-width corpora, sweep ELFs
`15557d79...` / `7aadcc3a...`): bulk wins at EVERY constant width — constant
span length makes memcpy size dispatch and sweep tails perfectly
predictable, erasing the penalty fused avoids on variable-length text. The
fused win exists only under real length variance; threshold 16 = two SWAR
words keeps the observed winning region (<= 15B mix) fused and routes the
settled >= 24B regression region to bulk.

## Raw artifacts

`scratchpad/ab3_results.txt` (full per-run lines + start/end uptime),
`scratchpad/emit_width_sweep_results.txt`, `scratchpad/realcorpus_check.txt`,
`scratchpad/ab3_stats.py` (aggregation), ELFs under `scratchpad/elfs-hybrid/`.
