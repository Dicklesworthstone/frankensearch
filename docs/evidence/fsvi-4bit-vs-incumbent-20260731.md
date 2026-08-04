# Claim conversion #1 — "the fastest lossless vector-search primitive" is REFUTED against a live third-party incumbent (2026-07-31, BlackThrush)

**Claim under test:** `origin/main:CHANGELOG.md:82` (this working tree: `:66`) —
*"Lossless quantized search: FSVI 4-bit two-pass, the fastest lossless
vector-search primitive"*.

Ranked **#1** of the 137 unconverted claims by public load-bearingness, per
`claim-coverage-audit-20260730.md` (CORRECTION section). It is the only
**unbounded superlative** on any public surface of this repository, and
`git grep fastest` confirms it is the sole public occurrence. It also travels as
the subject line of commit `f04074a4`.

## Why this claim was ranked first

Its entire evidentiary base is two ledger rows, and both compare the 4-bit
two-pass against **our own code**:

| source | comparator | class |
|---|---|---|
| `docs/PERF_LEDGER.md:825` | our int8 two-pass (1.07×), our flat f16 scan (2.56×) | SELF-SPEEDUP |
| `docs/PERF_LEDGER.md:827` | our int8 two-pass (1.40×), our flat f16 scan (3.09×) | SELF-SPEEDUP |

No third-party arm appears anywhere in its provenance. A user reading "the
fastest lossless vector-search primitive" could reasonably choose frankensearch
over faiss, usearch, or hnswlib on its strength; nothing in the repository
supported that.

## The incumbent, and why it is a fair one

`faiss::IndexFlatIP` — the reference implementation for exact ("lossless")
vector search — computes a dense score block `queries[nq×d] · corpusᵀ[d×N]`
with a cache-blocked SIMD GEMM, then reduces each row to top-k. The incumbent
arm reproduces exactly that structure using **`ndarray` 0.17.2 over
`matrixmultiply` 0.3.10**, verified third-party and BLAS-free:

```
$ awk '/^name = "ndarray"$/,/^$/' Cargo.lock     # -> dependencies = [ "matrixmultiply", ... ]
$ grep -cE '^name = "(blas-src|cblas-sys|openblas-src|netlib-src)"' Cargo.lock   # -> 0
```

With no BLAS feature in the tree, `Array2<f32>::dot` lowers to
`matrixmultiply::sgemm`, whose AVX2/FMA kernels are selected by runtime
detection. The GEMM is the dominant cost and is entirely third-party code; only
the small-k top-k reduction is written for this bench, which is the same
reduction faiss performs after its GEMM block.

**The incumbent is reported at its best.** Both a batched (`nq=32`, GEMM) and an
unbatched (`nq=1`, GEMV) shape are measured and the faster is used as the
official incumbent number.

## Thread asymmetry — the trap this bench is built to avoid

Our 4-bit pass-1 is Rayon-parallel; `matrixmultiply` is single-threaded (its
`threading` feature is not enabled — confirmed in `Cargo.lock`). Comparing them
as-shipped would measure thread count, not the primitive. So the candidate is
measured **twice**:

- `cand_4bit_mult5_threads1` — inside a pinned 1-thread Rayon pool via
  `pool.install()`. **This is the headline like-for-like ratio.**
- `cand_4bit_mult5_default` — the default global pool, as shipped.

CPU/wall and observed thread count are published per arm.

## Protocol

- **One binary, one invocation, one fixture.** 100k × 384 clustered vectors
  (64 centroids, noise 0.30) written once to an FSVI index; the identical f32
  vectors, in the identical order, form the incumbent's `Array2`. Positional id
  `i` therefore corresponds to `doc-{i:06}` in both arms.
- **Bracketed control (final design, run 3).** Every subject is measured
  *between* two incumbent runs and scored `X / mean(I_before, I_after)`, and the
  A/A null is an incumbent run scored by the **same rule** — so null and
  candidate ratios share one structure and one drift cancellation. Subject order
  rotates each round. This is the control `PERF_LEDGER.md:822-824` already
  validated on this fleet (B/A 0.9999).
  *Runs 1 and 2 used a whole-round A/A (replicates at opposite ends of the
  round) and its null was dirty both times; see "How the decidable answer was
  reached".*
- **Corrected null gate with the median clause.** The null is admissible only
  if its *median* lies within 1.000 ± 0.030; a candidate ratio is decidable only
  if it falls **outside** the null's [p5, p95]. A point estimate inside the band
  is a WASH, and no `n`-fishing is permitted.
- **Self-reported identity from inside the process:** ELF SHA-256 of
  `/proc/self/exe`, hostname from `/proc/sys/kernel/hostname`, CPU model from
  `/proc/cpuinfo`, and **observed** thread count from `/proc/self/task` (not the
  configured value). CPU time from `/proc/self/stat` accumulated per arm.
- **Losslessness is verified against two different definitions**, because they
  are not the same thing and the claim only ever tested the weaker one.

Build: strict-remote `rch exec --base 3debdf25 --clean-overlay` with only the
bench and its `Cargo.toml` overlaid. Bench source:
`frankensearch/benches/fsvi_4bit_vs_incumbent.rs`.

### Source-to-ELF provenance (stated because one entry is imperfect)

| run | ELF SHA-256 | source state | in git? |
|---|---|---|---|
| 1 (15 rounds) | `dff84b78…30a271` | as committed in `f23adfd0` **except** `unwrap_or(15)` instead of `unwrap_or(61)` | **no** — that literal was edited before the first commit |
| 2 (61 rounds) | `905fa959…bb7924` | exactly `f23adfd0` | yes |
| 3 (bracketed) | `3a99d494…cdcc6e` | the bracketed rewrite, as committed | yes |

**Run 1's exact source is not in git**, and that is a real defect in this card,
not a rounding of one. It differs from the committed file by a single integer
literal — the round count, which is also printed in run 1's own output
(`rounds=15`) — so it is reconstructible, but it should have been committed
before being measured. Recorded rather than quietly dropped.

> **Note on a known wart.** `cpu_time_ns()` routes through a pointless
> `american_paren()` helper where `rsplit_once(')')` would do. It is deliberately
> left in place: the committed source must keep producing the two ELF SHA-256s
> recorded below, and a cosmetic edit would break that correspondence. Clean it
> only in a commit that explicitly postdates and supersedes these measurements.

## Results — run 1 (15 rounds)

Strict-remote `rch exec` on worker `vmi1152480`, exit 0. Self-reported from
inside the process:

```
host                  = frankenlibc-test
cpu_model             = AMD EPYC Processor (with IBPB)
elf_sha256            = dff84b780e3c0ef296bb52102714c9d9fb1e7c12ccc1c80e5acb6234ff30a271
nproc                 = 10
rayon_global_threads  = 10
threads_at_start      = 11        threads_at_end = 12
N=100000 DIM=384 K=10 QUERIES=32 MULT=5 rounds=15
corpus_f32_bytes      = 153600000 (146.5 MiB)   [incumbent working set]
packed_4bit_bytes     =  19200000  (18.3 MiB)   [candidate pass-1 working set]
```

### Losslessness — the claim's weaker half is TRUE, and now verified against f32 truth

```
candidate == ours_exact_f16     : 32/32   <- the claim's own definition
candidate == incumbent_f32      : 32/32   <- the world's definition
ours_exact_f16 == incumbent_f32 : 32/32   <- our f16 storage vs f32 truth
```

The ledger only ever tested the first line. All three hold exactly at `mult=5`
on this fixture: the 4-bit two-pass returns the same top-10 as a full-precision
f32 exhaustive scan, and our f16 storage costs nothing in top-10 identity.
**"Lossless" is supported.**

### Per-arm timings

| arm | median µs/query | cpu/wall | n |
|---|---:|---:|---:|
| `incumbent_batch32_A` | 1833.79 | 1.01 | 15 |
| `incumbent_batch32_B` | 1884.00 | 1.00 | 15 |
| `incumbent_gemv_nq1` | 9257.59 | 1.00 | 15 |
| `cand_4bit_mult5_threads1` | 2990.58 | 0.98 | 15 |
| `cand_4bit_mult5_default` | 2182.07 | 2.82 | 15 |
| `ours_exact_flat_default` | 2282.52 | 3.37 | 15 |

`cpu/wall` confirms the thread discipline worked: both incumbent arms and the
pinned candidate sit at ~1.0 (genuinely single-threaded), the default candidate
at 2.82 and our flat scan at 3.37. **The `max_thr` column the bench prints is
useless and is omitted here** — `/proc/self/task` is process-wide, so it reads
12 for every arm. Per-arm concurrency is evidenced by `cpu/wall`, not by that
column.

The batched incumbent beats its own unbatched shape 5.0× (1833.79 vs 9257.59),
so `incumbent_batch32` is used as the official incumbent number — the incumbent
is reported at its best, as promised.

### The gate result

```
null_median = 1.0321   null_p5 = 0.7685   null_p95 = 1.5559
null_gate(median within 1.000 +/- 0.030) = DIRTY -- ratios are NOT decidable

ratio_like_for_like(1 thread both)          = 1.6226  (0.62x)  UNDECIDABLE
ratio_as_shipped(cand ~2.8 eff thr vs inc 1) = 1.1839  (0.84x)  UNDECIDABLE
ratio_ours_exact_flat                        = 1.2384  (0.81x)  UNDECIDABLE
```

**The null failed by 0.0021 and that is binding.** 1.0321 is not 1.030. The
pre-declared gate said a dirty null makes the ratios undecidable, so they are
undecidable, and no amount of favourable direction changes that. This is the
`1.0293 vs 1.03 = WASH` rule applied against my own result.

### What the point estimates say, pending a clean null

Every ratio points the same way, and it is not the way the claim points:

- At **equal threads** the 4-bit two-pass is **1.62× slower** than a
  single-threaded third-party BLAS-class exact GEMM.
- **Even given ~2.8 effective threads against the incumbent's 1**, it is still
  **1.18× slower**.
- Our exact flat scan is also slower than the incumbent (1.24×).

Note `ratio_like_for_like` (1.6226) actually lies *outside* the null band
`[0.7685, 1.5559]`; only the median clause blocks it. That is why a re-run at
higher round count is worth doing rather than abandoning.

### A second, unrelated finding: the ledger's own self-vs-self number does not reproduce here

```
self_vs_self_context: cand_default / ours_exact_flat = 0.9560 (1.05x)
```

`PERF_LEDGER.md:825/827` claim the 4-bit two-pass is **2.56×–3.22× faster than
our own flat scan**. On this host it is **1.05×**. This host has 10 cores and
neither parallel arm scaled past cpu/wall 3.4, so the gap is plausibly
host-dependent rather than wrong — but it means the published self-vs-self
ratio is not portable either, and it was never labelled with the host it was
measured on. Filed as a separate concern; not resolved by this card.

## Results — run 2 (61 rounds): more rounds made the null **worse**

Same worker (`vmi1152480`), same fixture, different binary
(`elf_sha256 = 905fa9597cbc940708dfeeeb09f9957fa82b6d1997168005a9f9f11e31bb7924`),
`rounds=61`, exit 0. Fleet was at 9 active builds vs 13–15 during run 1.

### Losslessness — reconfirmed on a second, independent ELF

```
candidate == ours_exact_f16     : 32/32
candidate == incumbent_f32      : 32/32
ours_exact_f16 == incumbent_f32 : 32/32
```

### Per-arm

| arm | median µs/query | cpu/wall | n |
|---|---:|---:|---:|
| `incumbent_batch32_A` | 2590.04 | 0.98 | 61 |
| `incumbent_batch32_B` | 2639.32 | 1.00 | 61 |
| `incumbent_gemv_nq1` | 12147.50 | 0.99 | 61 |
| `cand_4bit_mult5_threads1` | 4037.68 | 0.97 | 61 |
| `cand_4bit_mult5_default` | 4255.07 | 1.34 | 61 |
| `ours_exact_flat_default` | 5701.13 | 1.42 | 61 |

### The gate result — still DIRTY, and further from clean

```
null_median = 1.0598   null_p5 = 0.5788   null_p95 = 1.8289
null_gate(median within 1.000 +/- 0.030) = DIRTY

ratio_like_for_like(1 thread both)  = 1.5359  (0.65x)  UNDECIDABLE
ratio_as_shipped                    = 1.6186  (0.62x)  UNDECIDABLE
ratio_ours_exact_flat               = 2.1687  (0.46x)  UNDECIDABLE
self_vs_self_context: cand/flat     = 0.7464  (1.34x)
```

**Quadrupling the rounds moved the null the wrong way**: median 1.0321 → 1.0598,
band `[0.7685, 1.5559]` → `[0.5788, 1.8289]`. Every absolute time inflated too
(incumbent 1833.79 → 2590.04 µs/q) and both parallel arms lost CPU (`cand_default`
cpu/wall 2.82 → 1.34, `ours_exact_flat` 3.37 → 1.42). **The dispersion is
environmental, not sampling error** — this is a shared, contended VPS, and more
samples of a drifting process do not converge. That is the method lesson: when a
null is dirty because the host is noisy, `n` is the wrong lever.

## Results — run 3 (61 rounds, BRACKETED): **null CLEAN, claim DECIDABLY REFUTED at equal threads**

The whole-round A/A was replaced with a bracketing control (see the bench
header). Same worker `vmi1152480`, same fixture, `rounds=61`,
`elf_sha256 = 3a99d4940d987778e83cd80637995f24ec92a01bf2a710931e1091a1f7cdcc6e`,
exit 0. Losslessness reconfirmed a third time on a third independent ELF:
32/32 on all three definitions.

```
--- PER-ARM (median wall; 61 rounds x 5 slots, each = 32 queries) ---
arm                             median_us/q   cpu/wall        n
incumbent_batch32_bracket           1961.66       0.98      366
cand_4bit_mult5_threads1            3131.09       0.99       61
cand_4bit_mult5_default             2661.39       2.51       61
ours_exact_flat_default             2676.27       2.81       61
incumbent_null_AA                   1872.06       0.99       61
incumbent_gemv_nq1                  8382.91       1.00       61

--- A/A NULL (incumbent bracketed by incumbents, same rule as every ratio) ---
null_median = 0.9814   null_p5 = 0.7656   null_p95 = 1.4825
null_gate(median within 1.000 +/- 0.030) = CLEAN

--- BRACKETED RATIOS vs incumbent_batch32 (1961.66 us/query) ---
cand_4bit_mult5_threads1 = 1.5790 (0.63x) [p5 1.3040, p95 3.0279]  SLOWER (outside null)
cand_4bit_mult5_default  = 1.1575 (0.86x) [p5 0.6521, p95 2.1822]  WASH (inside null)
ours_exact_flat_default  = 1.3369 (0.75x) [p5 0.7371, p95 3.4275]  WASH (inside null)
incumbent_gemv_nq1       = 4.3293 (0.23x) [p5 3.6938, p95 6.6701]  SLOWER (outside null)

self_vs_self: cand_default / ours_exact_flat = 0.8658 (1.15x)
```

**The bracketing fixed the null**: median 1.0598 (whole-round, 61 rounds) →
**0.9814** (bracketed, same rounds, same worker). The dispersion really was
drift across the replicate gap, and closing that gap closed the null.

### The decision

> **At equal threads the FSVI 4-bit two-pass is DECIDABLY SLOWER than a
> third-party BLAS-class exact scan: ratio 1.5790, i.e. the incumbent is
> ~1.58× faster.** The candidate median 1.5790 lies above the null's p95 of
> 1.4825, so it clears the floor by the pre-declared rule.

**Margin honesty:** it clears p95 by 6.5%. That is a pass, not a rout. The
verdict is "decidably slower", and the *magnitude* (1.58×) is a point estimate
whose own p5–p95 is wide (`[1.3040, 3.0279]`).

**As shipped, the candidate reaches only parity.** Given ~2.5 effective threads
against the incumbent's 1 (`cpu/wall` 2.51 vs 0.98), the ratio is 1.1575 —
**inside the null, a WASH.** So the shipped configuration does not beat a
single-threaded third-party exact scan even with a 2.5× thread advantage.

**The incumbent was reported at its best, and that is now proven rather than
asserted:** the unbatched GEMV shape is decidably 4.33× slower than the batched
GEMM, so using batched as the incumbent number was the generous choice.

## Verdict

| half of the claim | outcome |
|---|---|
| **"lossless"** | **SUPPORTED.** 32/32 on all three definitions, twice, on two independent ELFs. This is a deterministic set comparison, so it needs no null and the fleet noise cannot touch it. It is also *stronger* than the ledger ever showed: the ledger only tested against our own f16 scan; it is now verified against an f32 exhaustive scan. |
| **"the fastest … primitive"** | **REFUTED.** Run 3's bracketed A/A null is CLEAN (0.9814) and the equal-threads ratio 1.5790 clears its p95 (1.4825). The primitive is **decidably slower** than a third-party BLAS-class exact scan at equal threads, and only a **WASH** even when given ~2.5× the threads. |

Runs 1 and 2 were undecidable (dirty null) and pointed the same way; run 3 made
it decidable. Nine ratio observations across three runs and three independent
binaries all put the candidate slower than or equal to the incumbent, and the one
run with a clean null decides it.

### The retraction does not depend on the measurement

This is the load-bearing point. The superlative must be withdrawn **regardless of
whether a clean null ever confirms the loss**, because:

1. It never had an incumbent arm at all. Its whole basis
   (`PERF_LEDGER.md:825/827`, and commit `f04074a4`'s subject line) is
   frankensearch-vs-frankensearch.
2. "Fastest" is unbounded. Even a decisive win against `matrixmultiply` would not
   establish it, with faiss, usearch and hnswlib unmeasured.

So the measurement's undecidability delays a *counter-claim*; it does not rescue
the original. Filed as `bd-retract-fastest-lossless-superlative-3ush8` with exact
replacement text, because the claim lives at `origin/main:82` and **cannot be
corrected from this checkout** — local `main` is 270 commits behind and its HEAD
does not contain the line, while the working-tree `CHANGELOG.md` carries an
unrelated peer's uncommitted 170-line draft that a commit here would sweep in.

### How the decidable answer was reached

Not by a quieter host — by a better control. Runs 1 and 2 blamed the fleet; the
real fault was the measurement design placing the two A/A replicates ~600 ms
apart, so the null absorbed drift the candidate ratio never saw. Bracketing each
subject between two incumbent runs, and scoring the null by that same rule,
brought the null from 1.0598 to 0.9814 on the same worker with the same round
count.

**Method lesson: when a null is dirty, check the control's geometry before
blaming the host or reaching for more samples.** Raising n from 15 to 61 made
this null worse; restructuring the control fixed it in one run.

## Disclosed asymmetries

These are stated because they favour the candidate and a hostile reviewer would
find them:

1. **Working set.** The incumbent holds f32 (146.5 MiB); the candidate's pass-1
   reads a packed 4-bit slab (18.3 MiB) and rescores from f16. Part of any
   candidate win is a storage-format advantage, not a kernel advantage. faiss
   would pay the same f32 toll only in `IndexFlatIP`; its own quantized indexes
   (SQ/PQ) are the like-for-like competitor and are **not** measured here.
2. **Fixture is clustered.** 64 centroids with 0.30 noise. Losslessness of the
   4-bit two-pass at `mult=5` is a property of clustered data; the ledger's own
   rows concede recall 0.96 at `mult=2`. A uniform-random corpus is not covered.
3. **Single incumbent.** Beating one third-party exact scan does not establish
   "fastest". faiss, usearch, and hnswlib remain unmeasured — none is in-tree.

