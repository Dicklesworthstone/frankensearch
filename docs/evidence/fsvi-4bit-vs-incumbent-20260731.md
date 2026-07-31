# Claim conversion #1 — "the fastest lossless vector-search primitive" vs a live third-party incumbent (2026-07-31, BlackThrush)

**Claim under test:** `CHANGELOG.md:66` —
*"Lossless quantized search: FSVI 4-bit two-pass, the fastest lossless
vector-search primitive"*.

Ranked **#1** of the 137 unconverted claims by public load-bearingness, per
`claim-coverage-audit-20260730.md` (CORRECTION section). It is the only
**unbounded superlative** on any public surface of this repository, and
`git grep fastest` confirms `CHANGELOG.md:66` is its sole public occurrence.

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
- **Interleaved rounds with order reversal.** Arm order is reversed on odd
  rounds (AB/BA) so linear drift cancels.
- **A/A null from two replicates of the incumbent arm** (`inc_a`, `inc_b`) run
  at opposite ends of each round. Null ratio distribution = `inc_b/inc_a` per
  round.
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

## Results — run 1 (15 rounds): **losslessness CONFIRMED, speed claim INVERTED but UNDECIDABLE**

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

## Results — run 2 (61 rounds)

<!-- RESULTS-2 -->

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

