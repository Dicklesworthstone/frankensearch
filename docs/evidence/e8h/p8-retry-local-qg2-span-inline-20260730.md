# P8 RETRY — Quill TermInterner span-inline bucket entries: KEEP at n=64 (`bd-e8h-w2-interner-arena-x9s38`)

Date: 2026-07-30. Executor: P8-retry subagent (SandyGrove orchestration).
Comparison class: **SELF-SPEEDUP** (both arms are frankensearch/Quill;
maintenance evidence only — no incumbent arm, no QG/campaign claim).

## Retry provenance chain (the graze history, honestly)

1. **Original run (2026-07-29, P8 of the E8-H loop):** arm A = P6 overlay ELF
   `454205fd…`, arm B = P6+P8 overlay ELF `d6410985…`. Pooled n=32 (two
   independent 16-pair batches, banked at `scratchpad/p8/ab_main{,2}.tsv`):
   paired docs/s ratio B/A **median 1.0293**, mean 1.0300, 32/32 pairs favor
   the lever; A/A null median 1.0007, span 0.9826-1.0249. Adjudicated
   **WASH** — grazed the 1.03 material line from below. Effect judged real
   (32/32 favor, mechanism visible) but sub-material at n=32.
2. **Frozen retry predicate (recorded at the WASH adjudication; authorized by
   YellowSparrow handoff #6886):** ONE permitted rerun once the P6 KEEP was
   landed in a published base — exact published base
   `d5ad5d59e3ca7a4000f97867d06f7ecc5fc59baa`, n=64 independent pairs, same
   provenance/null/parity gates, **KEEP iff the new independent median >=
   1.03**; otherwise TERMINAL REJECT (no further retries). (Agent-mail DB was
   returning errors during this retry; the predicate text is quoted from the
   orchestrator's launch authorization, which SandyGrove holds.)
3. **This run (2026-07-30):** the pre-registered n=64 retry, fresh worktree,
   fresh builds, all-new timing. **Pooled n=64 median 1.0325 >= 1.03 → KEEP.**

## Provenance

- Base: shared repo commit `d5ad5d59e3ca7a4000f97867d06f7ecc5fc59baa`
  ("docs(e8h): reconcile evidence cards with their publication state"),
  which CONTAINS the landed P6 identity-hasher KEEP (`bd16d35d`). Arm A =
  pristine d5ad5d59, so this A/B attributes the P8 span-inline lever alone,
  on top of landed P6.
- Landing parent:
  `6c70c86f7c852fb5e650fe44ff1b4939bb019fcc`. The intervening
  `d5ad5d59..6c70c86` train changes neither Cargo resolution nor Quill or
  gauntlet source; the pre-lever `scribe.rs` blob is `47cf87cd…` at both
  bases. The landing source hunk is byte-identical to the measured
  candidate. Hostile-review edits after measurement are confined to this
  card, the ledger, and the consolidated raw artifact.
- Worktree: `scratchpad/p8-retry-worktree` (`git worktree add … d5ad5d59
  --detach`), HEAD verified `d5ad5d59e3ca…`, clean before patching.
- Patch: banked `scratchpad/p8/p8-interner-arena.patch` (243 lines,
  scribe.rs only). `git apply --check` passed against pristine d5ad5d59
  with **zero fuzz and zero context adjustment** — the one-line doc-comment
  delta between the banked P6 patch and landed P6 ("SipHash" →
  "`SipHash`", scribe.rs ~1070) sits outside every hunk's context window.
  **No semantic rework of any kind.** Post-apply, the worktree diff's full
  +/- line set was diffed against the banked patch: identical.
  Applied copy banked at `scratchpad/p8-retry/p8-interner-arena.applied.patch`.
  Patched scribe.rs sha256 `cf1b42f533ecec437e356a7b2fa6f2d00268968315059054ae2673655e0cc6b2`.
- Build env: RCH disabled via wrapper script
  (`scratchpad/p8-retry/cargo-p8r.sh`), `CARGO_TARGET_DIR=/data/tmp/cargo-target-p8-retry`,
  `RUSTFLAGS="-C force-frame-pointers=yes"`, profile `release-perf`,
  gauntlet bench `perf_matrix` with `--features perf-harness`, builds under
  `taskset -c 16-31 … -j14`. Toolchain rustc 1.99.0-nightly (9f36de775
  2026-07-19) — same as the P6 landing gates.

## Executing ELFs (all four SHA-256, `scratchpad/p8-retry/elfs/`)

- Arm A `perf_matrix_p8r_base`  = `fdc7c5c7b7b1a0b2c7d107f10f26813bb39d02557534b3bbe8aff26a58120064`
- Arm B `perf_matrix_p8r_lever` = `3246d1b844467e50b83750abf4445e507ec3abcc27370350a2777c1724e68617`
- Probe base  `p8r_ingest_probe_base`  = `c8fd2dafdcfc6fe3be3eb14ea37eda904ff2de6a441a338abc92fc901ac48258`
- Probe lever `p8r_ingest_probe_lever` = `875f3324a27ab0c4190b96272f985eac9fbbff936f465da501fdaf88fb806b8f`

## Ladder

1. **Tests (B tree, strict pipefail transcript
   `scratchpad/p8-retry/test-lever-full.log`):** `cargo test -p
   frankensearch-quill` exit 0 — lib **484 passed / 0 failed / 1 ignored**
   (base at the train's landing gates was 483/0/1; the +1 is exactly the
   patch's own `bucket_entries_mirror_spans_across_collisions_and_reset`
   span-mirror pin, which passed), plus 3 integration + 2 doctests + 3
   process-isolated singles, 492 passed total, 0 failures anywhere.
2. **Clippy (landing-readiness extra):** `cargo clippy --no-deps -p
   frankensearch-quill` on the patched tree — finished clean, zero
   diagnostics.
3. **Byte identity:** the pass-8 deterministic 3-cycle ingest probe
   (9k docs, ~20k-term vocab, flush + interner reset per cycle;
   source banked at `scratchpad/p8-retry/p8_ingest_probe.rs`, built as a
   quill example from THIS worktree in both arms). Both arms, two runs
   each: all emitted FSLX segment files and the summary are
   SHA-256-identical across arms AND across repeated runs —
   `d3b2382d…` / `1a42f42e…` / `dc861ef5…` / `dd4257a8…` — and these are
   byte-for-byte the SAME artifact hashes the original 3684-era P8 overlay
   probe produced, closing the loop across bases. Flush-accounting pin
   (`const _: () = assert!(TERM_BUCKET_BYTES_ESTIMATE == 40)`) compiled,
   so flush boundaries cannot have moved.
4. **A/B (n=64 independent interleaved pairs):** QG-2 smoke memory child
   (`QUILL_PERF_CHILD_MODE=memory`, ENGINE=quill, COUNT=200000,
   HEAP=50000000, THREADS=1, POSITIONS=true, SCALE=smoke), external wall
   time, `taskset -c 8` (core ~91% idle at start), four independent
   16-pair batches via the session `ab_run.sh` (per-batch untimed warmups
   both arms, alternating A-first/B-first), plus one 16-pair A/A null.

## Result (VERDICT: KEEP)

- **A/A null (n=16, same invocation):** paired ratio median **0.9996**,
  mean 0.9982, 95% t-CI [0.9935, 1.0030], span 0.9754-1.0125.
- **Per-batch paired medians (replication):** 1.0349 / 1.0263 / 1.0328 /
  1.0314. Honest note: batch 2 alone sits below the line and its per-arm
  cv spiked to ~3% (transient host noise; other batches cv 0.6-1.6%) —
  the pre-registered statistic is the pooled independent median.
- **Pooled (n=64):** docs/s ratio lever/base **median 1.0325**, mean
  1.0306, 95% t-CI [1.0271, 1.0340], p5-p95 [1.0069, 1.0482], **63/64
  pairs favor the lever**; time-ratio convention (new/old) 0.9705,
  CI [0.9672, 0.9738]. ~38.9k → ~40.2k docs/s. The A/B CI is disjoint
  from the A/A null CI.
- **Frozen predicate check: 1.0325 >= 1.03 → KEEP** (point estimate sits
  just above the line; the effect is real and modest, consistent with the
  original graze at 1.0293).
- **CPU/wall asymmetry rule (`/usr/bin/time -v`, one sample per arm):**
  arm A user+sys 5.16s vs wall 5.17s; arm B 5.03s vs wall 5.04s —
  CPU ≈ wall both arms, single-threaded, no blocking asymmetry.
- **RSS diagnostic — NO CLAIM:** that same single sample observed arm B
  526.3 MiB versus arm A 500.3 MiB (+26 MiB), but it does not establish a
  causal or bounded-memory cost. Across all 64 paired A/B rows, median B/A
  RSS is 1.0097, median delta +4.84 MiB, mean delta +5.34 MiB,
  interpolated p5-p95 −14.00 to +22.86 MiB, and only 42/64 pairs have B>A;
  arm maxima are 555,061,248 and 561,410,048 bytes. `Bucket` remains 24
  bytes and the dominant `One` path retains the pinned 40-byte outer
  accounting estimate; only rare `Many` collision-vector capacity uses
  the larger entry. No QG-7 or high-collision/high-shard experiment ran.
  The +26 MiB sample is therefore preserved as a noisy observation, not
  attributed to inline span copies.

## Mechanism spot-check (perf, fp call-graph, one child run per arm)

Reproduces the original P8 signature: base flat self-time
`TermInterner::matches` **0.80%** + `find_in_bucket` 1.79%; lever: the
`matches` frame is **absent** from the profile (span verified inline from
the bucket entry — the dependent `spans[id]` load is gone), `find_in_bucket`
1.76%, `intern_accounted` 0.73% → 1.06% (absorbed inline work),
`hash_parts` ~unchanged (0.95% → 0.83%). Total samples 20355 → 19862 at
equal sample rate, consistent with the ~3% wall win.
Data: `scratchpad/p8-retry/mech-{base,lever}.perf.data`.

## Landing validation

- Protected landing parent:
  `6c70c86f7c852fb5e650fe44ff1b4939bb019fcc`.
- The focused
  `bucket_entries_mirror_spans_across_collisions_and_reset` test passed
  strictly remote on `ovh-a` at the rebased source state (1 passed, 484
  filtered; remote command exit 0).
- The full `cargo test -p frankensearch-quill` landing gate also passed
  strictly remote on `ovh-a`: 484 library tests passed / 0 failed / 1
  ignored, plus 3 cancellation-contract integration tests and 2 doctests.
  The strict-remote `cargo clippy -p frankensearch-quill --all-targets --
  -D warnings` gate passed on the same worker with zero diagnostics.
  `cargo fmt --all --check` and `git diff --check` are clean.
- Targeted UBS v5.3.7 exits 1 on both the exact parent and candidate because
  it treats the existing library/test panic and token-name heuristics as
  critical. Same-version totals are parent 47 critical / 1408 warning / 163
  info versus candidate 48 / 1415 / 165. The complete delta is reviewed:
  one false-positive constant-time warning on the existing public lexical
  `term` byte equality after its span parameter changes; five intentional
  compile-time/test assertions; three test-only `format!` allocations; two
  bounded test-only modulo-to-u16 casts; and one fewer production direct
  index (the load this lever removes). There is no new unreviewed production
  bug; the nonzero UBS receipt is retained rather than mislabeled as a pass.
- The consolidated artifact
  `p8-retry-local-qg2-span-inline-raw-20260730.json` retains every null and
  A/B pair as wall/RSS values, binds the five source TSV hashes plus runner
  and reconstruction-script hashes, and states the exact ratio and
  percentile reconstruction rules. It also carries the explicit RSS
  no-claim classification.

## Exact repro

```
SCRATCH=/data/tmp/claude-1000/-data-projects-frankensearch/6ba18fed-10c5-4855-b287-a7067faf4133/scratchpad
git -C /data/projects/frankensearch worktree add $SCRATCH/p8-retry-worktree d5ad5d59e3ca7a4000f97867d06f7ecc5fc59baa --detach
# arm A: build bench + probe via cargo-p8r.sh (RCH_DISABLE=1 wrapper, isolated target, frame pointers)
taskset -c 16-31 bash $SCRATCH/p8-retry/cargo-p8r.sh build --profile release-perf -p frankensearch-quill-gauntlet --bench perf_matrix --features perf-harness -j14
#   (probe: copy p8_ingest_probe.rs into crates/frankensearch-quill/examples/, build --example p8_ingest_probe)
# arm B: git apply scratchpad/p8/p8-interner-arena.patch ; rebuild both
# timing: ab_run.sh <A> <B> <tsv> 16 8   — null first (A vs A), then 4 batches
# stats:  pooled_stats.py ab_batch{1,2,3,4}.tsv
```

## Files

- Committed consolidated raw rows, source hashes, and reconstruction rules:
  `docs/evidence/e8h/p8-retry-local-qg2-span-inline-raw-20260730.json`.
- Original TSVs:
  `scratchpad/p8-retry/{aa_null,ab_batch1,ab_batch2,ab_batch3,ab_batch4}.tsv`
  (their SHA-256 identities are committed in the consolidated artifact).
- Pooled stats + script:
  `scratchpad/p8-retry/{pooled-result.txt,pooled_stats.py}` (hashes likewise
  committed).
- Test transcript: `scratchpad/p8-retry/test-lever-full.log`
- Probe artifacts: `scratchpad/p8-retry/out-probe-{base,base-run2,lever,lever-run2}/`
- Landing state: this commit on parent `6c70c86`; the production source hunk
  is identical to the measured `scratchpad/p8-retry-worktree` candidate.
