# NEGATIVE_EVIDENCE.md — frankensearch perf swarm

> Honest ledger of perf experiments that **did NOT pay off** (≈0 gain or regression)
> and were therefore **reverted**. The point of this file is to stop future agents
> (and future me) from re-attempting dead ends. Every entry must cite the measured
> ratio vs. the pre-change baseline on the same workload.

Conventions:
- **Workload** = the exact bench id (`cargo bench` group/function) measured head-to-head.
- **Ratio** = new_time / old_time. `< 1.0` is a speedup, `> 1.0` is a regression.
- A lever is **reverted** if ratio ∈ [0.97, 1.03] (noise) or > 1.03 (regression).
- Wins (ratio < 0.97, kept) go in `docs/PERF_LEDGER.md`, not here.

Build/bench protocol (per-crate ONLY):
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/<agent-lane> \
  rch exec -- cargo bench -p <crate> --profile release
```

---

## Measurement blockers

| Date | Owner | Workload | Evidence | Status |
|------|-------|----------|----------|--------|
| 2026-06-24 | BlueGull (`frankensearch-cod-a`) | `frankensearch/search_bench vector_search_topk/top10/10000` | `cargo bench --release -p frankensearch --bench search_bench vector_search_topk/top10/10000 -- --quiet` failed before measurement on rustc `1.98.0-nightly (f20a92ec0 2026-06-07)` because Cargo rejected `--release` for `cargo bench` as an unexpected argument. | Blocker tracked in `bd-ui41`; do not count as a perf ratio. |
| 2026-06-24 | BlueGull (`frankensearch-cod-a`) | `frankensearch/search_bench vector_search_topk/top10/10000` | Fallback optimized bench command without `--release` ran through RCH on `vmi1153651` but remained in cold compile/link with no Criterion timing output after more than 10 minutes; interrupted by the owner with exit 130. | No ratio produced; use `bd-ui41` to establish a reproducible harness and command contract. |
| 2026-06-24 | BlueGull (`frankensearch-cod-a`) | Tantivy/Lucene-class original comparison | README/AGENTS confirm frankensearch is a Tantivy BM25 + semantic/vector hybrid, but no current per-crate harness emits same-corpus ratios against a Tantivy-only incumbent. | Blocker tracked in `bd-ui41`; no dominance claim is valid until this exists. |
| 2026-06-24 | frankensearch-cod-b | `frankensearch/search_bench` requested protocol | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b rch exec -- cargo bench -p frankensearch --release --bench search_bench -- --sample-size 10` selected RCH worker `ovh-a`, then Cargo rejected `--release` for `cargo bench` with `unexpected argument '--release' found`; `cargo bench --help` lists `--profile <PROFILE-NAME>` instead. | Same protocol blocker as above; successful measurement used `--profile release` and remains per-crate. |
| 2026-06-24 | frankensearch-cod-b | `frankensearch/search_bench vector_search_topk/top10/{1000,5000,10000}` | Same-worker RCH run on `ovh-a`: `rch exec -- cargo bench -p frankensearch --profile release --bench search_bench -- --sample-size 10`. Results: 1K `944.07 us`, 5K `3.4640 ms`, 10K `1.6642 ms`. | Scaling order is unstable/noisy; use as routing evidence only, not as keep/reject proof. |
| 2026-06-24 | frankensearch-cod-b | `frankensearch-index/dot_product` release-profile comparison from detached baseline worktree | Three attempts with `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b rch exec -- cargo bench -p frankensearch-index --profile release --bench dot_product -- --sample-size 10 --warm-up-time 1 --measurement-time 3` fell open to local with `no admissible workers: insufficient_slots=8,health_below_fallback=2,hard_preflight=1`; each local fallback was interrupted before measurement. | No release-profile ratio. Bench-profile RCH runs may be routing evidence, but kept wins need an admitted remote release-profile run or an in-process head-to-head harness. |
| 2026-06-24 | BlueGull (`frankensearch-cod-a`) | `frankensearch-index/dot_product f32_bytes` vs Tantivy/Lucene/Meilisearch-class original | Kept microkernel proof is against the embedded pre-change frankensearch `f32_bytes_old` baseline: pinned RCH worker `vmi1149989` measured `dot/dim256/f32_bytes/10000` at 3.4835 ms old -> 0.66126 ms new (ratio 0.190) and `dot/dim384/f32_bytes/10000` at 5.1487 ms old -> 1.8811 ms new (ratio 0.365). There is still no same-corpus Tantivy/Lucene/Meilisearch-class comparator for this vector-byte kernel or end-to-end workload. | Original-comparator ratio remains blocked by `bd-ui41`; do not claim dominance over Lucene/Tantivy/Meilisearch-class from this microkernel win alone. |
| 2026-06-26 | BlackThrush | `frankensearch-index/dot_product f32_slice` vs Tantivy/Lucene/Meilisearch-class original | Kept microkernel proof is against the embedded pre-change frankensearch `f32_slice_old` baseline, not a search-engine incumbent: remote RCH worker `vmi1227854` measured `dot/dim256/f32_slice/10000` at 2.3594 ms old -> 750.64 us new (ratio **0.318**) and `dot/dim384/f32_slice/10000` at 3.7372 ms old -> 2.2784 ms new (ratio **0.610**) with `cargo bench -p frankensearch-index --profile release --bench dot_product f32_slice -- --sample-size 10 --warm-up-time 1 --measurement-time 3`. A same-command local fallback was mixed/noisy (dim256 **1.216x slower**, dim384 **0.935x faster**) and is not used for the keep decision. | Original-comparator ratio is not applicable for this vector microkernel. Do not claim new Lucene/Tantivy/Meilisearch-class search dominance from it; the residual BOLD lexical/materialization misses below still stand. |
| 2026-06-26 | BlackThrush | `frankensearch-index` WAL tests under parallel `cargo test -p frankensearch-index --lib` | **Flaky cluster** (`compaction_merges_wal_into_main`, `soft_delete_clears_pending_wal_updates_for_same_doc_id`, …): a *different* WAL test fails each parallel run (`assert_eq!(wal_record_count/compaction stats, 2)` panics), but **all 356 pass serially** (`--test-threads=1`) and in isolation (3/3). Not from this campaign (my only `frankensearch-index` changes are bench files; failure is on committed `origin/main`). Diagnosed: temp paths are unique (`name+pid+nanos`), **no global/shared mutable state** in `wal.rs`/`lib.rs`, and `wal_record_count()` is a `const fn` over an in-memory field — so the wrong count under parallel load points to an **IO/mmap read-after-write visibility gap** (WAL file/mmap length not consistently observed after writes when ~356 tests saturate the temp-dir IO), not a logic/state bug. | Conformance gate: `cargo test -p frankensearch-index --lib` is intermittently red in parallel → CI-flaky. **Owner action:** audit `wal.rs` read-after-write / mmap-remap-on-grow under concurrent IO (or mark the WAL tests `#[serial]`). Not fixed here (WAL-internals, another agent's active crate, correctness-critical). |
| 2026-06-26 | BlackThrush (`frankensearch-cod-a`) | `frankensearch-index/dot_product f16_slice` safe direct f16 lane-load probe | Fresh lever attempt: make `widen8_f16_slice` mirror the byte path by directly reinterpreting `[f16; 8]` to `u16x8`, then compare `f16_slice_new` against a bench-only pre-lane-load baseline. The literal requested `cargo bench --release -p frankensearch-index --bench dot_product f16_slice -- --sample-size 10 --warm-up-time 1 --measurement-time 1` failed on remote `ovh-a` with Cargo's known `unexpected argument '--release'`; fallback `cargo bench -p frankensearch-index --bench dot_product --profile release f16_slice -- --sample-size 10 --warm-up-time 1 --measurement-time 2` then failed before measurement because `half::f16: Pod` is not implemented for `bytemuck::cast::<[f16; 8], u16x8>`. | No ratio produced. Probe reverted. Do not retry without either enabling a vetted `half` bytemuck feature across the workspace or introducing unsafe, both of which require a larger proof surface than a one-lever BOLD pass. |
| 2026-06-26 | BlackThrush | **Shared-tree contention blocks clean `simd.rs` commits** + ready-to-land `f32_f32` 4-acc lever | Identified the next dot-kernel lever: `dot_product_f32_f32_unchecked` is still **single-accumulator** and on **real paths** — WAL-entry scoring (`search.rs:528,546`, `two_tier.rs:310,392,415`) and **MRL truncated search** (`mrl.rs:464,497`). It is cheap-decode (plain f32 load), so the 4-accumulator ILP pattern that won for `f32_bytes` (ledger) and `dot_i8_i8` (`0.856`/`0.939`) wins here too — **now MEASURED** (applied the edit, ran `f32_slice_new`=4acc vs `f32_slice_old`=single-acc in-process): **dim384 `820.9µs → 775.0µs = 0.944 (~1.06×)`, dim256 `440.3µs → 433.2µs = 0.984 (~1.02×)**. Smaller than i8/`f32_bytes` (f32 is more load-bound, so less ILP headroom); the dim256 gain is near noise. f32 correctness tests (`simd_matches_scalar_f32`, `large_256d_matches_scalar_f32`) pass with the edit. **Could not LAND it:** `simd.rs` is an active multi-agent battleground — across this + prior iterations agents committed identical 4-acc `dot_i8_i8`, escalated it to 8-acc in uncommitted WIP, and an aggressive edit/revert cycle **reverted my `f32_f32` edit to HEAD three separate times** within minutes, so the edit can't be held long enough to test+commit cleanly. | **Blocker = no file coordination across agents in one shared working dir** (edits don't survive). The win is modest (~1.06× on the conditional WAL/MRL f32 path), so it's **low priority** vs bigger levers. Whoever owns `simd.rs` in an isolated worktree: apply 4 accumulators to `dot_product_f32_f32_unchecked` (bench A/B already exists); the f32 sum reorder is the same accepted non-bit-identical trade as `f32_bytes`. Fleet fix: isolated git worktrees per agent, or `file_reservation` (agent-mail) before editing `simd.rs`/`dot_product.rs`/ledgers. |

---

## Gated levers (measured headroom that can't be landed as library code)

### 2026-06-25 — AVX2 build is the biggest remaining dot-kernel lever, but it's a build-config knob (BlackThrush)

**Finding:** every SIMD win so far is on an **SSE2-class build** (no AVX2/SSE4.1 — the ~1 ns/elem
f32 dots and the reverted `vpmaddwd` int8 experiment both confirm it). Rebuilding the dot bench
with `RUSTFLAGS="-C target-feature=+avx2,+fma,+f16c"` (separate target dir) measured, back-to-back:

| Workload | SSE2 | AVX2 | AVX2/SSE2 |
|----------|------|------|-----------|
| `dot/dim256/f16_bytes` | 2.09 ms | 0.77 ms | ~0.37 |
| `dot/dim256/f32_bytes` | 1.49 ms | 0.59 ms | ~0.40 |
| `dot/dim384/f16_bytes` | 3.20 ms | 1.94 ms | ~0.61 |
| `dot/dim384/f32_bytes` | 2.30 ms | 1.54 ms | ~0.67 |

**Honest caveat:** these are **cross-run on different rch workers** — the per-dim inconsistency
(dim256 ~2.5× vs dim384 ~1.6×) shows worker variance is mixed in. A clean same-worker pin was
attempted on `vmi1149989` but that worker was contended (SSE2 there ran +27% vs its own baseline
and the AVX2 leg didn't complete). So the real figure is a **~1.5–2.5× range, not a precise ratio**.

**Why it can't be landed as a code lever:**
- A *published* library cannot assume `-Ctarget-cpu`/`target-feature`; consumers compile with their
  own flags, and a workspace `.cargo/config.toml` `+avx2` would make the **released `fsfs` binary**
  crash (illegal instruction) on non-AVX2 hosts.
- Runtime AVX2 dispatch (`is_x86_feature_detected!` + `#[target_feature]`) needs `unsafe`. The crate
  is `deny(unsafe_code)` (opt-in allowed, but a hand-written AVX2 intrinsic dot kernel is a large,
  risky surface — and `wide` only uses compile-time features, so it can't help at runtime).

**Actionable recommendation (not a code change):** deploy targets known to have AVX2 should build
`fsfs` / the consuming app with `RUSTFLAGS=-Ctarget-cpu=x86-64-v3` (or `native`) for ~1.5–2.5×
faster vector search for free — `wide` then auto-selects its AVX2 paths. This belongs in the
packaging/deploy docs, not the library default. **Do not** add workspace-wide `+avx2` (breaks the
portable released binary).

---

## Residual comparator negatives

### 2026-06-26 — BOLD int8 two-pass vector wiring is not a Tantivy-class win (BlackThrush)

**Lever tested and reverted:** route the BOLD hash-hybrid challenger through a resident
`InMemoryVectorIndex::search_top_k_int8_two_pass(..., candidate_multiplier=5)` instead of the
file-backed exact FSVI vector scan. The candidate kept the Tantivy/Lucene-class incumbent
untouched, added a one-time exactness gate that compared int8 candidate doc-id order with exact
FSVI order for every BOLD query that reached the vector path, and passed that gate before timing.

**Measured command (per-crate, warm target dir; RCH local fallback because no worker was
admissible: `insufficient_slots=4,hard_preflight=1`):**
```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- env \
  FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
  FRANKENSEARCH_BOLD_VERIFY_SUMMARY_ONLY=1 \
  FRANKENSEARCH_BOLD_VERIFY_OUT=.scratch/bold_verify_int8_candidate_summary \
  RUST_LOG=off \
  cargo bench -p frankensearch --features lexical --profile release \
    --bench search_bench bold_verify_tantivy_class \
    -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact: `frankensearch/.scratch/bold_verify_int8_candidate_summary/summary.jsonl`
(`git_sha=28aa02207d1d81a836b645df80172f197e997437`, `worker=unknown`). The summary-only switch
was temporary candidate code to avoid the known Criterion tracing flood; it was reverted with the
int8 wiring. The literal `cargo bench --release` form remains invalid in this checkout, so the
successful per-crate run used Cargo's `--profile release` form.

**Decision:** rejected and reverted. Even with exact vector-candidate order preserved, the
Tantivy-class comparator is mostly worse: the int8 path only helps isolated zero-hit / 100k quoted
rows, while common 10k and 100k rows regress materially.

| Workload | Tantivy-class p50 | Candidate p50 | Candidate / Tantivy-class | Decision |
|----------|-------------------|---------------|----------------------------|----------|
| `top10_exact_identifier/10000` | 150 us | 155 us | 1.033 | zero-gain/reverted |
| `top10_short_keyword/10000` | 160 us | 193 us | **1.206x slower** | reverted |
| `top10_quoted_phrase/10000` | 273 us | 355 us | **1.300x slower** | reverted |
| `top10_natural_language/10000` | 223 us | 313 us | **1.404x slower** | reverted |
| `top10_high_fanout/10000` | 74 us | 108 us | **1.459x slower** | reverted |
| `top10_zero_hit/10000` | 36 us | 23 us | 0.639 | isolated/no keep |
| `limit_all/10000` | 14.385 ms | 16.565 ms | **1.152x slower** | reverted |
| `top10_exact_identifier/100000` | 1.129 ms | 1.229 ms | **1.089x slower** | reverted |
| `top10_short_keyword/100000` | 292 us | 286 us | 0.979 | zero-gain/reverted |
| `top10_quoted_phrase/100000` | 1.582 ms | 1.521 ms | 0.961 | isolated/no keep |
| `top10_natural_language/100000` | 1.086 ms | 1.333 ms | **1.227x slower** | reverted |
| `top10_high_fanout/100000` | 645 us | 911 us | **1.412x slower** | reverted |
| `top10_zero_hit/100000` | 30 us | 32 us | **1.067x slower** | reverted |

**Route next:** do not wire int8 two-pass into BOLD hybrid as a blanket replacement. The vector
primitive is still useful for standalone large-N vector scans, but the BOLD gap is dominated by
lexical materialization / tracing / RRF overhead on the mixed query stream, not by exact vector scan
cost alone.

### 2026-06-26 — Reusing `QueryClass` inside BOLD hybrid search is mixed/noise (BlackThrush)

**Lever tested and reverted:** compute `QueryClass::classify(query.text)` once in
`frankensearch_hash_hybrid_search`, then pass the enum into
`bold_verify_lexical_prefetch_limit` and `bold_verify_lexical_short_circuit` instead of
reclassifying the same query twice. This was scoped to the BOLD harness path and did not change
Tantivy incumbent behavior or product search semantics.

**Requested protocol blocker:** the literal requested command shape still fails before
measurement on current Cargo:
```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo bench --release -p frankensearch --features lexical \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

RCH selected `hz2`, then Cargo rejected `--release` for `cargo bench` with
`unexpected argument '--release' found`; use `--profile release` for this bench command.

**Measured command (per-crate, warm target dir; RCH shell-wrapper local fallback for output
capture):**
```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifacts:
- baseline summary: `frankensearch/.scratch/bold_queryclass_baseline_20260626T0134Z/summary.jsonl`
- candidate summary: `frankensearch/.scratch/bold_queryclass_candidate_20260626T0144Z/summary.jsonl`

Both summaries completed all 26 BOLD rows at `git_sha=6b70864826f76ac33a964dd561ef451b39fe88d2`.
The shell wrapper was needed to capture massive tracing output and therefore fell open locally
(`worker=unknown`); the result is valid as a same-machine reject, not as a remote-worker keep.
The post-summary Criterion loop was interrupted after summary emission to stop trace-log growth.

**Decision:** rejected and reverted. Median hybrid p50 movement was noise
(`candidate / baseline = 0.991`), with important short/exact rows regressing beyond the
3% rejection threshold. Some 100k rows improved, but the mixed profile is not a keep and several
candidate rows remain slower than the Tantivy/Lucene-class incumbent.

| Workload | Baseline p50 | Candidate p50 | Candidate / baseline | Candidate / Tantivy-class | Decision |
|----------|--------------|---------------|----------------------|----------------------------|----------|
| `top10_exact_identifier/10000` | 104 us | 132 us | **1.269x slower** | **1.282x slower** | reverted |
| `top10_short_keyword/10000` | 41 us | 49 us | **1.195x slower** | **1.065x slower** | reverted |
| `limit_all/10000` | 5.863 ms | 5.990 ms | 1.022 | **1.139x slower** | zero-gain/reverted |
| `top10_short_keyword/100000` | 189 us | 200 us | **1.058x slower** | **1.020x slower** | reverted |
| `top10_zero_hit/100000` | 24 us | 24 us | 1.000 | 0.960 | zero-gain/reverted |
| `top10_natural_language/10000` | 111 us | 110 us | 0.991 | **1.058x slower** | zero-gain/reverted |
| `top10_high_fanout/10000` | 69 us | 68 us | 0.986 | 1.000 | noise/reverted |
| `top10_natural_language/100000` | 798 us | 626 us | 0.784 | 0.817 | isolated/no keep |
| `top10_quoted_phrase/100000` | 1.194 ms | 965 us | 0.808 | 0.712 | isolated/no keep |

### 2026-06-26 — Early BOLD short-circuit before `ScoredResult` allocation is not a keep (BlackThrush)

**Lever tested and reverted:** move the BOLD hash-hybrid lexical short-circuit check before
`lexical_doc_ids_as_scored`, so saturated lexical rows can return the raw `LexicalIdHit` vector
instead of allocating temporary `ScoredResult` wrappers first. This was intentionally scoped to
the BOLD harness path after the rejected Tantivy fast-field attempt, leaving the Tantivy incumbent
and `TantivyIndex::search_doc_ids` comparator untouched.

**Measured command (RCH local fallback; no admissible workers:
`insufficient_slots=4,hard_preflight=1`; per-crate, warm target dir):**
```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- env \
  FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
  FRANKENSEARCH_BOLD_VERIFY_COMMAND='RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b rch exec -- env FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=off cargo bench -p frankensearch --features lexical --profile release --bench search_bench bold_verify_tantivy_class -- --sample-size 10 --warm-up-time 1 --measurement-time 1' \
  RUST_LOG=off \
  cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact: `/data/projects/.rch-targets/frankensearch-cod-b/criterion/bold_verify/summary.md`
and `summary.jsonl`. The BOLD summary completed; the remaining Criterion process was interrupted
afterward because the `#[instrument]` close-event tracing still emitted massive output despite
`RUST_LOG=off`.

**Decision:** rejected and reverted. The isolated 100k natural-language/high-fanout p50 wins are
not acceptable because the same lever leaves the main 10k rows slower and badly regresses the
100k zero-hit row. It is also only a harness allocation reorder, not a library search primitive.

| Workload | Tantivy-class p50 | Candidate frankensearch p50 | Candidate / Tantivy-class | Decision |
|----------|-------------------|------------------------------|----------------------------|----------|
| `top10_exact_identifier/10000` | 105 us | 105 us | 1.000 | zero-gain |
| `top10_short_keyword/10000` | 134 us | 197 us | **1.470x slower** | reverted |
| `top10_quoted_phrase/10000` | 121 us | 211 us | **1.744x slower** | reverted |
| `top10_natural_language/10000` | 120 us | 222 us | **1.850x slower** | reverted |
| `top10_high_fanout/10000` | 176 us | 188 us | **1.068x slower** | reverted |
| `top10_zero_hit/10000` | 18 us | 21 us | **1.167x slower** | reverted |
| `limit_all/10000` | 7.033 ms | 7.160 ms | 1.018 | zero-gain |
| `top10_exact_identifier/100000` | 1.256 ms | 1.262 ms | 1.005 | zero-gain |
| `top10_short_keyword/100000` | 324 us | 317 us | 0.978 | zero-gain/no keep |
| `top10_quoted_phrase/100000` | 1.148 ms | 1.290 ms | **1.124x slower** | reverted |
| `top10_natural_language/100000` | 988 us | 789 us | 0.799 | isolated/no keep |
| `top10_high_fanout/100000` | 677 us | 638 us | 0.942 | isolated/no keep |
| `top10_zero_hit/100000` | 67 us | 174 us | **2.597x slower** | reverted |

**Future route:** the next real lever is not another harness-side wrapper allocation skip. Target
the actual `search_doc_ids` materialization primitive without touching the incumbent comparator,
or suppress/avoid per-hit `#[instrument]` close-event logging in the measurement harness before
trusting very small microsecond deltas.

### 2026-06-26 — Tantivy fast `id` column is a comparator-poisoning loss (BlackThrush)

**Lever tested and reverted:** mark the Tantivy `id` text field as `FAST` and make
`TantivyIndex::search_doc_ids` pull IDs from the per-segment string fast field instead of loading
stored docs. This followed the prior stored-doc materialization hypothesis, but Tantivy text fast
fields are dictionary encoded; resolving each hit still requires `ord_to_str`, and large result
sets repeatedly pay dictionary lookup/decode costs.

**Measured command (RCH local fallback; no admissible workers:
`insufficient_slots=5,hard_preflight=1`; per-crate, warm target dir):**
```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- env \
  FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
  FRANKENSEARCH_BOLD_VERIFY_COMMAND='RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b rch exec -- env FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error cargo bench -p frankensearch --features lexical --profile release --bench search_bench bold_verify_tantivy_class -- --sample-size 10 --warm-up-time 1 --measurement-time 1' \
  RUST_LOG=error \
  cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact: `/data/projects/.rch-targets/frankensearch-cod-b/criterion/bold_verify/summary.md`
and `summary.jsonl`. The BOLD summary completed; the remaining Criterion process was interrupted
afterward because per-iteration tracing produced massive output.

**Why this is rejected:** the candidate changes the same `search_doc_ids` function used by both
the frankensearch path and the `tantivy_doc_ids` incumbent, so the emitted candidate/incumbent
ratio is comparator-poisoned. Against the prior mainline Tantivy-class ledger, the hot rows are
material regressions:

| Workload | Prior main Tantivy-class p50 | Candidate frankensearch p50 | Candidate / prior Tantivy-class | Decision |
|----------|------------------------------|------------------------------|----------------------------------|----------|
| `top10_short_keyword/10000` | 43 us | 310 us | **7.209x slower** | reverted |
| `top10_high_fanout/10000` | 171 us | 245 us | **1.433x slower** | reverted |
| `top10_zero_hit/10000` | 29 us | 140 us | **4.828x slower** | reverted |
| `limit_all/10000` | 9.832 ms | 135.282 ms | **13.76x slower** | reverted |
| `top10_quoted_phrase/100000` | 1.143 ms | 1.130 ms | **0.989x** | isolated/no keep |

Candidate-run examples showing the poisoned incumbent effect:

| Workload | Candidate mutated Tantivy p50 | Candidate frankensearch p50 | Emitted ratio | Why not accepted |
|----------|-------------------------------|------------------------------|---------------|------------------|
| `limit_all/10000` | 199.898 ms | 135.282 ms | 0.677 | both sides are >10x slower than prior mainline |
| `top10_short_keyword/10000` | 284 us | 310 us | 1.092 | still slower than the mutated incumbent and far slower than prior mainline |
| `top10_zero_hit/100000` | 87 us | 39 us | 0.448 | isolated win, but same code regresses 10k zero-hit and `limit_all` badly |

**Decision:** reverted all source changes. Text fast fields are not the right ID materialization
primitive for this workload. A future attempt needs an ID retrieval path that does not dictionary
decode per hit, or it must keep the comparator immutable and measure frankensearch-only changes.

### 2026-06-25 — BOLD-VERIFY after lexical prefetch budget gate: mixed, not universal (BlackThrush)

**Lever kept elsewhere:** the BOLD hash-hybrid harness now asks Tantivy for only `k` lexical
candidates, not `3k`, on classes that can legally short-circuit to lexical-only results. The
target natural-language rows became p50 wins vs the Tantivy/Lucene/Meilisearch-class incumbent and
are recorded in `docs/PERF_LEDGER.md`. The same run still has slower/noisy rows below.

**Measured command (RCH local fallback; no admissible workers:
`insufficient_slots=5,hard_preflight=1`; per-crate, warm target dir):**
```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- env \
  FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
  FRANKENSEARCH_BOLD_VERIFY_COMMAND='CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b rch exec -- env FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error cargo bench -p frankensearch --features lexical --profile release --bench search_bench bold_verify_tantivy_class -- --sample-size 10 --warm-up-time 1 --measurement-time 1' \
  RUST_LOG=error \
  cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact: `/data/projects/.rch-targets/frankensearch-cod-b/criterion/bold_verify/summary.md`
and `summary.jsonl`.

**Residual rows for `hash_hybrid_tantivy_vector_rrf` vs `tantivy_doc_ids`:**

| Workload | Corpus hash | Tantivy-class p50 | frankensearch p50 | Ratio vs Tantivy-class | Decision |
|----------|-------------|-------------------|-------------------|------------------------|----------|
| `top10_short_keyword/10000` | `2e78365a46a7c3b9` | 43 us | 177 us | **4.116x slower** | no dominance claim |
| `top10_quoted_phrase/10000` | `2e78365a46a7c3b9` | 301 us | 301 us | **1.000x** | p50 tie; tails slower |
| `top10_high_fanout/10000` | `2e78365a46a7c3b9` | 171 us | 228 us | **1.333x slower** | no dominance claim |
| `top10_zero_hit/10000` | `2e78365a46a7c3b9` | 29 us | 46 us | **1.586x slower** | no dominance claim |
| `limit_all/10000` | `2e78365a46a7c3b9` | 9.832 ms | 10.821 ms | **1.101x slower** | no dominance claim |
| `top10_quoted_phrase/100000` | `13f1b0153f5adec9` | 1.143 ms | 1.340 ms | **1.172x slower** | no dominance claim |

**Decision:** keep the scoped prefetch-budget gate because it produced clean p50 wins on the
target natural-language rows (`0.961x` at 10k, `0.962x` at 100k) and several lexical-saturated
rows. Do not generalize it: quoted phrases, 10k short keywords/high fanout/zero-hit, and
`limit_all` still need different levers.

### 2026-06-25 — BOLD-VERIFY after non-semantic zero-hit gate: still scoped, not universal (BlackThrush)

**Lever kept elsewhere:** non-semantic hash/no-quality searches now skip hash-vector work when
lexical returns zero candidates; the 100k zero-hit BOLD row is a real Tantivy-class win and is
recorded in `docs/PERF_LEDGER.md`. The same run shows the lever does not make the hash-hybrid path
universally faster than the Tantivy/Lucene/Meilisearch-class incumbent.

**Measured command (RCH worker `hz2`, per-crate, warm target dir):**
```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- env \
  FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
  RUST_LOG=error \
  cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact: `/data/projects/.rch-targets/frankensearch-cod-b/criterion/bold_verify/summary.md`
and `summary.jsonl`.

**Residual rows for `hash_hybrid_tantivy_vector_rrf` vs `tantivy_doc_ids`:**

| Workload | Corpus hash | Tantivy-class p50 | frankensearch p50 | Ratio vs Tantivy-class | Decision |
|----------|-------------|-------------------|-------------------|------------------------|----------|
| `top10_short_keyword/10000` | `2e78365a46a7c3b9` | 77 us | 80 us | **1.039x** | noise/tie; no dominance claim |
| `top10_quoted_phrase/10000` | `2e78365a46a7c3b9` | 146 us | 162 us | **1.110x slower** | no dominance claim |
| `top10_natural_language/10000` | `2e78365a46a7c3b9` | 148 us | 243 us | **1.642x slower** | needs lower-materialization lexical path |
| `top10_high_fanout/10000` | `2e78365a46a7c3b9` | 102 us | 122 us | **1.196x slower** | no dominance claim |
| `top10_zero_hit/10000` | `2e78365a46a7c3b9` | 44 us | 43 us | **0.977x** | p50 tie; tails improved, not a clean p50 win |
| `limit_all/10000` | `2e78365a46a7c3b9` | 5.720 ms | 5.975 ms | **1.045x slower** | no dominance claim |
| `top10_exact_identifier/100000` | `13f1b0153f5adec9` | 1.198 ms | 1.228 ms | **1.025x** | p50 noise; p95 regressed |
| `top10_short_keyword/100000` | `13f1b0153f5adec9` | 190 us | 214 us | **1.126x slower** | no dominance claim |
| `top10_quoted_phrase/100000` | `13f1b0153f5adec9` | 1.055 ms | 1.041 ms | **0.987x** | noise/tie |
| `top10_natural_language/100000` | `13f1b0153f5adec9` | 736 us | 768 us | **1.043x slower** | no dominance claim |
| `top10_high_fanout/100000` | `13f1b0153f5adec9` | 644 us | 685 us | **1.064x slower** | no dominance claim |

**Decision:** keep the zero-hit gate because it turns the 100k zero-hit BOLD row into a clean
incumbent win and removes the prior catastrophic empty-result vector scan. Do not generalize the
claim: saturated natural-language still spends too much in lexical over-fetch/materialization and
needs a separate lever before it can be called Tantivy/Lucene/Meilisearch-class faster.

### 2026-06-25 — BOLD-VERIFY after lexical short-circuit: still not universal dominance (BlackThrush)

**Lever kept elsewhere:** lexical-saturated `Identifier` / `ShortKeyword` queries now skip phase-1
vector scan + RRF once Tantivy has at least `k` hits. The same BOLD run produced two real
Tantivy-class p50 wins, recorded in `docs/PERF_LEDGER.md`.

**Measured command (per-crate, warm target dir; local fallback after RCH worker `vmi1153651`
stalled):**
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error \
  cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

**Residual rows where frankensearch remained slower than the Tantivy/Lucene/Meilisearch-class
incumbent, or where the ratio is noise rather than a clean win:**

| Workload | Corpus hash | Tantivy-class p50 | frankensearch p50 | Ratio vs Tantivy-class | Decision |
|----------|-------------|-------------------|-------------------|------------------------|----------|
| `top10_exact_identifier/10000` | `2e78365a46a7c3b9` | 143 us | 171 us | **1.196x slower** | no dominance claim |
| `top10_short_keyword/10000` | `2e78365a46a7c3b9` | 49 us | 66 us | **1.347x slower** | no dominance claim |
| `top10_quoted_phrase/10000` | `2e78365a46a7c3b9` | 174 us | 177 us | **1.017x** | noise/tie |
| `top10_natural_language/10000` | `2e78365a46a7c3b9` | 133 us | 893 us | **6.714x slower** | needs different lever |
| `top10_zero_hit/10000` | `2e78365a46a7c3b9` | 29 us | 624 us | **21.517x slower** | needs zero-hit semantic gate |
| `limit_all/10000` | `2e78365a46a7c3b9` | 6.068 ms | 7.324 ms | **1.207x slower** | no dominance claim |
| `top10_short_keyword/100000` | `13f1b0153f5adec9` | 165 us | 179 us | **1.085x slower** | no dominance claim |
| `top10_quoted_phrase/100000` | `13f1b0153f5adec9` | 968 us | 964 us | **0.996x** | noise/tie |
| `top10_natural_language/100000` | `13f1b0153f5adec9` | 655 us | 4.047 ms | **6.179x slower** | needs different lever |
| `top10_high_fanout/100000` | `13f1b0153f5adec9` | 542 us | 561 us | **1.035x slower** | p50/tail still not clean |
| `top10_zero_hit/100000` | `13f1b0153f5adec9` | 21 us | 2.776 ms | **132.190x slower** | needs zero-hit semantic gate |

**Decision:** keep the scoped short-circuit because it produced real incumbent wins on
`top10_high_fanout/10000` (0.711x) and `top10_exact_identifier/100000` (0.878x), but do not
generalize it. The next radical lever should target semantic gating for natural-language and
zero-hit queries rather than more dot-product work.

---

## Reverted experiments

### 2026-06-26 — 4-accumulator f16 dot kernel: neutral-to-1.49×-SLOWER (the single accumulator is deliberate) (BlackThrush)

**Cross-kernel-inconsistency probe → rejected, validates the existing design.** `dot_product_f32_bytes_f32`
uses **4 independent `f32x8` accumulators** (to break the FMA dependency chain — its decode is a cheap
load+cast, so the loop is sum-chain-bound and ILP helps), but `dot_product_f16_f32` uses a **single**
accumulator. Hypothesis: the f16 kernel left ILP on the table. Added a faithful `dot_product_f16_f32_4acc`
(same SIMD widen, 4 accumulators) and A/B'd it in-process against the single-acc kernel
(`benches/dot_product.rs`, `f16_slice_new` vs `f16_slice_4acc`):

| corpus | 1 accumulator | 4 accumulators | ratio |
|--------|---------------|----------------|-------|
| dim256, n=10k | 1589 µs | 1586 µs | 0.998 (neutral) |
| dim384, n=10k | 2271 µs | 3392 µs | **1.493 (49 % SLOWER)** |

The f16 kernel is **decode/widen-bound, not sum-chain-bound**: the Giesen magic-multiply widen is
register-heavy, so 4 live accumulators + the widen intermediates spill the SIMD register file — hurting
badly at dim384 (the production embedding dim). So the f32_bytes/f16 accumulator-count asymmetry is
**justified, not a missed lever**: cheap-decode kernels want 4 accumulators, expensive-widen kernels want
1. The single accumulator also **preserves the documented bit-identity** of the f16 dot (see `simd.rs`
header) — 4 accumulators would reorder the sum and break it. Probe reverted (zero-gain/regression); the
single-accumulator f16 kernel stays.

### 2026-06-26 — `ScalarQuantizer::dot_product_quantized` is a SIMD-lever TRAP (test-only, not wired) (BlackThrush)

**Not attempted — flagged to save effort.** `ScalarQuantizer::dot_product_quantized` /
`cosine_similarity_quantized` (`crates/frankensearch-index/src/quantization.rs`) are scalar,
single-accumulator per-dim dequant+`mul_add` loops — they *look* exactly like the f16/f32 dot
kernels that won ~3× from a SIMD-widen rewrite, so a profiling/extreme-optimization pass will be
tempted to SIMD them. **Don't:** every non-test caller is `#[cfg(test)]`, and `Quantization::Int8` /
`ScalarQuantizer` is **not** referenced in the production search scans (`search.rs`, `in_memory.rs`).
The production int8 vector search uses the in-memory **i8 corpus-max-abs slab** two-pass
(`search_top_k_int8_two_pass`, see `docs/PERF_LEDGER.md`), not this u8/per-dim quantizer. Optimizing
`dot_product_quantized` would be a zero-impact change (dead in production). If a future index format
wires `ScalarQuantizer` into search, re-evaluate then.

### 2026-06-26 — HNSW (ANN) vs flat parallel scan at 10k: a recall/latency trade, not a default win (BlackThrush)

**Lever evaluated (not wired):** the default vector search is a flat O(N) cosine scan
(`VectorIndex::search_top_k` — rayon-parallel + SIMD f16 dot + bounded-heap + cutoff). `HnswIndex`
(behind the `ann` feature, unwired) is an approximate O(log N) graph index — the obvious "radical
lever" for the scan. Validated head-to-head via the new gated bench `benches/hnsw_vs_flat.rs`
(latency + recall@10 sweep over `ef_search`), on **clustered** vectors (64 centroids + noise — the
realistic embedding structure HNSW exploits; uniform-random vectors are a worst case for ANN recall
and gave ~½ these recalls).

**Measured command (per-crate):**
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-index --features ann --bench hnsw_vs_flat
```

| N=10000, dim=128, k=10 | recall@10 | latency | vs flat |
|------------------------|-----------|---------|---------|
| flat `search_top_k` (exact) | 1.000 | 178.4 µs | 1.00× |
| HNSW `knn_search` ef=10 | 0.725 | 69.1 µs | **2.58× faster** |
| HNSW ef=20 | 0.850 | 110.3 µs | **1.62× faster** |
| HNSW ef=40 | 0.925 | 188.5 µs | 0.95× (≈break-even) |
| HNSW ef=100 | 0.950 | 425.3 µs | 0.42× (2.4× slower) |

**Interpretation — HNSW is a recall/latency *trade*, not a free speedup.** HNSW is faster only by
sacrificing recall: it beats the flat scan at `ef ≤ 20` (1.6–2.6×) but at **0.72–0.85 recall**; to
reach high recall (≥0.95) it needs `ef ≥ 100`, where it is **2.4× slower** than flat (break-even is
~ef=40 / recall 0.925). The flat path is hard to beat because it is **rayon-parallel + SIMD f16 dot +
sequential/cache-friendly**, whereas HNSW is a **serial** graph walk with **random** memory access.

**Decision:** do **not** wire HNSW as the **default** vector index — it would be a recall regression
for the exact/high-recall path (and slower if you keep recall high). It *is* a legitimate **opt-in
approximate fast-tier** option (e.g. ef≈20 → 1.6× faster at ~0.85 recall) for latency-sensitive
callers who accept lossy top-k; that's a feature/behaviour choice, not a perf lever for the exact
path. Bench kept (gated behind `ann`) for re-validation at larger N (the crossover improves for HNSW
as N grows). Corrects the earlier uniform-random measurement (which understated recall and called it
a flat loss).

### 2026-06-26 — fusing fingerprint content-hash + char-count is zero-gain (same per-byte work) (BlackThrush)

**Lever:** `Fingerprint::compute` scanned the document twice — `fnv1a_hash(text.as_bytes())` for the
content hash and `text.chars().count()` for the char count. Fused into one byte pass
(`content_hash_and_char_count`): accumulate FNV while counting non-continuation bytes
(`b & 0xC0 != 0x80`, which equals `chars().count()`). Bit-identical — proven by
`content_hash_and_char_count_matches_reference` across ASCII + 2/3/4-byte UTF-8 (896 core lib tests
green, +1 new test).

**Measured command (per-crate):**
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench fingerprint content_hash_charcount
```

| Workload | old (2 passes) | new (fused 1 pass) | ratio | verdict |
|----------|----------------|--------------------|-------|---------|
| `content_hash_charcount` (~2.2 KB doc) | 2868.3 ns | 2873.4 ns | **1.002** | noise / no gain |

**Why it fails:** fusing two byte passes does not reduce the *per-byte* work — the FNV loop is
~2 ops/byte and `chars().count()` is ~1 branch/byte; the fused loop does the same total per-byte
work, just in one loop. The only thing saved is one loop's iteration overhead (negligible), and the
second pass was already cache-warm (the bytes are hot from the first pass). Reverted source + bench
(stashed). Lesson: pass-fusion only helps when a pass is *eliminated* (data reused), not when both
passes touch every byte with comparable per-byte cost.

### 2026-06-26 — moving `LexicalIdHit` into `ScoredResult` is a mixed BOLD result, not a keep (BlackThrush)

**Lever:** the BOLD comparator's private lexical conversion changed from
`lexical_doc_ids_as_scored(&[LexicalIdHit])` cloning each `doc_id` into a `ScoredResult` to
consuming `Vec<LexicalIdHit>` and moving the owned `doc_id`. This targets frankensearch-only
phase-1 overhead that the Tantivy/Lucene/Meilisearch-class incumbent does not pay.

**Measured command (per-crate, warm `frankensearch-cod-a`; `rch exec` fell open to local because
no workers were admissible: `insufficient_slots=4,hard_preflight=1`):**
```bash
AGENT_NAME=BlackThrush RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- sh -c 'FRANKENSEARCH_BOLD_VERIFY_EMIT=1 \
  FRANKENSEARCH_BOLD_VERIFY_OUT=.scratch/bold_verify_blackthrush_move_scored \
  RUST_LOG=error cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1'
```

Artifact: `frankensearch/.scratch/bold_verify_blackthrush_move_scored/summary.jsonl`
at git `9650b7fe906af38a075048b3b09d1d7349461575`. Caveat: the harness still emitted
high-volume span-close logs despite `RUST_LOG=error`, so this is a reject/routing run, not a
dominance claim.

Key `hash_hybrid_tantivy_vector_rrf` rows vs `tantivy_doc_ids`:

| Workload | Corpus hash | Tantivy-class p50 | frankensearch p50 | Ratio vs Tantivy-class | Decision |
|----------|-------------|-------------------|-------------------|------------------------|----------|
| `top10_exact_identifier/10000` | `2e78365a46a7c3b9` | 104 us | 128 us | **1.231x slower** | reject |
| `top10_natural_language/10000` | `2e78365a46a7c3b9` | 119 us | 213 us | **1.790x slower** | reject |
| `limit_all/10000` | `2e78365a46a7c3b9` | 6.334 ms | 6.768 ms | **1.069x slower** | reject |
| `top10_short_keyword/100000` | `13f1b0153f5adec9` | 155 us | 177 us | **1.142x slower** | reject |
| `top10_high_fanout/100000` | `13f1b0153f5adec9` | 569 us | 547 us | 0.961x | noise / insufficient |
| `top10_zero_hit/100000` | `13f1b0153f5adec9` | 23 us | 20 us | 0.870x | win row, not enough |

**Decision:** reverted the source hunk. Moving owned `doc_id` strings can help individual rows,
but it does not survive the BOLD class mix: the shipped hybrid path regresses exact identifiers,
10k natural-language, `limit_all`, and 100k short-keyword rows. Future work should first fix the
BOLD logging/artifact harness, then target a deeper id-first fusion path rather than repeating this
conversion-only lever.

### 2026-06-25 — branchless f32 sign in `embed_jl` REGRESSES (compiler already selects constants) (BlackThrush)

**Lever:** mirror the SimHash branchless-vote win (`apply_hash_votes`, kept) onto the JL hash
embedder's inner loop. `embed_jl` does `let sign = if (state & 1) == 0 { 1.0 } else { -1.0 }; *dim +=
sign;` per dimension per token (O(tokens·dim)); the xorshift LSB is effectively random, so the branch
*looked* like the same ~50%-mispredict target. Replaced with `let sign = 1.0 - 2.0 * (state & 1) as
f32;`. Bit-identical (43 hash embedder tests green incl. JL determinism/orthogonality).

**Measured command (per-crate):**
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-embed --bench hash_embed jl_sign
```

In-process A/B (`jl_sign`, ~100-word doc, dim384; identical except the per-dim sign):

| Workload | branch (`if`/`else`) | branchless (`2*b-1` arith) | ratio | verdict |
|----------|----------------------|----------------------------|-------|---------|
| `jl_sign` | 97.324 µs | 104.990 µs | **1.079** | regression |

**Why it fails (contrast with the kept SimHash win):** SimHash accumulates into **i32** counters, so
`2*b - 1` is cheap integer arithmetic and beats the branch (kept, 0.870×). `embed_jl` selects an
**f32** sign (`+1.0`/`-1.0`); the compiler already lowers the conditional to a branchless **select of
two f32 constants**, whereas the arithmetic form forces an `int→f32` conversion (`cvtsi2ss`) + a float
mul + a float sub per element — strictly more work. **Rule:** branchless arithmetic helps integer
accumulators, not float constant-selection (the compiler handles the latter). Reverted source + bench
(stashed). Do not re-attempt branchless sign on f32 select paths.

### 2026-06-25 — `collapse_code_block` slice-join is zero-gain (join allocs dominate) (BlackThrush)

**Lever:** the long-block branch of `collapse_code_block` collected the head/tail lines into
intermediate `Vec<&str>` (`lines.iter().take(head).copied().collect()` etc.) before `join("\n")`.
Since `[&str]` joins directly, the candidate replaced those with `lines[..head].join("\n")` and
`lines[lines.len()-tail..].join("\n")` to drop the two scratch vectors. Byte-identical (34
canonicalize tests green incl. `collapse_long_code_block`).

**Measured command (per-crate):**
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench canonicalize collapse_code_block
```

In-process old-vs-new A/B (`collapse_code_block`, 60-line block, head=20/tail=10):

| Workload | old (Vec collect + join) | new (slice join) | ratio new/old | verdict |
|----------|--------------------------|------------------|---------------|---------|
| `collapse_code_block` | 254.6 ns | 257.2 ns | **1.010** | noise / no gain |

**Why it fails:** the two `Vec<&str>` collects are ~20 and ~10 pointer copies — negligible next to
the actual work: the two `join("\n")` calls (each allocates + copies the joined output) and the
final `format!`. `<[&str]>::join` iterates the slice the same way the `collect` did, so eliminating
the scratch vectors saves nothing measurable. Reverted source + bench (stashed). Code-block
collapsing is not allocation-bound on the scratch vectors; no lever here.

### 2026-06-25 — caching the Tantivy `QueryParser` is zero-gain (parse dominates) (BlackThrush)

**Lever:** `TantivyIndex::parse_query_lenient` rebuilt a `QueryParser::for_index(..)` +
`set_field_boost(..)` on every BM25 search. Since the schema/tokenizers are fixed for the index's
lifetime and `parse_query_lenient` takes `&self`, the parser was cached as a struct field (built
once in the constructor) and reused. Correct and `Sync` (compiled; 79 lexical lib tests green).

**Measured command (per-crate):**
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-lexical --bench query_parser
```

In-process old-vs-new A/B (`query_parser` group; `old` reconstructs the parser per query, `new`
reuses a cached one; both run the identical lenient parse of a 7-token query):

| Workload | old (reconstruct + parse) | new (cached + parse) | ratio new/old | verdict |
|----------|---------------------------|----------------------|---------------|---------|
| `query_parser` | 8384.1 ns | 8336.3 ns | **0.994** | noise / no gain |

**Why it fails:** `QueryParser::for_index` is cheap (~tens of ns: an `Arc` tokenizer-manager clone
+ a small default-fields `Vec` + a boost map). The actual lenient *parse* — tokenizing the query
through both field analyzers and building term queries — costs ~8.3 µs and dominates, so the
construction it eliminates is <1% of the per-query cost, lost in noise. Reverted source + bench
(stashed, not landed). The real lexical materialization gap is `load_doc` (full docstore
decompress per hit in `search`/`search_doc_ids`), which needs a fast/columnar `id` field — an
index-format change, tracked separately; **not** the parser.

### 2026-06-25 — `normalize_whitespace` ASCII byte-scan fast path is SLOWER (BlackThrush)

**Lever:** `normalize_whitespace` (runs on every document at index time) walks `text.chars()` and
pushes char-by-char, then does a trailing `trim_end`/`truncate` pass. The candidate added an
`is_ascii()`-guarded byte path that scanned bytes, bulk-copied each non-whitespace run with one
`push_str`, and emitted the separating space inline (no trailing-trim pass). A custom `is_ws_ascii`
predicate (`b' ' | 0x09..=0x0d`) was used — **not** `u8::is_ascii_whitespace`, which excludes
`\x0b` (vertical tab) and would diverge from `char::is_whitespace`. Byte-identity proven across
vertical-tab/form-feed/mixed cases (`normalize_whitespace_ascii_matches`, 34/34 canonicalize tests
green).

**Measured command (per-crate, local fallback — RCH had no admissible workers):**
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench canonicalize normalize_whitespace
```

In-process old-vs-new A/B (`normalize_whitespace` group, 40× multi-line doc with newline +
multi-space runs):

| Workload | old (char path) | new (ASCII byte scan) | ratio new/old | verdict |
|----------|-----------------|-----------------------|---------------|---------|
| `normalize_whitespace` (~2.2 KB) | 3.415 µs | 3.910 µs | **1.145** | regression |

**Why it fails:** safe Rust cannot bulk-append a known-ASCII byte slice to a `String` without
`std::str::from_utf8` **re-validating** every run — that per-run validation scan costs more than
the char-by-char path saves, and std's `chars()`/`String::push` already have ASCII fast paths so
the original is near-optimal. The only way to skip validation is `from_utf8_unchecked` (`unsafe`),
and the crate is `deny(unsafe_code)`. **Do not re-attempt** the byte-scan rewrite of
`normalize_whitespace` under the safe-Rust constraint. Reverted source + bench (stash, not landed).

### 2026-06-25 — `search_minimal` lexical trait hook regresses decisive BOLD rows (BlackThrush)

**Lever:** add a `LexicalSearch::search_minimal` hook and route the non-semantic hash-tier
lexical guard through Tantivy `search_doc_ids`, converting those id-only hits back to
`ScoredResult` without loading stored metadata. The goal was to keep the measured BOLD
lexical-guard wins while avoiding full stored-document materialization in product code.

**Measured command (per-crate, warm target dir):**
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error \
FRANKENSEARCH_BOLD_VERIFY_COMMAND="CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error cargo bench -p frankensearch --features lexical --profile release --bench search_bench bold_verify_tantivy_class -- --sample-size 10 --warm-up-time 1 --measurement-time 1" \
  cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

**Artifact:** `/data/projects/.rch-targets/frankensearch-cod-a/criterion/bold_verify/summary.jsonl`
at git `bd3f59e2bc40f2d048bee34feda74ccd1049959b` (`worker="unknown"`; local warm target lane).

| Workload | Corpus hash | Tantivy-class p50 | full guard p50 | minimal guard p50 | minimal/full | minimal/Tantivy-class | Decision |
|----------|-------------|-------------------|----------------|-------------------|--------------|-----------------------|----------|
| `top10_exact_identifier/10000` | `2e78365a46a7c3b9` | 141 us | 119 us | 172 us | **1.445** | **1.220x slower** | reject |
| `top10_high_fanout/10000` | `2e78365a46a7c3b9` | 122 us | 106 us | 122 us | **1.151** | 1.000x tie | reject |
| `top10_short_keyword/10000` | `2e78365a46a7c3b9` | 59 us | 35 us | 45 us | **1.286** | 0.763x faster | reject vs shipped guard |
| `top10_quoted_phrase/10000` | `2e78365a46a7c3b9` | 149 us | 145 us | 146 us | 1.007 | 0.980x noise | no keep signal |
| `top10_natural_language/10000` | `2e78365a46a7c3b9` | 143 us | 151 us | 135 us | 0.894 | 0.944x faster | insufficient |
| `top10_zero_hit/10000` | `2e78365a46a7c3b9` | 37 us | 36 us | 23 us | 0.639 | 0.622x faster | insufficient |
| `limit_all/10000` | `2e78365a46a7c3b9` | 11.923 ms | 14.395 ms | 8.208 ms | 0.570 | 0.688x faster | insufficient |
| `top10_exact_identifier/100000` | `13f1b0153f5adec9` | 1.299 ms | 1.238 ms | 1.246 ms | 1.006 | 0.959x faster | noise |
| `top10_high_fanout/100000` | `13f1b0153f5adec9` | 611 us | 864 us | 1.000 ms | **1.157** | **1.637x slower** | reject |
| `top10_natural_language/100000` | `13f1b0153f5adec9` | 737 us | 1.069 ms | 816 us | 0.763 | **1.107x slower** | reject |
| `top10_quoted_phrase/100000` | `13f1b0153f5adec9` | 1.095 ms | 1.122 ms | 1.049 ms | 0.935 | 0.958x faster | insufficient |
| `top10_short_keyword/100000` | `13f1b0153f5adec9` | 213 us | 69 us | 187 us | **2.710** | 0.878x faster | reject vs shipped guard |
| `top10_zero_hit/100000` | `13f1b0153f5adec9` | 69 us | 59 us | 61 us | **1.034** | 0.884x faster | reject vs shipped guard |

**Decision:** reverted the trait hook, Tantivy override, BOLD harness variant, and test. The
minimal path wins some broad/materialization-heavy rows, but it gives back or destroys the exact
identifier, high-fanout, and short-keyword rows that make the current lexical guard worth keeping.
Do not add a public minimal-scored trait method until the backend can skip stored-document loading
without hurting these high-selectivity paths. A future attempt should target a private id-first
fusion path that avoids rebuilding owned `ScoredResult` rows for phase 1, then bench against this
same BOLD harness.

### 2026-06-25 — BOLD-VERIFY: hash-hybrid does **not** beat Tantivy-class BM25 (BlackThrush)

**Comparator shipped:** `frankensearch/benches/search_bench.rs` now includes
`bold_verify_tantivy_class`, a same-corpus Tantivy/Lucene-class incumbent harness:
Tantivy `search_doc_ids` vs frankensearch hash embedding + FSVI vector search + Tantivy candidates
+ RRF fusion. This is a **negative dominance check**, not a reverted source lever.

**Measured command:** `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a
rch exec -- env FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error cargo bench -p frankensearch
--features lexical --profile release --bench search_bench bold_verify_tantivy_class
-- --sample-size 10 --warm-up-time 1 --measurement-time 3`

**Evidence:** the full per-crate Criterion pass completed on RCH worker `vmi1152480`. A filtered
summary rerun on RCH worker `hz2` emitted the machine-readable BOLD rows below; the emit flag must
be passed inside `rch exec -- env ...` because the wrapper does not preserve that outer environment
variable. The rows use the same fixed corpus/query harness and report
frankensearch p50 / Tantivy-class p50:

| Workload | Corpus hash | Tantivy-class p50 | frankensearch p50 | ratio |
|----------|-------------|-------------------|-------------------|-------|
| `top10_exact_identifier/10000` | `2e78365a46a7c3b9` | 449 us | 1.716 ms | **3.82x slower** |
| `top10_short_keyword/10000` | `2e78365a46a7c3b9` | 92 us | 1.616 ms | **17.57x slower** |
| `top10_quoted_phrase/10000` | `2e78365a46a7c3b9` | 145 us | 1.711 ms | **11.80x slower** |
| `top10_natural_language/10000` | `2e78365a46a7c3b9` | 352 us | 1.661 ms | **4.72x slower** |
| `top10_high_fanout/10000` | `2e78365a46a7c3b9` | 317 us | 1.749 ms | **5.52x slower** |
| `top10_zero_hit/10000` | `2e78365a46a7c3b9` | 80 us | 1.450 ms | **18.12x slower** |
| `limit_all_limit_all/10000` | `2e78365a46a7c3b9` | 6.324 ms | 12.318 ms | **1.95x slower** |
| `top10_exact_identifier/100000` | `13f1b0153f5adec9` | 2.260 ms | 8.143 ms | **3.60x slower** |
| `top10_short_keyword/100000` | `13f1b0153f5adec9` | 515 us | 4.082 ms | **7.93x slower** |
| `top10_quoted_phrase/100000` | `13f1b0153f5adec9` | 1.040 ms | 6.412 ms | **6.17x slower** |
| `top10_natural_language/100000` | `13f1b0153f5adec9` | 1.641 ms | 5.756 ms | **3.51x slower** |
| `top10_high_fanout/100000` | `13f1b0153f5adec9` | 647 us | 3.742 ms | **5.78x slower** |
| `top10_zero_hit/100000` | `13f1b0153f5adec9` | 48 us | 2.827 ms | **58.90x slower** |

**Decision:** no Lucene/Tantivy/Meilisearch-class win exists for the current hash-hybrid path.
Future bold claims need a new lever that changes the cost model (ANN/int8 slab reuse, lexical
short-circuiting, semantic gating, or a Meilisearch-class prefix/typo comparator), then must reuse
this head-to-head harness. Do not cite frankensearch hybrid as faster than Tantivy-class BM25 from
the current implementation.

### 2026-06-25 — binary-quantization ADC is too coarse for top-10; int8 ADC dominates (BlackThrush)

**Lever assessed (not built):** a Meilisearch-style **binary-quantization** first pass — pack
`sign(x_i)` to bits, rank by Hamming agreement (`popcnt`, fast even on SSE2, 1/16 the bytes of
f16), then exact f16 rescore. Faster pass-1 than int8, so the question is recall. Measured
(`binary_quant_recall_at_10`, random L2-normalized vectors, dim=128, n=3000, recall@10):

| candidate mult | 5 | 10 | 20 | 50 | 100 |
|----------------|------|------|------|------|------|
| binary recall@10 | 0.54 | 0.71 | 0.85 | 0.96 | **1.00** |
| (int8 recall@10) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

**Conclusion:** binary needs `mult ≈ 100` (k·mult = 1000 candidates) for recall ≈ 1.0, vs int8's
`mult = 2` (20 candidates) — ~50× coarser. At frankensearch's typical scale that means rescoring a
large fraction of the corpus, so the **already-shipped int8 ADC two-pass is strictly better** for
top-10. Binary only pulls ahead at very large N (≫1M), where the *fixed* ~1000-candidate rescore
is negligible and the super-fast `popcnt` pass-1 dominates. **Do not build binary ADC** unless
specifically targeting ≫1M-vector corpora. (Probe test kept for reproducibility; no production
code added.)

### 2026-06-26 — 8-accumulator int8 dot is not a robust successor to 4acc (BlackThrush)

**Lever tested and reverted:** extend the shipped `dot_i8_i8` loop from four independent
`i32x8` accumulators over 32 elements to eight accumulators over 64 elements, then try a narrower
384-dim-only gate after the global form regressed the 256-dim path. This is a primitive behind the
int8 ADC pass-1 vector scan, not a direct Lucene/Tantivy-class BM25 comparator; because all code
was reverted, the product-level Tantivy/Lucene/Meilisearch-class ratios remain the BOLD rows above.

The literal requested command shape with `cargo bench --release` was checked first and Cargo
rejected it (`unexpected argument '--release'`), so measurements used the equivalent release profile:
`AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a rch exec -- cargo bench -p frankensearch-index --bench dot_product --profile release i8_dot -- --sample-size 10 --warm-up-time 1 --measurement-time 2`.

**Remote `ovh-a`, global 8acc vs bench-local 4acc comparator (Criterion mean):**

| Workload | 4acc comparator | 8acc candidate | candidate/4acc | Decision |
|----------|-----------------|----------------|-----------------|----------|
| `dot/dim256/i8_dot` | 376.995 us | 395.577 us | **1.049** | regression |
| `dot/dim384/i8_dot` | 2.410 ms | 1.092 ms | 0.453 | apparent win, not landable because dim256 regressed |

**Local `rch` fallback, 384-dim-gated retry vs bench-local 4acc comparator (Criterion stdout
point estimate):**

| Workload | 4acc comparator | gated candidate | candidate/4acc | Decision |
|----------|-----------------|-----------------|-----------------|----------|
| `dot/dim256/i8_dot` | 619.51 us | 786.06 us | **1.269** | regression |
| `dot/dim384/i8_dot` | 1.0314 ms | 1.7042 ms | **1.652** | regression |

**Decision:** reverted the production kernel and the temporary bench-only 4acc comparator. The
current 4-accumulator implementation remains the only accepted int8 dot ILP lever; do not retry
8acc without a same-process no-regression proof for 256 and a stable 384 win.

### 2026-06-25 — SIMD-widen int8 dot (`i16x16::dot`/`vpmaddwd`) is SLOWER on this build (BlackThrush)

**Lever:** rewrite `dot_i8_i8` from scalar `i16::from` widening (16 `movsx` per 8 elems) to a
fully-SIMD 16-wide kernel: `[i8;16]` → `i8x16` → `i16x16` (sign-extend) → `i16x16::dot`
(`vpmaddwd`, pairwise products → `i32x8`). Correct (`dot_i8_i8_matches_scalar` green incl. the
all-(−128)/512-dim overflow case), but measured a **regression** vs the committed scalar-widen
kernel:

| Workload | int8/f16 ratio (scalar-widen, kept) | int8/f16 ratio (SIMD-widen) |
|----------|-------------------------------------|-----------------------------|
| `dot/dim256/i8_dot` | 0.331 | **0.508** |
| `dot/dim384/i8_dot` | 0.311 | **0.442** |

(in-process int8-vs-f16 ratio; higher = int8 got relatively slower). Normalizing for worker
speed, the SIMD-widen int8 dot is ~1.5× **slower** than scalar-widen.

**Why it fails:** this is an **SSE2-class build** (no AVX2/SSE4.1 — consistent with the ~1 ns/elem
f32 dots seen elsewhere). `i8→i16` `vpmovsxbw` and 256-bit `i16x16` are then *emulated* (unpack +
arithmetic-shift over two SSE registers), which costs more than 16 plain scalar `movsx`. "Portable
SIMD" loses to scalar when the target lacks the widening instruction. Reverted to scalar-widen.
**Do not re-attempt** without either a runtime AVX2/SSE4.1 dispatch or `-Ctarget-cpu` that enables
those features (which the published library cannot assume).

### 2026-06-24 — int8 ADC two-pass does NOT beat *parallel* exact at top-10/10k (BlackThrush)

**Self-correction of an earlier overstated result.** The `3ecfad8` bench reported the int8 ADC
two-pass ~2.6–3× faster than "exact f16", but that baseline (`topk_exact_f16`) was a **serial
full-sort** pipeline. The **real product** `InMemoryVectorIndex::search_top_k` is **rayon-parallel
+ bounded-heap + cutoff** — much faster. Benching the real shipped methods head-to-head
(`inmem_topk`, 10k vectors, top-10, mult=20, parallel pass-1):

| Workload | exact `search_top_k` (parallel) | `search_top_k_int8_two_pass` | ratio |
|----------|--------------------------------|------------------------------|-------|
| `inmem_topk/dim256` | 306 µs | 373 µs | **1.22 (regression)** |
| `inmem_topk/dim384` | ~400–700 µs (very noisy) | 393 µs | inconclusive |

**Root cause:** the int8 *kernel* is genuinely ~3× faster (that stands — `33fb45b`), but the
two-pass **method** materializes all N int8 scores into a `Vec` then selects serially, while the
exact path never materializes more than the top-k heap and runs across all cores. At 10k the
already-parallel+cutoff exact is ~300 µs; the two-pass's full-N materialize + serial select eats
the kernel win. So the int8 ADC two-pass is **not** a win at this scale/path.

**Honest scope of the kept results:** int8 dot ~3× (kernel, real), recall@10 = 1.0 (real). The
**search-level** speedup only holds vs a *serial* exact — it does **not** beat the product's
parallel exact at 10k. The lever's real upside is at larger N (100k+, where parallel exact also
slows and bandwidth matters more) or the mmap FSVI path (page-fault + decode overhead), or with a
**bounded-heap parallel pass-1** (avoid the full-N materialize). Filed as a follow-up.

**Decision:** the `search_top_k_int8_two_pass` method is kept (correct, opt-in, bit-identical when
recall=1 — proven by `int8_two_pass_matches_exact_topk`; a foundation), but it carries **no
verified perf-win claim at 10k**. PERF_LEDGER corrected accordingly.

### 2026-06-24 — multi-accumulator unrolling of the **f16** dot-product kernels (BlackThrush)

**Lever:** rewrite `dot_product_f16_f32` and `dot_product_f16_bytes_f32` to use 4 independent
`f32x8` accumulators (32 elements/iter) instead of 1, to break the SIMD-add latency chain.

**Measured head-to-head** (`benches/dot_product.rs`, `*_new` = 4-acc, `*_old` = original
single-acc, same process / same CPU, n=10 000 dots):

| Workload | old (median) | new (median) | ratio new/old | verdict |
|----------|-------------|-------------|---------------|---------|
| `dot/dim256/f16_bytes` | 12.483 ms | 13.504 ms | **1.082** | regression |
| `dot/dim384/f16_bytes` | 17.486 ms | 18.115 ms | **1.036** | regression |
| `dot/dim256/f16_slice` | 10.242 ms | 14.909 ms | **1.456** | regression (noise-inflated) |
| `dot/dim384/f16_slice` | 13.489 ms | 14.277 ms | **1.058** | regression |

**Why it fails:** the f16 paths are **decode-bound** — the per-element scalar `f16::to_f32()`
conversion dominates, so the accumulation latency the change targets is a negligible fraction.
The restructure (chunks_exact(32) + `try_into` per sub-block + two-phase remainder) only adds
setup overhead. Reverted these two kernels to the original single-accumulator form.

**Connects to historical revert `816963a`:** the human previously reverted `88c291b`
("…multi-accumulator unrolling"), which *bundled* this f16 kernel change with a
`select_nth_unstable` heap-merge (unstable sort → breaks deterministic tie ordering) and a
16-elem unroll. This measurement isolates the kernel part and confirms it is **not** a win on
the f16 (default-quantization) path independent of the heap change. Do not re-attempt
accumulator unrolling on f16 kernels without first making the f16→f32 decode SIMD (a
branchless `i32x8`/F16C widen), which is the actual bottleneck.

**Kept from the same experiment:** the `f32_bytes` kernel restructure was a genuine ~3× win
(decode-bound on open-ended slices, not accumulation) → see `docs/PERF_LEDGER.md`.

### 2026-06-24 — f32 slice multi-accumulator portability check (BlueGull)

**Lever:** apply the same 4-accumulator, 32-element loop shape to `dot_product_f32_f32`.

**Evidence:** after the f16 revert, this command fell back local because RCH had no admissible
workers:

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo bench -p frankensearch-index --profile release --bench dot_product
```

That local fallback measured `dot/dim256/f32_slice` at 1.4206 ms old vs 2.3436 ms new,
ratio **1.650** (regression). A later pinned-worker run with the source hunk reverted
measured the slice rows as noise-only routing checks, not a kept source change.

**Superseded 2026-06-26 by BlackThrush:** this local fallback was contradicted by an admitted
remote RCH same-process run on `vmi1227854`, which measured the f32 slice 4-accumulator kernel at
0.318 (dim256) and 0.610 (dim384) vs the embedded single-accumulator baseline. The current
decision is **ship the f32 slice rewrite** with the scope caveat above: it is a frankensearch
microkernel win, not a Lucene/Tantivy/Meilisearch-class end-to-end dominance claim.

### 2026-06-26 — BOLD re-check: f32/f32 4-accumulator slice dot is still not a win (BlackThrush)

**Lever tested and reverted:** a dirty worktree hunk rewrote
`dot_product_f32_f32_unchecked` from one `f32x8` accumulator over 8 elements to four accumulators
over 32 elements. This was the later `ready-to-land f32_f32 4-acc lever` note from the measurement
blockers table, but it had not been proven head-to-head on this tree. The hunk is a vector
primitive, not a direct Lucene/Tantivy-class BM25 comparator; because it was reverted, product-level
Lucene/Tantivy/Meilisearch-class ratios are unchanged.

The requested `cargo bench --release` form failed again with Cargo's
`unexpected argument '--release'`, so the measured command was:

```bash
AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo bench -p frankensearch-index --bench dot_product --profile release \
  f32_slice -- --sample-size 10 --warm-up-time 1 --measurement-time 2
```

RCH fell back local (`no admissible workers`) but the bench is in-process `new` vs `old`, so the
ratio is still same-run:

| Workload | old single-acc | 4acc candidate | candidate/old | Decision |
|----------|----------------|----------------|---------------|----------|
| `dot/dim256/f32_slice` | 2.4522 ms | 2.5077 ms | **1.023** | neutral / revert |
| `dot/dim384/f32_slice` | 2.9224 ms | 3.0636 ms | **1.048** | regression |

**Decision:** reverted the `simd.rs` hunk. Do not land the f32/f32 4-accumulator slice rewrite from
the shared worktree; the earlier ready-to-land note is superseded by this BOLD re-check.

### 2026-06-24 — detached-worktree f16 accumulator candidate (frankensearch-cod-b)

**Context:** a detached worktree at
`/data/projects/frankensearch-cod-b-baseline-20260624T221243Z` was used to compare
`HEAD` plus the `dot_product` bench harness against a 4-accumulator SIMD candidate that was
not on `main`. The accepted release-profile command could not be admitted by RCH (see
Measurement blockers), so the only completed same-worker numbers are Cargo's optimized bench
profile on worker `vmi1152480`.

**Bench-profile RCH before/after (`cargo bench -p frankensearch-index --bench dot_product
-- --sample-size 10 --warm-up-time 1 --measurement-time 3`):**

| Workload | before median | after median | ratio new/old | verdict |
|----------|---------------|--------------|---------------|---------|
| `dot/dim256/f16_bytes/10000` | 7.1007 ms | 3.9052 ms | 0.550 | routing win, not kept |
| `dot/dim256/f16_slice/10000` | 5.0532 ms | 4.0380 ms | 0.799 | routing win, not kept |
| `dot/dim384/f16_bytes/10000` | 7.0845 ms | 5.7334 ms | 0.809 | routing win, not kept |
| `dot/dim384/f16_slice/10000` | 5.6829 ms | 6.0270 ms | 1.061 | regression |

**Decision:** do not land this detached-worktree f16 accumulator candidate. It regresses a
tracked `f16_slice` workload and conflicts with the stronger in-process old-vs-new evidence
above, which isolates the f16 accumulator rewrite without cross-run Criterion noise. The
follow-up is `bd-gfzk`: attack the actual default-path bottleneck, scalar f16-to-f32 decode,
with an exhaustive correctness proof before rebenchmarking.

### 2026-06-24 — `f16_slice` u16x8 SIMD load (BlackThrush)

**Lever:** mirror the landed `f16_bytes` SIMD-load refinement (`c0e9c80`) onto the slice path
`widen8_f16_slice` — build a `u16x8` of `to_bits()` lanes and zero-extend to `u32x8` (16-byte
stack slot + `vpmovzxwd`) instead of materializing a `[u32; 8]`. Bit-exact (20/20 simd tests
green); the slice path feeds `dot_product_f16_f32` (`in_memory.rs:480`, two-tier quality rescore).

**Measured** (`f16_slice_new` crate vs `f16_slice_old` scalar, in-process; ratios are the only
worker-robust signal across runs):

| Workload | prior ratio (`[u32;8]`) | this change (`u16x8`) | verdict |
|----------|------------------------|-----------------------|---------|
| `dot/dim256/f16_slice` | 0.364 | **0.508** | no gain / hint of regression |
| `dot/dim384/f16_slice` | 0.394 | **0.420** | no gain (≈ noise) |

**Why reverted:** this run landed on a much slower/contended worker (absolute times ~2× the
prior run) and the cross-run ratio moved the **wrong way** on dim256 (0.364→0.508). With no
clean in-process A/B to isolate the load change from the fully-scalar baseline, there is no
demonstrated gain (and a hint of regression). Per "REVERT ~0-gain", reverted to the committed
`[u32; 8]` form. The byte-path SIMD load (`c0e9c80`) stays — it showed a consistent directional
improvement on both dims; the slice path did not. A clean keep/reject would need an in-process
"SIMD-widen + scalar-load" baseline added to `dot_product.rs` (left for a future pass).

### 2026-06-25 — non-semantic lexical guard is NOT a universal Tantivy-class win (BlackThrush)

**Kept change, bounded claim.** The hash-tier lexical guard is a real improvement over the old
hash-hybrid path (see `docs/PERF_LEDGER.md`), but the same BOLD-VERIFY run still shows several
rows slower than the `tantivy_doc_ids` lexical proxy used for Lucene/Tantivy/Meilisearch-class
comparison.

Command:

```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- env FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error \
    cargo bench -p frankensearch --features lexical --profile release \
    --bench search_bench bold_verify_tantivy_class \
    -- --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Worker: `vmi1152480` (`[RCH] remote vmi1152480 (804.5s)`). Incumbent:
`tantivy_doc_ids`.

| Workload | guarded p50 ratio | guarded p95 ratio | verdict |
|----------|-------------------|-------------------|---------|
| `top10/10000/exact_identifier` | 1.074 | 1.359 | near, but not a win |
| `top10/10000/quoted_phrase` | 1.031 | 1.330 | near, but not a win |
| `top10/10000/natural_language` | 1.040 | 1.067 | near, but not a win |
| `top10/10000/zero_hit` | 1.000 | 3.333 | p50 parity, tail miss |
| `top10/100000/exact_identifier` | 1.161 | 1.820 | miss |
| `top10/100000/short_keyword` | 1.208 | 1.439 | miss |
| `top10/100000/quoted_phrase` | 2.068 | 1.473 | miss |
| `top10/100000/high_fanout` | 1.010 | 1.236 | near, but not a win |

**Interpretation:** skipping hash embedding/vector/RRF is necessary but not sufficient for full
lexical-engine parity. The remaining overhead is in frankensearch's result materialization and
the fact that the incumbent comparator returns doc ids, while the guarded path produces
`ScoredResult`-class results. Do not claim Lucene/Tantivy/Meilisearch dominance from this change.

**Next lever:** avoid eager `ScoredResult` string/materialization for lexical-only Phase 1
(borrowed/id-first result lane or lazy metadata resolution), then rerun the same BOLD matrix.
