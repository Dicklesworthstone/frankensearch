# PERF_LEDGER.md — frankensearch measured wins

> Head-to-head measured performance wins **kept** in the tree. Each row cites the
> exact bench workload, the before/after timings, and the ratio (new/old; lower is
> faster). Dead-ends and regressions live in `docs/NEGATIVE_EVIDENCE.md`.

Build/bench protocol (per-crate ONLY, never workspace-wide):
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p <crate> --release
```

Current Cargo rejects `--release` for `cargo bench` in this checkout; successful
Criterion runs use the equivalent `--profile release` form and record that protocol
mismatch in `docs/NEGATIVE_EVIDENCE.md`.

The dominance baseline is a Tantivy/Lucene/Meilisearch-class original comparator
on identical corpora and query streams. The `bd-ui41` comparator harness now
lives in `frankensearch/benches/search_bench.rs`; its current result is negative
and recorded in `docs/NEGATIVE_EVIDENCE.md`. Rows below remain frankensearch
pre-change baselines or before/after local hot-path ratios unless explicitly
marked as original-comparator wins.

## Original-comparator wins

### 2026-06-25 — BOLD-VERIFY non-semantic zero-hit lexical gate (BlackThrush)

**Lever:** the BOLD hash-hybrid path and `TwoTierSearcher` now stop before hash-vector search
when a non-semantic fast embedder has no quality tier and lexical search produces zero candidates.
The same gate also permits lexical-only return for saturated natural-language rows in that
non-semantic/no-quality mode; that reduced the old catastrophic vector-scan residual but is still
not a universal Tantivy-class win, so the slower natural-language rows are recorded in
`docs/NEGATIVE_EVIDENCE.md`.

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

| Workload | Corpus hash | Tantivy-class p50 | frankensearch p50 | Ratio vs Tantivy-class | Status |
|----------|-------------|-------------------|-------------------|------------------------|--------|
| `bold_verify/top10/100000` `zero_hit` | `13f1b0153f5adec9` | 67 us | 62 us | **0.925** | KEEP |

**Scope:** this is a zero-hit incumbent win for the non-semantic hash/no-quality lane, not a
claim that hybrid search dominates Tantivy/Lucene/Meilisearch-class BM25 overall. The 10k zero-hit
row is a p50 tie (43 us vs 44 us; ratio 0.977) with better tails, and natural-language rows remain
slower at p50; both are recorded as residual evidence.

### 2026-06-25 — BOLD-VERIFY lexical-saturated short-circuit (BlackThrush)

**Lever:** after successful fast embedding and lexical search, `TwoTierSearcher` now returns
lexical-only initial results for `Identifier` / `ShortKeyword` queries when lexical already has
at least `k` hits and graph ranking is disabled. That skips the phase-1 vector scan and RRF work
for lexical-saturated queries while preserving the hybrid path for natural-language and zero-hit
queries.

**Measured command (local fallback after RCH worker `vmi1153651` stalled twice; per-crate,
warm target dir):**
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error \
  cargo bench -p frankensearch --features lexical --profile release \
  --bench search_bench bold_verify_tantivy_class \
  -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Artifact: `/data/projects/.rch-targets/frankensearch-cod-a/criterion/bold_verify/summary.md`
and `summary.jsonl`.

| Workload | Corpus hash | Tantivy-class p50 | frankensearch p50 | Ratio vs Tantivy-class | Status |
|----------|-------------|-------------------|-------------------|------------------------|--------|
| `bold_verify/top10/10000` `high_fanout` | `2e78365a46a7c3b9` | 114 us | 81 us | **0.711** | KEEP |
| `bold_verify/top10/100000` `exact_identifier` | `13f1b0153f5adec9` | 1.241 ms | 1.090 ms | **0.878** | KEEP |

**Scope:** this is a targeted lexical-saturation win vs a Tantivy/Lucene/Meilisearch-class
incumbent, not a universal dominance claim. The same BOLD run still shows slower/noisy rows for
10k exact identifiers, short-keyword rows, `limit_all`, natural-language, and zero-hit queries;
those ratios are recorded in `docs/NEGATIVE_EVIDENCE.md`.

### 2026-06-25 — BOLD-VERIFY lexical prefetch budget gate (BlackThrush)

**Lever:** the BOLD hash-hybrid harness now mirrors the product's lexical short-circuit budget:
for query classes that can legally return lexical-only (`Identifier`, `ShortKeyword`,
`NaturalLanguage`), it asks Tantivy for only `k` lexical candidates instead of prefetching `3k`
and then throwing the surplus away. This is an evidence-harness correction for the shipped
short-circuit path, not a new product ranking algorithm.

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

| Workload | Corpus hash | Tantivy-class p50 | frankensearch p50 | Ratio vs Tantivy-class | Status |
|----------|-------------|-------------------|-------------------|------------------------|--------|
| `bold_verify/top10/10000` `exact_identifier` | `2e78365a46a7c3b9` | 205 us | 178 us | **0.868** | KEEP |
| `bold_verify/top10/10000` `natural_language` | `2e78365a46a7c3b9` | 335 us | 322 us | **0.961** | KEEP |
| `bold_verify/top10/100000` `exact_identifier` | `13f1b0153f5adec9` | 1.596 ms | 1.540 ms | **0.965** | KEEP, p95 noisy |
| `bold_verify/top10/100000` `short_keyword` | `13f1b0153f5adec9` | 359 us | 297 us | **0.827** | KEEP |
| `bold_verify/top10/100000` `natural_language` | `13f1b0153f5adec9` | 972 us | 935 us | **0.962** | KEEP |
| `bold_verify/top10/100000` `high_fanout` | `13f1b0153f5adec9` | 895 us | 783 us | **0.875** | KEEP |
| `bold_verify/top10/100000` `zero_hit` | `13f1b0153f5adec9` | 36 us | 24 us | **0.667** | KEEP, tail regressed |

**Scope:** this fixes the BOLD evidence path for lexical-saturated and natural-language
short-circuit rows. It does not establish universal dominance over Tantivy/Lucene/Meilisearch-class
BM25; the 10k short-keyword/high-fanout/zero-hit rows, 100k quoted phrase, and `limit_all` remain
negative or noisy and are recorded in `docs/NEGATIVE_EVIDENCE.md`.

| Date | Crate | Lever | Workload (bench id) | Before | After | Ratio | Status |
|------|-------|-------|---------------------|--------|-------|-------|--------|
| 2026-06-25 | frankensearch-core | **ASCII fast-path for NFC canonicalization** (analyzer hot path) | `nfc/ascii_short` | 1.207 µs | 26.7 ns | **0.022 (~45×)** | KEEP (`9d7e8d0`) |
| 2026-06-25 | frankensearch-core | **ASCII fast-path for NFC canonicalization** (analyzer hot path) | `nfc/ascii_doc` (2.2 KB) | 56.35 µs | 153 ns | **0.0027 (~368×)** | KEEP (`9d7e8d0`) |
| 2026-06-25 | frankensearch-core | ASCII fast-path for NFC canonicalization | `nfc/non_ascii` | 40.86 µs | 41.32 µs | 1.011 (neutral) | KEEP |
| 2026-06-25 | frankensearch-core | **`filter_low_signal`: ASCII compare instead of full-doc `to_lowercase`** | `filter_low_signal/ascii_doc` (2.25 KB) | 137.8 ns | 9.2 ns | **0.067 (~15×)** | KEEP |
| 2026-06-25 | frankensearch-core | `filter_low_signal`: ASCII compare instead of full-doc `to_lowercase` | `filter_low_signal/ascii_short` | 16.7 ns | 8.2 ns | **0.49 (~2×)** | KEEP |
| 2026-06-25 | frankensearch-core | **`strip_markdown_line`: skip the 4-pass inline replace chain for lines with no inline-markdown chars** | `strip_markdown_inline` (80 plain lines) | 10.17 µs | 4.41 µs | **0.434 (~2.3×)** | KEEP |
| 2026-06-25 | frankensearch-core | `strip_markdown_line`: chain `#`/`>` prefix trims as one `&str` (1 alloc, not 2) | `prefix_trim` (80 lines) | 2.45 µs | 1.34 µs | **0.549 (~1.8×)** | KEEP |
| 2026-06-25 | frankensearch-core | **canonicalize tail: drop the 2nd whole-document copy** (`filter_low_signal`→`is_low_signal` bool predicate; pass the owned `ws_normalized` buffer straight to `truncate_to_chars`) | `pipeline_tail` (2.25 KB ascii_doc) | 91.4 ns | 59.4 ns | **0.649 (~1.54×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`strip_markdown_line` fast path: trim the borrowed line directly** (drop the `line.to_string()` copy; only `strip_list_marker` allocates) | `strip_markdown_fastpath` (80 plain lines) | 2.328 µs | 1.306 µs | **0.561 (~1.78×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`strip_markdown_line` → `Cow<str>`: zero-alloc plain lines** (`strip_list_marker` returns borrowed slices; plain line borrows straight to the caller's single `push_str`) | `strip_markdown_cow` (80 plain lines, full push_str loop) | 2.166 µs | 1.390 µs | **0.642 (~1.56×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`QueryClass::classify`: compute `has_whitespace` once + `rsplit_once`** (was up to 4× whitespace rescans + a `rsplitn().collect()` Vec, per query) | `query_class` (11-query mix) | 732.4 ns | 512.4 ns | **0.700 (~1.43×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`strip_italic_underscores`: single pass, 3 allocs → 1** (drop the `Vec<char>` + `Vec<bool>` + final `collect`; build output directly with `prev`/`peek`) | `strip_italic_underscores` (40 snake_case code lines) | 9.770 µs | 2.982 µs | **0.305 (~3.28×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`strip_markdown_line` slow path: guard each transform by its trigger char** (one scan sets `*`/`_`/`` ` ``/`[` flags; a snake_case line skips the `**`/`*`/`` ` ``/link no-op allocating passes) | `strip_markdown_slowpath` (snake_case line) | 307.9 ns | 48.3 ns | **0.157 (~6.37×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`nfc_normalize` → `Cow<str>`: ASCII borrows instead of `to_owned`** (eliminates the whole-document copy at the `canonicalize` entry; ASCII is already NFC and the next stage only needs `&str`) | `nfc_ascii_copy` (2.25 KB ascii_doc) | 39.45 ns | 1.17 ns | **0.030 (~33×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-core | **`apply_hash_votes` branchless SimHash vote** (`2*b-1` instead of a per-bit `if`/`else` on random hash bits → no ~50% branch mispredict) | `simhash_votes` (~300-token doc) | 18.480 µs | 16.073 µs | **0.870 (~1.15×)** | superseded by table |
| 2026-06-26 | frankensearch-core | **`apply_hash_votes` table-driven SimHash vote** (8 byte-indexed lookups into a compile-time `[[i32;8];256]` + vectorizable 8-wide slice adds, vs 64 per-bit `shift/mask/mul`) | `simhash_votes` (~300-token doc) | 15.810 µs (branchless) | 10.930 µs | **0.691 (~1.45×) vs branchless, ~0.60× vs original branch** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-core | **`ParsedQuery::parse` no-negation fast path** (queries without `-`/`"`/`\` skip the `Vec<char>` + char-by-char parse; whitespace-normalize via split + `push_str`) | `parsed_query` (plain multi-word query) | 503.4 ns | 109.6 ns | **0.218 (~4.59×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-embed | **hash embedder: drop 2 per-embed allocs** (lazy `tokenize` iterator + `l2_normalize_in_place` on the owned accumulator) | `hash_embed_fnv` (~100-word doc, dim384) | 2.318 µs | 1.961 µs | **0.846 (~1.18×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-embed | hash embedder alloc elision — JL path (compute-bound, alloc negligible) | `hash_embed_jl` (~100-word doc, dim384) | 100.27 µs | 102.07 µs | 1.018 (neutral) | KEEP (no regression) |
| 2026-06-27 | frankensearch-embed | **JL projection: interleave 4 independent xorshift token chains** — the JL path was still latency-bound after allocation elision because each token's xorshift recurrence serializes three shift/xor steps per dimension. Buffering four token-chain seeds and advancing them together exposes instruction-level parallelism while preserving the exact scalar output: each dimension accumulator is an exact small integer-valued `f32` sum of +/-1 signs before normalization. Verified by new bit-identical scalar-reference tests across dimensions, seeds, and token-count tails. The requested literal `cargo bench --release` form was tried first and Cargo rejected it; the successful final per-crate run used `--profile release` through `rch exec` with `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a` (local fallback because no RCH worker was admissible). | `hash_embed_jl` (~100-word doc, dim384) | 105.00 µs | 88.945 µs | **0.847 (~1.18×)** | KEEP (BlackThrush) |
| 2026-06-25 | frankensearch-fusion | **RRF fuse: one `entry` lookup instead of `get`+`entry`** (halve per-candidate hashing of the `AHashMap<&str,_>` accumulator) | `rrf_fuse` (1000 lexical + 1000 semantic, ~50% overlap) | 29.11 µs | 23.07 µs | **0.793 (~1.26×)** | KEEP (BlackThrush) |
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators | `dot/dim256/f32_bytes` | 10.839 ms | 3.647 ms | **0.336** | KEEP |
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators | `dot/dim384/f32_bytes` | 14.084 ms | 5.333 ms | **0.379** | KEEP |
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators (`BlueGull` pinned-worker confirmation) | `dot/dim256/f32_bytes/10000` | 3.4835 ms | 0.66126 ms | **0.190** | KEEP (`vmi1149989`) |
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators (`BlueGull` pinned-worker confirmation) | `dot/dim384/f32_bytes/10000` | 5.1487 ms | 1.8811 ms | **0.365** | KEEP (`vmi1149989`) |
| 2026-06-26 | frankensearch-index | **`dot_product_f32_f32`: 4 independent `f32x8` accumulators** over 32-element blocks (plain f32 slice path; cheap decode, real callers in WAL-entry scoring and MRL truncated search). Existing tests allow tiny f32-order differences; SIMD gate passed. BlackThrush landed after remote same-process A/B superseded the older local-fallback negative row. | `dot/dim256/f32_slice/10000` (remote `vmi1227854`, in-process vs single-acc bench baseline) | 2.3594 ms | 750.64 µs | **0.318 (~3.14×)** | KEEP |
| 2026-06-26 | frankensearch-index | `dot_product_f32_f32` 4 accumulators (MiniLM/quality dim) | `dot/dim384/f32_slice/10000` (remote `vmi1227854`) | 3.7372 ms | 2.2784 ms | **0.610 (~1.64×)** | KEEP |
| 2026-06-24 | frankensearch-index | **branchless SIMD f16→f32 widen** (default path) | `dot/dim256/f16_bytes` | 4.733 ms | 1.632 ms | **0.345** | KEEP |
| 2026-06-24 | frankensearch-index | **branchless SIMD f16→f32 widen** (default path) | `dot/dim384/f16_bytes` | 7.363 ms | 2.332 ms | **0.317** | KEEP |
| 2026-06-24 | frankensearch-index | branchless SIMD f16→f32 widen | `dot/dim256/f16_slice` | 3.699 ms | 1.348 ms | **0.364** | KEEP |
| 2026-06-24 | frankensearch-index | branchless SIMD f16→f32 widen | `dot/dim384/f16_slice` | 5.536 ms | 2.181 ms | **0.394** | KEEP |
| 2026-06-26 | frankensearch-index | **in-memory filtered scan: precomputed-hash prescreen** (`matches_doc_id_hash` with a lazy `doc_id_hashes` slab) instead of re-hashing each `doc_id` string per vector via `matches()` — matches the FSVI scan, which already did this | `filter_prescreen` (10k `BitsetFilter` checks) | 183.5 µs | 88.4 µs | **0.482 (~2.08×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-index | **`dot_i8_i8`: 4 independent `i32x8` accumulators** (int8 two-pass pass-1) — the i8→i16 decode is a cheap sign-extend, so the kernel is sum-chain-bound (unlike the decode-bound f16 dot, where 4 accs regress — see NEGATIVE_EVIDENCE). Integer sum is associative → bit-identical. Landed as `957d608`+`28aa022`; ledger row added by BlackThrush | `dot_dim384/i8_dot` (10k, in-process vs single-acc baseline) | 481.3 µs | 412.0 µs | **0.856 (~1.17×)** | KEEP |
| 2026-06-26 | frankensearch-index | `dot_i8_i8` 4 accumulators (smaller dim) | `dot_dim256/i8_dot` (10k) | 318.2 µs | 298.8 µs | **0.939 (~1.06×)** | KEEP |
| 2026-06-26 | frankensearch-index | **`search_top_k_int8_two_pass_filtered`** — extended the int8 ADC two-pass to accept a `SearchFilter` (pass-1 pre-screens by the precomputed `doc_id` hash). The fast int8 path was previously filter-incapable, so filtered large-N vector search fell back to the exact scan; now it gets the int8 speedup. Bit-identical to the exact filtered top-k (`int8_two_pass_filtered_matches_exact_filtered`) | `int8_two_pass` filtered (10k clustered, `BitsetFilter` ~½ corpus, mult=5): `flat_filtered` vs `int8_filtered_mult5` | 314.4 µs | 248.7 µs | **0.791 (~1.26×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-index | **int8 two-pass pass-1 cutoff fast-path** — the int8 pass-1 called `insert_candidate` for *every* vector; the exact `scan_range` skips it for sub-cutoff scores (`heap.len() < cap \|\| score_key(score) >= cutoff`). Added the same guard to pass-1 (track the bounded-heap min as cutoff). Result is bit-identical (a sub-cutoff score never enters the full heap; `int8_two_pass_matches_exact_topk` + filtered test pass). The selection overhead was large — without the cutoff, mult=10 (2278 µs) was **slower than the exact flat scan** (~2131 µs); the cutoff restores the int8 win. In-process A/B (100k clustered, cutoff vs no-cutoff baseline) | `int8_two_pass` mult=5 / mult=10 | 1170.4 / 2278.6 µs | 900.9 / 1140.2 µs | **0.770 (~1.30×) / 0.500 (~2.00×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-index | **MRL truncated-scan cutoff fast-path** — same fix applied to `mrl_truncated_scan`: it called `insert_mrl_candidate` for every vector; added the bounded-heap cutoff guard (`heap.len() < limit \|\| nan_safe(score) >= cutoff`) the exact scan + int8 pass-1 use. Bit-identical (the 46 `mrl::` tests pass unchanged). Smaller win than the int8 pass-1 (the MRL scan is single-threaded with a cheap 64-dim truncated dot, so `insert` is a smaller fraction). In-process A/B (100k FSVI F16, clustered, search_dims=64, limit=30): cutoff vs no-cutoff baseline | `mrl_cutoff` scan | 23056.9 µs | 21045.9 µs | **0.913 (~1.10×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-index | **MRL truncated-scan parallelization** — `mrl_truncated_scan` was **single-threaded** (a plain `for index in 0..record_count()` byte-walk), while the exact `scan_range` + int8 pass-1 are rayon-parallel. Refactored the byte-walk into a per-chunk `mrl_scan_chunk` (fn-pointer dot collapses the F16/F32 arms) and scan disjoint record ranges in parallel above `PARALLEL_THRESHOLD`, merging per-chunk bounded heaps (`merge_mrl_partials`). Order-independent (score+index total order), so **bit-identical** to the sequential top-k — verified by a new `mrl_search_parallel_path_covers_all_records` test (12k records, rescore-all == exact `search_top_k`) + all 47 `mrl::` tests. In-process A/B (100k FSVI F16, clustered, search_dims=64, limit=30): parallel vs forced-sequential | `mrl_parallel` scan | 5828.5 µs | 674.9 µs | **0.116 (~8.64×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-fusion | **`SyncTwoTierSearcher` fast-tier fetch → int8 two-pass (default path)** — the hybrid's vector candidate fetch (`search_fast_hits`) called the exact f16 `search_top_k`. The fast tier is a *reranked candidate generator* (quality-tier + RRF re-score its hits), so its contract is approximate candidates; routed the **default path** (no explicit `SearchParams`) to `search_top_k_int8_two_pass_filtered` (parallel + cutoff + filter, mult=10, recall=1.0). **Wires the validated int8 two-pass into a real production hybrid path** (it was previously unused by callers). Lossless → identical fused top-k (6 sync_searcher + 9 integration tests pass); explicit `SearchParams` still uses the exact scan. End-to-end hybrid A/B (100k in-memory two-tier, no lexical): exact-fetch vs int8-fetch | `sync_int8_fetch` | 6449.6 µs | 5498.6 µs | **0.853 (~1.17×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-index | **`InMemoryTwoTierIndex` quality rerank: O(hits·N) → O(hits)** — `quality_scores_for_hits` (the two-tier quality rerank, on the wired sync-hybrid path) did `quality.doc_ids.iter().position(\|id\| id == hit.doc_id)` — a **linear O(N) doc-id scan per hit** ⇒ O(hits·N) (≈3M string-compares at N=100k, hits=30). Replaced with a lazy `doc_id → index` `HashMap` (O(1) lookup, first-insert-wins = same as `position`; built on first rerank, search-only callers pay nothing). Also fixed the same O(N) scan in `InMemoryVectorIndex`'s by-doc-id hit scorer. Bit-identical (19 in-memory + 15 sync_searcher tests). End-to-end hybrid before→after (same `sync_int8_fetch` bench, only the rerank changed; both arms drop by a constant ~3.7 ms, isolating the rerank): int8 5498.6→1755.0 µs **3.13×**, exact 6449.6→2637.5 µs **2.45×** | `sync_int8_fetch` | 5498.6 µs | 1755.0 µs | **0.319 (~3.13×)** | KEEP (BlackThrush) |
| 2026-06-26 | frankensearch-fusion | **int8 fast-tier `mult` 10 → 5** — the wired `INT8_FAST_TIER_MULT` was 10, the *slowest* int8 mult (int8_two_pass bench: mult=5 862 µs vs mult=10 1187 µs at 100k, both recall=1.0). Since `fetch` is already a candidate over-fetch (`k·candidate_multiplier`), a 10× int8 multiplier is wasteful; mult=5 keeps the candidate set lossless (recall=1.0 validated at 100k) with a higher pass-1 cutoff (fewer inserts) + smaller pass-2 rescore. Lossless (15 sync_searcher tests pass). Cumulative sync hybrid vs the original exact + linear-rerank: **4.38×** | `sync_int8_fetch` int8 arm | 1755.0 µs | 1472.9 µs | **0.839 (~1.19×)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-index | **FSVI int8 two-pass for standalone large-N vector search** — new `VectorIndex::search_top_k_int8_two_pass`: a parallel int8 pass-1 over all main records (a lazily-built corpus-quantized i8 slab from the contiguous F16 region; tombstone-aware flag check + the same cutoff fast-path as the exact scan) keeps the top `k·mult` by approximate score, then an exact f16 rescore of just those candidates selects the final top-k. The file-backed twin of the in-memory two-pass; falls back to the exact `search_top_k` for WAL/non-F16 indexes (never silently degraded). **Lossless: recall@10 = 1.0000 at mult=2/3/5/10 on 100k clustered**, and bit-identical to `search_top_k` when the candidate set retains every record — verified by `int8_two_pass_keep_all_matches_exact` + 53 search tests. NOT wired into the BOLD hybrid (that gap is not vector-bound — see docs/NEGATIVE_EVIDENCE.md); targets pure vector-search latency. A/B (100k file-backed FSVI, in-process, same index): exact flat vs int8 mult=5 | `fsvi_int8_two_pass` | 14349.0 µs | 7408.2 µs | **0.516 (~1.94×)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-index | **FSVI 4-bit (16-level) two-pass — fastest lossless vector-search primitive** — new `VectorIndex::search_top_k_4bit_two_pass` + a fused SIMD nibble kernel `dot_packed_4bit` (load 16 packed bytes → `i16x16`, extract signed nibbles via arithmetic `(x<<12)>>12`/`(x<<8)>>12`, low·low + high·high). Packs each vector to signed 4-bit nibbles (`dim/2` bytes/vector — **half the int8 slab**), parallel pass-1 + exact f16 rescore of the top `k·mult`. **Lossless: recall@10 = 1.0000 at mult=5 on 100k clustered** (0.96 at mult=2); kernel verified against a scalar reference (`dot_packed_4bit_matches_scalar`) + bit-identical to `search_top_k` under keep-all (`four_bit_two_pass_keep_all_matches_exact`). The nibble-unpack compute offsets some of the 2× bandwidth saving, so the net vs int8 is modest, but 4-bit@mult5 is the **fastest lossless** option (mult=10 is slower → mult=5 is the sweet spot). Falls back to exact for WAL/non-F16; not wired into BOLD (not vector-bound). A/B (100k file-backed FSVI, in-process, same index; flat 2132.4 µs, int8 mult=5 888.2 µs): exact flat vs 4-bit mult=5 → **2.56×**; vs int8 mult=5 → **1.07×** | `fsvi_4bit_two_pass` | 888.2 µs | 831.4 µs | **0.936 (1.07× vs int8; 2.56× vs flat)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-index | **`dot_packed_4bit` kernel: vertical accumulation (1.07× → 1.36× vs int8)** — the 4-bit pass-1 is **compute-bound** (the parallel scan is not bandwidth-limited), and the kernel did a horizontal `prod.reduce_add()` **per 16-byte chunk** (12 reduces/vector at dim=384) — the dominant cost. Replaced with a single `i16x16` vertical accumulator (`acc += s_low·q_low + s_high·q_high`), flushed to the i32 sum every 16 chunks before any lane can exceed `i16` (16·98=1568/lane; 16-lane reduce 25088 < 32767). **Bit-identical** (scalar-match + keep-all tests pass; recall@10 = 1.0000 at mult=5 unchanged) — a pure speedup. A/B (100k file-backed FSVI, in-process, same run): 4-bit mult=5 vs int8 mult=5 | `fsvi_4bit_two_pass` | 1034.5 µs | 762.3 µs | **0.737 (1.36× vs int8; 3.22× vs flat)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-index | **In-memory 4-bit two-pass (in-memory twin of the FSVI 4-bit)** — `InMemoryVectorIndex::search_top_k_4bit_two_pass`: packed signed-4-bit slab (`dim/2` bytes/vector, half the int8 slab) + the fused `dot_packed_4bit` pass-1 (parallel, cutoff fast-path) + exact f16 rescore — the in-memory backend twin of the FSVI 4-bit, reusing the verified vertical-accumulation kernel. **Lossless: recall@10 = 1.0000 at mult=5** (0.98 at mult=2), 10k clustered; bit-identical under keep-all (`four_bit_two_pass_keep_all_matches_exact`). A/B (10k in-memory, in-process, same run; flat 352.9 µs, int8 mult=5 160.2 µs): 4-bit mult=5 vs int8 mult=5 → **1.40×**; vs flat → **3.09×** — the fastest lossless in-memory vector-search primitive. (Wiring it into the `SyncTwoTierSearcher` fast tier, which currently uses the int8 two-pass, is the follow-up: needs a `_filtered` variant.) | `int8_two_pass` 4-bit arm | 160.2 µs | 114.2 µs | **0.713 (1.40× vs int8; 3.09× vs flat)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-fusion | **`SyncTwoTierSearcher` fast tier wired int8 → 4-bit two-pass (deployment)** — `search_fast_hits` default path now routes to `search_top_k_4bit_two_pass_filtered` (added a `_filtered` 4-bit variant with the same doc_id-hash pre-screen as int8) instead of the int8 two-pass — deploying the fastest lossless primitive into the sync production hybrid. The fast tier is a reranked candidate generator, so the lossless 4-bit candidate set yields **identical fused top-k**: 6 sync_searcher lib + 9 integration tests pass unchanged. End-to-end same-run A/B (100k in-memory two-tier, no lexical): exact-fetch hybrid vs 4-bit-fetch hybrid → **2.36×** (up from the int8 wiring's 1.79× vs exact; the fast-tier fetch itself is 1.40× faster than int8, measured). Explicit `SearchParams` still uses the exact scan. | `sync_int8_fetch` | 2880.2 µs | 1219.8 µs | **0.424 (2.36× vs exact)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-fusion | **`SyncTwoTierSearcher` 4-bit fast-tier multiplier 5 → 3** — the default sync fast tier already over-fetches candidates before quality rerank/RRF, so the extra 4-bit pass-2 rescore from `mult=5` was wasteful. Tightened the candidate multiplier to 3 and added a deterministic clustered conformance test that compares the default 4-bit path against explicit exact scan top-k IDs for 12 queries. Remote same-worker proof on `ovh-a`, target dir `/data/projects/.rch-targets/frankensearch-cod-b`: baseline source restored to `FAST_TIER_MULT=5`, then candidate `FAST_TIER_MULT=3`, same command `cargo bench -p frankensearch-fusion --profile release --bench sync_int8_fetch -- --sample-size 10 --warm-up-time 1 --measurement-time 2`. Exact arm stayed neutral (2.3088 ms → 2.3120 ms). | `sync_int8_fetch/fast_fetch_4bit` | 769.95 µs | 639.75 µs | **0.831 (~1.20×)** | KEEP (BlackThrush) |
| 2026-06-27 | frankensearch-index | **Prepared 4-bit query lanes for repeated scan dots** — `prepare_4bit_query` decodes the query nibbles once into signed SIMD lane vectors, and `dot_4bit_prepared` reuses those lanes for every stored vector. This lands the alien-graveyard/vectorized-execution lever as a pure register-reuse win on the already-shipped 4-bit scan path: old path decoded both stored and query nibbles per dot; new path decodes only stored nibbles inside the hot loop. Behavior preserved by `dot_packed_4bit_matches_scalar` now checking both packed and prepared kernels, including the `0x99` signed-nibble extreme; the prepared loop also truncates to the shorter stored/query chunk stream instead of panicking on public mismatched slices. Remote RCH worker `hz2`, target dir `/data/projects/.rch-targets/frankensearch-cod-b`, command `cargo bench -p frankensearch-index --profile release --bench dot_product fourbit_prepared -- --sample-size 10 --warm-up-time 1 --measurement-time 3`. | `dot/dim256/fourbit_prepared/10000`; `dot/dim384/fourbit_prepared/10000` | 459.40 µs; 567.38 µs | 389.81 µs; 509.84 µs | **0.849 (~1.18×); 0.899 (~1.11×)** | KEEP (Codex) |

**Lever (ParsedQuery no-negation fast path):** `ParsedQuery::parse` runs per search query (the
searcher parses for `-term`/`NOT "phrase"` negations). The committed parser always materialized a
`Vec<char>` and re-collected each word via `chars[a..b].iter().collect()`. Negation syntax requires
one of `-` `"` `\`; with none present (the common query) there are no negations, so the positive
part is just the whitespace-normalized input. The new fast path returns it directly
(`split_whitespace` + `push_str` into one buffer), skipping the char materialization. Byte-identical
(the full parser collects the same whitespace-split words and `join(" ")`s them; 42 parsed_query
tests green). Measured on a plain multi-word query (`parsed_query`): 503.4 ns → 109.6 ns, **0.218
(~4.59×)**. Queries that *do* use negation syntax still take the exact char-based path.

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench parsed_query
```

**Lever (SimHash table-driven vote):** building on the branchless vote, `apply_hash_votes` now
processes the window hash **one byte at a time**: each byte (0..256) indexes a compile-time
`VOTE_TABLE: [[i32;8];256]` (`2*((b>>k)&1)-1`) for its 8 ±1 votes, added to the matching 8 bit
counters as a slice — the compiler vectorizes the 8-wide `i32` add (SSE2 `i32x4`). This replaces the
64 per-bit `shift/mask/mul/sub` of the branchless form. Bit-identical (byte `j`'s bit `k` is hash bit
`8j+k`; 20 fingerprint tests + 895 core lib tests green). Measured (`simhash_votes`): branchless
15.810 µs → table 10.930 µs, **0.691 (~1.45×)**, i.e. ~0.60× vs the original conditional. The table
is built `const` (zero runtime/init cost).

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench fingerprint
```

**Lever (SimHash branchless vote):** `Fingerprint::compute` builds a 64-bit SimHash over 3-token
shingles (per document, for incremental re-embedding decisions). The inner `apply_hash_votes`
accumulated ±1 into 64 bit counters per window via `if (bit set) { +1 } else { -1 }` — a
data-dependent branch on effectively-random hash bits (~50% misprediction). Replaced with the
branchless `*weight += 2*((hash>>bit)&1) - 1`. Bit-identical (`semantic_simhash` unchanged; 20
fingerprint tests green incl. `identical_text_produces_identical_fingerprint`,
`large_document_computes_without_overflow`). Measured over a ~300-token doc's shingle sweep
(`simhash_votes`): 18.480 µs → 16.073 µs, **0.870 (~1.15×)** — confirming LLVM did *not* already
emit a branchless form (measured rather than assumed). Determinism-safe: same vote, same final
`weight > 0` test.

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench fingerprint
```

**Lever (nfc_normalize → Cow):** `canonicalize`/`canonicalize_query` begin with NFC normalization.
ASCII text is already NFC, so the prior code returned `text.to_owned()` — a whole-document copy —
which then fed `strip_markdown_and_code` (which allocates its own buffer) and the trim path. Since
those stages only need a `&str`, `nfc_normalize` now returns `Cow<str>`: `Cow::Borrowed(text)` for
ASCII (zero copy), `Cow::Owned` for non-ASCII (the `nfc()` collect, unchanged). Byte-identical (the
`&str` view is identical; 34 canonicalize + 895 core lib tests green incl.
`nfc_normalize_ascii_fast_path_matches_reference`). Measured borrow vs `to_owned` on a 2.25 KB ascii
doc (`nfc_ascii_copy`): 39.45 ns → 1.17 ns, **0.030 (~33×)**. Stacks on the earlier NFC ASCII
fast-path (which had already replaced the `nfc()` state machine with `to_owned`); this removes the
remaining copy on the dominant indexing path's entry.

**Lever (strip_markdown_line slow-path guards):** the inline-markdown slow path previously ran the
**entire** transform chain — `replace("**")`, `replace("__")`, `replace('*')`,
`strip_italic_underscores`, `replace('`')`, `strip_markdown_links` — unconditionally, allocating a
whole-line copy at each step even when the relevant char was absent. A `snake_case` line (trigger =
`_` only) thus paid four no-op allocating passes (`**`, `*`, `` ` ``, links). Now one byte scan sets
`has_star`/`has_underscore`/`has_backtick`/`has_bracket` and each transform is guarded by its
trigger char (same order → byte-identical; skipping an absent-trigger transform is the same as the
no-op copy it produced). 34 canonicalize tests green (bold/italic, links, backticks, nested,
headings). Measured on a snake_case line (`strip_markdown_slowpath`, `strip_italic_underscores`
omitted from the A/B since it runs in both): 307.9 ns → 48.3 ns, **0.157 (~6.37×)**. Stacks with
the single-pass `strip_italic_underscores` lever for the dominant code-search line shape.

**Lever (strip_italic_underscores single pass):** this runs on the inline-markdown path of
`strip_markdown_line`, which is hit by **every line containing `_`** — i.e. every `snake_case`
identifier line, the common case in code search (where almost all underscores are kept). The prior
implementation materialized a `Vec<char>`, a `Vec<bool>` keep-mask, and a final `collect` — three
allocations and three passes. Rewritten as a single pass that builds the output `String` directly,
tracking the previous source char and peeking the next to apply the exact same word-boundary test
(`prev`/`next` is-word). Byte-identical — proven by `strip_italic_underscores_matches_reference`
(snake_case, `_italic_`, mixed, leading/trailing, non-ASCII word chars) + 33 canonicalize tests.
Measured over 40 snake_case code lines (`strip_italic_underscores`): 9.770 µs → 2.982 µs, **0.305
(~3.28×)**.

**Lever (QueryClass::classify):** `classify` runs on every search query (adaptive lexical/semantic
budget + the lexical short-circuit gate both call it). `looks_like_identifier` rescanned the query
with `chars().any(char::is_whitespace)` up to four times and allocated a `Vec` via
`rsplitn(2,'-').collect()` for the issue-ID check. Now whitespace is computed once and all the
no-whitespace heuristics are grouped under it (so whitespace queries skip the whole block after a
single early-stopping scan), and the issue-ID check uses allocation-free `rsplit_once('-')`.
Behaviour-identical (26 query_class tests + 894 core lib tests green; grouping is safe because every
branch returns `true`). Measured over an 11-query identifier/keyword/natural-language mix
(`query_class`, real `classify` as `new` vs the prior logic replicated as `old`): 732.4 ns →
512.4 ns, **0.700 (~1.43×)**.

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench query_class
```

**Lever (strip_markdown_line → Cow):** building on the fast-path lever above, the strip pipeline
now returns `Cow<str>`: `strip_list_marker` yields borrowed slices in every branch (unordered →
`strip_prefix` slice, ordered → `&trimmed[digit_count+2..]` since the marker is single-byte ASCII,
not-a-marker → the line), and `strip_markdown_line`'s fast path returns that borrowed `Cow`. So a
plain line (no inline markdown — prose, most code comments) flows through `strip_markdown_and_code`
with **zero per-line allocation**: the bytes are copied once by the existing `result.push_str`. The
inline-markdown slow path still returns `Cow::Owned`. Byte-identical (894 core lib tests green incl.
ordered/unordered list markers, `numbers_not_list_markers_preserved`, headings, blockquotes).
Measured over the full 80-plain-line `push_str` loop (`strip_markdown_cow`: `string` = prior
per-line owned String vs `cow` = borrowed): 2.166 µs → 1.390 µs, **0.642 (~1.56×)**, on top of the
fast-path win.

**Lever (strip_markdown_line fast path):** `strip_markdown_line` runs on every line of every
document at index time, and markdown stripping is the dominant `canonicalize` cost. The fast path
(lines with no inline-markdown chars — plain prose, most code-comment text) copied the whole line
with `line.to_string()` before the header/blockquote prefix trims, then `strip_list_marker`
allocated the final owned String — two allocations per line. Since the prefix trims and
`strip_list_marker` only need `&str`, the fast path now trims the borrowed `line` directly and only
`strip_list_marker` allocates (one allocation). Byte-identical (same trims in the same order; 33
canonicalize tests green incl. headings/blockquotes/list-markers/links/bold-italic). Measured
head-to-head in one process over 80 plain lines (`strip_markdown_fastpath`, `strip_list_marker`'s
final owned-String step modeled): 2.328 µs → 1.306 µs, **0.561 (~1.78×)**.

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench canonicalize
```

**Lever (RRF fuse single lookup):** `rrf_fuse_with_graph` accumulates per-doc scores in an
`AHashMap<&str, FusedHitScratch>` across the lexical, semantic, and graph candidate lists. Each
loop did a `hits.get(doc_id)` (per-source dedup probe) **then** a `hits.entry(doc_id)`
(insert/update) — two hashes of the same key per candidate, where the `get` almost never skips
(doc ids are unique within a source). Consolidated to a single `entry` match: `Occupied` keeps the
dedup-skip guard then updates via `get_mut`, `Vacant` inserts — semantically identical (47 rrf
tests + 816 fusion lib tests green), one hash lookup instead of two. Measured head-to-head in one
process (`benches/rrf_fuse.rs`, 1000 lexical + 1000 semantic, ~50% overlap; the private
`FusedHitScratch` is replicated with an equivalently-shaped struct): 29.11 µs → 23.07 µs, **0.793
(~1.26×)**. RRF fusion is a per-query cost on the hybrid path, so this directly trims hybrid search
latency. Frankensearch pre-change before/after ratio only (`bd-ui41`).

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-fusion --bench rrf_fuse
```

**Lever (hash embedder alloc elision):** `HashEmbedder::embed_sync` runs on every document at
index time and every query (the non-semantic `fnv1a-*`/`jl-*` fast tiers, and the frankensearch
side of the BOLD hash-hybrid comparison). The committed path did two dimension-sized allocations
per embed: `tokenize` collected a `Vec<&str>`, and `l2_normalize(&embedding)` returned a freshly
allocated `Vec<f32>`. Now `tokenize` returns a lazy iterator (each embedder consumes tokens exactly
once) and the new `frankensearch_core::traits::l2_normalize_in_place` normalizes the owned
accumulator in place — so the only allocation is the accumulator itself. Bit-identical output
(`l2_normalize_in_place_matches_allocating` proves the in-place form equals `l2_normalize` across
zero/near-zero/non-finite cases; 286 embed lib tests + the existing `output_is_l2_normalized` /
`jl_output_is_l2_normalized` green). Measured head-to-head in one process (old 2-alloc vs new
1-alloc, replicated in `benches/hash_embed.rs`): FNV path **0.846 (~1.18×)**; the JL path is
compute-bound (O(tokens·dim) xorshift), so its alloc savings vanish into noise (1.018, neutral —
never a regression). Frankensearch pre-change before/after ratio only (no dominance-vs-original
claim; `bd-ui41`).

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-embed --bench hash_embed
```

**Lever (canonicalize tail):** `DefaultCanonicalizer::canonicalize` runs on every document at
index time and every query. The committed tail did two whole-document allocations — the old
`filter_low_signal` returned `text.to_string()` on the common (non-ack) path, then
`truncate_to_chars` copied again. Converting the filter to an allocation-free `is_low_signal(&str)
-> bool` predicate (early-return `String::new()` for acks) lets `canonicalize` pass its already-owned
`ws_normalized` buffer straight to truncation — one copy, not two. Output is byte-identical (acks →
`""`, normal docs → same bytes; 893/893 `frankensearch-core` lib tests green, incl. all 33
canonicalize tests). Measured head-to-head in one process (`pipeline_tail/old` = 2 copies vs
`/new` = 1 copy, 2.25 KB ascii doc): 91.4 ns → 59.4 ns, **0.649 (~1.54×)**. Frankensearch
pre-change before/after ratio only (no dominance-vs-original claim; blocked by `bd-ui41`).

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench canonicalize
```

**Lever (in-memory filtered-scan prescreen):** the FSVI on-disk scan (`search.rs`) already
pre-screens filters with a **precomputed `doc_id_hash`** via `SearchFilter::matches_doc_id_hash`,
but `InMemoryVectorIndex::scan_range` called plain `matches(doc_id)` — which for a `BitsetFilter`
**re-hashes the doc_id string every vector** (`fnv1a_hash` per element). In a selective filtered
scan the filter check is the dominant per-(excluded-)vector cost, so this was a real waste. Added a
lazy `doc_id_hashes: OnceLock<Vec<u64>>` (built on first *filtered* search with the same
`frankensearch_core::filter::fnv1a_hash` that `BitsetFilter` uses — so it's bit-correct;
exact/unfiltered callers pay nothing) and switched the scan to `matches_doc_id_hash(precomputed)`
with a fallback to `matches()` for filters that can't decide by hash (e.g. metadata filters).
Behaviour-identical (17 in-memory + 15 filter tests + new `search_with_bitset_filter_uses_precomputed_hash_path`
+ 357 index lib serial all green). Measured `matches` re-hash vs `matches_doc_id_hash` precomputed
over 10k `BitsetFilter` checks (`benches/filter_prescreen.rs`): 183.5 µs → 88.4 µs, **0.482
(~2.08×)** — the per-vector filter-check cost the in-memory filtered scan now saves, bringing it to
parity with the FSVI scan.

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p frankensearch-core --bench filter_prescreen
```

**Lever (bd-gfzk):** `dot_product_f16_bytes_f32` / `dot_product_f16_f32` — the **default
quantization** path (f16 FSVI, `search.rs:288,346,441`). The f16 paths were decode-bound: the
per-element scalar `f16::to_f32()` dominated (confirmed in `docs/NEGATIVE_EVIDENCE.md` — that's
why accumulator unrolling failed here). Replaced the 8× scalar decode with a **branchless SIMD
widen** (Giesen magic-multiply over `wide::u32x8`/`f32x8` + pure-integer inf/nan fixup) →
**~2.5–3.2× faster** on the dominant path. The widen is **bit-exact** to `f16::to_f32()` for all
65 536 f16 values (exhaustive test `simd_f16_widen_is_bit_exact`), and the single accumulator /
lane assignment is unchanged, so every dot score is **bit-identical** on finite data → no
determinism/golden risk (all 19 simd + full index lib tests green). Measured head-to-head in one
process (`f16_*_new` SIMD vs `f16_*_old` scalar). No dominance-vs-original claim (blocked by
`bd-ui41`); pre-change before/after ratio only.

_Refinement (same lever):_ the byte path also replaces the 8 scalar `from_le_bytes` + stack
round-trip with a single little-endian SIMD load (`[u8;16]`→`u16x8`→`u32x8` zero-extend, BE
fallback retained). Still bit-exact (`simd_f16_bytes_load_matches_scalar`). The in-process
`f16_bytes_new/old` ratio improved 0.345→0.317 (dim256) / 0.317→0.305 (dim384), ~4–8% on top of
the SIMD widen. Modest and not cleanly isolated from worker noise, but kept because one SIMD
load strictly dominates 8 scalar loads (worst case neutral, never a regression).

**Lever:** `dot_product_f32_bytes_f32` (used by f32-quantized FSVI indexes, `search.rs:307,372,492`).
The old kernel decoded f32s from an open-ended `&stored_bytes[off..]` slice, which the
compiler could not prove a fixed width for and so did not vectorize the `from_le_bytes`
decode. Rewriting the decode over fixed `[u8; 32]` sub-blocks (+ 4 independent f32x8
accumulators) lets it vectorize → **~2.7–3.0× faster**. Measured head-to-head in one
process via `benches/dot_product.rs` (`f32_bytes_new` vs `f32_bytes_old`), so the ratio is
immune to which rch worker the run landed on. f16 paths were tried with the same restructure
but **regressed** (decode is scalar-bound there) and were reverted — see
`docs/NEGATIVE_EVIDENCE.md`. No dominance-vs-original claim (blocked by `bd-ui41`); this is a
frankensearch pre-change before/after ratio only.

`BlueGull` confirmation command:
```bash
RCH_WORKER=vmi1149989 \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo bench -p frankensearch-index --profile release --bench dot_product
```
Same worker conformance gate:
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo test -p frankensearch-index simd --profile release
```
Result: 18 relevant SIMD tests passed; RCH reported `remote vmi1149989`.

## Foundational primitives & lever validations (not yet product wins)

These are measured de-risking results for levers that are not yet wired into the search
path. They are **not** search speedups — they tell us whether a bigger build is worth it.

### int8 dot kernel (`dot_i8_i8`) — validates the `bd-b5wl` ADC pass-1 premise

Added a tested, exported symmetric int8 dot primitive (`crates/frankensearch-index/src/simd.rs`,
SIMD via `wide::i16x8::mul_widen`; exhaustive-ish correctness test incl. the i32 overflow
worst case). Benched head-to-head against the optimized f16 dot **in the same process/run**:

| Workload | `i8_dot` (int8) | `f16_bytes_new` (optimized f16) | int8/f16 ratio |
|----------|-----------------|---------------------------------|----------------|
| `dot/dim256` (n=10k) | 368 µs | 1.113 ms | **0.331** (~3.0×) |
| `dot/dim384` (n=10k) | 505 µs | 1.622 ms | **0.311** (~3.2×) |

**Conclusion:** scoring all N with int8 is ~3× faster than the current exact f16 scan (½ the
bytes + integer `mul_widen` MAC, no f16 decode). This **validates building the int8 ADC two-pass
scan** (`bd-b5wl`): a pass-1 int8 sweep over all N + an exact f16 rescore of only `k·mult`
candidates should net ~2.5–3× on the vector scan. **Caveat:** this is a kernel-throughput ratio,
**not** a search speedup — the product win requires the sidecar + two-pass wiring + a recall@10
gate (it's approximate). The primitive is unused by the search path until that lands.

**Recall gate measured (`int8_two_pass_recall_at_10`):** int8 pass-1 (top `k·mult`) + exact f16
rescore recovers the true f16 top-10 with **recall@10 = 1.0000** at `mult=20` (top-200 of 3000
random L2-normalized vectors, averaged over 25 queries). So at a modest multiplier the two-pass
is **lossless** here. Both ADC gates now pass: ~3× pass-1 speed **and** recall = 1.0. **Caveat:**
random vectors have wide angular spread; real (clustered) embeddings may need a higher `mult` —
the production wiring must re-measure recall on a real corpus and tune `mult`/quant granularity.

**End-to-end search-level ratio (`topk_exact_f16` vs `topk_int8_2pass`, k=10, mult=20, n=10k):**
full top-10 pipeline (score all N + select), measured same-process:

| dim | exact f16 top-k | int8 two-pass top-k | ratio |
|-----|-----------------|---------------------|-------|
| 256 | 1.213 ms | 469 µs | **0.387** (~2.6×) |
| 384 | 1.794 ms | 607 µs | **0.339** (~3.0×) |

So the int8 ADC two-pass is **~2.6–3.0× faster on the actual top-k search workload with recall@10
= 1.0** (mult=20, random vectors; both pipelines pay a conservative N-element sort, so production
bounded-heap selection would only widen the gap). This is the search-level proof, not just a
kernel ratio.

> **CORRECTION (see `docs/NEGATIVE_EVIDENCE.md`):** the `topk_exact_f16` baseline above is a
> *serial full-sort* pipeline. The real product `search_top_k` is **rayon-parallel + bounded-heap
> + cutoff** and is far faster, so the ~2.6–3× "search win" holds only vs a *serial* exact — NOT
> the product's parallel exact. The kernel ~3× (`33fb45b`) and recall@10 = 1.0 stand; the
> integrated-method ratios vs parallel exact are below.

**Wired into the product (`InMemoryVectorIndex::search_top_k_int8_two_pass`):** the in-memory index
precomputes an int8 slab (one corpus-wide max-abs scale, ranking-preserving) at construction and
exposes an opt-in two-pass method — a **parallel bounded-heap** int8 pass-1 (top `limit·mult` per
chunk, `PARALLEL_CHUNK_SIZE` chunking to match the exact path's core use) → exact f16 rescore with
the *same* bounded-heap selection as the exact path. **Bit-identical** to `search_top_k` whenever
pass-1 recall is 1 (`int8_two_pass_matches_exact_topk`: identical doc-ids + scores), exact
fallback if the int8 slab is absent. Existing exact paths untouched; 355/355 index lib tests green.

**Real-method head-to-head (`inmem_topk`, top-10) vs the parallel exact `search_top_k`:**

The candidate budget `limit·mult` is the selection-overhead knob. `int8_two_pass_recall_at_10`
shows recall@10 = **1.0 down to mult=2** for random (well-separated) vectors, so the original
mult=20 was 10× more candidates than needed. Re-benched at **mult=5**:

| Workload | exact `search_top_k` | `int8_two_pass` mult=5 | ratio | (mult=20 ratio) |
|----------|----------------------|------------------------|-------|-----------------|
| dim256/n10k | 328 µs | 223 µs | **0.68** | 0.72–1.09 (worker-var) |
| dim384/n10k | 397 µs | 278 µs | **0.70** | 0.66 |
| dim384/**n100k** | 2.02 ms | 1.34 ms | **0.67** | 0.75 |

At mult=5 the win is a **robust ~1.4–1.5× across 10k AND 100k** (consistent on both fast and slow
workers), no longer the worker-dependent tie seen at mult=20. Smaller candidate budget = less
top-`k·mult` selection + merge overhead, so more of the int8 kernel's 3× survives. Still
bit-identical to exact when recall=1. **Honest scope:** ~1.4–1.5× (not the 2.6–3× serial-baseline
figure); recall=1.0 is for *random* vectors — clustered real embeddings may need a higher mult, so
the production path must re-measure recall on a representative corpus. **Remaining (`bd-t8tv`):**
the mmap FSVI `search.rs` path (on-disk int8 sidecar, where exact also pays page-faults + decode —
likely a larger win); real-corpus recall + mult tuning.

**2026-06-26 — clustered-data validation (answers the "needs higher mult on real data" caveat):**
re-ran on **clustered** vectors (64 centroids + noise — the realistic embedding shape, vs the prior
uniform-random) via the new gated-free bench `benches/int8_two_pass.rs`
(`cargo bench -p frankensearch-index --bench int8_two_pass`). Result at N=10000, dim=384, k=10:

| candidate_mult | recall@10 (clustered) | latency | ratio vs flat |
|----------------|-----------------------|---------|---------------|
| flat exact | 1.000 | 309.7 µs | 1.00× |
| 2 | **1.0000** | — | — |
| 5 | **1.0000** | 166.5 µs | **0.538 (1.86×)** |
| 10 | **1.0000** | 207.2 µs | **0.669 (1.50×)** |
| 20 | **1.0000** | — | — |

So on clustered data the two-pass is **lossless (recall@10 = 1.0000) down to mult=2** — *easier* than
random vectors (clear cluster winners always land in the `k·mult` candidate set, and the exact
rescore fixes the order). It is a **lossless ~1.5–1.86× speedup over the flat parallel-SIMD exact
scan** at this scale. This is the strongest vector lever found and is the **opposite of HNSW** (which
trades recall — see `docs/NEGATIVE_EVIDENCE.md`). **Recommendation:** use `search_top_k_int8_two_pass`
(mult≈10 for safety margin) for in-memory vector search; it is strictly better than flat exact on
realistic data. Not made the unconditional default because recall=1.0 is empirical (exact iff the
true top-k ⊆ the int8 candidate set) — safe as an opt-in / large-N path, with a recall gate for
pathological corpora.

_Scale confirmation — and the win GROWS with N:_ re-ran the same clustered bench at **N=100000**
(dim=384, k=10 — the BOLD / Tantivy-comparison corpus size): recall@10 = **1.0000 at every mult
(2, 5, 10, 20)** (lossless at production scale; int8 quantization error is scale-independent, so a
modest `k·mult` candidate set still always contains the true top-10). Measured **clustered** latency
(single run): flat 2131 µs vs int8 mult=5 **862 µs (2.47×)**, mult=10 1187 µs (1.80×) — i.e. the
relative speedup **grows from ~1.86× at 10k to ~2.47× at 100k**, because the int8 pass-1's halved
bytes + no-f16-decode advantage scales with N (consistent with the `bd-t8tv` "larger win at scale"
hypothesis). (The 10k mult sweep is pass-1-bound and worker-noisy — mult 2–10 all land ~1.3–1.6×
within noise; the clean separation appears at 100k.) So `search_top_k_int8_two_pass` (mult≈10) is a
**lossless 1.86–2.47× win across 10k–100k, strongest at production scale** — the highest-value
vector lever, pending the wiring (opt-in / large-N default with an exactness gate).

_Wiring target (verified 2026-06-26, BlackThrush):_ the validated two-pass is **in-tree but unused
by callers** — `InMemoryTwoTierIndex::search` (`two_tier.rs:350,353`) and the fusion searcher call
the **exact** `search_top_k` / `search_top_k_with_params`. The integration point is there: route
large-N searches to `search_top_k_int8_two_pass(query, k, mult≈10)`. The blocker for a *default*
(rather than opt-in) switch is the exactness gate — a sound per-query int8-slab error bound (the i8
corpus-max-abs slab quantizes each component to `q=round(x/σ)`, `σ=max_abs/127`, so by
Cauchy–Schwarz `|true_dot − int8_dot| ≤ E` with `E ≈ √dim·σ`; if `β + E < θ` where `β` is the
`k·mult`-th int8 score and `θ` the k-th *exact* rescored score, the result is provably exact).
Deriving + verifying that bound (and its int8→f32 scaling) is the correctness-critical,
multi-iteration step that shouldn't be rushed into the default path.

_Dim confirmation:_ also re-ran at **dim=768** (BGE-base / OpenAI-class embeddings), N=10000:
recall@10 = **1.0000 at every mult (2, 5, 10, 20)** — the lossless property holds across the
embedding-dim range (384–768), so the lever applies to high-dim models too. (Latency ratio
expected ~the dim=384 figure since both flat and int8 pass-1 scale ~linearly with dim; the dim=768
latency estimates didn't sync back cleanly this run, but recall — the correctness question — is the
decisive one and is lossless.)

**Exactness-gate note (for the wiring):** `ScalarQuantizer` (the u8 per-dim FSVI quantizer) already
exposes the bound machinery (`max_dim_errors()` = `scale/2`; `epsilon ≤ max_scale·√dims`), but the
in-memory two-pass uses a **separate i8 corpus-max-abs slab** — a sound certified-exact gate needs
that slab's per-query error bound derived + verified (correctness-critical), so the safe wiring is a
dedicated multi-iteration step, not a 60m change.

These rows are routing evidence for future levers, not wins.

Command:
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- cargo bench -p frankensearch --profile release --bench search_bench -- --sample-size 10
```

Worker: `ovh-a`; RCH reported remote completion in `608.1s`.

| Date | Crate | Workload | Measurement | Original ratio | Status |
|------|-------|----------|-------------|----------------|--------|
| 2026-06-24 | frankensearch | `dot_product_f32/128` | `12.223 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `dot_product_f32/256` | `30.630 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `dot_product_f32/384` | `44.346 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `dot_product_f32/768` | `82.308 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `hash_embedder/short_10w` | `734.48 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `hash_embedder/medium_100w` | `2.3161 us` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `hash_embedder/long_1000w` | `17.225 us` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `vector_search_topk/top10/1000` | `944.07 us` | blocked by `bd-ui41` | noisy baseline |
| 2026-06-24 | frankensearch | `vector_search_topk/top10/5000` | `3.4640 ms` | blocked by `bd-ui41` | noisy baseline |
| 2026-06-24 | frankensearch | `vector_search_topk/top10/10000` | `1.6642 ms` | blocked by `bd-ui41` | noisy baseline |
| 2026-06-24 | frankensearch | `rrf_fusion/fuse/1000+1000` | `80.563 us` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `score_normalization/z_score/10000` | `43.159 us` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `index_io/write/10000` | `13.812 ms` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `index_io/open/10000` | `19.865 us` | blocked by `bd-ui41` | baseline only |

### 2026-06-25 — non-semantic hash tier lexical guard (BlackThrush)

**Lever:** when a lexical backend is present, no quality embedder is configured, and the fast
embedder is one of the shipped non-semantic hash tiers (`fnv1a-*`/`jl-*`), Phase 1 now returns
lexical results directly instead of paying hash embedding, vector scan, and RRF. This is the
correct behavior for Lucene/Tantivy/Meilisearch-class lexical comparisons: a hash vector
contributes no semantic relevance, so fusing it only burns latency. If lexical search fails
non-cancel, the old vector fallback still runs.

**BOLD-VERIFY command:**

```bash
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- env FRANKENSEARCH_BOLD_VERIFY_EMIT=1 RUST_LOG=error \
    cargo bench -p frankensearch --features lexical --profile release \
    --bench search_bench bold_verify_tantivy_class \
    -- --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Worker: `vmi1152480` (`[RCH] remote vmi1152480 (804.5s)`). Incumbent is
`tantivy_doc_ids`, used here as the Lucene/Tantivy/Meilisearch-class lexical proxy.

**Kept evidence:** the guard collapses the previous hash-hybrid penalty by removing useless
semantic work and wins several Tantivy-class rows outright:

| Workload | old hash-hybrid ratio | guarded ratio | guarded p95 ratio | status |
|----------|-----------------------|---------------|-------------------|--------|
| `top10/10000/high_fanout` | 18.180 | **0.978** | **0.756** | beats Tantivy-class |
| `limit_all/10000` | 2.086 | **0.873** | **0.611** | beats Tantivy-class |
| `top10/100000/natural_language` | 5.182 | **0.982** | 1.186 | p50 beats, p95 near |
| `top10/10000/zero_hit` | 54.333 | **1.000** | 3.333 | p50 parity; tail miss |
| `top10/100000/zero_hit` | 136.696 | **1.043** | **0.722** | near p50 parity; p95 win |

**Scope:** this is not a universal BOLD victory over Tantivy-class; remaining misses are recorded
in `docs/NEGATIVE_EVIDENCE.md`. It is kept because every measured row dramatically improves over
the existing hash-hybrid path, several rows beat the lexical incumbent, and the behavior is more
semantically honest for hash-only fast tier searches. Next lever should attack the residual
`ScoredResult` materialization and high-selectivity lexical overhead rather than vector/RRF.

## Local hot-path wins

### 2026-06-27 — count-free simple `search_doc_ids` top-k (BlackThrush)

**Lever:** `frankensearch-lexical::TantivyIndex::search_doc_ids` now uses a top-k-only Tantivy
collector for single plain-token queries. The API returns ranked document IDs and BM25 scores, not
total match counts, so the old `Count` collector was discarded work on this narrow path. Queries
with Boolean, phrase, field, wildcard, boost, path, quote, or hyphen syntax keep the counted path.

**Measured command:**

```bash
AGENT_NAME=BlackThrush \
RCH_ENV_ALLOWLIST=AGENT_NAME,CARGO_TARGET_DIR,RUST_LOG \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
RUST_LOG=off \
  rch exec -- cargo bench -p frankensearch-lexical --profile release \
    --bench doc_ids_topk -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

RCH executed on `hz2` (`[RCH] remote hz2 (382.8s)`). Artifact:
`/data/projects/.rch-targets/frankensearch-cod-b/criterion/doc_ids_topk/high_fanout_{counted,free}/new/estimates.json`.

| Workload | Baseline counted median | New count-free median | Ratio | Status |
|----------|-------------------------|-----------------------|-------|--------|
| `doc_ids_topk/high_fanout` | 499.086 us | 239.233 us | **0.479** | KEEP |

**Scope:** this is a local Tantivy-wrapper hot-path win, not an original-comparator BOLD win.
The required Lucene/Tantivy/Meilisearch-class caveat, N/A original-comparator ratio, and unclaimed
fallback rows are recorded in `docs/NEGATIVE_EVIDENCE.md`.

### 2026-06-27 — single-pass `QueryClass::classify` (fused case scan + `take(4)` word count) (BlackThrush)

**Lever:** `QueryClass::classify` runs on **every** search query (adaptive lexical/semantic budget +
the lexical short-circuit gate) — pure frankensearch per-query overhead Tantivy never pays, so cutting
it directly narrows the end-to-end gap vs a Tantivy/Lucene/Meilisearch-class incumbent. Two
behaviour-identical reductions over the already-grouped baseline:
1. `looks_like_identifier` collected the camel/Pascal case flags in **three** separate Unicode-aware
   `chars()` scans (`any(is_lowercase)`, `any(is_uppercase)`, `skip(1).all(is_lowercase)`); fused into
   **one** char pass (flags are order-independent, so the boolean result is unchanged).
2. The word count only needs the `<= 3` boundary, so `split_whitespace().take(4).count()` stops after
   the 4th word instead of scanning every word of a long natural-language query.

All 26 `frankensearch-core::query_class` tests stay green, including `classify_is_trim_invariant`
(proptest) and every classification case — the change is bit-identical.

**Measured command (per-crate, in-process A/B — host-independent ratio):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=BlackThrush \
  rch exec -- cargo bench -p frankensearch-core --profile release \
    --bench query_class -- --measurement-time 5
```

RCH worker `ovh-a`. `current` = the committed grouped baseline (replicated in the bench); `new` = the
shipped `classify`. The new arm name `current` doubled as a stale-binary freshness sentinel (it
appeared in the output).

| Workload (11-query mix) | current (grouped) | new (this change) | Ratio | Status |
|-------------------------|-------------------|-------------------|-------|--------|
| `query_class` | 484.88 ns | 424.50 ns | **0.875 (~1.14×)** | KEEP |

CIs do not overlap (`current [468.2, 505.2] ns`, `new [420.0, 429.9] ns`). The same run shows the
pre-grouping `old` at 805.3 ns (cumulative ~1.9×); the isolated ratio for **this** change is **0.875**.

**Scope:** a local per-query-overhead win on pure frankensearch work (query classification). The
original-comparator (BOLD vs Tantivy/Lucene/Meilisearch) ratio is **N/A** for this isolated function —
it reduces overhead the incumbent does not have, narrowing the end-to-end gap rather than beating a
Tantivy primitive head-to-head.

### 2026-06-27 — FSVI 4-bit pass-1 uses prepared query lanes (BlackThrush)

**Lever:** `VectorIndex::search_top_k_4bit_two_pass` packed the query once, but the file-backed
pass-1 still called `dot_packed_4bit` for every stored vector; that wrapper re-prepares the query
nibble lanes on every dot. The in-memory path already used the prepared-query kernel. This change
moves the FSVI path to the same loop-invariant query decode: pack once, `prepare_4bit_query` once,
then call `dot_4bit_prepared` for each stored vector. This is the `/alien-graveyard` vectorized
execution lever in its smallest form: hoist invariant decode out of a scan morsel while preserving
the exact same candidate heap, tombstone check, and f16 rescore.

**Measured command:** the literal requested form still fails in this checkout:

```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo bench --release -p frankensearch-index \
    --bench fsvi_4bit_two_pass -- --sample-size 10 --warm-up-time 1 --measurement-time 1
```

Cargo rejects `cargo bench --release` with `unexpected argument '--release'`; the valid per-crate
release-profile measurement was:

```bash
AGENT_NAME=BlackThrush \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo bench -p frankensearch-index --profile release \
    --bench fsvi_4bit_two_pass fourbit_mult5 \
    -- --sample-size 10 --warm-up-time 1 --measurement-time 2
```

RCH had no admissible worker for the bench and fell back locally. To avoid the known stale-binary
pitfall with the shared target dir, the candidate source mtime was bumped and the second run
confirmed a fresh `frankensearch-index` rebuild before timing.

| Workload | Baseline (`origin/main`) | New | Ratio | Recall | Status |
|----------|--------------------------|-----|-------|--------|--------|
| `fsvi_4bit_two_pass/fourbit_mult5` | 3.6991 ms | 1.5660 ms | **0.423 (~2.36x)** | recall@10 = 1.0000 at mult=5 | KEEP |

**Scope:** this is a local file-backed vector-search primitive win against the immediately prior
frankensearch FSVI 4-bit path. The original-comparator ratio vs Lucene/Tantivy/Meilisearch is
**N/A** for this isolated vector pass-1; the comparator caveat is recorded in
`docs/NEGATIVE_EVIDENCE.md`.
### 2026-06-27 — JL hash embedder xorshift ILP (interleave 4 token chains) (Cobaltmoth)

**Lever:** `HashEmbedder::embed_jl` (the `jl-*` fast tier) is the compute-bound embedder —
O(tokens·dim) xorshift64. A single token's chain is **latency-bound**: each step's three
*dependent* shift→xor ops must retire before the next, so one chain cannot saturate the CPU's
shift/ALU ports. (This ledger already noted the JL path is "compute-bound xorshift" with alloc
savings lost in noise; the only prior JL lever — branchless sign — regressed because LLVM already
cmov's it, see `docs/NEGATIVE_EVIDENCE.md`.) Each token seeds an **independent** chain, so
`embed_jl` now advances `JL_LANES = 4` token chains together per dimension (`jl_accumulate_lanes`),
exposing instruction-level parallelism that hides the single-chain latency; the (< 4) tail drains
one chain at a time via the original loop.

**Bit-identical:** the per-dimension accumulator only ever holds an exact integer-valued `f32`
(a sum of ±1 contributions, `|value| ≤ token count ≪ 2^24`), so f32 addition is exact and
reordering the lane contributions cannot change a single output bit. Proven by
`jl_ilp_matches_scalar_reference_bit_identical` (dims 256/384, two seeds) +
`jl_ilp_matches_scalar_across_token_counts` (token counts 0..=9 straddling the lane boundary); all
33 `hash_embedder` tests GREEN.

**Measured command (per-crate, in-process A/B — host-independent ratio):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=Cobaltmoth \
  rch exec -- cargo bench -p frankensearch-embed --bench hash_embed -- hash_embed_jl_ilp
```

`scalar` = the committed single-chain inner loop (replicated in the bench); `ilp2`/`ilp4` = the
2-/4-lane interleaved variants. ~100-word doc, dim 384. Medians:

| Workload | scalar | ilp2 | ilp4 | Ratio (ilp4/scalar) | Status |
|----------|--------|------|------|---------------------|--------|
| `hash_embed_jl_ilp` (dim384) | 102.88 µs | 69.81 µs | **50.19 µs** | **0.488 (~2.05×)** | KEEP |

CIs do not overlap (`scalar [100.99, 105.49] µs`, `ilp4 [49.70, 50.89] µs`). Independently
corroborated by a concurrent run on a different worker/target dir (`scalar` 98.0 µs → `ilp4`
51.2 µs, **~1.92×**). `ilp4` ships; `ilp2` (~1.47×) is the runner-up the bench keeps for the curve.

**Scope:** a local hot-path win on the `jl-*` non-semantic fast tier (the slowest embedder).
Original-comparator (BOLD vs Tantivy/Lucene/Meilisearch) ratio is **N/A** — this reduces
frankensearch-internal embedding compute the incumbents do not have; it narrows the end-to-end gap
on jl-tier searches rather than beating a Tantivy primitive head-to-head.

### 2026-06-27 — MMR re-rank: incremental running-max + hoisted norms (Cobaltmoth)

**Lever:** `mmr_rerank` (diversity re-ranking) greedily picks `k` of `n` candidates; its inner
sweep did two redundant things, both pure frankensearch per-rerank overhead:
1. **O(k²·n) → O(k·n) cosine evals.** Each round recomputed `max(sim(i, j) for j in selected)` from
   scratch over the whole growing `selected` set. Now a per-candidate running max
   (`max_sim_to_selected`) is updated once per selection, so every `sim(i, selected_doc)` is computed
   exactly once. `f64::max` is associative ⇒ the running max equals the per-round fold bit-for-bit.
2. **3 reductions → 1 per pair.** `cosine_sim` re-derived both vectors' L2 norms on every pair even
   though an embedding's norm is loop-invariant. Norms are hoisted once per candidate (`root_norms`),
   leaving only the dot reduction per pair (`cosine_sim_pre`). Bit-identical for uniform-dimension
   embeddings (the MMR contract); ragged dims fall back to per-pair `cosine_sim`.

**Bit-identical:** the dot loop and accumulation order are unchanged, `root_norms[i]·root_norms[j]`
reproduces the prior `norm_a.sqrt()·norm_b.sqrt()`, and the running max reproduces the per-round fold
— the selected index list is unchanged. Proven by `incremental_norm_hoist_matches_bruteforce`
(dims 8/64/384 × n 3/16/60 × k 1/5/n vs a brute-force reference) + all 32 `mmr` tests GREEN.

**Measured command (per-crate, in-process A/B — host-independent ratio):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=Cobaltmoth \
  rch exec -- cargo bench -p frankensearch-fusion --bench mmr_rerank
```

`old` = per-round full recompute + per-pair norms (replicated); `new` = running-max + hoisted norms.
dim 384. Medians:

| Workload (n cand, k results, dim384) | old | new | Ratio | Status |
|--------------------------------------|-----|-----|-------|--------|
| `mmr_rerank/n100_k20` | 5.387 ms | 581.9 µs | **0.108 (~9.3×)** | KEEP |
| `mmr_rerank/n200_k50` | 65.10 ms | 2.800 ms | **0.043 (~23.3×)** | KEEP |

The gain grows with `k` (the O(k²·n)→O(k·n) factor compounding with the 3→1 reduction hoist).

**Scope:** a local hot-path win on the diversity-rerank (MMR) path — pure frankensearch compute the
Tantivy/Lucene/Meilisearch lexical incumbents never perform. Original-comparator ratio is **N/A**
(recorded in `docs/NEGATIVE_EVIDENCE.md`); it narrows end-to-end latency when diversity rerank is
enabled rather than beating a comparator primitive head-to-head.

### 2026-06-27 — graph-rank PageRank: dense-index power iteration (Cobaltmoth)

**Lever:** `GraphRanker::rank_phase1` (query-biased PageRank, `graph` feature) ran its power
iteration over a `HashMap<GraphDocId=String, f64>` **rebuilt every iteration**, with a `String`
doc_id **clone on every teleport and every edge relaxation** (`next.entry(doc_id.clone())`) — and
because `add_edge` inserts both endpoints, every key those `entry()` calls touched was already
present, so the clones were pure dead work — plus a per-edge weight-finiteness check and `f64::from`
widen redone on every one of the (≤20) iterations. Replaced with a one-time dense index (every node
is an `adjacency()` key, so node set = `0..n`) + CSR edge arrays + two reused `Vec<f64>` buffers
(swapped per iteration). Per iteration this drops a HashMap allocation, all String clones, and a hash
probe per edge → array indexing; the weight filter + widen are hoisted into the one-time CSR build.

**Equivalent ranking:** PageRank converges to the same fixed point within `tolerance`, and the prior
`std::HashMap` accumulation order was already run-to-run non-deterministic, so no exact value was ever
pinned. Verified by `dense_rank_matches_reference_ranking` (40-node random graph, 5 seeds) which
asserts the dense result's doc_id **order matches a deterministic `BTreeMap` reference of the original
algorithm** with scores within 1e-4, plus the existing propagation/empty tests, all GREEN under
`--features graph`.

**Measured command (per-crate, in-process A/B; replicas of old HashMap vs new dense):**

```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
RCH_ENV_ALLOWLIST=CARGO_TARGET_DIR,AGENT_NAME AGENT_NAME=Cobaltmoth \
  rch exec -- cargo bench -p frankensearch-fusion --bench graph_rank
```

`old` = HashMap rebuilt per iteration + clone-keyed `entry`; `new` = dense `Vec<f64>` + CSR. 10 seeds,
limit 50, 20 iterations. Medians:

| Workload (n nodes, out-degree) | old | new | Ratio | Status |
|--------------------------------|-----|-----|-------|--------|
| `graph_rank/n500_deg6` | 2161.5 µs | 181.9 µs | **0.084 (~11.9×)** | KEEP |
| `graph_rank/n2000_deg8` | 11.661 ms | 982.7 µs | **0.084 (~11.9×)** | KEEP |

**Scope:** a local hot-path win on the optional `graph`-feature PageRank stage — pure frankensearch
compute the Tantivy/Lucene/Meilisearch lexical incumbents never perform, and off the default build
(feature-gated). Original-comparator ratio is **N/A** (recorded in `docs/NEGATIVE_EVIDENCE.md`); it
makes the graph-rank feature ~12× cheaper when enabled, not a head-to-head comparator win.
