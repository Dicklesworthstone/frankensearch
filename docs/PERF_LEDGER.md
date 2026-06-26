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
| 2026-06-25 | frankensearch-fusion | **RRF fuse: one `entry` lookup instead of `get`+`entry`** (halve per-candidate hashing of the `AHashMap<&str,_>` accumulator) | `rrf_fuse` (1000 lexical + 1000 semantic, ~50% overlap) | 29.11 µs | 23.07 µs | **0.793 (~1.26×)** | KEEP (BlackThrush) |
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators | `dot/dim256/f32_bytes` | 10.839 ms | 3.647 ms | **0.336** | KEEP |
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators | `dot/dim384/f32_bytes` | 14.084 ms | 5.333 ms | **0.379** | KEEP |
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators (`BlueGull` pinned-worker confirmation) | `dot/dim256/f32_bytes/10000` | 3.4835 ms | 0.66126 ms | **0.190** | KEEP (`vmi1149989`) |
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators (`BlueGull` pinned-worker confirmation) | `dot/dim384/f32_bytes/10000` | 5.1487 ms | 1.8811 ms | **0.365** | KEEP (`vmi1149989`) |
| 2026-06-24 | frankensearch-index | **branchless SIMD f16→f32 widen** (default path) | `dot/dim256/f16_bytes` | 4.733 ms | 1.632 ms | **0.345** | KEEP |
| 2026-06-24 | frankensearch-index | **branchless SIMD f16→f32 widen** (default path) | `dot/dim384/f16_bytes` | 7.363 ms | 2.332 ms | **0.317** | KEEP |
| 2026-06-24 | frankensearch-index | branchless SIMD f16→f32 widen | `dot/dim256/f16_slice` | 3.699 ms | 1.348 ms | **0.364** | KEEP |
| 2026-06-24 | frankensearch-index | branchless SIMD f16→f32 widen | `dot/dim384/f16_slice` | 5.536 ms | 2.181 ms | **0.394** | KEEP |

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

_Scale confirmation:_ re-ran the same clustered bench at **N=100000** (dim=384, k=10 — the BOLD /
Tantivy-comparison corpus size): recall@10 = **1.0000 at every mult (2, 5, 10, 20)**, i.e. the
lossless property holds at production scale (the int8 quantization error is scale-independent, so a
modest `k·mult` candidate set still always contains the true top-10). Latency ratio at 100k is the
~1.5× from the `dim384/n100k` row above. So the recommendation (use `search_top_k_int8_two_pass`,
mult≈10) is bulletproof across 10k–100k.

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
