# COMPREHENSIVE PLAN FOR THE DESIGN OF QUILL

*A blank-slate, memory-safe, ultra-high-performance lexical (BM25 full-text) indexing and search engine in pure Rust, built entirely on the frankensearch/asupersync foundation with zero new external dependencies, designed to completely replace Tantivy inside frankensearch and to leapfrog it on indexing throughput — especially on Apple Silicon and high-core-count AMD/Intel CPUs — while producing conformant results, with Tantivy retained as a pinned conformance oracle.*

**Status:** Design plan (pre-implementation). **Tracking:** beads epic family `quill` (see §19). **Oracle:** tantivy `0.26.1` (pinned, `Cargo.lock`).

---

## 0. The Thesis: What "Leapfrog" Actually Means Here

Tantivy is an excellent general-purpose search library — and that is precisely why it is beatable *inside frankensearch*. Tantivy is a faithful architectural descendant of Lucene, and it inherits Lucene's fossils: decisions that were correct for a 2005-era JVM server indexing the open web, transplanted into a 2020s Rust library, then adopted by frankensearch which uses a *narrow, well-bounded slice* of the surface (§3) on a *specific workload* (local codebases/monorepos, 10k–1M docs, watch-mode incremental updates) with a *specific runtime* (asupersync structured concurrency) that tantivy knows nothing about.

The measured facts that make the leapfrog credible:

1. **frankensearch exercises a narrow waist.** The complete tantivy surface census (§3) shows: default BM25 only, five collectors' worth of behavior expressible as one top-k kernel, six query types (four of which are trivial), one FAST column, one custom analyzer family, no facets, no aggregations, no custom scorers, no JSON/date/ip fields. A ground-up engine does not have to be "a better Tantivy" — it has to be a better *this*.

2. **Tantivy pays taxes frankensearch never uses.**
   - **The merge tax.** Tantivy's LSM-ish segment model merges by *decoding and re-encoding* every posting list. frankensearch's own `cass_compat.rs` had to grow `force_merge_bounded`, merge cooldowns, and custom merge-policy plumbing just to keep this tax survivable. Quill's docid-range discipline makes merge a *streaming concatenation of unchanged, self-delimiting posting blocks* (§7) — near-memcpy, with no posting decode or rebase.
   - **The doc-store tax.** Tantivy lz4-compresses stored fields into its doc store on the ingest path. frankensearch already owns canonical document storage (FrankenSQLite metadata + content); the CASS schema already marks `content` as indexed-not-stored and hydrates from canonical storage. Quill stores only what retrieval needs (id map, snippets source policy per field) and pays no generic compressed doc store.
   - **The visibility tax.** Tantivy requires `commit` + reader `reload` for searchability (frankensearch calls both, every batch, on the watch-mode hot path). Quill's in-memory delta segment is searchable immediately upon ingest with epoch-published snapshots — commit becomes a durability event, not a visibility event.
   - **The generality tax.** Tantivy's writer pipeline routes every document through a generic schema/`Value` machinery, per-field dynamic dispatch, and a stacker arena keyed by term bytes. Quill compiles frankensearch's two concrete schemas (default 5-field, chunk-policy fields) into monomorphic ingest paths with SIMD-classified tokenization feeding columnar term-id streams (§6.2).

3. **The runtime mismatch is real.** Today, tantivy's blocking calls run *inline inside async fns* on asupersync workers (`crates/frankensearch-lexical/src/lib.rs` — `_cx` is unused in the search paths), because tantivy predates and ignores structured concurrency. Quill is asupersync-native from byte zero: `Cx` threading, cancel-correct writer paths, `spawn_blocking` where appropriate, and — decisively for verification — the entire engine runs under `LabRuntime` with virtual time and deterministic scheduling, which tantivy structurally cannot.

4. **High-core-count CPUs punish tantivy's write path shape.** Tantivy parallelizes indexing across writer threads, but the endgame is always the same funnel: segment serialization, then merges that re-encode everything, coordinated through a single `IndexWriter`. frankensearch's CASS path already fights this with rayon `par_chunks` around `IndexWriter::run` and a 32-thread clamp. Quill's share-nothing shard-per-worker design (§6.1) has *no shared mutable state* on the ingest hot path, and its merge=concat property means adding cores adds throughput all the way through sealing, not just through tokenization. On Apple Silicon, work-stealing over small shard batches handles P/E-core asymmetry naturally; on 32–96-core x86, disjoint docid ranges mean workers never contend.

5. **Memory safety is a feature tantivy cannot retrofit.** Tantivy contains substantial `unsafe` and pulls ~40 transitive dependencies into the `lexical` feature. Quill is written under the workspace lint wall (`unsafe_code = "deny"`, clippy pedantic+nursery) with at most the single, ledgered mmap allowance the FSVI reader already uses — and zero new external dependencies (§1).

**The leapfrog is therefore a composition of five bets:**

| # | Bet | One-line statement | Why tantivy can't follow |
|---|-----|--------------------|--------------------------|
| **Q1** | **Merge = Concat** | Pairwise-disjoint, globally ordered segment intervals (holes legal) plus absolute docids make merge an ordered streaming concatenation: ordinary merge copies self-delimiting posting blocks unchanged, with no posting rebase or seam rewrite; compaction re-encodes only tombstone-dense segments. | Tantivy's docids are per-segment-dense by construction; merge must renumber every posting. This is load-bearing Lucene heritage. |
| **Q2** | **Columnar sort-based ingest** | Tokenize into flat `(term_id, doc_ordinal, pos)` columns in bump arenas; radix-partition by term at flush. Cache-shaped sequential passes replace per-token hashmap random access. | Tantivy's stacker/expull hashmap-per-term is fixed architecture; the write path cannot become columnar without a rewrite. |
| **Q3** | **Searchable delta, durable seal** | An always-searchable in-memory delta segment (epoch/Arc-swap published) decouples visibility from durability; `commit` = seal + fsync + manifest publish only. | Tantivy couples visibility to commit+reload; NRT-style reopen is its floor. |
| **Q4** | **Schema-specialized SIMD front-end** | The analyzer chain (fused simple+lowercase; CASS hyphen/CJK/edge-ngram family) compiles to SIMD/SWAR byte-classification kernels with byte-parity tests against the shipping scalar implementations. | Tantivy's tokenizer API is a generic char-iterator pipeline; per-byte dispatch is its floor. |
| **Q5** | **Verified conformance, honest speed** | Tantivy stays in-tree as a pinned oracle behind a non-default feature; a differential gauntlet (rank-conformance classes, metamorphic suites, crash matrices, LabRuntime determinism) plus keep-gated honest benchmarks certify "same results, faster" before the default flips. | Not a capability gap — a discipline tantivy consumers don't get for free. The gauntlet is what makes the replacement safe. |

**Anti-goals of this document.** No "temporary" second-class mode: Quill at 1.0 serves every query the default lexical path serves today, byte-compatible with frankensearch's public behavior (§5). No format compromises "to be fixed later": FSLX (§10) is versioned, checksummed, and repair-trailer-ready from the first byte. No benchmark theater: every performance claim in §14 follows the keep-gate rules (release-perf profile, cv_pct, pass-over-pass ratchet, MT8 attribution) and lands in `docs/PERF_LEDGER.md` / `docs/NEGATIVE_EVIDENCE.md` under the house rules. The cass tantivy-format interop (§3.4) is explicitly out of Quill's scope and stays tantivy-gated until the external cass tool migrates.

---

## 1. Constraints and Non-Negotiables

1. **The dependency universe is closed.** Quill adds **zero new external dependencies**. Allowed: the existing workspace dependency set — `asupersync` (runtime, channels, sync, LabRuntime), `rayon` (CPU-bound data parallelism only), `wide`/`bytemuck` (portable SIMD), `memmap2` (read-side mapping, same allowance discipline as FSVI), `compact_str`, `ahash`, `crc32fast`, `xxhash-rust` (xxh3), `thiserror`, `tracing`, `serde`/`serde_json` (**never** for durable formats — manifest and segment bytes are hand-rolled §10), `unicode-normalization`, and in-house family crates (`fsqlite-*` via frankensearch-storage, frankensearch-core/-durability). `tantivy 0.26.1` remains **only** as the pinned conformance oracle and the cass-interop backend, behind non-default features. No new crates.io deps, full stop.
2. **Memory safety is structural.** Quill crates carry `#![forbid(unsafe_code)]` at the crate root. The single exception class is the mmap read path, which follows the exact house precedent (`frankensearch-index`, `frankensearch-durability`): workspace lint `unsafe_code = "deny"` with a narrowly scoped, commented `#![allow(unsafe_code)]` **only** in the module that materializes a `memmap2` map, exposing a safe API. If Quill can reuse a shared safe mmap facade from `frankensearch-index`, it does that instead and stays at `forbid`. Every SIMD kernel is safe code (`wide`/SWAR); no `core::arch` intrinsics.
3. **asupersync only; `Cx` everywhere.** Every Quill operation that blocks, locks, or does I/O takes `&Cx` and honors cancellation. Long CPU/I-O work on async paths goes through `spawn_blocking` or executes inline *by documented contract* where the caller requires it (the fusion rayon path **requires** `search_fusion_candidates` to resolve without yielding `Pending` — `crates/frankensearch-fusion/src/searcher.rs:1270-1290`; Quill honors this with a fully synchronous read path wrapped in an immediately-ready future). Background work (merges, compaction) lives in asupersync regions with two-phase channels — no detached threads. Tokio and its ecosystem remain forbidden.
4. **Conformance before default.** The `lexical` feature default does not flip to Quill until the gauntlet gates in §15/§18 pass: rank-conformance on the differential corpus with all divergences classified, crash-matrix green, hybrid-fusion end-to-end quality unchanged (nDCG/MRR/Recall with bootstrap CIs), and the perf gates of §14 met with keep-gate-clean evidence.
5. **Determinism is a contract, not a vibe.** Same corpus + same ingest order + same config ⇒ byte-identical sealed segments and identical query results, including tie order (§8.4). Same corpus in *any* ingest order ⇒ identical result *sets* and identical scores; rank ties broken by the documented deterministic key. Concurrent ingest schedules may vary docid assignment across shards, but a `deterministic_ingest` mode (single-shard or round-robin-sealed) exists for replay/testing and is what LabRuntime suites use.
6. **Durable formats are hand-written and versioned.** FSLX files carry magic, format version, section table, per-section checksums (xxh3-64 fast path + crc32 trailer, matching the durability crate's conventions), little-endian fixed-width integers except the explicitly big-endian TERMDICT `field_ord` key prefix, and explicit alignment (§10). No serde-derived durable bytes. Every format change bumps the version and adds an entry to the format registry table in §10.6. Repair trailers / RaptorQ sidecars via the existing generic `FileProtector` work on day one.
7. **The house workflow binds this project.** Beads (`br`) for tracking with the epic/dependency structure of §19; `br sync --flush-only` + manual commits; UBS before commits; `cargo fmt --check`, `cargo check --workspace --all-targets`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test` gates; RCH offload for heavy builds/benches; Agent Mail file reservations when swarm-executed; negative results ledgered in `docs/NEGATIVE_EVIDENCE.md` with the Ratio convention; wins in `docs/PERF_LEDGER.md`.
8. **Prohibited shortcuts (constitutional).** No global-lock "interim" writer presented as the engine; no `HashMap<String, Vec<u32>>` presented as an index; no visibility-without-durability presented as commit; no benchmark run without the oracle side under identical analyzer/config; no conformance claim from a corpus the engine was tuned on without a held-out fixture; no silent semantic divergence — every intentional divergence from tantivy behavior is recorded in the Divergence Register (§15.6) with a classification and a consumer-impact note.
9. **Scope honesty.** Quill 1.0 replaces the **default frankensearch lexical path** (`TantivyIndex` + the `LexicalSearch` trait duties + fsfs/facade integration + snippets + the wildcard/boolean/range/phrase query surface of §5). Quill 1.0 does **not** reimplement the tantivy on-disk format: `cass_compat.rs` (external CASS tool interop, schema v8 on-disk tantivy indexes) remains tantivy-backed behind an opt-in `cass-compat` feature until the external tool migrates (§16.3). FTS5 remains the storage-integrated alternative it is today.

---

## 2. Foundation Audit: Exactly What We Stand On

This section maps *specific existing assets* to *specific Quill subsystems*. The leapfrog claim is only credible because most of the hard, non-lexical problems are already solved in-tree.

### 2.1 frankensearch-core: the seam and the contracts

| Asset | Location | Quill use |
|---|---|---|
| `LexicalSearch` trait (dyn-compatible, boxed `SearchFuture`) | `crates/frankensearch-core/src/traits.rs:600-692` | **The** integration seam. Quill implements it; fusion consumes `Arc<dyn LexicalSearch>` (`fusion/src/searcher.rs:129`) and never names a concrete engine. Two prior implementations prove the seam: `TantivyIndex` (`lexical/src/lib.rs:1335`), `Fts5LexicalSearch` (`storage/src/fts5_adapter.rs:343`). |
| `ScoredResult`, `DocId = CompactString`, `ScoreSource::Lexical`, `IndexableDocument` | `core/src/types.rs:15,31,179,198` | Quill's output/input types, unchanged. Rank is positional in the returned slice ⇒ Quill returns score-desc order with the deterministic tie-break of §8.4. |
| `SearchError` (`SubsystemError{subsystem}`, `IndexNotFound`, `Cancelled`, `Io`, `InvalidConfig`) | `core/src/error.rs` | Quill maps errors with `subsystem: "quill"` mirroring the uniform tantivy bridge; writer-lock cancellation → `Cancelled{phase}`. |
| Canonicalizer, `QueryClass` | `core/src/{canonicalize,query_class}.rs` | Upstream of Quill; unchanged. Quill receives already-canonicalized queries from the searcher. |
| `metrics_eval` (nDCG/MRR/Recall/MAP, bootstrap CI/compare, CV/outlier guards) | `core/src/metrics_eval.rs:25-624` | The statistical engine of the conformance/quality gates (§15.4, §18 G3). |
| Core SIMD module | `core/src/simd.rs` | Reference patterns for safe `wide` kernels. |

### 2.2 frankensearch-index: the format school

| Asset | Location | Quill use |
|---|---|---|
| FSVI binary format discipline: magic `b"FSVI"`, version, variable header + `header_crc32`, 64-byte alignment, tombstone record flags, mmap read path behind a scoped allow | `crates/frankensearch-index/src/lib.rs:6-113,234` | The house style FSLX imitates (§10): same magic/version/CRC/alignment/tombstone-flag conventions, same mmap allowance discipline, same `SearchError::IndexCorrupted` validation posture. |
| Append-only WAL module with compaction + `compaction_gen` staleness byte | `index/src/wal.rs` | Direct pattern for Quill's ingest journal option (§11.4) and the staleness handshake between manifest and segments. |
| Two-phase top-k (score first, materialize ids for winners), NaN-safe `total_cmp` heaps, `PARALLEL_THRESHOLD = 10_000` rayon gate | `index/src/search.rs` | The top-k collector shape Argus reuses (§9.3); the same threshold philosophy for parallel query fan-out. |
| Runtime SIMD dispatch precedent (`wide` portable + feature-detected kernels) | `index/src/simd.rs:6-232` | Kernel organization pattern for Quill's decode/score loops. |

### 2.3 frankensearch-lexical: the incumbent (and the oracle)

| Asset | Location | Quill use |
|---|---|---|
| Fused `FrankensearchTokenizer` (simple+lowercase, ASCII fast path) | `lexical/src/lib.rs:466` | The byte-parity **reference** for Quill's default analyzer (§8.1). The existing parity tests (vs `SimpleTokenizer`+`LowerCaser`) transfer to Quill's SIMD implementation verbatim. |
| CASS analyzer family: `CassTokenizer`, `HyphenDecompose`, `CjkBigramDecompose`, `CassNormalizeAndLimit`, edge n-grams, preview builder — all with fast/slow parity oracles | `lexical/src/cass_compat.rs:58-1674` | The token-level semantics Quill's analyzer framework must be able to express (§8.2); the `#[doc(hidden)]` slow oracles become conformance fixtures. |
| The complete query surface: lenient 2-field `QueryParser` w/ title boost 2.0, 10k-char truncation, count-free top-k gate; CASS boolean grammar (OR-tighter-than-AND, NOT via `Must(All)+MustNot`), `CassWildcardPattern` (glob→regex), i64 range filters | `lexical/src/lib.rs:226-365,913;` `cass_compat.rs:1798-2347` | The Language Contract input (§5, §8.3). Note the wildcard source language is **globs**, so Argus needs a glob-automaton term matcher, not a regex engine. |
| `ord` FAST column + `ord_table.json` sidecar id-materialization | `lexical/src/lib.rs:1184` | The problem Quill dissolves: docid↔`DocId` mapping is a first-class FSLX section (§10.3), no sidecar, no fallback path. |
| 90 unit tests incl. tokenizer byte-parity, CJK e2e, ranking/boost/upsert/delete semantics | `lexical/src/{lib,cass_compat}.rs` | Behavioral spec harvest: each test that encodes *engine-visible semantics* is ported to the engine-agnostic conformance suite (§15.3) and must pass against **both** engines. |
| 15 benches (5 pure-algorithm parity A/Bs; index-layout/id-materialization/count-free gates) | `lexical/benches/` | Baseline harnesses for §14; the `bench-internals` same-binary A/B convention carries over. |

### 2.4 frankensearch-durability, -storage, -fsfs, facade

| Asset | Location | Quill use |
|---|---|---|
| Generic `FileProtector`, repair trailer format (`FSDR`, xxh3-64 source hash, crc32, v2), atomic temp+rename sidecars, verify-and-repair-on-open | `durability/src/{repair_trailer,fsvi_protector}.rs` | FSLX segments and manifests get `.fec` protection through the *generic* path — no per-engine wrapper needed (the tantivy wrapper's per-segment-file enumeration complexity disappears because FSLX is one file per segment §10.1). |
| FrankenSQLite storage: canonical docs, content-hash dedup, durable embedding job queue, `claim_batch(32)` worker pattern | `storage/src/{pipeline,job_queue,content_hash}.rs` | Quill's ingest sits beside this pipeline (fsfs drives both); canonical content storage is why Quill needs no generic doc store (§6.4). |
| fsfs `LiveIngestPipeline`: per-batch upsert/delete + one commit per batch, adaptive `lexical_debounce_window_ms`, incremental change detection contract | `fsfs/src/runtime.rs:1531-2033,659-970;` `fsfs/src/incremental_change.rs` | The watch-mode driver Quill plugs into (§13.2). The per-batch-commit cadence is exactly what Q3's delta/seal split accelerates. |
| fsfs `lexical_pipeline.rs`: backend-agnostic `LexicalIndexBackend` trait, `LexicalChunkPolicy`, `tokenize_lexical`, and the **existing throughput contract** — 20k docs/s initial, 5k updates/s incremental, 25ms p95 | `fsfs/src/lexical_pipeline.rs:18-706` | The mutation-planning layer Quill implements; the contract numbers are Quill's *floor*, not its target (§14). |
| Quality/perf harnesses: `search_quality_harness` (v2, per-slice nDCG/MRR/Recall + CIs), `benchmark_baseline_matrix` (golden profiles, bootstrap compare, CV pre-gates, drift dashboards), `pressure_simulation_harness` (fault injection + goldens), shared fixtures (`tests/fixtures/{corpus,queries,relevance,edge_cases}.json`) | `fsfs/tests/`, `tests/fixtures/` | The conformance/quality gate machinery of §15/§18 reuses these wholesale — swap the lexical backend, diff the artifacts. |
| Facade feature wiring conventions (`lexical = ["dep:frankensearch-lexical"]`, additive bundles `hybrid/persistent/durable/full`) | `frankensearch/Cargo.toml:8-33` | The migration mechanics of §16. |

### 2.5 asupersync: the operating system

| Asset | Quill use |
|---|---|
| `Cx` capability contexts; regions; structured cancellation | Ingest/merge/compaction lifecycles; cancel-correct writer paths (§11). |
| `spawn_blocking` (panic-safe blocking pool) | Sealing, fsync, large merges — off the async workers (§11.2). |
| Two-phase reserve/commit channels (mpsc/oneshot/watch) | Shard feeder → sealer pipeline with cancel-safe backpressure (§6.1). |
| `asupersync::sync::Mutex`/`RwLock` (cancel-aware) | Writer-side coordination (manifest publish lock); read side is lock-free (epoch/Arc-swap snapshots). |
| `LabRuntime` (seeded deterministic scheduling, virtual time, oracles; `test-internals` dev feature) | The determinism suites and crash matrices of §15.5 — the capability tantivy structurally lacks. |

### 2.6 What the foundations do *not* provide (and we therefore build)

The inverted-index core itself: SIMD tokenization kernels, term interning/dictionaries (prefix-block + optional FST tier), posting-list encoding (FOR/bitpacked blocks + block-max metadata + roaring-style dense hybrid), the columnar ingest accumulator + radix flush, the delta segment, BM25 scoring with fieldnorm quantization, boolean/phrase/range/glob query evaluation with WAND/MaxScore top-k, snippet generation, the segment manifest + commit protocol, merge/compaction machinery, and the conformance gauntlet harness. That inventory — roughly 15–25 KLOC of new Rust across two crates (§17) — is the entire build; everything else is reuse.

---

## 3. The Used-Surface Audit: The Narrow Waist We Must Cover

Complete census of tantivy usage (workspace-wide, 2026-07-16, tantivy `0.26.1` default features; 44 files touch tantivy, of which the load-bearing ones are `lexical/src/lib.rs` (2451 LOC), `lexical/src/cass_compat.rs` (3505 LOC), `durability/src/tantivy_wrapper.rs` (1116 LOC), `fsfs/src/runtime.rs`, `frankensearch/src/index_builder.rs`).

### 3.1 What frankensearch actually uses (Quill 1.0 MUST cover)

| Area | Exercised surface | Quill disposition |
|---|---|---|
| **Schema** | Default 5-field: `id` STRING\|STORED (keyword, exact); `content`/`title` TEXT stored, `WithFreqsAndPositions`, tokenizer `frankensearch_default`; `metadata_json` STORED-only; `ord` u64 FAST\|STORED. Introspection by name + option flags. | Compiled-in schema descriptors (§6.4); positions on for text fields (phrase support), freq always; id map + stored metadata first-class. |
| **Analyzers** | Fused simple+lowercase (`FrankensearchTokenizer`); CASS family (hyphen decompose, CJK bigrams, normalize+256B limit, edge n-grams, preview) with byte-parity oracles. | §8.1–8.2; SIMD kernels with the same byte-parity tests. |
| **Queries** | `TermQuery`; `BooleanQuery` (Must/MustNot/Should, nested); `PhraseQuery`; `RangeQuery` (i64, `Bound`); `RegexQuery` **only via `CassWildcardPattern` globs** (Exact/Prefix/Suffix/Substring/Complex); `AllQuery`; lenient `QueryParser` over `[content, title]`, title boost 2.0, `MAX_QUERY_LENGTH=10_000` char truncation. | §8.3 Language Contract: term/boolean/phrase/range/glob/all + lenient parser with identical truncation and boost semantics. Glob-automaton term matching — no regex engine. |
| **Collectors** | `TopDocs::with_limit(+offset).order_by_score()`; `(TopDocs, Count)` tuple; `DocSetCollector`; count-free top-k gate for 1–2 plain terms. | One top-k kernel with optional exact-count accumulation and an id-set mode (§9.3). Count-free gate becomes structural (count is free when WAND is off; exact count accumulated only when requested). |
| **Scoring** | Default BM25 (k1=1.2, b=0.75), Lucene-style fieldnorms; multi-field sum with per-field boost (title 2.0). | §8.4: identical formula, identical fieldnorm quantization table for rank parity, documented FP discipline. |
| **Writer** | Heap budget (50MB default; CASS: threads×128MiB≥256MiB, threads=avail.clamp(1,32)); `add_document`; upsert = `delete_term(id)`+add; `delete_all_documents`; batched `run(UserOperation::Add)` under rayon; `commit`; merge policies (`LogMergePolicy` +`min_num_segments(256)`, `NoMergePolicy`, custom Arc adapter, `merge(...).wait()`, `force_merge_bounded` — *uncommitted working-tree feature with regression tests*); merge cooldown atomics. | §6/§12: shard budgets, upsert/delete/clear, batch ingest, seal/commit; merge policy becomes a small tier policy (§12.2) — the entire cooldown/bounded-merge apparatus dissolves under Q1 (merges are cheap concat; compaction is the only rewrite and is tombstone-triggered). |
| **Reader** | `ReloadPolicy::OnCommitWithDelay` + explicit `reload()` after commit; `searcher.num_docs`; `fast_fields().u64("ord")` (`Column::{first,max_value,num_docs}`); docstore fallback for id materialization; `ord_table.json` sidecar. | Dissolved: epoch-published snapshots (visibility is immediate §6.3); docid↔`DocId` map is an FSLX section; no sidecar, no fallback (§10.3). |
| **Snippets** | `SnippetGenerator::{create,set_max_num_chars,snippet_from_doc}`, HTML render with custom tags, default 200 chars. | Native snippet generator over canonical/stored content (§9.5) with output-parity fixtures. |
| **Segments (durability)** | `searchable_segment_metas()` → `SegmentMeta::list_files()` per-component protection; path-safety guard. | FSLX = one file per segment + one manifest ⇒ generic `FileProtector` applies directly (§11.3). |
| **Errors** | Uniform `SubsystemError{subsystem:"tantivy"}`; `IndexNotFound` on open-missing; `Cancelled` from writer lock. | Same taxonomy, `subsystem:"quill"`. |
| **Concurrency contract** | Writer behind `asupersync::sync::Mutex`; `TantivyIndex: Send+Sync`; **fusion requires `search_fusion_candidates` to resolve without `Pending`** (poll_immediate under `rayon::join`). | Quill read path is fully synchronous under the hood (§9.1) — the boxed future is immediately ready; `Send+Sync` by construction. |

### 3.2 What frankensearch does NOT use (Quill deliberately omits)

Facets; JSON/date/ip/bytes/f64/bool fields; fuzzy/MoreLikeThis/DisjunctionMax/ConstScore queries; custom `Weight`/`Scorer`; aggregation/histogram/filter collectors; custom similarities or per-field BM25 params; 2-phase `prepare_commit`; `delete_query`; index sorting; multi-value fast fields beyond the single u64 pattern. **These are explicitly out of Quill 1.0's surface.** Any future need re-enters through the Language Contract with its own conformance fixtures.

### 3.3 API-shape decision: narrow the seam, don't shim tantivy

The lexical crate currently **re-exports tantivy types** (`Field/Schema/TopDocs/Index/IndexWriter/...`, `lexical/src/lib.rs:34-40`) and fsfs/facade call concrete `TantivyIndex` methods. AGENTS.md forbids compatibility shims, and every consumer is in-repo. Therefore: **Quill defines its own minimal public API** (§9.6) shaped by what consumers actually do — not a tantivy look-alike. Consumers (`fsfs/src/runtime.rs`, `frankensearch/src/index_builder.rs`, facade re-exports) are ported directly (§16.2). The `LexicalSearch` trait needs **zero changes**; fusion needs **zero changes**.

### 3.4 The cass-compat carve-out

`cass_compat.rs` exists so frankensearch can open/write the external CASS tool's on-disk **tantivy-format** indexes (schema hash v8, `<base>/index/v8/`). That is an interop contract with a foreign tool's storage format — reimplementing tantivy's on-disk format is a non-goal (it is precisely the generality Quill escapes). Disposition: `cass-compat` becomes an opt-in feature that carries the tantivy dependency for exactly this module; the default `lexical` path carries no tantivy. The CASS *semantics* (analyzers, query grammar) are still implemented natively in Quill (§8.2–8.3) because fsfs's own indexes want them — only the *foreign on-disk format* stays tantivy-backed. When the external cass tool migrates to FSLX (out of scope here, tracked as a follow-up bead), the feature is deleted.

---

## 4. SOTA Distillation: The Field, and Where Quill Places Its Bets

Adopt / adapt / reject decisions over the indexing-engine literature and practice. (Alien-graveyard catalog references in parentheses.)

### 4.1 Ingest pipeline

| Source | Core idea | Verdict |
|---|---|---|
| **SPIMI** (Heinz & Zobel single-pass in-memory indexing) | Accumulate postings in memory per block, spill sorted runs, merge | **Adopt the skeleton** — it is the only sane shape for bounded-memory ingest — but replace the per-token dictionary probe with columnar accumulation + deferred radix partition (Q2). |
| **Radix hash join / cache-shaped partitioning** (graveyard 8.14: per-thread histograms → prefix sums → disjoint write ranges; software write-combining; skew handling) | Make random access sequential by partitioning to cache-sized buckets | **Adopt as the flush kernel**: `(term_id, doc, pos)` triples radix-partitioned by term_id in 1–2 passes; per-partition posting build is then purely sequential. Histogram+prefix-sum parallelizes across cores with zero contention. |
| **Lucene/tantivy stacker (hash-per-term + exponential-unrolled posting chains)** | Amortized per-token appends into term-keyed arenas | **Adapt for the delta segment only** (§6.3), where per-token random access is bounded by delta size (L2-resident by budget); **reject for bulk flush** where the columnar path wins on cache behavior and parallelism. This is a measured bet: E1 carries an explicit A/B bead (hash-accumulate vs columnar-radix) with the loser recorded in NEGATIVE_EVIDENCE (§19 quill-e1). |
| **Thread-per-core / share-nothing (Seastar lineage; graveyard 3.13)** | Eliminate shared mutable state; shard by data, not by lock | **Adopt** as shard-per-worker ingest with disjoint docid ranges (Q1 precondition). Work-stealing at the *batch* level (rayon / asupersync), share-nothing at the *shard* level. |
| **simdjson-school byte classification** | SIMD table lookups classify bytes in bulk; structural indexes | **Adapt**: tokenization = SIMD classify (alnum/CJK-lead/other) + boundary extraction + in-register ASCII lowercasing (§8.1). Full-scan classify is the right shape (memchr-style early-exit fusion is the known-rejected pattern — bd-5hz0 — but tokenization visits every byte regardless). |
| **LiveGraph-style append-only hot logs** | Sequential-only writes for bursty hot keys | **Adopt the lesson** for the delta segment's per-term chains: bump-arena chains, bounded, sealed into columnar form — never a long-lived structure. |

### 4.2 Storage & compression

| Source | Core idea | Verdict |
|---|---|---|
| **Frame-of-Reference / bitpacked blocks (Lucene FOR, PForDelta family)** | Fixed-size blocks (128) of delta-encoded docids bitpacked to the block's max bit-width; freqs likewise | **Adopt** as the primary posting format (§10.4): simple, vectorization-friendly, and — critically — directly concatenable because each self-delimiting block stores an absolute `first_doc`. Ordinary Q1 merge copies block bytes unchanged; no posting rebase or rewrite occurs. The canonical scalar unpacker is the production route; the bit-identical-output `wide` u32x8 candidate remains behind the internal differential/bench surface because the 2026-07-17 same-binary gate rejected a global wide-for-all policy. |
| **Elias–Fano posting lists** | Succinct monotone sequences with O(1) skip | **Reject for 1.0** (encode cost on the ingest path, marginal query win at this corpus scale); **retry condition**: query-side profile shows block-skip dominating on >10⁶-doc corpora. |
| **Roaring bitmaps (7.14)** | Density-adaptive containers for very common terms | **Adapt**: postings blocks whose local density exceeds a threshold switch to bitmap containers (§10.4); tombstone sets are roaring-style from day one (§10.5). |
| **FST term dictionaries (7.17, Lucene lineage)** | Minimal automata compress sorted term dicts massively | **Adapt as the cold tier**: fresh/hot segments use prefix-compressed sorted blocks (build-cheap, §10.2); the optional compaction tier can FST-ify large sealed dictionaries. **Not on the ingest hot path.** |
| **Minimal perfect hashing for static keysets (7.15)** | O(1) exact term lookup on sealed dicts | **Reject for 1.0** (prefix-block binary search + two-level index is within noise at our dict sizes; MPH kills prefix/range/glob scans which we need); retry if term-lookup profiles ≥0.1% self-time on real query mixes. |
| **Fieldnorm quantization (Lucene 1-byte norms)** | Doc length compressed to 256 buckets via lookup table | **Adopt tantivy's exact table** (§8.4) — 1 byte/doc/field, and it is *required* for score-parity with the oracle. |
| **LSM merge theory (Dostoevsky 8.6; tiered vs leveled)** | Merge policy trades write vs read vs space amplification | **Mostly dissolved by Q1**: when merge is concat, write amplification of merging collapses and the policy reduces to "bound segment count + bound tombstone debt" (§12.2). Adopt the *vocabulary* (tiering by size class) not the machinery. |

### 4.3 Query evaluation

| Source | Core idea | Verdict |
|---|---|---|
| **WAND / Block-Max WAND (Ding & Suel), MaxScore** | Skip non-competitive docs via per-term/per-block score upper bounds | **Adopt**: block-max metadata (max weighted term freq per block, 1 byte quantized) in FSLX; Argus uses MaxScore for short disjunctions, BMW above a clause-count/list-length threshold; exact-count mode disables pruning (§9.2). Conformance note: pruning must be provably rank-safe for top-k (it is, by construction) so oracle rank-parity holds. |
| **Galloping / SvS intersections** | Skip-heavy conjunction evaluation | **Adopt** for Must-clauses and phrase pre-filtering. |
| **AMAC / coroutine interleaving (8.17)** | Hide DRAM misses by interleaving independent lookups | **Defer to a post-1.0 lever bead**: batch term-dict probes across query clauses. Requires the batched shape anyway present in Argus's planning; measured entry criteria per graveyard contract (LLC-miss profile first). |
| **Impact-ordered / static index pruning** | Reorder postings by impact for early termination | **Reject**: breaks docid-order invariants Q1 depends on, complicates conformance; rank-safe BMW gets the win without reordering. |
| **Learned/adaptive indexes (8.4, 8.8)** | Data-dependent structures | **Reject** for 1.0 — determinism and conformance first. |

### 4.4 Runtime & verification

| Source | Core idea | Verdict |
|---|---|---|
| **FoundationDB-style deterministic simulation (6.20)** | The whole system under a deterministic scheduler with fault injection | **Adopt via LabRuntime** (§15.5): ingest/merge/crash interleavings are seed-replayable. This is the house specialty; tantivy cannot do this. |
| **Crash-only design (3.5)** | Recovery *is* the startup path | **Adopt**: opening an index = recovering its read-only snapshot (§11.4). The manifest points only at sealed, checksummed artifacts; anything else is invisible garbage retained by readers and GC'd only after writer admission. |
| **Property-based + metamorphic testing (6.12; gauntlet skill)** | Semantics-preserving transforms must preserve results | **Adopt** as the core of the conformance gauntlet (§15.3). |
| **E-graph / certified rewrites (6.6)** | Prove optimizations semantics-preserving | **Adapt the discipline, not the machinery**: every Argus pruning path carries a rank-safety argument and a differential fixture (pruned vs exhaustive) — §15.2. |

---

## 5. The Language Contract: Exactly What "Same Results" Means

Quill's conformance target is not "tantivy, bug for bug"; it is the **observable behavior of frankensearch's lexical paths**, pinned as a versioned contract (`docs/contracts/quill-language-contract.md`, produced in G0):

1. **Analyzer contract** (§8.1–8.2): for every analyzer in the family, token-stream equality (text, position, offsets) with the shipping implementations, verified byte-parity on the fixture corpora + fuzzed inputs. The shipping scalar implementations are the *specification*; tantivy's originals are the *lineage*.
2. **Query grammar contract** (§8.3): the default lenient parser (2 fields, title boost 2.0, 10k-char truncation, quoted phrases, `-negation` handled upstream) and the CASS boolean grammar (AND/OR/NOT with OR-tighter precedence, phrases, glob wildcards, range filters, CJK bigram expansion) — specified as grammars with golden parse-tree fixtures, differentially validated against the tantivy-backed implementations per query class.
3. **Scoring contract** (§8.4): BM25 with k1=1.2, b=0.75, Lucene idf `ln(1+(N−n+0.5)/(n+0.5))`, tantivy's fieldnorm quantization table, multi-field weighted sum, f32 accumulation discipline pinned. Conformance classes: **RankExact** (identical top-k order under the deterministic tie-break) is the gate for single-segment/equal-stats configurations; **ScoreEpsilon** (|Δscore| ≤ 1e-4 relative, identical result *sets*, rank flips only inside epsilon-tied groups) is the classified divergence budget for cross-segment-layout comparisons where tantivy's own scores vary by segment geometry. Every divergence in the differential corpus must be classified into exactly one of {RankExact-pass, ScoreEpsilon-tie, **DivergenceRegister entry**} — unclassified divergence blocks the gate.
4. **Behavioral contract**: upsert (delete-by-id then add) visibility; delete semantics; `doc_count` accuracy; empty/whitespace query ⇒ empty results; limit/offset pagination; deterministic tie order (§8.4); snippet output fixtures; error taxonomy.
5. **Out-of-contract**: anything in §3.2, plus tantivy-internal observable details (segment counts, file layouts, score values beyond epsilon class) that no frankensearch consumer depends on.

---

## 6. Architecture: Scribe, Grimoire, Quiver, Argus, Keeper

Quill is one engine crate (`crates/frankensearch-quill`) with five named subsystems and a conformance companion. Data flow:

```
                        ┌──────────────────────────────── SCRIBE (ingest) ───────────────────────────────┐
 IndexableDocument ──►  │ shard router (docid-range lease per shard)                                     │
                        │   └─► per-shard worker (share-nothing):                                        │
                        │        SIMD tokenize/fold (§8.1) ─► local intern (term → local u32)            │
                        │        ─► columnar (term_id, doc_ord, pos) triples in bump arenas              │
                        │        ─► ALSO applied to the live DELTA segment (searchable immediately)      │
                        │   flush trigger (arena budget) ─► radix partition by term ─► build mini-seg    │
                        └───────────────┬────────────────────────────────────────────────────────────────┘
                                        │ sealed mini-segments (FSLX §10)
                        ┌───────────────▼──────────────── KEEPER (lifecycle) ─────────────────────────────┐
                        │ manifest (versioned, checksummed, atomically published §11.4)                   │
                        │ tier policy: concat-merge same-tier runs (Q1, §7/§12) · tombstone compaction    │
                        │ durability: FileProtector .fec sidecars · crash-only recovery on open           │
                        └───────────────┬──────────────────────────────────────────────────────────────---┘
                                        │ epoch-published snapshot {delta, sealed segments, stats}
                        ┌───────────────▼──────────────── ARGUS (query) ──────────────────────────────────┐
                        │ parse (§8.3) ─► plan (clause classes, stats) ─► per-segment BM25 eval:          │
                        │   GRIMOIRE term dict probes ─► QUIVER posting cursors (FOR blocks, block-max)   │
                        │   MaxScore/BMW top-k (§9.2) ─► k-way segment merge ─► deterministic tie order   │
                        │ snippets (§9.5) · counts · id-set mode                                          │
                        └──────────────────────────────────────────────────────────────────────────────---┘
```

### 6.1 Scribe — the ingest pipeline (bets Q2, Q4; the throughput headline)

- **Shard-per-worker, share-nothing.** An ingest session (bulk build or watch batch) fans documents across `W` shard workers (default: physical cores, clamped by config). Each worker holds: a docid-range lease (§7), a local string-interner (`ahash` map term-bytes → local term id, arena-backed), columnar triple buffers, and the live delta-segment writer for its range. **No shared mutable state**; the only cross-worker artifact is the manifest update at seal time.
- **SIMD front-end.** Tokenization + case folding run as `wide`-vectorized byte classification with scalar tails (§8.1). Emits token spans directly against the borrowed input; no per-token `String` allocation on the hot path (`CompactString` only at dictionary-insert time for new terms).
- **Columnar accumulation.** Per token: append `(local_term_id: u32, doc_ord: u32, pos: u32)` to SoA arenas (position stream only for position-indexed fields; positions are u32 — a 2 MiB source file exceeds 65k tokens, so u16 would silently truncate phrase positions; tantivy likewise uses u32). Per document: one doc-length entry (fieldnorm byte) per field. This is a pure sequential write pattern — the cache-hostile per-token dictionary-chain append of the Lucene school is confined to the (small, L2-resident) delta segment.
- **Flush = radix + build.** When arenas hit the shard budget (default 64 MiB, config `scribe_shard_budget_bytes`): histogram + prefix-sum radix partition by `term_id` (1 pass at ≤2¹⁶ terms, 2-pass MSD beyond), then per-partition sequential posting-block build (delta+FOR bitpack, block-max computation), term dict build (sort local terms by bytes, prefix-block encode), doc-length column, docid-map section — one sealed FSLX mini-segment, one file, fsync'd by Keeper.
- **Batch API discipline.** `index_documents` is the primary path (fsfs already batches); single-doc `index_document` is a batch of one into the delta. Bulk build (index_builder) uses a high-watermark pipeline: tokenize/accumulate workers feed a sealer via two-phase channels — sealing (compression + fsync) overlaps tokenization of the next arenas.

### 6.2 Grimoire — term dictionaries

- **Hot format (all segments at 1.0):** sorted, field-namespaced prefix-compressed term blocks (canonical 4096-byte target including the entry-count header; a legal oversized term occupies one singleton block): restart-point layout (full composite key every R entries, longest-prefix suffix encoding between), two-level index (block-first-keys with offsets relative to the blocks-area start, plus a bounded in-memory restart directory built on open), per-term entry = {doc_freq: u32, postings offset/len: u64, conditional positions offset/len: u64, block-max summary offset/len: u64}. Terms are arbitrary bytes through the language-contract 65,530-byte ceiling. Positions presence comes from the schema; v1 has no per-entry flags word. O(log B + R) exact lookup with R=16, allocation-free amortized O(1) sequential scan, and natural half-open per-field prefix/range iteration (glob support §9.4).
- **Cold option (post-1.0 lever):** FST-encode dictionaries during deep compaction for large sealed segments; entry criteria: dictionary residency shows in memory profiles. Not on any 1.0 path.
- Term ids are **segment-local ordinals** (dense u32 in sorted order) — cross-segment queries join on term *bytes* via per-segment probes, exactly like Lucene/tantivy; no global dictionary to contend on (Q1 keeps ingest shards independent).

### 6.3 The delta segment — searchable-while-indexing (bet Q3)

- Per shard, a small mutable in-memory segment: term → unrolled exponential posting chains in a bump arena (the Lucene-school structure, correctly scoped: bounded by `delta_budget_bytes`, default 8 MiB per shard, L2/L3-resident).
- **Visibility:** every applied batch atomically swaps the shard's published delta snapshot (Arc-swap epoch pattern; readers never lock). A search snapshot = manifest's sealed segments + the current delta snapshots. `commit` no longer gates searchability — it seals + fsyncs + publishes (durability). This inverts tantivy's model and is the watch-mode latency win: p95 update→searchable becomes microseconds-scale in-memory work.
- **Sealing:** delta budget breach or explicit commit converts the delta into a standard FSLX mini-segment through the same radix/build path (the chains are drained in term-sorted order, so this is a sequential pass).
- **Honest cost note:** the delta is a second read-path implementation (chain cursors vs block cursors). Argus abstracts posting cursors behind one trait; conformance suites run every query class against delta-resident, sealed, and mixed states (§15.3).

### 6.4 Schemas without a schema engine

Quill compiles frankensearch's actual field shapes into **schema descriptors** (const tables, not dynamic `Value` dispatch): field id, kind (Keyword | Text{analyzer, positions} | StoredOnly | I64{indexed, fast} | U64{fast}), stored flag. The default schema and the fsfs chunk-policy schema are the two shipped descriptors; the CASS *semantic* fields (§8.2) form a third. Adding a field kind is a code change with a format-version note — deliberate rigidity in exchange for monomorphic ingest loops. `metadata_json` remains stored-opaque bytes exactly as today; `content` storage policy is per-descriptor (stored for snippet generation in the default schema; canonical-storage-backed for fsfs contexts that have it, mirroring the CASS pattern §2.4).

### 6.5 Naming

Subsystems are module-level names inside `frankensearch-quill`: `scribe`, `grimoire`, `quiver`, `argus`, `keeper` — plus `contract` (analyzer/query/scoring contract types shared with the gauntlet). The conformance harness lives in `crates/frankensearch-quill-gauntlet` (dev-only crate, never published, carries the tantivy oracle dep behind its own feature) — §15.1.

---

## 7. The Merge=Concat Invariant (Bet Q1) — Statement and Obligations

**Invariant Q1 (docid-range discipline).** Every segment S carries a half-open global docid interval `range(S) = [lo, hi)` in its header. At all times, the manifest's live segments have pairwise-disjoint ranges, and every posting, tombstone, doc-length entry, and docid-map entry in S refers only to docids in `range(S)`.

**Allocation.** Keeper owns a monotone docid allocator (persisted high-watermark in the manifest). Ingest shards lease disjoint contiguous blocks (e.g., 64k docids per lease, re-leased on exhaustion; leases are per-shard *session*, reused across watch-mode batches; the unused tail of a lease is *burned* at session end, never reused — docids are cheap, correctness is not). Upserts allocate a *new* docid and tombstone the old one (found via the id-map); docids are never reused. **Docid width:** postings encode docids as u32 (manifest/header fields are u64 for future-proofing). At the design scale (≤ ~1M live docs, watch-mode churn) monotone-with-burn consumes u32 space in thousands of years of realistic updates; a renumbering *deep compaction* mode is reserved as the u32-exhaustion escape hatch (format-registry note, never expected to run).

**Theorem (merge is concatenation).** For segments S₁…Sₙ with pairwise-disjoint ranges, sorted by `lo`, the merged segment M with `range(M) = [lo₁, hiₙ)` has, for every term t: `postings_M(t) = postings_S₁(t) ⧺ … ⧺ postings_Sₙ(t)` — already docid-sorted, no renumbering. *Proof sketch:* postings within each segment are docid-ascending; ranges are disjoint and ordered; concatenation of sorted sequences over ordered disjoint intervals is sorted. ∎

**What merge actually does (streaming, no posting decode):** k-way merge the n term dictionaries by term bytes (galloping over prefix blocks); for each term, copy self-delimiting posting blocks verbatim and in source order. Because FOR blocks store `first_doc` as an absolute u32 + internal deltas, "rebase" is unnecessary (docids are already global). A source stream's partial VINT block remains legal at an interior seam, so ordinary merge never fills a tail or shifts every later 128-posting boundary. Per-term work is recomputing aggregate stats (sum of doc_freqs) and re-emitting block-skip/block-max metadata in posting-block order; relative block offsets are adjusted for the merged stream while `first_doc` and bound fields are preserved. Doc-length columns and docid maps concatenate. Tombstone sets union. Each freshly sealed leaf segment contributes at most one partial block per term. A merge output preserves exactly the leaf partials already present in its inputs, so it may contain multiple partial blocks but creates none. Optional reblocking is an explicit deep-compaction lever, never hidden merge work. Measured expectation: merge cost ≈ sequential read + sequential write of the inputs — I/O-bound, not CPU-bound. This is the structural kill of tantivy's merge tax and of the entire cooldown/bounded-merge apparatus in `cass_compat.rs`.

**Two rules that keep intervals disjoint under shard interleaving** (fresh-eyes amendment): because shards acquire lease blocks alternately, a shard's later leases interleave with other shards' blocks in docid space. Without discipline, a segment spanning two leases — or a same-shard merge skipping an interleaved foreign segment — would produce a covering interval that overlaps another live segment. Therefore: **(R1) seal-at-lease-boundary** — a sealed segment's docid range is always a subinterval of a single lease block; lease exhaustion forces a segment cut even if the arena budget isn't reached. **(R2) merge only bound-consecutive runs** — the tier policy sorts live segments by `lo` and merges only consecutive runs in that order; cross-shard merges are the normal case and are exactly as cheap (concat is concat). Together R1+R2 preserve strict pairwise-disjoint covering intervals for the life of the index (burned lease tails are merely unused docids *inside* an interval, which was always legal).

**Obligations (checked by the gauntlet):**
- **Q1-OB1**: manifest validation rejects overlapping ranges (crash-recovery property tests §15.5); merge preconditions assert bound-consecutive inputs (R2) and seal asserts single-lease spans (R1).
- **Q1-OB2a**: merge(S₁…Sₙ) has decoded postings, aggregate stats, and query results identical to indexing the concatenated document streams into one segment with the same docid assignments (metamorphic fixture, including `df=100 + df=300`, §15.3).
- **Q1-OB2b**: every input posting block appears byte-identically and in source order in the merged stream; merge creates no new posting fragments (raw-block witness + arbitrary merge-schedule property fixture).
- **Q1-OB3**: BM25 global stats (N, avgdl, doc_freq) are snapshot-level aggregates over live segments minus tombstones (§8.4), so merging never changes scores — differential fixture: query results invariant under any merge schedule.
- **Q1-OB4**: compaction (the only re-encoding path) rewrites a segment to drop tombstoned docs **preserving surviving docids** (no renumbering — gaps are fine; FOR deltas absorb them). Triggered by tombstone-density threshold (default 20%, config), measured at realistic densities per the bd-6m8p lesson.

---

## 8. Semantics Contracts (Analyzer, Query, Scoring)

### 8.1 Default analyzer — SIMD with byte-parity

Specification = the shipping `FrankensearchTokenizer` (`lexical/src/lib.rs:466`): split on non-alphanumeric (`char::is_alphanumeric`), lowercase (ASCII fast path; full `char::to_lowercase` fallback), no length filter on the default path. Quill implementation: `wide` u8x16/u8x32 classification of ASCII alnum + high-bit detection; runs entirely in-register for pure-ASCII spans (the overwhelming case for code corpora); any high-byte falls back to the scalar char-walk *for that span* — preserving byte-exact semantics including Unicode alphanumerics and multi-char lowercasing. **Gate:** token-stream equality (text/position/offset) vs the shipping tokenizer on: fixture corpora, the existing parity-test cases, property-fuzzed inputs (mixed scripts, boundary alignments at SIMD lane edges), and the CJK fixtures. The existing `tokenizer_char_walk_ab` bench extends to a three-way A/B (legacy tantivy chain / shipping fused scalar / Quill SIMD).

### 8.2 CASS analyzer family — native

`CassTokenizer` (ASCII-alnum runs hyphen-joined + CJK runs), `HyphenDecompose` (compound + parts at same position), `CjkBigramDecompose` (overlapping bigrams, unigram singles), `CassNormalizeAndLimit` (ASCII lowercase + 256-byte limit), edge n-grams, preview builder — reimplemented as Quill token-pipeline stages with the **existing slow oracles** (`cass_char_walk_slow`, `cass_generate_edge_ngrams_slow`, etc.) as parity fixtures. These analyzers serve fsfs-native indexes wanting CASS semantics; the foreign-format interop remains carved out (§3.4). Positions: `HyphenDecompose` emits same-position tokens ⇒ Quill's position model supports position-duplicate tokens (u32 position stream with duplicates allowed; phrase matching treats same-position alternatives as OR — matching tantivy's behavior, pinned by fixtures).

### 8.3 Query surface

- **Default lenient parser:** grammar = whitespace-separated terms + quoted phrases over fields `[content, title(boost 2.0)]`; analyzer applied per-field; unparseable fragments dropped with a warning (lenient); 10k-char truncation at a char boundary. Multi-term default combinator pinned by differential fixtures against the tantivy parser on the recorded query-class corpus (identifier / short-keyword / natural-language slices from `search_quality_harness`) — the *fixtures*, not tantivy's source, are the contract.
- **Boolean algebra:** Must / Should / MustNot with nesting; CASS grammar (AND `&&`, OR `||` binding tighter, NOT/`-` via Must(All)+MustNot) as a parser producing the same algebra.
- **Phrase queries:** exact-position adjacency (slop 0 — the only mode used), evaluated via positions streams with galloping conjunction pre-filter.
- **Range queries:** i64 bounds over indexed numeric fields (`created_at` pattern) — implemented as a per-segment sorted numeric column scan-or-binary-search producing a doc filter (no generic numeric index machinery; the used surface is filter-shaped).
- **Glob term queries:** `CassWildcardPattern`{Exact, Prefix, Suffix, Substring, Complex} compiled to a glob automaton evaluated over Grimoire: Prefix = dictionary range scan; Suffix/Substring/Complex = bounded dictionary scan with SWAR memmem-style candidate rejection; expanded to a bounded term disjunction (limit + deterministic order, matching tantivy's expansion behavior on the fixtures; expansion-limit divergences → Divergence Register).
- **AllQuery, TermQuery (keyword fields), DocSet/id-set mode, (TopDocs, Count) tuple semantics** — as §3.1.

### 8.4 Scoring and determinism

- **BM25:** `idf(t) = ln(1 + (N − n_t + 0.5)/(n_t + 0.5))`; `tf_part = f·(k1+1)/(f + k1·(1−b+b·|d|/avgdl))` with k1=1.2, b=0.75; `|d|` from **tantivy's exact 1-byte fieldnorm table** (decode table vendored as a const, cited to tantivy 0.26.1 source, with a unit test pinning all 256 entries); per-field scores × field boost, summed. N/avgdl/doc_freq computed at **snapshot level** (sum over live segments; tombstones excluded from N at snapshot build, doc_freq left segment-exact — matching tantivy's delete behavior class; any residual divergence classified §15.6).
- **Float discipline:** f32 accumulation in a fixed clause order (parse order), no FMA-dependence in the contract (scalar and SIMD scoring paths must agree bit-for-bit on the conformance corpus — SIMD is used for decode/gather, scores accumulate scalar in contract mode; a `fast_scoring` config may relax to ScoreEpsilon class, default off until G3 evidence).
- **Deterministic tie-break:** total order = (score desc via `total_cmp`, docid asc); surfaced order additionally maps docid→`DocId` only after selection (two-phase, per house pattern). Since docid assignment is deterministic given ingest order (§1.5), end-to-end determinism holds; the conformance comparator canonicalizes tantivy-side ties by the same key before diffing.

---

## 9. Argus: Query Evaluation

### 9.1 Read path shape

Fully synchronous internals (mmap'd sealed segments + in-memory delta snapshots), wrapped in immediately-ready boxed futures for the `LexicalSearch` trait — satisfying the fusion `poll_immediate` requirement by construction. Per-query: resolve snapshot (Arc clone, no locks) → parse/plan → per-segment scored iterators → global top-k merge → id materialization → optional metadata hydration (Quill keeps `fusion_metadata_is_deferred = true` semantics and the hydrate path, matching the winning tantivy pattern the fusion crate already exploits).

### 9.2 Top-k kernels

- **Exhaustive kernel** (BinaryHeap guard, NaN-safe `total_cmp`, two-phase materialization — house pattern from `index/src/search.rs`) is the reference implementation and the exact-count path.
- **MaxScore** for disjunctions of 2–8 terms (the dominant frankensearch shape); **Block-Max WAND** above thresholds (clause count, list length) using FSLX block-max bytes. Both provably rank-safe for top-k; the gauntlet enforces pruned≡exhaustive on every fixture (§15.2). Count semantics: `(TopDocs, Count)`-tuple requests run counting mode (pruning off or count-corrected); count-free gate becomes the default for plain term queries structurally.
- **Segment parallelism:** rayon fan-out across segments above a work threshold (house `PARALLEL_THRESHOLD` philosophy); per-segment results merge with the deterministic key. Off in `deterministic_ingest`/Lab profiles unless the merge is order-insensitive (it is — merging is over a total order).

### 9.3 Collectors

One kernel, three modes: top-k (+offset), top-k+exact-count, id-set (DocSetCollector semantics for hydrate). `doc_count()` = snapshot live-doc count (O(1) from manifest stats minus tombstone cardinalities).

### 9.4 Glob/range machinery

Grimoire range scans + glob automaton (§8.3) expand to bounded disjunctions before scoring; range filters compile to per-segment docid predicates fused into cursors (filter-first, matching the CASS Must-filter shape).

### 9.5 Snippets

Native generator: given stored/canonical content + matched terms (post-analysis), find best window ≤ `max_chars` (200 default) by term coverage (greedy window over token hits — parity with tantivy's generator asserted on fixtures at the *output* level: same defaults, same tags `<b>/</b>`, documented divergence class if window choice differs on ties → Divergence Register with consumer impact "cosmetic").

### 9.6 Public API sketch (consumer-shaped, not tantivy-shaped)

```rust
pub struct QuillIndex { /* manifest, snapshots, keeper handle */ }
impl QuillIndex {
    pub fn create(path: &Path, schema: SchemaDescriptor) -> SearchResult<Self>;
    pub fn open(path: &Path) -> SearchResult<Self>;            // = recover (§11.4)
    pub fn in_memory(schema: SchemaDescriptor) -> SearchResult<Self>;
    pub async fn delete_document(&self, cx: &Cx, doc_id: &str) -> SearchResult<()>;
    pub async fn delete_all(&self, cx: &Cx) -> SearchResult<()>;
    pub fn path(&self) -> Option<&Path>;
    pub fn doc_count(&self) -> usize;
    pub fn search_with_snippets(&self, cx: &Cx, query: &str, limit: usize,
                                cfg: &SnippetConfig) -> SearchResult<Vec<LexicalHit>>;
    pub fn search_doc_ids(&self, cx: &Cx, query: &str, limit: usize)
                                -> SearchResult<Vec<LexicalIdHit>>;
    pub fn search_paginated(&self, cx: &Cx, query: &str, limit: usize, offset: usize)
                                -> SearchResult<PaginatedHits>;   // hits + exact total_count
                                                                  // (replaces (TopDocs, Count) tuple surface)
    pub fn segment_stats(&self) -> SegmentStats;               // ops/status surface
    pub async fn compact(&self, cx: &Cx, policy: CompactionPolicy) -> SearchResult<CompactionReport>;
}
impl LexicalSearch for QuillIndex { /* §3.1 duties, sync-in-async read paths */ }
```
`LexicalHit`/`LexicalIdHit`/`SnippetConfig`/`QueryExplanation` move to (or are re-defined in) engine-neutral form so fsfs/facade port without tantivy types (§16.2). `force_merge`/`optimize_if_idle`/merge-policy plumbing has **no Quill equivalent** — `compact` with a policy covers the real need; consumers are ported off the tantivy-shaped calls.

---

## 10. FSLX: The On-Disk Format (Normative Contract)

### 10.1 Files

A Quill index directory contains: `MANIFEST` (+ `MANIFEST.prev` for the two-slot publish protocol §11.4), `seg-<hex16>.fslx` (one file per sealed segment, immutable once referenced), optional `.fec` repair sidecars (durability crate), and nothing else. Segment ids are random u64 (collision-checked); temp files are `.tmp-` prefixed and are garbage by definition.

### 10.2 Segment file layout (`FSLX`, version 1)

```
[0..8)    magic "FSLXSEG\0"
[8..12)   format_version: u32 = 1
[12..16)  header_len: u32
[16..H)   header: {
            segment_id: u64, schema_id: u64 (descriptor hash),
            docid_lo: u64, docid_hi: u64,            # Q1 range
            doc_count: u32, tombstone_count_at_seal: u32,
            section_table: [ {section_kind: u16, flags: u16, offset: u64, len: u64, xxh3: u64} ],
            created_unix_s: i64, engine_version: u32
          }
[H..H+4)  header_crc32
sections (each 64-byte aligned, order fixed by kind):
  TERMDICT   prefix-compressed term blocks + two-level index (§6.2)
  POSTINGS   FOR/bitpacked doc blocks (+freq blocks) per term (§10.4)
  POSITIONS  optional; per-term position streams (vint blocks), present iff schema indexes positions
  BLOCKMAX   per-block (max_freq quantized-up u8, min_fieldnorm u8) bound pairs + skip entries
             (impact computed at query time with live idf/avgdl — see §10.4)
  DOCLEN     fieldnorm bytes per (field, doc) — 1 byte each
  IDMAP      docid → DocId (CompactString bytes): offsets + unseeded canonical-content
             xxh3-64 hashes + blob (registry wire contract is normative; §10.3)
  IDHASH     open-addressed hash (xxh3-64 of DocId bytes — stable, versioned; NEVER ahash,
             whose hashes are not stable across versions/platforms) → docid, for upsert/delete probes
  NUMERIC    optional sorted (value:i64, docid:u32) column per indexed numeric field
  STOREDMETA optional stored-field blob (metadata_json bytes; content iff descriptor stores it)
  STATS      per-field: total_tokens: u64, doc_count: u32 (avgdl inputs)
file trailer: file_xxh3: u64, trailer_crc32: u32
```
All integers are little-endian except the TERMDICT key's big-endian `field_ord` prefix, whose byte order preserves field-major lexical sort. All sections are independently checksummed (xxh3-64, verified lazily per-section on first touch + eagerly on `verify`); the durability `.fec` sidecar protects the whole file with a separate complete-source xxh3 witness. The trailer hash covers only the bytes before the trailer, so the encoder derives both hash domains in one traversal rather than equating them.

### 10.3 IDMAP/IDHASH dissolve the ord_table

Docid→`DocId` is a direct section (materialization = offset lookup + slice); `DocId`→docid probes (upsert/delete) hit IDHASH then verify against IDMAP. No JSON sidecar, no docstore fallback, no reopen penalty (the `reopen_id_materialize` bench class becomes trivially flat).

### 10.4 Posting encoding

Every posting block begins with the self-delimiting common header `{block_kind: u8, posting_count: u8, payload_len: u16 LE}`. FOR/BITMAP blocks have `posting_count=128`; VINT partials have `posting_count=1..=127`; readers require `payload_len` to match the exact consumed payload. Fresh seals emit full blocks with an absolute `first_doc: u32` and 127 `delta-1` values bitpacked at the minimal `doc_width: u8` (`0..=32`). Frequencies use `freq_kind: u8` (`0=all-one`, `1=bitpacked`), followed for kind 1 by the minimal `freq_width: u8` (`1..=32`) and packed `freq-1` values. A final partial block is vint-encoded. Concat merges may preserve those VINT partials at interior seams; readers consume blocks through the term's `postings_len` and validate summed counts against `doc_freq`. The fresh writer selects a 512-bit bitmap container when a full block's inclusive docid span is <512 (roaring lesson, §4.2); bitmap bits are LSB-first within each byte. Block-max: per block, the pair `(max_freq quantized up via monotone u8 table, min_fieldnorm byte)` — **not** a precomputed impact scalar, because `tf_part` depends on snapshot-level `avgdl`, which changes as the index grows (a seal-time impact bound would be unsound when avgdl later increases). Argus computes the block's impact upper bound at query time from the stored pair with live idf/avgdl: `tf_part` is increasing in freq and decreasing in `|d|`, so (max_freq, min_fieldnorm) dominates every doc in the block. Ordinary merge re-emits block-max entries in posting-block order, adjusting only their relative block offsets. Decode kernel: the canonical scalar reservoir is the production reader. A portable `wide` u32x8 implementation with bit-identical decoded output remains available to fixture `quiver-postings-bitpack-scalar-wide-v1` and same-binary benches, but the 2026-07-17 measurement rejected routing every width through it; any future per-width dispatcher requires its own admissible gate.

### 10.5 Tombstones live in the MANIFEST, not the segment

Deletes against sealed segments append to a per-segment tombstone set stored in the manifest generation (roaring-style serialized containers). Segments stay immutable; snapshot construction pairs segment ⊕ current tombstones. Compaction (§7 Q1-OB4) folds tombstones in and drops them.

### 10.6 MANIFEST (version 1)

```
magic "FSLXMAN\0" | format_version | generation: u64 | docid_high_watermark: u64
schema_id | engine_version (major:u8, minor:u8, patch:u16 packed into u32) |
flags: u32 | segment_count: u32 |
segment entries[segment_count] [{segment_id, seal_seq, file_len, file_xxh3, docid_lo, docid_hi,
doc_count, tombstones: roaring-bytes}] | field_count: u32 |
stats rollup[field_count] | crc32
```
Segments are ordered by ascending `docid_lo`; stats are ordered by ascending `field_ord`. Adjacent generations preserve retained segment metadata, grow retained tombstones monotonically, allocate every new `seal_seq` above the prior non-empty maximum, and leave at-seal stats unchanged when the immutable segment set is unchanged. An explicit empty generation starts a new seal-sequence epoch because no older segment remains queryable. Publish protocol in §11.3 and the exact crash windows in the format registry §6.2. A retry after a claim or rename failure reuses a durable temp only when the directory entry is a no-follow regular file whose bytes exactly match the proposal; an unexpected temp fails closed without overwrite. `KeeperWriter` retains the cross-process `LOCK` admission for every public mutation and acquires an O_EXCL generation claim around each two-slot publication; the in-process mutex remains the cancellation-safe serializer within one process. **Format registry:** every FSLX/MANIFEST change lands a row in `docs/contracts/fslx-format-registry.md` (version, change, migration note, gauntlet fixture id) — G0 creates the file with v1 rows.

---

## 11. Concurrency, Durability, and Crash-Only Recovery

### 11.1 Write path choreography (asupersync)

Ingest session = an asupersync region: shard workers (tasks) ← two-phase mpsc feeding batches; sealer task per shard; Keeper's manifest actor serializes publishes behind a cancel-aware `asupersync::sync::Mutex`. Cancellation drains: a cancelled session seals nothing and leaks no in-memory state (arenas dropped); any durable temp files remain inert until an admitted writer performs exact-name, grace-period GC — verified under LabRuntime with forced cancel points (§15.5). No detached threads; merges/compactions are region-scoped background tasks whose abandonment is always safe (outputs unreferenced until manifest publish).

### 11.2 Blocking discipline

Sealing (compress+write+fsync), merges, and compaction run under `spawn_blocking`; pure-CPU batch work inside rayon scopes. The trait read paths stay sync-in-async (§9.1) — bounded work, no I/O waits beyond page faults on mmap'd sections (documented, same posture as FSVI reads today).

### 11.3 Durability

`commit(cx)`: seal current deltas → fsync segment files → write MANIFEST(gen+1) to temp → fsync → atomic rename over `MANIFEST` (keeping `MANIFEST.prev`) → fsync dir. Optional (`durability` feature): `FileProtector` emits `.fec` for new segments and the temp MANIFEST, revalidates its complete-source witness against the exact immutable snapshot it encodes, rotates source/sidecar pairs together, and installs the new MANIFEST sidecar before the final directory fsync. Read-only `open` never mutates or repairs files; `open_writer` first acquires the persistent cross-process admission, then stages repairs at absent names, validates decoded MANIFEST/FSLX bytes against their external Quill witnesses, and only then installs them without clobbering an extant authority.

### 11.4 Crash-only recovery

Read-only `open` recovers a snapshot by reading MANIFEST (falling back to `MANIFEST.prev` on checksum failure), validating ranges (Q1-OB1), and checking section checksums lazily; it never creates, repairs, renames, removes, or timestamps files. `open_writer` acquires a no-follow regular `LOCK` inode with nonblocking exclusive `flock`, admits a residual owner record only when `kill(pid, 0)` returns `ESRCH` (never from mtime), then recovers claims/temps, repairs witness-matched sources, and runs grace-period GC before returning its post-GC snapshot. Every publish holds an O_EXCL `gen-N.claim` through the directory-fsynced two-slot transition. **GC deletes only files matching Quill's own naming schema inside the index dir** — never user files. Crash matrix (§15.5): kill-points at every arrow of §11.3, plus torn-manifest and partial-segment writes, a stopped live owner, and racing dead-owner takeover (fault-injected via LabRuntime VFS-style hooks plus real subprocesses for OS-lock semantics).

---

## 12. Keeper Policies

### 12.1 Snapshot/epoch management

Manifest generations publish via Arc-swap; readers pin generations by holding the Arc (no reader locks, no reload API — `ReloadPolicy` dissolves). Old generations retire when their Arc count drains; segment file deletion is deferred until no snapshot references them (drop-guard pattern; LabRuntime leak oracle).

### 12.2 Tiering (what remains of "merge policy")

Size-class tiers (delta → S/M/L by doc-range width); concat-merge when a tier holds ≥ `fanout` (default 8) runs; compaction when segment tombstone density > threshold (default 20%). Because concat-merge is I/O-cheap (§7), the policy's only real job is bounding segment count (query-side k-way width) and tombstone debt. `configure_bulk_load` mode = suppress merging during bulk build, one final concat pass at `finish` — replacing `LogMergePolicy::set_min_num_segments(256)` and `force_merge_bounded` (whose memory-pressure motivation evaporates when merge stops decoding postings).

---

## 13. Consumer Integration

### 13.1 Fusion

Zero changes: Quill implements `LexicalSearch` (+ `fusion_metadata_is_deferred=true` + hydrate), returns score-sorted `ScoredResult` with `ScoreSource::Lexical` and raw BM25 in `lexical_score`. `SyncLexicalSearch` for the sync searcher is a thin wrapper over the same sync internals.

### 13.2 fsfs

`LiveIngestPipeline` swaps `TantivyIndex` → `QuillIndex` (same upsert/delete/commit-per-batch shape); the adaptive lexical debounce window can shrink because visibility no longer waits on commit (Q3) — a measured, gated change (fsfs bead, §19 quill-e7). `lexical_pipeline::LexicalIndexBackend` gets a Quill implementation so the existing mutation planner/chunk policy drive it. Status/footprint reporting uses `segment_stats()`.

### 13.3 index_builder + facade

`protect_lexical_durability` switches to the generic `FileProtector` path over FSLX files. Facade re-exports Quill types under `frankensearch::lexical`; feature graph in §16.

---

## 14. Performance Doctrine & Targets

Reference machines: (a) Apple Silicon ≥ M2 Pro-class (P/E asymmetric, 10–12 cores), (b) high-core x86 (Zen 3+ ≥16C, the RCH fleet's Contabo class for CI trend lines with documented noise budgets). Every number is a provisional `EmpiricalGate`, activated only when its benchmark manifest pins: corpus fixture (docs, bytes, token distribution — the golden `tiny/small/medium` profiles + a new `xlarge` 1M-doc synthetic), schema (positions on/off), thread count, tantivy oracle version/config (identical analyzer semantics, identical heap budget), build profile (`release-perf`-equivalent: opt-level 3, lto thin, codegen-units 1, frame pointers), and statistical rule (≥10 runs, median + MAD, cv_pct < 5 or the result is noise). Both engines run in the same harness with symmetric setup/teardown outside timed windows.

| Gate | Target (provisional) |
|---|---|
| **QG-1 Bulk indexing, multi-core** (medium profile, positions on, 8 threads, commit included) | ≥ **3.0×** tantivy docs/s; and ≥ 60% of single-pass tokenization bandwidth ceiling (measured tokenize-only throughput) as the honesty denominator |
| **QG-2 Bulk indexing, single-thread** | ≥ **1.5×** tantivy. BINDING Q2C COMPARATOR CONTRACT 2026-07-31: QG-2 compares both arms symmetrically in memory with no durable storage. Continuous timing begins at the first document feed and ends only after terminal searchable visibility plus complete worker, merge, and queue quiescence. Commit is the searchable-visibility boundary, not durable publication. QG-2 excludes fsync, F_FULLFSYNC, crash recovery, durable publication, and on-disk-byte measurements. Durable gates and production-source durability nonregression remain mandatory outside QG-2. |
| **QG-3 Watch-mode incremental** (5k-update batches, upsert-heavy) | ≥ existing contract floor (5k updates/s, 25ms p95) with ≥ **4×** headroom on update→searchable latency vs tantivy commit+reload path |
| **QG-4 Commit latency** (100k-doc index, warm) | p99 ≤ 50ms sealed-commit; visibility lead (searchable-before-commit) demonstrated in the harness |
| **QG-5 Full compaction** (1M docs, 20% tombstones) | ≥ **5×** tantivy force-merge wall-clock |
| **QG-6 Query latency** (query-class mix from the quality harness, k∈{10,100}, 1M docs) | p50 parity (±10%), p99 ≤ tantivy per class; no class regresses >10% |
| **QG-7 Memory** | Peak ingest RSS ≤ tantivy at equal budget config; index bytes/doc ≤ 1.15× tantivy (positions on) and ≤ 0.8× (positions off vs tantivy positions-on default) |
| **QG-8 Scaling curve** | Indexing throughput at 16 threads ≥ 1.8× its own 4-thread number (contention honesty gate) on the x86 reference |
| **QG-9 Cold open** | `open()` of a 1M-doc index ≤ 50ms (manifest + lazy sections) vs tantivy reader open |
| **QG-10 Dependency surface** | Default `lexical` feature: tantivy + its transitive tree removed; `cargo tree` delta recorded in the plan's evidence bundle |

Method: the five standing laws — (1) no benchmark-only semantics (each gate uses its explicit comparator scope; both arms perform matched work and consume results symmetrically; durable gates match shipped durability, while QG-2 uses the symmetric in-memory terminal-visibility contract above; marker-only commit latency and unlabeled positions-off substitutions remain forbidden); (2) distributions, not averages (p50/p95/p99 + cv_pct always); (3) never hide maintenance (merge/compaction time inside the bulk-index window); (4) memory is first-class (bytes/doc includes tombstones, block-max, idmap); (5) one lever per change with MT8-style ≥0.1% frame attribution, ledgered per `NEGATIVE_EVIDENCE.md` ratio conventions (revert in [0.97,1.03]). `.bench-history` JSON per gate, pass-over-pass ratchet, committed baselines. RCH offload for the matrix; local runs only for flamegraph attribution.

---

## 15. The Conformance Gauntlet (Bet Q5)

### 15.1 Harness kernel

`crates/frankensearch-quill-gauntlet` (dev-only): **Subject** = QuillIndex; **Oracle** = TantivyIndex via the existing lexical crate behind feature `tantivy-oracle`; **Comparator** = canonicalizing differ (rank lists → (DocId, score, tie-group) with tie canonicalization §8.4; snippet/count/doc_count comparators). `EngineIdentity{Subject,Oracle}` stamped on every artifact; oracle version contract pinned to `tantivy 0.26.1` + the lexical crate's git rev. Artifacts: content-addressed JSON envelopes (xxh3 of canonical bytes) under `.gauntlet/` with first-divergence pointers.

### 15.2 Internal differentials (Quill vs Quill)

Pruned (MaxScore/BMW) ≡ exhaustive kernel; SIMD decode ≡ scalar decode; delta-resident ≡ sealed ≡ mixed; merged decoded behavior ≡ unmerged plus raw input-block preservation (Q1-OB2a/OB2b/OB3); compacted ≡ tombstoned. These run in ordinary `cargo test` (fast fixtures) + nightly larger corpora.

### 15.3 Oracle differentials & metamorphic suites

- **Corpus:** shared fixtures (100-doc relevance corpus, edge_cases.json) + synthetic generators (Zipf term distributions, doc-length spreads, unicode script mixes, pathological tokens: 256-byte terms, same-position stacks, empty/whitespace docs) + a real-code corpus snapshot (this repo).
- **Query corpus:** per-class (identifier/short-keyword/natural-language/phrase/boolean/glob/range) generated + harvested from the quality harness.
- **Metamorphic transforms:** document-order permutation (results invariant given docid-map canonicalization); duplicate-then-delete idempotence; index-twice equivalence; segment-boundary shuffling (flush budget perturbation must not change results); upsert≡delete+add; query whitespace/case invariances per analyzer contract.
- **Ported behavioral tests:** every engine-visible semantic among the 90 lexical unit tests becomes an engine-parameterized conformance test (both engines must pass identically).
- **Fusion end-to-end:** hybrid pipeline (RRF is rank-only) on the quality harness — per-slice nDCG@10/MRR/Recall@10 with bootstrap CIs must be **statistically indistinguishable or better** (G3 gate); known-item bench (`real_hybrid_knownitem`) re-run with Quill arm.

### 15.4 Statistical gates

`metrics_eval::bootstrap_compare` (2000 resamples, seeded, 0.95) for quality deltas; conformance pass-rate reported per query class with the release decision on the **lower** confidence bound, not the point estimate.

### 15.5 Determinism & crash matrices (LabRuntime)

Seeded deterministic schedules over: concurrent ingest + search; cancel-at-every-await ingest sessions; merge/compaction racing search; crash kill-points (§11.4) with recovery assertions (committed docs durable, uncommitted invisible, no file-format corruption, GC removes exactly garbage). Repeated-seed replay on failure; oracle checks via LabRuntime's invariant reports.

### 15.6 Divergence Register

`docs/contracts/quill-divergence-register.md`: every intentional or discovered-and-accepted divergence from oracle behavior — id, class (ScoreEpsilon / TieOrder / SnippetWindow / GlobExpansionLimit / …), root cause, consumer impact, fixture id, decision (accept/fix), reviewer. An empty register is not the goal; an *unclassified* divergence is the only failure.

---

## 16. Workspace, Features, Migration

### 16.1 Crates

- `crates/frankensearch-quill` — the engine (modules `scribe/grimoire/quiver/argus/keeper/contract`), deps: core, asupersync, wide, bytemuck, memmap2, rayon, compact_str, ahash, crc32fast, xxhash-rust, thiserror, tracing (+ serde_json only for the stored-metadata passthrough type). `#![forbid(unsafe_code)]` (module-scoped mmap allowance only if the shared facade route in §1.2 is unavailable).
- `crates/frankensearch-quill-gauntlet` — dev-only conformance/bench harness; features: `tantivy-oracle` (pulls frankensearch-lexical). Never a dependency of shipping crates. Workspace `default-members` excludes it.
- `frankensearch-lexical` — retains tantivy: the oracle impl + `cass-compat` interop. Post-G3 it leaves the default graph entirely.

### 16.2 Feature graph (end state)

Facade: `lexical = ["dep:frankensearch-quill"]` (the meaning of the flag flips at G3 — consumers keep using `lexical`/`hybrid`/`full` unchanged); `lexical-tantivy = ["dep:frankensearch-lexical"]` (oracle/interop, non-default); `cass-compat = ["lexical-tantivy"]`. During G1–G2 the new engine rides a non-default `quill` feature; the flip is a single, reviewed, gate-certified commit that also ports fsfs/index_builder/facade call sites off tantivy-shaped APIs (§9.6). Migration of existing on-disk indexes: none — fsfs rebuilds lexical indexes from canonical storage (already the recovery path); a `fsfs doctor` note + automatic rebuild-on-format-detect covers upgrades (bead quill-e7).

### 16.3 CI

Standard gates (fmt/check/clippy/test) + gauntlet fast suite on every PR touching quill/lexical/fusion; nightly full differential + crash matrix + perf trend (RCH); ratchet state committed. `cargo test -p frankensearch-quill` stays fast (<60s) — big corpora live behind `--ignored`/nightly lanes.

---

## 17. Build-It-Ourselves Inventory & Size Estimate

In-house (new): SIMD tokenizer kernels (~600 LOC), analyzer family (~1.2k), interner/arenas (~500), columnar accumulator + radix flush (~900), delta segment (~800), FSLX writer/reader + checksums (~1.5k), prefix-block term dict (~700), FOR/bitmap posting codecs + block-max (~900), BM25 + fieldnorm tables (~300), boolean/phrase/range/glob eval + MaxScore/BMW + collectors (~1.8k), snippets (~300), manifest/commit/recovery/GC (~900), tiering/compaction (~600), trait impls + public API (~500), gauntlet harness + generators + comparators (~2.5k), benches (~1.5k). **≈ 15–16 KLOC engine + ~4 KLOC harness/bench**, versus 5,956 LOC currently in the lexical crate (which stays for oracle/interop until retirement). Explicitly **not** built: async runtime, channels, deterministic lab, RaptorQ/repair, SQLite storage, RRF/fusion, eval statistics — all in-tree already.

---

## 18. Convergence Gates

- **G0 — Contracts on paper (no engine code merges before this lands):** Language Contract (§5) with fixture corpora committed; FSLX format registry v1 (§10.6); Q1 invariant + obligations doc; Divergence Register scaffold; gauntlet harness skeleton compiling with oracle wired (subject stubbed); perf-gate manifests drafted with pinned fixtures. *Exit: docs + fixtures + harness skeleton reviewed and committed.*
- **G1 — "The engine lives":** single-shard end-to-end (ingest → seal → open → query) passing the ported behavioral suite + internal differentials on the fast corpus; crash matrix green on the §11.3 kill-points; delta search live; deterministic-ingest replay byte-identical. *Exit: `cargo test -p frankensearch-quill` + fast gauntlet green.*
- **G2 — "Conformant at scale":** multi-shard ingest with Q1 obligations verified; merge/compaction live; full oracle differential + metamorphic suites green with all divergences registered; LabRuntime suites green; fsfs/index_builder integration behind `quill` feature; hybrid quality harness indistinguishable-or-better. *Exit: nightly gauntlet green two consecutive runs.*
- **G3 — "Leapfrog, published":** §14 gates QG-1..10 green with keep-gate-clean evidence committed (bench manifests, ledger entries, flamegraph artifacts); default `lexical` flip commit lands with consumer ports; tantivy out of the default graph; README/AGENTS/docs updated; retrospective + negative-evidence sweep. *Exit: the flip commit + evidence bundle.*

Workstream → gate mapping and full task decomposition live in the beads (§19); the graph is the source of truth for sequencing.

---

## 19. Beads Map (Epic Family `quill`)

Created via `br` with dependencies (validated acyclic via `bv --robot-insights`). Epic skeleton — the beads carry the self-contained detail:

| Epic | Contents | Gate |
|---|---|---|
| **quill-e0 Contracts & Gauntlet Skeleton** | Language Contract + fixtures; FSLX registry; Q1 obligations doc; Divergence Register; gauntlet kernel (oracle wiring, comparators, EngineIdentity); perf manifests | G0 |
| **quill-e1 Scribe** | engine-crate scaffolding (workspace membership, QuillConfig, errors, schema descriptors, tracing conventions); SIMD tokenizer (+3-way parity bench); CASS analyzer family; interner/arenas; columnar accumulator; radix flush; hash-vs-columnar A/B (ledgered); shard router + docid leases (R1 seal-at-lease-boundary) | G1 |
| **quill-e2 Grimoire & Quiver** | prefix-block dict (field-namespaced keys); FOR/bitmap codecs + canonical scalar unpack and a retained SIMD differential candidate; block-max pairs; positions streams (u32); DOCLEN/IDMAP/IDHASH/NUMERIC/STOREDMETA/STATS sections | G1 |
| **quill-e3 Keeper** | FSLX writer/reader; manifest + two-slot publish; read-only crash recovery plus writer-admitted GC; tombstones; concat-merge (Q1-OB2a/Q1-OB2b/Q1-OB3 fixtures); compaction (OB4); tiering; durability `.fec` integration | G1→G2 |
| **quill-e4 Argus** | parsers (default lenient + CASS grammar); planner; exhaustive kernel; MaxScore/BMW (+rank-safety differentials); phrase/range/glob; collectors; snippets; segment-parallel fan-out | G1→G2 |
| **quill-e5 Delta** | per-shard delta segment; epoch publish; seal path; mixed-state conformance | G1→G2 |
| **quill-e6 Gauntlet at scale** | corpus/query generators; metamorphic suites; ported behavioral tests; LabRuntime determinism + crash matrices; statistical gates; fusion e2e quality | G2 |
| **quill-e7 Integration & Flip** | LexicalSearch/Sync impls + public API assembly; fsfs LiveIngest + backend + debounce retune; index_builder/facade ports; feature graph; rebuild-on-upgrade; consumer-port e2e sweep (all-features matrix); the flip commit | G3 |
| **quill-e8 Perf doctrine** | bench matrix vs oracle (QG-1..10); .bench-history ratchet; flamegraph lanes; lever beads (one per optimization, ledger-disciplined); scaling-curve studies (Apple Silicon + x86) | G3 |
| **quill-e9 Docs & retirement** | README/AGENTS updates; format/ops docs; cass-compat carve-out tracking bead; tantivy default-graph removal; retrospective | G3 |

Every feature bead cluster carries explicit unit-test beads and e2e/logging beads (idea-wizard Phase-6 rule); every perf bead pre-flights the negative-evidence ledger.

---

## 20. Risks & Mitigations

| Risk | Reality check | Mitigation |
|---|---|---|
| **Rank-parity with tantivy is harder than it looks** (fieldnorm quantization, segment-geometry-dependent stats, parser corner cases) | The top correctness risk | Fieldnorm table vendored + pinned; ScoreEpsilon class with tie-canonicalizing comparator; parser pinned by differential *fixtures* not source-reading; Divergence Register makes residuals explicit and reviewable; default doesn't flip until G2/G3 gates |
| **Columnar-radix ingest underperforms the hashmap school on small batches** | Plausible — radix pays fixed costs | The delta segment *is* the hashmap school for small batches; radix owns bulk flush only; explicit A/B bead with ledger entry either way (quill-e1); SPIMI fallback is a config, not a rewrite |
| **Positions stream costs erase the indexing win** (phrases must work) | Positions ≈ largest section | Positions encoded lazily per term partition (sequential), skipped entirely for non-position fields; QG-1 is measured positions-ON so the claim is honest; positions-off mode is a separate documented config for corpora that never phrase-search |
| **Glob scans over big dictionaries regress query p99** | Substring/complex globs are O(dict) | Bounded expansion (as today), SWAR candidate rejection, per-class QG-6 gate; if needed, post-1.0 reversed-term tier — a registered retry-condition, not 1.0 scope |
| **Delta/sealed dual read paths breed divergence** | Two implementations of one semantics | One cursor trait, shared scoring; mixed-state conformance class runs every fixture in all three residency states (§15.2) |
| **Crash-recovery subtleties (torn manifest, partial seal)** | Classic | Two-slot manifest, crash-only open, LabRuntime kill-point matrix from G1 — not retrofitted |
| **mmap allowance vs `forbid(unsafe_code)` promise** | One real tension | Prefer a shared safe facade in frankensearch-index; else the single module-scoped allowance with ledger comment, exactly the FSVI precedent; zero unsafe anywhere else, enforced by lints |
| **Scope creep toward "generic tantivy replacement"** | The classic engine-rewrite death | §3.2 omissions are constitutional; Language Contract bounds the surface; new features require new contract rows + fixtures first |
| **cass on-disk interop confusion** | Users may expect Quill to read cass tantivy indexes | Explicit carve-out (§3.4), feature-gated, documented in fsfs doctor + README; migration tracked as a separate follow-up |
| **Benchmark self-deception** | Perennial | §14 laws + keep-gates + oracle-in-same-harness + NEGATIVE_EVIDENCE ratio discipline; regressions replay via seeded harness |
| **Concurrent-agent collisions during build-out** | This repo runs swarms | Beads dependency discipline + Agent Mail reservations per epic; crate boundary (`frankensearch-quill`) isolates the new work from the hot lexical crate |

---

## 21. Rejected Alternatives (Recorded Reasons)

1. **Fork tantivy and strip it.** Keeps the fossils (merge tax, doc store, visibility model, unsafe, dep tree), forfeits Q1–Q4, and creates a permanent divergent fork burden. The narrow used-surface makes ground-up *cheaper* than surgery.
2. **FTS5 (FrankenSQLite) as the engine.** Already in-tree as an alternative; contiguous B-tree FTS is the wrong shape for the multicore ingest and WAND query goals; keeping it as the storage-integrated option is strictly better.
3. **Reimplement tantivy's on-disk format.** Buys cass interop at the price of inheriting every format fossil; carve-out (§3.4) is cheaper and honest.
4. **Positions-off by default for speed.** Breaks phrase queries silently; rejected — positions on, made cheap, with an explicit off-mode.
5. **Global shared term dictionary during ingest.** A contention magnet that murders the share-nothing property; per-segment dictionaries + byte-keyed query joins are the Lucene lesson worth keeping.
6. **Elias-Fano / MPH / FST / learned structures at 1.0.** Encode-cost or flexibility losses on the hot path for wins our scale doesn't need; each has a registered retry condition (§4).
7. **mmap-as-durability.** Manifest+fsync+repair-trailers is the doctrine; mmap is a read optimization only.
8. **serde for durable bytes.** House rule; hand-rolled versioned sections only.
9. **An external-crates shortcut** (roaring, fst, memchr, regex…). Violates the closed-universe constraint; every needed piece is small at our scale and lands in-tree with tests.

---

*End of plan. The beads graph (§19) is the executable form; this document is the rationale of record. Amendments follow the format-registry discipline: change the plan, version the change, land the fixture.*
