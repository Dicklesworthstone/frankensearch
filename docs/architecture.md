# frankensearch Architecture Overview

This document is the contributor-facing map of the current `frankensearch` workspace. It focuses on runtime behavior, crate boundaries, and the design choices that matter when you are changing code.

## 1) Workspace Crate Map (15 Members)

| Crate | Purpose |
|---|---|
| `frankensearch-core` | Shared contracts: traits (`Embedder`, `LexicalRead`, …), errors, config, canonicalization, query classes, telemetry types |
| `frankensearch-embed` | Embedder implementations (hash control, Model2Vec/potion, FastEmbed/MiniLM, native multilingual) and model discovery/cache/download glue |
| `frankensearch-index` | FSVI vector format, search kernels (brute force, optional `ann` via `frankenhnsw`, in-tree `native_hnsw` not yet on the search path), index builders |
| `frankensearch-quill` | Native pure-Rust BM25 lexical engine (FSLX segments, delta-visible indexing); the default lexical backend behind the facade `lexical`/`quill` features and the only one `fsfs` ships |
| `frankensearch-lexical` | Tantivy-backed lexical implementation, retained as the pinned conformance oracle (`lexical-tantivy`) and for external CASS schema-v8 interop (`cass-compat`); not in default builds |
| `frankensearch-quill-gauntlet` | Differential conformance / metamorphic / perf gauntlet certifying Quill against the Tantivy oracle (`publish = false`) |
| `frankensearch-fusion` | RRF fusion, blending, progressive two-tier search orchestration (`TwoTierSearcher`) |
| `frankensearch-rerank` | Optional cross-encoder reranking: pure-Rust frankentorch `native` backend used by the library and `fsfs search --rerank`; FastEmbed/ONNX is a separate optional library backend |
| `frankensearch-storage` | FrankenSQLite metadata, dedup/content-hash, persistent embedding queue |
| `frankensearch-durability` | RaptorQ repair trailer and protect/verify/repair workflows for Quill segments and FSVI |
| `frankensearch-fsfs` | Standalone CLI/TUI search product (`fsfs`) built on library crates |
| `frankensearch-tui` | Shared TUI framework primitives (screens/shell/input/replay/theme) |
| `frankensearch-ops` | Fleet/operations console crate on top of `frankensearch-tui` (library + simulator-fed binary; no shipped telemetry source yet) |
| `frankensearch` | Facade crate that re-exports public APIs across the workspace |
| `tools/optimize_params` | Parameter search helper (`optimize-params`, `publish = false`) |

High-level dependency arrows:

```text
frankensearch-core
  -> frankensearch-embed
  -> frankensearch-index
  -> frankensearch-lexical
  -> frankensearch-fusion
  -> frankensearch-rerank
  -> frankensearch-storage
  -> frankensearch-durability
  -> frankensearch-tui

frankensearch-quill  -> (core, index formats; optional durability)
frankensearch-fusion -> (embed, index, optional quill/lexical/rerank)
frankensearch-quill-gauntlet -> (quill, lexical as the pinned Tantivy oracle, fusion)
frankensearch-fsfs   -> (core, embed, index, fusion, quill, storage, durability, tui, optional native rerank; lexical only behind `shadow-oracle`)
frankensearch-ops    -> (core, tui)
frankensearch facade -> (core, embed, index, fusion, quill by default for `lexical`; optional lexical-tantivy/rerank/storage/durability)
```

Mermaid dependency view:

```mermaid
graph TD
  core[frankensearch-core]
  embed[frankensearch-embed]
  index[frankensearch-index]
  quill[frankensearch-quill]
  lexical[frankensearch-lexical]
  fusion[frankensearch-fusion]
  rerank[frankensearch-rerank]
  storage[frankensearch-storage]
  durability[frankensearch-durability]
  tui[frankensearch-tui]
  fsfs[frankensearch-fsfs]
  ops[frankensearch-ops]
  facade[frankensearch]

  core --> embed
  core --> index
  core --> quill
  core --> lexical
  core --> fusion
  core --> rerank
  core --> storage
  core --> durability
  core --> tui

  embed --> fusion
  index --> fusion
  quill --> fusion
  lexical -. explicit oracle or interop .-> fusion
  rerank --> fusion

  core --> facade
  embed --> facade
  index --> facade
  fusion --> facade
  quill --> facade
  lexical -. lexical-tantivy or cass-compat .-> facade
  rerank --> facade
  storage --> facade
  durability --> facade

  tui --> fsfs
  core --> fsfs
  embed --> fsfs
  index --> fsfs
  fusion --> fsfs
  quill --> fsfs
  rerank -. rerank feature .-> fsfs
  tui --> ops
  core --> ops
```

## 2) End-to-End Data Flow

### Indexing path

```text
document
  -> canonicalize (NFC/cleanup)
  -> embed (fast tier; also quality tier in ordinary fsfs indexing)
  -> persist vectors in FSVI generations
  -> persist metadata/queue state in FrankenSQLite
  -> lexical indexing via Quill (facade lexical feature; always in fsfs)
```

### Query path

The following is the library's `TwoTierSearcher` pipeline. `fsfs` has a separate
orchestrator in `crates/frankensearch-fsfs/src/runtime.rs`; its command, daemon,
cache, streaming, and TUI paths must wire each policy explicitly. A library
builder or default alone does not establish product behavior.

The fsfs orchestrator retains independent fast and quality candidate pools and
calls `frankensearch_fusion::blend_two_tier` with the effective
`search.quality_weight` (default `0.7`, converted once to `f32`). The shared
helper normalizes each pool and joins scores by document ID. A document present
in both pools gets `weight * quality + (1 - weight) * fast`; a document present
in only one keeps that source's normalized score. fsfs then RRF-fuses the blended
semantic ranking with the lexical head and applies optional reranking. Daemon
requests acknowledge the caller's policy; cache keys retain the effective
weight, exact RRF configuration, and deadline. Explanation payloads retain the
source scores, blend weights, and joint lexical/vector RRF contributions.
Raw BM25 fallback tails carry explicit provenance instead of hypothetical RRF
contributions; a later quality promotion moves the hit into the refined head.

```text
query
  -> parse + classify
  -> embed query (fast tier)
  -> search vector index (FSVI; optional ANN)
  -> search lexical index (Quill by default; explicit Tantivy interop supported)
  -> RRF fuse
  -> emit initial results
  -> embed/refine with quality model
  -> blend scores
  -> optional cross-encoder rerank
  -> emit refined results
```

Mermaid search flow:

```mermaid
flowchart TD
  q[Query] --> c[Canonicalize + Classify]
  c --> fe[Fast Embed]
  c --> lx[Quill Lexical Search]
  fe --> vs[Vector Search FSVI]
  vs --> rrf[RRF Fusion K=60]
  lx --> rrf
  rrf --> init[SearchPhase::Initial]
  init --> qe[Quality Embed]
  qe --> blend[Two-tier Blend]
  blend --> rr[Optional Rerank]
  rr --> refined[SearchPhase::Refined]
```

## 3) Two-Tier Strategy (Why It Exists)

The architecture intentionally separates speed and quality:

- Fast tier gets early, useful answers quickly (interactive latency budget).
- Quality tier spends more compute to improve ranking after initial display.

Practical effect:

- better perceived latency for users and agent workflows
- an opportunity to improve final ranking, subject to corpus-specific quality evaluation
- graceful degradation when quality models are missing or fail

Progressive API contract:

- `SearchPhase::Initial`
- `SearchPhase::Refined`
- `SearchPhase::RefinementFailed`

The facade exposes the library phases. `fsfs` emits its own corresponding phase
artifacts and `fsfs.stream.query.v1` events. Native cross-encoder support is
compiled by its default `rerank` Cargo feature, but scoring is opt-in through
`fsfs search --rerank` or `search.rerank`. Provision its model with
`fsfs download-models ms-marco-minilm-l-6-v2`.

In fsfs, `search.quality_timeout_ms` starts after the Initial artifact is
produced and its streaming phase sink returns. It covers cold quality-model
initialization, waiting for model capacity, inference, and quality-index
retrieval. Expiry yields `RefinementFailed` with the Initial hits and a typed
timeout reason; caller cancellation propagates separately. Blending, lexical
fusion, and optional reranking follow successful retrieval outside this window.
This deadline does not bound total command duration or establish a measured
latency or quality improvement.

## 4) Storage Model

Three storage concerns are explicitly separated:

1. Vector storage: FSVI files (`frankensearch-index`)
2. Lexical storage: Quill FSLX segments (`frankensearch-quill`); Tantivy layouts are confined to the explicit oracle/CASS interop lane
3. Metadata/job state: FrankenSQLite (`frankensearch-storage`)

Why split:

- each subsystem can optimize for its access pattern
- vector search remains SIMD/mmap focused
- metadata and queues stay transactional and durable
- lexical ranking keeps BM25 semantics and query parsing isolated

## 5) Durability Layer

`frankensearch-durability` adds corruption detection/repair primitives around persistent files, including:

- RaptorQ FEC sidecar materialization/validation for recoverability
- repair trailer I/O
- file/segment verification and repair orchestration
- durability metrics and health reporting

This layer is deliberately optional so lightweight deployments can skip its overhead, while higher-durability environments can enable it.

## 6) Async Runtime Model (asupersync, not tokio)

The workspace uses `asupersync` for async/concurrency contracts.

To keep real-model inference from blocking the executor, callers supply a
blocking pool through `Cx`. The fsfs main runtime and each socket handler retain
a `SearchBlockingPool` outside the scheduler and attach its handle to the live
context. Quality-model initialization and index scans use the same bounded
capacity. A synchronous ONNX call already in progress cannot be preempted: it
retains its model permit until completion, and the pool owner drains every
admitted job at shutdown. Process exit can therefore follow the timeout
response later. The socket daemon flushes the response and closes its write
side before dropping its scheduler and draining the owned pool.

Operational implications:

- async functions receive a `Cx` capability context
- cancellation and scoped task lifetimes are part of normal control flow
- runtime behavior is explicit in API boundaries (especially embed/search/rerank paths)

Why it matters to contributors:

- do not add tokio/hyper/reqwest patterns
- preserve `Cx` plumbing in new async code
- keep cancellation-correct behavior when adding queues/workers/search phases

## 7) Key Design Decisions and Rationale

- f16 quantization for vector storage
  - reduces vector footprint materially while retaining ranking quality for cosine-style retrieval
- RRF with `K=60`
  - robust rank-based fusion across lexical and semantic lists without fragile score normalization coupling
- progressive iterator/phase model
  - enables fast-first UX with quality refinement as a second phase
- NaN-safe ordering in ranking operations
  - deterministic behavior even with problematic floating-point edge cases

These are foundational decisions; changes here require explicit measurement and migration planning.

## 8) Contributor Onramp: Where To Read Code First

Start with these files:

- `frankensearch/src/lib.rs` (facade surface and re-exports)
- `crates/frankensearch-core/src/lib.rs` (contracts/types)
- `crates/frankensearch-fusion/src/searcher.rs` (progressive orchestration)
- `crates/frankensearch-index/src/lib.rs` and `crates/frankensearch-index/src/two_tier.rs` (vector index/search)
- `crates/frankensearch-quill/src/index.rs` and `src/argus.rs` (default lexical storage and BM25 execution)
- `crates/frankensearch-lexical/src/lib.rs` (explicit Tantivy oracle/CASS integration)
- `crates/frankensearch-fsfs/src/main.rs` + `crates/frankensearch-fsfs/src/adapters/cli.rs` (standalone product surface)

Then inspect:

- `docs/fsfs-config-contract.md`
- `docs/fsfs-dual-mode-contract.md`
- `AGENTS.md`

## 9) Scope Notes

This document is intentionally a high-signal architecture map, not a full API reference. Detailed behavior, config invariants, and integration rules live in crate-level docs and the contracts under `docs/`.
