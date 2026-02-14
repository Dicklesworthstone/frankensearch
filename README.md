# frankensearch

<div align="center">
  <img src="frankensearch_illustration.webp" alt="frankensearch - Two-tier hybrid search for Rust">
</div>

<div align="center">

[![CI](https://github.com/Dicklesworthstone/frankensearch/actions/workflows/ci.yml/badge.svg)](https://github.com/Dicklesworthstone/frankensearch/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/frankensearch.svg)](https://crates.io/crates/frankensearch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-edition_2024-orange.svg)](https://doc.rust-lang.org/edition-guide/)

**Two-tier hybrid search for Rust: sub-millisecond initial results, quality-refined rankings in 150ms.**

Stitched together from the best parts of three battle-tested codebases — hence the name.

</div>

---

## TL;DR

**The Problem:** You want both speed *and* quality from local search, but fast embedding models sacrifice accuracy and quality models are too slow for interactive use. You also want lexical keyword matching combined with semantic understanding, not one or the other.

**The Solution:** frankensearch runs a two-tier progressive search pipeline. A fast static embedder (potion-128M, 0.57ms) delivers initial results instantly, then a quality transformer (MiniLM-L6-v2, 128ms) refines the rankings in the background. Lexical BM25 and semantic vector search are fused via Reciprocal Rank Fusion. Your UI gets results to display in under 15ms, with refined rankings arriving ~150ms later.

### Why frankensearch?

| Feature | What It Does |
|---------|--------------|
| **Progressive search** | Initial results in <15ms, refined in ~150ms via `SearchPhase` iterator |
| **Hybrid fusion** | Lexical (Tantivy BM25) + semantic (vector cosine) combined with RRF (K=60) |
| **Two-tier embedding** | Fast tier (potion-128M, 0.57ms) + quality tier (MiniLM-L6-v2, 128ms) |
| **Graceful degradation** | Quality model unavailable? Falls back to fast-only. No models at all? Hash embedder works everywhere |
| **Feature-gated** | Pay only for what you compile: `hash` (0 deps) through `full` (everything) |
| **SIMD vector search** | `wide::f32x8` portable SIMD across x86 SSE2/AVX2 and ARM NEON |
| **f16 quantization** | 50% memory savings on vector indices with <1% quality loss |
| **Zero unsafe code** | `#![forbid(unsafe_code)]` throughout |

---

## Quick Example

```rust
use frankensearch::prelude::*;

// Build a two-tier searcher with defaults
let config = TwoTierConfig::default();
let searcher = TwoTierSearcher::new(&index, &embedder_stack, config)
    .with_lexical(&tantivy_index);

// Progressive search: fast results first, then refined
for phase in searcher.search("distributed consensus algorithm", 10) {
    match phase {
        SearchPhase::Initial(results) => {
            // Display these immediately (~15ms)
            for hit in &results {
                println!("  {} (rrf: {:.4})", hit.doc_id, hit.rrf_score);
            }
        }
        SearchPhase::Refined(results) => {
            // Update the display with refined rankings (~150ms)
            for hit in &results {
                println!("  {} (rrf: {:.4}, blended)", hit.doc_id, hit.rrf_score);
            }
        }
        SearchPhase::RefinementFailed(results) => {
            // Quality model failed; initial results are still valid
        }
    }
}
```

### Indexing Documents

```rust
use frankensearch::prelude::*;

// Auto-detect the best available embedder
let embedder_stack = EmbedderStack::auto_detect()?;

// Build a vector index
let mut builder = VectorIndexBuilder::new(embedder_stack.fast_embedder());
for doc in &documents {
    let text = canonicalize(&doc.content); // NFC + markdown strip + code collapse
    let embedding = embedder_stack.fast_embedder().embed(&text)?;
    builder.add(&doc.id, &embedding)?;
}
builder.save("index.fsvi")?; // FSVI binary format, f16 quantized, fsync'd

// Build a Tantivy lexical index
let lexical = LexicalIndexBuilder::new()
    .add_documents(&documents)?
    .build("tantivy_index/")?;
```

### Minimal Setup (Hash Embedder Only)

```rust
// Zero dependencies beyond frankensearch-core
// Works everywhere, no model downloads needed
use frankensearch::hash::FnvHashEmbedder;

let embedder = FnvHashEmbedder::new(384); // 384-dim, deterministic
let embedding = embedder.embed("hello world")?;
```

---

## Design Philosophy

### 1. Progressive Over Blocking

Traditional search makes you wait for the best answer. frankensearch yields fast approximate results immediately (via `SearchPhase::Initial`), then upgrades them when the quality model finishes. The consumer decides how to present this — swap in place, animate a transition, or ignore the refinement entirely.

### 2. Hybrid Over Single-Signal

Pure semantic search misses exact keyword matches. Pure lexical search misses meaning. frankensearch fuses both via Reciprocal Rank Fusion (RRF), which is rank-based and doesn't depend on score normalization. Documents appearing in both lexical and semantic results get a natural boost.

### 3. Pay For What You Use

The default feature set (`hash`) compiles with zero ML dependencies. Add `model2vec` for the fast tier (~128MB model), `fastembed` for the quality tier (~90MB model + ONNX runtime), `lexical` for Tantivy, or `full` for everything. Feature flags control compilation, not runtime behavior.

### 4. No Domain Leakage

The vector index stores only `(doc_id, embedding)`. frankensearch doesn't know or care whether your documents are tweets, chat messages, code files, or research papers. Domain-specific metadata belongs in your storage layer.

### 5. Deterministic Results

NaN-safe `total_cmp()` ordering in the top-k heap. Four-level tie-breaking in RRF: score descending, in-both-sources preference, lexical score descending, doc_id ascending. Same input always produces the same output.

---

## How frankensearch Compares

| Feature | frankensearch | tantivy (alone) | qdrant | meilisearch |
|---------|---------------|-----------------|--------|-------------|
| Semantic search | Two-tier (fast + quality) | Via plugin | Single model | Experimental |
| Lexical search | Tantivy BM25 | Native | Basic | Native |
| Hybrid fusion | RRF built-in | Manual | RRF | Manual |
| Progressive results | Native iterator | N/A | N/A | N/A |
| Deployment | Embedded library | Embedded library | Server | Server |
| Model management | Auto-detect + download | N/A | External | Built-in |
| f16 quantization | Default | N/A | Optional | N/A |
| Portable SIMD | `wide` (x86 + ARM) | N/A | Platform-specific | N/A |
| Unsafe code | Forbidden | Minimal | Present | Present |

**Use frankensearch when:** you need an embedded search library with both speed and quality, hybrid lexical+semantic fusion, and progressive result delivery — all without running a server.

**Use something else when:** you need a distributed search cluster, GPU-accelerated inference, or a standalone search API with a REST interface.

---

## Installation

### As a Dependency

```toml
# Cargo.toml — pick your feature set

# Minimal: hash embedder only (zero ML deps, always works)
[dependencies]
frankensearch = "0.1"

# Fast semantic search (potion-128M, ~0.57ms embeddings)
[dependencies]
frankensearch = { version = "0.1", features = ["model2vec"] }

# Stateless hybrid search (semantic + lexical + RRF)
[dependencies]
frankensearch = { version = "0.1", features = ["hybrid"] }

# Persistent hybrid search (adds FrankenSQLite metadata + queue)
[dependencies]
frankensearch = { version = "0.1", features = ["persistent"] }

# Durable persistent search (adds RaptorQ self-healing)
[dependencies]
frankensearch = { version = "0.1", features = ["durable"] }

# Everything: durable + rerank + ANN + downloads
[dependencies]
frankensearch = { version = "0.1", features = ["full"] }
```

### Feature Flags

| Feature | Dependencies Added | What You Get |
|---------|--------------------|--------------|
| `hash` (default) | None | FNV-1a hash embedder, vector index, SIMD search |
| `model2vec` | safetensors, tokenizers | potion-128M static embedder (fast tier) |
| `fastembed` | fastembed (ONNX runtime) | MiniLM-L6-v2 transformer embedder (quality tier) |
| `lexical` | tantivy | BM25 full-text search + RRF fusion |
| `storage` | frankensearch-storage | FrankenSQLite metadata + persistent embedding queue |
| `durability` | frankensearch-durability | RaptorQ repair metadata + self-healing primitives |
| `fts5` | storage | Enable FrankenSQLite FTS5 lexical backend wiring |
| `rerank` | ort, tokenizers | FlashRank cross-encoder reranking |
| `ann` | hnsw_rs | HNSW approximate nearest neighbor index |
| `download` | asupersync/tls | Model auto-download from HuggingFace |
| `semantic` | hash + model2vec + fastembed | All embedding models |
| `hybrid` | semantic + lexical | Stateless hybrid search |
| `persistent` | hybrid + storage | Hybrid search with persistent metadata/queue |
| `durable` | persistent + durability | Persistent hybrid + self-healing durability |
| `full` | durable + rerank + ann + download | Kitchen-sink production bundle |
| `full-fts5` | full + fts5 | Full bundle plus FTS5 lexical backend |

### From Source

```bash
git clone https://github.com/Dicklesworthstone/frankensearch.git
cd frankensearch
cargo build --release --features full
cargo test --features full
```

Requires Rust nightly (edition 2024). A `rust-toolchain.toml` is included.

Feature-flag matrix regression check (used for CI/release gates):

```bash
bash scripts/check_feature_matrix.sh
```

---

## Architecture

```
                         ┌──────────────────┐
                         │   User Query     │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   Text Canonicalization      │
                    │   NFC → Markdown Strip →     │
                    │   Code Collapse → Truncate   │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   Query Classification      │
                    │   Empty│Identifier│Short│NL  │
                    │   → Adaptive candidate       │
                    │     budgets per class         │
                    └─────────────┬───────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                                       ▼
   ┌─────────────────────┐                 ┌─────────────────────┐
   │  Fast Tier Embed    │                 │  Tantivy BM25       │
   │  potion-128M        │                 │  Lexical Search     │
   │  ~0.57ms, 256d      │                 │                     │
   └──────────┬──────────┘                 └──────────┬──────────┘
              │                                       │
              ▼                                       │
   ┌─────────────────────┐                            │
   │  Vector Index       │                            │
   │  FSVI (f16, mmap)   │                            │
   │  SIMD dot product   │                            │
   └──────────┬──────────┘                            │
              │                                       │
              └──────────────┬────────────────────────┘
                             ▼
               ┌─────────────────────────────┐
               │    RRF Fusion (K=60)        │
               │    score = Σ 1/(K+rank+1)   │
               └─────────────┬───────────────┘
                             │
                             ▼
               ┌─────────────────────────────┐
               │  yield SearchPhase::Initial │  ← ~15ms
               └─────────────┬───────────────┘
                             │
                             ▼
               ┌─────────────────────────────┐
               │  Quality Tier Embed         │
               │  MiniLM-L6-v2, ~128ms       │
               │  Re-embed top candidates    │
               └─────────────┬───────────────┘
                             │
                             ▼
               ┌─────────────────────────────┐
               │  Two-Tier Blending          │
               │  0.7 quality + 0.3 fast     │
               └─────────────┬───────────────┘
                             │
                             ▼
               ┌─────────────────────────────┐
               │  Optional: FlashRank        │
               │  Cross-encoder reranking    │
               └─────────────┬───────────────┘
                             │
                             ▼
               ┌─────────────────────────────┐
               │  yield SearchPhase::Refined │  ← ~150ms
               └─────────────────────────────┘
```

### Crate Structure

```
frankensearch/                         # Facade crate (re-exports everything)
├── Cargo.toml                         # Workspace root
└── crates/
    ├── frankensearch-core/            # Zero-dep traits, types, errors
    │   └── src/lib.rs                 #   Embedder, Reranker, SearchError,
    │                                  #   ScoredResult, VectorHit, FusedHit,
    │                                  #   Canonicalizer, QueryClass
    │
    ├── frankensearch-embed/           # Embedder implementations
    │   └── src/
    │       ├── hash_embedder.rs       #   FNV-1a (0 deps, always available)
    │       ├── model2vec_embedder.rs  #   potion-128M (fast tier)
    │       ├── fastembed_embedder.rs  #   MiniLM-L6-v2 (quality tier)
    │       └── auto_detect.rs         #   EmbedderStack auto-detection
    │
    ├── frankensearch-index/           # Vector storage & search
    │   └── src/
    │       ├── format.rs              #   FSVI binary format I/O
    │       ├── simd.rs                #   wide::f32x8 dot product
    │       ├── search.rs              #   Brute-force top-k
    │       └── hnsw.rs                #   Optional HNSW ANN
    │
    ├── frankensearch-lexical/         # Full-text search
    │   └── src/lib.rs                 #   Tantivy schema, indexing, queries
    │
    ├── frankensearch-fusion/          # Result combination
    │   └── src/
    │       ├── rrf.rs                 #   Reciprocal Rank Fusion
    │       ├── normalize.rs           #   Score normalization (min-max)
    │       ├── blend.rs               #   Two-tier score blending
    │       ├── two_tier_searcher.rs   #   Progressive iterator orchestrator
    │       └── query_class.rs         #   Query classification & budgets
    │
    └── frankensearch-rerank/          # Reranking
        └── src/lib.rs                 #   FlashRank cross-encoder
```

### Dependency Graph Between Crates

```
frankensearch-core       ← everything depends on this (zero external deps)
    │
    ├── frankensearch-embed   (+ safetensors, tokenizers, fastembed)
    ├── frankensearch-index   (+ wide, half, memmap2)
    ├── frankensearch-lexical (+ tantivy)
    ├── frankensearch-fusion  (depends on embed + index + lexical)
    └── frankensearch-rerank  (+ ort, tokenizers)

frankensearch (facade)   ← re-exports from all crates
```

---

## Core Types

### SearchPhase — Progressive Results

```rust
pub enum SearchPhase {
    /// Fast results from potion-128M + BM25 + RRF (~15ms)
    Initial(Vec<FusedHit>),
    /// Quality-refined results from MiniLM-L6-v2 blending (~150ms)
    Refined(Vec<FusedHit>),
    /// Quality model failed; initial results are your final answer
    RefinementFailed(Vec<FusedHit>),
}
```

### FusedHit — Hybrid Search Result

```rust
pub struct FusedHit {
    pub doc_id: String,
    pub rrf_score: f64,
    pub lexical_rank: Option<usize>,
    pub semantic_rank: Option<usize>,
    pub lexical_score: Option<f32>,
    pub semantic_score: Option<f32>,
    pub in_both_sources: bool,
}
```

### Embedder Trait

```rust
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> SearchResult<Vec<f32>>;
    fn embed_batch(&self, texts: &[&str]) -> SearchResult<Vec<Vec<f32>>>;
    fn dimension(&self) -> usize;
    fn id(&self) -> &str;
    fn is_semantic(&self) -> bool;
    fn category(&self) -> ModelCategory;
    fn supports_mrl(&self) -> bool; // Matryoshka dim truncation
}
```

### TwoTierConfig

```rust
let config = TwoTierConfig {
    quality_weight: 0.7,        // 70% quality, 30% fast in blend
    rrf_k: 60.0,                // RRF constant (literature standard)
    candidate_multiplier: 3,    // Fetch 3x limit from each source
    quality_timeout_ms: 500,    // Max wait for quality model
    fast_only: false,           // Set true to skip quality refinement
    ..Default::default()
};
```

### SearchError

```rust
pub enum SearchError {
    EmbedderUnavailable { model: String, reason: String },
    EmbeddingFailed { model: String, source: Box<dyn Error> },
    IndexCorrupted { path: PathBuf, detail: String },
    DimensionMismatch { expected: usize, found: usize },
    QueryParseError { query: String, detail: String },
    SearchTimeout { elapsed_ms: u64, budget_ms: u64 },
    // ... and more
}
```

---

## FSVI Vector Index Format

frankensearch stores embeddings in a custom binary format optimized for memory-mapped SIMD search:

```
┌─────────────────────────────────────┐
│ Header                              │
│   magic: "FSVI" (4 bytes)           │
│   version: u16                      │
│   embedder_id: variable UTF-8       │
│   dimension: u32                    │
│   quantization: u8 (0=f32, 1=f16)  │
│   record_count: u64                 │
│   vectors_offset: u64               │
│   header_crc32: u32                 │
├─────────────────────────────────────┤
│ Record Table (16 bytes/record)      │
│   doc_id_hash: u64 (FNV-1a)        │
│   doc_id_offset: u32               │
│   doc_id_len: u16                  │
│   flags: u16                        │
├─────────────────────────────────────┤
│ String Table                        │
│   Concatenated UTF-8 doc_id strings │
├─────────────────────────────────────┤
│ Vector Slab (64-byte aligned)       │
│   f16 or f32 vectors, contiguous    │
│   Aligned for AVX2/cache lines      │
└─────────────────────────────────────┘
```

- **f16 by default**: 50% memory reduction, <1% quality loss for cosine similarity
- **Memory-mapped**: Zero-copy access via `memmap2`, OS handles page caching
- **64-byte aligned vectors**: Cache-line and AVX2 friendly
- **fsync on save**: Durability guarantee for the index file

---

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| FNV-1a hash embed | ~0.07ms | Pure algorithm, zero deps |
| potion-128M embed | ~0.57ms | Static token lookup + mean pool |
| MiniLM-L6-v2 embed | ~128ms | ONNX transformer inference |
| 384-dim f16 dot product | <2us | `wide::f32x8` SIMD |
| Phase 1 (Initial results) | <15ms | Fast embed + vector search + BM25 + RRF |
| Phase 2 (Refined results) | ~150ms | Quality embed + blend (quality embed is bottleneck) |
| Top-k search (10K docs) | <15ms | Brute-force with heap guard pattern |
| Top-k search (100K docs) | <150ms | Rayon parallel chunks of 1024 |

### Embedder Bakeoff Results

| Model | p50 Latency | Embeddings/sec | Dimensions | Semantic Quality |
|-------|-------------|----------------|------------|------------------|
| FNV-1a hash | 0.07ms | 14,000+ | 384 (configurable) | None (deterministic) |
| potion-multilingual-128M | 0.57ms | 52,144 | 256 | Good (223x faster than MiniLM) |
| all-MiniLM-L6-v2 | 128ms | 234 | 384 | Excellent (baseline) |

The 223x speed gap between potion and MiniLM is exactly why the two-tier design exists.

---

## Configuration

### Environment Variable Overrides

| Variable | Description | Default |
|----------|-------------|---------|
| `FRANKENSEARCH_MODEL_DIR` | Model file directory | `~/.cache/frankensearch/models/` |
| `FRANKENSEARCH_FAST_MODEL` | Fast tier model name | `potion-multilingual-128M` |
| `FRANKENSEARCH_QUALITY_MODEL` | Quality tier model name | `all-MiniLM-L6-v2` |
| `FRANKENSEARCH_FAST_ONLY` | Skip quality refinement | `false` |
| `FRANKENSEARCH_QUALITY_WEIGHT` | Blend factor (0.0-1.0) | `0.7` |
| `FRANKENSEARCH_RRF_K` | RRF constant | `60.0` |
| `FRANKENSEARCH_PARALLEL_SEARCH` | Enable rayon parallel search | `auto` |
| `FRANKENSEARCH_LOG` | Tracing filter directive | `info` |

### Model Search Paths

Models are located by checking these paths in order:

1. `$FRANKENSEARCH_MODEL_DIR` (explicit override)
2. `~/.cache/frankensearch/models/`
3. `~/.local/share/frankensearch/models/`
4. HuggingFace cache (`~/.cache/huggingface/hub/`)

### Embedder Auto-Detection

When you call `EmbedderStack::auto_detect()`, frankensearch probes for available models:

1. **FastEmbed** (MiniLM-L6-v2) — check for `model.onnx` in model paths
2. **Model2Vec** (potion-128M) — check for `model.safetensors` + `tokenizer.json`
3. **FNV-1a Hash** — always available (fallback)

The best available model becomes the quality tier; the fastest becomes the fast tier. If only one model is found, both tiers use it (no refinement phase). If only the hash embedder is available, search works but without semantic understanding.

---

## Text Canonicalization

All text is preprocessed before embedding to maximize search quality:

```rust
use frankensearch::core::DefaultCanonicalizer;

let canon = DefaultCanonicalizer::default();
let clean = canon.canonicalize(raw_text);
```

The default pipeline applies these steps in order:

1. **NFC Unicode normalization** — ensures hash stability across different Unicode representations
2. **Markdown stripping** — removes `#`, `**`, `*`, `_`, `[text](url)` → `text`
3. **Code block collapsing** — keeps first 20 + last 10 lines of fenced code blocks
4. **Low-signal filtering** — removes pure-URL lines, import-only blocks, empty sections
5. **Length truncation** — caps at 2000 characters (configurable)

Query canonicalization is simpler (no markdown stripping or code collapsing) since queries are typically short natural language.

---

## Query Classification

frankensearch adapts its retrieval strategy based on query type:

| Query Class | Example | Strategy |
|-------------|---------|----------|
| `Empty` | `""` | Return empty results |
| `Identifier` | `"br-123"`, `"src/main.rs"` | Lean heavily lexical (exact match matters) |
| `ShortKeyword` | `"error handling"` | Balanced lexical + semantic |
| `NaturalLanguage` | `"how does the search pipeline work?"` | Lean heavily semantic |

Each class gets adaptive candidate budgets — identifiers fetch more lexical candidates, natural language queries fetch more semantic candidates. This avoids wasting compute on the wrong retrieval path.

---

## Troubleshooting

### "EmbedderUnavailable: potion-multilingual-128M"

The Model2Vec model files aren't found. Either download them or use the hash embedder:

```bash
# Option 1: Download models (requires 'download' feature)
frankensearch download potion-multilingual-128M

# Option 2: Set model directory
export FRANKENSEARCH_MODEL_DIR=/path/to/models

# Option 3: Use hash-only (always works, no semantic understanding)
# In code: let embedder = FnvHashEmbedder::new(384);
```

### "DimensionMismatch: expected 256, found 384"

Your vector index was built with a different embedder than you're querying with. Rebuild the index with the correct embedder, or use the embedder that matches the index dimension.

### "IndexCorrupted: bad magic bytes"

The FSVI file is damaged or not a valid frankensearch index. Re-index your documents.

### Quality refinement never completes

Check that the quality model (MiniLM-L6-v2) is downloaded and accessible. If ONNX Runtime can't load it, frankensearch yields `SearchPhase::RefinementFailed` with the fast results as your final answer. Set `FRANKENSEARCH_FAST_ONLY=true` to skip refinement entirely.

### High memory usage with large indices

FSVI indices are memory-mapped. The OS manages page caching, so `top`/`htop` may show high virtual memory but actual resident memory depends on access patterns. For very large indices, ensure your system has enough RAM to keep the hot portion cached, or consider using HNSW (ANN) to avoid scanning every vector.

---

## Limitations

- **CPU-only inference**: ONNX Runtime runs on CPU; no GPU acceleration support yet
- **Single-node**: Designed as an embedded library, not a distributed search cluster
- **English-optimized**: MiniLM-L6-v2 works best on English text; potion-multilingual-128M handles multiple languages for the fast tier
- **Brute-force default**: Vector search scans all vectors by default; HNSW ANN is optional and trades accuracy for speed at large scale
- **No incremental index updates**: Adding documents requires rebuilding the FSVI index (append support planned)
- **Model download size**: Full setup requires ~220MB of model files (potion: ~128MB, MiniLM: ~90MB)

---

## FAQ

### Why "frankensearch"?

It's stitched together from the best parts of three separate codebases (cass, xf, mcp_agent_mail_rust) — like Frankenstein's monster, but for search. Each project independently developed similar hybrid search systems; frankensearch extracts and unifies the common core.

### Why two tiers instead of one model?

The 223x speed gap between potion-128M (0.57ms) and MiniLM-L6-v2 (128ms) means you can show results to the user *before* the quality model even starts. In interactive applications, perceived latency matters more than final quality — and with two tiers, you get both.

### Why RRF instead of learned fusion?

Reciprocal Rank Fusion is rank-based, so it doesn't need score calibration between lexical and semantic sources. It's simple, well-studied, and produces consistently good results across diverse query types. The K=60 constant comes from the original Cormack et al. literature and has been validated across all three source codebases.

### Can I use frankensearch without Tantivy?

Yes. Without the `lexical` feature, you get pure semantic search (vector similarity only). The RRF fusion layer is skipped, and you get ranked results from vector search alone.

### Can I bring my own embedding model?

Yes. Implement the `Embedder` trait for your model and pass it to the searcher. The trait is object-safe (`dyn Embedder`) for runtime polymorphism.

### Does frankensearch phone home or send telemetry?

No. Everything runs locally. No network calls are made unless you explicitly enable the `download` feature and call the model download function.

---

## Async Architecture (asupersync)

frankensearch uses [asupersync](https://github.com/Dicklesworthstone/asupersync) exclusively for all async and concurrent operations. Tokio, hyper, reqwest, and the entire tokio ecosystem are not used anywhere in the dependency tree.

### Why Not Tokio?

asupersync provides structured concurrency with a capability context (`Cx`) that flows through every async call. This gives frankensearch three properties that tokio cannot:

1. **Cancel-correctness by construction.** When a parent scope drops, all child tasks are cancelled and their resources cleaned up. No orphan tasks, no leaked futures, no silent background work after shutdown.

2. **Two-phase channels.** The `reserve()/send()` pattern ensures that data placed into a channel is never lost on cancellation. A `reserve()` call allocates a slot; `send()` fills it. If the sender is cancelled between reserve and send, the reservation is released cleanly — no data silently vanishes.

3. **Deterministic testing.** `LabRuntime` provides virtual time, Dependency-Partial-Order Reduction (DPOR) schedule exploration, and correctness oracles (`QuiescenceOracle`, `ObligationLeakOracle`, `TaskLeakOracle`). Every concurrent test runs deterministically and explores interleavings systematically.

### How It Looks in Practice

All async functions take `&Cx` as their first parameter. The `Cx` flows down from the consumer's runtime — frankensearch never creates its own runtime:

```rust
use frankensearch::prelude::*;
use asupersync::{Cx, region};

// The consumer creates the runtime
region(|cx| async {
    let searcher = TwoTierSearcher::auto(cx, "./data").await?;

    searcher.search(cx, "distributed consensus", &config, |phase| {
        match phase {
            SearchPhase::Initial(results) => display_results(&results),
            SearchPhase::Refined(results) => update_display(&results),
            SearchPhase::RefinementFailed { initial_results, .. } => { /* keep showing initial */ }
        }
    }).await?;

    Ok(())
});
```

**Rayon is retained** for CPU-bound data parallelism (SIMD dot product scans across vector chunks). Rayon's work-stealing thread pool composes cleanly with asupersync's async tasks — they operate on separate scheduling domains.

### Four-Valued Results

asupersync replaces `Result<T, E>` with `Outcome<T, E>` in concurrent contexts:

| Variant | Meaning |
|---------|---------|
| `Outcome::Ok(T)` | Success |
| `Outcome::Err(E)` | Application error |
| `Outcome::Cancelled` | Parent scope was dropped (structured cancellation) |
| `Outcome::Panicked(P)` | Task panicked (replaces `catch_unwind`) |

This eliminates an entire class of bugs where cancelled tasks silently produce partial or corrupted results.

---

## Cross-Encoder Reranking

The optional FlashRank cross-encoder reranker (`rerank` feature) applies a second-pass neural scoring to the top candidates from RRF fusion, producing more accurate relevance judgments than embedding cosine similarity alone.

### How It Works

Cross-encoders differ from bi-encoders (the embedding models) in a fundamental way: instead of comparing pre-computed embeddings, a cross-encoder processes the query and document *together* through a transformer, allowing direct token-level attention between them. This is dramatically more accurate but also much slower (you can't pre-compute anything).

```
RRF top-30 candidates
    │
    ▼
┌────────────────────────┐
│  For each candidate:   │
│  input = [query, doc]  │
│  logit = ONNX(input)   │
│  score = sigmoid(logit)│
└──────────┬─────────────┘
           │
           ▼
  Re-sort by cross-encoder score
  Return top-10
```

### Graceful Failure

The reranking step never blocks search results. If the cross-encoder fails (model unavailable, ONNX error, text lookup failure), the step logs a warning and returns candidates with their original RRF scores unchanged. This is critical for a library — the caller should never receive an error just because an optional reranker hiccuped.

Text for reranking is looked up by `doc_id` via a caller-supplied closure, since `ScoredResult` intentionally does not carry document text (it would be wasteful for the majority of use cases that only need IDs and scores).

---

## HNSW Approximate Nearest Neighbors

For indices beyond ~50K vectors, brute-force scanning becomes the latency bottleneck. The optional HNSW (Hierarchical Navigable Small World) index (`ann` feature) provides sub-linear approximate nearest neighbor search.

### When to Use It

| Index Size | Brute-Force Latency | HNSW Latency | Recommendation |
|-----------|---------------------|--------------|----------------|
| <10K | <15ms | ~1ms | Brute-force (simpler, exact) |
| 10K-50K | 15-75ms | ~1ms | Either works |
| 50K-100K | 75-150ms | ~2ms | HNSW recommended |
| >100K | >150ms | ~3ms | HNSW strongly recommended |

### Configuration

```rust
let config = TwoTierConfig {
    hnsw_ef_search: 100,        // Query-time beam width (higher = more accurate, slower)
    hnsw_ef_construction: 200,  // Build-time beam width (higher = better graph, slower build)
    hnsw_m: 16,                 // Max connections per node
    ..Default::default()
};
```

HNSW builds on top of the same FSVI vector data — it reads the f16 vectors from the slab and constructs a navigable graph structure. The graph is persisted alongside the FSVI file with a `CHSW` magic number header. Recall@10 is typically >0.95 with the default parameters.

---

## Quantization Ladder

frankensearch supports four levels of vector quantization, trading memory for quality with formal error bounds at each level:

| Level | Memory/Vector (384d) | Quality Loss | Use Case |
|-------|---------------------|--------------|----------|
| f32 | 1,536 bytes | None | Gold standard, benchmarks |
| **f16** (default) | **768 bytes** | **<1%** | **Production default** |
| int8 | 384 bytes | <3% | Memory-constrained, 100K+ docs |
| Product Quantization | 48 bytes | <8% | Extreme scale, 1M+ docs |

### f16 (Default)

Every FSVI index stores vectors as IEEE 754 half-precision floats. The f16→f32 conversion happens inside the SIMD dot product kernel — `wide::f32x8` loads and widens 8 f16 values per cycle. The quality loss is negligible for cosine similarity because the relative ordering of dot products is preserved even with reduced mantissa precision.

### int8 Scalar Quantization

Per-dimension affine quantization: each dimension gets its own `(min, scale)` pair, and values are mapped to `[0, 255]`:

```
quantized[i] = round((value[i] - min[i]) / scale[i] * 255)
```

The formal quality bound: for any query `q` and document `x`:

```
|cos_sim(q, x) - cos_sim(q, x')| <= sqrt(sum((scale_i/255)^2)) / ||x||
```

For typical 384-dim embeddings with unit-normalized vectors, this bound is <0.03.

### Product Quantization

Splits the 384-dim vector into 48 sub-vectors of 8 dimensions each. Each sub-vector is assigned to the nearest of 256 learned centroids, stored as a single byte. Total storage: 48 bytes per vector — a 32x compression over f32. Distances are computed via pre-computed lookup tables, making search faster despite the compression. Quality degrades more than int8 but is acceptable for candidate retrieval followed by re-scoring with full-precision vectors.

---

## Background Indexing Pipeline

Adding documents to frankensearch is not just "embed and store" — it's a pipeline with backpressure, deduplication, and atomic index replacement.

### Embedding Job Queue

New documents enter through a backpressure-controlled job queue. When the queue reaches capacity, `submit()` returns `SearchError::QueueFull { pending, capacity }` so the caller can implement retry logic (exponential backoff, drop-oldest, etc.).

Deduplication is built in: if you submit the same `doc_id` twice, the newer text replaces the older request in the queue. Additionally, a SHA-256 content hash is computed before embedding — if the document text hasn't changed since the last indexing, the embedding step is skipped entirely.

### Batch Coalescing

Individual embedding requests are coalesced into optimal batches before being sent to the model. ONNX inference has high fixed overhead (~128ms per call for MiniLM), but marginal cost per additional input is tiny (~0.4ms). Batching 32 texts together yields 29x throughput improvement.

Two priority lanes ensure interactive search queries (tight ~15ms budget) are never delayed by background indexing batches:

| Lane | Trigger | Max Wait |
|------|---------|----------|
| Interactive | Search query embedding | 5ms |
| Background | Document indexing | 10ms or batch full (32) |

### Index Refresh Worker

A dedicated asupersync region task acts as the **single writer** to vector indices. All reads happen through `RwLock::read()`. The worker:

1. Receives embedding batches from the job queue
2. Updates the FSVI index (append to WAL or full rebuild)
3. Updates the Tantivy lexical index
4. Writes the staleness sentinel file

Cancellation is handled via `Cx` — when the parent scope drops, the worker's receive loop returns `Outcome::Cancelled` and exits cleanly. Failed batches are logged and skipped; the worker continues processing subsequent commands.

### Staleness Detection

A sentinel file (`.frankensearch_sentinel.json`) records the last modification timestamp, document count, and embedder revision hash. On startup, the `StalenessDetector` checks whether the index is stale (embedder version changed, documents added/removed) and triggers a rebuild if needed. `ArcSwap` provides lock-free reads during atomic index replacement — existing readers continue using the old index until they finish, while new readers get the refreshed version.

---

## Incremental Index Updates

FSVI indices support append-only mutations via a Write-Ahead Log, eliminating the need to rebuild the entire index when adding documents:

```rust
// Append a single document (<100μs)
index.append(cx, "doc-42", &embedding).await?;

// Append a batch (<5ms for 100 docs)
index.append_batch(cx, &batch).await?;

// Background compaction merges WAL into main index
if index.needs_compaction() {
    index.compact(cx).await?; // Atomic rename swap
}
```

The WAL (`.fsvi.wal`) uses the same binary layout as the main FSVI file, so the search path simply merges results from both via `BinaryHeap`. Compaction is triggered automatically at 10% WAL-to-main size ratio or 1,000 accumulated records, and runs as a background task with atomic rename for crash safety. Soft-delete tombstones in the WAL are naturally cleaned up during compaction.

---

## Self-Healing Storage (FrankenSQLite + RaptorQ)

frankensearch integrates [FrankenSQLite](https://github.com/nicholasgasior/frankensqlite) for crash-safe metadata storage and [RaptorQ](https://tools.ietf.org/html/rfc6330) fountain codes for self-healing durability across all persistent files.
Enable this stack with `persistent` (storage) or `durable` (storage + repair).

### FrankenSQLite Integration

FrankenSQLite provides MVCC-isolated SQL storage for:

- **Document metadata** — `doc_id`, title, timestamps, tags, source, content hash
- **Persistent embedding job queue** — survives process crashes, resumes on restart
- **Index metadata** — embedder revision, dimension, record count, last compaction timestamp
- **Staleness detection** — storage-backed detector with atomic updates

All writes use WAL mode with `PRAGMA synchronous = NORMAL` for the durability/performance sweet spot. The content-hash deduplication layer queries FrankenSQLite to check whether a document's SHA-256 hash has changed before re-embedding.

### RaptorQ Erasure Coding

Every persistent binary file (FSVI vector indices, Tantivy segment files) is protected by RaptorQ repair symbols — a fountain code that can recover the original data from any K-of-(K+R) symbols:

```
┌─────────────────────────────────────────┐
│           Original FSVI File            │
│         (73 MB, 18,688 symbols)         │
├─────────────────────────────────────────┤
│        RaptorQ Repair Trailer           │
│  3,738 repair symbols (14.6 MB, ~20%)  │
│  xxh3_64 checksum per symbol            │
│  Deterministic seed for reproducibility │
└─────────────────────────────────────────┘
```

**Corruption detection** operates in three tiers:
1. **Fast** (<1ms): xxh3_64 whole-file checksum
2. **Medium** (~10ms): per-symbol CRC32 verification
3. **Full** (~100ms): complete RaptorQ decode and comparison

**Automatic repair** runs on index load when `verify_on_open` is enabled (the default). If corruption is detected, the repair pipeline backs up the corrupted file, reconstructs the original from the surviving symbols, verifies the reconstruction, and overwrites the file. Every repair event is logged as a JSONL record with timestamp, file path, corrupted byte count, decode time, and before/after hashes.

The 20% storage overhead buys remarkable resilience — a 73MB FSVI index can survive the loss of any 3,738 of its 22,426 total symbols (16.7% corruption tolerance).

### FTS5 Alternative Engine

In addition to Tantivy, frankensearch supports SQLite FTS5 as an alternative lexical search backend via the `LexicalSearch` trait. FTS5 is lighter-weight than Tantivy (no separate index directory, runs inside the same FrankenSQLite database) and is suitable for smaller corpora or environments where minimizing dependencies matters. The `LexicalSearch` trait in `frankensearch-core` defines the abstract interface; both `TantivyIndex` and the FTS5 adapter implement it.

---

## S3-FIFO Embedding Cache

Repeated queries and overlapping document sets benefit from caching embedding results. frankensearch uses the S3-FIFO algorithm (Yang et al., SOSP 2023), which achieves LRU-competitive hit rates with simpler, FIFO-based eviction:

```
┌──────────┐   promote on   ┌──────────┐
│  Small   │ ──── hit ────► │   Main   │
│  Queue   │   (10% cap)    │  Queue   │
│  (FIFO)  │                │  (FIFO)  │
└────┬─────┘                └────┬─────┘
     │ evict                     │ evict
     ▼                           ▼
┌──────────┐               (item dropped)
│  Ghost   │
│  Queue   │  ← tracks recently evicted keys
│(keys only)│    re-admission on ghost hit
└──────────┘
```

### Why S3-FIFO Over LRU?

LRU requires updating a linked list on every access — O(1) but with pointer-chasing and poor cache locality. S3-FIFO uses three FIFO queues with a simple promotion rule: items hit in the small queue get promoted to main. This produces comparable hit rates to LRU while being friendlier to concurrent access (no per-access list manipulation).

### Cache Bypass

A hit-rate exponential moving average (`alpha = 0.01`) continuously monitors cache effectiveness. If the hit rate drops below 30% sustained, the cache logs a warning and bypasses itself (direct computation without cache lookups), avoiding the overhead of maintaining a cache that isn't helping. This smooths over transient workload changes without thrashing.

Cache values are wrapped in `Arc<V>` to avoid cloning embedding vectors (each `Vec<f32>` is 1.5KB for 384 dimensions).

---

## Adaptive & Statistical Components

Beyond the core search pipeline, frankensearch includes a suite of adaptive and statistical components for production monitoring, safe experimentation, and automatic parameter tuning.

### Score Calibration

Raw scores from different sources (BM25, cosine similarity, cross-encoder logits) have different scales and distributions. The calibration service maps them to well-calibrated probabilities:

| Method | Best For | Training Cost |
|--------|----------|---------------|
| Temperature Scaling | Cross-encoder logits | O(1) parameter |
| Platt Scaling | BM25 / RRF scores | O(n) logistic regression |
| Isotonic Regression | Any source with sufficient data | O(n log n) |

Calibration quality is monitored via Expected Calibration Error (ECE) computed over rolling windows of 500 queries. If ECE exceeds 0.10 for 5 consecutive windows (2,500 queries of poor calibration), the system falls back to identity mapping and triggers a retrain.

### Bayesian Adaptive Fusion

The blend factor (how much to weight quality vs. fast scores) and RRF K constant don't need to be static. A Beta-Bernoulli Thompson sampling model learns optimal parameters online:

- **Blend factor** (`quality_weight`): Maintains separate Beta posteriors per query class (Identifier, ShortKeyword, NaturalLanguage). After each query, updates the posterior based on whether quality refinement improved NDCG. Over time, converges to the optimal per-class blend factor.

- **RRF K**: A Normal-Normal conjugate model adapts the RRF constant based on observed rank correlation between sources. Operates on a slower timescale (daily) since K changes affect the entire fusion layer.

The adaptive parameters are stored in the evidence ledger as JSONL, enabling full audit trail and rollback.

### Conformal Prediction Quality Guarantees

frankensearch provides distribution-free statistical guarantees on search quality via conformal prediction:

```rust
let calibrator = ConformalSearchCalibration::calibrate(cx, &searcher, &cal_set).await?;

// "How many results do I need to guarantee 95% recall?"
let required_k = calibrator.required_k(0.05); // alpha = 0.05 → 95% coverage

// "How surprising is this search result?"
let p_value = calibrator.p_value(query, &results);
```

The key guarantee: if you retrieve `required_k` results, the relevant document is in that set with probability at least `1 - alpha`. This holds regardless of the embedding model, query distribution, or corpus — it's a finite-sample guarantee from conformal inference.

Adaptive Conformal Inference (ACI) adjusts `alpha` online after distribution shifts (e.g., new documents indexed, embedder updated), re-establishing coverage within O(1/gamma) queries.

### Sequential Testing Gates (E-Processes)

Phase transitions (when to skip quality refinement, when to switch from exploration to exploitation) are governed by anytime-valid e-processes rather than fixed thresholds:

```
e_value = Π e_factor_i    (running product, one per query)

e_factor = Beta_pdf(tau; 2, 5) / Beta_pdf(tau; 5, 2)
```

where `tau` is the Kendall rank correlation between fast-only and quality-refined rankings, rescaled to [0, 1].

- **e_value > 20** → strong evidence that quality refinement helps → always refine
- **e_value < 0.05** → strong evidence that fast is sufficient → skip quality
- **In between** → keep gathering evidence

The key advantage over fixed thresholds: Ville's inequality guarantees `P(e_value ever exceeds 1/alpha) <= alpha` under the null hypothesis, providing continuous monitoring without multiple-testing correction.

### Off-Policy Evaluation

Before deploying a new ranking model or changing fusion parameters, frankensearch can evaluate the change offline using logged search data:

```rust
let evaluator = OffPolicyEvaluator::new(target_policy, logging_policy);

let estimate = evaluator.doubly_robust_estimate(&logged_data)?;
// DR = (1/N) Σ [ r_hat(x_i) + w_i * (r_i - r_hat(x_i)) ]

if estimate.ess > 100 && estimate.improvement > 0.05 {
    // Safe to deploy: sufficient overlap and meaningful improvement
}
```

The doubly robust estimator is unbiased if *either* the reward model or the importance weights are correct — a strict improvement over naive IPS. Weight clipping (default max 10.0) provides variance reduction with bounded bias.

---

## Monitoring & Observability

### Robust Statistics

Search latency distributions are heavy-tailed — a few slow queries dominate the mean. frankensearch uses robust estimators that aren't distorted by outliers:

| Primitive | Purpose | Complexity |
|-----------|---------|------------|
| **TDigest** | Streaming percentiles (p50, p95, p99) | O(1) per update, ~5KB memory |
| **MedianMAD** | Robust center and spread | O(1) via TDigest median |
| **Huber M-estimator** | Location estimation (95% efficient, bounded influence) | O(1) IRLS per update |
| **HyperLogLog** | Cardinality estimation (unique queries, docs) | 12KB for <2% error |

All primitives support merge operations for aggregating metrics across asupersync tasks, and run within a sub-500ns per-update budget.

### Structured Tracing

Every search operation emits structured spans via the `tracing` crate:

```
search
├── phase0::canonicalize          0.02ms
├── phase0::classify_query        0.01ms
├── phase1::fast_embed            0.57ms
├── phase1::vector_search         3.2ms
├── phase1::lexical_search        2.1ms
├── phase1::rrf_fusion            0.08ms
├── yield::initial                ← 6.0ms total
├── phase2::quality_embed         128ms
├── phase2::quality_search        3.5ms
├── phase2::blend                 0.04ms
├── phase2::rerank                12ms (optional)
└── yield::refined                ← 150ms total
```

Set `FRANKENSEARCH_LOG_FORMAT=json` for machine-parseable output. In release builds, TRACE and DEBUG events are stripped at compile time via `tracing`'s `max_level_info` feature for zero overhead.

### Quality-Tier Circuit Breaker

The circuit breaker pattern automatically skips quality refinement when it consistently fails to improve results:

```
Closed (normal) ──5 failures──► Open (skip quality)
                                     │
                               30s cooldown
                                     │
                                     ▼
                               HalfOpen (test)
                                     │
                              3 successes
                                     │
                                     ▼
                               Closed (normal)
```

A "failure" is any of: quality phase exceeds the latency threshold (500ms), returns an error, or produces Kendall tau improvement below 0.05 (quality didn't materially re-rank results). The circuit breaker adds <1 microsecond per query and can save 128ms+ on every skipped quality phase.

### FrankenTUI Observability Console

For operators running frankensearch across multiple projects, the FrankenTUI control plane provides:

- **Fleet Overview**: auto-discovered frankensearch instances with index sizes, embedding progress, and health status
- **Live Search Stream**: real-time query feed with latency sparklines and phase breakdowns
- **Index/Embedding/Resource Monitoring**: CPU, memory, I/O pressure per project with rolling aggregates (1m, 15m, 1h, 6h, 24h)
- **Historical Analytics**: trend analysis via FrankenSQLite persistence with configurable retention
- **Explainability Cockpit**: drill into any query to see per-hit score decomposition
- **Alerts & SLO Health**: error budget consumption, latency percentile tracking, anomaly detection

Telemetry is collected via lightweight instrumentation hooks in the search pipeline and stored in a FrankenSQLite database with batched ingestion and automatic downsampling for long-term retention.

---

## Result Explanations & Diversity

### Per-Hit Explanations

When `config.explain = true`, each search result carries a `HitExplanation` decomposing exactly why it ranked where it did:

```rust
HitExplanation {
    final_score: 0.847,
    components: [
        ScoreComponent::LexicalBm25 { matched_terms: ["consensus"], tf: 3, idf: 4.2, raw: 12.6, normalized: 0.82 },
        ScoreComponent::SemanticFast { embedder: "potion-128M", cosine_sim: 0.71 },
        ScoreComponent::SemanticQuality { embedder: "MiniLM-L6-v2", cosine_sim: 0.89 },
        ScoreComponent::Rerank { model: "ms-marco-MiniLM", logit: 2.3, sigmoid: 0.91 },
    ],
    rank_movement: RankMovement { initial: 4, refined: 1, delta: -3, reason: "promoted by quality embedder" },
}
```

Explanations have zero overhead when disabled — the explain flag is checked before any allocation. When enabled, they add ~2-5% latency overhead from collecting the component scores.

### MMR Diversified Ranking

Maximum Marginal Relevance prevents near-duplicate results from dominating the top-k by balancing relevance with diversity:

```
MMR(d) = λ × Sim(d, query) - (1-λ) × max(Sim(d, d_selected))
```

With the default λ=0.7 (2.3x relevance vs. diversity weighting), MMR iteratively selects documents that are both relevant to the query and dissimilar to already-selected results. The greedy selection runs in O(k×n) — under 1ms for typical top-10 from 30 candidates.

### Pseudo-Relevance Feedback

Between Phase 1 and Phase 2, frankensearch optionally uses the fast-tier results as pseudo-relevance signal to nudge the quality-tier query via Rocchio adaptation:

```
quality_query = α × original_query + (1-α) × centroid(top_k_feedback)
```

With α=0.8, the quality embedding is 80% original query and 20% shifted toward the neighborhood of fast-tier hits. This bridges the gap between different embedding spaces and improves recall@10 by 5-15% for ambiguous or multi-faceted queries. Guard rails ensure expansion only fires for NaturalLanguage queries with at least 3 feedback documents.

---

## Advanced Search Features

### Federated Search

`FederatedSearcher` queries multiple independent indices in parallel and fuses results via scatter-gather:

```rust
let federated = FederatedSearcher::new()
    .add_index("tweets", tweet_searcher, 1.0)
    .add_index("likes", likes_searcher, 0.8)
    .add_index("dms", dms_searcher, 1.2);

let results = federated.search(cx, "interesting thread", 10, FusionMethod::CombMNZ).await?;
// Results include source_index, source_rank, and appeared_in count
```

Three fusion methods are supported: RRF (rank-based), WeightedScore (normalized sum), and CombMNZ (score × count_of_indices). CombMNZ naturally boosts documents appearing in multiple indices, which strongly correlates with relevance. Per-index timeouts ensure a slow index doesn't block the entire search.

### MRL Adaptive Dimensionality

For Matryoshka Representation Learning models (where the first N dimensions capture most of the variance), frankensearch can truncate query vectors for an initial fast scan, then re-score the top candidates at full dimensionality:

```rust
let config = TwoTierConfig {
    mrl_search_dims: 64,        // Initial scan with first 64 dims (6x faster)
    mrl_rescore_top_k: 30,      // Re-score top 30 at full 384 dims
    ..Default::default()
};
```

For a 10K-vector index, searching at 64 dims + re-scoring 30 candidates takes ~3ms vs. ~15ms for a full 384-dim scan — a 5x speedup with negligible quality loss (the first 64 dimensions of MRL models capture >90% of variance).

### Typed Filter Predicates

Search results can be filtered by structured metadata without touching the ranking pipeline:

```rust
let results = searcher.search_filtered(cx, "error handling", 10, &[
    Filter::Eq("language", "rust"),
    Filter::Gte("created_at", "2025-01-01"),
    Filter::In("tags", &["async", "concurrency"]),
]).await?;
```

Filters are applied post-retrieval (after RRF fusion) to preserve ranking quality. For high-selectivity filters, a pre-filter hint can be passed to the lexical search backend to reduce candidate set size.

### Prefix-Optimized Incremental Search

For typeahead/autocomplete interfaces, `search_prefix()` provides optimized incremental search that reuses computation from the previous keystroke:

```rust
let mut state = IncrementalSearchState::new();
for query in ["d", "di", "dis", "dist", "distr", "distrib"] {
    let results = searcher.search_prefix(cx, query, 5, &mut state).await?;
    display(results);
}
```

The incremental state caches the previous query's embedding and Tantivy query plan. When the new query is a prefix extension of the previous one, only the delta is computed — avoiding redundant embedding calls for each keystroke.

### Negative Query Syntax

Queries support exclusion terms to filter out unwanted results:

```rust
// "consensus" but NOT "blockchain"
let results = searcher.search(cx, "consensus -blockchain", 10, &config, on_phase).await?;
```

Negative terms are parsed from the query string and applied as exclusion filters on the lexical side (Tantivy `BooleanQuery` with `MustNot` clauses). On the semantic side, the negative term's embedding is used to penalize similar vectors in the result set.

---

## Algorithms Deep Dive

### Reciprocal Rank Fusion (RRF)

RRF combines ranked lists from different sources without requiring score calibration:

```
RRF_score(d) = Σ_source 1 / (K + rank_source(d))
```

With K=60 (Cormack et al., 2009), a document ranked #1 in one source gets score `1/61 ≈ 0.0164`, while rank #10 gets `1/70 ≈ 0.0143`. The K constant controls how much top ranks dominate — higher K flattens the distribution.

**Four-level tie-breaking** when RRF scores are identical:
1. Documents appearing in both sources rank higher (stronger relevance signal)
2. Higher lexical score (BM25 captures explicit keyword relevance)
3. Higher semantic score (cosine captures meaning)
4. Lexicographic doc_id (deterministic, reproducible)

### Top-k Heap Guard Pattern

Vector search uses a min-heap of capacity k as a filter guard:

```rust
let mut heap = BinaryHeap::with_capacity(k);
for (idx, score) in dot_products.enumerate() {
    if heap.len() < k {
        heap.push(Reverse((score, idx)));
    } else if score > heap.peek().unwrap().0 .0 {
        heap.pop();
        heap.push(Reverse((score, idx)));
    }
}
```

The guard pattern avoids allocating a full sorted list — only k items are ever in memory. For 100K vectors with k=10, this avoids sorting 100K floats and instead does 100K comparisons against the heap minimum. All comparisons use `f32::total_cmp()` to handle NaN values deterministically (NaN sorts below all real values).

### SIMD Dot Product

The inner loop of vector search computes dot products between the query vector (f32) and stored vectors (f16), using `wide::f32x8` for 8-wide SIMD:

```rust
// Process 8 dimensions per iteration
for chunk in 0..dim/8 {
    let query_chunk = f32x8::from(&query[chunk*8..]);
    let stored_chunk = f16_to_f32x8(&vectors[offset + chunk*8..]);
    accumulator += query_chunk * stored_chunk;
}
let score = accumulator.reduce_add(); // Horizontal sum
```

This works portably across x86 (SSE2/AVX2) and ARM (NEON) — `wide` handles the architecture dispatch. For 384-dim vectors, this is 48 multiply-accumulate iterations per dot product, completing in under 2 microseconds.

### Two-Tier Score Blending

When both fast and quality embeddings produce scores for a document, they are blended:

```
blended_score = α × quality_score + (1-α) × fast_score
```

where `α = config.quality_weight` (default 0.7). For documents appearing in only one source:

- **Fast only**: `(1-α) × fast_score` — naturally penalized for missing quality confirmation
- **Quality only**: `α × quality_score` — naturally penalized for not being in fast results

This asymmetric penalty means documents found by both tiers get a natural boost — the same principle as RRF's `in_both_sources` preference, but applied at the score level rather than the rank level.

---

## Extending frankensearch

### Custom Embedder

Implement the `Embedder` trait to plug in any embedding model:

```rust
use frankensearch::core::{Embedder, ModelCategory, SearchResult};

struct MyEmbedder { /* ... */ }

impl Embedder for MyEmbedder {
    async fn embed(&self, cx: &Cx, text: &str) -> SearchResult<Vec<f32>> {
        // Your embedding logic here
    }

    fn dimension(&self) -> usize { 768 }
    fn id(&self) -> &str { "my-custom-embedder" }
    fn is_semantic(&self) -> bool { true }
    fn category(&self) -> ModelCategory { ModelCategory::Quality }
}
```

The trait is dyn-compatible via the `trait_variant` crate, which generates a `SendEmbedder` variant for use in `Box<dyn SendEmbedder>` contexts.

### Custom Canonicalizer

Replace or extend the text preprocessing pipeline:

```rust
use frankensearch::core::Canonicalizer;

struct DomainCanonicalizer;

impl Canonicalizer for DomainCanonicalizer {
    fn canonicalize(&self, text: &str) -> String {
        // Domain-specific preprocessing
        // e.g., expand abbreviations, normalize jargon
        let clean = DefaultCanonicalizer::default().canonicalize(text);
        expand_domain_abbreviations(&clean)
    }

    fn content_hash(&self, text: &str) -> [u8; 32] {
        // SHA-256 of canonicalized text for dedup
        sha256(self.canonicalize(text).as_bytes())
    }
}
```

### Custom LexicalSearch Backend

Implement `LexicalSearch` to use any full-text search engine:

```rust
use frankensearch::core::{LexicalSearch, ScoredResult, IndexableDocument, SearchError};

struct SqliteFts5Index { /* ... */ }

impl LexicalSearch for SqliteFts5Index {
    async fn search(&self, cx: &Cx, query: &str, limit: usize)
        -> Result<Vec<ScoredResult>, SearchError> { /* ... */ }

    async fn index_document(&self, cx: &Cx, doc: &IndexableDocument)
        -> Result<(), SearchError> { /* ... */ }

    async fn commit(&self, cx: &Cx) -> Result<(), SearchError> { /* ... */ }

    fn doc_count(&self) -> usize { /* ... */ }
}
```

The `LexicalSearch` trait lives in `frankensearch-core` alongside `Embedder` and `Reranker`, following the pattern: core defines interfaces, implementation crates provide concrete types.

---

## Companion Projects

### frankensearch-fast-search (fsfs)

A standalone machine-wide search product built on frankensearch. `fsfs` crawls and indexes text files across your entire machine — code, documentation, configuration, notes — with intelligent filtering of low-value content (logs, vendored dependencies, generated artifacts, binary files).

Two interfaces:
- **Agent CLI**: JSON/TOON output for programmatic access from AI coding agents
- **Deluxe TUI**: Interactive search with galaxy-brain explainability screens, indexing dashboards, and resource pressure monitoring

Key differentiators from generic file search: adaptive compute-pressure control (backs off when host is busy), evidence-ledger-backed ranking decisions, conformal/e-process calibrated adaptive controllers, and deterministic audit trails for every indexing and ranking decision.

Dual-mode contract spec: `docs/fsfs-dual-mode-contract.md`

### FrankenTUI Operations Console

Fleet-wide observability for frankensearch instances across projects. Auto-discovers running instances, collects telemetry via lightweight instrumentation hooks, and presents operational dashboards with live search streams, index health monitoring, and historical trend analysis. Built on the FrankenTUI pattern library with full keyboard navigation, command palette, progressive disclosure, and deterministic replay for debugging.

---

## Acknowledgments

Built with:
- [Tantivy](https://github.com/quickwit-oss/tantivy) — Full-text search engine for Rust
- [fastembed-rs](https://github.com/Anush008/fastembed-rs) — ONNX-based text embeddings
- [wide](https://crates.io/crates/wide) — Portable SIMD for Rust
- [half](https://crates.io/crates/half) — IEEE 754 half-precision floats
- [memmap2](https://crates.io/crates/memmap2) — Memory-mapped file I/O
- [Model2Vec](https://github.com/MinishLab/model2vec) — Static token embedding models

Informed by the hybrid search implementations in:
- [cass](https://github.com/Dicklesworthstone/cass) — Claude Agent Session Search
- [xf](https://github.com/Dicklesworthstone/xf) — X/Twitter archive search
- [mcp_agent_mail_rust](https://github.com/Dicklesworthstone/mcp_agent_mail_rust) — Agent coordination mail system

---

## About Contributions

Please don't take this the wrong way, but I do not accept outside contributions for any of my projects. I simply don't have the mental bandwidth to review anything, and it's my name on the thing, so I'm responsible for any problems it causes; thus, the risk-reward is highly asymmetric from my perspective. I'd also have to worry about other "stakeholders," which seems unwise for tools I mostly make for myself for free. Feel free to submit issues, and even PRs if you want to illustrate a proposed fix, but know I won't merge them directly. Instead, I'll have Claude or Codex review submissions via `gh` and independently decide whether and how to address them. Bug reports in particular are welcome. Sorry if this offends, but I want to avoid wasted time and hurt feelings. I understand this isn't in sync with the prevailing open-source ethos that seeks community contributions, but it's the only way I can move at this velocity and keep my sanity.

---

## License

MIT
