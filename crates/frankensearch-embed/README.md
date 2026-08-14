# frankensearch-embed

Embedder implementations for the frankensearch hybrid search library.

## Overview

This crate provides three tiers of text embedding, each feature-gated for granular dependency control:

- **Hash** (`hash` feature, default): FNV-1a hash embedder with zero ML dependencies. An explicit control/test double, not a semantic search engine.
- **Model2Vec** (`model2vec` feature): potion-128M static embedder (~0.57ms per embed). Serves as the fast tier in two-tier search.
- **FastEmbed** (`fastembed` feature): MiniLM-L6-v2 ONNX embedder (~128ms per embed). Serves as the quality tier in two-tier search.

The `EmbedderStack` auto-detection system probes for locally available models and configures the best fast+quality embedder pair automatically.

## Key Types

- `EmbedderStack` - auto-detected fast+quality embedder pair with optional dimension reduction
- `DimReduceEmbedder` - wrapper that truncates embeddings to a target dimensionality
- `TwoTierAvailability` - diagnostic report of which model tiers are available
- `HashEmbedder` - FNV-1a hash-based embedder (feature: `hash`)
- `Model2VecEmbedder` - potion-128M static model embedder (feature: `model2vec`)
- `FastEmbedEmbedder` - MiniLM-L6-v2 ONNX embedder (feature: `fastembed`)
- `CachedEmbedder` - transparent embedding cache wrapper with hit/miss stats
- `BatchCoalescer` - batches concurrent embedding requests for throughput optimization
- `ModelCacheLayout` - manages the on-disk model cache directory structure
- `ModelManifest` / `ModelManifestCatalog` - model metadata, lifecycle, and SHA-256 verification
- `ModelDownloader` - downloads models from HuggingFace with progress tracking (feature: `download`)
- `EmbedderRegistry` - registry of known embedder and reranker implementations

## Features

| Feature | Description |
|---------|-------------|
| `hash` (default) | FNV-1a hash embedder, zero dependencies |
| `model2vec` | potion-128M static embedder via safetensors + tokenizers |
| `fastembed` | MiniLM-L6-v2 ONNX embedder via fastembed |
| `download` | Model auto-download from HuggingFace |
| `bundled-default-models` | Enables `model2vec` + `fastembed` together |

## Usage

```rust
use std::path::Path;
use frankensearch_embed::{EmbedderStack, HashEmbedder};

// Explicit control double for tests — not semantic search
let embedder = HashEmbedder::default_256();

// Production: refuse a hash-only stack. `auto_detect` / `auto_detect_with`
// still exist for tests and explicit control policy; they silently fall
// back to FNV when no model is cached.
let stack = EmbedderStack::auto_detect_semantic_with(Some(Path::new("/path/to/model/cache")))
    .expect("semantic model must be present");
// stack.fast()    -> fastest semantic embedder
// stack.quality() -> highest quality embedder (if available)
```

## Dependency Graph Position

```
frankensearch-core
  ^
  |
frankensearch-embed
  ^
  |-- frankensearch-fusion
  |-- frankensearch-fsfs
  |-- frankensearch (root)
```

## License

MIT
