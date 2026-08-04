# frankensearch-lexical

Tantivy BM25 full-text search integration for frankensearch.

## Overview

This crate provides the `TantivyIndex` implementation of the split `LexicalRead` / `LexicalWrite` traits, wrapping Tantivy for BM25 keyword search. It handles schema creation, document indexing, query parsing, snippet generation, and result ranking. Title matches receive a 2x BM25 boost for improved relevance.

It also includes the CASS (Content-Addressable Search Schema) compatibility layer for advanced schema management, boolean query parsing, edge-ngram tokenization, and regex query caching.

## Schema

| Field | Tantivy Options | Source |
|-------|-----------------|--------|
| `id` | `STRING \| STORED` | `IndexableDocument::id` |
| `content` | `TEXT \| STORED` | `IndexableDocument::content` |
| `title` | `TEXT \| STORED` | `IndexableDocument::title` (empty if `None`) |
| `metadata_json` | `STORED` | Serialized `IndexableDocument::metadata` |

## Key Types

- `TantivyIndex` - main lexical search backend implementing `LexicalRead` + `LexicalWrite`
- `LexicalHit` - enriched search result with BM25 score, snippet, and query explanation
- `LexicalDocHit` - raw hit with BM25 score and Tantivy doc address for custom field extraction
- `LexicalIdHit` - lightweight hit with just doc ID and score for hot paths
- `QueryExplanation` - classification of parsed queries (Empty, Simple, Phrase, Boolean)
- `SnippetConfig` - configuration for highlighted snippet generation
- `CassTantivyIndex` - CASS-compatible Tantivy index with extended schema and query features

## Re-exported Tantivy Types

This crate re-exports commonly used Tantivy types so consumers do not need a direct Tantivy dependency:

- `Index`, `IndexReader`, `IndexWriter`, `Searcher`
- `Schema`, `Field`, `Term`, `DocAddress`
- `Query`, `BooleanQuery`, `TermQuery`, `TopDocs`

## Usage

```rust
use frankensearch_lexical::TantivyIndex;
use frankensearch_core::types::IndexableDocument;

// Create a lexical index
let index = TantivyIndex::create("/tmp/my_lexical_index")
    .expect("create index");

// Index documents
let doc = IndexableDocument {
    id: "doc-1".to_string(),
    content: "Rust ownership and borrowing explained".to_string(),
    title: Some("Memory Safety".to_string()),
    metadata: None,
};

// Search with snippets
// let results = index.search_with_snippets(&cx, "ownership", 10, &Default::default()).await?;
```

## Dependency Graph Position

```
frankensearch-core
  ^
  |
frankensearch-lexical
  ^
  |-- frankensearch-fusion (optional, feature: lexical)
  |-- frankensearch-fsfs
  |-- frankensearch (root, optional, feature: lexical)
```

## License

MIT
