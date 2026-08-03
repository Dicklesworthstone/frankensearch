# frankensearch-quill

Quill is frankensearch's native, deterministic lexical search engine.

It provides BM25-ranked term, phrase, boolean, range, set, prefix, and glob
queries without embedding Tantivy in the production engine. Quill owns its
on-disk segment format, term dictionary, postings and positions codecs,
snapshot publication, tombstone-aware compaction, snippets, and
search-while-indexing delta segments.

## Design

Quill is built around a few explicit contracts:

- compile-time schemas for the default frankensearch, fsfs, and CASS layouts;
- rank-exact BM25 scoring with deterministic total ordering;
- immutable search snapshots while a writer publishes newer generations;
- two-slot durable manifests and blue-green generation publication;
- bounded query work with typed cancellation and exhaustion outcomes;
- block-max and MaxScore pruning that preserve the reference result set;
- no async runtime ownership: callers supply an `asupersync::Cx`.

Tantivy is available only through the opt-in `tantivy-oracle` feature for
differential conformance and benchmark controls. It is not part of Quill's
default dependency graph.

## Core types

- `QuillIndex` creates, opens, updates, commits, deletes, and compacts an index.
- `QuillSearchSnapshot` is an immutable, generation-bound search view.
- `QuillSearchResult` and `QuillSnippetHit` expose ranked and highlighted hits.
- `QuillConfig` controls sharding, compaction, work budgets, and publication.
- `DefaultQueryParser` and `CassQueryParser` produce typed query trees.
- `SchemaDescriptor` describes the admitted compile-time field layout.
- `KeeperWriter` and `SnapshotPublisher` own durable generation publication.

## Minimal configuration

```rust
use frankensearch_quill::{QuillConfig, DEFAULT_SCHEMA};

let config = QuillConfig::default();
assert_eq!(config.tier_fanout, 8);
assert_ne!(DEFAULT_SCHEMA.schema_id().expect("valid schema"), 0);
```

Creating or opening an index is asynchronous and requires the caller's
structured-concurrency context:

```text
QuillIndex::create(&cx, path, QuillConfig::default()).await
QuillIndex::open(&cx, path, QuillConfig::default()).await
```

The facade crate is the preferred entry point for hybrid applications. Use this
crate directly when you need lexical-only lifecycle, schema, or storage
control.

## Features

| Feature | Purpose |
|---|---|
| `durability` | Repair/protection sidecars for Quill artifacts. |
| `tantivy-oracle` | Differential tests against the pinned Tantivy oracle. |
| `bench-internals` | Same-binary measurement helpers; never a shipping profile. |

Default features are empty.

## Dependency graph position

```text
frankensearch-core     frankensearch-index
          \                 /
           \               /
            frankensearch-quill
               /         \
  frankensearch-fusion   frankensearch
```

`frankensearch-durability` and `frankensearch-lexical` are optional
dependencies for the `durability` and `tantivy-oracle` features respectively.

## License

MIT
