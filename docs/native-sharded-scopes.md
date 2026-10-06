# Source-scoped native sharded queries

`NativeBuiltShardedHybridIndex::scope` freezes a fallible predicate over the
complete retained source cohort. The returned public type is
`native_ann::builder::scope::sharded::NativeScopedShardedHybridIndex`.
It supports fast, independently retrieved quality, refined, progressive and
cross-encoder-reranked queries using the existing sharded query engine.

```rust,ignore
// Retain one live snapshot for the complete query, including row interpretation.
let snapshot = live.snapshot(&cx).await?;
let scoped = snapshot.index().scope(&cx, |source| {
    Ok(source.metadata.get("project").is_some_and(|value| value == "alpha"))
})?;
let mut query = scoped.progressive(&cx, "retry network requests", 10)?;
while let Some(phase) = query.next_phase().await? {
    // Deliver before requesting the next phase; quality remains lazy.
}
// `scoped.document(id)` resolves only an eligible ORIGINAL source document.
```

The predicate runs once per retained source document when constructing the
scope. It can inspect original content, ID, title and metadata, and may fail.
An error or cancellation refuses the entire scope; no accepted prefix or
unrestricted fallback escapes. The predicate is not retained, need not be Send,
and is never re-evaluated during queries or candidate widening. Eligible IDs
are retained in an ordered set without copying document bodies or vector slabs.

## Filter before retrieval cutoffs, not after the result page

Every fast and quality vector partition uses the existing filtered ANN widening
or exact scan, before global candidate merging and tier normalization. Each
requested tier embeds the query once, regardless of its partition count.
The keyword lane uses the same immutable-reader widening implementation as
ordinary native source scopes: it widens until enough eligible candidates are
found or the original matching population is exhausted. It never concatenates
overlapping rank windows or transfers a hydration pin between readers.

All lanes are scoped before RRF fusion, hydration and optional reranking. The
global Quill population and its raw BM25 statistics remain unchanged. Very
selective keyword scopes may examine the complete matching population; this is
not constant-time filtering or a performance guarantee. Native ANN remains
approximate. Empty scopes skip both embeddings, lexical search and reranking,
but do not bypass configured producer or tier-topology admission.

`search` does not run quality. `search_quality` does not run fast and refuses a
missing quality tier even for empty/zero-result requests. `search_refined`
requires every configured phase; use `progressive` to deliver Initial before
quality work and preserve it on explicit refinement failure.
`progressive_with_reranker` uses only eligible original source bodies. Neither
excluded documents nor fresh filesystem contents enter the model's window.

## Snapshot and result ownership

Results retain distinct `fast_row` and `quality_row` coordinates; `result.index`
remains absent. Each coordinate belongs to the exact shard inventory borrowed
by this scope. A live successor may repartition every document and change its
metadata, but cannot rebind an already-created scope or change an old query's
source text, membership, scores or row interpretation.

Create a new scope from a newly acquired snapshot to observe updated membership.
An old scope is intentionally not a revocable authorization capability for the
latest generation. Search scoping does not hide the full corpus from code that
already owns the index and is not a mutation authorization policy. Preserve the
normal immutable-directory contract while any Quill reader remains alive.

The ordinary and sharded source wrappers share a closed lexical adapter that
accepts only completed immutable hybrid cohorts. It cannot be constructed from
an arbitrary independently refreshed `LexicalRead`. The internal query-engine
scope hook also refuses unexpected unscoped keyword candidates rather than
silently allowing a vector-only restriction.

This is library query support; it does not add sharded CLI dispatch, durable
CURRENT publication, persistent antirollback, file collection, model downloads
or a new runtime. The existing default fsfs and single-cohort query routes are
unchanged. Synchronous graph/model/filesystem calls remain non-preemptible.

Focused regressions use the ordinary scope's existing identified deterministic
providers with real native/exact shards, global Quill and live updates:

```sh
cargo test -p frankensearch --features quill --lib native_ann::builder::scope
cargo test -p frankensearch --features quill --lib native_ann::hybrid::sharded
```

These tests cover candidate underfill, independent quality discovery, exact
stored scores and physical coordinates, strict restart, empty/error scopes,
old-query reranking across repartitioning, and failed or dropped refinement.
They are not real-model relevance, recall, memory or throughput evidence.
