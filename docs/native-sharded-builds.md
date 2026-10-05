# Complete native partitioned builds

`NativeIndexBuilder::build_sharded(&cx, max_documents_per_shard)` builds a
complete source/vector inventory through the existing FSVI v2 and native-HNSW
builder. The public handle is
`native_ann::builder::sharded::NativeBuiltShardedIndex`.

All documents are sorted by canonical ID before partitioning. IDs must be
unique across the entire input. Every partition contains the same source
membership in fast and, when configured, quality; all partitions share one
complete generation identity and the original per-tier producer identities.
The providers can differ in dimension between tiers. Existing precision,
graph parameters/seed, batch size, byte limits and explicit batch-failure
splitting apply to every child. Empty input produces exactly one empty,
identity-bearing partition. A zero partition size or more than 1024 partitions
is rejected before creating the root or running a model.

The builder moves source bodies into partitions and shares admitted vector
owners and native graphs with the existing `NativeShardSet`. It does not retain
a second copy of every vector or graph. Each query embeds once per requested
tier, not once per partition. Global top-k merging, identity checks and physical
shard coordinates use the existing shard implementation. Quality-primary
retrieval runs no fast inference. Native ANN retains its approximate search
semantics; partitioning alone is not a recall or performance guarantee.

```rust,ignore
use frankensearch::native_ann::builder::{NativeIndexBuilder, sharded::{
    NativeBuiltShardedIndex, NativeShardedReopenLimits,
}};

// cx, generation, documents, fast and quality belong to the caller.
let built = NativeIndexBuilder::new(&new_directory, generation, fast.clone())?
    .with_quality_embedder(quality.clone())?
    .add_documents(documents)
    .build_sharded(&cx, 10_000)
    .await?;
let selected = built.seal_for_reopen(&cx)?;
// Keep `selected` independently in the caller's trusted catalog.
let reopened = NativeBuiltShardedIndex::open_selected(
    &cx, &new_directory, &selected, fast, Some(quality),
    NativeShardedReopenLimits::default(),
)?;
let hits = reopened.search_quality(&cx, "retry a failed request", 10).await?;
for hit in hits {
    // hit.row.shard plus physical_row, never a bare row across partitions.
    let source = reopened.document(hit.doc_id.as_str()).unwrap();
}
```

The vector/source seal is `native.sharded.json`, referring to exact ordered
child receipts in `shard-000000`, `shard-000001`, and so on. Names are derived
from ordinal positions, not accepted as paths in the descriptor. Each child is
sealed by the ordinary native source/vector snapshot implementation. The root
receipt is returned only after all children and the root descriptor are synced.
A caller cannot obtain a successful prefix when a later child fails. Failed or
dropped builds and partial seals remain on disk for inspection; they are not
adopted or overwritten by a retry. Choose fresh destinations.

Restart authenticates the externally selected root receipt, preflights every
child descriptor, its declared generation/quality topology and producer, and
checks aggregate limits before opening the first vector or graph. It then
performs the existing strict selected-reopen admission of every source, vector
and configured graph. A missing/corrupt native graph does not become exact
fallback. Ordered source membership and the total count must agree. The input
limits include maximum partitions, total documents, aggregate declared artifact
bytes, and existing per-partition source/record/vector/graph ceilings. Graph
receipt files conservatively consume their whole encoded-size allowance.

These are input accounting limits, not peak RSS or timing guarantees. All
source documents, vector owners and graphs stay resident. Synchronous model,
filesystem and graph operations run on the caller's lane and are not preempted
by cancellation. No hidden runtime, task, model download, CURRENT publisher,
external rollback floor or garbage collector is introduced. Directories remain
trusted and immutable; this is not descriptor-relative hostile-writer safety.

The vector/source handle alone does not certify an independently supplied keyword
reader. Use the complete hybrid API below for a joined lexical/source cohort.
Neither API provides a sharded mutation protocol or changes the CLI/default fsfs
layout. Ordinary unpartitioned formats and admission contracts stay unchanged.

## One global lexical population with partitioned vectors

With the existing `quill` feature, use
`NativeIndexBuilder::build_sharded_hybrid(&cx, max_documents_per_shard)` to obtain
`native_ann::builder::sharded::NativeBuiltShardedHybridIndex`.
It builds all fast/quality vector partitions first, then one global Quill index
from their exact original source bodies. The ordinary and partitioned builders
share the same deterministic bulk lexical implementation and stored-field
census. There is no new keyword engine or merger of shard-local BM25 scores.
Changing vector partition boundaries does not change the lexical population's
term frequencies or document-length statistics.

The builder requires every configured vector and lexical component to succeed
before returning a complete hybrid handle. After construction and every selected
restart, a full match-all census compares source IDs, original content, title and
canonical metadata with the retained lexical records, including empty and
punctuation-only bodies. A valid inventory with the same number of documents but
different stored fields is refused. The census uses one reference per source,
not a second copy of all bodies. The writer's publication lease remains held
through initial admission, then is released; serving keeps a read-only reader.

```rust,ignore
use frankensearch::native_ann::builder::sharded::{
    NativeBuiltShardedHybridIndex, NativeShardedHybridReopenLimits,
};

let built = NativeIndexBuilder::new(&new_directory, generation, fast.clone())?
    .with_quality_embedder(quality.clone())?
    .add_documents(documents)
    .build_sharded_hybrid(&cx, 10_000)
    .await?;
let receipt = built.seal_for_reopen(&cx)?;
// Retain this complete hybrid receipt in the trusted catalog.
let restarted = NativeBuiltShardedHybridIndex::open_selected(
    &cx, &new_directory, &receipt, fast, Some(quality),
    NativeShardedHybridReopenLimits::default(),
).await?;
let mut phases = restarted.progressive(&cx, "retry failed calls", 10)?;
while let Some(phase) = phases.next_phase().await? {
    // Deliver this phase before polling the next: quality inference is lazy.
}
```

`search` runs fast plus lexical retrieval, `search_refined` independently
retrieves quality and blends the global candidate union, and `search_quality`
skips fast inference entirely. The same existing native sharded engine handles
progressive phases and `progressive_with_reranker`: the cross-encoder receives
text only from this retained source cohort, once for its global candidate window.
A quality winner can be absent from every fast candidate. Scorer failure retains
the preceding page, and cancellation remains an error. Returned results carry
separate `fast_row` and `quality_row`; their `result.index` is deliberately None.
Rows travel with the original document through reranking, not with a scorer's
rank number. A query embeds once per requested tier, not once per partition.

The complete hybrid seal is `native.sharded-hybrid.json`. One trusted receipt
authenticates this descriptor, the exact global lexical inventory and the
`native.sharded.json` ordered vector/source inventory. The root hybrid receipt is
returned only after all children, lexical artifacts and final descriptor reach
their synchronization boundaries. Lexical bytes are rechecked against the seal
captured during construction, never recaptured from a later publication.
Only the completed hybrid receipt authorizes hybrid restart; a vector-only child
receipt is not promoted to hybrid authority.

Restart requires every selected source, vector, graph, descriptor and keyword
artifact. Missing or corrupt late partitions never become partial search success;
missing ANN never triggers exact fallback. Separate explicit budgets bound the
total vector/source inventory and total lexical bytes, with their existing
per-file/record limits. The combined descriptor itself is bounded to 32 MiB.
Ordinary source/graph/Quill admission is not replaced by descriptor self-checks.

All source/vector partitions remain resident, and Quill still maps its immutable
files. This is not disk-streaming query execution, a peak-memory guarantee,
distributed searching, a durable CURRENT/antirollback authority, or sharded CLI
activation/updates. Keep original artifacts immutable while any old reader lives.
No dependency, hidden runtime, model download, worker or destructive collector
is added. Real-model relevance, throughput and platform qualification remain
separate from this library integration.

Focused tests (model files are not required):

```sh
cargo test -p frankensearch --lib native_ann::builder::sharded
cargo test -p frankensearch --features quill --lib native_ann::builder::sharded::hybrid
```

The deterministic identified providers exercise the real vector/graph writers,
selected restart and shard queries. They do not establish semantic relevance,
real-model throughput, ANN recall, or platform qualification.
