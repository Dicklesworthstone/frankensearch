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

This first vector/source layer does not independently certify a keyword reader,
provide a sharded mutation protocol, or change the CLI/default fsfs layout.
Ordinary unpartitioned builders and selected snapshots keep their existing
formats and admission contracts.

Focused tests (model files are not required):

```sh
cargo test -p frankensearch --lib native_ann::builder::sharded
```

The deterministic identified providers exercise the real vector/graph writers,
selected restart and shard queries. They do not establish semantic relevance,
real-model throughput, ANN recall, or platform qualification.
