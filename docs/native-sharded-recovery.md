# Authenticated recovery of a complete sharded source cohort

`NativeBuiltShardedIndex::recover_selected_source` and
`NativeBuiltShardedHybridIndex::recover_selected_source` recover owned prepared
source documents from the caller's original receipt. They return rebuild input,
not a partially admitted searchable index. Ordinary selected reopen and update
still require all of their original search artifacts and producing identities.

The hybrid path authenticates the outer complete-hybrid descriptor, the ordered
shard inventory, every child source/vector descriptor, and every source stream.
The vector/source-only path starts at the ordered inventory. Neither path scans
for a newer generation, invents an expected hash, skips a missing partition, or
uses an individually valid child receipt as authority for a complete hybrid.

All child descriptors, generation identities, quality topology, document counts
and aggregate input budgets are checked before the first source is decoded.
The exact authenticated source digests remain pinned across that preflight.
Every full source stream then uses the ordinary native bounded decoder, including
hash, length, source schema, unique ordered IDs and exact count checks. Partition
boundaries must remain globally ordered and disjoint. A malformed or missing last
source is an error; no successful prefix escapes. Empty cohorts retain one empty,
identity-bearing partition, not a missing inventory.

```rust,ignore
use frankensearch::native_ann::builder::sharded::{
    NativeBuiltShardedHybridIndex, NativeShardedHybridReopenLimits,
};

let (old_generation, documents) =
    NativeBuiltShardedHybridIndex::recover_selected_source(
        &cx, &damaged_directory, &trusted_receipt,
        NativeShardedHybridReopenLimits::default(),
    )?;
// Supply a fresh destination, strictly newer generation and explicitly selected
// local models to NativeIndexBuilder. Rebuild ALL vectors, graphs and Quill,
// then seal and independently persist/select the completed successor receipt.
```

The original models need not be installed. Old FSVI, ANN, graph-receipt and Quill
files are neither opened nor reused: they may be missing or corrupt. Prepared
content, title, metadata and IDs are returned exactly, without normalization or
model relabelling. Damaged descriptors or source streams cannot be reconstructed
by this mechanism. It is not an alternate corpus-discovery or partial-salvage API.

`NativeShardedReopenLimits::max_artifact_bytes` counts all descriptors and encoded
source streams read for recovery, excluding derived artifact bytes that are not
read. Hybrid recovery also charges its outer descriptor to this same total.
Existing per-partition source/record ceilings and aggregate document/shard limits
apply. Hybrid lexical limits still validate the selected descriptor's structure
and declared inventory, not the missing/damaged lexical files. The limits bound
input, not decoded-object overhead, peak RSS, inference scratch, or latency.

No model, runtime, worker, publisher, repair writer or garbage collector is
created. All work is synchronous on the caller's I/O/CPU lane; cancellation is
checked at boundaries but cannot preempt a blocking filesystem call. Paths retain
the existing trusted, immutable-directory contract, not hostile-writer safety.
Recovery never changes an old receipt, descriptor, source or selected head.

Focused deterministic tests use actual native/exact partition writers and seals:

```sh
cargo test -p frankensearch --lib native_ann::builder::sharded::recovery
cargo test -p frankensearch --features quill --lib native_ann::builder::sharded::recovery
```
