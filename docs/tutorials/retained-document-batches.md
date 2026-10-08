# Atomic mixed document batches

`FsfsRuntime::apply_retained_batch` applies explicit document upserts and exact-ID
deletions to a complete-generation store with one publication. It is a library
operation, not a new CLI command or an automatic watcher mode.

A caller supplies an admitted `PublishedGeneration` as the expected base. The
operation acquires publication ownership, rechecks that exact store/ID/inventory
identity, and refuses a changed base rather than silently applying stale changes
to somebody else's successor. Reusing an old receipt after publication is a
conflict; reopen and explicitly reconsider the input instead of blindly retrying.

## Calling the API

Use the runtime's configured, compatible producers and caller-owned `Cx`/blocking
capacity, just as for retained rebuilds and searches. The store must already have
a complete generation. This function does not bootstrap a corpus.

```rust
use std::path::Path;
use asupersync::Cx;
use frankensearch_core::{SearchError, SearchResult};
use frankensearch_fsfs::FsfsRuntime;
use frankensearch_fsfs::generation_store::{
    CompleteGenerationStore, COMPLETE_GENERATION_POINTER,
};
use frankensearch_fsfs::runtime::retained_batch::{
    RetainedBatchResult, RetainedMutation,
};

async fn apply_changes(
    runtime: &FsfsRuntime,
    cx: &Cx,
    root: &Path,
) -> SearchResult<RetainedBatchResult> {
    let store = CompleteGenerationStore::open(cx, root)?;
    let expected = store.active(cx)?.ok_or_else(|| SearchError::IndexNotFound {
        path: store.root().join(COMPLETE_GENERATION_POINTER),
    })?;
    runtime.apply_retained_batch(cx, store.root(), &expected, &[
        RetainedMutation::Upsert {
            id: "src/cache.rs".to_owned(),
            text: "The complete replacement document body".to_owned(),
        },
        RetainedMutation::Delete { id: "src/obsolete.rs".to_owned() },
    ]).await
}
```

IDs are opaque. The API never opens `src/cache.rs`, scans a source directory, or
interprets a delete as a prefix. An upsert replaces the whole body using the same
bounded embedding canonicalization and full-text lexical canonicalization as
retained append-batch. It does not merge document properties. Filesystem discovery,
privacy/classification policy and source authority remain the caller's job.

Operations are ordered and the last operation for an ID wins, including mixed
upsert/delete sequences. Every supplied ID and raw body is validated even when
superseded. Canonicalization and inference run only for final upserts. A final empty
canonical body is rejected; it is not interpreted as deletion.

## Atomicity and identity

All inference required by final upserts completes before candidate artifact
mutation. Fast and quality are admitted independently against their corresponding
stored tiers. Bound responses must retain the complete admitted identity, exact
dimension, in-memory representation and finite values. Producer drift during either
successful or failed inference is terminal. Cancellation wins over a racing provider
success or error. There is no raw fallback, native-batch assumption, per-document
retry or successful prefix from an otherwise failed batch.

One independent copy receives lexical changes, both present vector tiers, catalog
updates and paired membership manifests. Replacing a windowed source with fewer
windows retires its old trailing rows. Deletions remove the source's rows from both
tiers. Existing partial quality coverage of untouched documents stays unchanged.
An existing quality tier cannot be silently discarded merely because its producer
is unavailable for an upsert. Vector WALs are compacted and vacuumed before sealing,
and ordinary repair protection and search-generation admission remain in the path.

Old admitted readers keep their original immutable generation. No predecessor or
source file is deleted by the batch. Failed candidates remain inert, following the
store's existing retention protocol.

`RetainedBatchResult` contains final distinct-ID upsert/delete counts and an optional
publication. Empty input does no filesystem/model work. A nonempty batch containing
only absent deletions still checks its base and does not publish; it may leave an
inert staging directory. No-op counts do not certify semantic readiness.

Inspect the publication variant:

- `Durable` confirms the new pointer's filesystem durability boundary.
- `VisibleButDurabilityUncertain` means the generation is already selected. Do not
  replay the mutation as though it aborted. Use the store's receipt-bound
  `confirm_retained_durability` when confirmation is needed.
- `None` means no selection was changed by this batch.

An error before pointer publication leaves the previous selection intact. The
internal precommit seam lets an integrating source owner recheck its authority
after sealing and before the pointer switch; it does not replace engine admission.

## Work bounds and remaining watch integration

Admission permits at most 4096 operations, 8 MiB per raw or canonical document
body, 64 MiB of aggregate supplied ID/body bytes, 64 MiB of retained canonical
payload, and 128 MiB of staged vector values. Superseded raw operations still count
against supplied-input work. These are payload/work limits, not a process-wide RSS
ceiling; model state and the predecessor/candidate index are additional resources.

This is the atomic mixed-mutation prerequisite for `bd-dnqgr`, not its completion.
The complete-generation watcher still uses its existing rebuild route. Before
routing observed filesystem changes here, preserve the normal discovery,
classification, extraction, metadata and source-precommit contracts; handle
startup, overflow and forced reconciliation separately. This API does not establish
that a native notification alone is authoritative permission to delete a document.

Whole-generation copying, integrity checks and vector maintenance remain. No
sublinear publication cost, measured latency improvement, default-layout flip or
release-quality acceptance is claimed.

## Validation status

The implementation includes admission/producer regressions and persisted two-tier
fixtures for atomic mixed updates, old-reader preservation, stale bases, failed
inference/precommit, and shrinking semantic windows. They use explicit synthetic
providers for wiring, not real-model quality evidence. The authoring environment
had no Rust compiler, Cargo or rustfmt: compilation, tests, formatting, Clippy and
the unchanged full quality gate still need execution on a supported build host.

Relevant targeted commands (run through the repository's RCH workflow):

```sh
cargo test -p frankensearch-fsfs --lib retained_batch
cargo test -p frankensearch-fsfs --no-default-features --lib retained_batch
cargo fmt --all --check
cargo clippy -p frankensearch-fsfs --all-targets -- -D warnings
```
