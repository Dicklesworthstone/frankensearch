# Atomic complete-generation rebuilds

The `fsfs` executable can opt an index root into complete-generation publication.
A rebuild writes its lexical, vector, catalog and content artifacts into a fresh
bundle and selects that bundle only after the existing indexing and search
admission checks succeed. Old bundles and abandoned builds are retained.

This route is currently Unix-only. Start with a fresh store outside the source
tree, with an existing parent directory. In-place migration of a legacy index
is not qualified by this workflow. Configure a generation-local catalog:

```toml
[storage]
db_path = "{index_dir}/catalog.sqlite"
```

Provision verified semantic models as for ordinary indexing. This route does not
turn a missing model into a hash fallback, bypass producer checks, or repair old
vectors merely because a new model is installed.

## Build and query

Opt in for the initial build (the longer `FRANKENSEARCH_COMPLETE_GENERATIONS`
spelling takes precedence over `FSFS_COMPLETE_GENERATIONS`):

```sh
FSFS_COMPLETE_GENERATIONS=1 fsfs index /work/source \
  --index-dir /work/search-store --config /work/fsfs.toml --format json
```

The success payload identifies the generation, its path and inventory digest,
and reports `publication: "durable"`. If the pointer rename succeeds but its
following directory synchronization fails, the command returns an error that
explicitly says the new generation is already visible and durability is
uncertain. It does not restore the old pointer or claim the build was aborted.

Once initialized, the store is recognized without the environment variable:

```sh
fsfs search "connection pooling" --index-dir /work/search-store \
  --config /work/fsfs.toml --no-daemon --format json

fsfs search "connection pooling" --index-dir /work/search-store \
  --config /work/fsfs.toml --no-daemon --stream --format jsonl

fsfs index /work/source --index-dir /work/search-store \
  --config /work/fsfs.toml --format json
```

The last command builds a successor. A search resolves and admits its bundle
once; it does not switch artifacts between progressive phases. Search does not
write a result cache or explanation session into the sealed bundle. `status`
and `doctor` inspect the selected bundle rather than the store container.

A malformed or missing selection in an initialized store is an error, not a
reason to scan for another generation or fall back to the legacy mutable index.
Setting the opt-in variable to false does not override that protection. Do not
point mutation commands at a sealed generation subdirectory.

## Warm standard-input serving

A long-lived process can reuse admitted models and indexes while following
complete-generation publication between requests:

```sh
fsfs serve --index-dir /work/search-store --config /work/fsfs.toml --format jsonl
```

Send the existing JSON request shape, for example
`{"query":"connection pooling","limit":10}`, or a plain query line. Each
request uses the existing buffered `fsfs.search.serve.v3` response shape, after
one ready event. Send `quit` or close stdin to stop. This is not the progressive
socket transport: use direct `search --stream` for immediate phase delivery.

Before executing a request, the server checks the selection and fully admits
any successor before swapping resources. The runtime's hydration paths and
its lexical/vector resources stay together. A changed generation invalidates
the in-memory result cache. A malformed selection or failed admission returns
an error for that request, even when the old query was cached; repairing the
selection permits a later retry without silently serving stale hits.

Input is bounded to 1 MiB before JSON parsing; responses are fully encoded under
the existing 4 MiB daemon-response limit before bytes are emitted. Output errors
return and release the retained reader. Like the existing stdio server, stdin
reads block on the command's owning lane: cancellation is observed between
reads and requests, not promised to interrupt an idle blocked read. No detached
input worker is introduced.

## Watching and current boundaries

Complete-store routing also supports `watch` and `index --watch`, using full
replacement builds without holding a serving-index writer between publications.
See [complete-generation watch](complete-generation-watch.md) for the bounded
notification queue, source recheck, framed receipts and remaining update costs.
The existing buffered Unix socket server and daemon shutdown path are preserved.

TUI, direct query expansion, automatic search-to-daemon forwarding and in-place
mutators remain outside this route. Legacy roots without a complete-generation
selection or staging tree retain their ordinary behavior when the new layout is
not explicitly selected. The legacy mutable watch/search exclusion issue is not
resolved by this opt-in route, nor is incremental artifact/embedding reuse.

There is no automatic garbage collection. Retained predecessors and failed
builds consume disk space. The store assumes cooperating writers and a trusted
directory tree; its inventory checks are not an anti-rollback authority or a
security boundary against hostile ancestor replacement.

## Focused validation

The command-level tests are in `runtime/complete_cli.rs`. Subprocess tests in
`tests/complete_generations_cli.rs` invoke the actual compiled executable and
bound each child lifetime. Refusal cases run without models. The positive
semantic subprocess case requires a verified Potion cache and must be invoked
explicitly; it covers the fast semantic tier, not quality-tier acceptance:

```sh
cargo test -p frankensearch-fsfs --no-default-features --lib complete_cli
cargo test -p frankensearch-fsfs --no-default-features --test complete_generations_cli
FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR=/verified/model-cache \
  cargo test -p frankensearch-fsfs --no-default-features --features semantic-support \
  --test complete_generations_cli complete_cli_binary_semantic_rebuild_search_and_stream \
  -- --ignored --exact
```

These tests were added without local Rust execution in the editing environment;
no compile, formatting, test, performance or full-release qualification is
asserted by this document. Warm-server tests replay genuine published bundle
selections at controlled output boundaries; they do not certify a real-model
cross-process watcher lifecycle.
