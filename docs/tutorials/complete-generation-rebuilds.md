# Atomic complete-generation rebuilds

The `fsfs` executable can opt an index root into complete-generation publication.
A rebuild writes its lexical, vector, catalog and content artifacts into a fresh
bundle and selects that bundle only after the existing indexing and search
admission checks succeed. Old bundles and abandoned builds are retained.

This route is currently Unix-only. Use a store outside the source tree, with an
existing parent directory. Configure a generation-local catalog:

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

## Current boundaries

Complete-store CLI routing currently supports one-shot `index`, direct `search`
(including JSONL/TOON streaming), `status`, and `doctor`. It refuses watch mode,
TUI, daemon/socket serving, query expansion and in-place mutators rather than
pretending that those paths have been migrated. Existing legacy roots without
a complete-generation selection or staging tree retain their ordinary behavior.
The legacy watch/search exclusion issue is not resolved by this opt-in route.

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
asserted by this document.
