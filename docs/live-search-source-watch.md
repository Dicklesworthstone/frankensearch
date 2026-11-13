# Index, watch, and stream one query

The explicitly mutating form of `fsfs live-search` runs the existing
complete-generation watcher and query subscriber in one process. Default lexical
watching works without installed models or configuration:

```sh
fsfs live-search --watch-source /path/to/source \
  --index-dir /path/to/separate-store --query 'connection pool' --max-updates 20
```

Add `--hybrid` to use the progressive pipeline and already installed models:

```sh
fsfs live-search --watch-source /path/to/source \
  --index-dir /path/to/separate-store --query 'connection pool' \
  --hybrid --config /path/to/fsfs.toml --max-updates 20
```

The store may not contain the source tree, and may sit inside it only beneath a
hidden directory, whose writes the watcher ignores. A new store can be
bootstrapped when its parent exists. An existing store is handled by the same complete-generation
writer admission and source checks as `fsfs watch`. Discovery, owner/privacy,
pressure, producer identity, and checkpoint-proven reuse policies remain in the
ordinary indexer. Lexical watching uses the built-in discovery/privacy policies
with lexical-only indexing forced; it does not load configuration or resolve
either embedding model. Hybrid watching loads normal fsfs configuration (or
`--config`) and requires already installed models. Both modes stay offline.
It never edits source files, rewrites sealed predecessors, or deletes artifacts.
`--watch-source` is currently Unix-only. Without this explicit flag, `live-search`
remains a read-only subscriber; `--config` still requires `--hybrid`.

The command registers native notifications before its initial build. After each
durable publication, lexical mode admits the immutable Quill reader and emits
the usual `fsfs.stream.live_search.v1` snapshot or delta. Created files, changed
matches, and removed source files become visible after publication. Hybrid mode
uses the retained pipeline, flushing Initial before its configured refinement
outcome in `fsfs.stream.live_search.hybrid.v1` records. Both modes check the
admitted generation ID and manifest digest against the watcher's receipt before
output. A competing publisher cannot silently substitute its results. Failed
hybrid refinement preserves the Initial window with its failure annotations.

Query and output backpressure suspend the owned watcher future, rather than queue
up completed generations or start another indexer. The native registration stays
alive and keeps the watcher's bounded change hints. After the complete phase
sequence, the same watcher resumes its existing debounce, reconciliation and
source-stability checks. There is at most one undelivered publication receipt.
A sink failure stops the entire command; there is no replay on a torn connection
and no independently running publisher after the command exits.

One of these limits is required: use `--once` to **build and query** one generation,
or `--max-updates N` to stop
after N complete publication/query cycles. Neither cuts off refinement just
because Initial was delivered. `--limit` and `--min-score-delta` control the
subscriber window as usual. Source timing belongs to the native watcher, so
`--poll-ms`, `--debounce-ms` and `--max-wait-ms` are rejected in this mode rather
than accepted without effect. Normal read-only subscriptions retain those flags.

Every publication retains a complete bundle. The existing unbounded-retention
issue (`bd-2op1d`) is not solved here: a finite successful-update limit is not a
byte quota, does not count abandoned candidates, and does not prune old data.
Budget disk space for full bundles and any failed builds. Indefinite unattended
source watching is deliberately not enabled by this command.

`--timeout-ms` and signals are cooperative, including while waiting for the next
publication. They cannot preempt a synchronous syscall, query or output write.
A timeout, cancellation, failed admission, or failed output may occur **after a
successful publication**. It does not undo that publication. Runtime errors on
stderr use the `watch_subscription` phase; stdout contains query frames only.
Inspect the selected store after an error rather than assuming the build aborted.

Add `--tui` for the existing terminal view and query editor in either mode. Query
edits reuse the same watcher and reader and do not consume the publication limit.

## Regression commands

```sh
cargo test -p frankensearch-fsfs --no-default-features --bin fsfs live_search_command
cargo test -p frankensearch-fsfs --no-default-features --test live_search_watch
FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR=/verified/model-cache \
  cargo test -p frankensearch-fsfs --test live_search_watch -- --ignored
```

Ordinary process tests bootstrap and watch a lexical store without models,
verify create/update/removal deltas, and check the publication limit. The ignored
process test uses the production binary and a real installed Potion
model. It bootstraps a store, waits for its initial streamed snapshot, renames a
source file, verifies the successor delta, and checks that the publisher stops
at the requested cycle limit. Ordinary tests require no model download.
