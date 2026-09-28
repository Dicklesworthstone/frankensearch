# Index, watch, and stream one query

The explicitly mutating form of `fsfs live-search` runs the existing
complete-generation watcher and the progressive query subscriber in one process:

```sh
fsfs live-search --watch-source /path/to/source \
  --index-dir /path/to/separate-store --query 'connection pool' \
  --hybrid --config /path/to/fsfs.toml --max-updates 20
```

Source and store trees must not overlap. A new store can be bootstrapped when its
parent exists. An existing store is handled by the same complete-generation
writer admission and source checks as `fsfs watch`. Discovery, owner/privacy,
pressure, producer identity, and checkpoint-proven reuse policies remain in the
ordinary indexer. Models must already be installed; this command stays offline.
It never edits source files, rewrites sealed predecessors, or deletes artifacts.
`--watch-source` requires `--hybrid` and is currently Unix-only. Without that flag,
`live-search` remains a read-only subscriber.

The command registers native notifications before its initial build. After each
durable publication it queries through the existing retained hybrid pipeline,
flushes Initial, and then delivers the refinement outcome when configured. The
output remains `fsfs.stream.live_search.hybrid.v1`; the receipt in every phase is
checked against the watcher's just-published generation. A competing publisher
cannot silently substitute its query results. A failed refinement preserves the
Initial window, with the ordinary failure annotations and no removal delta.

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

This adds the combined workflow through `live-search --watch-source`. It does not
add a `watch --stream-search` spelling or a TUI subscription.

## Regression commands

```sh
cargo test -p frankensearch-fsfs --no-default-features --bin fsfs live_search_command
cargo test -p frankensearch-fsfs --no-default-features --test live_search_watch
FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR=/verified/model-cache \
  cargo test -p frankensearch-fsfs --test live_search_watch -- --ignored
```

The ignored process test uses the production binary and a real installed Potion
model. It bootstraps a store, waits for its initial streamed snapshot, renames a
source file, verifies the successor delta, and checks that the publisher stops
at the requested cycle limit. Ordinary tests require no model download.
