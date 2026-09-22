# Watch while complete-generation searches remain available

This route is for the Unix complete-generation store. It does not change the
legacy mutable watcher or migrate an existing live index in place. Use a fresh
store outside the source tree; its parent must already exist. Configure verified
semantic models and a generation-local catalog as in
[complete-generation rebuilding](complete-generation-rebuilds.md).

```toml
[storage]
db_path = "{index_dir}/catalog.sqlite"
```

Start the first build and keep watching:

```sh
FSFS_COMPLETE_GENERATIONS=1 fsfs watch /work/source \
  --index-dir /work/search-store --config /work/fsfs.toml --format jsonl
```

`index /work/source --watch` uses the same complete-store route. After the first
initialization, the store is recognized without the environment variable. Watch
output accepts `table`, `jsonl`, or `toon`; unframed JSON/CSV output is refused
before creating the store. Each successful publication emits the existing index
receipt fields with command `watch`, including generation ID and durable status.

In another process, use the store rather than an individual generation directory:

```sh
fsfs search "connection pooling" --index-dir /work/search-store \
  --config /work/fsfs.toml --no-daemon --format json

fsfs serve --index-dir /work/search-store \
  --config /work/fsfs.toml --format jsonl
```

The watch registration stays alive across builds, but it owns no serving-index
writer. Each build acquires the complete-store publication lease and writes to
a fresh candidate. Already admitted readers retain their bundle; the existing
live server admits a successor between requests and invalidates its RAM query
cache. A failed initial build does not create usable search results. A failed
successor does not remove the already selected bundle.

## Changes and publication

Notifications coalesce for 500 milliseconds, capped at 5 seconds under a burst.
Registration happens before the initial scan. A consumed change window is taken
before rebuilding, never cleared after publication, so an event received during
the build remains pending. At most 128 bounded path hints are retained. A native
rescan indication, missing path, or hint overflow forces reconciliation instead
of silently losing change evidence. Ordinary access events do not trigger builds.

Source observation reuses the public discovery and mount policy used by ordinary
indexing. It collects device/inode, byte length, nanosecond modification time and
change time during traversal, not in a second stat pass over an older listing.
Cancellation is checked before and after each lazy directory entry. Opened
directory identities are checked again before accepting the observation, and the
original source-root descriptor stays alive for the whole watch session. An
incomplete observation is rejected wholesale rather than applied as deletions.

In-scope file and observed-directory hints still force a rebuild when all
metadata values happen to match. A renamed directory is checked against the
previous observation as well as the current one. Ambiguous parent-traversal
hints force reconciliation. Excluded-file-only hints need not re-embed an unchanged corpus. A periodic
30-second membership/metadata reconciliation recovers observable changes when a
notification is lost; this is not a 30-second end-to-end freshness guarantee.

The candidate runs the normal indexing and full-search admission path. After its
inventory and files are sealed, a read-only precommit hook rechecks source
membership/metadata and pending in-scope changes. Only then does the existing
publication code repeat cancellation, lease-fence, and predecessor checks and
rename the selection pointer. Engine admission is not replaced by this hook.

An unreadable subtree, replaced source root, or failed native backend is an
error, not evidence that documents should be deleted. A raced observation or
build is retried after another quiet window. Three consecutive raced attempts
pause candidate allocation, while keeping the watch registration and previous
selected generation alive. Quiet-source probes run five seconds apart without
creating generation directories. Two matching observations without relevant
hints queue a forced catch-up build, even if no new notification arrives and the
source metadata equals the old baseline. A hint received during a probe remains
pending and prevents that probe from counting as quiet. Cancellation, failed
native backends and ordinary admission failures still propagate; they are not
converted into successful recovery or deletions.

The watcher binds its source baseline to the generation it published. It checks
that selection on idle polls, after quiet-source probes and at precommit. Another
publisher selecting a generation, or removal of the selection pointer, stops the
session with an error rather than retargeting its baseline or overwriting the
other publication. This check uses the bounded selection descriptor; it does not
rehash every generation file on each idle poll.

Once a pointer rename happens, output failure or cancellation cannot roll it
back. If directory synchronization fails after the rename, the existing
visible-but-durability-uncertain error is preserved, and no false durable receipt
is emitted. The native watcher is owned by the watch future and released when it
returns or is dropped; no separate indexing task is left running.

## Boundaries

The retained builder can reuse checkpoint-proven embeddings in independently
copied artifacts from the same process session and resolved configuration.
Uncovered checkpoint tails and changed or incompatible inputs are recomputed by
the ordinary indexer. This is not cross-restart caching or delta-only traversal;
copying artifacts, full discovery and validation still cost work. Pausing hot
sources bounds repeated candidate creation until quiet observations agree, not
total lifetime disk use or event-to-visible latency. Retained and abandoned
generations consume disk space; automatic reclamation is not implemented.

The observation is not an atomic filesystem snapshot. There remains an ordinary
cooperative-writer race between the final source check and pointer rename; later
notifications/reconciliation drive subsequent publication. No claim is made to
detect a lost event for an edit that preserves every observed metadata field.
This is not hostile-directory, anti-rollback, or privileged-adversary protection.

Complete-store readers support buffered CLI forwarding and progressive socket
requests as described in [complete-generation serving](complete-generation-rebuilds.md).
Legacy watcher migration, cross-restart reuse, and TUI migration remain
unfinished. Ctrl-C cancellation
is wired through the existing shutdown coordinator; no hard real-time latency
is claimed for synchronous native operations or indexing work.

## Validation

```sh
cargo test -p frankensearch-fsfs --no-default-features --lib complete_watch
cargo test -p frankensearch-fsfs --no-default-features --test complete_generation_watch_cli
FSFS_COMPLETE_GENERATION_TEST_MODEL_DIR=/verified/model-cache \
  cargo test -p frankensearch-fsfs --no-default-features --features semantic-support \
  --test complete_generation_watch_cli complete_watch_binary_keeps_search_available \
  -- --ignored --exact
```

The ignored subprocess test requires real verified Potion model files. It keeps
the production watcher alive across independent search processes while checking
add/rename/delete visibility and predecessor retention. Its final child kill is
cleanup, not evidence of graceful signal handling. Library tests exercise the
owned lifetime, publication guard, missed-notification recovery, source identity,
mid-listing cancellation and partial-observation refusal. Recovery tests cover
paused candidate counts, catch-up without another event, old-reader retention,
hints/backend failures during probes, cancellation and outside publications at
the idle, settling and precommit boundaries. These are not a real-model
performance qualification or proof of bounded freshness under continuous churn.

These changes were prepared without Rust compiler, Cargo, rustfmt or RCH access.
None of these Rust tests were executed in the editing environment; compilation,
formatting, Clippy, model-quality, performance, and release acceptance remain
unverified.
