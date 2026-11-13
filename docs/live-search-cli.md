# Live search from the fsfs command line

`fsfs live-search` subscribes one query to an explicitly selected existing
complete-generation store. It uses native read-only Quill readers and requires
no embedding models by default. It does not index, discover roots, migrate a
sealed index, check for updates, or switch a published generation. Only explicit
`--hybrid` loads configuration and already available models.

```sh
fsfs live-search --index-dir /path/to/complete-store --query 'connection pool'
```

Run the complete-generation publisher separately. The root is the directory
containing `FSFS-CURRENT`, not its selected `generations/g-...` child or the
child's `lexical` directory. A missing root is an error. An existing store with
no selection is polled until the first publication, unless `--once` is used.
A mutable legacy index is not a complete-generation store; this command does
not silently rebuild or migrate one. The initial empty selection can therefore
wait indefinitely by default; use `--timeout-ms` for automation.

```sh
# One current, verified snapshot, or a nonzero exit when no generation is selected.
fsfs live-search --index-dir /path/to/complete-store --query alpha --once

# Up to three generation frames; stop with an error if the overall budget expires.
fsfs live-search --index-dir /path/to/complete-store --query alpha \
  --limit 100 --max-updates 3 --timeout-ms 60000

# Coalesce frequent publications without postponing query refresh forever.
fsfs live-search --index-dir /path/to/complete-store --query alpha \
  --poll-ms 50 --debounce-ms 100 --max-wait-ms 1000 --min-score-delta 0.001

# Keep the live window scoped to Rust or Markdown files beneath src/.
fsfs live-search --index-dir /path/to/complete-store --query 'connection pool' \
  --filter 'path:src/ lang:rust ext:md' --limit 20
```

## Fixed query scopes

`--filter EXPR` uses the same parser as ordinary `fsfs search`. The expression is
UTF-8 and at most 64 KiB; unsupported keys or empty clause values fail argument
validation before configuration, models, source paths or stores are opened.
Blank expressions impose no additional scope.

| Clause | Meaning |
| --- | --- |
| `path:src/` or bare `src/` | Case-insensitive substring of the document path |
| `lang:rust` or `type:rust` | Ordinary fsfs language aliases, including `.rs` |
| `ext:rs` or `extension:rs` | Literal file extension, ignoring case and a leading dot |

Every path clause must match. Extension, language and type selections form one
union, so `path:src/ lang:rust ext:md` selects Rust **or** Markdown beneath
`src/`. Matching uses document identities, not a separate display path in stored
metadata. Multiple `--filter` flags are rejected; put clauses in one expression.

The scope stays fixed for the subscription. It applies after every publication,
to all hybrid retrieval phases, and after interactive `--tui` query edits. A file
that leaves the scope is removed from the visible window. A valid scope with no
matches delivers an empty snapshot or the corresponding removals.

Lexical retrieval widens its ranked candidate prefix until enough eligible hits
fill `--limit` or the admitted corpus is exhausted. This avoids losing matching
files behind an arbitrary oversampling cutoff, but a highly selective scope can
require searching the full corpus. Metadata is read only for the selected
winners from the same admitted immutable reader. Hybrid search reuses the
ordinary filtered retrieval pipeline and its producer and degradation policy.
With `--watch-source`, filtering restricts the query results; the publisher still
indexes the source under its discovery and privacy policies.

## Output contract

Stdout contains only newline-delimited `fsfs.stream.live_search.v1` records.
Each connection begins with a complete `snapshot` at sequence 1. Later `delta`
frames contain additions, removals, and changes to rank, metadata, or score.
Apply all changes in one frame before rendering. Ranks describe the entire
visible top-k window, not a set of independently insertable list operations.
Each delta names its `previous_generation`; generation identifiers are opaque,
and an explicitly restored predecessor is a legitimate new update.

There is no replay log or resume token. Reconnect with a new subscription and
replace the local baseline with its first snapshot. `--max-updates` counts
successfully delivered generation frames, including empty deltas; it does not
count polling iterations or individual documents. The score threshold compares
against the last published baseline, so gradual changes eventually cross it.

An absent selection is not a committed empty result set. Once a baseline has
been delivered, its removal does not generate fabricated document removals.
Corruption, unsupported lexical layouts, failed queries and unreadable
selections terminate the subscription with an error instead of serving a guessed
fallback. Every result is read from an admitted immutable generation.

Errors are structured JSON on **stderr**, never normal command envelopes on
stdout. Any output error terminates the stream without retry: a failed flush or
partial write may already have exposed bytes. A closed output pipe exits 0;
Ctrl-C/SIGTERM exits 130; argument errors exit 2. Other errors use fsfs's normal
error-to-exit-code mapping. Cancellation and timeouts are cooperative: they
cannot preempt a running synchronous query or a blocked filesystem/output
syscall. Cancellation is checked between polls and before output operations.

The default stream is **lexical-only**. Unsupported semantic, reranking and
output-format flags are rejected instead of ignored. Use `--hybrid` for the
configured progressive pipeline, `--tui` for the interactive terminal view, and
`--watch-source` to opt into publishing source changes. See
`fsfs live-search --help` for all options.

## Progressive hybrid subscriptions

```sh
fsfs live-search --index-dir /path/to/complete-store --query 'connection pool' \
  --hybrid --config /path/to/fsfs.toml
```

`--hybrid` connects the command to `RetainedLiveSearchSession`, using the existing
retained reader's lexical/vector fusion, Initial/Refined phase sink, producer
admission and configured degradation policy. Without `--config`, standard
project/user/environment configuration is used. Ranking, `fast_only` and rerank
policy come from that configuration; the command does not introduce a second
ranking implementation. `--config` without `--hybrid` is rejected. The ordinary
model-free lexical command still returns before loading any configuration.

Hybrid startup forces offline operation and disables indexing/watch mode. Models
must already be available. This is an opt-in to the hybrid pipeline, not a claim
that every response is semantic: the pipeline's actual model admission, fallback,
skip, failure and rerank annotations are retained in each output frame.

Hybrid NDJSON has top-level schema `fsfs.stream.live_search.hybrid.v1`. Each frame
contains `generation_id`, `manifest_sha256`, `phase`, `annotations` (the ordinary
search payload without hits), and `update` (the result snapshot/delta). The
physical generation receipt stays constant through Initial and Refined;
`update.generation` additionally includes the phase as a **result revision**.
Refinement does not pretend to publish a different on-disk index. Follow the
nested update sequence and previous revision when applying deltas.

Initial is written and flushed before refinement is awaited. A
`RefinementFailed` frame has `update: null`: retain the delivered Initial window
and display its failure annotations. It is not an empty hit set, and it must not
clear the subscriber's results. A successful refined phase updates the same
window. A new publication observed during a query cannot mix its artifacts into
that query's phases; the next completed-generation poll admits the successor.

In hybrid mode `--once` completes one generation's entire phase sequence, and
`--max-updates N` completes N generations, not N phase frames. These limits do not
discard refinement after delivering Initial. A timeout, cancellation or transport
failure can still interrupt the sequence. Each record is fully serialized into a
16 MiB bounded buffer before any of its bytes are exposed. A failed write/flush
terminates the connection without replay; reconnect for a new full snapshot.

Unlike the default lexical session's absent-selection wait, loss of an established
hybrid selection is a typed error. Neither mode fabricates removals or silently
chooses a fallback generation. Hybrid admission errors preserve the old retained
reader internally, but the command stops instead of continuing with stale data.
