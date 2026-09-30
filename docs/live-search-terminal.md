# Terminal subscriptions

`fsfs live-search --tui` renders the existing subscription in a native Unix
alternate-screen terminal instead of writing NDJSON to stdout:

```sh
fsfs live-search --index-dir /path/to/store --query 'connection pool' --tui
fsfs live-search --index-dir /path/to/store --query 'connection pool' \
  --hybrid --config /path/to/fsfs.toml --tui
```

The ordinary lexical mode still loads no models or configuration. `--hybrid`
uses the same retained Initial/Refined pipeline as machine output. The view
shows committed results, rank, score, path/line and the selected result's stored
snippet when its payload includes one. It never rereads mutable source text to
fill a missing preview. A failed refinement displays its reason and keeps the
valid Initial window.

Up/Down or `j`/`k` select a result, Page Up/Down move ten rows, and Home/End jump
to the first/last result. Selection follows document identity when a new
publication or refined phase changes ranks. Removing the selected document
chooses a remaining neighboring rank. `q`/Escape stop the subscription; Ctrl-C
interrupts it. Resizing redraws the same window without requerying.

The query is fixed for the subscription. This is not the deluxe `fsfs tui`
query editor and it does not replace that screen. The native terminal backend
restores raw mode, cursor visibility and the alternate screen on exit. Both
stdin and stdout must be terminals; redirected `--tui` is refused before
configuration, model admission or source indexing. `--format` and `--tui`
cannot be combined. Omit `--tui` for the unchanged NDJSON protocol.

`--once`, `--max-updates` and `--timeout-ms` retain their existing stopping
semantics: reaching one exits and restores the terminal, not an indefinitely
held final screen. For an ongoing read-only view, omit those stopping limits.
`--watch-source` can be combined with `--tui`; its explicit `--hybrid` and
finite update-limit requirements remain unchanged, as do the source/store
separation, offline model, publication and retention policies.

The renderer consumes the same complete records as a machine client. It checks
query, schema, sequence, previous revision, physical generation, phase ordering,
identities, ranks and counts before installing an update. A delta is applied
atomically, never one row at a time. Failed decoding or presentation stops the
command without acknowledging that update. Both each record and the retained
result window have 16 MiB bounds. Frame-local grapheme pools prevent memory
growth from old rendered text. Display-only control/bidi characters are removed;
wire payloads and search scores are unchanged.

Input and the owned search/watch future share the existing request task. No
input thread, detached indexer or second runtime is started. Input checks occur
between asynchronous polls; cancellation cannot preempt a synchronous filesystem,
model, query or terminal-output syscall already running.
