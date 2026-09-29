# Interactive live-query terminal

`fsfs-live` is the native terminal frontend for the complete-generation live
search sessions. It is a separate Cargo binary in `frankensearch-fsfs`, not a
replacement for the existing `fsfs tui` onboarding and operations screens.

```sh
cargo build -p frankensearch-fsfs --bin fsfs-live
./target/debug/fsfs-live --index-dir /path/to/complete-store --query 'connection pool'
```

For progressive hybrid retrieval with already installed models:

```sh
./target/debug/fsfs-live --index-dir /path/to/complete-store --query 'connection pool' \
  --hybrid --config /path/to/fsfs.toml
```

A build with `--no-default-features` supports the model-free Quill frontend.
The command requires terminal stdin and stdout and the native Unix backend.
It never starts a publisher. Run indexing/watching separately; only complete,
admitted generations drive updates. No source edits, migration, persistent
query-cache writes, or model downloads are requested. The standard release
installer's existing binary selection is not changed by this addition.

## Interaction

Use Up/Down or j/k to select, PageUp/PageDown to move ten results, and Home/End
to jump. `/` edits the query, Ctrl-U clears it, Enter submits, and Escape
abandons the draft. `r` explicitly retries or refreshes the current query;
q/Escape exits when not editing. Ctrl-C returns cancellation. Bracketed paste
is text, not a sequence of navigation or quit commands. Query input and the
visible window are bounded (`--limit 1..=1000`, default 20).

Selection follows the document's stable identity when its rank changes. If it
leaves the top-k window, selection moves to the nearest remaining rank.
Metadata/snippet updates replace the visible payload. Initial is rendered from
the retained reader's phase sink before refinement is awaited. A failed
refinement preserves the Initial window and displays its actual skip/failure
annotations, rather than presenting an empty successful result.

While a query yields, input is checked on a 25 ms timer. A new query or quit
drops the owned old query future before replacing the session; its late phases
cannot overwrite the new query. This is cooperative, not preemption of a
synchronous model call, filesystem operation, or terminal write.

Admission/retrieval failure stops automatic refresh and labels the retained
window **STALE**. `r` or a new query starts a fresh subscription. There is no
implicit stale-result fallback or retry against a different generation.
Terminal input/output errors exit rather than entering a retry loop. Native
backend ownership restores the terminal on ordinary return/error/cancellation;
this does not promise recovery from SIGKILL or machine failure.

## Validation

```sh
cargo test -p frankensearch-fsfs --no-default-features --bin fsfs-live
scripts/quality-gate.sh
```

The binary's inline tests include actual Quill generation publication and query
switching through the same production driver with a headless screen, as well as
revision-chain refusal, selection stability, partial-delta atomicity, Unicode
and control-text handling, pending-future disposal, and terminal-error handling.
They do not certify a real terminal or real-model interactive session.
