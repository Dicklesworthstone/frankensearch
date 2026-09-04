# Setting up watch mode for a monorepo

For large repos, run a broad initial index once, then keep it fresh with `watch`.

## 1) Initial index pass

First complete the semantic binary and model setup in
[Indexing a Rust project](indexing-a-rust-project.md#1-install-fsfs). Loader-capable
builds need verified Potion and MiniLM installations; downloading those models
does not enable semantic execution in an explicit `--lite` binary.

```bash
cd /path/to/monorepo
fsfs index .
```

The initial command is intentionally one-shot and exits after sealing the
generation. Watch mode starts only in the explicit next step.

## 2) Start watch mode

```bash
fsfs watch .
```

`watch` listens for file changes and incrementally updates index state.

## 3) Stop the watcher before searching from another process

The current watcher holds an exclusive vector writer lock for its lifetime.
An independent `fsfs search` or query daemon can fail with `fsvi.map_lock` while
it runs. This was reproduced with the full Linux v1.8.0 binary: the watcher
remained alive and a separate `search --no-daemon` exited 2.

Press `Ctrl+C` in the watch terminal and wait for it to exit, then query:

```bash
fsfs search "ownership model in background workers" --no-daemon --limit 10
```

Concurrent searches that see durably published changes without stopping watch
remain required functionality under `bd-z2nfa`. Incremental ingestion alone does
not prove cross-process visibility. Until that work passes its concurrent-reader
probe, stop and restart the watcher when using this separate-terminal workflow.

## 4) Monorepo hygiene tips

- Keep generated directories (`target`, build artifacts, vendored deps) out of your index scope.
- Prefer smaller, focused roots if one massive root is too noisy.
- Use JSON output for observability scripts:

```bash
fsfs status --format json | jq
```

## 5) Graceful shutdown

Use `Ctrl+C`; watch mode should stop without corrupting index files.
