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

## 3) Searching from another terminal while `watch` runs

On Linux, a new index created in `.frankensearch` (or outside the source tree)
uses the **complete-generation store** by default (`FSFS-CURRENT` + immutable
`generations/<id>/` bundles). On a complete-generation store, `fsfs watch`
publishes atomic delta generations without holding a serving-index writer lock
between publications, so another terminal—and the background query daemon—can
search concurrently while `watch` is running:

```bash
fsfs search "ownership model in background workers" --limit 10
```

### Legacy index roots and non-Linux hosts (`bd-z2nfa`)

On Linux, running `fsfs index`, `fsfs watch`, or `fsfs index --watch` against a
completed legacy index root (`index_sentinel.json` with
`generation_complete = true`) automatically promotes it into a
complete-generation store while searches continue serving from the legacy
predecessor until `FSFS-CURRENT` publishes the first generation.

An index that remains on the **legacy mutable layout**—an incomplete/interrupted
legacy root, an index on macOS created without `FSFS_COMPLETE_GENERATIONS=1`, an
index built with `FSFS_COMPLETE_GENERATIONS=0`, or an index placed inside a
non-hidden directory of the source tree—still holds an exclusive vector writer
lock (`fsvi.map_lock`) for the lifetime of the watcher (`bd-z2nfa`).

To force-convert an incomplete legacy root or create a complete-generation store
on macOS, run one build or watch pass with `FSFS_COMPLETE_GENERATIONS=1`:

```bash
FSFS_COMPLETE_GENERATIONS=1 fsfs index .
```

When the legacy index has a compatible completion receipt, conversion reuses
unchanged embeddings and hard-links unchanged keyword segments in seconds. If
you intentionally keep a legacy root (`FSFS_COMPLETE_GENERATIONS=0`), press
`Ctrl+C` in the watch terminal and wait for it to exit before running
`fsfs search` from another process.

## 4) Monorepo hygiene tips

- Keep generated directories (`target`, build artifacts, vendored deps) out of your index scope.
- Prefer smaller, focused roots if one massive root is too noisy.
- Use JSON output for observability scripts:

```bash
fsfs status --format json | jq
```

## 5) Graceful shutdown

Use `Ctrl+C`; watch mode should stop without corrupting index files.
