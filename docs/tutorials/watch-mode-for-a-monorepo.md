# Setting up watch mode for a monorepo

For large repos, run a broad initial index once, then keep it fresh with `watch`.

## 1) Initial index pass

```bash
cd /path/to/monorepo
fsfs index .
```

## 2) Start watch mode

```bash
fsfs watch .
```

`watch` listens for file changes and incrementally updates index state.

## 3) Keep search in a second terminal

```bash
fsfs search "ownership model in background workers" --limit 10
```

Run this repeatedly while editing files; the freshest results should surface without a full reindex.

## 4) Monorepo hygiene tips

- Keep generated directories (`target`, build artifacts, vendored deps) out of your index scope.
- Prefer smaller, focused roots if one massive root is too noisy.
- Use JSON output for observability scripts:

```bash
fsfs status --format json | jq
```

## 5) Graceful shutdown

Use `Ctrl+C`; watch mode should stop without corrupting index files.
