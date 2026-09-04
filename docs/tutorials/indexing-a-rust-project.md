# Indexing a Rust project

This walkthrough indexes a Rust workspace with `fsfs`, then checks retrieval on a few queries.

## 1) Install `fsfs`

```bash
curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/frankensearch/main/install.sh | bash -s -- --easy-mode
fsfs version
```

The ordinary installer selects a full semantic artifact when available, otherwise
a loader-capable source build on supported platforms. A loader-capable binary
needs model files before indexing. Acquire and explicitly verify both tiers:

```bash
fsfs download-models potion-multilingual-128m
fsfs download-models all-minilm-l6-v2
fsfs download-models potion-multilingual-128m --verify
fsfs download-models all-minilm-l6-v2 --verify
fsfs doctor
```

The first download is roughly 621 MB; a verified cache can be reused. Embedded
builds supply those same model bytes. The explicit `--lite` profile omits the
semantic loaders, so downloading models cannot turn it into a semantic binary.
Ordinary Intel macOS installation currently returns `unsupported_platform`;
`--lite` is an explicit model-free choice, not completion of this semantic
walkthrough. See the [installation profiles](../../README.md#installation).

## 2) Index your repository

```bash
cd /path/to/your/rust-repo
fsfs index .
```

This plain command performs one complete pass, seals the index, and exits. It
does not silently enter watch mode.

Use JSON when you want to capture machine-readable stats:

```bash
fsfs index . --format json | jq
```

## 3) Run a few targeted searches

```bash
fsfs search "structured concurrency context propagation" --limit 5
fsfs search "Cargo feature flags and default features" --limit 5
fsfs search "how retries and backoff are implemented" --limit 5
```

## 4) Ask for an explanation when ranking surprises you

```bash
fsfs explain 1            # the rank printed by the last search
fsfs explain src/lib.rs   # or a path from that search (a unique file-name suffix works too)
```

The target is resolved against the last `fsfs search` in this index directory: the 1-based rank
from the table or JSON `rank` field, a path, or the session id (`R0` is rank 1). BM25 term
statistics are not exported by the lexical engine, so the explanation flags its `tf`/`idf`
placeholders with the `bm25_stats_unavailable` warning; the lexical raw score and RRF
contribution are real.

## 5) Recommended next step

If the repository changes constantly, switch to watch mode:

```bash
fsfs watch .
```

The current watcher prevents searches from other processes while it holds the
vector writer lock. Follow the [watch-mode limitation and shutdown steps](watch-mode-for-a-monorepo.md)
before querying from a separate terminal.
