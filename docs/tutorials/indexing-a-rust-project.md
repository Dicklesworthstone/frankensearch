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
walkthrough. See the [installation profiles](../../README.md#cargo-install-developer-path).

## 2) Index your repository

```bash
cd /path/to/your/rust-repo
fsfs index .
```

This plain command performs one complete pass, seals the index, and exits. It
does not silently enter watch mode.

After indexing, inspect machine-readable index and model status:

```bash
fsfs status --format json | jq
```

In v1.8.0 the one-shot index command still prints progress text even with
`--format json`; do not pipe that output into a JSON parser. The status command
provides a JSON envelope for automation.

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
from the table or JSON `rank` field, a path, or the session id (`R0` is rank 1). The BM25 score
is split by query word and field, and the parts add up to it. For the third hit of
`fsfs search "quantize vectors to f16"` on this repository's crates:

```text
Lexical (BM25): 24.158022
  content:quantize = 8.104690 (doc_freq 11, idf 3.8153)
  title:quantize = 10.972879 (doc_freq 2, idf 5.3414)
  content:to = 0.166461 (doc_freq 483, idf 0.0766)
  content:f16 = 4.913994 (doc_freq 54, idf 2.2595)
  unmatched: vectors
```

`title` is the file name, which counts double. `unmatched` lists query words the file does not
contain. If the index changed after the search, the split is left out with a
`bm25_stats_unavailable` warning (run the search again); the lexical score and its RRF
contribution are still shown.

## 5) Recommended next step

If the repository changes constantly, switch to watch mode:

```bash
fsfs watch .
```

The current watcher prevents searches from other processes while it holds the
vector writer lock. Follow the [watch-mode limitation and shutdown steps](watch-mode-for-a-monorepo.md)
before querying from a separate terminal.
