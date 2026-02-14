# frankensearch

[![CI](https://github.com/Dicklesworthstone/frankensearch/actions/workflows/ci.yml/badge.svg)](https://github.com/Dicklesworthstone/frankensearch/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/frankensearch.svg)](https://crates.io/crates/frankensearch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Two-tier hybrid local search for Rust and the `fsfs` standalone CLI: fast first-pass results, then quality refinement.

## Install In One Line

```bash
curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/frankensearch/main/install.sh | bash -s -- --easy-mode
```

Installer goals:
- zero-friction first run
- auto-configured model cache path
- sane defaults for interactive usage

## Cargo Install (Developer Path)

`fsfs` currently builds from this workspace and uses the pinned nightly toolchain (`rust-toolchain.toml`):

```bash
cargo +nightly install --path crates/frankensearch-fsfs
fsfs version
fsfs status --no-watch-mode --format json
```

## Quick Start (60 Seconds)

```bash
# 1) Install
curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/frankensearch/main/install.sh | bash -s -- --easy-mode

# 2) Index a directory
fsfs index ./my-project

# 3) Search
fsfs search "how does retry backoff work" --limit 5
```

Example output:

```text
PHASE 0 (fast): 5 hits in 12ms
  1. src/retry.rs      score=0.812
  2. docs/failures.md  score=0.774

PHASE 1 (refined): 5 hits in 151ms
  1. src/retry.rs      score=0.923
  2. src/http/client.rs score=0.901
```

## What It Does

`frankensearch` combines lexical and semantic retrieval with progressive delivery:
- lexical BM25 via Tantivy for exact keyword precision
- fast semantic tier for immediate relevant hits
- quality semantic tier for reranked refinement
- reciprocal rank fusion (RRF) to combine sources robustly

Result: responsive first answers plus better final ranking without blocking the UI.

## Core Features

- Auto-download and model fallback chain (`fastembed`/`model2vec`/`hash`)
- Progressive search phases (`Initial`, `Refined`, `RefinementFailed`)
- Agent-friendly streaming (`--stream`) with machine-readable output
- Result explanation surfaces (`fsfs explain <result-id>`)
- Multiple output formats: `table`, `json`, `jsonl`, `toon`, `csv`
- Watch/incremental indexing mode for local corpus updates
- Portable SIMD vector search + quantized FSVI storage
- Optional reranking and ANN paths via feature flags

## CLI At A Glance

```bash
# Basic search
fsfs search "structured concurrency" --limit 10

# Stream for agents/pipelines
fsfs search "query" --stream --format jsonl

# TOON mode
fsfs search "query" --stream --format toon

# Explain one result
fsfs explain result-123

# Keep index fresh
fsfs index ~/projects --watch

# Health checks
fsfs doctor
```

## Configuration

Configuration precedence:
1. CLI flags
2. project config file
3. user config file
4. environment variables
5. built-in defaults

Common environment variables:

| Variable | Purpose | Example |
|---|---|---|
| `FRANKENSEARCH_MODEL_DIR` | Override model location | `~/.cache/frankensearch/models` |
| `FRANKENSEARCH_FAST_ONLY` | Skip quality refinement | `true` |
| `FRANKENSEARCH_QUALITY_WEIGHT` | Blend quality vs fast tier | `0.7` |
| `FRANKENSEARCH_RRF_K` | RRF constant | `60` |
| `FRANKENSEARCH_LOG` | Tracing filter | `info` |

For full contracts and knobs:
- `docs/fsfs-config-contract.md`
- `docs/fsfs-dual-mode-contract.md`
- `docs/architecture/`

## How It Works

Pipeline summary:

```text
Query
  -> canonicalize
  -> classify
  -> fast embed + lexical BM25
  -> RRF fusion (initial)
  -> quality embed (top candidates)
  -> blend (and optional rerank)
  -> refined results
```

Model path used in the default quality lane:
- fast tier: potion-128M (or fallback)
- fusion: RRF over lexical + semantic ranks
- quality tier: MiniLM
- optional final rerank: FlashRank cross-encoder

## Why Not Just grep/ripgrep/ctags?

`grep`/`ripgrep`/`ctags` are excellent for exact text and symbol lookup. `frankensearch` solves a different problem: semantic intent search over mixed corpora.

| Tool | Strong At | Limitation vs frankensearch |
|---|---|---|
| `grep` | exact substrings | no semantic similarity |
| `ripgrep` | very fast regex search | no embedding-based recall |
| `ctags` | symbol navigation | not document-level semantic ranking |
| `frankensearch/fsfs` | hybrid semantic + lexical, progressive refinement | higher complexity/runtime footprint |

Use both: keep `rg` for exact matches and use `fsfs` for intent-level retrieval.

## FAQ

### Does it run fully local?
Yes. Search/indexing runs on your machine. Network access is only needed when downloading models.

### Can I use only the library and skip `fsfs`?
Yes. Add `frankensearch` as a dependency and wire your own app/runtime.

### What if the quality model is unavailable?
Search still works using fast-tier and lexical paths; you get `RefinementFailed` or fast-only behavior.

### Which output format should agents use?
Use `jsonl` for streaming automation and `toon` if your downstream stack expects TOON semantics.

### Is this tied to Tokio?
No. Async/concurrency is built around `asupersync` and `Cx`.

## Contributing

Project policy is no direct external merges, but issues and PRs are still useful for bug reports and proposal clarity.

If you are working inside this repository as an internal/automation agent:

```bash
cargo fmt --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Useful docs:
- `AGENTS.md`
- `docs/e2e-artifact-contract.md`
- `docs/dependency-semantics-policy.md`

## License

MIT
