# frankensearch-fsfs

Standalone `fsfs` CLI and runtime for two-tier hybrid local search.

## Overview

`frankensearch-fsfs` is the user-facing CLI application that provides a full-featured local search experience. It combines semantic vector search, BM25 lexical search, and two-tier progressive ranking into a terminal application with both a rich TUI mode and structured output formats (JSON, TOON). It includes filesystem watching for live index updates, configurable pressure sensing, query planning, and an explainability screen.

The crate is split into a library (`src/lib.rs`) for reusable runtime, configuration, and adapter logic, and a binary (`src/main.rs`) for the `fsfs` CLI entrypoint.

## Build Profiles

The default feature set is loader-capable: ordinary builds compile Model2Vec
and FastEmbed, while crates.io packaging remains independent of the large
Potion and MiniLM artifact bytes. Provision the registered models after a
plain build—no feature flags are needed:

```bash
cargo build -p frankensearch-fsfs
fsfs download-models potion-multilingual-128m
fsfs download-models all-minilm-l6-v2
fsfs download-models potion-multilingual-128m --verify
fsfs download-models all-minilm-l6-v2 --verify
fsfs status --format json
fsfs doctor --format json
```

Downloads are revision-pinned, SHA-verified, and atomically promoted. Explicit
verification refreshes the durable receipt and returns a typed nonzero error for
a missing, incomplete, or corrupt registered cache. Status is observational: it
may full-hash an uncached installation for that invocation but never mints a
receipt. Doctor additionally
opens verified caches through the compiled loaders, so a directory that merely
exists is never reported as semantic-ready. A hard doctor verdict emits one
`subsystem_error` report with the failing checks in its context and exits nonzero, while warning-only diagnostics remain
successful. A missing or offline cache is a typed, actionable failure; the
production indexing path never admits the hash control embedder. The explicit
`--no-default-features` profile is the model-free lite binary, and downloaded
files cannot add loaders to that deliberately stripped build.

`fsfs index <path>` performs one indexing pass and exits. Continuous indexing is
opt-in through `fsfs watch <path>` or `fsfs index <path> --watch`; a performance
profile may permit background indexing, but it never requests a watcher on the
user's behalf. Full search requires the sealed vector generation and matching
fast semantic loader. Only the explicit lexical-only mode bypasses that
readiness check, while quality-only failures preserve Initial results and emit
an actionable `RefinementFailed` phase.

The installer verifies every downloaded release archive and fails closed if a
checksum, checksum tool, or matching manifest entry is unavailable. When a full
artifact is unavailable, ordinary installation builds the loader-capable default
from source; it never silently substitutes lite. Only `install.sh --lite`
selects the model-free source profile. Intel macOS currently has no supported
FastEmbed/ONNX Runtime distribution: ordinary semantic installation returns the
typed `unsupported_platform` outcome, while an explicit `--lite` installation
remains available. The installer never pretends the unbuildable default source
route succeeded on that target.

Full release binaries explicitly select `embedded-models`. From the workspace
root, provision and verify the exact, pinned source files before building that
profile:

```bash
scripts/rch-ensure-deps.sh --models-only
cargo build -p frankensearch-fsfs --no-default-features --features embedded-models
```

Use `scripts/rch-ensure-deps.sh --all-workers --models-only` to prepare every
RCH worker without coupling model admission to sibling-dependency maintenance.
Both provisioning modes enforce the manifest byte lengths and SHA-256 digests;
Cargo's build script performs no network access.

## Key Types

### CLI and Configuration

- `CliCommand` / `CliInput` - parsed CLI commands and input
- `FsfsConfig` - layered configuration (CLI flags, project file, user file, defaults)
- `OutputFormat` - output format selection (JSON, TOON, TUI, plain)
- `OutputEnvelope` - structured output envelope for machine-readable output
- `Verbosity` - logging verbosity levels

### Runtime

- `FsfsRuntime` - main runtime coordinating search, indexing, and UI
- `ShutdownCoordinator` / `ShutdownReason` - graceful shutdown management
- `FsWatcher` / `WatcherStats` - filesystem watcher for live index updates

### Search Pipeline

- `QueryExecutionOrchestrator` - orchestrates query execution across retrieval stages
- `QueryPlanner` / `QueryPlannerConfig` - adaptive query planning with intent classification
- `LexicalPipeline` - manages the lexical indexing pipeline
- `FusionPolicy` / `FusedCandidate` - fusion strategy and merged candidates

### Catalog and Ingestion

- `bootstrap_catalog_schema` - catalog schema creation for document tracking
- `CatalogChangeKind` / `CatalogIngestionClass` - change and ingestion classification
- `WatchIngestPipeline` - pipeline connecting filesystem events to document ingestion

### TUI Adapters

- `FsfsScreen` / `FsfsTuiShellModel` - TUI screen and navigation model
- `ExplainabilityScreenState` - state for the search explainability view

### Output and Streaming

- `StreamFrame` / `StreamEvent` - streaming protocol for progressive result delivery
- `CompactEnvelope` / `CompactSearchResponse` - compact output for agent/IDE consumption

### Pressure and Lifecycle

- `PressureController` / `PressureSnapshot` - host resource pressure monitoring
- `LifecycleTracker` / `DaemonPhase` - daemon lifecycle state machine
- `PidFile` - process lock file management

### Reproducibility

- `ReproManifest` / `ReproInstance` - reproducibility artifacts for debugging
- `RedactionPolicy` - privacy-preserving output redaction

## Usage

```bash
# Search the current directory
fsfs "memory management"

# Search with JSON output
fsfs --format json "ownership borrowing"

# Index and search a specific directory
fsfs --path /path/to/project "error handling"

# Show configuration
fsfs config show

# Launch interactive TUI
fsfs --tui
```

## Dependency Graph Position

```
frankensearch-core
  ^     ^     ^     ^     ^
  |     |     |     |     |
embed index lexical storage tui
  \     |     /     /     /
   \    |    /     /     /
    frankensearch-fsfs (binary)
```

This is a leaf crate (no other crate depends on it). It pulls together most of the workspace to build the `fsfs` binary.

## License

MIT
