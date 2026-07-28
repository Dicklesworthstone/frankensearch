# frankensearch-fsfs

Standalone `fsfs` CLI and runtime for two-tier hybrid local search.

## Overview

`frankensearch-fsfs` is the user-facing CLI application that provides a full-featured local search experience. It combines semantic vector search, BM25 lexical search, and two-tier progressive ranking into a terminal application with both a rich TUI mode and structured output formats (JSON, TOON). It includes filesystem watching for live index updates, configurable pressure sensing, query planning, and an explainability screen.

The crate is split into a library (`src/lib.rs`) for reusable runtime, configuration, and adapter logic, and a binary (`src/main.rs`) for the `fsfs` CLI entrypoint.

## Build Profiles

The default feature set is model-free: ordinary builds and crates.io packaging
do not require Potion or MiniLM artifacts at compile time. It retains model
acquisition and verification commands, but it cannot execute Model2Vec or
FastEmbed. `fsfs download-models` can prepare verified build inputs; it cannot
add a backend that was not compiled into the binary.

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
