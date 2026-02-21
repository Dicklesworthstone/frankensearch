# AGENTS.md — frankensearch

> Guidelines for AI coding agents working in this Rust codebase.

---

## RULE 0 - THE FUNDAMENTAL OVERRIDE PREROGATIVE

If I tell you to do something, even if it goes against what follows below, YOU MUST LISTEN TO ME. I AM IN CHARGE, NOT YOU.

---

## RULE NUMBER 1: NO FILE DELETION

**YOU ARE NEVER ALLOWED TO DELETE A FILE WITHOUT EXPRESS PERMISSION.** Even a new file that you yourself created, such as a test code file. You have a horrible track record of deleting critically important files or otherwise throwing away tons of expensive work. As a result, you have permanently lost any and all rights to determine that a file or folder should be deleted.

**YOU MUST ALWAYS ASK AND RECEIVE CLEAR, WRITTEN PERMISSION BEFORE EVER DELETING A FILE OR FOLDER OF ANY KIND.**

---

## Irreversible Git & Filesystem Actions — DO NOT EVER BREAK GLASS

1. **Absolutely forbidden commands:** `git reset --hard`, `git clean -fd`, `rm -rf`, or any command that can delete or overwrite code/data must never be run unless the user explicitly provides the exact command and states, in the same message, that they understand and want the irreversible consequences.
2. **No guessing:** If there is any uncertainty about what a command might delete or overwrite, stop immediately and ask the user for specific approval. "I think it's safe" is never acceptable.
3. **Safer alternatives first:** When cleanup or rollbacks are needed, request permission to use non-destructive options (`git status`, `git diff`, `git stash`, copying to backups) before ever considering a destructive command.
4. **Mandatory explicit plan:** Even after explicit user authorization, restate the command verbatim, list exactly what will be affected, and wait for a confirmation that your understanding is correct. Only then may you execute it—if anything remains ambiguous, refuse and escalate.
5. **Document the confirmation:** When running any approved destructive command, record (in the session notes / final response) the exact user text that authorized it, the command actually run, and the execution time. If that record is absent, the operation did not happen.

---

## Git Branch: ONLY Use `main`, NEVER `master`

**The default branch is `main`. The `master` branch exists only for legacy URL compatibility.**

- **All work happens on `main`** — commits, PRs, feature branches all merge to `main`
- **Never reference `master` in code or docs** — if you see `master` anywhere, it's a bug that needs fixing
- **The `master` branch must stay synchronized with `main`** — after pushing to `main`, also push to `master`:
  ```bash
  git push origin main:master
  ```

**If you see `master` referenced anywhere:**
1. Update it to `main`
2. Ensure `master` is synchronized: `git push origin main:master`

---

## Toolchain: Rust & Cargo

We only use **Cargo** in this project, NEVER any other package manager.

- **Edition:** Rust 2024 (nightly required — see `rust-toolchain.toml`)
- **Dependency versions:** Explicit versions for stability
- **Configuration:** Cargo.toml workspace with `workspace = true` pattern
- **Unsafe code:** Forbidden (`#![forbid(unsafe_code)]`)

### Async Runtime: asupersync (MANDATORY — NO TOKIO)

**This project uses [asupersync](/dp/asupersync) exclusively for all async/concurrent operations. Tokio and the entire tokio ecosystem are FORBIDDEN.**

- **Structured concurrency**: `Cx`, `Scope`, `region()` — no orphan tasks
- **Cancel-correct channels**: Two-phase `reserve()/send()` — no data loss on cancellation
- **Sync primitives**: `asupersync::sync::Mutex`, `RwLock`, `OnceCell`, `Pool` — cancel-aware
- **Deterministic testing**: `LabRuntime` with virtual time, DPOR, oracles
- **Native HTTP**: `asupersync::http::h1` for model downloads (replaces reqwest)
- **Rayon is allowed**: For CPU-bound data parallelism (vector dot products). Rayon is not an async runtime.

**Forbidden crates**: `tokio`, `hyper`, `reqwest`, `axum`, `tower` (tokio adapter), `async-std`, `smol`, or any crate that transitively depends on tokio.

**Pattern**: All async functions take `&Cx` as first parameter. The `Cx` flows down from the consumer's runtime — frankensearch does NOT create its own runtime.

### Key Dependencies

| Crate | Purpose |
|-------|---------|
| `asupersync` | Structured async runtime (channels, sync, regions, HTTP, testing) |
| `half` | IEEE 754 f16 float support for quantized vectors |
| `wide` | Portable SIMD (`f32x8`) across x86 SSE2/AVX2 and ARM NEON |
| `memmap2` | Memory-mapped file I/O for zero-copy vector index access |
| `fastembed` | ONNX-based text embeddings (MiniLM-L6-v2 quality tier) |
| `safetensors` + `tokenizers` | Model2Vec static embeddings (potion-128M fast tier) |
| `tantivy` | Full-text BM25 search engine |
| `ort` | ONNX Runtime for cross-encoder reranking |
| `serde` + `serde_json` | Serialization |
| `thiserror` | Ergonomic error type derivation |
| `tracing` | Structured logging and diagnostics |
| `rayon` | Data parallelism for vector search (CPU-bound, not async) |
| `unicode-normalization` | NFC text canonicalization |

### Release Profile

The release build optimizes for performance (this is a library, not a binary):

```toml
[profile.release]
opt-level = 3       # Maximum performance optimization
lto = true          # Link-time optimization
codegen-units = 1   # Single codegen unit for better optimization
strip = true        # Remove debug symbols
```

---

## Code Editing Discipline

### No Script-Based Changes

**NEVER** run a script that processes/changes code files in this repo. Brittle regex-based transformations create far more problems than they solve.

- **Always make code changes manually**, even when there are many instances
- For many simple changes: use parallel subagents
- For subtle/complex changes: do them methodically yourself

### No File Proliferation

If you want to change something or add a feature, **revise existing code files in place**.

**NEVER** create variations like:
- `mainV2.rs`
- `main_improved.rs`
- `main_enhanced.rs`

New files are reserved for **genuinely new functionality** that makes zero sense to include in any existing file. The bar for creating new files is **incredibly high**.

---

## Backwards Compatibility

We do not care about backwards compatibility—we're in early development with no users. We want to do things the **RIGHT** way with **NO TECH DEBT**.

- Never create "compatibility shims"
- Never create wrapper functions for deprecated APIs
- Just fix the code directly

---

## Compiler Checks (CRITICAL)

**After any substantive code changes, you MUST verify no errors were introduced:**

```bash
# Check for compiler errors and warnings (workspace-wide)
cargo check --workspace --all-targets

# Check for clippy lints (pedantic + nursery are enabled)
cargo clippy --workspace --all-targets -- -D warnings

# Verify formatting
cargo fmt --check
```

If you see errors, **carefully understand and resolve each issue**. Read sufficient context to fix them the RIGHT way.

---

## Testing

### Testing Policy

Every component crate includes inline `#[cfg(test)]` unit tests alongside the implementation. Tests must cover:
- Happy path
- Edge cases (empty input, max values, boundary conditions)
- Error conditions

Cross-component integration tests live in the workspace `tests/` directory.

### Unit Tests

```bash
# Run all tests across the workspace
cargo test --workspace

# Run with output
cargo test --workspace -- --nocapture

# Run tests for a specific crate
cargo test -p frankensearch-core
cargo test -p frankensearch-embed
cargo test -p frankensearch-index
cargo test -p frankensearch-lexical
cargo test -p frankensearch-fusion
cargo test -p frankensearch-rerank
cargo test -p frankensearch-storage
cargo test -p frankensearch-durability
cargo test -p frankensearch-fsfs
cargo test -p frankensearch-tui
cargo test -p frankensearch-ops

# Run tests with all features enabled
cargo test --workspace --all-features
```

### Test Categories

| Crate | Focus Areas |
|-------|-------------|
| `frankensearch-core` | Error types, Embedder trait contracts, result type serialization, canonicalization pipeline, query classification |
| `frankensearch-embed` | Hash embedder determinism, Model2Vec tokenization + pooling, FastEmbed ONNX inference, auto-detection fallback chain |
| `frankensearch-index` | FSVI binary format round-trip, f16 quantization fidelity, SIMD dot product correctness, top-k heap ordering, NaN handling |
| `frankensearch-lexical` | Tantivy schema creation, document indexing, BM25 query parsing, search result ranking |
| `frankensearch-fusion` | RRF score calculation, 4-level tie-breaking, score normalization, two-tier blending, candidate budgeting |
| `frankensearch-rerank` | Cross-encoder scoring, sigmoid activation, pipeline integration |
| `frankensearch-storage` | SQLite schema bootstrap, dedup/content hashing, metadata persistence, embedding queue correctness |
| `frankensearch-durability` | Repair trailer I/O, file protection/repair pipelines, codec verification paths |
| `frankensearch-fsfs` | CLI config precedence, stream/output contracts, query execution pipeline, watcher lifecycle |
| `frankensearch-tui` | Shared shell/keymap/theme/determinism primitives, replay/evidence contracts |
| `frankensearch-ops` | Fleet telemetry ingestion/materialization, discovery logic, operations screens/state projection |
| `tests/` (workspace) | Cross-component integration, full search pipeline end-to-end, progressive iterator contract |
| `benches/` (workspace) | SIMD dot product throughput, top-k search scaling, embedding latency, RRF fusion overhead |

### Test Fixtures

A shared test fixture corpus (100 documents, 5 clusters, ground truth relevance data) is used across integration tests to ensure consistent cross-component validation.

---

## Third-Party Library Usage

If you aren't 100% sure how to use a third-party library, **SEARCH ONLINE** to find the latest documentation and current best practices.

---

## frankensearch — This Project

**This is the project you're working on.** frankensearch is a standalone, reusable Rust crate that extracts the 2-tier hybrid search system from three battle-tested codebases (cass, xf, mcp_agent_mail_rust) into a single drop-in library.

### What It Does

Combines lexical (Tantivy BM25) and semantic (vector cosine similarity) search via Reciprocal Rank Fusion, with a two-tier progressive model: fast results in <15ms, quality-refined results in ~150ms.

### Architecture

```
Query → Canonicalize → Classify → ┬─ Fast Embed (potion, 0.57ms) → Vector Search ─┐
                                   └─ Tantivy BM25 Lexical Search ─────────────────┤
                                                                                    ▼
                                                                        RRF Fusion (K=60)
                                                                                    │
                                                                    yield Initial (~15ms)
                                                                                    │
                                                                Quality Embed (MiniLM, 128ms)
                                                                                    │
                                                                Two-Tier Blend (0.7/0.3)
                                                                                    │
                                                             Optional FlashRank Rerank
                                                                                    │
                                                                   yield Refined (~150ms)
```

### Workspace Structure

```
frankensearch/
├── Cargo.toml                         # Workspace root
├── crates/
│   ├── frankensearch-core/            # Zero-dep traits, types, errors, canonicalization
│   ├── frankensearch-embed/           # Embedder impls (hash, model2vec, fastembed)
│   ├── frankensearch-index/           # FSVI vector index, SIMD dot product, top-k search
│   ├── frankensearch-lexical/         # Tantivy BM25 integration
│   ├── frankensearch-fusion/          # RRF, blending, TwoTierSearcher, telemetry/queue utilities
│   ├── frankensearch-rerank/          # FlashRank cross-encoder
│   ├── frankensearch-storage/         # FrankenSQLite metadata + job queue + optional FTS5
│   ├── frankensearch-durability/      # Repair/protection layer for index artifacts
│   ├── frankensearch-fsfs/            # Standalone CLI product
│   ├── frankensearch-tui/             # Shared TUI framework primitives
│   └── frankensearch-ops/             # Fleet observability/control-plane TUI
├── frankensearch/                     # Facade crate (re-exports everything)
├── tools/optimize_params/             # Parameter search/optimization helper tool
├── tests/                             # Cross-component integration tests
├── benches/                           # Performance benchmarks
└── examples/                          # Usage examples
```

### Key Files by Crate

| Crate | Key Files | Purpose |
|-------|-----------|---------|
| `frankensearch-core` | `src/lib.rs` | `Embedder` trait, `Reranker` trait, `SearchError`, `ScoredResult`, `VectorHit`, `FusedHit`, `Canonicalizer`, `QueryClass` |
| `frankensearch-embed` | `src/hash_embedder.rs` | FNV-1a hash embedder (zero deps, always available, test double) |
| `frankensearch-embed` | `src/model2vec_embedder.rs` | potion-128M static embedder (fast tier, ~0.57ms) |
| `frankensearch-embed` | `src/fastembed_embedder.rs` | MiniLM-L6-v2 ONNX embedder (quality tier, ~128ms) |
| `frankensearch-embed` | `src/auto_detect.rs` | `EmbedderStack` auto-detection and fallback chain |
| `frankensearch-index` | `src/format.rs` | FSVI binary format I/O (f16 quantized, memory-mapped) |
| `frankensearch-index` | `src/simd.rs` | `wide::f32x8` dot product (f16→f32 + SIMD multiply-accumulate) |
| `frankensearch-index` | `src/search.rs` | Brute-force top-k with BinaryHeap guard pattern, Rayon parallel |
| `frankensearch-lexical` | `src/lib.rs` | Tantivy schema, document indexing, BM25 query parsing |
| `frankensearch-fusion` | `src/rrf.rs` | Reciprocal Rank Fusion (K=60), 4-level tie-breaking |
| `frankensearch-fusion` | `src/blend.rs` | Two-tier score blending (0.7 quality / 0.3 fast) |
| `frankensearch-fusion` | `src/searcher.rs` | `TwoTierSearcher` progressive search orchestration and telemetry emission |
| `frankensearch-storage` | `src/pipeline.rs` | Storage-backed ingest/job pipeline and embedding vector sink contracts |
| `frankensearch-durability` | `src/fsvi_protector.rs` | FSVI protect/verify/repair flow for durability envelopes |
| `frankensearch-fsfs` | `src/runtime.rs` | CLI command execution lanes, search/index orchestration, stream protocol wiring |
| `frankensearch-tui` | `src/shell.rs` | Shared app-shell frame loop, navigation, overlays, and status plumbing |
| `frankensearch-ops` | `src/storage.rs` | Ops telemetry storage/materialization and control-plane persistence |
| `frankensearch-rerank` | `src/lib.rs` | FlashRank cross-encoder with sigmoid activation |

### Feature Flags

```toml
[features]
default = ['hash']
hash = ['frankensearch-embed/hash']                              # FNV-1a hash embedder (zero deps)
model2vec = ['frankensearch-embed/model2vec']                    # potion-128M fast embedder
fastembed = ['frankensearch-embed/fastembed']                    # MiniLM-L6-v2 quality embedder
lexical = ['dep:frankensearch-lexical', 'frankensearch-fusion/lexical']  # Tantivy BM25
storage = ['dep:frankensearch-storage']                          # FrankenSQLite persistence
durability = ['dep:frankensearch-durability']                    # Repair/protection layer
fts5 = ['storage', 'frankensearch-storage/fts5']                 # FTS5 storage backend
rerank = ['dep:frankensearch-rerank']                            # FlashRank cross-encoder
ann = ['frankensearch-index/ann']                                # HNSW approximate nearest neighbors
download = ['frankensearch-embed/download']                      # Model download via asupersync HTTP
semantic = ['hash', 'model2vec', 'fastembed']                    # All embedding models
hybrid = ['semantic', 'lexical']                                 # Semantic + lexical + RRF
persistent = ['hybrid', 'storage']                               # Hybrid + durable metadata/index queues
durable = ['persistent', 'durability']                           # Persistent + repair/protection
full = ['durable', 'rerank', 'ann', 'download']                  # Everything except FTS5
full-fts5 = ['full', 'fts5']                                     # Full stack + FTS5
```

### Core Types Quick Reference

| Type | Purpose |
|------|---------|
| `SearchPhase` | Progressive iterator yield: `Initial`, `Refined`, `RefinementFailed` |
| `FusedHit` | Hybrid search result with RRF score, per-source ranks, in_both_sources flag |
| `VectorHit` | Raw vector similarity result (index, score, doc_id) |
| `ScoredResult` | Final scored result with optional rerank score and metadata |
| `TwoTierSearcher` | Main orchestrator — progressive search via iterator protocol |
| `TwoTierConfig` | All tuning knobs (blend factor, RRF K, timeouts, fast_only mode) |
| `TwoTierMetrics` | Diagnostics: latencies, Kendall tau, promoted/demoted/stable counts, SkipReason |
| `Embedder` | Core async trait — `embed(&Cx, &str)`, `dimension()`, `is_semantic()`, `category()` |
| `Reranker` | Core async trait — `rerank(&Cx, query, docs)` cross-encoder scoring |
| `SearchError` | Unified error enum across all crates (thiserror-derived) |
| `Canonicalizer` | Text preprocessing trait (NFC, markdown strip, code collapse, truncation) |
| `QueryClass` | Query classification enum (Empty/Identifier/ShortKeyword/NaturalLanguage) |
| `EmbedderStack` | Auto-detected fast + quality embedder pair |
| `Cx` | asupersync capability context — passed to all async operations |
| `Outcome<T, E>` | Four-valued result: Ok, Err, Cancelled, Panicked |

### Performance Targets

| Operation | Target |
|-----------|--------|
| FNV-1a hash embed | ~0.07ms |
| potion-128M embed | ~0.57ms |
| MiniLM-L6-v2 embed | ~128ms |
| 384-dim f16 dot product | <2 microseconds |
| Phase 1 (Initial results) | <15ms total |
| Phase 2 (Refined results) | ~150ms total |
| Top-k search (1K vectors) | <1ms |
| Top-k search (10K vectors) | <15ms |
| Top-k search (100K vectors) | <150ms (with rayon) |

### Key Design Decisions

- **FSVI binary format** with `embedder_revision` field (generalized from CVVI/XFVI/AMVI formats across the 3 source codebases)
- **f16 quantization** by default — 50% memory savings, <1% quality loss for cosine similarity
- **`wide` and `half` are unconditional deps** of frankensearch-index (not feature-gated; SIMD and f16 are always available)
- **RRF does NOT depend on score normalization** — it's purely rank-based
- **NaN-safe `total_cmp()` ordering** in BinaryHeap top-k to prevent undefined sort behavior
- **Two-phase allocation** in top-k search: Phase 1 stores only `(index: u32, score: f32)`, Phase 2 resolves doc_id strings for winners only
- **fsync on vector index save** for durability guarantees
- **Structured tracing** throughout — every search emits spans with latency, counts, embedder info
- **asupersync exclusively** — NO tokio/reqwest/hyper. All async via `Cx` + structured concurrency
- **Rayon for data parallelism** — CPU-bound vector dot products use rayon's work-stealing; this composes with asupersync tasks
- **Cancel-correct lifecycle** — background workers use asupersync regions, channels use two-phase reserve/commit
- **LabRuntime for deterministic tests** — virtual time, DPOR schedule exploration, correctness oracles

---

## MCP Agent Mail — Multi-Agent Coordination

A mail-like layer that lets coding agents coordinate asynchronously via MCP tools and resources. Provides identities, inbox/outbox, searchable threads, and advisory file reservations with human-auditable artifacts in Git.

### Why It's Useful

- **Prevents conflicts:** Explicit file reservations (leases) for files/globs
- **Token-efficient:** Messages stored in per-project archive, not in context
- **Quick reads:** `resource://inbox/...`, `resource://thread/...`

### Same Repository Workflow

1. **Register identity:**
   ```
   ensure_project(project_key=<abs-path>)
   register_agent(project_key, program, model)
   ```

2. **Reserve files before editing:**
   ```
   file_reservation_paths(project_key, agent_name, ["src/**"], ttl_seconds=3600, exclusive=true)
   ```

3. **Communicate with threads:**
   ```
   send_message(..., thread_id="FEAT-123")
   fetch_inbox(project_key, agent_name)
   acknowledge_message(project_key, agent_name, message_id)
   ```

4. **Quick reads:**
   ```
   resource://inbox/{Agent}?project=<abs-path>&limit=20
   resource://thread/{id}?project=<abs-path>&include_bodies=true
   ```

### Macros vs Granular Tools

- **Prefer macros for speed:** `macro_start_session`, `macro_prepare_thread`, `macro_file_reservation_cycle`, `macro_contact_handshake`
- **Use granular tools for control:** `register_agent`, `file_reservation_paths`, `send_message`, `fetch_inbox`, `acknowledge_message`

### Common Pitfalls

- `"from_agent not registered"`: Always `register_agent` in the correct `project_key` first
- `"FILE_RESERVATION_CONFLICT"`: Adjust patterns, wait for expiry, or use non-exclusive reservation
- **Auth errors:** If JWT+JWKS enabled, include bearer token with matching `kid`

---

## Beads (br) — Dependency-Aware Issue Tracking

Beads provides a lightweight, dependency-aware issue database and CLI (`br` - beads_rust) for selecting "ready work," setting priorities, and tracking status. It complements MCP Agent Mail's messaging and file reservations.


### Conventions

- **Single source of truth:** Beads for task status/priority/dependencies; Agent Mail for conversation and audit
- **Shared identifiers:** Use Beads issue ID (e.g., `br-123`) as Mail `thread_id` and prefix subjects with `[br-123]`
- **Reservations:** When starting a task, call `file_reservation_paths()` with the issue ID in `reason`

### Typical Agent Flow

1. **Pick ready work (Beads):**
   ```bash
   bd ready --json  # Choose highest priority, no blockers
   ```

2. **Reserve edit surface (Mail):**
   ```
   file_reservation_paths(project_key, agent_name, ["src/**"], ttl_seconds=3600, exclusive=true, reason="br-123")
   ```

3. **Announce start (Mail):**
   ```
   send_message(..., thread_id="br-123", subject="[br-123] Start: <title>", ack_required=true)
   ```

4. **Work and update:** Reply in-thread with progress

5. **Complete and release:**
   ```bash
   bd close 123 --reason "Completed"
   bd sync  # Export to JSONL (no git operations)
   ```
   ```
   release_file_reservations(project_key, agent_name, paths=["src/**"])
   ```
   Final Mail reply: `[br-123] Completed` with summary

### Mapping Cheat Sheet

| Concept | Value |
|---------|-------|
| Mail `thread_id` | `br-###` |
| Mail subject | `[br-###] ...` |
| File reservation `reason` | `br-###` |
| Commit messages | Include `br-###` for traceability |

---

## bv — Graph-Aware Triage Engine

bv is a graph-aware triage engine for Beads projects (`.beads/beads.jsonl`). It computes PageRank, betweenness, critical path, cycles, HITS, eigenvector, and k-core metrics deterministically.

**Scope boundary:** bv handles *what to work on* (triage, priority, planning). For agent-to-agent coordination (messaging, work claiming, file reservations), use MCP Agent Mail.

**CRITICAL: Use ONLY `--robot-*` flags. Bare `bv` launches an interactive TUI that blocks your session.**

### The Workflow: Start With Triage

**`bv --robot-triage` is your single entry point.** It returns:
- `quick_ref`: at-a-glance counts + top 3 picks
- `recommendations`: ranked actionable items with scores, reasons, unblock info
- `quick_wins`: low-effort high-impact items
- `blockers_to_clear`: items that unblock the most downstream work
- `project_health`: status/type/priority distributions, graph metrics
- `commands`: copy-paste shell commands for next steps

```bash
bv --robot-triage        # THE MEGA-COMMAND: start here
bv --robot-next          # Minimal: just the single top pick + claim command
```

### Command Reference

**Planning:**
| Command | Returns |
|---------|---------|
| `--robot-plan` | Parallel execution tracks with `unblocks` lists |
| `--robot-priority` | Priority misalignment detection with confidence |

**Graph Analysis:**
| Command | Returns |
|---------|---------|
| `--robot-insights` | Full metrics: PageRank, betweenness, HITS, eigenvector, critical path, cycles, k-core, articulation points, slack |
| `--robot-label-health` | Per-label health: `health_level`, `velocity_score`, `staleness`, `blocked_count` |
| `--robot-label-flow` | Cross-label dependency: `flow_matrix`, `dependencies`, `bottleneck_labels` |
| `--robot-label-attention [--attention-limit=N]` | Attention-ranked labels |

**History & Change Tracking:**
| Command | Returns |
|---------|---------|
| `--robot-history` | Bead-to-commit correlations |
| `--robot-diff --diff-since <ref>` | Changes since ref: new/closed/modified issues, cycles |

**Other:**
| Command | Returns |
|---------|---------|
| `--robot-burndown <sprint>` | Sprint burndown, scope changes, at-risk items |
| `--robot-forecast <id\|all>` | ETA predictions with dependency-aware scheduling |
| `--robot-alerts` | Stale issues, blocking cascades, priority mismatches |
| `--robot-suggest` | Hygiene: duplicates, missing deps, label suggestions |
| `--robot-graph [--graph-format=json\|dot\|mermaid]` | Dependency graph export |
| `--export-graph <file.html>` | Interactive HTML visualization |

### Scoping & Filtering

```bash
bv --robot-plan --label backend              # Scope to label's subgraph
bv --robot-insights --as-of HEAD~30          # Historical point-in-time
bv --recipe actionable --robot-plan          # Pre-filter: ready to work
bv --recipe high-impact --robot-triage       # Pre-filter: top PageRank
bv --robot-triage --robot-triage-by-track    # Group by parallel work streams
bv --robot-triage --robot-triage-by-label    # Group by domain
```

### Understanding Robot Output

**All robot JSON includes:**
- `data_hash` — Fingerprint of source beads.jsonl
- `status` — Per-metric state: `computed|approx|timeout|skipped` + elapsed ms
- `as_of` / `as_of_commit` — Present when using `--as-of`

**Two-phase analysis:**
- **Phase 1 (instant):** degree, topo sort, density
- **Phase 2 (async, 500ms timeout):** PageRank, betweenness, HITS, eigenvector, cycles

### jq Quick Reference

```bash
bv --robot-triage | jq '.quick_ref'                        # At-a-glance summary
bv --robot-triage | jq '.recommendations[0]'               # Top recommendation
bv --robot-plan | jq '.plan.summary.highest_impact'        # Best unblock target
bv --robot-insights | jq '.status'                         # Check metric readiness
bv --robot-insights | jq '.Cycles'                         # Circular deps (must fix!)
```

---

## UBS — Ultimate Bug Scanner

**Golden Rule:** `ubs <changed-files>` before every commit. Exit 0 = safe. Exit >0 = fix & re-run.

### Commands

```bash
ubs file.rs file2.rs                    # Specific files (< 1s) — USE THIS
ubs $(git diff --name-only --cached)    # Staged files — before commit
ubs --only=rust,toml src/               # Language filter (3-5x faster)
ubs --ci --fail-on-warning .            # CI mode — before PR
ubs .                                   # Whole project (ignores target/, Cargo.lock)
```

### Output Format

```
⚠️  Category (N errors)
    file.rs:42:5 – Issue description
    💡 Suggested fix
Exit code: 1
```

Parse: `file:line:col` → location | 💡 → how to fix | Exit 0/1 → pass/fail

### Fix Workflow

1. Read finding → category + fix suggestion
2. Navigate `file:line:col` → view context
3. Verify real issue (not false positive)
4. Fix root cause (not symptom)
5. Re-run `ubs <file>` → exit 0
6. Commit

### Bug Severity

- **Critical (always fix):** Memory safety, use-after-free, data races, SQL injection
- **Important (production):** Unwrap panics, resource leaks, overflow checks
- **Contextual (judgment):** TODO/FIXME, println! debugging

---

## RCH — Remote Compilation Helper

RCH offloads `cargo build`, `cargo test`, `cargo clippy`, and other compilation commands to a fleet of 8 remote Contabo VPS workers instead of building locally. This prevents compilation storms from overwhelming csd when many agents run simultaneously.

**RCH is installed at `~/.local/bin/rch` and is hooked into Claude Code's PreToolUse automatically.** Most of the time you don't need to do anything if you are Claude Code — builds are intercepted and offloaded transparently.

To manually offload a build:
```bash
rch exec -- cargo build --release
rch exec -- cargo test
rch exec -- cargo clippy
```

Quick commands:
```bash
rch doctor                    # Health check
rch workers probe --all       # Test connectivity to all 8 workers
rch status                    # Overview of current state
rch queue                     # See active/waiting builds
```

If rch or its workers are unavailable, it fails open — builds run locally as normal.

### Sibling Dependency Bootstrap (bd-1pgv)

rch workers only sync the project directory — sibling path dependencies (`asupersync`, `frankensqlite`, `fast_cmaes`) referenced in Cargo.toml are **not** available on workers by default. Before offloading builds, ensure deps are set up:

```bash
# Check if deps are available (exit 1 if missing)
scripts/rch-ensure-deps.sh --check

# Auto-fix local sibling deps
scripts/rch-ensure-deps.sh

# Force refresh to pinned commits
scripts/rch-ensure-deps.sh --force

# Bootstrap deps on every configured rch worker
scripts/rch-ensure-deps.sh --all-workers

# Verify worker bootstrap (exit 1 if any worker is missing deps)
scripts/rch-ensure-deps.sh --all-workers --check
```

On the dev machine this is usually a no-op (deps are already at `/data/projects/`). In worker mode (`--all-workers` or `--worker`), it SSHes to workers and clones pinned sibling deps into `/tmp/rch/frankensearch/{asupersync,frankensqlite,fast_cmaes}` so path dependencies resolve during `rch exec -- cargo ...`. The pinned commit refs match `.github/workflows/ci.yml`.

**Note for Codex/GPT-5.2:** Codex does not have the automatic PreToolUse hook, but you can (and should) still manually offload compute-intensive compilation commands using `rch exec -- <command>`. This avoids local resource contention when multiple agents are building simultaneously.

---

## ast-grep vs ripgrep

**Use `ast-grep` when structure matters.** It parses code and matches AST nodes, ignoring comments/strings, and can **safely rewrite** code.

- Refactors/codemods: rename APIs, change import forms
- Policy checks: enforce patterns across a repo
- Editor/automation: LSP mode, `--json` output

**Use `ripgrep` when text is enough.** Fastest way to grep literals/regex.

- Recon: find strings, TODOs, log lines, config values
- Pre-filter: narrow candidate files before ast-grep

### Rule of Thumb

- Need correctness or **applying changes** → `ast-grep`
- Need raw speed or **hunting text** → `rg`
- Often combine: `rg` to shortlist files, then `ast-grep` to match/modify

### Rust Examples

```bash
# Find structured code (ignores comments)
ast-grep run -l Rust -p 'fn $NAME($$$ARGS) -> $RET { $$$BODY }'

# Find all unwrap() calls
ast-grep run -l Rust -p '$EXPR.unwrap()'

# Quick textual hunt
rg -n 'println!' -t rust

# Combine speed + precision
rg -l -t rust 'unwrap\(' | xargs ast-grep run -l Rust -p '$X.unwrap()' --json
```

---

## Morph Warp Grep — AI-Powered Code Search

**Use `mcp__morph-mcp__warp_grep` for exploratory "how does X work?" questions.** An AI agent expands your query, greps the codebase, reads relevant files, and returns precise line ranges with full context.

**Use `ripgrep` for targeted searches.** When you know exactly what you're looking for.

**Use `ast-grep` for structural patterns.** When you need AST precision for matching/rewriting.

### When to Use What

| Scenario | Tool | Why |
|----------|------|-----|
| "How does the two-tier searcher work?" | `warp_grep` | Exploratory; don't know where to start |
| "Where is the RRF fusion implemented?" | `warp_grep` | Need to understand architecture |
| "Find all uses of `Embedder::embed`" | `ripgrep` | Targeted literal search |
| "Find files with `println!`" | `ripgrep` | Simple pattern |
| "Replace all `unwrap()` with `expect()`" | `ast-grep` | Structural refactor |

### warp_grep Usage

```
mcp__morph-mcp__warp_grep(
  repoPath: "/data/projects/frankensearch",
  query: "How does the progressive search iterator work?"
)
```

Returns structured results with file paths, line ranges, and extracted code snippets.

### Anti-Patterns

- **Don't** use `warp_grep` to find a specific function name → use `ripgrep`
- **Don't** use `ripgrep` to understand "how does X work" → wastes time with manual reads
- **Don't** use `ripgrep` for codemods → risks collateral edits
## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Sync beads** - `bd sync` to export to JSONL
5. **Hand off** - Provide context for next session


---

Note for Codex/GPT-5.2:

You constantly bother me and stop working with concerned questions that look similar to this:

```
Unexpected changes (need guidance)

- Working tree still shows edits I did not make in Cargo.toml, Cargo.lock, src/cli/commands/upgrade.rs, src/storage/sqlite.rs, tests/conformance.rs, tests/storage_deps.rs. Please advise whether to keep/commit/revert these before any further work. I did not touch them.

Next steps (pick one)

1. Decide how to handle the unrelated modified files above so we can resume cleanly.
2. Triage beads_rust-orko (clippy/cargo warnings) and beads_rust-ydqr (rustfmt failures).
3. If you want a full suite run later, fix conformance/clippy blockers and re‑run cargo test --all.
```

NEVER EVER DO THAT AGAIN. The answer is literally ALWAYS the same: those are changes created by the potentially dozen of other agents working on the project at the same time. This is not only a common occurence, it happens multiple times PER MINUTE. The way to deal with it is simple: you NEVER, under ANY CIRCUMSTANCE, stash, revert, overwrite, or otherwise disturb in ANY way the work of other agents. Just treat those changes identically to changes that you yourself made. Just fool yourself into thinking YOU made the changes and simply don't recall it for some reason.

---

## Note on Built-in TODO Functionality

Also, if I ask you to explicitly use your built-in TODO functionality, don't complain about this and say you need to use beads. You can use built-in TODOs if I tell you specifically to do so. Always comply with such orders.

<!-- beads-agent-instructions-v2 -->

---

## Beads Workflow Integration

This project uses [beads](https://github.com/steveyegge/beads) for issue tracking. Issues live in `.beads/` and are tracked in git.

Two CLIs: **bd** (issue CRUD) and **bv** (graph-aware triage, read-only).

### bd: Issue Management

```bash
bd ready              # Unblocked issues ready to work
bd list --status=open # All open issues
bd show <id>          # Full details with dependencies
bd create --title="..." --type=task --priority=2
bd update <id> --status=in_progress
bd close <id>         # Mark complete
bd close <id1> <id2>  # Close multiple
bd dep add <a> <b>    # a depends on b
bd sync               # Sync with git
```

### bv: Graph Analysis (read-only)

**NEVER run bare `bv`** — it launches interactive TUI. Always use `--robot-*` flags:

```bash
bv --robot-triage     # Ranked picks, quick wins, blockers, health
bv --robot-next       # Single top pick + claim command
bv --robot-plan       # Parallel execution tracks
bv --robot-alerts     # Stale issues, cascades, mismatches
bv --robot-insights   # Full graph metrics: PageRank, betweenness, cycles
```

### Workflow

1. **Start**: `bd ready` (or `bv --robot-triage` for graph analysis)
2. **Claim**: `bd update <id> --status=in_progress`
3. **Work**: Implement the task
4. **Complete**: `bd close <id>`
5. **Sync**: `bd sync` at session end

### Session Close Protocol

```bash
git status            # Check what changed
git add <files>       # Stage code changes
bd sync               # Commit beads changes
git commit -m "..."   # Commit code
bd sync               # Commit any new beads changes
git push              # Push to remote
```

### Key Concepts

- **Priority**: P0=critical, P1=high, P2=medium, P3=low, P4=backlog (numbers only)
- **Types**: task, bug, feature, epic, question, docs
- **Dependencies**: `bd ready` shows only unblocked work

<!-- end-beads-agent-instructions -->
