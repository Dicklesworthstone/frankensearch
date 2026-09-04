# Quill E9.3 — Post-Flip Removal Proposal

Status: PROPOSAL ONLY. Nothing here has been deleted; per AGENTS.md Rule 1 the
user disposes, this document prepares. Prepared 2026-09-04 by PeachCliff
(bd-quill-e9-docs-retirement-3gul.3; supersedes the inline proposal comments on
bd-d7xk1).

## 0. QG-10 re-check (evidence for the sweep)

`cargo tree` on the published facade and the product binary, run 2026-09-04
via rch (`--job` lane):

| Feature set | tantivy in tree |
|---|---|
| `frankensearch --no-default-features` | 0 |
| `--features lexical` (post-flip default = Quill) | 0 |
| `--features hybrid` (670-line tree, quill present) | 0 |
| `--features full` | 0 |
| `--features full-fts5` | 0 |
| `frankensearch-fsfs` (default binary features) | 0 |
| `--features lexical-tantivy` (oracle lane, control) | present — `tantivy v0.26.1` |

Tantivy is reachable only through the `lexical-tantivy` / `cass-compat` oracle
and interop lanes. Facade consumers are feature-gated re-exports
(`frankensearch/src/lib.rs`: `pub use frankensearch_lexical as lexical_tantivy`
plus gated `TantivyIndex` / `cass_compat` re-exports; gates counted: lib.rs 3,
index_builder.rs 2). No default-path Tantivy plumbing remains in the facade or
fsfs.

## 1. Proposed removals (require explicit user approval)

### 1.1 `crates/frankensearch-durability/src/tantivy_wrapper.rs` — 1115 lines

- Retired `DurableTantivyIndex` / `TantivySegmentProtector` wrapper.
- No `mod` declaration in `crates/frankensearch-durability/src/lib.rs`
  (documented retired at line 12); compiled by nothing.
- The crate README states it was removed from the build in the Quill migration
  (bd-tkjm) and "remains on disk only until the quill-e9.3 retirement sweep
  lands its approved removal" — this document is that proposal.
- Engine-neutral durability crate: Tantivy appears nowhere in its dependency
  graph; the generic `FileProtector` boundary is the kept capability.
- Revival path if ever needed: git history keeps the bytes; the capability
  already exists engine-neutrally in `file_protector.rs`.

### 1.2 `crates/frankensearch-core/src/metrics.rs` — 1402 lines

- TDigest, MedianMAD, HuberEstimator, HyperLogLog, RobustMetrics.
- No `mod` declaration in `crates/frankensearch-core/src/lib.rs` (the crate
  registers `metrics_eval`, a different module); zero consumers workspace-wide.
- Registering it would add ~1.4k lines of public API surface to the published
  `frankensearch-core` crate with no product consumer. Revival path: git
  (present at 52b8c822); or extract to a future stats crate when a consumer
  exists.
- Also filed as bd-d7xk1; one approval covers both beads.

### 1.3 Repro scratch (four small files, 141 lines total)

- `crates/frankensearch-fusion/src/repro_blend.rs` (54) — two real regression
  tests over `blend_two_tier`, undeclared. If kept, the honest form is an
  integration test (`crates/frankensearch-fusion/tests/`, import conversion);
  if the behaviors are already covered by `blend.rs` inline tests, remove.
- `crates/frankensearch-fusion/src/repro_rrf.rs` (49) — same shape.
- `crates/frankensearch-tui/src/repro_input.rs` (32) — undeclared scratch.
- `crates/frankensearch-rerank/src/test_api.rs` (1, empty) and
  `crates/frankensearch-rerank/src/test_inputs.rs` (5, `fn main` scratch) —
  no non-destructive disposition exists; removal is the only clean option.

### 1.4 Acceptance once disposal is approved

A workspace test or lint asserting "no `.rs` under any crate `src/` is
unreferenced" (the bd-d7xk1 acceptance) goes green with the list above removed;
nothing else in `src/` is undeclared (verified by mod-declaration sweep,
2026-09-04).

## 2. Investigated and NOT proposed (census corrections)

- `ord_table` machinery (`frankensearch-lexical/src/lib.rs`:
  `ord_table: RwLock<Vec<DocId>>`, `load_ord_table_sidecar`,
  `persist_ord_table`, `ord_table.json` sidecar) — alive and load-bearing in
  the Tantivy oracle lane, including the `benches/reopen_id_materialize.rs`
  fast path. The e9.3 bead's original census named it as dead; that is stale.
- `frankensearch-lexical` itself — not dead: it is the pinned conformance
  oracle and CASS interop lane, deliberately retained per the flip contract
  (owner ruling 2026-09-01). Retirement of the whole crate is a separate
  future decision, out of scope here.
- `TantivyIndex` default-path plumbing — none exists anymore: the facade port
  (quill-e7.3) removed tantivy-shaped APIs without compat shims; fsfs talks to
  the lexical backend only through the Quill-shaped consumer API.

## 3. Disposal

Reply with explicit approval (file list is enough) and the removal + the
bd-d7xk1 acceptance lint/test land as one reviewed commit.
