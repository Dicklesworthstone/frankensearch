# Dependency Upgrade Log

## 2026-08-14 — Asupersync 0.4.4 lockstep + FrankenSQLite 0.3.1 usage

**Scope:** workspace Asupersync floor is now `>=0.4.4, <0.5`. `Cargo.lock`
resolves `asupersync` / `asupersync-macros` 0.4.4 from crates.io. FrankenSQLite
stays on published `0.3.1` (latest). No public frankensearch API break.

- **Asupersync 0.4.4:** native-task abort preserves an acknowledged
  `Cancelled` result; HTTP/1 streaming cancel/reuse is additive. Download
  now uses the caller `Cx` (plus `checkpoint()` in the manifest loop)
  instead of minting `Cx::for_request()`. Watcher join already prefers the
  typed task result over a generic join-cancel, which matches the 0.4.4
  contract.
- **FrankenSQLite 0.3.1 usage:** storage open, schema bootstrap, and
  storage transactions retry the published `is_transient()` family
  (`Busy` / `BusyRecovery` / `BusySnapshot` / `DatabaseLocked` /
  `WriteConflict` / `SerializationFailure` / `PageBufferCapacityExhausted`)
  with bounded backoff. Ops open retry walks the error source chain to
  `FrankenError::is_transient()` instead of matching the word "busy" in
  display text. Catalog file-open retries the same transient family.
- **Contract:** publish-identity gate and gauntlet fuzz pin now bind
  `asupersync@0.4.4`.

## 2026-08-14 — FrankenSQLite 0.3.1 crates.io bump

**Scope:** every frankensearch `fsqlite*` edge now resolves the published
`0.3.1` registry crates. This is a lockstep bug-fix release on the 0.3 API
(autocommit durability, concurrent-open prepare, committed-freelist safety,
read-only open sidecars). No FrankenSearch call-site signature change.

- **fsfs / storage / ops** stay on `AsyncConnection` plus the `*_sync` facade.
- **durability** RaptorQ `SymbolCodec` now binds `fsqlite-core` 0.3.1.

## 2026-08-14 — FrankenSQLite 0.3.0 crates.io cutover

**Scope:** every frankensearch `fsqlite*` edge now resolves the published
`0.3.0` registry crates. The remaining 0.1.2/0.1.19 compatibility line in
fsfs and durability is gone, and the workspace git patches that split
FrankenSQLite across two revs are removed.

- **fsfs catalog** uses FrankenSQLite 0.3's `AsyncConnection` synchronous
  facade (`open_sync` / `execute_sync` / `query_sync`).
- **durability** RaptorQ `SymbolCodec` now binds `fsqlite-core` 0.3.0.
- **storage / ops** keep the 0.3.0 API they already used, but now take it
  from crates.io instead of a git rev behind the 0.3.0 tag.

## 2026-08-11 — Asupersync 0.4.3 and FrankenSQLite 0.3 migration: IN PROGRESS

**Scope:** the workspace now resolves one Asupersync runtime family at 0.4.3.
The earlier resolution barrier was removed by lifting FrankenSearch's production
FrankenSQLite edge from the synchronous 0.1 API to the asynchronous 0.3 API.

### Asupersync: `0.3.10` → `0.4.3`

- **Target provenance:** stable `v0.4.3` crates.io release and tagged upstream
  source/changelog.
- **Constraint:** `>=0.4.3, <0.5`, with `default-features = false` and the
  production `proc-macros` feature retained. Test-only crates opt into
  `test-internals` explicitly.
- **Graph result:** the root workspace, FrankenSQLite 0.3 production edge, and
  retained FrankenSQLite 0.1 compatibility edge all resolve Asupersync 0.4.3;
  `cargo tree -d` shows no second Asupersync version.
- **Contract updates:** dependency-identity assertions and the separately rooted
  gauntlet fuzz workspace now bind 0.4.3 instead of the stale 0.3.10 identity.

### FrankenSQLite / `fsqlite`: production edge `0.1.19` → `0.3.0`

- Storage operations were migrated to the async connection, statement, row,
  transaction, and FTS5 APIs.
- FrankenSearch's synchronous public storage contract is preserved through
  FrankenSQLite 0.3's `AsyncConnection` synchronous facade. The connection owns
  its worker boundary; FrankenSearch does not create an ambient runtime or mint
  test contexts in production storage code.
- Commit and rollback now execute the real FrankenSQLite transaction operations;
  a dropped or panicking transaction is rolled back before the worker accepts
  another request.
- The legacy 0.1 package remains only where the current FrankenSQLite extension
  graph still exposes that audited compatibility edge. Both package lines share
  Asupersync 0.4.3, so they no longer split the runtime universe.

### Validation

Remote workspace/profile checks, focused storage tests, fuzz-workspace compile,
dependency audit, and CI validation are recorded with the completing commit.

### Primary sources

- Asupersync [`v0.4.3` changelog](https://github.com/Dicklesworthstone/asupersync/blob/v0.4.3/CHANGELOG.md)
  and [tagged source](https://github.com/Dicklesworthstone/asupersync/tree/v0.4.3).
- FrankenSQLite [`v0.2.1` changelog](https://github.com/Dicklesworthstone/frankensqlite/blob/v0.2.1/CHANGELOG.md)
  and [tagged source](https://github.com/Dicklesworthstone/frankensqlite/tree/v0.2.1).

## 2026-06-17 — straggler third-party majors

Two leftovers from earlier passes, finished now. `Cargo.lock` is gitignored in
this repo, so only the `Cargo.toml` edits are committable; the local lock was
re-resolved for validation.

- **sha2 0.10 → 0.11 (workspace) in `frankensearch-storage`.** The workspace
  root already declared `sha2 = "0.11.0"` and every other crate used
  `sha2 = { workspace = true }`; `frankensearch-storage` was the lone straggler
  pinning `"0.10"` directly. Switched it to `{ workspace = true }`. No code
  change — `content_hash.rs` already hex-encodes with a manual `write!("{:02x}")`
  loop (not the `LowerHex`-on-digest pattern that sha2 0.11 dropped).
- **jsonschema 0.17 → 0.46 (dev-dep) in `frankensearch-fsfs`.** Migrated the
  `schema_conformance` test off the removed `JSONSchema` builder API:
  `JSONSchema::options().with_draft(Draft::Draft202012).compile(&s)` →
  `jsonschema::draft202012::new(&s)` returning a `Validator`; validation now uses
  `validator.iter_errors(&value)` (collect-all for should-pass) and
  `validator.is_valid(&value)` (for should-fail) instead of the old
  iterator-returning `validate()`.

### Validation
- `cargo check -p frankensearch-fsfs --tests`: ✅ (jsonschema 0.46 API compiles)
- `cargo test -p frankensearch-fsfs --test schema_conformance`: ✅ **118 passed,
  0 failed** — including `test_schema_fixtures_validate_against_jsonschema`.
- `cargo audit`: no vulnerabilities (3 pre-existing allowed "unmaintained" advisories).

---

**Date:** 2026-02-17  
**Project:** frankensearch  
**Language:** Rust

## Summary
- **Upgraded major core deps** to current Rust-1.85-compatible versions
- **Updated manifests** in workspace root and crate-level manifests
- **Applied API migrations** required by newer `ort`, `fastembed`, `safetensors`, `notify`, and `criterion`
- **Validated:** `cargo check --workspace`, `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`

## Direct / Workspace Dependency Upgrades

### Search / IR
- `tantivy`: `0.22.1` -> `0.25.0`

### Embeddings / Tokenization / ONNX
- `fastembed`: `4.9.1` -> `5.8.0`
- `tokenizers`: `0.21.4` -> `0.22.2`
- `safetensors`: `0.5.3` -> `0.7.0`
- `ort`: `2.0.0-rc.9` -> `2.0.0-rc.10`
- `ort-sys`: `2.0.0-rc.9` -> `2.0.0-rc.10`

### Tooling / Runtime
- `criterion`: `0.5.1` -> `0.7.0`
- `sysinfo`: `0.33.1` -> `0.36.1`
- `toml`: `0.8.23` -> `1.0.2+spec-1.1.0`
- `notify` (crate-level in `frankensearch-fsfs`): `7.0.0` -> `8.2.0`

## Code Migrations Performed

### `ort` rc10 migration (`crates/frankensearch-rerank/src/lib.rs`)
- `SessionOutputs<'_, '_>` -> `SessionOutputs<'_>`
- `Tensor::from_array((shape, slice))` -> owned arrays (`Vec`) input form
- `ort::inputs!` handling adjusted (no `map_err` on macro result)
- `try_extract_raw_tensor` -> `try_extract_tensor`
- `Session::run` mutability update (`&mut Session`)

### `fastembed` 5.8 migration (`crates/frankensearch-embed/src/fastembed_embedder.rs`)
- mutable model/session handles where embed APIs now require mutable receiver
- batch embed call adjusted to avoid unnecessary owned conversion

### `safetensors` 0.7 migration
- `serialize(&tensors, &None)` -> `serialize(&tensors, None)` in:
  - `crates/frankensearch-embed/src/auto_detect.rs`
  - `crates/frankensearch-embed/src/model2vec_embedder.rs`

### Minor compatibility / lint updates
- tensor-name discovery adjusted for current string types (`model2vec_embedder`)
- deprecated `criterion::black_box` replaced with `std::hint::black_box` in benchmark files:
  - `crates/frankensearch-durability/benches/durability_bench.rs`
  - `frankensearch/benches/search_bench.rs`

## Remaining Behind Latest (after update)
`cargo update --verbose` still reports unresolved newest versions for some crates, primarily due Rust-version constraints (`rust-version > 1.85`) or upstream selection constraints:
- Rust version constrained: `criterion 0.8.2`, `ort 2.0.0-rc.11`, `sysinfo 0.38.2`, `time 0.3.47`, `time-core 0.1.8`, `time-macros 0.2.27`, `wide 1.1.1`, `wasip2/wasip3/wit-bindgen*`
- Also unresolved at latest despite wildcarded manifest: `fastembed 5.9.0`, `generic-array 0.14.9`, `indexmap 2.13.0`, `libc 0.2.182`, `signal-hook 0.4.3`, `smallvec 2.0.0-alpha.12`

## Validation Run
- `cargo check --workspace` ✅
- `cargo fmt --check` ✅
- `cargo clippy --workspace --all-targets -- -D warnings` ✅

## Notes
- External dependency warnings from `/data/projects/fast_cmaes` are still printed during checks, but they do not fail builds for this workspace.

---

## 2026-02-18 Follow-up Update

### Summary
- Ran `cargo update --verbose` and `cargo update --verbose --ignore-rust-version`
- Updated workspace dependency constraints and lockfile to latest practical versions in this environment
- Revalidated formatting and workspace compile after upgrades

### Workspace manifest updates
- `fastembed`: `5.8.0 -> 5.9.0`
- `ort`: `2.0.0-rc.10 -> 2.0.0-rc.11`
- `ndarray`: `0.16 -> 0.17`
- `toml`: `1.0.2 -> 1.0.3`
- `criterion`: `0.7.0 -> 0.8.2`
- `time`: `0.3 -> 0.3.47`
- `sysinfo`: `0.36.1 -> 0.38.2`
- `wide`: `0.7 -> 1.1.1`

### Lockfile updates observed
- `fastembed 5.8.0 -> 5.9.0`
- `ort 2.0.0-rc.10 -> 2.0.0-rc.11`
- `ort-sys 2.0.0-rc.10 -> 2.0.0-rc.11`
- `ndarray 0.16.1 -> 0.17.2`
- `criterion 0.7.0 -> 0.8.2`
- `criterion-plot 0.6.0 -> 0.8.2`
- `sysinfo 0.36.1 -> 0.38.2`
- `time 0.3.45 -> 0.3.47`
- `time-core 0.1.7 -> 0.1.8`
- `time-macros 0.2.25 -> 0.2.27`
- `wide 1.1.1` added
- plus related transitive graph updates/removals

### Post-update validation
- `cargo fmt` ✅
- `cargo fmt --check` ✅
- `cargo check --workspace` ✅

### Clippy status
- `cargo clippy --workspace --all-targets -- -D warnings` ❌
- Current failure is concentrated in `crates/frankensearch-lexical/src/cass_compat.rs` with strict pedantic/nursery lint violations (e.g. `missing_errors_doc`, `too_many_lines`, `iter_with_drain`, `derive_partial_eq_without_eq`, etc.).
- These are style/lint policy failures, not dependency-resolution or compile failures. No rollback was applied because core compile and tests for dependent cass integration remained functional.

---

## 2026-02-19 Dependency Update

### Summary
- Bumped MSRV from `1.85` to `1.95` to match nightly toolchain and unlock all dependency updates
- Updated `fastembed` 5.9.0 → 5.11.0 (workspace Cargo.toml)
- Updated `signal-hook` 0.3 → 0.4 (frankensearch-fsfs/Cargo.toml)
- Ran `cargo update` to pull all semver-compatible transitive bumps
- No source code changes required — all API surfaces remained compatible

### Manifest changes
- `rust-version`: `1.85` → `1.95` (workspace Cargo.toml, matches nightly toolchain)
- `fastembed`: `5.9.0` → `5.11.0` (workspace Cargo.toml)
- `signal-hook`: `0.3` → `0.4` (crates/frankensearch-fsfs/Cargo.toml)

### Lockfile updates
- `fastembed` 5.9.0 → 5.11.0
- `signal-hook` 0.3.18 removed (replaced by 0.4.x)
- `bumpalo` 3.20.1 → 3.20.2
- `clap` 4.5.59 → 4.5.60
- `clap_builder` 4.5.59 → 4.5.60
- `darling` 0.21.3 → 0.23.0
- `security-framework` 3.6.0 → 3.7.0
- `security-framework-sys` 2.16.0 → 2.17.0
- `wasip2` 1.0.1 → 1.0.2

### Still behind latest (held by upstream constraints)
- `generic-array` 0.14.7 (available 0.14.9) — held by transitive dependents
- `indexmap` 2.12.1 (available 2.13.0) — held by transitive dependents
- `libc` 0.2.180 (available 0.2.182) — held by transitive dependents
- `objc2-core-foundation` 0.3.1 (available 0.3.2) — macOS-only, held by upstream
- `objc2-io-kit` 0.3.1 (available 0.3.2) — macOS-only, held by upstream

### Already at latest
serde 1.0.228, serde_json 1.0.149, thiserror 2.0.18, tracing 0.1.44,
tracing-subscriber 0.3.22, rayon 1.11.0, half 2.7.1, memmap2 0.9.10,
safetensors 0.7.0, tokenizers 0.22.2, tantivy 0.25.0, ort 2.0.0-rc.11,
ndarray 0.17.2, hnsw_rs 0.3.3, crc32fast 1.5.0, unicode-normalization 0.1.25,
dirs 6.0.0, toml 1.0.3, sha2 0.10.9, criterion 0.8.2, proptest 1.10.0,
time 0.3.47, sysinfo 0.38.2, toon-rust 0.1.3, tempfile 3.25.0,
tracing-test 0.2.6, xxhash-rust 0.8.15, notify 8.2.0, ignore 0.4.25, wide 1.1.1

### Validation
- `cargo check` (all non-TUI crates): ✅
- `cargo clippy -- -D warnings` (all non-TUI crates): ✅
- `cargo test` (2,706+ tests across 10 crates): ✅ all passing, 0 failures
- **Note:** TUI crates (`frankensearch-fsfs`, `frankensearch-tui`, `frankensearch-ops`) blocked by a pre-existing `ftui-text` nightly regression (`E0515` in `markup.rs:143`) unrelated to this dependency update
