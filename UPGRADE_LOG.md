# Dependency Upgrade Log

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
