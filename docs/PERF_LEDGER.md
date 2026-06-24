# PERF_LEDGER.md — frankensearch measured wins

> Head-to-head measured performance wins **kept** in the tree. Each row cites the
> exact bench workload, the before/after timings, and the ratio (new/old; lower is
> faster). Dead-ends and regressions live in `docs/NEGATIVE_EVIDENCE.md`.

Build/bench protocol (per-crate ONLY, never workspace-wide):
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p <crate> --release
```

Current Cargo rejects `--release` for `cargo bench` in this checkout; successful
Criterion runs use the equivalent `--profile release` form and record that protocol
mismatch in `docs/NEGATIVE_EVIDENCE.md`.

The dominance baseline is a Tantivy/Lucene/Meilisearch-class original comparator
on identical corpora and query streams. Until `bd-ui41` lands that harness, rows
below are limited to frankensearch pre-change baselines or before/after local
hot-path ratios and must not be presented as original-comparator wins.

| Date | Crate | Lever | Workload (bench id) | Before | After | Ratio | Status |
|------|-------|-------|---------------------|--------|-------|-------|--------|
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators | `dot/dim256/f32_bytes` | 10.839 ms | 3.647 ms | **0.336** | KEEP |
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators | `dot/dim384/f32_bytes` | 14.084 ms | 5.333 ms | **0.379** | KEEP |
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators (`BlueGull` pinned-worker confirmation) | `dot/dim256/f32_bytes/10000` | 3.4835 ms | 0.66126 ms | **0.190** | KEEP (`vmi1149989`) |
| 2026-06-24 | frankensearch-index | `f32_bytes` fixed-array decode + 4 accumulators (`BlueGull` pinned-worker confirmation) | `dot/dim384/f32_bytes/10000` | 5.1487 ms | 1.8811 ms | **0.365** | KEEP (`vmi1149989`) |
| 2026-06-24 | frankensearch-index | **branchless SIMD f16→f32 widen** (default path) | `dot/dim256/f16_bytes` | 4.733 ms | 1.632 ms | **0.345** | KEEP |
| 2026-06-24 | frankensearch-index | **branchless SIMD f16→f32 widen** (default path) | `dot/dim384/f16_bytes` | 7.363 ms | 2.332 ms | **0.317** | KEEP |
| 2026-06-24 | frankensearch-index | branchless SIMD f16→f32 widen | `dot/dim256/f16_slice` | 3.699 ms | 1.348 ms | **0.364** | KEEP |
| 2026-06-24 | frankensearch-index | branchless SIMD f16→f32 widen | `dot/dim384/f16_slice` | 5.536 ms | 2.181 ms | **0.394** | KEEP |

**Lever (bd-gfzk):** `dot_product_f16_bytes_f32` / `dot_product_f16_f32` — the **default
quantization** path (f16 FSVI, `search.rs:288,346,441`). The f16 paths were decode-bound: the
per-element scalar `f16::to_f32()` dominated (confirmed in `docs/NEGATIVE_EVIDENCE.md` — that's
why accumulator unrolling failed here). Replaced the 8× scalar decode with a **branchless SIMD
widen** (Giesen magic-multiply over `wide::u32x8`/`f32x8` + pure-integer inf/nan fixup) →
**~2.5–3.2× faster** on the dominant path. The widen is **bit-exact** to `f16::to_f32()` for all
65 536 f16 values (exhaustive test `simd_f16_widen_is_bit_exact`), and the single accumulator /
lane assignment is unchanged, so every dot score is **bit-identical** on finite data → no
determinism/golden risk (all 19 simd + full index lib tests green). Measured head-to-head in one
process (`f16_*_new` SIMD vs `f16_*_old` scalar). No dominance-vs-original claim (blocked by
`bd-ui41`); pre-change before/after ratio only.

**Lever:** `dot_product_f32_bytes_f32` (used by f32-quantized FSVI indexes, `search.rs:307,372,492`).
The old kernel decoded f32s from an open-ended `&stored_bytes[off..]` slice, which the
compiler could not prove a fixed width for and so did not vectorize the `from_le_bytes`
decode. Rewriting the decode over fixed `[u8; 32]` sub-blocks (+ 4 independent f32x8
accumulators) lets it vectorize → **~2.7–3.0× faster**. Measured head-to-head in one
process via `benches/dot_product.rs` (`f32_bytes_new` vs `f32_bytes_old`), so the ratio is
immune to which rch worker the run landed on. f16 paths were tried with the same restructure
but **regressed** (decode is scalar-bound there) and were reverted — see
`docs/NEGATIVE_EVIDENCE.md`. No dominance-vs-original claim (blocked by `bd-ui41`); this is a
frankensearch pre-change before/after ratio only.

`BlueGull` confirmation command:
```bash
RCH_WORKER=vmi1149989 \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo bench -p frankensearch-index --profile release --bench dot_product
```
Same worker conformance gate:
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-a \
  rch exec -- cargo test -p frankensearch-index simd --profile release
```
Result: 18 relevant SIMD tests passed; RCH reported `remote vmi1149989`.

## Baselines only

These rows are routing evidence for future levers, not wins.

Command:
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b \
  rch exec -- cargo bench -p frankensearch --profile release --bench search_bench -- --sample-size 10
```

Worker: `ovh-a`; RCH reported remote completion in `608.1s`.

| Date | Crate | Workload | Measurement | Original ratio | Status |
|------|-------|----------|-------------|----------------|--------|
| 2026-06-24 | frankensearch | `dot_product_f32/128` | `12.223 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `dot_product_f32/256` | `30.630 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `dot_product_f32/384` | `44.346 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `dot_product_f32/768` | `82.308 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `hash_embedder/short_10w` | `734.48 ns` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `hash_embedder/medium_100w` | `2.3161 us` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `hash_embedder/long_1000w` | `17.225 us` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `vector_search_topk/top10/1000` | `944.07 us` | blocked by `bd-ui41` | noisy baseline |
| 2026-06-24 | frankensearch | `vector_search_topk/top10/5000` | `3.4640 ms` | blocked by `bd-ui41` | noisy baseline |
| 2026-06-24 | frankensearch | `vector_search_topk/top10/10000` | `1.6642 ms` | blocked by `bd-ui41` | noisy baseline |
| 2026-06-24 | frankensearch | `rrf_fusion/fuse/1000+1000` | `80.563 us` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `score_normalization/z_score/10000` | `43.159 us` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `index_io/write/10000` | `13.812 ms` | blocked by `bd-ui41` | baseline only |
| 2026-06-24 | frankensearch | `index_io/open/10000` | `19.865 us` | blocked by `bd-ui41` | baseline only |
