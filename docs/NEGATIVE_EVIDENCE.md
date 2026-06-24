# NEGATIVE_EVIDENCE.md — frankensearch perf swarm

> Honest ledger of perf experiments that **did NOT pay off** (≈0 gain or regression)
> and were therefore **reverted**. The point of this file is to stop future agents
> (and future me) from re-attempting dead ends. Every entry must cite the measured
> ratio vs. the pre-change baseline on the same workload.

Conventions:
- **Workload** = the exact bench id (`cargo bench` group/function) measured head-to-head.
- **Ratio** = new_time / old_time. `< 1.0` is a speedup, `> 1.0` is a regression.
- A lever is **reverted** if ratio ∈ [0.97, 1.03] (noise) or > 1.03 (regression).
- Wins (ratio < 0.97, kept) go in `docs/PERF_LEDGER.md`, not here.

Build/bench protocol (per-crate ONLY):
```bash
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
  rch exec -- cargo bench -p <crate> --release
```

---

## Measurement blockers

| Date | Owner | Workload | Evidence | Status |
|------|-------|----------|----------|--------|
| 2026-06-24 | BlueGull (`frankensearch-cod-a`) | `frankensearch/search_bench vector_search_topk/top10/10000` | `cargo bench --release -p frankensearch --bench search_bench vector_search_topk/top10/10000 -- --quiet` failed before measurement on rustc `1.98.0-nightly (f20a92ec0 2026-06-07)` because Cargo rejected `--release` for `cargo bench` as an unexpected argument. | Blocker tracked in `bd-ui41`; do not count as a perf ratio. |
| 2026-06-24 | BlueGull (`frankensearch-cod-a`) | `frankensearch/search_bench vector_search_topk/top10/10000` | Fallback optimized bench command without `--release` ran through RCH on `vmi1153651` but remained in cold compile/link with no Criterion timing output after more than 10 minutes; interrupted by the owner with exit 130. | No ratio produced; use `bd-ui41` to establish a reproducible harness and command contract. |
| 2026-06-24 | BlueGull (`frankensearch-cod-a`) | Tantivy/Lucene-class original comparison | README/AGENTS confirm frankensearch is a Tantivy BM25 + semantic/vector hybrid, but no current per-crate harness emits same-corpus ratios against a Tantivy-only incumbent. | Blocker tracked in `bd-ui41`; no dominance claim is valid until this exists. |
| 2026-06-24 | frankensearch-cod-b | `frankensearch/search_bench` requested protocol | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cod-b rch exec -- cargo bench -p frankensearch --release --bench search_bench -- --sample-size 10` selected RCH worker `ovh-a`, then Cargo rejected `--release` for `cargo bench` with `unexpected argument '--release' found`; `cargo bench --help` lists `--profile <PROFILE-NAME>` instead. | Same protocol blocker as above; successful measurement used `--profile release` and remains per-crate. |
| 2026-06-24 | frankensearch-cod-b | `frankensearch/search_bench vector_search_topk/top10/{1000,5000,10000}` | Same-worker RCH run on `ovh-a`: `rch exec -- cargo bench -p frankensearch --profile release --bench search_bench -- --sample-size 10`. Results: 1K `944.07 us`, 5K `3.4640 ms`, 10K `1.6642 ms`. | Scaling order is unstable/noisy; use as routing evidence only, not as keep/reject proof. |

---

## Reverted experiments

_(none yet — entries appended as experiments fail)_
