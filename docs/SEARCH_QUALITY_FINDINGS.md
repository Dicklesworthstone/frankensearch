# Search Quality: Historical Python Experiments and Current Implementation Scope

A consolidated, prioritized roadmap distilled from the real-embedding + real-benchmark
investigation recorded in `NEGATIVE_EVIDENCE.md` (commits `3833955`→, IronPetrel, 2026-07-02/03).
The numerical observations below are retained from those experiments on real
embeddings, BEIR datasets, and relevance judgments. The retrieval harness uses
Python `model2vec` and `rank_bm25`; it does **not** execute Tantivy, Quill, the Rust
searchers, or `fsfs`. `rank_bm25` is a BM25 proxy, not Tantivy's implementation.
The later stem/stop tokenizer improves that proxy's lexical analysis but does
not establish engine parity. Historical timings and test counts retain their
original scope and have not been remeasured here.

## Current implementation boundary (2026-09-04)

- Quill is the default lexical backend; Tantivy is an explicit oracle/CASS
  interoperability lane. These Python results do not certify Quill quality.
- `TwoTierConfig::default()` selects `FusionStrategy::Rrf`, `rrf_k = 60`, and
  `quality_weight = 0.7`. `PoolMinMax` is opt-in. Statements below proposing a
  pool-min-max default describe the historical experiment, not the current default.
- The library's `TwoTierSearcher` and `SyncTwoTierSearcher` install adaptive NQC
  weighting by default, neutral through its 128-query warm-up. Both expose
  `with_rrf_weights` and `with_rrf_tiebreak`; there is no missing builder plumbing
  for those controls. The Python nDCG gains do not prove gains in either Rust path.
- `fsfs` uses a separate orchestrator in `runtime.rs`; it does not inherit the
  library's NQC state or fusion strategy. Its native reranker is available via
  `search --rerank`, independently of these historical recommendations.
- Current product quality validation belongs to the existing Rust paths and
  `bd-quill-e6-gauntlet-scale-rm3q.7`. Real-model phase delivery is necessary but
  does not establish a relevance improvement. Use labeled, held-out queries,
  matched candidate coverage, and per-corpus paired comparisons before changing
  ranking defaults; retain harmful or undecidable slices.

Source: `crates/frankensearch-core/src/config.rs`,
`crates/frankensearch-fusion/src/{searcher,sync_searcher}.rs`, and
`crates/frankensearch-fsfs/src/runtime.rs`.

## Historical experiment verdict

The Python hybrid improved on its lexical proxy in the reported semantic
retrieval experiments. Model choice produced large differences on those English
datasets. Neither result is a live Rust-versus-Tantivy comparison or a universal
guarantee that refinement improves ranking.

## Historical recommendations, by experimental effect

### 1. Change the default embedder → a retrieval-distilled model. [biggest, ~free]
- **What:** `DEFAULT_MODEL_NAME`/`DEFAULT_HF_ID` in `crates/frankensearch-embed/src/model2vec_embedder.rs:35`
  is `potion-multilingual-128M` — **the worst of 4 model2vec models on English retrieval**
  (SciFact vector recall@10 = 0.598). A retrieval-distilled model (`minishlab/potion-retrieval-32M`
  class) scores **0.795 — +33% relative.**
- **Cost:** none — `model2vec` embed speed is ~flat across models (token-lookup + mean-pool, not a
  transformer forward pass), and `retrieval-32M` MRL-truncated to **dim=256** scores 0.757 (95% of
  full) at the **same scan cost, memory, and embed throughput** as the 256-dim default → a **free
  +27%** at identical cost. 32M < 128M, so it's smaller on disk too.
- **Caveat:** keep a multilingual option for non-English corpora; the recommendation is the
  *English default*. (This is the one genuine product decision — multilingual vs English-first.)
- **Premium tier (contextual):** frankensearch also ships a contextual ONNX path (`fastembed`).
  A contextual model (`BAAI/bge-small-en-v1.5`) beats the best static on **all 3 BEIR datasets** —
  **+14% nDCG SciFact, +9.6% NFCorpus, +28.8% ArguAna** (ArguAna recall 0.698→0.841) — but embeds
  **~650× slower** (transformer forward pass vs static lookup) and needs the onnxruntime. Since embed
  documents are embedded at index time and each query also needs an embedding, contextual is a candidate premium for
  quality-sensitive, rarely-reindexed corpora. **The premium is largest exactly where static embeddings
  are weakest** (ArguAna's argument→counter-argument task, which needs contextual understanding of
  argumentative structure/negation that static mean-pooling can't capture). Tiered guidance:
  **static `retrieval-32M` = fast default; contextual BGE = quality premium (+10-29% nDCG, multi-dataset
  validated), most compelling on semantically hard / argumentative corpora.**

### 2. Ship the hybrid as the default (it's the safe choice across domains).
- Hybrid ≥ the better single tier on all 3 BEIR datasets; wins meaningfully on semantic corpora
  (SciFact hybrid 0.834 vs BM25 0.776), ties on keyword-overlap ones. **Never worse than the best
  single tier by more than noise.** The vector tier alone beats real BM25 on all 3 datasets
  (including ArguAna, where BM25's long-doc length-penalty hurts it — a weakness embeddings lack).
- **Within this experiment: one strong static embedder + BM25, without a static-model ensemble.** Adding a
  second static embedder never helps and usually *hurts* (SciFact ret32+base32 0.769 < ret32 alone
  0.795; the triple 2-embedders+BM25 is worse than the pair on both datasets). The model2vec embedders
  are too correlated (a 2nd finds a doc the 1st misses in only ~3% of queries vs ~7% for BM25) *and*
  weaker, so RRF just dilutes. **The hybrid's power is MODALITY diversity (exact-term vs semantic →
  decorrelated errors), not signal count** — and the partner must be *comparably strong* (high
  unique-% alone doesn't help if the partner is weak; see NEGATIVE_EVIDENCE). To strengthen the vector
  tier, use a *better* single embedder (rec #1), not *more* embedders.

### 3. Tune RRF fusion: up-weight the STRONGER tier, small k, deep candidate feed.
- **Small RRF `k` (~10, not 60) is the key fusion knob.** k=60 is too flat over the top-10; k≈10 sharpens
  `1/(k+r)` so the top ranks separate (+2.6% nDCG vs k=60, `4cc3b47`). Given small-k, the **tier-weight is
  largely redundant**: equal weight ≈ 1.3× within noise, and over-weighting (≥1.7) *hurts* (`44566fd`) — small-k
  substitutes for the weight. So: **small k≈10 + a light weight (1.0-1.3, never more)**; the earlier "1.3×"
  mattered mainly at the old k=60. Up-weight the *stronger* tier for the workload if you weight at all.
- **Neutral (hash) tiebreak** — but *only if you run equal weights*. Tier-weighting **subsumes** it: with weights the
  fused contributions are distinct, eliminating ~95-97% of RRF ties (stock equal-weight has ties in 27-48% of queries;
  weighted has ~1%). So the lexical-favoring tiebreak biases 1/3-1/2 of *stock* results, but is nearly moot once you
  weight — **adopt tier-weighting and you can skip the tiebreak knob** (`396e2c6`).
- **Deep candidate feed is a RERANKER-only lever** (`candidate_multiplier`). Deeper feed grows recall@100
  (the reranker-feed metric — hybrid feeds 96% of relevant docs, SciFact recall@100=0.960, and the vector
  edge over BM25 grows with depth), but the **non-reranked hybrid top-10 plateaus at feed≈20** — a candidate
  at feed-rank 80 contributes ~`1/(k+80)`≈0, so it can't crack the fused top-10 (`b311ac3`). So: **~20-50/tier
  if NOT reranking; ~100/tier only when reranking** (the reranker is what turns deep recall into final quality).
- **Keep RRF (rank fusion) — don't switch to score-fusion.** Measured RRF vs score-fusion (normalize
  BM25+cosine, weighted sum) across all 3 datasets: they're **tied on quality** (RRF wins recall on 2/3;
  score-fusion wins nDCG by a hair, only NFCorpus's +0.014 above noise), but RRF is **scale-free** — no
  per-query normalizer to choose (z-norm vs min-max disagree and are brittle to BM25's unbounded vs
  cosine's bounded distributions). RRF's simplicity + recall edge win. This validates `rrf_fuse` as the
  fusion primitive across the whole stack (base hybrid *and* reranker-combine).

### 4. Reranker is a CONDITIONAL polish — there is NO safe default reranker (measured across 3 datasets).
The "best" reranker **flips completely by corpus**, and *stronger is not safer.* Same hybrid candidates,
both cross-encoders vs no-rerank, on all 3 BEIR datasets:

| dataset (query style) | `ms-marco-L6` (default) | `bge-reranker-base` (strong) |
|---|---|---|
| SciFact (scientific *claims*) | −2.1% | **+4.4%** |
| NFCorpus (health *questions*) | **+23.5%** | ~0 |
| ArguAna (arg → *counter*-arg) | **+7.0%** | **−23.4%** |

- **`bge-reranker-base` is NOT a safe default** (retracts an earlier single-dataset call): it wins only on
  SciFact, is inert on NFCorpus, and **loses 23% on ArguAna** — a strong *similarity* reranker demotes the
  *counter*-arguments ArguAna wants. When the task's relevance notion diverges from the reranker's training
  objective, a stronger reranker does *more* damage.
- **`ms-marco-L6` is the better generalist** (net-positive 2/3, +23.5% on web-question-style NFCorpus) but is
  mildly negative on claim-style SciFact.
- **Recommendation:** treat the reranker as a **per-corpus eval decision, not a default** — the downside of a
  mismatched reranker (−23%) exceeds the upside of the right one on most corpora. **The hybrid alone is never
  catastrophic; a mismatched reranker can be.** Cost adds to the case: `bge-reranker-base` (~278M) is ≈3× slower
  per pair than `ms-marco-L6` on CPU. When in doubt, **skip reranking** — or A/B a candidate reranker per corpus.
- **Rerank DEPTH is a second corpus-dependent knob — bias shallow.** How many candidates to feed the cross-encoder
  has opposite-signed optima: on NFCorpus/ms-marco reranking is monotonic-increasing (knee at D≈20-30 — captures 90%
  of the lift at top-20, never pay for >30); on SciFact/bge it's monotonic-**decreasing** past D=5 (deep reranking
  promotes false positives, flipping +0.019→−0.011). So **default to a shallow rerank depth (~top-10)** and only go
  deeper if a per-corpus eval shows the reranker is well-matched *and* relevant docs sit deep. Deeper reranking is
  neither free (linear cross-encoder cost) nor safe (an imperfect reranker injects false positives at depth).
- **Integrate the reranker as a THIRD RRF SOURCE — don't pure-reorder, don't even score-blend (this is the safety net,
  and it's native).** Rank by RRF-fusing the retrieval-order and the reranker-order, **not** by the reranker score
  alone. This resolves the downside risk above and beats the alternatives: on SciFact/bge pure-reorder *hurts* (−0.011),
  score-blend recovers it (+0.025), but **RRF-combine does best (+0.040)** — the rank fusion caps how far a deep false
  positive can climb (it must rank high by *both* retrieval and the reranker). It's **parameter-free** (no α, no score
  normalization, and k-insensitive: k=10≈k=60), so nothing to tune per corpus. It only trails pure-reorder when the
  reranker is strongly matched (NFCorpus +0.062 vs RRF +0.047) — the right trade for a default, since you can't know a
  priori if the reranker fits. **Zero new machinery: frankensearch already ships `rrf_fuse`** — feed the reranker as
  another ranked source (optionally tier-weighted). Net reranker verdict: **RRF-combine (never pure-reorder), bias
  shallow depth, still sanity-check the model per corpus — but RRF-combine makes the tier safe and tuning-free by
  default.** (Score-blend `α·reranker+(1-α)·retrieval`, α≈0.5, is the fallback if a non-RRF stage is combining.)

### 5. Historical int8 two-pass experiment (`39dd9be`)
- The original record reported **7.1× faster than flat exact @ recall 1.0** on
  its real-embedding workload and 820/820 fusion tests passing. That observation
  is not an exactness guarantee for other corpora or a current performance result.
  The present `SyncTwoTierSearcher` defaults to exact retrieval; the hidden
  approximate int8 seam is reserved for isolated benchmarks/tests.

## Total measured uplift (recommendations 1+3, end-to-end hybrid)
Refreshed with **all** validated retrieval-side knobs — retrieval-32M **@ dim-256** (`06fdef9`) + tier-weight 1.3 +
**RRF k=10** (`4cc3b47`) — vs stock (multilingual-128M + equal-weight k=60), both at dim-256 (equal scan cost, no
reranker). recall@10 / nDCG@10:

| BEIR dataset | stock (multilingual-128M + equal RRF k=60) | recommended (retrieval-32M@256 + w-1.3 + k=10) | Δrecall / ΔnDCG |
|---|---|---|---|
| SciFact | 0.756 / 0.590 | **0.842 / 0.676** | **+11% / +15%** |
| NFCorpus | 0.133 / 0.269 | **0.160 / 0.325** | **+20% / +21%** |
| ArguAna | vector already strong vs BM25's long-doc penalty | (embedder swap adds least) | ~+1-4% |

The compounding free/cheap changes (retrieval-distilled embedder + stronger-tier weight + small k, all at dim-256) lift
the semantic-search hybrid **+15-21% nDCG at zero inference cost** — ~3× the earlier estimate, which omitted the
validated small-k and full tuning. (Add the optional reranker on top for the capstone's further gains.)

## Historical Python capstone: pipeline (1+3+4) vs naive BM25 proxy

> **Historical baseline correction (2026-07-04, `NEGATIVE_EVIDENCE.md`,
> "HEADLINE CORRECTION"):** the baseline below uses naive-tokenizer
> `rank_bm25` without stemming/stopwords. Adding stem/stop analysis to the Python
> proxy changed the reported dense marginal value to **+0.6–3.0% nDCG /
> +0.5–5.3% recall** (mean ~+2.3%), versus the +12–22% naive-baseline headline.
> The stemmed proxy alone matched or beat the naive hybrid on all four corpora
> (ArguAna: +13.3%). Neither comparison ran production Tantivy; read these
> numbers as comparisons of the named Python configurations.

The experimental stack — retrieval-32M + tuned RRF hybrid **+ RRF-combine reranker** —
is compared with the naive BM25 proxy, showing recall@10 / nDCG@10. Similar Rust
configuration APIs exist, but this run did not exercise them:

| BEIR | Naive BM25 proxy | + hybrid | + RRF-combine rerank | full vs proxy |
|---|---|---|---|---|
| SciFact | 0.776 / 0.652 | 0.816 / 0.684 | **0.872 / 0.731** | **+12% / +12%** |
| NFCorpus | 0.152 / 0.306 | 0.159 / 0.327 | **0.167 / 0.346** | **+10% / +13%** |
| ArguAna (200q) | 0.565 / 0.259 | 0.620 / 0.294 | **0.680 / 0.316** | **+20% / +22%** |

The full-stack win is **largest on ArguAna (+22% nDCG)** — precisely because BM25 is structurally weakest there (its
long-doc length penalty), a weakness the dense vector tier is immune to. Headline across 3 datasets: **+12% to +22%
nDCG / +10% to +20% recall vs naive BM25 proxy.**

**Historical cost estimate (separate per-crate measurements):** the hybrid's added latency over lexical-only was estimated as *vector search + RRF
fusion* (the BM25 tier is shared) = **~250-290 µs (int8 vector, 10k×384) + ~149 µs (fusion) ≈ < 1 ms/query** — near-free
for the recall/nDCG win. The **reranker** is the only expensive stage (~ms/candidate cross-encoder inference →
100 ms-1 s for a top-100 window; its RRF-combine reorder is itself µs), so it's a quality-vs-latency-gated polish, not a
default cost. These separate measurements omit query embedding and do not establish
whole-product latency or a same-invocation performance win against Tantivy.

### Library configuration example (not an executed capstone receipt)

These builder calls express the experimental settings. The fragment assumes
the named index, models, and lexical implementation have already been created;
it is not a standalone runnable example or a claim that `fsfs` uses this recipe:
```rust
use frankensearch_rerank::{RerankCombine, DEFAULT_RRF_COMBINE_K};
// #1 embedder: load a retrieval-distilled static model (potion-retrieval-32M) as the fast/quality embedder.
let searcher = TwoTierSearcher::new(index, retrieval_distilled_embedder, config) // config.rrf_k = 10.0  (#3 small k)
    .with_lexical(lexical_index)                                    // Quill, or explicitly selected Tantivy interop
    .with_rrf_weights(1.0, 1.3)                                     // #3 up-weight the STRONGER tier (~1.3×)
    .with_rrf_tiebreak(RrfTiebreak::Hash)                          // #3 neutral tiebreak
    .with_reranker(cross_encoder)                                   // #4 corpus-appropriate cross-encoder
    .with_rerank_combine(RerankCombine::RrfCombine { k: DEFAULT_RRF_COMBINE_K }); // #4 RRF-combine (never pure-reorder)
```
`SyncTwoTierSearcher` exposes the same `with_rrf_weights` / `with_rrf_tiebreak`.
It currently uses exact fast retrieval. Selecting different ranking defaults
requires actual serving-path evidence; availability of the builders is not that
evidence.

## Gotchas ruled out (measured)
- **Query prefixes** (`query:`/`passage:`): do NOT add them — potion-retrieval-32M is a no-prefix
  (symmetric) model; prefixes cost 2-3 recall pts.
- **MRL dim-truncation:** viable (graceful) only on retrieval-distilled models (95% recall @2× smaller);
  catastrophic on the general default (0.545). `mrl.rs`'s "2-6× faster" is real+recall-safe *only* with
  a Matryoshka-trained embedder — another reason for recommendation #1.
- **Short queries:** the vector tier collapses on ≤3-word queries (recall 0.45) — a *fundamental*
  query-underspecification (query-embedding drift), model-invariant; lexical BM25 carries them.
- **Embedding-space PRF / query expansion (Rocchio):** do NOT add it for the static vector tier — it's
  net-harmful (SciFact: every config loses, up to −0.14 nDCG; harm grows monotonically with feedback =
  query drift). Static mean-pooled doc vectors averaged into the query blur it toward the corpus centroid;
  top-k false positives poison the centroid. Lexical BM25, not vector PRF, is the weak-query remedy.

## 2026-07-12 update — fusion design-space mapped cross-corpus + a NEW leverage (NQC dense down-weight)

Second investigation pass (cc_fse), rebuilt harness (`docs/quality_harness/`, model2vec potion + `rank_bm25`,
now with a **stem/stop Python proxy** via `snowballstemmer` + a fixed stopword set), across all 4 BEIR corpora
(SciFact / NFCorpus / ArguAna / SciDocs), nDCG@10. Full trail + self-corrections in `NEGATIVE_EVIDENCE.md`.

- **Proxy analyzer comparison.** Python Snowball stem+stop lifts lexical **+0.035 (+5.4%)** / hybrid **+0.024 (+3.6%)**
  on scifact vs basic tokenization. This measures the proxy tokenizer change, not
  production Tantivy or Quill execution. Measure proxy fusion levers on the
  stem+stop baseline, then validate the actual engine separately.
- **pool-min-max SCORE fusion (`a9e53b4`) validated cross-corpus:** beats RRF on **4/4** corpora (scifact +0.0146,
  nfcorpus +0.0123, arguana +0.0001, scidocs +0.0051) in this Python experiment.
  Current Rust defaults remain RRF; this does not confirm a shipped pool-min-max default.
- **Fusion knobs ruled out (measured, cross-corpus):** z-score normalization ≈ min-max (WASH, |Δ|≤0.002, loses
  scifact); candidate pool ≈100 is **Pareto** (larger pools give ~+0.001 nDCG for ~5× fusion+retrieval cost —
  don't enlarge; smaller costs quality).
- **★ MAJOR REFRAME — the dense tier is HIGH-VARIANCE.** Per-query, dense is net-neutral-or-**harmful** on **~71–81%**
  of queries on every corpus (it helps a ~20–30% minority a lot). Its marginal hybrid value ranges +0.0247 (scifact)
  down to +0.0017 (arguana ≈ nothing). An ORACLE that skips dense exactly where it doesn't help would drop ~¾ of the
  (dominant-cost) dense scans AND gain +0.017–0.042 nDCG. This quantifies the long-standing "hybrid absorbs dense
  micro-opts" theme: dense's marginal hybrid value (≤+0.025) upper-bounds any dense-*side* quality tweak.
- **★ NEW LEVERAGE (landable) — NQC-adaptive dense DOWN-WEIGHT.** Down-weight (not skip) the dense tier on
  high-commitment queries: `w_dense = clip(1 − β·CDF(cv), w_min, 1)`, `cv` = coefficient-of-variation of the top-k
  BM25 scores (NQC), β≈0.5. Beats equal-weight pool-min-max on **4/4** corpora; **statistically real POOLED**
  (3029 queries, +0.0022, bootstrap 95% CI **[+0.0008, +0.0035]**), deployment-validated leakage-free (held-out cv
  CDF). Small but real, **latency-neutral** (NQC is a single-pass reduction, 0.37× the per-tier min-max the fusion
  already does). The HARD gate (skip dense) is NOT robustly Pareto (nfcorpus needs dense on its committed queries) —
  the SOFT down-weight is, because it retains partial dense. LAND-SCOPING: must use a per-deployment cv-**percentile**
  CDF (streaming t-digest over the query stream) — a fixed `β·cv` does NOT transfer (cv scale is corpus-dependent).
  **Library implementation history (2026-07-13): default-on adaptive NQC.** The pipeline landed as 6 increments
  (`66592753` `NqcCvSampler` rolling sample → `749030c2` `AdaptiveNqcDenseWeight` self-driving cv→percentile
  sketch → `5fa9acfd`/`af1a7678` sync+async searcher wiring → `eb6fa58c` blessed defaults + disable escape →
  `ac081b7d` default-ON flip). Both searchers now install `AdaptiveNqcDenseWeight::production_default()` (β=0.5,
  2048-query rolling window, `min_samples`=128, rebuild every 64) in `new()`; it stays neutral (byte-identical
  fusion) through the 128-query warm-up, then realizes the down-weight online — no external sample management, no
  per-corpus calibration. The deployment-faithful cv-**percentile** CDF is the rolling empirical sample (not a
  fixed β·cv). Verified: fusion suite green under **default features** (lib 892 / integration 38, exit 0) **and `--all-features`**
  (graph/rerank/lexical/bench-internals paths, exit 0) — no feature-gated fallout; cold-start neutrality held
  everywhere (the integration tests run ≤12 queries/searcher, below the 128 warm-up). Opt out via
  `with_nqc_dense_downweight_disabled()` (byte-identical A/B). Remaining route-next (non-blocking, SCOPED
  2026-07-13): re-confirm the LIVE Rust default on a labeled corpus. NOTE the true scope — the `+0.0022` and every
  nDCG here is from the **Python** proxy harness (`docs/quality_harness/`, model2vec + rank_bm25 reimplementation),
  which does NOT exercise the shipped Rust code. Validate the actual Rust
  `TwoTierSearcher` with default-on versus `with_nqc_dense_downweight_disabled()`
  in the existing quality work item `bd-quill-e6-gauntlet-scale-rm3q.7`, recording
  warm-up and held-out query order. Unit coverage of cv → percentile → weight →
  fusion does not establish an end-to-end nDCG gain. `fsfs` has no corresponding
  adaptive-NQC wiring, so its product tests are a separate serving path.
- **Statistical caution (applies to every single-run number in this harness):** each nDCG figure is a POINT
  ESTIMATE; treat **±0.003-scale deltas as noise** unless pooled/CI'd. The large deltas above (stem+stop +0.024/
  +0.035, pool-min-max>RRF +0.017) are trustworthy; the sub-0.003 washes (z-score, pool-size) were correctly read
  as noise.

## 2026-07-14 update — bootstrap-CI on the pool-min-max vs RRF fusion default (corrects an over-cited single-run number)

The pool-min-max-beats-RRF result (a9e53b4) was a **single-run point estimate** on scifact ("+0.017, trustworthy").
Per this doc's own statistical caution (±0.003-scale single-run deltas are noise), that needed a bootstrap CI. New
harness `fusion_pmm_vs_rrf_ci.py` computes the **per-query nDCG@10 delta (pool-min-max − RRF)** with a 2000-resample
95% bootstrap CI, stem+stop lexical + model2vec `potion-retrieval-32M` dense, top-100 pool, per corpus + POOLED
(seed 12345):

| corpus | q | pool-min-max | RRF | Δ (mm−rrf) | 95% CI | CI excludes 0 |
|---|---:|---:|---:|---:|---|---|
| scifact | 300 | 0.7120 | 0.7003 | +0.0117 | [−0.0003, +0.0245] | **no (borderline)** |
| nfcorpus | 323 | 0.3471 | 0.3344 | +0.0127 | [+0.0068, +0.0186] | yes |
| arguana | 1406 | 0.3601 | 0.3604 | −0.0003 | [−0.0042, +0.0039] | no (wash) |
| scidocs | 1000 | 0.1627 | 0.1576 | +0.0051 | [+0.0023, +0.0080] | yes |
| **POOLED** | **3029** | — | — | **+0.0041** | **[+0.0016, +0.0066]** | **yes** |

**Verdict:** pool-min-max is **pooled-decisively but MODESTLY** better than RRF (+0.0041, CI excludes 0 over 3029
queries) and is **never decisively worse** on any single corpus — so it is a safe-or-better default fusion. BUT the
effect is corpus-dependent (decisive only on nfcorpus/scidocs; a wash on arguana) and, importantly, **the famous
scifact "+0.017 trustworthy" single-run number is actually BORDERLINE under bootstrap** — its 95% CI
[−0.0003, +0.0245] includes zero. So the honest case for pool-min-max as the fusion default rests on the *pooled*
cross-corpus CI (~+0.004), not the scifact headline (which overstated a borderline result). This reinforces the
doc's own caution: CI single-tier-vs-fusion deltas before quoting them as robust. No Rust change (Python proxy
harness; does not exercise shipped code). Harness: `docs/quality_harness/fusion_pmm_vs_rrf_ci.py`.

## 2026-07-14 update — bootstrap-CI on the DENSE tier's marginal value (dense is justified pooled + 3/4 corpora, a WASH on arguana)

Companion to the fusion CI above. `dense_marginal.py` reported single-run dense marginal value (hybrid − lexical);
this doc's caution flags ±0.003-scale single-run deltas as noise, and the arguana ~+0.002 was exactly there. New
harness `dense_marginal_ci.py` bootstrap-CIs the **per-query nDCG@10 delta (pool-min-max hybrid − lexical-only)**,
stem+stop + model2vec dense, top-100 pool, per corpus + POOLED (2000 resamples, seed 12345):

| corpus | q | hybrid | lexical | dense marginal (h−lex) | 95% CI | CI excludes 0 |
|---|---:|---:|---:|---:|---|---|
| scifact | 300 | 0.7120 | 0.6873 | +0.0247 | [+0.0039, +0.0465] | yes |
| nfcorpus | 323 | 0.3470 | 0.3269 | +0.0201 | [+0.0088, +0.0320] | yes |
| arguana | 1406 | 0.3601 | 0.3584 | +0.0017 | [−0.0063, +0.0098] | **no (wash)** |
| scidocs | 1000 | 0.1626 | 0.1562 | +0.0064 | [+0.0005, +0.0122] | yes (marginal) |
| **POOLED** | **3029** | — | — | **+0.0075** | **[+0.0026, +0.0122]** | **yes** |

**Verdict:** adding the dense tier to a strong stem+stop lexical tier is **statistically justified pooled (+0.0075,
CI excludes 0) and on 3/4 corpora** — but the value is MODEST and **corpus-dependent: on arguana (argument
retrieval) dense's marginal value is a bootstrap WASH** (CI [−0.0063, +0.0098] straddles 0). This CONFIRMS with
rigor the recurring single-run observation (dense's honest marginal value is small and near-zero on
argument-style corpora) and directly supports the **corpus/query-adaptive dense-gating** theme: on
argument-retrieval-style traffic the dense tier — the dominant per-query latency cost — can be skipped at no
measured quality cost. Note scidocs is decisive but only just (lower bound +0.0005). No Rust change (Python proxy
harness). Harness: `docs/quality_harness/dense_marginal_ci.py`.

## 2026-07-14 update — Python dense-reduction combination: +0.0068 pooled over RRF

Third CI in this batch. pool-min-max score fusion and the NQC dense down-weight BOTH damp the dense tier's
over-contribution — so the natural prior is they OVERLAP (sub-additive). Tested it: `stack_additivity_ci.py`
bootstrap-CIs, per query (stem+stop + model2vec, top-100 pool, 2000 resamples, seed 12345), two deltas —
**C = down-weighted-pool-min-max − RRF** (Python combined configuration vs equal-weight baseline) and
**B = down-weighted-pool-min-max − pool-min-max** (the down-weight's increment on top of pool-min-max):

| corpus | q | C = stack − RRF | C>0 | B = down-weight increment | B>0 |
|---|---:|---:|---|---:|---|
| scifact | 300 | +0.0198 [+0.0064, +0.0336] | yes | +0.0060 [−0.0012, +0.0136] | no |
| nfcorpus | 323 | +0.0142 [+0.0076, +0.0211] | yes | +0.0025 [−0.0004, +0.0052] | no |
| arguana | 1406 | +0.0015 [−0.0022, +0.0055] | no (wash) | +0.0011 [−0.0010, +0.0033] | no |
| scidocs | 1000 | +0.0078 [+0.0047, +0.0109] | yes | +0.0024 [+0.0005, +0.0044] | yes |
| **POOLED** | **3029** | **+0.0068 [+0.0042, +0.0093]** | **yes** | **+0.0022 [+0.0008, +0.0037]** | **yes** |

**Findings.** (1) **Approximately ADDITIVE, not overlapping:** A(fusion, pool-min-max−RRF = +0.0041 from the fusion
CI above) + B(down-weight = +0.0022) = **+0.0063 ≈ C(full stack) = +0.0068**. Despite both damping dense, the two
levers capture *different* slices of its over-contribution — so shipping both is justified (no cannibalization).
(2) **The Python pool-min-max + NQC configuration yields +0.0068 nDCG@10 pooled over the
RRF equal-weight baseline** (CI excludes 0, decisive on 3/4 corpora; arguana the recurring wash). (3) The
down-weight's shipped +0.0022 reproduces exactly at the pooled level (CI-decisive) but is **per-corpus mostly
BORDERLINE** (only scidocs CI-clean; scifact/nfcorpus straddle 0) — a modest, robust-in-aggregate lever, honest to
call pooled-real but not per-corpus-decisive. No Rust change (Python proxy harness).
Harness: `docs/quality_harness/stack_additivity_ci.py`.

## 2026-07-14 update — the tier ASYMMETRY, CI'd: lexical is decisive on 4/4 corpora; dense is ~4.4× smaller and a wash on arguana

Companion/mirror to the dense-marginal CI above. `lexical_marginal_ci.py` bootstrap-CIs the **per-query nDCG@10
delta (pool-min-max hybrid − dense-only)** — the lexical tier's marginal value — same protocol (stem+stop +
model2vec, top-100 pool, 2000 resamples, seed 12345):

| corpus | q | hybrid | dense-only | lexical marginal (h−dense) | 95% CI | CI excludes 0 |
|---|---:|---:|---:|---:|---|---|
| scifact | 300 | 0.7120 | 0.6331 | +0.0789 | [+0.0527, +0.1069] | yes |
| nfcorpus | 323 | 0.3461 | 0.3089 | +0.0372 | [+0.0252, +0.0489] | yes |
| arguana | 1406 | 0.3601 | 0.3328 | +0.0273 | [+0.0219, +0.0328] | yes |
| scidocs | 1000 | 0.1628 | 0.1369 | +0.0258 | [+0.0201, +0.0317] | yes |
| **POOLED** | **3029** | — | — | **+0.0330** | **[+0.0287, +0.0375]** | **yes** |

**The tier asymmetry, now rigorous (pair with the dense-marginal CI):**
- **Lexical marginal = +0.0330 pooled, CI-decisive on 4/4 corpora** (including arguana, +0.0273 decisive).
- **Dense marginal = +0.0075 pooled, CI-decisive on 3/4 — a WASH on arguana** (+0.0017, CI includes 0).

Lexical's contribution is **~4.4× larger** and universally decisive; dense's is small and corpus-dependent. So the
two tiers are NOT symmetric: **lexical is the indispensable primary tier, dense is the optional secondary tier.**
This is the rigorous, cross-corpus, CI-backed justification for the architecture the campaign already leans toward —
lexical-primary fusion with the stronger (lexical) tier up-weighted / dense down-weighted, and dense a candidate for
corpus/query-adaptive gating (skip it where its marginal value washes, e.g. argument retrieval, at the dominant
per-query latency cost). No Rust change (Python proxy harness). Harness: `docs/quality_harness/lexical_marginal_ci.py`.

## 2026-07-14 update — the pool-min-max > RRF advantage is POOL-SIZE-dependent: real at pool ≥50, a wash at pool 20

Robustness caveat on the fusion CI above (which used POOL=100). Pool size is the latency-relevant knob (smaller
candidate pool → less dense scan), so `fusion_pool_sweep_ci.py` encodes each corpus once and bootstrap-CIs the
POOLED per-query nDCG@10 delta (pool-min-max − RRF) at POOL ∈ {20, 50, 100} (stem+stop + model2vec, 3029 queries,
2000 resamples, seed 12345):

| pool | pooled Δ (mm − rrf) | 95% CI | CI excludes 0 |
|---:|---:|---|---|
| 20 | +0.0009 | [−0.0018, +0.0037] | **no (wash)** |
| 50 | +0.0038 | [+0.0014, +0.0064] | yes |
| 100 | +0.0043 | [+0.0017, +0.0070] | yes |

**Verdict:** pool-min-max's edge over RRF **only materializes at a candidate pool of ≥50** (+0.004, decisive, reproduces
the fusion-CI number) and **vanishes to a wash at pool 20** (+0.0009, CI includes 0). At small pools both fusion
methods pick essentially the same top-k — RRF's rank-only fusion loses little when there are few candidates to
discriminate, so pool-min-max's score-aware normalization has nothing to exploit. **Implication:** the "prefer
pool-min-max as the fusion default" recommendation is pool-size-conditional — worth it at the normal deep-candidate
pool, but at a small low-latency pool the fusion-method choice is immaterial (use the cheaper one). No Rust change
(Python proxy harness). Harness: `docs/quality_harness/fusion_pool_sweep_ci.py`.

## Implementation status

This table records library capabilities and their historical landing commits.
Availability of an API does not validate the historical Python effects in the
Rust implementation or select the same policy in `fsfs`:

| Lever | Status | Enable via |
|---|---|---|
| int8 fast-tier (#5) | Current `SyncTwoTierSearcher::new` selects exact retrieval; the hidden approximate int8 seam is documented for isolated benchmarks/tests only | `with_approximate_int8_fast_fetch_for_bench` |
| RRF-combine reranker (#4) | **Shipped, opt-in + wired** `235fb46`/`7ca8877` | `TwoTierSearcher::with_rerank_combine(RerankCombine::RrfCombine { k })` |
| Per-tier RRF weight (#3) | **Shipped, opt-in** `7ccda28` | `RrfConfig { semantic_weight: 1.3, .. }` (up-weight the *stronger* tier) |
| Neutral hash RRF tiebreak (#3) | **Shipped, opt-in** `05472cd` | `RrfConfig { tiebreak: RrfTiebreak::Hash, .. }` |
| RRF `k` (#3) | already configurable | `RrfConfig { k: 10.0, .. }` / `TwoTierConfig.rrf_k` |
| Deep candidate feed (#3) | already configurable | `candidate_multiplier` |
| NQC dense down-weight (2026-07-13) | Default-on in both library searchers; neutral through a 128-query warm-up. The +0.0022 nDCG observation is Python-only. | opt out `with_nqc_dense_downweight_disabled()`; explicit `with_nqc_dense_downweight(β, w_min, sketch)` overrides with a static sketch |

**Historical default-change candidates still require product evidence.** Turning
these settings on changes user-visible ranking. The line count of a default edit
does not establish its safety; actual Rust serving-path evaluation must precede
any decision or justified snapshot change:
- **#1 Embedder default → `potion-retrieval-32M` @ dim-256** (`model2vec_embedder.rs:35`): +33% English recall at
  equal cost; keep multilingual as an option. The single biggest lever — but a multilingual→English product call.
- **#4 Reranker default → `RrfCombine`**: evaluate the historical proxy's corpus-dependent effects on actual Rust retrieval candidates.
- **#3 Fusion defaults → stronger-tier weight ~1.3, `k`≈10, `Hash` tiebreak**: test per-corpus effects; strict dominance is not established for the product.

**Builder status:** both high-level library searchers expose `with_rrf_weights`
and `with_rrf_tiebreak`. The earlier builder-gap statement is obsolete. These
builders do not configure `fsfs`'s separate orchestration.

The full measurement trail, with self-corrections, is in `NEGATIVE_EVIDENCE.md`.
