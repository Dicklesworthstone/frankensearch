//! REAL-data "ratio vs Tantivy": does frankensearch's HYBRID (lexical + vector via
//! RRF) retrieve better than Tantivy-lexical-alone — the core value-add of the
//! hybrid tier — measured by label-free **known-item retrieval** on real text.
//!
//! Setup (all real): index a real English corpus in Tantivy (lexical) AND in the
//! vector index (real potion embeddings). Each query is the **first ~10 words** of a
//! held-out corpus doc (embedded separately, so it is a genuine partial-query, not a
//! trivial exact-vector match); the one relevant answer is that source doc. Measures
//! recall@10 + MRR for lexical-alone (Tantivy BM25) vs vector-alone vs hybrid (RRF).
//!
//! Run (LOCAL — needs the corpus/embedding artifacts on disk):
//! ```bash
//! FS_CORPUS_TXT=corpus.txt FS_CORPUS_SLAB=real.bin FS_QUERY_TXT=queries.txt \
//! FS_QUERY_SLAB=queries.bin FS_QUERY_IDS=query_srcids.txt FS_DIM=256 \
//!   cargo bench -p frankensearch-fusion --features tantivy-bench --bench real_hybrid_knownitem
//! ```

use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(feature = "tantivy-bench")]
#[allow(clippy::many_single_char_names)]
fn bench_real_hybrid_knownitem(c: &mut Criterion) {
    use std::hint::black_box;

    use frankensearch_core::traits::LexicalWrite;
    use frankensearch_core::types::{IndexableDocument, ScoreSource, ScoredResult};
    use frankensearch_fusion::rrf::{RrfConfig, rrf_fuse};
    use frankensearch_index::InMemoryVectorIndex;
    use frankensearch_lexical::TantivyIndex;

    const K: usize = 10;

    fn env(name: &str) -> Option<String> {
        std::env::var(name).ok()
    }
    let (Some(corpus_txt), Some(corpus_slab), Some(query_txt), Some(query_slab), Some(query_ids)) = (
        env("FS_CORPUS_TXT"),
        env("FS_CORPUS_SLAB"),
        env("FS_QUERY_TXT"),
        env("FS_QUERY_SLAB"),
        env("FS_QUERY_IDS"),
    ) else {
        eprintln!(
            "[real_hybrid_knownitem] FS_CORPUS_TXT/SLAB + FS_QUERY_TXT/SLAB/IDS unset — skipping."
        );
        return;
    };
    let dim: usize = env("FS_DIM").and_then(|s| s.parse().ok()).unwrap_or(256);

    fn read_lines(p: &str) -> Vec<String> {
        std::fs::read_to_string(p)
            .expect("read text")
            .lines()
            .map(str::to_owned)
            .collect()
    }
    fn read_slab(p: &str, dim: usize) -> Vec<Vec<f32>> {
        let bytes = std::fs::read(p).expect("read slab");
        let n = bytes.len() / (dim * 4);
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|d| {
                        let o = (i * dim + d) * 4;
                        f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
                    })
                    .collect()
            })
            .collect()
    }

    let corpus_texts = read_lines(&corpus_txt);
    let corpus_vecs = read_slab(&corpus_slab, dim);
    let n = corpus_texts.len().min(corpus_vecs.len());
    let query_texts = read_lines(&query_txt);
    let query_vecs = read_slab(&query_slab, dim);
    let query_src: Vec<usize> = read_lines(&query_ids)
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();
    let q = query_texts.len().min(query_vecs.len()).min(query_src.len());
    eprintln!("[real_hybrid_knownitem] corpus={n} docs dim={dim}  queries={q}  k={K}");

    let doc_ids: Vec<String> = (0..n).map(|i| format!("doc-{i:06}")).collect();

    // ── Vector index (real embeddings). ──
    let vindex = InMemoryVectorIndex::from_vectors(doc_ids.clone(), corpus_vecs[..n].to_vec(), dim)
        .expect("vector index");

    // ── Tantivy lexical index (real text) — async build. ──
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .build()
        .expect("runtime");
    let tantivy = TantivyIndex::in_memory().expect("tantivy in_memory");
    runtime.block_on(async {
        let cx = asupersync::Cx::for_testing();
        let docs: Vec<IndexableDocument> = (0..n)
            .map(|i| IndexableDocument {
                id: doc_ids[i].clone(),
                content: corpus_texts[i].clone(),
                title: None,
                metadata: std::collections::HashMap::default(),
            })
            .collect();
        tantivy.index_documents(&cx, &docs).await.expect("index");
        tantivy.commit(&cx).await.expect("commit");
    });

    let cx = asupersync::Cx::for_testing();
    let rank_of =
        |ids: &[String], target: &str| -> Option<usize> { ids.iter().position(|d| d == target) };
    let recall_mrr = |ids: &[String], target: &str| -> (f64, f64) {
        rank_of(ids, target).map_or((0.0, 0.0), |rank| (1.0, 1.0 / (rank as f64 + 1.0)))
    };

    // Precompute per-query lexical + vector rankings (k-independent); the hybrid is
    // re-fused per RRF-k below.
    let mut lex_all: Vec<Vec<ScoredResult>> = Vec::with_capacity(q);
    let mut vec_all: Vec<Vec<frankensearch_core::types::VectorHit>> = Vec::with_capacity(q);
    let mut targets: Vec<String> = Vec::with_capacity(q);
    let (mut rl, mut rv, mut ml, mut mv) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for qi in 0..q {
        let target = format!("doc-{:06}", query_src[qi]);
        let lex_hits = tantivy
            .search_doc_ids(&cx, &query_texts[qi], K)
            .expect("lex");
        let lex_ids: Vec<String> = lex_hits.iter().map(|h| h.doc_id.to_string()).collect();
        let lex_scored: Vec<ScoredResult> = lex_hits
            .iter()
            .map(|h| ScoredResult {
                doc_id: h.doc_id.clone(),
                score: h.bm25_score,
                source: ScoreSource::Lexical,
                index: None,
                fast_score: None,
                quality_score: None,
                lexical_score: Some(h.bm25_score),
                rerank_score: None,
                explanation: None,
                metadata: None,
            })
            .collect();
        let vec_hits = vindex.search_top_k(&query_vecs[qi], K, None).expect("vec");
        let vec_ids: Vec<String> = vec_hits.iter().map(|h| h.doc_id.to_string()).collect();
        let (a, b) = recall_mrr(&lex_ids, &target);
        rl += a;
        ml += b;
        let (a, b) = recall_mrr(&vec_ids, &target);
        rv += a;
        mv += b;
        lex_all.push(lex_scored);
        vec_all.push(vec_hits);
        targets.push(target);
    }
    let qf = q as f64;
    eprintln!(
        "[knownitem] lexical(Tantivy): recall@{K}={:.4} MRR@{K}={:.4}",
        rl / qf,
        ml / qf
    );
    eprintln!(
        "[knownitem] vector(potion):   recall@{K}={:.4} MRR@{K}={:.4}",
        rv / qf,
        mv / qf
    );

    // ── RRF-k sweep: does a sharper (lower-k) fusion keep hybrid's recall win AND
    //    recover the rank-1 MRR that the default k=60 blend dilutes below vector-alone? ──
    for k in [5.0f64, 10.0, 20.0, 30.0, 60.0, 100.0] {
        let cfg = RrfConfig {
            k,
            ..Default::default()
        };
        let (mut rh, mut mh) = (0.0f64, 0.0f64);
        for qi in 0..q {
            let fused = rrf_fuse(&lex_all[qi], &vec_all[qi], K, 0, &cfg);
            let hyb_ids: Vec<String> = fused.iter().map(|h| h.doc_id.to_string()).collect();
            let (a, b) = recall_mrr(&hyb_ids, &targets[qi]);
            rh += a;
            mh += b;
        }
        eprintln!(
            "[knownitem] hybrid RRF k={k:>5}: recall@{K}={:.4} MRR@{K}={:.4}",
            rh / qf,
            mh / qf
        );
    }

    // ── NQC dense DOWN-WEIGHT A/B (opt-in feature, end-to-end on REAL data): build the
    //    offline NQC sketch from every query's real Tantivy BM25 scores, then per query
    //    scale the dense tier's pool-min-max weight by clip(1 − beta·CDF(nqc_cv(lex)), w_min, 1).
    //    beta=0 is the equal-weight pool-min-max baseline. Known-item queries (first ~10 words
    //    of the target doc) are strongly committed-lexical → high NQC, so the thesis (dense is
    //    net-neutral/harmful on committed-lexical queries) predicts down-weighting helps/neutral.
    //    Validates the shipped nqc_cv/NqcDenseWeight/pool_minmax_fuse path vs the Python proxy. ──
    {
        use frankensearch_fusion::rrf::pool_minmax_fuse;
        use frankensearch_fusion::{NqcDenseWeight, nqc_cv};
        let lex_scores: Vec<Vec<f32>> = lex_all
            .iter()
            .map(|hits| hits.iter().map(|h| h.score).collect())
            .collect();
        let sketch = NqcDenseWeight::from_query_scores(lex_scores.iter().map(Vec::as_slice));
        let w_min = 0.1f32;
        for beta in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let (mut rh, mut mh) = (0.0f64, 0.0f64);
            for qi in 0..q {
                let cv = nqc_cv(&lex_scores[qi]);
                let sw = f64::from(sketch.dense_weight(cv, beta, w_min));
                let cfg = RrfConfig {
                    semantic_weight: sw,
                    ..Default::default()
                };
                let fused = pool_minmax_fuse(&lex_all[qi], &vec_all[qi], K, 0, &cfg);
                let ids: Vec<String> = fused.iter().map(|h| h.doc_id.to_string()).collect();
                let (a, b) = recall_mrr(&ids, &targets[qi]);
                rh += a;
                mh += b;
            }
            eprintln!(
                "[knownitem] pool-min-max NQC-downweight beta={beta:>4}: recall@{K}={:.4} MRR@{K}={:.4}",
                rh / qf,
                mh / qf
            );
        }
    }

    // ── Source-weighted RRF (manual; the knob rrf_fuse doesn't expose): up-weight the
    //    more-reliable vector tier. RRF-k is inert above, so this tests whether
    //    weighting recovers MRR ABOVE vector-alone (0.968) while keeping recall high. ──
    let k = 60.0f64;
    for wv in [1.0f64, 2.0, 4.0, 8.0, 16.0] {
        let (mut rh, mut mh) = (0.0f64, 0.0f64);
        for qi in 0..q {
            let mut score: std::collections::HashMap<String, f64> =
                std::collections::HashMap::new();
            for (r, h) in vec_all[qi].iter().enumerate() {
                *score.entry(h.doc_id.to_string()).or_insert(0.0) += wv / (k + r as f64 + 1.0);
            }
            for (r, h) in lex_all[qi].iter().enumerate() {
                *score.entry(h.doc_id.to_string()).or_insert(0.0) += 1.0 / (k + r as f64 + 1.0);
            }
            let mut ranked: Vec<(String, f64)> = score.into_iter().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            let ids: Vec<String> = ranked.iter().take(K).map(|(d, _)| d.clone()).collect();
            let (a, b) = recall_mrr(&ids, &targets[qi]);
            rh += a;
            mh += b;
        }
        eprintln!(
            "[knownitem] weighted-RRF vec_w={wv:>4} lex_w=1: recall@{K}={:.4} MRR@{K}={:.4}",
            rh / qf,
            mh / qf
        );
    }

    // ── Deterministic-tiebreak variants (equal weight): is a tiebreak-only fix (no
    //    source-weighting API) enough to recover the dominance the random tiebreak got?
    //    Compares vs production rrf_fuse (lexical-favoring tiebreak). "hash" = unbiased
    //    deterministic (FNV of doc_id); "doc_id" = ascending (deterministic, may be
    //    biased if ids correlate with relevance — they do in this bench by construction). ──
    fn fnv(s: &str) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for b in s.bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
    for tb in ["hash", "doc_id"] {
        let (mut rh, mut mh) = (0.0f64, 0.0f64);
        for qi in 0..q {
            let mut score: std::collections::HashMap<String, f64> =
                std::collections::HashMap::new();
            for (r, h) in vec_all[qi].iter().enumerate() {
                *score.entry(h.doc_id.to_string()).or_insert(0.0) += 1.0 / (60.0 + r as f64 + 1.0);
            }
            for (r, h) in lex_all[qi].iter().enumerate() {
                *score.entry(h.doc_id.to_string()).or_insert(0.0) += 1.0 / (60.0 + r as f64 + 1.0);
            }
            let mut ranked: Vec<(String, f64)> = score.into_iter().collect();
            ranked.sort_by(|a, b| {
                b.1.total_cmp(&a.1).then_with(|| match tb {
                    "hash" => fnv(&a.0).cmp(&fnv(&b.0)),
                    _ => a.0.cmp(&b.0),
                })
            });
            let ids: Vec<String> = ranked.iter().take(K).map(|(d, _)| d.clone()).collect();
            let (a, b) = recall_mrr(&ids, &targets[qi]);
            rh += a;
            mh += b;
        }
        eprintln!(
            "[knownitem] equal-RRF tiebreak={tb:>6}: recall@{K}={:.4} MRR@{K}={:.4}",
            rh / qf,
            mh / qf
        );
    }

    // ── Latency: lexical vs vector vs hybrid per query. ──
    let mut qi = 0usize;
    let mut g = c.benchmark_group("real_hybrid_knownitem");
    g.bench_function("lexical", |b| {
        b.iter(|| {
            let i = qi % q;
            qi += 1;
            black_box(
                tantivy
                    .search_doc_ids(&cx, black_box(&query_texts[i]), K)
                    .expect("lex"),
            )
        });
    });
    g.bench_function("vector", |b| {
        b.iter(|| {
            let i = qi % q;
            qi += 1;
            black_box(
                vindex
                    .search_top_k(black_box(&query_vecs[i]), K, None)
                    .expect("vec"),
            )
        });
    });
    g.finish();
}

#[cfg(not(feature = "tantivy-bench"))]
fn bench_real_hybrid_knownitem(_c: &mut Criterion) {
    // Tantivy lives behind a bench-only feature; build with `--features tantivy-bench`.
}

criterion_group!(benches, bench_real_hybrid_knownitem);
criterion_main!(benches);
