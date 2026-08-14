#![allow(clippy::cast_precision_loss)]

//! Profile and A/B harness for facade `IndexBuilder` lexical staging.
//!
//! The baseline profile measures the shipping facade path, including vector writes and the
//! subsequent Quill postings build, while excluding corpus construction and temporary-directory
//! creation. It separately measures the deep `IndexableDocument` staging clone so the candidate is
//! selected from observed cost rather than static inspection.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
//!   cargo bench -p frankensearch --profile release --features lexical,bench-internals \
//!   --bench index_builder_lexical_staging -- --noplot
//! ```

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use asupersync::Cx;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use frankensearch::{EmbedderStack, IndexBuilder, QuillConfig, QuillSearchIndex};
use frankensearch_core::traits::{Embedder, ModelCategory, SearchFuture};
use frankensearch_core::types::IndexableDocument;
use tempfile::TempDir;

const DOCS: usize = 20_000;
const PROFILE_ROUNDS: usize = 9;
const PAIRED_ROUNDS: usize = 21;
const DIMENSION: usize = 8;
const PARITY_K: usize = DOCS;

const VOCAB: &[&str] = &[
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "search",
    "engine",
    "vector",
    "lexical",
    "ranking",
    "relevance",
    "document",
    "query",
    "rust",
    "ownership",
    "borrowing",
    "lifetimes",
];

struct FixedEmbedder;

impl Embedder for FixedEmbedder {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            let mut vector = vec![0.0; DIMENSION];
            vector[text.len() % DIMENSION] = 1.0;
            Ok(vector)
        })
    }

    fn dimension(&self) -> usize {
        DIMENSION
    }

    fn id(&self) -> &'static str {
        "index-builder-staging-fixed-8"
    }

    fn model_name(&self) -> &'static str {
        "index-builder-staging-fixed-8"
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }
}

#[derive(Clone, Copy)]
struct BuildPhases {
    wall: Duration,
    reported_total: Duration,
    embed_and_stage: Duration,
}

#[derive(Clone, Copy)]
enum StagingMode {
    OriginalClone,
    CandidateMove,
}

#[derive(Clone, Copy)]
struct RatioSummary {
    median: f64,
    p5: f64,
    p95: f64,
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn make_documents() -> Vec<IndexableDocument> {
    (0..DOCS)
        .map(|i| {
            let mut state = u64::try_from(i).unwrap_or(u64::MAX - 1).saturating_add(1);
            let mut content = String::with_capacity(768);
            let vocab_len = u64::try_from(VOCAB.len()).unwrap_or(1);
            for _ in 0..(72 + i % 31) {
                let word_index = usize::try_from(xorshift(&mut state) % vocab_len).unwrap_or(0);
                content.push_str(VOCAB.get(word_index).copied().unwrap_or("alpha"));
                content.push(' ');
            }
            for _ in 0..(i % 13) {
                content.push_str("alpha ");
            }
            if i % 7 == 0 {
                content.push_str("rust ownership borrowing lifetimes search engine ");
            }

            IndexableDocument::new(format!("doc-{i:06}"), content)
                .with_title(format!("Rust search result {i}"))
                .with_metadata("shard", (i % 17).to_string())
                .with_metadata("cluster", (i % 7).to_string())
                .with_metadata("language", "rust")
                .with_metadata("source", format!("fixture-{}", i % 11))
        })
        .collect()
}

fn embedder_stack() -> EmbedderStack {
    EmbedderStack::from_parts(Arc::new(FixedEmbedder), None)
}

fn build_index(
    runtime: &asupersync::runtime::Runtime,
    documents: Vec<IndexableDocument>,
    mode: StagingMode,
) -> (TempDir, BuildPhases) {
    let dir = TempDir::new().expect("create index-builder benchmark directory");
    let mut builder = IndexBuilder::new(dir.path())
        .with_embedder_stack(embedder_stack())
        .add_documents(documents);
    if matches!(mode, StagingMode::OriginalClone) {
        builder = builder.with_clone_lexical_staging_for_benchmark();
    }
    let start = Instant::now();
    let stats = runtime.block_on(async {
        let cx = Cx::for_testing();
        builder.build(&cx).await.expect("build facade index")
    });
    let wall = start.elapsed();
    assert_eq!(stats.doc_count, DOCS);
    assert_eq!(stats.error_count, 0);
    (
        dir,
        BuildPhases {
            wall,
            reported_total: Duration::from_secs_f64(stats.total_ms / 1_000.0),
            embed_and_stage: Duration::from_secs_f64(stats.embed_ms / 1_000.0),
        },
    )
}

fn ranked_snapshot(
    runtime: &asupersync::runtime::Runtime,
    dir: &TempDir,
) -> Vec<Vec<(String, u32)>> {
    let index = runtime.block_on(async {
        let cx = Cx::for_testing();
        QuillSearchIndex::open(&cx, dir.path().join("lexical"), QuillConfig::default())
            .await
            .expect("open Quill lexical benchmark index")
    });
    let cx = Cx::for_testing();
    [
        "alpha",
        "rust ownership",
        "alpha beta gamma",
        "\"search engine\"",
        "title:rust",
        "alpha OR beta",
    ]
    .iter()
    .map(|query| {
        index
            .search_doc_ids(&cx, query, PARITY_K)
            .expect("run Quill profile query")
            .iter()
            .map(|hit| (hit.document_id.clone(), hit.score.to_bits()))
            .collect()
    })
    .collect()
}

fn median_duration(values: &mut [Duration]) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn ratio_summary(mut ratios: Vec<f64>) -> RatioSummary {
    ratios.sort_unstable_by(f64::total_cmp);
    let last = ratios.len() - 1;
    let percentile = |numerator: usize| ratios[(last * numerator + 50) / 100];
    RatioSummary {
        median: percentile(50),
        p5: percentile(5),
        p95: percentile(95),
    }
}

fn measure_build(
    runtime: &asupersync::runtime::Runtime,
    documents: &[IndexableDocument],
    mode: StagingMode,
) -> Duration {
    // Corpus ownership transfer and temporary-directory creation are setup, not part of the facade
    // build timer. Both arms receive identical owned values and differ only in the internal staging
    // choice selected above.
    let owned = documents.to_vec();
    let (dir, phases) = build_index(runtime, owned, mode);
    black_box(dir);
    phases.wall
}

fn paired_build_ratio(
    runtime: &asupersync::runtime::Runtime,
    documents: &[IndexableDocument],
    base: StagingMode,
    candidate: StagingMode,
) -> RatioSummary {
    for _ in 0..2 {
        black_box(measure_build(runtime, documents, base));
        black_box(measure_build(runtime, documents, candidate));
    }

    let mut ratios = Vec::with_capacity(PAIRED_ROUNDS);
    for round in 0..PAIRED_ROUNDS {
        let (base_time, candidate_time) = if round % 2 == 0 {
            (
                measure_build(runtime, documents, base),
                measure_build(runtime, documents, candidate),
            )
        } else {
            let candidate_time = measure_build(runtime, documents, candidate);
            let base_time = measure_build(runtime, documents, base);
            (base_time, candidate_time)
        };
        ratios.push(candidate_time.as_secs_f64() / base_time.as_secs_f64());
    }
    ratio_summary(ratios)
}

fn quality_against_original(original: &[(String, u32)], candidate: &[(String, u32)]) -> (f64, f64) {
    let original_rank: HashMap<&str, usize> = original
        .iter()
        .enumerate()
        .map(|(rank, (id, _))| (id.as_str(), rank))
        .collect();
    let recalled = candidate
        .iter()
        .filter(|(candidate_id, _)| original_rank.contains_key(candidate_id.as_str()))
        .count();
    let recall = recalled as f64 / original.len().max(1) as f64;
    let discount = |rank: usize| 1.0 / (rank as f64 + 2.0).log2();
    let dcg: f64 = candidate
        .iter()
        .enumerate()
        .filter_map(|(candidate_rank, (candidate_id, _))| {
            original_rank
                .get(candidate_id.as_str())
                .map(|original_rank| {
                    (original.len() - *original_rank) as f64 * discount(candidate_rank)
                })
        })
        .sum();
    let ideal: f64 = (0..original.len())
        .map(|rank| (original.len() - rank) as f64 * discount(rank))
        .sum();
    (recall, if ideal > 0.0 { dcg / ideal } else { 1.0 })
}

fn canonicalize_ties(hits: &[(String, u32)]) -> Vec<(String, u32)> {
    let mut canonical = hits.to_vec();
    canonical.sort_unstable_by(|(left_id, left_score), (right_id, right_score)| {
        f32::from_bits(*right_score)
            .total_cmp(&f32::from_bits(*left_score))
            .then_with(|| left_id.cmp(right_id))
    });
    canonical
}

fn profile_original(runtime: &asupersync::runtime::Runtime, documents: &[IndexableDocument]) {
    let mut clone_times = Vec::with_capacity(PROFILE_ROUNDS);
    for _ in 0..PROFILE_ROUNDS {
        let start = Instant::now();
        let cloned = documents.to_vec();
        clone_times.push(start.elapsed());
        black_box(cloned);
    }

    let mut walls = Vec::with_capacity(PROFILE_ROUNDS);
    let mut reported_totals = Vec::with_capacity(PROFILE_ROUNDS);
    let mut embed_and_stage = Vec::with_capacity(PROFILE_ROUNDS);
    let mut post_stage = Vec::with_capacity(PROFILE_ROUNDS);
    let mut first_snapshot = None;
    for round in 0..PROFILE_ROUNDS {
        let (dir, phases) = build_index(runtime, documents.to_vec(), StagingMode::OriginalClone);
        walls.push(phases.wall);
        reported_totals.push(phases.reported_total);
        embed_and_stage.push(phases.embed_and_stage);
        post_stage.push(phases.reported_total.saturating_sub(phases.embed_and_stage));
        if round == 0 {
            first_snapshot = Some(ranked_snapshot(runtime, &dir));
        }
        black_box(dir);
    }

    let clone_median = median_duration(&mut clone_times);
    let wall_median = median_duration(&mut walls);
    let reported_median = median_duration(&mut reported_totals);
    let embed_stage_median = median_duration(&mut embed_and_stage);
    let post_stage_median = median_duration(&mut post_stage);
    eprintln!(
        "[profile] ORIG median ms: deep_clone={:.3} embed_vector_and_stage={:.3} post_stage={:.3} reported_total={:.3} wall={:.3} clone_pct_wall={:.3}",
        clone_median.as_secs_f64() * 1_000.0,
        embed_stage_median.as_secs_f64() * 1_000.0,
        post_stage_median.as_secs_f64() * 1_000.0,
        reported_median.as_secs_f64() * 1_000.0,
        wall_median.as_secs_f64() * 1_000.0,
        clone_median.as_secs_f64() / wall_median.as_secs_f64().max(f64::EPSILON) * 100.0,
    );
    let snapshot = first_snapshot.expect("capture original BM25 snapshot");
    eprintln!(
        "[profile] ORIG BM25 snapshot: queries={} total_hits={} score_checksum={}",
        snapshot.len(),
        snapshot.iter().map(Vec::len).sum::<usize>(),
        snapshot
            .iter()
            .flatten()
            .fold(0u64, |acc, (_, score)| acc.wrapping_add(u64::from(*score))),
    );
}

fn verify_parity_and_measure(
    runtime: &asupersync::runtime::Runtime,
    documents: &[IndexableDocument],
) {
    let (original_dir, _) = build_index(runtime, documents.to_vec(), StagingMode::OriginalClone);
    let (candidate_dir, _) = build_index(runtime, documents.to_vec(), StagingMode::CandidateMove);
    let original = ranked_snapshot(runtime, &original_dir);
    let candidate = ranked_snapshot(runtime, &candidate_dir);
    let mut exact_raw_order = true;
    let mut exact_tie_aware_rank_ids = true;
    let mut exact_score_bits = true;
    let mut min_recall = 1.0f64;
    let mut min_ndcg = 1.0f64;
    for (original_hits, candidate_hits) in original.iter().zip(&candidate) {
        exact_raw_order &= original_hits
            .iter()
            .map(|(id, _)| id)
            .eq(candidate_hits.iter().map(|(id, _)| id));
        let canonical_original = canonicalize_ties(original_hits);
        let canonical_candidate = canonicalize_ties(candidate_hits);
        exact_tie_aware_rank_ids &= canonical_original
            .iter()
            .map(|(id, _)| id)
            .eq(canonical_candidate.iter().map(|(id, _)| id));
        exact_score_bits &= canonical_original == canonical_candidate;
        let (recall, ndcg) = quality_against_original(&canonical_original, &canonical_candidate);
        min_recall = min_recall.min(recall);
        min_ndcg = min_ndcg.min(ndcg);
    }
    eprintln!(
        "[parity] exact_raw_order={exact_raw_order} exact_tie_aware_rank_ids={exact_tie_aware_rank_ids} exact_score_bits={exact_score_bits} recall@{PARITY_K}={min_recall:.6} nDCG@{PARITY_K}={min_ndcg:.6}"
    );
    assert_eq!(
        min_recall.to_bits(),
        1.0_f64.to_bits(),
        "candidate changed recall"
    );
    assert!(
        min_ndcg >= 0.999_999,
        "candidate changed ranked relevance: nDCG={min_ndcg}"
    );

    let null = paired_build_ratio(
        runtime,
        documents,
        StagingMode::OriginalClone,
        StagingMode::OriginalClone,
    );
    let lever = paired_build_ratio(
        runtime,
        documents,
        StagingMode::OriginalClone,
        StagingMode::CandidateMove,
    );
    let decidable = lever.median < null.p5 || lever.median > null.p95;
    eprintln!(
        "[null] index_builder/{DOCS}: median {:.4} p5 {:.4} p95 {:.4} ({PAIRED_ROUNDS} rounds)",
        null.median, null.p5, null.p95,
    );
    eprintln!(
        "[lever] index_builder/{DOCS}: candidate/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
        lever.median,
        lever.p5,
        lever.p95,
        if decidable {
            if lever.median < 1.0 {
                "DECIDABLE WIN"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR"
        },
    );
}

fn bench(c: &mut Criterion) {
    let documents = make_documents();
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .build()
        .expect("build benchmark runtime");

    profile_original(&runtime, &documents);
    verify_parity_and_measure(&runtime, &documents);

    let mut group = c.benchmark_group("index_builder_lexical_staging");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(1));
    group.throughput(Throughput::Elements(DOCS as u64));
    group.bench_function("original_clone", |b| {
        b.iter_batched(
            || documents.clone(),
            |owned| {
                let (dir, phases) = build_index(&runtime, owned, StagingMode::OriginalClone);
                black_box(phases.wall);
                black_box(dir);
            },
            BatchSize::PerIteration,
        );
    });
    group.bench_function("candidate_move", |b| {
        b.iter_batched(
            || documents.clone(),
            |owned| {
                let (dir, phases) = build_index(&runtime, owned, StagingMode::CandidateMove);
                black_box(phases.wall);
                black_box(dir);
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
