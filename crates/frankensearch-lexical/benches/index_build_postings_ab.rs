#![allow(clippy::cast_precision_loss)]

//! Production-path index-build A/B for the generic Tantivy writer budget.
//!
//! Tantivy requires at least 15 MB per indexing worker. The shipping 50 MB writer budget therefore
//! caps a large host at three workers; 100 MB admits six while keeping nearly the same per-worker
//! arena. Both arms use the exact frankensearch schema, tokenizers, upsert path, commit path, and
//! BM25 search path. Ranked IDs and score bits are checked before the timing verdict.
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
//!   cargo bench -p frankensearch-lexical --features bench-internals \
//!   --bench index_build_postings_ab -- --noplot
//! ```

use std::collections::{HashMap, HashSet};
use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use frankensearch_core::bench_support::paired_median_ratio;
use frankensearch_core::traits::LexicalWrite;
use frankensearch_core::types::IndexableDocument;
use frankensearch_lexical::TantivyIndex;

const DOCS: usize = 20_000;
const PARITY_K: usize = 64;
const ORIGINAL_WRITER_HEAP_BYTES: usize = 50_000_000;
const CANDIDATE_WRITER_HEAP_BYTES: usize = 100_000_000;

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

#[derive(Clone, Copy)]
struct BuildPhases {
    create: Duration,
    enqueue: Duration,
    commit: Duration,
    total: Duration,
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
            let mut content = String::with_capacity(512);
            let word_count = 36 + (i % 29);
            let vocab_len = u64::try_from(VOCAB.len()).unwrap_or(1);
            for _ in 0..word_count {
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

            let mut metadata = HashMap::new();
            metadata.insert("shard".to_owned(), (i % 17).to_string());
            IndexableDocument {
                id: format!("doc-{i:06}"),
                content,
                title: (i % 5 == 0).then(|| format!("Rust search result {i}")),
                metadata,
            }
        })
        .collect()
}

fn build_index(
    runtime: &asupersync::runtime::Runtime,
    docs: &[IndexableDocument],
    writer_heap_bytes: usize,
) -> (TantivyIndex, BuildPhases) {
    let total_start = Instant::now();
    let create_start = Instant::now();
    let index = TantivyIndex::in_memory_with_writer_heap_bytes(writer_heap_bytes)
        .expect("create benchmark index");
    let create = create_start.elapsed();

    let mut enqueue = Duration::ZERO;
    let mut commit = Duration::ZERO;
    runtime.block_on(async {
        let cx = asupersync::Cx::for_testing();
        let enqueue_start = Instant::now();
        index
            .index_documents(&cx, docs)
            .await
            .expect("index benchmark documents");
        enqueue = enqueue_start.elapsed();

        let commit_start = Instant::now();
        index.commit(&cx).await.expect("commit benchmark index");
        commit = commit_start.elapsed();
    });

    let total = total_start.elapsed();
    (
        index,
        BuildPhases {
            create,
            enqueue,
            commit,
            total,
        },
    )
}

fn snapshot(index: &TantivyIndex, query: &str) -> Vec<(String, u32)> {
    let cx = asupersync::Cx::for_testing();
    index
        .search_doc_ids(&cx, query, PARITY_K)
        .expect("run BM25 parity query")
        .into_iter()
        .map(|hit| (hit.doc_id.to_string(), hit.bm25_score.to_bits()))
        .collect()
}

fn quality_against_original(original: &[(String, u32)], candidate: &[(String, u32)]) -> (f64, f64) {
    let original_ids: HashSet<&str> = original.iter().map(|(id, _)| id.as_str()).collect();
    let recalled = candidate
        .iter()
        .filter(|(id, _)| original_ids.contains(id.as_str()))
        .count();
    let recall = recalled as f64 / original.len().max(1) as f64;

    let original_rank: HashMap<&str, usize> = original
        .iter()
        .enumerate()
        .map(|(rank, (id, _))| (id.as_str(), rank))
        .collect();
    let discount = |rank: usize| 1.0 / (rank as f64 + 2.0).log2();
    let dcg: f64 = candidate
        .iter()
        .enumerate()
        .filter_map(|(candidate_rank, (id, _))| {
            original_rank
                .get(id.as_str())
                .map(|original_rank| (PARITY_K - *original_rank) as f64 * discount(candidate_rank))
        })
        .sum();
    let ideal: f64 = (0..original.len())
        .map(|rank| (PARITY_K - rank) as f64 * discount(rank))
        .sum();
    (recall, if ideal > 0.0 { dcg / ideal } else { 1.0 })
}

fn median_duration(values: &mut [Duration]) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn profile_original(runtime: &asupersync::runtime::Runtime, docs: &[IndexableDocument]) {
    let mut creates = Vec::with_capacity(9);
    let mut enqueues = Vec::with_capacity(9);
    let mut commits = Vec::with_capacity(9);
    let mut totals = Vec::with_capacity(9);
    for _ in 0..9 {
        let (index, phases) = build_index(runtime, docs, ORIGINAL_WRITER_HEAP_BYTES);
        creates.push(phases.create);
        enqueues.push(phases.enqueue);
        commits.push(phases.commit);
        totals.push(phases.total);
        black_box(index);
    }
    eprintln!(
        "[profile] ORIG median ms: create={:.3} enqueue={:.3} commit={:.3} total={:.3}",
        median_duration(&mut creates).as_secs_f64() * 1_000.0,
        median_duration(&mut enqueues).as_secs_f64() * 1_000.0,
        median_duration(&mut commits).as_secs_f64() * 1_000.0,
        median_duration(&mut totals).as_secs_f64() * 1_000.0,
    );
}

fn run_build_and_drop(
    runtime: &asupersync::runtime::Runtime,
    docs: &[IndexableDocument],
    writer_heap_bytes: usize,
) {
    let (index, phases) = build_index(runtime, docs, writer_heap_bytes);
    black_box(phases.total);
    black_box(index);
}

fn bench(c: &mut Criterion) {
    let docs = make_documents();
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .build()
        .expect("build benchmark runtime");

    profile_original(&runtime, &docs);

    let (original, _) = build_index(&runtime, &docs, ORIGINAL_WRITER_HEAP_BYTES);
    let (candidate, _) = build_index(&runtime, &docs, CANDIDATE_WRITER_HEAP_BYTES);
    let queries = [
        "alpha",
        "rust ownership",
        "alpha beta gamma",
        "\"search engine\"",
        "title:rust",
        "alpha OR beta",
    ];
    let mut exact_ids = true;
    let mut exact_scores = true;
    let mut min_recall = 1.0f64;
    let mut min_ndcg = 1.0f64;
    for query in queries {
        let original_hits = snapshot(&original, query);
        let candidate_hits = snapshot(&candidate, query);
        exact_ids &= original_hits
            .iter()
            .map(|(id, _)| id)
            .eq(candidate_hits.iter().map(|(id, _)| id));
        exact_scores &= original_hits
            .iter()
            .map(|(_, score)| score)
            .eq(candidate_hits.iter().map(|(_, score)| score));
        let (recall, ndcg) = quality_against_original(&original_hits, &candidate_hits);
        min_recall = min_recall.min(recall);
        min_ndcg = min_ndcg.min(ndcg);
    }

    let (original_segments, original_bytes) = original
        .benchmark_index_layout()
        .expect("read original index layout");
    let (candidate_segments, candidate_bytes) = candidate
        .benchmark_index_layout()
        .expect("read candidate index layout");
    let footprint_ratio = candidate_bytes as f64 / original_bytes.max(1) as f64;
    eprintln!(
        "[parity] exact_ids={exact_ids} exact_scores={exact_scores} recall@{PARITY_K}={min_recall:.6} nDCG@{PARITY_K}={min_ndcg:.6}"
    );
    eprintln!(
        "[footprint] orig_segments={original_segments} candidate_segments={candidate_segments} orig_bytes={original_bytes} candidate_bytes={candidate_bytes} ratio={footprint_ratio:.6}"
    );

    let null = paired_median_ratio(
        21,
        1,
        || run_build_and_drop(&runtime, &docs, ORIGINAL_WRITER_HEAP_BYTES),
        || run_build_and_drop(&runtime, &docs, ORIGINAL_WRITER_HEAP_BYTES),
    );
    let lever = paired_median_ratio(
        21,
        1,
        || run_build_and_drop(&runtime, &docs, ORIGINAL_WRITER_HEAP_BYTES),
        || run_build_and_drop(&runtime, &docs, CANDIDATE_WRITER_HEAP_BYTES),
    );
    eprintln!(
        "[null] index_build/{DOCS}: median {:.4} p5 {:.4} p95 {:.4} ({} rounds)",
        null.median, null.p5, null.p95, null.rounds
    );
    eprintln!(
        "[lever] index_build/{DOCS}: candidate/ORIG median {:.4} p5 {:.4} p95 {:.4} -> {}",
        lever.median,
        lever.p5,
        lever.p95,
        if lever.decidable_against(&null) {
            if lever.median < 1.0 {
                "DECIDABLE WIN"
            } else {
                "DECIDABLE REGRESSION"
            }
        } else {
            "INSIDE NULL FLOOR"
        }
    );

    let mut group = c.benchmark_group("index_build_postings");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(1));
    group.throughput(Throughput::Elements(DOCS as u64));
    group.bench_function("orig_50mb", |b| {
        b.iter(|| run_build_and_drop(&runtime, &docs, ORIGINAL_WRITER_HEAP_BYTES));
    });
    group.bench_function("candidate_100mb", |b| {
        b.iter(|| run_build_and_drop(&runtime, &docs, CANDIDATE_WRITER_HEAP_BYTES));
    });
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
