//! End-to-end A/B for the wired `ord` numeric-fast-field id materialization in
//! `search_doc_ids` (BM25 search + id materialization), same binary:
//!
//! - `fast`     : `search_doc_ids` (ord u64 FAST column + ordinal table).
//! - `docstore` : `search_doc_ids_via_docstore` (forces the old per-hit
//!   `searcher.doc(addr)` decompress) — the pre-wiring baseline.
//!
//! Both return identical ranked ids (asserted before timing). The win scales
//! with `limit`: at the BOLD top10 fetch (~30 candidates) materialization is a
//! small slice of BM25 search, but at large limits it dominates. Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/search-cc \
//!   rch exec -- cargo bench -p frankensearch-lexical --profile release \
//!   --bench search_doc_ids_materialize
//! ```

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::traits::LexicalWrite;
use frankensearch_core::types::IndexableDocument;
use frankensearch_lexical::TantivyIndex;

const N: usize = 100_000;
const KS: &[usize] = &[10, 100, 1000];

const VOCAB: &[&str] = &[
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "search",
    "engine",
    "vector",
    "lexical",
    "ranking",
    "relevance",
    "document",
    "query",
];

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn make_doc(i: usize) -> IndexableDocument {
    let mut state = u64::try_from(i).unwrap_or(u64::MAX - 1).saturating_add(1);
    let mut content = String::with_capacity(160);
    // High-fanout term in every doc so large-limit fetches are exercised.
    content.push_str("search ");
    let vocab_len = u64::try_from(VOCAB.len()).unwrap_or(1);
    for _ in 0..13 {
        let w = VOCAB[usize::try_from(xorshift(&mut state) % vocab_len).unwrap_or(0)];
        content.push_str(w);
        content.push(' ');
    }
    let id = format!("doc-{i:06}");
    content.push_str(&id);
    IndexableDocument {
        id,
        content,
        title: None,
        metadata: HashMap::new(),
    }
}

fn build_index() -> Arc<TantivyIndex> {
    let index = Arc::new(TantivyIndex::in_memory().expect("create in-memory tantivy index"));
    let docs: Vec<IndexableDocument> = (0..N).map(make_doc).collect();
    let idx = Arc::clone(&index);
    let rt = asupersync::runtime::RuntimeBuilder::current_thread()
        .build()
        .expect("runtime");
    rt.block_on(async move {
        let cx = asupersync::Cx::for_testing();
        idx.index_documents(&cx, &docs).await.expect("index");
        idx.commit(&cx).await.expect("commit");
    });
    index
}

fn bench(c: &mut Criterion) {
    let index = build_index();
    let cx = asupersync::Cx::for_testing();
    let query = "search";

    // Prove the two materialization paths agree at the largest k.
    let max_k = *KS.iter().max().unwrap();
    let fast = index.search_doc_ids(&cx, query, max_k).expect("fast");
    let docs = index
        .search_doc_ids_via_docstore(&cx, query, max_k)
        .expect("docstore");
    assert_eq!(fast, docs, "fast and docstore doc_ids must match");

    let mut g = c.benchmark_group("search_doc_ids_materialize");
    for &k in KS {
        g.bench_function(format!("docstore/k{k}"), |b| {
            b.iter(|| {
                black_box(
                    index
                        .search_doc_ids_via_docstore(&cx, black_box(query), k)
                        .expect("docstore"),
                )
            });
        });
        g.bench_function(format!("fast/k{k}"), |b| {
            b.iter(|| {
                black_box(
                    index
                        .search_doc_ids(&cx, black_box(query), k)
                        .expect("fast"),
                )
            });
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
