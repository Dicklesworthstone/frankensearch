//! Validates that a REOPENED on-disk index realizes the fast id-materialization
//! path via its persisted `ordinal→doc_id` sidecar (written on commit, loaded on
//! open). Builds an on-disk index, commits (persists the sidecar), drops it, and
//! reopens from the same directory — then A/Bs, on that reopened handle:
//!
//! - `fast`     : `search_doc_ids` (sidecar-restored ord table).
//! - `docstore` : `search_doc_ids_via_docstore` (forced per-hit decompress).
//!
//! Both return identical ranked ids (asserted before timing). Run with:
//! ```bash
//! CARGO_TARGET_DIR=/data/projects/.rch-targets/frankensearch-cc \
//!   rch exec -- cargo bench -p frankensearch-lexical --profile release \
//!   --bench reopen_id_materialize
//! ```

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::traits::LexicalWrite;
use frankensearch_core::types::IndexableDocument;
use frankensearch_lexical::TantivyIndex;

const N: usize = 20_000;
const KS: &[usize] = &[10, 1000];

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
    content.push_str("document ");
    let vlen = u64::try_from(VOCAB.len()).unwrap_or(1);
    for _ in 0..12 {
        content.push_str(VOCAB[usize::try_from(xorshift(&mut state) % vlen).unwrap_or(0)]);
        content.push(' ');
    }
    let id = format!("doc-{i:06}");
    content.push_str(&id);
    IndexableDocument {
        id,
        content,
        title: None,
        metadata: std::collections::HashMap::new(),
    }
}

fn build_and_reopen(dir: &std::path::Path) -> Arc<TantivyIndex> {
    let rt = asupersync::runtime::RuntimeBuilder::current_thread()
        .build()
        .expect("runtime");
    // Build on-disk, commit (persists the ord_table sidecar), then drop.
    {
        let idx = TantivyIndex::create(dir).expect("create on-disk");
        let docs: Vec<IndexableDocument> = (0..N).map(make_doc).collect();
        rt.block_on(async {
            let cx = asupersync::Cx::for_testing();
            idx.index_documents(&cx, &docs).await.expect("index");
            idx.commit(&cx).await.expect("commit");
        });
    }
    // Reopen: the sidecar restores the fast materialization path.
    Arc::new(TantivyIndex::open(dir).expect("reopen"))
}

fn bench(c: &mut Criterion) {
    let dir = tempfile::tempdir().expect("tempdir");
    let index = build_and_reopen(dir.path());
    let cx = asupersync::Cx::for_testing();
    let query = "document";

    let max_k = *KS.iter().max().unwrap();
    let fast = index.search_doc_ids(&cx, query, max_k).expect("fast");
    let docs = index
        .search_doc_ids_via_docstore(&cx, query, max_k)
        .expect("docstore");
    assert_eq!(fast, docs, "reopened fast and docstore doc_ids must match");
    assert!(!fast.is_empty(), "reopened index should return hits");

    let mut g = c.benchmark_group("reopen_id_materialize");
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
