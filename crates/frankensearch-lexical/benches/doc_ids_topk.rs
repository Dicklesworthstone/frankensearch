//! Per-crate A/B for the count-free `search_doc_ids` top-k path.
//!
//! `search_doc_ids` only needs the ranked top-k ids, never a total match count,
//! so it skips Tantivy's `Count` collector. This bench measures that path
//! against the counted baseline on a 100k-doc in-memory index. Both arms return
//! identical ranked ids; only the discarded count differs.

use std::collections::HashMap;
use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use frankensearch_core::{traits::LexicalWrite, types::IndexableDocument};
use frankensearch_lexical::TantivyIndex;

const N: usize = 100_000;
const K: usize = 10;

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
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
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
    let mut content = String::with_capacity(128);
    let vocab_len = u64::try_from(VOCAB.len()).unwrap_or(1);
    for _ in 0..12 {
        let word_index = usize::try_from(xorshift(&mut state) % vocab_len).unwrap_or(0);
        let word = VOCAB.get(word_index).copied().unwrap_or("alpha");
        content.push_str(word);
        content.push(' ');
    }
    if i % 6 == 0 {
        content.push_str("rust ownership borrowing lifetimes ");
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
    let index_to_build = Arc::clone(&index);
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .build()
        .expect("build benchmark runtime");
    runtime.block_on(async move {
        let cx = asupersync::Cx::for_testing();
        index_to_build
            .index_documents(&cx, &docs)
            .await
            .expect("index documents");
        index_to_build.commit(&cx).await.expect("commit");
    });
    index
}

fn bench_doc_ids_topk(c: &mut Criterion) {
    let index = build_index();
    let cx = asupersync::Cx::for_testing();

    let workloads: &[(&str, &str)] = &[
        ("high_fanout", "alpha"),
        ("short_keyword_bold", "rust ownership"),
        ("union3", "alpha beta gamma"),
        ("natural", "search engine vector ranking relevance"),
        ("phrase", "\"search engine\""),
    ];

    let mut group = c.benchmark_group("doc_ids_topk");
    for (label, query) in workloads {
        group.bench_function(format!("{label}/counted"), |bencher| {
            bencher.iter(|| {
                black_box(
                    index
                        .search_doc_ids_counted(&cx, black_box(query), K)
                        .expect("counted search"),
                )
            });
        });
        group.bench_function(format!("{label}/free"), |bencher| {
            bencher.iter(|| {
                black_box(
                    index
                        .search_doc_ids(&cx, black_box(query), K)
                        .expect("count-free search"),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_doc_ids_topk);
criterion_main!(benches);
