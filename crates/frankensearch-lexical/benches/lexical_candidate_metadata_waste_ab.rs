//! Quantify and gate deferred metadata work on the HYBRID fusion path.
//!
//! The first gate compares full candidate materialization with ordinal-fast-field
//! candidate acquisition followed by winner-only metadata hydration. The second gate
//! isolates that exact-ID hydration step: the retained baseline asks `TopDocs` to score
//! and heap-sort the matching IDs, while production uses an unscored document-set
//! collector because hydration consumes neither score nor hit order.
//!
//! This measures three per-query materialization shapes over the top-k hits of a
//! metadata-bearing index, isolating (a) the metadata deserialize share and (b) the
//! total win of the id-only fast path over the metadata-bearing docstore path:
//!   A `docstore_meta` : docstore doc → id String + `from_str(metadata)` → Value  (what fusion pays now)
//!   B `docstore_id`   : docstore doc → id String only                            (isolates metadata cost = A − B)
//!   C `fastfield_id`  : ord fast field → `table[ord]` clone, metadata dropped     (what the hybrid path could pay)
//!
//! ```bash
//! RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
//!   cargo bench -p frankensearch-lexical --profile release \
//!   --features bench-internals --bench lexical_candidate_metadata_waste_ab
//! ```
//!
//! Set `META_COLLECTOR_ONLY=1` to run only the exact-ID collector gate.

use std::hint::black_box;

use frankensearch_core::bench_support::{PairedRatio, paired_median_ratio};
use frankensearch_core::traits::{LexicalRead, LexicalWrite};
use frankensearch_core::types::{IndexableDocument, ScoredResult};
use frankensearch_lexical::TantivyIndex;
use tantivy::collector::TopDocs;
use tantivy::columnar::Column;
use tantivy::query::QueryParser;
use tantivy::schema::{FAST, STORED, STRING, Schema, TEXT, Value};
use tantivy::{DocAddress, Index, TantivyDocument, doc};

const N: usize = 30_000;
const KS: &[usize] = &[30, 100, 300];
const WINNERS: usize = 10;
const HYDRATION_WINNERS: &[usize] = &[10, 30, 100, 300];

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

const VOCAB: &[&str] = &[
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "search",
    "engine",
    "vector",
    "lexical",
    "ranking",
    "relevance",
    "document",
    "query",
    "hybrid",
    "fusion",
];

struct Fixture {
    index: Index,
    id_field: tantivy::schema::Field,
    meta_field: tantivy::schema::Field,
    table: Vec<String>,
}

fn build_fixture() -> Fixture {
    let mut sb = Schema::builder();
    let id_field = sb.add_text_field("id", STRING | STORED);
    let content = sb.add_text_field("content", TEXT | STORED);
    let meta_field = sb.add_text_field("metadata_json", STORED);
    let ord_field = sb.add_u64_field("ord", FAST | STORED);
    let schema = sb.build();

    let index = Index::create_in_ram(schema);
    let mut writer: tantivy::IndexWriter = index.writer(80_000_000).expect("writer");
    let mut table = Vec::with_capacity(N);
    let vocab_len = u64::try_from(VOCAB.len()).expect("VOCAB length fits u64");
    for i in 0..N {
        let mut state = (i as u64).wrapping_add(1);
        let doc_id = format!("doc-{i:06}");
        let mut body = String::with_capacity(120);
        body.push_str("search ");
        for _ in 0..12 {
            let word_index = usize::try_from(xorshift(&mut state) % vocab_len)
                .expect("VOCAB remainder fits usize");
            body.push_str(VOCAB[word_index]);
            body.push(' ');
        }
        // Realistic metadata JSON object (~8 fields, a few hundred bytes) — the payload
        // fusion deserializes per candidate and then throws away.
        let meta = format!(
            r#"{{"title":"Document {i}","author":"author_{a}","source":"repo/{a}/path/file_{i}.rs","lang":"rust","tags":["search","engine","tag_{t}"],"score":{s},"updated":"2026-07-{d:02}T12:00:00Z","length":{len}}}"#,
            a = i % 97,
            t = i % 41,
            s = (i % 1000) as f64 / 10.0,
            d = (i % 28) + 1,
            len = body.len(),
        );
        writer
            .add_document(doc!(
                id_field => doc_id.clone(),
                content => body,
                meta_field => meta,
                ord_field => i as u64,
            ))
            .expect("add");
        table.push(doc_id);
    }
    writer.commit().expect("commit");

    Fixture {
        index,
        id_field,
        meta_field,
        table,
    }
}

/// A: what the hybrid fusion path pays now — full docstore decompress + id + metadata deserialize.
fn materialize_docstore_meta(
    searcher: &tantivy::Searcher,
    id_field: tantivy::schema::Field,
    meta_field: tantivy::schema::Field,
    hits: &[(f32, DocAddress)],
) -> (Vec<String>, usize) {
    let mut ids = Vec::with_capacity(hits.len());
    let mut meta_nodes = 0usize;
    for &(_, addr) in hits {
        let doc: TantivyDocument = searcher.doc(addr).expect("doc");
        let id = doc
            .get_first(id_field)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned();
        if let Some(s) = doc.get_first(meta_field).and_then(|v| v.as_str())
            && let Ok(val) = serde_json::from_str::<serde_json::Value>(s)
        {
            meta_nodes += val.as_object().map_or(0, serde_json::Map::len);
        }
        ids.push(id);
    }
    (ids, meta_nodes)
}

/// B: docstore decompress + id only (no metadata deserialize) — isolates the metadata share.
fn materialize_docstore_id(
    searcher: &tantivy::Searcher,
    id_field: tantivy::schema::Field,
    hits: &[(f32, DocAddress)],
) -> Vec<String> {
    let mut ids = Vec::with_capacity(hits.len());
    for &(_, addr) in hits {
        let doc: TantivyDocument = searcher.doc(addr).expect("doc");
        let id = doc
            .get_first(id_field)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_owned();
        ids.push(id);
    }
    ids
}

/// C: ord fast field → `table[ord]`, metadata dropped — what the hybrid path could pay.
fn materialize_fastfield_id(
    searcher: &tantivy::Searcher,
    table: &[String],
    hits: &[(f32, DocAddress)],
) -> Vec<String> {
    let seg_count = searcher.segment_readers().len();
    let mut cols: Vec<Option<Column<u64>>> = Vec::with_capacity(seg_count);
    for sr in searcher.segment_readers() {
        cols.push(sr.fast_fields().u64("ord").ok());
    }
    let mut ids = Vec::with_capacity(hits.len());
    for &(_, addr) in hits {
        let segment_index = usize::try_from(addr.segment_ord).expect("segment ordinal fits usize");
        let column = cols
            .get(segment_index)
            .and_then(Option::as_ref)
            .expect("segment ordinal has an ord column");
        let ord = column.first(addr.doc_id).expect("document has an ord");
        let ord = usize::try_from(ord).expect("document ordinal fits usize");
        ids.push(
            table
                .get(ord)
                .expect("document ordinal is in range")
                .clone(),
        );
    }
    ids
}

fn ratio(a: &PairedRatio, null: &PairedRatio) -> &'static str {
    if a.decidable_against(null) {
        "decidable"
    } else {
        "inside floor"
    }
}

fn build_product_index(
    runtime: &asupersync::runtime::Runtime,
    cx: &asupersync::Cx,
) -> TantivyIndex {
    let index = TantivyIndex::in_memory().expect("product index");
    let docs: Vec<IndexableDocument> = (0..N)
        .map(|i| {
            IndexableDocument::new(
                format!("doc-{i:06}"),
                format!(
                    "search engine hybrid lexical ranking document query {}",
                    VOCAB[i % VOCAB.len()]
                ),
            )
            .with_title(format!("Document {i}"))
            .with_metadata("author", format!("author_{}", i % 97))
            .with_metadata("source", format!("repo/path/file_{i}.rs"))
            .with_metadata("lang", "rust")
            .with_metadata("tags", format!("search,engine,tag_{}", i % 41))
            .with_metadata("score", format!("{}", i % 1000))
            .with_metadata("updated", format!("2026-07-{:02}", (i % 28) + 1))
            .with_metadata("length", "128")
        })
        .collect();
    runtime.block_on(async {
        index.index_documents(cx, &docs).await.expect("index docs");
        index.commit(cx).await.expect("commit docs");
    });
    index
}

fn full_winners(
    runtime: &asupersync::runtime::Runtime,
    cx: &asupersync::Cx,
    index: &TantivyIndex,
    k: usize,
) -> Vec<ScoredResult> {
    runtime.block_on(async {
        let mut results = index.search(cx, "search engine", k).await.expect("full");
        results.truncate(WINNERS);
        results
    })
}

fn deferred_winners(
    runtime: &asupersync::runtime::Runtime,
    cx: &asupersync::Cx,
    index: &TantivyIndex,
    k: usize,
) -> Vec<ScoredResult> {
    runtime.block_on(async {
        let batch = index
            .search_candidates(cx, "search engine", k)
            .await
            .expect("candidates");
        let (mut results, pin) = batch.into_parts();
        results.truncate(WINNERS);
        // Hydrate through the batch's own pin, which is what the deferred arm
        // now measures: scoring and hydration share one searcher.
        index
            .hydrate_candidates(cx, pin.as_ref(), &mut results)
            .await
            .expect("hydrate");
        results
    })
}

fn hydration_candidates(
    runtime: &asupersync::runtime::Runtime,
    cx: &asupersync::Cx,
    index: &TantivyIndex,
    winners: usize,
) -> Vec<ScoredResult> {
    runtime.block_on(async {
        index
            .search_candidates(cx, "search engine", winners)
            .await
            .expect("hydration candidates")
            .into_parts()
            .0
    })
}

fn clear_hydrated_metadata(results: &mut [ScoredResult]) {
    for result in results {
        result.metadata = None;
    }
}

fn run_product_gate() {
    let runtime = asupersync::runtime::RuntimeBuilder::current_thread()
        .build()
        .expect("runtime");
    let cx = asupersync::Cx::for_testing();
    let index = build_product_index(&runtime, &cx);
    let inner = std::env::var("META_HYDRATE_AB_INNER")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4);

    if std::env::var_os("META_COLLECTOR_ONLY").is_none() {
        for &k in KS {
            let full = full_winners(&runtime, &cx, &index, k);
            let deferred = deferred_winners(&runtime, &cx, &index, k);
            assert_eq!(
                serde_json::to_vec(&full).expect("serialize full"),
                serde_json::to_vec(&deferred).expect("serialize deferred"),
                "full and deferred winner outputs differ at k={k}"
            );
            assert!(
                deferred.iter().all(|result| result.metadata.is_some()),
                "winner metadata was not restored at k={k}"
            );

            let run_full = || {
                black_box(full_winners(&runtime, &cx, &index, black_box(k)));
            };
            let run_deferred = || {
                black_box(deferred_winners(&runtime, &cx, &index, black_box(k)));
            };
            let null = paired_median_ratio(31, inner, run_full, run_full);
            let candidate = paired_median_ratio(31, inner, run_full, run_deferred);
            eprintln!(
                "[product k{k}->w{WINNERS}] null {:.4} [{:.4},{:.4}] | deferred/full {:.4} [{:.4},{:.4}] ({})",
                null.median,
                null.p5,
                null.p95,
                candidate.median,
                candidate.p5,
                candidate.p95,
                ratio(&candidate, &null),
            );
        }
    }

    let collector_inner = std::env::var("META_COLLECTOR_AB_INNER")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4);
    for &winners in HYDRATION_WINNERS {
        let seed = hydration_candidates(&runtime, &cx, &index, winners);
        assert_eq!(seed.len(), winners, "unexpected hydration candidate count");

        let mut scored = seed.clone();
        runtime.block_on(async {
            index
                .hydrate_fusion_metadata_scored_for_bench(&cx, &mut scored)
                .await
                .expect("scored hydration parity");
        });
        let mut unscored = seed.clone();
        runtime.block_on(async {
            index
                .hydrate_fusion_metadata_unscored_for_bench(&cx, &mut unscored)
                .await
                .expect("unscored hydration parity");
        });
        assert_eq!(
            serde_json::to_vec(&scored).expect("serialize scored hydration"),
            serde_json::to_vec(&unscored).expect("serialize unscored hydration"),
            "scored and unscored hydration outputs differ at winners={winners}"
        );

        let mut null_a = seed.clone();
        let mut null_b = seed.clone();
        let null = paired_median_ratio(
            31,
            collector_inner,
            || {
                runtime.block_on(async {
                    index
                        .hydrate_fusion_metadata_scored_for_bench(&cx, &mut null_a)
                        .await
                        .expect("scored hydration null A");
                });
                black_box(&null_a);
                clear_hydrated_metadata(&mut null_a);
            },
            || {
                runtime.block_on(async {
                    index
                        .hydrate_fusion_metadata_scored_for_bench(&cx, &mut null_b)
                        .await
                        .expect("scored hydration null B");
                });
                black_box(&null_b);
                clear_hydrated_metadata(&mut null_b);
            },
        );

        let mut original = seed.clone();
        let mut candidate = seed;
        let unscored_vs_scored = paired_median_ratio(
            31,
            collector_inner,
            || {
                runtime.block_on(async {
                    index
                        .hydrate_fusion_metadata_scored_for_bench(&cx, &mut original)
                        .await
                        .expect("scored hydration");
                });
                black_box(&original);
                clear_hydrated_metadata(&mut original);
            },
            || {
                runtime.block_on(async {
                    index
                        .hydrate_fusion_metadata_unscored_for_bench(&cx, &mut candidate)
                        .await
                        .expect("unscored hydration");
                });
                black_box(&candidate);
                clear_hydrated_metadata(&mut candidate);
            },
        );
        eprintln!(
            "[hydrate winners={winners}] null {:.4} [{:.4},{:.4}] | unscored/scored {:.4} [{:.4},{:.4}] ({})",
            null.median,
            null.p5,
            null.p95,
            unscored_vs_scored.median,
            unscored_vs_scored.p5,
            unscored_vs_scored.p95,
            ratio(&unscored_vs_scored, &null),
        );
    }
}

fn main() {
    if std::env::var_os("META_COLLECTOR_ONLY").is_some() {
        run_product_gate();
        return;
    }

    let fx = build_fixture();
    let reader = fx.index.reader().expect("reader");
    let searcher = reader.searcher();
    let parser = QueryParser::for_index(
        &fx.index,
        vec![fx.index.schema().get_field("content").unwrap()],
    );
    let query = parser.parse_query("search").expect("query");
    let max_k = *KS.iter().max().unwrap();
    let hits_all: Vec<(f32, DocAddress)> = searcher
        .search(&query, &TopDocs::with_limit(max_k).order_by_score())
        .expect("search");

    // Parity: id lists identical across all three; report metadata node count once.
    let (a_ids, meta_nodes) =
        materialize_docstore_meta(&searcher, fx.id_field, fx.meta_field, &hits_all);
    let b_ids = materialize_docstore_id(&searcher, fx.id_field, &hits_all);
    let c_ids = materialize_fastfield_id(&searcher, &fx.table, &hits_all);
    assert_eq!(a_ids, b_ids, "docstore_meta vs docstore_id ids differ");
    assert_eq!(a_ids, c_ids, "docstore_meta vs fastfield_id ids differ");
    eprintln!(
        "segments={} meta_nodes(top-{max_k})={meta_nodes}",
        searcher.segment_readers().len()
    );

    let inner = std::env::var("META_WASTE_AB_INNER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);

    for &k in KS {
        let hits = &hits_all[..k.min(hits_all.len())];
        let run_meta = || {
            black_box(materialize_docstore_meta(
                &searcher,
                fx.id_field,
                fx.meta_field,
                black_box(hits),
            ));
        };
        let run_docid = || {
            black_box(materialize_docstore_id(
                &searcher,
                fx.id_field,
                black_box(hits),
            ));
        };
        let run_ffid = || {
            black_box(materialize_fastfield_id(
                &searcher,
                &fx.table,
                black_box(hits),
            ));
        };
        let null = paired_median_ratio(41, inner, run_meta, run_meta);
        // metadata share of A: docstore_id / docstore_meta  (how much of A is the doc+id, rest is metadata)
        let meta_share = paired_median_ratio(41, inner, run_meta, run_docid);
        // total hybrid win: fastfield_id / docstore_meta
        let total = paired_median_ratio(41, inner, run_meta, run_ffid);
        eprintln!(
            "[k{k}] null median {:.4} [{:.4},{:.4}] | docid/meta {:.4} [{:.4},{:.4}] ({}) | ffid/meta {:.4} [{:.4},{:.4}] ({})",
            null.median,
            null.p5,
            null.p95,
            meta_share.median,
            meta_share.p5,
            meta_share.p95,
            ratio(&meta_share, &null),
            total.median,
            total.p5,
            total.p95,
            ratio(&total, &null),
        );
    }

    run_product_gate();
}
