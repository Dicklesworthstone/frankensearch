//! Exclusions share the real Quill, retained source and independently produced
//! fast/quality fixture used by the ordinary immutable scope tests.
use super::*;

use asupersync::runtime::blocking_pool::BlockingPool;

#[test]
fn exclusions_refill_native_tiers_and_intersect_explicit_scopes() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for ann in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("exclusion-fast", 2));
            let quality = Arc::new(Provider::new("exclusion-quality", 3));
            let index = build(
                &cx,
                &directory.path().join("index"),
                ann,
                &fast,
                Some(&quality),
            )
            .await;
            for query in [
                "needle -private",
                "needle not \"needle private\"",
                "needle -'needle private'",
            ] {
                for (quality_primary, hits) in [
                    (false, index.search(&cx, query, 3).await.unwrap()),
                    (false, index.search_refined(&cx, query, 3).await.unwrap()),
                    (true, index.search_quality(&cx, query, 3).await.unwrap()),
                ] {
                    assert_eq!(hits.len(), 3, "eligible tail must fill {query}");
                    for hit in &hits {
                        assert!(!hit.doc_id.starts_with("private-"));
                        assert!(hit.metadata.is_some(), "keep final lexical hydration pin");
                        assert_row(&index, hit, quality_primary);
                    }
                }
            }
            assert!(
                fast.query_texts
                    .lock()
                    .unwrap()
                    .iter()
                    .all(|text| text == "needle")
            );
            assert!(
                quality
                    .query_texts
                    .lock()
                    .unwrap()
                    .iter()
                    .all(|text| text == "needle")
            );

            let scope = index.scope(&cx, |doc| Ok(doc.id != "a-fast")).unwrap();
            let mut stream = scope
                .progressive(&cx, "needle -private -middle", 10)
                .unwrap();
            for expected_refined in [false, true] {
                let phase = stream.next_phase().await.unwrap().unwrap();
                let hits = match phase {
                    NativeSearchPhase::Initial { results, .. } if !expected_refined => results,
                    NativeSearchPhase::Refined { results, .. } if expected_refined => results,
                    other => panic!("unexpected phase: {other:?}"),
                };
                assert_eq!(hits.len(), 2);
                assert!(
                    hits.iter()
                        .all(|hit| matches!(hit.doc_id.as_str(), "c-near" | "d-quality"))
                );
            }
            assert!(stream.next_phase().await.unwrap().is_none());
        }
    });
}

#[test]
fn exclusion_only_queries_skip_models_but_preserve_admission_cancellation_and_zero_k() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("exclusion-fast", 2));
        let quality = Arc::new(Provider::new("exclusion-quality", 3));
        let index = build(
            &cx,
            &directory.path().join("index"),
            false,
            &fast,
            Some(&quality),
        )
        .await;
        fast.fail.store(true, Ordering::SeqCst);
        quality.fail.store(true, Ordering::SeqCst);
        let pool = BlockingPool::new(0, 1);
        for query in ["-private", "not \"needle private\""] {
            for hits in [
                index.search(&cx, query, 20).await.unwrap(),
                index.search_refined(&cx, query, 20).await.unwrap(),
                index.search_quality(&cx, query, 20).await.unwrap(),
                index
                    .search_refined_with_pool(&cx, query, 20, pool.handle())
                    .await
                    .unwrap(),
                index
                    .search_quality_with_pool(&cx, query, 20, pool.handle())
                    .await
                    .unwrap(),
            ] {
                assert_eq!(hits.len(), 4);
                assert!(hits.iter().all(|hit| !hit.doc_id.starts_with("private-")
                    && hit.fast_score.is_none()
                    && hit.quality_score.is_none()));
            }
        }
        assert!(
            index
                .search_refined(&cx, "needle -private", 0)
                .await
                .unwrap()
                .is_empty()
        );
        let empty = index.scope(&cx, |_| Ok(false)).unwrap();
        assert!(
            empty
                .search_with_pool(&cx, "needle -private", 10, pool.handle())
                .await
                .unwrap()
                .is_empty()
        );
        assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
        quality.advertise_foreign.store(true, Ordering::SeqCst);
        assert!(index.search_refined(&cx, "-private", 0).await.is_err());
        quality.advertise_foreign.store(false, Ordering::SeqCst);
        cx.set_cancel_requested(true);
        assert!(matches!(
            index.search_refined(&cx, "-private", 3).await,
            Err(SearchError::Cancelled { .. })
        ));
        cx.set_cancel_requested(false);
        assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
    });
}

struct QueryReranker {
    query: Mutex<String>,
    documents: Mutex<Vec<RerankDocument>>,
}

impl Reranker for QueryReranker {
    fn id(&self) -> &'static str {
        "exclusion-reranker"
    }
    fn model_name(&self) -> &str {
        self.id()
    }
    fn rerank<'a>(
        &'a self,
        _cx: &'a Cx,
        query: &'a str,
        documents: &'a [RerankDocument],
    ) -> SearchFuture<'a, Vec<RerankScore>> {
        Box::pin(async move {
            *self.query.lock().unwrap() = query.to_owned();
            *self.documents.lock().unwrap() = documents.to_vec();
            Ok(documents
                .iter()
                .enumerate()
                .map(|(original_rank, document)| RerankScore {
                    doc_id: document.doc_id.clone(),
                    original_rank,
                    raw_logit: None,
                    score: if document.doc_id == "d-quality" {
                        1.0
                    } else {
                        0.0
                    },
                })
                .collect())
        })
    }
}

#[test]
fn exclusions_keep_literal_quotes_no_match_ranks_and_positive_reranker_text() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("exclusion-fast", 2));
        let quality = Arc::new(Provider::new("exclusion-quality", 3));
        let index = build(
            &cx,
            &directory.path().join("index"),
            false,
            &fast,
            Some(&quality),
        )
        .await;
        for (original, excluded) in [
            (
                index.search(&cx, "needle", 3).await,
                index.search(&cx, "needle -unfindable", 3).await,
            ),
            (
                index.search_refined(&cx, "needle", 3).await,
                index.search_refined(&cx, "needle -unfindable", 3).await,
            ),
            (
                index.search_quality(&cx, "needle", 3).await,
                index.search_quality(&cx, "needle -unfindable", 3).await,
            ),
        ] {
            assert_eq!(
                serde_json::to_value(original.unwrap()).unwrap(),
                serde_json::to_value(excluded.unwrap()).unwrap()
            );
        }
        let literal = r#""needle -private" -unfindable"#;
        let hits = index.search(&cx, literal, 1).await.unwrap();
        assert!(hits[0].doc_id.starts_with("private-"));
        assert_eq!(
            fast.query_texts.lock().unwrap().last().unwrap(),
            r#""needle -private""#
        );
        let probe = QueryReranker {
            query: Mutex::new(String::new()),
            documents: Mutex::new(Vec::new()),
        };
        let mut stream = index
            .progressive_with_reranker(&cx, "needle -private", 2, &probe, 4)
            .unwrap();
        while stream.next_phase().await.unwrap().is_some() {}
        assert_eq!(*probe.query.lock().unwrap(), "needle");
        let docs = probe.documents.lock().unwrap();
        assert_eq!(docs.len(), 4);
        assert!(docs.iter().all(|doc| !doc.doc_id.starts_with("private-")));
        drop(docs);
    });
}

#[test]
fn exclusion_census_keeps_boolean_words_literal_when_combining_terms() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for ann in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let fast = Arc::new(Provider::new("literal-exclusion-fast", 2));
            let quality = Arc::new(Provider::new("literal-exclusion-quality", 3));
            let index = NativeIndexBuilder::new(
                directory.path().join("index"),
                generation(1),
                fast.clone(),
            )
            .unwrap()
            .add_documents(
                ["AND", "OR", "NOT", "private", "public"]
                    .map(|word| IndexableDocument::new(word, format!("needle {word}"))),
            )
            .with_fast_storage(
                NativeBuildPrecision::F16,
                if ann {
                    NativeBuildRetrieval::Hnsw {
                        params: HnswParams::default(),
                        seed: 7,
                    }
                } else {
                    NativeBuildRetrieval::Exact
                },
            )
            .with_quality_embedder(quality.clone())
            .unwrap()
            .with_quality_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Exact)
            .unwrap()
            .build_hybrid(&cx)
            .await
            .unwrap();
            assert_eq!(index.search(&cx, "needle", 10).await.unwrap().len(), 5);
            for word in ["AND", "OR", "NOT"] {
                let expected: BTreeSet<_> = ["AND", "OR", "NOT", "private", "public"]
                    .into_iter()
                    .filter(|id| *id != word)
                    .collect();
                for query in [
                    format!("needle -{word} -absent"),
                    format!("needle -absent -{word}"),
                    format!("needle -\"{word}\" -absent"),
                ] {
                    for hits in [
                        index.search(&cx, &query, 10).await.unwrap(),
                        index.search_refined(&cx, &query, 10).await.unwrap(),
                        index.search_quality(&cx, &query, 10).await.unwrap(),
                    ] {
                        let actual: BTreeSet<_> =
                            hits.iter().map(|hit| hit.doc_id.as_str()).collect();
                        assert_eq!(actual, expected, "ann={ann}, query={query}");
                    }
                }
            }
        }
    });
}

#[test]
fn exclusion_census_preserves_trailing_terms_when_or_rendering_expands_the_query() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("long-exclusion-fast", 2));
        let quality = Arc::new(Provider::new("long-exclusion-quality", 3));
        let index = build(
            &cx,
            &directory.path().join("index"),
            false,
            &fast,
            Some(&quality),
        )
        .await;
        let mut query = String::from("needle ");
        for term in 0..800 {
            use std::fmt::Write;
            write!(query, "-absent{term:03} ").unwrap();
        }
        query.push_str("-private");
        assert!(query.chars().count() < frankensearch_quill::query::MAX_QUERY_LENGTH);
        let expanded = frankensearch_core::ParsedQuery::parse(&query)
            .negative_terms
            .join(" OR ");
        assert!(expanded.chars().count() > frankensearch_quill::query::MAX_QUERY_LENGTH);
        for hits in [
            index.search(&cx, &query, 28).await.unwrap(),
            index.search_refined(&cx, &query, 28).await.unwrap(),
            index.search_quality(&cx, &query, 28).await.unwrap(),
        ] {
            assert_eq!(hits.len(), 4);
            assert!(hits.iter().all(|hit| !hit.doc_id.starts_with("private-")));
        }
        // Matching the same ID in both census groups is valid membership,
        // rather than a duplicate candidate returned within one batch.
        let overlapping = format!("-private {query}");
        let hits = index.search(&cx, &overlapping, 28).await.unwrap();
        assert_eq!(hits.len(), 4);
        assert!(hits.iter().all(|hit| !hit.doc_id.starts_with("private-")));
        assert_eq!(fast.queries.load(Ordering::SeqCst), 3);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 2);
        assert!(
            fast.query_texts
                .lock()
                .unwrap()
                .iter()
                .all(|text| text == "needle")
        );
        assert!(
            quality
                .query_texts
                .lock()
                .unwrap()
                .iter()
                .all(|text| text == "needle")
        );
    });
}

#[test]
fn exclusion_census_refuses_one_oversized_escaped_phrase_before_inference() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("escaped-exclusion-fast", 2));
        let index = build(&cx, &directory.path().join("index"), false, &fast, None).await;
        let query = format!(
            "needle -'{}'",
            "\"".repeat(frankensearch_quill::query::MAX_QUERY_LENGTH / 2 + 1)
        );
        assert!(query.chars().count() < frankensearch_quill::query::MAX_QUERY_LENGTH);
        let error = index.search(&cx, &query, 28).await.unwrap_err();
        assert!(
            matches!(&error, SearchError::InvalidConfig { field, value, .. }
            if field == "native_ann.query.exclusions" && value == "oversized-clause"),
            "unexpected oversized-clause refusal: {error:?}"
        );
        assert_eq!(fast.queries.load(Ordering::SeqCst), 0);

        // The parser bound counts Unicode scalars, not UTF-8 bytes. A long
        // unmatched Unicode phrase still fits and must remain searchable.
        let unicode_query = format!(
            "needle -'{}'",
            "é".repeat(frankensearch_quill::query::MAX_QUERY_LENGTH / 2 + 1)
        );
        assert!(unicode_query.len() > frankensearch_quill::query::MAX_QUERY_LENGTH);
        assert!(unicode_query.chars().count() < frankensearch_quill::query::MAX_QUERY_LENGTH);
        let hits = index.search(&cx, &unicode_query, 28).await.unwrap();
        assert_eq!(hits.len(), 28);
        assert_eq!(fast.queries.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn global_exclusions_widen_mixed_boolean_lexical_branches() {
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        let directory = tempfile::tempdir().unwrap();
        let fast = Arc::new(Provider::new("boolean-fast", 2));
        let mut docs: Vec<_> = (0..24)
            .map(|i| IndexableDocument::new(format!("blocked-{i:02}"), "alpha banned"))
            .collect();
        docs.extend([
            IndexableDocument::new("allowed-a", format!("alpha {}", "padding ".repeat(24))),
            IndexableDocument::new("allowed-b", format!("alpha {}", "padding ".repeat(24))),
        ]);
        let index = NativeIndexBuilder::new(directory.path().join("index"), generation(1), fast)
            .unwrap()
            .add_documents(docs)
            .with_fast_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Exact)
            .build_hybrid(&cx)
            .await
            .unwrap();
        for query in ["alpha OR beta AND -banned", "alpha OR -banned"] {
            let hits = index.search(&cx, query, 2).await.unwrap();
            assert_eq!(hits.len(), 2);
            assert!(hits.iter().all(|hit| hit.doc_id.starts_with("allowed-")));
            assert!(
                hits.iter().all(|hit| hit.lexical_score.is_some()),
                "the lexical lane must refill its eligible tail, not leave semantic-only survivors"
            );
        }
    });
}
