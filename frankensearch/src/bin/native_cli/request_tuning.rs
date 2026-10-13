//! Explicit, startup-authorized request tuning over the existing phase engine.
//! This envelope keeps ordinary flat requests and control grants unchanged.

use super::{Cx, Deserialize, Frames, Mode, Result, Write, bad, cohort, filter, query};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Operation {
    Search,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    #[serde(rename = "op")]
    _operation: Operation,
    request: super::Request,
    #[serde(deserialize_with = "query::deserialize_query_override")]
    full_query: query::FullQueryOverride,
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn execute<W: Write>(
    index: cohort::Index<'_>,
    cx: &Cx,
    message: &Request,
    ordinal: u64,
    defaults: (Mode, usize),
    output: &mut W,
    base_filter: Option<&filter::Filter>,
    policy: &query::Policy,
) -> Result<()> {
    let admitted = (|| {
        // Permission and cancellation precede all request-driven model work.
        let effective = policy.for_query_override(cx, &message.full_query)?;
        message.request.validate()?;
        if message.request.mode.unwrap_or(defaults.0) != Mode::Full {
            return Err(bad("request full-query tuning requires mode full"));
        }
        Ok(effective)
    })();
    let effective = match admitted {
        Ok(effective) => effective,
        Err(error) => {
            let mut terminal = query::failure(error.as_ref());
            terminal["event"] = serde_json::json!("terminal");
            terminal["ok"] = serde_json::json!(false);
            terminal["partial_results"] = serde_json::json!(false);
            // Do not reflect an invalid/oversized ID into a rejection frame.
            let id = message
                .request
                .id
                .as_deref()
                .filter(|id| super::validate_id(Some(id)).is_ok());
            return Frames::new(output, ordinal, id, index.generation()).send(terminal);
        }
    };
    // One retained cohort, intersected scopes, unchanged deadline ceiling and
    // reranker. A delivery error remains fatal; reported query failures do not.
    super::stream_one(
        index, cx, &message.request, ordinal, defaults, output, base_filter, &effective,
    )
    .await?;
    Ok(())
}

#[cfg(test)]
mod wire_tests {
    use super::*;

    #[test]
    fn tuning_envelope_is_unambiguous_and_retains_duplicate_field_rejection() {
        let accepted = r#"{"op":"search","request":{"query":"needle"},"full_query":{}}"#;
        assert!(matches!(serde_json::from_str::<super::super::Message>(accepted).unwrap(),
            super::super::Message::TunedSearch(_)));
        assert!(matches!(serde_json::from_str::<super::super::Message>(
            r#"{"query":"needle"}"#).unwrap(), super::super::Message::Search(_)));
        for full_query in [
            "null", "[]", "[1,null,null,60,1,1,0.7]",
            r#"{"candidate_multiplier":1,"candidate_multiplier":2}"#,
            r#"{"fast_ef":null,"fast_ef":2}"#,
            r#"{"request_overrides":true}"#, r#"{"timeout_ms":1000}"#,
            r#"{"filter":{}}"#, r#"{"model_dir":"private-canary"}"#,
        ] {
            let message = format!(
                r#"{{"op":"search","request":{{"query":"needle"}},"full_query":{full_query}}}"#,
            );
            assert!(serde_json::from_str::<super::super::Message>(&message).is_err());
        }
        for message in [
            r#"{"query":"needle","full_query":{}}"#,
            r#"{"op":"search","request":{"query":"needle"}}"#,
            r#"{"op":"search","request":{"query":"needle"},"full_query":{},"other":1}"#,
            r#"{"op":"status","request":{"query":"needle"},"full_query":{}}"#,
        ] {
            assert!(serde_json::from_str::<super::super::Message>(message).is_err());
        }
        let request: Request = serde_json::from_str(
            r#"{"op":"search","request":{"query":"needle"},"full_query":{"fast_ef":null}}"#,
        ).unwrap();
        assert!(query::Policy::default().for_query_override(&Cx::for_testing(), &request.full_query).is_err());
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
mod tests {
    use super::*;
    use crate::{
        ArtifactGenerationIdentityV1, Embedder, IndexableDocument, NativeBuildPrecision,
        NativeBuildRetrieval, NativeIndexBuilder, live, sharded,
    };
    use asupersync::test_utils::run_test_with_cx;
    use frankensearch_core::{
        EmbeddingIdentityBundleV1, ModelCategory, SearchError, SearchFuture, SearchResult,
    };
    use frankensearch_index::native_hnsw::HnswParams;
    use std::io::Cursor;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    // Synthetic identified providers exercise actual native/Quill/stdio paths,
    // not real-model relevance or a graph throughput benchmark.
    struct Model {
        identity: EmbeddingIdentityBundleV1,
        quality: bool,
        calls: Arc<AtomicUsize>,
        fail: Arc<AtomicBool>,
    }

    impl Embedder for Model {
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> { Ok(&self.identity) }
        fn dimension(&self) -> usize { 3 }
        fn id(&self) -> &str { &self.identity.space.logical_model_id }
        fn model_name(&self) -> &str { self.id() }
        fn is_semantic(&self) -> bool { false }
        fn category(&self) -> ModelCategory { ModelCategory::HashEmbedder }
        fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                if self.fail.load(Ordering::SeqCst) {
                    return Err(SearchError::EmbeddingFailed {
                        model: self.id().to_owned(), source: "controlled quality failure".into(),
                    });
                }
                Ok(match text {
                    "needle" => vec![1.0, 0.0, 0.0],
                    "vector content" if !self.quality => vec![1.0, 0.0, 0.0],
                    "quality content" if self.quality => vec![1.0, 0.0, 0.0],
                    "needle keyword" => vec![0.0, 1.0, 0.0],
                    "filler content" => vec![-1.0, 0.0, 0.0],
                    _ => vec![0.0, 0.0, 1.0],
                })
            })
        }
    }

    struct Fixture {
        _directory: tempfile::TempDir,
        index: sharded::Opened,
        fast_calls: Arc<AtomicUsize>,
        quality_calls: Arc<AtomicUsize>,
        fail_quality: Arc<AtomicBool>,
    }

    async fn fixture(cx: &Cx, partitioned: bool, exact: bool, quality: bool) -> Fixture {
        let directory = tempfile::tempdir().unwrap();
        let fast_calls = Arc::new(AtomicUsize::new(0));
        let quality_calls = Arc::new(AtomicUsize::new(0));
        let fail_quality = Arc::new(AtomicBool::new(false));
        let retrieval = if exact { NativeBuildRetrieval::Exact } else {
            NativeBuildRetrieval::Hnsw { params: HnswParams::default(), seed: 42 }
        };
        let fast = Arc::new(Model {
            identity: EmbeddingIdentityBundleV1::explicit_test_model("request-fast", 3),
            quality: false, calls: Arc::clone(&fast_calls),
            fail: Arc::new(AtomicBool::new(false)),
        });
        let documents = [
            ("vector", "vector content"), ("keyword", "needle keyword"),
            ("quality", "quality content"), ("filler", "filler content"),
        ].map(|(id, text)| IndexableDocument::new(id, text).with_metadata("original_id", id));
        let mut builder = NativeIndexBuilder::new(
            directory.path().join("cohort"),
            ArtifactGenerationIdentityV1::new(1, [41; 16]).unwrap(), fast,
        ).unwrap().with_fast_storage(NativeBuildPrecision::F32, retrieval).add_documents(documents);
        if quality {
            builder = builder.with_quality_embedder(Arc::new(Model {
                identity: EmbeddingIdentityBundleV1::explicit_test_model("request-quality", 3),
                quality: true, calls: Arc::clone(&quality_calls), fail: Arc::clone(&fail_quality),
            })).unwrap().with_quality_storage(NativeBuildPrecision::F32, retrieval).unwrap();
        }
        let index = if partitioned {
            sharded::Opened::Sharded(Box::new(builder.build_sharded_hybrid(cx, 2).await.unwrap()))
        } else {
            sharded::Opened::Single(Box::new(builder.build_hybrid(cx).await.unwrap()))
        };
        fast_calls.store(0, Ordering::SeqCst);
        quality_calls.store(0, Ordering::SeqCst);
        Fixture { _directory: directory, index, fast_calls, quality_calls, fail_quality }
    }

    fn policy() -> query::Policy {
        query::Policy {
            full_query: Some(query::FullQuery::parse(
                r#"{"request_overrides":true,"candidate_multiplier":3,"lexical_weight":100,"fast_ef":128,"quality_ef":128}"#,
            ).unwrap()),
            ..query::Policy::default()
        }
    }

    fn frames(bytes: &[u8]) -> Vec<serde_json::Value> {
        bytes.split(|byte| *byte == b'\n').filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap()).collect()
    }

    #[test]
    fn warm_overrides_change_ranking_and_budget_without_leaking_into_following_requests() {
        run_test_with_cx(|cx| async move {
            for partitioned in [false, true] {
                for exact in [false, true] {
                    let f = fixture(&cx, partitioned, exact, false).await;
                    let server = live::Serving::new(&cx, f.index).unwrap();
                    let policy = policy();
                    let before = serde_json::to_value(policy.full_query.as_ref().unwrap()).unwrap();
                    let mut input = Cursor::new(concat!(
                        "{\"op\":\"search\",\"request\":{\"id\":\"vector-query\",\"query\":\"needle\",\"limit\":1},\"full_query\":{\"candidate_multiplier\":1,\"lexical_weight\":0.01,\"fast_ef\":8}}\n",
                        "{\"id\":\"ordinary\",\"query\":\"needle\",\"limit\":1}\n",
                        "{\"op\":\"search\",\"request\":{\"id\":\"over-limit\",\"query\":\"needle\"},\"full_query\":{\"candidate_multiplier\":4}}\n",
                        "{\"id\":\"fast\",\"query\":\"needle\",\"mode\":\"fast\",\"limit\":1}\n",
                    ));
                    let mut output = Vec::new();
                    super::super::run_with_controls(
                        server.borrow(), &cx, &mut input, &mut output, (Mode::Full, 1),
                        super::super::Controls::default(), None, &policy,
                    ).await.unwrap();
                    let output = frames(&output);
                    assert_eq!(output[0]["query_tuning_enabled"], true);
                    let page = |id: &str| output.iter().find(|f| f["id"] == id && f["event"] == "results").unwrap();
                    assert_eq!(page("vector-query")["results"][0]["doc_id"], "vector");
                    assert_eq!(page("vector-query")["candidates"]["fast"], 1);
                    assert_eq!(page("vector-query")["full_query"]["fast_ef"], 8);
                    assert_eq!(page("ordinary")["results"][0]["doc_id"], "keyword");
                    assert_eq!(page("ordinary")["candidates"]["fast"], 3);
                    assert_eq!(page("ordinary")["full_query"], before);
                    assert!(page("fast").get("full_query").is_none());
                    let rejected: Vec<_> = output.iter().filter(|f| f["id"] == "over-limit").collect();
                    assert_eq!(rejected.len(), 1);
                    assert_eq!(rejected[0]["event"], "terminal");
                    assert_eq!(rejected[0]["seq"], 0);
                    assert_eq!(rejected[0]["partial_results"], false);
                    assert_eq!(f.fast_calls.load(Ordering::SeqCst), 3);
                    assert_eq!(f.quality_calls.load(Ordering::SeqCst), 0);
                    assert_eq!(serde_json::to_value(policy.full_query.as_ref().unwrap()).unwrap(), before);
                    if partitioned {
                        assert_eq!(page("vector-query")["layout"], "sharded");
                        assert!(page("vector-query")["results"][0]["fast_row"].is_object());
                    }
                }
            }
        });
    }

    #[test]
    fn disabled_invalid_and_nonfull_tuning_reject_before_any_provider_work() {
        run_test_with_cx(|cx| async move {
            let f = fixture(&cx, false, true, true).await;
            for (policy, mode, override_json) in [
                (query::Policy::default(), "full", "{}"),
                (policy(), "fast", "{}"), (policy(), "quality", "{}"),
                (policy(), "full", r#"{"quality_ef":129}"#),
                (policy(), "full", r#"{"lexical_weight":0}"#),
            ] {
                let request: Request = serde_json::from_str(&format!(
                    r#"{{"op":"search","request":{{"query":"needle","mode":"{mode}"}},"full_query":{override_json}}}"#,
                )).unwrap();
                let mut output = Vec::new();
                execute(f.index.borrow(), &cx, &request, 1, (Mode::Full, 1), &mut output, None, &policy).await.unwrap();
                let output = frames(&output);
                assert_eq!(output.len(), 1);
                assert_eq!(output[0]["status"], "failed");
                assert_eq!(output[0]["partial_results"], false);
                assert_eq!(f.fast_calls.load(Ordering::SeqCst), 0);
                assert_eq!(f.quality_calls.load(Ordering::SeqCst), 0);
            }
            let request: Request = serde_json::from_str(
                r#"{"op":"search","request":{"query":"needle"},"full_query":{}}"#,
            ).unwrap();
            cx.set_cancel_requested(true);
            let mut output = Vec::new();
            execute(f.index.borrow(), &cx, &request, 1, (Mode::Full, 1), &mut output, None, &policy()).await.unwrap();
            assert_eq!(frames(&output)[0]["status"], "cancelled");
            cx.set_cancel_requested(false);
            assert_eq!(f.fast_calls.load(Ordering::SeqCst), 0);
            assert_eq!(f.quality_calls.load(Ordering::SeqCst), 0);
        });
    }

    #[test]
    fn request_scopes_still_intersect_and_zero_weight_cannot_disable_required_quality() {
        run_test_with_cx(|cx| async move {
            for partitioned in [false, true] {
                let f = fixture(&cx, partitioned, true, true).await;
                let base_filter = filter::Filter::parse(r#"{"ids":["keyword","quality"]}"#).unwrap();
                let request: Request = serde_json::from_str(
                    r#"{"op":"search","request":{"query":"needle","limit":1,"filter":{"ids":["keyword","vector"]}},"full_query":{"quality_weight":0,"candidate_multiplier":1}}"#,
                ).unwrap();
                f.fail_quality.store(true, Ordering::SeqCst);
                let mut output = Vec::new();
                execute(f.index.borrow(), &cx, &request, 7, (Mode::Full, 1), &mut output, Some(&base_filter), &policy()).await.unwrap();
                let output = frames(&output);
                assert_eq!(output.len(), 4);
                assert_eq!(output[1]["scope"]["eligible_documents"], 1);
                assert_eq!(output[1]["results"][0]["doc_id"], "keyword");
                assert_eq!(output[1]["results"][0]["metadata"]["original_id"], "keyword");
                assert_eq!(output[1]["phase"], "initial");
                assert_eq!(output[2]["phase"], "refinement_failed");
                assert_eq!(output[2]["results"], output[1]["results"]);
                assert_eq!(output[3]["status"], "degraded");
                assert_eq!(output[3]["partial_results"], true);
                for (seq, frame) in output.iter().enumerate() {
                    assert_eq!(frame["seq"], seq);
                    assert_eq!(frame["request"], 7);
                    assert_eq!(frame["generation"], output[0]["generation"]);
                }
                if partitioned {
                    assert!(output[1]["results"][0]["fast_row"].is_object());
                    assert!(output[1]["results"][0].get("quality_row").is_some());
                }
                assert_eq!(f.fast_calls.load(Ordering::SeqCst), 1);
                assert_eq!(f.quality_calls.load(Ordering::SeqCst), 1);
            }
        });
    }

    #[test]
    fn output_failures_remain_fatal_before_provider_work() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "controlled delivery failure"))
            }
            fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
        }
        run_test_with_cx(|cx| async move {
            let f = fixture(&cx, false, true, true).await;
            let request: Request = serde_json::from_str(
                r#"{"op":"search","request":{"query":"needle"},"full_query":{}}"#,
            ).unwrap();
            // Both the permission-rejection terminal and successful admission's
            // started frame propagate delivery failure rather than attempting
            // another result/terminal after a potentially partial write.
            for policy in [query::Policy::default(), policy()] {
                assert!(execute(f.index.borrow(), &cx, &request, 1, (Mode::Full, 1), &mut Broken, None, &policy).await.is_err());
                assert_eq!(f.fast_calls.load(Ordering::SeqCst), 0);
                assert_eq!(f.quality_calls.load(Ordering::SeqCst), 0);
            }
        });
    }
}
