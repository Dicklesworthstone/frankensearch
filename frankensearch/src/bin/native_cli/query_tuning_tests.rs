//! CLI parsing and actual buffered/stdio dispatch over native/Quill fixtures.
//! Identified control models isolate policy wiring, not semantic quality.

use super::*;
use asupersync::test_utils::run_test_with_cx;
use asupersync::time::{TimerDriverHandle, VirtualClock};
use frankensearch::{ModelCategory, RerankDocument, RerankScore, Reranker, SearchFuture};
use frankensearch_core::{EmbeddingIdentityBundleV1, SearchError, SearchResult};
use std::future::Future;
use std::io::Cursor;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Waker};

const TUNED: &str = r#"{"candidate_multiplier":1,"fast_ef":8,"quality_ef":7,"rrf_k":0,"lexical_weight":7,"quality_weight":0}"#;

fn parsed(extra: &[&str]) -> Result<Option<Options>> {
    Options::parse(
        ["search", "--receipt", "unused", "--query", "needle"]
            .into_iter()
            .chain(extra.iter().copied())
            .map(str::to_owned),
    )
}

fn policy(cx: &Cx) -> query::Policy {
    let options = parsed(&["--full-query", TUNED]).unwrap().unwrap();
    let mut policy = query::Policy::new(cx, None).unwrap();
    policy.full_query = options.full_query;
    policy
}

#[test]
fn full_query_parsing_is_bounded_strict_and_explicitly_full_mode() {
    let parsed = parsed(&["--full-query", TUNED]).unwrap().unwrap();
    let encoded = serde_json::to_value(parsed.full_query.as_ref().unwrap()).unwrap();
    assert_eq!(encoded["candidate_multiplier"], 1);
    assert_eq!(encoded["quality_ef"], 7);
    assert_eq!(encoded["semantic_weight"], 1.0);
    assert!(
        Options::parse(
            [
                "serve",
                "--receipt",
                "r",
                "--mode",
                "fast",
                "--full-query",
                TUNED
            ]
            .into_iter()
            .map(str::to_owned)
        )
        .is_ok()
    );
    for mode in ["fast", "quality"] {
        assert!(self::parsed(&["--mode", mode, "--full-query", TUNED]).is_err());
    }
    assert!(self::parsed(&["--lexical-only", "--full-query", "{}"]).is_err());
    assert!(self::parsed(&["--full-query", "{}", "--full-query", "{}"]).is_err());
    for command in ["index", "update", "rebuild"] {
        assert!(
            Options::parse(
                [
                    command,
                    "--receipt",
                    "r",
                    "--index-dir",
                    "new",
                    "--full-query",
                    "{}"
                ]
                .into_iter()
                .map(str::to_owned)
            )
            .is_err()
        );
    }
    for invalid in [
        r#"{"candidate_multiplier":0}"#,
        r#"{"candidate_multiplier":101}"#,
        r#"{"fast_ef":0}"#,
        r#"{"quality_ef":1000001}"#,
        r#"{"rrf_k":-1}"#,
        r#"{"lexical_weight":0}"#,
        r#"{"semantic_weight":-1}"#,
        r#"{"semantic_weight":1e100}"#,
        r#"{"quality_weight":1.1}"#,
        r#"{"quality_weight":null}"#,
        r#"{"candidate_multiplier":1,"candidate_multiplier":2}"#,
        r#"{"unknown":"private-policy-canary"}"#,
        r#"{"candidate_multiplier":"private-policy-canary"}"#,
        "null",
        "[]",
        "NaN",
    ] {
        let error = query::FullQuery::parse(invalid).unwrap_err();
        assert!(!error.to_string().contains("private-policy-canary"));
    }
    assert!(query::FullQuery::parse(&" ".repeat(8193)).is_err());
    let boundary = query::FullQuery::parse(
        r#"{"candidate_multiplier":100,"fast_ef":1000000,"quality_ef":null,"quality_weight":1}"#,
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(boundary).unwrap()["quality_weight"],
        1.0
    );
    assert!(self::parsed(&[]).unwrap().unwrap().full_query.is_none());
}

struct Model {
    identity: EmbeddingIdentityBundleV1,
    calls: AtomicUsize,
    drops: AtomicUsize,
    mode: AtomicUsize,
}

impl Model {
    fn new(name: &str) -> Self {
        Self {
            identity: EmbeddingIdentityBundleV1::explicit_test_model(name, 2),
            calls: AtomicUsize::new(0),
            drops: AtomicUsize::new(0),
            mode: AtomicUsize::new(0),
        }
    }
}

struct Dropped<'a>(&'a AtomicUsize);
impl Drop for Dropped<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

impl Embedder for Model {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }
    fn dimension(&self) -> usize {
        2
    }
    fn id(&self) -> &str {
        &self.identity.space.logical_model_id
    }
    fn model_name(&self) -> &str {
        self.id()
    }
    fn is_semantic(&self) -> bool {
        false
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::HashEmbedder
    }
    fn embed<'a>(&'a self, _cx: &'a Cx, _text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let _drop = Dropped(&self.drops);
            match self.mode.load(Ordering::SeqCst) {
                1 => Err(SearchError::EmbeddingFailed {
                    model: self.id().to_owned(),
                    source: "controlled quality failure".into(),
                }),
                2 => std::future::pending().await,
                _ => Ok(vec![1.0, 0.0]),
            }
        })
    }
}

async fn fixture(
    cx: &Cx,
    partitioned: bool,
    exact: bool,
) -> (tempfile::TempDir, sharded::Opened, Arc<Model>, Arc<Model>) {
    let parent = tempfile::tempdir().unwrap();
    let fast = Arc::new(Model::new("cli-tuning-fast"));
    let quality = Arc::new(Model::new("cli-tuning-quality"));
    let generation = ArtifactGenerationIdentityV1::new(1, [19; 16]).unwrap();
    let retrieval = if exact {
        NativeBuildRetrieval::Exact
    } else {
        NativeBuildRetrieval::Hnsw {
            params: HnswParams::default(),
            seed: 19,
        }
    };
    let builder = NativeIndexBuilder::new(parent.path().join("index"), generation, fast.clone())
        .unwrap()
        .with_quality_embedder(quality.clone())
        .unwrap()
        .with_fast_storage(NativeBuildPrecision::F32, retrieval)
        .with_quality_storage(NativeBuildPrecision::F32, retrieval)
        .unwrap()
        .add_documents([
            IndexableDocument::new("a", "ordinary alpha").with_metadata("epoch", "old"),
            IndexableDocument::new("b", "ordinary beta").with_metadata("epoch", "old"),
            IndexableDocument::new("z", "needle").with_metadata("epoch", "old"),
        ]);
    let index = if partitioned {
        sharded::Opened::Sharded(Box::new(builder.build_sharded_hybrid(cx, 1).await.unwrap()))
    } else {
        sharded::Opened::Single(Box::new(builder.build_hybrid(cx).await.unwrap()))
    };
    (parent, index, fast, quality)
}

fn frames(bytes: &[u8]) -> Vec<serde_json::Value> {
    bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect()
}

#[test]
fn buffered_and_streaming_tuning_reaches_both_layouts_and_changes_real_ranking() {
    run_test_with_cx(|cx| async move {
        for partitioned in [false, true] {
            for exact in [false, true] {
                let (_parent, index, fast, quality) = fixture(&cx, partitioned, exact).await;
                let defaults = query::Policy::new(&cx, None).unwrap();
                let ordinary = query::buffered(
                    index.borrow(),
                    &cx,
                    "needle",
                    Mode::Full,
                    1,
                    [None, None],
                    &defaults,
                )
                .await
                .unwrap();
                assert!(ordinary.get("full_query").is_none());
                assert!(ordinary["results"][0]["score"].as_f64().unwrap() < 1.0);
                let policy = policy(&cx);
                let before = (
                    fast.calls.load(Ordering::SeqCst),
                    quality.calls.load(Ordering::SeqCst),
                );
                let tuned = query::buffered(
                    index.borrow(),
                    &cx,
                    "needle",
                    Mode::Full,
                    1,
                    [None, None],
                    &policy,
                )
                .await
                .unwrap();
                assert_eq!(tuned["phase"], "refined");
                assert_eq!(tuned["results"][0]["doc_id"], "z");
                assert_eq!(tuned["results"][0]["score"], 7.0);
                assert_eq!(tuned["full_query"]["candidate_multiplier"], 1);
                assert_eq!(fast.calls.load(Ordering::SeqCst), before.0 + 1);
                assert_eq!(quality.calls.load(Ordering::SeqCst), before.1 + 1);
                let request: serve::Request =
                    serde_json::from_str(r#"{"query":"needle","limit":1}"#).unwrap();
                let mut output = Vec::new();
                assert!(
                    serve::stream_one(
                        index.borrow(),
                        &cx,
                        &request,
                        1,
                        (Mode::Full, 1),
                        &mut output,
                        None,
                        &policy
                    )
                    .await
                    .unwrap()
                );
                let frames = frames(&output);
                assert_eq!(frames.last().unwrap()["status"], "complete");
                let pages: Vec<_> = frames.iter().filter(|f| f["event"] == "results").collect();
                assert_eq!(pages.len(), 2);
                assert_eq!(pages[0]["candidates"]["fast"], 1);
                assert_eq!(pages[1]["candidates"]["quality"], 1);
                assert_eq!(pages[1]["candidates"]["lexical"], 1);
                assert_eq!(pages[1]["results"], tuned["results"]);
                assert_eq!(pages[1]["full_query"], tuned["full_query"]);
                if partitioned {
                    assert_eq!(tuned["layout"], "sharded");
                    assert!(tuned["results"][0]["index"].is_null());
                }
            }
        }
    });
}

#[derive(Default)]
struct Scorer {
    ids: Mutex<Vec<String>>,
}
impl Reranker for Scorer {
    #[allow(clippy::unnecessary_literal_bound)]
    fn id(&self) -> &str {
        "cli-tuning-scorer"
    }
    fn model_name(&self) -> &str {
        self.id()
    }
    fn rerank<'a>(
        &'a self,
        _cx: &'a Cx,
        _text: &'a str,
        documents: &'a [RerankDocument],
    ) -> SearchFuture<'a, Vec<RerankScore>> {
        Box::pin(async move {
            *self.ids.lock().unwrap() = documents.iter().map(|d| d.doc_id.clone()).collect();
            Ok(documents
                .iter()
                .enumerate()
                .map(|(original_rank, document)| RerankScore {
                    doc_id: document.doc_id.clone(),
                    original_rank,
                    score: if document.doc_id == "a" { 1.0 } else { 0.0 },
                    raw_logit: None,
                })
                .collect())
        })
    }
}

#[test]
fn tuned_rerank_windows_and_intersected_filters_preserve_rows_and_owned_sources() {
    run_test_with_cx(|cx| async move {
        for partitioned in [false, true] {
            let (_parent, index, _, _) = fixture(&cx, partitioned, false).await;
            let mut policy = policy(&cx);
            let scorer = Arc::new(Scorer::default());
            policy.rerank = Some(query::Rerank::new(scorer.clone(), 2).unwrap());
            let unfiltered = query::buffered(
                index.borrow(),
                &cx,
                "needle",
                Mode::Full,
                1,
                [None, None],
                &policy,
            )
            .await
            .unwrap();
            assert_eq!(unfiltered["phase"], "reranked");
            assert_eq!(unfiltered["results"][0]["doc_id"], "a");
            assert_eq!(scorer.ids.lock().unwrap().len(), 2);
            if partitioned {
                assert_eq!(unfiltered["results"][0]["fast_row"]["shard"], 0);
                assert_eq!(unfiltered["results"][0]["quality_row"]["shard"], 0);
            }
            let base = filter::Filter::parse(r#"{"ids":["z"]}"#).unwrap();
            let requested = filter::Filter::parse(r#"{"ids":["a","z"]}"#).unwrap();
            let scoped = query::buffered(
                index.borrow(),
                &cx,
                "needle",
                Mode::Full,
                1,
                [Some(&base), Some(&requested)],
                &policy,
            )
            .await
            .unwrap();
            assert_eq!(scoped["results"][0]["doc_id"], "z");
            assert_eq!(*scorer.ids.lock().unwrap(), ["z"]);
            assert_eq!(scoped["scope"]["eligible_documents"], 1);
            if partitioned {
                assert_eq!(scoped["results"][0]["fast_row"]["shard"], 2);
                assert_eq!(scoped["results"][0]["quality_row"]["shard"], 2);
            }
            assert_eq!(scoped["results"][0]["metadata"]["epoch"], "old");
        }
    });
}

#[test]
fn warm_session_reports_fixed_full_policy_without_changing_primary_modes() {
    run_test_with_cx(|cx| async move {
        for partitioned in [false, true] {
            let (_parent, index, fast, quality) = fixture(&cx, partitioned, true).await;
            let serving = live::Serving::new(&cx, index).unwrap();
            let policy = policy(&cx);
            let before = (
                fast.calls.load(Ordering::SeqCst),
                quality.calls.load(Ordering::SeqCst),
            );
            let mut input = Cursor::new(concat!(
                "{\"query\":\"needle\",\"mode\":\"fast\"}\n",
                "{\"query\":\"needle\",\"mode\":\"quality\"}\n",
                "{\"query\":\"needle\",\"mode\":\"full\"}\n",
                "{\"query\":\"needle\",\"full_query\":{\"candidate_multiplier\":100}}\n",
                "{\"query\":\"needle\",\"mode\":\"full\"}\nexit\n",
            ));
            let mut output = Vec::new();
            serve::run_with_controls(
                serving.borrow(),
                &cx,
                &mut input,
                &mut output,
                (Mode::Full, 1),
                serve::Controls::default(),
                None,
                &policy,
            )
            .await
            .unwrap();
            let frames = frames(&output);
            assert_eq!(frames[0]["event"], "ready");
            assert_eq!(frames[0]["full_query"]["candidate_multiplier"], 1);
            for frame in &frames {
                match frame["request"].as_u64() {
                    Some(1 | 2) => assert!(frame.get("full_query").is_none()),
                    Some(3 | 5) if frame["event"] == "results" => {
                        assert_eq!(frame["full_query"]["candidate_multiplier"], 1);
                        assert_eq!(frame["candidates"]["fast"], 1);
                    }
                    Some(4) => assert_eq!(frame["ok"], false),
                    _ => {}
                }
            }
            assert_eq!(fast.calls.load(Ordering::SeqCst), before.0 + 3);
            assert_eq!(quality.calls.load(Ordering::SeqCst), before.1 + 3);
        }
    });
}

#[test]
fn tuned_zero_weight_quality_failure_is_not_buffered_success_or_stream_completion() {
    run_test_with_cx(|cx| async move {
        for partitioned in [false, true] {
            let (_parent, index, _, quality) = fixture(&cx, partitioned, true).await;
            let policy = policy(&cx);
            quality.mode.store(1, Ordering::SeqCst);
            assert!(
                query::buffered(
                    index.borrow(),
                    &cx,
                    "needle",
                    Mode::Full,
                    1,
                    [None, None],
                    &policy
                )
                .await
                .is_err()
            );
            let request: serve::Request = serde_json::from_str(r#"{"query":"needle"}"#).unwrap();
            let mut output = Vec::new();
            assert!(
                !serve::stream_one(
                    index.borrow(),
                    &cx,
                    &request,
                    1,
                    (Mode::Full, 1),
                    &mut output,
                    None,
                    &policy
                )
                .await
                .unwrap()
            );
            let frames = frames(&output);
            assert_eq!(frames[2]["phase"], "refinement_failed");
            assert_eq!(frames[1]["results"], frames[2]["results"]);
            assert_eq!(frames.last().unwrap()["status"], "degraded");
        }
    });
}

#[test]
fn tuned_queries_keep_the_original_timer_and_release_pending_quality_on_expiry() {
    run_test_with_cx(|cx| async move {
        for partitioned in [false, true] {
            let (_parent, index, _, quality) = fixture(&cx, partitioned, true).await;
            let mut policy = policy(&cx);
            let clock = Arc::new(VirtualClock::new());
            let timer = TimerDriverHandle::with_virtual_clock(clock.clone());
            policy.maximum_ms = Some(10);
            policy.timer = Some(timer.clone());
            quality.mode.store(2, Ordering::SeqCst);
            let before = quality.drops.load(Ordering::SeqCst);
            let mut waiting = Box::pin(query::buffered(
                index.borrow(),
                &cx,
                "needle",
                Mode::Full,
                1,
                [None, None],
                &policy,
            ));
            assert!(
                waiting
                    .as_mut()
                    .poll(&mut Context::from_waker(Waker::noop()))
                    .is_pending()
            );
            assert_eq!(timer.pending_count(), 1);
            clock.advance(10_000_000);
            let _ = timer.process_timers();
            let error = waiting.await.unwrap_err();
            assert_eq!(query::failure(error.as_ref())["code"], "search_timeout");
            assert_eq!(quality.drops.load(Ordering::SeqCst), before + 1);
            assert!(timer.is_empty());
            assert!(!cx.is_cancel_requested());
        }
    });
}
