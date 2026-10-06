//! Executable paths with real partition writers, global Quill and receipt reopen.
//! Identified deterministic providers test lifecycle and dispatch, not relevance.
use super::*;
use crate::{
    Arc, Mode, Options, SCHEMA, Selection, encode, filter, fs, new_path, query, save_selection,
    serve,
};
use asupersync::test_utils::run_test_with_cx;
use frankensearch::{
    Embedder, ModelCategory, RerankDocument, RerankScore, Reranker, SearchError, SearchFuture,
    SearchResult,
};
use frankensearch_core::generation::EmbeddingIdentityBundleV1;
use frankensearch_core::traits::IdentityBoundEmbedding;
use frankensearch_index::ValidatedFsviBytes;
use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Provider {
    identity: EmbeddingIdentityBundleV1,
    queries: AtomicUsize,
    submitted: AtomicUsize,
    fail: AtomicBool,
}

impl Provider {
    fn new(name: &str, dimension: u32) -> Arc<Self> {
        Arc::new(Self {
            identity: EmbeddingIdentityBundleV1::explicit_test_model(name, dimension),
            queries: AtomicUsize::new(0),
            submitted: AtomicUsize::new(0),
            fail: AtomicBool::new(false),
        })
    }

    fn values(&self, text: &str) -> Vec<f32> {
        let mut values = vec![0.0; self.dimension()];
        values[0] = if text.contains("private") { 2.0 } else { 1.0 };
        values[1] = if text.contains("last") { 0.5 } else { 0.0 };
        values
    }
}

impl Embedder for Provider {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.queries.fetch_add(1, Ordering::SeqCst);
            if self.fail.load(Ordering::SeqCst) {
                return Err(SearchError::EmbeddingFailed {
                    model: self.id().to_owned(),
                    source: "injected query failure".into(),
                });
            }
            Ok(self.values(text))
        })
    }

    fn embed_batch_bound<'a>(
        &'a self,
        _cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<IdentityBoundEmbedding>> {
        Box::pin(async move {
            self.submitted.fetch_add(texts.len(), Ordering::SeqCst);
            Ok(texts.iter().map(|text| IdentityBoundEmbedding {
                values: self.values(text), identity: self.identity.clone(),
            }).collect())
        })
    }

    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> { Ok(&self.identity) }
    fn id(&self) -> &str { &self.identity.space.logical_model_id }
    fn model_name(&self) -> &str { self.id() }
    fn dimension(&self) -> usize { usize::try_from(self.identity.space.dimension).unwrap() }
    fn is_semantic(&self) -> bool { false }
    fn category(&self) -> ModelCategory { ModelCategory::HashEmbedder }
}

fn options() -> Options {
    Options::parse(["index", "--index-dir", "unused", "--receipt", "unused.json", "--shard-size", "2"]
        .into_iter().map(str::to_owned)).unwrap().unwrap()
}

fn models() -> (Models, Arc<Provider>, Arc<Provider>) {
    let fast = Provider::new("cli-sharded-fast", 2);
    let quality = Provider::new("cli-sharded-quality", 3);
    (Models { fast: fast.clone(), quality: Some(quality.clone()) }, fast, quality)
}

fn documents() -> Vec<IndexableDocument> {
    let mut documents = (0..12).map(|id| {
        IndexableDocument::new(format!("private-{id:02}"), "needle private")
            .with_metadata("tenant", "private")
    }).collect::<Vec<_>>();
    for id in ["public-a", "public-b", "public-c", "public-last"] {
        documents.push(IndexableDocument::new(id, format!("needle {id} {}", "padding ".repeat(20)))
            .with_title(format!("title {id}"))
            .with_metadata("tenant", "public"));
    }
    documents
}

fn frames(bytes: &[u8]) -> Vec<serde_json::Value> {
    bytes.split(|byte| *byte == b'\n').filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap()).collect()
}

fn assert_rows(index: &NativeBuiltShardedHybridIndex, page: &serde_json::Value) {
    for hit in page["results"].as_array().unwrap() {
        assert!(hit["index"].is_null(), "a bare index cannot identify a shard");
        for (field, quality) in [("fast_row", false), ("quality_row", true)] {
            let row = &hit[field];
            if row.is_null() { continue; }
            let ordinal = usize::try_from(row["shard"].as_u64().unwrap()).unwrap();
            let physical = usize::try_from(row["physical_row"].as_u64().unwrap()).unwrap();
            let partition = &index.vectors().partitions()[ordinal];
            let tier = if quality { partition.quality().unwrap() } else { partition.fast() };
            // Interpret the emitted coordinates against the independently parsed
            // persisted FSVI image, not a position in the ranked result page.
            let owner = ValidatedFsviBytes::from_arc(
                fs::read(tier.vector_path()).unwrap().into(), tier.binding(),
            ).unwrap();
            assert_eq!(owner.doc_id_at(physical).unwrap(), hit["doc_id"].as_str().unwrap());
        }
    }
}

#[test]
fn shard_option_and_complete_partition_budget_are_explicit() {
    assert_eq!(options().shard_size, Some(2));
    for args in [
        vec!["index", "--index-dir", "x", "--receipt", "r", "--shard-size", "0"],
        vec!["index", "--index-dir", "x", "--receipt", "r", "--shard-size", "100001"],
        vec!["search", "--receipt", "r", "--query", "q", "--shard-size", "2"],
        vec!["serve", "--receipt", "r", "--shard-size", "2"],
    ] {
        assert!(Options::parse(args.into_iter().map(str::to_owned)).is_err());
    }
    assert!(validate_size(1, 1024).is_ok());
    assert!(validate_size(1, 1025).is_err());
    assert!(validate_size(2, 0).is_ok());
    assert!(validate_size(2, MAX_DOCUMENTS + 1).is_err());
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (models, fast, quality) = models();
        let path = root.path().join("rejected");
        let docs = (0..1025).map(|i| IndexableDocument::new(i.to_string(), "needle")).collect();
        assert!(build(&cx, &options(), &path, docs, models, 1).await.is_err());
        assert!(!path.exists());
        assert_eq!(fast.submitted.load(Ordering::SeqCst), 0);
        assert_eq!(quality.submitted.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn persisted_sharded_receipts_drive_all_modes_and_embed_once_per_tier() {
    run_test_with_cx(|cx| async move {
        for exact in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let mut opts = options();
            opts.exact = exact;
            let (models, fast, quality) = models();
            let (built, selected) = build(&cx, &opts, &root.path().join("index"), documents(), models.clone(), 2)
                .await.unwrap();
            assert_eq!(built.vectors().partitions().len(), 8);
            assert!(built.vectors().partitions().iter().all(|part|
                part.fast().graph_path().is_some() != exact));
            let receipt = root.path().join("selection.json");
            save_selection(&selected, &receipt).unwrap();
            drop(built);
            let selected = Selection::read(&receipt).unwrap();
            assert_eq!(selected.schema, SELECTION_SCHEMA);
            assert!(selected.open(&cx, models.clone()).await.is_err());
            let reopened = Opened::open(&cx, &selected, models.clone()).await.unwrap();
            let Opened::Sharded(index) = reopened else { panic!("receipt must select all shards"); };
            assert_eq!(fast.submitted.load(Ordering::SeqCst), 16);
            assert_eq!(quality.submitted.load(Ordering::SeqCst), 16);
            for (mode, expected_phase, fast_delta, quality_delta) in [
                (Mode::Fast, "initial", 1, 0), (Mode::Quality, "quality", 0, 1),
                (Mode::Full, "refined", 1, 1),
            ] {
                let before = (fast.queries.load(Ordering::SeqCst), quality.queries.load(Ordering::SeqCst));
                let page = query::buffered(index.as_ref(), &cx, "needle", mode, 4,
                    [None, None], &query::Policy::default()).await.unwrap();
                assert_eq!(page["schema"], SCHEMA);
                assert_eq!(page["phase"], expected_phase);
                assert_eq!(page["layout"], "sharded");
                assert_eq!(page["partitions"], 8);
                assert_eq!(page["generation"], serde_json::to_value(selected.generation).unwrap());
                assert_eq!(page["results"].as_array().unwrap().len(), 4);
                assert_rows(&index, &page);
                assert_eq!(fast.queries.load(Ordering::SeqCst), before.0 + fast_delta);
                assert_eq!(quality.queries.load(Ordering::SeqCst), before.1 + quality_delta);
            }
            let foreign = Models { fast: Provider::new("different-producer", 2), quality: models.quality.clone() };
            assert!(open(&cx, &selected, foreign).await.is_err());
            let mut malformed = selected;
            malformed.snapshot.sha256[0] ^= 1;
            assert!(Opened::open(&cx, &malformed, models).await.is_err());
            assert_eq!(index.vectors().document_count(), 16);
        }
    });
}

#[test]
fn sharded_filters_refill_before_cutoffs_and_intersect_without_widening() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (models, fast, quality) = models();
        let (index, _) = build(&cx, &options(), &root.path().join("index"), documents(), models, 3).await.unwrap();
        let global = index.lexical().search(&cx, "needle", 3).await.unwrap();
        assert!(global.iter().all(|hit| hit.doc_id.starts_with("private-")));
        let base = filter::Filter::parse(r#"{"metadata":{"tenant":"public"}}"#).unwrap();
        let request = filter::Filter::parse(r#"{"ids":["public-a","public-c","public-last","private-00"]}"#).unwrap();
        for mode in [Mode::Fast, Mode::Quality, Mode::Full] {
            let page = query::buffered(&index, &cx, "needle", mode, 3, [Some(&base), Some(&request)],
                &query::Policy::default()).await.unwrap();
            assert_eq!(page["scope"]["eligible_documents"], 3);
            assert_eq!(page["results"].as_array().unwrap().len(), 3);
            for hit in page["results"].as_array().unwrap() {
                assert!(hit["doc_id"].as_str().unwrap().starts_with("public-"));
                assert_ne!(hit["doc_id"], "public-b");
            }
            assert_rows(&index, &page);
        }
        let none = filter::Filter::parse(r#"{"ids":[]}"#).unwrap();
        let before = (fast.queries.load(Ordering::SeqCst), quality.queries.load(Ordering::SeqCst));
        let page = query::buffered(&index, &cx, "needle", Mode::Full, 3, [Some(&base), Some(&none)],
            &query::Policy::default()).await.unwrap();
        assert!(page["results"].as_array().unwrap().is_empty());
        assert_eq!(fast.queries.load(Ordering::SeqCst), before.0);
        assert_eq!(quality.queries.load(Ordering::SeqCst), before.1);
    });
}

#[derive(Default)]
struct Scorer { calls: AtomicUsize, seen: Mutex<Vec<RerankDocument>> }
impl Reranker for Scorer {
    fn id(&self) -> &'static str { "sharded-cli-scorer" }
    fn model_name(&self) -> &str { self.id() }
    fn rerank<'a>(&'a self, _cx: &'a Cx, _query: &'a str, docs: &'a [RerankDocument])
        -> SearchFuture<'a, Vec<RerankScore>> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            *self.seen.lock().unwrap() = docs.to_vec();
            Ok(docs.iter().enumerate().map(|(original_rank, doc)| RerankScore {
                doc_id: doc.doc_id.clone(), original_rank,
                score: if doc.doc_id == "public-last" { 1.0 } else { 0.0 }, raw_logit: None,
            }).collect())
        })
    }
}

struct Delivery {
    bytes: Vec<u8>, quality: Arc<Provider>, scorer: Arc<Scorer>,
    clock: Option<Arc<asupersync::time::VirtualClock>>,
}
impl Write for Delivery {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.bytes.extend_from_slice(bytes); Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        let frames = frames(&self.bytes);
        let last = frames.last().unwrap();
        if last["phase"] == "initial" {
            assert_eq!(self.quality.queries.load(Ordering::SeqCst), 0);
            assert_eq!(self.scorer.calls.load(Ordering::SeqCst), 0);
            if let Some(clock) = &self.clock { clock.advance(10_000_000); }
        }
        if last["phase"] == "refined" { assert_eq!(self.scorer.calls.load(Ordering::SeqCst), 0); }
        Ok(())
    }
}

fn request() -> serve::Request {
    serde_json::from_value(serde_json::json!({
        "id": "q", "query": "needle", "mode": "full", "limit": 1,
        "filter": {"metadata": {"tenant": "public"}},
    })).unwrap()
}

#[test]
fn shared_stream_delivers_before_each_model_and_preserves_reranked_shard_rows() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (models, fast, quality) = models();
        let (index, _) = build(&cx, &options(), &root.path().join("index"), documents(), models, 2).await.unwrap();
        let scorer = Arc::new(Scorer::default());
        let policy = query::Policy { rerank: Some(query::Rerank::new(scorer.clone(), 4).unwrap()),
            ..query::Policy::default() };
        let mut delivery = Delivery { bytes: Vec::new(), quality: quality.clone(), scorer: scorer.clone(), clock: None };
        assert!(serve::stream_one(&index, &cx, &request(), 1, (Mode::Full, 1), &mut delivery, None, &policy).await.unwrap());
        let output = frames(&delivery.bytes);
        assert_eq!(output.len(), 5);
        assert_eq!(output[0]["event"], "started");
        for (seq, phase) in [(1, "initial"), (2, "refined"), (3, "reranked")] {
            assert_eq!(output[seq]["phase"], phase);
            assert_eq!(output[seq]["seq"], seq);
            assert_eq!(output[seq]["layout"], "sharded");
            assert_rows(&index, &output[seq]);
        }
        assert_eq!(output[3]["results"][0]["doc_id"], "public-last");
        assert_eq!(output[3]["evaluated"], 4);
        assert_eq!(output[4]["status"], "complete");
        assert_eq!(fast.queries.load(Ordering::SeqCst), 1);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 1);
        let seen = scorer.seen.lock().unwrap();
        assert_eq!(seen.len(), 4);
        for doc in seen.iter() {
            assert!(doc.doc_id.starts_with("public-"));
            assert_eq!(doc.text, index.vectors().document(&doc.doc_id).unwrap().content);
        }
    });
}

#[test]
fn sharded_timeout_after_initial_starts_no_quality_or_rerank_and_later_query_recovers() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (models, _, quality) = models();
        let (index, _) = build(&cx, &options(), &root.path().join("index"), documents(), models, 2).await.unwrap();
        let clock = Arc::new(asupersync::time::VirtualClock::new());
        let timer = asupersync::time::TimerDriverHandle::with_virtual_clock(clock.clone());
        let scorer = Arc::new(Scorer::default());
        let policy = query::Policy { maximum_ms: Some(10), timer: Some(timer),
            rerank: Some(query::Rerank::new(scorer.clone(), 4).unwrap()) };
        let mut output = Delivery { bytes: Vec::new(), quality: quality.clone(), scorer: scorer.clone(), clock: Some(clock) };
        assert!(!serve::stream_one(&index, &cx, &request(), 1, (Mode::Full, 1), &mut output, None, &policy).await.unwrap());
        let frames = frames(&output.bytes);
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[1]["phase"], "initial");
        assert_eq!(frames[2]["status"], "timed_out");
        assert_eq!(frames[2]["partial_results"], true);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
        assert_eq!(scorer.calls.load(Ordering::SeqCst), 0);
        assert!(!cx.is_cancel_requested());
        let mut next = Vec::new();
        assert!(serve::stream_one(&index, &cx, &request(), 2, (Mode::Full, 1), &mut next, None, &policy).await.unwrap());
        assert_eq!(quality.queries.load(Ordering::SeqCst), 1);
        assert_eq!(scorer.calls.load(Ordering::SeqCst), 1);
    });
}

#[test]
fn missing_late_graph_and_occupied_or_sealed_destinations_never_trigger_fallback() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (models, fast, _) = models();
        let (index, selected) = build(&cx, &options(), &root.path().join("index"), documents(), models.clone(), 2).await.unwrap();
        assert!(new_path(&selected.directory.join("new-file")).is_err());
        assert!(new_path(&selected.directory.join("lexical/new-file")).is_err());
        let graph = index.vectors().partitions().last().unwrap().fast().graph_path().unwrap();
        let moved = root.path().join("saved-graph");
        fs::rename(graph, &moved).unwrap();
        assert!(Opened::open(&cx, &selected, models.clone()).await.is_err());
        assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
        assert!(query::buffered(&index, &cx, "needle", Mode::Fast, 1, [None, None], &query::Policy::default()).await.is_ok());
        fs::rename(&moved, graph).unwrap();
        assert!(Opened::open(&cx, &selected, models).await.is_ok());
        let path = root.path().join("unknown.json");
        let mut unknown = selected;
        unknown.schema = "frankensearch.native.sharded-selection.v999".to_owned();
        fs::write(&path, encode(&unknown, 64 * 1024).unwrap()).unwrap();
        assert!(Selection::read(&path).is_err());
    });
}

#[test]
fn ordinary_output_remains_ordinary_through_the_shared_command_dispatch() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (models, _, _) = models();
        let mut opts = options();
        opts.shard_size = None;
        let (index, selected) = crate::build(&cx, &opts, &root.path().join("single"), documents(), models.clone()).await.unwrap();
        let expected = index.search_refined(&cx, "needle", 3).await.unwrap();
        let opened = Opened::open(&cx, &selected, models).await.unwrap();
        let actual = query::buffered(opened.borrow(), &cx, "needle", Mode::Full, 3,
            [None, None], &query::Policy::default()).await.unwrap();
        assert_eq!(actual["results"], serde_json::to_value(expected).unwrap());
        assert!(actual.get("layout").is_none());
        for hit in actual["results"].as_array().unwrap() {
            assert!(hit.get("fast_row").is_none());
            assert!(hit.get("quality_row").is_none());
        }
    });
}

async fn successor(
    cx: &Cx,
    root: &Path,
    index: &NativeBuiltShardedHybridIndex,
) -> (NativeBuiltShardedHybridIndex, Selection, std::path::PathBuf) {
    let mut revised = index.vectors().document("public-a").unwrap().clone();
    revised.metadata.insert("tenant".to_owned(), "private".to_owned());
    let generation = new_generation(index.vectors().generation().sequence + 1).unwrap();
    let next = index.begin_update(cx, root.join("successor"), generation).unwrap()
        .upsert_document(revised)
        .delete_document("public-last")
        .upsert_document(IndexableDocument::new("public-arrived", "needle arrival")
            .with_metadata("tenant", "public"))
        .build_sharded_hybrid(cx, 5).await.unwrap();
    let receipt = next.seal_for_reopen(cx).unwrap();
    let first = &next.vectors().partitions()[0];
    let selected = Selection {
        schema: SELECTION_SCHEMA.to_owned(),
        directory: next.vectors().directory().to_path_buf(), generation,
        snapshot: SnapshotReceipt { byte_len: receipt.byte_len, sha256: receipt.sha256 },
        documents: next.vectors().document_count(),
        fast_producer: first.fast().embedder().identity().unwrap().fingerprint(),
        quality_producer: first.quality().map(|tier| tier.embedder().identity().unwrap().fingerprint()),
    };
    let path = root.join("successor.json");
    save_selection(&selected, &path).unwrap();
    (next, selected, path)
}

fn messages(values: &[serde_json::Value]) -> std::io::Cursor<Vec<u8>> {
    let mut input = Vec::new();
    for value in values { input.extend(encode(value, 1024 * 1024).unwrap()); }
    std::io::Cursor::new(input)
}

fn activation(old: &Selection, receipt: &Path) -> serde_json::Value {
    serde_json::json!({"op": "activate", "id": "activate", "receipt": receipt,
        "expected_generation": old.generation})
}

#[test]
fn warm_sharded_server_activates_complete_successors_and_refreshes_scopes_without_model_loading() {
    use frankensearch::native_ann::builder::sharded::live::NativeLiveShardedHybridIndex;
    run_test_with_cx(|cx| async move {
        for exact in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let mut opts = options(); opts.exact = exact;
            let (models, fast, quality) = models();
            let (index, selected) = build(&cx, &opts, &root.path().join("initial"), documents(), models, 2).await.unwrap();
            let (next, next_selection, receipt) = successor(&cx, root.path(), &index).await;
            let receipt_bytes = fs::read(&receipt).unwrap();
            let submissions = (fast.submitted.load(Ordering::SeqCst), quality.submitted.load(Ordering::SeqCst));
            drop(next);
            let live = NativeLiveShardedHybridIndex::new(&cx, index).unwrap();
            let old = live.snapshot(&cx).await.unwrap();
            let mut input = messages(&[
                serde_json::json!({"id": "before", "query": "needle", "limit": 100}),
                activation(&selected, &receipt),
                serde_json::json!({"id": "after", "query": "needle", "mode": "quality", "limit": 100}),
                serde_json::json!({"op": "status", "id": "status"}),
                activation(&selected, &receipt),
                serde_json::json!({"id": "again", "query": "needle", "limit": 100}),
            ]);
            let base_filter = filter::Filter::parse(r#"{"metadata":{"tenant":"public"}}"#).unwrap();
            let mut output = Vec::new();
            serve::run_with_controls(&live, &cx, &mut input, &mut output, (Mode::Full, 10),
                serve::Controls { activation: true, updates: false }, Some(&base_filter),
                &query::Policy::default()).await.unwrap();
            let frames = frames(&output);
            assert_eq!(frames[0]["event"], "ready");
            assert_eq!(frames[0]["layout"], "sharded");
            assert_eq!(frames[0]["partitions"], 8);
            assert_eq!(frames[0]["fast_native_hnsw"], !exact);
            assert_eq!(frames[0]["updates_enabled"], false);
            let current = live.snapshot(&cx).await.unwrap();
            assert_eq!(current.generation(), next_selection.generation);
            for frame in frames.iter().filter(|frame| frame["event"] == "results") {
                let before = frame["id"] == "before";
                assert_eq!(frame["generation"], serde_json::to_value(if before { selected.generation } else { next_selection.generation }).unwrap());
                assert_eq!(frame["scope"]["eligible_documents"], if before { 4 } else { 3 });
                assert_eq!(frame["results"].as_array().unwrap().len(), if before { 4 } else { 3 });
                assert_rows(if before { old.index() } else { current.index() }, frame);
                if !before {
                    assert!(frame["results"].as_array().unwrap().iter().all(|hit|
                        hit["doc_id"] != "public-a" && hit["doc_id"] != "public-last"));
                }
            }
            let activations = frames.iter().filter(|frame| frame["operation"] == "activate").collect::<Vec<_>>();
            assert_eq!(activations.len(), 2);
            assert_eq!(activations[0]["selection_changed"], true);
            assert_eq!(activations[0]["partitions"], 4);
            assert_eq!(activations[1]["ok"], false);
            assert_eq!(activations[1]["selection_changed"], false);
            assert_eq!(fast.queries.load(Ordering::SeqCst), 2);
            assert_eq!(quality.queries.load(Ordering::SeqCst), 3);
            assert_eq!(fast.submitted.load(Ordering::SeqCst), submissions.0);
            assert_eq!(quality.submitted.load(Ordering::SeqCst), submissions.1);
            assert_eq!(fs::read(&receipt).unwrap(), receipt_bytes);
            assert!(old.index().vectors().document("public-last").is_some());
            assert!(current.index().vectors().document("public-last").is_none());
        }
    });
}

#[test]
fn sharded_controls_fail_closed_before_paths_and_invalid_messages_do_not_become_queries() {
    use frankensearch::native_ann::builder::sharded::live::NativeLiveShardedHybridIndex;
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (models, fast, _) = models();
        let (index, selected) = build(&cx, &options(), &root.path().join("index"), documents(), models, 3).await.unwrap();
        let live = NativeLiveShardedHybridIndex::new(&cx, index).unwrap();
        let mut output = Vec::new();
        let mut input = messages(&[serde_json::json!({"query":"needle"})]);
        assert!(serve::run_with_controls(&live, &cx, &mut input, &mut output, (Mode::Full, 1),
            serve::Controls { activation: false, updates: true }, None, &query::Policy::default()).await.is_err());
        assert_eq!(input.position(), 0);
        assert!(output.is_empty());
        let absent = root.path().join("missing.json");
        let mut input = messages(&[
            activation(&selected, &absent),
            serde_json::json!({"op":"update", "id":"denied", "expected_generation":selected.generation,
                "index_dir":root.path().join("must-not-exist"), "new_receipt":root.path().join("must-not-exist.json"),
                "changes":[{"op":"upsert", "id":"new", "content":"needle"}]}),
            serde_json::json!({"op":"activate", "query":"needle"}),
            serde_json::json!({"query":"needle", "mode":"fast"}),
        ]);
        serve::run_with_controls(&live, &cx, &mut input, &mut output, (Mode::Full, 1),
            serve::Controls::default(), None, &query::Policy::default()).await.unwrap();
        let frames = frames(&output);
        assert!(frames[1]["error"].as_str().unwrap().contains("activation is disabled"));
        assert_eq!(frames[2]["operation"], "update");
        assert_eq!(frames[2]["id"], "denied");
        assert_eq!(frames[2]["selection_changed"], false);
        assert_eq!(frames[3]["ok"], false);
        assert_eq!(frames.last().unwrap()["status"], "complete");
        assert_eq!(fast.queries.load(Ordering::SeqCst), 1);
        assert_eq!(live.snapshot(&cx).await.unwrap().generation(), selected.generation);
        assert!(!root.path().join("must-not-exist").exists());
        assert!(!root.path().join("must-not-exist.json").exists());
    });
}

#[test]
fn sharded_activation_refuses_missing_late_artifacts_and_then_recovers_without_partial_install() {
    use frankensearch::native_ann::builder::sharded::live::NativeLiveShardedHybridIndex;
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (models, fast, quality) = models();
        let (index, selected) = build(&cx, &options(), &root.path().join("initial"), documents(), models, 2).await.unwrap();
        let (next, next_selection, receipt) = successor(&cx, root.path(), &index).await;
        let last = next.vectors().partitions().last().unwrap().quality().unwrap().vector_path().to_path_buf();
        drop(next);
        let saved = root.path().join("saved-quality");
        fs::rename(&last, &saved).unwrap();
        let live = NativeLiveShardedHybridIndex::new(&cx, index).unwrap();
        let mut output = Vec::new();
        serve::run_with_controls(&live, &cx, &mut messages(&[
            activation(&selected, &receipt), serde_json::json!({"op":"status"}),
        ]), &mut output, (Mode::Full, 1), serve::Controls { activation: true, updates: false },
            None, &query::Policy::default()).await.unwrap();
        let failed = frames(&output);
        assert_eq!(failed[1]["ok"], false);
        assert_eq!(failed[1]["selection_changed"], false);
        assert_eq!(failed[2]["generation"], serde_json::to_value(selected.generation).unwrap());
        fs::rename(&saved, &last).unwrap();
        let mut repaired = Vec::new();
        serve::run_with_controls(&live, &cx, &mut messages(&[activation(&selected, &receipt)]),
            &mut repaired, (Mode::Full, 1), serve::Controls { activation: true, updates: false },
            None, &query::Policy::default()).await.unwrap();
        assert_eq!(frames(&repaired)[1]["ok"], true);
        assert_eq!(live.snapshot(&cx).await.unwrap().generation(), next_selection.generation);
        assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
        assert_eq!(quality.queries.load(Ordering::SeqCst), 0);
    });
}

#[test]
fn sharded_activation_lost_acknowledgement_never_rolls_back_or_consumes_another_request() {
    use frankensearch::native_ann::builder::sharded::live::NativeLiveShardedHybridIndex;
    struct BrokenActivationOutput { writes: usize }
    impl Write for BrokenActivationOutput {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.writes += 1;
            let frame: serde_json::Value = serde_json::from_slice(bytes).unwrap();
            if frame["operation"] == "activate" {
                assert_eq!(frame["ok"], true);
                return Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "lost acknowledgement"));
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (models, fast, _) = models();
        let (index, selected) = build(&cx, &options(), &root.path().join("initial"), documents(), models, 2).await.unwrap();
        let (next, next_selection, receipt) = successor(&cx, root.path(), &index).await;
        drop(next);
        let live = NativeLiveShardedHybridIndex::new(&cx, index).unwrap();
        let activate = activation(&selected, &receipt);
        let expected_read = encode(&activate, 1024 * 1024).unwrap().len() as u64;
        let mut input = messages(&[activate, serde_json::json!({"query":"needle"})]);
        let mut output = BrokenActivationOutput { writes: 0 };
        assert!(serve::run_with_controls(&live, &cx, &mut input, &mut output, (Mode::Full, 1),
            serve::Controls { activation: true, updates: false }, None, &query::Policy::default()).await.is_err());
        assert_eq!(output.writes, 2, "no contradictory error appended after failed success delivery");
        assert_eq!(input.position(), expected_read);
        assert_eq!(live.snapshot(&cx).await.unwrap().generation(), next_selection.generation);
        assert_eq!(fast.queries.load(Ordering::SeqCst), 0);
        let mut replay = Vec::new();
        serve::run_with_controls(&live, &cx, &mut messages(&[
            activation(&selected, &receipt), serde_json::json!({"op":"status"}),
        ]), &mut replay, (Mode::Full, 1), serve::Controls { activation: true, updates: false },
            None, &query::Policy::default()).await.unwrap();
        let replay = frames(&replay);
        assert_eq!(replay[1]["selection_changed"], false);
        assert_eq!(replay[2]["generation"], serde_json::to_value(next_selection.generation).unwrap());
    });
}

#[test]
fn sharded_external_activation_never_rebinds_an_inflight_scoped_rerank() {
    use frankensearch::native_ann::builder::sharded::live::NativeLiveShardedHybridIndex;
    use frankensearch::native_ann::NativeShardedSearchPhase;
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let (models, _, _) = models();
        let (index, selected) = build(&cx, &options(), &root.path().join("initial"), documents(), models, 2).await.unwrap();
        let (next, _, receipt) = successor(&cx, root.path(), &index).await;
        drop(next);
        let live = NativeLiveShardedHybridIndex::new(&cx, index).unwrap();
        let old = live.snapshot(&cx).await.unwrap();
        let scope = old.index().scope(&cx, |doc| Ok(doc.metadata["tenant"] == "public")).unwrap();
        let scorer = Scorer::default();
        let mut phases = scope.progressive_with_reranker(&cx, "needle", 1, &scorer, 4).unwrap();
        assert!(matches!(phases.next_phase().await.unwrap(), Some(NativeShardedSearchPhase::Initial { .. })));
        let mut output = Vec::new();
        serve::run_with_controls(&live, &cx, &mut messages(&[activation(&selected, &receipt)]),
            &mut output, (Mode::Full, 1), serve::Controls { activation: true, updates: false },
            None, &query::Policy::default()).await.unwrap();
        assert_eq!(frames(&output)[1]["ok"], true);
        assert!(matches!(phases.next_phase().await.unwrap(), Some(NativeShardedSearchPhase::Refined { .. })));
        let Some(NativeShardedSearchPhase::Reranked { results, .. }) = phases.next_phase().await.unwrap() else {
            panic!("old query's final stage must remain available");
        };
        assert_eq!(results[0].result.doc_id, "public-last");
        let page = serde_json::json!({"results": crate::cohort::Rows::from(results)});
        assert_rows(old.index(), &page);
        let current = live.snapshot(&cx).await.unwrap();
        assert!(current.index().vectors().document("public-last").is_none());
        let seen = scorer.seen.lock().unwrap();
        for input in seen.iter() {
            assert_eq!(input.text, scope.document(&input.doc_id).unwrap().content);
        }
    });
}
