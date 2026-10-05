//! Exercise source scoping through the executable's real native search routes.
use super::*;

#[test]
fn filter_option_is_validated_before_index_or_model_access() {
    let requested = options(&[
        "search",
        "--receipt",
        "unused",
        "--query",
        "retry",
        "--filter",
        r#"{"ids":[]}"#,
    ]);
    assert!(requested.filter.is_some());
    assert!(
        options(&[
            "serve",
            "--receipt",
            "unused",
            "--filter",
            r#"{"metadata":{"tenant":"A"}}"#
        ])
        .filter
        .is_some()
    );
    for args in [
        vec![
            "search",
            "--receipt",
            "unused",
            "--query",
            "retry",
            "--filter",
            "null",
        ],
        vec![
            "serve",
            "--receipt",
            "unused",
            "--filter",
            r#"{"unknown":"ignore"}"#,
        ],
        vec![
            "index",
            "--index-dir",
            "unused",
            "--receipt",
            "unused",
            "--filter",
            "{}",
        ],
    ] {
        assert!(Options::parse(args.into_iter().map(str::to_owned)).is_err());
    }
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn selective_scope_fills_pages_before_fusion_in_fast_quality_and_progressive_routes() {
    run_test_with_cx(|cx| async move {
        for exact in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let mut opts = options(&["index", "--index-dir", "unused", "--receipt", "unused"]);
            opts.exact = exact;
            let mut documents = (0..32)
                .map(|i| IndexableDocument::new(format!("a-{i:02}"), "retry network requests"))
                .collect::<Vec<_>>();
            for id in ["z-one", "z-two"] {
                let mut document = IndexableDocument::new(id, "retry network requests");
                document
                    .metadata
                    .insert("tenant".to_owned(), "allowed".to_owned());
                documents.push(document);
            }
            let (models, _, _) = fixture_models(true);
            let (index, _) = build(&cx, &opts, &root.path().join("index"), documents, models)
                .await
                .unwrap();
            // The eligible documents are outside the ordinary three-times-k
            // lexical window: post-filtering that window would return no hits.
            let ordinary_candidates = index
                .lexical()
                .search(&cx, "retry network", 6)
                .await
                .unwrap();
            assert_eq!(ordinary_candidates.len(), 6);
            assert!(
                ordinary_candidates
                    .iter()
                    .all(|hit| hit.doc_id.starts_with("a-"))
            );
            let restricted =
                filter::Filter::parse(r#"{"id_prefix":"z-","metadata":{"tenant":"allowed"}}"#)
                    .unwrap();
            for mode in [Mode::Fast, Mode::Quality, Mode::Full] {
                let page = search(
                    &index,
                    &cx,
                    "retry network",
                    mode,
                    2,
                    [None, Some(&restricted)],
                )
                .await
                .unwrap();
                let ids = page["results"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|hit| hit["doc_id"].as_str().unwrap())
                    .collect::<BTreeSet<_>>();
                assert_eq!(ids, BTreeSet::from(["z-one", "z-two"]));
                assert_eq!(page["scope"]["eligible_documents"], 2);
            }
            let request = serve::Request {
                id: None,
                query: "retry network".to_owned(),
                mode: None,
                limit: Some(2),
                filter: Some(restricted),
                timeout_ms: None,
            };
            let mut output = Vec::new();
            assert!(
                serve::stream_one(
                    &index,
                    &cx,
                    &request,
                    1,
                    (Mode::Full, 2),
                    &mut output,
                    None,
                    &query::Policy::default()
                )
                .await
                .unwrap()
            );
            let frames = output_frames(&output);
            for frame in &frames {
                if let Some(results) = frame["results"].as_array() {
                    assert_eq!(results.len(), 2);
                    assert!(
                        results
                            .iter()
                            .all(|hit| hit["doc_id"].as_str().unwrap().starts_with("z-"))
                    );
                    assert_eq!(frame["scope"]["eligible_documents"], 2);
                }
            }
            assert_eq!(frames[1]["phase"], "initial");
            assert_eq!(frames[2]["phase"], "refined");
            let ordinary = search(&index, &cx, "retry network", Mode::Full, 2, [None, None])
                .await
                .unwrap();
            assert!(
                ordinary.get("scope").is_none(),
                "unscoped output remains unchanged"
            );
        }
    });
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn empty_and_intersected_scopes_skip_inference_without_widening() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let opts = options(&["index", "--index-dir", "unused", "--receipt", "unused"]);
        let (models, fast, quality) = fixture_models(true);
        let (index, _) = build(&cx, &opts, &root.path().join("index"), source(), models)
            .await
            .unwrap();
        let before = (
            fast.load(Ordering::Relaxed),
            quality.load(Ordering::Relaxed),
        );
        let empty = filter::Filter::parse(r#"{"ids":[]}"#).unwrap();
        let owner = filter::Filter::parse(r#"{"metadata":{"language":"rust"}}"#).unwrap();
        let other = filter::Filter::parse(r#"{"ids":["garden.md"]}"#).unwrap();
        for filters in [[None, Some(&empty)], [Some(&owner), Some(&other)]] {
            for mode in [Mode::Fast, Mode::Quality, Mode::Full] {
                let page = search(&index, &cx, "retry", mode, 10, filters)
                    .await
                    .unwrap();
                assert!(page["results"].as_array().unwrap().is_empty());
                assert_eq!(page["scope"]["eligible_documents"], 0);
            }
            let request = serve::Request {
                id: None,
                query: "retry".to_owned(),
                mode: None,
                limit: None,
                filter: filters[1].cloned(),
                timeout_ms: None,
            };
            let mut output = Vec::new();
            assert!(
                serve::stream_one(
                    &index,
                    &cx,
                    &request,
                    1,
                    (Mode::Full, 10),
                    &mut output,
                    filters[0],
                    &query::Policy::default()
                )
                .await
                .unwrap()
            );
            for frame in output_frames(&output) {
                if let Some(results) = frame["results"].as_array() {
                    assert!(results.is_empty());
                }
            }
        }
        assert_eq!(
            (
                fast.load(Ordering::Relaxed),
                quality.load(Ordering::Relaxed)
            ),
            before
        );
        cx.set_cancel_requested(true);
        let rejected = search(&index, &cx, "retry", Mode::Full, 10, [Some(&owner), None]).await;
        cx.set_cancel_requested(false);
        assert!(rejected.is_err());
        assert_eq!(
            (
                fast.load(Ordering::Relaxed),
                quality.load(Ordering::Relaxed)
            ),
            before
        );
    });
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn server_default_scope_cannot_be_overridden_and_is_recomputed_after_activation() {
    run_test_with_cx(|cx| async move {
        let root = tempfile::tempdir().unwrap();
        let opts = options(&["index", "--index-dir", "unused", "--receipt", "unused"]);
        let (models, fast, quality) = fixture_models(true);
        let (old, old_selection) = build(&cx, &opts, &root.path().join("old"), source(), models)
            .await
            .unwrap();
        let edits = update::read_edits(&mut Cursor::new(
            r#"{"op":"upsert","id":"retry.rs","content":"retry network requests with backoff","metadata":{"language":"python"}}"#,
        )).unwrap();
        let (new, next) = update::apply(
            &cx,
            &old_selection,
            &old,
            &root.path().join("new"),
            edits,
            8,
        )
        .await
        .unwrap();
        let receipt = root.path().join("next.json");
        save_selection(&next, &receipt).unwrap();
        drop(new);
        let live = serve::NativeLiveHybridIndex::new(&cx, old).unwrap();
        let scope = filter::Filter::parse(r#"{"metadata":{"language":"rust"}}"#).unwrap();
        let messages = [
            serde_json::json!({"id":"before", "query":"retry", "filter":{}}),
            serde_json::json!({"id":"escape", "query":"garden", "filter":{"ids":["garden.md"]}}),
            serde_json::json!({"id":"bad", "query":"retry", "filter":{"ignored":true}}),
            serde_json::json!({"op":"activate", "receipt":receipt, "expected_generation":old_selection.generation}),
            serde_json::json!({"id":"after", "query":"retry", "filter":{}}),
        ];
        let mut bytes = Vec::new();
        for message in messages {
            bytes.extend(encode(&message, MAX_OUTPUT_BYTES).unwrap());
        }
        let before = (
            fast.load(Ordering::Relaxed),
            quality.load(Ordering::Relaxed),
        );
        let mut output = Vec::new();
        serve::run(
            &live,
            &cx,
            &mut Cursor::new(bytes),
            &mut output,
            (Mode::Full, 10),
            true,
            Some(&scope),
            &query::Policy::default(),
        )
        .await
        .unwrap();
        let frames = output_frames(&output);
        assert_eq!(frames[0]["default_filter_applied"], true);
        for frame in &frames {
            if let Some(results) = frame["results"].as_array() {
                if frame["id"] == "before" {
                    assert_eq!(results.len(), 1);
                    assert_eq!(results[0]["doc_id"], "retry.rs");
                } else {
                    assert!(
                        results.is_empty(),
                        "request cannot escape or carry old scope membership"
                    );
                }
            }
        }
        assert_eq!(
            live.snapshot(&cx).await.unwrap().generation(),
            next.generation
        );
        assert!(
            frames
                .iter()
                .any(|frame| frame["operation"] == "activate" && frame["ok"] == true)
        );
        assert_eq!(
            (
                fast.load(Ordering::Relaxed),
                quality.load(Ordering::Relaxed)
            ),
            (before.0 + 1, before.1 + 1)
        );
    });
}
