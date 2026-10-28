use super::*;
use frankensearch::native_ann::{NativeExactReason, NativeRetrievalMode};
use frankensearch_core::generation::{
    EmbeddingArtifactIdentityV1, EmbeddingIdentityBundleV1, EmbeddingSpaceKindV1,
    QuantizationFormat,
};
use frankensearch_core::{ModelCategory, SearchFuture, SearchResult};
use frankensearch_index::generation_root::authority_publisher::AttemptStageV1;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Provider {
    identity: EmbeddingIdentityBundleV1,
    calls: AtomicUsize,
}

impl Embedder for Provider {
    fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
        Ok(&self.identity)
    }
    fn embed<'a>(&'a self, _: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Box::pin(async move {
            Ok(if text.contains("retry") {
                vec![1.0, 0.0]
            } else {
                vec![0.0, 1.0]
            })
        })
    }
    fn dimension(&self) -> usize {
        2
    }
    fn id(&self) -> &'static str {
        "authority-cli-test"
    }
    fn model_name(&self) -> &str {
        self.id()
    }
    fn is_semantic(&self) -> bool {
        true
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::TransformerEmbedder
    }
}

fn parse(args: &[&str]) -> Result<Option<Options>> {
    Options::parse(args.iter().map(|argument| (*argument).to_owned()))
}

fn index_arguments() -> Vec<&'static str> {
    vec![
        "authority-index",
        "--root",
        "new-root",
        "--work-dir",
        "new-work",
        "--receipt",
        "new.json",
        "--tier",
        "fast",
        "--source-low",
        "1",
        "--source-high",
        "5",
        "--cooperative-local",
    ]
}

fn rebuild_arguments() -> Vec<&'static str> {
    vec![
        "authority-rebuild",
        "--from-receipt",
        "selected.json",
        "--work-dir",
        "new-work",
        "--receipt",
        "new.json",
        "--source-low",
        "1",
        "--source-high",
        "9",
    ]
}

fn update_arguments() -> Vec<&'static str> {
    vec![
        "authority-update",
        "--from-receipt",
        "selected.json",
        "--work-dir",
        "new-work",
        "--receipt",
        "new.json",
        "--input",
        "edits.jsonl",
        "--source-low",
        "2",
        "--source-high",
        "8",
    ]
}

fn retained_rebuild_arguments() -> Vec<&'static str> {
    vec![
        "authority-rebuild",
        "--from-receipt",
        "selected.json",
        "--work-dir",
        "new-work",
        "--receipt",
        "new.json",
        "--retained-source",
    ]
}

fn rollback_arguments() -> Vec<&'static str> {
    vec![
        "authority-rollback",
        "--from-receipt",
        "current.json",
        "--target-receipt",
        "predecessor.json",
        "--receipt",
        "rolled-back.json",
    ]
}

#[test]
fn historical_selection_requires_explicit_read_commands_without_control_grants() {
    for mut arguments in [
        vec![
            "authority-search",
            "--receipt",
            "old.json",
            "--query",
            "retry",
        ],
        vec![
            "authority-fetch",
            "--receipt",
            "old.json",
            "--doc-id",
            "zeta",
        ],
        vec!["authority-serve", "--receipt", "old.json"],
    ] {
        assert!(
            parse(&arguments)
                .unwrap()
                .unwrap()
                .current_receipt
                .is_none()
        );
        arguments.extend(["--current-receipt", "current.json"]);
        let options = parse(&arguments).unwrap().unwrap();
        assert_eq!(
            options.current_receipt.as_deref(),
            Some(Path::new("current.json"))
        );
        assert_eq!(options.receipt, Path::new("old.json"));
        let mut duplicate = arguments.clone();
        duplicate.extend(["--current-receipt", "another.json"]);
        assert!(parse(&duplicate).is_err());
        for grant in [
            "--allow-activation",
            "--allow-updates",
            "--allow-model-activation",
        ] {
            let mut granted = arguments.clone();
            granted.push(grant);
            assert!(
                parse(&granted).is_err(),
                "historical reads cannot grant {grant}"
            );
        }
    }
    for mut arguments in [
        index_arguments(),
        rebuild_arguments(),
        update_arguments(),
        rollback_arguments(),
        vec!["authority-recover", "--receipt", "selected.json"],
    ] {
        arguments.extend(["--current-receipt", "current.json"]);
        assert!(parse(&arguments).is_err());
    }
    for suffix in [vec!["--current-receipt"], vec!["--current-receipt", ""]] {
        let mut arguments = vec!["authority-serve", "--receipt", "old.json"];
        arguments.extend(suffix);
        assert!(parse(&arguments).is_err());
    }
}

#[test]
fn authority_lexical_options_require_explicit_read_only_execution() {
    for command in ["authority-search", "authority-serve"] {
        let mut arguments = vec![command, "--receipt", "selected.json", "--lexical-only"];
        if command == "authority-search" {
            arguments.extend(["--query", "retry"]);
        }
        let options = parse(&arguments).unwrap().unwrap();
        assert!(options.lexical_only);
        for extra in [
            vec!["--model-dir", "models"],
            vec!["--quality-backend", "onnx"],
            vec!["--quality-model-dir", "models"],
            vec!["--allow-updates"],
            vec!["--lexical-only"],
        ] {
            let mut rejected = arguments.clone();
            rejected.extend(extra);
            assert!(parse(&rejected).is_err(), "{rejected:?}");
        }
        if command == "authority-serve" {
            arguments.push("--allow-activation");
            assert!(parse(&arguments).unwrap().unwrap().allow_activation);
        }
    }
    for command in [
        "authority-index",
        "authority-update",
        "authority-rebuild",
        "authority-fetch",
    ] {
        assert!(parse(&[command, "--receipt", "selected.json", "--lexical-only"]).is_err());
    }
    for mode in ["lexical", "fast", "quality", "full"] {
        assert!(
            serde_json::from_value::<AuthorityRequest>(serde_json::json!({
                "query": "retry", "mode": mode,
            }))
            .is_ok()
        );
    }
    for mode in ["semantic", "hybrid", "auto", "lexical_fallback"] {
        assert!(
            serde_json::from_value::<AuthorityRequest>(serde_json::json!({
                "query": "retry", "mode": mode,
            }))
            .is_err()
        );
    }
}

#[test]
fn authority_model_activation_is_an_independent_explicit_startup_grant() {
    let arguments = [
        "authority-serve",
        "--receipt",
        "selected.json",
        "--allow-model-activation",
    ];
    let options = parse(&arguments).unwrap().unwrap();
    assert!(options.allow_model_activation);
    assert!(!options.allow_activation);
    assert!(!options.allow_updates);
    for extra in ["--lexical-only", "--allow-model-activation"] {
        let mut rejected = arguments.to_vec();
        rejected.push(extra);
        assert!(parse(&rejected).is_err());
    }
    for command in [
        "authority-search",
        "authority-index",
        "authority-rebuild",
        "authority-update",
        "authority-fetch",
        "authority-recover",
        "authority-rollback",
    ] {
        assert!(
            parse(&[
                command,
                "--receipt",
                "selected.json",
                "--allow-model-activation"
            ])
            .is_err()
        );
    }
    let request = serde_json::from_value::<AuthorityRequest>(serde_json::json!({
        "op":"activate_model", "receipt":"/absolute/candidate.json", "expected_authority":[],
    }))
    .unwrap();
    assert!(
        matches!(request, AuthorityRequest::ModelActivation(_)),
        "missing model paths must reach the model activation validator, not generic activation"
    );
}

#[test]
fn authority_fetch_options_select_windows_without_model_or_writer_grants() {
    let base = [
        "authority-fetch",
        "--receipt",
        "selected.json",
        "--doc-id",
        "notes/../retained",
    ];
    let options = parse(&base).unwrap().unwrap();
    assert_eq!(options.command, Command::Fetch);
    assert_eq!(options.doc_id.as_deref(), Some("notes/../retained"));
    assert_eq!(options.offset_bytes, 0);
    assert_eq!(options.max_bytes, 8192);
    assert!(!options.include_metadata);
    assert!(options.models.is_none());
    assert!(options.work.is_none());
    assert!(is_command("authority-fetch"));
    assert!(parse(&["authority-fetch", "--help"]).unwrap().is_none());
    assert!(parse(&["authority-fetch", "--receipt", "selected.json"]).is_err());
    assert!(parse(&["authority-fetch", "--doc-id", "retained"]).is_err());
    for doc_id in ["", " ", "invalid\0id"] {
        assert!(
            parse(&[
                "authority-fetch",
                "--receipt",
                "selected.json",
                "--doc-id",
                doc_id
            ])
            .is_err()
        );
    }
    let mut bounded = base.to_vec();
    bounded.extend([
        "--offset-bytes",
        "0",
        "--max-bytes",
        "4",
        "--include-metadata",
        "--filter",
        r#"{"metadata":{"tenant":"blue"}}"#,
        "--max-source-bytes",
        "4096",
        "--max-image-bytes",
        "8192",
    ]);
    let options = parse(&bounded).unwrap().unwrap();
    assert_eq!(options.offset_bytes, 0);
    assert_eq!(options.max_bytes, 4);
    assert!(options.include_metadata);
    assert!(options.filter.is_some());
    assert_eq!(options.max_source_bytes, 4096);
    assert_eq!(options.max_image_bytes, 8192);
    for extra in [
        vec!["--max-bytes", "0"],
        vec!["--max-bytes", "3"],
        vec!["--max-bytes", "65537"],
        vec!["--offset-bytes", "-1"],
        vec!["--max-source-bytes", "0"],
        vec!["--max-image-bytes", "0"],
        vec!["--include-metadata", "--include-metadata"],
        vec!["--doc-id", "duplicate"],
        vec!["--model-dir", "models"],
        vec!["--quality-model-dir", "models"],
        vec!["--quality-backend", "native-f32"],
        vec!["--query", "retry"],
        vec!["--limit", "1"],
        vec!["--batch-size", "1"],
        vec!["--work-dir", "new-work"],
        vec!["--input", "source.jsonl"],
        vec!["--root", "root"],
        vec!["--from-receipt", "old.json"],
        vec!["--target-receipt", "old.json"],
        vec!["--source-low", "1"],
        vec!["--tier", "fast"],
        vec!["--retrieval", "exact"],
        vec!["--retained-source"],
        vec!["--allow-activation"],
        vec!["--allow-updates"],
        vec!["--allow-damaged-current"],
    ] {
        let mut rejected = base.to_vec();
        rejected.extend(extra);
        assert!(parse(&rejected).is_err(), "{rejected:?}");
    }
    for command in ["authority-search", "authority-serve", "authority-recover"] {
        assert!(parse(&[command, "--receipt", "selected.json", "--doc-id", "id"]).is_err());
    }
}

#[test]
fn authority_rollback_requires_exact_receipts_and_rejects_build_or_model_options() {
    let arguments = rollback_arguments();
    let options = parse(&arguments).unwrap().unwrap();
    assert_eq!(options.command, Command::Rollback);
    assert_eq!(
        options.from_receipt.as_deref(),
        Some(Path::new("current.json"))
    );
    assert_eq!(
        options.target_receipt.as_deref(),
        Some(Path::new("predecessor.json"))
    );
    assert!(options.models.is_none());
    assert!(options.work.is_none());
    assert!(!options.allow_damaged_current);
    assert!(is_command("authority-rollback"));
    assert!(parse(&["authority-rollback", "--help"]).unwrap().is_none());
    for required in ["--from-receipt", "--target-receipt", "--receipt"] {
        let mut missing = arguments.clone();
        let position = missing
            .iter()
            .position(|argument| *argument == required)
            .unwrap();
        missing.drain(position..position + 2);
        assert!(parse(&missing).is_err(), "{required}");
    }
    let mut bounded = arguments.clone();
    bounded.extend(["--max-image-bytes", "8192"]);
    assert_eq!(parse(&bounded).unwrap().unwrap().max_image_bytes, 8192);
    let mut recovery = arguments.clone();
    recovery.push("--allow-damaged-current");
    assert!(parse(&recovery).unwrap().unwrap().allow_damaged_current);
    recovery.push("--allow-damaged-current");
    assert!(parse(&recovery).is_err());
    for extra in [
        vec!["--model-dir", "models"],
        vec!["--quality-model-dir", "models"],
        vec!["--quality-backend", "native-f32"],
        vec!["--work-dir", "new-work"],
        vec!["--input", "source.jsonl"],
        vec!["--source-low", "1"],
        vec!["--source-high", "9"],
        vec!["--query", "retry"],
        vec!["--limit", "1"],
        vec!["--batch-size", "1"],
        vec!["--root", "root"],
        vec!["--tier", "fast"],
        vec!["--retrieval", "exact"],
        vec!["--retained-source"],
        vec!["--max-source-bytes", "4096"],
        vec!["--migrate-model"],
        vec!["--allow-activation"],
        vec!["--cooperative-local"],
        vec!["--max-image-bytes", "0"],
    ] {
        let mut rejected = arguments.clone();
        rejected.extend(extra);
        assert!(parse(&rejected).is_err(), "{rejected:?}");
    }
    for mut rejected in [
        index_arguments(),
        rebuild_arguments(),
        retained_rebuild_arguments(),
    ] {
        rejected.extend(["--target-receipt", "predecessor.json"]);
        assert!(parse(&rejected).is_err(), "{rejected:?}");
    }
    for command in ["authority-search", "authority-serve", "authority-recover"] {
        assert!(
            parse(&[
                command,
                "--receipt",
                "r.json",
                "--target-receipt",
                "old.json"
            ])
            .is_err()
        );
    }
    for mut rejected in [
        index_arguments(),
        rebuild_arguments(),
        retained_rebuild_arguments(),
        update_arguments(),
        vec![
            "authority-search",
            "--receipt",
            "selected.json",
            "--query",
            "retry",
        ],
        vec!["authority-serve", "--receipt", "selected.json"],
        vec!["authority-recover", "--receipt", "selected.json"],
    ] {
        rejected.push("--allow-damaged-current");
        assert!(parse(&rejected).is_err(), "{rejected:?}");
    }
}

#[test]
fn authority_update_requires_an_explicit_checkpoint_and_inherits_selected_policy() {
    let arguments = update_arguments();
    let options = parse(&arguments).unwrap().unwrap();
    assert_eq!(options.command, Command::Update);
    assert_eq!(
        options.from_receipt.as_deref(),
        Some(Path::new("selected.json"))
    );
    assert_eq!(options.input.as_deref(), Some(Path::new("edits.jsonl")));
    assert_eq!(
        options.source_cut().unwrap(),
        CommitRange { low: 2, high: 8 }
    );
    assert_eq!(options.max_source_bytes, crate::MAX_INPUT_BYTES);
    assert!(options.root.is_none());
    assert!(options.tier.is_none());
    assert!(options.retrieval.is_none());
    assert!(!options.migrate_model);
    assert!(!options.retained_source);
    assert!(is_command("authority-update"));
    assert!(parse(&["authority-update", "--help"]).unwrap().is_none());

    let mut stdin = arguments.clone();
    let input = stdin
        .iter()
        .position(|argument| *argument == "--input")
        .unwrap();
    stdin.drain(input..input + 2);
    assert!(parse(&stdin).unwrap().unwrap().input.is_none());

    let mut bounded = arguments.clone();
    bounded.extend(["--max-source-bytes", "4096", "--batch-size", "2"]);
    let options = parse(&bounded).unwrap().unwrap();
    assert_eq!(options.max_source_bytes, 4096);
    assert_eq!(options.batch_size, 2);
    for required in [
        "--from-receipt",
        "--work-dir",
        "--receipt",
        "--source-low",
        "--source-high",
    ] {
        let mut missing = arguments.clone();
        let position = missing
            .iter()
            .position(|argument| *argument == required)
            .unwrap();
        missing.drain(position..position + 2);
        assert!(parse(&missing).is_err(), "accepted missing {required}");
    }
    for extra in [
        vec!["--root", "other-root"],
        vec!["--tier", "quality"],
        vec!["--retrieval", "exact"],
        vec!["--retrieval", "hnsw"],
        vec!["--migrate-model"],
        vec!["--retained-source"],
        vec!["--cooperative-local"],
        vec!["--allow-activation"],
        vec!["--max-source-bytes", "0"],
        vec!["--batch-size", "0"],
        vec!["--source-high", "9"],
    ] {
        let mut rejected = arguments.clone();
        rejected.extend(extra);
        assert!(parse(&rejected).is_err(), "{rejected:?}");
    }
}

#[test]
fn retained_rebuild_options_preserve_the_selected_cut_and_bound_source_export() {
    let arguments = retained_rebuild_arguments();
    let options = parse(&arguments).unwrap().unwrap();
    assert_eq!(options.command, Command::Rebuild);
    assert!(options.retained_source);
    assert!(options.input.is_none());
    assert_eq!(options.max_source_bytes, crate::MAX_INPUT_BYTES);
    let previous = CommitRange { low: 9, high: 37 };
    assert_eq!(options.rebuild_source_cut(&previous).unwrap(), previous);
    let supplied = parse(&rebuild_arguments()).unwrap().unwrap();
    assert!(!supplied.retained_source);
    assert_eq!(
        supplied.rebuild_source_cut(&previous).unwrap(),
        CommitRange { low: 1, high: 9 }
    );

    let mut bounded = arguments.clone();
    bounded.extend([
        "--max-source-bytes",
        "4096",
        "--migrate-model",
        "--retrieval",
        "hnsw",
    ]);
    let options = parse(&bounded).unwrap().unwrap();
    assert_eq!(options.max_source_bytes, 4096);
    assert!(options.migrate_model);
    assert!(matches!(
        options.build_retrieval(None),
        NativeBuildRetrieval::Hnsw { .. }
    ));
    // The budget may precede the source-selection flag; admission depends on
    // the complete option set, not argument order.
    let mut reordered = arguments.clone();
    reordered.pop();
    reordered.extend(["--max-source-bytes", "4096", "--retained-source"]);
    assert_eq!(parse(&reordered).unwrap().unwrap().max_source_bytes, 4096);

    for extra in [
        vec!["--input", "source.jsonl"],
        vec!["--source-low", "1"],
        vec!["--source-high", "9"],
        vec!["--max-source-bytes", "0"],
        vec!["--max-source-bytes", "-1"],
        vec!["--retained-source"],
    ] {
        let mut rejected = arguments.clone();
        rejected.extend(extra);
        assert!(parse(&rejected).is_err(), "{rejected:?}");
    }
    let mut unselected_budget = rebuild_arguments();
    unselected_budget.extend(["--max-source-bytes", "4096"]);
    assert!(parse(&unselected_budget).is_err());
    let mut mixed_sources = rebuild_arguments();
    mixed_sources.push("--retained-source");
    assert!(parse(&mixed_sources).is_err());
    for command in [
        "authority-index",
        "authority-search",
        "authority-serve",
        "authority-recover",
    ] {
        let rejected = [command, "--receipt", "selected.json", "--retained-source"];
        assert!(parse(&rejected).is_err(), "{rejected:?}");
    }
    for command in ["authority-index", "authority-recover"] {
        let rejected = [
            command,
            "--receipt",
            "selected.json",
            "--max-source-bytes",
            "4096",
        ];
        assert!(parse(&rejected).is_err(), "{rejected:?}");
    }
}

#[test]
fn authority_filters_are_bounded_and_only_apply_to_search_and_serve() {
    let filter = r#"{"ids":["garden-old"],"id_prefix":"garden-","metadata":{"version":"old"}}"#;
    for arguments in [
        vec![
            "authority-search",
            "--receipt",
            "selected.json",
            "--query",
            "retry",
        ],
        vec!["authority-serve", "--receipt", "selected.json"],
    ] {
        let options = parse(&arguments).unwrap().unwrap();
        assert!(options.filter.is_none());
        assert_eq!(options.max_source_bytes, crate::MAX_INPUT_BYTES);
        let mut configured = arguments.clone();
        configured.extend(["--filter", filter, "--max-source-bytes", "4096"]);
        let options = parse(&configured).unwrap().unwrap();
        let filter = options.filter.as_ref().unwrap();
        assert_eq!(options.max_source_bytes, 4096);
        assert!(filter.matches(
            &IndexableDocument::new("garden-old", "garden flowers").with_metadata("version", "old")
        ));
        assert!(!filter.matches(
            &IndexableDocument::new("retry-old", "retry requests").with_metadata("version", "old")
        ));
        assert!(!filter.matches(
            &IndexableDocument::new("garden-old", "garden flowers").with_metadata("version", "new")
        ));
        let mut budget_only = arguments.clone();
        budget_only.extend(["--max-source-bytes", "1"]);
        let options = parse(&budget_only).unwrap().unwrap();
        assert!(options.filter.is_none());
        assert_eq!(options.max_source_bytes, 1);
        for invalid in [
            "null",
            "[]",
            r#"{"unknown":"private-filter-sentinel"}"#,
            r#"{"metadata":{"version":7}}"#,
            r#"{"ids":[" "]}"#,
            r#"{"id_prefix":""}"#,
            r#"{"ids":["invalid\u0000id"]}"#,
        ] {
            let mut rejected = arguments.clone();
            rejected.extend(["--filter", invalid]);
            assert!(parse(&rejected).is_err(), "{rejected:?}");
        }
        for extra in [
            vec!["--max-source-bytes", "0"],
            vec!["--max-source-bytes", "-1"],
            vec!["--filter", "{}", "--filter", "{}"],
        ] {
            let mut rejected = arguments.clone();
            rejected.extend(extra);
            assert!(parse(&rejected).is_err(), "{rejected:?}");
        }
        for explicit in ["{}", r#"{"ids":[]}"#] {
            let mut configured = arguments.clone();
            configured.extend(["--filter", explicit]);
            assert!(parse(&configured).unwrap().unwrap().filter.is_some());
        }
        let oversized = serde_json::json!({"id_prefix": "x".repeat(64 * 1024)}).to_string();
        let mut rejected = arguments.clone();
        rejected.extend(["--filter", &oversized]);
        assert!(parse(&rejected).is_err());
    }
    for mut arguments in [
        index_arguments(),
        rebuild_arguments(),
        retained_rebuild_arguments(),
        update_arguments(),
        rollback_arguments(),
        vec!["authority-recover", "--receipt", "selected.json"],
    ] {
        arguments.extend(["--filter", "{}"]);
        assert!(parse(&arguments).is_err(), "{arguments:?}");
    }
    for malformed in [
        serde_json::json!({"query": "retry", "filter": {"unknown": true}}),
        serde_json::json!({"query": "retry", "filter": {"metadata": {"tenant": 7}}}),
        serde_json::json!({"query": "retry", "filter": []}),
        serde_json::json!({
            "op": "activate", "receipt": "/selected.json",
            "expected_authority": [], "filter": {},
        }),
    ] {
        assert!(serde_json::from_value::<AuthorityRequest>(malformed).is_err());
    }
    for filter in [serde_json::Value::Null, serde_json::json!({})] {
        let request = serde_json::from_value::<AuthorityRequest>(
            serde_json::json!({"query": "retry", "filter": filter}),
        )
        .unwrap();
        let AuthorityRequest::Search(request) = request else {
            panic!("a search request cannot become an activation");
        };
        assert_eq!(request.filter.is_none(), filter.is_null());
    }
}

#[test]
fn authority_serve_requires_a_separate_grant_for_warm_updates_and_batch_limits() {
    let base = ["authority-serve", "--receipt", "selected.json"];
    let mut enabled = base.to_vec();
    enabled.extend([
        "--allow-updates",
        "--batch-size",
        "2",
        "--max-source-bytes",
        "4096",
    ]);
    let options = parse(&enabled).unwrap().unwrap();
    assert_eq!(options.command, Command::Serve);
    assert_eq!(options.batch_size, 2);
    assert_eq!(options.max_source_bytes, 4096);
    assert!(!options.allow_activation);
    let mut reordered = base.to_vec();
    reordered.extend(["--max-source-bytes", "4096", "--allow-updates"]);
    assert_eq!(parse(&reordered).unwrap().unwrap().max_source_bytes, 4096);
    for extra in [
        vec!["--batch-size", "2"],
        vec!["--allow-activation", "--batch-size", "2"],
        vec!["--allow-updates", "--batch-size", "0"],
        vec!["--allow-updates", "--max-source-bytes", "0"],
        vec!["--allow-updates", "--allow-updates"],
    ] {
        let mut rejected = base.to_vec();
        rejected.extend(extra);
        assert!(parse(&rejected).is_err(), "{rejected:?}");
    }
    for command in [
        "authority-search",
        "authority-recover",
        "authority-rollback",
        "authority-index",
        "authority-rebuild",
        "authority-update",
    ] {
        assert!(parse(&[command, "--receipt", "selected.json", "--allow-updates"]).is_err());
    }
}

#[test]
fn authority_serve_options_select_one_warm_receipt_and_explicit_activation_grant() {
    let arguments = ["authority-serve", "--receipt", "selected.json"];
    let options = parse(&arguments).unwrap().unwrap();
    assert_eq!(options.command, Command::Serve);
    assert!(!options.allow_activation);
    assert_eq!(options.limit, 10);
    assert!(is_command("authority-serve"));
    assert!(parse(&["authority-serve", "--help"]).unwrap().is_none());
    let mut configured = arguments.to_vec();
    configured.extend([
        "--allow-activation",
        "--limit",
        "2",
        "--max-image-bytes",
        "8192",
    ]);
    let options = parse(&configured).unwrap().unwrap();
    assert!(options.allow_activation);
    assert_eq!(options.limit, 2);
    assert_eq!(options.max_image_bytes, 8192);
    for extra in [
        vec!["--query", "retry"],
        vec!["--retrieval", "hnsw"],
        vec!["--root", "other-root"],
        vec!["--tier", "quality"],
        vec!["--migrate-model"],
        vec!["--cooperative-local"],
        vec!["--limit", "0"],
        vec!["--limit", "1001"],
    ] {
        let mut rejected = arguments.to_vec();
        rejected.extend(extra);
        assert!(parse(&rejected).is_err(), "{rejected:?}");
    }
    for command in ["authority-search", "authority-recover"] {
        assert!(parse(&[command, "--receipt", "r.json", "--allow-activation"]).is_err());
    }
}

#[test]
fn authority_retrieval_is_explicit_for_new_indexes_and_cannot_override_search() {
    let options = parse(&index_arguments()).unwrap().unwrap();
    assert!(options.retrieval.is_none());
    assert!(matches!(
        options.build_retrieval(None),
        NativeBuildRetrieval::Exact
    ));
    for mut arguments in [index_arguments(), rebuild_arguments()] {
        arguments.extend(["--retrieval", "hnsw"]);
        let options = parse(&arguments).unwrap().unwrap();
        let NativeBuildRetrieval::Hnsw { params, seed } = options.build_retrieval(None) else {
            panic!("explicit HNSW must select a graph");
        };
        assert_eq!(params, HnswParams::default());
        assert_eq!(seed, 42);
        arguments.extend(["--retrieval", "exact"]);
        assert!(parse(&arguments).is_err(), "duplicate retrieval selection");
    }
    for retrieval in ["auto", "ann", "", "EXACT"] {
        let mut arguments = index_arguments();
        arguments.extend(["--retrieval", retrieval]);
        assert!(parse(&arguments).is_err(), "accepted {retrieval:?}");
    }
    for arguments in [
        vec![
            "authority-search",
            "--receipt",
            "selected.json",
            "--query",
            "retry",
            "--retrieval",
            "exact",
        ],
        vec![
            "authority-recover",
            "--receipt",
            "selected.json",
            "--retrieval",
            "hnsw",
        ],
    ] {
        assert!(parse(&arguments).is_err());
    }
}

#[test]
fn rebuild_inherits_full_authenticated_retrieval_policy_unless_explicitly_changed() {
    let inherited_params = HnswParams {
        m: 4,
        m0: 8,
        ef_construction: 24,
        ef_search: 12,
    };
    let inherited = NativeBuildRetrieval::Hnsw {
        params: inherited_params,
        seed: 9876,
    };
    let options = parse(&rebuild_arguments()).unwrap().unwrap();
    let NativeBuildRetrieval::Hnsw { params, seed } = options.build_retrieval(Some(inherited))
    else {
        panic!("a rebuild must preserve its admitted graph policy");
    };
    assert_eq!(params, inherited_params);
    assert_eq!(seed, 9876);
    assert!(matches!(
        options.build_retrieval(Some(NativeBuildRetrieval::Exact)),
        NativeBuildRetrieval::Exact
    ));

    let mut arguments = rebuild_arguments();
    arguments.extend(["--retrieval", "exact"]);
    let options = parse(&arguments).unwrap().unwrap();
    assert!(matches!(
        options.build_retrieval(Some(inherited)),
        NativeBuildRetrieval::Exact
    ));
    *arguments.last_mut().unwrap() = "hnsw";
    let options = parse(&arguments).unwrap().unwrap();
    let NativeBuildRetrieval::Hnsw { params, seed } = options.build_retrieval(Some(inherited))
    else {
        panic!("explicit HNSW must select the default graph policy");
    };
    assert_eq!(params, HnswParams::default());
    assert_eq!(seed, 42);
}

#[test]
fn rebuild_parser_selects_an_existing_receipt_without_overriding_its_root_or_tier() {
    let options = parse(&rebuild_arguments()).unwrap().unwrap();
    assert_eq!(options.command, Command::Rebuild);
    assert_eq!(
        options.from_receipt.as_deref(),
        Some(Path::new("selected.json"))
    );
    assert!(options.root.is_none());
    assert!(options.tier.is_none());
    assert!(!options.migrate_model);
    assert!(is_command("authority-rebuild"));
    assert!(parse(&["authority-rebuild", "--help"]).unwrap().is_none());
    for extra in [
        vec!["--root", "replacement-root"],
        vec!["--tier", "quality"],
        vec!["--cooperative-local"],
        vec!["--query", "retry"],
        vec!["--limit", "1"],
        vec!["--from-receipt", "different.json"],
    ] {
        let mut arguments = rebuild_arguments();
        arguments.extend(extra);
        assert!(parse(&arguments).is_err(), "{arguments:?}");
    }
    for required in [
        "--from-receipt",
        "--work-dir",
        "--receipt",
        "--source-low",
        "--source-high",
    ] {
        let mut arguments = rebuild_arguments();
        let position = arguments
            .iter()
            .position(|argument| *argument == required)
            .unwrap();
        arguments.drain(position..position + 2);
        assert!(parse(&arguments).is_err(), "{required}");
    }
    let mut migration = rebuild_arguments();
    migration.extend([
        "--migrate-model",
        "--quality-backend",
        "native-int8",
        "--quality-model-dir",
        "model",
    ]);
    assert!(parse(&migration).unwrap().unwrap().migrate_model);
    for command in ["authority-search", "authority-recover"] {
        assert!(parse(&[command, "--receipt", "r", "--from-receipt", "old"]).is_err());
        assert!(parse(&[command, "--receipt", "r", "--migrate-model"]).is_err());
    }
}

#[test]
fn rebuild_query_requires_explicit_migration_and_a_distinct_compensation_receipt() {
    for base in [rebuild_arguments(), retained_rebuild_arguments()] {
        for extra in [
            vec!["--query", "retry requests"],
            vec!["--query", "retry requests", "--migrate-model"],
            vec![
                "--query",
                "retry requests",
                "--rollback-receipt",
                "undo.json",
            ],
            vec!["--rollback-receipt", "undo.json"],
            vec!["--limit", "4"],
            vec!["--stream"],
            vec!["--filter", "{\"ids\":[\"a\"]}"],
        ] {
            let mut arguments = base.clone();
            arguments.extend(extra);
            assert!(parse(&arguments).is_err(), "{arguments:?}");
        }
        let mut arguments = base;
        arguments.extend([
            "--query",
            "retry requests",
            "--migrate-model",
            "--rollback-receipt",
            "undo.json",
            "--filter",
            "{\"ids\":[\"a\"]}",
            "--limit",
            "4",
            "--stream",
            "--max-source-bytes",
            "4096",
        ]);
        let options = parse(&arguments).unwrap().unwrap();
        assert_eq!(options.query.as_deref(), Some("retry requests"));
        assert_eq!(
            options.rollback_receipt.as_deref(),
            Some(Path::new("undo.json"))
        );
        assert!(options.filter.is_some());
        assert!(options.migrate_model && options.stream);
        assert_eq!(options.limit, 4);
        assert_eq!(options.max_source_bytes, 4096);
        assert!(options.mode.is_none());
        for extra in [
            vec!["--mode", "lexical"],
            vec!["--query", "second request"],
            vec!["--rollback-receipt", "other.json"],
            vec!["--allow-model-activation"],
        ] {
            let mut duplicate = arguments.clone();
            duplicate.extend(extra);
            assert!(parse(&duplicate).is_err(), "{duplicate:?}");
        }
    }
    for command in ["authority-search", "authority-serve", "authority-recover"] {
        assert!(
            parse(&[
                command,
                "--receipt",
                "current.json",
                "--rollback-receipt",
                "undo.json"
            ])
            .is_err()
        );
    }
    let mut blank = retained_rebuild_arguments();
    blank.extend([
        "--query",
        " \t\n",
        "--migrate-model",
        "--rollback-receipt",
        "undo.json",
    ]);
    assert!(parse(&blank).is_err());
}

#[test]
fn rebuild_source_boundaries_are_monotone_and_allow_an_unchanged_cut() {
    let previous = CommitRange { low: 2, high: 7 };
    for source in [
        CommitRange { low: 2, high: 7 },
        CommitRange { low: 2, high: 8 },
        CommitRange { low: 3, high: 9 },
    ] {
        validate_successor_source(&previous, &source).unwrap();
    }
    for source in [
        CommitRange { low: 1, high: 9 },
        CommitRange { low: 2, high: 6 },
        CommitRange { low: 1, high: 6 },
    ] {
        assert!(validate_successor_source(&previous, &source).is_err());
    }
}

#[test]
fn rebuild_requires_explicit_producer_migration_before_any_inference() {
    let selected = receipt();
    let mut provider = Provider {
        identity: provider_identity(),
        calls: AtomicUsize::new(0),
    };
    assert_eq!(
        rebuild_identity(&selected.identity, &provider, false, false).unwrap(),
        selected.identity
    );
    provider.identity.producer.implementation_revision =
        "changed-same-dimension-producer".to_owned();
    provider.identity.validate().unwrap();
    assert!(rebuild_identity(&selected.identity, &provider, false, false).is_err());
    let migrated = rebuild_identity(&selected.identity, &provider, true, false).unwrap();
    assert_ne!(migrated, selected.identity);
    assert_eq!(migrated.identity.producer, provider.identity.producer);
    assert_eq!(migrated.identity.space, provider.identity.space);
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
    provider.identity.space.dimension = 3;
    assert!(rebuild_identity(&selected.identity, &provider, true, false).is_err());
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn retained_rebuild_migrates_producers_and_spaces_without_relabeling_prepared_input() {
    let expected = identity();
    let mut provider = Provider {
        identity: provider_identity(),
        calls: AtomicUsize::new(0),
    };
    provider.identity.producer.implementation_revision = "replacement-producer".to_owned();
    provider.identity.space.immutable_revision = "replacement-weights".to_owned();
    provider.identity.space.artifacts[0].sha256 = "c".repeat(64);
    provider.identity.producer.space_fingerprint = provider.identity.space.fingerprint();
    provider.identity.validate().unwrap();
    let migrated = rebuild_identity(&expected, &provider, true, true).unwrap();
    assert_eq!(migrated.identity.input, expected.identity.input);
    assert_ne!(migrated.identity.space, expected.identity.space);
    assert_ne!(migrated.identity.producer, expected.identity.producer);

    provider
        .identity
        .input
        .canonicalization
        .push_str("-new-preparation");
    provider.identity.space.input_contract_fingerprint = provider.identity.input.fingerprint();
    provider.identity.producer.space_fingerprint = provider.identity.space.fingerprint();
    provider.identity.validate().unwrap();
    assert!(
        rebuild_identity(&expected, &provider, true, true)
            .unwrap_err()
            .to_string()
            .contains("retained source cannot migrate its prepared-input contract")
    );
    let fresh = rebuild_identity(&expected, &provider, true, false).unwrap();
    assert_eq!(fresh.identity.input, provider.identity.input);
    assert!(rebuild_identity(&expected, &provider, false, false).is_err());
    assert_eq!(
        provider.calls.load(Ordering::Relaxed),
        0,
        "the complete input contract is admitted before any document inference"
    );
}

#[test]
fn parser_requires_explicit_topology_source_and_security_profile() {
    let options = parse(&index_arguments()).unwrap().unwrap();
    assert_eq!(options.tier, Some(EmbedderTierTag::Fast));
    assert_eq!(
        options.source_cut().unwrap(),
        CommitRange { low: 1, high: 5 }
    );
    let mut absent_profile = index_arguments();
    absent_profile.pop();
    assert!(parse(&absent_profile).is_err());
    for extra in [
        vec!["--quality-backend", "onnx"],
        vec!["--root", "different-root"],
        vec!["--mode", "full"],
        vec!["--shard-size", "5"],
        vec!["--batch-size", "0"],
        vec!["--max-image-bytes", "0"],
    ] {
        let mut arguments = index_arguments();
        arguments.extend(extra);
        assert!(parse(&arguments).is_err(), "{arguments:?}");
    }
    let mut quality = index_arguments();
    let position = quality.iter().position(|value| *value == "fast").unwrap();
    quality[position] = "quality";
    quality.extend([
        "--quality-backend",
        "native-int8",
        "--quality-model-dir",
        "verified-model",
    ]);
    assert_eq!(
        parse(&quality).unwrap().unwrap().tier,
        Some(EmbedderTierTag::Quality)
    );
    let mut invalid_range = index_arguments();
    let low = invalid_range
        .iter()
        .position(|value| *value == "1")
        .unwrap();
    let high = invalid_range
        .iter()
        .position(|value| *value == "5")
        .unwrap();
    invalid_range[low] = "6";
    assert!(parse(&invalid_range).is_err());
    invalid_range[low] = "0";
    invalid_range[high] = "18446744073709551615";
    assert!(
        parse(&invalid_range).is_err(),
        "inclusive-range overflow must fail before indexing"
    );
}

#[test]
fn search_options_do_not_grant_publication_or_change_the_selected_tier() {
    let base = [
        "authority-search",
        "--receipt",
        "trusted.json",
        "--query",
        "retry network",
    ];
    assert_eq!(parse(&base).unwrap().unwrap().command, Command::Search);
    for extra in [
        vec!["--root", "different-root"],
        vec!["--cooperative-local"],
        vec!["--tier", "quality"],
        vec!["--input", "documents.jsonl"],
        vec!["--limit", "1001"],
        vec!["--query", "second query"],
    ] {
        let mut arguments = base.to_vec();
        arguments.extend(extra);
        assert!(parse(&arguments).is_err());
    }
    assert!(parse(&["authority-search", "--help"]).unwrap().is_none());
    assert!(parse(&["authority-search", "--receipt", "r", "--query", "   "]).is_err());
}

#[test]
fn recovery_requires_only_the_saved_attempt_and_rejects_model_or_publish_options() {
    let base = ["authority-recover", "--receipt", "trusted.json"];
    assert_eq!(parse(&base).unwrap().unwrap().command, Command::Recover);
    assert!(is_command("authority-recover"));
    assert!(parse(&["authority-recover"]).is_err());
    assert!(parse(&["authority-recover", "--help"]).unwrap().is_none());
    for extra in [
        vec!["--root", "different-root"],
        vec!["--cooperative-local"],
        vec!["--tier", "quality"],
        vec!["--input", "documents.jsonl"],
        vec!["--model-dir", "unavailable"],
        vec!["--quality-backend", "onnx"],
        vec!["--quality-model-dir", "unavailable"],
        vec!["--query", "retry"],
        vec!["--limit", "10"],
        vec!["--receipt", "different.json"],
        vec!["--max-image-bytes", "0"],
    ] {
        let mut arguments = base.to_vec();
        arguments.extend(extra);
        assert!(parse(&arguments).is_err(), "{arguments:?}");
    }
    let mut arguments = base.to_vec();
    arguments.extend(["--max-image-bytes", "4096"]);
    assert_eq!(parse(&arguments).unwrap().unwrap().max_image_bytes, 4096);
}

fn provider_identity() -> EmbeddingIdentityBundleV1 {
    let mut identity = EmbeddingIdentityBundleV1::explicit_test_model("authority-cli-test", 2);
    // Synthetic semantic-shaped test provider; no production inference or
    // model-availability claim is made by these filesystem/identity tests.
    identity.space.kind = EmbeddingSpaceKindV1::Semantic;
    identity.space.hash_control = None;
    identity.space.artifact_manifest_fingerprint = "a".repeat(64);
    identity.space.artifacts = vec![EmbeddingArtifactIdentityV1 {
        role: "weights".to_owned(),
        sha256: "b".repeat(64),
        size: 1,
    }];
    identity.producer.space_fingerprint = identity.space.fingerprint();
    identity
}

fn identity() -> FrozenEmbeddingIdentityBundleV1 {
    // Only the persisted receipt uses the on-disk storage contract. Providers
    // return native f32 values; the builder binds their persisted encoding.
    let mut identity = provider_identity();
    identity.storage.format = "fsvi-v2".to_owned();
    identity.storage.quantization = QuantizationFormat::F32;
    identity.storage.endianness = "little-endian".to_owned();
    identity.freeze().unwrap()
}

fn receipt() -> Receipt {
    let authority = AuthorityRefV1::new(1, [0x52; 16], 256, [0x53; 32], None).unwrap();
    let permit = AttemptPermitV1::new(
        [0x51; 16],
        [0x61; 16],
        [0x62; 16],
        authority,
        ExpectedAuthorityPairV1::default(),
        [0x63; 16],
    )
    .unwrap();
    Receipt {
        schema: RECEIPT_SCHEMA.to_owned(),
        root: PathBuf::from("/private/generation"),
        root_id: [0x51; 16],
        profile: PROFILE.as_str().to_owned(),
        authority: authority.canonical_bytes().to_vec(),
        generation: ArtifactGenerationIdentityV1::new(1, [0x54; 16]).unwrap(),
        identity: identity(),
        tier: EmbedderTierTag::Fast,
        quality_identity: None,
        documents: 2,
        image_bytes: 4096,
        attempt: Some(SavedAttempt::from_permit(&permit, false)),
    }
}

fn progressive_receipt() -> Receipt {
    let mut selected = receipt();
    selected.schema = PROGRESSIVE_RECEIPT_SCHEMA.to_owned();
    let mut quality = identity().identity;
    quality.producer.implementation_revision = "independent-quality-producer".to_owned();
    selected.quality_identity = Some(quality.freeze().unwrap());
    selected
}

#[test]
fn full_options_accept_exact_or_hnsw_and_require_explicit_query_modes() {
    let mut arguments = index_arguments();
    let tier = arguments.iter().position(|value| *value == "fast").unwrap();
    arguments[tier] = "full";
    arguments.extend([
        "--quality-backend",
        "native-int8",
        "--quality-model-dir",
        "verified-quality",
    ]);
    let full = parse(&arguments).unwrap().unwrap();
    assert!(full.progressive);
    assert_eq!(full.tier, Some(EmbedderTierTag::Fast));
    assert!(!full.stream);
    assert!(full.mode.is_none());
    assert_eq!(full.build_retrieval(None), NativeBuildRetrieval::Exact);
    let mut duplicate = arguments.clone();
    duplicate.extend(["--tier", "quality"]);
    assert!(parse(&duplicate).is_err());
    arguments.extend(["--retrieval", "hnsw"]);
    assert_eq!(
        parse(&arguments).unwrap().unwrap().build_retrieval(None),
        NativeBuildRetrieval::Hnsw {
            params: HnswParams::default(),
            seed: 42
        },
    );

    for (name, mode) in [
        ("fast", QueryMode::Fast),
        ("quality", QueryMode::Quality),
        ("full", QueryMode::Full),
    ] {
        let options = parse(&[
            "authority-search",
            "--receipt",
            "selected.json",
            "--query",
            "retry",
            "--mode",
            name,
            "--stream",
        ])
        .unwrap()
        .unwrap();
        assert_eq!(options.mode, Some(mode));
        assert!(options.stream);
        let request: AuthoritySearchRequest = serde_json::from_value(serde_json::json!({
            "query":"retry", "mode":name, "stream":true,
        }))
        .unwrap();
        assert_eq!(request.mode, Some(mode));
        assert!(request.stream);
    }
    let default: AuthoritySearchRequest = serde_json::from_str(r#"{"query":"retry"}"#).unwrap();
    assert!(default.mode.is_none());
    assert!(!default.stream);
    assert!(
        serde_json::from_str::<AuthoritySearchRequest>(r#"{"query":"retry","mode":"automatic"}"#)
            .is_err()
    );
    assert!(
        parse(&[
            "authority-search",
            "--receipt",
            "r",
            "--query",
            "retry",
            "--mode",
            "automatic",
        ])
        .is_err()
    );
    for mut arguments in [
        index_arguments(),
        rebuild_arguments(),
        vec!["authority-recover", "--receipt", "r"],
    ] {
        arguments.push("--stream");
        assert!(parse(&arguments).is_err());
    }
}

#[test]
fn progressive_receipt_preserves_both_bindings_and_refuses_ambiguous_schema() {
    let selected = progressive_receipt();
    selected.validate().unwrap();
    let decoded: Receipt = serde_json::from_slice(&serde_json::to_vec(&selected).unwrap()).unwrap();
    decoded.validate().unwrap();
    assert_eq!(decoded.binding().unwrap(), selected.binding().unwrap());
    assert_eq!(
        decoded.quality_binding().unwrap(),
        selected.quality_binding().unwrap()
    );
    assert_ne!(
        decoded.binding().unwrap(),
        decoded.quality_binding().unwrap().unwrap()
    );
    assert_eq!(
        attempt_report(&decoded.recovery_attempt().unwrap()),
        attempt_report(&selected.recovery_attempt().unwrap())
    );
    assert_eq!(
        decoded.topology(),
        NativeGenerationTopology::FullProgressive
    );
    for schema in [RECEIPT_SCHEMA, SELECTION_ONLY_RECEIPT_SCHEMA] {
        let mut changed = selected.clone();
        changed.schema = schema.to_owned();
        assert!(changed.validate().is_err());
    }
    let mut changed = selected.clone();
    changed.quality_identity = None;
    assert!(changed.validate().is_err());
    changed = selected.clone();
    changed.tier = EmbedderTierTag::Quality;
    assert!(changed.validate().is_err());
    changed = selected.clone();
    changed
        .quality_identity
        .as_mut()
        .unwrap()
        .identity
        .producer
        .implementation_revision
        .push_str("-tampered");
    assert!(changed.validate().is_err());
    changed = selected;
    changed.attempt = None;
    assert!(changed.validate().is_err());
}

#[test]
fn query_modes_cannot_invent_a_missing_authority_tier() {
    let fast = receipt();
    let mut quality = receipt();
    quality.tier = EmbedderTierTag::Quality;
    let full = progressive_receipt();
    for (selected, default, allowed) in [
        (fast, QueryMode::Fast, vec![QueryMode::Fast]),
        (quality, QueryMode::Quality, vec![QueryMode::Quality]),
        (
            full,
            QueryMode::Full,
            vec![QueryMode::Fast, QueryMode::Quality, QueryMode::Full],
        ),
    ] {
        selected.validate().unwrap();
        assert_eq!(selected.query_mode(None).unwrap(), default);
        for mode in [QueryMode::Fast, QueryMode::Quality, QueryMode::Full] {
            assert_eq!(
                selected.query_mode(Some(mode)).is_ok(),
                allowed.contains(&mode)
            );
        }
    }
}

#[test]
fn full_serving_and_rebuild_admit_quality_independently_before_inference() {
    let selected = progressive_receipt();
    let fast = Arc::new(Provider {
        identity: selected.identity.identity.clone(),
        calls: AtomicUsize::new(0),
    });
    let quality = Arc::new(Provider {
        identity: selected.quality_identity.as_ref().unwrap().identity.clone(),
        calls: AtomicUsize::new(0),
    });
    let loaded = Models {
        fast: fast.clone(),
        quality: Some(quality.clone()),
    };
    admit_serving_models(&selected, &loaded).unwrap();
    assert!(
        admit_serving_models(
            &selected,
            &Models {
                fast: fast.clone(),
                quality: None
            }
        )
        .is_err()
    );
    assert!(
        admit_serving_models(
            &selected,
            &Models {
                fast: fast.clone(),
                quality: Some(fast.clone())
            }
        )
        .is_err()
    );
    assert!(admit_serving_models(&receipt(), &loaded).is_err());
    let expected_quality = selected.quality_identity.as_ref().unwrap();
    rebuild_identity(&selected.identity, fast.as_ref(), false, false).unwrap();
    rebuild_identity(expected_quality, quality.as_ref(), false, false).unwrap();
    assert!(rebuild_identity(expected_quality, fast.as_ref(), false, false).is_err());
    let migrated = rebuild_identity(expected_quality, fast.as_ref(), true, false).unwrap();
    assert_eq!(migrated, selected.identity);
    assert_ne!(&migrated, expected_quality);
    assert!(!same_query_contract(expected_quality, &migrated));
    assert_eq!(fast.calls.load(Ordering::Relaxed), 0);
    assert_eq!(quality.calls.load(Ordering::Relaxed), 0);
}

#[test]
fn receipt_preserves_independent_binding_and_rejects_downgrades_and_tampering() {
    let original = receipt();
    original.validate().unwrap();
    let encoded = serde_json::to_vec(&original).unwrap();
    let decoded: Receipt = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded.binding().unwrap(), original.binding().unwrap());
    assert_eq!(decoded.authority().unwrap(), original.authority().unwrap());
    let mut changed = original.clone();
    changed.profile = "required_external".to_owned();
    assert!(changed.validate().is_err());
    changed = original.clone();
    changed.root_id = [0; 16];
    assert!(changed.validate().is_err());
    changed = original.clone();
    changed.authority.push(0);
    assert!(changed.validate().is_err());
    changed = original.clone();
    changed.identity.identity.storage.dimension += 1;
    assert!(changed.validate().is_err());
    changed = original.clone();
    changed.image_bytes = 1;
    assert!(changed.validate().is_err());
    changed = original;
    changed.root = PathBuf::from("relative/root");
    assert!(changed.validate().is_err());
}

#[test]
fn saved_attempt_round_trip_preserves_both_slots_and_the_original_identities() {
    let mut selected = receipt();
    let first = selected.authority().unwrap();
    let second =
        AuthorityRefV1::new(2, [0x71; 16], 256, [0x72; 32], Some(first.fingerprint())).unwrap();
    let third =
        AuthorityRefV1::new(3, [0x73; 16], 256, [0x74; 32], Some(second.fingerprint())).unwrap();
    let permit = AttemptPermitV1::new(
        selected.root_id,
        [0x81; 16],
        [0x82; 16],
        third,
        ExpectedAuthorityPairV1 {
            first: Some(AuthoritySlotV1::new(0, selected.root_id, second).unwrap()),
            second: Some(AuthoritySlotV1::new(1, selected.root_id, first).unwrap()),
        },
        [0x83; 16],
    )
    .unwrap();
    selected.authority = third.canonical_bytes().to_vec();
    selected.attempt = Some(SavedAttempt::from_permit(&permit, false));
    let encoded = serde_json::to_vec(&selected).unwrap();
    let restored: Receipt = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(restored.recovery_attempt().unwrap(), permit);
    assert_eq!(
        restored.recovery_attempt().unwrap().stage,
        AttemptStageV1::AttemptFrame
    );
    let mut changed = restored.clone();
    changed.attempt.as_mut().unwrap().requires_floor = true;
    assert!(changed.validate().is_err());
    changed = restored.clone();
    changed.attempt.as_mut().unwrap().writer_id = [0; 16];
    assert!(changed.validate().is_err());
    changed = restored.clone();
    changed.attempt.as_mut().unwrap().attempt_id = [0; 16];
    assert!(changed.validate().is_err());
    changed = restored.clone();
    changed.attempt.as_mut().unwrap().idempotency_key = [0; 16];
    assert!(changed.validate().is_err());
    changed = restored.clone();
    changed.attempt.as_mut().unwrap().expected_first = None;
    assert!(
        changed.validate().is_err(),
        "the complete predecessor pair is required"
    );
    changed = restored;
    changed.attempt.as_mut().unwrap().expected_second = Some(vec![0; 128]);
    assert!(changed.validate().is_err());
}

#[test]
fn selection_only_receipts_remain_readable_but_cannot_authorize_recovery() {
    let mut selected = receipt();
    selected.schema = SELECTION_ONLY_RECEIPT_SCHEMA.to_owned();
    assert!(
        selected.validate().is_err(),
        "a v2 attempt cannot be relabelled v1"
    );
    selected.attempt = None;
    selected.validate().unwrap();
    let encoded = serde_json::to_vec(&selected).unwrap();
    let value: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
    assert!(value.get("attempt").is_none());
    let restored: Receipt = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(restored.binding().unwrap(), selected.binding().unwrap());
    assert_eq!(restored.authority().unwrap(), selected.authority().unwrap());
    assert!(restored.recovery_attempt().is_err());
    selected.schema = RECEIPT_SCHEMA.to_owned();
    assert!(
        selected.validate().is_err(),
        "v2 requires its prepared attempt"
    );
}

#[test]
fn unknown_commit_reports_the_exact_attempt_instead_of_ordinary_failure() {
    let receipt = receipt();
    let permit = AttemptPermitV1 {
        root_id: receipt.root_id,
        writer_id: [0x61; 16],
        attempt_id: [0x62; 16],
        target_slot: 1,
        candidate: receipt.authority().unwrap(),
        expected: ExpectedAuthorityPairV1::default(),
        stage: AttemptStageV1::SlotReread,
        idempotency_key: [0x63; 16],
    };
    let error = publication_error(
        &receipt,
        "unknown",
        "not_activated",
        "reconciliation_required",
        Some(&permit),
    );
    let payload = publication_failure(error.as_ref()).unwrap();
    assert_eq!(payload["publication"], "unknown");
    assert_eq!(payload["serving"], "not_activated");
    let candidate: Vec<u8> =
        serde_json::from_value(payload["attempt"]["candidate"].clone()).unwrap();
    assert_eq!(
        AuthorityRefV1::from_canonical_bytes(&candidate).unwrap(),
        permit.candidate
    );
    assert_eq!(payload["attempt"]["stage"], "SlotReread");
    let committed = publication_error(
        &receipt,
        "committed",
        "not_activated",
        "engine_admission_refused",
        None,
    );
    assert_eq!(
        publication_failure(committed.as_ref()).unwrap()["publication"],
        "committed"
    );
    assert!(publication_failure(bad("ordinary input failure").as_ref()).is_none());
}

#[test]
fn root_receipt_and_work_destinations_are_disjoint() {
    separate_destinations(&[
        Path::new("/a/root"),
        Path::new("/a/work"),
        Path::new("/a/receipt"),
    ])
    .unwrap();
    for paths in [
        [Path::new("/a/root"), Path::new("/a/root/work")],
        [Path::new("/a/root/receipt"), Path::new("/a/root")],
        [Path::new("/a/root"), Path::new("/a/root")],
    ] {
        assert!(separate_destinations(&paths).is_err());
    }
}

#[test]
fn recovery_reports_every_verdict_without_turning_uncertainty_into_an_abort() {
    use frankensearch_core::generation::GenerationAuthorityErrorV1;

    let selected = receipt();
    let permit = selected.recovery_attempt().unwrap();
    let slot = AuthoritySlotV1::new(1, selected.root_id, permit.candidate).unwrap();
    let other = AuthorityRefV1::new(1, [0x91; 16], 256, [0x92; 32], None).unwrap();
    let fork = AuthoritySlotV1::new(1, selected.root_id, other).unwrap();
    let successor = AuthorityRefV1::new(
        2,
        [0x93; 16],
        256,
        [0x94; 32],
        Some(permit.candidate.fingerprint()),
    )
    .unwrap();
    let head = AuthoritySlotV1::new(0, selected.root_id, successor).unwrap();
    for (outcome, verdict, publication, serving, resolved) in [
        (
            ReconcileOutcomeV1::Committed { slot, floor: None },
            "committed",
            "committed",
            "not_activated",
            true,
        ),
        (
            ReconcileOutcomeV1::NotCommitted,
            "not_committed",
            "not_committed",
            "not_activated",
            true,
        ),
        (
            ReconcileOutcomeV1::StillUnknown(permit),
            "still_unknown",
            "unknown",
            "not_activated",
            false,
        ),
        (
            ReconcileOutcomeV1::Fork { observed: fork },
            "fork",
            "unknown",
            "not_activated",
            false,
        ),
        (
            ReconcileOutcomeV1::RepairRequired(GenerationAuthorityErrorV1::UnresolvedAttempt),
            "repair_required",
            "unknown",
            "not_activated",
            false,
        ),
        (
            ReconcileOutcomeV1::CommittedThenSuperseded { head },
            "committed_then_superseded",
            "committed",
            "superseded",
            true,
        ),
    ] {
        let report = reconciliation_report(&selected, &permit, &outcome);
        assert_eq!(report["reconciliation"], verdict);
        assert_eq!(report["publication"], publication);
        assert_eq!(report["serving"], serving);
        assert_eq!(report["ok"], resolved);
        assert_eq!(report["attempt"], attempt_report(&permit));
        let mut output = Vec::new();
        let result = emit_recovery(&mut output, report.clone());
        if resolved {
            result.unwrap();
            let emitted: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(emitted, report);
        } else {
            let error = result.unwrap_err();
            assert!(output.is_empty());
            assert_eq!(publication_failure(error.as_ref()).unwrap(), report);
        }
    }
}

#[test]
fn recovery_output_failure_preserves_the_exact_reconciliation_and_commit_knowledge() {
    struct Closed;
    impl Write for Closed {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed output"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let selected = receipt();
    let permit = selected.recovery_attempt().unwrap();
    let slot = AuthoritySlotV1::new(1, selected.root_id, permit.candidate).unwrap();
    let mut report = reconciliation_report(
        &selected,
        &permit,
        &ReconcileOutcomeV1::Committed { slot, floor: None },
    );
    report["serving"] = serde_json::json!("activated");
    let error = emit_recovery(&mut Closed, report).unwrap_err();
    let failure = publication_failure(error.as_ref()).unwrap();
    assert_eq!(failure["publication"], "committed");
    assert_eq!(failure["reconciliation"], "committed");
    assert_eq!(failure["serving"], "activated");
    assert_eq!(failure["reason"], "result_delivery_failed");
    assert_eq!(failure["attempt"], attempt_report(&permit));
}

#[test]
fn new_destinations_cannot_add_files_inside_an_existing_authority_root() {
    let parent = tempfile::tempdir().unwrap().keep();
    let root = parent.join("existing-root");
    fs::create_dir(&root).unwrap();
    fs::write(root.join("AUTHORITY"), [0; 8192]).unwrap();
    for destination in ["new-root", "new-work", "receipt.json"] {
        assert!(new_path(&root.join(destination)).is_err());
    }
    #[cfg(unix)]
    {
        let alias = parent.join("alias");
        std::os::unix::fs::symlink(&root, &alias).unwrap();
        assert!(new_path(&alias.join("receipt.json")).is_err());
    }
    assert_eq!(
        fs::read_dir(&root).unwrap().count(),
        1,
        "refusal must precede any destination mutation"
    );
    assert_eq!(fs::read(root.join("AUTHORITY")).unwrap(), vec![0; 8192]);
}

#[test]
fn failed_result_delivery_preserves_known_commit_and_activation() {
    struct BrokenOutput;
    impl Write for BrokenOutput {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed output"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let error = emit_indexed(
        &mut BrokenOutput,
        &receipt(),
        Path::new("/private/receipt.json"),
        NativeBuildRetrieval::Hnsw {
            params: HnswParams::default(),
            seed: 42,
        },
        None,
    )
    .unwrap_err();
    let payload = publication_failure(error.as_ref()).unwrap();
    assert_eq!(payload["publication"], "committed");
    assert_eq!(payload["serving"], "activated");
    assert_eq!(payload["reason"], "result_delivery_failed");
}

#[test]
fn indexed_report_exposes_the_realized_retrieval_without_changing_the_receipt() {
    let ann = NativeBuildRetrieval::Hnsw {
        params: HnswParams::default(),
        seed: 42,
    };
    let exact = NativeBuildRetrieval::Exact;
    for (primary, quality, label) in [
        (ann, None, "hnsw"),
        (exact, None, "exact"),
        (exact, Some(ann), "mixed"),
        (ann, Some(exact), "mixed"),
        (ann, Some(ann), "hnsw"),
    ] {
        let selected = if quality.is_some() {
            progressive_receipt()
        } else {
            receipt()
        };
        let mut output = Vec::new();
        emit_indexed(
            &mut output,
            &selected,
            Path::new("/receipt.json"),
            primary,
            quality,
        )
        .unwrap();
        let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(report["retrieval"], label);
        assert_eq!(report["retrieval_by_tier"]["fast"], retrieval_name(primary));
        if let Some(quality) = quality {
            assert_eq!(
                report["retrieval_by_tier"]["quality"],
                retrieval_name(quality)
            );
        } else {
            assert!(report["retrieval_by_tier"].get("quality").is_none());
        }
        assert_eq!(
            report["selection"],
            serde_json::to_value(&selected).unwrap()
        );
        assert_eq!(report["publication"], "committed");
        assert_eq!(report["serving"], "activated");
    }
}

#[cfg(target_os = "linux")]
#[test]
fn qualified_authority_cli_publishes_and_reopens_fast_and_quality_generations() {
    use frankensearch::IndexableDocument;
    use std::os::unix::fs::DirBuilderExt as _;
    use std::time::{SystemTime, UNIX_EPOCH};

    // Same real-host discipline as generation_root's tests. Qualification
    // failures remain failures: no overlay/tmpfs bypass or automatic skip.
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let parent = PathBuf::from(std::env::var_os("HOME").unwrap()).join(format!(
        ".frankensearch-authority-cli-tests-{}-{stamp}",
        std::process::id(),
    ));
    fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        // A durable receipt can precede every attempt write. Recovery must prove
        // non-commit without requiring model files or staged component images.
        let unissued_path = parent.join("unissued-root");
        drop(create_root(&unissued_path).expect("qualified Linux root is required"));
        let mut unissued = receipt();
        unissued.root = unissued_path.clone();
        let unissued_receipt = parent.join("unissued.json");
        save_receipt(&unissued, &unissued_receipt).unwrap();
        let unissued_options = Options::parse([
            "authority-recover".to_owned(),
            "--receipt".to_owned(),
            unissued_receipt.to_str().unwrap().to_owned(),
        ])
        .unwrap()
        .unwrap();
        let mut output = Vec::new();
        recover(&cx, &unissued_options, &mut output).unwrap();
        let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(report["publication"], "not_committed");
        assert_eq!(report["reconciliation"], "not_committed");
        assert_eq!(report["serving"], "not_activated");
        assert_eq!(
            fs::read(unissued_path.join("AUTHORITY")).unwrap(),
            vec![0; 8192]
        );
        assert_eq!(fs::read(unissued_path.join("LOCK")).unwrap(), vec![0; 8192]);

        for (name, tier, ann) in [
            ("fast-exact", EmbedderTierTag::Fast, false),
            ("quality-exact", EmbedderTierTag::Quality, false),
            ("fast-hnsw", EmbedderTierTag::Fast, true),
            ("quality-hnsw", EmbedderTierTag::Quality, true),
        ] {
            let retrieval = if ann {
                NativeBuildRetrieval::Hnsw {
                    params: HnswParams::default(),
                    seed: 42,
                }
            } else {
                NativeBuildRetrieval::Exact
            };
            let mode = if ann {
                NativeRetrievalMode::Ann
            } else {
                NativeRetrievalMode::Exact {
                    reason: NativeExactReason::Requested,
                }
            };
            let root_path = parent.join(format!("{name}-root"));
            let root = create_root(&root_path).expect("qualified Linux root is required");
            let work = parent.join(format!("{name}-work"));
            let receipt_path = parent.join(format!("{name}.json"));
            let provider = Arc::new(Provider {
                identity: provider_identity(),
                calls: AtomicUsize::new(0),
            });
            let model: Arc<dyn Embedder> = provider.clone();
            let built = NativeIndexBuilder::new(&work, new_generation(1).unwrap(), model)
                .unwrap()
                .with_fast_storage(NativeBuildPrecision::F32, retrieval)
                .add_documents([
                    IndexableDocument::new("zeta", "retry network requests")
                        .with_metadata("version", "original"),
                    IndexableDocument::new("alpha", "garden flowers"),
                ])
                .build_exact_generation(
                    &cx,
                    tier,
                    CommitRange { low: 2, high: 7 },
                    DEFAULT_IMAGE_BYTES,
                )
                .await
                .unwrap();
            assert_eq!(built.retrieval_mode(), mode);
            let published = stage_and_publish(
                &cx,
                &root,
                &root_path,
                &receipt_path,
                &built,
                None,
                PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
            )
            .unwrap()
            .receipt;
            let successor = built
                .candidate(
                    new_generation(1).unwrap().nonce,
                    Some(published.authority().unwrap()),
                    GenerationAuthorityActionV1::Rollback,
                    [0x71; 32],
                )
                .unwrap();
            drop(published);
            drop(built);
            drop(root);
            let selected = Receipt::read(&receipt_path).unwrap();
            let recorded = selected.recovery_attempt().unwrap();
            // Reproduce the on-disk cut after the slot barrier but before the
            // owner frame. Only the pre-publication receipt is used to recover.
            let mut lock = OpenOptions::new()
                .write(true)
                .open(root_path.join("LOCK"))
                .unwrap();
            lock.write_all(&[0; 4096]).unwrap();
            lock.sync_all().unwrap();
            drop(lock);
            assert!(selected.open(&cx, DEFAULT_IMAGE_BYTES).is_err());
            let authority_before_recovery = fs::read(root_path.join("AUTHORITY")).unwrap();
            let lock_before_recovery = fs::read(root_path.join("LOCK")).unwrap();
            let mut altered = selected.clone();
            altered.attempt.as_mut().unwrap().writer_id[0] ^= 1;
            let altered_path = parent.join(format!("{name}-altered.json"));
            save_receipt(&altered, &altered_path).unwrap();
            let mut recovery = Options::parse([
                "authority-recover".to_owned(),
                "--receipt".to_owned(),
                altered_path.to_str().unwrap().to_owned(),
            ])
            .unwrap()
            .unwrap();
            let error = recover(&cx, &recovery, &mut Vec::new()).unwrap_err();
            let refused = publication_failure(error.as_ref()).unwrap();
            assert_eq!(refused["publication"], "unknown");
            assert_eq!(refused["reconciliation"], "repair_required");
            assert_eq!(
                fs::read(root_path.join("AUTHORITY")).unwrap(),
                authority_before_recovery
            );
            assert_eq!(
                fs::read(root_path.join("LOCK")).unwrap(),
                lock_before_recovery
            );

            recovery.receipt = receipt_path.clone();
            let calls_before_recovery = provider.calls.load(Ordering::Relaxed);
            let mut recovered_output = Vec::new();
            recover(&cx, &recovery, &mut recovered_output).unwrap();
            let recovered: serde_json::Value = serde_json::from_slice(&recovered_output).unwrap();
            assert_eq!(recovered["publication"], "committed");
            assert_eq!(recovered["reconciliation"], "committed");
            assert_eq!(recovered["serving"], "activated");
            assert_eq!(recovered["retrieval"], retrieval_name(retrieval));
            assert_eq!(recovered["attempt"], attempt_report(&recorded));
            assert_eq!(
                provider.calls.load(Ordering::Relaxed),
                calls_before_recovery
            );
            assert_eq!(
                fs::read(root_path.join("AUTHORITY")).unwrap(),
                authority_before_recovery
            );
            let owner = admit_root(&root_path)
                .unwrap()
                .authority_publisher(selected.root_id, [0xa1; 16], PROFILE)
                .unwrap()
                .owner_frame()
                .unwrap()
                .unwrap();
            assert_eq!(owner.writer_id, recorded.writer_id);
            assert_eq!(owner.attempt_id, recorded.attempt_id);
            recovery.max_image_bytes = 1;
            let error = recover(&cx, &recovery, &mut Vec::new()).unwrap_err();
            let bounded = publication_failure(error.as_ref()).unwrap();
            assert_eq!(bounded["publication"], "committed");
            assert_eq!(bounded["reconciliation"], "committed");
            assert_eq!(bounded["serving"], "not_activated");
            assert_eq!(bounded["reason"], "snapshot_activation_deferred_or_refused");
            assert_eq!(
                provider.calls.load(Ordering::Relaxed),
                calls_before_recovery
            );
            recovery.max_image_bytes = DEFAULT_IMAGE_BYTES;
            assert!(selected.open(&cx, 1).is_err());
            let before = provider.calls.load(Ordering::Relaxed);
            let snapshot = selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            assert_eq!(snapshot.retrieval_mode(), mode);
            assert_eq!(
                provider.calls.load(Ordering::Relaxed),
                before,
                "reopening must not re-embed documents"
            );
            let results = snapshot
                .search_text(&cx, provider.as_ref(), "retry network", 2)
                .await
                .unwrap();
            assert_eq!(results[0].doc_id, "zeta");
            assert_eq!(snapshot.tier(), tier);
            assert_eq!(snapshot.metadata().total_documents, 2);
            assert_eq!(provider.calls.load(Ordering::Relaxed), before + 1);
            let mut wrong = selected.clone();
            wrong.tier = if tier == EmbedderTierTag::Fast {
                EmbedderTierTag::Quality
            } else {
                EmbedderTierTag::Fast
            };
            assert!(wrong.open(&cx, DEFAULT_IMAGE_BYTES).is_err());
            wrong = selected.clone();
            wrong.generation.nonce[0] ^= 1;
            assert!(wrong.open(&cx, DEFAULT_IMAGE_BYTES).is_err());
            // A valid new authority selecting the same image bytes still must
            // not be adopted through an old caller-owned selection receipt.
            let root = admit_root(&root_path).unwrap();
            write_immutable(
                &cx,
                &root_path.join(successor.manifest_name()),
                successor.manifest_bytes(),
            )
            .unwrap();
            let mut activation_writer =
                GenerationPublisherV1::new(&root, selected.root_id, [0x72; 16], PROFILE).unwrap();
            assert!(matches!(
                activation_writer
                    .publish(
                        &cx,
                        &successor,
                        snapshot.retained_snapshot().authority_pair(),
                        None,
                        [0x73; 16],
                        DEFAULT_IMAGE_BYTES as u64,
                    )
                    .unwrap()
                    .publication,
                PublicationOutcomeV1::Committed { .. }
            ));
            assert!(selected.open(&cx, DEFAULT_IMAGE_BYTES).is_err());
            let before_superseded = provider.calls.load(Ordering::Relaxed);
            let mut superseded_output = Vec::new();
            recover(&cx, &recovery, &mut superseded_output).unwrap();
            let superseded: serde_json::Value = serde_json::from_slice(&superseded_output).unwrap();
            assert_eq!(superseded["publication"], "committed");
            assert_eq!(superseded["reconciliation"], "committed_then_superseded");
            assert_eq!(superseded["serving"], "superseded");
            assert_eq!(provider.calls.load(Ordering::Relaxed), before_superseded);
            assert!(
                !snapshot
                    .search_text(&cx, provider.as_ref(), "retry", 1)
                    .await
                    .unwrap()
                    .is_empty()
            );
        }
    });
}

#[cfg(target_os = "linux")]
#[test]
fn qualified_rebuild_reuses_images_replaces_the_cohort_and_fences_a_losing_publisher() {
    use frankensearch::IndexableDocument;
    use frankensearch_core::generation::GenerationComponentRole;
    use std::os::unix::fs::{DirBuilderExt as _, MetadataExt as _};
    use std::time::{SystemTime, UNIX_EPOCH};

    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let parent = PathBuf::from(std::env::var_os("HOME").unwrap()).join(format!(
        ".frankensearch-authority-rebuild-tests-{}-{stamp}",
        std::process::id(),
    ));
    fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
    asupersync::test_utils::run_test_with_cx(|cx| async move {
        for (name, tier) in [
            ("fast", EmbedderTierTag::Fast),
            ("quality", EmbedderTierTag::Quality),
        ] {
            let root_path = parent.join(format!("{name}-root"));
            let root = create_root(&root_path).expect("qualified Linux root is required");
            let provider = Arc::new(Provider {
                identity: provider_identity(),
                calls: AtomicUsize::new(0),
            });
            let model: Arc<dyn Embedder> = provider.clone();
            let original_documents = || {
                [
                    IndexableDocument::new("zeta", "retry network requests")
                        .with_metadata("version", "old"),
                    IndexableDocument::new("alpha", "garden flowers"),
                ]
            };
            let first = NativeIndexBuilder::new(
                parent.join(format!("{name}-first-work")),
                new_generation(1).unwrap(),
                model.clone(),
            )
            .unwrap()
            .add_documents(original_documents())
            .build_exact_generation(
                &cx,
                tier,
                CommitRange { low: 2, high: 7 },
                DEFAULT_IMAGE_BYTES,
            )
            .await
            .unwrap();
            let first_path = parent.join(format!("{name}-first.json"));
            let selected = stage_and_publish(
                &cx,
                &root,
                &root_path,
                &first_path,
                &first,
                None,
                PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
            )
            .unwrap()
            .receipt;
            let retained = selected
                .open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES)
                .unwrap();
            let lexical_receipt = retained.retained_snapshot().manifest().components().lexical;
            let lexical_path = root_path.join(
                frankensearch_index::generation_root::generation_reader::component_object_name_v1(
                    GenerationComponentRole::Lexical,
                    lexical_receipt.sha256,
                ),
            );
            let original_inode = fs::metadata(&lexical_path).unwrap().ino();
            let original_lexical = fs::read(&lexical_path).unwrap();
            let first_receipt_bytes = fs::read(&first_path).unwrap();

            // An unchanged full cohort shares its exact lexical object. A
            // create_new-only pathname writer would fail this valid successor.
            let unchanged = retained
                .begin_model_migration(
                    &cx,
                    parent.join(format!("{name}-unchanged-work")),
                    new_generation(2).unwrap(),
                    model.clone(),
                    None,
                    crate::MAX_INPUT_BYTES,
                )
                .unwrap()
                .build_exact_generation(
                    &cx,
                    tier,
                    CommitRange { low: 2, high: 7 },
                    DEFAULT_IMAGE_BYTES,
                )
                .await
                .unwrap();
            assert_eq!(
                first.component_bytes(GenerationComponentRole::Lexical),
                unchanged.component_bytes(GenerationComponentRole::Lexical)
            );
            assert!(
                !parent
                    .join(format!("{name}-unchanged-work/lexical"))
                    .exists(),
                "unchanged source must use its admitted bytes without rebuilding Quill",
            );
            for change in ["content", "title", "metadata", "source-cut"] {
                let mut document = original_documents()[0].clone();
                match change {
                    "content" => document.content.push_str(" changed"),
                    "title" => document.title = Some("Changed title".to_owned()),
                    "metadata" => {
                        document
                            .metadata
                            .insert("version".to_owned(), "changed".to_owned());
                    }
                    "source-cut" => {}
                    _ => unreachable!(),
                }
                let changed = retained
                    .begin_model_migration(
                        &cx,
                        parent.join(format!("{name}-lexical-{change}")),
                        new_generation(2).unwrap(),
                        model.clone(),
                        None,
                        crate::MAX_INPUT_BYTES,
                    )
                    .unwrap()
                    .upsert_document(document)
                    .build_exact_generation(
                        &cx,
                        tier,
                        CommitRange {
                            low: 2,
                            high: if change == "source-cut" { 8 } else { 7 },
                        },
                        DEFAULT_IMAGE_BYTES,
                    )
                    .await
                    .unwrap();
                assert_ne!(
                    first.component_bytes(GenerationComponentRole::Lexical),
                    changed.component_bytes(GenerationComponentRole::Lexical),
                    "{name}: {change}"
                );
                let candidate = changed
                    .candidate(
                        new_generation(1).unwrap().nonce,
                        Some(retained.authority()),
                        GenerationAuthorityActionV1::Activate,
                        [0x44; 32],
                    )
                    .unwrap();
                changed.validate_candidate(&cx, &candidate).unwrap();
            }
            let second_path = parent.join(format!("{name}-second.json"));
            let second = stage_and_publish(
                &cx,
                &root,
                &root_path,
                &second_path,
                &unchanged,
                Some(&retained),
                PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
            )
            .unwrap()
            .receipt;
            assert_eq!(second.root_id, selected.root_id);
            assert_eq!(second.authority().unwrap().sequence, 2);
            assert_eq!(fs::metadata(&lexical_path).unwrap().ino(), original_inode);
            assert_eq!(fs::read(&lexical_path).unwrap(), original_lexical);
            let second_snapshot = second
                .open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES)
                .unwrap();
            assert_eq!(
                second.recovery_attempt().unwrap().expected,
                retained.retained_snapshot().authority_pair()
            );

            let winner = NativeIndexBuilder::new(
                parent.join(format!("{name}-winner-work")),
                new_generation(3).unwrap(),
                model.clone(),
            )
            .unwrap()
            .add_documents([
                IndexableDocument::new("zeta", "retry requests safely")
                    .with_metadata("version", "new"),
                IndexableDocument::new("beta", "fresh garden vegetables"),
            ])
            .build_exact_generation(
                &cx,
                tier,
                CommitRange { low: 2, high: 8 },
                DEFAULT_IMAGE_BYTES,
            )
            .await
            .unwrap();
            let loser = NativeIndexBuilder::new(
                parent.join(format!("{name}-loser-work")),
                new_generation(3).unwrap(),
                model,
            )
            .unwrap()
            .add_documents([IndexableDocument::new("loser", "retry obsolete requests")])
            .build_exact_generation(
                &cx,
                tier,
                CommitRange { low: 2, high: 9 },
                DEFAULT_IMAGE_BYTES,
            )
            .await
            .unwrap();
            let winner_path = parent.join(format!("{name}-winner.json"));
            let committed = stage_and_publish(
                &cx,
                &root,
                &root_path,
                &winner_path,
                &winner,
                Some(&second_snapshot),
                PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
            )
            .unwrap()
            .receipt;
            let authority_after_winner = fs::read(root_path.join("AUTHORITY")).unwrap();
            let lock_after_winner = fs::read(root_path.join("LOCK")).unwrap();
            let loser_path = parent.join(format!("{name}-loser.json"));
            let error = stage_and_publish(
                &cx,
                &root,
                &root_path,
                &loser_path,
                &loser,
                Some(&second_snapshot),
                PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
            )
            .err()
            .expect("the competing publication must lose");
            assert_eq!(
                publication_failure(error.as_ref()).unwrap()["publication"],
                "not_committed"
            );
            assert_eq!(
                fs::read(root_path.join("AUTHORITY")).unwrap(),
                authority_after_winner
            );
            assert_eq!(fs::read(root_path.join("LOCK")).unwrap(), lock_after_winner);
            assert_eq!(fs::read(&first_path).unwrap(), first_receipt_bytes);
            assert_eq!(fs::read(&lexical_path).unwrap(), original_lexical);
            let lost = Receipt::read(&loser_path).unwrap();
            assert_eq!(
                lost.recovery_attempt().unwrap().expected,
                second_snapshot.retained_snapshot().authority_pair()
            );
            assert_ne!(lost.authority().unwrap(), committed.authority().unwrap());
            assert!(selected.open(&cx, DEFAULT_IMAGE_BYTES).is_err());
            assert!(second.open(&cx, DEFAULT_IMAGE_BYTES).is_err());

            let active = committed.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            let current_results = active
                .search_text(&cx, provider.as_ref(), "retry garden", 10)
                .await
                .unwrap();
            let current_ids: BTreeSet<_> = current_results
                .iter()
                .map(|hit| hit.doc_id.as_str())
                .collect();
            assert_eq!(current_ids, BTreeSet::from(["beta", "zeta"]));
            let old_results = retained
                .search_text(&cx, provider.as_ref(), "retry garden", 10)
                .await
                .unwrap();
            let old_ids: BTreeSet<_> = old_results.iter().map(|hit| hit.doc_id.as_str()).collect();
            assert_eq!(old_ids, BTreeSet::from(["alpha", "zeta"]));
            assert_eq!(
                active.metadata().commit_range,
                CommitRange { low: 2, high: 8 }
            );
            assert_eq!(
                retained.metadata().commit_range,
                CommitRange { low: 2, high: 7 }
            );
            let calls = provider.calls.load(Ordering::Relaxed);
            let recovery = Options::parse([
                "authority-recover".to_owned(),
                "--receipt".to_owned(),
                winner_path.to_str().unwrap().to_owned(),
            ])
            .unwrap()
            .unwrap();
            let mut output = Vec::new();
            recover(&cx, &recovery, &mut output).unwrap();
            let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(report["publication"], "committed");
            assert_eq!(report["serving"], "activated");
            assert_eq!(provider.calls.load(Ordering::Relaxed), calls);

            // A stale receipt is refused before even trying the deliberately
            // missing input or models, and cannot create new work/receipt files.
            let mut options = parse(&rebuild_arguments()).unwrap().unwrap();
            options.from_receipt = Some(first_path);
            let refused_work = parent.join(format!("{name}-stale-work"));
            let refused_receipt = parent.join(format!("{name}-stale.json"));
            options.work = Some(refused_work.clone());
            options.receipt = refused_receipt.clone();
            options.input = Some(parent.join("missing-input.jsonl"));
            options.models = Some(parent.join("missing-models"));
            let error = rebuild(&cx, &options, &mut Vec::new(), None)
                .await
                .unwrap_err();
            assert!(error.to_string().contains("head differs"));
            assert!(!refused_work.exists());
            assert!(!refused_receipt.exists());
            assert_eq!(
                fs::read(root_path.join("AUTHORITY")).unwrap(),
                authority_after_winner
            );
        }
    });
}

#[cfg(target_os = "linux")]
mod warm_authority {
    mod model_activation_tests {
        include!("authority_model_activation_tests.rs");
    }

    mod reindex_tests {
        include!("authority_reindex_tests.rs");
    }

    use std::io::{BufRead, Cursor, Read};
    use std::os::unix::fs::DirBuilderExt as _;
    use std::time::{SystemTime, UNIX_EPOCH};

    use frankensearch::IndexableDocument;
    use frankensearch_core::generation::GenerationComponentRole;
    use frankensearch_index::generation_root::generation_reader::{
        activation_manifest_name_v1, component_object_name_v1,
    };

    use super::*;

    fn private_parent(label: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let parent = PathBuf::from(std::env::var_os("HOME").unwrap()).join(format!(
            ".frankensearch-authority-serve-{label}-{}-{stamp}",
            std::process::id()
        ));
        // Qualification uses the actual host filesystem. Unsupported roots
        // remain test failures; no fixture bypass admits overlay/tmpfs.
        fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
        parent
    }

    async fn publish(
        cx: &Cx,
        parent: &Path,
        previous: Option<&NativeGenerationSnapshot>,
        provider: &Arc<Provider>,
        tier: EmbedderTierTag,
        label: &str,
        ann: bool,
    ) -> Receipt {
        let root_path = parent.join("root");
        let root = if previous.is_some() {
            admit_root(&root_path).unwrap()
        } else {
            create_root(&root_path).expect("qualified Linux root is required")
        };
        let sequence = previous.map_or(1, |snapshot| snapshot.authority().sequence + 1);
        let model: Arc<dyn Embedder> = provider.clone();
        let retrieval = if ann {
            NativeBuildRetrieval::Hnsw {
                params: HnswParams::default(),
                seed: 17,
            }
        } else {
            NativeBuildRetrieval::Exact
        };
        let built = NativeIndexBuilder::new(
            parent.join(format!("{label}-work")),
            new_generation(sequence).unwrap(),
            model,
        )
        .unwrap()
        .with_fast_storage(NativeBuildPrecision::F32, retrieval)
        .add_documents([
            IndexableDocument::new(format!("retry-{label}"), "retry network requests")
                .with_metadata("version", label),
            IndexableDocument::new(format!("garden-{label}"), "garden flowers")
                .with_metadata("version", label),
        ])
        .build_exact_generation(
            cx,
            tier,
            CommitRange {
                low: 1,
                high: sequence + 4,
            },
            DEFAULT_IMAGE_BYTES,
        )
        .await
        .unwrap();
        stage_and_publish(
            cx,
            &root,
            &root_path,
            &parent.join(format!("{label}.json")),
            &built,
            previous,
            PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
        )
        .unwrap()
        .receipt
    }

    fn open_session(
        cx: &Cx,
        selected: Receipt,
        provider: &Arc<Provider>,
        allow_activation: bool,
    ) -> AuthoritySession {
        let root = admit_root(&selected.root).unwrap();
        let snapshot = selected
            .open_on_root(cx, &root, DEFAULT_IMAGE_BYTES)
            .unwrap();
        let model: Arc<dyn Embedder> = provider.clone();
        AuthoritySession::new(
            cx,
            root,
            selected,
            snapshot,
            Models {
                fast: model,
                quality: None,
            },
            (1, DEFAULT_IMAGE_BYTES),
            allow_activation,
        )
        .unwrap()
    }

    fn frames(bytes: &[u8]) -> Vec<serde_json::Value> {
        bytes
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect()
    }

    async fn exchange(
        cx: &Cx,
        session: &mut AuthoritySession,
        messages: &[serde_json::Value],
    ) -> Vec<serde_json::Value> {
        let mut input = Vec::new();
        for message in messages {
            serde_json::to_writer(&mut input, message).unwrap();
            input.push(b'\n');
        }
        let mut output = Vec::new();
        serve_session(cx, session, &mut Cursor::new(input), &mut output)
            .await
            .unwrap();
        let frames = frames(&output);
        assert_eq!(frames.len(), messages.len() + 1);
        assert_eq!(frames[0]["event"], "ready");
        frames
    }

    fn provider() -> Arc<Provider> {
        Arc::new(Provider {
            identity: provider_identity(),
            calls: AtomicUsize::new(0),
        })
    }

    async fn publish_source_fixture(
        cx: &Cx,
        parent: &Path,
        provider: &Arc<Provider>,
        tier: EmbedderTierTag,
        ann: bool,
        documents: Vec<IndexableDocument>,
    ) -> Receipt {
        let root_path = parent.join("root");
        let root = create_root(&root_path).expect("qualified Linux root is required");
        let model: Arc<dyn Embedder> = provider.clone();
        let retrieval = if ann {
            NativeBuildRetrieval::Hnsw {
                params: HnswParams::default(),
                seed: 17,
            }
        } else {
            NativeBuildRetrieval::Exact
        };
        let built = NativeIndexBuilder::new(
            parent.join("source-work"),
            new_generation(1).unwrap(),
            model,
        )
        .unwrap()
        .with_fast_storage(NativeBuildPrecision::F32, retrieval)
        .add_documents(documents)
        .build_exact_generation(
            cx,
            tier,
            CommitRange { low: 1, high: 5 },
            DEFAULT_IMAGE_BYTES,
        )
        .await
        .unwrap();
        stage_and_publish(
            cx,
            &root,
            &root_path,
            &parent.join("source.json"),
            &built,
            None,
            PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
        )
        .unwrap()
        .receipt
    }

    fn fetch_message(
        selected: &Receipt,
        id: &str,
        documents: &serde_json::Value,
    ) -> serde_json::Value {
        serde_json::json!({
            "op":"fetch", "id":id, "expected_authority":selected.authority,
            "documents":documents,
        })
    }

    #[test]
    fn authority_fetch_requests_require_an_exact_selection_and_strict_window_fields() {
        let valid = fetch_message(&receipt(), "source", &serde_json::json!([{"doc_id":"a"}]));
        assert!(matches!(
            serde_json::from_value::<AuthorityRequest>(valid.clone()).unwrap(),
            AuthorityRequest::Fetch(_)
        ));
        for field in ["op", "expected_authority", "documents"] {
            let mut missing = valid.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(
                serde_json::from_value::<AuthorityRequest>(missing).is_err(),
                "{field}"
            );
        }
        for field in [
            "expected_generation",
            "query",
            "receipt",
            "path",
            "timeout_ms",
            "model_dir",
        ] {
            let mut unknown = valid.clone();
            unknown[field] = serde_json::json!("private-request-sentinel");
            assert!(
                serde_json::from_value::<AuthorityRequest>(unknown).is_err(),
                "{field}"
            );
        }
        for document in [
            serde_json::json!({"doc_id":"a", "path":"private-request-sentinel"}),
            serde_json::json!({"doc_id":"a", "offset_bytes":-1}),
            serde_json::json!({"doc_id":"a", "include_metadata":"true"}),
        ] {
            let mut malformed = valid.clone();
            malformed["documents"] = serde_json::json!([document]);
            assert!(serde_json::from_value::<AuthorityRequest>(malformed).is_err());
        }
    }

    #[test]
    fn authority_fetch_reads_selective_utf8_windows_in_order_without_model_work_or_grants() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (label, tier, ann) in [
                ("fetch-fast-exact", EmbedderTierTag::Fast, false),
                ("fetch-quality-hnsw", EmbedderTierTag::Quality, true),
            ] {
                let parent = private_parent(label);
                let provider = provider();
                let content = "  café🦀\n東京\u{0}<script>&\r\n";
                let requested = IndexableDocument::new("unicode", content)
                    .with_title("Retained title")
                    .with_metadata("tenant", "A");
                let source_budget = requested.id.len()
                    + requested.content.len()
                    + requested.title.as_deref().map_or(0, str::len)
                    + serde_json::to_vec(&requested.metadata).unwrap().len();
                let selected = publish_source_fixture(
                    &cx,
                    &parent,
                    &provider,
                    tier,
                    ann,
                    vec![
                        requested,
                        IndexableDocument::new("unrequested", "garden ".repeat(1_024)),
                    ],
                )
                .await;
                // The complete corpus and repeated full copies of unicode do
                // not fit this budget. Its one distinct requested source does.
                let mut session = open_session(&cx, selected.clone(), &provider, false)
                    .with_filter(&cx, None, source_budget)
                    .unwrap();
                let request = fetch_message(
                    &selected,
                    "windows",
                    &serde_json::json!([
                        {"doc_id":"unicode", "max_bytes":4, "include_metadata":true},
                        {"doc_id":"missing-first"},
                        {"doc_id":"unicode", "offset_bytes":4, "max_bytes":4},
                        {"doc_id":"unicode", "offset_bytes":7, "max_bytes":4},
                        {"doc_id":"unicode"},
                        {"doc_id":"unicode", "offset_bytes":content.len(), "max_bytes":4},
                        {"doc_id":"missing-last", "include_metadata":true}
                    ]),
                );
                let before = provider.calls.load(Ordering::Relaxed);
                let output = exchange(&cx, &mut session, &[request]).await;
                assert_eq!(output[0]["activation_enabled"], false);
                assert_eq!(output[0]["updates_enabled"], false);
                assert_eq!(output[0]["fetch_enabled"], true);
                assert_eq!(output[0]["max_fetch_documents"], 32);
                assert_eq!(output[0]["max_fetch_content_bytes"], 65_536);
                assert_eq!(output[0]["max_fetch_response_bytes"], 1024 * 1024);
                let result = &output[1];
                assert_eq!(result["event"], "terminal");
                assert_eq!(result["operation"], "fetch");
                assert_eq!(result["status"], "complete");
                assert_eq!(result["source"], "retained_snapshot");
                assert_eq!(result["partial_results"], false);
                assert_eq!(result["authority"], serde_json::json!(selected.authority));
                assert_eq!(result["generation"], serde_json::json!(selected.generation));
                assert_eq!(result["request"], 1);
                assert_eq!(result["id"], "windows");
                assert!(result.get("scope").is_none());
                let rows = result["documents"].as_array().unwrap();
                assert_eq!(rows.len(), 7);
                assert_eq!(rows[0]["content"], "  ca");
                assert_eq!(rows[0]["end_offset_bytes"], 4);
                assert_eq!(rows[0]["next_offset_bytes"], 4);
                assert_eq!(rows[0]["title"], "Retained title");
                assert_eq!(rows[0]["metadata"], serde_json::json!({"tenant":"A"}));
                assert_eq!(
                    rows[1],
                    serde_json::json!({"status":"unavailable", "doc_id":"missing-first"})
                );
                assert_eq!(rows[2]["content"], "fé");
                assert_eq!(rows[2]["offset_bytes"], 4);
                assert_eq!(rows[2]["end_offset_bytes"], 7);
                assert_eq!(rows[2]["next_offset_bytes"], 7);
                assert_eq!(rows[3]["content"], "🦀");
                assert_eq!(rows[3]["end_offset_bytes"], 11);
                assert_eq!(rows[4]["content"], content);
                assert_eq!(rows[4]["total_bytes"], content.len());
                assert!(rows[4]["next_offset_bytes"].is_null());
                assert_eq!(rows[5]["content"], "");
                assert!(rows[5]["next_offset_bytes"].is_null());
                for row in &rows[2..6] {
                    assert!(row.get("metadata").is_none());
                    assert!(row.get("title").is_none());
                }
                assert_eq!(
                    rows[6],
                    serde_json::json!({"status":"unavailable", "doc_id":"missing-last"})
                );
                assert_eq!(provider.calls.load(Ordering::Relaxed), before);
            }
        });
    }

    #[test]
    fn authority_fetch_intersects_scopes_and_hides_deleted_missing_and_excluded_sources() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let parent = private_parent("fetch-scopes");
            let provider = provider();
            let old = publish_source_fixture(
                &cx,
                &parent,
                &provider,
                EmbedderTierTag::Quality,
                true,
                vec![
                    IndexableDocument::new("allowed", "retry allowed source")
                        .with_title("Allowed title")
                        .with_metadata("tenant", "A"),
                    IndexableDocument::new("excluded", "PRIVATE_EXCLUDED_SOURCE")
                        .with_title("PRIVATE_EXCLUDED_TITLE")
                        .with_metadata("tenant", "B"),
                    IndexableDocument::new("deleted", "PRIVATE_DELETED_SOURCE")
                        .with_metadata("tenant", "A"),
                ],
            )
            .await;
            let mut writer = open_session(&cx, old.clone(), &provider, false)
                .with_updates(2, crate::MAX_INPUT_BYTES)
                .unwrap();
            let before = provider.calls.load(Ordering::Relaxed);
            let deletion = update_message(
                &parent,
                "deleted-source",
                &old,
                6,
                &serde_json::json!([{"op":"delete", "id":"deleted"}]),
            );
            assert_eq!(
                exchange(&cx, &mut writer, &[deletion]).await[1]["event"],
                "updated"
            );
            let selected = Receipt::read(&parent.join("deleted-source.json")).unwrap();
            let fixed = crate::filter::Filter::parse(r#"{"metadata":{"tenant":"A"}}"#).unwrap();
            let mut session = open_session(&cx, selected.clone(), &provider, false)
                .with_filter(&cx, Some(fixed), crate::MAX_INPUT_BYTES)
                .unwrap();
            let windows = serde_json::json!([
                {"doc_id":"excluded", "offset_bytes":usize::MAX, "include_metadata":true},
                {"doc_id":"missing", "offset_bytes":usize::MAX, "include_metadata":true},
                {"doc_id":"deleted", "offset_bytes":usize::MAX, "include_metadata":true},
                {"doc_id":"allowed", "include_metadata":true},
                {"doc_id":"allowed"}
            ]);
            let ordinary = fetch_message(&selected, "fixed", &windows);
            let mut narrow = fetch_message(&selected, "narrow", &windows);
            narrow["filter"] = serde_json::json!({"ids":["excluded"]});
            let mut broad = fetch_message(&selected, "broad", &windows);
            broad["filter"] = serde_json::json!({});
            let output = exchange(&cx, &mut session, &[ordinary, narrow, broad]).await;
            for position in [1, 3] {
                assert_eq!(output[position]["event"], "terminal");
                assert_eq!(
                    output[position]["scope"],
                    serde_json::json!({"filtered":true})
                );
                let rows = output[position]["documents"].as_array().unwrap();
                for (row, id) in rows[..3].iter().zip(["excluded", "missing", "deleted"]) {
                    assert_eq!(
                        *row,
                        serde_json::json!({"status":"unavailable", "doc_id":id})
                    );
                }
                assert_eq!(rows[3]["content"], "retry allowed source");
                assert_eq!(rows[3]["metadata"]["tenant"], "A");
                assert_eq!(rows[3]["title"], "Allowed title");
                assert!(rows[4].get("metadata").is_none());
                assert!(rows[4].get("title").is_none());
            }
            assert_eq!(output[2]["scope"], serde_json::json!({"filtered":true}));
            assert!(
                output[2]["documents"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|row| row["status"] == "unavailable")
            );
            let encoded = serde_json::to_string(&output).unwrap();
            assert!(!encoded.contains("PRIVATE_EXCLUDED"));
            assert!(!encoded.contains("PRIVATE_DELETED"));
            let mut unscoped = open_session(&cx, selected.clone(), &provider, false);
            let mut requested_filter = fetch_message(
                &selected,
                "request-filter",
                &serde_json::json!([
                    {"doc_id":"excluded", "offset_bytes":usize::MAX, "include_metadata":true},
                    {"doc_id":"allowed", "include_metadata":true}
                ]),
            );
            requested_filter["filter"] = serde_json::json!({"metadata":{"tenant":"A"}});
            let output = exchange(&cx, &mut unscoped, &[requested_filter]).await;
            assert_eq!(output[1]["scope"], serde_json::json!({"filtered":true}));
            assert_eq!(
                output[1]["documents"][0],
                serde_json::json!({"status":"unavailable", "doc_id":"excluded"})
            );
            assert_eq!(output[1]["documents"][1]["content"], "retry allowed source");
            assert!(
                !serde_json::to_string(&output)
                    .unwrap()
                    .contains("PRIVATE_EXCLUDED")
            );
            assert_eq!(provider.calls.load(Ordering::Relaxed), before);
        });
    }

    #[test]
    fn authority_fetch_keeps_its_original_source_after_publication_and_path_movement() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let parent = private_parent("fetch-retained-paths");
            let provider = provider();
            let old = publish(
                &cx,
                &parent,
                None,
                &provider,
                EmbedderTierTag::Fast,
                "old",
                true,
            )
            .await;
            let mut session = open_session(&cx, old.clone(), &provider, false);
            let winner = publish(
                &cx,
                &parent,
                Some(&session.snapshot),
                &provider,
                EmbedderTierTag::Fast,
                "winner",
                true,
            )
            .await;
            let head = fs::read(old.root.join("AUTHORITY")).unwrap();
            fs::rename(parent.join("old-work"), parent.join("archived-work")).unwrap();
            fs::rename(&old.root, parent.join("relocated-root")).unwrap();
            let before = provider.calls.load(Ordering::Relaxed);
            let old_request = fetch_message(
                &old,
                "retained",
                &serde_json::json!([
                    {"doc_id":"retry-old", "include_metadata":true}, {"doc_id":"garden-old"}
                ]),
            );
            let new_request = fetch_message(
                &winner,
                "wrong-owner",
                &serde_json::json!([
                    {"doc_id":"retry-winner"}
                ]),
            );
            let output = exchange(&cx, &mut session, &[old_request, new_request]).await;
            assert_eq!(output[1]["event"], "terminal");
            assert_eq!(output[1]["authority"], serde_json::json!(old.authority));
            assert_eq!(
                output[1]["documents"][0]["content"],
                "retry network requests"
            );
            assert_eq!(output[1]["documents"][0]["metadata"]["version"], "old");
            assert_eq!(output[1]["documents"][1]["content"], "garden flowers");
            assert_eq!(output[2]["event"], "failed");
            assert_eq!(output[2]["operation"], "fetch");
            assert_eq!(output[2]["code"], "stale_expected_authority");
            assert!(output[2].get("documents").is_none());
            assert_eq!(provider.calls.load(Ordering::Relaxed), before);
            assert_eq!(
                fs::read(parent.join("relocated-root").join("AUTHORITY")).unwrap(),
                head
            );
            assert_eq!(session.receipt.authority, old.authority);
        });
    }

    #[test]
    fn authority_fetch_follows_session_updates_activation_and_rollback_authority() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let parent = private_parent("fetch-authority-transitions");
            let provider = provider();
            let old = publish(
                &cx,
                &parent,
                None,
                &provider,
                EmbedderTierTag::Quality,
                "old",
                true,
            )
            .await;
            let mut session = open_session(&cx, old.clone(), &provider, true)
                .with_updates(2, crate::MAX_INPUT_BYTES)
                .unwrap();
            let before = provider.calls.load(Ordering::Relaxed);
            let original_fetch = fetch_message(
                &old,
                "original-source",
                &serde_json::json!([
                    {"doc_id":"retry-old", "include_metadata":true}
                ]),
            );
            let output = exchange(
                &cx,
                &mut session,
                &[serde_json::json!({"query":"retry"}), original_fetch.clone()],
            )
            .await;
            assert_eq!(output[1]["results"][0]["doc_id"], "retry-old");
            assert_eq!(output[2]["authority"], output[1]["authority"]);
            assert_eq!(
                output[2]["documents"][0]["content"],
                "retry network requests"
            );
            let update = update_message(
                &parent,
                "edited-source",
                &old,
                6,
                &serde_json::json!([
                    {"op":"upsert", "id":"retry-old", "content":"retry changed source",
                     "title":"Changed title", "metadata":{"version":"edited"}},
                    {"op":"delete", "id":"garden-old"}
                ]),
            );
            assert_eq!(
                exchange(&cx, &mut session, &[update]).await[1]["event"],
                "updated"
            );
            let edited = Receipt::read(&parent.join("edited-source.json")).unwrap();
            let updated_fetch = fetch_message(
                &edited,
                "updated-source",
                &serde_json::json!([
                    {"doc_id":"retry-old", "include_metadata":true}, {"doc_id":"garden-old"}
                ]),
            );
            let output = exchange(
                &cx,
                &mut session,
                &[
                    original_fetch,
                    serde_json::json!({"query":"retry changed"}),
                    updated_fetch.clone(),
                ],
            )
            .await;
            assert_eq!(output[1]["code"], "stale_expected_authority");
            assert!(output[1].get("documents").is_none());
            assert_eq!(output[3]["authority"], output[2]["authority"]);
            assert_eq!(output[3]["documents"][0]["content"], "retry changed source");
            assert_eq!(output[3]["documents"][0]["title"], "Changed title");
            assert_eq!(output[3]["documents"][0]["metadata"]["version"], "edited");
            assert_eq!(
                output[3]["documents"][1],
                serde_json::json!({"status":"unavailable", "doc_id":"garden-old"})
            );
            assert_eq!(
                provider.calls.load(Ordering::Relaxed),
                before + 3,
                "two queries and one changed document infer; all source fetches remain model-free"
            );

            let external = publish(
                &cx,
                &parent,
                Some(&session.snapshot),
                &provider,
                EmbedderTierTag::Quality,
                "external",
                false,
            )
            .await;
            let before = provider.calls.load(Ordering::Relaxed);
            let external_fetch = fetch_message(
                &external,
                "external-source",
                &serde_json::json!([
                    {"doc_id":"retry-external", "include_metadata":true}
                ]),
            );
            let output = exchange(
                &cx,
                &mut session,
                &[
                    serde_json::json!({"op":"activate", "receipt":parent.join("external.json"),
                    "expected_authority":edited.authority}),
                    updated_fetch.clone(),
                    external_fetch.clone(),
                ],
            )
            .await;
            assert_eq!(output[1]["event"], "activated");
            assert_eq!(output[2]["code"], "stale_expected_authority");
            assert_eq!(output[3]["documents"][0]["metadata"]["version"], "external");

            let mut rollback_options = parse(&rollback_arguments()).unwrap().unwrap();
            rollback_options.from_receipt = Some(parent.join("external.json"));
            rollback_options.target_receipt = Some(parent.join("edited-source.json"));
            rollback_options.receipt = parent.join("restored-source.json");
            rollback(&cx, &rollback_options, &mut Vec::new()).unwrap();
            let restored = Receipt::read(&rollback_options.receipt).unwrap();
            assert_eq!(restored.generation, edited.generation);
            assert_ne!(restored.authority, edited.authority);
            let restored_fetch = fetch_message(
                &restored,
                "restored-source",
                &serde_json::json!([
                    {"doc_id":"retry-old", "include_metadata":true}
                ]),
            );
            let output = exchange(
                &cx,
                &mut session,
                &[
                    serde_json::json!({"op":"activate", "receipt":rollback_options.receipt,
                    "expected_authority":external.authority}),
                    updated_fetch,
                    external_fetch,
                    restored_fetch,
                ],
            )
            .await;
            assert_eq!(output[1]["event"], "activated");
            for position in [2, 3] {
                assert_eq!(output[position]["code"], "stale_expected_authority");
                assert!(output[position].get("documents").is_none());
            }
            assert_eq!(output[4]["event"], "terminal");
            assert_eq!(
                output[4]["authority"],
                serde_json::json!(restored.authority)
            );
            assert_eq!(output[4]["documents"][0]["content"], "retry changed source");
            assert_eq!(output[4]["documents"][0]["metadata"]["version"], "edited");
            assert_eq!(
                provider.calls.load(Ordering::Relaxed),
                before,
                "activation, rollback and retained source fetches require no inference"
            );
        });
    }

    #[test]
    fn authority_fetch_refusals_are_atomic_bounded_and_leave_the_next_request_usable() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let parent = private_parent("fetch-refusals");
            let provider = provider();
            let selected = publish_source_fixture(
                &cx,
                &parent,
                &provider,
                EmbedderTierTag::Fast,
                false,
                vec![IndexableDocument::new(
                    "unicode",
                    "🦀retained-source-canary",
                )],
            )
            .await;
            let mut session = open_session(&cx, selected.clone(), &provider, false);
            let before = provider.calls.load(Ordering::Relaxed);
            let valid = fetch_message(
                &selected,
                "valid",
                &serde_json::json!([{"doc_id":"unicode"}]),
            );
            let mut bad_authority = valid.clone();
            bad_authority["expected_authority"] = serde_json::json!([1, 2]);
            let mut stale = valid.clone();
            stale["expected_authority"] = serde_json::json!(receipt().authority);
            let mut bad_id = valid.clone();
            bad_id["id"] = serde_json::json!("bad\u{0}private-request-sentinel");
            let mut wrong_op = valid.clone();
            wrong_op["op"] = serde_json::json!("unsupported-source-operation");
            let mut rejected = vec![
                (bad_authority, "invalid_expected_authority"),
                (stale.clone(), "stale_expected_authority"),
                (bad_id, "invalid_request_id"),
                (wrong_op, "invalid_fetch_request"),
            ];
            for windows in [
                serde_json::json!([]),
                serde_json::json!([{"doc_id":" "}]),
                serde_json::json!([{"doc_id":"a\u{0}b"}]),
                serde_json::json!([{"doc_id":"unicode", "max_bytes":3}]),
                serde_json::json!([{"doc_id":"unicode", "max_bytes":65_537}]),
                serde_json::json!(vec![serde_json::json!({"doc_id":"unicode"}); 33]),
                serde_json::json!([{"doc_id":"x".repeat(65_536)}]),
                serde_json::json!([{"doc_id":"x".repeat(40_000)}, {"doc_id":"y".repeat(40_000)}]),
            ] {
                rejected.push((
                    fetch_message(&selected, "bad-windows", &windows),
                    "invalid_fetch_windows",
                ));
            }
            // The valid first window must never escape when a later available
            // document has an offset inside a UTF-8 scalar or beyond its end.
            for offset in [1, usize::MAX] {
                rejected.push((
                    fetch_message(
                        &selected,
                        "bad-offset",
                        &serde_json::json!([
                            {"doc_id":"unicode"}, {"doc_id":"unicode", "offset_bytes":offset}
                        ]),
                    ),
                    "fetch_refused",
                ));
            }
            let mut requests = rejected
                .iter()
                .map(|(request, _)| request.clone())
                .collect::<Vec<_>>();
            requests.push(valid.clone());
            let output = exchange(&cx, &mut session, &requests).await;
            for (frame, (_, code)) in output[1..output.len() - 1].iter().zip(&rejected) {
                assert_eq!(frame["event"], "failed");
                assert_eq!(frame["operation"], "fetch");
                assert_eq!(frame["status"], "failed");
                assert_eq!(frame["partial_results"], false);
                assert_eq!(frame["code"], *code);
                assert!(frame.get("documents").is_none());
                let encoded = serde_json::to_string(frame).unwrap();
                assert!(encoded.len() < 2_048);
                assert!(!encoded.contains("retained-source-canary"));
                assert!(!encoded.contains("private-request-sentinel"));
            }
            assert_eq!(output.last().unwrap()["event"], "terminal");
            assert_eq!(
                output.last().unwrap()["documents"][0]["content"],
                "🦀retained-source-canary"
            );
            assert_eq!(provider.calls.load(Ordering::Relaxed), before);

            let mut bounded = open_session(&cx, selected.clone(), &provider, false)
                .with_filter(&cx, None, 16)
                .unwrap();
            let mut small_window = valid;
            small_window["documents"][0]["max_bytes"] = serde_json::json!(4);
            let missing = fetch_message(
                &selected,
                "missing",
                &serde_json::json!([{"doc_id":"absent"}]),
            );
            let output = exchange(&cx, &mut bounded, &[stale, small_window, missing]).await;
            assert_eq!(
                output[1]["code"], "stale_expected_authority",
                "authority fencing must precede the source-budget lookup"
            );
            assert_eq!(
                output[2]["code"], "fetch_refused",
                "a four-byte output window still requires admission of the complete stored source"
            );
            assert!(output[2].get("documents").is_none());
            assert_eq!(output[3]["event"], "terminal");
            assert_eq!(
                output[3]["documents"][0],
                serde_json::json!({"status":"unavailable", "doc_id":"absent"})
            );
            assert_eq!(provider.calls.load(Ordering::Relaxed), before);
            assert_eq!(bounded.receipt.authority, selected.authority);
        });
    }

    #[test]
    fn authority_fetch_bounds_encoded_output_and_stops_on_delivery_failure() {
        struct RejectFetch {
            writes: usize,
            delivered: Vec<u8>,
        }
        impl Write for RejectFetch {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.writes += 1;
                if self.writes > 1 {
                    return Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed"));
                }
                self.delivered.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let parent = private_parent("fetch-output-bound");
            let provider = provider();
            // Each full window is at most 64 KiB raw, but JSON escaping expands
            // its NULs sixfold. Three copies exceed the encoded 1 MiB cap.
            let content = format!("retry{}", "\u{0}".repeat(65_532));
            let selected = publish_source_fixture(
                &cx,
                &parent,
                &provider,
                EmbedderTierTag::Fast,
                true,
                vec![
                    IndexableDocument::new("escaped", content)
                        .with_title("PRIVATE_OPTIONAL_TITLE")
                        .with_metadata("note", "PRIVATE_OPTIONAL_METADATA"),
                ],
            )
            .await;
            let mut session = open_session(&cx, selected.clone(), &provider, false)
                .with_filter(&cx, None, 100_000)
                .unwrap();
            let single = fetch_message(
                &selected,
                "single",
                &serde_json::json!([
                    {"doc_id":"escaped", "max_bytes":65_536, "include_metadata":true}
                ]),
            );
            let overflow = fetch_message(
                &selected,
                "overflow",
                &serde_json::json!(vec![
                    serde_json::json!({"doc_id":"escaped", "max_bytes":65_536, "include_metadata":true});
                    3
                ]),
            );
            let small = fetch_message(
                &selected,
                "small",
                &serde_json::json!([
                    {"doc_id":"escaped", "max_bytes":4}
                ]),
            );
            let before = provider.calls.load(Ordering::Relaxed);
            let output = exchange(&cx, &mut session, &[single, overflow, small.clone()]).await;
            assert_eq!(output[1]["event"], "terminal");
            assert_eq!(
                output[1]["documents"][0]["content"].as_str().unwrap().len(),
                65_536
            );
            assert_eq!(
                output[1]["documents"][0]["metadata"]["note"],
                "PRIVATE_OPTIONAL_METADATA"
            );
            assert_eq!(output[2]["event"], "failed");
            assert_eq!(output[2]["code"], "fetch_refused");
            assert_eq!(output[2]["partial_results"], false);
            assert!(output[2].get("documents").is_none());
            let failure = serde_json::to_string(&output[2]).unwrap();
            assert!(failure.len() < 2_048);
            assert!(!failure.contains("PRIVATE_OPTIONAL"));
            assert_eq!(output[3]["event"], "terminal");
            assert_eq!(output[3]["documents"][0]["content"], "retr");
            assert!(output[3]["documents"][0].get("metadata").is_none());
            assert_eq!(provider.calls.load(Ordering::Relaxed), before);

            let mut input = serde_json::to_vec(&small).unwrap();
            input.extend_from_slice(b"\n{\"query\":\"retry\"}\n");
            let mut output = RejectFetch {
                writes: 0,
                delivered: Vec::new(),
            };
            let error = serve_session(&cx, &mut session, &mut Cursor::new(input), &mut output)
                .await
                .unwrap_err();
            assert!(publication_failure(error.as_ref()).is_none());
            assert_eq!(output.writes, 2);
            assert_eq!(
                frames(&output.delivered).len(),
                1,
                "only readiness was delivered; a broken source reply does not emit another frame"
            );
            assert_eq!(
                provider.calls.load(Ordering::Relaxed),
                before,
                "the next query cannot execute after a fetch delivery failure"
            );
            assert_eq!(session.receipt.authority, selected.authority);
        });
    }

    fn fetch_options(receipt: &Path, doc_id: &str) -> Options {
        parse(&[
            "authority-fetch",
            "--receipt",
            receipt.to_str().unwrap(),
            "--doc-id",
            doc_id,
        ])
        .unwrap()
        .unwrap()
    }

    #[test]
    fn one_shot_fetch_reads_selected_utf8_source_with_independent_payload_and_window_limits() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (label, tier, ann) in [
                ("one-shot-fetch-fast-exact", EmbedderTierTag::Fast, false),
                (
                    "one-shot-fetch-quality-exact",
                    EmbedderTierTag::Quality,
                    false,
                ),
                ("one-shot-fetch-fast-hnsw", EmbedderTierTag::Fast, true),
                (
                    "one-shot-fetch-quality-hnsw",
                    EmbedderTierTag::Quality,
                    true,
                ),
            ] {
                let parent = private_parent(label);
                let provider = provider();
                let doc_id = "notes/../🦀";
                let content = "retry 🙂éZ\noriginal";
                let requested = IndexableDocument::new(doc_id, content)
                    .with_title("Source \"title\"")
                    .with_metadata("tenant", "blue")
                    .with_metadata("escaped", "\"\n\\\t");
                let source_bytes = requested.id.len()
                    + requested.content.len()
                    + requested.title.as_deref().map_or(0, str::len)
                    + serde_json::to_vec(&requested.metadata).unwrap().len();
                let selected = publish_source_fixture(
                    &cx,
                    &parent,
                    &provider,
                    tier,
                    ann,
                    vec![
                        requested,
                        IndexableDocument::new("unrequested", "garden ".repeat(2048)),
                    ],
                )
                .await;
                let authority_before = fs::read(selected.root.join("AUTHORITY")).unwrap();
                let lock_before = fs::read(selected.root.join("LOCK")).unwrap();
                let receipt_path = parent.join("source.json");
                let receipt_before = fs::read(&receipt_path).unwrap();
                let before = provider.calls.load(Ordering::Relaxed);
                let mut options = fetch_options(&receipt_path, doc_id);
                // Only this complete source fits. Exporting the unrelated
                // large document would fail even though the output is tiny.
                options.max_source_bytes = source_bytes;
                options.offset_bytes = 6;
                options.max_bytes = 5;
                let mut output = Vec::new();
                fetch(&cx, &options, &mut output).unwrap();
                assert!(output.len() <= source::MAX_RESPONSE_BYTES);
                let responses = frames(&output);
                assert_eq!(responses.len(), 1);
                let response = &responses[0];
                assert_eq!(response["event"], "terminal");
                assert_eq!(response["operation"], "fetch");
                assert_eq!(response["status"], "complete");
                assert_eq!(response["ok"], true);
                assert_eq!(response["partial_results"], false);
                assert_eq!(response["authority"], serde_json::json!(selected.authority));
                assert_eq!(
                    response["generation"],
                    serde_json::json!(selected.generation)
                );
                assert_eq!(response["source"], "retained_snapshot");
                let document = &response["documents"][0];
                assert_eq!(document["doc_id"], doc_id);
                assert_eq!(document["status"], "found");
                assert_eq!(document["content"], "🙂");
                assert_eq!(document["offset_bytes"], 6);
                assert_eq!(document["end_offset_bytes"], 10);
                assert_eq!(document["next_offset_bytes"], 10);
                assert_eq!(document["total_bytes"], content.len());
                assert!(document.get("title").is_none());
                assert!(document.get("metadata").is_none());

                options.offset_bytes = 10;
                options.max_bytes = 64;
                options.include_metadata = true;
                output.clear();
                fetch(&cx, &options, &mut output).unwrap();
                let response: serde_json::Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(response["documents"][0]["content"], "éZ\noriginal");
                assert!(response["documents"][0]["next_offset_bytes"].is_null());
                assert_eq!(response["documents"][0]["title"], "Source \"title\"");
                assert_eq!(response["documents"][0]["metadata"]["escaped"], "\"\n\\\t");

                options.filter = Some(Filter::parse(r#"{"metadata":{"tenant":"red"}}"#).unwrap());
                output.clear();
                fetch(&cx, &options, &mut output).unwrap();
                let response: serde_json::Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(
                    response["documents"][0],
                    serde_json::json!({"status":"unavailable", "doc_id":doc_id})
                );
                options.doc_id = Some("absent".to_owned());
                options.max_source_bytes = 1;
                output.clear();
                fetch(&cx, &options, &mut output).unwrap();
                let response: serde_json::Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(
                    response["documents"][0],
                    serde_json::json!({"status":"unavailable", "doc_id":"absent"})
                );

                options.doc_id = Some(doc_id.to_owned());
                options.filter = None;
                options.offset_bytes = 0;
                options.max_bytes = 4;
                options.include_metadata = false;
                options.max_source_bytes = source_bytes - 1;
                output.clear();
                assert!(
                    fetch(&cx, &options, &mut output).is_err(),
                    "a small output window still requires the complete selected stored payload"
                );
                assert!(output.is_empty());
                assert_eq!(provider.calls.load(Ordering::Relaxed), before);
                assert_eq!(
                    fs::read(selected.root.join("AUTHORITY")).unwrap(),
                    authority_before
                );
                assert_eq!(fs::read(selected.root.join("LOCK")).unwrap(), lock_before);
                assert_eq!(fs::read(receipt_path).unwrap(), receipt_before);
            }
        });
    }

    #[test]
    fn one_shot_fetch_refuses_stale_cancelled_and_invalid_windows_without_partial_output() {
        struct PartialOutput {
            writes: usize,
            prefix: Vec<u8>,
        }
        impl Write for PartialOutput {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.writes += 1;
                match self.writes {
                    1 => {
                        let count = bytes.len().min(7);
                        self.prefix.extend_from_slice(&bytes[..count]);
                        Ok(count)
                    }
                    2 => Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "interrupted fetch",
                    )),
                    _ => panic!("source delivery failure cannot emit another frame"),
                }
            }
            fn flush(&mut self) -> io::Result<()> {
                panic!("an incomplete source frame cannot reach flush")
            }
        }
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let parent = private_parent("one-shot-fetch-refusal");
            let provider = provider();
            let selected = publish_source_fixture(
                &cx,
                &parent,
                &provider,
                EmbedderTierTag::Fast,
                true,
                vec![IndexableDocument::new("unicode", "🦀retained source")],
            )
            .await;
            let authority_before = fs::read(selected.root.join("AUTHORITY")).unwrap();
            let lock_before = fs::read(selected.root.join("LOCK")).unwrap();
            let before = provider.calls.load(Ordering::Relaxed);
            let mut options = fetch_options(&parent.join("source.json"), "unicode");
            let mut output = Vec::new();
            cx.set_cancel_requested(true);
            assert!(fetch(&cx, &options, &mut output).is_err());
            cx.set_cancel_requested(false);
            assert!(output.is_empty());
            for offset in [1, usize::MAX] {
                options.offset_bytes = offset;
                assert!(fetch(&cx, &options, &mut output).is_err());
                assert!(output.is_empty());
            }
            options.offset_bytes = 0;
            let mut partial = PartialOutput {
                writes: 0,
                prefix: Vec::new(),
            };
            let error = fetch(&cx, &options, &mut partial).unwrap_err();
            assert_eq!(
                error.downcast_ref::<io::Error>().unwrap().kind(),
                io::ErrorKind::BrokenPipe
            );
            assert!(publication_failure(error.as_ref()).is_none());
            assert_eq!(partial.writes, 2);
            assert_eq!(partial.prefix.len(), 7);
            assert_eq!(provider.calls.load(Ordering::Relaxed), before);
            assert_eq!(
                fs::read(selected.root.join("AUTHORITY")).unwrap(),
                authority_before
            );
            assert_eq!(fs::read(selected.root.join("LOCK")).unwrap(), lock_before);

            let retained = selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            let successor = publish(
                &cx,
                &parent,
                Some(&retained),
                &provider,
                EmbedderTierTag::Fast,
                "new",
                false,
            )
            .await;
            let before = provider.calls.load(Ordering::Relaxed);
            let authority_after = fs::read(selected.root.join("AUTHORITY")).unwrap();
            let lock_after = fs::read(selected.root.join("LOCK")).unwrap();
            assert!(
                fetch(&cx, &options, &mut output).is_err(),
                "an old receipt cannot follow a newer authority"
            );
            assert!(output.is_empty());
            options.receipt = parent.join("new.json");
            options.doc_id = Some("retry-new".to_owned());
            fetch(&cx, &options, &mut output).unwrap();
            let response: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(
                response["authority"],
                serde_json::json!(successor.authority)
            );
            assert_eq!(
                response["documents"][0]["content"],
                "retry network requests"
            );
            assert_eq!(provider.calls.load(Ordering::Relaxed), before);
            assert_eq!(
                fs::read(selected.root.join("AUTHORITY")).unwrap(),
                authority_after
            );
            assert_eq!(fs::read(selected.root.join("LOCK")).unwrap(), lock_after);
        });
    }

    fn component_path(
        root: &Path,
        snapshot: &NativeGenerationSnapshot,
        role: GenerationComponentRole,
    ) -> PathBuf {
        let receipt = snapshot
            .retained_snapshot()
            .manifest()
            .components()
            .get(role)
            .unwrap();
        root.join(component_object_name_v1(role, receipt.sha256))
    }

    fn search_options(receipt: &Path, filter: Option<&str>) -> Options {
        let mut options = parse(&[
            "authority-search",
            "--receipt",
            "selected.json",
            "--query",
            "retry",
            "--limit",
            "1",
        ])
        .unwrap()
        .unwrap();
        options.receipt = receipt.to_path_buf();
        options.filter = filter
            .map(crate::filter::Filter::parse)
            .transpose()
            .unwrap();
        options
    }

    #[test]
    fn authority_lexical_search_uses_no_model_and_filters_before_the_cutoff() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (label, tier, ann) in [
                ("lexical-fast-exact", EmbedderTierTag::Fast, false),
                ("lexical-quality-exact", EmbedderTierTag::Quality, false),
                ("lexical-fast-hnsw", EmbedderTierTag::Fast, true),
                ("lexical-quality-hnsw", EmbedderTierTag::Quality, true),
            ] {
                let parent = private_parent(label);
                let provider = provider();
                let selected = publish_source_fixture(
                    &cx,
                    &parent,
                    &provider,
                    tier,
                    ann,
                    vec![
                        IndexableDocument::new("blocked", "retry retry retry retry")
                            .with_metadata("tenant", "red"),
                        IndexableDocument::new("allowed", "retry network requests with a backoff")
                            .with_title("Retry guide")
                            .with_metadata("tenant", "blue"),
                        IndexableDocument::new("garden", "grow flowers")
                            .with_metadata("tenant", "blue"),
                    ],
                )
                .await;
                let receipt_path = parent.join("source.json");
                let authority_before = fs::read(selected.root.join("AUTHORITY")).unwrap();
                let lock_before = fs::read(selected.root.join("LOCK")).unwrap();
                let receipt_before = fs::read(&receipt_path).unwrap();
                let calls = provider.calls.load(Ordering::Relaxed);
                let mut options =
                    search_options(&receipt_path, Some(r#"{"metadata":{"tenant":"blue"}}"#));
                options.lexical_only = true;
                let mut output = Vec::new();
                search_with_loader(&cx, &options, &mut output, |_| {
                    panic!("explicit lexical search must never load a model")
                })
                .await
                .unwrap();
                let response: serde_json::Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(response["mode"], "lexical");
                assert_eq!(response["retrieval"], "lexical");
                assert_eq!(response["semantic_used"], false);
                assert_eq!(response["semantic_components_verified"], true);
                assert_eq!(response["authority"], serde_json::json!(selected.authority));
                assert_eq!(response["scope"]["eligible_documents"], 2);
                let hits = response["results"].as_array().unwrap();
                assert_eq!(hits.len(), 1);
                assert_eq!(hits[0]["doc_id"], "allowed");
                assert_eq!(hits[0]["source"], "Lexical");
                assert_eq!(hits[0]["score"], hits[0]["lexical_score"]);
                assert!(hits[0]["score"].as_f64().unwrap().is_finite());
                assert_eq!(hits[0]["metadata"]["tenant"], "blue");
                for field in ["fast_score", "quality_score", "rerank_score", "index"] {
                    assert!(hits[0][field].is_null(), "{field}");
                }

                // The same scope cannot create a semantic fallback on failure.
                options.lexical_only = false;
                output.clear();
                assert!(
                    search_with_loader(&cx, &options, &mut output, |_| {
                        Err(bad("model unavailable"))
                    })
                    .await
                    .is_err()
                );
                assert!(output.is_empty());
                options.lexical_only = true;
                options.max_source_bytes = 1;
                assert!(
                    search_with_loader(&cx, &options, &mut output, |_| {
                        panic!("source admission failure cannot load a model")
                    })
                    .await
                    .is_err()
                );
                assert!(output.is_empty());
                options.max_source_bytes = MAX_INPUT_BYTES;
                cx.set_cancel_requested(true);
                assert!(
                    search_with_loader(&cx, &options, &mut output, |_| {
                        panic!("cancelled lexical search cannot load a model")
                    })
                    .await
                    .is_err()
                );
                cx.set_cancel_requested(false);
                assert!(output.is_empty());
                assert_eq!(provider.calls.load(Ordering::Relaxed), calls);
                assert_eq!(
                    fs::read(selected.root.join("AUTHORITY")).unwrap(),
                    authority_before
                );
                assert_eq!(fs::read(selected.root.join("LOCK")).unwrap(), lock_before);
                assert_eq!(fs::read(receipt_path).unwrap(), receipt_before);

                let semantic_mode = match tier {
                    EmbedderTierTag::Fast => "fast",
                    EmbedderTierTag::Quality => "quality",
                };
                let mut hybrid = open_session(&cx, selected.clone(), &provider, false);
                let responses = exchange(&cx, &mut hybrid, &[
                    serde_json::json!({"query":"retry", "mode":"lexical", "filter":{"ids":["allowed"]}}),
                    serde_json::json!({"query":"retry", "mode":semantic_mode}),
                ]).await;
                let modes = responses[0]["query_modes"].as_array().unwrap();
                assert_eq!(modes.len(), 2);
                assert!(modes.contains(&serde_json::json!(semantic_mode)));
                assert!(modes.contains(&serde_json::json!("lexical")));
                assert_eq!(responses[1]["results"][0]["doc_id"], "allowed");
                assert_eq!(responses[1]["retrieval"], "lexical");
                assert_eq!(responses[1]["semantic_used"], false);
                assert_eq!(responses[2]["mode"], semantic_mode);
                assert_eq!(provider.calls.load(Ordering::Relaxed), calls + 1);
            }
        });
    }

    #[test]
    fn lexical_authority_sessions_retain_scope_and_activate_without_semantic_privileges() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let parent = private_parent("lexical-activation");
            let old_provider = provider();
            let selected = publish(
                &cx,
                &parent,
                None,
                &old_provider,
                EmbedderTierTag::Quality,
                "old",
                true,
            )
            .await;
            let root = admit_root(&selected.root).unwrap();
            let snapshot = Arc::new(
                selected
                    .open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES)
                    .unwrap(),
            );
            let old_snapshot = Arc::clone(&snapshot);
            let mut session = AuthoritySession::new(
                &cx,
                root,
                selected.clone(),
                snapshot,
                AuthorityModels::Lexical,
                (10, DEFAULT_IMAGE_BYTES),
                true,
            )
            .unwrap()
            .with_filter(
                &cx,
                Some(Filter::parse(r#"{"id_prefix":"retry-"}"#).unwrap()),
                MAX_INPUT_BYTES,
            )
            .unwrap();
            let mut next_identity = provider_identity();
            next_identity.producer.implementation_revision =
                "next-keyword-only-producer".to_owned();
            let new_provider = Arc::new(Provider {
                identity: next_identity,
                calls: AtomicUsize::new(0),
            });
            let next = publish(
                &cx,
                &parent,
                Some(&old_snapshot),
                &new_provider,
                EmbedderTierTag::Quality,
                "new",
                false,
            )
            .await;
            let old_calls = old_provider.calls.load(Ordering::Relaxed);
            let new_calls = new_provider.calls.load(Ordering::Relaxed);
            let authority_after = fs::read(selected.root.join("AUTHORITY")).unwrap();
            let lock_after = fs::read(selected.root.join("LOCK")).unwrap();
            let responses = exchange(
                &cx,
                &mut session,
                &[
                    serde_json::json!({"query":"retry"}),
                    serde_json::json!({"query":"retry", "mode":"quality"}),
                    serde_json::json!({"query":"garden", "filter":{"ids":["garden-old"]}}),
                    serde_json::json!({"op":"activate", "receipt":parent.join("new.json"),
                    "expected_authority":selected.authority}),
                    serde_json::json!({"query":"retry"}),
                    serde_json::json!({"op":"fetch", "expected_authority":next.authority,
                    "documents":[{"doc_id":"retry-new"}]}),
                ],
            )
            .await;
            assert_eq!(responses[0]["query_modes"], serde_json::json!(["lexical"]));
            assert_eq!(responses[0]["updates_enabled"], false);
            assert_eq!(responses[1]["results"][0]["doc_id"], "retry-old");
            assert_eq!(responses[2]["code"], "semantic_search_disabled");
            assert!(responses[3]["results"].as_array().unwrap().is_empty());
            assert_eq!(responses[3]["scope"]["eligible_documents"], 0);
            assert_eq!(responses[4]["event"], "activated");
            assert_eq!(responses[5]["results"][0]["doc_id"], "retry-new");
            assert_eq!(responses[5]["authority"], serde_json::json!(next.authority));
            assert_eq!(
                responses[6]["documents"][0]["content"],
                "retry network requests"
            );
            assert_eq!(
                old_snapshot
                    .search_lexical_text(&cx, "retry", 10)
                    .await
                    .unwrap()[0]
                    .doc_id,
                "retry-old"
            );
            assert_eq!(old_provider.calls.load(Ordering::Relaxed), old_calls);
            assert_eq!(new_provider.calls.load(Ordering::Relaxed), new_calls);
            assert_eq!(
                fs::read(selected.root.join("AUTHORITY")).unwrap(),
                authority_after
            );
            assert_eq!(fs::read(selected.root.join("LOCK")).unwrap(), lock_after);
            assert!(session.with_updates(2, MAX_INPUT_BYTES).is_err());
        });
    }

    #[test]
    fn filtered_search_and_sessions_keep_the_owner_scope_on_retained_images() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (label, tier, ann) in [
                ("filtered-fast-exact", EmbedderTierTag::Fast, false),
                ("filtered-quality-exact", EmbedderTierTag::Quality, false),
                ("filtered-fast-hnsw", EmbedderTierTag::Fast, true),
                ("filtered-quality-hnsw", EmbedderTierTag::Quality, true),
            ] {
                let parent = private_parent(label);
                let provider = provider();
                let selected = publish(&cx, &parent, None, &provider, tier, "old", ann).await;
                let receipt_path = parent.join("old.json");
                let options = search_options(&receipt_path, Some(r#"{"id_prefix":"garden-"}"#));
                let before = provider.calls.load(Ordering::Relaxed);
                let model: Arc<dyn Embedder> = provider.clone();
                let mut output = Vec::new();
                search_with_loader(&cx, &options, &mut output, |requested| {
                    assert_eq!(requested, tier);
                    Ok(Arc::clone(&model))
                })
                .await
                .unwrap();
                let filtered: serde_json::Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(filtered["event"], "results");
                assert_eq!(filtered["scope"]["filtered"], true);
                assert_eq!(filtered["scope"]["eligible_documents"], 1);
                assert_eq!(filtered["results"].as_array().unwrap().len(), 1);
                assert_eq!(filtered["results"][0]["doc_id"], "garden-old");
                assert_eq!(provider.calls.load(Ordering::Relaxed), before + 1);

                let options = search_options(&receipt_path, None);
                let model: Arc<dyn Embedder> = provider.clone();
                output.clear();
                search_with_loader(&cx, &options, &mut output, |_| Ok(Arc::clone(&model)))
                    .await
                    .unwrap();
                let unrestricted: serde_json::Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(unrestricted["results"][0]["doc_id"], "retry-old");
                assert!(unrestricted.get("scope").is_none());
                assert_eq!(provider.calls.load(Ordering::Relaxed), before + 2);

                let options = search_options(&receipt_path, Some(r#"{"ids":[]}"#));
                let model: Arc<dyn Embedder> = provider.clone();
                output.clear();
                search_with_loader(&cx, &options, &mut output, |_| Ok(Arc::clone(&model)))
                    .await
                    .unwrap();
                let empty: serde_json::Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(empty["scope"]["eligible_documents"], 0);
                assert!(empty["results"].as_array().unwrap().is_empty());
                assert_eq!(
                    provider.calls.load(Ordering::Relaxed),
                    before + 2,
                    "an admitted empty scope does not submit a query to the provider"
                );

                let fixed = crate::filter::Filter::parse(
                    r#"{"id_prefix":"garden-","metadata":{"version":"old"}}"#,
                )
                .unwrap();
                let mut session = open_session(&cx, selected.clone(), &provider, false)
                    .with_filter(&cx, Some(fixed), crate::MAX_INPUT_BYTES)
                    .unwrap();
                let before_queries = provider.calls.load(Ordering::Relaxed);
                let output = exchange(
                    &cx,
                    &mut session,
                    &[
                        serde_json::json!({"id": "fixed", "query": "retry"}),
                        serde_json::json!({"id": "null", "query": "retry", "filter": null}),
                        serde_json::json!({"id": "broad", "query": "retry", "filter": {}}),
                        serde_json::json!({
                            "id": "excluded", "query": "retry",
                            "filter": {"ids": ["retry-old"]},
                        }),
                        serde_json::json!({
                            "id": "narrow", "query": "retry",
                            "filter": {"ids": ["garden-old"], "metadata": {"version": "old"}},
                        }),
                        serde_json::json!({
                            "id": "wrong-metadata", "query": "retry",
                            "filter": {"metadata": {"version": "new"}},
                        }),
                    ],
                )
                .await;
                for position in [1, 2, 3, 5] {
                    assert_eq!(output[position]["event"], "results");
                    assert_eq!(output[position]["scope"]["filtered"], true);
                    assert_eq!(output[position]["scope"]["eligible_documents"], 1);
                    assert_eq!(output[position]["results"][0]["doc_id"], "garden-old");
                    assert_eq!(
                        output[position]["authority"],
                        serde_json::json!(selected.authority)
                    );
                }
                for position in [4, 6] {
                    assert_eq!(output[position]["event"], "results");
                    assert_eq!(output[position]["scope"]["eligible_documents"], 0);
                    assert!(output[position]["results"].as_array().unwrap().is_empty());
                }
                assert_eq!(provider.calls.load(Ordering::Relaxed), before_queries + 4);
                assert_eq!(session.receipt.authority, selected.authority);

                fs::rename(parent.join("old-work"), parent.join("archived-work")).unwrap();
                fs::rename(&selected.root, parent.join("relocated-root")).unwrap();
                let output = exchange(
                    &cx,
                    &mut session,
                    &[serde_json::json!({
                        "query": "retry", "filter": {"ids": ["garden-old"]},
                    })],
                )
                .await;
                assert_eq!(output[1]["results"][0]["doc_id"], "garden-old");
                assert_eq!(output[1]["scope"]["eligible_documents"], 1);
                assert_eq!(provider.calls.load(Ordering::Relaxed), before_queries + 5);
            }
        });
    }

    #[test]
    fn authority_filter_source_budgets_are_lazy_and_fail_before_model_or_inference() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let parent = private_parent("filter-source-budget");
            let provider = provider();
            let selected = publish(
                &cx,
                &parent,
                None,
                &provider,
                EmbedderTierTag::Fast,
                "old",
                true,
            )
            .await;
            let receipt_path = parent.join("old.json");
            let before = provider.calls.load(Ordering::Relaxed);
            let mut options = search_options(&receipt_path, Some("{}"));
            options.max_source_bytes = 1;
            let loads = AtomicUsize::new(0);
            let mut output = Vec::new();
            let error = search_with_loader(&cx, &options, &mut output, |_| {
                loads.fetch_add(1, Ordering::Relaxed);
                Err(bad("source admission must precede local model loading"))
            })
            .await
            .unwrap_err();
            assert!(error.to_string().contains("stored-payload-limit"));
            assert_eq!(loads.load(Ordering::Relaxed), 0);
            assert_eq!(provider.calls.load(Ordering::Relaxed), before);
            assert!(output.is_empty());
            assert!(
                open_session(&cx, selected.clone(), &provider, false)
                    .with_filter(&cx, Some(crate::filter::Filter::parse("{}").unwrap()), 1)
                    .is_err()
            );
            assert_eq!(provider.calls.load(Ordering::Relaxed), before);

            options.filter = None;
            let model: Arc<dyn Embedder> = provider.clone();
            search_with_loader(&cx, &options, &mut output, |_| Ok(Arc::clone(&model)))
                .await
                .unwrap();
            let page: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(page["results"][0]["doc_id"], "retry-old");
            assert!(page.get("scope").is_none());
            let mut session = open_session(&cx, selected.clone(), &provider, false)
                .with_filter(&cx, None, 1)
                .unwrap();
            let before_queries = provider.calls.load(Ordering::Relaxed);
            let output = exchange(
                &cx,
                &mut session,
                &[
                    serde_json::json!({"id": "ordinary", "query": "retry"}),
                    serde_json::json!({"id": "bounded", "query": "retry", "filter": {}}),
                    serde_json::json!({"id": "still-ordinary", "query": "retry", "filter": null}),
                ],
            )
            .await;
            assert_eq!(output[1]["event"], "results");
            assert_eq!(output[2]["event"], "failed");
            assert_eq!(output[3]["event"], "results");
            assert_eq!(output[3]["results"][0]["doc_id"], "retry-old");
            assert!(output[1].get("scope").is_none());
            assert!(output[3].get("scope").is_none());
            assert_eq!(provider.calls.load(Ordering::Relaxed), before_queries + 2);
            assert_eq!(session.receipt.authority, selected.authority);

            // Empty membership still requires the selected query producer.
            let wrong = Arc::new(Provider {
                identity: {
                    let mut identity = provider_identity();
                    identity.producer.implementation_revision = "wrong-filter-producer".to_owned();
                    identity
                },
                calls: AtomicUsize::new(0),
            });
            let options = search_options(&receipt_path, Some(r#"{"ids":[]}"#));
            let model: Arc<dyn Embedder> = wrong.clone();
            assert!(
                search_with_loader(&cx, &options, &mut Vec::new(), |_| Ok(Arc::clone(&model)))
                    .await
                    .is_err()
            );
            assert_eq!(wrong.calls.load(Ordering::Relaxed), 0);
        });
    }

    #[test]
    fn bad_filter_requests_and_query_failures_leave_the_next_scoped_query_usable() {
        struct QueryFailureProvider {
            inner: Arc<Provider>,
        }
        impl Embedder for QueryFailureProvider {
            fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
                self.inner.identity()
            }
            fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
                if text == "private-inference-failure" {
                    self.inner.calls.fetch_add(1, Ordering::Relaxed);
                    return Box::pin(async {
                        Err(SearchError::SubsystemError {
                            subsystem: "authority_filter_test",
                            source: "private-provider-error-must-not-escape".into(),
                        })
                    });
                }
                self.inner.embed(cx, text)
            }
            fn dimension(&self) -> usize {
                self.inner.dimension()
            }
            fn id(&self) -> &str {
                self.inner.id()
            }
            fn model_name(&self) -> &str {
                self.inner.model_name()
            }
            fn is_semantic(&self) -> bool {
                true
            }
            fn category(&self) -> ModelCategory {
                self.inner.category()
            }
        }
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let parent = private_parent("bad-filters");
            let provider = provider();
            let selected = publish(
                &cx,
                &parent,
                None,
                &provider,
                EmbedderTierTag::Quality,
                "old",
                true,
            )
            .await;
            let root = admit_root(&selected.root).unwrap();
            let snapshot = selected
                .open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES)
                .unwrap();
            let model: Arc<dyn Embedder> = Arc::new(QueryFailureProvider {
                inner: provider.clone(),
            });
            let mut session = AuthoritySession::new(
                &cx,
                root,
                selected.clone(),
                snapshot,
                Models {
                    fast: model,
                    quality: None,
                },
                (1, DEFAULT_IMAGE_BYTES),
                false,
            )
            .unwrap()
            .with_filter(
                &cx,
                Some(crate::filter::Filter::parse(r#"{"metadata":{"version":"old"}}"#).unwrap()),
                crate::MAX_INPUT_BYTES,
            )
            .unwrap();
            let too_many_ids = (0..4097)
                .map(|index| format!("doc-{index}"))
                .collect::<Vec<_>>();
            let too_many_metadata = (0..65)
                .map(|index| (format!("field-{index}"), "value"))
                .collect::<std::collections::BTreeMap<_, _>>();
            let mut messages = vec![
                serde_json::json!({"query": "retry", "filter": {"ids": too_many_ids}}),
                serde_json::json!({"query": "retry", "filter": {"metadata": too_many_metadata}}),
                serde_json::json!({"query": "retry", "filter": {"id_prefix": "x".repeat(65536)}}),
                serde_json::json!({"query": "retry", "filter": {"ids": [" "]}}),
                serde_json::json!({"query": "retry", "filter": {"ids": ["private\0id"]}}),
                serde_json::json!({
                    "query": "retry", "filter": {"unknown": "private-filter-sentinel"},
                }),
            ];
            let invalid_count = messages.len();
            messages.extend([
                serde_json::json!({"id": "inference", "query": "private-inference-failure"}),
                serde_json::json!({
                    "id": "empty", "query": "private-inference-failure", "filter": {"ids": []},
                }),
                serde_json::json!({
                    "id": "recovered", "query": "retry", "filter": {"ids": ["garden-old"]},
                }),
            ]);
            let before = provider.calls.load(Ordering::Relaxed);
            let output = exchange(&cx, &mut session, &messages).await;
            for frame in &output[1..=invalid_count + 1] {
                assert_eq!(frame["event"], "failed");
                assert_eq!(frame["authority"], serde_json::json!(selected.authority));
                assert!(serde_json::to_vec(frame).unwrap().len() < 2048);
            }
            let empty = &output[invalid_count + 2];
            assert_eq!(empty["event"], "results");
            assert_eq!(empty["scope"]["eligible_documents"], 0);
            assert!(empty["results"].as_array().unwrap().is_empty());
            let recovered = &output[invalid_count + 3];
            assert_eq!(recovered["event"], "results");
            assert_eq!(recovered["id"], "recovered");
            assert_eq!(recovered["scope"]["eligible_documents"], 1);
            assert_eq!(recovered["results"][0]["doc_id"], "garden-old");
            assert_eq!(provider.calls.load(Ordering::Relaxed), before + 2);
            assert_eq!(session.receipt.authority, selected.authority);
            let encoded = serde_json::to_string(&output).unwrap();
            assert!(!encoded.contains("private-provider-error-must-not-escape"));
            assert!(!encoded.contains("private-filter-sentinel"));
        });
    }

    #[test]
    fn filtered_activation_refreshes_membership_and_budget_refusal_keeps_the_prior_scope() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (label, tier) in [
                ("filtered-activation-fast", EmbedderTierTag::Fast),
                ("filtered-activation-quality", EmbedderTierTag::Quality),
            ] {
                let parent = private_parent(label);
                let provider = provider();
                let old = publish(&cx, &parent, None, &provider, tier, "old", true).await;
                let fixed =
                    || crate::filter::Filter::parse(r#"{"metadata":{"version":"new"}}"#).unwrap();
                let mut session = open_session(&cx, old.clone(), &provider, true)
                    .with_filter(&cx, Some(fixed()), 256)
                    .unwrap();
                let mut retained = open_session(&cx, old.clone(), &provider, false)
                    .with_filter(&cx, Some(fixed()), 256)
                    .unwrap();
                let next = publish(
                    &cx,
                    &parent,
                    Some(&session.snapshot),
                    &provider,
                    tier,
                    "new",
                    false,
                )
                .await;
                let before = provider.calls.load(Ordering::Relaxed);
                let output = exchange(
                    &cx,
                    &mut session,
                    &[
                        serde_json::json!({"query": "retry"}),
                        serde_json::json!({
                            "op": "activate", "receipt": parent.join("new.json"),
                            "expected_authority": old.authority,
                        }),
                        serde_json::json!({"query": "retry"}),
                    ],
                )
                .await;
                assert_eq!(output[1]["scope"]["eligible_documents"], 0);
                assert!(output[1]["results"].as_array().unwrap().is_empty());
                assert_eq!(output[2]["event"], "activated");
                assert_eq!(output[2]["authority"], serde_json::json!(next.authority));
                assert_eq!(output[3]["scope"]["eligible_documents"], 2);
                assert_eq!(output[3]["results"][0]["doc_id"], "retry-new");
                assert_eq!(provider.calls.load(Ordering::Relaxed), before + 1);
                let held = exchange(
                    &cx,
                    &mut retained,
                    &[serde_json::json!({"query": "retry", "filter": {}})],
                )
                .await;
                assert_eq!(held[1]["authority"], serde_json::json!(old.authority));
                assert_eq!(held[1]["scope"]["eligible_documents"], 0);
                assert!(held[1]["results"].as_array().unwrap().is_empty());
                assert_eq!(provider.calls.load(Ordering::Relaxed), before + 1);

                // A valid successor can exceed the server's retained-source
                // budget even when its fixed predicate would reject every row.
                // That refusal must leave BOTH the reader and frozen scope intact.
                let large_label = "oversized".repeat(20);
                let oversized = publish(
                    &cx,
                    &parent,
                    Some(&session.snapshot),
                    &provider,
                    tier,
                    &large_label,
                    true,
                )
                .await;
                let authority_after = fs::read(old.root.join("AUTHORITY")).unwrap();
                let before_refusal = provider.calls.load(Ordering::Relaxed);
                let output = exchange(
                    &cx,
                    &mut session,
                    &[
                        serde_json::json!({
                            "op": "activate", "receipt": parent.join(format!("{large_label}.json")),
                            "expected_authority": next.authority,
                        }),
                        serde_json::json!({"query": "retry", "filter": {"ids": []}}),
                        serde_json::json!({"query": "retry", "filter": {}}),
                    ],
                )
                .await;
                assert_eq!(output[1]["event"], "failed");
                assert_eq!(output[1]["authority"], serde_json::json!(next.authority));
                assert_ne!(
                    output[1]["authority"],
                    serde_json::json!(oversized.authority)
                );
                assert_eq!(output[2]["scope"]["eligible_documents"], 0);
                assert!(output[2]["results"].as_array().unwrap().is_empty());
                assert_eq!(output[3]["scope"]["eligible_documents"], 2);
                assert_eq!(output[3]["results"][0]["doc_id"], "retry-new");
                assert_eq!(provider.calls.load(Ordering::Relaxed), before_refusal + 1);
                assert_eq!(session.receipt.authority, next.authority);
                assert_eq!(
                    fs::read(old.root.join("AUTHORITY")).unwrap(),
                    authority_after
                );
            }
        });
    }

    #[test]
    fn rollback_reopens_the_explicit_predecessor_without_models_or_inference() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (label, tier, ann) in [
                ("rollback-fast-exact", EmbedderTierTag::Fast, false),
                ("rollback-quality-exact", EmbedderTierTag::Quality, false),
                ("rollback-fast-hnsw", EmbedderTierTag::Fast, true),
                ("rollback-quality-hnsw", EmbedderTierTag::Quality, true),
            ] {
                let parent = private_parent(label);
                let old_provider = provider();
                let old = publish(&cx, &parent, None, &old_provider, tier, "old", ann).await;
                let old_reader = old.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                let changed_provider = Arc::new(Provider {
                    identity: {
                        let mut identity = provider_identity();
                        identity.producer.implementation_revision =
                            "before-rollback-migration".to_owned();
                        identity
                    },
                    calls: AtomicUsize::new(0),
                });
                let current = publish(
                    &cx,
                    &parent,
                    Some(&old_reader),
                    &changed_provider,
                    tier,
                    "current",
                    !ann,
                )
                .await;
                let current_reader = current.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_ne!(old.identity, current.identity);
                let original_receipt = fs::read(parent.join("old.json")).unwrap();
                let current_receipt = fs::read(parent.join("current.json")).unwrap();
                let old_calls = old_provider.calls.load(Ordering::Relaxed);
                let current_calls = changed_provider.calls.load(Ordering::Relaxed);
                let mut options = parse(&rollback_arguments()).unwrap().unwrap();
                options.from_receipt = Some(parent.join("current.json"));
                options.target_receipt = Some(parent.join("old.json"));
                options.receipt = parent.join("rolled-back.json");
                let mut output = Vec::new();
                rollback(&cx, &options, &mut output).unwrap();
                let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(report["event"], "rolled_back");
                assert_eq!(report["publication"], "committed");
                assert_eq!(report["serving"], "activated");
                assert_eq!(report["current_admission"], "complete_generation");
                assert_eq!(report["retrieval"], if ann { "hnsw" } else { "exact" });
                assert_eq!(report["replaces"], serde_json::json!(current.authority));
                assert_eq!(report["restores"], serde_json::json!(old.authority));
                assert_eq!(old_provider.calls.load(Ordering::Relaxed), old_calls);
                assert_eq!(
                    changed_provider.calls.load(Ordering::Relaxed),
                    current_calls
                );
                assert_eq!(fs::read(parent.join("old.json")).unwrap(), original_receipt);
                assert_eq!(
                    fs::read(parent.join("current.json")).unwrap(),
                    current_receipt
                );

                // Only receipt paths cross the command boundary. No retained
                // predecessor handle or provider is passed into rollback.
                let restored = Receipt::read(&options.receipt).unwrap();
                assert_eq!(
                    report["selection"],
                    serde_json::to_value(&restored).unwrap()
                );
                assert_eq!(
                    restored.authority().unwrap().sequence,
                    current.authority().unwrap().sequence + 1
                );
                assert_eq!(restored.generation, old.generation);
                assert_eq!(restored.identity, old.identity);
                assert_eq!(restored.documents, old.documents);
                assert_eq!(restored.tier, tier);
                assert_eq!(restored.root_id, old.root_id);
                assert_eq!(
                    restored.recovery_attempt().unwrap().expected,
                    current_reader.retained_snapshot().authority_pair()
                );
                let reopened = restored.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                let manifest = reopened.retained_snapshot().manifest();
                assert_eq!(manifest.predecessor(), Some(current.authority().unwrap()));
                assert_eq!(
                    manifest.components(),
                    old_reader.retained_snapshot().manifest().components()
                );
                assert_eq!(
                    reopened.metadata().commit_range,
                    old_reader.metadata().commit_range
                );
                match manifest {
                    frankensearch_core::generation::ActivationManifest::V1(manifest) => {
                        assert_eq!(manifest.action, GenerationAuthorityActionV1::Rollback);
                    }
                    frankensearch_core::generation::ActivationManifest::V2(manifest) => {
                        assert_eq!(manifest.action, GenerationAuthorityActionV1::Rollback);
                    }
                }

                let mut recovery = parse(&["authority-recover", "--receipt", "r.json"])
                    .unwrap()
                    .unwrap();
                recovery.receipt = options.receipt.clone();
                let mut recovered_output = Vec::new();
                recover(&cx, &recovery, &mut recovered_output).unwrap();
                let recovered: serde_json::Value =
                    serde_json::from_slice(&recovered_output).unwrap();
                assert_eq!(recovered["publication"], "committed");
                assert_eq!(recovered["serving"], "activated");
                assert_eq!(old_provider.calls.load(Ordering::Relaxed), old_calls);
                assert_eq!(
                    changed_provider.calls.load(Ordering::Relaxed),
                    current_calls
                );
                assert!(
                    reopened
                        .search_text(&cx, changed_provider.as_ref(), "retry", 1)
                        .await
                        .is_err()
                );
                assert_eq!(
                    changed_provider.calls.load(Ordering::Relaxed),
                    current_calls
                );
                assert_eq!(
                    reopened
                        .search_text(&cx, old_provider.as_ref(), "retry", 1)
                        .await
                        .unwrap()[0]
                        .doc_id,
                    "retry-old"
                );
                assert_eq!(
                    old_reader
                        .search_text(&cx, old_provider.as_ref(), "retry", 1)
                        .await
                        .unwrap()[0]
                        .doc_id,
                    "retry-old"
                );
                assert_eq!(
                    current_reader
                        .search_text(&cx, changed_provider.as_ref(), "retry", 1)
                        .await
                        .unwrap()[0]
                        .doc_id,
                    "retry-current"
                );

                // The original authority is no longer an immediate physical
                // predecessor after rollback, although its old reader lives.
                let authority_after = fs::read(old.root.join("AUTHORITY")).unwrap();
                let lock_after = fs::read(old.root.join("LOCK")).unwrap();
                options.from_receipt = Some(options.receipt.clone());
                options.receipt = parent.join("out-of-window.json");
                assert!(rollback(&cx, &options, &mut Vec::new()).is_err());
                assert!(!options.receipt.exists());
                assert_eq!(
                    fs::read(old.root.join("AUTHORITY")).unwrap(),
                    authority_after
                );
                assert_eq!(fs::read(old.root.join("LOCK")).unwrap(), lock_after);
            }
        });
    }

    #[test]
    fn explicit_rollback_recovers_damaged_current_components_without_inference_or_rewriting_them() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (label, tier, ann, damaged_role, missing) in [
                (
                    "recover-fast-exact-vector",
                    EmbedderTierTag::Fast,
                    false,
                    GenerationComponentRole::Vector,
                    false,
                ),
                (
                    "recover-quality-exact-lexical",
                    EmbedderTierTag::Quality,
                    false,
                    GenerationComponentRole::Lexical,
                    true,
                ),
                (
                    "recover-fast-hnsw-metadata",
                    EmbedderTierTag::Fast,
                    true,
                    GenerationComponentRole::Metadata,
                    false,
                ),
                (
                    "recover-quality-hnsw-ann",
                    EmbedderTierTag::Quality,
                    true,
                    GenerationComponentRole::Ann,
                    false,
                ),
            ] {
                let parent = private_parent(label);
                let old_provider = provider();
                let old = publish(&cx, &parent, None, &old_provider, tier, "old", ann).await;
                let old_reader = old.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                let current_provider = Arc::new(Provider {
                    identity: {
                        let mut identity = provider_identity();
                        identity.producer.implementation_revision =
                            "damaged-current-migrated-producer".to_owned();
                        identity
                    },
                    calls: AtomicUsize::new(0),
                });
                // A producer-only change can legitimately reuse the exact ANN
                // bytes. Add a current-only row so damaging this graph cannot
                // also damage the retained predecessor's shared component.
                let current_root = admit_root(&parent.join("root")).unwrap();
                let built = NativeIndexBuilder::new(
                    parent.join("current-work"),
                    new_generation(old_reader.authority().sequence + 1).unwrap(),
                    current_provider.clone(),
                )
                .unwrap()
                .with_fast_storage(
                    NativeBuildPrecision::F32,
                    NativeBuildRetrieval::Hnsw {
                        params: HnswParams::default(),
                        seed: 17,
                    },
                )
                .add_documents([
                    IndexableDocument::new("retry-current", "retry network requests"),
                    IndexableDocument::new("garden-current", "garden flowers"),
                    IndexableDocument::new("current-only", "retry current graph membership"),
                ])
                .build_exact_generation(
                    &cx,
                    tier,
                    CommitRange { low: 1, high: 6 },
                    DEFAULT_IMAGE_BYTES,
                )
                .await
                .unwrap();
                let current = stage_and_publish(
                    &cx,
                    &current_root,
                    &parent.join("root"),
                    &parent.join("current.json"),
                    &built,
                    Some(&old_reader),
                    PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
                )
                .unwrap()
                .receipt;
                let current_reader = current.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                let expected_pair = current_reader.retained_snapshot().authority_pair();
                let damaged = component_path(&current.root, &current_reader, damaged_role);
                assert_ne!(
                    damaged,
                    component_path(&old.root, &old_reader, damaged_role),
                    "this fixture must damage only the current component"
                );
                let retained_original = parent.join("original-damaged-component");
                let original_bytes = fs::read(&damaged).unwrap();
                // Preserve the healthy inode and create the fault as a sealed
                // replacement, so fault injection also works without root.
                fs::rename(&damaged, &retained_original).unwrap();
                if !missing {
                    write_immutable(&cx, &damaged, b"damaged-current-component-evidence").unwrap();
                }
                let current_objects = current_reader
                    .retained_snapshot()
                    .manifest()
                    .components()
                    .iter()
                    .map(|(role, _)| {
                        let path = component_path(&current.root, &current_reader, role);
                        let bytes = fs::read(&path).ok();
                        (path, bytes)
                    })
                    .collect::<Vec<_>>();
                let authority_before = fs::read(current.root.join("AUTHORITY")).unwrap();
                let lock_before = fs::read(current.root.join("LOCK")).unwrap();
                let old_receipt = fs::read(parent.join("old.json")).unwrap();
                let current_receipt = fs::read(parent.join("current.json")).unwrap();
                let old_calls = old_provider.calls.load(Ordering::Relaxed);
                let current_calls = current_provider.calls.load(Ordering::Relaxed);
                assert!(current.open(&cx, DEFAULT_IMAGE_BYTES).is_err());
                let search = search_options(&parent.join("current.json"), None);
                let model_loads = AtomicUsize::new(0);
                assert!(
                    search_with_loader(&cx, &search, &mut Vec::new(), |_| {
                        model_loads.fetch_add(1, Ordering::Relaxed);
                        Err(bad("damaged current search must fail before model loading"))
                    })
                    .await
                    .is_err()
                );
                assert_eq!(model_loads.load(Ordering::Relaxed), 0);
                let mut options = parse(&rollback_arguments()).unwrap().unwrap();
                options.from_receipt = Some(parent.join("current.json"));
                options.target_receipt = Some(parent.join("old.json"));
                options.receipt = parent.join("restored.json");
                let mut output = Vec::new();
                assert!(rollback(&cx, &options, &mut output).is_err());
                assert!(output.is_empty());
                assert!(!options.receipt.exists());
                assert_eq!(
                    fs::read(current.root.join("AUTHORITY")).unwrap(),
                    authority_before
                );
                assert_eq!(fs::read(current.root.join("LOCK")).unwrap(), lock_before);

                options.allow_damaged_current = true;
                rollback(&cx, &options, &mut output).unwrap();
                let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(report["event"], "rolled_back");
                assert_eq!(report["publication"], "committed");
                assert_eq!(report["serving"], "activated");
                assert_eq!(report["current_admission"], "control_manifest_only");
                assert_eq!(report["retrieval"], if ann { "hnsw" } else { "exact" });
                assert_eq!(report["replaces"], serde_json::json!(current.authority));
                assert_eq!(report["restores"], serde_json::json!(old.authority));
                assert_eq!(old_provider.calls.load(Ordering::Relaxed), old_calls);
                assert_eq!(
                    current_provider.calls.load(Ordering::Relaxed),
                    current_calls
                );
                assert_eq!(fs::read(parent.join("old.json")).unwrap(), old_receipt);
                assert_eq!(
                    fs::read(parent.join("current.json")).unwrap(),
                    current_receipt
                );
                for (path, bytes) in &current_objects {
                    assert_eq!(fs::read(path).ok(), *bytes, "{}", path.display());
                }
                if missing {
                    assert!(!damaged.exists());
                }
                assert_eq!(fs::read(&retained_original).unwrap(), original_bytes);
                let restored = Receipt::read(&options.receipt).unwrap();
                assert_eq!(restored.generation, old.generation);
                assert_eq!(restored.identity, old.identity);
                assert_eq!(restored.tier, old.tier);
                assert_eq!(restored.documents, old.documents);
                assert_eq!(restored.authority().unwrap().sequence, 3);
                assert_eq!(restored.recovery_attempt().unwrap().expected, expected_pair);
                let reopened = restored.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(
                    reopened.retained_snapshot().manifest().predecessor(),
                    Some(current.authority().unwrap())
                );
                assert_eq!(
                    reopened.retained_snapshot().manifest().components(),
                    old_reader.retained_snapshot().manifest().components()
                );
                assert_eq!(
                    reopened.metadata().commit_range,
                    old_reader.metadata().commit_range
                );
                assert_eq!(
                    reopened
                        .search_text(&cx, old_provider.as_ref(), "retry", 1)
                        .await
                        .unwrap()[0]
                        .doc_id,
                    "retry-old"
                );
                assert!(
                    reopened
                        .search_text(&cx, current_provider.as_ref(), "retry", 1)
                        .await
                        .is_err()
                );
                assert_eq!(
                    current_provider.calls.load(Ordering::Relaxed),
                    current_calls
                );
                assert_eq!(
                    current_reader
                        .search_text(&cx, current_provider.as_ref(), "retry", 1)
                        .await
                        .unwrap()[0]
                        .doc_id,
                    "retry-current",
                    "the previously admitted reader retains its original healthy bytes"
                );
            }
        });
    }

    #[test]
    fn damaged_current_recovery_refuses_untrusted_selection_and_preserves_a_lost_response_receipt()
    {
        struct ClosedOutput;
        impl Write for ClosedOutput {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let parent = private_parent("damaged-recovery-refusal");
            let provider = provider();
            let ancient = publish(
                &cx,
                &parent,
                None,
                &provider,
                EmbedderTierTag::Fast,
                "ancient",
                false,
            )
            .await;
            let ancient_reader = ancient.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            let target = publish(
                &cx,
                &parent,
                Some(&ancient_reader),
                &provider,
                EmbedderTierTag::Fast,
                "old",
                true,
            )
            .await;
            let target_reader = target.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            let current = publish(
                &cx,
                &parent,
                Some(&target_reader),
                &provider,
                EmbedderTierTag::Fast,
                "current",
                true,
            )
            .await;
            let current_reader = current.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            let damaged = component_path(
                &current.root,
                &current_reader,
                GenerationComponentRole::Vector,
            );
            let damaged_bytes = b"preserved-current-vector-failure";
            fs::rename(&damaged, parent.join("original-current-vector")).unwrap();
            write_immutable(&cx, &damaged, damaged_bytes).unwrap();
            let authority_before = fs::read(current.root.join("AUTHORITY")).unwrap();
            let lock_before = fs::read(current.root.join("LOCK")).unwrap();
            let target_receipt_before = fs::read(parent.join("old.json")).unwrap();
            let current_receipt_before = fs::read(parent.join("current.json")).unwrap();
            let calls = provider.calls.load(Ordering::Relaxed);
            let mut options = parse(&rollback_arguments()).unwrap().unwrap();
            options.allow_damaged_current = true;
            options.from_receipt = Some(parent.join("current.json"));
            options.target_receipt = Some(parent.join("old.json"));
            let refuse = |options: &Options, label: &str| {
                let mut output = Vec::new();
                assert!(rollback(&cx, options, &mut output).is_err(), "{label}");
                assert!(output.is_empty(), "{label}");
                assert!(!options.receipt.exists(), "{label}");
                assert_eq!(
                    fs::read(current.root.join("AUTHORITY")).unwrap(),
                    authority_before,
                    "{label}"
                );
                assert_eq!(
                    fs::read(current.root.join("LOCK")).unwrap(),
                    lock_before,
                    "{label}"
                );
                assert_eq!(provider.calls.load(Ordering::Relaxed), calls, "{label}");
                assert_eq!(
                    fs::read(parent.join("old.json")).unwrap(),
                    target_receipt_before,
                    "{label}"
                );
                assert_eq!(
                    fs::read(parent.join("current.json")).unwrap(),
                    current_receipt_before,
                    "{label}"
                );
            };
            for label in [
                "not-immediate",
                "same-head",
                "stale-current",
                "current-generation",
                "target-producer",
                "combined-budget",
            ] {
                options.from_receipt = Some(parent.join("current.json"));
                options.target_receipt = Some(parent.join("old.json"));
                options.max_image_bytes = DEFAULT_IMAGE_BYTES;
                options.receipt = parent.join(format!("refused-{label}.json"));
                match label {
                    "not-immediate" => {
                        options.target_receipt = Some(parent.join("ancient.json"));
                    }
                    "same-head" => options.target_receipt = Some(parent.join("current.json")),
                    "stale-current" => {
                        options.from_receipt = Some(parent.join("old.json"));
                        options.target_receipt = Some(parent.join("ancient.json"));
                    }
                    "current-generation" => {
                        let mut wrong = current.clone();
                        wrong.generation.nonce[0] ^= 1;
                        let path = parent.join("wrong-current-generation.json");
                        fs::write(&path, serde_json::to_vec(&wrong).unwrap()).unwrap();
                        options.from_receipt = Some(path);
                    }
                    "target-producer" => {
                        let mut wrong = target.clone();
                        let mut identity = wrong.identity.identity.clone();
                        identity.producer.implementation_revision =
                            "wrong-damaged-recovery-target".to_owned();
                        wrong.identity = identity.freeze().unwrap();
                        let path = parent.join("wrong-target-producer.json");
                        fs::write(&path, serde_json::to_vec(&wrong).unwrap()).unwrap();
                        options.target_receipt = Some(path);
                    }
                    "combined-budget" => {
                        // The target itself fits exactly. Recovery must also
                        // charge the independently authenticated current manifest.
                        options.max_image_bytes = usize::try_from(target.image_bytes).unwrap();
                    }
                    _ => unreachable!(),
                }
                refuse(&options, label);
            }
            options.from_receipt = Some(parent.join("current.json"));
            options.target_receipt = Some(parent.join("old.json"));
            options.max_image_bytes = DEFAULT_IMAGE_BYTES;

            let current_manifest = current.root.join(activation_manifest_name_v1(
                current.authority().unwrap().object_id,
            ));
            let manifest_bytes = fs::read(&current_manifest).unwrap();
            let preserved_manifest = parent.join("original-current-manifest");
            fs::rename(&current_manifest, &preserved_manifest).unwrap();
            write_immutable(&cx, &current_manifest, b"invalid-current-manifest-evidence").unwrap();
            options.receipt = parent.join("refused-current-manifest.json");
            refuse(&options, "current-manifest");
            assert_eq!(
                fs::read(&current_manifest).unwrap(),
                b"invalid-current-manifest-evidence"
            );
            fs::rename(&current_manifest, parent.join("damaged-current-manifest")).unwrap();
            fs::rename(&preserved_manifest, &current_manifest).unwrap();

            // Skipping current component payload reads cannot waive the root's
            // no-follow/no-alias contract. Keep both evidence and alias intact.
            let saved_damage = parent.join("saved-current-vector");
            fs::rename(&damaged, &saved_damage).unwrap();
            std::os::unix::fs::symlink(&saved_damage, &damaged).unwrap();
            options.receipt = parent.join("refused-current-alias.json");
            refuse(&options, "current-component-alias");
            assert!(
                fs::symlink_metadata(&damaged)
                    .unwrap()
                    .file_type()
                    .is_symlink()
            );
            fs::rename(&damaged, parent.join("saved-current-vector-alias")).unwrap();
            fs::rename(&saved_damage, &damaged).unwrap();
            assert_eq!(fs::read(&damaged).unwrap(), damaged_bytes);

            let target_metadata = component_path(
                &target.root,
                &target_reader,
                GenerationComponentRole::Metadata,
            );
            let metadata_bytes = fs::read(&target_metadata).unwrap();
            let preserved_metadata = parent.join("original-target-metadata");
            fs::rename(&target_metadata, &preserved_metadata).unwrap();
            write_immutable(&cx, &target_metadata, b"invalid-target-metadata-evidence").unwrap();
            options.receipt = parent.join("refused-target-payload.json");
            refuse(&options, "target-payload");
            assert_eq!(
                fs::read(&target_metadata).unwrap(),
                b"invalid-target-metadata-evidence"
            );
            fs::rename(&target_metadata, parent.join("damaged-target-metadata")).unwrap();
            fs::rename(&preserved_metadata, &target_metadata).unwrap();

            options.receipt = parent.join("refused-cancelled.json");
            cx.set_cancel_requested(true);
            refuse(&options, "cancelled");
            cx.set_cancel_requested(false);

            // All preflight failures left the authentic predecessor usable.
            // A broken output after the valid repair is committed, not a reason
            // to republish; the saved attempt is sufficient for fresh recovery.
            options.receipt = parent.join("committed-without-response.json");
            let error = rollback(&cx, &options, &mut ClosedOutput).unwrap_err();
            let failure = publication_failure(error.as_ref()).unwrap();
            assert_eq!(failure["publication"], "committed");
            assert_eq!(failure["serving"], "activated");
            assert_eq!(failure["reason"], "result_delivery_failed");
            assert_eq!(provider.calls.load(Ordering::Relaxed), calls);
            let restored = Receipt::read(&options.receipt).unwrap();
            assert_eq!(restored.generation, target.generation);
            assert_eq!(restored.identity, target.identity);
            assert_eq!(restored.authority().unwrap().sequence, 4);
            assert_eq!(
                restored.recovery_attempt().unwrap().expected,
                current_reader.retained_snapshot().authority_pair()
            );
            let mut recovery = parse(&["authority-recover", "--receipt", "selected.json"])
                .unwrap()
                .unwrap();
            recovery.receipt = options.receipt.clone();
            let mut output = Vec::new();
            recover(&cx, &recovery, &mut output).unwrap();
            let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(report["publication"], "committed");
            assert_eq!(report["serving"], "activated");
            assert_eq!(provider.calls.load(Ordering::Relaxed), calls);
            assert_eq!(fs::read(&damaged).unwrap(), damaged_bytes);
            assert_eq!(fs::read(&current_manifest).unwrap(), manifest_bytes);
            assert_eq!(fs::read(&target_metadata).unwrap(), metadata_bytes);
            let reopened = restored.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            assert_eq!(
                reopened
                    .search_text(&cx, provider.as_ref(), "retry", 1)
                    .await
                    .unwrap()[0]
                    .doc_id,
                "retry-old"
            );
        });
    }

    #[test]
    fn rollback_preflight_refuses_wrong_receipts_and_limits_before_publication() {
        struct ClosedOutput;
        impl Write for ClosedOutput {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed"))
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let parent = private_parent("rollback-refusal");
            let provider = provider();
            let old = publish(
                &cx,
                &parent,
                None,
                &provider,
                EmbedderTierTag::Fast,
                "old",
                true,
            )
            .await;
            let old_reader = old.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            let current = publish(
                &cx,
                &parent,
                Some(&old_reader),
                &provider,
                EmbedderTierTag::Fast,
                "current",
                false,
            )
            .await;
            let authority_before = fs::read(old.root.join("AUTHORITY")).unwrap();
            let lock_before = fs::read(old.root.join("LOCK")).unwrap();
            let calls = provider.calls.load(Ordering::Relaxed);
            let mut options = parse(&rollback_arguments()).unwrap().unwrap();
            options.from_receipt = Some(parent.join("current.json"));
            options.target_receipt = Some(parent.join("old.json"));
            for label in [
                "same-head",
                "stale-current",
                "budget",
                "tier",
                "root",
                "root-id",
                "identity",
                "documents",
                "profile",
            ] {
                options.from_receipt = Some(parent.join("current.json"));
                options.target_receipt = Some(parent.join("old.json"));
                options.max_image_bytes = DEFAULT_IMAGE_BYTES;
                options.receipt = parent.join(format!("refused-{label}.json"));
                match label {
                    "same-head" => options.target_receipt = Some(parent.join("current.json")),
                    "stale-current" => options.from_receipt = Some(parent.join("old.json")),
                    "budget" => options.max_image_bytes = 1,
                    _ => {
                        let mut wrong = old.clone();
                        match label {
                            "tier" => wrong.tier = EmbedderTierTag::Quality,
                            "root" => wrong.root = parent.join("foreign-root"),
                            "root-id" => wrong.root_id[0] ^= 1,
                            "identity" => {
                                let mut identity = wrong.identity.identity.clone();
                                identity.producer.implementation_revision =
                                    "wrong-rollback-producer".to_owned();
                                wrong.identity = identity.freeze().unwrap();
                            }
                            "documents" => wrong.documents = 1,
                            "profile" => wrong.profile = "unadmitted-profile".to_owned(),
                            _ => unreachable!(),
                        }
                        let path = parent.join(format!("wrong-{label}.json"));
                        fs::write(&path, serde_json::to_vec(&wrong).unwrap()).unwrap();
                        options.target_receipt = Some(path);
                    }
                }
                let mut output = Vec::new();
                assert!(rollback(&cx, &options, &mut output).is_err(), "{label}");
                assert!(output.is_empty(), "{label}");
                assert!(!options.receipt.exists(), "{label}");
                assert_eq!(
                    fs::read(old.root.join("AUTHORITY")).unwrap(),
                    authority_before,
                    "{label}"
                );
                assert_eq!(
                    fs::read(old.root.join("LOCK")).unwrap(),
                    lock_before,
                    "{label}"
                );
                assert_eq!(provider.calls.load(Ordering::Relaxed), calls, "{label}");
            }
            options.from_receipt = Some(parent.join("current.json"));
            options.target_receipt = Some(parent.join("old.json"));
            options.max_image_bytes = DEFAULT_IMAGE_BYTES;
            options.receipt = parent.join("cancelled.json");
            cx.set_cancel_requested(true);
            assert!(rollback(&cx, &options, &mut Vec::new()).is_err());
            cx.set_cancel_requested(false);
            assert!(!options.receipt.exists());
            options.receipt = parent.join("old.json");
            let saved = fs::read(&options.receipt).unwrap();
            assert!(rollback(&cx, &options, &mut Vec::new()).is_err());
            assert_eq!(fs::read(&options.receipt).unwrap(), saved);
            assert_eq!(
                fs::read(old.root.join("AUTHORITY")).unwrap(),
                authority_before
            );
            assert_eq!(fs::read(old.root.join("LOCK")).unwrap(), lock_before);
            assert_eq!(provider.calls.load(Ordering::Relaxed), calls);
            let selected = current.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            assert_eq!(
                selected
                    .search_text(&cx, provider.as_ref(), "retry", 1)
                    .await
                    .unwrap()[0]
                    .doc_id,
                "retry-current"
            );
            assert_eq!(
                old_reader
                    .search_text(&cx, provider.as_ref(), "retry", 1)
                    .await
                    .unwrap()[0]
                    .doc_id,
                "retry-old"
            );
            let calls = provider.calls.load(Ordering::Relaxed);
            options.receipt = parent.join("lost-response.json");
            let error = rollback(&cx, &options, &mut ClosedOutput).unwrap_err();
            let failure = publication_failure(error.as_ref()).unwrap();
            assert_eq!(failure["publication"], "committed");
            assert_eq!(failure["serving"], "activated");
            let restored = Receipt::read(&options.receipt).unwrap();
            assert_eq!(restored.generation, old.generation);
            assert_eq!(restored.authority().unwrap().sequence, 3);
            assert!(restored.open(&cx, DEFAULT_IMAGE_BYTES).is_ok());
            assert_eq!(provider.calls.load(Ordering::Relaxed), calls);
        });
    }

    fn update_options(parent: &Path, previous: &Path, label: &str, input: &Path) -> Options {
        let mut options = parse(&update_arguments()).unwrap().unwrap();
        options.from_receipt = Some(previous.to_path_buf());
        options.work = Some(parent.join(format!("{label}-work")));
        options.receipt = parent.join(format!("{label}.json"));
        options.input = Some(input.to_path_buf());
        options
    }

    #[test]
    fn authority_updates_reuse_unchanged_documents_and_publish_complete_successors() {
        use frankensearch_core::generation::GenerationComponentRole;
        use frankensearch_index::generation_root::generation_reader::component_object_name_v1;

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (tier_name, tier) in [
                ("fast", EmbedderTierTag::Fast),
                ("quality", EmbedderTierTag::Quality),
            ] {
                for ann in [false, true] {
                    let parent = private_parent(&format!("update-{tier_name}-{ann}"));
                    let root_path = parent.join("root");
                    let root = create_root(&root_path).expect("qualified Linux root is required");
                    let provider = provider();
                    let model: Arc<dyn Embedder> = provider.clone();
                    let params = HnswParams {
                        m: 4,
                        m0: 8,
                        ef_construction: 24,
                        ef_search: 12,
                    };
                    let retrieval = if ann {
                        NativeBuildRetrieval::Hnsw { params, seed: 671 }
                    } else {
                        NativeBuildRetrieval::Exact
                    };
                    let precision = if ann {
                        NativeBuildPrecision::F16
                    } else {
                        NativeBuildPrecision::F32
                    };
                    let mut metadata =
                        IndexableDocument::new("metadata", "retry persistent source")
                            .with_metadata("version", "old");
                    metadata.title = Some("Original title".to_owned());
                    let initial_work = parent.join("initial-work");
                    let initial =
                        NativeIndexBuilder::new(&initial_work, new_generation(1).unwrap(), model)
                            .unwrap()
                            .with_fast_storage(precision, retrieval)
                            .add_documents([
                                IndexableDocument::new("keep", "garden flowers")
                                    .with_metadata("version", "unchanged"),
                                IndexableDocument::new("changed", "garden older chapter"),
                                IndexableDocument::new("removed", "retry obsolete requests"),
                                metadata,
                            ])
                            .build_exact_generation(
                                &cx,
                                tier,
                                CommitRange { low: 2, high: 7 },
                                DEFAULT_IMAGE_BYTES,
                            )
                            .await
                            .unwrap();
                    assert_eq!(provider.calls.load(Ordering::Relaxed), 4);
                    let first_path = parent.join("initial.json");
                    let selected = stage_and_publish(
                        &cx,
                        &root,
                        &root_path,
                        &first_path,
                        &initial,
                        None,
                        PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
                    )
                    .unwrap()
                    .receipt;
                    let retained = selected
                        .open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES)
                        .unwrap();
                    let first_receipt_bytes = fs::read(&first_path).unwrap();
                    let components = retained.retained_snapshot().manifest().components();
                    let old_objects = [
                        (GenerationComponentRole::Vector, components.vector),
                        (GenerationComponentRole::Lexical, components.lexical),
                    ]
                    .map(|(role, receipt)| {
                        let path = root_path.join(component_object_name_v1(role, receipt.sha256));
                        let bytes = fs::read(&path).unwrap();
                        (path, bytes)
                    });
                    drop(initial);
                    fs::rename(&initial_work, parent.join("archived-initial-work")).unwrap();

                    // These are document inference calls, not a claim that a
                    // freshly loaded registered model skips its own self-test.
                    let metadata_input = parent.join("metadata.jsonl");
                    fs::write(
                        &metadata_input,
                        concat!(
                            "{\"op\":\"upsert\",\"id\":\"metadata\",\"content\":\"retry persistent source\",",
                            "\"title\":\"Refreshed unique title\",\"metadata\":{\"version\":\"new\"}}\n",
                            "{\"op\":\"upsert\",\"id\":\"keep\",\"content\":\"garden flowers\",",
                            "\"metadata\":{\"version\":\"unchanged\"}}\n",
                            "{\"op\":\"delete\",\"id\":\"already-absent\"}\n",
                        ),
                    )
                    .unwrap();
                    let mut metadata_options =
                        update_options(&parent, &first_path, "metadata", &metadata_input);
                    metadata_options.source_high = Some(7);
                    let before_metadata = provider.calls.load(Ordering::Relaxed);
                    let model: Arc<dyn Embedder> = provider.clone();
                    let mut output = Vec::new();
                    update_generation_with_loader(
                        &cx,
                        &metadata_options,
                        &mut output,
                        |requested| {
                            assert_eq!(requested, tier);
                            Ok(Arc::clone(&model))
                        },
                    )
                    .await
                    .unwrap();
                    assert_eq!(provider.calls.load(Ordering::Relaxed), before_metadata);
                    let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
                    assert_eq!(report["event"], "updated");
                    assert_eq!(report["publication"], "committed");
                    assert_eq!(report["serving"], "activated");
                    let second = Receipt::read(&metadata_options.receipt).unwrap();
                    assert_eq!(second.identity, selected.identity);
                    let second_snapshot = second
                        .open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES)
                        .unwrap();
                    assert_eq!(
                        second_snapshot.metadata().commit_range,
                        CommitRange { low: 2, high: 7 }
                    );
                    let title_hits = second_snapshot
                        .search_text(&cx, provider.as_ref(), "Refreshed unique title", 4)
                        .await
                        .unwrap();
                    assert!(
                        title_hits
                            .iter()
                            .find(|hit| hit.doc_id == "metadata")
                            .unwrap()
                            .lexical_score
                            .is_some_and(|score| score > 0.0)
                    );

                    let edits = parent.join("mixed.jsonl");
                    fs::write(
                        &edits,
                        concat!(
                            "{\"op\":\"upsert\",\"id\":\"changed\",\"content\":\"superseded temporary body\"}\n",
                            "{\"op\":\"upsert\",\"id\":\"fresh\",\"content\":\"superseded temporary new body\"}\n",
                            "{\"op\":\"delete\",\"id\":\"fresh\"}\n",
                            "{\"op\":\"delete\",\"id\":\"removed\"}\n",
                            "{\"op\":\"upsert\",\"id\":\"changed\",\"content\":\"retry revised network\"}\n",
                            "{\"op\":\"upsert\",\"id\":\"fresh\",\"content\":\"garden fresh flowers\"}\n",
                        ),
                    )
                    .unwrap();
                    let options =
                        update_options(&parent, &metadata_options.receipt, "mixed", &edits);
                    let before_mixed = provider.calls.load(Ordering::Relaxed);
                    let model: Arc<dyn Embedder> = provider.clone();
                    output.clear();
                    update_generation_with_loader(&cx, &options, &mut output, |requested| {
                        assert_eq!(requested, tier);
                        Ok(Arc::clone(&model))
                    })
                    .await
                    .unwrap();
                    assert_eq!(
                        provider.calls.load(Ordering::Relaxed),
                        before_mixed + 2,
                        "only final changed/new bodies may be submitted to the provider"
                    );
                    let committed = Receipt::read(&options.receipt).unwrap();
                    assert_eq!(committed.identity, selected.identity);
                    assert_eq!(committed.tier, tier);
                    assert_eq!(committed.documents, 4);
                    assert_eq!(committed.authority().unwrap().sequence, 3);
                    let active = committed.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                    assert_eq!(
                        active.metadata().commit_range,
                        CommitRange { low: 2, high: 8 }
                    );
                    match active.retrieval_configuration() {
                        NativeBuildRetrieval::Hnsw {
                            params: actual,
                            seed,
                        } => {
                            assert!(ann);
                            assert_eq!(actual, params);
                            assert_eq!(seed, 671);
                        }
                        NativeBuildRetrieval::Exact => assert!(!ann),
                    }
                    let source = active
                        .source_documents(&cx, crate::MAX_INPUT_BYTES)
                        .unwrap();
                    assert_eq!(
                        source
                            .iter()
                            .map(|document| document.id.as_str())
                            .collect::<BTreeSet<_>>(),
                        BTreeSet::from(["changed", "fresh", "keep", "metadata"])
                    );
                    let kept = source
                        .iter()
                        .find(|document| document.id == "keep")
                        .unwrap();
                    assert_eq!(kept.content, "garden flowers");
                    assert_eq!(kept.metadata.get("version").unwrap(), "unchanged");
                    let metadata = source
                        .iter()
                        .find(|document| document.id == "metadata")
                        .unwrap();
                    assert_eq!(metadata.content, "retry persistent source");
                    assert_eq!(metadata.title.as_deref(), Some("Refreshed unique title"));
                    assert_eq!(metadata.metadata.get("version").unwrap(), "new");
                    assert_eq!(
                        source
                            .iter()
                            .find(|document| document.id == "changed")
                            .unwrap()
                            .content,
                        "retry revised network"
                    );
                    let current_hits = active
                        .search_text(&cx, provider.as_ref(), "retry revised network", 4)
                        .await
                        .unwrap();
                    assert_eq!(
                        current_hits
                            .iter()
                            .map(|hit| hit.doc_id.as_str())
                            .collect::<BTreeSet<_>>(),
                        BTreeSet::from(["changed", "fresh", "keep", "metadata"])
                    );
                    let changed = current_hits
                        .iter()
                        .find(|hit| hit.doc_id == "changed")
                        .unwrap();
                    let semantic_score = match tier {
                        EmbedderTierTag::Fast => changed.fast_score,
                        EmbedderTierTag::Quality => changed.quality_score,
                    };
                    assert_eq!(semantic_score, Some(1.0));
                    assert!(changed.lexical_score.is_some_and(|score| score > 0.0));
                    let old_hits = retained
                        .search_text(&cx, provider.as_ref(), "retry revised network", 4)
                        .await
                        .unwrap();
                    assert_eq!(
                        old_hits
                            .iter()
                            .map(|hit| hit.doc_id.as_str())
                            .collect::<BTreeSet<_>>(),
                        BTreeSet::from(["changed", "keep", "metadata", "removed"])
                    );
                    let old_changed = old_hits.iter().find(|hit| hit.doc_id == "changed").unwrap();
                    assert_eq!(
                        match tier {
                            EmbedderTierTag::Fast => old_changed.fast_score,
                            EmbedderTierTag::Quality => old_changed.quality_score,
                        },
                        Some(0.0)
                    );
                    let old_source = retained
                        .source_documents(&cx, crate::MAX_INPUT_BYTES)
                        .unwrap();
                    let old_metadata = old_source
                        .iter()
                        .find(|document| document.id == "metadata")
                        .unwrap();
                    assert_eq!(old_metadata.title.as_deref(), Some("Original title"));
                    assert_eq!(old_metadata.metadata.get("version").unwrap(), "old");
                    assert_eq!(fs::read(&first_path).unwrap(), first_receipt_bytes);
                    for (path, bytes) in &old_objects {
                        assert_eq!(fs::read(path).unwrap(), *bytes);
                    }

                    // Superseded command receipts must fail before attempting
                    // the deliberately missing delta or local model directory.
                    let mut stale =
                        update_options(&parent, &first_path, "stale", &parent.join("absent.jsonl"));
                    stale.models = Some(parent.join("absent-models"));
                    output.clear();
                    let before_stale = provider.calls.load(Ordering::Relaxed);
                    let authority_before = fs::read(root_path.join("AUTHORITY")).unwrap();
                    let error = update_generation(&cx, &stale, &mut output, None)
                        .await
                        .unwrap_err();
                    assert!(error.to_string().contains("head differs"), "{error}");
                    assert!(output.is_empty());
                    assert!(!stale.work.as_ref().unwrap().exists());
                    assert!(!stale.receipt.exists());
                    assert_eq!(provider.calls.load(Ordering::Relaxed), before_stale);
                    assert_eq!(
                        fs::read(root_path.join("AUTHORITY")).unwrap(),
                        authority_before
                    );
                }
            }
        });
    }

    #[test]
    fn authority_update_preflight_preserves_selection_before_loading_models_or_staging() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (label, tier) in [
                ("update-preflight-fast", EmbedderTierTag::Fast),
                ("update-preflight-quality", EmbedderTierTag::Quality),
            ] {
                let parent = private_parent(label);
                let provider = provider();
                let selected =
                    publish(&cx, &parent, None, &provider, tier, "baseline", false).await;
                let first_path = parent.join("baseline.json");
                let root_path = parent.join("root");
                let authority_before = fs::read(root_path.join("AUTHORITY")).unwrap();
                let receipt_before = fs::read(&first_path).unwrap();
                let calls = provider.calls.load(Ordering::Relaxed);
                let ordinary = parent.join("ordinary.jsonl");
                fs::write(&ordinary, "{\"op\":\"delete\",\"id\":\"absent\"}\n").unwrap();
                let invalid_delete = parent.join("invalid-delete.jsonl");
                fs::write(&invalid_delete, "{\"op\":\"delete\",\"id\":\" \"}\n").unwrap();
                let malformed = parent.join("malformed.jsonl");
                fs::write(
                    &malformed,
                    concat!(
                        "{\"op\":\"upsert\",\"id\":\"retry-baseline\",\"content\":7}\n",
                        "{\"op\":\"upsert\",\"id\":\"retry-baseline\",\"content\":\"valid final edit\"}\n",
                    ),
                )
                .unwrap();
                let absent = parent.join("absent.jsonl");
                for (name, input, low, high, budget, reason) in [
                    (
                        "backward",
                        &absent,
                        0,
                        5,
                        crate::MAX_INPUT_BYTES,
                        "backward",
                    ),
                    ("source-budget", &ordinary, 1, 5, 1, "stored-payload-limit"),
                    (
                        "invalid-delete",
                        &invalid_delete,
                        1,
                        5,
                        crate::MAX_INPUT_BYTES,
                        "nonblank",
                    ),
                    (
                        "invalid-overwritten",
                        &malformed,
                        1,
                        5,
                        crate::MAX_INPUT_BYTES,
                        "invalid update",
                    ),
                ] {
                    let mut options = update_options(&parent, &first_path, name, input);
                    options.source_low = Some(low);
                    options.source_high = Some(high);
                    options.max_source_bytes = budget;
                    let attempts = AtomicUsize::new(0);
                    let mut output = Vec::new();
                    let error = update_generation_with_loader(&cx, &options, &mut output, |_| {
                        attempts.fetch_add(1, Ordering::Relaxed);
                        Err(bad("preflight must finish before model loading"))
                    })
                    .await
                    .unwrap_err();
                    assert!(error.to_string().contains(reason), "{name}: {error}");
                    assert_eq!(attempts.load(Ordering::Relaxed), 0, "{name}");
                    assert_eq!(provider.calls.load(Ordering::Relaxed), calls, "{name}");
                    assert!(output.is_empty(), "{name}");
                    assert!(!options.work.as_ref().unwrap().exists(), "{name}");
                    assert!(!options.receipt.exists(), "{name}");
                    assert_eq!(
                        fs::read(root_path.join("AUTHORITY")).unwrap(),
                        authority_before
                    );
                    assert_eq!(fs::read(&first_path).unwrap(), receipt_before);
                }

                // A source checkpoint may advance without any document edits.
                // The complete unchanged cohort still receives a new authority.
                let empty = parent.join("empty.jsonl");
                fs::write(&empty, b"\n \t\n").unwrap();
                let mut options = update_options(&parent, &first_path, "checkpoint", &empty);
                options.source_low = Some(1);
                options.source_high = Some(6);
                let model: Arc<dyn Embedder> = provider.clone();
                update_generation_with_loader(&cx, &options, &mut Vec::new(), |_| {
                    Ok(Arc::clone(&model))
                })
                .await
                .unwrap();
                assert_eq!(provider.calls.load(Ordering::Relaxed), calls);
                let advanced = Receipt::read(&options.receipt).unwrap();
                assert_eq!(advanced.identity, selected.identity);
                assert_eq!(advanced.documents, 2);
                assert_eq!(advanced.authority().unwrap().sequence, 2);
                let snapshot = advanced.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(
                    snapshot.metadata().commit_range,
                    CommitRange { low: 1, high: 6 }
                );
                assert_eq!(
                    snapshot
                        .source_documents(&cx, crate::MAX_INPUT_BYTES)
                        .unwrap()
                        .iter()
                        .map(|document| document.id.as_str())
                        .collect::<BTreeSet<_>>(),
                    BTreeSet::from(["garden-baseline", "retry-baseline"])
                );
            }
        });
    }

    #[test]
    fn authority_update_loses_to_publication_after_its_predecessor_was_admitted() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (label, tier) in [
                ("update-race-fast", EmbedderTierTag::Fast),
                ("update-race-quality", EmbedderTierTag::Quality),
            ] {
                let parent = private_parent(label);
                let provider = provider();
                let selected = publish(&cx, &parent, None, &provider, tier, "baseline", true).await;
                let root_path = parent.join("root");
                let root = admit_root(&root_path).unwrap();
                let retained = selected
                    .open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES)
                    .unwrap();
                let model: Arc<dyn Embedder> = provider.clone();
                let competitor = retained
                    .begin_update(
                        &cx,
                        parent.join("winner-work"),
                        new_generation(2).unwrap(),
                        model,
                        crate::MAX_INPUT_BYTES,
                    )
                    .unwrap()
                    .upsert_document(IndexableDocument::new("winner", "retry concurrent winner"))
                    .build_exact_generation(
                        &cx,
                        tier,
                        CommitRange { low: 1, high: 6 },
                        DEFAULT_IMAGE_BYTES,
                    )
                    .await
                    .unwrap();
                let input = parent.join("loser.jsonl");
                fs::write(
                    &input,
                    "{\"op\":\"upsert\",\"id\":\"loser\",\"content\":\"retry losing update\"}\n",
                )
                .unwrap();
                let first_path = parent.join("baseline.json");
                let receipt_before = fs::read(&first_path).unwrap();
                let mut options = update_options(&parent, &first_path, "loser", &input);
                options.source_low = Some(1);
                options.source_high = Some(7);
                let winner_path = parent.join("winner.json");
                let mut winner = None;
                let mut authority_after_winner = Vec::new();
                let model: Arc<dyn Embedder> = provider.clone();
                let mut output = Vec::new();
                let before = provider.calls.load(Ordering::Relaxed);
                let error =
                    update_generation_with_loader(&cx, &options, &mut output, |requested| {
                        assert_eq!(requested, tier);
                        // The real command has already admitted its predecessor
                        // and inputs. Publish the competing complete candidate
                        // before allowing its model/load/build lane to proceed.
                        winner = Some(
                            stage_and_publish(
                                &cx,
                                &root,
                                &root_path,
                                &winner_path,
                                &competitor,
                                Some(&retained),
                                PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
                            )?
                            .receipt,
                        );
                        authority_after_winner = fs::read(root_path.join("AUTHORITY"))?;
                        Ok(Arc::clone(&model))
                    })
                    .await
                    .unwrap_err();
                assert_eq!(
                    publication_failure(error.as_ref()).unwrap()["publication"],
                    "not_committed"
                );
                assert!(output.is_empty());
                assert_eq!(provider.calls.load(Ordering::Relaxed), before + 1);
                assert_eq!(
                    fs::read(root_path.join("AUTHORITY")).unwrap(),
                    authority_after_winner
                );
                assert_eq!(fs::read(&first_path).unwrap(), receipt_before);
                let losing_receipt = Receipt::read(&options.receipt).unwrap();
                assert_eq!(
                    losing_receipt.recovery_attempt().unwrap().expected,
                    retained.retained_snapshot().authority_pair()
                );
                let winner = winner.unwrap();
                assert_ne!(
                    losing_receipt.authority().unwrap(),
                    winner.authority().unwrap()
                );
                let active = winner.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(
                    active.metadata().commit_range,
                    CommitRange { low: 1, high: 6 }
                );
                for snapshot in [&active, &retained] {
                    let source = snapshot
                        .source_documents(&cx, crate::MAX_INPUT_BYTES)
                        .unwrap();
                    let hits = snapshot
                        .search_text(&cx, provider.as_ref(), "retry", 10)
                        .await
                        .unwrap();
                    assert!(source.iter().all(|document| document.id != "loser"));
                    assert!(hits.iter().all(|hit| hit.doc_id != "loser"));
                }
                assert!(
                    active
                        .source_documents(&cx, crate::MAX_INPUT_BYTES)
                        .unwrap()
                        .iter()
                        .any(|document| document.id == "winner")
                );
                assert!(
                    retained
                        .source_documents(&cx, crate::MAX_INPUT_BYTES)
                        .unwrap()
                        .iter()
                        .all(|document| document.id != "winner")
                );
            }
        });
    }

    #[test]
    fn retained_source_rebuild_preserves_the_cohort_migrates_the_producer_and_fences_stale_work() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (label, tier) in [
                ("retained-fast", EmbedderTierTag::Fast),
                ("retained-quality", EmbedderTierTag::Quality),
            ] {
                let parent = private_parent(label);
                let root_path = parent.join("root");
                let root = create_root(&root_path).expect("qualified Linux root is required");
                let provider = provider();
                let model: Arc<dyn Embedder> = provider.clone();
                let mut retry =
                    IndexableDocument::new("zeta", "retry network requests\nexact source")
                        .with_metadata("version", "original")
                        .with_metadata("quoted", "line one\n\"line two\" – λ");
                retry.title = Some("Retries: λ and \"quoted\" text".to_owned());
                let garden = IndexableDocument::new("alpha", "garden flowers")
                    .with_metadata("kind", "guide");
                let original_work = parent.join("original-work");
                let first =
                    NativeIndexBuilder::new(&original_work, new_generation(1).unwrap(), model)
                        .unwrap()
                        .with_fast_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Exact)
                        .add_documents([retry, garden])
                        .build_exact_generation(
                            &cx,
                            tier,
                            CommitRange { low: 2, high: 7 },
                            DEFAULT_IMAGE_BYTES,
                        )
                        .await
                        .unwrap();
                let first_path = parent.join("first.json");
                let selected = stage_and_publish(
                    &cx,
                    &root,
                    &root_path,
                    &first_path,
                    &first,
                    None,
                    PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
                )
                .unwrap()
                .receipt;
                drop(first);
                let retained = selected
                    .open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES)
                    .unwrap();
                // Neither the original build directory nor a source JSONL
                // pathname is available to this retained-source selection.
                fs::rename(&original_work, parent.join("archived-original-work")).unwrap();
                assert!(!original_work.exists());
                let before = provider.calls.load(Ordering::Relaxed);
                let authority_before = fs::read(root_path.join("AUTHORITY")).unwrap();
                let source =
                    retained_rebuild_documents(&cx, &retained, crate::MAX_INPUT_BYTES).unwrap();
                assert_eq!(source.len(), 2);
                assert_eq!(
                    source
                        .iter()
                        .map(|doc| doc.id.as_str())
                        .collect::<BTreeSet<_>>(),
                    BTreeSet::from(["alpha", "zeta"])
                );
                let garden = source.iter().find(|doc| doc.id == "alpha").unwrap();
                assert_eq!(garden.content, "garden flowers");
                assert!(garden.title.is_none());
                assert_eq!(garden.metadata.get("kind").unwrap(), "guide");
                let retry = source.iter().find(|doc| doc.id == "zeta").unwrap();
                assert_eq!(retry.content, "retry network requests\nexact source");
                assert_eq!(
                    retry.title.as_deref(),
                    Some("Retries: λ and \"quoted\" text")
                );
                assert_eq!(retry.metadata.get("version").unwrap(), "original");
                assert_eq!(
                    retry.metadata.get("quoted").unwrap(),
                    "line one\n\"line two\" – λ"
                );
                assert_eq!(provider.calls.load(Ordering::Relaxed), before);
                let error = retained_rebuild_documents(&cx, &retained, 1).unwrap_err();
                assert!(error.to_string().contains("stored-payload-limit"));

                // Exercise the real CLI rebuild entry point. Its undersized
                // retained payload budget must win over the missing model
                // directory and must not create a build or publication.
                let mut refused = parse(&retained_rebuild_arguments()).unwrap().unwrap();
                refused.from_receipt = Some(first_path.clone());
                let refused_work = parent.join("budget-refused-work");
                let refused_receipt = parent.join("budget-refused.json");
                refused.work = Some(refused_work.clone());
                refused.receipt = refused_receipt.clone();
                refused.models = Some(parent.join("absent-models"));
                refused.max_source_bytes = 1;
                let mut output = Vec::new();
                let error = rebuild(&cx, &refused, &mut output, None).await.unwrap_err();
                assert!(error.to_string().contains("stored-payload-limit"));
                assert!(output.is_empty());
                assert!(!refused_work.exists());
                assert!(!refused_receipt.exists());
                assert_eq!(
                    fs::read(root_path.join("AUTHORITY")).unwrap(),
                    authority_before
                );
                assert_eq!(provider.calls.load(Ordering::Relaxed), before);

                let migrated = Arc::new(Provider {
                    identity: {
                        let mut identity = provider_identity();
                        identity.producer.implementation_revision =
                            "retained-source-migration".to_owned();
                        identity
                    },
                    calls: AtomicUsize::new(0),
                });
                assert!(
                    rebuild_identity(&selected.identity, migrated.as_ref(), false, true).is_err()
                );
                assert_eq!(migrated.calls.load(Ordering::Relaxed), 0);
                let admitted =
                    rebuild_identity(&selected.identity, migrated.as_ref(), true, true).unwrap();
                let options = parse(&retained_rebuild_arguments()).unwrap().unwrap();
                let source_cut = options
                    .rebuild_source_cut(&retained.metadata().commit_range)
                    .unwrap();
                assert_eq!(source_cut, CommitRange { low: 2, high: 7 });
                let migrated_model: Arc<dyn Embedder> = migrated.clone();
                let successor = NativeIndexBuilder::new(
                    parent.join("migrated-work"),
                    new_generation(2).unwrap(),
                    migrated_model,
                )
                .unwrap()
                .with_fast_storage(
                    NativeBuildPrecision::F32,
                    NativeBuildRetrieval::Hnsw {
                        params: HnswParams::default(),
                        seed: 19,
                    },
                )
                .add_documents(source)
                .build_exact_generation(&cx, tier, source_cut, DEFAULT_IMAGE_BYTES)
                .await
                .unwrap();
                assert_eq!(successor.binding().frozen_identity(), &admitted);
                assert_eq!(migrated.calls.load(Ordering::Relaxed), 2);
                assert_eq!(provider.calls.load(Ordering::Relaxed), before);

                // A second captured rebuild is valid in isolation but loses
                // publication after the migrated producer's successor wins.
                let stale_model: Arc<dyn Embedder> = provider.clone();
                let loser = NativeIndexBuilder::new(
                    parent.join("stale-candidate-work"),
                    new_generation(2).unwrap(),
                    stale_model,
                )
                .unwrap()
                .with_fast_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Exact)
                .add_documents(
                    retained_rebuild_documents(&cx, &retained, crate::MAX_INPUT_BYTES).unwrap(),
                )
                .build_exact_generation(
                    &cx,
                    tier,
                    CommitRange { low: 2, high: 7 },
                    DEFAULT_IMAGE_BYTES,
                )
                .await
                .unwrap();
                let committed = stage_and_publish(
                    &cx,
                    &root,
                    &root_path,
                    &parent.join("migrated.json"),
                    &successor,
                    Some(&retained),
                    PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
                )
                .unwrap()
                .receipt;
                assert_eq!(committed.root_id, selected.root_id);
                assert_eq!(committed.tier, tier);
                assert_eq!(committed.documents, 2);
                assert_eq!(committed.identity, admitted);
                let authority_after = fs::read(root_path.join("AUTHORITY")).unwrap();
                let error = stage_and_publish(
                    &cx,
                    &root,
                    &root_path,
                    &parent.join("stale-candidate.json"),
                    &loser,
                    Some(&retained),
                    PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
                )
                .err()
                .expect("the competing publication must lose");
                assert_eq!(
                    publication_failure(error.as_ref()).unwrap()["publication"],
                    "not_committed"
                );
                assert_eq!(
                    fs::read(root_path.join("AUTHORITY")).unwrap(),
                    authority_after
                );

                let active = committed
                    .open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES)
                    .unwrap();
                assert_eq!(
                    active.metadata().commit_range,
                    retained.metadata().commit_range
                );
                assert_eq!(active.retrieval_mode(), NativeRetrievalMode::Ann);
                let current_source =
                    retained_rebuild_documents(&cx, &active, crate::MAX_INPUT_BYTES).unwrap();
                let old_source =
                    retained_rebuild_documents(&cx, &retained, crate::MAX_INPUT_BYTES).unwrap();
                assert_eq!(current_source.len(), old_source.len());
                assert_eq!(current_source.len(), 2);
                for (current, old) in current_source.iter().zip(&old_source) {
                    assert_eq!(current.id, old.id);
                    assert_eq!(current.content, old.content);
                    assert_eq!(current.title, old.title);
                    assert_eq!(current.metadata, old.metadata);
                }
                let old_calls = provider.calls.load(Ordering::Relaxed);
                assert!(
                    active
                        .search_text(&cx, provider.as_ref(), "retry", 1)
                        .await
                        .is_err()
                );
                assert_eq!(provider.calls.load(Ordering::Relaxed), old_calls);
                assert_eq!(
                    active
                        .search_text(&cx, migrated.as_ref(), "retry", 1)
                        .await
                        .unwrap()[0]
                        .doc_id,
                    "zeta"
                );
                assert_eq!(
                    retained
                        .search_text(&cx, provider.as_ref(), "retry", 1)
                        .await
                        .unwrap()[0]
                        .doc_id,
                    "zeta"
                );

                refused.max_source_bytes = crate::MAX_INPUT_BYTES;
                let error = rebuild(&cx, &refused, &mut Vec::new(), None)
                    .await
                    .unwrap_err();
                assert!(error.to_string().contains("head differs"), "{error}");
                assert!(!refused_work.exists());
                assert!(!refused_receipt.exists());
                assert_eq!(
                    fs::read(root_path.join("AUTHORITY")).unwrap(),
                    authority_after
                );
            }
        });
    }

    fn update_message(
        parent: &Path,
        label: &str,
        previous: &Receipt,
        source_high: u64,
        changes: &serde_json::Value,
    ) -> serde_json::Value {
        serde_json::json!({
            "op": "update", "id": label, "expected_authority": previous.authority,
            "work_dir": parent.join(format!("{label}-work")),
            "new_receipt": parent.join(format!("{label}.json")),
            "source_low": 1, "source_high": source_high, "changes": changes,
        })
    }

    #[test]
    fn warm_authority_updates_reuse_the_model_and_install_complete_successors() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (label, tier, ann) in [
                ("update-fast-exact", EmbedderTierTag::Fast, false),
                ("update-quality-exact", EmbedderTierTag::Quality, false),
                ("update-fast-ann", EmbedderTierTag::Fast, true),
                ("update-quality-ann", EmbedderTierTag::Quality, true),
            ] {
                let parent = private_parent(label);
                let provider = provider();
                let old = publish(&cx, &parent, None, &provider, tier, "old", ann).await;
                let retained = old.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                let original_receipt = fs::read(parent.join("old.json")).unwrap();
                let mut session = open_session(&cx, old.clone(), &provider, false)
                    .with_filter(&cx, None, 1)
                    .unwrap()
                    .with_updates(2, crate::MAX_INPUT_BYTES)
                    .unwrap();
                let metadata = update_message(
                    &parent,
                    "metadata",
                    &old,
                    5,
                    &serde_json::json!([
                        {"op":"upsert", "id":"retry-old", "content":"retry network requests",
                         "title":"Unique refreshed title", "metadata":{"version":"new"}},
                        {"op":"upsert", "id":"garden-old", "content":"garden flowers",
                         "metadata":{"version":"old"}},
                        {"op":"delete", "id":"absent"}
                    ]),
                );
                let before = provider.calls.load(Ordering::Relaxed);
                let output = exchange(
                    &cx,
                    &mut session,
                    &[
                        serde_json::json!({"id":"before", "query":"retry"}),
                        metadata,
                        serde_json::json!({"id":"after", "query":"Unique refreshed title", "limit":2}),
                    ],
                )
                .await;
                assert_eq!(output[0]["updates_enabled"], true);
                assert_eq!(output[0]["activation_enabled"], false);
                assert!(
                    output.iter().all(|frame| frame.get("scope").is_none()),
                    "unfiltered updates must not export a new query scope against its tiny source budget"
                );
                assert_eq!(output[1]["results"][0]["doc_id"], "retry-old");
                assert_eq!(output[2]["event"], "updated");
                assert_eq!(output[2]["publication"], "committed");
                assert_eq!(output[2]["serving"], "activated");
                assert_eq!(output[2]["documents"], 2);
                assert_eq!(output[2]["edited_ids"], 3);
                assert_eq!(
                    output[2]["previous_authority"],
                    serde_json::json!(old.authority)
                );
                assert_eq!(
                    output[2]["receipt"],
                    serde_json::json!(parent.join("metadata.json"))
                );
                assert_eq!(
                    provider.calls.load(Ordering::Relaxed),
                    before + 2,
                    "only the two queries infer; metadata and no-op document edits reuse vectors"
                );
                let refreshed = output[3]["results"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|hit| hit["doc_id"] == "retry-old")
                    .unwrap();
                assert!(
                    refreshed["lexical_score"]
                        .as_f64()
                        .is_some_and(|score| score > 0.0)
                );
                let second = Receipt::read(&parent.join("metadata.json")).unwrap();
                assert_eq!(session.receipt.authority, second.authority);
                assert_eq!(session.receipt.identity, old.identity);
                assert_eq!(session.snapshot.authority(), second.authority().unwrap());
                let source = session
                    .snapshot
                    .source_documents(&cx, crate::MAX_INPUT_BYTES)
                    .unwrap();
                let retry = source
                    .iter()
                    .find(|document| document.id == "retry-old")
                    .unwrap();
                assert_eq!(retry.title.as_deref(), Some("Unique refreshed title"));
                assert_eq!(retry.metadata.get("version").unwrap(), "new");

                let mixed = update_message(
                    &parent,
                    "mixed",
                    &second,
                    6,
                    &serde_json::json!([
                        {"op":"upsert", "id":"garden-old", "content":"discarded input"},
                        {"op":"upsert", "id":"fresh", "content":"discarded new input"},
                        {"op":"delete", "id":"fresh"},
                        {"op":"delete", "id":"retry-old"},
                        {"op":"upsert", "id":"garden-old", "content":"retry revised garden"},
                        {"op":"upsert", "id":"fresh", "content":"garden fresh flowers"}
                    ]),
                );
                let before = provider.calls.load(Ordering::Relaxed);
                let output = exchange(
                    &cx,
                    &mut session,
                    &[
                        mixed,
                        serde_json::json!({"query":"retry revised", "limit":2}),
                    ],
                )
                .await;
                assert_eq!(output[1]["event"], "updated");
                assert_eq!(output[1]["publication"], "committed");
                assert_eq!(output[1]["edited_ids"], 3);
                assert_eq!(
                    provider.calls.load(Ordering::Relaxed),
                    before + 3,
                    "only two final document inputs and the following query infer"
                );
                assert_eq!(
                    output[2]["results"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|hit| hit["doc_id"].as_str().unwrap())
                        .collect::<BTreeSet<_>>(),
                    BTreeSet::from(["fresh", "garden-old"])
                );
                assert_eq!(session.snapshot.tier(), tier);
                assert_eq!(session.snapshot.retrieval_mode(), retained.retrieval_mode());
                assert_eq!(session.receipt.identity, old.identity);
                let third = Receipt::read(&parent.join("mixed.json")).unwrap();
                let reopened = third.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(reopened.authority(), session.snapshot.authority());
                assert_eq!(
                    reopened.metadata().commit_range,
                    CommitRange { low: 1, high: 6 }
                );
                assert_eq!(
                    reopened
                        .source_documents(&cx, crate::MAX_INPUT_BYTES)
                        .unwrap()
                        .iter()
                        .map(|document| document.id.as_str())
                        .collect::<BTreeSet<_>>(),
                    BTreeSet::from(["fresh", "garden-old"])
                );
                let old_source = retained
                    .source_documents(&cx, crate::MAX_INPUT_BYTES)
                    .unwrap();
                assert_eq!(
                    old_source
                        .iter()
                        .map(|document| document.id.as_str())
                        .collect::<BTreeSet<_>>(),
                    BTreeSet::from(["garden-old", "retry-old"])
                );
                assert_eq!(
                    old_source
                        .iter()
                        .find(|document| document.id == "retry-old")
                        .unwrap()
                        .metadata
                        .get("version")
                        .unwrap(),
                    "old"
                );
                assert_eq!(fs::read(parent.join("old.json")).unwrap(), original_receipt);
            }
        });
    }

    #[test]
    fn warm_updates_refresh_fixed_filter_membership_without_widening_retained_scopes() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (label, tier, ann) in [
                ("warm-scoped-fast-exact", EmbedderTierTag::Fast, false),
                ("warm-scoped-quality-hnsw", EmbedderTierTag::Quality, true),
            ] {
                let parent = private_parent(label);
                let provider = provider();
                let old = publish(&cx, &parent, None, &provider, tier, "old", ann).await;
                let fixed =
                    || crate::filter::Filter::parse(r#"{"metadata":{"version":"old"}}"#).unwrap();
                let mut session = open_session(&cx, old.clone(), &provider, false)
                    .with_filter(&cx, Some(fixed()), crate::MAX_INPUT_BYTES)
                    .unwrap()
                    .with_updates(2, crate::MAX_INPUT_BYTES)
                    .unwrap();
                let mut retained = open_session(&cx, old.clone(), &provider, false)
                    .with_filter(&cx, Some(fixed()), crate::MAX_INPUT_BYTES)
                    .unwrap();
                let request = update_message(
                    &parent,
                    "scoped-next",
                    &old,
                    6,
                    &serde_json::json!([
                        {"op":"upsert", "id":"retry-old", "content":"retry network requests",
                         "metadata":{"version":"new"}},
                        {"op":"delete", "id":"garden-old"},
                        {"op":"upsert", "id":"garden-new", "content":"garden new flowers",
                         "metadata":{"version":"old"}},
                        {"op":"upsert", "id":"retry-new", "content":"retry outside the fixed filter",
                         "metadata":{"version":"new"}}
                    ]),
                );
                let before = provider.calls.load(Ordering::Relaxed);
                let output = exchange(&cx, &mut session, &[
                    request,
                    serde_json::json!({"query":"retry", "limit":3, "filter":{}}),
                    serde_json::json!({"query":"retry", "filter":{"ids":["retry-old"]}}),
                    serde_json::json!({"query":"retry", "filter":{"metadata":{"version":"new"}}}),
                ]).await;
                assert_eq!(output[0]["scope"]["eligible_documents"], 2);
                assert_eq!(output[1]["event"], "updated");
                assert_eq!(output[1]["publication"], "committed");
                assert_eq!(output[1]["serving"], "activated");
                assert_eq!(output[1]["documents"], 3);
                assert_eq!(output[1]["scope"]["eligible_documents"], 1);
                assert_eq!(output[2]["event"], "results");
                assert_eq!(output[2]["scope"]["eligible_documents"], 1);
                assert_eq!(output[2]["results"].as_array().unwrap().len(), 1);
                assert_eq!(output[2]["results"][0]["doc_id"], "garden-new");
                for position in [3, 4] {
                    assert_eq!(output[position]["event"], "results");
                    assert_eq!(output[position]["scope"]["eligible_documents"], 0);
                    assert!(output[position]["results"].as_array().unwrap().is_empty());
                }
                assert_eq!(
                    provider.calls.load(Ordering::Relaxed),
                    before + 3,
                    "only two new document inputs and one nonempty scoped query infer"
                );
                let next = Receipt::read(&parent.join("scoped-next.json")).unwrap();
                assert_eq!(session.receipt.authority, next.authority);
                assert_eq!(output[2]["authority"], serde_json::json!(next.authority));
                assert_eq!(
                    next.open(&cx, DEFAULT_IMAGE_BYTES)
                        .unwrap()
                        .source_documents(&cx, crate::MAX_INPUT_BYTES)
                        .unwrap()
                        .iter()
                        .map(|document| document.id.as_str())
                        .collect::<BTreeSet<_>>(),
                    BTreeSet::from(["garden-new", "retry-new", "retry-old"])
                );
                let empty = update_message(
                    &parent,
                    "scope-empty",
                    &next,
                    7,
                    &serde_json::json!([
                        {"op":"upsert", "id":"garden-new", "content":"garden new flowers",
                         "metadata":{"version":"new"}}
                    ]),
                );
                let output = exchange(
                    &cx,
                    &mut session,
                    &[empty, serde_json::json!({"query":"retry", "limit":3})],
                )
                .await;
                assert_eq!(output[1]["event"], "updated");
                assert_eq!(output[1]["scope"]["eligible_documents"], 0);
                assert_eq!(output[2]["scope"]["eligible_documents"], 0);
                assert!(output[2]["results"].as_array().unwrap().is_empty());
                assert_eq!(
                    provider.calls.load(Ordering::Relaxed),
                    before + 3,
                    "an empty fixed scope remains a restriction and skips query inference"
                );
                let empty = Receipt::read(&parent.join("scope-empty.json")).unwrap();
                let refill = update_message(
                    &parent,
                    "scope-refill",
                    &empty,
                    8,
                    &serde_json::json!([
                        {"op":"upsert", "id":"retry-old", "content":"retry network requests",
                         "metadata":{"version":"old"}}
                    ]),
                );
                let output = exchange(
                    &cx,
                    &mut session,
                    &[
                        refill,
                        serde_json::json!({"query":"retry", "limit":3}),
                        serde_json::json!({"query":"retry", "filter":{"ids":["retry-old"]}}),
                    ],
                )
                .await;
                assert_eq!(output[1]["event"], "updated");
                assert_eq!(output[1]["scope"]["eligible_documents"], 1);
                for position in [2, 3] {
                    assert_eq!(output[position]["scope"]["eligible_documents"], 1);
                    assert_eq!(output[position]["results"].as_array().unwrap().len(), 1);
                    assert_eq!(output[position]["results"][0]["doc_id"], "retry-old");
                }
                assert_eq!(
                    provider.calls.load(Ordering::Relaxed),
                    before + 5,
                    "metadata-only membership transitions reuse the existing document vectors"
                );
                let held = exchange(
                    &cx,
                    &mut retained,
                    &[serde_json::json!({"query":"retry", "limit":3, "filter":{}})],
                )
                .await;
                assert_eq!(held[1]["authority"], serde_json::json!(old.authority));
                assert_eq!(held[1]["scope"]["eligible_documents"], 2);
                assert_eq!(held[1]["results"][0]["doc_id"], "retry-old");
                assert_eq!(
                    held[1]["results"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|hit| hit["doc_id"].as_str().unwrap())
                        .collect::<BTreeSet<_>>(),
                    BTreeSet::from(["garden-old", "retry-old"])
                );
                assert_eq!(provider.calls.load(Ordering::Relaxed), before + 6);
            }
        });
    }

    #[test]
    fn warm_update_scope_budget_refusal_precedes_publication_and_preserves_membership() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let parent = private_parent("warm-scoped-update-budget");
            let provider = provider();
            let old = publish(
                &cx,
                &parent,
                None,
                &provider,
                EmbedderTierTag::Quality,
                "old",
                true,
            )
            .await;
            let filter = crate::filter::Filter::parse(r#"{"metadata":{"version":"old"}}"#).unwrap();
            let mut session = open_session(&cx, old.clone(), &provider, false)
                .with_filter(&cx, Some(filter), 256)
                .unwrap()
                .with_updates(2, crate::MAX_INPUT_BYTES)
                .unwrap();
            let head = fs::read(old.root.join("AUTHORITY")).unwrap();
            let request = update_message(
                &parent,
                "too-large-scope",
                &old,
                6,
                &serde_json::json!([
                    {"op":"upsert", "id":"retry-old", "content":"retry network requests",
                     "metadata":{"version":"old", "note":"x".repeat(1_024)}}
                ]),
            );
            let before = provider.calls.load(Ordering::Relaxed);
            let output = exchange(
                &cx,
                &mut session,
                &[
                    request,
                    serde_json::json!({"query":"retry", "limit":2, "filter":{}}),
                ],
            )
            .await;
            assert_eq!(output[1]["event"], "failed");
            assert_eq!(output[1]["publication"], "not_committed");
            assert_eq!(output[1]["serving"], "not_activated");
            assert_eq!(output[1]["authority"], serde_json::json!(old.authority));
            assert_eq!(output[1]["scope"]["eligible_documents"], 2);
            assert_eq!(session.receipt.authority, old.authority);
            assert_eq!(fs::read(old.root.join("AUTHORITY")).unwrap(), head);
            assert!(!parent.join("too-large-scope.json").exists());
            assert_eq!(output[2]["event"], "results");
            assert_eq!(output[2]["scope"]["eligible_documents"], 2);
            assert_eq!(output[2]["results"][0]["doc_id"], "retry-old");
            assert_eq!(
                provider.calls.load(Ordering::Relaxed),
                before + 1,
                "a metadata-only update reuses document vectors even when its final scope is refused"
            );
            assert!(
                !session
                    .snapshot
                    .source_documents(&cx, crate::MAX_INPUT_BYTES)
                    .unwrap()
                    .iter()
                    .find(|document| document.id == "retry-old")
                    .unwrap()
                    .metadata
                    .contains_key("note")
            );
        });
    }

    #[test]
    fn warm_update_refusals_preserve_the_session_and_reject_before_inference() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let parent = private_parent("warm-update-refusals");
            let provider = provider();
            let old = publish(
                &cx,
                &parent,
                None,
                &provider,
                EmbedderTierTag::Fast,
                "old",
                true,
            )
            .await;
            let root_path = parent.join("root");
            let original_authority = fs::read(root_path.join("AUTHORITY")).unwrap();
            let before = provider.calls.load(Ordering::Relaxed);
            let changes = serde_json::json!([
                {"op":"upsert", "id":"retry-old", "content":"retry new content"}
            ]);
            let mut disabled = open_session(&cx, old.clone(), &provider, false);
            let request = update_message(&parent, "disabled", &old, 6, &changes);
            let output = exchange(&cx, &mut disabled, &[request]).await;
            assert_eq!(output[0]["updates_enabled"], false);
            assert_eq!(output[1]["event"], "failed");
            assert_eq!(output[1]["code"], "updates_disabled");
            assert!(!parent.join("disabled-work").exists());
            assert!(!parent.join("disabled.json").exists());
            assert_eq!(provider.calls.load(Ordering::Relaxed), before);
            let mut session = open_session(&cx, old.clone(), &provider, false)
                .with_updates(2, crate::MAX_INPUT_BYTES)
                .unwrap();
            let mut stale = update_message(&parent, "stale", &old, 6, &changes);
            stale["expected_authority"] = serde_json::json!(receipt().authority);
            let mut backward = update_message(&parent, "backward", &old, 6, &changes);
            backward["source_low"] = serde_json::json!(0);
            let mut inverted = update_message(&parent, "inverted", &old, 8, &changes);
            inverted["source_low"] = serde_json::json!(9);
            let mut overflow = update_message(&parent, "overflow", &old, u64::MAX, &changes);
            overflow["source_low"] = serde_json::json!(0);
            let mut inside = update_message(&parent, "inside", &old, 6, &changes);
            inside["work_dir"] = serde_json::json!(root_path.join("forbidden-work"));
            let mut relative = update_message(&parent, "relative", &old, 6, &changes);
            relative["work_dir"] = serde_json::json!("relative-work");
            let invalid_delete = update_message(
                &parent,
                "invalid-delete",
                &old,
                6,
                &serde_json::json!([
                    {"op":"delete", "id":" "}
                ]),
            );
            let too_many = update_message(
                &parent,
                "too-many",
                &old,
                6,
                &serde_json::Value::Array(vec![
                    serde_json::json!({"op":"delete", "id":"absent"});
                    1_001
                ]),
            );
            for (label, request, code) in [
                ("stale", stale, "stale_expected_authority"),
                ("backward", backward, "backward_source_checkpoint"),
                ("inverted", inverted, "invalid_source_checkpoint"),
                ("overflow", overflow, "invalid_source_checkpoint"),
                ("inside", inside, "update_refused"),
                ("relative", relative, "invalid_update_request"),
                ("invalid-delete", invalid_delete, "invalid_update_mutations"),
                ("too-many", too_many, "update_mutation_limit"),
            ] {
                let output = exchange(&cx, &mut session, &[request]).await;
                assert_eq!(output[1]["event"], "failed", "{label}");
                assert_eq!(output[1]["code"], code, "{label}");
                assert_eq!(output[1]["publication"], "not_committed", "{label}");
                assert_eq!(output[1]["serving"], "not_activated", "{label}");
                assert_eq!(session.receipt.authority, old.authority, "{label}");
                assert_eq!(provider.calls.load(Ordering::Relaxed), before, "{label}");
                assert!(!parent.join(format!("{label}-work")).exists(), "{label}");
                assert!(!parent.join(format!("{label}.json")).exists(), "{label}");
                assert_eq!(
                    fs::read(root_path.join("AUTHORITY")).unwrap(),
                    original_authority
                );
            }
            assert!(!root_path.join("forbidden-work").exists());
            let invalid = update_message(
                &parent,
                "invalid-overwritten",
                &old,
                6,
                &serde_json::json!([
                    {"op":"upsert", "id":"retry-old", "content":7},
                    {"op":"upsert", "id":"retry-old", "content":"valid final replacement"}
                ]),
            );
            let output = exchange(&cx, &mut session, &[invalid]).await;
            assert_eq!(output[1]["event"], "failed");
            assert_eq!(provider.calls.load(Ordering::Relaxed), before);
            assert!(!parent.join("invalid-overwritten-work").exists());
            assert!(!parent.join("invalid-overwritten.json").exists());
            let mut bounded = open_session(&cx, old.clone(), &provider, false)
                .with_updates(2, 1)
                .unwrap();
            let request = update_message(&parent, "bounded", &old, 6, &changes);
            let output = exchange(&cx, &mut bounded, &[request]).await;
            assert_eq!(output[1]["event"], "failed");
            assert_eq!(output[1]["publication"], "not_committed");
            assert_eq!(provider.calls.load(Ordering::Relaxed), before);
            assert!(!parent.join("bounded-work").exists());
            assert!(!parent.join("bounded.json").exists());

            // A matching in-memory expectation is insufficient when another
            // process has already published a different physical-root head.
            let winner = publish(
                &cx,
                &parent,
                Some(&session.snapshot),
                &provider,
                EmbedderTierTag::Fast,
                "external",
                true,
            )
            .await;
            let head = fs::read(root_path.join("AUTHORITY")).unwrap();
            let before = provider.calls.load(Ordering::Relaxed);
            let request = update_message(&parent, "external-stale", &old, 6, &changes);
            let output = exchange(&cx, &mut session, &[request]).await;
            assert_eq!(output[1]["event"], "failed");
            assert_eq!(output[1]["publication"], "not_committed");
            assert_eq!(provider.calls.load(Ordering::Relaxed), before);
            assert_eq!(session.receipt.authority, old.authority);
            assert_eq!(fs::read(root_path.join("AUTHORITY")).unwrap(), head);
            assert!(!parent.join("external-stale-work").exists());
            assert!(!parent.join("external-stale.json").exists());
            assert_eq!(
                winner.open(&cx, DEFAULT_IMAGE_BYTES).unwrap().authority(),
                winner.authority().unwrap()
            );
            let output = exchange(&cx, &mut session, &[serde_json::json!({"query":"retry"})]).await;
            assert_eq!(output[1]["results"][0]["doc_id"], "retry-old");
        });
    }

    #[test]
    fn warm_update_losing_publication_keeps_the_original_reader_and_recovery_receipt() {
        use std::future::Future as _;
        use std::sync::atomic::AtomicBool;
        use std::task::Poll;

        struct GatedProvider {
            inner: Arc<Provider>,
            started: AtomicBool,
            released: AtomicBool,
        }
        impl Embedder for GatedProvider {
            fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
                self.inner.identity()
            }
            fn embed<'a>(&'a self, cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
                self.started.store(true, Ordering::Relaxed);
                let work = self.inner.embed(cx, text);
                Box::pin(async move {
                    std::future::poll_fn(|_| {
                        if self.released.load(Ordering::Relaxed) {
                            Poll::Ready(())
                        } else {
                            Poll::Pending
                        }
                    })
                    .await;
                    work.await
                })
            }
            fn dimension(&self) -> usize {
                self.inner.dimension()
            }
            fn id(&self) -> &str {
                self.inner.id()
            }
            fn model_name(&self) -> &str {
                self.inner.model_name()
            }
            fn is_semantic(&self) -> bool {
                self.inner.is_semantic()
            }
            fn category(&self) -> ModelCategory {
                self.inner.category()
            }
        }

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let parent = private_parent("warm-update-publication-race");
            let provider = provider();
            let old = publish(
                &cx,
                &parent,
                None,
                &provider,
                EmbedderTierTag::Quality,
                "old",
                true,
            )
            .await;
            let root_path = parent.join("root");
            let root = admit_root(&root_path).unwrap();
            let retained = old.open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES).unwrap();
            let model: Arc<dyn Embedder> = provider.clone();
            let competitor = NativeIndexBuilder::new(
                parent.join("winner-work"),
                new_generation(2).unwrap(),
                model,
            )
            .unwrap()
            .with_fast_storage(
                NativeBuildPrecision::F32,
                retained.retrieval_configuration(),
            )
            .add_documents([IndexableDocument::new(
                "external",
                "retry concurrent publication",
            )])
            .build_exact_generation(
                &cx,
                EmbedderTierTag::Quality,
                CommitRange { low: 1, high: 6 },
                DEFAULT_IMAGE_BYTES,
            )
            .await
            .unwrap();
            let gated = Arc::new(GatedProvider {
                inner: provider.clone(),
                started: AtomicBool::new(false),
                released: AtomicBool::new(false),
            });
            let mut session = open_session(&cx, old.clone(), &provider, false)
                .with_updates(2, crate::MAX_INPUT_BYTES)
                .unwrap();
            session.models = Models {
                fast: gated.clone(),
                quality: None,
            }
            .into();
            let request = update_message(
                &parent,
                "loser",
                &old,
                7,
                &serde_json::json!([
                    {"op":"upsert", "id":"loser", "content":"retry losing change"}
                ]),
            );
            let mut input = serde_json::to_vec(&request).unwrap();
            input.extend_from_slice(b"\n{\"query\":\"retry\",\"limit\":2}\n");
            let mut input = Cursor::new(input);
            let mut output = Vec::new();
            let before = provider.calls.load(Ordering::Relaxed);
            let mut serving = Box::pin(serve_session(&cx, &mut session, &mut input, &mut output));
            std::future::poll_fn(|task| match serving.as_mut().poll(task) {
                Poll::Ready(result) => panic!("update must wait in its provider: {result:?}"),
                Poll::Pending if gated.started.load(Ordering::Relaxed) => Poll::Ready(()),
                Poll::Pending => Poll::Pending,
            })
            .await;
            assert_eq!(provider.calls.load(Ordering::Relaxed), before + 1);
            let winner = stage_and_publish(
                &cx,
                &root,
                &root_path,
                &parent.join("winner.json"),
                &competitor,
                Some(&retained),
                PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
            )
            .unwrap()
            .receipt;
            let winning_head = fs::read(root_path.join("AUTHORITY")).unwrap();
            gated.released.store(true, Ordering::Relaxed);
            serving.await.unwrap();
            let output = frames(&output);
            assert_eq!(output.len(), 3);
            assert_eq!(output[1]["event"], "failed");
            assert_eq!(output[1]["publication"], "not_committed");
            assert_eq!(output[1]["serving"], "not_activated");
            assert_eq!(output[1]["authority"], serde_json::json!(old.authority));
            assert_eq!(
                output[1]["receipt"],
                serde_json::json!(parent.join("loser.json"))
            );
            assert_eq!(output[2]["event"], "results");
            assert_eq!(session.receipt.authority, old.authority);
            assert_eq!(provider.calls.load(Ordering::Relaxed), before + 2);
            assert_eq!(
                output[2]["results"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|hit| hit["doc_id"].as_str().unwrap())
                    .collect::<BTreeSet<_>>(),
                BTreeSet::from(["garden-old", "retry-old"])
            );
            assert_eq!(fs::read(root_path.join("AUTHORITY")).unwrap(), winning_head);
            let losing = Receipt::read(&parent.join("loser.json")).unwrap();
            assert_eq!(
                losing.recovery_attempt().unwrap().expected,
                retained.retained_snapshot().authority_pair()
            );
            assert_ne!(losing.authority().unwrap(), winner.authority().unwrap());
            assert_eq!(
                winner
                    .open(&cx, DEFAULT_IMAGE_BYTES)
                    .unwrap()
                    .source_documents(&cx, crate::MAX_INPUT_BYTES)
                    .unwrap()[0]
                    .id,
                "external"
            );
        });
    }

    #[test]
    fn committed_warm_updates_preserve_their_effect_when_delivery_fails_or_is_cancelled() {
        struct CancelAtUpdate<'a> {
            cx: &'a Cx,
            fail: bool,
            bytes: Vec<u8>,
            attempted_update: Option<serde_json::Value>,
        }
        impl Write for CancelAtUpdate<'_> {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                let frame: serde_json::Value = serde_json::from_slice(bytes).unwrap();
                if frame["event"] == "updated" {
                    self.attempted_update = Some(frame);
                    self.cx.set_cancel_requested(true);
                    if self.fail {
                        return Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed"));
                    }
                }
                self.bytes.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for fail in [false, true] {
                let parent = private_parent(if fail {
                    "warm-delivery-failure"
                } else {
                    "warm-delivery-cancel"
                });
                let provider = provider();
                let old = publish(
                    &cx,
                    &parent,
                    None,
                    &provider,
                    EmbedderTierTag::Fast,
                    "old",
                    false,
                )
                .await;
                let retained = old.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                let mut session = open_session(&cx, old.clone(), &provider, false)
                    .with_updates(2, crate::MAX_INPUT_BYTES)
                    .unwrap();
                let request = update_message(
                    &parent,
                    "committed",
                    &old,
                    6,
                    &serde_json::json!([
                        {"op":"upsert", "id":"retry-old", "content":"retry committed replacement"}
                    ]),
                );
                let mut input = serde_json::to_vec(&request).unwrap();
                input.extend_from_slice(b"\n{\"query\":\"retry\"}\n");
                let mut input = Cursor::new(input);
                let mut output = CancelAtUpdate {
                    cx: &cx,
                    fail,
                    bytes: Vec::new(),
                    attempted_update: None,
                };
                let before = provider.calls.load(Ordering::Relaxed);
                let error = serve_session(&cx, &mut session, &mut input, &mut output)
                    .await
                    .unwrap_err();
                let acknowledgement = output.attempted_update.as_ref().unwrap();
                assert_eq!(acknowledgement["publication"], "committed");
                assert_eq!(acknowledgement["serving"], "activated");
                assert_eq!(
                    acknowledgement["authority"],
                    serde_json::json!(session.receipt.authority)
                );
                assert_ne!(session.receipt.authority, old.authority);
                assert_eq!(
                    provider.calls.load(Ordering::Relaxed),
                    before + 1,
                    "delivery failure or cancellation stops before the next query"
                );
                let delivered = frames(&output.bytes);
                if fail {
                    let effect = publication_failure(error.as_ref()).unwrap();
                    assert_eq!(effect["publication"], "committed");
                    assert_eq!(effect["serving"], "activated");
                    assert_eq!(effect["reason"], "result_delivery_failed");
                    assert_eq!(
                        effect["authority"],
                        serde_json::json!(session.receipt.authority)
                    );
                    assert_eq!(
                        effect["receipt"],
                        serde_json::json!(parent.join("committed.json"))
                    );
                    assert_eq!(delivered.len(), 1);
                } else {
                    assert!(publication_failure(error.as_ref()).is_none());
                    assert_eq!(delivered.len(), 2);
                    assert_eq!(delivered[1], *acknowledgement);
                }
                assert_eq!(delivered[0]["event"], "ready");
                // The writer sets cancellation at delivery, after installation.
                // Clear it only to independently reopen the committed receipt.
                cx.set_cancel_requested(false);
                let committed = Receipt::read(&parent.join("committed.json")).unwrap();
                let reopened = committed.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(reopened.authority(), session.snapshot.authority());
                let source = reopened
                    .source_documents(&cx, crate::MAX_INPUT_BYTES)
                    .unwrap();
                assert_eq!(
                    source
                        .iter()
                        .find(|document| document.id == "retry-old")
                        .unwrap()
                        .content,
                    "retry committed replacement"
                );
                let old_source = retained
                    .source_documents(&cx, crate::MAX_INPUT_BYTES)
                    .unwrap();
                assert_eq!(
                    old_source
                        .iter()
                        .find(|document| document.id == "retry-old")
                        .unwrap()
                        .content,
                    "retry network requests"
                );
            }
        });
    }

    #[test]
    fn warm_updates_after_rollback_follow_the_restored_source_and_current_authority() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let parent = private_parent("warm-update-after-rollback");
            let provider = provider();
            let old = publish(
                &cx,
                &parent,
                None,
                &provider,
                EmbedderTierTag::Quality,
                "old",
                true,
            )
            .await;
            let mut session = open_session(&cx, old.clone(), &provider, true)
                .with_updates(2, crate::MAX_INPUT_BYTES)
                .unwrap();
            let first = update_message(
                &parent,
                "before-rollback",
                &old,
                9,
                &serde_json::json!([
                    {"op":"delete", "id":"retry-old"},
                    {"op":"upsert", "id":"later", "content":"retry before rollback"}
                ]),
            );
            let before = provider.calls.load(Ordering::Relaxed);
            let output = exchange(&cx, &mut session, &[first]).await;
            assert_eq!(output[1]["event"], "updated");
            assert_eq!(provider.calls.load(Ordering::Relaxed), before + 1);
            let replaced = Receipt::read(&parent.join("before-rollback.json")).unwrap();
            assert_eq!(session.receipt.authority, replaced.authority);
            assert_eq!(
                session.snapshot.metadata().commit_range,
                CommitRange { low: 1, high: 9 }
            );

            let mut options = parse(&rollback_arguments()).unwrap().unwrap();
            options.from_receipt = Some(parent.join("before-rollback.json"));
            options.target_receipt = Some(parent.join("old.json"));
            options.receipt = parent.join("restored.json");
            rollback(&cx, &options, &mut Vec::new()).unwrap();
            let restored = Receipt::read(&options.receipt).unwrap();
            let restored_reader = restored.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            assert_eq!(restored.generation, old.generation);
            assert!(
                restored.authority().unwrap().sequence > replaced.authority().unwrap().sequence
            );
            assert_eq!(provider.calls.load(Ordering::Relaxed), before + 1);
            let activate = serde_json::json!({
                "op":"activate", "id":"restore", "receipt":options.receipt,
                "expected_authority":replaced.authority,
            });
            // An empty change batch can advance the restored source checkpoint.
            // Its cut may precede the rolled-back cut, but must extend the actual
            // restored source and bind the fresh, advancing authority reference.
            let checkpoint = update_message(
                &parent,
                "after-rollback",
                &restored,
                6,
                &serde_json::json!([]),
            );
            let output = exchange(
                &cx,
                &mut session,
                &[
                    activate,
                    checkpoint,
                    serde_json::json!({"query":"retry", "limit":2}),
                ],
            )
            .await;
            assert_eq!(output[1]["event"], "activated");
            assert_eq!(
                output[1]["authority"],
                serde_json::json!(restored.authority)
            );
            assert_eq!(output[2]["event"], "updated");
            assert_eq!(output[2]["publication"], "committed");
            assert_eq!(output[2]["serving"], "activated");
            assert_eq!(output[2]["edited_ids"], 0);
            assert_eq!(
                output[2]["previous_authority"],
                serde_json::json!(restored.authority)
            );
            assert_eq!(output[3]["results"][0]["doc_id"], "retry-old");
            assert_eq!(
                provider.calls.load(Ordering::Relaxed),
                before + 2,
                "rollback, activation, and an empty update reuse the admitted document vectors"
            );
            let updated = Receipt::read(&parent.join("after-rollback.json")).unwrap();
            assert_eq!(
                updated.authority().unwrap().sequence,
                restored.authority().unwrap().sequence + 1
            );
            assert_eq!(updated.generation.sequence, old.generation.sequence + 1);
            assert_eq!(updated.generation.sequence, replaced.generation.sequence);
            assert_ne!(
                updated.generation, replaced.generation,
                "artifact nonces prevent reuse after rollback"
            );
            assert_eq!(updated.identity, old.identity);
            assert_eq!(
                updated.recovery_attempt().unwrap().expected,
                restored_reader.retained_snapshot().authority_pair()
            );
            let reopened = updated.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            assert_eq!(reopened.authority(), session.snapshot.authority());
            assert_eq!(
                reopened.metadata().commit_range,
                CommitRange { low: 1, high: 6 }
            );
            assert_eq!(
                reopened
                    .source_documents(&cx, crate::MAX_INPUT_BYTES)
                    .unwrap()
                    .iter()
                    .map(|document| document.id.as_str())
                    .collect::<BTreeSet<_>>(),
                BTreeSet::from(["garden-old", "retry-old"])
            );
            assert_eq!(
                restored_reader.metadata().commit_range,
                CommitRange { low: 1, high: 5 }
            );
        });
    }

    #[test]
    fn warm_queries_reuse_the_admitted_model_and_images_for_both_tiers() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (label, tier, ann) in [
                ("fast-exact", EmbedderTierTag::Fast, false),
                ("quality-exact", EmbedderTierTag::Quality, false),
                ("fast-ann", EmbedderTierTag::Fast, true),
                ("quality-ann", EmbedderTierTag::Quality, true),
            ] {
                let parent = private_parent(label);
                let provider = provider();
                let selected = publish(&cx, &parent, None, &provider, tier, "old", ann).await;
                let mut session = open_session(&cx, selected.clone(), &provider, false);
                let before = provider.calls.load(Ordering::Relaxed);
                let output = exchange(
                    &cx,
                    &mut session,
                    &[
                        serde_json::json!({"id": "first", "query": "retry"}),
                        serde_json::json!({"id": "second", "query": "garden", "limit": 2}),
                    ],
                )
                .await;
                assert_eq!(output[0]["retrieval"], if ann { "hnsw" } else { "exact" });
                assert_eq!(output[1]["event"], "results");
                assert_eq!(output[1]["id"], "first");
                assert_eq!(output[1]["results"].as_array().unwrap().len(), 1);
                assert_eq!(output[1]["results"][0]["doc_id"], "retry-old");
                assert_eq!(output[2]["id"], "second");
                assert_eq!(output[2]["results"].as_array().unwrap().len(), 2);
                assert_eq!(output[2]["results"][0]["doc_id"], "garden-old");
                assert_eq!(provider.calls.load(Ordering::Relaxed), before + 2);

                // The session keeps the complete admitted images. Neither a
                // model reload nor a component pathname reopen is necessary.
                fs::rename(&selected.root, parent.join("relocated-root")).unwrap();
                let output = exchange(
                    &cx,
                    &mut session,
                    &[serde_json::json!({"query": "retry", "limit": 2})],
                )
                .await;
                assert_eq!(output[1]["results"][0]["doc_id"], "retry-old");
                assert_eq!(
                    output[1]["authority"],
                    serde_json::json!(selected.authority)
                );
                assert_eq!(provider.calls.load(Ordering::Relaxed), before + 3);
            }
        });
    }

    #[test]
    fn explicit_activation_is_fenced_and_failures_keep_the_warm_predecessor() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (label, tier) in [
                ("fast", EmbedderTierTag::Fast),
                ("quality", EmbedderTierTag::Quality),
            ] {
                let parent = private_parent(label);
                let provider = provider();
                let old = publish(&cx, &parent, None, &provider, tier, "old", true).await;
                let mut session = open_session(&cx, old.clone(), &provider, true);
                let mut disabled = open_session(&cx, old.clone(), &provider, false);
                let next = publish(
                    &cx,
                    &parent,
                    Some(&session.snapshot),
                    &provider,
                    tier,
                    "new",
                    false,
                )
                .await;
                let before = provider.calls.load(Ordering::Relaxed);
                let activate = serde_json::json!({
                    "op": "activate", "id": "switch", "receipt": parent.join("new.json"),
                    "expected_authority": old.authority,
                });
                let output = exchange(
                    &cx,
                    &mut disabled,
                    &[activate.clone(), serde_json::json!({"query": "retry"})],
                )
                .await;
                assert_eq!(output[1]["event"], "failed");
                assert_eq!(output[2]["results"][0]["doc_id"], "retry-old");
                assert_eq!(disabled.receipt.authority, old.authority);
                assert_eq!(provider.calls.load(Ordering::Relaxed), before + 1);

                let mut wrong = next.clone();
                wrong.tier = if tier == EmbedderTierTag::Fast {
                    EmbedderTierTag::Quality
                } else {
                    EmbedderTierTag::Fast
                };
                save_receipt(&wrong, &parent.join("wrong-tier.json")).unwrap();
                let mut wrong_producer = next.clone();
                let mut identity = wrong_producer.identity.identity.clone();
                identity.producer.implementation_revision = "different-producer".to_owned();
                wrong_producer.identity = identity.freeze().unwrap();
                save_receipt(&wrong_producer, &parent.join("wrong-producer.json")).unwrap();
                let mut stale = activate.clone();
                stale["expected_authority"] = serde_json::json!(next.authority);
                let mut wrong_tier = activate.clone();
                wrong_tier["receipt"] = serde_json::json!(parent.join("wrong-tier.json"));
                let mut wrong_identity = activate.clone();
                wrong_identity["receipt"] = serde_json::json!(parent.join("wrong-producer.json"));
                let mut missing = activate.clone();
                missing["receipt"] = serde_json::json!(parent.join("absent.json"));
                let output = exchange(
                    &cx,
                    &mut session,
                    &[
                        stale,
                        wrong_tier,
                        wrong_identity,
                        missing,
                        serde_json::json!({"query": "retry"}),
                        activate.clone(),
                        serde_json::json!({"id": "new-query", "query": "retry"}),
                        activate,
                        serde_json::json!({"query": "garden"}),
                    ],
                )
                .await;
                for frame in &output[1..5] {
                    assert_eq!(frame["event"], "failed");
                    assert_eq!(frame["authority"], serde_json::json!(old.authority));
                }
                assert_eq!(output[5]["results"][0]["doc_id"], "retry-old");
                assert_eq!(output[6]["event"], "activated");
                assert_eq!(output[6]["authority"], serde_json::json!(next.authority));
                assert_eq!(output[6]["retrieval"], "exact");
                assert_eq!(output[7]["results"][0]["doc_id"], "retry-new");
                assert_eq!(output[8]["event"], "failed");
                assert_eq!(output[9]["results"][0]["doc_id"], "garden-new");
                assert_eq!(
                    provider.calls.load(Ordering::Relaxed),
                    before + 4,
                    "only queries perform inference; rejected and accepted activations never do"
                );
                assert_eq!(session.receipt.authority, next.authority);
                assert_eq!(disabled.receipt.authority, old.authority);
            }
        });
    }

    #[test]
    fn request_failures_are_bounded_while_oversize_cancellation_and_output_end_the_session() {
        struct FailAfterReady {
            writes: usize,
        }
        impl Write for FailAfterReady {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.writes += 1;
                if self.writes > 1 {
                    return Err(io::Error::new(io::ErrorKind::BrokenPipe, "closed"));
                }
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        struct CancelOnRead<'a> {
            cx: &'a Cx,
            inner: Cursor<Vec<u8>>,
        }
        impl Read for CancelOnRead<'_> {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                self.inner.read(buffer)
            }
        }
        impl BufRead for CancelOnRead<'_> {
            fn fill_buf(&mut self) -> io::Result<&[u8]> {
                self.cx.set_cancel_requested(true);
                self.inner.fill_buf()
            }
            fn consume(&mut self, amount: usize) {
                self.inner.consume(amount);
            }
        }
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let parent = private_parent("protocol");
            let provider = provider();
            let selected = publish(
                &cx,
                &parent,
                None,
                &provider,
                EmbedderTierTag::Fast,
                "old",
                true,
            )
            .await;
            let before = provider.calls.load(Ordering::Relaxed);
            let wrong = Arc::new(Provider {
                identity: {
                    let mut identity = provider_identity();
                    identity.producer.implementation_revision = "wrong-warm-producer".to_owned();
                    identity
                },
                calls: AtomicUsize::new(0),
            });
            let root = admit_root(&selected.root).unwrap();
            let snapshot = selected
                .open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES)
                .unwrap();
            let wrong_model: Arc<dyn Embedder> = wrong.clone();
            assert!(
                AuthoritySession::new(
                    &cx,
                    root,
                    selected.clone(),
                    snapshot,
                    Models {
                        fast: wrong_model,
                        quality: None
                    },
                    (1, DEFAULT_IMAGE_BYTES),
                    false
                )
                .is_err()
            );
            assert_eq!(wrong.calls.load(Ordering::Relaxed), 0);

            let mut session = open_session(&cx, selected.clone(), &provider, false);
            let input = concat!(
                "{invalid}\n",
                "{\"query\":\"retry\",\"limit\":0}\n",
                "{\"query\":\"retry\",\"limit\":1001}\n",
                "{\"query\":\"retry\",\"retrieval\":\"exact\"}\n",
                "{\"op\":\"unknown\",\"query\":\"retry\"}\n",
                "{\"id\":\"valid\",\"query\":\"retry\",\"limit\":1}\n",
            );
            let mut output = Vec::new();
            serve_session(&cx, &mut session, &mut Cursor::new(input), &mut output)
                .await
                .unwrap();
            let output = frames(&output);
            assert_eq!(output.len(), 7);
            for frame in &output[1..6] {
                assert_eq!(frame["event"], "failed");
            }
            assert_eq!(output[6]["event"], "results");
            assert_eq!(output[6]["id"], "valid");
            assert_eq!(provider.calls.load(Ordering::Relaxed), before + 1);

            let mut oversized = vec![b'x'; crate::serve::MAX_REQUEST_BYTES + 2];
            oversized.extend_from_slice(b"\n{\"query\":\"retry\"}\n");
            let length = oversized.len();
            let mut oversized = Cursor::new(oversized);
            let mut output = Vec::new();
            assert!(
                serve_session(&cx, &mut session, &mut oversized, &mut output)
                    .await
                    .is_err()
            );
            assert!(usize::try_from(oversized.position()).unwrap() < length);
            assert_eq!(
                frames(&output).len(),
                1,
                "only readiness precedes a terminal frame refusal"
            );
            assert_eq!(provider.calls.load(Ordering::Relaxed), before + 1);

            let mut cancelled = CancelOnRead {
                cx: &cx,
                inner: Cursor::new(b"{\"query\":\"retry\"}\n".to_vec()),
            };
            assert!(
                serve_session(&cx, &mut session, &mut cancelled, &mut Vec::new())
                    .await
                    .is_err()
            );
            assert_eq!(provider.calls.load(Ordering::Relaxed), before + 1);
            assert_eq!(session.receipt.authority, selected.authority);
            cx.set_cancel_requested(false);

            let mut output = FailAfterReady { writes: 0 };
            let mut input = Cursor::new(b"{\"query\":\"retry\"}\n{\"query\":\"garden\"}\n");
            assert!(
                serve_session(&cx, &mut session, &mut input, &mut output)
                    .await
                    .is_err()
            );
            assert_eq!(
                provider.calls.load(Ordering::Relaxed),
                before + 2,
                "delivery failure terminates before the next query"
            );
            assert_eq!(output.writes, 2);
        });
    }
}

#[cfg(target_os = "linux")]
mod progressive_authority {
    use std::io::Cursor;
    use std::os::unix::fs::DirBuilderExt as _;
    use std::time::{SystemTime, UNIX_EPOCH};

    use frankensearch_core::generation::GenerationComponentRole;
    use frankensearch_index::generation_root::generation_reader::component_object_name_v1;

    use super::*;

    fn private_parent() -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let parent = PathBuf::from(std::env::var_os("HOME").unwrap()).join(format!(
            ".frankensearch-authority-full-{}-{stamp}",
            std::process::id(),
        ));
        // These are real qualified-root lifecycle tests. Unsupported filesystems
        // remain failures, without an overlay/tmpfs bypass or automatic skip.
        fs::DirBuilder::new().mode(0o700).create(&parent).unwrap();
        parent
    }

    fn provider(quality: bool) -> Arc<Provider> {
        let mut identity = provider_identity();
        if quality {
            identity.producer.implementation_revision = "independent-complete-quality".to_owned();
        }
        identity.validate().unwrap();
        Arc::new(Provider {
            identity,
            calls: AtomicUsize::new(0),
        })
    }

    struct FailingQuality {
        inner: Arc<Provider>,
    }

    impl Embedder for FailingQuality {
        fn identity(&self) -> SearchResult<&EmbeddingIdentityBundleV1> {
            self.inner.identity()
        }

        fn embed<'a>(&'a self, _: &'a Cx, _: &'a str) -> SearchFuture<'a, Vec<f32>> {
            self.inner.calls.fetch_add(1, Ordering::Relaxed);
            Box::pin(async {
                Err(SearchError::SubsystemError {
                    subsystem: "authority_progressive_test",
                    source: "private-quality-inference-failure".into(),
                })
            })
        }

        fn dimension(&self) -> usize {
            self.inner.dimension()
        }

        fn id(&self) -> &str {
            self.inner.id()
        }

        fn model_name(&self) -> &str {
            self.inner.model_name()
        }

        fn is_semantic(&self) -> bool {
            true
        }

        fn category(&self) -> ModelCategory {
            self.inner.category()
        }
    }

    fn models(fast: &Arc<Provider>, quality: &Arc<Provider>) -> Models {
        Models {
            fast: fast.clone(),
            quality: Some(quality.clone()),
        }
    }

    fn calls(fast: &Provider, quality: &Provider) -> (usize, usize) {
        (
            fast.calls.load(Ordering::Relaxed),
            quality.calls.load(Ordering::Relaxed),
        )
    }

    fn plan(mode: QueryMode, stream: bool) -> QueryPlan<'static> {
        QueryPlan {
            text: "how do I retry failed network requests",
            limit: 2,
            mode,
            stream,
        }
    }

    fn frames(bytes: &[u8]) -> Vec<serde_json::Value> {
        bytes
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect()
    }

    fn update_options(
        parent: &Path,
        predecessor: &Path,
        label: &str,
        input: &Path,
        high: u64,
    ) -> Options {
        let mut options = parse(&update_arguments()).unwrap().unwrap();
        options.from_receipt = Some(predecessor.to_path_buf());
        options.work = Some(parent.join(format!("{label}-work")));
        options.receipt = parent.join(format!("{label}.json"));
        options.input = Some(input.to_path_buf());
        options.source_low = Some(2);
        options.source_high = Some(high);
        options
    }

    async fn publish_full(
        cx: &Cx,
        parent: &Path,
        fast: &Arc<Provider>,
        quality: &Arc<Provider>,
        documents: Vec<IndexableDocument>,
        retrieval: [NativeBuildRetrieval; 2],
    ) -> Receipt {
        let root_path = parent.join("root");
        let root = create_root(&root_path).expect("qualified Linux root is required");
        let built = NativeIndexBuilder::new(
            parent.join("initial-work"),
            new_generation(1).unwrap(),
            fast.clone(),
        )
        .unwrap()
        .with_fast_storage(NativeBuildPrecision::F32, retrieval[0])
        .with_quality_embedder(quality.clone())
        .unwrap()
        .with_quality_storage(NativeBuildPrecision::F32, retrieval[1])
        .unwrap()
        .add_documents(documents)
        .build_progressive_generation(cx, CommitRange { low: 2, high: 7 }, DEFAULT_IMAGE_BYTES)
        .await
        .unwrap();
        stage_and_publish(
            cx,
            &root,
            &root_path,
            &parent.join("initial.json"),
            &built,
            None,
            PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
        )
        .unwrap()
        .receipt
    }

    async fn exchange(
        cx: &Cx,
        session: &mut AuthoritySession,
        messages: &[serde_json::Value],
    ) -> Vec<serde_json::Value> {
        let mut input = Vec::new();
        for message in messages {
            serde_json::to_writer(&mut input, message).unwrap();
            input.push(b'\n');
        }
        let mut output = Vec::new();
        serve_session(cx, session, &mut Cursor::new(input), &mut output)
            .await
            .unwrap();
        frames(&output)
    }

    fn topology_models(mode: QueryMode, fast: &Arc<Provider>, quality: &Arc<Provider>) -> Models {
        Models {
            fast: if mode == QueryMode::Quality {
                quality.clone()
            } else {
                fast.clone()
            },
            quality: (mode == QueryMode::Full).then(|| quality.clone() as Arc<dyn Embedder>),
        }
    }

    fn retrieval_parameters(snapshot: &NativeGenerationSnapshot) -> Option<(HnswParams, u64)> {
        match snapshot.retrieval_configuration() {
            NativeBuildRetrieval::Exact => None,
            NativeBuildRetrieval::Hnsw { params, seed } => Some((params, seed)),
        }
    }

    fn empty_lifecycle_index_options(
        parent: &Path,
        mode: QueryMode,
        ann: bool,
        input: &Path,
    ) -> Options {
        let mut options = parse(&index_arguments()).unwrap().unwrap();
        options.root = Some(parent.join("root"));
        options.work = Some(parent.join("initial-work"));
        options.receipt = parent.join("initial.json");
        options.input = Some(input.to_path_buf());
        options.tier = Some(if mode == QueryMode::Quality {
            EmbedderTierTag::Quality
        } else {
            EmbedderTierTag::Fast
        });
        options.progressive = mode == QueryMode::Full;
        options.retrieval = Some(if ann {
            NativeBuildRetrieval::Hnsw {
                params: HnswParams::default(),
                seed: 17,
            }
        } else {
            NativeBuildRetrieval::Exact
        });
        options.source_low = Some(2);
        options.source_high = Some(7);
        options
    }

    async fn publish_history_successor(
        cx: &Cx,
        parent: &Path,
        previous: &NativeGenerationSnapshot,
        provider: &Arc<Provider>,
        label: &str,
    ) -> Receipt {
        let root = admit_root(&parent.join("root")).unwrap();
        let builder = NativeIndexBuilder::new(
            parent.join(format!("{label}-work")),
            new_generation(previous.authority().sequence + 1).unwrap(),
            provider.clone(),
        )
        .unwrap()
        .with_fast_storage(NativeBuildPrecision::F32, NativeBuildRetrieval::Exact)
        .add_documents([IndexableDocument::new(
            format!("live-{label}"),
            "retry current contents",
        )]);
        let source = CommitRange {
            low: 2,
            high: previous.metadata().commit_range.high + 1,
        };
        let built = builder
            .build_exact_generation(cx, EmbedderTierTag::Fast, source, DEFAULT_IMAGE_BYTES)
            .await
            .unwrap();
        // Historical readers must accept a fully authenticated predecessor with
        // a different topology. An ordinary CLI rebuild correctly refuses that
        // change, so this fixture publishes through the lower-level authority
        // API rather than pretending the rebuild grant authorizes it.
        if previous.topology() != NativeGenerationTopology::FastOnly {
            assert!(prepare_publication(&built, Some(previous), DEFAULT_IMAGE_BYTES).is_err());
        }
        let writer = WriterIdentityV1::for_current_process().unwrap();
        let candidate = built
            .candidate(
                new_generation(1).unwrap().nonce,
                Some(previous.authority()),
                GenerationAuthorityActionV1::Activate,
                [0x33; 32],
            )
            .unwrap();
        let prepared = PreparedPublication {
            candidate,
            root_id: previous.retained_snapshot().root_id(),
            expected: previous.retained_snapshot().authority_pair(),
            writer,
        };
        stage_prepared_and_publish(
            cx,
            &root,
            &parent.join("root"),
            &parent.join(format!("{label}.json")),
            &built,
            prepared,
            PublicationOptions::unscoped(DEFAULT_IMAGE_BYTES),
        )
        .unwrap()
        .receipt
    }

    struct HistoricalFixture {
        parent: PathBuf,
        fast: Arc<Provider>,
        quality: Arc<Provider>,
        current_model: Arc<Provider>,
        target: Receipt,
        current: Receipt,
    }

    async fn historical_fixture(cx: &Cx, mode: QueryMode, empty: bool) -> HistoricalFixture {
        let parent = private_parent();
        let fast = provider(false);
        let quality = provider(true);
        let input = parent.join("original.jsonl");
        fs::write(&input, if empty { "\n" } else { concat!(
            "{\"id\":\"zeta\",\"content\":\"retry café archived notes\",\"metadata\":{\"tenant\":\"A\"}}\n",
            "{\"id\":\"alpha\",\"content\":\"retry private archive\",\"metadata\":{\"tenant\":\"B\"}}\n",
        ) }).unwrap();
        let options = empty_lifecycle_index_options(&parent, mode, true, &input);
        index_with_loader(cx, &options, &mut Vec::new(), |requested| {
            Ok(topology_models(requested, &fast, &quality))
        })
        .await
        .unwrap();
        let target = Receipt::read(&options.receipt).unwrap();
        let mut identity = provider_identity();
        identity.producer.implementation_revision = "historical-current-producer".to_owned();
        let current_model = Arc::new(Provider {
            identity,
            calls: AtomicUsize::new(0),
        });
        let previous = target.open(cx, DEFAULT_IMAGE_BYTES).unwrap();
        let current =
            publish_history_successor(cx, &parent, &previous, &current_model, "current").await;
        HistoricalFixture {
            parent,
            fast,
            quality,
            current_model,
            target,
            current,
        }
    }

    fn root_file_bytes(root: &Path) -> std::collections::BTreeMap<std::ffi::OsString, Vec<u8>> {
        fs::read_dir(root)
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                (entry.file_name(), fs::read(entry.path()).unwrap())
            })
            .collect()
    }

    #[test]
    fn historical_commands_read_only_the_explicit_predecessor_with_its_own_producers() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (mode, empty) in [
                (QueryMode::Fast, false),
                (QueryMode::Quality, false),
                (QueryMode::Full, false),
                (QueryMode::Full, true),
            ] {
                let fixture = historical_fixture(&cx, mode, empty).await;
                let target_path = fixture.parent.join("initial.json");
                let current_path = fixture.parent.join("current.json");
                let before = root_file_bytes(&fixture.target.root);
                let receipts_before = [
                    fs::read(&target_path).unwrap(),
                    fs::read(&current_path).unwrap(),
                ];
                let current_calls = fixture.current_model.calls.load(Ordering::Relaxed);
                assert_ne!(fixture.target.identity, fixture.current.identity);
                if mode != QueryMode::Fast {
                    assert_ne!(fixture.target.topology(), fixture.current.topology());
                }
                let combined =
                    usize::try_from(fixture.target.image_bytes + fixture.current.image_bytes)
                        .unwrap();
                let mut options = parse(&[
                    "authority-search",
                    "--receipt",
                    "unused",
                    "--query",
                    "retry",
                ])
                .unwrap()
                .unwrap();
                options.receipt = target_path.clone();
                options.filter = Some(Filter::parse(r#"{"metadata":{"tenant":"A"}}"#).unwrap());
                options.max_image_bytes = combined;
                let mut output = Vec::new();
                assert!(
                    search_with_loader(&cx, &options, &mut output, |_| {
                        panic!("a stale receipt alone cannot open its former generation")
                    })
                    .await
                    .is_err()
                );
                assert!(output.is_empty());
                options.current_receipt = Some(current_path.clone());
                let mut modes = if mode == QueryMode::Full {
                    vec![QueryMode::Fast, QueryMode::Quality, QueryMode::Full]
                } else {
                    vec![mode]
                };
                modes.push(QueryMode::Lexical);
                for requested_mode in modes {
                    options.mode = Some(requested_mode);
                    options.lexical_only = requested_mode == QueryMode::Lexical;
                    options.stream = true;
                    output.clear();
                    let mut requested = Vec::new();
                    search_with_loader(&cx, &options, &mut output, |tier| {
                        assert_ne!(requested_mode, QueryMode::Lexical);
                        requested.push(tier);
                        Ok(match tier {
                            EmbedderTierTag::Fast => fixture.fast.clone(),
                            EmbedderTierTag::Quality => fixture.quality.clone(),
                        })
                    })
                    .await
                    .unwrap();
                    assert_eq!(
                        requested,
                        match requested_mode {
                            QueryMode::Fast => vec![EmbedderTierTag::Fast],
                            QueryMode::Quality => vec![EmbedderTierTag::Quality],
                            QueryMode::Full =>
                                vec![EmbedderTierTag::Fast, EmbedderTierTag::Quality],
                            QueryMode::Lexical => vec![],
                        }
                    );
                    let responses = frames(&output);
                    assert_eq!(
                        responses.len(),
                        if requested_mode == QueryMode::Full {
                            2
                        } else {
                            1
                        }
                    );
                    for response in responses {
                        assert_eq!(response["event"], "results");
                        assert_eq!(
                            response["authority"],
                            serde_json::json!(fixture.target.authority)
                        );
                        assert_eq!(
                            response["generation"],
                            serde_json::json!(fixture.target.generation)
                        );
                        assert_eq!(response["selection_kind"], "retained_predecessor");
                        assert_eq!(
                            response["selected_predecessor_of"],
                            serde_json::json!(fixture.current.authority)
                        );
                        assert_eq!(response["scope"]["eligible_documents"], usize::from(!empty));
                        let hits = response["results"].as_array().unwrap();
                        assert_eq!(hits.len(), usize::from(!empty));
                        if !empty {
                            assert_eq!(hits[0]["doc_id"], "zeta");
                        }
                    }
                }
                assert_eq!(
                    fixture.current_model.calls.load(Ordering::Relaxed),
                    current_calls
                );
                if empty {
                    assert_eq!(calls(&fixture.fast, &fixture.quality), (0, 0));
                }

                let mut source =
                    parse(&["authority-fetch", "--receipt", "unused", "--doc-id", "zeta"])
                        .unwrap()
                        .unwrap();
                source.receipt = target_path;
                source.current_receipt = Some(current_path);
                source.max_image_bytes = combined;
                source.include_metadata = true;
                output.clear();
                let inference_before_fetch = calls(&fixture.fast, &fixture.quality);
                fetch(&cx, &source, &mut output).unwrap();
                let response = &frames(&output)[0];
                assert_eq!(response["selection_kind"], "retained_predecessor");
                assert_eq!(
                    response["selected_predecessor_of"],
                    serde_json::json!(fixture.current.authority)
                );
                assert_eq!(
                    response["authority"],
                    serde_json::json!(fixture.target.authority)
                );
                if empty {
                    assert_eq!(
                        response["documents"][0],
                        serde_json::json!({"doc_id":"zeta", "status":"unavailable"})
                    );
                } else {
                    assert_eq!(
                        response["documents"][0]["content"],
                        "retry café archived notes"
                    );
                    assert_eq!(response["documents"][0]["metadata"]["tenant"], "A");
                }
                assert_eq!(
                    calls(&fixture.fast, &fixture.quality),
                    inference_before_fetch
                );
                assert_eq!(
                    fixture.current_model.calls.load(Ordering::Relaxed),
                    current_calls
                );
                assert_eq!(root_file_bytes(&fixture.target.root), before);
                assert_eq!(
                    fs::read(fixture.parent.join("initial.json")).unwrap(),
                    receipts_before[0]
                );
                assert_eq!(
                    fs::read(fixture.parent.join("current.json")).unwrap(),
                    receipts_before[1]
                );
            }
        });
    }

    #[test]
    fn historical_admission_checks_both_receipts_and_the_combined_authenticated_budget() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fixture = historical_fixture(&cx, QueryMode::Full, false).await;
            let target_reader = fixture.current.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            let latest = publish_history_successor(
                &cx,
                &fixture.parent,
                &target_reader,
                &fixture.current_model,
                "latest",
            )
            .await;
            let latest_reader = latest.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            let combined =
                usize::try_from(fixture.current.image_bytes + latest.image_bytes).unwrap();
            let before = root_file_bytes(&latest.root);
            let calls_before = fixture.current_model.calls.load(Ordering::Relaxed);
            for label in [
                "stale-current",
                "same-head",
                "ancient-target",
                "target-root",
                "current-root-id",
                "target-documents",
                "current-documents",
                "target-producer",
                "target-topology",
                "target-bound",
                "current-bound",
                "combined-budget",
            ] {
                let mut target = fixture.current.clone();
                let mut current = latest.clone();
                let mut budget = combined;
                match label {
                    "stale-current" => {
                        target = fixture.target.clone();
                        current = fixture.current.clone();
                    }
                    "same-head" => target = latest.clone(),
                    "ancient-target" => target = fixture.target.clone(),
                    "target-root" => target.root = fixture.parent.join("unrelated-root"),
                    "current-root-id" => current.root_id[0] ^= 1,
                    "target-documents" => target.documents = 0,
                    "current-documents" => current.documents = 0,
                    "target-producer" => {
                        let mut identity = target.identity.identity.clone();
                        identity.producer.implementation_revision =
                            "forged-history-producer".to_owned();
                        target.identity = identity.freeze().unwrap();
                    }
                    "target-topology" => {
                        target.schema = PROGRESSIVE_RECEIPT_SCHEMA.to_owned();
                        target.quality_identity = Some(target.identity.clone());
                    }
                    "target-bound" => target.image_bytes -= 1,
                    "current-bound" => current.image_bytes -= 1,
                    "combined-budget" => budget -= 1,
                    _ => unreachable!(),
                }
                let target_path = fixture.parent.join(format!("{label}-target.json"));
                let current_path = fixture.parent.join(format!("{label}-current.json"));
                fs::write(&target_path, serde_json::to_vec(&target).unwrap()).unwrap();
                fs::write(&current_path, serde_json::to_vec(&current).unwrap()).unwrap();
                let mut options = parse(&[
                    "authority-search",
                    "--receipt",
                    "unused",
                    "--query",
                    "retry",
                ])
                .unwrap()
                .unwrap();
                options.receipt = target_path.clone();
                options.current_receipt = Some(current_path.clone());
                options.max_image_bytes = budget;
                let mut output = Vec::new();
                let mut loads = 0;
                assert!(
                    search_with_loader(&cx, &options, &mut output, |_| {
                        loads += 1;
                        Err(bad("refused history must not load a producer"))
                    })
                    .await
                    .is_err(),
                    "{label}"
                );
                assert_eq!(loads, 0, "{label}");
                assert!(output.is_empty(), "{label}");
                let mut source = parse(&[
                    "authority-fetch",
                    "--receipt",
                    "unused",
                    "--doc-id",
                    "live-current",
                ])
                .unwrap()
                .unwrap();
                source.receipt = target_path;
                source.current_receipt = Some(current_path);
                source.max_image_bytes = budget;
                assert!(fetch(&cx, &source, &mut output).is_err(), "{label}");
                assert!(output.is_empty(), "{label}");
                assert_eq!(root_file_bytes(&latest.root), before, "{label}");
            }

            // A receipt's permitted maximum is not its authenticated allocation.
            // Charge actual admitted current bytes, then the complete target.
            let mut inflated = latest.clone();
            inflated.image_bytes += 1_000_000;
            let inflated_path = fixture.parent.join("inflated-current.json");
            fs::write(&inflated_path, serde_json::to_vec(&inflated).unwrap()).unwrap();
            let mut options = parse(&[
                "authority-search",
                "--receipt",
                "unused",
                "--query",
                "retry",
            ])
            .unwrap()
            .unwrap();
            options.receipt = fixture.parent.join("current.json");
            options.current_receipt = Some(inflated_path);
            options.max_image_bytes = combined;
            options.lexical_only = true;
            let mut output = Vec::new();
            search_with_loader(&cx, &options, &mut output, |_| {
                panic!("lexical history does not load models")
            })
            .await
            .unwrap();
            assert_eq!(frames(&output)[0]["results"][0]["doc_id"], "live-current");
            options.max_image_bytes = combined - 1;
            output.clear();
            assert!(
                search_with_loader(&cx, &options, &mut output, |_| panic!(
                    "budget refusal precedes models"
                ))
                .await
                .is_err()
            );
            assert!(output.is_empty());
            options.max_image_bytes = combined;

            // Historical reads require healthy complete current admission;
            // damaged-current rollback is a separate explicit operation.
            let component = latest_reader
                .retained_snapshot()
                .manifest()
                .components()
                .get(GenerationComponentRole::Vector)
                .unwrap();
            let current_vector = latest.root.join(component_object_name_v1(
                GenerationComponentRole::Vector,
                component.sha256,
            ));
            let saved_vector = fixture.parent.join("healthy-current-vector");
            fs::rename(&current_vector, &saved_vector).unwrap();
            write_immutable(&cx, &current_vector, b"damaged-current-vector").unwrap();
            let damaged = root_file_bytes(&latest.root);
            options.lexical_only = false;
            assert!(
                search_with_loader(&cx, &options, &mut output, |_| panic!(
                    "damaged current cannot authorize model loading"
                ))
                .await
                .is_err()
            );
            assert!(output.is_empty());
            assert_eq!(root_file_bytes(&latest.root), damaged);
            fs::rename(
                &current_vector,
                fixture.parent.join("damaged-current-vector-evidence"),
            )
            .unwrap();
            fs::rename(&saved_vector, &current_vector).unwrap();
            assert_eq!(root_file_bytes(&latest.root), before);
            cx.set_cancel_requested(true);
            assert!(
                search_with_loader(&cx, &options, &mut output, |_| panic!(
                    "cancelled history cannot load models"
                ))
                .await
                .is_err()
            );
            cx.set_cancel_requested(false);
            assert!(output.is_empty());
            assert_eq!(
                fixture.current_model.calls.load(Ordering::Relaxed),
                calls_before
            );

            // Authority sequence, not artifact sequence, defines history.
            // A rollback can make the immediate predecessor's artifact newer.
            let mut rollback_options = parse(&rollback_arguments()).unwrap().unwrap();
            rollback_options.from_receipt = Some(fixture.parent.join("latest.json"));
            rollback_options.target_receipt = Some(fixture.parent.join("current.json"));
            rollback_options.receipt = fixture.parent.join("rollback.json");
            rollback(&cx, &rollback_options, &mut Vec::new()).unwrap();
            let rolled_back = Receipt::read(&rollback_options.receipt).unwrap();
            assert!(latest.generation.sequence > rolled_back.generation.sequence);
            assert!(
                latest.authority().unwrap().sequence < rolled_back.authority().unwrap().sequence
            );
            options.receipt = fixture.parent.join("latest.json");
            options.current_receipt = Some(rollback_options.receipt);
            options.lexical_only = true;
            options.max_image_bytes =
                usize::try_from(latest.image_bytes + rolled_back.image_bytes).unwrap();
            search_with_loader(&cx, &options, &mut output, |_| {
                panic!("lexical history does not load models")
            })
            .await
            .unwrap();
            let response = &frames(&output)[0];
            assert_eq!(response["results"][0]["doc_id"], "live-latest");
            assert_eq!(response["authority"], serde_json::json!(latest.authority));
            assert_eq!(
                response["selected_predecessor_of"],
                serde_json::json!(rolled_back.authority)
            );
        });
    }

    #[test]
    fn historical_warm_sessions_pin_source_and_scopes_and_cannot_activate_or_publish() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fixture = historical_fixture(&cx, QueryMode::Full, false).await;
            let current_path = fixture.parent.join("current.json");
            let combined =
                usize::try_from(fixture.target.image_bytes + fixture.current.image_bytes).unwrap();
            let (root, snapshot) =
                open_read_selection(&cx, &fixture.target, Some(&current_path), combined).unwrap();
            assert_eq!(
                snapshot.retained_snapshot().selected_predecessor_of(),
                Some(fixture.current.authority().unwrap())
            );
            assert!(
                AuthoritySession::new(
                    &cx,
                    admit_root(&fixture.target.root).unwrap(),
                    fixture.target.clone(),
                    Arc::clone(&snapshot),
                    models(&fixture.fast, &fixture.quality),
                    (2, combined),
                    true,
                )
                .is_err()
            );
            let no_updates = AuthoritySession::new(
                &cx,
                admit_root(&fixture.target.root).unwrap(),
                fixture.target.clone(),
                Arc::clone(&snapshot),
                models(&fixture.fast, &fixture.quality),
                (2, combined),
                false,
            )
            .unwrap();
            assert!(no_updates.with_updates(2, MAX_INPUT_BYTES).is_err());
            let mut session = AuthoritySession::new(
                &cx,
                root,
                fixture.target.clone(),
                snapshot,
                models(&fixture.fast, &fixture.quality),
                (2, combined),
                false,
            )
            .unwrap()
            .with_filter(
                &cx,
                Some(Filter::parse(r#"{"metadata":{"tenant":"A"}}"#).unwrap()),
                MAX_INPUT_BYTES,
            )
            .unwrap();
            let current_reader = fixture.current.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            let latest = publish_history_successor(
                &cx,
                &fixture.parent,
                &current_reader,
                &fixture.current_model,
                "latest",
            )
            .await;
            fs::rename(
                fixture.parent.join("original.jsonl"),
                fixture.parent.join("archived-input.jsonl"),
            )
            .unwrap();
            fs::rename(
                fixture.parent.join("initial-work"),
                fixture.parent.join("archived-initial-work"),
            )
            .unwrap();
            let root_before = root_file_bytes(&latest.root);
            let before = calls(&fixture.fast, &fixture.quality);
            let current_calls = fixture.current_model.calls.load(Ordering::Relaxed);
            assert!(
                open_read_selection(&cx, &fixture.target, Some(&current_path), combined).is_err()
            );
            let response = exchange(&cx, &mut session, &[
                serde_json::json!({"query":"retry", "mode":"full", "stream":true, "filter":{}}),
                serde_json::json!({"op":"fetch", "expected_authority":fixture.target.authority, "documents":[{"doc_id":"zeta"},{"doc_id":"alpha"},{"doc_id":"live-latest"}], "filter":{}}),
                serde_json::json!({"query":"retry", "mode":"lexical", "filter":{"ids":["alpha"]}}),
            ]).await;
            assert_eq!(response.len(), 5);
            for frame in &response {
                assert_eq!(
                    frame["authority"],
                    serde_json::json!(fixture.target.authority)
                );
                assert_eq!(frame["selection_kind"], "retained_predecessor");
                assert_eq!(
                    frame["selected_predecessor_of"],
                    serde_json::json!(fixture.current.authority)
                );
            }
            assert_eq!(response[0]["event"], "ready");
            for frame in &response[1..=2] {
                assert_eq!(frame["scope"]["eligible_documents"], 1);
                assert_eq!(frame["results"].as_array().unwrap().len(), 1);
                assert_eq!(frame["results"][0]["doc_id"], "zeta");
            }
            assert_eq!(
                response[3]["documents"][0]["content"],
                "retry café archived notes"
            );
            assert_eq!(
                response[3]["documents"][1],
                serde_json::json!({"doc_id":"alpha", "status":"unavailable"})
            );
            assert_eq!(
                response[3]["documents"][2],
                serde_json::json!({"doc_id":"live-latest", "status":"unavailable"})
            );
            assert_eq!(response[4]["scope"]["eligible_documents"], 0);
            assert!(response[4]["results"].as_array().unwrap().is_empty());
            assert_eq!(
                calls(&fixture.fast, &fixture.quality),
                (before.0 + 1, before.1 + 1)
            );

            // Private callers cannot bypass the read-only marker by toggling
            // grants after startup. Each operation independently checks it.
            session.allow_activation = true;
            session.allow_model_activation = true;
            session.update_limits = Some(AuthorityUpdateLimits {
                batch_size: 2,
                max_source_bytes: MAX_INPUT_BYTES,
            });
            let before = calls(&fixture.fast, &fixture.quality);
            let model_request = serde_json::json!({
                "op":"activate_model", "receipt":fixture.parent.join("absent.json"),
                "expected_authority":fixture.target.authority, "model_dir":fixture.parent.join("absent-models"),
            });
            let request: model_activation::Request =
                serde_json::from_value(model_request.clone()).unwrap();
            let mut loads = 0;
            let error = session
                .activate_model_with_loader(&cx, &request, |_, _, _, _| {
                    loads += 1;
                    Err(bad(
                        "historical model activation must refuse before loading",
                    ))
                })
                .unwrap_err();
            assert!(error.to_string().contains("historical_selection_read_only"));
            assert_eq!(loads, 0);
            let response = exchange(&cx, &mut session, &[
                serde_json::json!({"op":"activate", "receipt":fixture.parent.join("latest.json"), "expected_authority":fixture.target.authority}),
                model_request,
                serde_json::json!({"op":"update", "expected_authority":fixture.target.authority, "work_dir":fixture.parent.join("forbidden-work"), "new_receipt":fixture.parent.join("forbidden.json"), "source_low":2, "source_high":10, "changes":[{"op":"delete","id":"zeta"}]}),
            ]).await;
            for frame in &response[1..] {
                assert_eq!(frame["event"], "failed");
                assert_eq!(frame["code"], "historical_selection_read_only");
                assert_eq!(
                    frame["authority"],
                    serde_json::json!(fixture.target.authority)
                );
                assert_eq!(
                    frame["selected_predecessor_of"],
                    serde_json::json!(fixture.current.authority)
                );
            }
            assert_eq!(calls(&fixture.fast, &fixture.quality), before);
            assert_eq!(
                fixture.current_model.calls.load(Ordering::Relaxed),
                current_calls
            );
            assert!(!fixture.parent.join("forbidden-work").exists());
            assert!(!fixture.parent.join("forbidden.json").exists());
            assert_eq!(root_file_bytes(&latest.root), root_before);
        });
    }

    #[test]
    fn empty_authority_commands_index_rebuild_and_reopen_without_inference() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (mode, ann) in [
                (QueryMode::Fast, false),
                (QueryMode::Fast, true),
                (QueryMode::Quality, false),
                (QueryMode::Quality, true),
                (QueryMode::Full, false),
                (QueryMode::Full, true),
            ] {
                let parent = private_parent();
                let fast = provider(false);
                let quality = provider(true);
                let input = parent.join("empty.jsonl");
                fs::write(&input, b"\n \t\n").unwrap();
                let options = empty_lifecycle_index_options(&parent, mode, ann, &input);
                let mut loads = 0;
                let mut output = Vec::new();
                index_with_loader(&cx, &options, &mut output, |requested| {
                    assert_eq!(requested, mode);
                    loads += 1;
                    Ok(topology_models(requested, &fast, &quality))
                })
                .await
                .unwrap();
                assert_eq!(loads, 1, "even an empty index admits its producer topology");
                assert_eq!(calls(&fast, &quality), (0, 0));
                let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(report["event"], "indexed");
                assert_eq!(report["publication"], "committed");
                let initial = Receipt::read(&options.receipt).unwrap();
                assert_eq!(initial.documents, 0);
                assert!(initial.image_bytes > 0);
                assert_eq!(initial.quality_identity.is_some(), mode == QueryMode::Full);
                assert_eq!(initial.query_mode(None).unwrap(), mode);
                fs::rename(
                    options.work.as_ref().unwrap(),
                    parent.join("archived-initial-work"),
                )
                .unwrap();
                let opened = initial.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(opened.metadata().total_documents, 0);
                assert_eq!(
                    opened.metadata().commit_range,
                    CommitRange { low: 2, high: 7 }
                );
                assert!(opened.source_documents(&cx, 0).unwrap().is_empty());
                assert!(opened.source_document(&cx, "absent", 0).unwrap().is_none());
                let retrieval = retrieval_parameters(&opened);
                let quality_retrieval = opened.quality_retrieval_configuration();
                assert_eq!(
                    quality_retrieval,
                    (mode == QueryMode::Full).then(|| options.build_retrieval(None))
                );
                let authority_bytes = fs::read(initial.root.join("AUTHORITY")).unwrap();
                let mut forged = initial.clone();
                forged.documents = 1;
                assert!(forged.open(&cx, DEFAULT_IMAGE_BYTES).is_err());
                assert_eq!(
                    fs::read(initial.root.join("AUTHORITY")).unwrap(),
                    authority_bytes
                );

                // A fresh command admits each requested producer but performs
                // no query inference, including both progressive phases.
                let mut query_modes = if mode == QueryMode::Full {
                    vec![QueryMode::Fast, QueryMode::Quality, QueryMode::Full]
                } else {
                    vec![mode]
                };
                query_modes.push(QueryMode::Lexical);
                for query_mode in query_modes {
                    let mut query = parse(&[
                        "authority-search",
                        "--receipt",
                        "unused",
                        "--query",
                        "retry",
                    ])
                    .unwrap()
                    .unwrap();
                    query.receipt = options.receipt.clone();
                    query.mode = Some(query_mode);
                    query.lexical_only = query_mode == QueryMode::Lexical;
                    query.stream = true;
                    query.max_source_bytes = 1;
                    query.filter = Some(Filter::parse(r#"{"metadata":{"tenant":"A"}}"#).unwrap());
                    output.clear();
                    let mut requested = Vec::new();
                    search_with_loader(&cx, &query, &mut output, |tier| {
                        assert_ne!(
                            query_mode,
                            QueryMode::Lexical,
                            "empty lexical queries must not load a model"
                        );
                        requested.push(tier);
                        Ok(match tier {
                            EmbedderTierTag::Fast => fast.clone(),
                            EmbedderTierTag::Quality => quality.clone(),
                        })
                    })
                    .await
                    .unwrap();
                    assert_eq!(
                        requested,
                        match query_mode {
                            QueryMode::Fast => vec![EmbedderTierTag::Fast],
                            QueryMode::Quality => vec![EmbedderTierTag::Quality],
                            QueryMode::Full =>
                                vec![EmbedderTierTag::Fast, EmbedderTierTag::Quality],
                            QueryMode::Lexical => vec![],
                        }
                    );
                    let responses = frames(&output);
                    assert_eq!(
                        responses.len(),
                        if query_mode == QueryMode::Full { 2 } else { 1 }
                    );
                    for response in &responses {
                        assert_eq!(response["event"], "results");
                        assert_eq!(response["authority"], serde_json::json!(initial.authority));
                        assert_eq!(response["scope"]["eligible_documents"], 0);
                        assert!(response["results"].as_array().unwrap().is_empty());
                    }
                    if query_mode == QueryMode::Full {
                        assert_eq!(responses[0]["phase"], "initial");
                        assert_eq!(responses[1]["phase"], "refined");
                    } else if query_mode == QueryMode::Lexical {
                        assert_eq!(responses[0]["phase"], "lexical");
                        assert_eq!(responses[0]["retrieval"], "lexical");
                        assert_eq!(responses[0]["semantic_used"], false);
                        assert_eq!(responses[0]["semantic_components_verified"], true);
                    }
                    assert_eq!(calls(&fast, &quality), (0, 0));
                }
                let mut source = parse(&[
                    "authority-fetch",
                    "--receipt",
                    "unused",
                    "--doc-id",
                    "absent",
                ])
                .unwrap()
                .unwrap();
                source.receipt = options.receipt.clone();
                source.max_source_bytes = 1;
                output.clear();
                fetch(&cx, &source, &mut output).unwrap();
                assert_eq!(
                    frames(&output)[0]["documents"],
                    serde_json::json!([{"doc_id":"absent", "status":"unavailable"}])
                );
                assert_eq!(calls(&fast, &quality), (0, 0));

                // Both explicit empty input and retained empty source pass
                // through the actual rebuild publication/receipt path.
                let mut previous = initial;
                let mut predecessor_path = options.receipt.clone();
                for (label, retained_source) in
                    [("input-rebuild", false), ("retained-rebuild", true)]
                {
                    let arguments = if retained_source {
                        retained_rebuild_arguments()
                    } else {
                        rebuild_arguments()
                    };
                    let mut rebuild_options = parse(&arguments).unwrap().unwrap();
                    rebuild_options.from_receipt = Some(predecessor_path.clone());
                    rebuild_options.work = Some(parent.join(format!("{label}-work")));
                    rebuild_options.receipt = parent.join(format!("{label}.json"));
                    rebuild_options.max_source_bytes = 1;
                    if !retained_source {
                        rebuild_options.input = Some(input.clone());
                        rebuild_options.source_low = Some(2);
                        rebuild_options.source_high = Some(8);
                    }
                    output.clear();
                    rebuild_with_loader(&cx, &rebuild_options, &mut output, |requested| {
                        assert_eq!(requested, mode);
                        loads += 1;
                        Ok(topology_models(requested, &fast, &quality))
                    })
                    .await
                    .unwrap();
                    let report = &frames(&output)[0];
                    assert_eq!(report["event"], "rebuilt");
                    assert_eq!(report["model_migrated"], false);
                    assert_eq!(
                        report["source"],
                        if retained_source {
                            "retained_generation"
                        } else {
                            "input"
                        }
                    );
                    let successor = Receipt::read(&rebuild_options.receipt).unwrap();
                    assert_eq!(successor.documents, 0);
                    assert_eq!(successor.identity, previous.identity);
                    assert_eq!(successor.quality_identity, previous.quality_identity);
                    assert_eq!(
                        successor.authority().unwrap().sequence,
                        previous.authority().unwrap().sequence + 1
                    );
                    let successor_reader = successor.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                    assert_eq!(
                        successor_reader.metadata().commit_range,
                        CommitRange { low: 2, high: 8 }
                    );
                    assert_eq!(retrieval_parameters(&successor_reader), retrieval);
                    assert_eq!(
                        successor_reader.quality_retrieval_configuration(),
                        quality_retrieval
                    );
                    assert!(
                        successor_reader
                            .source_documents(&cx, 0)
                            .unwrap()
                            .is_empty()
                    );
                    assert_eq!(calls(&fast, &quality), (0, 0));
                    previous = successor;
                    predecessor_path = rebuild_options.receipt;
                }
                assert_eq!(loads, 3);
                assert!(opened.source_documents(&cx, 0).unwrap().is_empty());
            }
        });
    }

    #[test]
    fn authority_delete_all_refills_and_rolls_back_complete_empty_generations() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (mode, ann) in [
                (QueryMode::Fast, true),
                (QueryMode::Quality, false),
                (QueryMode::Full, false),
                (QueryMode::Full, true),
            ] {
                let parent = private_parent();
                let fast = provider(false);
                let quality = provider(true);
                let input = parent.join("original.jsonl");
                fs::write(&input, concat!(
                    "{\"id\":\"zeta\",\"content\":\"retry network requests\",\"metadata\":{\"version\":\"original\"}}\n",
                    "{\"id\":\"alpha\",\"content\":\"garden flowers\"}\n",
                )).unwrap();
                let initial_options = empty_lifecycle_index_options(&parent, mode, ann, &input);
                index_with_loader(&cx, &initial_options, &mut Vec::new(), |requested| {
                    Ok(topology_models(requested, &fast, &quality))
                })
                .await
                .unwrap();
                let initial = Receipt::read(&initial_options.receipt).unwrap();
                let initial_bytes = fs::read(&initial_options.receipt).unwrap();
                let retained = initial.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                let images: Vec<_> = [
                    GenerationComponentRole::Vector,
                    GenerationComponentRole::Lexical,
                    GenerationComponentRole::Metadata,
                    GenerationComponentRole::Ann,
                ]
                .into_iter()
                .filter_map(|role| {
                    retained
                        .retained_snapshot()
                        .manifest()
                        .components()
                        .get(role)
                        .map(|component| {
                            let path = initial
                                .root
                                .join(component_object_name_v1(role, component.sha256));
                            let bytes = fs::read(&path).unwrap();
                            (path, bytes)
                        })
                })
                .collect();
                let initial_calls = match mode {
                    QueryMode::Fast => (2, 0),
                    QueryMode::Quality => (0, 2),
                    QueryMode::Full => (2, 2),
                    QueryMode::Lexical => unreachable!("index construction requires a producer"),
                };
                assert_eq!(calls(&fast, &quality), initial_calls);
                let deletes = parent.join("delete-all.jsonl");
                fs::write(&deletes, concat!(
                    "{\"op\":\"upsert\",\"id\":\"zeta\",\"content\":\"discarded intermediate content\"}\n",
                    "{\"op\":\"delete\",\"id\":\"zeta\"}\n",
                    "{\"op\":\"delete\",\"id\":\"alpha\"}\n",
                )).unwrap();
                let delete_options =
                    update_options(&parent, &initial_options.receipt, "deleted", &deletes, 8);
                let mut output = Vec::new();
                update_generation_with_loader(&cx, &delete_options, &mut output, |tier| {
                    Ok(match tier {
                        EmbedderTierTag::Fast => fast.clone(),
                        EmbedderTierTag::Quality => quality.clone(),
                    })
                })
                .await
                .unwrap();
                let response = &frames(&output)[0];
                assert_eq!(response["event"], "updated");
                assert_eq!(response["publication"], "committed");
                assert_eq!(response["documents"], 0);
                assert_eq!(response["edited_ids"], 2);
                assert_eq!(calls(&fast, &quality), initial_calls);
                let empty = Receipt::read(&delete_options.receipt).unwrap();
                assert_eq!(empty.documents, 0);
                assert_eq!(empty.identity, initial.identity);
                assert_eq!(empty.quality_identity, initial.quality_identity);
                assert_eq!(
                    empty.authority().unwrap().sequence,
                    initial.authority().unwrap().sequence + 1
                );
                let empty_reader = empty.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(
                    retrieval_parameters(&empty_reader),
                    retrieval_parameters(&retained)
                );
                assert_eq!(
                    empty_reader.quality_retrieval_configuration(),
                    retained.quality_retrieval_configuration()
                );
                assert!(empty_reader.source_documents(&cx, 0).unwrap().is_empty());
                assert!(
                    empty_reader
                        .source_document(&cx, "zeta", 0)
                        .unwrap()
                        .is_none()
                );

                let refill = parent.join("refill.jsonl");
                fs::write(&refill, concat!(
                    "{\"op\":\"upsert\",\"id\":\"zeta\",\"content\":\"retry repopulated requests\"}\n",
                    "{\"op\":\"upsert\",\"id\":\"new\",\"content\":\"garden replanted\"}\n",
                )).unwrap();
                let refill_options =
                    update_options(&parent, &delete_options.receipt, "refilled", &refill, 9);
                update_generation_with_loader(&cx, &refill_options, &mut Vec::new(), |tier| {
                    Ok(match tier {
                        EmbedderTierTag::Fast => fast.clone(),
                        EmbedderTierTag::Quality => quality.clone(),
                    })
                })
                .await
                .unwrap();
                assert_eq!(
                    calls(&fast, &quality),
                    (initial_calls.0 * 2, initial_calls.1 * 2)
                );
                let refilled = Receipt::read(&refill_options.receipt).unwrap();
                let refilled_reader = refilled.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(refilled.documents, 2);
                assert_eq!(refilled.identity, empty.identity);
                assert_eq!(refilled.quality_identity, empty.quality_identity);
                assert_eq!(
                    refilled_reader.quality_retrieval_configuration(),
                    retained.quality_retrieval_configuration()
                );
                assert_eq!(
                    refilled_reader
                        .source_document(&cx, "zeta", MAX_INPUT_BYTES)
                        .unwrap()
                        .unwrap()
                        .content,
                    "retry repopulated requests"
                );
                let mut forged = refilled.clone();
                forged.documents = 0;
                assert!(
                    forged.open(&cx, DEFAULT_IMAGE_BYTES).is_err(),
                    "zero in a receipt cannot relabel a populated cohort"
                );

                let before = calls(&fast, &quality);
                let loaded = topology_models(mode, &fast, &quality);
                let mut phases = Vec::new();
                query_admitted(
                    &cx,
                    &empty_reader,
                    None,
                    &loaded,
                    plan(mode, true),
                    |phase, hits| {
                        phases.push(phase);
                        assert!(hits.is_empty());
                        Ok(())
                    },
                )
                .await
                .unwrap();
                assert_eq!(phases.len(), if mode == QueryMode::Full { 2 } else { 1 });
                assert_eq!(
                    calls(&fast, &quality),
                    before,
                    "the retained empty reader stays empty after refill"
                );
                assert_eq!(
                    retained
                        .source_document(&cx, "zeta", MAX_INPUT_BYTES)
                        .unwrap()
                        .unwrap()
                        .content,
                    "retry network requests"
                );
                assert_eq!(fs::read(&initial_options.receipt).unwrap(), initial_bytes);
                for (path, bytes) in &images {
                    assert_eq!(fs::read(path).unwrap(), *bytes);
                }

                // Rollback reuses either complete target at a new authority
                // sequence, including a target with no documents or vectors.
                let mut rollback_options = parse(&rollback_arguments()).unwrap().unwrap();
                rollback_options.from_receipt = Some(refill_options.receipt.clone());
                rollback_options.target_receipt = Some(delete_options.receipt.clone());
                rollback_options.receipt = parent.join("restore-empty.json");
                rollback(&cx, &rollback_options, &mut Vec::new()).unwrap();
                let restored_empty = Receipt::read(&rollback_options.receipt).unwrap();
                assert_eq!(restored_empty.generation, empty.generation);
                assert_eq!(restored_empty.documents, 0);
                assert_eq!(
                    restored_empty.authority().unwrap().sequence,
                    refilled.authority().unwrap().sequence + 1
                );
                let restored_empty_reader = restored_empty.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(
                    restored_empty_reader
                        .retained_snapshot()
                        .manifest()
                        .components(),
                    empty_reader.retained_snapshot().manifest().components()
                );
                rollback_options.from_receipt = Some(rollback_options.receipt.clone());
                rollback_options.target_receipt = Some(refill_options.receipt);
                rollback_options.receipt = parent.join("restore-populated.json");
                rollback(&cx, &rollback_options, &mut Vec::new()).unwrap();
                let restored_populated = Receipt::read(&rollback_options.receipt).unwrap();
                assert_eq!(restored_populated.generation, refilled.generation);
                assert_eq!(restored_populated.documents, 2);
                assert_eq!(
                    restored_populated.authority().unwrap().sequence,
                    restored_empty.authority().unwrap().sequence + 1
                );
                let restored_reader = restored_populated.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(
                    restored_reader.retained_snapshot().manifest().components(),
                    refilled_reader.retained_snapshot().manifest().components()
                );
                assert_eq!(calls(&fast, &quality), before);
            }
        });
    }

    #[test]
    fn warm_authority_delete_all_and_refill_preserve_fixed_scopes_and_empty_checkpoints() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            for (mode, ann) in [
                (QueryMode::Fast, true),
                (QueryMode::Quality, false),
                (QueryMode::Full, false),
                (QueryMode::Full, true),
            ] {
                let parent = private_parent();
                let fast = provider(false);
                let quality = provider(true);
                let input = parent.join("original.jsonl");
                fs::write(&input, concat!(
                    "{\"id\":\"zeta\",\"content\":\"retry network requests\",\"metadata\":{\"tenant\":\"A\"}}\n",
                    "{\"id\":\"alpha\",\"content\":\"garden flowers\",\"metadata\":{\"tenant\":\"B\"}}\n",
                )).unwrap();
                let options = empty_lifecycle_index_options(&parent, mode, ann, &input);
                index_with_loader(&cx, &options, &mut Vec::new(), |requested| {
                    Ok(topology_models(requested, &fast, &quality))
                })
                .await
                .unwrap();
                let initial = Receipt::read(&options.receipt).unwrap();
                let root = admit_root(&initial.root).unwrap();
                let retained = Arc::new(initial.open(&cx, DEFAULT_IMAGE_BYTES).unwrap());
                let fixed = Filter::parse(r#"{"metadata":{"tenant":"A"}}"#).unwrap();
                let original_scope = retained
                    .scope(&cx, MAX_INPUT_BYTES, |document| Ok(fixed.matches(document)))
                    .unwrap();
                let mut session = AuthoritySession::new(
                    &cx,
                    root,
                    initial.clone(),
                    Arc::clone(&retained),
                    topology_models(mode, &fast, &quality),
                    (2, DEFAULT_IMAGE_BYTES),
                    false,
                )
                .unwrap()
                .with_filter(&cx, Some(fixed), MAX_INPUT_BYTES)
                .unwrap()
                .with_updates(2, MAX_INPUT_BYTES)
                .unwrap();
                let before = calls(&fast, &quality);
                let emptied_path = parent.join("emptied.json");
                let response = exchange(
                    &cx,
                    &mut session,
                    &[serde_json::json!({
                        "op":"update", "id":"delete-all", "expected_authority":initial.authority,
                        "work_dir":parent.join("emptied-work"), "new_receipt":emptied_path,
                        "source_low":2, "source_high":8,
                        "changes":[{"op":"delete","id":"zeta"},{"op":"delete","id":"alpha"}],
                    })],
                )
                .await;
                assert_eq!(response[0]["documents"], 2);
                assert_eq!(response[0]["scope"]["eligible_documents"], 1);
                assert_eq!(response[1]["event"], "updated");
                assert_eq!(response[1]["publication"], "committed");
                assert_eq!(response[1]["serving"], "activated");
                assert_eq!(response[1]["documents"], 0);
                assert_eq!(response[1]["scope"]["eligible_documents"], 0);
                assert_eq!(calls(&fast, &quality), before);
                let emptied = Receipt::read(&emptied_path).unwrap();
                assert_eq!(session.receipt.authority, emptied.authority);
                assert_eq!(emptied.identity, initial.identity);
                assert_eq!(emptied.quality_identity, initial.quality_identity);
                let empty_reader = Arc::clone(&session.snapshot);
                assert!(empty_reader.source_documents(&cx, 0).unwrap().is_empty());
                let response = exchange(&cx, &mut session, &[
                    serde_json::json!({"query":"retry", "mode":mode.as_str(), "stream":true, "filter":{}}),
                    serde_json::json!({
                        "op":"fetch", "expected_authority":emptied.authority,
                        "documents":[{"doc_id":"zeta"},{"doc_id":"alpha"}], "filter":{},
                    }),
                    serde_json::json!({"query":"retry", "mode":"lexical", "filter":{}}),
                ]).await;
                assert_eq!(response[0]["documents"], 0);
                assert_eq!(response[0]["scope"]["eligible_documents"], 0);
                let expected_phases = if mode == QueryMode::Full { 2 } else { 1 };
                assert_eq!(response.len(), expected_phases + 3);
                for frame in &response[1..=expected_phases] {
                    assert_eq!(frame["event"], "results");
                    assert_eq!(frame["authority"], serde_json::json!(emptied.authority));
                    assert_eq!(frame["scope"]["eligible_documents"], 0);
                    assert!(frame["results"].as_array().unwrap().is_empty());
                }
                assert_eq!(
                    response[expected_phases + 1]["documents"],
                    serde_json::json!([
                        {"doc_id":"zeta","status":"unavailable"},
                        {"doc_id":"alpha","status":"unavailable"},
                    ])
                );
                let lexical = response.last().unwrap();
                assert_eq!(lexical["phase"], "lexical");
                assert_eq!(lexical["scope"]["eligible_documents"], 0);
                assert_eq!(lexical["semantic_used"], false);
                assert!(lexical["results"].as_array().unwrap().is_empty());
                assert_eq!(calls(&fast, &quality), before);

                let checkpoint_path = parent.join("checkpoint.json");
                let response = exchange(
                    &cx,
                    &mut session,
                    &[serde_json::json!({
                        "op":"update", "id":"checkpoint", "expected_authority":emptied.authority,
                        "work_dir":parent.join("checkpoint-work"), "new_receipt":checkpoint_path,
                        "source_low":2, "source_high":9, "changes":[],
                    })],
                )
                .await;
                assert_eq!(response[1]["event"], "updated");
                assert_eq!(response[1]["documents"], 0);
                assert_eq!(response[1]["edited_ids"], 0);
                assert_eq!(response[1]["scope"]["eligible_documents"], 0);
                assert_eq!(calls(&fast, &quality), before);
                let checkpoint = Receipt::read(&checkpoint_path).unwrap();
                assert_eq!(
                    checkpoint.authority().unwrap().sequence,
                    emptied.authority().unwrap().sequence + 1
                );
                let checkpoint_reader = checkpoint.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(
                    checkpoint_reader.metadata().commit_range,
                    CommitRange { low: 2, high: 9 }
                );
                assert!(
                    checkpoint_reader
                        .source_documents(&cx, 0)
                        .unwrap()
                        .is_empty()
                );

                let refill_path = parent.join("refilled.json");
                let response = exchange(&cx, &mut session, &[
                    serde_json::json!({
                        "op":"update", "id":"refill", "expected_authority":checkpoint.authority,
                        "work_dir":parent.join("refilled-work"), "new_receipt":refill_path,
                        "source_low":2, "source_high":10, "changes":[
                            {"op":"upsert","id":"visible","content":"retry allowed refill","metadata":{"tenant":"A"}},
                            {"op":"upsert","id":"private","content":"retry PRIVATE_SOURCE","metadata":{"tenant":"B"}},
                        ],
                    }),
                    serde_json::json!({"query":"retry", "mode":mode.as_str(), "stream":true, "filter":{}}),
                    serde_json::json!({"query":"retry", "mode":"lexical", "filter":{}}),
                ]).await;
                assert_eq!(response[0]["documents"], 0);
                assert_eq!(response[1]["event"], "updated");
                assert_eq!(response[1]["documents"], 2);
                assert_eq!(response[1]["scope"]["eligible_documents"], 1);
                assert_eq!(response.len(), expected_phases + 3);
                for frame in &response[2..] {
                    assert_eq!(frame["event"], "results");
                    assert_eq!(frame["scope"]["eligible_documents"], 1);
                    assert_eq!(frame["results"].as_array().unwrap().len(), 1);
                    assert_eq!(frame["results"][0]["doc_id"], "visible");
                }
                assert_eq!(
                    calls(&fast, &quality),
                    match mode {
                        QueryMode::Fast => (before.0 + 3, before.1),
                        QueryMode::Quality => (before.0, before.1 + 3),
                        QueryMode::Full => (before.0 + 3, before.1 + 3),
                        QueryMode::Lexical =>
                            unreachable!("index construction requires a producer"),
                    },
                    "only two new documents and the nonempty scoped query invoke each selected tier"
                );
                let refilled = Receipt::read(&refill_path).unwrap();
                assert_eq!(refilled.identity, initial.identity);
                assert_eq!(refilled.quality_identity, initial.quality_identity);
                let before = calls(&fast, &quality);
                let response = exchange(
                    &cx,
                    &mut session,
                    &[serde_json::json!({
                        "op":"fetch", "expected_authority":refilled.authority,
                        "documents":[{"doc_id":"visible"},{"doc_id":"private"},{"doc_id":"zeta"}],
                        "filter":{},
                    })],
                )
                .await;
                assert_eq!(
                    response[1]["documents"][0]["content"],
                    "retry allowed refill"
                );
                assert_eq!(
                    response[1]["documents"][1],
                    serde_json::json!({"doc_id":"private","status":"unavailable"})
                );
                assert_eq!(
                    response[1]["documents"][2],
                    serde_json::json!({"doc_id":"zeta","status":"unavailable"})
                );
                assert_eq!(calls(&fast, &quality), before);

                let response = exchange(&cx, &mut session, &[
                    serde_json::json!({
                        "op":"update", "id":"empty-again", "expected_authority":refilled.authority,
                        "work_dir":parent.join("empty-again-work"), "new_receipt":parent.join("empty-again.json"),
                        "source_low":2, "source_high":11,
                        "changes":[{"op":"delete","id":"visible"},{"op":"delete","id":"private"}],
                    }),
                    serde_json::json!({"query":"retry", "mode":mode.as_str(), "stream":true, "filter":{}}),
                    serde_json::json!({"query":"retry", "mode":"lexical", "filter":{}}),
                ]).await;
                assert_eq!(response[1]["event"], "updated");
                assert_eq!(response[1]["documents"], 0);
                assert_eq!(response[1]["scope"]["eligible_documents"], 0);
                for frame in &response[2..] {
                    assert_eq!(frame["scope"]["eligible_documents"], 0);
                    assert!(frame["results"].as_array().unwrap().is_empty());
                }
                assert_eq!(calls(&fast, &quality), before);
                assert_eq!(session.snapshot.metadata().total_documents, 0);
                assert!(empty_reader.source_documents(&cx, 0).unwrap().is_empty());
                assert!(
                    checkpoint_reader
                        .source_documents(&cx, 0)
                        .unwrap()
                        .is_empty()
                );
                let loaded = topology_models(mode, &fast, &quality);
                query_admitted(
                    &cx,
                    &retained,
                    Some(&original_scope),
                    &loaded,
                    plan(mode, true),
                    |_, hits| {
                        assert_eq!(hits.len(), 1);
                        assert_eq!(hits[0].doc_id, "zeta");
                        Ok(())
                    },
                )
                .await
                .unwrap();
            }
        });
    }

    #[test]
    fn full_authority_queries_updates_activation_recovery_and_rollback_share_both_owners() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let parent = private_parent();
            let root_path = parent.join("root");
            let fast = provider(false);
            let quality = provider(true);
            let work = parent.join("initial-work");
            let first = publish_full(
                &cx,
                &parent,
                &fast,
                &quality,
                vec![
                    IndexableDocument::new("zeta", "retry network requests")
                        .with_metadata("version", "original"),
                    IndexableDocument::new("alpha", "garden flowers"),
                ],
                [NativeBuildRetrieval::Exact; 2],
            )
            .await;
            assert_eq!(calls(&fast, &quality), (2, 2));
            let first_path = parent.join("initial.json");
            fs::rename(&work, parent.join("archived-initial-work")).unwrap();
            assert_eq!(first.schema, PROGRESSIVE_RECEIPT_SCHEMA);
            assert!(first.quality_identity.is_some());
            assert_ne!(first.quality_identity.as_ref().unwrap(), &first.identity);
            let old_reader = first.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            assert_eq!(
                old_reader.topology(),
                NativeGenerationTopology::FullProgressive
            );
            assert_eq!(
                calls(&fast, &quality),
                (2, 2),
                "reopen uses retained images without document inference"
            );
            let initial_receipt = fs::read(&first_path).unwrap();

            let loaded = models(&fast, &quality);
            let before = calls(&fast, &quality);
            let mut options = parse(&[
                "authority-search",
                "--receipt",
                "unused",
                "--query",
                "how do I retry failed network requests",
                "--mode",
                "quality",
            ])
            .unwrap()
            .unwrap();
            options.receipt = first_path.clone();
            let mut requested = Vec::new();
            let mut output = Vec::new();
            search_with_loader(&cx, &options, &mut output, |tier| {
                requested.push(tier);
                assert_eq!(tier, EmbedderTierTag::Quality);
                Ok(quality.clone())
            })
            .await
            .unwrap();
            let page: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(page["phase"], "quality");
            assert_eq!(page["mode"], "quality");
            assert_eq!(page["topology"], "full_progressive");
            assert_eq!(page["results"][0]["doc_id"], "zeta");
            assert_eq!(requested, [EmbedderTierTag::Quality]);
            assert_eq!(calls(&fast, &quality), (before.0, before.1 + 1));

            let before = calls(&fast, &quality);
            let wrong_models = models(&fast, &fast);
            assert!(
                query_admitted(
                    &cx,
                    &old_reader,
                    None,
                    &wrong_models,
                    plan(QueryMode::Full, true),
                    |_, _| Ok(())
                )
                .await
                .is_err()
            );
            assert_eq!(
                calls(&fast, &quality),
                before,
                "wrong quality is rejected before either inference"
            );
            let mut initial_deliveries = 0;
            assert!(
                query_admitted(
                    &cx,
                    &old_reader,
                    None,
                    &loaded,
                    plan(QueryMode::Full, true),
                    |phase, _| {
                        assert_eq!(phase, "initial");
                        initial_deliveries += 1;
                        Err(bad("output is closed"))
                    }
                )
                .await
                .is_err()
            );
            assert_eq!(initial_deliveries, 1);
            assert_eq!(
                calls(&fast, &quality),
                (before.0 + 1, before.1),
                "undelivered Initial must not start quality inference"
            );

            let before = calls(&fast, &quality);
            let failed_quality = Models {
                fast: fast.clone(),
                quality: Some(Arc::new(FailingQuality {
                    inner: Arc::clone(&quality),
                })),
            };
            let mut phases = Vec::new();
            assert!(
                query_admitted(
                    &cx,
                    &old_reader,
                    None,
                    &failed_quality,
                    plan(QueryMode::Full, true),
                    |phase, _| {
                        phases.push(phase);
                        Ok(())
                    }
                )
                .await
                .is_err()
            );
            assert_eq!(
                phases,
                ["initial"],
                "failed refinement must not synthesize successful completion"
            );
            assert_eq!(calls(&fast, &quality), (before.0 + 1, before.1 + 1));

            let root = admit_root(&root_path).unwrap();
            let snapshot = first.open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES).unwrap();
            let mut session = AuthoritySession::new(
                &cx,
                root,
                first.clone(),
                snapshot,
                models(&fast, &quality),
                (2, DEFAULT_IMAGE_BYTES),
                true,
            )
            .unwrap();
            let mut output = Vec::new();
            let request = b"{\"id\":\"progressive\",\"query\":\"how do I retry failed network requests\",\"stream\":true}\n";
            serve_session(&cx, &mut session, &mut Cursor::new(request), &mut output)
                .await
                .unwrap();
            let responses = frames(&output);
            assert_eq!(responses.len(), 3);
            assert_eq!(responses[0]["event"], "ready");
            assert_eq!(responses[0]["default_mode"], "full");
            assert_eq!(responses[0]["progressive_available"], true);
            assert_eq!(responses[1]["phase"], "initial");
            assert_eq!(responses[2]["phase"], "refined");
            assert_eq!(responses[1]["authority"], responses[2]["authority"]);
            assert_eq!(responses[1]["generation"], responses[2]["generation"]);
            assert_eq!(responses[2]["topology"], "full_progressive");
            for frame in &responses[1..] {
                assert_eq!(frame["id"], "progressive");
                assert_eq!(frame["request"], 1);
                assert_eq!(frame["authority"], serde_json::json!(first.authority));
            }

            struct ClosedAfterReady {
                writes: usize,
            }
            impl Write for ClosedAfterReady {
                fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                    self.writes += 1;
                    if self.writes == 1 {
                        Ok(bytes.len())
                    } else {
                        Err(std::io::Error::new(
                            std::io::ErrorKind::BrokenPipe,
                            "closed",
                        ))
                    }
                }

                fn flush(&mut self) -> std::io::Result<()> {
                    Ok(())
                }
            }
            let before = calls(&fast, &quality);
            let mut closed = ClosedAfterReady { writes: 0 };
            let input = [request.as_slice(), request.as_slice()].concat();
            assert!(
                serve_session(&cx, &mut session, &mut Cursor::new(input), &mut closed)
                    .await
                    .is_err()
            );
            assert_eq!(
                closed.writes, 2,
                "no failure frame is written after a broken result stream"
            );
            assert_eq!(
                calls(&fast, &quality),
                (before.0 + 1, before.1),
                "failed initial delivery stops quality and the next request"
            );

            let metadata_input = parent.join("metadata.jsonl");
            fs::write(
                &metadata_input,
                concat!(
                    "{\"op\":\"upsert\",\"id\":\"zeta\",\"content\":\"retry network requests\",",
                    "\"title\":\"Refreshed title\",\"metadata\":{\"version\":\"metadata-only\"}}\n",
                ),
            )
            .unwrap();
            let metadata_options =
                update_options(&parent, &first_path, "metadata", &metadata_input, 7);
            let before = calls(&fast, &quality);
            let mut requested = Vec::new();
            update_generation_with_loader(&cx, &metadata_options, &mut Vec::new(), |tier| {
                requested.push(tier);
                Ok(match tier {
                    EmbedderTierTag::Fast => fast.clone(),
                    EmbedderTierTag::Quality => quality.clone(),
                })
            })
            .await
            .unwrap();
            assert_eq!(requested, [EmbedderTierTag::Fast, EmbedderTierTag::Quality]);
            assert_eq!(
                calls(&fast, &quality),
                before,
                "metadata-only edit reuses both admitted vectors"
            );
            let second = Receipt::read(&metadata_options.receipt).unwrap();
            assert_eq!(second.identity, first.identity);
            assert_eq!(second.quality_identity, first.quality_identity);
            let second_reader = second.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            assert_eq!(
                session.snapshot.authority(),
                first.authority().unwrap(),
                "warm serving never adopts a head implicitly"
            );
            session
                .activate(
                    &cx,
                    &AuthorityActivateRequest {
                        op: "activate".to_owned(),
                        id: Some("adopt-metadata".to_owned()),
                        receipt: metadata_options.receipt.clone(),
                        expected_authority: first.authority.clone(),
                    },
                )
                .unwrap();
            assert_eq!(session.snapshot.authority(), second.authority().unwrap());
            assert_eq!(
                calls(&fast, &quality),
                before,
                "activation requires no model inference"
            );

            let changed_input = parent.join("changed.jsonl");
            fs::write(&changed_input, concat!(
                "{\"op\":\"upsert\",\"id\":\"zeta\",\"content\":\"retry changed network requests\",\"metadata\":{\"version\":\"changed\"}}\n",
                "{\"op\":\"delete\",\"id\":\"alpha\"}\n",
            )).unwrap();
            let changed_options = update_options(
                &parent,
                &metadata_options.receipt,
                "changed",
                &changed_input,
                8,
            );
            update_generation_with_loader(&cx, &changed_options, &mut Vec::new(), |tier| {
                Ok(match tier {
                    EmbedderTierTag::Fast => fast.clone(),
                    EmbedderTierTag::Quality => quality.clone(),
                })
            })
            .await
            .unwrap();
            assert_eq!(
                calls(&fast, &quality),
                (before.0 + 1, before.1 + 1),
                "only the changed document is embedded in each tier"
            );
            let third = Receipt::read(&changed_options.receipt).unwrap();
            let third_reader = third.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            assert_eq!(third.documents, 1);
            assert_eq!(third.quality_identity, first.quality_identity);
            assert_eq!(
                third_reader.source_documents(&cx, MAX_INPUT_BYTES).unwrap()[0].content,
                "retry changed network requests"
            );
            let original = old_reader.source_documents(&cx, MAX_INPUT_BYTES).unwrap();
            assert_eq!(original.len(), 2);
            assert_eq!(
                original
                    .iter()
                    .find(|document| document.id == "zeta")
                    .unwrap()
                    .content,
                "retry network requests"
            );
            assert_eq!(fs::read(&first_path).unwrap(), initial_receipt);
            assert_eq!(session.snapshot.authority(), second.authority().unwrap());

            let before = calls(&fast, &quality);
            let mut recovery = parse(&["authority-recover", "--receipt", "unused"])
                .unwrap()
                .unwrap();
            recovery.receipt = changed_options.receipt.clone();
            let mut output = Vec::new();
            recover(&cx, &recovery, &mut output).unwrap();
            let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(report["publication"], "committed");
            assert_eq!(report["serving"], "activated");
            assert_eq!(calls(&fast, &quality), before);

            // Only the saved receipts cross this rollback command boundary.
            let mut rollback_options = parse(&rollback_arguments()).unwrap().unwrap();
            rollback_options.from_receipt = Some(changed_options.receipt.clone());
            rollback_options.target_receipt = Some(metadata_options.receipt.clone());
            rollback_options.receipt = parent.join("restored.json");
            rollback(&cx, &rollback_options, &mut Vec::new()).unwrap();
            assert_eq!(calls(&fast, &quality), before);
            let restored = Receipt::read(&rollback_options.receipt).unwrap();
            assert_eq!(restored.schema, PROGRESSIVE_RECEIPT_SCHEMA);
            assert_eq!(restored.generation, second.generation);
            assert_eq!(restored.identity, second.identity);
            assert_eq!(restored.quality_identity, second.quality_identity);
            assert_eq!(
                restored.authority().unwrap().sequence,
                third.authority().unwrap().sequence + 1
            );
            let restored_reader = restored.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
            assert_eq!(
                restored_reader.retained_snapshot().manifest().components(),
                second_reader.retained_snapshot().manifest().components()
            );
            assert_eq!(
                serde_json::to_value(
                    restored_reader
                        .source_documents(&cx, MAX_INPUT_BYTES)
                        .unwrap()
                )
                .unwrap(),
                serde_json::to_value(
                    second_reader
                        .source_documents(&cx, MAX_INPUT_BYTES)
                        .unwrap()
                )
                .unwrap(),
            );
            assert_eq!(
                third_reader
                    .source_documents(&cx, MAX_INPUT_BYTES)
                    .unwrap()
                    .len(),
                1
            );
            session
                .activate(
                    &cx,
                    &AuthorityActivateRequest {
                        op: "activate".to_owned(),
                        id: Some("adopt-rollback".to_owned()),
                        receipt: rollback_options.receipt.clone(),
                        expected_authority: second.authority.clone(),
                    },
                )
                .unwrap();
            assert_eq!(session.snapshot.authority(), restored.authority().unwrap());
            assert_eq!(calls(&fast, &quality), before);
            recovery.receipt = rollback_options.receipt;
            recover(&cx, &recovery, &mut Vec::new()).unwrap();
            assert_eq!(calls(&fast, &quality), before);
        });
    }

    #[test]
    fn full_ann_policies_survive_queries_rebuilds_warm_updates_and_rollback() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let fast_ann = NativeBuildRetrieval::Hnsw {
                params: HnswParams {
                    m: 4,
                    m0: 8,
                    ef_construction: 24,
                    ef_search: 12,
                },
                seed: 11,
            };
            let quality_ann = NativeBuildRetrieval::Hnsw {
                params: HnswParams {
                    m: 6,
                    m0: 12,
                    ef_construction: 48,
                    ef_search: 40,
                },
                seed: 29,
            };
            for (policies, aggregate, names) in [
                ([fast_ann, quality_ann], "hnsw", ["hnsw", "hnsw"]),
                (
                    [NativeBuildRetrieval::Exact, quality_ann],
                    "mixed",
                    ["exact", "hnsw"],
                ),
                (
                    [fast_ann, NativeBuildRetrieval::Exact],
                    "mixed",
                    ["hnsw", "exact"],
                ),
            ] {
                let parent = private_parent();
                let fast = provider(false);
                let quality = provider(true);
                let first = publish_full(
                    &cx,
                    &parent,
                    &fast,
                    &quality,
                    vec![
                        IndexableDocument::new("zeta", "retry network requests")
                            .with_metadata("tenant", "A"),
                        IndexableDocument::new("alpha", "garden flowers")
                            .with_metadata("tenant", "B"),
                    ],
                    policies,
                )
                .await;
                let original = first.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(original.retrieval_configuration(), policies[0]);
                assert_eq!(
                    original.quality_retrieval_configuration(),
                    Some(policies[1])
                );
                assert!(
                    original
                        .retained_snapshot()
                        .manifest()
                        .components()
                        .ann
                        .is_some()
                );
                let tiers = serde_json::json!({"fast":names[0], "quality":names[1]});

                // The selected mode must report and execute its own graph, even
                // when the other stored tier uses a different retrieval policy.
                for (mode, expected, requested) in [
                    (QueryMode::Fast, names[0], vec![EmbedderTierTag::Fast]),
                    (QueryMode::Quality, names[1], vec![EmbedderTierTag::Quality]),
                    (QueryMode::Lexical, "lexical", vec![]),
                    (
                        QueryMode::Full,
                        aggregate,
                        vec![EmbedderTierTag::Fast, EmbedderTierTag::Quality],
                    ),
                ] {
                    let mut options = parse(&[
                        "authority-search",
                        "--receipt",
                        "unused",
                        "--query",
                        "retry",
                        "--mode",
                        mode.as_str(),
                        "--stream",
                        "--filter",
                        r#"{"metadata":{"tenant":"A"}}"#,
                    ])
                    .unwrap()
                    .unwrap();
                    options.receipt = parent.join("initial.json");
                    let mut loaded = Vec::new();
                    let mut output = Vec::new();
                    search_with_loader(&cx, &options, &mut output, |tier| {
                        loaded.push(tier);
                        Ok(match tier {
                            EmbedderTierTag::Fast => fast.clone(),
                            EmbedderTierTag::Quality => quality.clone(),
                        })
                    })
                    .await
                    .unwrap();
                    assert_eq!(loaded, requested);
                    let results = frames(&output);
                    assert_eq!(results.len(), if mode == QueryMode::Full { 2 } else { 1 });
                    for frame in &results {
                        assert_eq!(frame["retrieval"], expected);
                        assert_eq!(frame["retrieval_by_tier"], tiers);
                        if mode == QueryMode::Lexical {
                            assert_eq!(frame["semantic_used"], false);
                            assert_eq!(frame["semantic_components_verified"], true);
                        }
                        assert_eq!(frame["authority"], serde_json::json!(first.authority));
                        assert_eq!(frame["scope"]["eligible_documents"], 1);
                        assert_eq!(frame["results"].as_array().unwrap().len(), 1);
                        assert_eq!(frame["results"][0]["doc_id"], "zeta");
                    }
                }

                let mut rebuild_options = parse(&[
                    "authority-rebuild",
                    "--from-receipt",
                    "unused",
                    "--receipt",
                    "unused",
                    "--work-dir",
                    "unused",
                    "--retained-source",
                ])
                .unwrap()
                .unwrap();
                rebuild_options.from_receipt = Some(parent.join("initial.json"));
                rebuild_options.receipt = parent.join("inherited.json");
                rebuild_options.work = Some(parent.join("inherited-work"));
                let mut requested = Vec::new();
                let mut output = Vec::new();
                Box::pin(rebuild_with_loader(
                    &cx,
                    &rebuild_options,
                    &mut output,
                    |mode| {
                        requested.push(mode);
                        Ok(models(&fast, &quality))
                    },
                ))
                .await
                .unwrap();
                assert_eq!(requested, [QueryMode::Full]);
                let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(report["retrieval"], aggregate);
                assert_eq!(report["retrieval_by_tier"], tiers);
                assert_eq!(report["source"], "retained_generation");
                assert_eq!(report["model_migrated"], false);
                let inherited = Receipt::read(&rebuild_options.receipt).unwrap();
                let root = admit_root(&inherited.root).unwrap();
                let inherited_reader = Arc::new(
                    inherited
                        .open_on_root(&cx, &root, DEFAULT_IMAGE_BYTES)
                        .unwrap(),
                );
                assert_eq!(inherited_reader.retrieval_configuration(), policies[0]);
                assert_eq!(
                    inherited_reader.quality_retrieval_configuration(),
                    Some(policies[1])
                );
                assert_eq!(
                    inherited_reader.metadata().commit_range,
                    original.metadata().commit_range
                );
                let mut session = AuthoritySession::new(
                    &cx,
                    root,
                    inherited.clone(),
                    Arc::clone(&inherited_reader),
                    models(&fast, &quality),
                    (2, DEFAULT_IMAGE_BYTES),
                    true,
                )
                .unwrap()
                .with_updates(2, MAX_INPUT_BYTES)
                .unwrap()
                .with_filter(
                    &cx,
                    Some(Filter::parse(r#"{"metadata":{"tenant":"A"}}"#).unwrap()),
                    MAX_INPUT_BYTES,
                )
                .unwrap();

                let before = calls(&fast, &quality);
                let updated_path = parent.join("warm.json");
                let response = exchange(
                    &cx,
                    &mut session,
                    &[serde_json::json!({
                        "op":"update", "id":"metadata", "expected_authority":inherited.authority,
                        "work_dir":parent.join("warm-work"), "new_receipt":updated_path,
                        "source_low":2, "source_high":8,
                        "changes":[{"op":"upsert", "id":"zeta", "content":"retry network requests",
                            "title":"Metadata update", "metadata":{"tenant":"A"}}],
                    })],
                )
                .await;
                assert_eq!(response[0]["retrieval"], aggregate);
                assert_eq!(response[1]["event"], "updated");
                assert_eq!(response[1]["retrieval_by_tier"], tiers);
                assert_eq!(
                    calls(&fast, &quality),
                    before,
                    "both tier vectors are reused without inference"
                );
                assert_eq!(session.snapshot.retrieval_configuration(), policies[0]);
                assert_eq!(
                    session.snapshot.quality_retrieval_configuration(),
                    Some(policies[1])
                );
                let updated = Receipt::read(&updated_path).unwrap();
                let response = exchange(
                    &cx,
                    &mut session,
                    &[
                        serde_json::json!({"query":"retry", "mode":"quality"}),
                        serde_json::json!({"query":"retry", "mode":"lexical"}),
                        serde_json::json!({"op":"fetch", "expected_authority":updated.authority,
                        "documents":[{"doc_id":"zeta", "include_metadata":true}]}),
                    ],
                )
                .await;
                assert_eq!(response[1]["retrieval"], names[1]);
                assert_eq!(response[1]["retrieval_by_tier"], tiers);
                assert_eq!(response[2]["retrieval"], "lexical");
                assert_eq!(response[2]["retrieval_by_tier"], tiers);
                assert_eq!(response[2]["semantic_used"], false);
                assert_eq!(response[3]["retrieval"], aggregate);
                assert_eq!(response[3]["documents"][0]["title"], "Metadata update");
                assert_eq!(calls(&fast, &quality), (before.0, before.1 + 1));

                let mut rollback_options = parse(&rollback_arguments()).unwrap().unwrap();
                rollback_options.from_receipt = Some(updated_path);
                rollback_options.target_receipt = Some(rebuild_options.receipt.clone());
                rollback_options.receipt = parent.join("restored.json");
                let before = calls(&fast, &quality);
                let mut output = Vec::new();
                rollback(&cx, &rollback_options, &mut output).unwrap();
                let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(report["retrieval"], aggregate);
                assert_eq!(report["retrieval_by_tier"], tiers);
                let restored = Receipt::read(&rollback_options.receipt).unwrap();
                let restored_reader = restored.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                assert_eq!(restored_reader.retrieval_configuration(), policies[0]);
                assert_eq!(
                    restored_reader.quality_retrieval_configuration(),
                    Some(policies[1])
                );
                assert_eq!(
                    restored_reader.retained_snapshot().manifest().components(),
                    inherited_reader.retained_snapshot().manifest().components()
                );
                assert_eq!(calls(&fast, &quality), before);
                let mut recovery = parse(&["authority-recover", "--receipt", "unused"])
                    .unwrap()
                    .unwrap();
                recovery.receipt = rollback_options.receipt.clone();
                let mut output = Vec::new();
                recover(&cx, &recovery, &mut output).unwrap();
                let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(report["retrieval_by_tier"], tiers);
                assert_eq!(calls(&fast, &quality), before);

                // Explicit CLI retrieval changes every stored tier. Without
                // this override, the custom graph parameters above survived.
                let mut previous_path = rollback_options.receipt;
                for (label, requested) in [
                    ("all-exact", NativeBuildRetrieval::Exact),
                    (
                        "all-hnsw",
                        NativeBuildRetrieval::Hnsw {
                            params: HnswParams::default(),
                            seed: 42,
                        },
                    ),
                ] {
                    rebuild_options.from_receipt = Some(previous_path);
                    rebuild_options.receipt = parent.join(format!("{label}.json"));
                    rebuild_options.work = Some(parent.join(format!("{label}-work")));
                    rebuild_options.retrieval = Some(requested);
                    let mut output = Vec::new();
                    Box::pin(rebuild_with_loader(
                        &cx,
                        &rebuild_options,
                        &mut output,
                        |mode| {
                            assert_eq!(mode, QueryMode::Full);
                            Ok(models(&fast, &quality))
                        },
                    ))
                    .await
                    .unwrap();
                    let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
                    let name = if label == "all-exact" {
                        "exact"
                    } else {
                        "hnsw"
                    };
                    assert_eq!(report["retrieval"], name);
                    assert_eq!(
                        report["retrieval_by_tier"],
                        serde_json::json!({"fast":name, "quality":name})
                    );
                    let selected = Receipt::read(&rebuild_options.receipt).unwrap();
                    let reader = selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                    assert_eq!(reader.retrieval_configuration(), requested);
                    assert_eq!(reader.quality_retrieval_configuration(), Some(requested));
                    assert_eq!(
                        reader
                            .retained_snapshot()
                            .manifest()
                            .components()
                            .ann
                            .is_some(),
                        label == "all-hnsw"
                    );
                    previous_path = rebuild_options.receipt.clone();
                }

                let selected = Receipt::read(&previous_path).unwrap();
                let retained = selected.open(&cx, DEFAULT_IMAGE_BYTES).unwrap();
                let ann = retained
                    .retained_snapshot()
                    .manifest()
                    .components()
                    .ann
                    .unwrap();
                let ann_path = selected.root.join(component_object_name_v1(
                    GenerationComponentRole::Ann,
                    ann.sha256,
                ));
                fs::rename(&ann_path, parent.join("saved-ann-image")).unwrap();
                write_immutable(&cx, &ann_path, b"corrupt-declared-ann-cohort").unwrap();
                let mut query_options = parse(&[
                    "authority-search",
                    "--receipt",
                    "unused",
                    "--query",
                    "retry",
                    "--mode",
                    "quality",
                ])
                .unwrap()
                .unwrap();
                query_options.receipt = previous_path;
                let before = calls(&fast, &quality);
                for mode in [QueryMode::Quality, QueryMode::Lexical] {
                    query_options.mode = Some(mode);
                    query_options.lexical_only = mode == QueryMode::Lexical;
                    let mut loaded = false;
                    let mut output = Vec::new();
                    assert!(
                        search_with_loader(&cx, &query_options, &mut output, |_| {
                            loaded = true;
                            Ok(quality.clone())
                        })
                        .await
                        .is_err()
                    );
                    assert!(
                        !loaded,
                        "declared ANN damage refuses before any model loading"
                    );
                    assert!(output.is_empty());
                }
                assert_eq!(calls(&fast, &quality), before);
                let results =
                    Box::pin(retained.search_quality_text(&cx, quality.as_ref(), "retry", 1))
                        .await
                        .unwrap();
                assert_eq!(results[0].doc_id, "zeta");
                assert_eq!(
                    original
                        .source_documents(&cx, MAX_INPUT_BYTES)
                        .unwrap()
                        .len(),
                    2
                );
            }
        });
    }

    #[test]
    fn full_warm_updates_refresh_both_tiers_scopes_and_source_then_recover_a_damaged_cohort() {
        asupersync::test_utils::run_test_with_cx(|cx| async move {
            let parent = private_parent();
            let fast = provider(false);
            let quality = provider(true);
            let first = publish_full(
                &cx,
                &parent,
                &fast,
                &quality,
                vec![
                    IndexableDocument::new("zeta", "retry network requests")
                        .with_metadata("tenant", "A"),
                    IndexableDocument::new("alpha", "garden flowers").with_metadata("tenant", "B"),
                ],
                [NativeBuildRetrieval::Exact; 2],
            )
            .await;
            assert_eq!(calls(&fast, &quality), (2, 2));
            let original_receipt = fs::read(parent.join("initial.json")).unwrap();
            let retained = Arc::new(first.open(&cx, DEFAULT_IMAGE_BYTES).unwrap());
            let root = admit_root(&first.root).unwrap();
            let fixed = Filter::parse(r#"{"metadata":{"tenant":"A"}}"#).unwrap();
            let mut session = AuthoritySession::new(
                &cx,
                root,
                first.clone(),
                Arc::clone(&retained),
                models(&fast, &quality),
                (2, DEFAULT_IMAGE_BYTES),
                true,
            )
            .unwrap()
            .with_filter(&cx, Some(fixed), MAX_INPUT_BYTES)
            .unwrap()
            .with_updates(2, MAX_INPUT_BYTES)
            .unwrap();

            let empty = retained.scope(&cx, MAX_INPUT_BYTES, |_| Ok(false)).unwrap();
            let before = calls(&fast, &quality);
            let wrong = models(&fast, &fast);
            let mut phases = Vec::new();
            assert!(
                query_admitted(
                    &cx,
                    &retained,
                    Some(&empty),
                    &wrong,
                    plan(QueryMode::Full, true),
                    |phase, _| {
                        phases.push(phase);
                        Ok(())
                    },
                )
                .await
                .is_err()
            );
            assert!(
                phases.is_empty(),
                "empty scopes still admit both identities before delivery"
            );
            for cancel_after_initial in [false, true] {
                let loaded = models(&fast, &quality);
                let mut phases = Vec::new();
                let outcome = query_admitted(
                    &cx,
                    &retained,
                    Some(&empty),
                    &loaded,
                    plan(QueryMode::Full, true),
                    |phase, hits| {
                        phases.push(phase);
                        assert!(hits.is_empty());
                        if cancel_after_initial {
                            cx.set_cancel_requested(true);
                            Ok(())
                        } else {
                            Err(bad("empty Initial delivery failed"))
                        }
                    },
                )
                .await;
                cx.set_cancel_requested(false);
                assert!(outcome.is_err());
                assert_eq!(
                    phases,
                    ["initial"],
                    "cancellation or broken output must not synthesize Refined"
                );
            }
            assert_eq!(calls(&fast, &quality), before);

            let before = calls(&fast, &quality);
            let response = exchange(
                &cx,
                &mut session,
                &[serde_json::json!({
                    "id":"quality", "query":"retry", "mode":"quality", "filter":{},
                })],
            )
            .await;
            assert_eq!(response.len(), 2);
            assert_eq!(response[1]["phase"], "quality");
            assert_eq!(response[1]["results"][0]["doc_id"], "zeta");
            assert_eq!(calls(&fast, &quality), (before.0, before.1 + 1));

            let before = calls(&fast, &quality);
            let response = exchange(
                &cx,
                &mut session,
                &[serde_json::json!({
                    "id":"scoped-phases", "query":"retry", "stream":true,
                    "filter":{"ids":["zeta"]},
                })],
            )
            .await;
            assert_eq!(response.len(), 3);
            assert_eq!(response[1]["phase"], "initial");
            assert_eq!(response[2]["phase"], "refined");
            for frame in &response[1..] {
                assert_eq!(frame["authority"], serde_json::json!(first.authority));
                assert_eq!(frame["id"], "scoped-phases");
                assert_eq!(frame["scope"]["eligible_documents"], 1);
                assert_eq!(frame["results"].as_array().unwrap().len(), 1);
                assert_eq!(frame["results"][0]["doc_id"], "zeta");
            }
            assert_eq!(calls(&fast, &quality), (before.0 + 1, before.1 + 1));
            let before = calls(&fast, &quality);
            let response = exchange(
                &cx,
                &mut session,
                &[serde_json::json!({
                    "id":"empty-phases", "query":"retry", "mode":"full", "stream":true,
                    "filter":{"ids":["alpha"]},
                })],
            )
            .await;
            assert_eq!(response.len(), 3);
            assert_eq!(response[1]["phase"], "initial");
            assert_eq!(response[2]["phase"], "refined");
            for frame in &response[1..] {
                assert_eq!(frame["id"], "empty-phases");
                assert_eq!(frame["request"], 1);
                assert_eq!(frame["authority"], serde_json::json!(first.authority));
                assert_eq!(frame["scope"]["eligible_documents"], 0);
                assert!(frame["results"].as_array().unwrap().is_empty());
            }
            assert_eq!(
                calls(&fast, &quality),
                before,
                "an empty intersection starts neither producer"
            );
            let response = exchange(
                &cx,
                &mut session,
                &[serde_json::json!({
                    "id":"after-empty", "query":"retry", "mode":"fast",
                })],
            )
            .await;
            assert_eq!(response.len(), 2);
            assert_eq!(response[1]["event"], "results");
            assert_eq!(response[1]["results"][0]["doc_id"], "zeta");
            assert_eq!(calls(&fast, &quality), (before.0 + 1, before.1));

            let before = calls(&fast, &quality);
            let metadata_path = parent.join("metadata.json");
            let response = exchange(&cx, &mut session, &[serde_json::json!({
                "op":"update", "id":"metadata", "expected_authority":first.authority,
                "work_dir":parent.join("metadata-work"), "new_receipt":metadata_path,
                "source_low":2, "source_high":7, "changes":[
                    {"op":"upsert", "id":"zeta", "content":"retry network requests", "metadata":{"tenant":"B"}},
                    {"op":"upsert", "id":"alpha", "content":"garden flowers", "title":"Now allowed", "metadata":{"tenant":"A"}}
                ],
            })]).await;
            assert_eq!(response[1]["event"], "updated");
            assert_eq!(response[1]["publication"], "committed");
            assert_eq!(response[1]["serving"], "activated");
            assert_eq!(response[1]["scope"]["eligible_documents"], 1);
            assert_eq!(
                calls(&fast, &quality),
                before,
                "membership changes reuse each tier's unchanged content vectors"
            );
            let second = Receipt::read(&metadata_path).unwrap();
            assert_eq!(second.identity, first.identity);
            assert_eq!(second.quality_identity, first.quality_identity);
            assert_eq!(session.receipt.authority, second.authority);
            let metadata_reader = Arc::clone(&session.snapshot);

            let windows = serde_json::json!([
                {"doc_id":"zeta", "include_metadata":true},
                {"doc_id":"alpha", "include_metadata":true},
            ]);
            let response = exchange(&cx, &mut session, &[
                serde_json::json!({"op":"fetch", "expected_authority":first.authority, "documents":windows}),
                serde_json::json!({"op":"fetch", "expected_authority":second.authority, "documents":windows}),
                serde_json::json!({"query":"retry", "mode":"quality", "filter":{}}),
            ]).await;
            assert_eq!(response[1]["event"], "failed");
            assert!(response[1].get("documents").is_none());
            assert_eq!(response[2]["event"], "terminal");
            assert_eq!(
                response[2]["documents"][0],
                serde_json::json!({"doc_id":"zeta", "status":"unavailable"})
            );
            assert_eq!(response[2]["documents"][1]["content"], "garden flowers");
            assert_eq!(response[2]["documents"][1]["title"], "Now allowed");
            assert_eq!(
                response[2]["authority"],
                serde_json::json!(second.authority)
            );
            assert_eq!(response[3]["results"].as_array().unwrap().len(), 1);
            assert_eq!(response[3]["results"][0]["doc_id"], "alpha");
            assert_eq!(
                calls(&fast, &quality),
                (before.0, before.1 + 1),
                "source access does no inference and quality never invokes fast"
            );

            let before = calls(&fast, &quality);
            let changed_path = parent.join("changed.json");
            let response = exchange(&cx, &mut session, &[serde_json::json!({
                "op":"update", "id":"changed", "expected_authority":second.authority,
                "work_dir":parent.join("changed-work"), "new_receipt":changed_path,
                "source_low":2, "source_high":8, "changes":[
                    {"op":"upsert", "id":"alpha", "content":"discarded input"},
                    {"op":"upsert", "id":"alpha", "content":"retry revised garden", "metadata":{"tenant":"A"}},
                    {"op":"delete", "id":"zeta"},
                    {"op":"upsert", "id":"private", "content":"PRIVATE_SOURCE", "metadata":{"tenant":"B"}}
                ],
            })]).await;
            assert_eq!(response[1]["event"], "updated");
            assert_eq!(response[1]["documents"], 2);
            assert_eq!(response[1]["edited_ids"], 3);
            assert_eq!(response[1]["scope"]["eligible_documents"], 1);
            assert_eq!(
                calls(&fast, &quality),
                (before.0 + 2, before.1 + 2),
                "only final changed and new content is inferred by each retained producer"
            );
            let third = Receipt::read(&changed_path).unwrap();
            assert_eq!(third.identity, first.identity);
            assert_eq!(third.quality_identity, first.quality_identity);
            assert_eq!(session.snapshot.authority(), third.authority().unwrap());
            let before = calls(&fast, &quality);
            let response = exchange(
                &cx,
                &mut session,
                &[serde_json::json!({
                    "op":"fetch", "expected_authority":third.authority,
                    "documents":[{"doc_id":"alpha"},{"doc_id":"private"},{"doc_id":"zeta"}],
                })],
            )
            .await;
            assert_eq!(
                response[1]["documents"][0]["content"],
                "retry revised garden"
            );
            for (row, id) in response[1]["documents"].as_array().unwrap()[1..]
                .iter()
                .zip(["private", "zeta"])
            {
                assert_eq!(
                    *row,
                    serde_json::json!({"doc_id":id, "status":"unavailable"})
                );
            }
            assert_eq!(calls(&fast, &quality), before);
            let old_source = retained.source_documents(&cx, MAX_INPUT_BYTES).unwrap();
            assert_eq!(
                old_source
                    .iter()
                    .find(|document| document.id == "alpha")
                    .unwrap()
                    .content,
                "garden flowers"
            );
            assert_eq!(
                old_source
                    .iter()
                    .find(|document| document.id == "alpha")
                    .unwrap()
                    .metadata["tenant"],
                "B"
            );
            assert_eq!(
                fs::read(parent.join("initial.json")).unwrap(),
                original_receipt
            );

            let current_vector = session
                .snapshot
                .retained_snapshot()
                .manifest()
                .components()
                .get(GenerationComponentRole::Vector)
                .unwrap();
            let damaged_path = third.root.join(component_object_name_v1(
                GenerationComponentRole::Vector,
                current_vector.sha256,
            ));
            assert_ne!(
                current_vector.sha256,
                metadata_reader
                    .retained_snapshot()
                    .manifest()
                    .components()
                    .get(GenerationComponentRole::Vector)
                    .unwrap()
                    .sha256,
                "damage must affect only the current complete pair"
            );
            let healthy_image = parent.join("retained-current-vector");
            fs::rename(&damaged_path, &healthy_image).unwrap();
            write_immutable(&cx, &damaged_path, b"damaged-full-vector-evidence").unwrap();
            assert!(third.open(&cx, DEFAULT_IMAGE_BYTES).is_err());
            let mut options = parse(&rollback_arguments()).unwrap().unwrap();
            options.from_receipt = Some(changed_path);
            options.target_receipt = Some(metadata_path);
            options.receipt = parent.join("restored.json");
            assert!(rollback(&cx, &options, &mut Vec::new()).is_err());
            assert!(!options.receipt.exists());
            options.allow_damaged_current = true;
            let mut output = Vec::new();
            rollback(&cx, &options, &mut output).unwrap();
            let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
            assert_eq!(report["current_admission"], "control_manifest_only");
            assert_eq!(report["publication"], "committed");
            assert_eq!(report["serving"], "activated");
            let restored = Receipt::read(&options.receipt).unwrap();
            assert_eq!(restored.schema, PROGRESSIVE_RECEIPT_SCHEMA);
            assert_eq!(restored.generation, second.generation);
            assert_eq!(restored.identity, second.identity);
            assert_eq!(restored.quality_identity, second.quality_identity);
            assert_eq!(
                fs::read(&damaged_path).unwrap(),
                b"damaged-full-vector-evidence"
            );
            assert!(healthy_image.exists());
            assert_eq!(calls(&fast, &quality), before);

            let response = exchange(&cx, &mut session, &[serde_json::json!({
                "op":"fetch", "expected_authority":third.authority, "documents":[{"doc_id":"alpha"}],
            })]).await;
            assert_eq!(
                response[1]["documents"][0]["content"],
                "retry revised garden"
            );
            assert_eq!(response[1]["authority"], serde_json::json!(third.authority));
            session
                .activate(
                    &cx,
                    &AuthorityActivateRequest {
                        op: "activate".to_owned(),
                        id: None,
                        receipt: options.receipt.clone(),
                        expected_authority: third.authority.clone(),
                    },
                )
                .unwrap();
            assert_eq!(
                session.snapshot.retained_snapshot().manifest().components(),
                metadata_reader.retained_snapshot().manifest().components(),
                "rollback reuses both exact vector images and the original lexical source"
            );
            let response = exchange(&cx, &mut session, &[
                serde_json::json!({"op":"fetch", "expected_authority":third.authority, "documents":[{"doc_id":"alpha"}]}),
                serde_json::json!({"op":"fetch", "expected_authority":restored.authority, "documents":[{"doc_id":"alpha", "include_metadata":true},{"doc_id":"zeta"}]}),
            ]).await;
            assert_eq!(response[1]["event"], "failed");
            assert_eq!(response[2]["documents"][0]["content"], "garden flowers");
            assert_eq!(response[2]["documents"][0]["title"], "Now allowed");
            assert_eq!(
                response[2]["documents"][1],
                serde_json::json!({"doc_id":"zeta", "status":"unavailable"})
            );
            assert_eq!(
                response[2]["authority"],
                serde_json::json!(restored.authority)
            );
            assert_eq!(
                calls(&fast, &quality),
                before,
                "rollback, activation and fetching retain both models without inference"
            );
        });
    }
}
