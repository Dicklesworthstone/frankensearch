use super::*;
use frankensearch::native_ann::NativeExactReason;
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
    fn id(&self) -> &str {
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
        identity: identity().identity,
        calls: AtomicUsize::new(0),
    };
    assert_eq!(
        rebuild_identity(&selected, &provider, false).unwrap(),
        selected.identity
    );
    provider.identity.producer.implementation_revision =
        "changed-same-dimension-producer".to_owned();
    provider.identity.validate().unwrap();
    assert!(rebuild_identity(&selected, &provider, false).is_err());
    let migrated = rebuild_identity(&selected, &provider, true).unwrap();
    assert_ne!(migrated, selected.identity);
    assert_eq!(migrated.identity.producer, provider.identity.producer);
    assert_eq!(migrated.identity.space, provider.identity.space);
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
    provider.identity.space.dimension = 3;
    assert!(rebuild_identity(&selected, &provider, true).is_err());
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
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

fn identity() -> FrozenEmbeddingIdentityBundleV1 {
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
        documents: 2,
        image_bytes: 4096,
        attempt: Some(SavedAttempt::from_permit(&permit, false)),
    }
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
        NativeRetrievalMode::Ann,
    )
    .unwrap_err();
    let payload = publication_failure(error.as_ref()).unwrap();
    assert_eq!(payload["publication"], "committed");
    assert_eq!(payload["serving"], "activated");
    assert_eq!(payload["reason"], "result_delivery_failed");
}

#[test]
fn indexed_report_exposes_the_realized_retrieval_without_changing_the_receipt() {
    let selected = receipt();
    for (mode, label) in [
        (NativeRetrievalMode::Ann, "hnsw"),
        (
            NativeRetrievalMode::Exact {
                reason: NativeExactReason::Requested,
            },
            "exact",
        ),
    ] {
        let mut output = Vec::new();
        emit_indexed(&mut output, &selected, Path::new("/receipt.json"), mode).unwrap();
        let report: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(report["retrieval"], label);
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
                identity: identity().identity,
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
            assert_eq!(recovered["retrieval"], retrieval_name(mode));
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
            let mut publisher =
                GenerationPublisherV1::new(&root, selected.root_id, [0x72; 16], PROFILE).unwrap();
            assert!(matches!(
                publisher
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
                identity: identity().identity,
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
            let unchanged = NativeIndexBuilder::new(
                parent.join(format!("{name}-unchanged-work")),
                new_generation(2).unwrap(),
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
            assert_eq!(
                first.component_bytes(GenerationComponentRole::Lexical),
                unchanged.component_bytes(GenerationComponentRole::Lexical)
            );
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
    use std::io::{BufRead, Cursor, Read};
    use std::os::unix::fs::DirBuilderExt as _;
    use std::time::{SystemTime, UNIX_EPOCH};

    use frankensearch::IndexableDocument;

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
            model,
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
            identity: identity().identity,
            calls: AtomicUsize::new(0),
        })
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
                    Ok(model)
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
                search_with_loader(&cx, &options, &mut output, |_| Ok(model))
                    .await
                    .unwrap();
                let unrestricted: serde_json::Value = serde_json::from_slice(&output).unwrap();
                assert_eq!(unrestricted["results"][0]["doc_id"], "retry-old");
                assert!(unrestricted.get("scope").is_none());
                assert_eq!(provider.calls.load(Ordering::Relaxed), before + 2);

                let options = search_options(&receipt_path, Some(r#"{"ids":[]}"#));
                let model: Arc<dyn Embedder> = provider.clone();
                output.clear();
                search_with_loader(&cx, &options, &mut output, |_| Ok(model))
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
            search_with_loader(&cx, &options, &mut output, |_| Ok(model))
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
                    let mut identity = identity().identity;
                    identity.producer.implementation_revision = "wrong-filter-producer".to_owned();
                    identity
                },
                calls: AtomicUsize::new(0),
            });
            let options = search_options(&receipt_path, Some(r#"{"ids":[]}"#));
            let model: Arc<dyn Embedder> = wrong.clone();
            assert!(
                search_with_loader(&cx, &options, &mut Vec::new(), |_| Ok(model))
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
                model,
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
                        let mut identity = identity().identity;
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
                            Ok(model)
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
                        Ok(model)
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
                    assert!(error.to_string().contains("head differs"));
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
                let deletes = parent.join("deletes.jsonl");
                fs::write(
                    &deletes,
                    concat!(
                        "{\"op\":\"upsert\",\"id\":\"retry-baseline\",\"content\":\"discarded\"}\n",
                        "{\"op\":\"delete\",\"id\":\"retry-baseline\"}\n",
                        "{\"op\":\"delete\",\"id\":\"garden-baseline\"}\n",
                    ),
                )
                .unwrap();
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
                        "delete-all",
                        &deletes,
                        1,
                        5,
                        crate::MAX_INPUT_BYTES,
                        "nonempty",
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
                update_generation_with_loader(&cx, &options, &mut Vec::new(), |_| Ok(model))
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
                        Ok(model)
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
                        let mut identity = identity().identity;
                        identity.producer.implementation_revision =
                            "retained-source-migration".to_owned();
                        identity
                    },
                    calls: AtomicUsize::new(0),
                });
                assert!(rebuild_identity(&selected, migrated.as_ref(), false).is_err());
                assert_eq!(migrated.calls.load(Ordering::Relaxed), 0);
                let admitted = rebuild_identity(&selected, migrated.as_ref(), true).unwrap();
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
                assert!(error.to_string().contains("head differs"));
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
        changes: serde_json::Value,
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
                    serde_json::json!([
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
                    serde_json::json!([
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
                    serde_json::json!([
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
                    serde_json::json!([
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
                    serde_json::json!([
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
                serde_json::json!([
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
            let request = update_message(&parent, "disabled", &old, 6, changes.clone());
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
            let mut stale = update_message(&parent, "stale", &old, 6, changes.clone());
            stale["expected_authority"] = serde_json::json!(receipt().authority);
            let mut backward = update_message(&parent, "backward", &old, 6, changes.clone());
            backward["source_low"] = serde_json::json!(0);
            let mut inverted = update_message(&parent, "inverted", &old, 8, changes.clone());
            inverted["source_low"] = serde_json::json!(9);
            let mut overflow = update_message(&parent, "overflow", &old, u64::MAX, changes.clone());
            overflow["source_low"] = serde_json::json!(0);
            let mut inside = update_message(&parent, "inside", &old, 6, changes.clone());
            inside["work_dir"] = serde_json::json!(root_path.join("forbidden-work"));
            let mut relative = update_message(&parent, "relative", &old, 6, changes.clone());
            relative["work_dir"] = serde_json::json!("relative-work");
            let delete_all = update_message(
                &parent,
                "delete-all",
                &old,
                6,
                serde_json::json!([
                    {"op":"delete", "id":"retry-old"}, {"op":"delete", "id":"garden-old"}
                ]),
            );
            let too_many = update_message(
                &parent,
                "too-many",
                &old,
                6,
                serde_json::Value::Array(vec![
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
                ("delete-all", delete_all, "update_refused"),
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
                serde_json::json!([
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
            let request = update_message(&parent, "bounded", &old, 6, changes.clone());
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
            let request = update_message(&parent, "external-stale", &old, 6, changes);
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
            session.model = gated.clone();
            let request = update_message(
                &parent,
                "loser",
                &old,
                7,
                serde_json::json!([
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
                    serde_json::json!([
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
                serde_json::json!([
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
                serde_json::json!([]),
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
                    let mut identity = identity().identity;
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
                    wrong_model,
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
